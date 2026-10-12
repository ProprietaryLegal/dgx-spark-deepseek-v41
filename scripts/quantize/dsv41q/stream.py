# SPDX-License-Identifier: Apache-2.0
"""Layer-streaming forward of the reference DeepSeek-V4.1 over a fixed token set: every sequence through layer l
before any goes through layer l+1, so only one block (~7 GiB with MXFP4 experts) is resident at a time.

What crosses layers, per sequence, is exactly what the reference Transformer.forward carries:
  h        [1, L, hc_mult, dim]  the hyper-connection residual stream (bf16)
  pre_mix  [1, L, hc_mult]       the next block's attention input mix (fp32)
  shared   the SharedAttentionRuntime slots (compress_kv, index_k, topk_idxs, candidates) written by kv / index /
           candidate source layers and read by the layers after them. The reference keeps them in one global for a
           single forward; here they are cloned per sequence after every layer that writes them.

Hooks see each MoE call (post-norm input, routing) so calibration passes can capture statistics or Hessians.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch

from .refmodel import Checkpoint, engram_hasher, load_block, load_reference, model_args


@dataclass
class SeqState:
    ids: torch.Tensor                      # [1, L] int64 (device)
    h: torch.Tensor | None = None          # [1, L, hc, dim] bf16
    pre_mix: torch.Tensor | None = None    # [1, L, hc] fp32
    hashes: torch.Tensor | None = None     # [1, L, n_engram_layers, n_hash_cols] int64
    shared: dict = field(default_factory=dict)


class Streamer:
    def __init__(self, ckpt_dir: str, seqs: list[list[int]], device="cuda", max_seq_len: int | None = None):
        self.ckpt = Checkpoint(ckpt_dir)
        self.model, self.kernel, _ = load_reference(ckpt_dir)
        # GB10: CUDA memory is system memory; a runaway allocation must fail in this process, not hang the box
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("DSV41_CUDA_MEM_FRACTION", "0.70")))
        # the reference builds index tensors with bare torch.arange (generate.py sets the default device to cuda)
        torch.set_default_device(device)
        L = max(len(s) for s in seqs)
        self.args = model_args(ckpt_dir, 1, max_seq_len or max(L, 128))
        self.device = torch.device(device)
        self.hasher, self.layout = engram_hasher(ckpt_dir, 1, self.args.max_seq_len)
        embed = self.ckpt.get("embed.weight", self.device)                   # [vocab, dim] bf16
        self.states = []
        for s in seqs:
            ids = torch.tensor([s], dtype=torch.long, device=self.device)
            st = SeqState(ids=ids)
            st.h = embed[ids].unsqueeze(2).repeat(1, 1, self.args.hc_mult, 1)
            st.pre_mix = self.model.make_identity_pre_mix(st.h, self.args.hc_mult)
            st.hashes = self.hasher(ids, 0, None).clone()
            self.states.append(st)
        del embed
        torch.cuda.empty_cache()
        self.layer = 0

    # -- shared attention runtime, per sequence ------------------------------------------------------------------
    _SLOTS = ("compress_kv", "index_k", "topk_idxs", "candidates")

    def _restore_shared(self, st: SeqState):
        sa = self.model.shared_attn
        for k in self._SLOTS:
            setattr(sa, k, st.shared.get(k))

    def _save_shared(self, st: SeqState, layer_id: int, seqlen: int):
        sa = self.model.shared_attn
        for k in self._SLOTS:
            v = getattr(sa, k)
            if v is None or v is st.shared.get(k):
                continue
            if k in ("compress_kv", "index_k"):
                ratio = self.args.compress_ratios[layer_id]
                v = v[:1, : seqlen // ratio]                     # the valid rows of the source layer's cache
            st.shared[k] = v.clone()

    # -- one layer for every sequence -------------------------------------------------------------------------------
    def groups(self, max_tokens: int):
        """Consecutive runs of sequences holding at most max_tokens tokens (a longer sequence is a group by itself)."""
        out, cur, n = [], [], 0
        for st in self.states:
            L = st.ids.size(1)
            if cur and n + L > max_tokens:
                out.append(cur)
                cur, n = [], 0
            cur.append(st)
            n += L
        if cur:
            out.append(cur)
        return out

    @torch.inference_mode()
    def run_layer(self, block, moe=None, moe_chunk_tokens: int = 65536, block_hook=None):
        """Block `self.layer` for every sequence, split the way Block.forward is (inference/model.py), group by group
        (consecutive sequences of <= moe_chunk_tokens tokens), so the MoE sees large token batches while only one
        group's intermediates are alive (on GB10 the CUDA allocator draws from system RAM: holding every sequence's
        post-attention stream at once took a 548K-token pass past 120 GB and the box down, Oct 8):

          A (per sequence): engram, attention half, hc_post, then hc_mixes / hc_pre / ffn_norm -> the MoE input
          B (the group's tokens): y = moe(x, block)          (default: block.ffn, the reference MoE)
          C (per sequence): hc_post(y, residual, ffn_post, ffn_comb), next pre_mix = ffn_pre

        moe is called once per group, in sequence order, on that group's tokens concatenated (callers can map rows to
        a global token index with a running offset). block_hook(seq_idx, h_in, h_out) sees each sequence's stream."""
        moe = moe or (lambda x, b: b.ffn(x))
        base = 0
        for grp in self.groups(moe_chunk_tokens):
            flat, pend = self.phase_a(block, states=grp, keep_input=block_hook is not None)
            ys = moe(flat, block)
            del flat
            self.phase_c(block, ys, pend, states=grp, block_hook=block_hook, index_base=base, advance=False)
            base += len(grp)
            del ys, pend
        self.layer += 1

    @torch.inference_mode()
    def phase_a(self, blk, states=None, keep_input: bool = False, keep_pend: bool = True):
        """Everything before the MoE for the given sequences (default all). Returns (x [N_tokens, dim] bf16: their MoE
        input concatenated in order, pend: per-sequence state phase_c needs, or None with keep_pend=False)."""
        pend, xs = [], []
        for st in (self.states if states is None else states):
            self._restore_shared(st)
            h = st.h
            if blk.engram is not None:
                h = blk.engram(h, st.hashes[:, :, blk.engram.layer_hash_index, :], None)
            residual = h
            attn_pre, attn_post, attn_comb = blk.hc_mixes(h, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
            x = blk.attn_norm(blk.hc_pre(h, st.pre_mix))
            x = blk.attn(x, 0)
            x = blk.hc_post(x, residual, attn_post, attn_comb)
            self._save_shared(st, self.layer, st.ids.size(1))
            residual = x
            ffn_pre, ffn_post, ffn_comb = blk.hc_mixes(x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
            x = blk.ffn_norm(blk.hc_pre(x, attn_pre))
            xs.append(x.view(-1, x.size(-1)))
            if keep_pend:
                pend.append((h if keep_input else None, residual, ffn_pre, ffn_post, ffn_comb))
        return torch.cat(xs), (pend if keep_pend else None)

    @torch.inference_mode()
    def phase_c(self, blk, ys, pend, states=None, block_hook=None, index_base: int = 0, advance: bool = True):
        """hc_post with the MoE output ys [N_tokens, dim] for the given sequences (default all)."""
        off = 0
        sts = self.states if states is None else states
        for i, (st, (h_in, residual, ffn_pre, ffn_post, ffn_comb)) in enumerate(zip(sts, pend)):
            n = st.ids.size(1)
            y = ys[off: off + n].view(1, n, -1)
            off += n
            h_out = blk.hc_post(y, residual, ffn_post, ffn_comb)
            if block_hook is not None:
                block_hook(index_base + i, h_in, h_out)
            st.h, st.pre_mix = h_out, ffn_pre
        if advance:
            self.layer += 1

    def load(self, layer_id: int, experts: bool = True):
        return load_block(self.ckpt, self.args, layer_id, self.layout, self.device, experts=experts)

    @torch.inference_mode()
    def iter_logits(self, last_block, extra_heads: dict | None = None):
        """Yield (seq_idx, fp32 logits [L, vocab]) one sequence at a time (all positions of a 2K-token sequence are
        ~1 GB, so callers reduce each before the next). With extra_heads {name: fn(head_w) -> weight}, yield
        (seq_idx, logits, {name: logits with that head}) instead (e.g. an MXFP8 head, scored in the same pass)."""
        norm_w = self.ckpt.get("norm.weight", self.device).float()
        base = self.ckpt.get("head.weight", self.device).float()   # the checkpoint's BF16 head
        over = getattr(self, "head_override", None)                # an EXL3 head (dsv41q.overlay.head_weight)
        head_w = base if over is None else over.float()
        heads = dict(extra_heads or {})
        if over is not None and heads:
            heads.setdefault("head_bf16", lambda w: w)             # the BF16 head as a variant, for comparison
        extra = {k: fn(base) for k, fn in heads.items()}          # variants are built from the BF16 head
        extra_heads = heads if heads else extra_heads
        for i, st in enumerate(self.states):
            x = last_block.hc_pre(st.h, st.pre_mix).float()
            x = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.args.norm_eps) * norm_w).to(torch.bfloat16)
            lg = torch.nn.functional.linear(x[0].float(), head_w)
            if extra_heads is None:
                yield i, lg
            else:
                yield i, lg, {k: torch.nn.functional.linear(x[0].float(), w) for k, w in extra.items()}

    @torch.inference_mode()
    def logits(self, last_block, positions: str = "all") -> list[torch.Tensor]:
        """Final norm + head after the last streamed layer (fp32 logits, [L, vocab] per sequence). An EXL3 head in
        self.head_override (dsv41q.overlay.head_weight) replaces the checkpoint's BF16 head, as in iter_logits."""
        norm_w = self.ckpt.get("norm.weight", self.device)
        over = getattr(self, "head_override", None)
        head_w = (self.ckpt.get("head.weight", self.device) if over is None else over).float()
        out = []
        for st in self.states:
            x = last_block.hc_pre(st.h, st.pre_mix)
            x = x.float()
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.args.norm_eps) * norm_w.float()
            x = x.to(torch.bfloat16)                            # RMSNorm returns the input dtype
            lg = torch.nn.functional.linear(x[0].float(), head_w)
            out.append(lg if positions == "all" else lg[-1:])
        return out


def run_truncated(ckpt_dir: str, seqs: list[list[int]], n_layers: int, verbose: bool = True, tensors_dir: str = "",
                  dense_dir: str = ""):
    """Reference forward through the first n_layers, then norm + head: what a vLLM boot with
    text_config.num_hidden_layers = n_layers computes. Returns per-sequence fp32 logits [L, vocab]. With tensors_dir,
    routed experts of layers that have pass-1 files there run as EXL3 (reconstructed, fp16 activations); with
    dense_dir (p4_dense.py output), the dense linears it covers (and its head.*, when present) run from their EXL3
    reconstruction too: the same weights a vLLM boot of the assembled checkpoint serves."""
    import glob
    s = Streamer(ckpt_dir, seqs)
    dense = None
    if dense_dir:
        from .overlay import Exl3Checkpoint, head_weight
        dense = Exl3Checkpoint(dense_dir)
        hw = head_weight(dense, s.device)                   # the EXL3 head when the dense pass encoded one (head.*)
        if hw is not None:
            s.head_override = hw
            if verbose:
                print(f"head: EXL3 from {dense_dir}", flush=True)
    blk = None
    for l in range(n_layers):
        t0 = time.time()
        blk = s.load(l)
        if dense is not None:
            from .overlay import overlay_block
            done = overlay_block(blk, l, dense.tensors(f"layers.{l}.", s.device))
            if verbose:
                print(f"layer {l}: EXL3 dense {done}", flush=True)
        t1 = time.time()
        moe = None
        files = sorted(glob.glob(os.path.join(tensors_dir, f"layer_{l:02d}.part*.safetensors"))) if tensors_dir else []
        if files:
            from safetensors.torch import load_file
            from .exl3 import quant_moe
            from .refmodel import drop_page_cache
            tens = {}
            for f in files:
                tens.update(load_file(f, device=str(s.device)))
                drop_page_cache(f)
            moe = quant_moe(l, tens, s.args.swiglu_limit)
        s.run_layer(blk, moe=moe)
        torch.cuda.synchronize()
        if verbose:
            print(f"layer {l}: load {t1 - t0:.1f}s, forward {time.time() - t1:.1f}s", flush=True)
        if l < n_layers - 1:
            del blk
            torch.cuda.empty_cache()
    return s.logits(blk)
