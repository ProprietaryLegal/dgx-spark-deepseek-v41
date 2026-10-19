# SPDX-License-Identifier: Apache-2.0
"""P4 dense pass: EXL3 encoding of DeepSeek-V4.1's dense linears (attention wq_a + wkv / wq_b / wo_a / wo_b, shared
expert w1 / w3 / w2) on the calibration stream through pass 1's EXL3 experts.

Why: the FP8 dense linears are ~34% of a TP=2 decode step on GB10 (dummy-weight boot, Oct 8): Marlin reads ~4.3 GB per
rank per step at ~240 GB/s, bandwidth-bound, so the lever is bytes. EXL3 K5 is 5 bits per weight against MXFP8's 8.25
and adds ~1% relative output error to the FP8 weights (the per-K figures measured on the experts).

Per layer: one forward of the whole calibration set with pass 1's EXL3 experts and the native dense weights (not
sequential for the dense part: K5 errors are ~1%, far below the experts', so they barely move later inputs), summing
input Hessians over all tokens:
  attn.wq_a / wkv         the attention input (both read it: one LDLQ pass over both; vLLM fuses them as fused_wqa_wkv,
                          ~45 us per layer per decode step at FP8, replicated on both ranks)   [5120]
  attn.wq_b               qr after q_norm                                   [1280]
  attn.wo_a.slice.G       head group G of the attention output after the inverse RoPE, G = 0..7   [4096] each
                          (the reference applies wo_a with an einsum, so it is captured there)
  attn.wo_b               the wo_a output                                   [8192]
  ffn.shared_experts.w1 / w3   the FFN input (one LDLQ pass over both, as for the routed experts)    [5120]
  ffn.shared_experts.w2        silu(gate) * up                              [2304]
then EXL3-encodes this part's layers (layer % n == i) with exllamav3's quantizer (mul1 codebook). Tensor names follow
the EXL3 pack convention (Mia-AiLab's 2.9 bpw pack uses the same): layers.L.attn.wq_b.{trellis,suh,svh,mul1},
layers.L.attn.wo_a.slice.G.*, layers.L.attn.wo_b.*, layers.L.ffn.shared_experts.wN.*

  python3 p4_dense.py /model /cal.safetensors /pass1_dir /out [--k 5] [--k-map attn.wo_a=6,...] [--part 0/2] [--no-wqa-wkv]
                     [--out-alt /out2 --k-alt 4 --k-map-alt attn.wo_a=5]   (a second K set from the same Hessians)
                     [--head 6 --head-alt 5]   (the head at K into /out and, with --out-alt, at another K into /out2)
  python3 p4_dense.py --index /out          (after both parts' files are in /out: model.safetensors.index.json)
Outputs: /out/dense_LXX.safetensors (EXL3 tensors) + /out/dense_LXX.json (Hessian-weighted error per matrix).
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.exl3 import h_data, quant_args, quant_moe, reconstruct  # noqa: E402
from dsv41q.refmodel import _dequant_block_fp8, drop_page_cache  # noqa: E402
from dsv41q.stream import Streamer  # noqa: E402

MODULES = ("attn.wq_a", "attn.wkv", "attn.wq_b", "attn.wo_a", "attn.wo_b", "ffn.shared_experts")


class HAcc:
    """Sum of x^T x per key (fp32) and the row count."""

    def __init__(self):
        self.H: dict[str, torch.Tensor] = {}
        self.n: dict[str, int] = {}

    def add(self, key: str, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        if key not in self.H:
            self.H[key] = torch.zeros(x.shape[-1], x.shape[-1], dtype=torch.float32, device=x.device)
            self.n[key] = 0
        self.H[key].addmm_(x.t(), x)
        self.n[key] += x.shape[0]


def install_capture(blk, acc: HAcc, wqa_wkv: bool = True):
    """Pre-hooks (each returns None: a pre-hook's return value would replace the module input) + the wo_a einsum."""
    hooks = []

    def pre(key):
        def hook(_module, inputs):
            acc.add(key, inputs[0])
        return hook
    if wqa_wkv:                                    # wkv reads the same x (Attention._window_kv): one Hessian for both
        hooks.append(blk.attn.wq_a.register_forward_pre_hook(pre("attn.wqa_wkv")))
    hooks.append(blk.attn.wq_b.register_forward_pre_hook(pre("attn.wq_b")))
    hooks.append(blk.attn.wo_b.register_forward_pre_hook(pre("attn.wo_b")))
    hooks.append(blk.ffn.shared_experts.w1.register_forward_pre_hook(pre("ffn.shared_experts.w13")))
    hooks.append(blk.ffn.shared_experts.w2.register_forward_pre_hook(pre("ffn.shared_experts.w2")))
    orig = torch.einsum

    def einsum(eq, *ops):
        if eq == "bsgd,grd->bsgr":                     # Attention.forward: o [b, s, groups, 4096] x wo_a
            o = ops[0]
            for g in range(o.shape[2]):
                acc.add(f"attn.wo_a.slice.{g}", o[:, :, g, :])
        return orig(eq, *ops)
    torch.einsum = einsum

    def remove():
        for h in hooks:
            h.remove()
        torch.einsum = orig
    return remove


def fp32_weight(mod) -> torch.Tensor:
    """nn.Linear-orientation fp32 weight of a reference Linear (block-FP8 dequantized exactly, or already bf16)."""
    w = mod.weight
    if w.dtype == torch.float8_e4m3fn:
        return _dequant_block_fp8(w, mod.scale).float()
    return w.float()


def h_err(w: torch.Tensor, wq: torch.Tensor, H: torch.Tensor) -> float:
    """Relative output error under the calibration inputs: tr(D H D^T) / tr(W H W^T), D = Wq - W ([out, in])."""
    d = wq.float() - w
    return ((d @ H) * d).sum().div(((w @ H) * w).sum().clamp_min(1e-30)).sqrt().item()


def encode_layer(blk, acc: HAcc, l: int, kmap: dict, work: str, seed: int, wqa_wkv: bool = True):
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, quantize_exl3_batch
    out, metrics = {}, {}
    # exllamav3 finalizes (scales, damps) and rotates the Hessian it is given in place: keep untouched copies for the
    # error metric, normalized per row so every key reads as a mean x^T x
    H0 = {k: (v / max(acc.n[k], 1)).clone() for k, v in acc.H.items()}

    def put(name, t, w):
        k_in, n_out = w.shape[1], w.shape[0]
        for piece, v in t.items():
            out[f"layers.{l}.{name}.{piece}"] = v.contiguous()
        metrics[name] = {"k": kmap_get(name), "rel_err": h_err(w, reconstruct(t, k_in, n_out), H0[hkey(name)])}

    def kmap_get(name):
        for prefix, k in kmap.items():
            if name.startswith(prefix):
                return k
        return kmap["*"]

    def hkey(name):
        if name in ("attn.wq_a", "attn.wkv"):
            return "attn.wqa_wkv"
        return "ffn.shared_experts.w13" if name in ("ffn.shared_experts.w1", "ffn.shared_experts.w3") else name

    def one(name, w, sd):
        H = h_data(acc.H[hkey(name)], acc.n[hkey(name)], name)
        _, pe, t = quantize_exl3(w.t().contiguous(), H, quant_args(kmap_get(name), sd, work), return_weight_q=False)
        put(name, t, w)

    if wqa_wkv:                                    # one K for the fused pair: the plugin serves them as one module
        wq_a, wkv = fp32_weight(blk.attn.wq_a), fp32_weight(blk.attn.wkv)
        kq = kmap_get("attn.wq_a")
        Hq = h_data(acc.H["attn.wqa_wkv"], acc.n["attn.wqa_wkv"], "wqa_wkv")
        (_, tq), (_, tkv) = quantize_exl3_batch([wq_a.t().contiguous(), wkv.t().contiguous()], [Hq, Hq],
                                                [quant_args(kq, seed + 20, work), quant_args(kq, seed + 21, work)])
        put("attn.wq_a", tq, wq_a)
        put("attn.wkv", tkv, wkv)
    one("attn.wq_b", fp32_weight(blk.attn.wq_b), seed)
    wo_a = fp32_weight(blk.attn.wo_a)
    r = wo_a.shape[0] // 8
    for g in range(8):
        one(f"attn.wo_a.slice.{g}", wo_a[g * r:(g + 1) * r].contiguous(), seed + 1 + g)
    one("attn.wo_b", fp32_weight(blk.attn.wo_b), seed + 9)
    se = blk.ffn.shared_experts
    w1, w3 = fp32_weight(se.w1), fp32_weight(se.w3)
    k13 = kmap_get("ffn.shared_experts.w1")
    H13 = h_data(acc.H["ffn.shared_experts.w13"], acc.n["ffn.shared_experts.w13"], "w13")
    (_, t1), (_, t3) = quantize_exl3_batch([w1.t().contiguous(), w3.t().contiguous()], [H13, H13],
                                           [quant_args(k13, seed + 10, work), quant_args(k13, seed + 11, work)])
    put("ffn.shared_experts.w1", t1, w1)
    put("ffn.shared_experts.w3", t3, w3)
    one("ffn.shared_experts.w2", fp32_weight(se.w2), seed + 12)
    return out, metrics


def encode_head(s, last_blk, targets, work: str, seed: int, say) -> None:
    """lm_head [vocab, dim] at each (K, out_dir) of `targets` from one H = sum x^T x of the final normed hidden states
    (the head's input); exllamav3 finalizes a Hessian in place, so every target but the last gets a copy."""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    t0 = time.time()
    norm_w = s.ckpt.get("norm.weight", s.device).float()
    H, n = None, 0
    for st in s.states:                                   # same final mix + norm as Streamer.iter_logits
        x = last_blk.hc_pre(st.h, st.pre_mix).float()
        x = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + s.args.norm_eps) * norm_w).to(torch.bfloat16)
        x = x.reshape(-1, x.shape[-1]).float()
        H = x.t() @ x if H is None else H.addmm_(x.t(), x)
        n += x.shape[0]
    w = s.ckpt.get("head.weight", s.device).float()        # [vocab, dim]
    H0 = (H / n).clone()
    # the raw sum and row count (hcap.py's format), before exllamav3 finalizes it in place: a later head encode at
    # another K needs only this file, not a full calibration stream
    torch.save({"H": H.cpu(), "rows": n}, os.path.join(targets[0][1], "head_H.pt"))
    for i, (k, out_dir) in enumerate(targets):
        t1 = time.time()
        Hk = H.clone() if i < len(targets) - 1 else H
        _, _, t = quantize_exl3(w.t().contiguous(), h_data(Hk, n, "head"), quant_args(k, seed, work),
                                return_weight_q=False)
        rel = h_err(w, reconstruct(t, w.shape[1], w.shape[0]), H0)
        p = os.path.join(out_dir, "dense_head.safetensors")
        save_file({f"head.{piece}": v.contiguous() for piece, v in t.items()}, p + ".tmp")
        os.chmod(p + ".tmp", 0o644)
        os.replace(p + ".tmp", p)
        with open(os.path.join(out_dir, "dense_head.json"), "w") as fh:
            json.dump({"k": k, "rel_err": rel, "tokens": n, "seconds": round(time.time() - t1, 1)}, fh, indent=1)
        say(f"head: K{k} -> {out_dir}, rel err {rel:.4f} over {n:,} tokens, {time.time() - t1:.0f}s")
        del t, Hk
        torch.cuda.empty_cache()
    say(f"head done in {time.time() - t0:.0f}s")


def write_index(d: str) -> None:
    wm = {}
    for f in sorted(glob.glob(os.path.join(d, "dense_*.safetensors"))):
        from safetensors import safe_open
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                wm[k] = os.path.basename(f)
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"format": "exl3 dense linears (p4_dense.py)"}, "weight_map": wm}, fh, indent=1)
    print(f"index: {len(wm)} tensors in {len(set(wm.values()))} files")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("cal", nargs="?")
    ap.add_argument("experts", nargs="?", help="pass 1 output dir (layer_XX.part*.safetensors)")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--k", type=float, default=5)
    ap.add_argument("--k-map", default="", help="per-module K by name prefix, e.g. attn.wo_a=6,ffn.shared_experts=4")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=914)
    ap.add_argument("--index", default="", help="only write model.safetensors.index.json for this dir")
    ap.add_argument("--no-wqa-wkv", action="store_true",
                    help="leave wq_a / wkv in FP8 (by default they are encoded too, at the K of attn.wq_a)")
    ap.add_argument("--out-alt", default="", help="also encode every module at --k-alt / --k-map-alt into this dir, from "
                    "the same captured Hessians (the stream, most of the time, is shared)")
    ap.add_argument("--k-alt", type=float, default=4)
    ap.add_argument("--head-alt", type=int, default=0, help="with --head and --out-alt: the head also at this K in --out-alt")
    ap.add_argument("--k-map-alt", default="")
    ap.add_argument("--head", type=int, default=0,
                    help="also encode the lm_head at this K from the final hidden states (part 0 only): head.* tensors "
                         "in dense_head.safetensors")
    a = ap.parse_args()
    if a.index:
        return write_index(a.index)
    part, nparts = (int(x) for x in a.part.split("/"))
    def parse_kmap(k_all, spec):
        km = {"*": int(k_all) if float(k_all).is_integer() else k_all}
        for kv in filter(None, spec.split(",")):
            name, k = kv.split("=")
            km[name] = int(float(k)) if float(k).is_integer() else float(k)
        return km
    kmap = parse_kmap(a.k, a.k_map)
    kmap_alt = parse_kmap(a.k_alt, a.k_map_alt) if a.out_alt else None
    if a.out_alt:
        os.makedirs(a.out_alt, exist_ok=True)
    l0, l1 = (int(x) for x in a.layers.split("-"))
    n_layers = json.load(open(os.path.join(a.model, "inference", "config.json")))["n_layers"]
    if a.head and part == 0 and l1 != n_layers - 1:      # the head's H comes from the stream after the last layer
        raise SystemExit(f"--head needs the stream through the last layer: --layers {l0}-{n_layers - 1}, not {a.layers}")
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "work")
    os.makedirs(work, exist_ok=True)
    torch.set_default_dtype(torch.bfloat16)
    d = load_file(a.cal)
    toks, offs = d["tokens"].tolist(), d["offsets"].tolist()
    seqs = [toks[offs[i]: offs[i + 1]] for i in range(len(offs) - 1)]
    seqs = seqs[:a.max_seqs] if a.max_seqs else seqs
    s = Streamer(a.model, seqs)
    limit = s.args.swiglu_limit
    log = open(os.path.join(a.out, f"dense.part{part}.log"), "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
    say(f"dense pass part {part}/{nparts}: {len(seqs)} sequences, {sum(map(len, seqs)):,} tokens, layers {l0}-{l1}, K {kmap}"
        + (f", alt K {kmap_alt} -> {a.out_alt}" if kmap_alt else ""))
    t_all = time.time()
    for l in range(0, l1 + 1):
        t0 = time.time()
        blk = s.load(l)
        tensors = {}
        for f in sorted(glob.glob(os.path.join(a.experts, f"layer_{l:02d}.part*.safetensors"))):
            tensors.update(load_file(f, device="cuda"))
            drop_page_cache(f)
        n_exp = len({k.split(".")[4] for k in tensors if ".ffn.experts." in k})
        if n_exp != 384:
            raise SystemExit(f"layer {l}: {n_exp} experts in pass 1's tensors in {a.experts}, need all 384")
        mine = l0 <= l and l % nparts == part
        acc = HAcc()
        remove = install_capture(blk, acc, not a.no_wqa_wkv) if mine else (lambda: None)
        try:
            s.run_layer(blk, moe=quant_moe(l, tensors, limit), moe_chunk_tokens=131072)   # fewer expert rebuilds
        finally:
            remove()
        del tensors
        t1 = time.time()
        if mine:
            sets = [(a.out, kmap)] + ([(a.out_alt, kmap_alt)] if kmap_alt else [])
            for i, (odir, km) in enumerate(sets):
                t2 = time.time()
                # exllamav3 finalizes the Hessians it is given in place: every set but the last gets copies
                use = acc
                if i < len(sets) - 1:
                    use = HAcc()
                    use.H, use.n = {k: v.clone() for k, v in acc.H.items()}, dict(acc.n)
                out, metrics = encode_layer(blk, use, l, km, work, a.seed + 100 * l, not a.no_wqa_wkv)
                del use
                p = os.path.join(odir, f"dense_L{l:02d}.safetensors")
                save_file(out, p + ".tmp")
                os.chmod(p + ".tmp", 0o644)
                os.replace(p + ".tmp", p)
                drop_page_cache(p)
                metrics.update(tokens={k: v for k, v in acc.n.items()}, forward_s=round(t1 - t0, 1),
                               encode_s=round(time.time() - t2, 1))
                with open(os.path.join(odir, f"dense_L{l:02d}.json"), "w") as fh:
                    json.dump(metrics, fh, indent=1)
                worst = max((v["rel_err"], k) for k, v in metrics.items() if isinstance(v, dict) and "rel_err" in v)
                say(f"layer {l:2d}{' (alt)' if i else ''}: forward {t1 - t0:.0f}s encode {time.time() - t2:.0f}s, "
                    f"worst rel err {worst[0]:.4f} ({worst[1]})")
        else:
            say(f"layer {l:2d}: forward {t1 - t0:.0f}s (stream only)")
        del acc
        if l < l1 or not (a.head and part == 0):
            del blk
        torch.cuda.empty_cache()
    if a.head and part == 0:
        targets = [(a.head, a.out)] + ([(a.head_alt, a.out_alt)] if a.head_alt and a.out_alt else [])
        encode_head(s, blk, targets, work, a.seed + 9999, say)
    say(f"DONE in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
