# SPDX-License-Identifier: Apache-2.0
"""Fidelity on the published panel: KL(native || ours), top-1 agreement and NLL, at every scored (assistant)
position of every panel conversation, both models run by the same layer-streamed reference implementation, so the
number measures the weights, not two different engines.

  --mode ref    native checkpoint; writes ref.safetensors: per scored position the top-128 token ids + logprobs and
                the NLL of the true next token
  --mode quant  routed experts replaced by our EXL3 tensors (--tensors DIR: layer_XX.part*.safetensors from pass 1,
                missing layers stay native); reads ref.safetensors, writes kld.json

KL uses the native top-128 support plus one tail bucket: sum_top P log(P/Q) + P_rest log(P_rest / Q_rest), exact when
the top-128 holds all the mass and a tight estimate otherwise.

  python3 p4_kld.py /model /caldata/panel.safetensors /caldata/panel_index.json /out --mode ref
  python3 p4_kld.py /model /caldata/panel.safetensors /caldata/panel_index.json /out --mode quant --tensors /pass1
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.exl3 import quant_moe  # noqa: E402
from dsv41q.overlay import Exl3Checkpoint, head_weight, overlay_block  # noqa: E402
from dsv41q.refmodel import drop_page_cache  # noqa: E402
from dsv41q.stream import Streamer  # noqa: E402

TOPK = 128


_ACT = {"qdq": None}


def set_act_qdq(fn) -> None:
    """--act-qdq-layers: switch the round trip on (fn) or off (None) for the next layer's linears and experts."""
    from dsv41q import exl3 as exl3_mod
    _ACT["qdq"] = fn
    exl3_mod.ACT_QDQ = fn


def install_tap_dump(s, out_dir: str, seqs) -> None:
    """--dump-taps: the reference's main_hidden pieces (Transformer.forward: h.mean(dim=2) entering each
    dspark_target_layer_ids layer) for every position, written as they pass (no host copy of a whole layer)."""
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    layers = list(getattr(s.args, "dspark_target_layer_ids", ()) or ())
    offs = [0]
    for q in seqs:
        offs.append(offs[-1] + len(q))
    dim = s.args.dim
    maps = {l: np.lib.format.open_memmap(os.path.join(out_dir, f"taps_L{l}.npy"), mode="w+", dtype=np.uint16,
                                         shape=(offs[-1], dim)) for l in layers}
    json.dump({"layers": layers, "offsets": offs, "dim": dim, "dtype": "bf16 bits in uint16"},
              open(os.path.join(out_dir, "taps.json"), "w"))
    index = {id(st): i for i, st in enumerate(s.states)}

    def hook(layer, st, h):
        if layer in maps:
            i = index[id(st)]
            t = h.mean(dim=2)[0].to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)
            maps[layer][offs[i]:offs[i + 1]] = t
            if i == len(offs) - 2:
                maps[layer].flush()

    s.tap_hook = hook
    print(f"dumping taps of layers {layers} for {offs[-1]:,} positions to {out_dir}", flush=True)


BLOCKS = (5120, 128, 16)       # --head-check: the error bound's block sizes over the hidden dim


def head_check(s, blk, score, offs, seqs, index, specs: str, ks, out_dir: str, hessian_path: str = "") -> dict:
    """--head-check: could a cheap first-stage head find the served head's argmax exactly? Per scored position, x = the
    normed final hidden (the head's bf16 input), L5 = x . W5 (this pack's head as served) and, per stage-1 head W1,
    L1 = x . W1; per vocab shard (TP=2 halves) the top-k of L1 are rescored with W5; the shard's exact max is certified
    when it beats every non-candidate's bound L1[r] + |x| e_r (e_r = |W5[r] - W1[r]|, + a 0.05 margin), and the
    position's greedy token is certified when both shards are. Reported per stage-1 head and k: recall (the W5 argmax
    among the candidates), certified (per-row bounds), certified with the coarse bound (the k+1-th L1 + |x| max e),
    the median |x| max e; and, with hessian_path, X^T X of x over the scored positions (a stage-1 head's encode)."""
    from safetensors import safe_open
    from dsv41q.exl3 import reconstruct
    from dsv41q.overlay import _dims, _modules
    dev = "cuda"
    norm_w = s.ckpt.get("norm.weight", dev).float()
    w5 = s.head_override.float() if getattr(s, "head_override", None) is not None else s.ckpt.get("head.weight", dev).float()
    V = w5.shape[0]
    shards = [(0, V // 2), (V // 2, V)]
    heads = {}
    for spec in specs.split(","):
        name, src = spec.split("=", 1)
        src, _, prefix = src.partition(":")
        prefix = prefix or "head."
        if os.path.isdir(src):
            from dsv41q.overlay import Exl3Checkpoint
            t = Exl3Checkpoint(src).tensors(prefix, dev)
        else:
            with safe_open(src, "pt", device=dev) as f:
                t = {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
        mods = _modules(t, prefix[: -len("head.")] if prefix.endswith("head.") else prefix)
        w1 = reconstruct(mods["head"], *_dims(mods["head"])).float()
        d = w5 - w1
        # per-row error norms over blocks of the hidden dim: |x . d_r| <= sum_b |x_b| |d_r,b| (Cauchy-Schwarz per block,
        # rigorous, tighter as blocks shrink): the whole row (5120), 128-wide and 16-wide blocks
        eb = {bs: d.view(d.shape[0], -1, bs).norm(dim=2) for bs in BLOCKS}
        heads[name] = (w1, eb)
        del d
        print(f"head-check: stage-1 {name} from {src} ({prefix}*): K {mods['head']['trellis'].shape[-1] // 16}, "
              f"max row error {float(eb[w5.shape[1]].max()):.4f}", flush=True)
    SIGMAS = (6.0, 8.0, 10.0)      # statistical margins: the error x . d_r ~ N(0, (|x| |d_r| / sqrt(D))^2)
    zmax = {n: 0.0 for n in heads}
    stat = {(n, k): {sg: 0 for sg in SIGMAS} for n in heads for k in ks}
    H = torch.zeros(w5.shape[1], w5.shape[1], device=dev, dtype=torch.float32) if hessian_path else None
    n_h = 0
    st_ = {(n, k): {"n": 0, "recall": 0, "cert": {bs: 0 for bs in BLOCKS}, "dom": {}} for n in heads for k in ks}
    gaps = {n: [] for n in heads}
    for i, stt in enumerate(s.states):
        sc = score[offs[i]: offs[i + 1]].bool().to(dev)
        pos = sc[1:].nonzero().squeeze(-1)
        if pos.numel() == 0:
            continue
        x = blk.hc_pre(stt.h, stt.pre_mix).float()
        x = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + s.args.norm_eps) * norm_w).to(torch.bfloat16)
        x = x[0][pos].float()                                              # [P, dim]
        if H is not None:
            H.addmm_(x.t(), x)
            n_h += x.shape[0]
        xn = x.norm(dim=1)                                                 # [P]
        l5 = x @ w5.t()                                                    # [P, V]
        top5 = l5.argmax(-1)
        dom = index[i]["domain"].split("_")[0]
        for name, (w1, eb) in heads.items():
            l1 = x @ w1.t()
            gaps[name].append((xn * eb[x.shape[1]].max()).cpu())
            xb = {bs: x.view(x.shape[0], -1, bs).norm(dim=2) for bs in BLOCKS}       # [P, blocks]
            sig = xn[:, None] * eb[x.shape[1]][:, 0][None, :] / math.sqrt(x.shape[1])   # [P, V]
            zmax[name] = max(zmax[name], float(((l5 - l1).abs() / sig.clamp_min(1e-12)).max()))
            for k in ks:
                cert = {bs: torch.ones(len(pos), dtype=torch.bool, device=dev) for bs in BLOCKS}
                rec = torch.zeros(len(pos), dtype=torch.bool, device=dev)
                for lo, hi in shards:
                    a1, a5 = l1[:, lo:hi], l5[:, lo:hi]
                    idx = a1.topk(k, dim=-1).indices
                    best = a5.gather(1, idx).max(1).values
                    for bs in BLOCKS:
                        bound = a1 + xb[bs] @ eb[bs][lo:hi].t()                    # [P, shard]
                        bound = bound.scatter(1, idx, float("-inf")).max(1).values
                        cert[bs] &= best >= bound + 0.05
                    for sg in SIGMAS:
                        sb = (a1 + sg * sig[:, lo:hi]).scatter(1, idx, float("-inf")).max(1).values
                        okm = best >= sb
                        if lo == 0:
                            stat_tmp = {}
                        stat_tmp[sg] = stat_tmp.get(sg, torch.ones_like(okm)) & okm
                    in_shard = (top5 >= lo) & (top5 < hi)
                    rec |= in_shard & (idx == (top5 - lo)[:, None]).any(1)
                r = st_[(name, k)]
                for sg in SIGMAS:
                    stat[(name, k)][sg] += int(stat_tmp[sg].sum())
                r["n"] += len(pos)
                r["recall"] += int(rec.sum())
                for bs in BLOCKS:
                    r["cert"][bs] += int(cert[bs].sum())
                d = r["dom"].setdefault(dom, [0, 0])
                d[0] += len(pos)
                d[1] += int(cert[min(BLOCKS)].sum())
    res = {}
    for (name, k), r in st_.items():
        n = max(r["n"], 1)
        res[f"{name}@{k}"] = {"positions": r["n"], "recall": r["recall"] / n,
                              "certified_by_block": {str(bs): r["cert"][bs] / n for bs in BLOCKS},
                              "certified_by_domain": {d: v[1] / max(v[0], 1) for d, v in sorted(r["dom"].items())},
                              "median_x_maxe": float(torch.cat(gaps[name]).median()),
                              "certified_by_sigma": {str(sg): stat[(name, k)][sg] / n for sg in SIGMAS},
                              "max_standardized_error": zmax[name]}
        print(f"head-check {name} k={k}: recall {100 * r['recall'] / n:.3f}%  certified (block 5120 / 128 / 16) "
              + " / ".join(f"{100 * r['cert'][bs] / n:.3f}%" for bs in BLOCKS)
              + f"  (median |x| max e {res[f'{name}@{k}']['median_x_maxe']:.2f}); at 6 / 8 / 10 sigma "
              + " / ".join(f"{100 * stat[(name, k)][sg] / n:.3f}%" for sg in SIGMAS)
              + f", the largest |error| seen {zmax[name]:.2f} sigma", flush=True)
    if H is not None:
        torch.save({"H": H.cpu(), "rows": n_h}, hessian_path)
        print(f"head-check: X^T X of the head's input over {n_h:,} positions -> {hessian_path}", flush=True)
    with open(os.path.join(out_dir, "head_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def enable_act_qdq(ckpt_dir: str, scope: str = "all"):
    """--act-qdq: the reference's act_quant round trip (block fp8_block_size, ue8m0 scales) before every overlaid linear
    whose official weight was FP8 / FP4 (overlay.py marks them `_act_qdq`) and inside our EXL3 experts (dsv41q.exl3)."""
    from dsv41q import exl3 as exl3_mod
    from dsv41q.refmodel import load_reference
    model, kernel, _ = load_reference(ckpt_dir)
    bs, fmt, sdt = model.fp8_block_size, model.scale_fmt, model.scale_dtype

    def qdq(x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.bfloat16).contiguous().clone()
        kernel.act_quant(y, bs, fmt, sdt, True)                # in place: quantize + dequantize back to bf16
        return y

    exl3_mod.ACT_QDQ = qdq
    exl3_mod.ACT_QDQ_H = scope == "all"
    _ACT["qdq"] = qdq
    ref_linear = model.linear

    def linear_qdq(x, weight, bias=None):
        on = _ACT["qdq"] is not None and getattr(weight, "_act_qdq", False) and (
            scope == "all" or getattr(weight, "_act_qdq_input", False))
        return ref_linear(qdq(x) if on else x, weight, bias)

    model.linear = linear_qdq
    print(f"act-qdq on: block {bs}, scale {fmt} ({sdt}), scope {scope}", flush=True)
    return qdq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("panel")
    ap.add_argument("index")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["ref", "quant"], required=True)
    ap.add_argument("--tensors", default="")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--head-mxfp8", action="store_true",
                    help="quant mode: also score the same model with an MXFP8 lm_head (vLLM's own quantizer, what a "
                         "Marlin MXFP8 head would serve), in the same pass")
    ap.add_argument("--exl3-ckpt", default="",
                    help="quant mode: a sharded EXL3 pack (e.g. Mia-AiLab's 2.9 bpw): its routed experts AND every dense "
                         "linear / head it stores as EXL3 replace the official ones (dsv41q.overlay)")
    ap.add_argument("--max-layers", type=int, default=0, help="debug: run only the first N layers (smoke tests)")
    ap.add_argument("--dense", default="",
                    help="quant mode: EXL3 dense linears from our dense pass (same layout), on top of --tensors experts")
    ap.add_argument("--act-qdq", action="store_true",
                    help="quant mode: our EXL3 linears and experts take the reference's activation round trip (act_quant: "
                         "FP8 with a power-of-two scale per 32 values) wherever the reference's linear() quantizes the "
                         "activation of the FP8 / FP4 weight they replace, as the official kernels compute")
    ap.add_argument("--act-qdq-scope", choices=["all", "inputs"], default="all",
                    help="with --act-qdq: every quantized GEMM input (all), or not the experts' intermediate (inputs)")
    ap.add_argument("--act-qdq-layers", default="",
                    help="with --act-qdq: only layers LO-HI (inclusive) take the round trip, e.g. 0-9")
    ap.add_argument("--hc-bf16", action="store_true",
                    help="every layer's mHC mixing weights (hc_attn_fn / hc_ffn_fn, fp32 [24, 4 x hidden]) rounded to "
                         "bf16 first: what a bf16 copy of them would cost in quality (they stream ~170 MB a round a rank)")
    ap.add_argument("--dump-taps", default="",
                    help="a directory: the hc mean of the stream entering the DSpark target layers (37, 38, 39), every "
                         "position of every sequence, as bf16 memmaps taps_L<layer>.bin [tokens, dim] + taps.json "
                         "(sequence offsets): the drafter's inputs, for training it")
    ap.add_argument("--dump-top", type=int, default=0,
                    help="quant mode: also save each scored position's top-N token ids + logprobs (quant_top.safetensors, "
                         "same order as ref.safetensors), for comparing this harness with a served path position by position")
    ap.add_argument("--head-check", default="",
                    help="quant mode: name=SRC[:PREFIX],... stage-1 heads (a pack dir or an EXL3 .safetensors; PREFIX "
                         "head. or mtp.head.): can a cheap head find this pack's head argmax exactly (head_check.json)")
    ap.add_argument("--head-check-k", default="32,64,128", help="candidate counts per vocab shard for --head-check")
    ap.add_argument("--head-hessian", default="", help="with --head-check: save X^T X of the head's input here (.pt)")
    ap.add_argument("--save-kl", action="store_true",
                    help="quant mode: also write kl_positions.safetensors (every scored position's KL, the sequences' "
                         "lengths), for paired comparisons between variants")
    ap.add_argument("--no-ref", action="store_true",
                    help="quant mode with --dump-top: no reference and no KLD, only this model's top-N at every scored "
                         "position (quant_top.safetensors) and its top-1 agreement with the text (labels for training a drafter on a "
                         "panel without a native reference)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load_file(a.panel)
    toks, offs, score = d["tokens"].tolist(), d["offsets"].tolist(), d["score"]
    n = len(offs) - 1 if not a.max_seqs else min(a.max_seqs, len(offs) - 1)
    seqs = [toks[offs[i]: offs[i + 1]] for i in range(n)]
    index = json.load(open(a.index))[:n]
    torch.set_default_dtype(torch.bfloat16)
    t0 = time.time()
    s = Streamer(a.model, seqs)
    limit = s.args.swiglu_limit
    packs = [Exl3Checkpoint(p) for p in (a.exl3_ckpt, a.dense) if p]
    qdq_layers = None
    if a.act_qdq:
        qdq_fn = enable_act_qdq(a.model, a.act_qdq_scope)
        if a.act_qdq_layers:
            lo, hi = (int(v) for v in a.act_qdq_layers.split("-"))
            qdq_layers = (lo, hi)
            print(f"act-qdq on layers {lo}-{hi} only", flush=True)
    if a.dump_taps:
        install_tap_dump(s, a.dump_taps, seqs)
    overlaid = {}
    if a.mode == "quant":
        for pk in packs:                                   # an EXL3 head in the pack or in our dense output
            hw = head_weight(pk)
            if hw is not None:
                s.head_override = hw
                overlaid["head"] = pk.dir
    blk = None
    n_layers = a.max_layers or s.args.n_layers
    for l in range(n_layers):
        t1 = time.time()
        files = sorted(glob.glob(os.path.join(a.tensors, f"layer_{l:02d}.part*.safetensors"))) if a.tensors else []
        exl3_experts = a.mode == "quant" and (files or any(pk.has_experts(l) for pk in packs))
        blk = s.load(l, experts=not exl3_experts)          # native MXFP4 experts only when nothing replaces them
        if a.hc_bf16:
            for name in ("hc_attn_fn", "hc_ffn_fn"):
                w = getattr(blk, name, None)
                if w is not None:
                    w.data.copy_(w.data.to(torch.bfloat16).to(w.dtype))
        if qdq_layers is not None:                         # --act-qdq-layers: the round trip on this layer or not
            set_act_qdq(qdq_fn if qdq_layers[0] <= l <= qdq_layers[1] else None)
        moe = None
        if a.mode == "quant":
            tensors = {}
            for f in files:
                tensors.update(load_file(f, device="cuda"))
                drop_page_cache(f)
            for pk in packs:
                tensors.update(pk.tensors(f"layers.{l}."))
            if any(".experts." in k for k in tensors):
                moe = quant_moe(l, tensors, limit)
            if packs:
                overlaid[l] = overlay_block(blk, l, tensors)
        s.run_layer(blk, moe=moe)
        print(f"layer {l:2d} {'EXL3' if moe else 'native'}"
              f"{' + dense ' + ','.join(overlaid[l]) if overlaid.get(l) else ''}: {time.time() - t1:.0f}s", flush=True)
        if l < n_layers - 1:
            del blk
            torch.cuda.empty_cache()
    ref_path = os.path.join(a.out, "ref.safetensors")
    if a.head_check and a.mode == "quant":
        head_check(s, blk, score, offs, seqs, index, a.head_check, [int(k) for k in a.head_check_k.split(",")], a.out,
                   a.head_hessian)
        return
    if a.mode == "ref":
        ids, lps, nll = [], [], []
        for i, lg in s.iter_logits(blk):                    # position t predicts token t+1
            sc = score[offs[i]: offs[i + 1]].bool().to(lg.device)
            pos = sc[1:].nonzero().squeeze(-1)              # predict scored token t+1 from position t
            lp = torch.log_softmax(lg[pos], -1)
            top = torch.topk(lp, TOPK, -1)
            ids.append(top.indices.int().cpu())
            lps.append(top.values.half().cpu())
            tgt = torch.tensor(seqs[i], device=lg.device)[1:][pos]
            nll.append((-lp.gather(-1, tgt[:, None]).squeeze(-1)).float().cpu())
        lens = torch.tensor([x.size(0) for x in ids], dtype=torch.int64)
        save_file({"ids": torch.cat(ids), "logprobs": torch.cat(lps), "nll": torch.cat(nll), "lens": lens}, ref_path)
        print(f"ref saved: {int(lens.sum()):,} scored positions, mean NLL {torch.cat(nll).mean().item():.4f}, "
              f"{time.time() - t0:.0f}s")
        return
    if a.no_ref:
        if not a.dump_top:
            raise SystemExit("--no-ref needs --dump-top")
        ids, lps, lens, agree = [], [], [], []
        for i, lg in s.iter_logits(blk):                    # position t predicts token t+1, as in ref mode
            sc = score[offs[i]: offs[i + 1]].bool().to(lg.device)
            pos = sc[1:].nonzero().squeeze(-1)
            top = torch.topk(torch.log_softmax(lg[pos].float(), -1), a.dump_top, -1)
            ids.append(top.indices.int().cpu())
            lps.append(top.values.float().cpu())
            lens.append(pos.numel())
            tgt = torch.tensor(seqs[i], device=lg.device)[1:][pos]
            agree.append((top.indices[:, 0] == tgt).float().cpu())
        save_file({"ids": torch.cat(ids), "logprobs": torch.cat(lps), "lens": torch.tensor(lens, dtype=torch.int64)},
                  os.path.join(a.out, "quant_top.safetensors"))
        res = {"positions": sum(lens), "top1_vs_text": torch.cat(agree).mean().item(), "exl3_ckpt": a.exl3_ckpt,
               "overlaid": {str(k): v for k, v in overlaid.items()}, "seconds": round(time.time() - t0)}
        with open(os.path.join(a.out, "kld.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))
        return
    ref = load_file(ref_path)
    lens = ref["lens"].tolist()

    def kl_vs_ref(lq, pid, plp):                            # native top-128 support + one tail bucket
        P, Q = plp.exp(), lq.gather(-1, pid).exp()
        p_rest = (1 - P.sum(-1)).clamp_min(1e-12)
        q_rest = (1 - Q.sum(-1)).clamp_min(1e-12)
        return (P * (plp - Q.clamp_min(1e-30).log())).sum(-1) + p_rest * (p_rest.log() - q_rest.log())

    heads = None
    if a.head_mxfp8:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (dequant_mxfp8_to_bf16,
                                                                              mxfp8_e4m3_quantize)

        def mxfp8_head(w):
            q, sc = mxfp8_e4m3_quantize(w.to(torch.bfloat16), is_sf_swizzled_layout=False)
            return dequant_mxfp8_to_bf16(q, sc).float()
        heads = {"head_mxfp8": mxfp8_head}
    o, kls, agree, nll_q, by_dom, var = 0, [], [], [], {}, {}
    dump_ids, dump_lps = [], []
    for item in s.iter_logits(blk, heads):
        i, lg, extra = item if heads else (*item, {})
        k = lens[i]
        sc = score[offs[i]: offs[i + 1]].bool().to(lg.device)
        pos = sc[1:].nonzero().squeeze(-1)
        lq = torch.log_softmax(lg[pos].float(), -1)
        pid = ref["ids"][o: o + k].to(lg.device).long()
        plp = ref["logprobs"][o: o + k].to(lg.device).float()
        o += k
        kl = kl_vs_ref(lq, pid, plp)
        kls.append(kl.cpu())
        for name, lgx in extra.items():
            lqx = torch.log_softmax(lgx[pos].float(), -1)
            v = var.setdefault(name, {"kl": [], "agree": [], "kl_main_vs_variant": []})
            v["kl"].append(kl_vs_ref(lqx, pid, plp).cpu())
            v["agree"].append((lqx.argmax(-1) == pid[:, 0]).float().cpu())
            v["kl_main_vs_variant"].append((lq.exp() * (lq - lqx)).sum(-1).cpu())
        agree.append((lq.argmax(-1) == pid[:, 0]).float().cpu())
        if a.dump_top:
            top = torch.topk(lq, a.dump_top, -1)
            dump_ids.append(top.indices.int().cpu())
            dump_lps.append(top.values.float().cpu())
        tgt = torch.tensor(seqs[i], device=lg.device)[1:][pos]
        nll_q.append((-lq.gather(-1, tgt[:, None]).squeeze(-1)).cpu())
        dom = index[i]["domain"].split("_")[0]
        by_dom.setdefault(dom, []).append((kl.sum().item(), agree[-1].sum().item(), k))
    kl_all, ag_all = torch.cat(kls), torch.cat(agree)
    res = {"positions": int(kl_all.numel()), "kld_mean": kl_all.mean().item(), "kld_p99": kl_all.quantile(0.99).item(),
           "top1_agreement": ag_all.mean().item(), "nll_native": ref["nll"].mean().item(),
           "nll_ours": torch.cat(nll_q).mean().item(), "tensors": a.tensors, "exl3_ckpt": a.exl3_ckpt,
           "dense": a.dense, "overlaid": {str(k): v for k, v in overlaid.items()},
           "by_domain": {k: {"kld": sum(x[0] for x in v) / sum(x[2] for x in v),
                             "top1": sum(x[1] for x in v) / sum(x[2] for x in v),
                             "positions": sum(x[2] for x in v)} for k, v in sorted(by_dom.items())}}
    res["ppl_native"], res["ppl_ours"] = math.exp(res["nll_native"]), math.exp(res["nll_ours"])
    for name, v in var.items():
        res[name] = {"kld_mean": torch.cat(v["kl"]).mean().item(), "top1_agreement": torch.cat(v["agree"]).mean().item(),
                     "kl_main_vs_variant_mean": torch.cat(v["kl_main_vs_variant"]).mean().item()}
    if a.dump_top:
        save_file({"ids": torch.cat(dump_ids), "logprobs": torch.cat(dump_lps), "lens": ref["lens"]},
                  os.path.join(a.out, "quant_top.safetensors"))
    if a.save_kl:
        save_file({"kl": kl_all.float().contiguous(), "lens": ref["lens"].cpu().contiguous()},
                  os.path.join(a.out, "kl_positions.safetensors"))
    with open(os.path.join(a.out, "kld.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
