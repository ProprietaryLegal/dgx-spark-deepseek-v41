# SPDX-License-Identifier: Apache-2.0
"""EXL3 encode of the DSpark drafter's dense linears (mtp.0-2: wq_a + wkv, wq_b, wo_b, shared w1 / w3 / w2, and
mtp.0.main_proj), from the official checkpoint's FP8 tensors and the input Hessians the serve captured
(VLLM_EXL3_HCAP, vllm_exl3_moe/hcap.py: X^T X of each module's real inputs during drafting).

Why: the drafter's dense layers are the last FP8 Marlin linears in a decode step (~2.8 ms of it on GB10, well under
the DRAM ceiling at these row counts); as EXL3 on exl3_mrow they stream ~40% fewer bytes at full speed. The drafter
only proposes tokens (the target verifies every one), so its precision moves acceptance, never the output.

Hessians: column-parallel and replicated modules see the same input on every rank (rank 0's file); row-parallel ones
(wo_b, w2) see half the input channels a rank, so their H is the block diagonal of the two ranks' captures (the
cross-rank blocks are unknown: EXL3 rotates inputs in 128-wide blocks, and each rank's slice keeps its own block).
wo_a stays FP8 (its grouped FP8 path).

  python3 p4_drafter.py /official_ckpt /hcap_dir /out.safetensors [--k 5] [--k-map main_proj=6,attn.wq_b=5]
Writes mtp.<i>.<module>.{trellis,suh,svh,mul1} under the checkpoint's names (the plugin's dense loader convention)
and <out>.json with each matrix's Hessian-weighted relative error.
"""
import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.exl3 import h_data, quant_args, reconstruct  # noqa: E402
from dsv41q.refmodel import _dequant_block_fp8  # noqa: E402

LAYERS = 3
# checkpoint module -> (vLLM module prefix inside the drafter, how the capture saw its input)
VLLM_NAME = {"attn.wq_a": "attn.fused_wqa_wkv", "attn.wkv": "attn.fused_wqa_wkv", "attn.wq_b": "attn.wq_b",
             "attn.wo_b": "attn.wo_b", "ffn.shared_experts.w1": "ffn.shared_experts.gate_up_proj",
             "ffn.shared_experts.w3": "ffn.shared_experts.gate_up_proj",
             "ffn.shared_experts.w2": "ffn.shared_experts.down_proj"}
ROW_PARALLEL = {"attn.wo_b", "ffn.shared_experts.w2"}


def tensor_index(ckpt: str) -> dict:
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def load(ckpt: str, wmap: dict, name: str) -> torch.Tensor:
    with safe_open(os.path.join(ckpt, wmap[name]), "pt", device="cuda") as f:
        return f.get_tensor(name)


def fp32(ckpt, wmap, base) -> torch.Tensor:
    w = load(ckpt, wmap, base + ".weight")
    if w.dtype == torch.float8_e4m3fn:
        return _dequant_block_fp8(w, load(ckpt, wmap, base + ".scale")).float()
    return w.float()


def hessian(hdir: str, prefix_tail: str, row: bool):
    """(H sum, rows) for the module whose captured name ends with prefix_tail."""
    files = sorted(glob.glob(os.path.join(hdir, f"*{prefix_tail}.rank*.pt")))
    if not files:
        raise SystemExit(f"no captured Hessian for {prefix_tail} in {hdir}")
    parts = [torch.load(f, map_location="cuda") for f in files]
    if not row:
        return parts[0]["H"], parts[0]["rows"]
    n = sum(p["H"].shape[0] for p in parts)
    H = torch.zeros((n, n), dtype=torch.float32, device="cuda")
    o = 0
    for p in parts:                      # rank 0 holds the first input channels, rank 1 the next
        k = p["H"].shape[0]
        H[o:o + k, o:o + k] = p["H"] * (parts[0]["rows"] / max(p["rows"], 1))
        o += k
    return H, parts[0]["rows"]


def h_err(w, wq, H):
    d = wq.float() - w
    return float(torch.trace(d @ H @ d.t()) / torch.trace(w @ H @ w.t()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("hcap")
    ap.add_argument("out")
    ap.add_argument("--k", type=float, default=5)
    ap.add_argument("--k-map", default="")
    ap.add_argument("--seed", type=int, default=4711)
    a = ap.parse_args()
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, quantize_exl3_batch
    kmap = {}
    for kv in filter(None, a.k_map.split(",")):
        name, k = kv.split("=")
        kmap[name] = float(k)

    def K(name):
        return int(kmap.get(name, a.k))

    wmap = tensor_index(a.ckpt)
    work = os.path.dirname(os.path.abspath(a.out))
    out, metrics = {}, {}

    def put(ck, t, w, H):
        for piece, v in t.items():
            out[f"{ck}.{piece}"] = v.contiguous().cpu()
        metrics[ck] = {"k": None, "rel_err": h_err(w, reconstruct(t, w.shape[1], w.shape[0]), H)}

    def hd(H, rows, key):
        return h_data(H.clone(), rows, key)

    seed = a.seed
    for i in range(LAYERS):
        base = f"mtp.{i}"
        # wq_a + wkv: one input (the drafter layer's attention input), one K: the plugin serves them as one module
        H, n = hessian(a.hcap, f"layers.{40 + i}.attn.fused_wqa_wkv", False)
        Hm = H / max(n, 1)
        wq_a, wkv = fp32(a.ckpt, wmap, f"{base}.attn.wq_a"), fp32(a.ckpt, wmap, f"{base}.attn.wkv")
        k = K("attn.wq_a")
        Hq = hd(H, n, "wqa_wkv")                  # one H object for the pair: exllamav3's shared-H batch path
        (_, tq), (_, tkv) = quantize_exl3_batch([wq_a.t().contiguous(), wkv.t().contiguous()], [Hq, Hq],
                                                [quant_args(k, seed, work), quant_args(k, seed + 1, work)])
        put(f"{base}.attn.wq_a", tq, wq_a, Hm)
        put(f"{base}.attn.wkv", tkv, wkv, Hm)
        for ck in ("attn.wq_b", "attn.wo_b"):
            H, n = hessian(a.hcap, f"layers.{40 + i}.{VLLM_NAME[ck]}", ck in ROW_PARALLEL)
            w = fp32(a.ckpt, wmap, f"{base}.{ck}")
            _, _, t = quantize_exl3(w.t().contiguous(), hd(H, n, ck), quant_args(K(ck), seed + 2, work),
                                    return_weight_q=False)
            put(f"{base}.{ck}", t, w, H / max(n, 1))
            seed += 1
        H, n = hessian(a.hcap, f"layers.{40 + i}.ffn.shared_experts.gate_up_proj", False)
        w1 = fp32(a.ckpt, wmap, f"{base}.ffn.shared_experts.w1")
        w3 = fp32(a.ckpt, wmap, f"{base}.ffn.shared_experts.w3")
        k = K("ffn.shared_experts.w1")
        H13 = hd(H, n, "w13")
        (_, t1), (_, t3) = quantize_exl3_batch([w1.t().contiguous(), w3.t().contiguous()], [H13, H13],
                                               [quant_args(k, seed + 3, work), quant_args(k, seed + 4, work)])
        put(f"{base}.ffn.shared_experts.w1", t1, w1, H / max(n, 1))
        put(f"{base}.ffn.shared_experts.w3", t3, w3, H / max(n, 1))
        H, n = hessian(a.hcap, f"layers.{40 + i}.ffn.shared_experts.down_proj", True)
        w2 = fp32(a.ckpt, wmap, f"{base}.ffn.shared_experts.w2")
        _, _, t = quantize_exl3(w2.t().contiguous(), hd(H, n, "w2"), quant_args(K("ffn.shared_experts.w2"),
                                                                                 seed + 5, work), return_weight_q=False)
        put(f"{base}.ffn.shared_experts.w2", t, w2, H / max(n, 1))
        seed += 10
        print(f"mtp.{i}: " + ", ".join(f"{k.split('.', 2)[2]} {100 * v['rel_err']:.2f}%"
                                         for k, v in metrics.items() if k.startswith(base)), flush=True)
    if any(k.startswith("mtp.0.main_proj") for k in wmap):
        H, n = hessian(a.hcap, "main_proj", False)
        w = fp32(a.ckpt, wmap, "mtp.0.main_proj")
        _, _, t = quantize_exl3(w.t().contiguous(), hd(H, n, "main_proj"), quant_args(K("main_proj"), seed, work),
                                return_weight_q=False)
        put("mtp.0.main_proj", t, w, H / max(n, 1))
        print(f"main_proj {100 * metrics['mtp.0.main_proj']['rel_err']:.2f}%", flush=True)
    for ck in metrics:
        metrics[ck]["k"] = K(ck.split(".", 2)[2]) if not ck.endswith("main_proj") else K("main_proj")
    save_file(out, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)
    with open(a.out + ".json", "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"wrote {len(out)} tensors to {a.out}")


if __name__ == "__main__":
    main()
