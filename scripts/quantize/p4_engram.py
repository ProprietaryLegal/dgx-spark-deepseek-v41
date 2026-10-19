# SPDX-License-Identifier: Apache-2.0
"""EXL3 encode of DeepSeek-V4.1's Engram projections (layers.<1|14>.engram.wkv: 24 hash columns x 256 -> 5120 x 5, the
last big FP8 GEMMs of a decode step: 78.6 MB a rank each on Marlin, ~386 us a call on GB10, measured Oct 11), from the
official checkpoint's FP8 tensors and the input Hessians a serve captured (VLLM_EXL3_HCAP with
VLLM_EXL3_HCAP_MATCH='engram\\.wkv$', scripts/quantize/hcap_engram.sh). Column-parallel: every rank sees the whole
input, so rank 0's capture is the module's H.

  python3 p4_engram.py /official_ckpt /hcap_dir /out.safetensors [--k 4]
Writes layers.<i>.engram.wkv.{trellis,suh,svh,mul1} (the plugin's dense loader convention) and <out>.json with each
matrix's Hessian-weighted relative error; the variant dir comes from p4_drafter_variant.py (same tensor-swap rules).
"""
import argparse
import json
import os
import re
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.exl3 import h_data, quant_args, reconstruct  # noqa: E402
from p4_drafter import fp32, h_err, hessian, tensor_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("hcap")
    ap.add_argument("out")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5150)
    a = ap.parse_args()
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    wmap = tensor_index(a.ckpt)
    layers = sorted({int(m.group(1)) for n in wmap if (m := re.match(r"^layers\.(\d+)\.engram\.wkv\.weight$", n))})
    if not layers:
        raise SystemExit("no layers.<i>.engram.wkv.weight in the checkpoint")
    work = os.path.dirname(os.path.abspath(a.out))
    out, metrics = {}, {}
    for i, l in enumerate(layers):
        ck = f"layers.{l}.engram.wkv"
        H, n = hessian(a.hcap, f"layers.{l}.engram.wkv", False)
        w = fp32(a.ckpt, wmap, ck)                                       # [out, in]
        _, _, t = quantize_exl3(w.t().contiguous(), h_data(H.clone(), n, ck), quant_args(a.k, a.seed + i, work),
                                return_weight_q=False)
        for piece, v in t.items():
            out[f"{ck}.{piece}"] = v.contiguous().cpu()
        metrics[ck] = {"k": a.k, "rows": int(n),
                       "rel_err": h_err(w, reconstruct(t, w.shape[1], w.shape[0]), H / max(n, 1))}
        print(f"{ck}: K{a.k}, {n} rows of H, relative error {100 * metrics[ck]['rel_err']:.3f}%", flush=True)
    save_file(out, a.out + ".tmp")
    os.chmod(a.out + ".tmp", 0o644)          # readable by the non-root rsync to the other box (it was 600, Oct 11)
    os.replace(a.out + ".tmp", a.out)
    with open(a.out + ".json", "w") as f:
        json.dump(metrics, f, indent=1)
    os.chmod(a.out + ".json", 0o644)
    print(f"wrote {len(out)} tensors to {a.out}")


if __name__ == "__main__":
    main()
