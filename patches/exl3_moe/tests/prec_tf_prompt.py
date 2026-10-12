# SPDX-License-Identifier: Apache-2.0
"""Which prompt path rounds closer to the quantized model's exact math: our deterministic prefill (exllamav3's
reconstruct + GEMM tiers) or TensorFold's grouped kernels (tf_prompt.py)? Both against an fp64 evaluation of the same
EXL3 weights (exllamav3's reconstruct gives W_q exactly in fp16; the rotations, SwiGLU with the limit, the down and the
weighted sum in fp64), on random EXL3 banks at a TP=2 rank's shapes. Prints relative errors; asserts nothing.
    CUDA_VISIBLE_DEVICES=0 python3 prec_tf_prompt.py [rows]
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_kernels_gpu import KS, LIMIT, make_bank, random_experts, routing  # noqa: E402

import vllm_exl3_moe.forward as F  # noqa: E402
from vllm_exl3_moe import kernels as K  # noqa: E402
from vllm_exl3_moe import tf_prompt  # noqa: E402

H128 = None


def blockh(v):
    """v [..., n] times the 128-point Sylvester Hadamard in 128-column blocks, / sqrt(128), fp64."""
    global H128
    if H128 is None:
        h = torch.ones((1, 1), dtype=torch.float64)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        H128 = h.to(v.device) / math.sqrt(128)
    s = v.shape
    return (v.reshape(-1, s[-1] // 128, 128) @ H128).reshape(s)


def wq(trellis, k):
    w = torch.empty((trellis.shape[0] * 16, trellis.shape[1] * 16), dtype=torch.float16, device=trellis.device)
    K.ext().reconstruct(w, trellis, float(k), False, True)
    return w.double()


def lin64(x, t):
    trellis, suh, svh, k = t
    return blockh(blockh(x * suh.double()) @ wq(trellis, k)) * svh.double()


def exact(b, x, w, ids, rows):
    bank, perm = b["bank"], b["perm"]
    out = torch.zeros((len(rows), x.shape[1]), dtype=torch.float64, device=x.device)
    for i, r in enumerate(rows):
        xr = x[r:r + 1].double()
        for j in range(ids.shape[1]):
            e = int(ids[r, j])
            if e < 0:
                continue
            w1, w3, w2 = bank.views(int(perm[e]))
            g, u = lin64(xr, w1), lin64(xr, w3)
            a = torch.nn.functional.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT)
            out[i] += float(w[r, j]) * lin64(a, w2)[0]
    return out


def rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    b = make_bank(random_experts(KS, seed=3), KS, tp_size=2, tp_rank=0)
    x = torch.randn((m, 5120), device="cuda", dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=61)
    F._DET_PREFILL, F._TF_PROMPT = True, False
    ours = F.routed_moe_forward(x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"], LIMIT, False,
                                True).float()
    tf = tf_prompt.routed(x, w, ids, b["perm"], b["bank"], LIMIT, False, True)
    rows = list(range(0, m, max(1, m // 32)))[:32]
    ref = exact(b, x, w, ids, rows)
    print(f"m={m}, {len(rows)} rows vs fp64: ours (bf16 out) {rel(ours[rows], ref):.3e}, "
          f"TensorFold (fp32 out) {rel(tf[rows], ref):.3e}, TensorFold rounded to bf16 {rel(tf[rows].bfloat16(), ref):.3e}")


if __name__ == "__main__":
    main()
