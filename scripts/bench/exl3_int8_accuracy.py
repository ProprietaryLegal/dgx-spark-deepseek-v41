# SPDX-License-Identifier: Apache-2.0
"""Output error of exllamav3's EXL3 linear forward at decode sizes against the exact product with the same (dequantized)
weight, under the activation mode exllamav3 picked from EXL3_INT8_GEMV for this process (read once per process):
  0 = fp16 activations, 1 = int8 + error-feedback residual pass, 2 = plain int8 (the CUDA default, for K <= 6 on Blackwell).
Isolates what the serving path adds on top of the weight error that p4_kld.py measures (the KLD harness runs the
reconstructed weights in BF16, so it cannot see this). Activations: Gaussian, and Gaussian with a few outlier channels
(x50) as real LLM activations have. Shapes: V4.1's dense linears per rank at TP=2 (fused_wqa_wkv is replicated) and the
head shard. Random trellis words: the kernels decode any bit pattern.
  EXL3_INT8_GEMV=2 python3 exl3_int8_accuracy.py [--m 1,6]
"""
import argparse
import os

import torch
from exllamav3.modules.quant.exl3 import LinearEXL3

dev = torch.device("cuda")
MUL1 = 0x83DCD12D - 2 ** 32
SHAPES = {"fused_wqa_wkv: wq_a": (5120, 1280), "fused_wqa_wkv: wkv": (5120, 512), "wq_b (col)": (1280, 16384),
          "wo_b (row)": (4096, 5120), "shared gate|up (col)": (5120, 2304), "shared down (row)": (1152, 5120),
          "head shard": (5120, 64640)}


def linear(in_f, out_f, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return LinearEXL3(None, in_f, out_f,
                      trellis=torch.randint(-32768, 32768, (in_f // 16, out_f // 16, 16 * k), dtype=torch.int16,
                                            device=dev, generator=g),
                      suh=((torch.randint(0, 2, (in_f,), device=dev, generator=g) * 2 - 1) * 0.05).half(),
                      svh=((torch.randint(0, 2, (out_f,), device=dev, generator=g) * 2 - 1) * 0.05).half(),
                      mul1=torch.tensor(MUL1, dtype=torch.int32, device=dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,6")
    ap.add_argument("--k", default="5,6")
    a = ap.parse_args()
    mode = os.environ.get("EXL3_INT8_GEMV", "unset (2 on CUDA)")
    print(f"EXL3_INT8_GEMV={mode}: relative output error vs the exact product with the same weight (fp32)")
    for k in (int(v) for v in a.k.split(",")):
        for name, (in_f, out_f) in SHAPES.items():
            lin = linear(in_f, out_f, k, sum(map(ord, name)))          # same weights in every mode's process
            W = lin.get_weight_tensor().float()                          # [in, out]
            for m in (int(v) for v in a.m.split(",")):
                g = torch.Generator(device=dev).manual_seed(m)
                x = torch.randn(m, in_f, device=dev, generator=g)
                xo = x.clone()
                xo[:, torch.randperm(in_f, generator=torch.Generator().manual_seed(1))[:8].to(dev)] *= 50.0
                errs = []
                for xx in (x, xo):
                    xh = xx.half()
                    y = lin.forward(xh, {}, out_dtype=torch.float32)
                    ref = xh.float() @ W
                    errs.append(((y - ref).norm() / ref.norm()).item())
                print(f"K{k}  {name:22s} m={m}:  gaussian {errs[0]:.2e}   outliers {errs[1]:.2e}")


if __name__ == "__main__":
    main()
