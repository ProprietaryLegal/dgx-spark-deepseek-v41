# SPDX-License-Identifier: Apache-2.0
"""Decode-size linear cost on this GPU: FP8 (FlashInfer CUTLASS mm_mxfp8, what vLLM runs for V4.1's MXFP8 linears on
SM12x) vs EXL3 at K = 3 / 4 / 5 (exllamav3's fused small-batch path, LinearEXL3 -> BC_LinearEXL3.run_alloc), at the
per-rank (TP=2) shapes of DeepSeek-V4.1's attention projections and shared expert. CUDA-graph replays, us per call.
Random weights: kernels decode any bit pattern, so this is a pure speed test.
  python3 linear_decode_bench.py
"""
import time

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale
from vllm.utils import flashinfer as vfi

dev = torch.device("cuda")
SHAPES = {  # name: (in, out) per rank at TP=2
    "wq_b (col)": (1280, 16384),
    "wo_b (row)": (4096, 5120),
    "wo_a group x4": (4096, 1024),
    "shared gate|up (col)": (5120, 2304),
    "shared down (row)": (1152, 5120),
}


def graph_time(fn, reps=300):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


def fp8_case(k, n, m):
    w = (torch.randn(n, k, device=dev) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(118, 126, (n, k // 32), device=dev, dtype=torch.uint8)   # vLLM MXFP8 scales are raw uint8 exponents
    sw = swizzle_mxfp8_scale(s, M=n, K=k)
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)

    def f():
        q, qs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        vfi.mm_mxfp8(q, w.t(), qs, sw, out_dtype=torch.bfloat16, backend="cutlass")
    return graph_time(f), n * k                                   # bytes ~ 1 per weight


def marlin_case(k, n, m):
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter((torch.randn(n, k, device=dev) * 0.5).to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.randint(118, 126, (n, k // 32), device=dev, dtype=torch.uint8),
                                            requires_grad=False)
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    kern = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kern.process_weights_after_loading(layer)
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)

    def f():
        kern.apply_weights(layer, x)
    return graph_time(f), n * k


def bf16_case(k, n, m):
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)

    def f():
        torch.nn.functional.linear(x, w)
    return graph_time(f), 2 * n * k


def exl3_case(k, n, m, K):
    from exllamav3.modules.quant.exl3 import LinearEXL3
    trellis = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev)
    suh = ((torch.randint(0, 2, (k,), device=dev) * 2 - 1) * 0.05).half()
    svh = ((torch.randint(0, 2, (n,), device=dev) * 2 - 1) * 0.05).half()
    mul1 = torch.tensor(0x83DCD12D - 2 ** 32, dtype=torch.int32, device=dev)
    lin = LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis, mul1=mul1)
    x = torch.randn(m, k, device=dev, dtype=torch.float16)

    def f():
        lin.forward(x, {})
    return graph_time(f), n * k * K / 8


try:
    import b12x  # noqa: F401
    print("b12x importable:", getattr(b12x, "__version__", "?"))
except Exception as e:  # noqa: BLE001
    print("b12x not available:", type(e).__name__)
print(f"{'shape':24s} {'m':>2s} | {'FP8 cutlass':>11s} | {'FP8 marlin':>11s} | {'BF16 linear':>11s} | "
      + " | ".join(f"EXL3 K{K}" for K in (3, 4, 5)) + "   (us / GB/s)")
import sys
MS = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1,6,8").split(",")]
for name, (k, n) in SHAPES.items():
    for m in MS:
        tf, bf = fp8_case(k, n, m)
        tm, bm = marlin_case(k, n, m)
        tb, bb = bf16_case(k, n, m)
        row = (f"{name:24s} {m:4d} | {tf:6.1f} {bf / tf / 1e3:4.0f} | {tm:6.1f} {bm / tm / 1e3:4.0f} | "
               f"{tb:6.1f} {bb / tb / 1e3:4.0f} | ")
        cells = []
        for K in (3, 4, 5):
            try:
                te, be = exl3_case(k, n, m, K)
                cells.append(f"{te:9.1f} {be / te / 1e3:5.0f}")
            except Exception as e:  # noqa: BLE001
                cells.append(f"err {type(e).__name__}")
        print(row + " | ".join(cells), flush=True)
