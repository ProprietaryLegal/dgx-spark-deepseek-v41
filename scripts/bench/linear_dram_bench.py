# SPDX-License-Identifier: Apache-2.0
"""DRAM-bound linear cost at decode sizes: Marlin MXFP8 (what vLLM serves V4.1's dense linears with on SM12x, with
VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel) vs exllamav3 EXL3 at K = 4 / 5 / 6, at the per-rank TP=2
shapes of V4.1's attention projections and shared expert.

linear_decode_bench.py replays ONE weight, so the weight sits in L2 and every kernel looks faster than it is in the model
(the dummy-weight TP=2 profile showed Marlin at ~180-200 GB/s, 2-4x the microbenchmark's per-call times). Here each
case cycles through enough distinct weights (>= 256 MB) inside one CUDA graph that every call reads from DRAM.
Random weights: the kernels decode any bit pattern, this is a pure speed test.
  python3 linear_dram_bench.py [m_list]
"""
import sys
import time

import torch

dev = torch.device("cuda")
SHAPES = {  # name: (in, out) per rank at TP=2
    "wqa+wkv (repl)": (5120, 1792),        # fused_wqa_wkv, both ranks (EXL3 serves it as two matrices, 1280 + 512)
    "wq_b (col)": (1280, 16384),
    "wo_b (row)": (4096, 5120),
    "wo_a group": (4096, 1024),
    "shared gate|up (col)": (5120, 2304),
    "shared down (row)": (1152, 5120),
    "head (vocab shard)": (5120, 64640),   # 129,280 / 2 rows per rank; ~2 calls per DSpark step
}
MIN_BYTES = 256 << 20


def graph_time(fns, reps=30):
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / len(fns) * 1e6      # us per call


def marlin_fns(k, n, m):
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
    count = max(2, -(-MIN_BYTES // (n * k)))
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    fns = []
    for _ in range(count):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter((torch.randn(n, k, device=dev) * 0.5).to(torch.float8_e4m3fn),
                                          requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.randint(118, 126, (n, k // 32), device=dev, dtype=torch.uint8),
                                                requires_grad=False)
        layer.output_size_per_partition, layer.input_size_per_partition = n, k
        kern = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
        kern.process_weights_after_loading(layer)
        fns.append(lambda kern=kern, layer=layer: kern.apply_weights(layer, x))
    return fns, n * k * (1 + 1 / 32)


def cutlass_fns(k, n, m):
    """FlashInfer CUTLASS mm_mxfp8: vLLM's default MXFP8 linear on SM12x (activations quantized to MXFP8 per call)."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale
    from vllm.utils import flashinfer as vfi
    count = max(2, -(-MIN_BYTES // (n * k)))
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    fns = []
    for _ in range(count):
        w = (torch.randn(n, k, device=dev) * 0.5).to(torch.float8_e4m3fn)
        s = torch.randint(118, 126, (n, k // 32), device=dev, dtype=torch.uint8)
        sw = swizzle_mxfp8_scale(s, M=n, K=k)

        def f(w=w, sw=sw):
            q, qs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            vfi.mm_mxfp8(q, w.t(), qs, sw, out_dtype=torch.bfloat16, backend="cutlass")
        fns.append(f)
    return fns, n * k * (1 + 1 / 32)


def exl3_fns(k, n, m, K):
    from exllamav3.modules.quant.exl3 import LinearEXL3
    count = max(2, -(-MIN_BYTES // int(n * k * K / 8)))
    x = torch.randn(m, k, device=dev, dtype=torch.float16)
    mul1 = torch.tensor(0x83DCD12D - 2 ** 32, dtype=torch.int32, device=dev)
    fns = []
    for _ in range(count):
        trellis = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev)
        suh = ((torch.randint(0, 2, (k,), device=dev) * 2 - 1) * 0.05).half()
        svh = ((torch.randint(0, 2, (n,), device=dev) * 2 - 1) * 0.05).half()
        lin = LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis, mul1=mul1)
        fns.append(lambda lin=lin: lin.forward(x, {}))
    return fns, n * k * K / 8


MS = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1,6,8").split(",")]
print(f"{'shape':22s} {'m':>3s} | {'CUTLASS FP8':>13s} | {'Marlin FP8':>13s} | " + " | ".join(f"{'EXL3 K' + str(K):>13s}" for K in (4, 5, 6))
      + "   (us per call, GB/s, distinct weights cycled from DRAM)")
for name, (k, n) in SHAPES.items():
    for m in MS:
        cells = []
        fns, b = cutlass_fns(k, n, m)
        t = graph_time(fns)
        cells.append(f"{t:7.1f} {b / t / 1e3:4.0f}")
        del fns
        fns, b = marlin_fns(k, n, m)
        t = graph_time(fns)
        cells.append(f"{t:7.1f} {b / t / 1e3:4.0f}")
        del fns
        for K in (4, 5, 6):
            fns, b = exl3_fns(k, n, m, K)
            t = graph_time(fns)
            cells.append(f"{t:7.1f} {b / t / 1e3:4.0f}")
            del fns
        torch.cuda.empty_cache()
        print(f"{name:22s} {m:3d} | " + " | ".join(cells), flush=True)
