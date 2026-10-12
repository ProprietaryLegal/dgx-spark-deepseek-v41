# SPDX-License-Identifier: Apache-2.0
"""exl3_router vs what vLLM runs for the V4.1 gate on family-120 Blackwell (cuBLAS bf16 x bf16 -> fp32 via torch.mm's
out_dtype, GateLinear tier 4): per call device time over LAYERS distinct [N, K] gates in one CUDA graph (their bytes
exceed L2, so the weights stream from DRAM as in a decode step), at each row count.
    python3 bench_router.py [layers=48] [rows=1,2,4,6,8] [N=384] [K=5120]
"""
import sys

import torch

from vllm_exl3_moe.jit import load

DEV = "cuda"


def graph_us(fn, calls, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / reps / calls


def main():
    layers = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    rows = [int(r) for r in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,6,8").split(",")]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 384
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 5120
    ext = load("exl3_router", "EXL3_ROUTER_FLAGS")
    ws = [(torch.randn((N, K), device=DEV) * 0.02).to(torch.bfloat16) for _ in range(layers)]
    bias = torch.randn(N, device=DEV) * 0.05
    mib = N * K * 2 / 2**20
    print(f"{torch.cuda.get_device_name(0)}: {layers} gates [{N}, {K}] bf16 ({mib:.2f} MiB each, {layers * mib:.0f} MiB a pass)")
    print(f"{'rows':>4} {'cuBLAS mm us':>13} {'ours logits us':>15} {'ours +topk us':>14} {'ours GB/s':>10} {'speedup':>8}")
    for m in rows:
        x = torch.randn((m, K), device=DEV).to(torch.bfloat16)
        outs = [torch.empty((m, N), device=DEV, dtype=torch.float32) for _ in range(layers)]
        tw = torch.empty((m, 6), device=DEV, dtype=torch.float32)
        ti = torch.empty((m, 6), device=DEV, dtype=torch.int32)
        cnt = [torch.zeros(1, device=DEV, dtype=torch.int32) for _ in range(layers)]
        ef = torch.empty(0, device=DEV, dtype=torch.float32)
        el = torch.empty(0, device=DEV, dtype=torch.int64)
        ws_ = torch.empty(2 * m * N, device=DEV, dtype=torch.float32)

        def mm():
            for w, o in zip(ws, outs):
                torch.mm(x, w.t(), out_dtype=torch.float32, out=o)

        def ours():
            for w, o in zip(ws, outs):
                ext.router_logits(x, w, o)

        def ours_topk():
            for w, o, c in zip(ws, outs, cnt):
                ext.router_topk(x, w, o, bias, ef, el, 0, 1.0, tw, ti, c, ws_)

        a, b, c = graph_us(mm, layers), graph_us(ours, layers), graph_us(ours_topk, layers)
        print(f"{m:4d} {a:13.2f} {b:15.2f} {c:14.2f} {N * K * 2 / b / 1e3:10.0f} {a / b:7.2f}x")


if __name__ == "__main__":
    main()
