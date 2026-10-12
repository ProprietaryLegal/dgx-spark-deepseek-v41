# SPDX-License-Identifier: Apache-2.0
"""exl3_mhc post_collapse (mHC post + carried-pre collapse + RMSNorm, one launch) against vLLM's TileLang math in
torch (vllm_exl3_moe.mhc.reference): same fp32 FMA chain and bf16 rounding points, so residual_out and layer_input
agree but for a handful of near-tie elements one bf16 ulp apart (the reference's fp64 FMA emulation double-rounds;
layer_input's sum of squares is also added in another order).
Also: graph capture + replays with new inputs (the per-token counters reset themselves), and a timing line.
    PYTHONPATH=.. python3 -m pytest -q -s test_mhc_gpu.py
"""
import pytest
import torch

torch_ = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from vllm_exl3_moe import mhc  # noqa: E402

dev = torch.device("cuda")
H = 5120


def inputs(n, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn(n, H, device=dev, generator=g) * 3).bfloat16()
    res = (torch.randn(n, 4, H, device=dev, generator=g) * 20).bfloat16()
    comb = torch.rand(n, 4, 4, device=dev, generator=g)
    comb = comb / comb.sum(1, keepdim=True)                    # doubly-stochastic-ish, as Sinkhorn gives
    post = torch.rand(n, 4, device=dev, generator=g) * 2
    pre = torch.rand(n, 4, device=dev, generator=g) + 1e-2
    w = (torch.rand(H, device=dev, generator=g) * 2).bfloat16()
    return x, res, comb, post, pre, w


@pytest.mark.parametrize("n", [1, 4, 6, 8, 16])
def test_matches_tilelang_math(n):
    x, res, comb, post, pre, w = inputs(n, 10 + n)
    r_out, li = mhc.post_collapse(x, res, comb, post, pre, w, 1e-6)
    ref_r, ref_li = mhc.reference(x, res, comb, post, pre, w, 1e-6)
    # same fp32 FMA chain and bf16 rounding; the reference's fp64 emulation of the FMA double-rounds at near-ties, so
    # a few elements in ~10^4 land one bf16 ulp apart
    rd = (r_out.float() - ref_r.float()).abs()
    r_ulp = ref_r.float().abs().clamp_min(1e-30) * 2 ** -7
    assert float((rd / r_ulp).max()) <= 1.0 and int((rd > 0).sum()) <= r_out.numel() // 1000
    diff = (li.float() - ref_li.float()).abs()
    ulp = ref_li.float().abs().clamp_min(1e-30) * 2 ** -7
    flips = int((diff > 0).sum())
    print(f"n={n}: layer_input differs in {flips} of {li.numel()} elements, max {float((diff / ulp).max()):.2f} ulp")
    assert float((diff / ulp).max()) <= 1.0 and flips <= li.numel() // 100


def test_graph_replays_reset_counters():
    n = 6
    x, res, comb, post, pre, w = inputs(n, 99)
    static = [t.clone() for t in (x, res, comb, post, pre)]
    mhc.post_collapse(*static, w, 1e-6)                         # warm-up outside the capture (scratch, JIT)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        out_r, out_li = mhc.post_collapse(*static, w, 1e-6)
    for rep in range(5):
        new = inputs(n, 200 + rep)
        for dst, src in zip(static, new[:5]):
            dst.copy_(src)
        g.replay()
        torch.cuda.synchronize()
        ref_r, ref_li = mhc.reference(*static, w, 1e-6)
        assert int((out_r != ref_r).sum()) <= out_r.numel() // 1000
        assert float((out_li.float() - ref_li.float()).abs().max()) <= float(ref_li.float().abs().max()) * 2 ** -7


def test_timing_line():
    n = 6
    x, res, comb, post, pre, w = inputs(n, 7)
    for _ in range(3):
        mhc.post_collapse(x, res, comb, post, pre, w, 1e-6)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(200):
        mhc.post_collapse(x, res, comb, post, pre, w, 1e-6)
    ev1.record()
    torch.cuda.synchronize()
    print(f"post_collapse n={n}: {ev0.elapsed_time(ev1) / 200 * 1e3:.1f} us a call (eager, launch included)")
