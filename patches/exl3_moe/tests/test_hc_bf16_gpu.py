# SPDX-License-Identifier: Apache-2.0
"""VLLM_DSV41_HC_BF16 (hc_bf16.py): the mHC mixing GEMM on a bf16 copy of its weight, tf32_hc_prenorm_gemm's contract.

1. out[s] = x[:, slice s].float() @ bf16(fn)[:, slice s].T and sqrsum[s] = the slice's sum of x^2, for the decode row
   counts (1-8, a 16-row tile) and prompt ones (a 64-row tile), the splits vLLM picks (1, 4, 16) and V4.1's shapes
   (24 x 20480 for the layers, 24 x 5120 for the head's pre-norm); against fp32 math on the bf16 weight (only the
   summation order differs), and the distance to the fp32 weight's result printed for scale;
2. a weight first met inside a CUDA graph capture is not copied there: the original GEMM runs (once warned), and the
   copy is made at the next eager call; shapes outside the contract go to the original.

    python3 -m pytest -q test_hc_bf16_gpu.py
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("vllm")   # hc_bf16 logs through vllm.logger
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_exl3_moe import hc_bf16  # noqa: E402

DEV = "cuda"


def reference(x, fn, splits):
    n, K = x.shape
    w = fn.to(torch.bfloat16).float()
    xs = x.float().view(n, splits, K // splits)
    out = torch.einsum("nsk,osk->sno", xs, w.view(fn.shape[0], splits, K // splits))
    sq = xs.square().sum(-1).t().contiguous()
    return out.contiguous(), sq


@pytest.mark.parametrize("n", [1, 3, 6, 8, 17, 70])
@pytest.mark.parametrize("K,splits", [(20480, 16), (20480, 4), (20480, 1), (5120, 4)])
def test_matches_bf16_weight_math(n, K, splits):
    g = torch.Generator(device=DEV).manual_seed(n * 7 + splits)
    x = (torch.randn((n, K), device=DEV, generator=g) * 3).to(torch.bfloat16)
    fn = torch.randn((24, K), device=DEV, generator=g) * 0.01
    out = torch.full((splits, n, 24), float("nan"), device=DEV)
    sq = torch.full((splits, n), float("nan"), device=DEV)
    hc_bf16._copies.clear()
    hc_bf16.hc_prenorm_gemm(x, fn, out, sq, splits)
    ro, rs = reference(x, fn, splits)
    torch.testing.assert_close(out, ro, rtol=2e-5, atol=1e-3)   # 20480 fp32 products in another order
    torch.testing.assert_close(sq, rs, rtol=2e-5, atol=1e-3)
    exact = (x.float() @ fn.t()).unsqueeze(0)
    rel = ((out.sum(0) - exact[0]).abs().max() / exact.abs().max()).item()
    print(f"n {n} K {K} splits {splits}: max |bf16-weight - fp32-weight| / max |out| = {rel:.2e}")
    assert rel < 1e-2


def test_capture_keeps_the_original_for_a_new_weight(monkeypatch):
    calls = []

    def orig(x, fn, out, sqrsum, num_split):
        calls.append(fn.data_ptr())
        out.zero_()
        sqrsum.zero_()
        return out

    monkeypatch.setattr(hc_bf16, "_orig", orig)
    monkeypatch.setattr(hc_bf16, "_fallbacks", 0)
    hc_bf16._copies.clear()
    x = torch.randn((4, 20480), device=DEV).to(torch.bfloat16)
    fn = torch.randn((24, 20480), device=DEV) * 0.01
    out = torch.empty((16, 4, 24), device=DEV)
    sq = torch.empty((16, 4), device=DEV)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g):
        hc_bf16.hc_prenorm_gemm(x, fn, out, sq, 16)
    torch.cuda.current_stream().wait_stream(s)
    assert calls == [fn.data_ptr()] and not hc_bf16._copies and hc_bf16._fallbacks == 1
    hc_bf16.hc_prenorm_gemm(x, fn, out, sq, 16)                  # eager: the copy is made, the kernel runs
    torch.cuda.synchronize()
    assert len(hc_bf16._copies) == 1 and len(calls) == 1
    torch.testing.assert_close(out, reference(x, fn, 16)[0], rtol=2e-5, atol=2e-4)


def test_outside_the_contract_goes_to_the_original(monkeypatch):
    calls = []
    monkeypatch.setattr(hc_bf16, "_orig", lambda *a: calls.append(1) or a[2])
    x = torch.randn((4, 20480), device=DEV).to(torch.bfloat16)
    fn = torch.randn((24, 20480), device=DEV)
    hc_bf16.hc_prenorm_gemm(x, fn, torch.empty((3, 4, 24), device=DEV), torch.empty((3, 4), device=DEV), 3)  # 20480 % 3
    hc_bf16.hc_prenorm_gemm(x.float(), fn, torch.empty((4, 4, 24), device=DEV), torch.empty((4, 4), device=DEV), 4)
    assert len(calls) == 2
