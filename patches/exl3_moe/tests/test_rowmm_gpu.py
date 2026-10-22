# SPDX-License-Identifier: Apache-2.0
"""rowmm (csrc/exl3_router.cu, vllm_exl3_moe/rowmm.py): decode-size bf16 projections for the V4.1 attention inputs.

(1) rowmm matches x @ W^T in fp64 to fp32-accumulation error for 1..8 rows on the shapes patch 0029 sends it (the
    compressor's wkv / wgate [1024, 5120] and wkv alone [512, 5120] -> fp32, the indexer's weights_proj [32, 5120] and
    wk [128, 512] -> bf16), and is bit-identical run to run.
(2) The bf16 output is the fp32 output rounded to nearest even, bit for bit (one kernel, one sum order).
(3) rowmm's fp32 output is router_logits' bit for bit (the router path keeps its numbers).
(4) rowmm.ok refuses what the kernel cannot take, so the caller keeps cuBLAS: 9 rows, fp16, a column view, an
    unaligned row start, K not a multiple of 8, the flag off.
(5) A captured CUDA graph replays it with fresh inputs.

Run on a GPU (sm_120 / sm_121; set EXL3_KERNEL_ARCH to the device's, e.g. 120 on an RTX 50 card):
    python3 -m pytest -q test_rowmm_gpu.py
"""
import importlib
import sys
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = "cuda"
SHAPES = [(1024, 5120, torch.float32), (512, 5120, torch.float32), (32, 5120, torch.bfloat16),
          (128, 512, torch.bfloat16)]


@pytest.fixture(scope="module")
def ext():
    from vllm_exl3_moe.jit import load
    return load("exl3_router", "EXL3_ROUTER_FLAGS")


@pytest.fixture()
def rowmm(monkeypatch, ext):
    """vllm_exl3_moe.rowmm with the flag on, its kernel this test's build, and no vLLM needed for its log line."""
    monkeypatch.setenv("EXL3_ROWMM", "1")
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda name: __import__("logging").getLogger(name)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    router = types.ModuleType("vllm_exl3_moe.router")
    router.ext = lambda: ext
    monkeypatch.setitem(sys.modules, "vllm_exl3_moe.router", router)
    import vllm_exl3_moe.rowmm as m
    m = importlib.reload(m)
    assert m.ENABLED
    return m


def make(M, N, K, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn((M, K), device=DEV, generator=g).to(torch.bfloat16)
    w = (torch.randn((N, K), device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    return x, w


@pytest.mark.parametrize("N,K,dt", SHAPES)
@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6, 7, 8])
def test_matches_fp64(ext, M, N, K, dt):
    x, w = make(M, N, K, 10 * M + N)
    out = torch.empty((M, N), device=DEV, dtype=dt)
    ext.rowmm(x, w, out)
    ref = x.double() @ w.double().T
    tol = 2e-4 if dt == torch.float32 else 8e-3          # bf16 output: half an ulp of 2^-8 relative
    err = (out.double() - ref).abs().max().item()
    assert err < tol * max(1.0, ref.abs().max().item()), err
    again = torch.empty_like(out)
    ext.rowmm(x, w, again)
    assert torch.equal(out, again), "rowmm not deterministic"


@pytest.mark.parametrize("N,K", [(32, 5120), (128, 512), (1024, 5120)])
def test_bf16_is_fp32_rounded(ext, N, K):
    x, w = make(6, N, K, N + K)
    f = torch.empty((6, N), device=DEV, dtype=torch.float32)
    b = torch.empty((6, N), device=DEV, dtype=torch.bfloat16)
    ext.rowmm(x, w, f)
    ext.rowmm(x, w, b)
    assert torch.equal(b, f.to(torch.bfloat16))


def test_fp32_is_router_logits(ext):
    x, w = make(5, 384, 5120, 7)
    a = torch.empty((5, 384), device=DEV, dtype=torch.float32)
    r = torch.empty_like(a)
    ext.rowmm(x, w, a)
    ext.router_logits(x, w, r)
    assert torch.equal(a, r)


def test_module_mm_and_strided_rows(rowmm):
    """rowmm.mm on a row slice of a wider buffer (vLLM hands the projections views like this)."""
    big = torch.randn((4, 5120 + 64), device=DEV).to(torch.bfloat16)
    x = big[:, :5120]
    _, w = make(4, 1024, 5120, 11)
    assert rowmm.ok(x, w)
    out = rowmm.mm(x, w, torch.float32)
    ref = x.double() @ w.double().T
    assert out.dtype == torch.float32 and out.shape == (4, 1024)
    assert (out.double() - ref).abs().max().item() < 2e-4 * max(1.0, ref.abs().max().item())
    assert torch.equal(out, rowmm.mm(x.contiguous(), w, torch.float32))


def test_ok_refuses(rowmm, monkeypatch):
    x, w = make(8, 32, 5120, 1)
    assert rowmm.ok(x, w)
    x9, _ = make(9, 32, 5120, 2)
    assert not rowmm.ok(x9, w)                                       # 9 rows
    assert not rowmm.ok(x.half(), w)                                 # fp16 input
    assert not rowmm.ok(x, w.half())                                 # fp16 weight
    assert not rowmm.ok(x.t().contiguous().t(), w)                   # column-major view
    off = torch.empty(8 * 5120 + 8, device=DEV, dtype=torch.bfloat16)[1:1 + 8 * 5120].view(8, 5120)
    assert not rowmm.ok(off, w)                                      # row start not 16-byte aligned
    x12, w12 = make(4, 32, 5124, 3)
    assert not rowmm.ok(x12, w12)                                    # K not a multiple of 8
    assert not rowmm.ok(x, make(8, 32, 4096, 4)[1])                  # K mismatch
    assert not rowmm.ok(x, w.t().contiguous().t())                   # weight not contiguous
    monkeypatch.setattr(rowmm, "ENABLED", False)
    assert not rowmm.ok(x, w)                                        # flag off


def test_graph_replay(rowmm):
    x, w = make(3, 1024, 5120, 21)
    static_x = x.clone()
    rowmm.mm(static_x, w, torch.float32)                             # build and warm outside capture
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            static_out = rowmm.mm(static_x, w, torch.float32)
    torch.cuda.current_stream().wait_stream(s)
    for seed in (22, 23):
        nx, _ = make(3, 1024, 5120, seed)
        static_x.copy_(nx)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_out, rowmm.mm(nx, w, torch.float32))
