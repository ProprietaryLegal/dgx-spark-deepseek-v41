# SPDX-License-Identifier: Apache-2.0
"""exl3_smoe's fused launch (SmoeLayer.fused: router GEMV + DeepSeek-V4 top-k + routed + shared experts in one
persistent launch) against the pair of launches it replaces: exl3_router.router_topk, then
SmoeLayer(..., shared). They run the same arithmetic in the same order, so routing ids, routing weights and the output
must be bit-identical: 1..8 tokens, padded rows, with and without the shared expert, image-sentinel tokens on
bias_vl, and CUDA-graph replays on new inputs (the launch zeroes its own sync words for the next one). A TP=2 rank-0
cut of 32 routed experts (K 2..5) plus a K5 shared expert, a 32-expert router.
    EXL3_KERNEL_ARCH=120 python3 -m pytest -q test_smoe_fused_gpu.py
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_kernels_gpu import DEV, H, LIMIT, TOPK, make_bank, random_experts  # noqa: E402
from test_smoe_shared_gpu import cut  # noqa: E402

N = 32                          # router experts = routed experts in the bank
KS = [2, 3, 4, 5] * 8
SCALE = 1.5
SENT = 128_000


@pytest.fixture(scope="module")
def setup():
    from vllm_exl3_moe.jit import load
    from vllm_exl3_moe.smoe import SmoeLayer

    experts = random_experts(KS + [5], seed=21)
    b = make_bank(experts[:-1], KS, 2, 0)
    sl = SmoeLayer(b["bank"], TOPK)
    sl.attach_shared(*cut(experts[-1]))
    g = torch.Generator().manual_seed(3)
    gw = (torch.randn(N, H, generator=g) * 0.02).to(DEV, torch.bfloat16)
    bias = (torch.randn(N, generator=g) * 0.05).to(DEV)
    bias_vl = (torch.randn(N, generator=g) * 0.05).to(DEV)
    return dict(b=b, sl=sl, gw=gw, bias=bias, bias_vl=bias_vl, rx=load("exl3_router", "EXL3_ROUTER_FLAGS"))


def reference(st, x, pad, shared, vl=False, tok=None):
    m = x.shape[0]
    logits = torch.empty(m, N, dtype=torch.float32, device=DEV)
    tw = torch.empty(m, TOPK, dtype=torch.float32, device=DEV)
    ti = torch.empty(m, TOPK, dtype=torch.int32, device=DEV)
    counter = torch.zeros(1, dtype=torch.int32, device=DEV)
    ws = torch.empty(2 * m * N, dtype=torch.float32, device=DEV)
    st["rx"].router_topk(x, st["gw"], logits, st["bias"], st["bias_vl"] if vl else torch.empty(0, device=DEV),
                         tok if vl else torch.empty(0, dtype=torch.int64, device=DEV), SENT, SCALE, tw, ti, counter, ws)
    out = st["sl"](x, ti, tw, st["b"]["perm"], LIMIT, pad, shared=shared)
    return out, ti, tw


def fused(st, x, pad, shared, vl=False, tok=None):
    return st["sl"].fused(x, st["gw"], st["bias"], st["b"]["perm"], LIMIT, SCALE, pad, shared,
                          st["bias_vl"] if vl else None, tok if vl else None, SENT if vl else 0)


def check(a, b, tag):
    (oa, ia, wa), (ob, ib, wb) = a, b
    assert torch.equal(ia, ib), f"{tag}: ids {ia.tolist()} vs {ib.tolist()}"
    assert torch.equal(wa, wb), f"{tag}: weights differ by {(wa - wb).abs().max().item()}"
    assert torch.equal(oa, ob), f"{tag}: output differs by {(oa.float() - ob.float()).abs().max().item()}"


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
def test_fused_equals_router_then_smoe(setup, m, shared):
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    for pad_rows in ([], [m - 1] if m > 1 else [], list(range(1, m))):
        pad = torch.zeros(m, dtype=torch.bool, device=DEV)
        pad[pad_rows] = True
        ref = reference(setup, x, pad, shared)
        got = fused(setup, x, pad, shared)
        check(got, (ref[0], ref[1], ref[2]), f"m={m} shared={shared} pad={pad_rows}")
        for t in pad_rows:
            assert torch.count_nonzero(got[0][t]).item() == 0, f"padded row {t} not zero"


def test_image_sentinel_rows(setup):
    m = 6
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    tok = torch.tensor([5, SENT, SENT + 4, 77, SENT + 5, SENT + 2], dtype=torch.int64, device=DEV)
    check(fused(setup, x, None, True, True, tok), reference(setup, x, None, True, True, tok), "bias_vl")


def test_graph_replays(setup):
    m = 6
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    pad = torch.zeros(m, dtype=torch.bool, device=DEV)
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    fused(setup, x, pad, True)                                  # warm-up (builds, sizes the grid)
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s):
        out, ids, w = fused(setup, x, pad, True)
    for rep in range(6):
        x.copy_(torch.randn((m, H), device=DEV, dtype=torch.bfloat16))
        pad.zero_()
        if rep % 2:
            pad[rep % m] = True
        g.replay()
        torch.cuda.synchronize()
        ref = reference(setup, x, pad, True)
        check((out, ids, w), ref, f"replay {rep}")
