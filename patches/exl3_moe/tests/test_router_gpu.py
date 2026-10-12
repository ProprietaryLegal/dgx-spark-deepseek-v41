# SPDX-License-Identifier: Apache-2.0
"""exl3_router (csrc/exl3_router.cu): the MoE router at decode sizes.

(1) router_logits matches x @ W^T in fp64 to fp32-accumulation error, for 1..8 rows, the V4.1 gate [384, 5120] and the
    drafter's [128, 5120], and is bit-identical run to run (fixed summation order).
(2) router_topk picks exactly the experts DeepSeek-V4's routing picks from the same logits (the math of vLLM's
    dsv4_topk, ported to torch below: sqrt(softplus) + correction bias, lowest index on equal scores, NaN -> -1e30,
    weights renormalized x routed_scaling_factor), including exact ties, NaN logits and image-sentinel rows that route
    with bias_vl; weights to fp32 rounding.
(3) router_topk replays in a CUDA graph (the last block resets the grid counter): same outputs on every replay.

Run on a GPU (sm_120 / sm_121; set EXL3_KERNEL_ARCH to the device's, e.g. 120 on an RTX 50 card):
    python3 -m pytest -q test_router_gpu.py
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = "cuda"
SENT = 128_000                     # image sentinel ids SENT .. SENT + 4 in these tests


@pytest.fixture(scope="module")
def ext():
    from vllm_exl3_moe.jit import load
    return load("exl3_router", "EXL3_ROUTER_FLAGS")


def ref_topk(logits, bias, k, scale, bias_vl=None, ids=None, sent=SENT):
    """DeepSeek-V4 routing on fp32 logits [M, N], as vLLM's _dsv4_topk_kernel (fused_moe/router/dsv4_topk.py)."""
    l = logits.float()
    w = torch.sqrt(torch.where(l > 20.0, l, torch.log(1.0 + torch.exp(l))))
    b = bias.expand_as(w).clone()
    if bias_vl is not None:
        img = (ids >= sent) & (ids < sent + 5)
        b[img] = bias_vl
    cur = w + b
    cur = torch.where(cur == cur, cur, torch.full_like(cur, -1e30))
    M, N = l.shape
    out_i = torch.empty((M, k), dtype=torch.int32, device=l.device)
    out_w = torch.empty((M, k), dtype=torch.float32, device=l.device)
    ar = torch.arange(N, device=l.device)
    for s in range(k):
        mx = cur.max(dim=1, keepdim=True).values
        e = torch.where(cur == mx, ar, torch.full_like(ar, N)).min(dim=1).values
        out_i[:, s] = e.int()
        out_w[:, s] = w.gather(1, e[:, None])[:, 0]
        cur.scatter_(1, e[:, None], float("-inf"))
    out_w = out_w * (scale / torch.where(out_w.sum(1, keepdim=True) > 0, out_w.sum(1, keepdim=True), 1.0))
    return out_w, out_i


def make(M, N, K, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn((M, K), device=DEV, generator=g).to(torch.bfloat16)
    w = (torch.randn((N, K), device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    return x, w


@pytest.mark.parametrize("N,K", [(384, 5120), (128, 5120), (256, 4096)])
@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6, 7, 8])
def test_logits(ext, M, N, K):
    x, w = make(M, N, K, 10 * M + N)
    out = torch.empty((M, N), device=DEV, dtype=torch.float32)
    ext.router_logits(x, w, out)
    ref = x.double() @ w.double().T
    err = (out.double() - ref).abs().max().item()
    assert err < 2e-4 * max(1.0, ref.abs().max().item()), err
    again = torch.empty_like(out)
    ext.router_logits(x, w, again)
    assert torch.equal(out, again), "router_logits not deterministic"


def test_logits_strided_rows(ext):
    """x as a row slice of a wider buffer (row stride > K): vLLM hands the gate views like this."""
    big = torch.randn((6, 5120 + 64), device=DEV).to(torch.bfloat16)
    x = big[:, :5120]
    _, w = make(6, 384, 5120, 3)
    out = torch.empty((6, 384), device=DEV, dtype=torch.float32)
    ext.router_logits(x, w, out)
    ref = x.double() @ w.double().T
    assert (out.double() - ref).abs().max().item() < 2e-4 * ref.abs().max().item()


def run_topk(ext, x, w, bias, k, scale, bias_vl=None, ids=None, counter=None):
    M, N = x.shape[0], w.shape[0]
    logits = torch.empty((M, N), device=DEV, dtype=torch.float32)
    tw = torch.empty((M, k), device=DEV, dtype=torch.float32)
    ti = torch.empty((M, k), device=DEV, dtype=torch.int32)
    counter = torch.zeros(1, device=DEV, dtype=torch.int32) if counter is None else counter
    empty_f = torch.empty(0, device=DEV, dtype=torch.float32)
    empty_l = torch.empty(0, device=DEV, dtype=torch.int64)
    ws = torch.empty(2 * M * N, device=DEV, dtype=torch.float32)
    ext.router_topk(x, w, logits, bias, empty_f if bias_vl is None else bias_vl, empty_l if ids is None else ids,
                    SENT, scale, tw, ti, counter, ws)
    return logits, tw, ti, counter


@pytest.mark.parametrize("M", [1, 3, 6, 8])
@pytest.mark.parametrize("N,k,scale", [(384, 6, 1.0), (384, 6, 2.5), (256, 6, 1.0), (128, 3, 1.0)])
def test_topk_matches_routing(ext, M, N, k, scale):
    x, w = make(M, N, 5120, 100 + M + N + k)
    bias = (torch.randn(N, device=DEV) * 0.05).float()
    logits, tw, ti, counter = run_topk(ext, x, w, bias, k, scale)
    rw, ri = ref_topk(logits, bias, k, scale)
    assert torch.equal(ti, ri), (ti, ri)
    torch.testing.assert_close(tw, rw, rtol=2e-6, atol=1e-7)
    assert counter.item() == 0, "counter not reset"


def test_topk_ties_nan_and_image_rows(ext):
    """Exact ties (lowest index wins), NaN logits (score -1e30) and image-sentinel rows routed with bias_vl. The weight
    matrix is built so the logits carry the cases: rows of x pick out rows of w (one-hot x)."""
    M, N, K = 6, 384, 5120
    x = torch.zeros((M, K), device=DEV, dtype=torch.bfloat16)
    for m in range(M):
        x[m, m] = 1.0                                           # logits[m, e] = w[e, m]
    w = torch.zeros((N, K), device=DEV, dtype=torch.bfloat16)
    g = torch.Generator(device=DEV).manual_seed(7)
    w[:, :M] = (torch.randn((N, M), device=DEV, generator=g) * 2).to(torch.bfloat16)
    w[[10, 20, 300], 0] = 9.0                                   # row 0: a three-way tie at the top
    w[[5, 6], 1] = float("nan")                                 # row 1: NaN logits never win
    w[[33, 34], 2] = 7.0                                        # row 2: tie, bias decides nothing (bias equal)
    bias = (torch.randn(N, device=DEV, generator=g) * 0.05).float()
    bias[[10, 20, 300, 33, 34]] = 0.25
    bias_vl = (torch.randn(N, device=DEV, generator=g) * 0.5).float()
    ids = torch.tensor([1, 2, 3, SENT, SENT + 4, SENT + 5], device=DEV, dtype=torch.int64)   # rows 3, 4 image; 5 not
    logits, tw, ti, _ = run_topk(ext, x, w, bias, 6, 1.0, bias_vl, ids)
    rw, ri = ref_topk(logits, bias, 6, 1.0, bias_vl, ids)
    assert torch.equal(ti, ri), (ti, ri)
    torch.testing.assert_close(tw, rw, rtol=2e-6, atol=1e-7)
    assert ti[0, :3].tolist() == [10, 20, 300]
    assert 5 not in ti[1].tolist() and 6 not in ti[1].tolist()
    assert ti[2, :2].tolist() == [33, 34]
    _, ri_plain = ref_topk(logits, bias, 6, 1.0)
    assert not torch.equal(ri[3], ri_plain[3]), "bias_vl rows should route differently in this draw"


def test_topk_cuda_graph_replay(ext):
    M, N, k = 6, 384, 6
    x, w = make(M, N, 5120, 42)
    bias = (torch.randn(N, device=DEV) * 0.05).float()
    logits = torch.empty((M, N), device=DEV, dtype=torch.float32)
    tw = torch.empty((M, k), device=DEV, dtype=torch.float32)
    ti = torch.empty((M, k), device=DEV, dtype=torch.int32)
    counter = torch.zeros(1, device=DEV, dtype=torch.int32)
    ef = torch.empty(0, device=DEV, dtype=torch.float32)
    el = torch.empty(0, device=DEV, dtype=torch.int64)
    ws = torch.empty(2 * M * N, device=DEV, dtype=torch.float32)
    call = lambda: ext.router_topk(x, w, logits, bias, ef, el, SENT, 1.0, tw, ti, counter, ws)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()                                                  # warm-up / JIT outside the capture
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(4):
            call()
    outs = []
    for r in range(3):
        x.copy_(make(M, N, 5120, 1000 + r)[0])
        graph.replay()
        torch.cuda.synchronize()
        rw, ri = ref_topk(logits, bias, k, 1.0)
        assert torch.equal(ti, ri) and counter.item() == 0
        outs.append(ti.clone())
    assert not torch.equal(outs[0], outs[1]), "inputs changed, routing should too"
