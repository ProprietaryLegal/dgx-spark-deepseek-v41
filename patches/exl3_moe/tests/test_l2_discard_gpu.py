# SPDX-License-Identifier: Apache-2.0
"""EXL3_L2_DISCARD: the builds that drop dead fp32 scratch from L2 (discard.global.L2) instead of letting it be written
back give the bits of the default builds.

exl3_mrow (-DMROW_L2_DISCARD=1, the k-split partials): the serve's per-rank dense shapes at K4 / K5, 1-6 rows (and 8,
16), with GB10's k-split choice (1.5 items x 48 SMs) and a finer one, called back to back with new inputs and replayed
from a CUDA graph. exl3_smoe (-DSMOE_L2_DISCARD=1, the per-slot outputs): 1-6 rows (and 8, 16) with an unrouted pick
and a padded row, with and without the shared-expert slot, the fused launch, the run-flags build the serve uses, graph
replays. Canary runs fill the scratch with NaN and push it to DRAM first: a partial read after its line was dropped
would come back as NaN and show in the output. They also print how much of the summed scratch reads back as the canary
afterwards, i.e. how many lines the drop kept from DRAM (a line written back before its drop keeps its value), and fail
if that is none: the instruction would then save nothing on that GPU (RTX 5090 mobile, sm_120: 100% of the lines).

    python3 -m pytest -q -s test_l2_discard_gpu.py
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_mrow_gpu import rand_linear  # noqa: E402
from vllm_exl3_moe import mrow, smoe  # noqa: E402
from vllm_exl3_moe.mrow import MrowGroup  # noqa: E402

DEV = torch.device("cuda", 0)
GB10_ITEMS = 72                 # mrow.py's want_items on GB10 (48 SMs): the k-splits the serve runs
SHAPES = {                      # name: (k, n) linears of one launch, per rank at TP=2 (test_mrow_gpu.SHAPES)
    "wq_a + wkv": [(5120, 1280), (5120, 512)],
    "wq_b": [(1280, 16384)],                     # one k-slice at GB10's split: nothing to drop, same bits
    "wo_a (4 groups)": [(4096, 1024)] * 4,
    "wo_b": [(4096, 5120)],
    "shared gate + up": [(5120, 1152), (5120, 1152)],
    "shared down": [(1152, 5120)],
}
ROWS = [1, 2, 3, 4, 5, 6, 8, 16]


@pytest.fixture(scope="module")
def flush():
    """Streams 4x the L2 through it: dirty lines (a canary just written) end up in DRAM, nothing of them in L2."""
    buf = torch.empty(4 * torch.cuda.get_device_properties(DEV).L2_cache_size // 4, dtype=torch.float32, device=DEV)

    def run():
        buf.fill_(1.0)
        buf.sum()
        torch.cuda.synchronize()
    return run


def _group(name, K, items, discard, seed):
    lins = SHAPES[name]
    ws = [rand_linear(k, n, K, seed + i) for i, (k, n) in enumerate(lins)]
    xcs = [i * k for i, (k, _) in enumerate(lins)] if name.startswith("wo_a") else [0] * len(lins)
    ocs = [sum(n for _, n in lins[:i]) for i in range(len(lins))]
    g = MrowGroup([(t, su, sv, xc, oc) for (t, su, sv), xc, oc in zip(ws, xcs, ocs)], K, want_items=items,
                  discard=discard)
    return g, max(xc + k for xc, (k, _) in zip(xcs, lins)), sum(n for _, n in lins)


def test_builds_are_distinct(monkeypatch):
    assert mrow.ext(True) is not mrow.ext(False) and smoe.ext(True) is not smoe.ext(False)
    t, su, sv = rand_linear(1024, 512, 4, 1)
    monkeypatch.setenv("EXL3_L2_DISCARD", "1")
    assert MrowGroup([(t, su, sv, 0, 0)], 4).ext is mrow.ext(True)
    monkeypatch.setenv("EXL3_L2_DISCARD", "0")
    assert MrowGroup([(t, su, sv, 0, 0)], 4).ext is mrow.ext(False)


@pytest.mark.parametrize("K", [4, 5])
@pytest.mark.parametrize("name", list(SHAPES))
def test_mrow_discard_bit_identical(name, K):
    for items in (GB10_ITEMS, 4 * GB10_ITEMS):
        off, k_in, n_out = _group(name, K, items, False, 100 * K)
        on, _, _ = _group(name, K, items, True, 100 * K)
        assert on.ks == off.ks
        if name != "wq_b" or items > GB10_ITEMS:
            assert any(k // ks > 1 for (k, _), ks in zip(SHAPES[name], on.ks)), (name, items, on.ks)
        for rep in range(2):                     # back to back with new inputs: dropped lines are written before read
            for m in ROWS:
                x = torch.randn(m, k_in, device=DEV).bfloat16()
                a = off(x, torch.empty(m, n_out, device=DEV, dtype=torch.bfloat16))
                b = on(x, torch.empty(m, n_out, device=DEV, dtype=torch.bfloat16))
                assert torch.equal(a, b), (name, K, items, m, rep, (a.float() - b.float()).abs().max().item())


@pytest.mark.parametrize("m", [1, 4, 6])
def test_mrow_discard_graph_replay(m):
    K = 4
    off, k_in, n_out = _group("wo_a (4 groups)", K, GB10_ITEMS, False, 7)
    on, _, _ = _group("wo_a (4 groups)", K, GB10_ITEMS, True, 7)
    x = torch.randn(m, k_in, device=DEV).bfloat16()
    out = torch.empty(m, n_out, device=DEV, dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        on(x, out)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        on(x, out)
    for _ in range(4):
        x.copy_(torch.randn(m, k_in, device=DEV).bfloat16())
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, off(x, torch.empty_like(out)))


@pytest.mark.parametrize("items", [GB10_ITEMS, 4 * GB10_ITEMS])
def test_mrow_canary(flush, items):
    """wo_b (one problem, so its partials sit at the start of ws: [slices][m][n]) with ws full of NaN in DRAM."""
    K, m, (k, n) = 4, 6, SHAPES["wo_b"][0]
    res = {}
    for d in (False, True):
        g, _, _ = _group("wo_b", K, items, d, 11)
        x = torch.randn(m, k, device=DEV, generator=torch.Generator(device=DEV).manual_seed(5)).bfloat16()
        ws = g.ext.workspace(g.handle)
        used = ws[: (k // g.ks[0]) * m * n]
        assert k // g.ks[0] > 1
        for rep in range(3):
            ws.fill_(float("nan"))
            flush()
            out = g(x, torch.empty(m, n, device=DEV, dtype=torch.bfloat16))
            torch.cuda.synchronize()
            assert not torch.isnan(out.float()).any(), (d, rep)
            res.setdefault(d, out.clone())
            assert torch.equal(out, res[d])
            frac = torch.isnan(used).float().mean().item()
            if not d:
                assert frac == 0.0                # the default build leaves its partials, in L2 or DRAM
        print(f"\nmrow wo_b items={items} slices={k // g.ks[0]} discard={d}: {frac:.0%} of the summed partials read "
              f"back as the DRAM canary after the call")
        assert not d or frac > 0, "discard.global.L2 dropped no line on this GPU: the build saves no DRAM writes here"
    assert torch.equal(res[False], res[True])


# ---- exl3_smoe -------------------------------------------------------------------------------------------------------

N = 32                          # routed experts in the bank = router experts of the fused launch (a multiple of 32)
KS = [2, 3, 4, 5] * 8
SCALE = 1.5


def _smoe_pair(st, flags=None):
    """(default build, discard build) SmoeLayers of the fixture's bank, both with the shared slot attached. flags: an
    EXL3_SMOE_FLAGS value both builds are made with (their own module cache, restored after)."""
    from test_smoe_shared_gpu import cut
    from vllm_exl3_moe.smoe import SmoeLayer
    saved_env, saved = os.environ.get("EXL3_SMOE_FLAGS"), smoe._ext
    try:
        if flags is not None:
            os.environ["EXL3_SMOE_FLAGS"] = flags
            smoe._ext = {}
        pair = {}
        for d in (False, True):
            pair[d] = SmoeLayer(st["b"]["bank"], st["topk"], discard=d)
            pair[d].attach_shared(*cut(st["experts"][-1]))
        return pair
    finally:
        if flags is not None:
            smoe._ext = saved
            if saved_env is None:
                os.environ.pop("EXL3_SMOE_FLAGS", None)
            else:
                os.environ["EXL3_SMOE_FLAGS"] = saved_env


@pytest.fixture(scope="module")
def st():
    from test_kernels_gpu import TOPK, make_bank, random_experts
    experts = random_experts(KS + [5], seed=41)
    s = dict(experts=experts, b=make_bank(experts[:-1], KS, 2, 0), topk=TOPK)   # a TP=2 rank-0 cut, as served
    g = torch.Generator().manual_seed(5)
    s["gw"] = (torch.randn(N, 5120, generator=g) * 0.02).to(DEV, torch.bfloat16)
    s["bias"] = (torch.randn(N, generator=g) * 0.05).to(DEV)
    s["sl"] = _smoe_pair(s)
    return s


def _inputs(m, seed):
    from test_kernels_gpu import H, routing
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, N, seed=seed)
    if m >= 3:
        ids[1, 2] = -1                                       # an unrouted pick
    pad = torch.zeros(m, dtype=torch.bool, device=DEV)
    if m >= 4:
        pad[m - 1] = True                                    # a padded row: no routed pick, no shared pick
    return x, w, ids, pad


def _run(sl, st, x, w, ids, pad, shared):
    from test_kernels_gpu import LIMIT
    return sl(x, ids, w, st["b"]["perm"], LIMIT, pad, shared=shared)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("m", ROWS)
def test_smoe_discard_bit_identical(st, m, shared):
    for rep in range(2):
        x, w, ids, pad = _inputs(m, 1000 + 10 * m + rep)
        a = _run(st["sl"][False], st, x, w, ids, pad, shared)
        b = _run(st["sl"][True], st, x, w, ids, pad, shared)
        assert torch.equal(a, b), (m, shared, rep, (a.float() - b.float()).abs().max().item())
        if m >= 4:
            assert torch.count_nonzero(b[m - 1]).item() == 0


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 8])
def test_smoe_fused_discard_bit_identical(st, m):
    from test_kernels_gpu import LIMIT
    x, _, _, pad = _inputs(m, 2000 + m)
    for shared in (False, True):
        res = []
        for d in (False, True):
            out, ids, w = st["sl"][d].fused(x, st["gw"], st["bias"], st["b"]["perm"], LIMIT, SCALE, pad, shared)
            res.append((out.clone(), ids.clone(), w.clone()))  # ids / w are views of the shared routing scratch
        (oa, ia, wa), (ob, ib, wb) = res
        assert torch.equal(ia, ib) and torch.equal(wa, wb) and torch.equal(oa, ob), (m, shared)


@pytest.mark.parametrize("m", [1, 4, 6, 16])
def test_smoe_runflags_discard_bit_identical(st, m):
    """The serve's build (EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1: down waits per run) with and without the discard, against
    the plain default build too."""
    if st.get("rf") is None:
        st["rf"] = _smoe_pair(st, "-DSMOE_RUNFLAGS=1")
    rf = st["rf"]
    for shared in (False, True):
        x, w, ids, pad = _inputs(m, 3000 + m)
        a = _run(rf[False], st, x, w, ids, pad, shared)
        b = _run(rf[True], st, x, w, ids, pad, shared)
        c = _run(st["sl"][False], st, x, w, ids, pad, shared)
        assert torch.equal(a, b) and torch.equal(a, c), (m, shared)


def test_smoe_discard_graph_replay(st):
    m = 6
    x, w, ids, pad = _inputs(m, 4000)
    sl = st["sl"][True]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run(sl, st, x, w, ids, pad, True)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _run(sl, st, x, w, ids, pad, True)
    for rep in range(4):
        x2, w2, ids2, _ = _inputs(m, 4001 + rep)
        x.copy_(x2)
        w.copy_(w2)
        ids.copy_(ids2)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _run(st["sl"][False], st, x, w, ids, pad, True)), rep


@pytest.mark.parametrize("fused", [False, True])
def test_smoe_canary(st, flush, fused):
    from test_kernels_gpu import H, LIMIT, routing
    m = 6
    kp = st["topk"] + 1
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, N, seed=77)
    ys = st["sl"][True].scratch[2]
    assert ys.data_ptr() == st["sl"][False].scratch[2].data_ptr()       # one scratch for every layer and build
    live = ys[: m * kp * H]                                             # every pick of every row is summed here
    res = {}
    for d in (False, True):
        for rep in range(3):
            ys.fill_(float("nan"))
            flush()
            if fused:
                out = st["sl"][d].fused(x, st["gw"], st["bias"], st["b"]["perm"], LIMIT, SCALE, None, True)[0]
            else:
                out = _run(st["sl"][d], st, x, w, ids, None, True)
            torch.cuda.synchronize()
            assert not torch.isnan(out.float()).any(), (d, rep)
            res.setdefault(d, out.clone())
            assert torch.equal(out, res[d])
            frac = torch.isnan(live).float().mean().item()
            if not d:
                assert frac == 0.0
        print(f"\nsmoe {'fused' if fused else '3-launch'} discard={d}: {frac:.0%} of the summed per-slot outputs read "
              f"back as the DRAM canary after the call")
        assert not d or frac > 0, "discard.global.L2 dropped no line on this GPU: the build saves no DRAM writes here"
    assert torch.equal(res[False], res[True])
