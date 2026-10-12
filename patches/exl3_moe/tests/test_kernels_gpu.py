# SPDX-License-Identifier: Apache-2.0
"""Kernel tests on a GPU with exllamav3_ext built (a Spark; skipped elsewhere). No vLLM needed.

Weights are random EXL3 bit patterns (random int16 trellis words + random +-scale sign vectors): the
kernels decode any pattern deterministically, so this checks kernel plumbing, the K-group launches,
the TP channel split and CUDA graph replay at DeepSeek-V4.1 shapes without a quantizer. Model
quality is a separate test (the KLD panel, scripts/quantize/p4_kld.py).

Run on a Spark, one GPU, nothing else heavy running:
    CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_kernels_gpu.py
    EXL3_BENCH=1 python -m pytest -q -s tests/test_kernels_gpu.py -k bench      # decode GB/s
"""

import os
import sys
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_exl3_moe import kernels as K  # noqa: E402

if not K.have_ext():
    pytest.skip("exllamav3_ext (v1.6.0) not importable", allow_module_level=True)

from vllm_exl3_moe.bank import Exl3Bank  # noqa: E402
from vllm_exl3_moe.forward import reference_moe, routed_moe_forward  # noqa: E402
from vllm_exl3_moe.plan import ExpertBits, LayerPlan, trellis_shape  # noqa: E402

H, I, TOPK, LIMIT = 5120, 2304, 6, 10.0
DEV = torch.device("cuda", 0)


def random_experts(ks, seed=0, scale=0.1):
    """Full (unsplit) exllamav3 tensors per expert: {proj: (trellis, suh, svh)} on CPU."""
    g = torch.Generator().manual_seed(seed)

    def lin(rows, cols, k):
        t = torch.randint(-32768, 32768, trellis_shape(rows, cols, k), dtype=torch.int16, generator=g)
        suh = ((torch.randint(0, 2, (rows,), generator=g) * 2 - 1) * scale).half()
        svh = ((torch.randint(0, 2, (cols,), generator=g) * 2 - 1) * scale).half()
        return t, suh, svh

    return [{"w1": lin(H, I, k), "w3": lin(H, I, k), "w2": lin(I, H, k)} for k in ks]


def make_bank(experts, ks, tp_size=1, tp_rank=0):
    lp = LayerPlan.build(0, [ExpertBits(float(k), float(k)) for k in ks])
    bank = Exl3Bank(lp, H, I, tp_size, tp_rank)
    bank.allocate(DEV)
    for e, ex in enumerate(experts):
        for proj, (t, suh, svh) in ex.items():
            bank.load_piece("trellis", proj, e, t)
            bank.load_piece("suh", proj, e, suh)
            bank.load_piece("svh", proj, e, svh)
    groups, full = bank.tables()
    perm = torch.tensor(lp.perm, dtype=torch.int64, device=DEV)
    coop = K.get_coop_scratch(DEV, 32, TOPK, H, bank.inter_local)
    fused = K.get_fused_scratch(DEV, 256, H, bank.inter_local)
    K.prime(DEV, groups, coop, fused, H, TOPK, LIMIT, False, True)
    return dict(bank=bank, groups=groups, full=full, perm=perm, coop=coop, fused=fused)


def fwd(b, x, w, ids, kernel="coop"):
    return routed_moe_forward(x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"],
                              LIMIT, False, True, decode_kernel=kernel)


def routing(m, n_exp, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(n_exp, generator=g)[:TOPK] for _ in range(m)]).to(DEV, torch.int32)
    w = torch.rand((m, TOPK), generator=g).to(DEV)
    return w / w.sum(-1, keepdim=True) * 1.5, ids


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


KS = [2, 3, 4, 5] * 3          # 12 experts, four K groups


@pytest.fixture(scope="module")
def experts():
    return random_experts(KS, seed=1)


@pytest.fixture(scope="module")
def full_bank(experts):
    return make_bank(experts, KS)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_coop_matches_reconstruct_reference(full_bank, m):
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=m)
    out = fwd(full_bank, x, w, ids)
    ref = reference_moe(x, w, ids, full_bank["bank"], LIMIT, False, True)
    assert rel_err(out, ref) < 2e-2, rel_err(out, ref)


@pytest.mark.parametrize("m", [1, 8])
def test_fused_path_matches_coop(full_bank, m):
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=100 + m)
    a = fwd(full_bank, x, w, ids, "coop")
    b = fwd(full_bank, x, w, ids, "fused")
    assert rel_err(b, a) < 2e-2


def test_prefill_path_matches_reference(full_bank):
    m = 600                    # > fused rows (256) for some experts: exercises the reconstruct tier
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=7)
    out = fwd(full_bank, x, w, ids)
    sub = slice(0, 24)
    ref = reference_moe(x[sub], w[sub], ids[sub], full_bank["bank"], LIMIT, False, True)
    assert rel_err(out[sub], ref) < 2e-2


def test_prefill_deterministic_slots(full_bank, monkeypatch):
    """VLLM_EXL3_MOE_DET_PREFILL: the eager prefill path (fused tiers + the reconstruct tier for hot experts) gives the
    same bits on every run, and the same values as the atomic path within rounding."""
    import vllm_exl3_moe.forward as F
    m = 600
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=11)
    ids[5, 1] = -1                                  # an unrouted pick
    monkeypatch.setattr(F, "_DET_PREFILL", True)
    runs = [fwd(full_bank, x, w, ids) for _ in range(3)]
    assert all(torch.equal(r, runs[0]) for r in runs[1:]), "deterministic prefill differs run to run"
    monkeypatch.setattr(F, "_DET_PREFILL", False)
    atomic = fwd(full_bank, x, w, ids)
    assert rel_err(runs[0], atomic) < 1e-2
    sub = slice(0, 24)
    ref = reference_moe(x[sub], w[sub], ids[sub], full_bank["bank"], LIMIT, False, True)
    assert rel_err(runs[0][sub], ref) < 2e-2


@pytest.mark.parametrize("m", [1, 4, 16])
def test_tp2_channel_split_sums_to_full(experts, full_bank, m):
    """Rank 0 + rank 1 (each 1152 intermediate channels of every expert) == the unsplit expert."""
    r0, r1 = make_bank(experts, KS, 2, 0), make_bank(experts, KS, 2, 1)
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=200 + m)
    full = fwd(full_bank, x, w, ids).float()
    parts = fwd(r0, x, w, ids).float() + fwd(r1, x, w, ids).float()
    assert rel_err(parts, full) < 2e-2


def test_group_with_no_picks_passes_running_sum_through(full_bank):
    # route only to K=2 experts: the K=3/4/5 launches see no active slot and must copy sh_out through
    m = 4
    k2 = [e for e, k in enumerate(KS) if k == 2]
    ids = torch.tensor([[k2[(t + j) % len(k2)] for j in range(TOPK)] for t in range(m)], device=DEV, dtype=torch.int32)
    ids = ids[:, :len(k2)]
    w = torch.full(ids.shape, 0.5, device=DEV)
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    out = fwd(full_bank, x, w, ids)
    ref = reference_moe(x, w, ids, full_bank["bank"], LIMIT, False, True)
    assert rel_err(out, ref) < 2e-2


def test_unrouted_slots_are_ignored(full_bank):
    m = 2
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=9)
    ids2 = ids.clone()
    ids2[:, -1] = -1
    out = fwd(full_bank, x, w, ids2)
    w_ref = w.clone()
    w_ref[:, -1] = 0
    ref = reference_moe(x, w_ref, ids2, full_bank["bank"], LIMIT, False, True)
    assert rel_err(out, ref) < 2e-2


@pytest.mark.parametrize("m", [1, 8])
def test_cuda_graph_replay_is_bit_exact(full_bank, m):
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=300 + m)
    eager = fwd(full_bank, x, w, ids).clone()
    sx, sw, sids = x.clone(), w.clone(), ids.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fwd(full_bank, sx, sw, sids)            # warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sout = fwd(full_bank, sx, sw, sids)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(sout, eager)
    # new inputs through the same graph
    x2 = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w2, ids2 = routing(m, len(KS), seed=400 + m)
    sx.copy_(x2), sw.copy_(w2), sids.copy_(ids2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(sout, fwd(full_bank, x2, w2, ids2))


@pytest.mark.skipif(os.environ.get("EXL3_BENCH") != "1", reason="set EXL3_BENCH=1")
def test_bench_decode_bandwidth():
    """384 experts, one layer, TP=2 rank-0 shard (I = 1152), K mix close to 2.7 bpw.
    Prints achieved GB/s over the bytes of the unique experts touched vs the 273 GB/s roofline."""
    n = 384
    ks = ([2] * 182 + [3] * 158 + [5] * 44)[:n]             # ~2.76 bpw, the reference mix's shape
    exps = random_experts(ks, seed=5)
    b = make_bank(exps, ks, 2, 0)
    bpe = {k: 3 * H * (I // 2) * k / 8 for k in (2, 3, 4, 5)}
    for m in (1, 2, 4, 8, 16):
        x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
        w, ids = routing(m, n, seed=m)
        uniq = torch.unique(ids).tolist()
        nbytes = sum(bpe[ks[e]] for e in uniq)
        for _ in range(5):
            fwd(b, x, w, ids)
        torch.cuda.synchronize()
        it = 50
        t0 = time.perf_counter()
        for _ in range(it):
            fwd(b, x, w, ids)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / it
        print(f"m={m:2d} unique={len(uniq):3d} bytes={nbytes / 1e6:7.1f} MB  {dt * 1e6:8.1f} us/layer  "
              f"{nbytes / dt / 1e9:6.1f} GB/s (eager, incl. launch overhead; roofline 273 GB/s)")


@pytest.mark.skipif(os.environ.get("EXL3_BENCH") != "1", reason="set EXL3_BENCH=1")
@pytest.mark.parametrize("mix", ["2/3/5 (~2.76 bpw, 3 K groups)", "uniform 3 (1 K group)"])
def test_bench_decode_graph(mix):
    """Same shard as test_bench_decode_bandwidth, but one layer captured in a CUDA graph and replayed (what the
    serve does with FULL_DECODE_ONLY), plus a single-K bank: separates launch overhead from K-group cost."""
    n = 384
    ks = ([2] * 182 + [3] * 158 + [5] * 44)[:n] if mix.startswith("2/3/5") else [3] * n
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    bpe = {k: 3 * H * (I // 2) * k / 8 for k in (2, 3, 4, 5)}
    for m in (1, 2, 4, 8, 16):
        x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
        w, ids = routing(m, n, seed=m)
        nbytes = sum(bpe[ks[e]] for e in torch.unique(ids).tolist())
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fwd(b, x, w, ids)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fwd(b, x, w, ids)
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        it = 200
        t0 = time.perf_counter()
        for _ in range(it):
            g.replay()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / it
        print(f"[graph, {mix}] m={m:2d} bytes={nbytes / 1e6:7.1f} MB  {dt * 1e6:8.1f} us/layer  "
              f"{nbytes / dt / 1e9:6.1f} GB/s  (x40 layers = {dt * 40e3:5.2f} ms/step)")
