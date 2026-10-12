# SPDX-License-Identifier: Apache-2.0
"""exl3_smoe (our decode MoE kernel, smoe.py) against the reconstruct reference and the exllamav3 coop path, on random
EXL3 banks with every K group (2..5), whole and TP=2-split experts; routing corner cases (unrouted picks, a token with
none, an expert picked by more than 8 tokens), determinism under CUDA-graph replay, then a decode timing against the
coop path on a plan-v2-like layer (EXL3_BENCH=1).

    python3 -m pytest -q -s test_smoe_gpu.py
    EXL3_BENCH=1 python3 -m pytest -q -s test_smoe_gpu.py -k bench      (idle GPU only)
"""
import os
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from test_kernels_gpu import DEV, H, KS, LIMIT, TOPK, make_bank, random_experts, rel_err, routing  # noqa: E402
from vllm_exl3_moe import kernels as K  # noqa: E402
from vllm_exl3_moe.forward import reference_moe, routed_moe_forward  # noqa: E402
from vllm_exl3_moe.smoe import SmoeLayer  # noqa: E402


@pytest.fixture(scope="module")
def experts():
    return random_experts(KS, seed=1)


@pytest.fixture(scope="module")
def banks(experts):
    out = {}
    for name, tp in (("full", (1, 0)), ("r0", (2, 0)), ("r1", (2, 1))):
        b = make_bank(experts, KS, *tp)
        b["smoe"] = SmoeLayer(b["bank"], TOPK)
        out[name] = b
    return out


def smoe(b, x, w, ids):
    return b["smoe"](x, ids, w, b["perm"], LIMIT)


def coop(b, x, w, ids):
    return routed_moe_forward(x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"], LIMIT, False, True,
                              decode_kernel="coop")


@pytest.mark.parametrize("m", [1, 2, 4, 6, 8, 16])
def test_matches_reference(banks, m):
    b = banks["full"]
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=m)
    out = smoe(b, x, w, ids)
    ref = reference_moe(x, w, ids, b["bank"], LIMIT, False, True)
    e = rel_err(out, ref)
    print(f"m={m}: smoe vs reference rel {e:.2e}")
    assert out.dtype == torch.bfloat16 and out.shape == (m, H) and e < 2e-2


@pytest.mark.parametrize("m", [1, 4, 6, 12])
def test_matches_coop_tp2(banks, m):
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=50 + m)
    for name in ("r0", "r1"):
        b = banks[name]
        e = rel_err(smoe(b, x, w, ids), coop(b, x, w, ids))
        print(f"m={m} {name}: smoe vs coop rel {e:.2e}")
        assert e < 1e-2
    full = smoe(banks["full"], x, w, ids).float()
    parts = smoe(banks["r0"], x, w, ids).float() + smoe(banks["r1"], x, w, ids).float()
    assert rel_err(parts, full) < 2e-2


def test_unrouted_and_empty_tokens(banks):
    b = banks["full"]
    m = 5
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=3)
    ids[1, 2] = -1
    ids[3, :] = -1                     # token 3: no pick at all -> zeros
    ids[4, 0] = -1
    ids64 = ids.to(torch.int64)
    out = smoe(b, x, w, ids64)
    ref = reference_moe(x, w, ids, b["bank"], LIMIT, False, True)
    assert torch.all(out[3] == 0)
    keep = [0, 1, 2, 4]
    assert rel_err(out[keep], ref[keep]) < 2e-2


@pytest.mark.parametrize("live", [1, 3, 5])
def test_dead_rows_leave_live_rows_bit_identical(banks, live):
    """Patch 0010's verify depth: rows past a request's depth reach the MoE with every pick -1 (vLLM's router on
    is_padding rows). The live rows must come out bit-identical to the full window's, so accepted tokens never depend
    on the depth, and the dead rows must cost no expert weights (all zeros out)."""
    b = banks["r0"]
    m = 6
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=70 + live)
    full = smoe(b, x, w, ids)
    cut = ids.clone()
    cut[live:] = -1
    part = smoe(b, x, w, cut)
    assert torch.equal(part[:live], full[:live]), "a live row changed with the depth"
    assert torch.all(part[live:] == 0)
    # the same through the pad mask (vLLM's is_padding, which its DSv4 top-k kernel ignores on non-hash layers):
    # real ids on the dead rows, still no experts for them
    pad = torch.zeros(m + 2, dtype=torch.bool, device=DEV)
    pad[live:m] = True
    padded = b["smoe"](x, ids, w, b["perm"], LIMIT, pad)
    assert torch.equal(padded, part)


def test_expert_shared_by_many_tokens(banks):
    """16 tokens all picking the same 6 experts: runs of 16 rows, split into two 8-row runs each."""
    b = banks["full"]
    m = 16
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 3, 5, 7, 9, 11]] * m, dtype=torch.int32, device=DEV)
    w = torch.rand((m, TOPK), device=DEV) + 0.1
    out = smoe(b, x, w, ids)
    ref = reference_moe(x, w, ids, b["bank"], LIMIT, False, True)
    e = rel_err(out, ref)
    print(f"shared experts: rel {e:.2e}")
    assert e < 2e-2


def test_graph_replay_deterministic(banks):
    b = banks["r0"]
    m = 6
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=9)
    eager = smoe(b, x, w, ids)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        smoe(b, x, w, ids)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = smoe(b, x, w, ids)
    for _ in range(5):
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), "graph replay differs from eager (determinism)"


@pytest.mark.skipif(not os.environ.get("EXL3_BENCH"), reason="timing: set EXL3_BENCH=1 on an idle GPU")
def test_bench_vs_coop():
    """Plan v2 layer 0 mix (K2 35, K3 185, K4 163, K5 1), TP=2 rank-0 shard; 8 routings per graph so the picked experts
    change call to call (the L2 cannot hold a layer's picks)."""
    ks = [2] * 35 + [3] * 185 + [4] * 163 + [5] * 1
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    sl = SmoeLayer(b["bank"], TOPK)
    par = K.get_parallel_coop(DEV, 32, TOPK, H, b["bank"].inter_local, K.MAX_K_GROUPS)
    bpe = {k: 3 * H * b["bank"].inter_local * k / 8 for k in set(ks)}
    for m in (1, 4, 6, 8, 12):
        x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
        calls = [routing(m, len(ks), seed=1000 * m + i) for i in range(8)]
        nbytes = sum(sum(bpe[ks[e]] for e in set(ids.flatten().tolist())) for _, ids in calls) / len(calls)

        def coop_call(w, ids):
            return routed_moe_forward(x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"], LIMIT,
                                      False, True, decode_kernel="coop", par=par)

        res = {}
        for name, fn in (("coop parallel", coop_call), ("smoe", lambda w, ids: sl(x, ids, w, b["perm"], LIMIT))):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for w, ids in calls:
                    fn(w, ids)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for w, ids in calls:
                    fn(w, ids)
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                g.replay()
            torch.cuda.synchronize()
            res[name] = (time.perf_counter() - t0) / 30 / len(calls) * 1e6
        print(f"m={m:2d}: {nbytes / 1e6:6.1f} MB/layer  coop parallel {res['coop parallel']:7.1f} us "
              f"({nbytes / res['coop parallel'] / 1e3:4.0f} GB/s)  smoe {res['smoe']:7.1f} us "
              f"({nbytes / res['smoe'] / 1e3:4.0f} GB/s)  {res['coop parallel'] / res['smoe']:.2f}x", flush=True)
