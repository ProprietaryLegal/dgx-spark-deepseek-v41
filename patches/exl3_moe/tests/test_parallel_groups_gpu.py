# SPDX-License-Identifier: Apache-2.0
"""Concurrent K groups (kernels.run_decode_coop_parallel, VLLM_EXL3_MOE_PARALLEL_GROUPS=1) == the chained launches,
eager and under CUDA-graph capture, plus a decode timing of both on a plan-v2-like layer (EXL3_BENCH=1).

  python3 -m pytest -q -s test_parallel_groups_gpu.py                   (correctness, small bank, seconds)
  EXL3_BENCH=1 python3 -m pytest -q -s test_parallel_groups_gpu.py -k bench   (idle GPU only: timing)
"""
import os
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from test_kernels_gpu import DEV, H, KS, LIMIT, TOPK, make_bank, random_experts, rel_err, routing  # noqa: E402
from vllm_exl3_moe import kernels as K  # noqa: E402
from vllm_exl3_moe.forward import routed_moe_forward  # noqa: E402


def run(b, x, w, ids, par):
    return routed_moe_forward(x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"],
                              LIMIT, False, True, decode_kernel="coop", par=par)


@pytest.fixture(scope="module")
def bank():
    return make_bank(random_experts(KS, seed=7), KS)


@pytest.fixture(scope="module")
def par(bank):
    return K.get_parallel_coop(DEV, 32, TOPK, H, bank["bank"].inter_local, K.MAX_K_GROUPS)


@pytest.mark.parametrize("m", [1, 2, 6, 8, 16])
def test_parallel_matches_chained(bank, par, m):
    assert len(bank["groups"]) == 4
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=100 + m)
    ref = run(bank, x, w, ids, None).float()
    out = run(bank, x, w, ids, par).float()
    e = rel_err(out, ref)
    print(f"m={m}: parallel vs chained rel {e:.2e}")
    assert out.shape == ref.shape and e < 2e-3


def test_parallel_under_graph_capture(bank, par):
    m = 6
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=11)
    eager = run(bank, x, w, ids, par).float()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run(bank, x, w, ids, par)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run(bank, x, w, ids, par)
    outs = []
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        outs.append(out.float().clone())
    assert all(torch.equal(outs[0], o) for o in outs[1:]), "graph replays differ"
    e = rel_err(outs[0], eager)
    print(f"graph vs eager rel {e:.2e}")
    assert e < 2e-3


@pytest.mark.skipif(not os.environ.get("EXL3_BENCH"), reason="timing: set EXL3_BENCH=1 on an idle GPU")
def test_bench_parallel_vs_chained():
    """Plan v2 layer 0 mix (K2 35, K3 185, K4 163, K5 1: four groups), TP=2 rank-0 shard, graph replay per layer."""
    ks = [2] * 35 + [3] * 185 + [4] * 163 + [5] * 1
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    par = K.get_parallel_coop(DEV, 32, TOPK, H, b["bank"].inter_local, K.MAX_K_GROUPS)
    for m in (1, 2, 4, 6, 8):
        x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
        w, ids = routing(m, len(ks), seed=m)
        res = {}
        for name, p in (("chained", None), ("parallel", par)):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run(b, x, w, ids, p)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run(b, x, w, ids, p)
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                g.replay()
            torch.cuda.synchronize()
            res[name] = (time.perf_counter() - t0) / 200 * 1e6
        print(f"m={m}: chained {res['chained']:7.1f} us/layer, parallel {res['parallel']:7.1f} us/layer "
              f"({res['chained'] / res['parallel']:.2f}x)")
