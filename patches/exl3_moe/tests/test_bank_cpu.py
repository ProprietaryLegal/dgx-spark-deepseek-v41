# SPDX-License-Identifier: Apache-2.0
"""Bank layout, TP cut and pointer tables on CPU tensors (needs torch, not CUDA / vLLM / exllamav3)."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_exl3_moe.bank import TENSOR_NAMES, Exl3Bank  # noqa: E402
from vllm_exl3_moe.plan import ExpertBits, LayerPlan, trellis_shape  # noqa: E402

H, I = 5120, 2304
KS = [5, 2, 3, 2, 4, 3]


def full_tensors(seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for k in KS:
        ex = {}
        for proj, (r, c) in (("w1", (H, I)), ("w3", (H, I)), ("w2", (I, H))):
            ex[proj] = (
                torch.randint(-32768, 32768, trellis_shape(r, c, k), dtype=torch.int16, generator=g),
                torch.randn(r, generator=g).half(),
                torch.randn(c, generator=g).half(),
            )
        out.append(ex)
    return out


def build(tp, rank, exps):
    lp = LayerPlan.build(0, [ExpertBits(k, k) for k in KS])
    b = Exl3Bank(lp, H, I, tp, rank)
    b.allocate("cpu")
    for e, ex in enumerate(exps):
        for proj, (t, suh, svh) in ex.items():
            for kind, w in (("trellis", t), ("suh", suh), ("svh", svh)):
                b.load_piece(kind, proj, e, w)
    return b


@pytest.fixture(scope="module")
def exps():
    return full_tensors()


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1)])
def test_views_hold_this_ranks_cut(exps, tp, rank):
    b = build(tp, rank, exps)
    il = I // tp
    i0, i1 = rank * il, (rank + 1) * il
    t0, t1 = i0 // 16, i1 // 16
    for e in range(len(KS)):
        p = b.lp.perm[e]
        (g_t, g_suh, g_svh, kg), (u_t, u_suh, u_svh, ku), (d_t, d_suh, d_svh, kd) = b.views(p)
        assert kg == ku == kd == KS[e]
        ex = exps[e]
        assert torch.equal(g_t, ex["w1"][0][:, t0:t1])
        assert torch.equal(u_t, ex["w3"][0][:, t0:t1])
        assert torch.equal(d_t, ex["w2"][0][t0:t1])
        assert torch.equal(g_suh, ex["w1"][1]) and torch.equal(u_suh, ex["w3"][1])
        assert torch.equal(g_svh, ex["w1"][2][i0:i1]) and torch.equal(u_svh, ex["w3"][2][i0:i1])
        assert torch.equal(d_suh, ex["w2"][1][i0:i1]) and torch.equal(d_svh, ex["w2"][2])


def test_pointer_tables_match_views(exps):
    b = build(2, 1, exps)
    groups, (g, u, d) = b.tables()
    for p in range(len(KS)):
        (gt, gs, gv, _), (ut, us, uv, _), (dt, ds, dv, _) = b.views(p)
        assert int(g.trellis[p]) == gt.data_ptr() and int(g.suh[p]) == gs.data_ptr() and int(g.svh[p]) == gv.data_ptr()
        assert int(u.trellis[p]) == ut.data_ptr() and int(u.suh[p]) == us.data_ptr() and int(u.svh[p]) == uv.data_ptr()
        assert int(d.trellis[p]) == dt.data_ptr() and int(d.suh[p]) == ds.data_ptr() and int(d.svh[p]) == dv.data_ptr()
    # one group per K, contiguous K-sorted slots, group tables are slices of the full ones
    assert [(gr.k_gu, gr.lo, gr.hi) for gr in groups] == [(2.0, 0, 2), (3.0, 2, 4), (4.0, 4, 5), (5.0, 5, 6)]
    for gr in groups:
        assert torch.equal(gr.g.trellis, g.trellis[gr.lo:gr.hi]) and gr.g.trellis.is_contiguous()


def test_plan_checkpoint_mismatch_is_loud(exps):
    lp = LayerPlan.build(0, [ExpertBits(k, k) for k in KS])
    b = Exl3Bank(lp, H, I, 2, 0)
    b.allocate("cpu")
    wrong_k = torch.zeros(trellis_shape(H, I, 4), dtype=torch.int16)    # expert 0 is K=5 in the plan
    with pytest.raises(ValueError, match="plan and checkpoint disagree"):
        b.load_piece("trellis", "w1", 0, wrong_k)
    with pytest.raises(ValueError):
        b.load_piece("suh", "w1", 0, torch.zeros(H, dtype=torch.float32))   # wrong dtype


def test_bind_external_tensors(exps):
    b = build(2, 0, exps)
    other = Exl3Bank(b.lp, H, I, 2, 0)
    other.bind({n: b.t[n] for n in TENSOR_NAMES})
    assert all(other.t[n].data_ptr() == b.t[n].data_ptr() for n in TENSOR_NAMES)
