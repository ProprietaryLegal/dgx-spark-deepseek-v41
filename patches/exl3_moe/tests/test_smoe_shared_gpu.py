# SPDX-License-Identifier: Apache-2.0
"""exl3_smoe's shared-expert slot (attach_shared + shared=True) on a GPU with exllamav3_ext (a Spark, or any sm_12x card
with EXL3_KERNEL_ARCH set).

The same expert two ways, on a TP=2 rank-0 cut:
  (A) in the bank as routed expert 12, picked by every token as a 7th pick with weight 1 (top_k = 7);
  (B) attached as the shared slot, every token taking it as pick 6 of 6 + 1 (shared=True).
Run order differs (the shared slot runs first), each slot's math and the pick-order sum do not: the outputs must be
bit-identical, padded rows included. Plus smoe with shared=True vs the reconstruct reference of routed + shared.
    python3 -m pytest -q test_smoe_shared_gpu.py
"""
import os
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_kernels_gpu import DEV, H, I, LIMIT, TOPK, make_bank, random_experts, rel_err, routing  # noqa: E402
from vllm_exl3_moe.forward import reference_moe  # noqa: E402
from vllm_exl3_moe.plan import shard_slices  # noqa: E402
from vllm_exl3_moe.smoe import SmoeLayer  # noqa: E402

KS = [2, 3, 4, 5] * 3           # the routed experts
SHARED_K = 5
TP, RANK = 2, 0


def cut(ex):
    """This rank's part of one expert, as the bank cuts it (gate / up columns, down rows)."""
    out = []
    for proj in ("w1", "w3", "w2"):
        t, suh, svh = ex[proj]
        sl = shard_slices(proj, I, TP, RANK)
        a, z = sl.trellis_range
        su = suh if sl.suh_range is None else suh[sl.suh_range[0]:sl.suh_range[1]]
        sv = svh if sl.svh_range is None else svh[sl.svh_range[0]:sl.svh_range[1]]
        out.append(SimpleNamespace(trellis=t.narrow(sl.trellis_dim, a, z - a).contiguous().to(DEV),
                                   suh=su.contiguous().to(DEV), svh=sv.contiguous().to(DEV)))
    return out


@pytest.fixture(scope="module")
def setup():
    experts = random_experts(KS + [SHARED_K], seed=11)
    plus = make_bank(experts, KS + [SHARED_K], TP, RANK)          # (A): the shared expert as expert 12
    base = make_bank(experts[:-1], KS, TP, RANK)                   # (B): routed only, shared attached
    sl_a = SmoeLayer(plus["bank"], TOPK + 1)
    sl_b = SmoeLayer(base["bank"], TOPK)
    sl_b.attach_shared(*cut(experts[-1]))
    return plus, base, sl_a, sl_b


@pytest.mark.parametrize("m", [1, 2, 4, 6, 8, 16])
def test_shared_slot_equals_seventh_pick(setup, m):
    plus, base, sl_a, sl_b = setup
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=500 + m)
    ids7 = torch.cat([ids, torch.full((m, 1), len(KS), dtype=ids.dtype, device=DEV)], 1)
    w7 = torch.cat([w, torch.ones((m, 1), dtype=w.dtype, device=DEV)], 1)
    pad = torch.zeros(m, dtype=torch.bool, device=DEV)
    if m >= 4:
        pad[1] = True                                               # a padded row: no routed pick, no shared pick
    a = sl_a(x, ids7, w7, plus["perm"], LIMIT, pad)
    b = sl_b(x, ids, w, base["perm"], LIMIT, pad, shared=True)
    assert torch.equal(a, b), (a.float() - b.float()).abs().max().item()
    if m >= 4:
        assert torch.count_nonzero(b[1]).item() == 0


@pytest.mark.parametrize("m", [1, 6])
def test_shared_slot_vs_reference(setup, m):
    plus, base, sl_a, sl_b = setup
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=900 + m)
    ids7 = torch.cat([ids, torch.full((m, 1), len(KS), dtype=ids.dtype, device=DEV)], 1)
    w7 = torch.cat([w, torch.ones((m, 1), dtype=w.dtype, device=DEV)], 1)
    ref = reference_moe(x, w7, ids7, plus["bank"], LIMIT, False, True)
    got = sl_b(x, ids, w, base["perm"], LIMIT, None, shared=True)
    assert rel_err(got, ref) < 2e-2, rel_err(got, ref)


def test_without_shared_flag_unchanged(setup):
    """attach_shared does not change calls made without shared=True."""
    plus, base, sl_a, sl_b = setup
    fresh = SmoeLayer(base["bank"], TOPK)
    x = torch.randn((6, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(6, len(KS), seed=77)
    assert torch.equal(sl_b(x, ids, w, base["perm"], LIMIT), fresh(x, ids, w, base["perm"], LIMIT))
