# SPDX-License-Identifier: Apache-2.0
"""Plan / bank layout / TP slicing math. Pure Python: runs anywhere (no torch, vLLM or exllamav3)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_exl3_moe.plan import (  # noqa: E402
    ALIGN_WORDS,
    BankLayout,
    Exl3MoEPlan,
    ExpertBits,
    LayerPlan,
    PlanError,
    k_code,
    parse_k,
    shard_slices,
    synthetic_plan,
    tp_shard,
    trellis_shape,
    trellis_words,
)

H, I, E, L = 5120, 2304, 384, 40


def test_parse_k_and_codes():
    assert parse_k("2") == 2.0 and parse_k(5) == 5.0 and parse_k("b") == 2.5 and parse_k(3.5) == 3.5
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 1.5, 2.5, 3.5):
        assert parse_k(k_code(k)) == float(k)
    for bad in (0, 9, 2.25, "x", 4.5):
        with pytest.raises(PlanError):
            parse_k(bad)


def test_trellis_words_and_shape():
    # w1 at K=2: (5120/16) x (2304/16) tiles x 32 words
    assert trellis_words(H, I, 2) == 320 * 144 * 32
    assert trellis_shape(H, I, 3) == (320, 144, 48)
    assert trellis_shape(I, H, 2.5) == (144, 320, 40)   # 16*K + 8 words for half K (exl3.py:66)
    assert trellis_words(H, I, 2.5) * 16 == H * I * 2.5   # bits check: words * 16 bits = weights * K


def test_layer_plan_groups_and_perm():
    ks = [5, 2, 3, 2, 5, 3, 2, 4]
    lp = LayerPlan.build(0, [ExpertBits(k, k) for k in ks])
    assert [(g.k_gu, g.lo, g.hi) for g in lp.groups] == [(2, 0, 3), (3, 3, 5), (4, 5, 6), (5, 6, 8)]
    # perm/inv_perm are inverse permutations and slots are K-sorted
    assert sorted(lp.perm) == list(range(len(ks)))
    for e, p in enumerate(lp.perm):
        assert lp.inv_perm[p] == e
    for g in lp.groups:
        assert all(ks[e] == g.k_gu for e in g.experts)
        assert list(g.experts) == [lp.inv_perm[p] for p in range(g.lo, g.hi)]


def test_separate_down_bits_make_pair_groups():
    bits = [ExpertBits(2, 3), ExpertBits(2, 2), ExpertBits(2, 3), ExpertBits(3, 3)]
    lp = LayerPlan.build(0, bits)
    assert [(g.k_gu, g.k_d, g.size) for g in lp.groups] == [(2, 2, 1), (2, 3, 2), (3, 3, 1)]


def test_bank_layout_offsets_aligned_and_disjoint():
    plan = synthetic_plan(num_layers=2, counts={2: 300, 3: 300, 5: 168})
    for Lx in plan.layers:
        bl = plan.bank_layout(Lx, tp_size=2)
        spans = []
        for p in range(E):
            for off, n in ((bl.off_w1[p], bl.words_w1[p]), (bl.off_w3[p], bl.words_w1[p])):
                assert off % ALIGN_WORDS == 0
                spans.append(("13", off, off + n))
            assert bl.off_w2[p] % ALIGN_WORDS == 0
            spans.append(("2", bl.off_w2[p], bl.off_w2[p] + bl.words_w2[p]))
        for bank in ("13", "2"):
            s = sorted((a, b) for k, a, b in spans if k == bank)
            assert all(s[i][1] <= s[i + 1][0] for i in range(len(s) - 1))
        assert max(b for k, a, b in spans if k == "13") <= bl.total13
        assert max(b for k, a, b in spans if k == "2") <= bl.total2


def test_bytes_per_rank_matches_bpw():
    # uniform K=3: trellis bytes per rank = params/2 * 3 bits / 8 (alignment is exact at these shapes)
    plan = synthetic_plan(num_layers=1, uniform_k=3)
    bl = plan.bank_layout(0, tp_size=2)
    assert bl.trellis_bytes == 3 * H * I * E * 3 // 8 // 2
    assert abs(plan.bpw() - 3.0) < 1e-12


def test_reference_mix_bpw():
    # the published K2 7,279 / K3 6,299 / K5 1,782 mix averages 2.758 bpw over 15,360 experts
    plan = synthetic_plan(counts={2: 7279, 3: 6299, 5: 1782}, seed=1)
    assert abs(plan.bpw() - (2 * 7279 + 3 * 6299 + 5 * 1782) / 15360) < 1e-9
    gib = plan.bytes_per_rank(2) / 2**30
    assert 87.0 < gib < 88.5, gib     # ~87.3 GiB trellis + ~0.54 GiB sign/scale vectors per rank


def test_usage_weighted_bpw():
    plan = synthetic_plan(num_layers=1, num_experts=4, counts={2: 2, 5: 2}, seed=0)
    lp = plan.layer(0)
    hot = [1.0 if b.k_gu == 5 else 0.0 for b in lp.bits]
    assert plan.usage_weighted_bpw({0: hot}) == 5.0
    assert plan.bpw() == 3.5


def test_tp_shard_rules():
    assert tp_shard(I, 2, 0) == (0, 1152) and tp_shard(I, 2, 1) == (1152, 2304)
    assert tp_shard(I, 1, 0) == (0, 2304)
    for tp in (3, 6, 9, 18):
        tp_shard(I, tp, 0)
    with pytest.raises(PlanError):
        tp_shard(I, 4, 0)      # 576 is not a multiple of the 128-wide Hadamard block
    with pytest.raises(PlanError):
        tp_shard(I, 8, 0)


def test_shard_slices():
    s = shard_slices("w1", I, 2, 1)
    assert s.trellis_dim == 1 and s.trellis_range == (72, 144) and s.suh_range is None and s.svh_range == (1152, 2304)
    s = shard_slices("w3", I, 2, 0)
    assert s.trellis_range == (0, 72) and s.svh_range == (0, 1152)
    s = shard_slices("w2", I, 2, 1)
    assert s.trellis_dim == 0 and s.trellis_range == (72, 144) and s.suh_range == (1152, 2304) and s.svh_range is None


def test_roundtrip_dict_and_inline_lists():
    plan = synthetic_plan(num_layers=3, counts={2: 500, 3: 500, 4: 100, 5: 52}, seed=3)
    d = plan.to_dict()
    again = Exl3MoEPlan.from_dict(d)
    assert again.to_dict() == d
    d2 = plan.to_dict(compact=False)
    assert Exl3MoEPlan.from_dict(d2).to_dict() == d


def test_plan_validation():
    base = dict(format="exl3_moe_plan", version=1, num_layers=2, num_experts=4, hidden_size=H,
                moe_intermediate_size=I, layers={"0": {"k": "2222"}})
    Exl3MoEPlan.from_dict(base)
    with pytest.raises(PlanError):
        Exl3MoEPlan.from_dict({**base, "layers": {"0": {"k": "222"}}})        # wrong expert count
    with pytest.raises(PlanError):
        Exl3MoEPlan.from_dict({**base, "layers": {"5": {"k": "2222"}}})       # layer out of range
    with pytest.raises(PlanError):
        Exl3MoEPlan.from_dict({**base, "codebook": "mcg", "layers": {"0": {"k": "2b22"}}})  # half K needs mul1
    with pytest.raises(PlanError):
        Exl3MoEPlan.from_dict({**base, "codebook": "3inst"})


def test_layers_absent_from_plan_stay_native():
    d = dict(format="exl3_moe_plan", version=1, num_layers=40, num_experts=4, hidden_size=H,
             moe_intermediate_size=I, layers={"3": {"k": "2345"}})
    plan = Exl3MoEPlan.from_dict(d)
    assert plan.has_layer(3) and not plan.has_layer(0) and not plan.has_layer(40)
