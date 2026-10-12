# SPDX-License-Identifier: Apache-2.0
"""Patch 0010: per-round DSpark verify depth inside a fixed-shape verify window.

The depth rule against a plain-Python reference (TensorFold's ``_choose_k``: the k with the most expected tokens per
ms, expected tokens = 1 + prefix survivals of the confidence head), the dead-row mask that feeds is_padding and the
rejection sampler, the policy's fixed / file modes, and the mechanism the speed comes from: this build's V4.1 router op
gives rows marked as padding top-k ids of -1 (no experts) and leaves every other row's picks bit-identical.

Run inside the dsv41 image on a GPU:  python3 -m pytest -q test_dspark_depth_policy.py
"""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pytest.importorskip("vllm")   # runs in the serve image
DEV = "cuda"


def ref_depth(conf, cost):
    best, best_v, surv, exp_tokens = 0, -1.0, 1.0, 1.0
    for d in range(len(cost)):
        if d > 0:
            surv *= conf[d - 1]
            exp_tokens += surv
        v = exp_tokens / cost[d]
        if v > best_v:
            best, best_v = d, v
    return best


def test_choose_depth_matches_reference():
    from vllm.v1.worker.gpu.spec_decode.depth_policy import choose_depth

    g = torch.Generator().manual_seed(0)
    k = 5
    cost = [42.5, 45.5, 48.5, 51.0, 53.0, 55.0]
    conf = torch.rand((256, k), generator=g) ** 0.5
    conf[:16] = 1.0
    conf[16:32] = 0.0
    got = choose_depth(conf.to(DEV), torch.tensor(cost, device=DEV)).cpu().tolist()
    want = [ref_depth(row.tolist(), cost) for row in conf]
    assert got == want
    assert got[:16] == [k] * 16 and got[16:32] == [0] * 16
    assert len(set(got)) > 2  # the rule really varies the depth


def test_round_cost_table(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.depth_policy import round_cost_ms

    monkeypatch.setenv("VLLM_DSPARK_ROUND_MS", "30,35,39")
    monkeypatch.setenv("VLLM_DSPARK_DRAFT_MS", "3.5")
    assert round_cost_ms(4) == pytest.approx([33.5, 38.5, 42.5, 46.5, 50.5])


def make_policy(monkeypatch, policy, k=5, n=8, **env):
    from vllm.v1.worker.gpu.spec_decode.depth_policy import DSparkDepthPolicy

    monkeypatch.setenv("VLLM_DSPARK_DEPTH_POLICY", policy)
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    return DSparkDepthPolicy(k, n, torch.device(DEV), has_confidence=True)


def test_dead_rows_by_request(monkeypatch):
    p = make_policy(monkeypatch, "conf")
    k = 5
    # three requests in batch order: slot 2 verifies 5 drafts, slot 0 is a prefill chunk (one logits row), slot 5
    # verifies 5 drafts
    idx = torch.tensor([2, 0, 5], dtype=torch.int32, device=DEV)
    conf = torch.zeros((3, k), device=DEV)                    # slot 0: no confidence at all -> depth 0
    conf[0] = torch.tensor([0.99, 0.95, 0.05, 0.05, 0.05])   # shallow
    conf[2] = 1.0                                             # deep
    p.record(conf, idx, 3)
    depth = p.depth.cpu().tolist()
    assert depth[5] == k and 0 < depth[2] < k and depth[0] == 0 and depth[1] == k   # slot 1: never recorded
    exp_idx = torch.tensor([2] * 6 + [0] + [5] * 6, dtype=torch.int32, device=DEV)
    exp_pos = torch.tensor(list(range(6)) + [0] + list(range(6)), dtype=torch.int32, device=DEV)
    dead = p.dead_rows(exp_idx, exp_pos).cpu().tolist()
    d2 = depth[2]
    assert dead[:6] == [pos > d2 for pos in range(6)]
    assert dead[6] is False and dead[7:] == [False] * 6
    p.reset(2)
    assert p.depth[2].item() == k


def test_fixed_policy_and_file_switch(monkeypatch, tmp_path):
    f = tmp_path / "depth"
    f.write_text("fixed:1")
    p = make_policy(monkeypatch, "fixed:3", VLLM_DSPARK_DEPTH_FILE=str(f))
    idx = torch.arange(4, dtype=torch.int32, device=DEV)
    conf = torch.ones((4, 5), device=DEV)
    p.record(conf, idx, 4)                     # record 0 reads the file first
    assert p.describe() == "fixed:1" and p.depth[:4].tolist() == [1] * 4
    f.write_text("conf")
    for _ in range(64):
        p.record(conf, idx, 4)
    assert p.describe() == "conf" and p.depth[:4].tolist() == [5] * 4
    # a measured table rides along: flat costs make every confident draft worth verifying, a steep one none
    f.write_text("conf:10,100,200,300,400,500")
    half = torch.full((4, 5), 0.5, device=DEV)
    for _ in range(64):
        p.record(half, idx, 4)
    assert p.cost_ms.tolist() == pytest.approx([10, 100, 200, 300, 400, 500]) and p.depth[:4].tolist() == [0] * 4


def test_router_padding_rows_get_no_experts():
    """The MoE skip: is_padding rows come out of the V4.1 top-k op with ids -1 and weight 0, every other row's picks
    and weights are bit-identical to an unmasked call (biased sqrt-softplus layers and hash layers)."""
    from vllm import _custom_ops as ops

    m, e, topk, vocab = 6, 384, 6, 1000          # V4.1-Flash: 384 routed experts, top-6
    g = torch.Generator(device=DEV).manual_seed(1)
    gating = torch.randn((m, e), device=DEV, generator=g)
    bias = torch.randn((e,), device=DEV, generator=g) * 0.1
    tokens = torch.randint(0, vocab, (m,), device=DEV, generator=g, dtype=torch.int32)
    tid2eid = torch.stack([torch.randperm(e, device=DEV, generator=g)[:topk] for _ in range(vocab)]).to(torch.int32)
    pad = torch.tensor([False, False, False, True, True, True], device=DEV)

    def run(is_padding, hashed):
        w = torch.empty((m, topk), device=DEV, dtype=torch.float32)
        ids = torch.empty((m, topk), device=DEV, dtype=torch.int32)
        tei = torch.empty((m, topk), device=DEV, dtype=torch.int32)
        ops.topk_hash_softplus_sqrt(
            w, ids, tei, gating, True, 1.5, None if hashed else bias, tokens if hashed else None,
            tid2eid if hashed else None, is_padding=is_padding)
        return w, ids

    for hashed in (False, True):
        w0, ids0 = run(None, hashed)
        w1, ids1 = run(pad, hashed)
        assert torch.all(ids1[3:] == -1), (hashed, ids1)
        assert torch.all(ids1[:3] == ids0[:3]) and torch.equal(w1[:3], w0[:3]), hashed
        assert torch.all(ids0[3:] >= 0)
