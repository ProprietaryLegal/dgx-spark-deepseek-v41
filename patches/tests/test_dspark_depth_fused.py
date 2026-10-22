# SPDX-License-Identifier: Apache-2.0
"""Patch 0038: VLLM_DSPARK_DEPTH_FUSED, the conf rule and its write into the depth slots in one launch
(_choose_depth_kernel) and dead_rows in one (_dead_rows_kernel).

Against the torch path (choose_depth + index_copy_, the compare over a gather), on random confidences, cost tables and
request slots: the same depths row for row (random fp32 inputs leave no near-ties for the scans' association to flip),
ties to the shallowest depth, clamped confidences outside [0, 1], and the policy's record() / dead_rows() with the flag
on and off.

Run inside the dsv41 image on a GPU:  python3 -m pytest -q test_dspark_depth_fused.py
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = "cuda"


@pytest.mark.parametrize("k,n", [(5, 1), (5, 7), (3, 33), (8, 4)])
def test_kernel_matches_choose_depth(k, n):
    from vllm.v1.worker.gpu.spec_decode import depth_policy as D

    g = torch.Generator(device=DEV).manual_seed(10 * k + n)
    for trial in range(20):
        conf = torch.rand((n, k), device=DEV, generator=g)
        if trial % 4 == 1:
            conf = conf * 1.4 - 0.2                              # outside [0, 1]: clamped by both paths
        cost = torch.sort(torch.rand(k + 1, device=DEV, generator=g) * 20 + 25).values
        idx = torch.randperm(64, device=DEV, generator=g)[:n].to(torch.int32)
        want = torch.full((64,), -7, dtype=torch.int32, device=DEV)
        want.index_copy_(0, idx.long(), D.choose_depth(conf, cost))
        got = torch.full((64,), -7, dtype=torch.int32, device=DEV)
        D._choose_depth_kernel[(n,)](conf.contiguous(), k, idx, cost, got, K=k)
        assert torch.equal(got, want), (trial, got[idx.long()].tolist(), want[idx.long()].tolist())


def test_ties_keep_the_shallowest_depth():
    from vllm.v1.worker.gpu.spec_decode import depth_policy as D

    k = 5
    cost = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], device=DEV)
    conf = torch.ones((2, k), device=DEV)                       # expected d + 1 tokens: every depth at 0.1 a ms
    conf[1] = 0.0                                               # depth 0 strictly best
    idx = torch.tensor([3, 5], dtype=torch.int32, device=DEV)
    got = torch.full((8,), -1, dtype=torch.int32, device=DEV)
    D._choose_depth_kernel[(2,)](conf, k, idx, cost, got, K=k)
    want = D.choose_depth(conf, cost)
    assert got[3].item() == want[0].item() == 0 and got[5].item() == want[1].item() == 0


@pytest.mark.parametrize("fused", [False, True])
def test_policy_record_and_dead_rows(monkeypatch, fused):
    from vllm.v1.worker.gpu.spec_decode import depth_policy as D

    monkeypatch.setattr(D, "_DEPTH_FUSED", fused)
    monkeypatch.setenv("VLLM_DSPARK_DEPTH_POLICY", "conf")
    monkeypatch.setenv("VLLM_DSPARK_ROUND_MS", "27.9,32.2,35.8,39.3,42.3,45.4")
    monkeypatch.delenv("VLLM_DSPARK_DEPTH_FILE", raising=False)
    p = D.DSparkDepthPolicy(5, 16, torch.device(DEV), has_confidence=True)
    g = torch.Generator(device=DEV).manual_seed(4)
    conf = torch.rand((16, 5), device=DEV, generator=g)
    idx = torch.tensor([9, 2, 14, 0, 7], dtype=torch.int32, device=DEV)
    p.record(conf, idx, 5)
    want = D.choose_depth(conf[:5], p.cost_ms)
    assert torch.equal(p.depth[idx.long()], want)
    exp_idx = idx.repeat_interleave(6).contiguous()
    exp_pos = torch.arange(6, dtype=torch.int32, device=DEV).repeat(5).contiguous()
    dead = p.dead_rows(exp_idx, exp_pos)
    assert dead.dtype == torch.bool
    assert torch.equal(dead, exp_pos > p.depth[exp_idx.long()])
