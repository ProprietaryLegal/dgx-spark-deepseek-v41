# SPDX-License-Identifier: Apache-2.0
"""Patch 0034: the drafts' logit margin and top-1 probability in the DSpark round log (VLLM_DSPARK_CONF_LOG_FEATURES=1).

(1) draft_logit_features on two vocabulary shards with a stand-in all-gather equals the full-vocabulary (top-1 - top-2)
    margin and softmax top-1 probability, with the maxima in either shard and an exact tie across shards (margin 0).
(2) The round log carries "f" (2 values a position) when the flag is on, taken from the features recorded with the
    drafts, and no "f" without it.

Run inside the dsv41 image:  python3 -m pytest -q test_dspark_conf_features.py
"""
import json

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")   # runs in the serve image
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 5


def shard_gather(shards):
    """Each shard's (top-1, top-2, logsumexp), concatenated in rank order: what the all-gather returns."""
    locals_ = []
    for sh in shards:
        lf = sh.float()
        locals_.append(torch.cat([lf.topk(2, dim=-1).values, torch.logsumexp(lf, -1, keepdim=True)], -1))
    return torch.cat(locals_, dim=-1)


@pytest.mark.parametrize("where", ["rank0", "rank1", "tie"])
def test_features_match_full_vocab(where):
    from vllm.models.deepseek_v41.nvidia.dspark import draft_logit_features

    g = torch.Generator(device=DEV).manual_seed(len(where))
    full = (torch.randn((4, 2 * 1000), device=DEV, generator=g) * 2).to(torch.bfloat16)
    if where == "rank0":
        full[:, 17] = 12.0
    elif where == "rank1":
        full[:, 1500] = 12.0
    else:
        full[:, 17] = 12.0
        full[:, 1500] = 12.0
    shards = [full[:, :1000], full[:, 1000:]]
    gathered = shard_gather(shards)
    got = draft_logit_features(shards[0], lambda t: gathered, 2)   # rank 0's view; the gather is the same on both
    f = full.float()
    top = f.topk(2, dim=-1).values
    ref = torch.stack([top[:, 0] - top[:, 1], torch.softmax(f, -1).max(-1).values], -1)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    if where == "tie":
        assert (got[:, 0] == 0).all()


def make_policy(monkeypatch, path, feats):
    from vllm.v1.worker.gpu.spec_decode.depth_policy import DSparkDepthPolicy

    for key in ("VLLM_DSPARK_CONF_TEMPS", "VLLM_DSPARK_CONF_BIAS", "VLLM_DSPARK_DEPTH_FILE", "VLLM_DSPARK_DEPTH_LOG",
                "VLLM_DSPARK_CONF_LOG_TAG", "VLLM_DSPARK_CONF_LOG_FEATURES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_DSPARK_DEPTH_POLICY", "fixed:5")
    monkeypatch.setenv("VLLM_DSPARK_ROUND_MS", "28.9,32.85,36.57,40.13,42.62,45.69")
    monkeypatch.setenv("VLLM_DSPARK_DRAFT_MS", "0")
    monkeypatch.setenv("VLLM_DSPARK_CONF_LOG", str(path))
    if feats:
        monkeypatch.setenv("VLLM_DSPARK_CONF_LOG_FEATURES", "1")
    return DSparkDepthPolicy(K, 8, torch.device(DEV), has_confidence=True)


@pytest.mark.parametrize("feats", [True, False])
def test_log_lines_carry_features(monkeypatch, tmp_path, feats):
    path = tmp_path / "conf.jsonl"
    p = make_policy(monkeypatch, path, feats)
    conf = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5], [0.5] * 5], device=DEV)
    fe = torch.arange(2 * K * 2, dtype=torch.float32, device=DEV).view(2, K, 2) / 10
    idx = torch.tensor([5, 2], dtype=torch.int32, device=DEV)
    p.record(conf, idx, 2, feats=fe)
    p.observe(idx, torch.tensor([3, 1], dtype=torch.int32, device=DEV), np.array([5, 5]), 2, req_ids=["a", "b"])
    p._conf_log.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()][1:]
    assert [r["q"] for r in rows] == ["a", "b"] and [r["a"] for r in rows] == [2, 0]
    for r, f in zip(rows, fe):
        if feats:
            assert r["f"] == pytest.approx(f.flatten().tolist(), abs=1e-5)
        else:
            assert "f" not in r


def test_log_hidden_side_file(monkeypatch, tmp_path):
    """VLLM_DSPARK_CONF_LOG_HIDDEN: the confidence head's inputs recorded with the drafts land in <log>.hid as bf16, one
    k x width record a written line in line order, and each line's "h" is its record index; a request with nothing
    verified writes neither."""
    path = tmp_path / "conf.jsonl"
    p = make_policy(monkeypatch, path, feats=False)
    width = 6
    conf = torch.tensor([[0.9] * K, [0.5] * K, [0.7] * K], device=DEV)
    hid = torch.randn(3, K, width, device=DEV).to(torch.bfloat16)
    idx = torch.tensor([4, 0, 7], dtype=torch.int32, device=DEV)
    p.record(conf, idx, 3, hidden=hid)
    p.observe(idx, torch.tensor([2, 1, 0], dtype=torch.int32, device=DEV), np.array([5, 5, 0]), 3,
              req_ids=["a", "b", "c"])
    hid2 = torch.randn(1, K, width, device=DEV).to(torch.bfloat16)
    p.record(conf[:1], idx[:1], 1, hidden=hid2)
    p.observe(idx[:1], torch.tensor([4], dtype=torch.int32, device=DEV), np.array([5]), 1, req_ids=["a"])
    p._conf_log.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()][1:]
    assert [(r["q"], r["h"]) for r in rows] == [("a", 0), ("b", 1), ("a", 2)]
    raw = np.fromfile(str(path) + ".hid", dtype=np.uint16).reshape(-1, K, width)
    got = torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32))        # bf16 bits -> fp32
    want = torch.stack([hid[0], hid[1], hid2[0]]).float().cpu()
    assert torch.equal(got, want)
