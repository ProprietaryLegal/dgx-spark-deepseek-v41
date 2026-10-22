# SPDX-License-Identifier: Apache-2.0
"""Patch 0023: confidence temperatures and the round log for the DSpark depth policy (patch 0010).

calibrate() is a temperature on the confidence head's logit (the policy receives sigmoid(logit) in fp32), unset knobs
leave the policy as 0010 ships it, temperatures move the depth and nothing else, and VLLM_DSPARK_CONF_LOG writes a
header plus one line per request per verify round (raw confidences, the depth that round used, drafts in the window,
drafts accepted), in step order, with a new header when the depth file switches the policy.

Run inside the dsv41 image:  python3 -m pytest -q test_dspark_conf_calibration.py
"""
import json

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")   # runs in the serve image
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 5
TABLE = [28.90, 32.85, 36.57, 40.13, 42.62, 45.69]   # the shipping round-cost table, ms by depth 0..5
KNOBS = ("VLLM_DSPARK_CONF_TEMPS", "VLLM_DSPARK_CONF_BIAS", "VLLM_DSPARK_CONF_LOG", "VLLM_DSPARK_CONF_LOG_TAG",
         "VLLM_DSPARK_DEPTH_FILE", "VLLM_DSPARK_DEPTH_LOG")


def make_policy(monkeypatch, policy="conf", n=8, dev=DEV, **env):
    from vllm.v1.worker.gpu.spec_decode.depth_policy import DSparkDepthPolicy

    for key in KNOBS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_DSPARK_DEPTH_POLICY", policy)
    monkeypatch.setenv("VLLM_DSPARK_ROUND_MS", ",".join(map(str, TABLE)))
    monkeypatch.setenv("VLLM_DSPARK_DRAFT_MS", "0")
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    return DSparkDepthPolicy(K, n, torch.device(dev), has_confidence=True)


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_calibrate_is_a_temperature_on_the_logit():
    from vllm.v1.worker.gpu.spec_decode.depth_policy import calibrate

    g = torch.Generator().manual_seed(0)
    z = torch.rand((512, K), generator=g) * 12 - 6          # the head's logits
    t = torch.tensor([0.7, 1.0, 1.3, 2.0, 3.5])
    b = torch.tensor([0.0, -0.3, 0.2, -1.0, 0.5])
    got = calibrate(torch.sigmoid(z).to(DEV), (1 / t).to(DEV), b.to(DEV)).cpu()
    assert torch.allclose(got, torch.sigmoid(z / t + b), atol=2e-5, rtol=0)
    same = calibrate(torch.sigmoid(z).to(DEV), torch.ones(K, device=DEV), torch.zeros(K, device=DEV)).cpu()
    assert torch.allclose(same, torch.sigmoid(z), atol=2e-6, rtol=0)
    # saturated scores stay finite: the logit is clamped 1e-6 from 0 and 1
    edge = calibrate(torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.5]], device=DEV), (1 / t).to(DEV), b.to(DEV))
    assert torch.isfinite(edge).all()


def test_unset_knobs_leave_the_policy_alone(monkeypatch):
    p = make_policy(monkeypatch)
    assert p.inv_temp is None and p.bias is None and p._conf_log is None


def test_knob_parsing(monkeypatch):
    p = make_policy(monkeypatch, VLLM_DSPARK_CONF_TEMPS="1.5")
    assert p._temps == [1.5] * K and p._bias == [0.0] * K
    p = make_policy(monkeypatch, VLLM_DSPARK_CONF_TEMPS="1.2,1.3", VLLM_DSPARK_CONF_BIAS="-0.5")
    assert p._temps == [1.2] + [1.3] * 4 and p._bias == [-0.5] * K
    assert p.inv_temp.cpu().tolist() == pytest.approx([1 / 1.2] + [1 / 1.3] * 4)
    p = make_policy(monkeypatch, VLLM_DSPARK_CONF_BIAS="0.1")       # a bias alone: unit temperatures
    assert p._temps == [1.0] * K and p.bias.cpu().tolist() == pytest.approx([0.1] * K)
    with pytest.raises(ValueError):
        make_policy(monkeypatch, VLLM_DSPARK_CONF_TEMPS="1,1,1,1,1,1")   # 6 values for 5 positions
    with pytest.raises(ValueError):
        make_policy(monkeypatch, VLLM_DSPARK_CONF_TEMPS="1.2,0")


def test_temperatures_move_only_the_depth(monkeypatch):
    """Raw 0.8 everywhere verifies all 5 drafts on the shipping table; at temperature 4 the same drafts read 0.59 and
    verify fewer. Fixed depths ignore the knob."""
    from vllm.v1.worker.gpu.spec_decode.depth_policy import calibrate, choose_depth

    conf = torch.full((4, K), 0.8, device=DEV)
    idx = torch.tensor([0, 2, 5, 7], dtype=torch.int32, device=DEV)
    raw = make_policy(monkeypatch)
    raw.record(conf, idx, 4)
    hot = make_policy(monkeypatch, VLLM_DSPARK_CONF_TEMPS="4")
    hot.record(conf, idx, 4)
    d_raw, d_hot = raw.depth[idx.long()].tolist(), hot.depth[idx.long()].tolist()
    assert d_raw == [K] * 4 and d_hot == [3] * 4
    assert d_hot == choose_depth(calibrate(conf, hot.inv_temp, hot.bias), hot.cost_ms).tolist()
    assert hot.depth[1].item() == K                              # untouched slots keep the full window
    fixed = make_policy(monkeypatch, "fixed:2", VLLM_DSPARK_CONF_TEMPS="4")
    fixed.record(conf, idx, 4)
    assert fixed.depth[idx.long()].tolist() == [2] * 4


def test_conf_log_lines(monkeypatch, tmp_path):
    """Two verify rounds with the batch reordered in between: each line carries the raw confidences (the temperature
    only steers the depth), the depth that round used, the drafts in the window and the drafts accepted; a request with
    nothing verified writes no line."""
    path = tmp_path / "conf.jsonl"
    p = make_policy(monkeypatch, VLLM_DSPARK_CONF_LOG=str(path), VLLM_DSPARK_CONF_TEMPS="2",
                    VLLM_DSPARK_CONF_LOG_TAG="chat")
    rid = "chatcmpl-dom-chat-p0374-1a2b3c4d-9f8e7d6c"
    conf = torch.tensor([[0.99, 0.97, 0.9, 0.8, 0.7], [0.6, 0.3, 0.2, 0.1, 0.05], [0.9] * 5], device=DEV)
    idx = torch.tensor([3, 1, 6], dtype=torch.int32, device=DEV)
    p.record(conf, idx, 3)                       # drafts proposed: each request's depth for its next verify
    d1 = p.depth[idx.long()].tolist()
    p.observe(idx, torch.tensor([3, 1, 0], dtype=torch.int32, device=DEV), np.array([5, 5, 0]), 3,
              req_ids=[rid, "r1", "r6"])          # 2 accepted, 0 accepted, slot 6 verified nothing
    conf2 = torch.tensor([[0.5] * 5, [0.95] * 5], device=DEV)
    idx2 = torch.tensor([6, 3], dtype=torch.int32, device=DEV)
    p.record(conf2, idx2, 2)
    d2 = p.depth[idx2.long()].tolist()
    p.observe(idx2, torch.tensor([1, 2], dtype=torch.int32, device=DEV), np.array([5, 4]), 2, req_ids=["r6", rid])
    p._conf_log.close()
    hdr, *rows = read_log(path)
    assert hdr["hdr"] == 1 and hdr["k"] == K and hdr["policy"] == "conf" and hdr["tag"] == "chat"
    assert hdr["temps"] == [2.0] * K and hdr["bias"] == [0.0] * K and hdr["cost_ms"] == pytest.approx(TABLE)
    assert [r["q"] for r in rows] == [rid, "r1", "r6", rid] and [r["s"] for r in rows] == [0, 0, 1, 1]
    assert d1[:2] == [4, 2] and d2 == [2, 5]     # the depths at temperature 2 (raw would verify deeper)
    want = [(conf[0], d1[0], 5, 2), (conf[1], d1[1], 5, 0), (conf2[0], d2[0], 5, 0), (conf2[1], d2[1], 4, 1)]
    for r, (c, d, n, a) in zip(rows, want):
        assert r["c"] == pytest.approx(c.tolist(), abs=1e-6)
        assert (r["d"], r["n"], r["a"]) == (d, n, a)


def test_conf_log_header_on_policy_switch(monkeypatch, tmp_path):
    """The depth file switches fixed:5 -> conf at record 64: every round logged before the new header used depth 5,
    every round after it the conf rule's depth."""
    path, f = tmp_path / "conf.jsonl", tmp_path / "depth"
    f.write_text("fixed:5")
    p = make_policy(monkeypatch, "fixed:5", VLLM_DSPARK_CONF_LOG=str(path), VLLM_DSPARK_DEPTH_FILE=str(f))
    idx = torch.arange(2, dtype=torch.int32, device=DEV)
    conf = torch.full((2, K), 0.5, device=DEV)    # conf rule: depth 2 on the shipping table
    ns = torch.tensor([3, 1], dtype=torch.int32, device=DEV)
    for step in range(70):
        if step == 10:
            f.write_text("conf")
        p.record(conf, idx, 2)
        p.observe(idx, ns, np.array([5, 5]), 2, req_ids=["a", "b"])
    p._conf_log.close()
    lines = read_log(path)
    heads = [i for i, r in enumerate(lines) if r.get("hdr")]
    assert len(heads) == 2 and lines[heads[0]]["policy"] == "fixed:5" and lines[heads[1]]["policy"] == "conf"
    before, after = lines[1:heads[1]], lines[heads[1] + 1:]
    assert len(before) == 2 * 64 and {r["d"] for r in before} == {5}
    assert len(after) == 2 * 6 and {r["d"] for r in after} == {2}
    assert [r["s"] for r in before + after] == [s for s in range(70) for _ in range(2)]


class LateEvent:
    """A device-to-host copy that has not landed until someone waits for it."""

    def __init__(self):
        self.done = False

    def record(self):
        self.done = False

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


def test_conf_log_ring_waits_only_when_full(monkeypatch, tmp_path):
    """Copies still in flight stay queued (no wait); when the 8-slot ring is full the oldest is waited for, and every
    step lands in the file in order. CPU tensors with stand-in events, so nothing real is in flight."""
    path = tmp_path / "conf.jsonl"
    p = make_policy(monkeypatch, "fixed:5", dev="cpu", VLLM_DSPARK_CONF_LOG=str(path))
    log = p._conf_log
    log.events = [LateEvent() for _ in range(log.SLOTS)]
    idx = torch.tensor([0], dtype=torch.int32)
    for step in range(20):
        p.record(torch.full((1, K), step / 20), idx, 1)
        p.observe(idx, torch.tensor([1 + step % 6], dtype=torch.int32), np.array([5]), 1, req_ids=["x"])
        assert len(log.pending) == min(step + 1, log.SLOTS)
    log.close()
    rows = read_log(path)[1:]
    assert [r["s"] for r in rows] == list(range(20))
    assert [r["a"] for r in rows] == [s % 6 for s in range(20)]
    assert [r["c"][0] for r in rows] == pytest.approx([s / 20 for s in range(20)])
