# SPDX-License-Identifier: Apache-2.0
"""Several reuse checkpoints for a partial pass 1 (p4_pass1.Reuse) and extra expert sources for an overlay variant
(p4_overlay_variant.py --extra), on tiny fake assembled dirs (CPU only).
  python3 -m pytest scripts/quantize/test_multi_reuse.py -q
"""
import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p4_pass1 import Reuse  # noqa: E402

N_LAYERS, N_EXP = 2, 6


def plan(ks):
    """ks: one K string per layer ('3342..'), k_down = k."""
    return {"format": "exl3_moe_plan", "version": 1, "layers": {str(l): {"k": k} for l, k in enumerate(ks)}}


def expert_tensors(l, e, k, tag):
    """Fake EXL3 tensors: trellis [2, 2, 16K] int16 filled with `tag` (which build wrote it), suh / svh fp16."""
    out = {}
    for proj in ("w1", "w3", "w2"):
        pre = f"layers.{l}.ffn.experts.{e}.{proj}."
        out[pre + "trellis"] = torch.full((2, 2, 16 * int(k)), tag, dtype=torch.int16)
        out[pre + "suh"] = torch.ones(32, dtype=torch.float16)
        out[pre + "svh"] = torch.ones(32, dtype=torch.float16)
    return out


def assembled(d, p, tag):
    """An assembled dir: one model-exl3-LXX.safetensors a layer at plan p, a plan file, one other file."""
    os.makedirs(d)
    for l, v in p["layers"].items():
        t = {}
        for e, k in enumerate(v["k"]):
            t.update(expert_tensors(int(l), e, k, tag))
        save_file(t, os.path.join(d, f"model-exl3-L{int(l):02d}.safetensors"))
    json.dump(p, open(os.path.join(d, "exl3_moe_plan.json"), "w"))
    open(os.path.join(d, "config.json"), "w").write("{}")
    save_file({"x": torch.zeros(1)}, os.path.join(d, "model.safetensors.index.json.fake"))
    return d


def part_files(d, l, experts, tag):
    """A pass 1 run's part file for layer l holding `experts` {e: K}."""
    os.makedirs(d, exist_ok=True)
    t = {}
    for e, k in experts.items():
        t.update(expert_tensors(l, e, k, tag))
    save_file(t, os.path.join(d, f"layer_{l:02d}.part0.safetensors"))


def read_expert(d, l, e):
    with safe_open(os.path.join(d, f"model-exl3-L{l:02d}.safetensors"), "pt") as f:
        v = f.get_tensor(f"layers.{l}.ffn.experts.{e}.w2.trellis")
    return v.shape[-1] // 16, int(v.flatten()[0])


SHIP = plan(["332233", "323232"])          # tag 1
Q280 = plan(["443233", "423242"])          # tag 2: ship + some experts one K up
NEW = plan(["432234", "323252"])           # mixes: q280 K, ship K, new K (expert 5 L0 K4, expert 4 L1 K5)


@pytest.fixture
def dirs(tmp_path):
    return {"ship": assembled(str(tmp_path / "ship"), SHIP, 1), "q280": assembled(str(tmp_path / "q280"), Q280, 2)}


def test_reuse_sources_and_changed(dirs, tmp_path):
    Reuse.device = "cpu"
    r = Reuse(f'{dirs["q280"]},{dirs["ship"]}', [Q280, SHIP], NEW)
    # layer 0: new 432234 vs q280 443233 / ship 332233
    assert [r.source(0, e) for e in range(N_EXP)] == [0, 1, 1, 0, 0, None]
    assert r.changed(NEW, 0) == [5]
    # layer 1: new 323252 vs q280 423242 / ship 323232
    assert [r.source(1, e) for e in range(N_EXP)] == [1, 0, 0, 0, None, 0]
    assert r.changed(NEW, 1) == [4]
    one = r.expert(0, 1)                      # from ship (tag 1) at K3
    assert one["w2"]["trellis"].shape[-1] == 48 and int(one["w2"]["trellis"].flatten()[0]) == 1
    two = r.expert(0, 0)                      # from q280 (tag 2) at K4
    assert two["w1"]["trellis"].shape[-1] == 64 and int(two["w1"]["trellis"].flatten()[0]) == 2
    with pytest.raises(SystemExit):
        r.expert(0, 5)                        # nobody holds it at K4


def test_reuse_single_checkpoint_matches_old_behaviour(dirs):
    Reuse.device = "cpu"
    r = Reuse(dirs["ship"], [SHIP], Q280)
    assert r.changed(Q280, 0) == [0, 1, 2]    # 332233 -> 443233
    with pytest.raises(SystemExit):
        Reuse(f'{dirs["ship"]},{dirs["q280"]}', [SHIP], Q280)   # unpaired lists


def run_overlay(*args):
    return subprocess.run([sys.executable, os.path.join(HERE, "p4_overlay_variant.py"), *args],
                          capture_output=True, text=True)


def test_overlay_with_extra(dirs, tmp_path):
    p1 = str(tmp_path / "p1")
    # the run encoded every expert no checkpoint held at the 3.10-like plan, plus one the new plan does not take
    part_files(p1, 0, {5: "4", 2: "3"}, 3)    # L0 expert 2 at K3 (the NEW plan wants K2 there: from ship instead)
    part_files(p1, 1, {4: "5"}, 3)
    pf = str(tmp_path / "new.json")
    json.dump(NEW, open(pf, "w"))
    dst = str(tmp_path / "dst")
    r = run_overlay(dirs["q280"], p1, pf, dst, "--extra", dirs["ship"])
    assert r.returncode == 0, r.stdout + r.stderr
    want = {0: [(4, 2), (3, 1), (2, 1), (2, 2), (3, 2), (4, 3)], 1: [(3, 1), (2, 2), (3, 2), (2, 2), (5, 3), (2, 2)]}
    for l in range(N_LAYERS):
        assert [read_expert(dst, l, e) for e in range(N_EXP)] == want[l], l
    assert json.load(open(os.path.join(dst, "exl3_moe_plan.json"))) == NEW
    assert os.stat(os.path.join(dst, "config.json")).st_ino == os.stat(os.path.join(dirs["q280"], "config.json")).st_ino


def test_overlay_missing_k_stops(dirs, tmp_path):
    p1 = str(tmp_path / "p1")
    part_files(p1, 0, {2: "3"}, 3)            # L0 expert 5 at K4 is in no source
    part_files(p1, 1, {4: "5"}, 3)
    pf = str(tmp_path / "new.json")
    json.dump(NEW, open(pf, "w"))
    r = run_overlay(dirs["q280"], p1, pf, str(tmp_path / "dst"), "--extra", dirs["ship"])
    assert r.returncode != 0 and "neither the part files nor --extra" in (r.stdout + r.stderr)
    assert not os.path.exists(str(tmp_path / "dst"))


def test_overlay_strict_without_extra(dirs, tmp_path):
    p1 = str(tmp_path / "p1")
    part_files(p1, 0, {5: "4", 2: "3"}, 3)    # expert 2 does not differ from q280's K the plan wants: strict refuses
    part_files(p1, 1, {4: "5"}, 3)
    pf = str(tmp_path / "new.json")
    json.dump(NEW, open(pf, "w"))
    r = run_overlay(dirs["q280"], p1, pf, str(tmp_path / "dst"))
    assert r.returncode != 0 and "part files hold" in (r.stdout + r.stderr)
