# SPDX-License-Identifier: Apache-2.0
"""p4_drafter_head_variant.py on tiny fake model dirs (CPU only): the variant hardlinks every SRC file, adds
model-exl3-mtp-head.safetensors with HEAD_SRC's head tensors under the mtp.head.* names patch 0026 loads, lists them in
the index, sets exl3_moe.drafter_head_k and keeps the target head; it refuses a K / shape mismatch, a HEAD_SRC without
an EXL3 head, an existing DST and a stale drafter head in another file, and replaces a SRC's own drafter head.
  python3 -m pytest scripts/quantize/test_drafter_head_variant.py -q
"""
import json
import os
import re
import subprocess
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "p4_drafter_head_variant.py")
D, V = 256, 512                                         # hidden, vocab: whole 16-wide tiles
NAME = "model-exl3-mtp-head.safetensors"
OWN_RE = re.compile(r"^mtp\.head\.(trellis|suh|svh|mul1|mcg)$")   # what patch 0026's drafter loads (dspark.py)


def head(k, seed, vocab=V):
    g = torch.Generator().manual_seed(seed)
    return {"head.trellis": torch.randint(-32768, 32768, (D // 16, vocab // 16, 16 * k), dtype=torch.int16, generator=g),
            "head.suh": torch.randn(D, generator=g).half(), "head.svh": torch.randn(vocab, generator=g).half(),
            "head.mul1": torch.tensor(-2083270355, dtype=torch.int32)}


def model_dir(d, k, seed, vocab=V, head_kind="exl3", cfg_k=None):
    """A model dir: an official-style shard (backbone + drafter tensors), the EXL3 head file (or a BF16 head.weight in
    the shard), plan, tokenizer, manifest, a subdir and a relative symlink, an index and a config like ours."""
    os.makedirs(d)
    shard = {"layers.0.attn.wq_a.weight": torch.zeros(4, 4), "mtp.0.attn.wq_a.weight": torch.ones(4, 4),
             "embed.weight": torch.zeros(vocab, 8)}
    if head_kind == "bf16":
        shard["head.weight"] = torch.zeros(vocab, D, dtype=torch.bfloat16)
    save_file(shard, os.path.join(d, "model-00001-of-00001.safetensors"))
    wmap = {n: "model-00001-of-00001.safetensors" for n in shard}
    if head_kind == "exl3":
        h = head(k, seed, vocab)
        save_file(h, os.path.join(d, "model-exl3-head.safetensors"))
        wmap.update({n: "model-exl3-head.safetensors" for n in h})
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    q = {"quant_method": "deepseek_v41_exl3", "exl3_moe": {"plan_file": "exl3_moe_plan.json"}}
    if head_kind == "exl3":
        q["exl3_moe"].update(head="exl3", head_k=cfg_k or k)
    json.dump({"architectures": ["DeepseekV41ForCausalLM"], "quantization_config": q},
              open(os.path.join(d, "config.json"), "w"))
    for name in ("exl3_moe_plan.json", "tokenizer.json", "MANIFEST.sha256"):
        open(os.path.join(d, name), "w").write("{}")
    os.makedirs(os.path.join(d, "assets"))
    open(os.path.join(d, "assets", "x.txt"), "w").write("x")
    os.symlink("tokenizer.json", os.path.join(d, "tokenizer.model"))
    return d


def run(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


def tensors(path):
    with safe_open(path, "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


@pytest.fixture
def dirs(tmp_path):
    return {"src": model_dir(str(tmp_path / "k6"), 6, 1), "k4": model_dir(str(tmp_path / "k4"), 4, 2),
            "k3": model_dir(str(tmp_path / "k3"), 3, 3), "tmp": tmp_path}


def test_variant(dirs):
    src, dst = dirs["src"], str(dirs["tmp"] / "dst")
    r = run(src, dirs["k4"], dst)
    assert r.returncode == 0, r.stdout + r.stderr
    # every SRC file but the three rewritten ones, hardlinked (dirs and symlinks as symlinks to the same place)
    assert sorted(os.listdir(dst)) == sorted(set(os.listdir(src)) - {"MANIFEST.sha256"} | {NAME})
    for name in ("model-00001-of-00001.safetensors", "model-exl3-head.safetensors", "exl3_moe_plan.json"):
        assert os.stat(os.path.join(dst, name)).st_ino == os.stat(os.path.join(src, name)).st_ino, name
    assert os.path.realpath(os.path.join(dst, "assets")) == os.path.realpath(os.path.join(src, "assets"))
    assert os.path.realpath(os.path.join(dst, "tokenizer.model")) == os.path.realpath(os.path.join(src, "tokenizer.json"))
    # the drafter head: HEAD_SRC's tensors under the names the drafter loads, bit for bit
    new = tensors(os.path.join(dst, NAME))
    want = head(4, 2)
    assert sorted(new) == sorted("mtp." + n for n in want) and all(OWN_RE.match(n) for n in new)
    for n, v in want.items():
        assert new["mtp." + n].dtype == v.dtype and torch.equal(new["mtp." + n], v), n
    # the index: SRC's entries untouched, the new ones in the new file
    old_map = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    new_map = json.load(open(os.path.join(dst, "model.safetensors.index.json")))["weight_map"]
    assert {n: f for n, f in new_map.items() if n not in new} == old_map
    assert {n: f for n, f in new_map.items() if n in new} == {n: NAME for n in new}
    # config: drafter_head_k set, the target head and everything else as in SRC
    cs = json.load(open(os.path.join(src, "config.json")))
    cd = json.load(open(os.path.join(dst, "config.json")))
    assert cd["quantization_config"]["exl3_moe"].pop("drafter_head_k") == 4
    assert cd == cs and cs["quantization_config"]["exl3_moe"]["head_k"] == 6
    assert not os.path.exists(os.path.join(src, NAME))                  # SRC untouched
    assert "drafter head EXL3 K4" in r.stdout and "target head EXL3 K6 kept" in r.stdout


def test_replaces_a_drafter_head(dirs):
    v4, v3 = str(dirs["tmp"] / "v4"), str(dirs["tmp"] / "v3")
    assert run(dirs["src"], dirs["k4"], v4).returncode == 0
    r = run(v4, dirs["k3"], v3)
    assert r.returncode == 0, r.stdout + r.stderr
    new = tensors(os.path.join(v3, NAME))
    assert torch.equal(new["mtp.head.trellis"], head(3, 3)["head.trellis"])
    assert json.load(open(os.path.join(v3, "config.json")))["quantization_config"]["exl3_moe"]["drafter_head_k"] == 3
    m4 = json.load(open(os.path.join(v4, "model.safetensors.index.json")))["weight_map"]
    m3 = json.load(open(os.path.join(v3, "model.safetensors.index.json")))["weight_map"]
    assert m3 == m4                                                     # same names, same files
    assert os.stat(os.path.join(v3, NAME)).st_ino != os.stat(os.path.join(v4, NAME)).st_ino
    assert "was K4" in r.stdout


def test_bf16_target_head(tmp_path):
    src = model_dir(str(tmp_path / "bf16"), 0, 0, head_kind="bf16")
    k4 = model_dir(str(tmp_path / "k4"), 4, 2)
    r = run(src, k4, str(tmp_path / "dst"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "target head bf16 kept" in r.stdout


@pytest.mark.parametrize("case", ["k_mismatch", "vocab", "bf16_head_src", "exists", "stale"])
def test_refusals(dirs, case):
    src, hs, dst = dirs["src"], dirs["k4"], str(dirs["tmp"] / "dst")
    if case == "k_mismatch":                                            # config says K3, the trellis is K4
        hs = model_dir(str(dirs["tmp"] / "bad"), 4, 5, cfg_k=3)
        msg = "want (16, 32, 48) int16"
    elif case == "vocab":                                               # another model's head
        hs = model_dir(str(dirs["tmp"] / "bad"), 4, 5, vocab=2 * V)
        msg = "want (16, 32, 64) int16"
    elif case == "bf16_head_src":
        hs = model_dir(str(dirs["tmp"] / "bad"), 0, 5, head_kind="bf16")
        msg = "is not EXL3"
    elif case == "exists":
        os.makedirs(dst)
        msg = "exists"
    else:                                                               # mtp.head.* already mapped to another file
        idx = os.path.join(src, "model.safetensors.index.json")
        m = json.load(open(idx))
        m["weight_map"]["mtp.head.trellis"] = "model-00001-of-00001.safetensors"
        json.dump(m, open(idx, "w"))
        msg = "outside"
    r = run(src, hs, dst)
    assert r.returncode != 0 and msg in r.stdout + r.stderr, r.stdout + r.stderr
    assert case == "exists" or not os.path.exists(dst)
