# SPDX-License-Identifier: Apache-2.0
"""p4_dense_mix.py on tiny fake dense passes (CPU only): the named groups and their riders come from pass B with B's
metrics, everything else and the head from pass A; p4_dense_variant.py then builds a model dir whose config gives the
moved groups B's K. It refuses an unknown group, passes from two captures, differing layer files or keys, a moved
module whose shapes differ and an existing DST.
  python3 -m pytest scripts/quantize/test_dense_mix.py -q
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
MIX = os.path.join(HERE, "p4_dense_mix.py")
VARIANT = os.path.join(HERE, "p4_dense_variant.py")
D = 64                                                   # every matrix D x D: whole 16-wide tiles
MODULES = ["attn.wq_a", "attn.wkv", "attn.wq_b", "attn.wo_a.slice.0", "attn.wo_a.slice.1", "attn.wo_b",
           "ffn.shared_experts.w1", "ffn.shared_experts.w3", "ffn.shared_experts.w2"]
K_A = {m: 5 for m in MODULES}
K_B = {m: (5 if m.startswith("attn.wo_a") else 4) for m in MODULES}
CFG_K = {"attn.wo_a": 5, "attn.wo_b": 5, "attn.wq_a": 5, "attn.wq_b": 5, "ffn.shared_experts.w1": 5,
         "ffn.shared_experts.w2": 5}


def exl3(prefix, k, seed, d=D):
    g = torch.Generator().manual_seed(seed)
    return {f"{prefix}.trellis": torch.randint(-32768, 32768, (d // 16, D // 16, 16 * k), dtype=torch.int16, generator=g),
            f"{prefix}.suh": torch.randn(d, generator=g).half(), f"{prefix}.svh": torch.randn(D, generator=g).half(),
            f"{prefix}.mul1": torch.tensor(seed, dtype=torch.int32)}


def dense_pass(d, ks, head_k, seed, layers=2, tokens=1000, skip=None):
    os.makedirs(d)
    wm = {}
    for l in range(layers):
        t, metrics = {}, {"tokens": {"attn.wqa_wkv": tokens}, "forward_s": 1.0, "encode_s": 2.0}
        for i, m in enumerate(MODULES):
            if (l, m) == skip:
                continue
            t.update(exl3(f"layers.{l}.{m}", ks[m], seed + 100 * l + i))
            metrics[m] = {"k": ks[m], "rel_err": 0.01 * ks[m] + l}
        f = f"dense_L{l:02d}.safetensors"
        save_file(t, os.path.join(d, f))
        json.dump(metrics, open(os.path.join(d, f"dense_L{l:02d}.json"), "w"))
        wm.update({k: f for k in t})
    save_file(exl3("head", head_k, seed + 9), os.path.join(d, "dense_head.safetensors"))
    json.dump({"k": head_k, "rel_err": 0.01}, open(os.path.join(d, "dense_head.json"), "w"))
    wm.update({k: "dense_head.safetensors" for k in exl3("head", head_k, 0)})
    json.dump({"metadata": {"format": "exl3 dense linears (p4_dense.py)"}, "weight_map": wm},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))


@pytest.fixture
def passes(tmp_path):
    a, b = str(tmp_path / "dense_k5"), str(tmp_path / "dense_k4")
    dense_pass(a, K_A, 6, 1)
    dense_pass(b, K_B, 5, 2)
    return tmp_path, a, b


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True)


def tensors(path):
    with safe_open(path, "pt") as sf:
        return {k: sf.get_tensor(k) for k in sf.keys()}


def test_moves_groups_and_riders(passes):
    tmp, a, b = passes
    dst = str(tmp / "mix")
    r = run(MIX, a, b, dst, "--from-b", "attn.wq_a", "ffn.shared_experts.w1")
    assert r.returncode == 0, r.stderr
    moved = {"attn.wq_a", "attn.wkv", "ffn.shared_experts.w1", "ffn.shared_experts.w3"}
    for l in range(2):
        f = f"dense_L{l:02d}.safetensors"
        got, ta, tb = tensors(os.path.join(dst, f)), tensors(os.path.join(a, f)), tensors(os.path.join(b, f))
        assert set(got) == set(ta)
        for k, v in got.items():
            mod = k.split(".", 2)[2].rsplit(".", 1)[0]
            assert torch.equal(v, (tb if mod in moved else ta)[k]), k
        m = json.load(open(os.path.join(dst, f"dense_L{l:02d}.json")))
        mb, ma = json.load(open(os.path.join(b, f"dense_L{l:02d}.json"))), json.load(open(os.path.join(a, f"dense_L{l:02d}.json")))
        for mod in MODULES:
            assert m[mod] == (mb if mod in moved else ma)[mod], mod
        assert m["tokens"] == ma["tokens"] and m["encode_s"] == ma["encode_s"]
    assert os.stat(os.path.join(dst, "dense_head.safetensors")).st_ino == os.stat(os.path.join(a, "dense_head.safetensors")).st_ino
    assert json.load(open(os.path.join(dst, "model.safetensors.index.json"))) == json.load(open(os.path.join(a, "model.safetensors.index.json")))


def test_dense_variant_takes_the_mix(passes):
    tmp, a, b = passes
    # a source model dir like q310r: its dense and head files are the all-K5 pass's, plus an expert part file
    src = str(tmp / "q310r")
    os.makedirs(src)
    wm = {}
    for f in sorted(os.listdir(a)):
        if f.startswith("dense_L") and f.endswith(".safetensors"):
            n = f"model-exl3-dense-L{int(f[7:9]):02d}.safetensors"
        elif f == "dense_head.safetensors":
            n = "model-exl3-head.safetensors"
        else:
            continue
        os.link(os.path.join(a, f), os.path.join(src, n))
        with safe_open(os.path.join(src, n), "pt") as sf:
            wm.update({k: n for k in sf.keys()})
    save_file(exl3("layers.0.ffn.experts.0.w1", 3, 7), os.path.join(src, "model-exl3-L00-p0.safetensors"))
    wm.update({k: "model-exl3-L00-p0.safetensors" for k in exl3("layers.0.ffn.experts.0.w1", 3, 0)})
    json.dump({"weight_map": wm}, open(os.path.join(src, "model.safetensors.index.json"), "w"))
    json.dump({"quantization_config": {"exl3_moe": {"dense": {"k": dict(CFG_K), "layers": [0, 1]}, "head_k": 6}}},
              open(os.path.join(src, "config.json"), "w"))
    mix, dst = str(tmp / "mix"), str(tmp / "q310r-wqb4")
    assert run(MIX, a, b, mix, "--from-b", "attn.wq_b").returncode == 0
    r = run(VARIANT, src, mix, dst)
    assert r.returncode == 0, r.stderr + r.stdout
    ex = json.load(open(os.path.join(dst, "config.json")))["quantization_config"]["exl3_moe"]
    assert ex["dense"]["k"] == {**CFG_K, "attn.wq_b": 4} and ex["head_k"] == 6
    got = tensors(os.path.join(dst, "model-exl3-dense-L01.safetensors"))
    assert torch.equal(got["layers.1.attn.wq_b.trellis"], tensors(os.path.join(b, "dense_L01.safetensors"))["layers.1.attn.wq_b.trellis"])
    assert torch.equal(got["layers.1.attn.wo_b.trellis"], tensors(os.path.join(a, "dense_L01.safetensors"))["layers.1.attn.wo_b.trellis"])
    assert os.stat(os.path.join(dst, "model-exl3-L00-p0.safetensors")).st_ino == os.stat(os.path.join(src, "model-exl3-L00-p0.safetensors")).st_ino


def test_per_layer_mix_and_variant(passes):
    """--layers moves a group only in the named layers; --head-from-b takes B's head; the variant config carries the
    per-layer K in dense.k_layers and checks each trellis at its own layer's K."""
    tmp, a, b = passes
    mix = str(tmp / "mixl")
    r = run(MIX, a, b, mix, "--from-b", "attn.wq_b", "attn.wo_b", "ffn.shared_experts.w1",
            "--layers", "attn.wo_b=1", "ffn.shared_experts.w1=1", "--head-from-b")
    assert r.returncode == 0, r.stderr
    for l in range(2):
        f = f"dense_L{l:02d}.safetensors"
        got, ta, tb = tensors(os.path.join(mix, f)), tensors(os.path.join(a, f)), tensors(os.path.join(b, f))
        moved = {"attn.wq_b"} | ({"attn.wo_b", "ffn.shared_experts.w1", "ffn.shared_experts.w3"} if l == 1 else set())
        for k, v in got.items():
            mod = k.split(".", 2)[2].rsplit(".", 1)[0]
            assert torch.equal(v, (tb if mod in moved else ta)[k]), (l, k)
        m = json.load(open(os.path.join(mix, f"dense_L{l:02d}.json")))
        assert m["attn.wo_b"]["k"] == (4 if l == 1 else 5) and m["ffn.shared_experts.w3"]["k"] == (4 if l == 1 else 5)
    assert os.stat(os.path.join(mix, "dense_head.safetensors")).st_ino == os.stat(os.path.join(b, "dense_head.safetensors")).st_ino
    src = str(tmp / "src")
    os.makedirs(src)
    wm = {}
    for f in sorted(os.listdir(a)):
        if f.startswith("dense_L") and f.endswith(".safetensors"):
            nm = f"model-exl3-dense-L{int(f[7:9]):02d}.safetensors"
        elif f == "dense_head.safetensors":
            nm = "model-exl3-head.safetensors"
        else:
            continue
        os.link(os.path.join(a, f), os.path.join(src, nm))
        with safe_open(os.path.join(src, nm), "pt") as sf:
            wm.update({k: nm for k in sf.keys()})
    json.dump({"weight_map": wm}, open(os.path.join(src, "model.safetensors.index.json"), "w"))
    json.dump({"quantization_config": {"exl3_moe": {"dense": {"k": dict(CFG_K), "layers": [0, 1]}, "head_k": 6}}},
              open(os.path.join(src, "config.json"), "w"))
    dst = str(tmp / "src-late4")
    r = run(VARIANT, src, mix, dst)
    assert r.returncode == 0, r.stderr + r.stdout
    ex = json.load(open(os.path.join(dst, "config.json")))["quantization_config"]["exl3_moe"]
    assert ex["dense"]["k"]["attn.wq_b"] == 4 and ex["head_k"] == 5
    # two layers, one moved: the most common K is a tie, either base is consistent with its overrides
    kl = ex["dense"].get("k_layers", {})
    for mod in ("attn.wo_b", "ffn.shared_experts.w1"):
        k_of = {l: int(kl.get(mod, {}).get(str(l), ex["dense"]["k"][mod])) for l in range(2)}
        assert k_of == {0: 5, 1: 4}, (mod, ex["dense"])
    assert "attn.wo_a" not in kl and ex["dense"]["k"]["ffn.shared_experts.w2"] == 5


def test_refusals(passes, tmp_path):
    tmp, a, b = passes
    r = run(MIX, a, b, str(tmp / "x1"), "--from-b", "attn.wkv")
    assert r.returncode != 0 and "unknown groups" in r.stderr + r.stdout
    other = str(tmp / "other_capture")
    dense_pass(other, K_B, 5, 3, tokens=999)
    r = run(MIX, a, other, str(tmp / "x2"), "--from-b", "attn.wq_b")
    assert r.returncode != 0 and "not one capture" in r.stderr + r.stdout
    short = str(tmp / "short")
    dense_pass(short, K_B, 5, 4, layers=1)
    r = run(MIX, a, short, str(tmp / "x3"), "--from-b", "attn.wq_b")
    assert r.returncode != 0 and "layer files differ" in r.stderr + r.stdout
    holed = str(tmp / "holed")
    dense_pass(holed, K_B, 5, 5, skip=(1, "attn.wo_b"))
    r = run(MIX, a, holed, str(tmp / "x4"), "--from-b", "attn.wq_b")
    assert r.returncode != 0 and "key sets differ" in r.stderr + r.stdout
    wide = str(tmp / "wide")
    dense_pass(wide, K_B, 5, 6)
    f = os.path.join(wide, "dense_L00.safetensors")
    t = tensors(f)
    t["layers.0.attn.wq_b.suh"] = torch.zeros(D + 16).half()
    save_file(t, f)
    r = run(MIX, a, wide, str(tmp / "x5"), "--from-b", "attn.wq_b")
    assert r.returncode != 0 and "shape" in r.stderr + r.stdout
    os.makedirs(str(tmp / "taken"))
    r = run(MIX, a, b, str(tmp / "taken"), "--from-b", "attn.wq_b")
    assert r.returncode != 0 and "exists" in r.stderr + r.stdout
    for d in ("x1", "x2", "x3", "x4", "x5"):
        assert not os.path.exists(str(tmp / d)), d
