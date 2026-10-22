# SPDX-License-Identifier: Apache-2.0
"""p1_complete.py on fake pass 1 run dirs (headers only, no torch): full runs, partial (--reuse) runs whose part files
hold only the re-encoded experts and whose jsons list the reused ones, a part cut before its json, a solo layer.
  python3 -m pytest -q scripts/quantize/test_p1_complete.py"""
import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "p1_complete.py")


def tensors(path, l, experts):
    hdr = {f"layers.{l}.ffn.experts.{e}.{p}.trellis": {"dtype": "I16", "shape": [1], "data_offsets": [0, 2]}
           for e in experts for p in ("w1", "w2", "w3")}
    b = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(b)) + b + b"\0\0")


def part(d, l, p, mine, encoded, json_too=True):
    name = f"layer_{l:02d}.part{p}"
    tensors(os.path.join(d, name + ".safetensors"), l, encoded)
    if json_too:
        ex = {str(e): ({"n": 1} if e in encoded else {"n": 1, "reused": True}) for e in mine}
        json.dump({"layer": l, "part": p, "experts": ex, "files": [name + ".safetensors"]},
                  open(os.path.join(d, name + ".json"), "w"))


def run(d, arg):
    r = subprocess.run([sys.executable, TOOL, d, arg], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def halves(l):
    return [e for e in range(384) if e % 2 == 0], [e for e in range(384) if e % 2 == 1]


def test_full_run(tmp_path):
    d = str(tmp_path)
    for l in range(3):
        for p, mine in enumerate(halves(l)):
            part(d, l, p, mine, mine)
    assert run(d, "--prefix") == (0, "2")
    assert run(d, "2")[0] == 0


def test_full_run_without_jsons_counts_tensors(tmp_path):   # the old rule still holds for a full run
    d = str(tmp_path)
    for p, mine in enumerate(halves(0)):
        part(d, 0, p, mine, mine, json_too=False)
    assert run(d, "--prefix") == (0, "0")


def test_partial_run_counts_reused_experts_from_the_jsons(tmp_path):
    d = str(tmp_path)
    for l in range(2):
        for p, mine in enumerate(halves(l)):
            part(d, l, p, mine, mine[:8])          # 8 re-encoded a part, the rest reused
    assert run(d, "--prefix") == (0, "1")
    assert run(d, "1") == (0, "layers 0-1: all 384 experts present")


def test_partial_layer_cut_before_its_json_is_incomplete(tmp_path):
    d = str(tmp_path)
    for p, mine in enumerate(halves(0)):
        part(d, 0, p, mine, mine[:8])
    a, b = halves(1)
    part(d, 1, 1, b, b[:8])
    part(d, 1, 0, a, a[:8], json_too=False)       # the stop cut part 0 after its tensors
    assert run(d, "--prefix") == (0, "0")
    rc, out = run(d, "1")
    assert rc == 1 and out.startswith("layer 1: ")


def test_solo_layer_one_file_all_experts(tmp_path):
    d = str(tmp_path)
    for p, mine in enumerate(halves(0)):
        part(d, 0, p, mine, mine[:8])
    part(d, 1, 0, list(range(384)), list(range(16)))   # --solo-from: part 0/1 holds every expert
    assert run(d, "--prefix") == (0, "1")


def test_reused_listing_does_not_cover_a_missing_part(tmp_path):
    d = str(tmp_path)
    a, _ = halves(0)
    part(d, 0, 0, a, a[:8])                       # part 1 never published
    assert run(d, "--prefix") == (0, "-1")
