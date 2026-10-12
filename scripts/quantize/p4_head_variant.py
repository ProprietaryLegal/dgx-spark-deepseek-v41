# SPDX-License-Identifier: Apache-2.0
"""A model-dir variant with another EXL3 lm_head (a dense_head.safetensors from p4_dense.py or p4_pass1.py --head):
every file of the source dir hardlinked except model-exl3-head.safetensors and config.json; the new head hardlinked in
under that name (same head.* tensor names, so the index stays valid); config.json's exl3_moe.head_k set to its K (read
from dense_head.json next to it). Checks: the same tensor names as the source's head, and each trellis at 16 * K words
per tile. Same filesystem for the hardlinks; no GPU (run as root in the quant image: the source files are root-owned).

  python3 p4_head_variant.py /models/dsv41-recipe-exl3-v3b-dk5 /runs/p1_v3s280/dense_head.safetensors /models/<variant>
"""
import argparse
import json
import os

from safetensors import safe_open

NAME = "model-exl3-head.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("head")
    ap.add_argument("dst")
    a = ap.parse_args()
    with open(os.path.join(os.path.dirname(os.path.abspath(a.head)), "dense_head.json")) as f:
        k = int(json.load(f)["k"])
    with safe_open(os.path.join(a.src, NAME), "pt") as f:
        old = set(f.keys())
    with safe_open(a.head, "pt") as f:
        new = set(f.keys())
        tr = f.get_slice("head.trellis").get_shape()
    if old != new:
        raise SystemExit(f"head tensor names differ: {sorted(old ^ new)}")
    if tr[-1] != 16 * k:
        raise SystemExit(f"head.trellis has {tr[-1]} words a tile, K{k} needs {16 * k}")
    os.makedirs(a.dst, exist_ok=False)
    for name in sorted(os.listdir(a.src)):
        s, d = os.path.join(a.src, name), os.path.join(a.dst, name)
        if name in (NAME, "config.json", "MANIFEST.sha256"):
            continue
        if os.path.isdir(s):
            os.symlink(os.path.abspath(s), d)
        else:
            os.link(s, d)
    os.link(os.path.abspath(a.head), os.path.join(a.dst, NAME))
    with open(os.path.join(a.src, "config.json")) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config") or cfg["text_config"]["quantization_config"]
    was = qc["exl3_moe"].get("head_k")
    qc["exl3_moe"]["head_k"] = k
    with open(os.path.join(a.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"variant {a.dst}: head K{was} -> K{k} ({len(new)} tensors)")


if __name__ == "__main__":
    main()
