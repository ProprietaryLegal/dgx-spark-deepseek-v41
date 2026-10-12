# SPDX-License-Identifier: Apache-2.0
"""A model-dir variant whose Engram projections (layers.<i>.engram.wkv) run on EXL3 (p4_engram.py output), by hardlinks
only: every file of the source dir hardlinked (V4.1's shards 47 / 48 hold the FP8 wkv next to the 101.5 GB Engram
embedding tables, so rewriting them is out: LOG Oct 11 12:40), the EXL3 tensors linked in as model-exl3-engram.safetensors,
the index extended with them and config.json given exl3_moe.dense.engram.wkv = K. The FP8 wkv tensors stay in their
shards and in the index: they still stream in at load, into the empty sink parameters the plugin's EXL3 engram.wkv method
registers (vllm_exl3_moe/dense.py, plugin of Oct 11 12:45 on; an older plugin fails the load, it does not misload).
One node, no GPU (run as root in the quant image: the source files are root-owned).

  python3 p4_engram_variant.py /models/dsv41-recipe-exl3-v3-dk5 /runs/engram/engram_exl3_k4.safetensors /models/<variant>
"""
import argparse
import json
import os
import re

from safetensors import safe_open

FP8 = re.compile(r"^layers\.(\d+)\.engram\.wkv\.(weight|scale)$")
NAME = "model-exl3-engram.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("exl3")
    ap.add_argument("dst")
    a = ap.parse_args()
    with open(os.path.join(a.src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    wmap = index["weight_map"]
    with safe_open(a.exl3, "pt") as f:
        exl3_names = list(f.keys())
    covered = {re.sub(r"\.(trellis|suh|svh|mul1|mcg)$", "", n) for n in exl3_names}
    fp8 = {re.sub(r"\.(weight|scale)$", "", n) for n in wmap if FP8.match(n)}
    if not covered or not covered <= fp8:
        raise SystemExit(f"the EXL3 file covers {sorted(covered)}, the source's FP8 engram.wkv are {sorted(fp8)}")
    if any(n in wmap for n in exl3_names):
        raise SystemExit("the source already maps EXL3 engram tensors")
    os.makedirs(a.dst, exist_ok=False)
    for name in sorted(os.listdir(a.src)):
        s, d = os.path.join(a.src, name), os.path.join(a.dst, name)
        if name in ("model.safetensors.index.json", "config.json", "MANIFEST.sha256", NAME):
            continue
        if os.path.isdir(s):
            os.symlink(os.path.abspath(s), d)
        else:
            os.link(s, d)
    os.link(os.path.abspath(a.exl3), os.path.join(a.dst, NAME))
    for n in exl3_names:
        wmap[n] = NAME
    with open(os.path.join(a.dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=1)
    with open(os.path.join(a.src, "config.json")) as f:
        cfg = json.load(f)
    with open(a.exl3 + ".json") as f:
        ks = {int(v["k"]) for v in json.load(f).values()}
    if len(ks) != 1:
        raise SystemExit(f"one K for every engram.wkv, got {sorted(ks)}")
    qc = cfg.get("quantization_config") or cfg["text_config"]["quantization_config"]
    qc["exl3_moe"].setdefault("dense", {})["engram"] = {"wkv": ks.pop()}
    with open(os.path.join(a.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"variant {a.dst}: {len(exl3_names)} EXL3 tensors added for {sorted(covered)} (FP8 kept in place, hardlinks "
          f"only), engram {qc['exl3_moe']['dense']['engram']}")


if __name__ == "__main__":
    main()
