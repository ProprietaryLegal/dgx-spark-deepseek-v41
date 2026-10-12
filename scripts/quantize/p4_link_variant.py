# SPDX-License-Identifier: Apache-2.0
"""A model dir that serves a complete new pass-1 encode without copying it: every pass-1 part file is hardlinked in as
model-exl3-LXX-pN.safetensors (the plugin and vLLM find tensors through the index, not by file name), the dense EXL3
tensors that p4_assemble.py had merged into the old model-exl3-LXX files are collected into one
model-exl3-dense.safetensors (~4 GB), the index is rewritten to match, exl3_moe_plan.json becomes the new plan, and every
other file of the source dir (official shards, head, drafter, tokenizer) is hardlinked. ~4 GB of new data instead of
p4_assemble.py's ~185 GB; the part files must stay on the same filesystem (hardlinks share them).

Checks before writing: each layer's part files hold all experts, every trellis has 16 * K words per tile at the plan's K,
and the old files' dense keys all land in the dense file.

  python3 p4_link_variant.py /models/dsv41-recipe-exl3-v2.1-dk5 /runs/p1_v3 plan_v3.json /models/<variant>
Runs as root in a container when files are root-owned (hardlinks need ownership); no GPU.
"""
import argparse
import glob
import json
import os
import re
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(\w+)$")
OLD_LAYER_FILE = re.compile(r"^model-exl3-L(\d+)\.safetensors$")


def kval(c):
    return {"a": 1.5, "b": 2.5, "c": 3.5}.get(c) or float(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("pass1")
    ap.add_argument("plan")
    ap.add_argument("dst")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    n_layers, n_exp = len(plan["layers"]), plan.get("num_experts", 384)
    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    wmap = index["weight_map"]
    links, new_map = {}, {}                            # dst name -> src path; key -> dst file
    for l in range(n_layers):
        parts = sorted(glob.glob(os.path.join(a.pass1, f"layer_{l:02d}.part*.safetensors")))
        seen = set()
        ks = plan["layers"][str(l)]["k"]
        kd = plan["layers"][str(l)].get("k_down", ks)
        for p in parts:
            name = f"model-exl3-L{l:02d}-p{re.search(r'part(\d+)', p).group(1)}.safetensors"
            links[name] = p
            with safe_open(p, "pt") as f:
                for k in f.keys():
                    m = EXPERT.match(k)
                    if not m or int(m.group(1)) != l:
                        raise SystemExit(f"{p}: unexpected tensor {k}")
                    e, proj, piece = int(m.group(2)), m.group(3), m.group(4)
                    seen.add(e)
                    if piece == "trellis":
                        want = 16 * kval((kd if proj == "w2" else ks)[e])
                        if f.get_slice(k).get_shape()[-1] != want:
                            raise SystemExit(f"{k}: trellis {f.get_slice(k).get_shape()} is not K{(kd if proj == 'w2' else ks)[e]}")
                    new_map[k] = name
        if seen != set(range(n_exp)):
            raise SystemExit(f"layer {l}: part files hold {len(seen)} experts, need {n_exp}")
    dense, old_files = {}, set()
    for k, f in wmap.items():
        if OLD_LAYER_FILE.match(f):
            old_files.add(f)
            if not EXPERT.match(k):
                dense.setdefault(f, []).append(k)
    stale = [k for k, f in wmap.items() if OLD_LAYER_FILE.match(f) and EXPERT.match(k) and k not in new_map]
    if stale:
        raise SystemExit(f"{len(stale)} expert keys of the source have no new tensor, e.g. {stale[:3]}")
    print(f"{len(new_map):,} expert tensors in {len(links)} part files; {sum(map(len, dense.values()))} dense tensors "
          f"from {len(old_files)} old layer files", flush=True)
    os.makedirs(a.dst, exist_ok=False)
    for name in sorted(os.listdir(a.src)):
        s, d = os.path.join(a.src, name), os.path.join(a.dst, name)
        if name in old_files or name in ("model.safetensors.index.json", "exl3_moe_plan.json", "MANIFEST.sha256"):
            continue
        if os.path.islink(s):
            os.symlink(os.readlink(s), d)
        elif os.path.isdir(s):
            os.symlink(os.path.abspath(s), d)
        else:
            os.link(s, d)
    for name, p in links.items():
        os.link(p, os.path.join(a.dst, name))
    tens = {}
    for f, keys in sorted(dense.items()):
        with safe_open(os.path.join(a.src, f), "pt") as sf:
            for k in keys:
                tens[k] = sf.get_tensor(k)
    if tens:
        save_file(tens, os.path.join(a.dst, "model-exl3-dense.safetensors.tmp"),
                  metadata={"format": "pt", "quant": "exl3", "encoder": "exllamav3 1.6.0"})
        os.replace(os.path.join(a.dst, "model-exl3-dense.safetensors.tmp"),
                   os.path.join(a.dst, "model-exl3-dense.safetensors"))
    for k in [k for k, f in wmap.items() if f in old_files]:
        del wmap[k]
    wmap.update(new_map)
    wmap.update({k: "model-exl3-dense.safetensors" for k in tens})
    with open(os.path.join(a.dst, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    shutil.copyfile(a.plan, os.path.join(a.dst, "exl3_moe_plan.json"))
    print(f"variant {a.dst}: plan {os.path.basename(a.plan)} {plan.get('summary', {}).get('hist')}, "
          f"{len(tens)} dense tensors in model-exl3-dense.safetensors")


if __name__ == "__main__":
    main()
