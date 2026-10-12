# SPDX-License-Identifier: Apache-2.0
"""A model-dir variant whose DSpark drafter serves EXL3 dense linears (p4_drafter.py output): every file of the source
dir hardlinked, except the shards holding the converted FP8 drafter tensors, rewritten without them; the EXL3 tensors
added as model-exl3-mtp.safetensors; the index and config.json (exl3_moe.dense.drafter) updated. Runs on one node, no
GPU; ~10 GB of new files (three ~3 GB shards rewritten).

  python3 p4_drafter_variant.py /models/dsv41-recipe-exl3-v2.1 /runs/drafter/mtp_exl3.safetensors /models/<variant>
A source whose drafter is already partly EXL3 (e.g. v3-dk5, main_proj still FP8) works too: its model-exl3-mtp file and
index entries are replaced by the new file's (e.g. mtp_exl3_k5.safetensors with main_proj: one shard rewritten).
"""
import argparse
import json
import os
import re

from safetensors import safe_open
from safetensors.torch import save_file

CONVERTED = re.compile(r"^mtp\.(\d+)\.(attn\.(wq_a|wkv|wq_b|wo_b)|ffn\.shared_experts\.(w1|w2|w3)|main_proj)\.(weight|scale)$")


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
    # the modules the EXL3 file covers decide which FP8 tensors go
    covered = {re.sub(r"\.(trellis|suh|svh|mul1|mcg)$", "", n) for n in exl3_names}
    drop = {n for n in wmap if CONVERTED.match(n) and re.sub(r"\.(weight|scale)$", "", n) in covered}
    rewrite = sorted({wmap[n] for n in drop})
    os.makedirs(a.dst, exist_ok=False)
    for name in sorted(os.listdir(a.src)):
        s, d = os.path.join(a.src, name), os.path.join(a.dst, name)
        if name in rewrite or name in ("model.safetensors.index.json", "config.json", "MANIFEST.sha256",
                                       "model-exl3-mtp.safetensors"):    # a source that already has one: replaced
            continue
        if os.path.isdir(s):
            os.symlink(os.path.abspath(s), d)
        else:
            os.link(s, d)
    for shard in rewrite:
        keep = {}
        with safe_open(os.path.join(a.src, shard), "pt") as f:
            for n in f.keys():
                if n not in drop:
                    keep[n] = f.get_tensor(n)
        save_file(keep, os.path.join(a.dst, shard), metadata={"format": "pt"})
        print(f"{shard}: kept {len(keep)} tensors, dropped {sum(1 for n in drop if wmap[n] == shard)}", flush=True)
    os.link(os.path.abspath(a.exl3), os.path.join(a.dst, "model-exl3-mtp.safetensors"))
    for n in drop:
        del wmap[n]
    for n in [n for n, f in wmap.items() if f == "model-exl3-mtp.safetensors"]:   # the source's EXL3 drafter tensors
        del wmap[n]
    for n in exl3_names:
        wmap[n] = "model-exl3-mtp.safetensors"
    with open(os.path.join(a.dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=1)
    with open(os.path.join(a.src, "config.json")) as f:
        cfg = json.load(f)
    with open(a.exl3 + ".json") as f:
        ks = {k: v["k"] for k, v in json.load(f).items()}
    drafter = {}
    for ck, k in ks.items():                       # mtp.<i>.<module> -> the module's K (one K a module across layers)
        mod = ck.split(".", 2)[2]
        mod = {"attn.wkv": "attn.wq_a", "ffn.shared_experts.w3": "ffn.shared_experts.w1"}.get(mod, mod)
        drafter[mod] = int(k)
    qc = cfg.get("quantization_config") or cfg["text_config"]["quantization_config"]
    qc["exl3_moe"].setdefault("dense", {})["drafter"] = drafter
    with open(os.path.join(a.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"variant {a.dst}: {len(drop)} FP8 tensors replaced by {len(exl3_names)} EXL3 tensors, drafter K {drafter}")


if __name__ == "__main__":
    main()
