# SPDX-License-Identifier: Apache-2.0
"""A model dir whose dense EXL3 linears and head come from another dense pass (p4_dense.py output), without copying:
every file of the source dir is hardlinked except its dense and head files; the dense pass's per-layer files are
hardlinked in as model-exl3-dense-LXX.safetensors and its head as model-exl3-head.safetensors; the index is rewritten;
config.json gets the new pass's K per module (exl3_moe.dense.k, read from its dense_LXX.json metrics) and head_k.
The DSpark drafter's EXL3 tensors (model-exl3-mtp.safetensors) and everything else stay as they are.

For the bytes variant of a recipe: the same experts with dense K4 (wo_a K5) and a K5 head, both from one dense pass
(run_dense_2box.sh OUT_ALT / K_ALT / KMAP_ALT / HEAD_K_ALT).

Checks before writing: the dense pass holds exactly the dense and head keys the source serves (same names, nothing
missing, nothing extra), every trellis has 16 * K words per tile at the K the new config gives its module, and each
module's K is the same in every layer. Same filesystem for the hardlinks; no GPU.

  python3 p4_dense_variant.py /models/dsv41-recipe-exl3-v3-dk5 /runs/dense_v3_k4 /models/dsv41-recipe-exl3-v3-dk5-b4
"""
import argparse
import glob
import json
import os
import re

from safetensors import safe_open

DENSE_KEY = re.compile(r"^layers\.(\d+)\.(.+)\.(trellis|suh|svh|mul1|su|sv)$")
HEAD_KEY = re.compile(r"^head\.(trellis|suh|svh|mul1|su|sv)$")


def module_k(kmap_keys, dense_dir):
    """K per config module prefix (e.g. attn.wo_a), from the dense pass's per-layer metrics; uniform across layers."""
    ks = {}
    for f in sorted(glob.glob(os.path.join(dense_dir, "dense_L*.json"))):
        m = json.load(open(f))
        for prefix in kmap_keys:
            vals = {v["k"] for name, v in m.items() if name == prefix or name.startswith(prefix + ".")}
            if not vals:
                continue
            if len(vals) > 1 or ks.get(prefix, next(iter(vals))) != next(iter(vals)):
                raise SystemExit(f"{prefix}: K differs across {os.path.basename(f)}'s modules or layers: {vals}, {ks.get(prefix)}")
            ks[prefix] = next(iter(vals))
    missing = [p for p in kmap_keys if p not in ks]
    if missing:
        raise SystemExit(f"no K in the dense pass metrics for {missing}")
    return ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dense")
    ap.add_argument("dst")
    a = ap.parse_args()
    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    wmap = index["weight_map"]
    dwm = json.load(open(os.path.join(a.dense, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    ex = cfg["quantization_config"]["exl3_moe"]
    old_k = ex["dense"]["k"]

    # the source's dense + head keys: layer-numbered non-expert EXL3 keys in model-exl3-* files outside the drafter's
    src_keys = {k for k, f in wmap.items() if f.startswith("model-exl3-") and f != "model-exl3-mtp.safetensors"
                and (HEAD_KEY.match(k) or (DENSE_KEY.match(k) and ".ffn.experts." not in k))}
    new_keys = set(dwm)
    if src_keys != new_keys:
        only_src, only_new = sorted(src_keys - new_keys), sorted(new_keys - src_keys)
        raise SystemExit(f"key sets differ: {len(only_src)} only in the source (e.g. {only_src[:3]}), "
                         f"{len(only_new)} only in the dense pass (e.g. {only_new[:3]})")
    old_files = {wmap[k] for k in src_keys}
    for f in old_files:   # the swapped files must hold nothing else (experts live in their own part files)
        other = [k for k, ff in wmap.items() if ff == f and k not in src_keys]
        if other:
            raise SystemExit(f"{f} also holds {len(other)} other tensors (e.g. {other[:2]}): not a dense-only file")

    new_k = module_k(list(old_k), a.dense)
    head_meta = os.path.join(a.dense, "dense_head.json")
    head_k = json.load(open(head_meta))["k"] if os.path.exists(head_meta) else ex.get("head_k")

    def want_k(key):
        if HEAD_KEY.match(key):
            return head_k
        mod = DENSE_KEY.match(key).group(2)
        hits = [p for p in new_k if mod == p or mod.startswith(p + ".")]
        if not hits:
            # modules the config groups under another name (wkv rides with wq_a, w3 with w1): same K as their group
            group = {"attn.wkv": "attn.wq_a", "ffn.shared_experts.w3": "ffn.shared_experts.w1"}.get(mod.split(".slice")[0])
            hits = [group] if group in new_k else []
        if not hits:
            raise SystemExit(f"{key}: no K for module {mod} in {sorted(new_k)}")
        return new_k[max(hits, key=len)]

    by_file = {}
    for k, f in dwm.items():
        by_file.setdefault(f, []).append(k)
    for f, keys in sorted(by_file.items()):
        with safe_open(os.path.join(a.dense, f), "pt") as sf:
            for k in keys:
                if k.endswith(".trellis"):
                    shape = sf.get_slice(k).get_shape()
                    if shape[-1] != 16 * want_k(k):
                        raise SystemExit(f"{f}: {k} trellis {shape} is not K{want_k(k)}")
    print(f"{len(new_keys)} dense + head tensors in {len(by_file)} files replace {len(old_files)} source files; "
          f"K {old_k} -> {new_k}, head K{ex.get('head_k')} -> K{head_k}", flush=True)

    os.makedirs(a.dst, exist_ok=False)
    for name in sorted(os.listdir(a.src)):
        s, d = os.path.join(a.src, name), os.path.join(a.dst, name)
        if name in old_files or name in ("model.safetensors.index.json", "config.json", "MANIFEST.sha256"):
            continue
        if os.path.islink(s):
            os.symlink(os.readlink(s), d)
        elif os.path.isdir(s):
            os.symlink(os.path.abspath(s), d)
        else:
            os.link(s, d)
    names = {}
    for f in by_file:
        m = re.match(r"^dense_L(\d+)\.safetensors$", f)
        names[f] = (f"model-exl3-dense-L{int(m.group(1)):02d}.safetensors" if m else
                    "model-exl3-head.safetensors" if f == "dense_head.safetensors" else f"model-exl3-{f}")
        os.link(os.path.join(a.dense, f), os.path.join(a.dst, names[f]))
    for k in src_keys:
        wmap[k] = names[dwm[k]]
    with open(os.path.join(a.dst, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    ex["dense"]["k"] = new_k
    ex["head_k"] = head_k
    with open(os.path.join(a.dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"variant {a.dst}: dense K {new_k}, head K{head_k}")


if __name__ == "__main__":
    main()
