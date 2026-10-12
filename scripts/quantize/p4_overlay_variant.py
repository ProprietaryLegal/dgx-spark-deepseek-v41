# SPDX-License-Identifier: Apache-2.0
"""A model dir whose routed experts follow a new plan, from an assembled one plus a partial pass 1 (p4_pass1.py
--reuse): each model-exl3-LXX.safetensors with re-encoded experts is rewritten with those experts' tensors swapped in
(same key names, so the index does not change), exl3_moe_plan.json becomes the new plan, and every other file (other
layers, dense, head, drafter, official shards, tokenizer) is hardlinked. Checks before writing: the part files hold
exactly the experts whose K differs between the two plans, and every trellis has 16 * K words per tile at its new K.

Runs as root in a container when the source dir has root-owned files (hardlinks need ownership); no GPU. Writes
~4.6 GB per changed layer.

  python3 p4_overlay_variant.py /models/dsv41-recipe-exl3-v2.1-dk5 /runs/p1_v3 plan_v3.json /models/<variant>
"""
import argparse
import glob
import json
import os
import re
import shutil

from safetensors import safe_open
from safetensors.torch import load_file, save_file

EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(\w+)$")


def changed(old, new, l):
    o, n = old["layers"][str(l)], new["layers"][str(l)]
    od, nd = o.get("k_down", o["k"]), n.get("k_down", n["k"])
    return {e for e in range(len(n["k"])) if (n["k"][e], nd[e]) != (o["k"][e], od[e])}


def kval(c):
    return {"a": 1.5, "b": 2.5, "c": 3.5}.get(c) or float(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("pass1")
    ap.add_argument("plan")
    ap.add_argument("dst")
    a = ap.parse_args()
    old = json.load(open(os.path.join(a.src, "exl3_moe_plan.json")))
    new = json.load(open(a.plan))
    n_layers = len(new["layers"])
    updates = {}                                       # layer -> {key: tensor}
    for l in range(n_layers):
        want = changed(old, new, l)
        t = {}
        for f in sorted(glob.glob(os.path.join(a.pass1, f"layer_{l:02d}.part*.safetensors"))):
            t.update(load_file(f))
        have = {int(EXPERT.match(k).group(2)) for k in t}
        if have != want:
            raise SystemExit(f"layer {l}: part files hold {len(have)} experts, the plans differ on {len(want)} "
                             f"(missing {sorted(want - have)[:8]}, extra {sorted(have - want)[:8]})")
        for k, v in t.items():
            m = EXPERT.match(k)
            e, proj, piece = int(m.group(2)), m.group(3), m.group(4)
            if piece == "trellis":
                ks = new["layers"][str(l)]["k"] if proj != "w2" else new["layers"][str(l)].get("k_down",
                                                                                                new["layers"][str(l)]["k"])
                if v.shape[-1] != 16 * kval(ks[e]):
                    raise SystemExit(f"{k}: trellis {tuple(v.shape)} is not K{ks[e]}")
        if t:
            updates[l] = t
    print(f"{sum(len({int(EXPERT.match(k).group(2)) for k in t}) for t in updates.values())} experts re-encoded "
          f"in {len(updates)} of {n_layers} layers", flush=True)
    # which file holds which re-encoded tensor: model-exl3-LXX.safetensors (p4_assemble.py) or its 2-box parts
    # model-exl3-LXX-p*.safetensors (the v3 dirs); only the files holding a swapped tensor are rewritten
    swaps = {}                                         # file name -> {key: tensor}
    for l, t in updates.items():
        one = f"model-exl3-L{l:02d}.safetensors"
        files = [one] if os.path.exists(os.path.join(a.src, one)) else sorted(
            os.path.basename(f) for f in glob.glob(os.path.join(a.src, f"model-exl3-L{l:02d}-p*.safetensors")))
        if not files:
            raise SystemExit(f"layer {l}: no model-exl3-L{l:02d}[-p*].safetensors in {a.src}")
        left = dict(t)
        for name in files:
            with safe_open(os.path.join(a.src, name), "pt") as f:
                ks = set(f.keys())
            mine = {k: v for k, v in left.items() if k in ks}
            if mine:
                swaps[name] = mine
                for k in mine:
                    del left[k]
        if left:
            raise SystemExit(f"layer {l}: {len(left)} re-encoded tensors are in none of {files} ({next(iter(left))})")
    os.makedirs(a.dst, exist_ok=False)
    rewrite = set(swaps)
    for name in sorted(os.listdir(a.src)):
        s, d = os.path.join(a.src, name), os.path.join(a.dst, name)
        if name in rewrite or name in ("exl3_moe_plan.json", "MANIFEST.sha256"):
            continue
        if os.path.islink(s):
            os.symlink(os.readlink(s), d)
        elif os.path.isdir(s):
            os.symlink(os.path.abspath(s), d)
        else:
            os.link(s, d)
    for name, t in sorted(swaps.items()):
        src = os.path.join(a.src, name)
        with safe_open(src, "pt") as f:
            meta = f.metadata()
            tens = {k: f.get_tensor(k) for k in f.keys()}
        for k, v in t.items():
            if k not in tens:
                raise SystemExit(f"{k} not in {src}")
            tens[k] = v
        save_file(tens, os.path.join(a.dst, name + ".tmp"), metadata=meta)
        os.replace(os.path.join(a.dst, name + ".tmp"), os.path.join(a.dst, name))
        print(f"{name}: {len({int(EXPERT.match(k).group(2)) for k in t})} experts swapped", flush=True)
    shutil.copyfile(a.plan, os.path.join(a.dst, "exl3_moe_plan.json"))
    print(f"variant {a.dst}: plan {os.path.basename(a.plan)} {new.get('summary', {}).get('hist')}")


if __name__ == "__main__":
    main()
