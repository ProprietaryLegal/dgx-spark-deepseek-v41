# SPDX-License-Identifier: Apache-2.0
"""A model dir whose routed experts follow a new plan, from an assembled one plus a partial pass 1 (p4_pass1.py
--reuse): each model-exl3-LXX.safetensors with re-encoded experts is rewritten with those experts' tensors swapped in
(same key names, so the index does not change), exl3_moe_plan.json becomes the new plan, and every other file (other
layers, dense, head, drafter, official shards, tokenizer) is hardlinked. Checks before writing: the part files hold
exactly the experts whose K differs between the two plans, and every trellis has 16 * K words per tile at its new K.

--extra DIR (repeatable): more assembled dirs (their exl3_moe_plan.json says each expert's K). An expert whose new K
differs from src's comes from the part files when they hold it at that K, else from the first extra dir whose plan has
it there; the part files may then hold experts the new plan does not take (one pass 1 serving several plans, e.g. the
3.05 and 3.10 decode caps from one 3.10 run), and an expert at a K that nothing holds stops the build.

Runs as root in a container when the source dir has root-owned files (hardlinks need ownership); no GPU. Writes
~4.6 GB per changed layer.

  python3 p4_overlay_variant.py /models/dsv41-recipe-exl3-v2.1-dk5 /runs/p1_v3 plan_v3.json /models/<variant>
  python3 p4_overlay_variant.py /models/q280r /runs/p1_q310 plan_305.json /models/<variant> --extra /models/1017-asm
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


def kpair(p, l, e):
    x = p["layers"][str(l)]
    return x["k"][e], x.get("k_down", x["k"])[e]


def trellis_k_ok(t, l, e, pair):
    """True when the part tensors of expert e hold every projection's trellis at K `pair` (gate/up, down)."""
    pre = f"layers.{l}.ffn.experts.{e}."
    for proj, k in (("w1", pair[0]), ("w3", pair[0]), ("w2", pair[1])):
        v = t.get(pre + proj + ".trellis")
        if v is None or v.shape[-1] != 16 * kval(k):
            return False
    return True


def layer_file(d, l):
    """Layer l's expert file(s) in an assembled dir: model-exl3-LXX.safetensors or its 2-box parts."""
    one = os.path.join(d, f"model-exl3-L{l:02d}.safetensors")
    if os.path.exists(one):
        return [one]
    parts = sorted(glob.glob(os.path.join(d, f"model-exl3-L{l:02d}-p*.safetensors")))
    if not parts:
        raise SystemExit(f"layer {l}: no model-exl3-L{l:02d}[-p*].safetensors in {d}")
    return parts


def expert_tensors(d, l, e):
    """{key: tensor} of expert e of layer l from an assembled dir."""
    pre = f"layers.{l}.ffn.experts.{e}."
    out = {}
    for f in layer_file(d, l):
        with safe_open(f, "pt") as g:
            for k in g.keys():
                if k.startswith(pre):
                    out[k] = g.get_tensor(k)
    if {EXPERT.match(k).group(3) for k in out} != {"w1", "w2", "w3"}:
        raise SystemExit(f"layer {l} expert {e}: incomplete in {d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("pass1")
    ap.add_argument("plan")
    ap.add_argument("dst")
    ap.add_argument("--extra", action="append", default=[], help="another assembled dir to take experts from")
    a = ap.parse_args()
    old = json.load(open(os.path.join(a.src, "exl3_moe_plan.json")))
    new = json.load(open(a.plan))
    extras = [(d, json.load(open(os.path.join(d, "exl3_moe_plan.json")))) for d in a.extra]
    n_layers = len(new["layers"])
    updates = {}                                       # layer -> {key: tensor}
    from_extra = 0
    for l in range(n_layers):
        want = changed(old, new, l)
        t = {}
        for f in sorted(glob.glob(os.path.join(a.pass1, f"layer_{l:02d}.part*.safetensors"))):
            t.update(load_file(f))
        have = {int(EXPERT.match(k).group(2)) for k in t}
        if extras:                                     # each changed expert from the part files or an extra dir
            take, missing = set(), []
            for e in sorted(want):
                pair = kpair(new, l, e)
                if e in have and trellis_k_ok(t, l, e, pair):
                    take.add(e)
                    continue
                d = next((d for d, p in extras if kpair(p, l, e) == pair), None)
                if d is None:
                    missing.append(e)
                    continue
                t.update(expert_tensors(d, l, e))
                take.add(e)
                from_extra += 1
            if missing:
                raise SystemExit(f"layer {l}: {len(missing)} experts at a K neither the part files nor --extra hold "
                                 f"({missing[:8]})")
            t = {k: v for k, v in t.items() if int(EXPERT.match(k).group(2)) in take}
            have = take
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
    print(f"{sum(len({int(EXPERT.match(k).group(2)) for k in t}) for t in updates.values())} experts swapped "
          f"in {len(updates)} of {n_layers} layers ({from_extra} from --extra dirs)", flush=True)
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
