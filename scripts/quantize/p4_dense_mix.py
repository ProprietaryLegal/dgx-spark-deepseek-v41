# SPDX-License-Identifier: Apache-2.0
"""A dense pass in p4_dense.py's layout, mixed per config module group from two passes of one Hessian capture (a run's
--out and --out-alt): the groups named by --from-b from pass B, every other module and the head from pass A. The result
is a dense dir for p4_dense_variant.py, e.g. the all-K5 pass with attn.wq_b from the K4 pass, to measure what one
group's K5 buys (KLD) and costs (bytes a round).

Groups are the config's exl3_moe.dense.k prefixes (attn.wq_a, attn.wq_b, attn.wo_a, attn.wo_b, ffn.shared_experts.w1,
ffn.shared_experts.w2). attn.wkv rides with attn.wq_a and ffn.shared_experts.w3 with ffn.shared_experts.w1 (the
config gives them their group's K), so they move with it. Every layer file is written new: A's tensors, the moved
modules' from B, and the per-layer metrics follow the tensors. The head files and the index come from A unchanged.

Checks before writing: A and B come from the same capture (each layer's metrics carry the same token counts), both
hold the same layer files with the same keys, a moved module's suh / svh shapes agree, and each moved group has one K
in B across layers. No GPU; ~3 GB of writes for the whole model at K5 (run as root in the image when the passes are
root-owned).

  python3 p4_dense_mix.py /runs/r1017/dense_k5 /runs/r1017/dense_k4 /runs/r1017/dense_mix_wqb4 --from-b attn.wq_b

--layers GROUP=L,L-L ... moves a group only in those layers (the rest of its layers stay A's): a per-layer K, which
the config carries as exl3_moe.dense.k_layers (p4_dense_variant.py writes it). --head-from-b takes B's head instead.
"""
import argparse
import glob
import json
import math
import os
import re

from safetensors import safe_open
from safetensors.torch import save_file

GROUPS = ("attn.wq_a", "attn.wq_b", "attn.wo_a", "attn.wo_b", "ffn.shared_experts.w1", "ffn.shared_experts.w2")
RIDERS = {"attn.wq_a": ("attn.wkv",), "ffn.shared_experts.w1": ("ffn.shared_experts.w3",)}
KEY = re.compile(r"^layers\.(\d+)\.(.+)\.(trellis|suh|svh|mul1|mcg|su|sv)$")
HEAD_FILES = ("dense_head.safetensors", "dense_head.json")
DTYPE_BYTES = {"I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4, "U8": 1, "I8": 1}


def in_group(mod, prefixes):
    return any(mod == p or mod.startswith(p + ".") for p in prefixes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="dense pass whose modules and head stay (e.g. the all-K5 pass)")
    ap.add_argument("b", help="dense pass the named groups come from (e.g. the K4 pass of the same capture)")
    ap.add_argument("dst")
    ap.add_argument("--from-b", nargs="+", required=True, metavar="GROUP")
    ap.add_argument("--layers", nargs="*", default=[], metavar="GROUP=L,L-L",
                    help="move a group only in these layers (default: every layer)")
    ap.add_argument("--head-from-b", action="store_true", help="B's head files instead of A's")
    a = ap.parse_args()
    only = {}
    for spec in a.layers:
        g, rng = spec.split("=")
        ls = set()
        for part in rng.split(","):
            lo, _, hi = part.partition("-")
            ls.update(range(int(lo), int(hi or lo) + 1))
        only[g] = ls
    if set(only) - set(a.from_b):
        raise SystemExit(f"--layers names groups not in --from-b: {sorted(set(only) - set(a.from_b))}")
    bad = [g for g in a.from_b if g not in GROUPS]
    if bad:
        raise SystemExit(f"unknown groups {bad}: the config's are {list(GROUPS)}")
    prefixes = [p for g in a.from_b for p in (g,) + RIDERS.get(g, ())]
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.a, "dense_L*.safetensors")))
    files_b = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.b, "dense_L*.safetensors")))
    if not files or files != files_b:
        raise SystemExit(f"layer files differ: {len(files)} in {a.a}, {len(files_b)} in {a.b}")
    if os.path.exists(a.dst):
        raise SystemExit(f"{a.dst} exists")

    ks_b, plan = {}, []
    for f in files:
        ma = json.load(open(os.path.join(a.a, f.replace(".safetensors", ".json"))))
        mb = json.load(open(os.path.join(a.b, f.replace(".safetensors", ".json"))))
        if ma.get("tokens") != mb.get("tokens"):
            raise SystemExit(f"{f}: token counts differ, not one capture: {ma.get('tokens')} vs {mb.get('tokens')}")
        layer = int(re.match(r"^dense_L(\d+)\.safetensors$", f).group(1))
        here = [p for g in a.from_b if g not in only or layer in only[g] for p in (g,) + RIDERS.get(g, ())]
        with safe_open(os.path.join(a.a, f), "pt") as sa, safe_open(os.path.join(a.b, f), "pt") as sb:
            if set(sa.keys()) != set(sb.keys()):
                raise SystemExit(f"{f}: key sets differ ({len(set(sa.keys()) ^ set(sb.keys()))} keys)")
            moved = sorted(k for k in sa.keys() if KEY.match(k) and in_group(KEY.match(k).group(2), here))
            for k in moved:
                if not k.endswith(".trellis") and sa.get_slice(k).get_shape() != sb.get_slice(k).get_shape():
                    raise SystemExit(f"{f}: {k} shape {sa.get_slice(k).get_shape()} vs {sb.get_slice(k).get_shape()}")
        for name, v in mb.items():
            if isinstance(v, dict) and "k" in v and in_group(name, here):
                g = next(p for p in prefixes if name == p or name.startswith(p + "."))
                ks_b.setdefault(g, set()).add(v["k"])
        if not moved and here:
            raise SystemExit(f"{f}: no tensors of {a.from_b}")
        plan.append((f, set(moved), ma, mb, here))
    mixed = {g: s for g, s in ks_b.items() if len(s) > 1}
    if mixed:
        raise SystemExit(f"a moved group has more than one K in {a.b}: {mixed}")
    print(f"{len(files)} layer files: {', '.join(prefixes)} from {a.b} (K {sorted({k for s in ks_b.values() for k in s})}), "
          f"the rest and the head from {a.a}", flush=True)

    os.makedirs(a.dst)
    before = after = 0
    for f, moved, ma, mb, here in plan:
        out = {}
        with safe_open(os.path.join(a.a, f), "pt") as sa, safe_open(os.path.join(a.b, f), "pt") as sb:
            for k in sa.keys():
                out[k] = (sb if k in moved else sa).get_tensor(k)
                if k in moved:
                    sl = sa.get_slice(k)
                    before += math.prod(sl.get_shape()) * DTYPE_BYTES[sl.get_dtype()]
                    after += out[k].nbytes
        p = os.path.join(a.dst, f)
        save_file(out, p + ".tmp")
        os.chmod(p + ".tmp", 0o644)
        os.replace(p + ".tmp", p)
        metrics = {n: (mb[n] if isinstance(v, dict) and "k" in v and in_group(n, here) else v) for n, v in ma.items()}
        with open(p.replace(".safetensors", ".json"), "w") as fh:
            json.dump(metrics, fh, indent=1)
    for f in HEAD_FILES:
        src = a.b if a.head_from_b else a.a
        if os.path.exists(os.path.join(src, f)):
            os.link(os.path.join(src, f), os.path.join(a.dst, f))
    index = json.load(open(os.path.join(a.a, "model.safetensors.index.json")))
    with open(os.path.join(a.dst, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"mixed dense pass {a.dst}: moved tensors {before / 1e9:.3f} GB -> {after / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
