# SPDX-License-Identifier: Apache-2.0
"""A model-dir variant whose DSpark drafter projects with its own, lighter EXL3 lm_head (patch 0026 with
VLLM_DSPARK_OWN_HEAD=1): every file of SRC hardlinked, plus model-exl3-mtp-head.safetensors holding HEAD_SRC's head
tensors (head.trellis / suh / svh / mul1 of another build, e.g. the shipping K4 head) renamed mtp.head.*, the index
extended with them and config.json given exl3_moe.drafter_head_k = that head's K (HEAD_SRC's exl3_moe.head_k). The
target keeps SRC's head (head.*), so target outputs do not change. Without the flag, or on an image without patch 0026,
the drafter skips mtp.head.* and shares the target's head: the variant then serves exactly as SRC does.

Checks before writing: HEAD_SRC's head is EXL3 with head_k set, its trellis int16 at 16 * K words a tile, its geometry
(hidden, vocab) that of SRC's head (EXL3 or BF16), and SRC's own mtp.head.* tensors, if it has a drafter head already,
live only in model-exl3-mtp-head.safetensors, which is then replaced (vLLM reads every tensor of every file the index
lists, so a stale copy in another file would load beside the new one). New data: the full head, ~331 MB at K4 (each
rank keeps half); every other file is a hardlink (same filesystem). One node, no GPU (run as root in the quant image
when the source files are root-owned).

  python3 p4_drafter_head_variant.py /models/dsv41-recipe-exl3-1017q280r-dk5-dx3 \
      /models/dsv41-recipe-exl3-1017b-dk5-h4-dx3 /models/dsv41-recipe-exl3-1017q280r-dk5-dx3-dh4
"""
import argparse
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NAME = "model-exl3-mtp-head.safetensors"
INDEX = "model.safetensors.index.json"
HEAD = re.compile(r"^head\.(trellis|suh|svh|mul1|mcg)$")      # the target's EXL3 head pieces (patch 0007)
OWN = "mtp.head."                                             # the drafter's own head (patch 0026's mtp.head.*)


def load_json(d, name):
    with open(os.path.join(d, name)) as f:
        return json.load(f)


def exl3_settings(cfg):
    """config.json's quantization_config.exl3_moe block (text_config's on a VL config), {} when there is none."""
    qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config") or {}
    return qc.get("exl3_moe") or {}


def head_geometry(d, wmap):
    """(hidden, vocab) of model dir d's target head, from head.trellis (EXL3) or head.weight (BF16)."""
    for name in ("head.trellis", "head.weight"):
        if name in wmap:
            with safe_open(os.path.join(d, wmap[name]), "pt") as f:
                s = f.get_slice(name).get_shape()
            return (s[0] * 16, s[1] * 16) if name == "head.trellis" else (s[1], s[0])
    raise SystemExit(f"{d}: the index has neither head.trellis nor head.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="model dir served as the target (keeps its own head)")
    ap.add_argument("head_src", help="model dir whose EXL3 head becomes the drafter's")
    ap.add_argument("dst")
    a = ap.parse_args()
    if os.path.lexists(a.dst):
        raise SystemExit(f"{a.dst} exists: refusing to overwrite")
    index = load_json(a.src, INDEX)
    wmap = index["weight_map"]
    cfg = load_json(a.src, "config.json")
    if not exl3_settings(cfg):
        raise SystemExit(f"{a.src}: config.json has no quantization_config.exl3_moe (the drafter head needs the "
                         f"vllm_exl3_moe plugin)")
    hs = exl3_settings(load_json(a.head_src, "config.json"))
    if hs.get("head") != "exl3" or not hs.get("head_k"):
        raise SystemExit(f"{a.head_src}: its head is not EXL3 with a head_k (exl3_moe.head {hs.get('head')!r}, "
                         f"head_k {hs.get('head_k')!r})")
    k = int(hs["head_k"])
    hmap = load_json(a.head_src, INDEX)["weight_map"]
    pieces = {HEAD.match(n).group(1): f for n, f in hmap.items() if HEAD.match(n)}
    if not {"trellis", "suh", "svh"} <= set(pieces) or not {"mul1", "mcg"} & set(pieces):
        raise SystemExit(f"{a.head_src}: head pieces {sorted(pieces)} in its index, want trellis, suh, svh and "
                         f"mul1 or mcg")
    t = {}
    for p, fn in sorted(pieces.items()):
        with safe_open(os.path.join(a.head_src, fn), "pt") as f:
            t[OWN + p] = f.get_tensor("head." + p).contiguous()
    hidden, vocab = head_geometry(a.src, wmap)
    tr = t[OWN + "trellis"]
    want = (hidden // 16, vocab // 16, 16 * k)
    if tr.dtype != torch.int16 or tuple(tr.shape) != want:
        raise SystemExit(f"{a.head_src}: head.trellis {tuple(tr.shape)} {tr.dtype}, want {want} int16 (SRC's head "
                         f"is hidden {hidden} x vocab {vocab}; K{k} is 16 * K = {16 * k} words a tile)")
    for p, n in (("suh", hidden), ("svh", vocab)):
        if tuple(t[OWN + p].shape) != (n,):
            raise SystemExit(f"{a.head_src}: head.{p} {tuple(t[OWN + p].shape)}, want ({n},)")
    old = {n: f for n, f in wmap.items() if n.startswith(OWN)}
    if any(f != NAME for f in old.values()) or any(f == NAME and not n.startswith(OWN) for n, f in wmap.items()):
        raise SystemExit(f"{a.src}: its index maps mtp.head.* outside {NAME} (or other tensors into it): only "
                         f"{NAME} holding the drafter head alone can be replaced")
    os.makedirs(a.dst)
    for name in sorted(os.listdir(a.src)):
        s, d = os.path.join(a.src, name), os.path.join(a.dst, name)
        if name in (NAME, INDEX, "config.json", "MANIFEST.sha256"):     # a source's drafter head is replaced
            continue
        if os.path.islink(s):
            os.symlink(os.path.realpath(s), d)
        elif os.path.isdir(s):
            os.symlink(os.path.abspath(s), d)
        else:
            os.link(s, d)
    tmp = os.path.join(a.dst, NAME + ".tmp")
    save_file(t, tmp, metadata={"format": "pt", "quant": "exl3", "drafter_head_k": str(k),
                                "head_from": os.path.abspath(a.head_src)})
    os.chmod(tmp, 0o644)
    os.replace(tmp, os.path.join(a.dst, NAME))
    for n in old:
        del wmap[n]
    for n in t:
        wmap[n] = NAME
    with open(os.path.join(a.dst, INDEX), "w") as f:
        json.dump(index, f, indent=1)
    q = exl3_settings(cfg)                                 # the dict inside cfg: set in place
    was = q.get("drafter_head_k")
    q["drafter_head_k"] = k
    with open(os.path.join(a.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    target = f"EXL3 K{q['head_k']}" if q.get("head") == "exl3" else q.get("head", "bf16")
    print(f"variant {a.dst}: drafter head EXL3 K{k} from {a.head_src} ({len(t)} tensors in {NAME}"
          f"{f', was K{was}' if was else ''}), target head {target} kept. Serve it with VLLM_DSPARK_OWN_HEAD=1 "
          f"(patch 0026); without the flag it serves as SRC.")


if __name__ == "__main__":
    main()
