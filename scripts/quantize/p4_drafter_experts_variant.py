# SPDX-License-Identifier: Apache-2.0
"""A model-dir variant whose DSpark drafter runs its routed experts on EXL3 (p4_drafter_experts.py output): every file
of the source dir hardlinked, except
  - the shards holding the drafter's MXFP4 expert tensors (mtp.<i>.ffn.experts.<e>.w{1,2,3}.{weight,scale}), rewritten
    without them: vLLM's loader yields every tensor of a listed shard, and the EXL3 method registers no MXFP4 params;
  - model.safetensors.index.json: those entries dropped, the EXL3 file's added;
  - exl3_moe_plan.json: a "drafter" section (num_experts, layers num_hidden_layers + i, each expert's K and K_down from
    the encode's json), which the plugin (plan.py, config.py, Oct 11) reads to put layers 40-42 on its EXL3 method;
and the EXL3 tensors linked in as model-exl3-mtp-experts.safetensors. One node, no GPU; run as root in the quant image
(the source files are root-owned). Needs a serve image whose plugin knows the plan's "drafter" section.

  python3 p4_drafter_experts_variant.py /models/<src> /runs/drafter/mtp_experts_exl3_k3.safetensors /models/<dst>
"""
import argparse
import json
import os
import re

from safetensors import safe_open
from safetensors.torch import save_file

NAME = "model-exl3-mtp-experts.safetensors"
MXFP4 = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.w[123]\.(weight|scale)$")
EXL3 = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.w[123]\.(trellis|suh|svh|mul1|mcg)$")


def kchar(k: float) -> str:
    if not float(k).is_integer() or not 1 <= k <= 8:
        raise SystemExit(f"K {k}: this builder writes whole bitrates 1-8 only")
    return str(int(k))


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
        names = list(f.keys())
    bad = [n for n in names if not EXL3.match(n)]
    if bad:
        raise SystemExit(f"{len(bad)} tensors outside mtp.<i>.ffn.experts.<e>.w*.*, e.g. {bad[:3]}")
    with open(a.exl3 + ".json") as f:
        enc = json.load(f)["experts"]
    with open(os.path.join(a.src, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    n_hidden = int(tc["num_hidden_layers"])
    n_exp = int(tc.get("dspark_n_routed_experts") or tc["n_routed_experts"])
    layers: dict[int, dict[int, dict]] = {}
    for key, m in enc.items():                                   # "mtp.<i>.ffn.experts.<e>" -> {k, k_down, ...}
        i, e = (int(x) for x in re.match(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)$", key).groups())
        layers.setdefault(i, {})[e] = m
    for i, ex in layers.items():
        if sorted(ex) != list(range(n_exp)):
            raise SystemExit(f"mtp.{i}: the encode has {len(ex)} experts, the drafter has {n_exp}")
    drop = {n for n in wmap if (m := MXFP4.match(n)) and int(m.group(1)) in layers}
    rewrite = sorted({wmap[n] for n in drop})
    os.makedirs(a.dst, exist_ok=False)
    for name in sorted(os.listdir(a.src)):
        s, d = os.path.join(a.src, name), os.path.join(a.dst, name)
        if name in rewrite or name in ("model.safetensors.index.json", "exl3_moe_plan.json", "MANIFEST.sha256", NAME):
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
    os.link(os.path.abspath(a.exl3), os.path.join(a.dst, NAME))
    for n in drop:
        del wmap[n]
    for n in [n for n, f in wmap.items() if f == NAME]:
        del wmap[n]
    for n in names:
        wmap[n] = NAME
    with open(os.path.join(a.dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=1)
    with open(os.path.join(a.src, "exl3_moe_plan.json")) as f:
        plan = json.load(f)
    plan["drafter"] = {"num_experts": n_exp, "layers": {}}
    for i, ex in sorted(layers.items()):
        entry = {"k": "".join(kchar(ex[e]["k"]) for e in range(n_exp))}
        kd = "".join(kchar(ex[e]["k_down"]) for e in range(n_exp))
        if kd != entry["k"]:
            entry["k_down"] = kd
        plan["drafter"]["layers"][str(n_hidden + i)] = entry
    with open(os.path.join(a.dst, "exl3_moe_plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    ks = sorted({ex[e]["k"] for ex in layers.values() for e in ex})
    print(f"variant {a.dst}: {len(drop)} MXFP4 expert tensors replaced by {len(names)} EXL3 tensors, drafter layers "
          f"{sorted(n_hidden + i for i in layers)}, K {ks}")


if __name__ == "__main__":
    main()
