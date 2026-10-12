# SPDX-License-Identifier: Apache-2.0
"""exl3_smoe built with and without -DSMOE_RUNFLAGS=1 (down waits per run instead of per grid) on one TP=2 rank-0
layer: the same bank and routings through both builds must give bit-identical outputs (same math, same sums), then the
per-call time of each in CUDA graphs (8 routings a graph, as bench_smoe.py).
    python3 ab_smoe_runflags.py "2:35,3:185,4:163,5:1" 1:6,4:16,6:22,8:28
AB_FLAGS sets the second build's flags (default "-DSMOE_RUNFLAGS=1"; e.g. add -DSMOE_RUNFLAGS_SLEEP=64).
"""
import os
import sys

import torch

from bench_smoe import graph_us, shared_routing
from test_kernels_gpu import DEV, H, LIMIT, TOPK, make_bank, random_experts
from vllm_exl3_moe import smoe as S
from vllm_exl3_moe.jit import load


def build(flags):
    os.environ["EXL3_SMOE_FLAGS"] = flags
    return load("exl3_smoe", "EXL3_SMOE_FLAGS")


def main():
    mix = [(int(k), int(n)) for k, n in (p.split(":") for p in sys.argv[1].split(","))]
    cases = [(int(a), int(b)) for a, b in (p.split(":") for p in sys.argv[2].split(","))]
    ks = [k for k, n in mix for _ in range(n)]
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    mods = {"grid": build(""), "runflags": build(os.environ.get("AB_FLAGS", "-DSMOE_RUNFLAGS=1"))}
    layers = {}
    for name, mod in mods.items():
        S._ext = mod
        layers[name] = S.SmoeLayer(b["bank"], TOPK)

    def run(name, x, ids, w):
        S._ext = mods[name]
        return layers[name](x, ids, w, b["perm"], LIMIT)

    ok = True
    for m, distinct in cases:
        x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
        calls = [shared_routing(m, len(ks), distinct, 1000 * m + i) for i in range(8)]
        same = all(torch.equal(run("grid", x, ids, w), run("runflags", x, ids, w)) for w, ids in calls)
        ok &= same
        t = {}
        for name in mods:
            S._ext = mods[name]
            t[name] = graph_us([lambda w=w, ids=ids, n=name: layers[n](x, ids, w, b["perm"], LIMIT) for w, ids in calls])
        print(f"m={m:2d} distinct={distinct:2d}: bit-identical {same}; grid {t['grid']:7.1f} us, runflags "
              f"{t['runflags']:7.1f} us ({t['grid'] / t['runflags']:.3f}x)", flush=True)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
