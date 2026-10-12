#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write an exl3_moe_plan.json and print its static bpw and per-rank memory.

Synthetic plans are for plumbing and dummy-load tests only (the real plan comes from the P4
calibration: per-expert sensitivity x usage, allocated to the memory budget).

  # uniform K=3 everywhere
  tools/make_plan.py --uniform 3 -o /tmp/plan_k3.json
  # the published reference mix's shape (K2 7,279 / K3 6,299 / K5 1,782 over 15,360 experts)
  tools/make_plan.py --counts 2:7279,3:6299,5:1782 -o /tmp/plan_mix.json
  # inspect an existing plan
  tools/make_plan.py --show exl3_moe_plan.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_exl3_moe.plan import Exl3MoEPlan, synthetic_plan  # noqa: E402

BASE = "deepseek-ai/DeepSeek-V4.1-Flash@2cba9e42aa026125f3ed06c6d98c1db82f7ca027"


def report(plan: Exl3MoEPlan, tp: int) -> None:
    per_rank = plan.bytes_per_rank(tp)
    print(f"layers {len(plan.layers)}/{plan.num_layers}, experts/layer {plan.num_experts}, codebook {plan.codebook}")
    print(f"static bpw {plan.bpw():.4f}")
    print(f"per rank at TP={tp}: {per_rank / 2**30:.2f} GiB (trellis + fp16 suh/svh)")
    print("experts per (K_gate/up, K_down):", {f"{a:g}/{b:g}": n for (a, b), n in plan.group_histogram().items()})
    groups = [len(lp.groups) for lp in plan.layers.values()]
    print(f"K groups per layer (kernel launches per MoE layer): min {min(groups)} max {max(groups)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", type=float)
    ap.add_argument("--counts", help="K:count,... summing to num_layers * num_experts")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--show")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    if a.show:
        report(Exl3MoEPlan.from_json_file(a.show), a.tp)
        return
    counts = None
    if a.counts:
        counts = {float(k): int(n) for k, n in (kv.split(":") for kv in a.counts.split(","))}
    plan = synthetic_plan(num_layers=a.layers, num_experts=a.experts, counts=counts, uniform_k=a.uniform,
                          seed=a.seed, base_model=BASE)
    report(plan, a.tp)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(plan.to_dict(), f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
