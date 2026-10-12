#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Routed-expert decode cost model for DeepSeek-V4.1-Flash on 2x GB10 (docs/exl3_moe_design.md section 8).

Bytes per decode step per rank for the routed experts only, uniform independent routing (top-6 of 384),
channel split (each rank reads half of every unique expert) vs expert parallel inside TP (each rank
reads whole experts of its half, the slower rank sets the pace). Model numbers, not measurements.

  tools/cost_model.py                 # 2.7 bpw, 273 GB/s
  tools/cost_model.py --bpw 2.758 --bw 250 --trials 20000
"""

import argparse
import random
import statistics

H, I, E, TOPK, L = 5120, 2304, 384, 6, 40


def simulate(n: int, trials: int, rng: random.Random) -> tuple[float, float]:
    uniq, slow = [], []
    for _ in range(trials):
        s: set[int] = set()
        for _ in range(n):
            s.update(rng.sample(range(E), TOPK))
        lo = sum(1 for e in s if e < E // 2)
        uniq.append(len(s))
        slow.append(max(lo, len(s) - lo))
    return statistics.mean(uniq), statistics.mean(slow)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpw", type=float, default=2.7)
    ap.add_argument("--bw", type=float, default=273.0, help="GB/s per box")
    ap.add_argument("--eff", type=float, default=0.8, help="achieved fraction of --bw for the second column")
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    expert = 3 * H * I * a.bpw / 8
    bw = a.bw * 1e9
    print(f"whole expert {expert / 1e6:.2f} MB at {a.bpw} bpw; {a.bw:.0f} GB/s per box; uniform routing")
    print(f"{'n':>3} {'unique/layer':>12} {'GB/rank/step':>13} {'ms @bw':>8} {'ms @eff':>8} {'EP slow GB':>11} {'EP ms @bw':>10}")
    for n in (1, 2, 4, 8, 16):
        u, slow = simulate(n, a.trials, rng)
        tp = u * expert / 2 * L
        ep = slow * expert * L
        print(f"{n:>3} {u:>12.1f} {tp / 1e9:>13.2f} {tp / bw * 1e3:>8.2f} {tp / (a.eff * bw) * 1e3:>8.2f} "
              f"{ep / 1e9:>11.2f} {ep / bw * 1e3:>10.2f}")


if __name__ == "__main__":
    main()
