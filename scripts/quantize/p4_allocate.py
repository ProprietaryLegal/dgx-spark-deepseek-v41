# SPDX-License-Identifier: Apache-2.0
"""P4 bit allocation: one K per routed expert (gate/up and down share it) from pass-0 statistics.

Model: a K-bit EXL3 expert adds output noise with relative energy eps(K) (--eps: measured per K on the full
calibration set by pass 1, e.g. 2:0.026,3:0.0042,4:0.00058,5:0.000096; default 0.0286 * 4^-(K-2) from the bridge).
p4_sensitivity.py measures alpha_l = KL per unit of per-token relative noise energy in layer l's routed output, so on
token t expert e costs alpha_l * eps(K_e) * |y_e,t|^2 / |r_t|^2 (r_t = the routed output), and its importance is
alpha_l * tshare_e / tokens_l with tshare_e = sum_t |y_e,t|^2 / |r_t|^2 (pass 0, "tshare").

Plan v1 priced experts by their share of the layer's routed-output ENERGY instead. That is wrong for V4.1: one
"super expert" per layer writes huge outputs on a handful of tokens (layer 1: 92% of the energy, 1.4% of the
per-token share), so v1 gave it K5 and pushed the rest of layer 1 to K2 (pass 1 v1 aborted, Oct 8).
--protect-share keeps those experts at --protect-k anyway: their few tokens are massive-activation tokens, cheap
insurance (~40 experts).

Minimise the total cost subject to a memory budget (average bpw over all routed experts) and, optionally, a cap on
the usage-weighted bpw (what decode reads per token): greedy marginal upgrades (the cost is convex in K, so this is
the Lagrangian optimum on the K grid), the decode cap by bisection on a usage penalty.

  python3 p4_allocate.py /stats --bpw 2.75 --decode-bpw 2.77 --alpha alpha.json --eps ... -o plan.json
  python3 p4_allocate.py /stats --alpha alpha.json --eps ... --score plan.json -o /dev/null
"""
import argparse
import heapq
import json
import math
import os

KS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
CHAR = {1.5: "a", 2.0: "2", 2.5: "b", 3.0: "3", 3.5: "c", 4.0: "4", 5.0: "5", 6.0: "6"}
PARAMS = 3 * 5120 * 2304                              # one routed expert
OVERHEAD_BITS = 3 * (5120 + 2304) * 16                # suh + svh fp16 per projection


EPS_TABLE = None          # measured relative output MSE per K (set by --eps), else the 4x-per-bit model


def eps(K):
    if EPS_TABLE is not None:
        return EPS_TABLE[K]
    return 0.0286 * 4.0 ** (-(K - 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--bpw", type=float, required=True, help="average bits per routed-expert weight (memory)")
    ap.add_argument("--decode-bpw", type=float, default=0.0, help="cap on usage-weighted bpw (0 = none)")
    ap.add_argument("--kmin", type=float, default=2.0)
    ap.add_argument("--kmax", type=float, default=5.0)
    ap.add_argument("--grid", default="2,3,4,5",
                    help="allowed K values. Integers only by default: half-integer K encodes ~2.7x slower than K3 on GB10 "
                         "(ldlq, exllamav3 1.6.0) and every distinct K is one more decode launch per MoE layer")
    ap.add_argument("--alpha", default="", help="JSON {layer: sensitivity}, default 1 / residual energy")
    ap.add_argument("--eps", default="", help="measured eps per K, e.g. 2:0.026,3:0.0042,4:0.00058,5:0.000096")
    ap.add_argument("--score", default="", help="score an existing plan under this eps model instead of allocating")
    ap.add_argument("--importance", choices=["auto", "tshare", "energy"], default="auto",
                    help="tshare = per-token share (needs --alpha and pass-0 tshare); energy = plan v1's energy share")
    ap.add_argument("--protect-share", type=float, default=0.0,
                    help="experts with at least this share of their layer's routed-output energy get K >= --protect-k")
    ap.add_argument("--protect-k", type=float, default=5.0)
    ap.add_argument("--floor-layers", default="",
                    help="minimum K for every expert of some layers, e.g. 37,38,39:3 (the DSpark drafter reads the hidden "
                         "states of layers 37-39: final-output KL cannot see what K2 there costs in draft acceptance)")
    ap.add_argument("--usage-stats", default="",
                    help="per-domain pass-0 stats (layer_XX.json with a 'domains' dict, e.g. data/p0_stats_1017/dom): the "
                         "decode cap's usage comes from these domains' routing, weighted by --usage-weights, instead of the "
                         "main stats' counts (the importance stays the main stats')")
    ap.add_argument("--usage-weights", default="",
                    help="domain:weight list for --usage-stats, e.g. writing:1,chat:1,code:1,std:1 (per-token usage of "
                         "each domain, mixed by these weights)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    global EPS_TABLE
    if a.eps:
        EPS_TABLE = {float(k): float(v) for k, v in (x.split(":") for x in a.eps.split(","))}
    grid = [float(x) for x in a.grid.split(",")]
    ks = [k for k in KS if a.kmin <= k <= a.kmax and k in grid]

    layers = sorted(int(f[6:8]) for f in os.listdir(a.stats) if f.startswith("layer_") and f.endswith(".json"))
    st = {l: json.load(open(os.path.join(a.stats, f"layer_{l:02d}.json"))) for l in layers}
    use_tshare = a.importance == "tshare" or (a.importance == "auto" and a.alpha and all("tshare" in st[l] for l in layers))
    if use_tshare and not a.alpha:
        raise SystemExit("--importance tshare needs a measured --alpha (KL per unit per-token relative noise)")
    if a.alpha:                                        # measured (p4_sensitivity.py): KL per unit relative noise
        meas = {int(k): v for k, v in json.load(open(a.alpha))["alpha"].items()}
    layer_floor = {}
    if a.floor_layers:
        ls_, kf = a.floor_layers.split(":")
        layer_floor = {int(l): float(kf) for l in ls_.split(",")}
    usage_of = None
    if a.usage_stats:                                  # the decode cap on another routing mix (see --usage-stats)
        wts = {d: float(w) for d, w in (x.split(":") for x in a.usage_weights.split(","))}
        usage_of = {}
        for l in layers:
            us = json.load(open(os.path.join(a.usage_stats, f"layer_{l:02d}.json")))["domains"]
            mix = [0.0] * len(st[l]["count"])
            for d, w in wts.items():
                tok_d = max(us[d]["tokens"], 1)
                for e, c in enumerate(us[d]["count"]):
                    mix[e] += w * c / tok_d
            tot_w = sum(wts.values())
            usage_of[l] = [m / tot_w for m in mix]
    experts = []                                       # (layer, idx, importance, usage, floor K)
    for l in layers:
        tok = max(st[l]["tokens"], 1)
        etot = max(sum(st[l]["out_energy"]), 1e-12)
        for e, (E, c) in enumerate(zip(st[l]["out_energy"], st[l]["count"])):
            if use_tshare:
                imp = meas[l] * st[l]["tshare"][e] / tok
            elif a.alpha:
                imp = meas[l] * E / etot
            else:                                      # heuristic: relative perturbation of the residual stream
                imp = E / max(st[l]["resid_out_energy"], 1e-12)
            floor = a.protect_k if a.protect_share and E / etot >= a.protect_share else ks[0]
            floor = max(floor, layer_floor.get(l, ks[0]))
            use = usage_of[l][e] if usage_of is not None else c / tok
            experts.append((l, e, imp, use, min(max(floor, ks[0]), ks[-1])))
    n = len(experts)
    if a.score:
        inv = {v: k for k, v in CHAR.items()}
        pl = json.load(open(a.score))["layers"]
        Ks = [inv[pl[str(x[0])]["k"][x[1]]] for x in experts]
        cost = sum(x[2] * eps(k) for x, k in zip(experts, Ks))
        per_layer = {}
        for x, k in zip(experts, Ks):
            per_layer[x[0]] = per_layer.get(x[0], 0.0) + x[2] * eps(k)
        usage_sum = sum(x[3] for x in experts)
        dec = sum(x[3] * k for x, k in zip(experts, Ks)) / usage_sum
        print(json.dumps({"plan": a.score, "importance": "tshare" if use_tshare else "energy", "predicted_cost": cost,
                          "bpw": sum(Ks) / n, "decode_bpw": dec}))
        print("per layer:", {l: round(c, 5) for l, c in per_layer.items()})
        return
    budget = a.bpw * n                                 # in K units (bits per weight summed over experts)
    usage_sum = sum(x[3] for x in experts)

    # lambda for the decode cap: bisect on a usage penalty so the cap holds (0 if no cap or already satisfied)
    def run(lam):
        Kl = [x[4] for x in experts]
        heap = []
        for i in range(n):
            j = ks.index(Kl[i])
            if j + 1 < len(ks):
                g = experts[i][2] * (eps(ks[j]) - eps(ks[j + 1])) / (ks[j + 1] - ks[j]) - lam * experts[i][3]
                heapq.heappush(heap, (-g, i))
        tot = sum(Kl)
        while heap:
            g, i = heapq.heappop(heap)
            if -g <= 0:
                break
            j = ks.index(Kl[i])
            if j + 1 >= len(ks):
                continue
            dk = ks[j + 1] - ks[j]
            if tot + dk > budget + 1e-9:
                continue
            Kl[i] = ks[j + 1]
            tot += dk
            if j + 2 < len(ks):
                dk2 = ks[j + 2] - ks[j + 1]
                g2 = experts[i][2] * (eps(ks[j + 1]) - eps(ks[j + 2])) / dk2 - lam * experts[i][3]
                heapq.heappush(heap, (-g2, i))
        return Kl

    lam = 0.0
    K = run(0.0)
    dec = sum(x[3] * k for x, k in zip(experts, K)) / usage_sum
    if a.decode_bpw and dec > a.decode_bpw:
        lo, hi = 0.0, max(x[2] * eps(ks[0]) / x[3] for x in experts if x[3] > 0)   # every upgrade gain < 0 at hi
        for _ in range(60):
            mid = (lo + hi) / 2
            Km = run(mid)
            dm = sum(x[3] * k for x, k in zip(experts, Km)) / usage_sum
            if dm > a.decode_bpw:
                lo = mid
            else:
                hi, K = mid, Km
        lam = hi
        dec = sum(x[3] * k for x, k in zip(experts, K)) / usage_sum
        if dec > a.decode_bpw + 1e-6:
            raise SystemExit(f"decode cap {a.decode_bpw} unreachable (floors alone give {dec:.4f})")

    cost = sum(x[2] * eps(k) for x, k in zip(experts, K))
    uni = sum(x[2] * eps(a.bpw if a.bpw in ks else min(ks, key=lambda q: abs(q - a.bpw))) for x in experts)
    plan = {"format": "exl3_moe_plan", "version": 1, "codebook": "mul1", "num_layers": 40, "num_experts": 384,
            "hidden_size": 5120, "moe_intermediate_size": 2304, "layers": {}}
    for l in layers:
        row = [K[i] for i, x in enumerate(experts) if x[0] == l]
        plan["layers"][str(l)] = {"k": "".join(CHAR[k] for k in row)}
    avg = sum(K) / n
    gib_rank = (sum(K) * PARAMS + n * OVERHEAD_BITS) / 8 / 2 / 2 ** 30
    hist = {k: K.count(k) for k in ks}
    plan["summary"] = {"bpw": round(avg, 4), "decode_bpw": round(dec, 4), "gib_per_rank_tp2": round(gib_rank, 2),
                       "hist": {str(k): v for k, v in hist.items()}, "lambda_decode": lam,
                       "predicted_cost": cost, "uniform_cost_at_target": uni,
                       "alpha": os.path.basename(a.alpha) if a.alpha else "1/residual_energy",
                       "importance": "tshare" if use_tshare else "energy",
                       "protected": sum(1 for x in experts if x[4] > ks[0]), "eps": EPS_TABLE or "4x per bit",
                       "floor_layers": a.floor_layers or None, "protect_share": a.protect_share or None,
                       "layers_with_stats": len(layers)}
    with open(a.out, "w") as f:
        json.dump(plan, f, indent=1)
    print(json.dumps(plan["summary"], indent=1))
    for l in layers:
        row = [K[i] for i, x in enumerate(experts) if x[0] == l]
        print(f"layer {l:2d}: avg K {sum(row) / len(row):.3f}  " + " ".join(f"{k}:{row.count(k)}" for k in ks if row.count(k)))


if __name__ == "__main__":
    main()
