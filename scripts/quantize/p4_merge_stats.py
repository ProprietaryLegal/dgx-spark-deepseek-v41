# SPDX-License-Identifier: Apache-2.0
"""Domain-weighted pass-0 statistics for p4_allocate.py: the allocator's importance is alpha_l * tshare_e / tokens_l, a
per-token average over the calibration tokens, so a code-heavy token mix makes it a code-heavy average (plan v2: 42% of
the calibration tokens were code, 1.2% writing; build v2.1 then trailed Mia's pack on chat / writing / multilingual KLD
and on IFEval). This writes stats whose per-token averages are a chosen mix of domains instead:

  x_merged[e] / tokens_merged = sum_d w_d * x_d[e] / tokens_d          (x = count, sum_g, sum_g2, out_energy, tshare)

with each domain's sums pooled across the input runs first (pass 0 --index / --domain output: "domains" per layer).
Domains missing from the weights are dropped; weights are renormalised over the domains present.

  python3 p4_merge_stats.py /out/stats --weights code=1,chat=1,...,random=0.1 /p0_dom /p0_instruct [...]
"""
import argparse
import json
import os

FIELDS = ("count", "sum_g", "sum_g2", "out_energy", "tshare")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--weights", required=True, help="domain=weight,... (renormalised)")
    a = ap.parse_args()
    w = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
    os.makedirs(a.out, exist_ok=True)
    layers = sorted({f for r in a.runs for f in os.listdir(r) if f.startswith("layer_") and f.endswith(".json")})
    report = None
    for f in layers:
        recs = [json.load(open(os.path.join(r, f))) for r in a.runs if os.path.exists(os.path.join(r, f))]
        if len(recs) != len(a.runs):
            raise SystemExit(f"{f}: missing in some runs")
        pooled = {}                                  # domain -> {"tokens", fields...} summed over runs
        for rec in recs:
            if "domains" not in rec:
                raise SystemExit(f"{f}: no per-domain stats (run pass 0 with --index or --domain)")
            for d, s in rec["domains"].items():
                p = pooled.setdefault(d, {"tokens": 0.0, **{k: [0.0] * len(s["count"]) for k in FIELDS}})
                p["tokens"] += s["tokens"]
                for k in FIELDS:
                    p[k] = [x + y for x, y in zip(p[k], s[k])]
        use = {d: w[d] for d in pooled if w.get(d, 0) > 0 and pooled[d]["tokens"] > 0}
        tot_w = sum(use.values())
        T = sum(pooled[d]["tokens"] for d in use)
        n_e = len(next(iter(pooled.values()))["count"])
        out = {"tokens": T, "layer": recs[0]["layer"]}
        for k in FIELDS:
            out[k] = [T * sum(use[d] / tot_w * pooled[d][k][e] / pooled[d]["tokens"] for d in use) for e in range(n_e)]
        for k in ("moe_out_energy", "shared_out_energy", "resid_in_energy", "resid_out_energy"):   # token-weighted
            out[k] = sum(r[k] for r in recs) * T / sum(r["tokens"] for r in recs)
        out["merge"] = {"runs": [os.path.basename(r.rstrip("/")) for r in a.runs],   # names only: no host paths
                        "weights": {d: round(use[d] / tot_w, 4) for d in sorted(use)},
                        "tokens": {d: pooled[d]["tokens"] for d in sorted(pooled)},
                        "dropped": sorted(set(pooled) - set(use))}
        with open(os.path.join(a.out, f), "w") as fh:
            json.dump(out, fh)
        report = out["merge"]
    print(f"{len(layers)} layers merged into {a.out}")
    print("weights:", report["weights"])
    print("tokens per domain:", {d: int(t) for d, t in report["tokens"].items()}, "dropped:", report["dropped"])


if __name__ == "__main__":
    main()
