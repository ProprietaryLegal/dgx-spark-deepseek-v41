#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-position temperatures for DSpark's confidence head, fitted from a VLLM_DSPARK_CONF_LOG round log (patch 0023).
Oct 20 2026.

The log, written by rank 0 of the serve: header lines {"hdr": 1, "k", "policy", "cost_ms", "temps", "bias", "tag"} and
one line per request per verify round {"q": request id, "s": step, "c": the head's raw confidences c_1..c_k, "d": the
depth that round used, "n": drafts in the window, "a": drafts accepted}. Draft i is observed when it was verified
(i <= min(d, n)) and drafts 1..i-1 were accepted, the event the head models (DSpark, arXiv 2607.05147, 3.2.1), so each
position's fit is a logistic regression of "accepted >= i" on z_i = logit(c_i): p_i = sigmoid(z_i / t_i + b_i), b_i = 0
unless --bias, by minimum negative log-likelihood (damped Newton, convex in 1/t and b; a pull toward t = 1, b = 0 worth
about one round keeps small or one-sided samples finite). The serve computes the same formula from
VLLM_DSPARK_CONF_TEMPS / VLLM_DSPARK_CONF_BIAS (torch.logit clamped 1e-6 from 0 and 1, as here).

Reported, cross-fitted by request (--folds: each request is scored with temperatures fitted without it):
  per position: rounds observed, acceptance, mean probability, ECE (15 equal-width bins) of the conditional
  probability and of the prefix survival prod_{j<=i} p_j, and the NLL a round, raw vs fitted;
  a replay of patch 0010's depth rule over the logged rounds with raw vs fitted probabilities under the cost table (the
  log header's, or --round-ms): mean depth, predicted and logged tokens a round, ms a round and model tok/s
  (1000 x tokens / ms), next to fixed:k, the best fixed depth and the hindsight oracle (verify exactly the drafts that
  get accepted), overall and per domain.
The env line at the end is fitted on every round.

Log under VLLM_DSPARK_DEPTH_POLICY=fixed:<k>: every draft is then verified whenever its prefix survives. Under the conf
policy the depth was chosen from the same scores, so what lies past it is unobserved in a biased way; the replay then
drops rounds whose outcome past the logged depth is unknown and says how many. Greedy requests only: sampled ones
accept differently. The replay holds the logged rounds fixed (a policy that accepts fewer tokens a round would also
run more rounds, from other states), so read it as a per-round estimate.

Domains: a request id containing dom-<name> (accept_by_domain.py sends X-Request-Id dom-<group>-<id>-<hex>), else the
header's tag (VLLM_DSPARK_CONF_LOG_TAG), else "all".
  python3 fit_conf_temps.py conf.jsonl [--bias] [--folds 4] [--round-ms 28.90,32.85,...] [--out fit.json]
"""
import argparse
import json
import re
import sys
import zlib
from collections import Counter

import numpy as np

EPS = 1e-6              # the serve's torch.logit clamp (patch 0023)
BINS = 15               # ECE bins (Guo et al. 2017; the DSpark paper reports ECE the same way)
T_MIN, T_MAX = 0.05, 50.0
RIDGE = 1.0             # pull toward t = 1, b = 0, worth about one round
MIN_ROUNDS = 50         # fewer observed rounds at a position: it stays raw (t = 1, b = 0)
TAG_RE = r"(?:^|[^A-Za-z0-9])dom-([A-Za-z0-9_]+)"


def logit(c):
    c = np.clip(np.asarray(c, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(c) - np.log1p(-c)


def sigmoid(u):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(u, dtype=np.float64)))


def calibrated(c, temps, bias):
    """[R, k] raw confidences -> sigmoid(logit(c) / t + b) per position, the serve's formula."""
    return sigmoid(logit(c) / np.asarray(temps, dtype=np.float64) + np.asarray(bias, dtype=np.float64))


def conditional(V, A, i):
    """Rounds where draft i (0-based) was verified with drafts 0..i-1 accepted, and whether draft i was accepted."""
    m = (V > i) & (A >= i)
    return m, (A[m] > i).astype(np.float64)


def nll(z, y, a=1.0, b=0.0):
    """Negative log-likelihood of outcomes y under sigmoid(a z + b), summed."""
    u = a * np.asarray(z, dtype=np.float64) + b
    return float(np.sum(np.logaddexp(0.0, u) - y * u))


def fit_position(z, y, bias=False, ridge=RIDGE, iters=100):
    """(t, b) minimizing NLL(sigmoid(z / t + b); y) + ridge / 2 ((1/t - 1)^2 + b^2), damped Newton in a = 1/t."""
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    a_lo, a_hi = 1.0 / T_MAX, 1.0 / T_MIN

    def obj(a, b):
        return nll(z, y, a, b) + 0.5 * ridge * ((a - 1.0) ** 2 + b * b)

    a, b = 1.0, 0.0
    f = obj(a, b)
    for _ in range(iters):
        p = sigmoid(a * z + b)
        r, w = p - y, p * (1.0 - p)
        ga, haa = float(r @ z) + ridge * (a - 1.0), float(w @ (z * z)) + ridge
        if bias:
            gb, hab, hbb = float(r.sum()) + ridge * b, float(w @ z), float(w.sum()) + ridge
            det = haa * hbb - hab * hab
            da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        else:
            da, db = ga / haa, 0.0
        step = 1.0
        while step >= 1e-8:
            na, nb = min(max(a - step * da, a_lo), a_hi), b - step * db
            nf = obj(na, nb)
            if nf <= f:
                break
            step *= 0.5
        else:
            break
        gain, (a, b, f) = f - nf, (na, nb, nf)
        if gain <= 1e-12 * max(1.0, abs(f)):
            break
    return 1.0 / a, b


def fit(C, V, A, bias=False, min_rounds=MIN_ROUNDS):
    """Per-position temperatures, biases and observed rounds from rounds C [R, k], verified V, accepted A."""
    k = C.shape[1]
    temps, biases, counts = np.ones(k), np.zeros(k), np.zeros(k, dtype=np.int64)
    for i in range(k):
        m, y = conditional(V, A, i)
        counts[i] = int(m.sum())
        if counts[i] >= min_rounds:
            temps[i], biases[i] = fit_position(logit(C[m, i]), y, bias=bias)
    return temps, biases, counts


def ece(p, y, bins=BINS):
    """Expected calibration error, equal-width bins: sum over bins of |sum y - sum p| / rounds."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    b = np.minimum((p * bins).astype(np.int64), bins - 1)
    return float(np.abs(np.bincount(b, y, bins) - np.bincount(b, p, bins)).sum() / p.size)


def choose_depth(q, cost):
    """Patch 0010's rule on [R, k] probabilities: the depth d in 0..k with the most expected tokens per ms, expected
    = 1 + sum_{j<=d} prod_{i<=j} q_i over cost[d] ms, the shallowest among equal rates. Returns (depth, expected)."""
    surv = np.cumprod(np.clip(q, 0.0, 1.0), axis=1)
    expected = np.concatenate([np.ones((len(q), 1)), 1.0 + np.cumsum(surv, axis=1)], axis=1)
    return np.argmax(expected / np.asarray(cost, dtype=np.float64), axis=1), expected


def replay(depth, A, N, cost, expected=None):
    """The rounds had they verified `depth` (capped at their drafts): tokens = 1 + min(accepted, depth) under greedy
    verification, ms = cost[depth]."""
    e = np.minimum(np.asarray(depth), N)
    tokens = 1 + np.minimum(A, e)
    ms = np.asarray(cost, dtype=np.float64)[e]
    row = {"rounds": int(e.size), "depth": float(e.mean()), "tok_round": float(tokens.mean()),
           "ms_round": float(ms.mean()), "tok_s": float(1000.0 * tokens.sum() / ms.sum())}
    if expected is not None:
        row["pred_tok_round"] = float(expected[np.arange(e.size), e].mean())
    return row


def cost_table(text, k):
    """ms by depth 0..k from "ms,ms,..." (the depth file's conf:<table> form), extended by the last step as the serve
    does."""
    rows = [float(x) for x in str(text).split(",") if x.strip()]
    while len(rows) < k + 1:
        rows.append(rows[-1] + (rows[-1] - rows[-2] if len(rows) > 1 else 0.0))
    return rows[: k + 1]


def load(path, tag_re=TAG_RE):
    """The rounds of a VLLM_DSPARK_CONF_LOG file: arrays C [R, k], D, N, A, V = min(D, N), and per round the request
    id, domain and the policy at log time; plus the headers and a count of skipped lines."""
    rx = re.compile(tag_re)
    hdr, heads, bad = {}, [], Counter()
    k = None
    C, D, N, A, req, dom, pol = [], [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad["unparsable"] += 1
                continue
            if r.get("hdr"):
                hdr = r
                heads.append(r)
                k = k or r.get("k")
                continue
            try:
                c, d, n, a, q = r["c"], int(r["d"]), int(r["n"]), int(r["a"]), str(r["q"])
            except (KeyError, TypeError, ValueError):
                bad["fields"] += 1
                continue
            k = k or len(c)
            if len(c) != k:
                bad["k"] += 1
                continue
            if not (n >= 1 and 0 <= d <= k and 0 <= a <= min(d, n)):
                bad["outcome"] += 1
                continue
            m = rx.search(q)
            C.append(c)
            D.append(d)
            N.append(n)
            A.append(a)
            req.append(q)
            dom.append(m.group(1) if m else hdr.get("tag") or "all")
            pol.append(hdr.get("policy", "unknown") + (" + temps" if hdr.get("temps") else ""))
    C = np.asarray(C, dtype=np.float64).reshape(len(C), k or 1)
    D, N, A = (np.asarray(x, dtype=np.int64) for x in (D, N, A))
    return {"C": C, "D": D, "N": N, "A": A, "V": np.minimum(D, N), "req": req, "dom": dom, "policy": pol,
            "headers": heads, "bad": bad, "k": k or 0}


def cross_fit(C, V, A, req, folds, bias=False, min_rounds=MIN_ROUNDS):
    """Calibrated probabilities with each request scored by a fit that never saw it (folds by request id); folds <= 1
    or a single request: in-sample. Returns (q, folds used)."""
    ids = sorted(set(req))
    folds = min(folds, len(ids))
    if folds <= 1:
        t, b, _ = fit(C, V, A, bias, min_rounds)
        return calibrated(C, t, b), 1
    fold = np.array([zlib.crc32(q.encode()) % folds for q in req])
    q = np.empty_like(C)
    for f in range(folds):
        test = fold == f
        if test.any():
            t, b, _ = fit(C[~test], V[~test], A[~test], bias, min_rounds)
            q[test] = calibrated(C[test], t, b)
    return q, folds


def position_table(C, Q, V, A):
    """Per position: observed rounds, acceptance, mean probability, conditional and survival ECE, NLL a round; raw
    (C) vs calibrated (Q)."""
    out = []
    surv_raw, surv_fit = np.cumprod(C, axis=1), np.cumprod(Q, axis=1)
    for i in range(C.shape[1]):
        m, y = conditional(V, A, i)
        s = V > i
        ys = (A[s] > i).astype(np.float64)
        row = {"pos": i + 1, "rounds": int(m.sum()), "accept": float(y.mean()) if y.size else float("nan")}
        for name, P, S in (("raw", C, surv_raw), ("fit", Q, surv_fit)):
            p = np.clip(P[m, i], EPS, 1.0 - EPS)
            row[f"mean_{name}"] = float(p.mean()) if p.size else float("nan")
            row[f"ece_cond_{name}"] = ece(p, y)
            row[f"ece_surv_{name}"] = ece(S[s, i], ys)
            row[f"nll_{name}"] = nll(logit(p), y) / max(p.size, 1)
        out.append(row)
    return out


def replay_block(C, Q, A, N, V, cost, k, best_fixed=None):
    """Replay rows for one set of rounds: raw and fitted conf rule, logged depths, fixed:k, best fixed, oracle."""
    ok = (A < V) | (V >= N)            # outcome known at every depth up to the window
    if not ok.any():
        return None, int(ok.size), best_fixed
    C, Q, A, N, V = C[ok], Q[ok], A[ok], N[ok], V[ok]
    rows = {}
    for name, P in (("raw", C), ("fitted", Q)):
        d, expected = choose_depth(P, cost)
        rows[name] = replay(d, A, N, cost, expected)
    if (V < N).any():
        rows["logged"] = replay(V, A, N, cost)
    rows[f"fixed:{k}"] = replay(np.full(A.size, k), A, N, cost)
    if best_fixed is None:
        best_fixed = max(range(k + 1), key=lambda d: replay(np.full(A.size, d), A, N, cost)["tok_s"])
    if best_fixed != k:
        rows[f"fixed:{best_fixed} (best fixed)"] = replay(np.full(A.size, best_fixed), A, N, cost)
    rows["oracle"] = replay(A, A, N, cost)
    return rows, int((~ok).sum()), best_fixed


def fmt_pct(x):
    return "   -  " if x != x else f"{100 * x:5.1f}%"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", help="VLLM_DSPARK_CONF_LOG .jsonl file")
    ap.add_argument("--bias", action="store_true", help="fit a per-position bias next to the temperature")
    ap.add_argument("--folds", type=int, default=4, help="cross-fit folds by request (1 = in-sample)")
    ap.add_argument("--round-ms", help="cost table, ms by depth 0..k, drafter included (default: the log header's)")
    ap.add_argument("--min-rounds", type=int, default=MIN_ROUNDS, help="observed rounds a position needs for a fit")
    ap.add_argument("--tag-re", default=TAG_RE, help="regex whose group 1 is the domain in a request id")
    ap.add_argument("--out", help="write the fit, the tables and the replay as JSON")
    a = ap.parse_args(argv)

    L = load(a.log, a.tag_re)
    C, D, N, A, V, k = L["C"], L["D"], L["N"], L["A"], L["V"], L["k"]
    if not len(A):
        print(f"{a.log}: no rounds ({dict(L['bad']) or 'empty'})")
        return 1
    tables = [h["cost_ms"] for h in L["headers"] if h.get("cost_ms")]
    if a.round_ms:
        cost = cost_table(a.round_ms, k)
    elif tables:
        cost = cost_table(",".join(map(str, tables[-1])), k)
    else:
        print("no cost table in the log: pass --round-ms (ms by depth 0..k, drafter included)")
        return 1

    print(f"log {a.log}: {len(A):,} rounds from {len(set(L['req'])):,} requests, k={k}"
          + (f"; skipped {dict(L['bad'])}" if L["bad"] else ""))
    for p, n in Counter(L["policy"]).most_common():
        print(f"  policy at log time {p}: {n:,} rounds")
    if any(not p.startswith(f"fixed:{k}") for p in L["policy"]):
        print(f"  note: rounds not logged under fixed:{k} pick their depth from the scores being fitted; log under "
              f"VLLM_DSPARK_DEPTH_POLICY=fixed:{k} for an unbiased fit")
    if len({tuple(t) for t in tables}) > 1 and not a.round_ms:
        print(f"  note: the log holds {len({tuple(t) for t in tables})} cost tables; the replay uses the last one")
    print("cost table, ms by depth 0..%d: %s%s" % (k, " ".join(f"{c:.2f}" for c in cost),
                                                   " (--round-ms)" if a.round_ms else " (log header)"))

    Q, folds = cross_fit(C, V, A, L["req"], a.folds, a.bias, a.min_rounds)
    temps, biases, counts = fit(C, V, A, a.bias, a.min_rounds)
    kind = "temperature + bias" if a.bias else "temperature"
    print(f"fit: {kind} per position, minimum NLL; tables below "
          + (f"cross-fitted over {folds} folds by request (each request scored without its own rounds)"
             if folds > 1 else "in-sample"))

    pos = position_table(C, Q, V, A)
    print("\npos   rounds  accepted |  raw: mean  ECE cond  ECE surv  NLL   |  fit: mean  ECE cond  ECE surv  NLL   "
          "|  t (all)  b (all)")
    for r in pos:
        i = r["pos"] - 1
        print(f"{r['pos']:3d} {r['rounds']:8,d}   {fmt_pct(r['accept'])}  | "
              f"{r['mean_raw']:10.3f}    {fmt_pct(r['ece_cond_raw'])}    {fmt_pct(r['ece_surv_raw'])}  "
              f"{r['nll_raw']:.3f} | {r['mean_fit']:10.3f}    {fmt_pct(r['ece_cond_fit'])}    "
              f"{fmt_pct(r['ece_surv_fit'])}  {r['nll_fit']:.3f} | {temps[i]:7.3f}  {biases[i]:7.3f}"
              + ("   (too few rounds: raw)" if counts[i] < a.min_rounds else ""))

    env = "VLLM_DSPARK_CONF_TEMPS=" + ",".join(f"{t:.3f}" for t in temps)
    if a.bias:
        env += " VLLM_DSPARK_CONF_BIAS=" + ",".join(f"{b:.3f}" for b in biases)

    doms = np.asarray(L["dom"])
    blocks = ["all"] + sorted(set(L["dom"]) - {"all"})
    out_replay, best_fixed = {}, None
    print("\nreplay of 0010's depth rule over the logged rounds (model tok/s = 1000 x tokens / ms on the cost table)")
    print(f"{'domain':14s} {'rounds':>7s}  {'policy':24s} {'depth':>5s} {'pred tok':>8s} {'tok/rd':>7s} "
          f"{'ms/rd':>7s} {'model tok/s':>13s} {'vs raw':>7s}")
    for blk in blocks:
        sel = np.ones(len(A), dtype=bool) if blk == "all" else doms == blk
        if blk != "all" and len(blocks) == 2 and sel.all():
            continue                     # one domain only: the same rounds as "all"
        rows, dropped, bf = replay_block(C[sel], Q[sel], A[sel], N[sel], V[sel], cost, k, best_fixed)
        best_fixed = bf if blk == "all" else best_fixed
        out_replay[blk] = {"rows": rows, "dropped_censored": dropped}
        if rows is None:
            print(f"{blk:14s} {0:7d}  (all {dropped:,} rounds have an unknown outcome past the logged depth)")
            continue
        base = rows["raw"]["tok_s"]
        for j, (name, r) in enumerate(rows.items()):
            head = f"{blk:14s} {r['rounds']:7,d}" if j == 0 else " " * 22
            pred = f"{r['pred_tok_round']:8.2f}" if "pred_tok_round" in r else f"{'-':>8s}"
            vs = "" if name == "raw" else f"{100 * (r['tok_s'] / base - 1):+6.1f}%"
            print(f"{head}  {name:24s} {r['depth']:5.2f} {pred} {r['tok_round']:7.2f} {r['ms_round']:7.2f} "
                  f"{r['tok_s']:7.1f} tok/s {vs:>7s}")
        if dropped:
            print(" " * 24 + f"({dropped:,} rounds dropped: all verified drafts accepted below the window, "
                  "outcome past the logged depth unknown)")

    print(f"\n{env}   (fit on all {len(A):,} rounds)")
    if a.out:
        res = {"log": a.log, "rounds": int(len(A)), "requests": len(set(L["req"])), "k": k, "cost_ms": cost,
               "policies_at_log_time": dict(Counter(L["policy"])), "skipped": dict(L["bad"]),
               "fit": {"kind": kind, "folds": folds, "temps": temps.tolist(), "bias": biases.tolist(),
                       "rounds_observed": counts.tolist(), "min_rounds": a.min_rounds},
               "env": env, "positions": pos, "replay": out_replay}
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
