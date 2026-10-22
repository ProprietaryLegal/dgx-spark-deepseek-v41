#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Can more than the confidence head predict acceptance? Offline replay on a fixed:k VLLM_DSPARK_CONF_LOG round log
(Oct 20 2026).

Per position i, a logistic model of "draft i accepted given 1..i-1 were" on features known when the depth is picked:
the head's logits z_1..z_k, the request's last two rounds (drafts accepted, all accepted), the step. Fitted per fold
with the rounds of the other requests (cross-fit by request, as fit_conf_temps.py), then patch 0010's depth rule on the
predicted probabilities, replayed under the cost table: tok/s next to the raw head, the head alone refitted (z_i only)
and the hindsight oracle, overall and per domain. Reads fit_conf_temps.py's loader and replay.
  python3 predictor_research.py conf.jsonl [--folds 4] [--round-ms ...]
"""
import argparse, sys, zlib
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fit_conf_temps import EPS, choose_depth, cost_table, load, replay  # noqa: E402

RIDGE = 1.0


def features(L, which):
    """[R, F] per-round features (no look-ahead: only rounds before this one in its request)."""
    C, A, D, req = L["C"], L["A"], L["V"], L["req"]
    z = np.log(np.clip(C, EPS, 1 - EPS) / (1 - np.clip(C, EPS, 1 - EPS)))
    R, k = C.shape
    prev = np.zeros((R, 4))       # a_prev / k, all_prev, a_prev2 / k, first round
    step = np.zeros(R)
    last = {}
    for r in range(R):            # the log is in round order per request
        h = last.get(req[r], [])
        if h:
            prev[r, 0] = h[-1][0] / k; prev[r, 1] = float(h[-1][0] >= h[-1][1])
            prev[r, 2] = (h[-2][0] if len(h) > 1 else h[-1][0]) / k
        else:
            prev[r, 3] = 1.0
        step[r] = np.log1p(len(h))
        last.setdefault(req[r], []).append((A[r], D[r]))
    cols = {"z": z, "prev": prev, "step": step[:, None], "zmean": z.mean(1, keepdims=True)}
    return {name: np.concatenate([cols[c] for c in spec], axis=1) for name, spec in which.items()}, z


def logreg(X, y, ridge=RIDGE, iters=50):
    """Ridge logistic regression by Newton's method; X gets an intercept column. Returns weights."""
    X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    w = np.zeros(X.shape[1])
    reg = ridge * np.eye(X.shape[1]); reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + reg @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    return 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))


def cross_predict(F, z, A, req, folds, pos_feats):
    """[R, k] conditional acceptance probabilities, each request scored by models fitted without it."""
    R, k = z.shape
    fold = np.array([zlib.crc32(q.encode()) % folds for q in req])
    Q = np.zeros((R, k))
    for f in range(folds):
        tr, te = fold != f, fold == f
        for i in range(k):
            Xi = pos_feats(F, z, i)
            obs = tr & (A >= i)             # draft i+1 observed: drafts 1..i accepted (fixed:k verifies all)
            w = logreg(Xi[obs], (A[obs] >= i + 1).astype(np.float64))
            Q[te, i] = predict(w, Xi[te])
    return Q


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("--folds", type=int, default=4); ap.add_argument("--round-ms")
    a = ap.parse_args(argv)
    L = load(a.log)
    keep = np.array([not q.startswith("_warmup") for q in L["req"]])
    for key in ("C", "D", "N", "A", "V"):
        L[key] = L[key][keep]
    for key in ("req", "dom", "policy"):
        L[key] = [x for x, m in zip(L[key], keep) if m]
    k = L["k"]
    hdr = L["headers"][-1] if L["headers"] else {}
    cost = cost_table(a.round_ms, k) if a.round_ms else cost_table(",".join(map(str, hdr["cost_ms"])), k)
    if (L["V"] < k).any():
        sys.exit("needs a fixed:k log (every draft verified while its prefix survives)")
    A, N, req = L["A"], L["N"], L["req"]
    F, z = features(L, {"head": ["z"], "head+prev": ["z", "prev"], "head+prev+step": ["z", "prev", "step"]})
    models = {
        "head refit (z_i)": lambda F, z, i: z[:, [i]],
        "head (all z)": lambda F, z, i: F["head"],
        "head + last rounds": lambda F, z, i: F["head+prev"],
        "head + last rounds + step": lambda F, z, i: F["head+prev+step"],
    }
    rows = {"raw head": choose_depth(L["C"], cost)[0]}
    for name, pf in models.items():
        rows[name] = choose_depth(cross_predict(F, z, A, req, a.folds, pf), cost)[0]
    rows["oracle"] = A
    doms = sorted(set(L["dom"]))
    print(f"{len(A):,} rounds, {len(set(req))} requests, k {k}, cost {','.join(f'{c:.2f}' for c in cost)}")
    base = replay(rows["raw head"], A, N, cost)["tok_s"]
    print(f"{'policy':28s} {'depth':>6s} {'tok/rd':>7s} {'ms/rd':>7s} {'tok/s':>7s} {'vs head':>8s}")
    for name, d in rows.items():
        r = replay(d, A, N, cost)
        print(f"{name:28s} {r['depth']:6.2f} {r['tok_round']:7.3f} {r['ms_round']:7.2f} {r['tok_s']:7.1f} "
              f"{100 * (r['tok_s'] / base - 1):+7.1f}%")
    print("\nby domain (tok/s, vs the raw head):")
    print(f"{'domain':14s} {'rounds':>7s} " + " ".join(f"{n[:14]:>15s}" for n in rows))
    for dmn in doms:
        m = np.array([x == dmn for x in L["dom"]])
        b = replay(rows["raw head"][m], A[m], N[m], cost)["tok_s"]
        cells = []
        for name, d in rows.items():
            t = replay(d[m], A[m], N[m], cost)["tok_s"]
            cells.append(f"{t:7.1f} {100 * (t / b - 1):+5.1f}%")
        print(f"{dmn:14s} {int(m.sum()):7d} " + " ".join(f"{c:>15s}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
