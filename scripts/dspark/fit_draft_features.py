#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Do the drafts' own logit margins and top-1 probabilities predict acceptance beyond DSpark's confidence head?
Offline, on a fixed:k VLLM_DSPARK_CONF_LOG round log with VLLM_DSPARK_CONF_LOG_FEATURES=1 (patch 0034: each line's "f"
holds margin_i, p1_i for every draft position). Oct 21 2026.

Per position i, a ridge logistic model of "draft i accepted given 1..i-1 were" on: the head's logit z_i alone (a refit),
z_i + the draft's (top-1 - top-2) logit margin + logit(p1_i), the draft features alone, and z of every position + the
draft features of position i; cross-fitted by request (each request scored by models fitted without it), as
predictor_research.py. Reported: AUC per position (all rounds, writing, chat) and patch 0010's depth rule replayed on each
model's probabilities under the cost table, tok/s against the raw head, overall and by domain, next to the oracle.
  python3 fit_draft_features.py conflog.jsonl [--folds 4] [--round-ms 27.9,32.2,...] [--export feat_head.npz]

--export fits "head + draft" on every round (no cross-fit) and writes w [k, 4] per position (intercept, z, margin,
logit p1): the file patch 0036 serves (VLLM_DSPARK_CONF_FEAT_HEAD).
"""
import argparse
import json
import re
import sys
import zlib

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fit_conf_temps import EPS, TAG_RE, choose_depth, cost_table, replay  # noqa: E402
from predictor_research import logreg, predict  # noqa: E402


def load(path):
    """Rounds with features: C [R, k], F [R, k, 2] (margin, p1), D, N, A, request ids, domains; headers."""
    rx = re.compile(TAG_RE)
    heads, C, F, D, N, A, req, dom = [], [], [], [], [], [], [], []
    hdr = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("hdr"):
            hdr = r
            heads.append(r)
            continue
        if "f" not in r or str(r["q"]).startswith("_warmup"):
            continue
        k = len(r["c"])
        if len(r["f"]) != 2 * k:
            continue
        C.append(r["c"]); F.append(np.asarray(r["f"], dtype=np.float64).reshape(k, 2))
        D.append(int(r["d"])); N.append(int(r["n"])); A.append(int(r["a"])); req.append(str(r["q"]))
        m = rx.search(req[-1])
        dom.append(m.group(1) if m else hdr.get("tag") or "all")
    return {"C": np.asarray(C, dtype=np.float64), "F": np.asarray(F), "D": np.asarray(D), "N": np.asarray(N),
            "A": np.asarray(A), "req": req, "dom": np.asarray(dom), "headers": heads}


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort"); ps = p[order]; ranks = np.empty(len(p)); i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cross_predict(X_of, A, req, k, folds):
    """[R, k] probabilities, each request scored by per-position models fitted without it. X_of(i) -> [R, f]."""
    fold = np.array([zlib.crc32(q.encode()) % folds for q in req])
    Q = np.zeros((len(A), k))
    for f in range(folds):
        tr, te = fold != f, fold == f
        for i in range(k):
            X = X_of(i)
            obs = tr & (A >= i)
            w = logreg(X[obs], (A[obs] >= i + 1).astype(np.float64))
            Q[te, i] = predict(w, X[te])
    return Q


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("--folds", type=int, default=4); ap.add_argument("--round-ms")
    ap.add_argument("--export", help="write the head + draft model fitted on every round as an .npz (patch 0036)")
    a = ap.parse_args(argv)
    L = load(a.log)
    C, F, A, N, req, dom = L["C"], L["F"], L["A"], L["N"], L["req"], L["dom"]
    if not len(C):
        sys.exit("no rounds with features (VLLM_DSPARK_CONF_LOG_FEATURES=1)")
    if (np.minimum(L["D"], N) < N).any():
        sys.exit("needs a fixed:k log (every draft verified while its prefix survives)")
    k = C.shape[1]
    hdr = L["headers"][-1] if L["headers"] else {}
    cost = cost_table(a.round_ms, k) if a.round_ms else cost_table(",".join(map(str, hdr["cost_ms"])), k)
    z = logit(C)
    marg = F[:, :, 0]
    zp1 = logit(F[:, :, 1])
    models = {
        "head refit": lambda i: z[:, [i]],
        "head + draft": lambda i: np.stack([z[:, i], marg[:, i], zp1[:, i]], 1),
        "draft only": lambda i: np.stack([marg[:, i], zp1[:, i]], 1),
        "all z + draft": lambda i: np.concatenate([z, np.stack([marg[:, i], zp1[:, i]], 1)], 1),
    }
    probs = {"raw head": C}
    for name, X_of in models.items():
        probs[name] = cross_predict(X_of, A, req, k, a.folds)
    print(f"{len(A):,} rounds with features, {len(set(req))} requests, k {k}, cost {','.join(f'{c:.2f}' for c in cost)}")
    print("\nAUC of draft i accepted given 1..i-1 were (per position):")
    for dname in ("ALL", "writing", "chat", "code"):
        m0 = np.ones(len(A), bool) if dname == "ALL" else dom == dname
        if not m0.any():
            continue
        for name, Q in probs.items():
            cells = []
            for i in range(k):
                obs = m0 & (A >= i)
                cells.append(f"{auc(Q[obs, i], (A[obs] >= i + 1)):.3f}")
            print(f"  {dname:8s} {name:14s} " + " ".join(cells))
    rows = {name: choose_depth(Q, cost)[0] for name, Q in probs.items()}
    rows["oracle"] = np.minimum(A, N)
    base = replay(rows["raw head"], A, N, cost)["tok_s"]
    print(f"\n{'policy':16s} {'depth':>6s} {'tok/rd':>7s} {'ms/rd':>7s} {'tok/s':>7s} {'vs head':>8s}")
    for name, d in rows.items():
        r = replay(d, A, N, cost)
        print(f"{name:16s} {r['depth']:6.2f} {r['tok_round']:7.3f} {r['ms_round']:7.2f} {r['tok_s']:7.1f} "
              f"{100 * (r['tok_s'] / base - 1):+7.1f}%")
    print("\nby domain (tok/s, vs the raw head):")
    print(f"{'domain':14s} {'rounds':>7s} " + " ".join(f"{n[:14]:>15s}" for n in rows))
    for dmn in sorted(set(dom)):
        m = dom == dmn
        b = replay(rows["raw head"][m], A[m], N[m], cost)["tok_s"]
        cells = []
        for name, d in rows.items():
            t = replay(d[m], A[m], N[m], cost)["tok_s"]
            cells.append(f"{t:7.1f} {100 * (t / b - 1):+5.1f}%")
        print(f"{dmn:14s} {int(m.sum()):7d} " + " ".join(f"{c:>15s}" for c in cells))
    if a.export:
        X_of = models["head + draft"]
        W = []
        for i in range(k):
            obs = A >= i
            W.append(logreg(X_of(i)[obs], (A[obs] >= i + 1).astype(np.float64)))
        W = np.asarray(W)                                  # [k, 4]: intercept, z, margin, logit p1
        np.savez(a.export, w=W)
        # the serve's arithmetic (fp32 clamps, as torch.logit) against the fit's
        c32 = np.clip(C.astype(np.float32), np.float32(EPS), np.float32(1 - EPS)).astype(np.float64)
        p32 = np.clip(F[:, :, 1].astype(np.float32), np.float32(EPS), np.float32(1 - EPS)).astype(np.float64)
        u = W[:, 0] + W[:, 1] * (np.log(c32) - np.log1p(-c32)) + W[:, 2] * marg + W[:, 3] * (np.log(p32) - np.log1p(-p32))
        fit = np.stack([predict(W[i], X_of(i)) for i in range(k)], 1)
        print(f"\nexported {a.export}: w {W.shape}, max |serve - fit| {np.abs(1 / (1 + np.exp(-u)) - fit).max():.2e}")
        for i in range(k):
            print(f"  position {i + 1}: " + " ".join(f"{x:+.4f}" for x in W[i]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
