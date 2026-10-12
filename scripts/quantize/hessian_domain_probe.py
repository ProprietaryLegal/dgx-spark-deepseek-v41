# SPDX-License-Identifier: Apache-2.0
"""Is an EXL3 expert encode tuned to its calibration mix? One layer, a sample of experts, two encodes each at the
plan's K, three Hessians: "v2like" (the original calibration sequences as they come, token-weighted: plan v2's encode,
42% code), "tokw_all" (plus the v3 supplement, token-weighted) and "bal" (calibration + supplement re-weighted so every
domain counts equally); then each encode's relative output error on HELD-OUT tokens, per domain. If the token-weighted encode is much worse on chat / writing / multilingual tokens than on code and the
balanced one evens that out, the fix for build v2.1's domain gap is a re-encode with balanced Hessians, not only a
re-allocation (the allocator's eps(K) is one number per K, measured on the calibration mix).

Native stream (reference implementation) up to the layer; sequences split per domain into a Hessian half and a held-out
half (seeded); the tau prior of pass 1 (all-token H) included, weighted the same way as the expert Hessians.

  python3 hessian_domain_probe.py /model cal_v3.safetensors cal_v3_index.json plan.json /out --layer 10 --experts 24
Writes /out/probe_L<layer>.json and prints per-domain errors per K.
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.exl3 import encode_expert, expert_forward_dense, h_data, mxfp4_dequant, reconstruct  # noqa: E402
from dsv41q.stream import Streamer  # noqa: E402
from p4_pass0 import coarse  # noqa: E402

DOMAINS_EVAL = ["code", "agentic", "long", "chat", "writing", "knowledge", "math", "multilingual", "instruct", "std"]


def kval(c):
    k = {"a": 1.5, "b": 2.5, "c": 3.5}.get(c) or float(c)
    return int(k) if float(k).is_integer() else k


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cal")
    ap.add_argument("index")
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--experts", type=int, default=24)
    ap.add_argument("--frac", type=float, default=0.5, help="fraction of each domain's sequences kept (speed)")
    ap.add_argument("--tau", type=float, default=512.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load_file(a.cal)
    t, o = d["tokens"].tolist(), d["offsets"].tolist()
    seqs = [t[o[i]: o[i + 1]] for i in range(len(o) - 1)]
    idx = json.load(open(a.index))
    assert len(idx) == len(seqs)
    rng = random.Random(a.seed)
    by = {}
    for i, r in enumerate(idx):
        by.setdefault(coarse(r["domain"]), []).append(i)
    keep, role = [], {}
    for dom, ids in sorted(by.items()):
        if dom == "random":
            continue
        rng.shuffle(ids)
        ids = ids[: max(2, round(len(ids) * a.frac))]
        for j, i in enumerate(ids):
            keep.append(i)
            role[i] = "fit" if j % 2 == 0 else "test"
    keep.sort()
    seqs_k = [seqs[i] for i in keep]
    dom_tok, fit_tok, cal_tok = [], [], []
    names = sorted({coarse(idx[i]["domain"]) for i in keep})
    did = {n: j for j, n in enumerate(names)}
    for i in keep:
        n = len(seqs[i])
        dom_tok.append(torch.full((n,), did[coarse(idx[i]["domain"])], dtype=torch.long))
        fit_tok.append(torch.full((n,), role[i] == "fit", dtype=torch.bool))
        cal_tok.append(torch.full((n,), idx[i].get("source") != "instruct_v3", dtype=torch.bool))
    dom_tok, fit_tok, cal_tok = torch.cat(dom_tok).cuda(), torch.cat(fit_tok).cuda(), torch.cat(cal_tok).cuda()
    torch.set_default_dtype(torch.bfloat16)
    s = Streamer(a.model, seqs_k)
    N = sum(map(len, seqs_k))
    print(f"{len(seqs_k)} sequences, {N:,} tokens (fit {int(fit_tok.sum()):,}), domains {names}", flush=True)
    t0 = time.time()
    for l in range(a.layer):
        blk = s.load(l)
        s.run_layer(blk)
        del blk
        torch.cuda.empty_cache()
    print(f"streamed to layer {a.layer} in {time.time() - t0:.0f}s", flush=True)
    blk = s.load(a.layer)
    flat, _ = s.phase_a(blk, keep_pend=False)
    W_, I_ = [], []
    for c in range(0, N, 65536):
        w, ix = blk.ffn.gate(flat[c: c + 65536], None)
        W_.append(w.float())
        I_.append(ix)
    Wr, Ir = torch.cat(W_), torch.cat(I_)
    limit = s.args.swiglu_limit
    # per-token weights over the fit half: token-weighted = 1; balanced = (1 / n_domain) scaled to mean 1
    fit_dom = dom_tok[fit_tok]
    cnt = torch.bincount(fit_dom, minlength=len(names)).float()
    present = (cnt > 0).float()
    w_bal_dom = present / cnt.clamp_min(1) / present.sum() * fit_tok.sum()
    weights = {"v2like": (fit_tok & cal_tok).float(), "tokw_all": fit_tok.float(), "bal": w_bal_dom[dom_tok] * fit_tok}
    plan = json.load(open(a.plan))["layers"][str(a.layer)]["k"]
    # experts with enough routed tokens in both halves, spread over the plan's K values
    counts = torch.bincount(Ir.flatten(), minlength=384)
    cand = [e for e in range(384) if counts[e] > 2000]
    per_k = {}
    for e in cand:
        per_k.setdefault(plan[e], []).append(e)
    pick = []
    for kc in sorted(per_k):
        r2 = random.Random(a.seed + ord(kc))
        pick += r2.sample(per_k[kc], min(len(per_k[kc]), max(2, a.experts * len(per_k[kc]) // len(cand))))
    pick = sorted(pick)[: a.experts]
    print(f"experts {pick} (K {[plan[e] for e in pick]})", flush=True)
    H_all = {}
    for wk, wt in weights.items():
        H = torch.zeros(5120, 5120, dtype=torch.float32, device="cuda")
        for c in range(0, N, 32768):
            xc = flat[c: c + 32768].float() * wt[c: c + 32768, None].sqrt()
            H.addmm_(xc.t(), xc)
        H_all[wk] = H
    mean_g2 = Wr.square().mean().item()
    gsamp = torch.Generator(device="cpu").manual_seed(1000 + a.layer)
    fit_idx = fit_tok.nonzero().flatten()
    xs = flat[fit_idx[torch.randperm(fit_idx.numel(), generator=gsamp, device="cpu")[:1024].to(fit_idx.device)]].float()
    res = {"layer": a.layer, "domains": names, "experts": {}}
    for e in pick:
        ex = blk.ffn.experts[e]
        W1 = mxfp4_dequant(ex.w1.weight, ex.w1.scale)
        W3 = mxfp4_dequant(ex.w3.weight, ex.w3.scale)
        W2 = mxfp4_dequant(ex.w2.weight, ex.w2.scale)
        tok, slot = (Ir == e).nonzero(as_tuple=True)
        xe, ge = flat[tok], Wr[tok, slot]
        te = ~fit_tok[tok]
        yn = expert_forward_dense(xe, ge, W1.half(), W3.half(), W2.half(), limit)
        K = kval(plan[e])
        rec = {"K": K, "n": int(tok.numel()), "errors": {}}

        def inter(x, g):
            gate = (x @ W1.t()).clamp(max=limit)
            up = (x @ W3.t()).clamp(-limit, limit)
            return (F.silu(gate) * up * g[:, None]).to(torch.bfloat16).float()

        for wk, wt in weights.items():
            we = wt[tok]
            n_eff = float(we.sum())
            xg = xe.float() * (ge * we.sqrt())[:, None]
            H_gu = xg.t() @ xg
            H_gu.add_(H_all[wk], alpha=a.tau * mean_g2 / float(wt.sum()))
            act = inter(xe.float(), ge) * we.sqrt()[:, None]
            H_d = act.t() @ act
            ap_ = inter(xs, torch.full((xs.size(0),), mean_g2 ** 0.5, device=xs.device))
            H_d.add_(ap_.t() @ ap_, alpha=a.tau / xs.size(0))
            tens, _ = encode_expert(W1, W3, W2, h_data(H_gu, int(n_eff + a.tau), f"e{e}gu"),
                                    h_data(H_d, int(n_eff + a.tau), f"e{e}d"), K, K, seed=a.layer * 1000 + e,
                                    work_dir=a.out)
            q = [reconstruct(tens[p], *((5120, 2304) if p != "w2" else (2304, 5120))) for p in ("w1", "w3", "w2")]
            yq = expert_forward_dense(xe, ge, *q, limit)
            err = (yq - yn).square().sum(-1)
            en = yn.square().sum(-1)
            dd = dom_tok[tok]
            per = {}
            for j, nm in enumerate(names):
                m = te & (dd == j)
                if int(m.sum()) >= 20:
                    per[nm] = [float(err[m].sum() / en[m].sum().clamp_min(1e-30)), int(m.sum())]
            per["all_test"] = [float(err[te].sum() / en[te].sum().clamp_min(1e-30)), int(te.sum())]
            rec["errors"][wk] = per
        res["experts"][e] = rec
        print(f"e{e:3d} K{K} n={rec['n']}: " + "  ".join(
            f"{nm} " + "/".join(f"{rec['errors'][wk].get(nm, [float('nan')])[0]:.4f}" for wk in weights)
            for nm in ("code", "agentic", "chat", "writing", "multilingual", "instruct", "all_test")), flush=True)
    json.dump(res, open(os.path.join(a.out, f"probe_L{a.layer:02d}.json"), "w"), indent=1)
    # summary: per K and domain, mean relative error (token-weighted vs balanced) over the sampled experts
    print("summary: mean rel output error on held-out tokens, Hessian v2like / tokw_all / bal")
    for kk in sorted({r["K"] for r in res["experts"].values()}) + ["all"]:
        rs = [r for r in res["experts"].values() if kk == "all" or r["K"] == kk]
        cells, dom_means = [], {wk: [] for wk in weights}
        for nm in DOMAINS_EVAL + ["all_test"]:
            vals = {}
            for wk in weights:
                v = [r["errors"][wk][nm][0] for r in rs if nm in r["errors"][wk]]
                if v:
                    vals[wk] = sum(v) / len(v)
                    if nm != "all_test":
                        dom_means[wk].append(vals[wk])
            if vals:
                cells.append(f"{nm} " + "/".join(f"{vals[wk]:.4f}" for wk in weights if wk in vals))
        bal_line = "/".join(f"{sum(v) / len(v):.4f}" for v in dom_means.values() if v)
        print(f"  K{kk} ({len(rs)} experts): " + ", ".join(cells) + f" | domain mean {bal_line}")


if __name__ == "__main__":
    main()
