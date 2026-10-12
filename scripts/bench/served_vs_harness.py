#!/usr/bin/env python3
"""Served logits vs the layer-streamed reference harness, position by position, on one kit_bench oracle file.

kit_bench's oracle holds two served views of the same positions: the decode path (`gen_top`, top-20 at each of the
greedy steps) and the prefill path (`tf_top`, teacher-forced `prompt_logprobs` top-20 over prompt + continuation).
Score both against the native checkpoint (p4_kld.py --mode ref: top-128) and against our own weights run through the
same harness (p4_kld.py --mode quant --exl3-ckpt ... --dump-top 20). If the decode path sits as close to the harness as
the prefill path does, run-to-run spread is the model's own sensitivity (routing near-ties); if only the prefill path
strays, a prefill kernel computes something else.

  python3 served_vs_harness.py panel ORACLE.jsonl OUTDIR       -> OUTDIR/panel.safetensors + panel_index.json
      (prompt + continuation per prompt, the continuation's tokens scored), then on a Spark:
      p4_kld.py /official OUTDIR/panel.safetensors OUTDIR/panel_index.json OUTDIR --mode ref
      p4_kld.py /official ... OUTDIR --mode quant --exl3-ckpt /models/dsv41-recipe-exl3-v2.1 --dump-top 20
  python3 served_vs_harness.py compare ORACLE.jsonl OUTDIR      -> the table
"""
import json
import math
import os
import sys

import torch
from safetensors.torch import load_file, save_file


def kl_trunc(p, q):
    """KL(p || q) over p's listed tokens, both renormalised; q's unlisted tokens get q's lowest listed logprob - 1."""
    floor = min(q.values()) - 1.0
    zp = math.log(sum(math.exp(v) for v in p.values()))
    zq = math.log(sum(math.exp(q.get(t, floor)) for t in p))
    return sum(math.exp(v - zp) * ((v - zp) - (q.get(t, floor) - zq)) for t, v in p.items())


def records(path):
    return [json.loads(line) for line in open(path)]


def panel(oracle, out):
    os.makedirs(out, exist_ok=True)
    toks, offs, score, idx = [], [0], [], []
    for r in records(oracle):
        seq = r["prompt_ids"] + r["gen_ids"]
        toks += seq
        offs.append(len(toks))
        score += [0] * len(r["prompt_ids"]) + [1] * len(r["gen_ids"])
        idx.append({"id": f"oracle{r['index']}", "domain": f"oracle_{r['kind']}", "length": len(seq)})
    save_file({"tokens": torch.tensor(toks, dtype=torch.int32), "offsets": torch.tensor(offs, dtype=torch.int64),
               "score": torch.tensor(score, dtype=torch.uint8)}, os.path.join(out, "panel.safetensors"))
    json.dump(idx, open(os.path.join(out, "panel_index.json"), "w"), indent=0)
    print(f"panel: {len(idx)} sequences, {len(toks)} tokens, {sum(score)} scored")


def compare(oracle, out):
    ref, qt = load_file(os.path.join(out, "ref.safetensors")), load_file(os.path.join(out, "quant_top.safetensors"))
    rows = {"decode": [], "prefill": [], "harness": []}
    o = 0
    for r in records(oracle):
        P, n = len(r["prompt_ids"]), len(r["gen_ids"])
        for j in range(n):          # scored token P + j, predicted from position P + j - 1
            nat = {str(int(t)): float(v) for t, v in zip(ref["ids"][o + j], ref["logprobs"][o + j])}
            har = {str(int(t)): float(v) for t, v in zip(qt["ids"][o + j], qt["logprobs"][o + j])}
            views = {"harness": har}
            if j < len(r["gen_top"]) and r["gen_top"][j]:
                views["decode"] = {k: float(v) for k, v in r["gen_top"][j].items()}
            if P + j < len(r["tf_top"]) and r["tf_top"][P + j]:
                views["prefill"] = {k: float(v) for k, v in r["tf_top"][P + j].items()}
            nat_top = max(nat, key=nat.get)
            for name, v in views.items():
                rows[name].append((kl_trunc(nat, v), kl_trunc(har, v) if name != "harness" else 0.0,
                                   max(v, key=v.get) == nat_top, max(v, key=v.get) == max(har, key=har.get)))
        o += n
    print(f"{'path':8s} {'n':>5s} {'KL(native||path)':>17s} {'p99':>7s} {'top1=native':>12s} "
          f"{'KL(harness||path)':>18s} {'p99':>7s} {'max':>7s} {'top1=harness':>13s}")
    for name, v in rows.items():
        if not v:
            continue
        kn = sorted(x[0] for x in v)
        kh = sorted(x[1] for x in v)
        print(f"{name:8s} {len(v):5d} {sum(kn) / len(v):17.5f} {kn[int(0.99 * len(v))]:7.4f} "
              f"{100 * sum(x[2] for x in v) / len(v):11.2f}% {sum(kh) / len(v):18.5f} {kh[int(0.99 * len(v))]:7.4f} "
              f"{kh[-1]:7.3f} {100 * sum(x[3] for x in v) / len(v):12.2f}%")


if __name__ == "__main__":
    {"panel": panel, "compare": compare}[sys.argv[1]](sys.argv[2], sys.argv[3])
