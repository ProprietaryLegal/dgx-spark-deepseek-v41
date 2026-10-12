# SPDX-License-Identifier: Apache-2.0
"""Per-round GPU time by kernel family for one rank's decode capture, from either engine, on one ruler: TensorFold
under Nsight Systems (the CSV of `nsys stats --report cuda_gpu_trace --format csv`) or ours under vLLM's torch
profiler (`*.pt.trace.json.gz`). Kernel families are matched on names by the regex table below (both engines' names);
`--top` lists the biggest kernels per round so unmatched names can be mapped.

The window is the decode part of the capture: from the first kernel + `--skip-ms` (or the client's time to first
token, `--client prof_stream.json`, which also gives the rounds as chunks - 1) to the last kernel; or, for our traces,
`--after REGEX` starts it at the first kernel after the last one matching REGEX (the prefill MoE: exl3_moe_kernel).

Per round it prints: the span, the busy union (time with at least one kernel running), idle = span - busy, the number
of kernels, and each family's summed kernel time (kernels on concurrent streams overlap, so families can sum past the
busy time) and its exclusive share of the busy union (time where that family is the only one running, the part a
faster kernel of that family would take off the round).
    python3 trace_rounds.py tf_spark_code_cuda_gpu_trace.csv --client samebox_code.json [--top 30]
    python3 trace_rounds.py rank0.pt.trace.json.gz --rounds 30 --after exl3_moe_kernel [--top 30]
"""
import argparse
import collections
import csv
import gzip
import json
import re

FAM = [
    ("experts", r"smoe_|grouped_cp|grouped_kernel|grouped_rows|grouped_mma|exl3_moe|moe_|expert"),
    ("dense linears", r"mrow_kernel|glinear|linear_grouped|exl3_gemm|exl3_gemv|gemv|marlin|rot_many|deep_gemm"),
    ("router / top-k", r"route|router|rowmm|_dsv4_topk|topk_kernel|splitKreduce|cutlass.*16x16_128x2"),
    ("attention", r"sparse_attn|sparse_mla|attn_merge|flashinfer|fmha|attention|flash_"),
    ("indexer / compressor", r"indexer|compress|topk_sel|global_topk|kv_norm|QNormRope|qnorm"),
    ("mHC", r"_hc_|mhc|sinkhorn|hc_mix|hc_post|hc_pre|hc_finish|hyper"),
    ("communication", r"nccl|rdma|ring_|collect|gather|allreduce|AllReduce|all_reduce|reduce_scatter|sum_kernel"),
    ("L2 prefetch", r"prefetch|l2pf"),
    ("drafter / Markov", r"markov|draft|dspark|mtp|confidence"),
    ("head / logits", r"lm_head|vocab|logits|argmax|sampl"),
    ("norm / rope / act", r"norm|rope|silu|act_and_mul|act_|swiglu|gelu"),
    ("copy / elementwise", r"copy|fill|cast|elementwise|memcpy|memset|CUDA mem"),
    ("cuBLAS / CUTLASS (other)", r"cutlass|cublas|gemm|Kernel2"),
]


def fam(name):
    for f, rx in FAM:
        if re.search(rx, name, re.I):
            return f
    return "other"


def load(path):
    """[(start_ns, dur_ns, name, stream)] sorted by start"""
    rows = []
    if path.endswith(".csv"):
        with open(path) as f:
            for r in csv.DictReader(f):
                name = r.get("Name") or r.get("name") or ""
                try:
                    s = float(r.get("Start (ns)") or r.get("Start") or "nan")
                    d = float(r.get("Duration (ns)") or r.get("Duration") or "nan")
                except ValueError:
                    continue
                if s == s and d == d:
                    rows.append((s, d, name, r.get("Strm") or r.get("Stream") or ""))
    else:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt") as f:
            ev = json.load(f)["traceEvents"]
        for e in ev:
            if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e:
                rows.append((e["ts"] * 1e3, e["dur"] * 1e3, e.get("name", ""), str(e.get("args", {}).get("stream", ""))))
    rows.sort()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--rounds", type=int, help="verify rounds in the window (default: the client's chunks - 1)")
    ap.add_argument("--client", help="prof_stream.py's JSON line for this capture: rounds and time to first token")
    ap.add_argument("--skip-ms", type=float, help="drop this much from the start of the capture (prefill)")
    ap.add_argument("--after", help="start after the last kernel whose name matches this regex (our prefill MoE)")
    ap.add_argument("--top", type=int, default=0)
    a = ap.parse_args()
    rows = load(a.trace)
    if not rows:
        raise SystemExit("no kernel rows")
    rounds, skip = a.rounds, a.skip_ms
    if a.client:
        c = json.loads(open(a.client).read().strip().splitlines()[-1])
        rounds = rounds or c["chunks"] - 1
        skip = skip if skip is not None else c["ttft_s"] * 1e3
    if not rounds:
        raise SystemExit("need --rounds or --client")
    t_first = rows[0][0]
    if a.after:
        last = max((i for i, r in enumerate(rows) if re.search(a.after, r[2])), default=-1)
        rows = rows[last + 1:]
    elif skip:
        rows = [r for r in rows if r[0] >= t_first + skip * 1e6]
    t0 = rows[0][0]
    t1 = max(s + d for s, d, _, _ in rows)
    span = t1 - t0
    # busy union and each family's exclusive time, by sweeping kernel start / end events
    evs = []
    for s, d, n, _ in rows:
        f = fam(n)
        evs.append((s, 1, f))
        evs.append((s + d, -1, f))
    evs.sort(key=lambda e: (e[0], e[1]))
    active = collections.Counter()
    busy = 0.0
    excl = collections.defaultdict(float)
    prev = t0
    for t, kind, f in evs:
        dt = t - prev
        if dt > 0:
            live = [k for k, v in active.items() if v > 0]
            if live:
                busy += dt
                if len(live) == 1:
                    excl[live[0]] += dt
                else:
                    excl["(overlapped)"] += dt
            prev = t
        active[f] += kind
        prev = max(prev, t)
    tot = collections.defaultdict(float)
    cnt = collections.Counter()
    by_name = collections.defaultdict(float)
    by_name_n = collections.Counter()
    for s, d, n, _ in rows:
        f = fam(n)
        tot[f] += d
        cnt[f] += 1
        by_name[n[:100]] += d
        by_name_n[n[:100]] += 1
    R = rounds
    print(f"{a.trace.split('/')[-1]}: {R} rounds, {len(rows)} GPU ops in the window")
    print(f"per round: span {span / 1e6 / R:6.2f} ms | busy {busy / 1e6 / R:6.2f} | idle {(span - busy) / 1e6 / R:5.2f} | "
          f"kernels {len(rows) / R:5.0f}")
    print(f"  {'family':28s} {'sum ms':>7s} {'excl ms':>7s} {'n/round':>8s}")
    for f, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {f:28s} {v / 1e6 / R:7.3f} {excl.get(f, 0) / 1e6 / R:7.3f} {cnt[f] / R:8.1f}")
    print(f"  {'(overlapped)':28s} {'':7s} {excl.get('(overlapped)', 0) / 1e6 / R:7.3f}")
    if a.top:
        print(f"top {a.top} kernels by time per round:")
        for n, v in sorted(by_name.items(), key=lambda kv: -kv[1])[: a.top]:
            print(f"  {v / 1e6 / R:7.3f} ms  x{by_name_n[n] / R:6.1f}  [{fam(n)}] {n}")


if __name__ == "__main__":
    main()
