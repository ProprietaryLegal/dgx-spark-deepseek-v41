"""GPU kernel timeline of a vLLM torch-profiler trace (profile_decode.sh writes them), per decode step: the span, busy
time as the union of kernel intervals (concurrent streams counted once), per-family union time, NCCL per-call stats,
and an exclusive attribution where overlapping time goes to the earliest-started kernel (with PDL the next kernel
launches early and waits inside, so its own duration includes the previous kernel's tail).

  python3 trace_union.py <rank0 trace .json.gz> <decode steps in the window>
"""
import gzip, json, re, sys
path, steps = sys.argv[1], int(sys.argv[2])
d = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
FAM = [("routed experts (smoe, ours)", r"smoe_"), ("routed experts (coop decode)", r"exl3_moe_coop"), ("routed experts (prefill exl3_moe)", r"exl3_moe_kernel"),
       ("head (EXL3 K6 gemm)", r"exl3_gemm_kernel<6"), ("dense EXL3 linears (K5)", r"exl3_gemm_kernel<5|exl3_gemv"), ("dense exl3_mrow (ours)", r"mrow_kernel"), ("fused MoE ops (ours)", r"_moe_prep_kernel|_sum_groups_kernel"),
       ("drafter MXFP4 experts (deep_gemm)", r"deep_gemm|fp8_fp4"), ("Marlin MXFP8 linears", r"[Mm]arlin"),
       ("nccl", r"nccl"), ("RDMA ring (ours)", r"stage_kernel|sum_kernel"), ("L2 prefetch (ours)", r"l2pf_kernel"),
       ("Markov bias (ours)", r"_markov_bias_kernel"),
       ("cuBLAS / CUTLASS bf16", r"cutlass|gemm|Gemm|cublas|nvjet"), ("mHC", r"mhc|sinkhorn|tilelang"),
       ("attention", r"mla|attention|flashinfer|fmha"), ("indexer / top-k", r"topk|top_k|indexer|radix|sort"),
       ("engram", r"engram"), ("copy / elementwise", r"elementwise|copy|Copy|fill|cat|index|scatter|gather|reduce|Memcpy|Memset")]
def fam(n):
    for f, rx in FAM:
        if re.search(rx, n):
            return f
    return "other"
def union(iv):
    iv = sorted(iv); tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if cs is not None else 0)
iv = [(e["ts"], e["ts"] + e["dur"]) for e in ev]
span = max(e for _, e in iv) - min(s for s, _ in iv)
busy = union(iv); total = sum(e - s for s, e in iv)
print(f"{len(ev)} GPU events, {steps} steps: span {span / 1e3 / steps:.2f} ms/step, busy (union) {busy / 1e3 / steps:.2f} ms/step, "
      f"kernel sum {total / 1e3 / steps:.2f} ms/step (overlap {100 * (1 - busy / total):.1f}%)")
by = {}
for e in ev:
    by.setdefault(fam(e["name"]), []).append((e["ts"], e["ts"] + e["dur"]))
rows = sorted(((union(v), sum(b - a for a, b in v), f, len(v)) for f, v in by.items()), reverse=True)
print(f"{'family':38s} {'union ms/step':>13s} {'sum ms/step':>12s} {'calls/step':>10s}")
for u, s, f, n in rows:
    print(f"{f:38s} {u / 1e3 / steps:13.2f} {s / 1e3 / steps:12.2f} {n / steps:10.1f}")
streams = {}
for e in ev:
    streams.setdefault(e.get("tid"), 0); streams[e.get("tid")] += e["dur"]
print("busy by stream (ms/step):", {k: round(v / 1e3 / steps, 2) for k, v in sorted(streams.items(), key=lambda x: -x[1])[:8]})

# NCCL per call (all-reduce kernels), to compare in-graph and eager
nc = sorted(e["dur"] for e in ev if re.search(r"nccl", e.get("name", "")))
if nc:
    print(f"nccl: {len(nc) / steps:.1f} calls/step, mean {sum(nc) / len(nc):.1f} us, median {nc[len(nc) // 2]:.1f} us, "
          f"p90 {nc[int(len(nc) * 0.9)]:.1f} us")

# Exclusive attribution: a time slice where several kernels overlap goes to the one that started first (with PDL the
# next kernel launches early and waits inside, so its trace duration includes the previous kernel's tail)
pts = sorted(((e["ts"], e["ts"] + e["dur"], fam(e["name"])) for e in ev))
excl = {}
import heapq
active = []          # (start, end, family) of running kernels, earliest start first
t_prev = None
events = sorted([(s, 0, i) for i, (s, e, f) in enumerate(pts)] + [(e, 1, i) for i, (s, e, f) in enumerate(pts)])
running = set()
for t, kind, i in events:
    if t_prev is not None and running:
        first = min(running, key=lambda j: pts[j][0])
        excl[pts[first][2]] = excl.get(pts[first][2], 0) + (t - t_prev)
    if kind == 0:
        running.add(i)
    else:
        running.discard(i)
    t_prev = t
print(f"{'family (exclusive: overlap to the earliest-started kernel)':58s} {'ms/step':>8s}")
for f, v in sorted(excl.items(), key=lambda x: -x[1]):
    print(f"  {f:56s} {v / 1e3 / steps:8.2f}")
