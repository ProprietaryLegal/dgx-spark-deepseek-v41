# SPDX-License-Identifier: Apache-2.0
"""Where a prefill's GPU time goes, by kernel family, for one rank of either engine on one ruler: ours from vLLM's
torch profiler (`*.pt.trace.json.gz`, the window between /start_profile and /stop_profile) or TensorFold's from Nsight
Systems (`nsys stats --report cuda_gpu_trace --format csv`). Per trace: the span (first kernel start to last kernel
end), the busy union (time with at least one kernel running), each family's kernel-time sum and its exclusive share of
the busy union (time where only that family runs), and the top kernels by time.
    python3 prefill_compare.py ours=<rank0 trace.json.gz> tf=<rank0 cuda_gpu_trace.csv> [--top 25]
"""
import argparse
import collections
import csv
import gzip
import json
import re

FAM = [   # (family, regex on the kernel name), first match wins; both engines' names
    ("routed experts", r"exl3_moe|smoe_|grouped_mma|grouped_rows|grouped_kernel|grouped_cp|experts|expert|"
                       r"reconstruct_kernel<[23],|nvjet_sm121_hss|had_ff_r_128|had_hf_r_128|swiglu|combine"),
    ("dense linears", r"mrow_kernel|glinear|linear_kernel|linear_grouped|unpack_kernel|rot_in|rot_many|_gemm\b|"
                      r"exl3_gemm|reconstruct_had|reconstruct_kernel<4,|cutlass_80_tensorop_s16816gemm_f16|[Mm]arlin|deep_gemm"),
    ("attention", r"sparse_mla|sparse_attn|attn_merge|prefill_attention|flash|fmha|attention"),
    ("indexer / compressor", r"indexer|compress|topk|global_topk|kv_norm|QNormRope|qnorm|_dsv4_topk"),
    ("mHC", r"mhc|_hc_|hc_prenorm|sinkhorn|hyper|tilelang"),
    ("communication", r"nccl|rdma|ring_|allreduce|AllReduce|all_reduce|AllGather|\bsum_kernel\(|\bstage_kernel\("),
    ("engram", r"engram"),
    ("router", r"router|route|gate|rowmm"),
    ("norm / rope / act", r"norm|rope|silu|act_and_mul|act_"),
    ("casts / copies / elementwise", r"copy|Copy|cast|elementwise|fill|Fill|memcpy|memset|gather|scatter|index|"
                                     r"reduce_kernel|BinaryFunctor|CUDA mem"),
]
FAM_RX = [(f, re.compile(p)) for f, p in FAM]


def family(name):
    for f, rx in FAM_RX:
        if rx.search(name):
            return f
    return "other"


def load(path):
    """[(start_us, end_us, name)] of the GPU work in the trace."""
    if path.endswith(".csv"):
        out = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                s, d = float(r["Start (ns)"]) / 1e3, float(r["Duration (ns)"] or 0) / 1e3
                out.append((s, s + d, r["Name"]))
        return out
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    ev = d["traceEvents"] if isinstance(d, dict) else d
    return [(e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]


def report(label, ev, top):
    ev.sort()
    span = ev[-1][1] - ev[0][0]
    # sweep: busy union and exclusive time per family
    points = []
    for s, e, n in ev:
        points.append((s, 1, family(n)))
        points.append((e, -1, family(n)))
    points.sort(key=lambda p: (p[0], -p[1]))
    active = collections.Counter()
    busy = 0.0
    excl = collections.Counter()
    last = points[0][0]
    for t, delta, fam in points:
        if t > last and sum(active.values()) > 0:
            busy += t - last
            live = [f for f, c in active.items() if c > 0]
            if len(live) == 1:
                excl[live[0]] += t - last
            else:
                excl["(two or more at once)"] += t - last
        active[fam] += delta
        last = t
    fam_sum = collections.Counter()
    name_sum = collections.Counter()
    name_n = collections.Counter()
    for s, e, n in ev:
        fam_sum[family(n)] += e - s
        k = re.sub(r"\(.*", "", n.replace("(anonymous namespace)::", "anon::").replace("<unnamed>::", "anon::"))[:100]
        name_sum[k] += e - s
        name_n[k] += 1
    print(f"== {label}: span {span / 1e3:.1f} ms, busy {busy / 1e3:.1f} ms, {len(ev)} GPU ops")
    print(f"   {'family':32s} {'sum ms':>9s} {'excl ms':>9s}")
    for f in [x for x, _ in FAM] + ["other", "(two or more at once)"]:
        if fam_sum[f] or excl[f]:
            print(f"   {f:32s} {fam_sum[f] / 1e3:9.1f} {excl[f] / 1e3:9.1f}")
    print(f"   top {top} kernels:")
    for k, v in name_sum.most_common(top):
        print(f"   {v / 1e3:8.1f} ms x{name_n[k]:5d}  [{family(k)}] {k}")
    return fam_sum, excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", help="label=path")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    for spec in a.traces:
        label, path = spec.split("=", 1)
        report(label, load(path), a.top)


if __name__ == "__main__":
    main()
