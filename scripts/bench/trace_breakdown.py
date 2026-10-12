# SPDX-License-Identifier: Apache-2.0
"""GPU time per decode step by kernel family, from a vLLM torch-profiler trace (chrome JSON, .json or .json.gz).

  python3 trace_breakdown.py rank0.pt.trace.json.gz [--steps N] [--top 25]

Sums every GPU kernel / memcpy / memset event, groups kernels by family (first matching pattern below), and divides by
the number of decode steps (--steps, else the count of the model's per-step marker kernel: the lm_head GEMM runs once per
step). Wall time per step is not the GPU sum: kernels on side streams overlap (the L2 prefetch runs beside the work it
feeds) and host work sits between kernels; trace_rounds.py gives the busy union and each family's exclusive share.
"""
import argparse
import collections
import gzip
import json
import re

FAMILIES = [  # (family, regex on the kernel name), first match wins
    ("routed experts (smoe, ours)", r"smoe_"),
    ("dense EXL3 linears (mrow, ours)", r"mrow_kernel"),
    ("L2 prefetch (side stream, ours)", r"l2pf|prefetch"),
    ("RDMA all-reduce ring (ours)", r"\bsum_kernel\(|\bstage_kernel\(|rdma"),
    ("router + top-k", r"router_kernel|_dsv4_topk|rowmm"),
    ("mHC", r"mhc|sinkhorn|hc_prenorm|tilelang|tl_"),
    ("drafter Markov bias", r"markov"),
    ("nccl all-reduce / all-gather", r"nccl|allreduce|all_reduce|AllReduce|AllGather|cross_device"),
    ("exllamav3 kernels (head GEMM, prefill MoE)", r"exl3|exllama|trellis|moe_coop|reconstruct"),
    ("Marlin MXFP8 linears", r"[Mm]arlin"),
    ("CUTLASS / cuBLAS GEMM", r"cutlass|gemm|Gemm|cublas|sm90_|sm100_|sm120_|nvjet|ampere_|xmma"),
    ("sparse MLA attention (FlashInfer)", r"flashinfer|mla|MLA|attention|Attention|fmha|paged"),
    ("indexer / top-k", r"topk|top_k|TopK|indexer|deepselect|radix|sort"),
    ("engram", r"engram"),
    ("norm / rope / act quant", r"norm|rope|rotary|quant|silu|act_|swiglu"),
    ("routing / gate", r"gate|router|routing|softplus|sqrtsoftplus"),
    ("elementwise / copy", r"elementwise|vectorized|copy|Copy|fill|cat|index|scatter|gather|reduce|Reduce"),
]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    ev = load(a.trace)
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    fam_t, name_t, name_n = collections.Counter(), collections.Counter(), collections.Counter()
    pats = [(f, re.compile(p)) for f, p in FAMILIES]
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        name, dur = e.get("name", ""), float(e.get("dur", 0.0))
        fam = "memcpy / memset" if e["cat"] != "kernel" else next((f for f, p in pats if p.search(name)), "other")
        fam_t[fam] += dur
        name_t[name[:110]] += dur
        name_n[name[:110]] += 1
    steps = a.steps
    if not steps:
        head = [n for n in name_n if re.search(r"[Mm]arlin|gemm|Gemm", n)]
        steps = max(1, min((name_n[n] for n in head), default=1))
        print(f"(steps not given: using {steps}, the smallest call count among GEMM kernels; pass --steps to fix)")
    total = sum(fam_t.values())
    print(f"GPU time {total / 1e3:.1f} ms over {steps} steps = {total / steps / 1e3:.2f} ms per step")
    for f, t in fam_t.most_common():
        print(f"  {f:38s} {t / steps / 1e3:8.3f} ms/step  {100 * t / total:5.1f}%")
    print(f"top {a.top} kernels:")
    for n, t in name_t.most_common(a.top):
        print(f"  {t / steps:9.1f} us/step  x{name_n[n] / steps:6.1f}/step  {n}")


if __name__ == "__main__":
    main()
