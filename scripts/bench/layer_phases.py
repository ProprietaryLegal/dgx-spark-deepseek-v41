# SPDX-License-Identifier: Apache-2.0
"""Where a decode layer's time goes on one rank, from our torch-profiler trace: each backbone layer of the decode
window is cut at its routed-expert down kernel (smoe_down), and the time from the previous cut is split into phases by
the first kernel of each phase on the main stream:
    glue_in   previous down end -> attention's first dense launch (all-reduce, mHC post + pre, norm)
    attn      first dense launch -> the last attention dense (wo_b) end (wq_a+wkv, q/kv norm + rope, wq_b, indexer,
              sparse attention, wo_a, wo_b)
    glue_mid  wo_b end -> router GEMM start (all-reduce, mHC post + pre)
    route     router start -> smoe_gu start (router, top-k, rotation; the shared expert overlaps on its own stream)
    experts   smoe_gu start -> smoe_down end
Medians and means over the window's layers, per layer and times 40 for a round's backbone; plus the kernels per
layer and the idle (no kernel running) inside each phase. Layers whose sequence does not parse are skipped and counted.
    python3 layer_phases.py rank0.pt.trace.json.gz [--after exl3_moe_kernel]
"""
import argparse
import re
import statistics

from trace_rounds import load

ENGINES = {   # per engine: attention dense, router start, first expert kernel, the layer's last expert kernel
    "ours": dict(dense=r"mrow_kernel|exl3_gemm|marlin", router=r"cutlass.*16x16_128x2|exl3_router|router_topk",
                 gu=r"smoe_gu|smoe_fused", down=r"smoe_down|smoe_fused", after="exl3_moe_kernel"),
    # TensorFold (tp2 branch): grouped EXL3 linears, router GEMV parts, grouped_cp gate/up then down (its commit history)
    "tf": dict(dense=r"glinear|linear_grouped|exl3_gemm|gemv", router=r"rowmm|route_parts|router",
               gu=r"grouped_cp|grouped_kernel|grouped_rows", down=r"grouped_cp|grouped_kernel|grouped_rows", after=None),
}


def idle(rows, a, b):
    """time in [a, b] with no kernel running"""
    busy, end = 0.0, a
    for s, d, _, _ in rows:
        s2, e2 = max(s, a), min(s + d, b)
        if e2 <= end:
            continue
        busy += e2 - max(s2, end)
        end = e2
    return (b - a) - busy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--engine", choices=sorted(ENGINES), default="ours")
    ap.add_argument("--after", help="start after the last kernel matching this (default: the engine's prefill marker)")
    ap.add_argument("--skip-ms", type=float, default=0.0, help="drop this much from the capture's start (prefill)")
    a = ap.parse_args()
    E = ENGINES[a.engine]
    DENSE, ROUTER, GU, DOWN = (re.compile(E[k]) for k in ("dense", "router", "gu", "down"))
    rows = load(a.trace)
    after = a.after or E["after"]
    if after:
        last = max((i for i, r in enumerate(rows) if re.search(after, r[2])), default=-1)
        rows = rows[last + 1:]
    if a.skip_ms:
        t_first = rows[0][0]
        rows = [r for r in rows if r[0] >= t_first + a.skip_ms * 1e6]
    downs = [i for i, r in enumerate(rows) if DOWN.search(r[2])]
    if a.engine == "tf":     # gate/up and down are both grouped_cp: a layer ends at every second one
        downs = downs[1::2]
    ph = {k: [] for k in ("glue_in", "attn", "glue_mid", "route", "experts", "layer")}
    idl = {k: [] for k in ph}
    nk, skipped = [], 0
    for a_i, b_i in zip(downs, downs[1:]):
        t0 = rows[a_i][0] + rows[a_i][1]
        seg = [r for r in rows[a_i + 1: b_i + 1] if r[0] >= t0]     # kernels that started under the down belong to it
        if not seg or seg[-1][0] + seg[-1][1] - t0 > 5e6:       # > 5 ms: a step boundary (head, drafter), not a layer
            skipped += 1
            continue
        try:
            i_dense = next(i for i, r in enumerate(seg) if DENSE.search(r[2]))
            i_router = next(i for i, r in enumerate(seg) if ROUTER.search(r[2]))
            i_gu = next(i for i, r in enumerate(seg) if GU.search(r[2]))
        except StopIteration:
            skipped += 1
            continue
        t_dense = seg[i_dense][0]
        t_attn_end = max(r[0] + r[1] for r in seg[i_dense:i_router] if DENSE.search(r[2]))
        t_router, t_gu = seg[i_router][0], seg[i_gu][0]
        t_end = seg[-1][0] + seg[-1][1]
        if not (t0 <= t_dense <= t_attn_end <= t_router <= t_gu <= t_end):
            skipped += 1
            continue
        cuts = {"glue_in": (t0, t_dense), "attn": (t_dense, t_attn_end), "glue_mid": (t_attn_end, t_router),
                "route": (t_router, t_gu), "experts": (t_gu, t_end), "layer": (t0, t_end)}
        for k, (x, y) in cuts.items():
            ph[k].append((y - x) / 1e3)
            idl[k].append(idle(seg, x, y) / 1e3)
        nk.append(len(seg))
    n = len(ph["layer"])
    print(f"{n} layers parsed ({skipped} segments skipped: step boundaries or unparsed)")
    print(f"  {'phase':9s} {'median us':>10s} {'mean us':>9s} {'idle us':>8s} {'x40 ms':>7s}")
    for k in ph:
        if not ph[k]:
            continue
        print(f"  {k:9s} {statistics.median(ph[k]):10.1f} {statistics.mean(ph[k]):9.1f} {statistics.mean(idl[k]):8.1f} "
              f"{statistics.mean(ph[k]) * 40 / 1e3:7.2f}")
    if nk:
        print(f"  kernels a layer: median {statistics.median(nk):.0f}")


if __name__ == "__main__":
    main()
