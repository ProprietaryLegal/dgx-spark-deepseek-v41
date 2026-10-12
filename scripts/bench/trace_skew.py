"""Rank-to-rank skew in a TP=2 decode profile (profile_decode.sh writes one torch-profiler trace per rank): which rank
waits at the all-reduces, and which compute segments make the other rank slow. The two ranks replay the same CUDA graph,
so their kernel sequences align by index (checked by name). For each all-reduce call the longer duration of the two
includes waiting for the peer: wait = dur - min(dur_rank0, dur_rank1). Each segment (the kernels between two
consecutive all-reduces, timed from one all-reduce's end to the next one's start on each rank's own clock) is compared
across ranks. Clocks are not synchronized between the boxes, so only durations are compared.

  python3 trace_skew.py <rank0 trace .json.gz> <rank1 trace .json.gz> <decode steps> [top]
"""
import gzip, json, re, sys
from collections import defaultdict

COMM = re.compile(r"nccl|stage_kernel|sum_kernel|allreduce|AllReduce|AllGather")


def kernels(path):
    d = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
    ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    ev.sort(key=lambda e: e["ts"])
    return [(e["name"], e["ts"], e["dur"]) for e in ev]


def short(n):
    n = re.sub(r"\(.*", "", n)
    n = re.sub(r"<.*", "", n)
    return n.split("::")[-1][:40]


def main():
    k0, k1, steps = kernels(sys.argv[1]), kernels(sys.argv[2]), int(sys.argv[3])
    top = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    n = min(len(k0), len(k1))
    same = sum(k0[i][0] == k1[i][0] for i in range(n))
    print(f"kernels: rank0 {len(k0)}, rank1 {len(k1)}; same name at {same / n:.1%} of {n} aligned positions")
    comm = [i for i in range(n) if COMM.search(k0[i][0]) and k0[i][0] == k1[i][0]]
    w0 = w1 = 0.0
    for i in comm:
        m = min(k0[i][2], k1[i][2])
        w0 += k0[i][2] - m
        w1 += k1[i][2] - m
    print(f"all-reduce / gather calls: {len(comm) / steps:.1f} per step; waiting for the peer per step: "
          f"rank0 {w0 / 1e3 / steps:.2f} ms, rank1 {w1 / 1e3 / steps:.2f} ms "
          f"(the rank that waits more is the faster one)")
    # segments between consecutive comm calls: compute time on each rank, keyed by the segment's kernel signature
    seg = defaultdict(lambda: [0, 0.0, 0.0])
    tot0 = tot1 = 0.0
    for a, b in zip(comm, comm[1:]):
        t0 = k0[b][1] - (k0[a][1] + k0[a][2])
        t1 = k1[b][1] - (k1[a][1] + k1[a][2])
        names = [short(k0[j][0]) for j in range(a + 1, b)]
        key = " > ".join(dict.fromkeys(names))[:110] or "(none)"
        s = seg[key]
        s[0] += 1; s[1] += t0; s[2] += t1
        tot0 += t0; tot1 += t1
    print(f"compute between all-reduces per step: rank0 {tot0 / 1e3 / steps:.2f} ms, rank1 {tot1 / 1e3 / steps:.2f} ms")
    print(f"{'segments by |rank0 - rank1| per step':<60} {'n/step':>7} {'r0 ms':>7} {'r1 ms':>7} {'r0-r1':>7}")
    for key, (c, a0, a1) in sorted(seg.items(), key=lambda kv: -abs(kv[1][1] - kv[1][2]))[:top]:
        print(f"  {key[:58]:<58} {c / steps:7.1f} {a0 / 1e3 / steps:7.2f} {a1 / 1e3 / steps:7.2f} {(a0 - a1) / 1e3 / steps:+7.2f}")
    # same kernel, different duration: the per-kernel-name sums outside comm
    by = defaultdict(lambda: [0, 0.0, 0.0])
    for i in range(n):
        if k0[i][0] == k1[i][0] and not COMM.search(k0[i][0]):
            b = by[short(k0[i][0])]
            b[0] += 1; b[1] += k0[i][2]; b[2] += k1[i][2]
    print(f"{'kernels by |rank0 - rank1| per step (own durations)':<60} {'n/step':>7} {'r0 ms':>7} {'r1 ms':>7} {'r0-r1':>7}")
    for key, (c, a0, a1) in sorted(by.items(), key=lambda kv: -abs(kv[1][1] - kv[1][2]))[:top]:
        print(f"  {key[:58]:<58} {c / steps:7.1f} {a0 / 1e3 / steps:7.2f} {a1 / 1e3 / steps:7.2f} {(a0 - a1) / 1e3 / steps:+7.2f}")


if __name__ == "__main__":
    main()
