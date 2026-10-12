"""Host-side gaps in a decode profile: stretches where no kernel, copy or memset runs on any stream (the union of busy
time, so concurrent streams do not make fake gaps), keyed by the kernels on either side, per rank. With full decode
graphs a gap is time the host spends between launches (scheduling, sampling, the drafter's host work, the broadcast
to the other node's worker), and a gap that ends in an all-reduce is waiting for the other rank to start.
Per gap kind it also splits the cause: the host call that issued the work ending the gap (the last kernel launch / copy
/ graph launch that began before it) and how long before the gap's end it began. Issued just before the end: the host
was late. Issued long before (the GPU sat idle with the work queued): a dependency held it, an event on another stream
or a copy, e.g. the next step's inputs waiting for the previous step's sampled and drafted tokens.

  python3 trace_gaps.py <decode steps> <rank trace .json.gz> [<rank trace> ...] [--min-us 50] [--top 10]
"""
import argparse, bisect, collections, gzip, json, re


def short(n):
    n = re.sub(r"\(.*", "", n)
    n = re.sub(r"<.*", "", n)
    return n.split("::")[-1][:36]


LAUNCH = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cudaGraphLaunch", "cudaMemcpyAsync", "cudaLaunchCooperativeKernel")


def events(path):
    d = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
    x = [e for e in d["traceEvents"] if e.get("ph") == "X"]
    ev = sorted((e for e in x if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    calls = sorted((e["ts"], e["name"]) for e in x if e.get("cat") == "cuda_runtime" and e["name"] in LAUNCH)
    return [(e["name"], e["ts"], e["dur"]) for e in ev], calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", type=int)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--min-us", type=float, default=50.0)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    for path in a.traces:
        k, calls = events(path)
        call_ts = [c[0] for c in calls]
        gaps = collections.defaultdict(lambda: [0, 0.0])
        lead = collections.defaultdict(list)          # per gap kind: how long before the gap's end its work was issued
        total, end = 0.0, 0.0
        for i in range(len(k) - 1):
            end = max(end, k[i][1] + k[i][2])
            g = k[i + 1][1] - end
            if g > a.min_us:
                key = f"{short(k[i][0])} -> {short(k[i + 1][0])}"
                gaps[key][0] += 1
                gaps[key][1] += g
                total += g
                j = bisect.bisect_right(call_ts, k[i + 1][1]) - 1
                if j >= 0:
                    lead[key].append(k[i + 1][1] - call_ts[j])
        span = (max(t + d for _, t, d in k) - k[0][1]) / 1e3 / a.steps
        print(f"== {path.split('/')[-1][:60]}: span {span:.2f} ms/step, gaps > {a.min_us:.0f} us "
              f"{total / 1e3 / a.steps:.2f} ms/step")
        for key, (c, s) in sorted(gaps.items(), key=lambda kv: -kv[1][1])[:a.top]:
            ld = sorted(lead.get(key, []))
            why = (f"issued {ld[len(ld) // 2]:7.0f} us before it started (median): "
                   + ("host late" if ld[len(ld) // 2] < 50 else "queued behind a dependency")) if ld else ""
            print(f"   {s / 1e3 / a.steps:6.2f} ms/step  {c / a.steps:5.1f}/step  mean {s / c:7.0f} us  {key}   {why}")


if __name__ == "__main__":
    main()
