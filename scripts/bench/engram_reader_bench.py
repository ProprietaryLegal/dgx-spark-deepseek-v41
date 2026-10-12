# SPDX-License-Identifier: Apache-2.0
"""Raw speed of patch 0002's Engram disk reader (engram_disk.cpp) on one Spark: random rows of layer 1's table read
from the official shard into pinned staging, as a prefill lookup does (tokens x local hash heads rows), at several
thread counts. Rows are drawn fresh per case from the ~100 GB table, so reads are cold (page cache mostly misses).
No GPU work. The dummy-weight TP=2 profile showed a 512-token prefill step waiting ~74 ms on these reads (32 threads).

  python3 engram_reader_bench.py /model [--layer 1] [--rows 4096,16384] [--threads 32,64,96,128] [--cache-rows 0]
"""
import argparse
import json
import os
import struct
import time

import torch


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--rows", default="4096,16384")
    ap.add_argument("--threads", default="32,64,96,128")
    ap.add_argument("--cache-rows", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="0 = from the clock, so a rerun reads cold rows")
    a = ap.parse_args()
    wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    names = {s: f"layers.{a.layer}.engram.embed.{s}" for s in ("weight", "scale")}
    files = {s: os.path.join(a.model, wm[n]) for s, n in names.items()}
    meta = {}
    for s in names:
        base, h = header(files[s])
        meta[s] = (base + h[names[s]]["data_offsets"][0], h[names[s]]["shape"], h[names[s]]["dtype"])
    rows, dim = meta["weight"][1]
    sdim = meta["scale"][1][1]
    print(f"layer {a.layer} Engram table: {rows:,} rows x {dim} ({meta['weight'][2]}) + scales x {sdim}: "
          f"{rows * (dim + sdim) / 2**30:.1f} GiB", flush=True)
    from torch.utils.cpp_extension import load
    import vllm.models.deepseek_v41.common.engram as eng
    ext = load(name="vllm_engram_disk", sources=[os.path.join(os.path.dirname(eng.__file__), "engram_disk.cpp")],
               extra_cflags=["-O2"], with_cuda=True, verbose=False)
    g = torch.Generator().manual_seed(a.seed if a.seed else int(time.time() * 1e6) % (2 ** 31))
    for nthreads in (int(t) for t in a.threads.split(",")):
        table = ext.EngramDiskTable(files["weight"], meta["weight"][0], files["scale"], meta["scale"][0],
                                    0, rows, dim, sdim, a.cache_rows, nthreads)
        for n in (int(r) for r in a.rows.split(",")):
            ids = torch.empty(n, dtype=torch.int64, pin_memory=True)
            w = torch.empty(n, dim, dtype=torch.uint8, pin_memory=True)
            s = torch.empty(n, sdim, dtype=torch.uint8, pin_memory=True)
            ts = []
            for _ in range(a.reps):
                ids.copy_(torch.randint(0, rows, (n,), generator=g))
                t0 = time.perf_counter()
                table.read_rows(ids.data_ptr(), w.data_ptr(), s.data_ptr(), n)
                ts.append(time.perf_counter() - t0)
            best = min(ts)
            print(f"threads {nthreads:3d}  rows {n:6,d}: {best * 1e3:7.1f} ms best of {a.reps} "
                  f"({n / best / 1e3:6.1f} K rows/s, {2 * n / best / 1e3:6.1f} K reads/s)", flush=True)
        del table


if __name__ == "__main__":
    main()
