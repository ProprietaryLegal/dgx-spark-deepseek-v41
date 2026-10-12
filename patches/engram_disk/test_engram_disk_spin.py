"""Standalone test of engram_disk's spin mode (enable_spin; no vLLM needed): bit-exact rows through the proxy in eager
stream order, in CUDA graph replays, two tables in one graph, the synchronous serve_rows path, a table destroyed while
the proxy runs; then the GPU-side latency of a decode-sized lookup, spin mode vs today's host node, rows cold.

  python3 test_engram_disk_spin.py [--dir DIR_ON_DISK] [--rows 4000000] [--threads 256]
"""
import argparse, gc, os, time
import numpy as np
import torch

from test_engram_disk import ROW, SROW, build, make_shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/engram_disk_test")
    ap.add_argument("--rows", type=int, default=4_000_000)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    ext = build()
    path = os.path.join(a.dir, "shard.bin")
    off_w, off_s = make_shard(path, a.rows)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    ref_w = mm[off_w: off_w + a.rows * ROW].reshape(a.rows, ROW)
    ref_s = mm[off_s: off_s + a.rows * SROW].reshape(a.rows, SROW)
    row_start, num_rows = a.rows // 3, a.rows // 2
    rng = np.random.default_rng(3)

    def table(cache_rows, spin):
        t = ext.EngramDiskTable(path, off_w, path, off_s, row_start, num_rows, ROW, SROW, cache_rows=cache_rows,
                                threads=a.threads)
        if spin:
            print("   mode:", t.enable_spin(1000, 256))
        return t

    def check(ids_np, w, s, tag):
        for i, local in enumerate(ids_np):
            if local < 0 or local >= num_rows:
                assert not w[i].any() and not s[i].any(), f"{tag}: row {i} should be zero"
            else:
                g = row_start + local
                assert np.array_equal(w[i], ref_w[g]) and np.array_equal(s[i], ref_s[g]), f"{tag}: row {i} mismatch"

    keep = []

    def buffers(n):
        # kept alive for the whole run, as vLLM keeps its staging: torch 2.11's pinned-memory cache fails the next
        # capture (cudaErrorInvalidValue in _host_emptyCache) after a block copied under capture is released, host
        # node or spin mode alike
        b = (torch.empty(n, dtype=torch.int64, pin_memory=True), torch.empty(n, ROW, dtype=torch.uint8, pin_memory=True),
             torch.empty(n, SROW, dtype=torch.uint8, pin_memory=True))
        keep.append(b)
        return b

    n = 1024
    s = torch.cuda.Stream()
    for cache in (0, 4096):
        t = table(cache, True)
        ids, st_w, st_s = buffers(n)
        d_w = torch.empty(n, ROW, dtype=torch.uint8, device="cuda")
        d_s = torch.empty(n, SROW, dtype=torch.uint8, device="cuda")
        # 1. eager stream order: device ids -> D2H -> spin request -> H2D of the staging rows; twice, so cache hits too
        x = rng.integers(0, num_rows, n); x[::97] = -1; x[5] = num_rows + 10
        d_ids = torch.from_numpy(x).cuda()
        for rep in range(2):
            with torch.cuda.stream(s):
                ids.copy_(d_ids, non_blocking=True)
                t.enqueue_read(s.cuda_stream, ids.data_ptr(), st_w.data_ptr(), st_s.data_ptr(), n)
                d_w.copy_(st_w, non_blocking=True)
                d_s.copy_(st_s, non_blocking=True)
            s.synchronize()
            check(x, d_w.cpu().numpy(), d_s.cpu().numpy(), f"cache {cache} eager {rep}")
        # 2. graph: capture once, replay with new device ids, fewer rows than the buffers hold
        g = torch.cuda.CUDAGraph()
        m = 300
        d_in = torch.zeros(m, dtype=torch.int64, device="cuda")
        with torch.cuda.graph(g, stream=s):
            ids[:m].copy_(d_in, non_blocking=True)
            t.enqueue_read(torch.cuda.current_stream().cuda_stream, ids.data_ptr(), st_w.data_ptr(), st_s.data_ptr(), m)
            d_w[:m].copy_(st_w[:m], non_blocking=True)
            d_s[:m].copy_(st_s[:m], non_blocking=True)
        for rep in range(5):
            x = rng.integers(-1, num_rows + 3, m)
            d_in.copy_(torch.from_numpy(x))
            g.replay()
            torch.cuda.synchronize()
            check(x, d_w[:m].cpu().numpy(), d_s[:m].cpu().numpy(), f"cache {cache} graph {rep}")
        # 3. serve_rows: the proxy's read path in this thread
        x = rng.integers(0, num_rows, n)
        ids.copy_(torch.from_numpy(x))
        t.serve_rows(ids.data_ptr(), st_w.data_ptr(), st_s.data_ptr(), n)
        check(x, st_w.numpy(), st_s.numpy(), f"cache {cache} serve_rows")
        print(f"cache {cache}: eager x2, graph x5, serve_rows bit-exact OK; stats {t.stats()}, spin {t.spin_stats()}")
        del g, t
        gc.collect()

    # 4. two tables, one graph: both requests on one stream, each table its own buffers
    t1, t2 = table(0, True), table(512, True)
    b1, b2 = buffers(64), buffers(64)
    o1 = torch.empty(64, ROW, dtype=torch.uint8, device="cuda")
    o2 = torch.empty(64, ROW, dtype=torch.uint8, device="cuda")
    d1 = torch.zeros(64, dtype=torch.int64, device="cuda")
    d2 = torch.zeros(64, dtype=torch.int64, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        cs = torch.cuda.current_stream()
        for t, (ids, w, sc), d, o in ((t1, b1, d1, o1), (t2, b2, d2, o2)):
            ids.copy_(d, non_blocking=True)
            t.enqueue_read(cs.cuda_stream, ids.data_ptr(), w.data_ptr(), sc.data_ptr(), 64)
            o.copy_(w, non_blocking=True)
    for rep in range(20):
        x1, x2 = rng.integers(0, num_rows, 64), rng.integers(0, num_rows, 64)
        d1.copy_(torch.from_numpy(x1)); d2.copy_(torch.from_numpy(x2))
        g.replay()
        torch.cuda.synchronize()
        check(x1, o1.cpu().numpy(), b1[2].numpy(), f"two tables t1 {rep}")
        check(x2, o2.cpu().numpy(), b2[2].numpy(), f"two tables t2 {rep}")
    print("two tables in one graph x20: bit-exact OK")
    # 5. destroy one table while the proxy serves the other
    del t2, t
    gc.collect()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, stream=s):
        b1[0].copy_(d1, non_blocking=True)
        t1.enqueue_read(torch.cuda.current_stream().cuda_stream, b1[0].data_ptr(), b1[1].data_ptr(), b1[2].data_ptr(), 64)
        o1.copy_(b1[1], non_blocking=True)
    x1 = rng.integers(0, num_rows, 64)
    d1.copy_(torch.from_numpy(x1))
    g2.replay()
    torch.cuda.synchronize()
    check(x1, o1.cpu().numpy(), b1[2].numpy(), "after removing a table")
    print("table removed while the proxy runs: OK")
    del g, g2, t1
    gc.collect()

    # 6. latency, rows cold: one graph per mode = D2H ids, the read, H2D rows; fresh random ids every replay, and the
    # file's cached pages dropped before each replay (O_DIRECT skips the page cache anyway)
    fd = os.open(path, os.O_RDONLY)
    for m in (12, 48, 96, 384):
        res = {}
        for name in ("host node", "spin"):
            t = table(0, name == "spin")
            ids, w, sc = buffers(m)
            d_in = torch.zeros(m, dtype=torch.int64, device="cuda")
            o = torch.empty(m, ROW, dtype=torch.uint8, device="cuda")
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                ids.copy_(d_in, non_blocking=True)
                t.enqueue_read(torch.cuda.current_stream().cuda_stream, ids.data_ptr(), w.data_ptr(), sc.data_ptr(), m)
                o.copy_(w, non_blocking=True)
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            us = []
            for rep in range(a.reps):
                d_in.copy_(torch.from_numpy(rng.integers(0, num_rows, m)))
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                torch.cuda.synchronize()
                time.sleep(0.02)                              # idle between steps, as in decode
                with torch.cuda.stream(s):
                    ev0.record(s); g.replay(); ev1.record(s)
                s.synchronize()
                us.append(ev0.elapsed_time(ev1) * 1e3)
            us.sort()
            res[name] = us
            extra = ""
            if name == "spin":
                sv, ns, mx, err = t.spin_stats()
                extra = f"  (proxy serve mean {ns / max(sv, 1) / 1e3:.0f} us, max {mx / 1e3:.0f} us, errors {err})"
            print(f"rows {m:4d} {name:9s}: p50 {us[len(us) // 2]:7.0f} us  p90 {us[int(len(us) * .9)]:7.0f} us  "
                  f"max {us[-1]:7.0f} us{extra}", flush=True)
            del g, t
            gc.collect()
        print(f"rows {m:4d}: spin p50 {res['host node'][len(us) // 2] / res['spin'][len(us) // 2]:.2f}x faster than the host node")
    os.close(fd)


if __name__ == "__main__":
    main()
