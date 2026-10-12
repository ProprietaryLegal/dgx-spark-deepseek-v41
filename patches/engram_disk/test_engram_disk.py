"""Standalone test of the engram_disk extension (no vLLM needed): bit-exact rows via a direct read, via a CUDA stream
host node, and via CUDA graph capture + replays, plus a latency print for decode-sized reads.

  python3 test_engram_disk.py [--dir /tmp/engram_disk_test] [--rows 4000000]
"""
import argparse, os, time
import numpy as np
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ROW, SROW = 256, 8


def build():
    return load(name="engram_disk", sources=[os.path.join(HERE, "engram_disk.cpp")],
                extra_cflags=["-O2"], with_cuda=True, verbose=False)


def make_shard(path, rows, header_gap=4096 + 123):
    """A fake safetensors-like file: gap, weight table [rows, 256], gap, scale table [rows, 8]."""
    off_w = header_gap
    off_s = off_w + rows * ROW + 777
    if not (os.path.exists(path) and os.path.getsize(path) >= off_s + rows * SROW):
        rng = np.random.default_rng(7)
        with open(path, "wb") as f:
            f.write(rng.integers(0, 256, header_gap, dtype=np.uint8).tobytes())
            chunk = 1 << 18
            for start in range(0, rows, chunk):
                n = min(chunk, rows - start)
                f.write(rng.integers(0, 256, n * ROW, dtype=np.uint8).tobytes())
            f.write(b"\x00" * 777)
            for start in range(0, rows, chunk):
                n = min(chunk, rows - start)
                f.write(rng.integers(0, 256, n * SROW, dtype=np.uint8).tobytes())
    return off_w, off_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/engram_disk_test")
    ap.add_argument("--rows", type=int, default=4_000_000)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    ext = build()
    path = os.path.join(a.dir, "shard.bin")
    off_w, off_s = make_shard(path, a.rows)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    ref_w = mm[off_w: off_w + a.rows * ROW].reshape(a.rows, ROW)
    ref_s = mm[off_s: off_s + a.rows * SROW].reshape(a.rows, SROW)
    row_start = a.rows // 3            # this "rank" owns rows [row_start, row_start + num_rows)
    num_rows = a.rows // 2
    t = ext.EngramDiskTable(path, off_w, path, off_s, row_start, num_rows, ROW, SROW, cache_rows=4096, threads=16)

    def check(ids_np, w, s, tag):
        for i, local in enumerate(ids_np):
            if local < 0 or local >= num_rows:
                assert not w[i].any() and not s[i].any(), f"{tag}: row {i} should be zero"
            else:
                g = row_start + local
                assert np.array_equal(w[i], ref_w[g]) and np.array_equal(s[i], ref_s[g]), f"{tag}: row {i} mismatch"

    n = 1024
    ids = torch.empty(n, dtype=torch.int64, pin_memory=True)
    st_w = torch.empty(n, ROW, dtype=torch.uint8, pin_memory=True)
    st_s = torch.empty(n, SROW, dtype=torch.uint8, pin_memory=True)
    rng = np.random.default_rng(1)

    # 1. direct read (includes out-of-range and -1 rows)
    x = rng.integers(0, num_rows, n); x[::97] = -1; x[5] = num_rows + 10
    ids.copy_(torch.from_numpy(x))
    t.read_rows(ids.data_ptr(), st_w.data_ptr(), st_s.data_ptr(), n)
    check(x, st_w.numpy(), st_s.numpy(), "direct")
    print("direct read: bit-exact OK")

    # 2. stream: D2H ids copy -> host node -> H2D staging copy, all ordered on one stream
    s = torch.cuda.Stream()
    d_ids = torch.from_numpy(rng.integers(0, num_rows, n)).cuda()
    d_w = torch.empty(n, ROW, dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(s):
        ids.copy_(d_ids, non_blocking=True)
        t.enqueue_read(s.cuda_stream, ids.data_ptr(), st_w.data_ptr(), st_s.data_ptr(), n)
        d_w.copy_(st_w, non_blocking=True)
    s.synchronize()
    check(d_ids.cpu().numpy(), d_w.cpu().numpy(), st_s.numpy(), "stream")
    print("stream host node: bit-exact OK")

    # 3. CUDA graph: capture once, replay with new device ids
    g = torch.cuda.CUDAGraph()
    d_in = torch.zeros(n, dtype=torch.int64, device="cuda")
    with torch.cuda.graph(g, stream=s):
        ids.copy_(d_in, non_blocking=True)
        t.enqueue_read(torch.cuda.current_stream().cuda_stream, ids.data_ptr(), st_w.data_ptr(), st_s.data_ptr(), n)
        d_w.copy_(st_w, non_blocking=True)
    for rep in range(3):
        x = rng.integers(0, num_rows, n)
        d_in.copy_(torch.from_numpy(x))
        g.replay(); torch.cuda.synchronize()
        check(x, d_w.cpu().numpy(), st_s.numpy(), f"graph replay {rep}")
    print("CUDA graph capture + 3 replays: bit-exact OK")

    # 4. latency: decode-sized reads (12 local rows per layer per rank at TP=2, batch 1), larger batches
    for m in (12, 48, 384, 1024):
        times = []
        for _ in range(200):
            x = rng.integers(0, num_rows, m)
            ids[:m].copy_(torch.from_numpy(x))
            t0 = time.perf_counter()
            t.read_rows(ids.data_ptr(), st_w.data_ptr(), st_s.data_ptr(), m)
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        print(f"read {m:5d} rows: p50 {times[100]:.3f} ms, p90 {times[180]:.3f} ms, max {times[-1]:.3f} ms")
    hits, misses = t.stats()
    print(f"cache hits {hits}, misses {misses}")


if __name__ == "__main__":
    main()
