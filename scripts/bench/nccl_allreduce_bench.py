# SPDX-License-Identifier: Apache-2.0
"""Two-node NCCL all-reduce latency / bandwidth at the sizes a TP=2 DeepSeek-V4.1 serve uses on 2x GB10 (bf16 rows of
5120): decode verify (6 rows, 61 KB), prefill chunks (512 / 2048 rows: 5.2 / 21 MB). Eager and inside a CUDA graph
(vLLM captures its decode all-reduces). The dummy-weight serve profile (Oct 8) showed NCCL picking RING_LL for every
size: 52 us at 61 KB and ~785 us at 5.2 MB (~6.7 GB/s on a 200 Gb/s link). Run the same command on both nodes with the
NCCL env under test (NCCL_PROTO, NCCL_ALGO, ...); rank 0 prints.

  RANK=0|1 MASTER_ADDR=<rank 0's ConnectX IP> MASTER_PORT=29511 WORLD_SIZE=2 python3 nccl_allreduce_bench.py [--rows 6,512,2048]
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def bench(fn, reps):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="6,512,2048")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    rank = dist.get_rank()
    env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_") and k not in ("NCCL_IB_HCA", "NCCL_SOCKET_IFNAME")}
    if rank == 0:
        print(f"NCCL {torch.cuda.nccl.version()} env {env}", flush=True)
    for rows in (int(r) for r in a.rows.split(",")):
        x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
        reps = a.reps if rows <= 64 else max(20, a.reps // 10)
        t_eager = bench(lambda: dist.all_reduce(x), reps)
        # 10 all-reduces captured in one graph, the way a decode graph holds them
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                dist.all_reduce(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(10):
                dist.all_reduce(x)
        t_graph = bench(g.replay, max(5, reps // 10)) / 10
        mb = x.numel() * 2 / 1e6
        if rank == 0:
            print(f"rows {rows:5d} ({mb:7.3f} MB): eager {t_eager:8.1f} us ({mb / t_eager * 1e3:6.2f} GB/s)  "
                  f"graph {t_graph:8.1f} us ({mb / t_graph * 1e3:6.2f} GB/s)", flush=True)
    torch.cuda.synchronize()
    sys.stdout.flush()
    os._exit(0)   # destroy_process_group() hangs after graph-captured NCCL all-reduces (Oct 9): skip teardown


if __name__ == "__main__":
    main()
