# SPDX-License-Identifier: Apache-2.0
"""Two-node check of the RDMA all-reduce (vllm_exl3_moe/rdma.py, csrc/exl3_rdma.cu) against NCCL, one process a node.

Correctness: for decode shapes, the ring's sum equals the rank-order fp32 sum of both ranks' tensors rounded once to
bf16, and both ranks hold the same bits. Speed: 96 all-reduces of a [6, 5120] bf16 tensor (a k=5 verify step's count)
captured in one CUDA graph, ring vs NCCL, median over replays.

STRESS=N (P2 stage 0, the flag-write mode's ordering check): N all-reduces in graphs of 96 chained calls, every result
checked. Rank 0 contributes y * 0.5 and rank 1 y * 0.25 (exact in bf16), so each call's sum is y * 0.75 rounded once,
which every rank computes locally; a flag seen before its data, or a slot read stale, shows as a wrong value. Run it
with EXL3_RDMA_FLAGW=0 and =1 (the same on both nodes) to compare the latency too.

On each node, in the serve image (rank 0 on the head, MASTER_ADDR = the head's fabric address):
  docker run --rm --gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 \
    -e RANK=R -e MASTER_ADDR=<head IP> -e MASTER_PORT=29610 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=<gid> \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
    -v ~/dsv41_recipe/patches:/opt/p:ro -v ~/.cache/dsv41-serve:/root/.cache --entrypoint python3 dsv41-serve:v3 \
    /opt/p/exl3_moe/tests/rdma_ar_2node.py
"""
import os
import sys
import statistics

import torch
import torch.distributed as dist

from vllm_exl3_moe import rdma


def graph_of(fn, x, calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y = x
            for _ in range(calls):
                y = fn(y) * 0.5                    # keep values bounded; each call depends on the last
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = x
        for _ in range(calls):
            y = fn(y) * 0.5
    return g


def time_graph(g, reps=20):
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        dist.barrier()
        torch.cuda.synchronize()
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def stress_check(ring, rank, total, calls=96):
    """total all-reduces in graphs of `calls` chained ones; every intermediate result compared with the local expectation."""
    bad = 0
    for shape in [(6, 5120), (1, 5120)]:
        x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        outs = [torch.empty_like(x) for _ in range(calls)]
        scale = 0.5 if rank == 0 else 0.25

        def body():
            y = x
            for o in outs:
                y = ring.all_reduce(y * scale)
                o.copy_(y)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            x.normal_()
            body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        reps = max(1, total // (2 * calls))
        gen = torch.Generator(device="cuda")
        for r in range(reps):
            gen.manual_seed(r)                         # the same input on both ranks: each knows the other's part
            x.copy_((torch.randn(shape, device="cuda", generator=gen) * 4).to(torch.bfloat16))
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
            y = x
            for o in outs:
                y = ((y * 0.5).float() + (y * 0.25).float()).to(torch.bfloat16)   # rank order, rounded once
                if not torch.equal(o, y):
                    bad += 1
                    if bad <= 3:
                        print(f"rank {rank}: replay {r} shape {shape}: wrong sum", flush=True)
                    y = o                              # continue the chain from what the ring produced
        if rank == 0:
            print(f"stress {shape}: {reps * calls} all-reduces checked, {bad} wrong so far (flag-write {rdma.FLAGW})",
                  flush=True)
    return bad == 0


def main():
    rank, world = int(os.environ["RANK"]), 2
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda:0"))
    gloo = dist.new_group(backend="gloo")
    ring = rdma.RdmaAllReduce(gloo, rank, world)

    def nccl_ar(x):
        y = x.clone()
        dist.all_reduce(y)
        return y

    ok = True
    for shape in [(1, 5120), (4, 5120), (6, 5120), (16, 5120), (6, 4096), (2, 7168)]:
        g = torch.Generator(device="cuda").manual_seed(1000 * rank + shape[0])
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g)
        for _ in range(3):                         # several sequences through the slots
            y = ring.all_reduce(x)
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x)
        ref = (parts[0].float() + parts[1].float()).to(torch.bfloat16)
        ys = [torch.empty_like(y) for _ in range(world)]
        dist.all_gather(ys, y)
        good = torch.equal(y, ref) and torch.equal(ys[0], ys[1])
        nccl_same = torch.equal(nccl_ar(x), y)
        ok &= good
        if rank == 0:
            print(f"{shape}: ring == rank-order fp32 sum {good}, == NCCL {nccl_same}", flush=True)
    x = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
    calls = 96
    g_ring = graph_of(ring.all_reduce, x, calls)
    g_nccl = graph_of(nccl_ar, x, calls)
    for _ in range(3):
        t_ring, t_nccl = time_graph(g_ring), time_graph(g_nccl)
    if rank == 0:
        print(f"{calls} all-reduces of [6, 5120] bf16 in a graph: ring {t_ring:.3f} ms ({1000 * t_ring / calls:.1f} us "
              f"each), NCCL {t_nccl:.3f} ms ({1000 * t_nccl / calls:.1f} us each), flag-write mode {rdma.FLAGW}",
              flush=True)
    stress = int(os.environ.get("STRESS", "0"))
    if stress:
        ok &= stress_check(ring, rank, stress)
    if rank == 0:
        print("PASS" if ok else "FAIL", flush=True)
    dist.barrier()
    # no destroy_process_group: with the ring's pinned proxy thread alive, ProcessGroupNCCL's teardown hung past the
    # run's timeout after a passing stress run (seen Oct 11); the OS frees the queue pairs and registrations
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
