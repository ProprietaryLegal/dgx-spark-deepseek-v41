# SPDX-License-Identifier: Apache-2.0
"""Two-node check of the RDMA all-reduce (vllm_exl3_moe/rdma.py, csrc/exl3_rdma.cu) against NCCL, one process a node.

Correctness: for decode shapes, the ring's sum equals the rank-order fp32 sum of both ranks' tensors rounded once to
bf16, and both ranks hold the same bits. Speed: 96 all-reduces of a [6, 5120] bf16 tensor (a k=5 verify step's count)
captured in one CUDA graph, ring vs NCCL, median over replays.

STRESS=N (P2 stage 0, the flag-write mode's ordering check): N all-reduces in graphs of 96 chained calls, every result
checked. Rank 0 contributes y * 0.5 and rank 1 y * 0.25 (exact in bf16), so each call's sum is y * 0.75 rounded once,
which every rank computes locally; a flag seen before its data, or a slot read stale, shows as a wrong value. Run it
with EXL3_RDMA_FLAGW=0 and =1 (the same on both nodes) to compare the latency too.
AG=1 checks the ring's all-gather (EXL3_RDMA_AG): bit-equal to NCCL's all_gather_into_tensor + vLLM's movedim /
reshape for fp32 (value, index) pairs, bf16 rows, int32 and 1-D tensors along the last dimension and dim 0, then (with
STRESS) graphs that interleave all-reduces and all-gathers on the shared slots, every result checked.
WIDE=1 adds Engram row-parallel's all-reduce, [6 | 12, 25600] bf16 (307 / 614 KB, patch 0028), to the correctness list,
the timing and the stress run; it needs EXL3_RDMA_AR_MAX_KB >= 600 on both nodes (the serve's default 256 sends it to
NCCL), and refuses to run below that.

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


WIDE_SHAPES = [(6, 25600), (12, 25600)] if os.environ.get("WIDE", "0") == "1" else []
AG = os.environ.get("AG", "0") == "1"
AG_CASES = [((1, 2), torch.float32, -1), ((5, 2), torch.float32, -1), ((6, 2), torch.float32, -1),
            ((6, 12800), torch.bfloat16, -1), ((3, 7), torch.int32, 0), ((40,), torch.float32, 0), ((2, 3, 4), torch.float32, -1)]


def nccl_gather(x, dim, world):
    """vLLM's base-class all_gather (concat style) on NCCL: the reference bits."""
    d = dim % x.dim()
    out = torch.empty((x.shape[0] * world,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.contiguous())
    out = out.reshape((world,) + tuple(x.shape)).movedim(0, d)
    return out.reshape(tuple(x.shape[:d]) + (world * x.shape[d],) + tuple(x.shape[d + 1:]))


def gather_check(ring, rank, world):
    ok = True
    for shape, dtype, dim in AG_CASES:
        g = torch.Generator(device="cuda").manual_seed(7 + 100 * rank + len(shape))
        x = (torch.randn(shape, device="cuda", generator=g) * 1000).to(dtype)
        if not rdma.RdmaAllReduce.fits_gather(x, dim):
            if rank == 0:
                print(f"all-gather {shape} {dtype} dim {dim}: does not fit the ring", flush=True)
            ok = False
            continue
        y = ring.all_gather(x, dim)
        ref = nccl_gather(x, dim, world)
        same = y.shape == ref.shape and torch.equal(y, ref)
        ys = [torch.empty_like(y) for _ in range(world)]
        dist.all_gather(ys, y)
        same &= all(torch.equal(ys[0], z) for z in ys)
        ok &= same
        if rank == 0:
            print(f"all-gather {tuple(shape)} {str(dtype).split('.')[-1]} dim {dim}: == NCCL {same}", flush=True)
    return ok


def stress_mixed(ring, rank, world, total, calls=48):
    """Graphs of `calls` (all-reduce of [6, 5120] bf16, all-gather of a [1, 2] fp32 pair) pairs: the two share the
    ring's slots and sequence; every result checked against the local expectation."""
    x = torch.empty((6, 5120), device="cuda", dtype=torch.bfloat16)
    pr = torch.empty((1, 2), device="cuda", dtype=torch.float32)
    outs, gouts = [torch.empty_like(x) for _ in range(calls)], [torch.empty((1, 2 * world), device="cuda") for _ in range(calls)]
    scale = 0.5 if rank == 0 else 0.25

    def body():
        y = x
        for o, go in zip(outs, gouts):
            y = ring.all_reduce(y * scale)
            o.copy_(y)
            go.copy_(ring.all_gather(pr + float(rank), -1))

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        x.normal_(); pr.normal_()
        body()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        body()
    bad, reps = 0, max(1, total // (2 * calls))
    gen = torch.Generator(device="cuda")
    for r in range(reps):
        gen.manual_seed(r)
        x.copy_((torch.randn(x.shape, device="cuda", generator=gen) * 4).to(torch.bfloat16))
        pr.copy_(torch.randn(pr.shape, device="cuda", generator=gen))
        dist.barrier()
        gr.replay()
        torch.cuda.synchronize()
        y = x
        want_g = torch.cat([pr + float(p) for p in range(world)], dim=-1)
        for o, go in zip(outs, gouts):
            y = ((y * 0.5).float() + (y * 0.25).float()).to(torch.bfloat16)
            if not torch.equal(o, y) or not torch.equal(go, want_g):
                bad += 1
                if bad <= 3:
                    print(f"rank {rank}: replay {r}: wrong mixed result", flush=True)
                y = o
    if rank == 0:
        print(f"stress mixed: {reps * calls} all-reduce + all-gather pairs checked, {bad} wrong", flush=True)
    return bad == 0


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
    for shape in [(6, 5120), (1, 5120)] + WIDE_SHAPES:
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

    too_big = [sh for sh in WIDE_SHAPES if sh[0] * sh[1] * 2 > rdma.MAX_BYTES]
    if too_big:
        if rank == 0:
            print(f"WIDE=1: {too_big} exceed EXL3_RDMA_AR_MAX_KB ({rdma.MAX_BYTES // 1024} KB): set it >= 600 on both nodes",
                  flush=True)
        sys.stdout.flush()
        os._exit(2)
    ok = True
    for shape in [(1, 5120), (4, 5120), (6, 5120), (16, 5120), (6, 4096), (2, 7168)] + WIDE_SHAPES:
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
    if WIDE_SHAPES:
        xw = torch.randn(*WIDE_SHAPES[0], device="cuda", dtype=torch.bfloat16)
        gw_ring, gw_nccl = graph_of(ring.all_reduce, xw, calls), graph_of(nccl_ar, xw, calls)
        for _ in range(3):
            tw_ring, tw_nccl = time_graph(gw_ring), time_graph(gw_nccl)
        if rank == 0:
            print(f"{calls} all-reduces of {list(WIDE_SHAPES[0])} bf16 in a graph: ring {tw_ring:.3f} ms "
                  f"({1000 * tw_ring / calls:.1f} us each), NCCL {tw_nccl:.3f} ms ({1000 * tw_nccl / calls:.1f} us each)",
                  flush=True)
    if AG:
        ok &= gather_check(ring, rank, world)
    stress = int(os.environ.get("STRESS", "0"))
    if stress:
        ok &= stress_check(ring, rank, stress)
        if AG:
            ok &= stress_mixed(ring, rank, world, stress)
    if rank == 0:
        print("PASS" if ok else "FAIL", flush=True)
    dist.barrier()
    # no destroy_process_group: with the ring's pinned proxy thread alive, ProcessGroupNCCL's teardown hung past the
    # run's timeout after a passing stress run (seen Oct 11); the OS frees the queue pairs and registrations
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
