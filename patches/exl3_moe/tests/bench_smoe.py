# SPDX-License-Identifier: Apache-2.0
"""exl3_smoe vs the exllamav3 coop path (parallel K groups, EXL3_MOE_COOP_WIDE as set) on one TP=2 rank-0 layer, decode
routing with DSpark-like sharing (m tokens draw their picks from a pool of `distinct` experts, every pool expert used),
8 routings per CUDA graph so the picked experts change call to call. EXL3_SMOE_FLAGS picks the smoe build.
    python3 bench_smoe.py "2:35,3:185,4:163,5:1" 1:6,4:16,6:22,8:28,12:40 [coop]
"""
import os
import sys
import time

import torch

from test_kernels_gpu import DEV, H, LIMIT, TOPK, make_bank, random_experts
from vllm_exl3_moe import kernels as K
from vllm_exl3_moe.forward import routed_moe_forward
from vllm_exl3_moe.smoe import SmoeLayer


def shared_routing(m, n_exp, distinct, seed):
    g = torch.Generator().manual_seed(seed)
    pool = torch.randperm(n_exp, generator=g)[:distinct].tolist()
    rows, used = [], set()
    for t in range(m):
        unused = [e for e in pool if e not in used]
        pick = unused[:TOPK]
        if len(pick) < TOPK:
            rest = [e for e in pool if e not in pick]
            pick += [rest[i] for i in torch.randperm(len(rest), generator=g)[:TOPK - len(pick)].tolist()]
        used.update(pick)
        rows.append(pick)
    w = torch.rand((m, TOPK), generator=g)
    return (w / w.sum(-1, keepdim=True) * 1.5).to(DEV), torch.tensor(rows, dtype=torch.int32, device=DEV)


def graph_us(fns, reps=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / len(fns) * 1e6


if __name__ == "__main__":
    mix = [(int(k), int(n)) for k, n in (p.split(":") for p in sys.argv[1].split(","))]
    cases = [(int(a), int(b)) for a, b in (p.split(":") for p in sys.argv[2].split(","))]
    with_coop = len(sys.argv) > 3 and sys.argv[3] == "coop"
    ks = [k for k, n in mix for _ in range(n)]
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    sl = SmoeLayer(b["bank"], TOPK)
    par = K.get_parallel_coop(DEV, 32, TOPK, H, b["bank"].inter_local, K.MAX_K_GROUPS) if with_coop else None
    bpe = {k: 3 * H * b["bank"].inter_local * k / 8 for k in set(ks)}
    print(f"mix {sys.argv[1]}  smoe flags: {os.environ.get('EXL3_SMOE_FLAGS', '(defaults)')}", flush=True)
    for m, distinct in cases:
        x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
        calls = [shared_routing(m, len(ks), distinct, 1000 * m + i) for i in range(8)]
        nbytes = sum(sum(bpe[ks[e]] for e in set(ids.flatten().tolist())) for _, ids in calls) / len(calls)
        ts = graph_us([lambda w=w, ids=ids: sl(x, ids, w, b["perm"], LIMIT) for w, ids in calls])
        line = f"m={m:2d} distinct={distinct:2d} {nbytes / 1e6:6.1f} MB  smoe {ts:7.1f} us {nbytes / ts / 1e3:4.0f} GB/s"
        if with_coop:
            tc = graph_us([lambda w=w, ids=ids: routed_moe_forward(
                x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"], LIMIT, False, True,
                decode_kernel="coop", par=par) for w, ids in calls])
            line += f"  coop {tc:7.1f} us {nbytes / tc / 1e3:4.0f} GB/s  {tc / ts:4.2f}x"
        print(line, flush=True)
