# SPDX-License-Identifier: Apache-2.0
"""What does splitting a layer's experts into K groups cost the decode MoE? m=4 rows routed onto 16 shared experts (the
serve's measured overlap: 16.13 distinct per verify step at k=3), concurrent K groups + wide coop tiles (the serve's
settings), one layer in a CUDA graph, GB/s on the distinct experts' bytes. Run with EXL3_MOE_COOP_WIDE=1."""
import time

import torch

from test_kernels_gpu import DEV, H, I, LIMIT, TOPK, make_bank, random_experts
from vllm_exl3_moe import kernels as K
from vllm_exl3_moe.forward import routed_moe_forward


def run(ks, label, m=4, distinct=16, reps=200):
    n = len(ks)
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    par = K.get_parallel_coop(DEV, 32, TOPK, H, b["bank"].inter_local, K.MAX_K_GROUPS)
    g = torch.Generator().manual_seed(1)
    pool = torch.randperm(n, generator=g)[:distinct]
    ids = torch.stack([pool[torch.randperm(distinct, generator=g)[:TOPK]] for _ in range(m)]).to(DEV, torch.int32)
    w = torch.rand((m, TOPK), generator=g).to(DEV)
    w = w / w.sum(-1, keepdim=True)
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    used = torch.unique(ids).tolist()
    nbytes = sum(3 * H * (I // 2) * ks[e] / 8 for e in used)
    f = lambda: routed_moe_forward(x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"], LIMIT,
                                   False, True, decode_kernel="coop", par=par)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        f()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        f()
    for _ in range(20):
        gr.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        gr.replay()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    groups = sorted({ks[e] for e in used})
    print(f"{label:28s} groups used {groups}  {len(used)} experts  {nbytes / 1e6:6.1f} MB  {dt * 1e6:6.1f} us/layer  "
          f"{nbytes / dt / 1e9:6.1f} GB/s", flush=True)


n = 384
run([2, 3, 4, 5] * 96, "mix 2/3/4/5 (4 groups)")
run([2, 3] * 192, "mix 2/3 (2 groups)")
run([3] * n, "uniform K3 (1 group)")
run([2] * n, "uniform K2 (1 group)")
run([4] * n, "uniform K4 (1 group)")
