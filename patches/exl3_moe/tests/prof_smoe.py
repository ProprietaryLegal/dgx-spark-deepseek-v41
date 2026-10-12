# SPDX-License-Identifier: Apache-2.0
"""Driver for Nsight Compute: one plan-v2-like layer (K2 35, K3 185, K4 163, K5 1; TP=2 rank 0), m tokens with
DSpark-like sharing (the m tokens' picks drawn from a pool of `distinct` experts), exl3_smoe and the exllamav3 coop path
(parallel K groups) a few times each.
    ncu --kernel-name regex:smoe_ --launch-skip 6 --launch-count 3 ... python3 prof_smoe.py [m] [distinct]
"""
import sys

import torch

from test_kernels_gpu import DEV, H, LIMIT, TOPK, make_bank, random_experts
from vllm_exl3_moe import kernels as K
from vllm_exl3_moe.forward import routed_moe_forward
from vllm_exl3_moe.smoe import SmoeLayer

m = int(sys.argv[1]) if len(sys.argv) > 1 else 4
distinct = int(sys.argv[2]) if len(sys.argv) > 2 else 16


def shared_routing(m, n_exp, distinct, seed):
    """Each token picks TOPK distinct experts from one pool of `distinct` experts, every pool expert used."""
    g = torch.Generator().manual_seed(seed)
    pool = torch.randperm(n_exp, generator=g)[:distinct].tolist()
    rows = []
    for t in range(m):
        if t == 0:
            rows.append(pool[:TOPK])
            continue
        rest = pool[TOPK + (t - 1) * (TOPK // 2): TOPK + t * (TOPK // 2)]
        pick = rest + [e for e in torch.randperm(distinct, generator=g).tolist() if pool[e] not in rest][:TOPK - len(rest)]
        rows.append([p if isinstance(p, int) and p in pool and pool.index(p) >= TOPK else pool[p] for p in pick][:TOPK])
    ids = torch.tensor([sorted(set(r))[:TOPK] + [-1] * (TOPK - len(set(r))) for r in rows], dtype=torch.int32, device=DEV)
    w = torch.rand((m, TOPK), generator=g).to(DEV)
    return w / w.sum(-1, keepdim=True) * 1.5, ids


if __name__ == "__main__":
    ks = [2] * 35 + [3] * 185 + [4] * 163 + [5] * 1
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    sl = SmoeLayer(b["bank"], TOPK)
    par = K.get_parallel_coop(DEV, 32, TOPK, H, b["bank"].inter_local, K.MAX_K_GROUPS)
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = shared_routing(m, len(ks), distinct, 3)
    print("distinct experts:", len(set(ids[ids >= 0].tolist())), flush=True)
    for _ in range(4):
        sl(x, ids, w, b["perm"], LIMIT)
        routed_moe_forward(x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"], LIMIT, False, True,
                           decode_kernel="coop", par=par)
    torch.cuda.synchronize()
