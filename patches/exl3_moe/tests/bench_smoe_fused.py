# SPDX-License-Identifier: Apache-2.0
"""The fused MoE launch (SmoeLayer.fused: router + top-k + routed + shared experts, one persistent launch) against the
launches it replaces with the same arithmetic: exl3_router.router_topk, then smoe with the shared expert as a slot.
One TP=2 rank-0 layer (a K mix of 384 routed experts + a K5 shared expert, a [384, H] bf16 router), 8 inputs a CUDA
graph so the routing changes call to call. Tokens share experts as DSpark verify rows do: each row is a common vector
plus `spread` x its own noise (smaller spread, more shared picks); the distinct experts a call are printed.
    python3 bench_smoe_fused.py "2:35,3:185,4:163,5:1" 1,4,6,8 [spread]
"""
import sys

import torch

from bench_smoe import graph_us
from test_kernels_gpu import DEV, H, LIMIT, TOPK, make_bank, random_experts
from test_smoe_shared_gpu import cut
from vllm_exl3_moe.jit import load
from vllm_exl3_moe.smoe import SmoeLayer

N, SCALE = 384, 1.5


def main():
    mix = [(int(k), int(n)) for k, n in (p.split(":") for p in sys.argv[1].split(","))]
    ms = [int(v) for v in sys.argv[2].split(",")]
    spread = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    ks = [k for k, n in mix for _ in range(n)]
    assert len(ks) == N, f"the mix has {len(ks)} experts, the router {N}"
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    sl = SmoeLayer(b["bank"], TOPK)
    sl.attach_shared(*cut(random_experts([5], seed=9)[0]))
    g = torch.Generator().manual_seed(11)
    gw = (torch.randn(N, H, generator=g) * 0.02).to(DEV, torch.bfloat16)
    bias = (torch.randn(N, generator=g) * 0.02).to(DEV)
    rx = load("exl3_router", "EXL3_ROUTER_FLAGS")
    e32, e64 = torch.empty(0, device=DEV), torch.empty(0, dtype=torch.int64, device=DEV)
    print(f"mix {sys.argv[1]}, router [{N}, {H}] bf16, shared expert K5, spread {spread}; fused grid "
          f"{sl_blocks(sl)} blocks", flush=True)
    for m in ms:
        xs = []
        for i in range(8):
            base = torch.randn((1, H), generator=g)
            xs.append((base + spread * torch.randn((m, H), generator=g)).to(DEV, torch.bfloat16))
        logits = torch.empty(m, N, dtype=torch.float32, device=DEV)
        tw = torch.empty(m, TOPK, dtype=torch.float32, device=DEV)
        ti = torch.empty(m, TOPK, dtype=torch.int32, device=DEV)
        counter = torch.zeros(1, dtype=torch.int32, device=DEV)
        ws = torch.empty(2 * m * N, dtype=torch.float32, device=DEV)

        def pair(x):
            rx.router_topk(x, gw, logits, bias, e32, e64, 0, SCALE, tw, ti, counter, ws)
            return sl(x, ti, tw, b["perm"], LIMIT, None, shared=True)

        def fused(x):
            return sl.fused(x, gw, bias, b["perm"], LIMIT, SCALE, None, True)[0]

        distinct = []
        for x in xs:
            pair(x)
            distinct.append(len(set(ti.flatten().tolist())))
        diff = max((pair(x).float() - fused(x).float()).abs().max().item() for x in xs)
        tp = graph_us([lambda x=x: pair(x) for x in xs])
        tf = graph_us([lambda x=x: fused(x) for x in xs])
        print(f"m={m}: distinct experts {sum(distinct) / len(distinct):5.1f} + shared; router_topk + smoe {tp:7.1f} us, "
              f"fused {tf:7.1f} us ({tp / tf:.3f}x); max |fused - pair| {diff:.3g}", flush=True)


def sl_blocks(sl):
    from vllm_exl3_moe.smoe import ext
    return ext().fused_blocks(sl.handle)


if __name__ == "__main__":
    main()
