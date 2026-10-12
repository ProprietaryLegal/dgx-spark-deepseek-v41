# SPDX-License-Identifier: Apache-2.0
"""The shared expert inside exl3_smoe (attach_shared, shared=True) vs today's path, one TP=2 rank-0 layer, per call in
CUDA graphs (8 routings a graph, as bench_smoe.py):
  routed     smoe, routed experts only (the floor)
  today      smoe on the main stream, the shared expert on a side stream (exl3_mrow gate_up -> clamped SiLU x up ->
             exl3_mrow down, as vLLM runs the EXL3 shared MLP on its aux stream), joined, then added
  serial     the same, no side stream
  fused      smoe with the shared expert as one more pick per token
    python3 bench_smoe_shared.py "2:35,3:185,4:163,5:1" 1:6,4:16,6:22
"""
import sys

import torch

from bench_smoe import graph_us, shared_routing
from test_kernels_gpu import DEV, H, I, LIMIT, TOPK, make_bank, random_experts
from test_smoe_shared_gpu import cut
from vllm_exl3_moe.mrow import MrowGroup
from vllm_exl3_moe.smoe import SmoeLayer


def main():
    mix = [(int(k), int(n)) for k, n in (p.split(":") for p in sys.argv[1].split(","))]
    cases = [(int(a), int(b)) for a, b in (p.split(":") for p in sys.argv[2].split(","))]
    ks = [k for k, n in mix for _ in range(n)]
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    g, u, d = cut(random_experts([5], seed=9)[0])
    il = b["bank"].inter_local
    sl = SmoeLayer(b["bank"], TOPK)
    sl.attach_shared(g, u, d)
    gu = MrowGroup([(g.trellis, g.suh, g.svh, 0, 0), (u.trellis, u.suh, u.svh, 0, il)], 5)
    dn = MrowGroup([(d.trellis, d.suh, d.svh, 0, 0)], 5)
    side = torch.cuda.Stream()

    def shared_mlp(x):
        y = gu(x, torch.empty((x.shape[0], 2 * il), dtype=torch.bfloat16, device=DEV))
        gate = y[:, :il].clamp(max=LIMIT)
        up = y[:, il:].clamp(min=-LIMIT, max=LIMIT)
        act = (torch.nn.functional.silu(gate) * up).contiguous()
        return dn(act, torch.empty((x.shape[0], H), dtype=torch.bfloat16, device=DEV))

    print(f"mix {sys.argv[1]}, shared expert K5 (gate/up {H}->{il} x2, down {il}->{H} on this rank)", flush=True)
    for m, distinct in cases:
        x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
        calls = [shared_routing(m, len(ks), distinct, 1000 * m + i) for i in range(8)]

        def today(w, ids, overlap):
            if overlap:
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    s = shared_mlp(x)
                r = sl(x, ids, w, b["perm"], LIMIT)
                torch.cuda.current_stream().wait_stream(side)
            else:
                s = shared_mlp(x)
                r = sl(x, ids, w, b["perm"], LIMIT)
            return r + s

        t = {
            "routed": graph_us([lambda w=w, ids=ids: sl(x, ids, w, b["perm"], LIMIT) for w, ids in calls]),
            "today": graph_us([lambda w=w, ids=ids: today(w, ids, True) for w, ids in calls]),
            "serial": graph_us([lambda w=w, ids=ids: today(w, ids, False) for w, ids in calls]),
            "fused": graph_us([lambda w=w, ids=ids: sl(x, ids, w, b["perm"], LIMIT, None, shared=True)
                               for w, ids in calls]),
        }
        diff = max((sl(x, ids, w, b["perm"], LIMIT, None, shared=True).float() - today(w, ids, True).float()).abs().max().item()
                   for w, ids in calls)
        print(f"m={m:2d} distinct={distinct:2d}: routed {t['routed']:6.1f} us, today {t['today']:6.1f}, serial "
              f"{t['serial']:6.1f}, fused {t['fused']:6.1f} ({t['today'] / t['fused']:.3f}x today); "
              f"max |fused - today| {diff:.3g}", flush=True)


if __name__ == "__main__":
    main()
