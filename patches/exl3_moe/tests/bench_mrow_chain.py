# SPDX-License-Identifier: Apache-2.0
"""A layer's dense chain on exl3_mrow, back to back as the serve runs it (wq_a + wkv, wq_b, wo_a x4, wo_b, shared
gate + up, shared down; K5, per rank at TP=2), over enough distinct layers that every weight comes from DRAM, in one
CUDA graph with PDL. Prints us per layer and the stream rate; compare builds with EXL3_MROW_FLAGS, e.g.
"-DMROW_L2PF=32" (L2 prefetch of each item's next k-rows before griddepcontrol.wait).
    EXL3_MROW_FLAGS="-DMROW_L2PF=32" python3 bench_mrow_chain.py [m] [layers]
"""
import os
import sys

import torch

from test_mrow_gpu import SHAPES, _graph_us, _xcols, rand_linear
from vllm_exl3_moe.mrow import MrowGroup

m = int(sys.argv[1]) if len(sys.argv) > 1 else 4
layers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
K = int(os.environ.get("K", "5"))
groups, total = [], 0
for li in range(layers):
    for gi, (name, lins) in enumerate(SHAPES.items()):
        xcs = _xcols(name, lins)
        k_in = max(xc + k for xc, (k, _) in zip(xcs, lins))
        n_out = sum(n for _, n in lins)
        ocs = [sum(n for _, n in lins[:i]) for i in range(len(lins))]
        ws = [rand_linear(k, n, K, 100000 * li + 1000 * gi + i) for i, (k, n) in enumerate(lins)]
        grp = MrowGroup([(t, su, sv, xc, oc) for (t, su, sv), xc, oc in zip(ws, xcs, ocs)], K)
        x = torch.randn(m, k_in, device="cuda").bfloat16()
        out = torch.empty(m, n_out, device="cuda", dtype=torch.bfloat16)
        groups.append((grp, x, out))
        total += sum(k * n * K / 8 for k, n in lins)


def chain():
    for grp, x, out in groups:
        grp(x, out)


us = _graph_us([chain], reps=20) / layers
print(f"m={m} flags {os.environ.get('EXL3_MROW_FLAGS', '(defaults)')}: {us:7.1f} us per layer "
      f"({total / layers / 2**20:.1f} MiB a layer, {total / layers / us / 1e3:.0f} GB/s)", flush=True)
