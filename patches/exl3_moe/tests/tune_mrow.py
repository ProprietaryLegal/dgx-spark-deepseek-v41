# SPDX-License-Identifier: Apache-2.0
"""exl3_mrow work-split sweep at the serve's dense shapes (K5, per rank at TP=2), DRAM-cycled in a CUDA graph: for each
shape, exllamav3's time (+ the casts dense.py runs) once, then mrow at each target work-item count (EXL3_MROW_FLAGS
picks the build, e.g. "-DMROW_PIPE=4").
    python3 tune_mrow.py [m] [items,items,...]
"""
import os
import sys

import torch
from exllamav3.modules.quant.exl3 import LinearEXL3

from test_mrow_gpu import SHAPES, _graph_us, _xcols, rand_linear
from vllm_exl3_moe.mrow import MrowGroup

m = int(sys.argv[1]) if len(sys.argv) > 1 else 4
wants = [int(w) for w in (sys.argv[2] if len(sys.argv) > 2 else "96,192,288,600,1200").split(",")]
K = 5
mul1 = torch.tensor(0x83DCD12D - 2 ** 32, dtype=torch.int32, device="cuda")
print(f"m={m} flags: {os.environ.get('EXL3_MROW_FLAGS', '(defaults)')}", flush=True)
print(f"{'shape':18s} {'exllamav3':>14s}  " + "  ".join(f"{'items>=' + str(w):>22s}" for w in wants), flush=True)
for name, lins in SHAPES.items():
    xcs = _xcols(name, lins)
    k_in = max(xc + k for xc, (k, _) in zip(xcs, lins))
    n_out = sum(n for _, n in lins)
    bytes_ = sum(k * n * K / 8 for k, n in lins)
    copies = max(2, int((256 << 20) // bytes_) + 1)
    x = torch.randn(m, k_in, device="cuda").bfloat16()
    out = torch.empty(m, n_out, device="cuda", dtype=torch.bfloat16)
    ocs = [sum(n for _, n in lins[:i]) for i in range(len(lins))]
    weights = [[rand_linear(k, n, K, 1000 * c + i) for i, (k, n) in enumerate(lins)] for c in range(copies)]
    e_fns = []
    for ws in weights:
        exl = [LinearEXL3(None, k, n, suh=su, svh=sv, trellis=t, mul1=mul1) for (t, su, sv), (k, n) in zip(ws, lins)]

        def e(exl=exl):
            for lin, xc, oc, (k, n) in zip(exl, xcs, ocs, lins):
                out[:, oc:oc + n].copy_(lin.forward(x[:, xc:xc + k].half().contiguous(), {}))
        e_fns.append(e)
    te = _graph_us(e_fns)
    cells = []
    for w in wants:
        fns = []
        for ws in weights:
            grp = MrowGroup([(t, su, sv, xc, oc) for (t, su, sv), xc, oc in zip(ws, xcs, ocs)], K, want_items=w)
            fns.append(lambda grp=grp: grp(x, out))
        tm = _graph_us(fns)
        cells.append(f"{tm:6.1f} {bytes_ / tm / 1e3:4.0f}GB/s {te / tm:4.2f}x {grp.items:4d}")
    print(f"{name:18s} {te:6.1f} {bytes_ / te / 1e3:4.0f}GB/s  " + "  ".join(f"{c:>22s}" for c in cells), flush=True)
