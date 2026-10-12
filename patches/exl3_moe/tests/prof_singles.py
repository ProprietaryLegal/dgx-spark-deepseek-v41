# SPDX-License-Identifier: Apache-2.0
"""Driver for Nsight Compute: one single K5 EXL3 linear at a serve shape, run through exl3_mrow and through exllamav3's
LinearEXL3 (what dense.py runs for wq_b / wo_b / the shared down today), warm-up then a few launches of each.
    ncu --kernel-name regex:mrow_kernel --launch-skip 3 --launch-count 1 ... python3 prof_singles.py m k n
    ncu --kernel-name regex:exl3_gemm --launch-skip 3 --launch-count 1 ... python3 prof_singles.py m k n
"""
import sys

import torch
from exllamav3.modules.quant.exl3 import LinearEXL3

from test_mrow_gpu import rand_linear
from vllm_exl3_moe.mrow import MrowGroup

m, k, n = (int(a) for a in sys.argv[1:4])
K = 5
t, su, sv = rand_linear(k, n, K, 1)
grp = MrowGroup([(t, su, sv, 0, 0)], K)
print("items", grp.items, "ks", grp.ks, flush=True)
mul1 = torch.tensor(0x83DCD12D - 2 ** 32, dtype=torch.int32, device="cuda")
lin = LinearEXL3(None, k, n, suh=su, svh=sv, trellis=t, mul1=mul1)
x = torch.randn(m, k, device="cuda").bfloat16()
xh = x.half()
out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
for _ in range(6):
    grp(x, out)
    lin.forward(xh, {})
torch.cuda.synchronize()
