# SPDX-License-Identifier: Apache-2.0
"""Driver for Nsight Compute: one wo_b-shaped (4096 -> 5120, K5) exl3_mrow group, warm-up then a few launches.
    ncu --kernel-name regex:mrow_kernel --launch-skip 3 --launch-count 1 --section ... python3 prof_mrow.py [m] [k] [n]
"""
import sys

import torch

from test_mrow_gpu import rand_linear
from vllm_exl3_moe.mrow import MrowGroup

m = int(sys.argv[1]) if len(sys.argv) > 1 else 4
k = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5120
t, su, sv = rand_linear(k, n, 5, 1)
grp = MrowGroup([(t, su, sv, 0, 0)], 5)
print("items", grp.items, "ks", grp.ks, flush=True)
x = torch.randn(m, k, device="cuda").bfloat16()
out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
for _ in range(6):
    grp(x, out)
torch.cuda.synchronize()
