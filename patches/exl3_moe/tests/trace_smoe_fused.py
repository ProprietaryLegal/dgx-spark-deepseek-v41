# SPDX-License-Identifier: Apache-2.0
"""Timeline of one fused MoE launch (build with EXL3_SMOE_FLAGS="-DSMOE_FUSED_TRACE=1"): per task kind, when tasks
were claimed, when their dependencies cleared and when they ended, relative to the first claim; plus the critical
milestones (routing ready, last shared gate / up, last routed gate / up, last down).
    EXL3_SMOE_FLAGS="-DSMOE_FUSED_TRACE=1" python3 trace_smoe_fused.py "2:35,3:185,4:163,5:1" 4 [spread]
"""
import sys

import torch

from test_kernels_gpu import DEV, H, LIMIT, TOPK, make_bank, random_experts
from test_smoe_shared_gpu import cut
from vllm_exl3_moe.smoe import SmoeLayer, ext

N, SCALE, RX, ROT_U, WARPS, OBI = 384, 1.5, 8, 4, 8, 2


def main():
    mix = [(int(k), int(n)) for k, n in (p.split(":") for p in sys.argv[1].split(","))]
    m = int(sys.argv[2])
    spread = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    ks = [k for k, n in mix for _ in range(n)]
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    sl = SmoeLayer(b["bank"], TOPK)
    sl.attach_shared(*cut(random_experts([5], seed=9)[0]))
    g = torch.Generator().manual_seed(11)
    gw = (torch.randn(N, H, generator=g) * 0.02).to(DEV, torch.bfloat16)
    bias = (torch.randn(N, generator=g) * 0.02).to(DEV)
    x = (torch.randn((1, H), generator=g) + spread * torch.randn((m, H), generator=g)).to(DEV, torch.bfloat16)
    tr = torch.zeros(4096 * 4, dtype=torch.int64, device=DEV)
    for _ in range(3):
        sl.fused(x, gw, bias, b["perm"], LIMIT, SCALE, None, True)
    torch.cuda.synchronize()
    ext().set_fused_trace(tr)
    out, ids, w = sl.fused(x, gw, bias, b["perm"], LIMIT, SCALE, None, True)
    torch.cuda.synchronize()
    ext().set_fused_trace(torch.empty(0, dtype=torch.int64))
    rows = tr.view(4096, 4).cpu()
    hbn, nI, nob, kp = H // 128, sl.I // 128, H // 128 // OBI, TOPK + 1
    ns_max = (m + RX - 1) // RX
    per = WARPS * ROT_U
    n_rsh, n_rt = (m * 2 * hbn + per - 1) // per, N // WARPS
    n_rrt = (m * TOPK * 2 * hbn + per - 1) // per
    b_rsh, b_rt = 1, 1 + n_rsh
    b_gush = b_rt + n_rt
    b_dnsh = b_gush + ns_max * nI
    b_rrt = b_dnsh + ns_max * nob
    b_dyn = b_rrt + n_rrt
    rc = len(set(ids.flatten().tolist())) + 1                   # + the shared run (m <= 8: one shared run)
    n_gu = (rc - 1) * nI

    def kind(i):
        if i == 0: return "setup"
        if i < b_rt: return "shared rot"
        if i < b_gush: return "router"
        if i < b_dnsh: return "shared gu"
        if i < b_rrt: return "shared down"
        if i < b_dyn: return "routed rot"
        return "routed gu" if i - b_dyn < n_gu else "routed down"

    t0 = None
    stats = {}
    for i in range(4096):
        r = rows[i]
        if r[3] == 0:
            continue
        claim, ready, end = int(r[2]), int(r[1]), int(r[3])
        t0 = claim if t0 is None else min(t0, claim)
        stats.setdefault(kind(i), []).append((claim, ready, end))
    print(f"m={m}: {rc - 1} routed runs + 1 shared; blocks {ext().fused_blocks(sl.handle)}; times in us from the first claim")
    order = ["setup", "shared rot", "router", "shared gu", "shared down", "routed rot", "routed gu", "routed down"]
    for k in order:
        v = stats.get(k, [])
        if not v:
            continue
        cl = sorted(c - t0 for c, _, _ in v)
        rd = sorted(r - t0 for _, r, _ in v)
        en = sorted(e - t0 for _, _, e in v)
        work = sorted(e - r for _, r, e in v)
        print(f"  {k:12s} n={len(v):4d}  claimed {cl[0] / 1e3:7.1f}..{cl[-1] / 1e3:7.1f}  deps clear {rd[0] / 1e3:7.1f}.."
              f"{rd[-1] / 1e3:7.1f}  end {en[0] / 1e3:7.1f}..{en[-1] / 1e3:7.1f}  work p50 {work[len(work) // 2] / 1e3:6.1f}"
              f" max {work[-1] / 1e3:6.1f}")


if __name__ == "__main__":
    main()
