# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE hang probe: routed_moe_forward on the eager prefill path (m > coop rows) at serving shapes (plan v2
layer 0 mix, TP=2 rank-0 shard, mul1), skewed routing so per-expert counts reach the 32- and 64-row tile tiers.
Every call is synced under a watchdog; a call that does not finish in WATCHDOG_S seconds prints its tier split and
exits 3. Run inside the serve image on an idle GPU:  python3 hang_probe_prefill.py [m ...]
SHARED=nojoin / SHARED=join: also run a shared-expert-like exllamav3 dense GEMM pair (TP=2 shard, K5) on a side stream
right before each routed call, as vLLM's runner does for <= 256 tokens; join = the routed prefill waits for that stream
(prefill_wait), nojoin = the old behaviour. SHARED_REPS (default 4) GEMM pairs per call: enough (e.g. 100) keeps the side
stream busy past the routed call's first exl3_moe launch.
"""
import os
import sys
import threading
import time

import torch

from test_kernels_gpu import DEV, H, LIMIT, TOPK, make_bank, random_experts
from vllm_exl3_moe import kernels as K
from vllm_exl3_moe.forward import routed_moe_forward

WATCHDOG_S = float(os.environ.get("WATCHDOG_S", 30))
SKEW = float(os.environ.get("SKEW", 1.1))
SHARED = os.environ.get("SHARED", "")
state = {"busy": None}


def watchdog():
    while True:
        time.sleep(1)
        b = state["busy"]
        if b is not None and time.time() - b[0] > WATCHDOG_S:
            print(f"HANG: {b[1]} not done after {WATCHDOG_S:.0f} s", flush=True)
            os._exit(3)


def skewed_routing(m, n_exp, seed):
    g = torch.Generator().manual_seed(seed)
    p = 1.0 / torch.arange(1, n_exp + 1, dtype=torch.float64) ** SKEW
    p = p[torch.randperm(n_exp, generator=g)]
    ids = torch.stack([torch.multinomial(p, TOPK, replacement=False, generator=g) for _ in range(m)])
    w = torch.rand((m, TOPK), generator=g)
    return (w / w.sum(-1, keepdim=True) * 1.5).to(DEV), ids.to(DEV, torch.int32)


def tiers(ids, b):
    phys = b["perm"][ids.long()].reshape(-1)
    out = []
    for g in b["groups"]:
        c = torch.bincount(phys[(phys >= g.lo) & (phys < g.lo + g.size)] - g.lo, minlength=g.size).tolist()
        out.append((g.k_gu, sum(1 for x in c if 0 < x <= K.MTILE_T1), sum(1 for x in c if K.MTILE_T1 < x <= K.MTILE_T2),
                    sum(1 for x in c if K.MTILE_T2 < x <= 256), sum(1 for x in c if x > 256), max(c)))
    return out


def main():
    ms = [int(a) for a in sys.argv[1:]] or [33, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    threading.Thread(target=watchdog, daemon=True).start()
    ks = [2] * 35 + [3] * 185 + [4] * 163 + [5] * 1
    t0 = time.time()
    b = make_bank(random_experts(ks, seed=5), ks, 2, 0)
    print(f"bank: {len(ks)} experts, groups {[(g.k_gu, g.size) for g in b['groups']]}, inter_local "
          f"{b['bank'].inter_local}, built in {time.time() - t0:.0f} s", flush=True)
    shared = None
    if SHARED:
        from exllamav3.modules.quant.exl3 import LinearEXL3
        from test_dense_gpu import full_tensors
        t_gu, t_dn = full_tensors(H, 2304, 5, 1), full_tensors(1152, H, 5, 2)
        gu = LinearEXL3(None, H, 2304, suh=t_gu["suh"], svh=t_gu["svh"], trellis=t_gu["trellis"], mul1=t_gu["mul1"])
        dn = LinearEXL3(None, 1152, H, suh=t_dn["suh"], svh=t_dn["svh"], trellis=t_dn["trellis"], mul1=t_dn["mul1"])
        shared = (gu, dn, torch.cuda.Stream())
        print(f"shared-expert GEMMs on a side stream, {SHARED}", flush=True)
    main_s = torch.cuda.current_stream()
    for m in ms:
        for seed in (1, 2):
            x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
            w, ids = skewed_routing(m, len(ks), seed)
            tag = f"m={m} seed={seed} (K, t0, t1, t2, reconstruct, max count) {tiers(ids, b)}"
            state["busy"] = (time.time(), tag)
            t = time.time()
            wait = None
            if shared is not None:
                gu, dn, side = shared
                side.wait_stream(main_s)
                with torch.cuda.stream(side):
                    x16 = x.half()
                    for _ in range(int(os.environ.get("SHARED_REPS", 4))):
                        y = gu.forward(x16, {})
                        dn.forward(y[:, :1152].contiguous(), {})
                wait = side if SHARED == "join" else None
            out = routed_moe_forward(x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"],
                                     LIMIT, False, True, decode_kernel="coop", prefill_wait=wait)
            if shared is not None:
                main_s.wait_stream(shared[2])
            torch.cuda.synchronize()
            state["busy"] = None
            print(f"ok {tag}: {1e3 * (time.time() - t):.1f} ms, finite {bool(torch.isfinite(out).all())}", flush=True)
    print("PROBE DONE: no hang", flush=True)


if __name__ == "__main__":
    main()
