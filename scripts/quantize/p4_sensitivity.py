# SPDX-License-Identifier: Apache-2.0
"""Measured per-layer sensitivity alpha_l for the allocator: how much the final distribution moves (KL vs the clean
model) per unit of relative noise energy in layer l's routed-expert output.

One layer-streamed pass carries a clean stream plus one copy per layer l, branched off the clean stream at l's input;
copy l gets y_routed + sigma * rms(y_routed per token) * N(0, 1) in layer l's MoE only, then runs clean to the end.
alpha_l = mean KL(clean || copy_l) / sigma^2. The allocator then prices expert e of layer l at
alpha_l * out_energy_e / sum_e out_energy_e * eps(K_e) (relative noise energy of the routed output).

Calibration tokens only (never the KLD panel). ~8K tokens: 41 streams of [L, 4, 5120] bf16 fit in ~20 GB.
  python3 p4_sensitivity.py /model /caldata/cal.safetensors /out/alpha.json [--seqs 16 --len 512 --sigma 0.1]
"""
import argparse
import copy
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.stream import Streamer  # noqa: E402


def clone_states(states):
    out = []
    for st in states:
        c = copy.copy(st)
        c.h, c.pre_mix = st.h.clone(), st.pre_mix.clone()
        c.shared = {k: v.clone() for k, v in st.shared.items()}
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cal")
    ap.add_argument("out")
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=0.1)
    a = ap.parse_args()
    d = load_file(a.cal)
    t, o = d["tokens"].tolist(), d["offsets"].tolist()
    idx = json.load(open(os.path.join(os.path.dirname(a.cal), "cal_index.json")))
    chat = [i for i, x in enumerate(idx) if x["source"] == "chat"]
    step = max(1, len(chat) // a.seqs)
    pick = chat[::step][: a.seqs]                                  # spread over domains (index is domain-ordered)
    seqs = [t[o[i]: o[i + 1]][: a.len] for i in pick]
    torch.set_default_dtype(torch.bfloat16)
    s = Streamer(a.model, seqs)
    L = s.args.n_layers
    clean = s.states
    streams = {}                                                   # layer l -> states of copy l
    gen = torch.Generator(device="cuda").manual_seed(4242)
    t0 = time.time()
    blk = None
    for j in range(L):
        blk = s.load(j)
        streams[j] = clone_states(clean)                           # branch copy j at layer j's input

        def noisy_moe(x, b):
            y = b.ffn(x).float()
            sh = b.ffn.shared_experts(x).float()
            r = y - sh
            rms = r.square().mean(-1, keepdim=True).sqrt()
            r = r + a.sigma * rms * torch.randn(r.shape, generator=gen, device=r.device)
            return (sh + r).to(torch.bfloat16)

        for key in [None] + sorted(streams):
            s.states = clean if key is None else streams[key]
            s.layer = j
            s.run_layer(blk, moe=noisy_moe if key == j else None)
        print(f"layer {j:2d}: {len(streams) + 1} streams, {time.time() - t0:.0f}s", flush=True)
        if j < L - 1:
            del blk
            torch.cuda.empty_cache()
    # KL(clean || copy_l) over all positions
    s.states = clean
    ref = [torch.log_softmax(lg, -1) for _, lg in s.iter_logits(blk)]
    alpha = {}
    for l in range(L):
        s.states = streams[l]
        kl_sum, n = 0.0, 0
        for i, lg in s.iter_logits(blk):
            lq = torch.log_softmax(lg, -1)
            p = ref[i].exp()
            kl_sum += (p * (ref[i] - lq)).sum().item()
            n += lg.size(0)
        alpha[l] = kl_sum / n / a.sigma ** 2
        print(f"alpha[{l:2d}] = {alpha[l]:.5f}  (KL {kl_sum / n:.6f})", flush=True)
    with open(a.out, "w") as f:
        json.dump({"sigma": a.sigma, "tokens": sum(map(len, seqs)), "alpha": alpha}, f, indent=1)
    print(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
