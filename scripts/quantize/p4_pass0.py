# SPDX-License-Identifier: Apache-2.0
"""P4 pass 0: stream the native model (reference implementation) over the calibration set and record, per layer,
what the bit allocator needs. No weights change.

Per routed expert e:   count (routed token slots), sum_g, sum_g2 (routing weights), out_energy = sum ||g * expert(x)||^2
                       (the energy the expert writes into the MoE output, i.e. what a relative weight error scales)
Per layer:             tokens, moe_out_energy, shared_out_energy, attn wq_b / wo_a / wo_b output energies,
                       residual-stream energy before / after the layer (hc copies averaged), seconds

  python3 p4_pass0.py /model /caldata/cal.safetensors /out/stats [--layers 0-39] [--max-seqs 0]
                     [--index cal_index.json | --domain NAME]
Writes /out/stats/layer_XX.json (resumable: finished layers are kept, but the residual stream has to be recomputed
from layer 0, so a restart re-runs earlier layers without re-writing their files).

--index (build.py's per-sequence index) or --domain (one label for every sequence) adds "domains": the per-expert
count / sum_g / sum_g2 / out_energy / tshare and the token count of each coarse domain (code, long, agentic, chat,
writing, knowledge, math, multilingual, std, random, ...), so p4_merge_stats.py can weight domains instead of tokens.
"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.stream import Streamer  # noqa: E402


def load_seqs(path, max_seqs=0):
    d = load_file(path)
    t, o = d["tokens"].tolist(), d["offsets"].tolist()
    seqs = [t[o[i]: o[i + 1]] for i in range(len(o) - 1)]
    return seqs[:max_seqs] if max_seqs else seqs


COARSE = {"code_debug": "code", "long_context": "long", "wiki": "std", "c4": "std", "technical": "std", "tiny": "std"}


def coarse(d):
    """build.py's domain names -> the groups allocation weights (every multilingual_* and exllamav3's multilingual
    text are one group; exllamav3's wiki / c4 / technical / tiny text is "std")."""
    return "multilingual" if d.startswith("multilingual") else COARSE.get(d, d)


def domain_tokens(seqs, index, label, dev):
    """(names, per-token domain id [T] on dev) in sequence order (the order run_layer hands tokens to moe), or None."""
    if not index and not label:
        return [], None
    if index:
        ix = json.load(open(index))[: len(seqs)]
        if len(ix) != len(seqs) or any(r["length"] != len(x) for r, x in zip(ix, seqs)):
            raise SystemExit(f"{index} does not describe these {len(seqs)} sequences")
        labels = [coarse(r["domain"]) for r in ix]
    else:
        labels = [label] * len(seqs)
    names = sorted(set(labels))
    did = {d: i for i, d in enumerate(names)}
    ids = torch.cat([torch.full((len(x),), did[d], dtype=torch.long) for d, x in zip(labels, seqs)])
    return names, ids.to(dev)


def parse_layers(s, n):
    if not s:
        return list(range(n))
    a, b = s.split("-") if "-" in s else (s, s)
    return list(range(int(a), int(b) + 1))


class LayerStats:
    def __init__(self, n_experts, dev, domains=()):
        z = lambda: torch.zeros(n_experts, dtype=torch.float64, device=dev)  # noqa: E731
        self.count, self.sum_g, self.sum_g2, self.out_energy, self.tshare = z(), z(), z(), z(), z()
        self.domains, self.E = list(domains), n_experts
        zd = lambda: torch.zeros(len(self.domains) * n_experts, dtype=torch.float64, device=dev)  # noqa: E731
        self.d = {k: zd() for k in ("count", "sum_g", "sum_g2", "out_energy", "tshare")}   # [domain * E + e]
        self.d_tokens = torch.zeros(len(self.domains), dtype=torch.float64, device=dev)
        self.scalars = {"tokens": 0, "moe_out_energy": 0.0, "shared_out_energy": 0.0, "wq_b_out_energy": 0.0,
                        "wo_a_out_energy": 0.0, "wo_b_out_energy": 0.0, "resid_in_energy": 0.0,
                        "resid_out_energy": 0.0}

    def add(self, k, v):
        self.scalars[k] += float(v)

    def to_json(self):
        rec = {**self.scalars, "count": self.count.tolist(), "sum_g": self.sum_g.tolist(),
               "sum_g2": self.sum_g2.tolist(), "out_energy": self.out_energy.tolist(), "tshare": self.tshare.tolist()}
        if self.domains:
            E = self.E
            rec["domains"] = {name: {"tokens": self.d_tokens[i].item(),
                                     **{k: v[i * E:(i + 1) * E].tolist() for k, v in self.d.items()}}
                              for i, name in enumerate(self.domains)}
        return rec


def sq(t):
    return t.float().square().sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cal")
    ap.add_argument("out")
    ap.add_argument("--layers", default="")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--moe-chunk", type=int, default=65536)
    ap.add_argument("--index", default="", help="build.py's cal_index.json: per-domain stats too")
    ap.add_argument("--domain", default="", help="one domain label for every sequence (instead of --index)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = load_seqs(a.cal, a.max_seqs)
    torch.set_default_dtype(torch.bfloat16)
    t0 = time.time()
    s = Streamer(a.model, seqs)
    n_tok = sum(len(x) for x in seqs)
    dom_names, dom_tok = domain_tokens(seqs, a.index, a.domain, s.device)
    print(f"{len(seqs)} sequences, {n_tok:,} tokens, streamer ready in {time.time() - t0:.0f}s"
          + (f", domains {dom_names}" if dom_names else ""), flush=True)
    layers = parse_layers(a.layers, s.args.n_layers)
    for l in range(max(layers) + 1):
        t1 = time.time()
        blk = s.load(l)
        t2 = time.time()
        st = LayerStats(blk.ffn.n_routed_experts, s.device, dom_names)
        off = [0]                                    # tokens seen this layer: run_layer goes in sequence order
        hooks = []
        # routed experts: output energy per expert (the reference passes the routing weight into Expert.forward)
        hooks.append(blk.ffn.shared_experts.register_forward_hook(lambda m, i, o: st.add("shared_out_energy", sq(o))))
        hooks.append(blk.attn.wq_b.register_forward_hook(lambda m, i, o: st.add("wq_b_out_energy", sq(o))))
        def wo_b_hook(m, i, o):                      # a forward hook must return None or it replaces the output
            st.add("wo_a_out_energy", sq(i[0]))
            st.add("wo_b_out_energy", sq(o))

        hooks.append(blk.attn.wo_b.register_forward_hook(wo_b_hook))

        def moe(x, b):
            """The reference MoE, with the routed experts run explicitly so each token's routed output r_t is known
            before the per-expert shares: tshare_e += ||g y_e(x_t)||^2 / ||r_t||^2 over e's routed tokens (the
            allocator's importance, consistent with p4_sensitivity's per-token relative noise; energy shares are
            dominated by one massive-activation expert per layer)."""
            w, idx = b.ffn.gate(x, None)
            flat = idx.flatten()
            st.count.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
            st.sum_g.index_add_(0, flat, w.flatten().double())
            st.sum_g2.index_add_(0, flat, w.flatten().double().square())
            d = None
            if dom_tok is not None:
                d = dom_tok[off[0]: off[0] + x.size(0)]
                key = (d[:, None] * st.E + idx.long()).flatten()
                st.d["count"].index_add_(0, key, torch.ones_like(key, dtype=torch.float64))
                st.d["sum_g"].index_add_(0, key, w.flatten().double())
                st.d["sum_g2"].index_add_(0, key, w.flatten().double().square())
                st.d_tokens.index_add_(0, d, torch.ones_like(d, dtype=torch.float64))
            off[0] += x.size(0)
            routed = torch.zeros(x.size(0), x.size(1), dtype=torch.float32, device=x.device)
            outs = []
            for e in range(b.ffn.n_routed_experts):
                tok, slot = (idx == e).nonzero(as_tuple=True)
                if tok.numel() == 0:
                    continue
                ye = b.ffn.experts[e](x[tok], w[tok, slot, None]).float()
                routed.index_add_(0, tok, ye)
                outs.append((e, tok, ye))
            rn = routed.square().sum(-1).clamp_min(1e-30)
            for e, tok, ye in outs:
                en = ye.square().sum(-1)
                st.out_energy[e] += en.sum().double()
                st.tshare[e] += (en / rn[tok]).sum().double()
                if d is not None:
                    key = d[tok] * st.E + e
                    st.d["out_energy"].index_add_(0, key, en.double())
                    st.d["tshare"].index_add_(0, key, (en / rn[tok]).double())
            y = (routed + b.ffn.shared_experts(x).float()).to(x.dtype)
            st.add("moe_out_energy", sq(y))
            st.add("tokens", x.size(0))
            return y

        def block_hook(i, h_in, h_out):
            st.add("resid_in_energy", sq(h_in) / h_in.size(2))
            st.add("resid_out_energy", sq(h_out) / h_out.size(2))

        s.run_layer(blk, moe=moe, moe_chunk_tokens=a.moe_chunk, block_hook=block_hook)
        torch.cuda.synchronize()
        assert off[0] == n_tok, (off[0], n_tok)
        for h in hooks:
            h.remove()
        rec = st.to_json()
        rec.update(layer=l, load_s=round(t2 - t1, 1), run_s=round(time.time() - t2, 1))
        path = os.path.join(a.out, f"layer_{l:02d}.json")
        if l in layers and not os.path.exists(path):
            with open(path + ".tmp", "w") as f:
                json.dump(rec, f)
            os.replace(path + ".tmp", path)
        c = st.count
        print(f"layer {l:2d}: load {t2 - t1:5.1f}s run {time.time() - t2:6.1f}s | experts used {(c > 0).sum().item()}"
              f"/{c.numel()}, min/med/max count {c.min().item():.0f}/{c.median().item():.0f}/{c.max().item():.0f}"
              f" | moe/resid energy {rec['moe_out_energy'] / max(rec['resid_out_energy'], 1e-9):.4f}"
              f" | top expert energy share {max(rec['out_energy']) / max(sum(rec['out_energy']), 1e-30):.3f},"
              f" top token share {max(rec['tshare']) / max(rec['tokens'], 1):.4f}", flush=True)
        del blk
        torch.cuda.empty_cache()
    print(f"DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
