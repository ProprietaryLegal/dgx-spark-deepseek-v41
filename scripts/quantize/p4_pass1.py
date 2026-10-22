# SPDX-License-Identifier: Apache-2.0
"""P4 pass 1: sequential EXL3 quantization of the routed experts, layer by layer, on the reference implementation.

Per layer l (the stream entering l already went through the quantized layers < l):
  1. phase A (attention, native) for every calibration sequence -> MoE input x [N, 5120] and routing (top-6, weights)
  2. per expert e (this part's share): routing-weighted Hessians from the tokens routed to e
       H_gu = sum (g x)(g x)^T + tau * mean(g^2) * H_all / N      (all-token prior: cold experts stay well posed)
       H_d  = sum a a^T, a = g * silu(gate) * up from the native expert (+ tau * prior from 1024 sampled tokens)
     EXL3 encode at the plan's K (exllamav3: mul1 codebook, out scales auto), reconstruct, and add the quantized
     expert's output on its routed tokens into Y; error vs the native expert accumulated per expert.
  3. the stream advances through layer l with every expert's EXL3 tensors (this part's and the peers' part files, which
     exchange_sync.sh pushes between the boxes): phase A again per group, the quantized MoE (+ the native shared
     expert) reconstructed from the tensors, phase C -> the next layer's stream, identical on every box.
Multi-box: --part i/n gives each box a share of the experts; the boxes swap encoded part files (~1-2 GB a layer), not
partial MoE outputs. Until Oct 10 the boxes swapped partial outputs Y [tokens, 5120] (fp32 in memory, 8.7 GB of bf16 a
layer on disk at 853K tokens); without Y a layer holds ~49 GiB at 853K tokens (16 GiB less).

  python3 p4_pass1.py /model /cal.safetensors plan.json /out [--layers 0-39] [--part 0/1] [--exchange DIR] [--tau 512]
Outputs: /out/layer_XX.partI.safetensors (EXL3 tensors, plugin key names), /out/layer_XX.partI.json (metrics).

Partial re-encode (--reuse CKPT --old-plan PLAN): an expert whose K (gate/up and down) is the same in both plans is not
re-encoded: its EXL3 tensors come from the assembled checkpoint CKPT (model-exl3-LXX.safetensors, encoded with PLAN)
and only its quantized output enters Y, so the stream still runs through every quantized expert of earlier layers.
Comma lists (--reuse A,B --old-plan PA,PB) reuse from the first checkpoint that holds an expert at its new K.
The changed experts are split over the parts by K (the encode cost), the unchanged ones round-robin, and the part files
hold only the re-encoded experts (a resume or an assembly takes the rest from CKPT).

Domain-weighted Hessians (--index cal_index.json --domain-weights code=1,chat=1,...): every token's rows enter the
Hessians (and the all-token prior) with weight w_d / n_d of its domain, scaled to mean 1, so an encode is fitted to a
chosen domain mix instead of the token mix of the calibration set (plan v2: 42% code; an EXL3 encode fits its
Hessian's input statistics, and build v2.1's KLD gap to Mia's pack grew with how thin a domain was in calibration).
The down projection's prior tokens are drawn with the same weights. Reported errors stay unweighted.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.exl3 import encode_expert, expert_forward_dense, h_data, mxfp4_dequant, reconstruct  # noqa: E402
from dsv41q.refmodel import drop_page_cache  # noqa: E402
from dsv41q.stream import Streamer  # noqa: E402

KCHAR = {"a": 1.5, "b": 2.5, "c": 3.5}


def kval(c):
    """Plan character -> bitrate. Integer rates must be Python ints: exllamav3's bindings reject 5.0 for K."""
    k = KCHAR.get(c, None) or float(c)
    return int(k) if float(k).is_integer() else k


def load_seqs(path, max_seqs=0):
    d = load_file(path)
    t, o = d["tokens"].tolist(), d["offsets"].tolist()
    seqs = [t[o[i]: o[i + 1]] for i in range(len(o) - 1)]
    return seqs[:max_seqs] if max_seqs else seqs


class _Parts:
    """Several open safetensors files as one: keys() and get_tensor() over all of them."""

    def __init__(self, files):
        self.where = {k: f for f in files for k in f.keys()}

    def keys(self):
        return self.where.keys()

    def get_tensor(self, k):
        return self.where[k].get_tensor(k)


class Reuse:
    """Old EXL3 experts from assembled checkpoints (p4_assemble.py layout: model-exl3-LXX.safetensors, plugin key names)
    and the plans they were encoded with. Several pairs (--reuse A,B --old-plan PA,PB): an expert comes from the first
    checkpoint whose plan gives it the new plan's K (gate/up and down), so only the experts that no checkpoint holds at
    their new K are re-encoded (e.g. a plan that moves some experts up past one build and others back to an older one)."""

    device = "cuda"

    def __init__(self, ckpts, old_plans, plan):
        ckpts = [c for c in ckpts.split(",") if c] if isinstance(ckpts, str) else list(ckpts)
        olds = [p for p in old_plans.split(",") if p] if isinstance(old_plans, str) else list(old_plans)
        if not ckpts or len(ckpts) != len(olds):
            raise SystemExit(f"--reuse lists {len(ckpts)} checkpoints and --old-plan {len(olds)} plans: they pair up")
        self.ckpts, self.plan = ckpts, plan
        self.olds = [p if isinstance(p, dict) else json.load(open(p)) for p in olds]
        self._open = {}                                # (checkpoint, layer) -> open file, one layer at a time

    @staticmethod
    def kpair(p, l, e):
        x = p["layers"][str(l)]
        return x["k"][e], x.get("k_down", x["k"])[e]

    def source(self, l, e):
        """Index of the first checkpoint whose plan gives expert e of layer l the new plan's K, or None."""
        want = self.kpair(self.plan, l, e)
        return next((i for i, o in enumerate(self.olds) if self.kpair(o, l, e) == want), None)

    def changed(self, plan, l):
        """Experts of layer l that no checkpoint holds at the new plan's K (gate/up or down): the ones to encode."""
        if plan is not self.plan and plan != self.plan:
            raise SystemExit("reuse: changed() asked about another plan than the one given at construction")
        return [e for e in range(len(plan["layers"][str(l)]["k"])) if self.source(l, e) is None]

    def _layer_file(self, i, l):
        if (i, l) not in self._open:
            self._open = {k: v for k, v in self._open.items() if k[1] == l}
            self._open[(i, l)] = self.open_layer(self.ckpts[i], l, self.device)
        return self._open[(i, l)]

    @staticmethod
    def open_layer(ckpt, l, device="cuda"):
        """Layer l's experts: model-exl3-LXX.safetensors (p4_assemble.py) or its 2-box parts model-exl3-LXX-p*.safetensors
        (the v3 dirs), the latter behind one keys() / get_tensor() view."""
        from safetensors import safe_open
        one = os.path.join(ckpt, f"model-exl3-L{l:02d}.safetensors")
        if os.path.exists(one):
            return safe_open(one, "pt", device=device)
        parts = sorted(glob.glob(os.path.join(ckpt, f"model-exl3-L{l:02d}-p*.safetensors")))
        if not parts:
            raise SystemExit(f"reuse: no model-exl3-L{l:02d}[-p*].safetensors in {ckpt}")
        return _Parts([safe_open(f, "pt", device=device) for f in parts])

    def expert(self, l, e):
        """{proj: {piece: tensor}} of expert e of layer l from the checkpoint that holds it at the new plan's K."""
        i = self.source(l, e)
        if i is None:
            raise SystemExit(f"reuse: layer {l} expert {e} is at its new K in none of {self.ckpts}")
        f = self._layer_file(i, l)
        pre = f"layers.{l}.ffn.experts.{e}."
        out = {}
        for k in f.keys():
            if k.startswith(pre):
                proj, piece = k[len(pre):].split(".", 1)
                out.setdefault(proj, {})[piece] = f.get_tensor(k)
        if set(out) != {"w1", "w3", "w2"}:
            raise SystemExit(f"layer {l} expert {e}: {sorted(out)} in the reuse checkpoint {self.ckpts[i]}")
        return out


def assign_parts(plan, l, changed, nparts):
    """Expert -> part for a partial re-encode: changed experts dealt by K (highest first, the slowest to encode) so both
    boxes encode about the same bits, then the unchanged ones round-robin. Identical on every box."""
    ks = plan["layers"][str(l)]["k"]
    order = sorted(changed, key=lambda e: (-kval(ks[e]), e))
    part_of = {e: i % nparts for i, e in enumerate(order)}
    rest = [e for e in range(len(ks)) if e not in part_of]
    part_of.update({e: i % nparts for i, e in enumerate(rest)})
    return part_of


def token_weights(seqs, index, spec, dev):
    """Per-token Hessian weights in stream order from build.py's index and domain=weight pairs (p4_pass0's coarse
    domains), mean 1 over all tokens; domains without a weight get 0."""
    from p4_pass0 import coarse
    w = {k: float(v) for k, v in (x.split("=") for x in spec.split(","))}
    ix = json.load(open(index))[: len(seqs)]
    if len(ix) != len(seqs) or any(r["length"] != len(x) for r, x in zip(ix, seqs)):
        raise SystemExit(f"{index} does not describe these {len(seqs)} sequences")
    doms = [coarse(r["domain"]) for r in ix]
    n = {}
    for d_, x in zip(doms, seqs):
        n[d_] = n.get(d_, 0) + len(x)
    per_tok = torch.cat([torch.full((len(x),), w.get(d_, 0.0) / n[d_], dtype=torch.float32) for d_, x in zip(doms, seqs)])
    if float(per_tok.sum()) <= 0:
        raise SystemExit("--domain-weights gives every token weight 0")
    per_tok *= per_tok.numel() / per_tok.sum()
    tot = sum(w.get(d_, 0.0) for d_ in n)
    shares = {d_: round(w.get(d_, 0.0) / tot, 4) for d_ in sorted(n)}          # each domain's share of the Hessians
    return per_tok.to(dev), shares


def wait_parts(out_dir, l, part, nparts):
    """Block until every other part has published layer l: its metrics json (written after its tensor file), then
    the tensor files the json lists (exchange_sync.sh pushes both; rsync renames into place, so present = complete)."""
    for p in range(nparts):
        if p == part:
            continue
        mpath = os.path.join(out_dir, f"layer_{l:02d}.part{p}.json")
        while not os.path.exists(mpath):
            time.sleep(2)
        for name in json.load(open(mpath)).get("files", []):
            while not os.path.exists(os.path.join(out_dir, name)):
                time.sleep(2)


MEMLOG = int(os.environ.get("DSV41_MEMLOG", "0"))   # > 0: log CUDA memory at each step of the first MEMLOG experts


def mem(log, tag):
    if MEMLOG:
        log(f"    mem {tag}: allocated {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB, "
            f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}, reserved {torch.cuda.memory_reserved() / 2 ** 30:.2f}")


EXPERT_CHUNK = int(os.environ.get("DSV41_EXPERT_CHUNK", "32768"))   # routed tokens per chunk in an expert's Hessians and error forwards: a hot expert (154K tokens at
                       # layer 3 of calibration v3) otherwise materializes ~15 GB of fp32 temporaries at once


def chunked_err(flat, Wr, tok, slot, wq, wn, limit):
    """(sum |yq - yn|^2, sum |yn|^2) of one expert over its routed tokens, CHUNK rows at a time."""
    err = energy = 0.0
    for c in range(0, tok.numel(), EXPERT_CHUNK):
        t_c, s_c = tok[c: c + EXPERT_CHUNK], slot[c: c + EXPERT_CHUNK]
        x_c, g_c = flat[t_c], Wr[t_c, s_c]
        yq = expert_forward_dense(x_c, g_c, *wq, limit)
        yn = expert_forward_dense(x_c, g_c, *wn, limit)
        err += (yq - yn).square().sum().item()
        energy += yn.square().sum().item()
    return err, energy


@torch.inference_mode()        # without it every op on the block's nn.Parameters records autograd and keeps its inputs
def quantize_layer(s, blk, l, plan, part, nparts, out_dir, tau, exchange, limit, log, reuse=None, tok_w=None):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    mem(log, "layer start")
    flat, _ = s.phase_a(blk, keep_pend=False)                       # [N, 5120] bf16, every calibration token
    N = flat.size(0)
    mem(log, "after phase A")
    t_a = time.time()
    W_, I_ = [], []
    for c in range(0, N, 65536):
        w, idx = blk.ffn.gate(flat[c: c + 65536], None)
        W_.append(w.float())
        I_.append(idx)
    Wr, Ir = torch.cat(W_), torch.cat(I_)                          # [N, 6]
    del W_, I_
    mem(log, "after routing")
    H_all = torch.zeros(5120, 5120, dtype=torch.float32, device=flat.device)
    for c in range(0, N, 32768):
        xc = flat[c: c + 32768].float()
        if tok_w is not None:
            xc = xc * tok_w[c: c + 32768, None].sqrt()
        H_all.addmm_(xc.t(), xc)
    mean_g2 = Wr.square().mean().item()
    gsamp = torch.Generator(device="cpu").manual_seed(1000 + l)
    if tok_w is None:
        samp = torch.randperm(N, generator=gsamp, device="cpu")[:1024].to(flat.device)
    else:                                                          # prior tokens drawn with the Hessian weights
        samp = torch.multinomial(tok_w.cpu(), 1024, replacement=False, generator=gsamp).to(flat.device)
    xs = flat[samp].float()                                        # generic tokens for the down-projection prior
    mem(log, "after routing + H_all")
    ks = plan["layers"][str(l)]["k"]
    kd = plan["layers"][str(l)].get("k_down", ks)
    tensors, metrics = {}, {"layer": l, "part": part, "experts": {}}
    t_h = t_enc = t_fwd = 0.0
    if reuse is None:
        mine, changed = [e for e in range(384) if e % nparts == part], None
    else:
        changed = set(reuse.changed(plan, l))
        part_of = assign_parts(plan, l, changed, nparts)
        mine = [e for e in range(384) if part_of[e] == part]
        log(f"  layer {l:2d}: {len(changed)} experts change K, this part re-encodes "
            f"{sum(1 for e in mine if e in changed)} and reuses {sum(1 for e in mine if e not in changed)}")
    for e in mine:
        ex = blk.ffn.experts[e]
        W1 = mxfp4_dequant(ex.w1.weight, ex.w1.scale)              # fp32 [2304, 5120]
        W3 = mxfp4_dequant(ex.w3.weight, ex.w3.scale)
        W2 = mxfp4_dequant(ex.w2.weight, ex.w2.scale)              # fp32 [5120, 2304]
        t1 = time.time()
        tok, slot = (Ir == e).nonzero(as_tuple=True)
        n_e = tok.numel()
        if changed is not None and e not in changed:               # same K as the reuse checkpoint: its tensors
            old = reuse.expert(l, e)
            t2 = time.time()
            t_h += t2 - t1
            Wq1 = reconstruct(old["w1"], 5120, 2304)
            Wq3 = reconstruct(old["w3"], 5120, 2304)
            Wq2 = reconstruct(old["w2"], 2304, 5120)
            err, energy = chunked_err(flat, Wr, tok, slot, (Wq1, Wq3, Wq2), (W1.half(), W3.half(), W2.half()), limit)
            t_fwd += time.time() - t2
            metrics["experts"][e] = {"n": n_e, "K": kval(ks[e]), "K_down": kval(kd[e]), "reused": True,
                                     "out_err": err, "out_energy": energy}
            del old, Wq1, Wq3, Wq2, W1, W3, W2               # free this expert's tensors before the next one's
            continue
        we = tok_w[tok] if tok_w is not None else None
        n_h = float(we.sum()) if we is not None else n_e             # effective rows in this expert's Hessians

        def inter(x, g):
            gate = (x @ W1.t()).clamp(max=limit)
            up = (x @ W3.t()).clamp(-limit, limit)
            return (F.silu(gate) * up * g[:, None]).to(torch.bfloat16).float()

        # H_gu = sum (g x)(g x)^T and H_d = sum a a^T over the routed tokens, in chunks (one chunk for most experts:
        # the same single GEMM as before; hot experts sum chunk GEMMs instead of holding [n, 5120] fp32 copies)
        H_gu = H_d = None
        for c in range(0, max(n_e, 1), EXPERT_CHUNK):
            t_c, s_c = tok[c: c + EXPERT_CHUNK], slot[c: c + EXPERT_CHUNK]
            x_c, g_c = flat[t_c].float(), Wr[t_c, s_c]
            w_c = we[c: c + EXPERT_CHUNK].sqrt() if we is not None else None
            xg = x_c * (g_c if w_c is None else g_c * w_c)[:, None]
            H_gu = xg.t() @ xg if H_gu is None else H_gu.addmm_(xg.t(), xg)
            a = inter(x_c, g_c)
            if w_c is not None:
                a = a * w_c[:, None]
            H_d = a.t() @ a if H_d is None else H_d.addmm_(a.t(), a)
            del x_c, xg, a
        if len(metrics["experts"]) < MEMLOG:
            mem(log, f"e{e} n={n_e} after H_gu")
        H_gu.add_(H_all, alpha=tau * mean_g2 / N)
        ap_ = inter(xs, torch.full((xs.size(0),), mean_g2 ** 0.5, device=xs.device))
        H_d.add_(ap_.t() @ ap_, alpha=tau / xs.size(0))
        t2 = time.time()
        t_h += t2 - t1
        K_gu, K_d = kval(ks[e]), kval(kd[e])
        if len(metrics["experts"]) < MEMLOG:
            mem(log, f"e{e} before encode K{K_gu}")
        tens, perr = encode_expert(W1, W3, W2, h_data(H_gu, n_h + tau, f"L{l}.e{e}.gu"),
                                   h_data(H_d, n_h + tau, f"L{l}.e{e}.d"), K_gu, K_d, seed=l * 1000 + e,
                                   work_dir=out_dir)
        t3 = time.time()
        t_enc += t3 - t2
        if len(metrics["experts"]) < MEMLOG:
            mem(log, f"e{e} after encode")
        Wq1 = reconstruct(tens["w1"], 5120, 2304)
        Wq3 = reconstruct(tens["w3"], 5120, 2304)
        Wq2 = reconstruct(tens["w2"], 2304, 5120)
        err, energy = chunked_err(flat, Wr, tok, slot, (Wq1, Wq3, Wq2), (W1.half(), W3.half(), W2.half()), limit)
        t_fwd += time.time() - t3
        for proj in ("w1", "w3", "w2"):
            for k, v in tens[proj].items():
                tensors[f"layers.{l}.ffn.experts.{e}.{proj}.{k}"] = v.contiguous().cpu()
        metrics["experts"][e] = {"n": n_e, "K": K_gu, "K_down": K_d, "proxy": perr, "out_err": err,
                                 "out_energy": energy}
        del ap_, H_gu, H_d, tens, Wq1, Wq3, Wq2, W1, W3, W2   # before the next expert's working set
        done = len(metrics["experts"])
        if done % 32 == 0:
            e_err = sum(m["out_err"] for m in metrics["experts"].values())
            e_en = sum(m["out_energy"] for m in metrics["experts"].values())
            log(f"  layer {l:2d}: {done}/{len(mine)} experts, {time.time() - t0:.0f}s, "
                f"routed-output rel err so far {(e_err / max(e_en, 1e-12)) ** 0.5:.4f}")
    files = []
    tname = f"layer_{l:02d}.part{part}.safetensors"
    tpath = os.path.join(out_dir, tname)
    if tensors or reuse is None:                                   # a partial run with nothing re-encoded writes none
        save_file(tensors, tpath + ".tmp")
        os.chmod(tpath + ".tmp", 0o644)                            # host-side sync runs as the normal user
        os.replace(tpath + ".tmp", tpath)
        drop_page_cache(tpath)
        files.append(tname)
    mem_encode = torch.cuda.max_memory_allocated() / 2 ** 30
    del flat, Wr, Ir, H_all, xs, tensors
    torch.cuda.empty_cache()
    tot_err = sum(m["out_err"] for m in metrics["experts"].values())
    tot_en = sum(m["out_energy"] for m in metrics["experts"].values())
    metrics.update(tokens=N, files=files, routed_out_rel_err=(tot_err / max(tot_en, 1e-12)) ** 0.5,
                   max_mem_gib_encode=round(mem_encode, 1),
                   seconds={"phase_a": round(t_a - t0, 1), "hessians": round(t_h, 1), "encode": round(t_enc, 1),
                            "fwd": round(t_fwd, 1), "total_encode": round(time.time() - t0, 1)})
    mpath = os.path.join(out_dir, f"layer_{l:02d}.part{part}.json")
    with open(mpath + ".tmp", "w") as f:                           # the marker the peers wait for, after the tensors
        json.dump(metrics, f)
    os.chmod(mpath + ".tmp", 0o644)
    os.replace(mpath + ".tmp", mpath)
    t_w = time.time()
    if nparts > 1:
        wait_parts(out_dir, l, part, nparts)
    t_s = time.time()
    torch.cuda.reset_peak_memory_stats()
    replay_layer(s, blk, l, out_dir, nparts, limit, reuse, moe_chunk=STREAM_CHUNK)
    log(f"layer {l:2d} part {part}/{nparts}: {len(mine)} experts, routed-output rel err "
        f"{metrics['routed_out_rel_err']:.4f}, seconds {metrics['seconds']}, wait {t_s - t_w:.1f}, "
        f"stream {time.time() - t_s:.1f}, max mem GiB encode {mem_encode:.1f} stream "
        f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cal")
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--exchange", default="")
    ap.add_argument("--tau", type=float, default=512.0)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--index", default="", help="build.py's index of the calibration set (for --domain-weights)")
    ap.add_argument("--domain-weights", default="", help="domain=weight,...: domain-weighted Hessians")
    ap.add_argument("--reuse", default="", help="assembled checkpoint(s), comma list: experts are kept where one holds "
                                                "them at the new K")
    ap.add_argument("--old-plan", default="", help="the plan(s) --reuse was encoded with, same order")
    ap.add_argument("--head", type=int, default=0,
                    help="also encode the lm_head at this K from the stream's final hidden states (part 0 of a run "
                         "through the last layer; p4_dense.encode_head): <out>/dense_head.safetensors, its H in head_H.pt")
    ap.add_argument("--solo-from", type=int, default=-1,
                    help="from this layer on, one box does every expert (part 0/1, no exchange): continues a 2-box run "
                         "on one Spark when the other is down; layers before it keep --part / --exchange")
    a = ap.parse_args()
    part, nparts = (int(x) for x in a.part.split("/"))
    lo, hi = (int(x) for x in a.layers.split("-"))
    os.makedirs(a.out, exist_ok=True)
    plan = json.load(open(a.plan))
    if bool(a.reuse) != bool(a.old_plan):
        raise SystemExit("--reuse and --old-plan go together")
    reuse = Reuse(a.reuse, a.old_plan, plan) if a.reuse else None
    logf = open(os.path.join(a.out, f"pass1.part{part}.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.set_default_dtype(torch.bfloat16)
    seqs = load_seqs(a.cal, a.max_seqs)
    s = Streamer(a.model, seqs)
    log(f"pass 1 part {part}/{nparts}: {len(seqs)} sequences, {sum(map(len, seqs)):,} tokens, layers {lo}-{hi}"
        + (f", re-encoding only K changes vs {a.old_plan} (the rest from {a.reuse})" if reuse else ""))
    tok_w = None
    if a.domain_weights:
        if not a.index:
            raise SystemExit("--domain-weights needs --index")
        tok_w, shares = token_weights(seqs, a.index, a.domain_weights, s.device)
        log(f"domain-weighted Hessians, shares {shares}")
    limit = s.args.swiglu_limit
    for l in range(0, hi + 1):
        t0 = time.time()
        blk = s.load(l)
        if l < lo:                     # replay already-quantized layers from their saved tensors (resume)
            replay_layer(s, blk, l, a.out, nparts, limit, reuse)
            log(f"layer {l:2d}: replayed from saved tensors in {time.time() - t0:.0f}s")
        elif 0 <= a.solo_from <= l:
            quantize_layer(s, blk, l, plan, 0, 1, a.out, a.tau, "", limit, log, reuse, tok_w)
        else:
            quantize_layer(s, blk, l, plan, part, nparts, a.out, a.tau, a.exchange, limit, log, reuse, tok_w)
        if not (a.head and part == 0 and l == hi == s.args.n_layers - 1):
            del blk                    # the last block stays for the head's final collapse + norm
        torch.cuda.empty_cache()
    if a.head and part == 0 and hi == s.args.n_layers - 1:
        try:                           # never fail pass 1 (its part files are done) over the head
            from p4_dense import encode_head
            work = os.path.join(a.out, "work_head")
            os.makedirs(work, exist_ok=True)
            encode_head(s, blk, [(a.head, a.out)], work, 9999, log)
        except Exception as e:         # noqa: BLE001
            log(f"head encode failed (pass 1 unaffected): {type(e).__name__}: {e}")
    log("DONE")


STREAM_CHUNK = 131072   # tokens per MoE call when the stream advances on EXL3 tensors (each call reconstructs every
                        # expert once: bigger groups, fewer reconstructions; ~10 GB of group intermediates)


@torch.inference_mode()
def replay_layer(s, blk, l, out_dir, nparts, limit, reuse=None, moe_chunk=65536):
    """Run layer l with its saved EXL3 experts (all parts) so a resumed run continues from the quantized stream; in a
    partial run the experts not in the part files come from the reuse checkpoint."""
    tens = {}
    files = sorted(glob.glob(os.path.join(out_dir, f"layer_{l:02d}.part*.safetensors")))   # however the layer was split
    for f in files:
        tens.update(load_file(f, device="cuda"))
        drop_page_cache(f)
    have = {int(k.split(".")[4]) for k in tens if ".ffn.experts." in k}
    if reuse is not None:
        for e in sorted(set(range(384)) - have):
            for proj, pieces in reuse.expert(l, e).items():
                for piece, v in pieces.items():
                    tens[f"layers.{l}.ffn.experts.{e}.{proj}.{piece}"] = v
        have = set(range(384))
    n_exp = len(have)
    if n_exp != 384:
        raise SystemExit(f"replay layer {l}: {n_exp} experts in {len(files)} part file(s), need all 384")

    def moe(x, b):
        w, idx = b.ffn.gate(x, None)
        y = b.ffn.shared_experts(x).float()
        for e in range(384):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            pre = f"layers.{l}.ffn.experts.{e}."
            q = {pj: reconstruct({k[len(pre) + 3:]: v for k, v in tens.items() if k.startswith(pre + pj + ".")},
                                 *((5120, 2304) if pj != "w2" else (2304, 5120))) for pj in ("w1", "w3", "w2")}
            y.index_add_(0, tok, expert_forward_dense(x[tok], w[tok, slot].float(), q["w1"], q["w3"], q["w2"], limit))
        return y.to(torch.bfloat16)

    s.run_layer(blk, moe=moe, moe_chunk_tokens=moe_chunk)


if __name__ == "__main__":
    main()
