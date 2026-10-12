# SPDX-License-Identifier: Apache-2.0
"""Fidelity on the published panel: KL(native || ours), top-1 agreement and NLL, at every scored (assistant)
position of every panel conversation, both models run by the same layer-streamed reference implementation, so the
number measures the weights, not two different engines.

  --mode ref    native checkpoint; writes ref.safetensors: per scored position the top-128 token ids + logprobs and
                the NLL of the true next token
  --mode quant  routed experts replaced by our EXL3 tensors (--tensors DIR: layer_XX.part*.safetensors from pass 1,
                missing layers stay native); reads ref.safetensors, writes kld.json

KL uses the native top-128 support plus one tail bucket: sum_top P log(P/Q) + P_rest log(P_rest / Q_rest), exact when
the top-128 holds all the mass and a tight estimate otherwise.

  python3 p4_kld.py /model /caldata/panel.safetensors /caldata/panel_index.json /out --mode ref
  python3 p4_kld.py /model /caldata/panel.safetensors /caldata/panel_index.json /out --mode quant --tensors /pass1
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.exl3 import quant_moe  # noqa: E402
from dsv41q.overlay import Exl3Checkpoint, head_weight, overlay_block  # noqa: E402
from dsv41q.refmodel import drop_page_cache  # noqa: E402
from dsv41q.stream import Streamer  # noqa: E402

TOPK = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("panel")
    ap.add_argument("index")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["ref", "quant"], required=True)
    ap.add_argument("--tensors", default="")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--head-mxfp8", action="store_true",
                    help="quant mode: also score the same model with an MXFP8 lm_head (vLLM's own quantizer, what a "
                         "Marlin MXFP8 head would serve), in the same pass")
    ap.add_argument("--exl3-ckpt", default="",
                    help="quant mode: a sharded EXL3 pack (e.g. Mia-AiLab's 2.9 bpw): its routed experts AND every dense "
                         "linear / head it stores as EXL3 replace the official ones (dsv41q.overlay)")
    ap.add_argument("--max-layers", type=int, default=0, help="debug: run only the first N layers (smoke tests)")
    ap.add_argument("--dense", default="",
                    help="quant mode: EXL3 dense linears from our dense pass (same layout), on top of --tensors experts")
    ap.add_argument("--dump-top", type=int, default=0,
                    help="quant mode: also save each scored position's top-N token ids + logprobs (quant_top.safetensors, "
                         "same order as ref.safetensors), for comparing this harness with a served path position by position")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load_file(a.panel)
    toks, offs, score = d["tokens"].tolist(), d["offsets"].tolist(), d["score"]
    n = len(offs) - 1 if not a.max_seqs else min(a.max_seqs, len(offs) - 1)
    seqs = [toks[offs[i]: offs[i + 1]] for i in range(n)]
    index = json.load(open(a.index))[:n]
    torch.set_default_dtype(torch.bfloat16)
    t0 = time.time()
    s = Streamer(a.model, seqs)
    limit = s.args.swiglu_limit
    packs = [Exl3Checkpoint(p) for p in (a.exl3_ckpt, a.dense) if p]
    overlaid = {}
    if a.mode == "quant":
        for pk in packs:                                   # an EXL3 head in the pack or in our dense output
            hw = head_weight(pk)
            if hw is not None:
                s.head_override = hw
                overlaid["head"] = pk.dir
    blk = None
    n_layers = a.max_layers or s.args.n_layers
    for l in range(n_layers):
        t1 = time.time()
        files = sorted(glob.glob(os.path.join(a.tensors, f"layer_{l:02d}.part*.safetensors"))) if a.tensors else []
        exl3_experts = a.mode == "quant" and (files or any(pk.has_experts(l) for pk in packs))
        blk = s.load(l, experts=not exl3_experts)          # native MXFP4 experts only when nothing replaces them
        moe = None
        if a.mode == "quant":
            tensors = {}
            for f in files:
                tensors.update(load_file(f, device="cuda"))
                drop_page_cache(f)
            for pk in packs:
                tensors.update(pk.tensors(f"layers.{l}."))
            if any(".experts." in k for k in tensors):
                moe = quant_moe(l, tensors, limit)
            if packs:
                overlaid[l] = overlay_block(blk, l, tensors)
        s.run_layer(blk, moe=moe)
        print(f"layer {l:2d} {'EXL3' if moe else 'native'}"
              f"{' + dense ' + ','.join(overlaid[l]) if overlaid.get(l) else ''}: {time.time() - t1:.0f}s", flush=True)
        if l < n_layers - 1:
            del blk
            torch.cuda.empty_cache()
    ref_path = os.path.join(a.out, "ref.safetensors")
    if a.mode == "ref":
        ids, lps, nll = [], [], []
        for i, lg in s.iter_logits(blk):                    # position t predicts token t+1
            sc = score[offs[i]: offs[i + 1]].bool().to(lg.device)
            pos = sc[1:].nonzero().squeeze(-1)              # predict scored token t+1 from position t
            lp = torch.log_softmax(lg[pos], -1)
            top = torch.topk(lp, TOPK, -1)
            ids.append(top.indices.int().cpu())
            lps.append(top.values.half().cpu())
            tgt = torch.tensor(seqs[i], device=lg.device)[1:][pos]
            nll.append((-lp.gather(-1, tgt[:, None]).squeeze(-1)).float().cpu())
        lens = torch.tensor([x.size(0) for x in ids], dtype=torch.int64)
        save_file({"ids": torch.cat(ids), "logprobs": torch.cat(lps), "nll": torch.cat(nll), "lens": lens}, ref_path)
        print(f"ref saved: {int(lens.sum()):,} scored positions, mean NLL {torch.cat(nll).mean().item():.4f}, "
              f"{time.time() - t0:.0f}s")
        return
    ref = load_file(ref_path)
    lens = ref["lens"].tolist()

    def kl_vs_ref(lq, pid, plp):                            # native top-128 support + one tail bucket
        P, Q = plp.exp(), lq.gather(-1, pid).exp()
        p_rest = (1 - P.sum(-1)).clamp_min(1e-12)
        q_rest = (1 - Q.sum(-1)).clamp_min(1e-12)
        return (P * (plp - Q.clamp_min(1e-30).log())).sum(-1) + p_rest * (p_rest.log() - q_rest.log())

    heads = None
    if a.head_mxfp8:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (dequant_mxfp8_to_bf16,
                                                                              mxfp8_e4m3_quantize)

        def mxfp8_head(w):
            q, sc = mxfp8_e4m3_quantize(w.to(torch.bfloat16), is_sf_swizzled_layout=False)
            return dequant_mxfp8_to_bf16(q, sc).float()
        heads = {"head_mxfp8": mxfp8_head}
    o, kls, agree, nll_q, by_dom, var = 0, [], [], [], {}, {}
    dump_ids, dump_lps = [], []
    for item in s.iter_logits(blk, heads):
        i, lg, extra = item if heads else (*item, {})
        k = lens[i]
        sc = score[offs[i]: offs[i + 1]].bool().to(lg.device)
        pos = sc[1:].nonzero().squeeze(-1)
        lq = torch.log_softmax(lg[pos].float(), -1)
        pid = ref["ids"][o: o + k].to(lg.device).long()
        plp = ref["logprobs"][o: o + k].to(lg.device).float()
        o += k
        kl = kl_vs_ref(lq, pid, plp)
        kls.append(kl.cpu())
        for name, lgx in extra.items():
            lqx = torch.log_softmax(lgx[pos].float(), -1)
            v = var.setdefault(name, {"kl": [], "agree": [], "kl_main_vs_variant": []})
            v["kl"].append(kl_vs_ref(lqx, pid, plp).cpu())
            v["agree"].append((lqx.argmax(-1) == pid[:, 0]).float().cpu())
            v["kl_main_vs_variant"].append((lq.exp() * (lq - lqx)).sum(-1).cpu())
        agree.append((lq.argmax(-1) == pid[:, 0]).float().cpu())
        if a.dump_top:
            top = torch.topk(lq, a.dump_top, -1)
            dump_ids.append(top.indices.int().cpu())
            dump_lps.append(top.values.float().cpu())
        tgt = torch.tensor(seqs[i], device=lg.device)[1:][pos]
        nll_q.append((-lq.gather(-1, tgt[:, None]).squeeze(-1)).cpu())
        dom = index[i]["domain"].split("_")[0]
        by_dom.setdefault(dom, []).append((kl.sum().item(), agree[-1].sum().item(), k))
    kl_all, ag_all = torch.cat(kls), torch.cat(agree)
    res = {"positions": int(kl_all.numel()), "kld_mean": kl_all.mean().item(), "kld_p99": kl_all.quantile(0.99).item(),
           "top1_agreement": ag_all.mean().item(), "nll_native": ref["nll"].mean().item(),
           "nll_ours": torch.cat(nll_q).mean().item(), "tensors": a.tensors, "exl3_ckpt": a.exl3_ckpt,
           "dense": a.dense, "overlaid": {str(k): v for k, v in overlaid.items()},
           "by_domain": {k: {"kld": sum(x[0] for x in v) / sum(x[2] for x in v),
                             "top1": sum(x[1] for x in v) / sum(x[2] for x in v),
                             "positions": sum(x[2] for x in v)} for k, v in sorted(by_dom.items())}}
    res["ppl_native"], res["ppl_ours"] = math.exp(res["nll_native"]), math.exp(res["nll_ours"])
    for name, v in var.items():
        res[name] = {"kld_mean": torch.cat(v["kl"]).mean().item(), "top1_agreement": torch.cat(v["agree"]).mean().item(),
                     "kl_main_vs_variant_mean": torch.cat(v["kl_main_vs_variant"]).mean().item()}
    if a.dump_top:
        save_file({"ids": torch.cat(dump_ids), "logprobs": torch.cat(dump_lps), "lens": ref["lens"]},
                  os.path.join(a.out, "quant_top.safetensors"))
    with open(os.path.join(a.out, "kld.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
