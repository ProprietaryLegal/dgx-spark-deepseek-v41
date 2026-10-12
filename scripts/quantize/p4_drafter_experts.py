# SPDX-License-Identifier: Apache-2.0
"""EXL3 encode of the DSpark drafter's routed experts (mtp.0-2: 128 experts each, top-3), from the official MXFP4
tensors, so the drafter's MoE can run on our EXL3 kernels instead of deep_gemm MXFP4 (~1.5 ms a round at 4.25 bpw;
K3 streams ~29% fewer bytes).

No routed-token capture exists for these experts (vLLM runs them on deep_gemm, which HCAP can't wrap), so each expert
gets pass 1's priors alone (p4_pass1.py quantize_layer):
  - gate / up: H_all, the layer's MoE-input X^T X. The drafter's shared expert reads the same hidden states the router
    and the routed experts do, so its captured gate_up_proj input Hessian (VLLM_EXL3_HCAP, Oct 10: 18.5-44K rows a
    layer) is that matrix.
  - down: the gated intermediate silu(clamp(x W1^T)) * clamp(x W3^T) of samples x ~ N(0, H_all / rows), pass 1's
    generic-token prior with Gaussian draws of the captured covariance in place of real tokens.
The router's gate scales a whole expert's rows by one factor, which EXL3's Hessian finalize ignores (it divides by the
count and damps relative to the diagonal), so it's left out. The drafter only proposes tokens: its precision moves
acceptance, never the target's output (set b decides).

  python3 p4_drafter_experts.py /ckpt_dir /hcap_dir /out.safetensors [--k 3] [--k-down 3] [--samples 8192]
/ckpt_dir needs model.safetensors.index.json and the shards holding mtp.*.ffn.experts.* (the official 44-46 suffice).
Writes mtp.<i>.ffn.experts.<e>.w{1,3,2}.{trellis,suh,svh,mul1} (the plugin's expert-bank names, mtp prefix: the DSpark
loader maps mtp.<i> to layer 40 + i) and <out>.json with each expert's K and exllamav3 proxy errors.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.exl3 import encode_expert, h_data, mxfp4_dequant, reconstruct  # noqa: E402

LAYERS, EXPERTS, HIDDEN, INTER, LIMIT = 3, 128, 5120, 2304, 10.0      # dspark_n_routed_experts, swiglu_limit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("hcap")
    ap.add_argument("out")
    ap.add_argument("--k", type=float, default=3)
    ap.add_argument("--k-down", type=float, default=None)
    ap.add_argument("--samples", type=int, default=8192)
    ap.add_argument("--layers", default="0-2")
    ap.add_argument("--experts", type=int, default=EXPERTS)          # a smaller count for a smoke test
    a = ap.parse_args()
    def kval(k):                                                     # exllamav3 wants an int K for whole bitrates
        return int(k) if float(k).is_integer() else float(k)

    a.k = kval(a.k)
    kd = a.k if a.k_down is None else kval(a.k_down)
    lo, hi = (int(x) for x in a.layers.split("-"))
    dev = torch.device("cuda")
    with open(os.path.join(a.ckpt, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    work = os.path.dirname(os.path.abspath(a.out))
    handles = {}

    def get(name):
        fn = wmap[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(a.ckpt, fn), "pt", device="cuda")
        return handles[fn].get_tensor(name)

    out, metrics = {}, {}
    t0 = time.time()
    for i in range(lo, hi + 1):
        hc = torch.load(os.path.join(a.hcap, f"model.layers.{40 + i}.ffn.shared_experts.gate_up_proj.rank0.pt"),
                        map_location=dev)
        H_all, rows = hc["H"].float(), int(hc["rows"])
        cov = H_all / max(rows, 1)
        cov = (cov + cov.t()) / 2
        jitter = 1e-6 * cov.diagonal().mean()
        Lc = torch.linalg.cholesky(cov + jitter * torch.eye(HIDDEN, device=dev))
        g = torch.Generator(device="cuda").manual_seed(4711 + i)
        xs = torch.randn(a.samples, HIDDEN, device=dev, generator=g) @ Lc.t()          # [S, 5120], covariance cov
        print(f"mtp.{i}: H_all from {rows} rows; samples {a.samples}, sample/captured diag ratio "
              f"{(xs.square().mean(0).mean() / cov.diagonal().mean()).item():.3f}", flush=True)
        for e in range(a.experts):
            base = f"mtp.{i}.ffn.experts.{e}"
            W1 = mxfp4_dequant(get(f"{base}.w1.weight"), get(f"{base}.w1.scale"))        # fp32 [2304, 5120]
            W3 = mxfp4_dequant(get(f"{base}.w3.weight"), get(f"{base}.w3.scale"))
            W2 = mxfp4_dequant(get(f"{base}.w2.weight"), get(f"{base}.w2.scale"))        # fp32 [5120, 2304]
            gate = (xs @ W1.t()).clamp(max=LIMIT)
            up = (xs @ W3.t()).clamp(-LIMIT, LIMIT)
            act = (F.silu(gate) * up).to(torch.bfloat16).float()                         # what w2 reads, bf16 like serving
            H_d = act.t() @ act
            tens, perr = encode_expert(W1, W3, W2, h_data(H_all.clone(), rows, f"d{i}.e{e}.gu"),
                                       h_data(H_d, a.samples, f"d{i}.e{e}.d"), a.k, kd,
                                       seed=(40 + i) * 1000 + e, work_dir=work)
            # output error of the whole expert on the samples (fp16 weights, the serving kernels' reconstruction)
            Wq = [reconstruct(tens["w1"], HIDDEN, INTER), reconstruct(tens["w3"], HIDDEN, INTER),
                  reconstruct(tens["w2"], INTER, HIDDEN)]
            y_ref = act @ W2.t()
            gq = (xs.half() @ Wq[0].t()).float().clamp(max=LIMIT)
            uq = (xs.half() @ Wq[1].t()).float().clamp(-LIMIT, LIMIT)
            y_q = (F.silu(gq) * uq).half() @ Wq[2].t()
            rel = ((y_q.float() - y_ref).norm() / y_ref.norm()).item()
            for proj in ("w1", "w3", "w2"):
                for k, v in tens[proj].items():
                    out[f"{base}.{proj}.{k}"] = v.contiguous().cpu()
            metrics[base] = {"k": a.k, "k_down": kd, "proxy": perr, "out_rel_err": rel}
            del W1, W3, W2, gate, up, act, H_d, tens, Wq, y_ref, gq, uq, y_q
            if (e + 1) % 32 == 0:
                errs = [m["out_rel_err"] for k_, m in metrics.items() if k_.startswith(f"mtp.{i}.")]
                print(f"  mtp.{i}: {e + 1}/{a.experts} experts, {time.time() - t0:.0f}s, output rel err mean "
                      f"{sum(errs) / len(errs):.4f} max {max(errs):.4f}", flush=True)
        del H_all, cov, Lc, xs
        torch.cuda.empty_cache()
    save_file(out, a.out + ".tmp")
    os.chmod(a.out + ".tmp", 0o644)
    os.replace(a.out + ".tmp", a.out)
    with open(a.out + ".json", "w") as f:
        json.dump({"k": a.k, "k_down": kd, "samples": a.samples, "experts": metrics}, f, indent=1)
    print(f"wrote {len(out)} tensors ({sum(v.numel() * v.element_size() for v in out.values()) / 2**20:.0f} MiB) to "
          f"{a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
