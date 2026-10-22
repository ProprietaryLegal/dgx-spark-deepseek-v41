# SPDX-License-Identifier: Apache-2.0
"""EXL3 encoding of DeepSeek-V4.1 routed experts with turboderp's exllamav3 (v1.6.0, MIT) quantizer, called through its
public functions (quantize_exl3 / quantize_exl3_batch / LinearEXL3); nothing from exllamav3 is copied here.

Orientation: exllamav3 works on (in_features, out_features) row-major weights, the transpose of nn.Linear. The
reference checkpoint stores routed experts as MXFP4: int8 [out, in/2] (two e2m1 per byte, LOW nibble first, measured on
GB10 against the reference fp4_gemm) + e8m0 scales [out, in/32].
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def mxfp4_dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """MXFP4 [out, in/2] (+ e8m0 [out, in/32]) -> fp32 [out, in], exactly the values the reference fp4_gemm uses."""
    b = w.view(torch.uint8)
    lut = _FP4.to(b.device)
    vals = torch.stack([lut[(b & 15).long()], lut[(b >> 4).long()]], -1).flatten(-2)
    sc = (s.view(torch.uint8).int() << 23).view(torch.float32)
    return vals * sc.repeat_interleave(32, -1)


def h_data(H_sum: torch.Tensor, count: int, key: str) -> dict:
    """exllamav3's H_data dict: H is a SUM of x^T x over `count` rows (finalize divides by count, then damps the
    diagonal by sigma_reg x its mean)."""
    return {"H": H_sum, "count": int(count), "finalized": False, "num_total": int(count) * H_sum.shape[0],
            "inf_nan": torch.zeros(2, dtype=torch.long, device=H_sum.device), "device": H_sum.device,
            "first_key": key}


def quant_args(K: float, seed: int, work_dir: str, codebook: str = "mul1", device: int = 0) -> dict:
    qa = {"K": K, "seed": seed, "devices": [device], "device_ratios": None, "apply_out_scales": None,
          "debug_dir": os.path.join(work_dir, "x3_debug")}
    qa[codebook] = True
    return qa


def encode_expert(w1, w3, w2, H_gu: dict, H_d: dict, K_gu: float, K_d: float, seed: int, work_dir: str):
    """w1/w3/w2: fp32 nn.Linear-orientation weights [out, in] (gate, up: [2304, 5120]; down: [5120, 2304]).
    Gate and up share one Hessian (same input), so they go through quantize_exl3_batch's shared-H path (one LDLQ pass
    over the concatenated columns). Returns ({"w1": tensors, "w3": ..., "w2": ...}, {"w1": proxy_err, ...})."""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, quantize_exl3_batch
    qa_g = quant_args(K_gu, seed, work_dir)
    qa_u = quant_args(K_gu, seed + 1, work_dir)
    res = quantize_exl3_batch([w1.t().contiguous(), w3.t().contiguous()], [H_gu, H_gu], [qa_g, qa_u])
    (pe1, t1), (pe3, t3) = res
    _, pe2, t2 = quantize_exl3(w2.t().contiguous(), H_d, quant_args(K_d, seed + 2, work_dir), return_weight_q=False)
    return {"w1": t1, "w3": t3, "w2": t2}, {"w1": float(pe1), "w3": float(pe3), "w2": float(pe2)}


def reconstruct(t: dict, in_features: int, out_features: int) -> torch.Tensor:
    """Stored EXL3 tensors -> fp16 [out, in] (nn.Linear orientation), the same reconstruction the serving kernels use."""
    from exllamav3.modules.quant.exl3 import LinearEXL3
    lin = LinearEXL3(None, in_features, out_features, suh=t["suh"], svh=t["svh"], trellis=t["trellis"],
                     mcg=t.get("mcg"), mul1=t.get("mul1"))
    return lin.get_weight_tensor().t()


# p4_kld.py --act-qdq: a bf16 -> bf16 round trip through the reference's activation quantizer (act_quant, FP8 with a
# power-of-two scale per 32 values), applied where the reference's linear() quantizes activations for its FP4 experts.
ACT_QDQ = None
ACT_QDQ_H = True       # --act-qdq-scope inputs: the experts' inputs only, not the routing-weighted intermediate


def expert_forward_dense(x: torch.Tensor, g: torch.Tensor, w1, w3, w2, limit: float) -> torch.Tensor:
    """Reference Expert.forward semantics (swiglu clamps, routing weight applied before w2) with dense fp16 weights and
    fp16 activations, i.e. what the EXL3 serving kernels compute. x [n, 5120], g [n] -> [n, 5120] fp32. With ACT_QDQ
    set, the expert's input and the routing-weighted intermediate (cast to bf16 first, as the reference does) take the
    reference's activation round trip before their GEMMs."""
    xh = (ACT_QDQ(x.to(torch.bfloat16)) if ACT_QDQ else x).to(torch.float16)
    gate = F.linear(xh, w1).float()
    up = F.linear(xh, w3).float()
    if limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    h = F.silu(gate) * up * g.unsqueeze(-1)
    h = (ACT_QDQ(h.to(torch.bfloat16)) if ACT_QDQ and ACT_QDQ_H else h).to(torch.float16)
    return F.linear(h, w2).float()


def quant_moe(l: int, tensors: dict, limit: float):
    """A Streamer moe(x, block) that runs layer l's routed experts from EXL3 tensors (keys
    layers.{l}.ffn.experts.{e}.w{1,2,3}.{trellis,suh,svh,mul1|mcg}) and the native shared expert."""
    by = {}                                                   # (expert, proj) -> {piece: tensor}, built once
    for k, v in tensors.items():
        parts = k.split(".")                                  # layers.L.ffn.experts.E.wN.piece
        if len(parts) == 7 and parts[1] == str(l) and parts[3] == "experts":
            by.setdefault((int(parts[4]), parts[5]), {})[parts[6]] = v
    shapes = {"w1": (5120, 2304), "w3": (5120, 2304), "w2": (2304, 5120)}

    def moe(x, b):
        w, idx = b.ffn.gate(x, None)
        y = b.ffn.shared_experts(x).float()
        for e in range(b.ffn.n_routed_experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            q = {pj: reconstruct(by[(e, pj)], *shapes[pj]) for pj in shapes}
            y.index_add_(0, tok, expert_forward_dense(x[tok], w[tok, slot].float(), q["w1"], q["w3"], q["w2"], limit))
        return y.to(torch.bfloat16)
    return moe
