# SPDX-License-Identifier: Apache-2.0
"""VLLM_DSV41_WOA_QDQ=1 (default 0): the FP8 round trip DeepSeek's reference puts on the attention output `o` before wo_a,
on the serve too, so it computes the function our KLD harness (scripts/quantize/p4_kld.py, which runs the reference
file) measured.

Semantics, from deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170, inference/:
  model.py:786-789   Attention.forward: o.view(bsz, seqlen, n_local_groups, -1), then act_quant(o, fp8_block_size,
                     scale_fmt, scale_dtype, True) on every call (prefill and decode alike), then the bf16 wo_a einsum.
  model.py:1072-1075 DSparkAttention.forward (the DSpark drafter): the same on its o.
  model.py:27-30     fp8_block_size = 32, scale_fmt = "ue8m0", scale_dtype = float8_e8m0fnu.
  kernel.py:98-122   act_quant(..., inplace=True): round_scale = scale_fmt is not None (117), the result is copied back
                     into o (122), so o stays bf16 and only its values change.
  kernel.py:41-95    act_quant_kernel, per row and 32-value group along the last dim (each head group's 4096 values),
                     in fp32 on the bf16 input:
                       amax = max |x| over the group, at least 1e-4 (74-76);
                       s = 2^ceil(log2(amax * fl32(1/448))), the exponent read off the IEEE bits (78, fast_round_scale
                           22-38: exponent field - 127, + 1 if the mantissa is not zero), so a power of two in
                           [2^-22, 2^120] for bf16 input;
                       out = bf16(fp32(e4m3(clamp(x / s, -448, 448))) * s) (85): e4m3 round to nearest even, saturating.
                     x / s is exact (s is a power of two) and |x / s| <= 448 always, so the clamp never binds; the bf16
                     store is exact too, except past bf16's range: |x| >= 1.9375 * 2^127 rounds to 256 * 2^120 = inf, in
                     the reference as here.
Checked bit for bit against DeepSeek's own TileLang kernel (kernel.py, tilelang 0.1.15, RTX 5090): random input at
scales 1e-30 .. 1e36, every finite bf16 value, signed zeros, subnormals, groups at the 448 * 2^k scale boundaries,
e4m3 ties, values near the bf16 maximum (inf out) and inf input (NaN out at the same places).

Where it runs (patch 0006's grouped_bmm hook in deep_gemm_fp8_o_proj, after the inverse RoPE):
  - EXL3 wo_a (dense.py): at decode sizes (<= 16 rows) exl3_mrow's wo_a build (-DMROW_QDQ=1, mrow.py) rounds each
    32-value group as it loads the input (8 lanes a group, no extra launch); above that, `qdq` below writes the rounded
    o straight as the fp16 the exllamav3 path reads (one pass, in place of the old .to(float16)). Same bits either way
    (tests/test_woa_qdq_gpu.py).
  - FP8 wo_a (the DSpark drafter's; a backbone wo_a without an EXL3 K): `hook_fp8` adds `install_fp8` to the
    checkpoint method's loading, which gives the layer a grouped_bmm = `qdq` + the SM12x Marlin bmm (patch 0004) that
    o_proj.py would have run.
"""

from __future__ import annotations

import os
import re

import torch
import triton
import triton.language as tl

try:
    from vllm.logger import init_logger
except ImportError:                    # the kernel tests run without vLLM
    from logging import getLogger as init_logger

logger = init_logger(__name__)

GROUP = 32                             # values per ue8m0 scale (model.py:27 fp8_block_size)
FP8_MAX = 448.0                        # e4m3fn
AMAX_MIN = 1e-4                        # kernel.py:76
_WO_A = re.compile(r"(?:^|\.)attn\.wo_a$")


def enabled() -> bool:
    return os.environ.get("VLLM_DSV41_WOA_QDQ", "0") == "1"


def is_wo_a(prefix: str) -> bool:
    return _WO_A.search(prefix) is not None


def act_quant_ref(x: torch.Tensor) -> torch.Tensor:
    """act_quant(x, 32, "ue8m0", float8_e8m0fnu, inplace=True) in plain torch, step by step as above (tests)."""
    if x.dtype != torch.bfloat16 or x.shape[-1] % GROUP:
        raise ValueError(f"act_quant_ref: bf16 with a last dim divisible by {GROUP}, got {x.dtype} {tuple(x.shape)}")
    xf = x.float().unflatten(-1, (-1, GROUP))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(AMAX_MIN)
    t = (amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x.device)).view(torch.int32)
    e = ((t >> 23) & 0xFF) - 127 + ((t & 0x7FFFFF) != 0).to(torch.int32)
    s = ((e + 127) << 23).view(torch.float32)
    q = (xf / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * s).to(torch.bfloat16).flatten(-2)


@triton.jit(do_not_specialize=["n_groups"])
def _qdq_kernel(x_ptr, y_ptr, n_groups, BLOCK_G: tl.constexpr):
    g = tl.program_id(0).to(tl.int64) * BLOCK_G + tl.arange(0, BLOCK_G)
    offs = g[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = g[:, None] < n_groups
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    t = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    e = ((t >> 23) & 0xFF) - 127 + ((t & 0x7FFFFF) != 0).to(tl.int32)
    s = ((e + 127) << 23).to(tl.float32, bitcast=True)
    # x / s as x * 2^-e: the same correctly rounded value (2^-e is a normal float for e in [-22, 120]); an inf amax
    # (inf input) takes 0, which gives the reference's NaN group (0 * inf)
    inv = tl.where(e < 128, ((127 - e) << 23).to(tl.float32, bitcast=True), 0.0)
    q = (x * inv[:, None]).to(tl.float8e4nv)                   # round to nearest even, satfinite (|x / s| <= 448)
    y = (q.to(tl.float32) * s[:, None]).to(tl.bfloat16)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def _dense(x: torch.Tensor) -> bool:
    """Non-overlapping and dense: the elements fill [data_ptr, data_ptr + numel) exactly, in some dim order."""
    want = 1
    for d in sorted(range(x.dim()), key=x.stride):
        if x.size(d) == 1:
            continue
        if x.stride(d) != want:
            return False
        want *= x.size(d)
    return True


def qdq(x: torch.Tensor, dtype: torch.dtype | None = None, block_g: int = 64) -> torch.Tensor:
    """The round trip of bf16 `x` [..., D] (D % 32 == 0, groups along the last dim) as a new tensor of `dtype`
    (default bf16; float16 = the rounded bf16 values cast to fp16, as `.to(float16)` would). A dense input (e.g.
    o_rot, the [T, G, D] view of the inverse-RoPE kernel's [G, T, D] buffer) keeps its layout: one flat pass, and each
    group's [T, D] slice of the output is still contiguous. Anything else is made contiguous first."""
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.shape[-1] % GROUP:
        raise ValueError(f"woa qdq: CUDA bf16 with a last dim divisible by {GROUP}, got {x.dtype} {tuple(x.shape)} "
                         f"on {x.device}")
    if x.stride(-1) != 1 or not _dense(x):
        x = x.contiguous()
    y = torch.empty_strided(x.shape, x.stride(), dtype=dtype or x.dtype, device=x.device)
    n = x.numel() // GROUP
    if n:
        _qdq_kernel[(triton.cdiv(n, block_g),)](x, y, n, BLOCK_G=block_g)
    return y


def install_fp8(layer, bmm=None) -> bool:
    """An FP8 wo_a on the SM12x Marlin bmm (patch 0004 leaves `sm12x_bmm_groups` / `sm12x_bmm_nk` on it): set its
    `grouped_bmm` (patch 0006 takes that hook before the Marlin branch) to the round trip + the same Marlin bmm, the
    weights read at call time. False, and nothing set, on any other path. `bmm`: the bmm to call (tests)."""
    groups = getattr(layer, "sm12x_bmm_groups", None)
    if not groups:
        return False
    if bmm is None:
        from vllm.model_executor.kernels.linear.mxfp8.sm12x_bmm import sm12x_marlin_bmm as bmm
    n, k = layer.sm12x_bmm_nk

    def grouped_bmm(o_rot: torch.Tensor) -> torch.Tensor:              # [T, G_loc, k] -> [T, G_loc, n]
        return bmm(qdq(o_rot), layer.weight, layer.weight_scale, groups, n, k)
    layer.grouped_bmm = grouped_bmm
    return True


def hook_fp8(method, prefix: str):
    """A wo_a the checkpoint's FP8 method serves: `install_fp8` runs after the method's own process_weights_after_loading.
    The step goes on the method instance, not a wrapper class: vLLM picks a linear's weight loader by the method's class
    name (WEIGHT_LOADER_V2_SUPPORTED, model_executor/layers/linear.py:49 at 6e517b15), which a wrapper would change."""
    inner = method.process_weights_after_loading

    def process_weights_after_loading(layer):
        inner(layer)
        if install_fp8(layer):
            logger.info("VLLM_DSV41_WOA_QDQ: %s rounds o to FP8 (32 values a ue8m0 scale) before its Marlin bmm", prefix)
        else:
            logger.warning("VLLM_DSV41_WOA_QDQ: %s is not on the SM12x Marlin bmm (patch 0004), its input is not "
                           "rounded", prefix)
    method.process_weights_after_loading = process_weights_after_loading
    return method
