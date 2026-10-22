# SPDX-License-Identifier: Apache-2.0
"""VLLM_DSV41_HC_BF16=1 (default 0): DeepSeek-V4.1's mHC mixing GEMM reads a bf16 copy of its weight.

vllm.utils.deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_split) computes, split over K, out = x.float() @ fn.T
and sqrsum = x.float().square().sum(-1) for every layer's attention and FFN mixing (x the bf16 residual streams
[tokens, 4 x 5120], fn fp32 [24, 20480]). The weights stream from DRAM at every call: ~2 MB each, ~170 MB a decode round
a rank, always beside a weight-streaming kernel (the Oct 21 prose trace: the DeepGEMM TF32 kernel at ~98 GB/s, 81 calls a
round, overlapping smoe / mrow / l2pf the whole time), so their bytes come out of the weights' bandwidth. Here the same
split layout (out [num_split, tokens, 24], sqrsum [num_split, tokens]) comes from one Triton launch that reads a bf16 copy
of fn (made once per weight, outside graph capture) and multiplies bf16 x bf16 into fp32 on the tensor cores: half the
bytes. Numerics: the weights at bf16 (8 significant bits) where TF32 keeps 11, so outputs move in the rounding class;
p4_kld.py --hc-bf16 measures what that costs against native 1017.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_DSV41_HC_BF16", "0") == "1"
BLOCK_N, BLOCK_K = 32, 128              # outputs (24 used) and K a step; rows a program: 16 at decode, 64 above

_copies: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}   # (fn.data_ptr(), shape) -> bf16 copy
_orig = None
_fallbacks = 0

try:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _hc_gemm_kernel(x_ptr, x_stride, w_ptr, out_ptr, sq_ptr, n, K, k_per_split, n_out,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        s = tl.program_id(0)
        rows = tl.program_id(1) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        rmask = rows < n
        cmask = cols < n_out
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        sq = tl.zeros((BM,), dtype=tl.float32)
        k0 = s * k_per_split
        for kk in range(0, k_per_split, BK):
            ks = k0 + kk + tl.arange(0, BK)
            xv = tl.load(x_ptr + rows[:, None].to(tl.int64) * x_stride + ks[None, :], mask=rmask[:, None], other=0.0)
            wv = tl.load(w_ptr + cols[:, None].to(tl.int64) * K + ks[None, :], mask=cmask[:, None], other=0.0)
            acc = tl.dot(xv, tl.trans(wv), acc)             # bf16 x bf16, exact products, fp32 sums
            xf = xv.to(tl.float32)
            sq += tl.sum(xf * xf, axis=1)
        base = s.to(tl.int64) * n
        tl.store(out_ptr + (base + rows[:, None]) * n_out + cols[None, :], acc, mask=rmask[:, None] & cmask[None, :])
        tl.store(sq_ptr + base + rows, sq, mask=rmask)
except ImportError:  # pragma: no cover (no Triton: the hook never installs)
    _hc_gemm_kernel = None


def _weight(fn: torch.Tensor) -> torch.Tensor | None:
    key = (fn.data_ptr(), tuple(fn.shape))
    w = _copies.get(key)
    if w is None and not torch.cuda.is_current_stream_capturing():
        w = fn.detach().to(torch.bfloat16).contiguous()
        _copies[key] = w
    return w


def supported(x: torch.Tensor, fn: torch.Tensor, out: torch.Tensor, sqrsum: torch.Tensor, num_split: int) -> bool:
    if _hc_gemm_kernel is None or x.dtype != torch.bfloat16 or x.dim() != 2 or x.stride(1) != 1:
        return False
    n, K = x.shape
    return (fn.dim() == 2 and fn.shape[1] == K and fn.shape[0] <= BLOCK_N and num_split >= 1
            and K % num_split == 0 and (K // num_split) % BLOCK_K == 0
            and out.dtype == torch.float32 and out.is_contiguous() and tuple(out.shape) == (num_split, n, fn.shape[0])
            and sqrsum.dtype == torch.float32 and sqrsum.is_contiguous() and tuple(sqrsum.shape) == (num_split, n))


def hc_prenorm_gemm(x: torch.Tensor, fn: torch.Tensor, out: torch.Tensor, sqrsum: torch.Tensor,
                    num_split: int) -> torch.Tensor:
    """tf32_hc_prenorm_gemm's contract on a bf16 copy of fn; anything outside it goes to the original."""
    global _fallbacks
    w = _weight(fn) if supported(x, fn, out, sqrsum, num_split) else None
    if w is None:
        if supported(x, fn, out, sqrsum, num_split):
            _fallbacks += 1
            if _fallbacks == 1:
                logger.warning("VLLM_DSV41_HC_BF16: a weight first seen inside a graph capture keeps the TF32 GEMM")
        return _orig(x, fn, out, sqrsum, num_split)
    n, K = x.shape
    if n:
        bm = 16 if n <= 16 else 64
        _hc_gemm_kernel[(num_split, triton.cdiv(n, bm))](
            x, x.stride(0), w, out, sqrsum, n, K, K // num_split, fn.shape[0],
            BM=bm, BN=BLOCK_N, BK=BLOCK_K, num_warps=4,
        )
    return out


def install() -> None:
    global _orig
    if not ENABLED or _orig is not None or _hc_gemm_kernel is None:
        return
    try:
        import vllm.utils.deep_gemm as dg
    except ImportError as e:
        logger.warning("VLLM_DSV41_HC_BF16: vllm.utils.deep_gemm not importable (%s): off", e)
        return
    _orig = dg.tf32_hc_prenorm_gemm
    dg.tf32_hc_prenorm_gemm = hc_prenorm_gemm
    logger.info("VLLM_DSV41_HC_BF16: the mHC mixing GEMM reads bf16 copies of its weights (one Triton launch a call)")
