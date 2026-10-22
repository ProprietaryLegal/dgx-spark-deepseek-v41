# SPDX-License-Identifier: Apache-2.0
"""Decode-size bf16 projections on exl3_router's row GEMM (csrc/exl3_router.cu `rowmm`), for the V4.1 attention inputs
cuBLAS runs at a fraction of DRAM rate on GB10 (sm_121). From an Oct 21 prose profile (rank 0):
  - the compressor's fused wkv / wgate [1024, 5120] -> fp32 (layers 2, 8, 14, 20): 87-97 us on cuBLAS's split-K
    path for 10.5 MB;
  - the indexer's weights_proj [32, 5120] -> bf16 (the 8 index layers): 40-55 us for 0.33 MB;
  - the indexer's wk [128, 512] -> bf16 on the compressor latent (layers 2, 8, 14, 20): ~8 us.
The first two run on aux streams beside wq_a + wkv (31-42 us), and the default stream joins them before wq_b, so the
slowest of the three is the layer's: index layers ran ~47 us and compressor layers ~105 us longer than plain ones,
~0.6 ms a round. exl3_router streams the rows at DRAM rate (one warp a row, the weights' first loads before the PDL
wait) for up to 8 rows (decode with DSpark's verify rows); any other shape keeps vLLM's path. The sums are fp32 in a
fixed order: deterministic, not bit-equal to cuBLAS (a rounding-class change), and a bf16 output is rounded to nearest
even as F.linear's epilogue does.

EXL3_ROWMM=1 turns it on (default off until measured on the serve); patch 0029 calls `ok` / `mm` from the V4.1
attention.
"""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("EXL3_ROWMM", "0") == "1"
MAX_ROWS = 8
SMEM_BYTES = 96 * 1024             # x is staged whole: rows * K * 2 bytes

_sm12 = {}                         # device index -> compute capability major is 12
_logged = False


def _on_sm12(dev: torch.device) -> bool:
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    if i not in _sm12:
        _sm12[i] = torch.cuda.get_device_capability(i)[0] == 12
    return _sm12[i]


def ok(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether x @ w.T can run on rowmm: on, sm_12x, bf16 x [M <= 8, K] with contiguous 16-byte aligned rows, bf16 w
    [N, K] contiguous and aligned, K a multiple of 8, x fitting shared memory."""
    return (ENABLED and x.is_cuda and w.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
            and x.dim() == 2 and w.dim() == 2 and 0 < x.shape[0] <= MAX_ROWS and x.shape[1] == w.shape[1]
            and x.shape[1] % 8 == 0 and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
            and w.is_contiguous() and w.data_ptr() % 16 == 0 and x.shape[0] * x.shape[1] * 2 <= SMEM_BYTES
            and _on_sm12(x.device))


def mm(x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """x @ w.T as out_dtype (fp32 or bf16) on rowmm; the caller checked `ok`."""
    global _logged
    from .router import ext
    out = torch.empty((x.shape[0], w.shape[0]), dtype=out_dtype, device=x.device)
    ext().rowmm(x, w, out)
    if not _logged:
        _logged = True
        from vllm.logger import init_logger
        init_logger(__name__).info("EXL3 rowmm: decode-size V4.1 attention input projections on exl3_router "
                                   "(EXL3_ROWMM=1, up to %d rows)", MAX_ROWS)
    return out
