# SPDX-License-Identifier: Apache-2.0
"""MoE router GEMM at decode sizes on family-120 Blackwell (csrc/exl3_router.cu), in place of vLLM's GateLinear.

vLLM's GateLinear tiers skip GB10 (sm_121): its gate [384, 5120] bf16 -> fp32 runs through cuBLAS with a split-K reduce,
~30 us + a ~5 us reduce kernel per layer for 3.75 MiB (k=3 profile, Oct 10 02:27), about half of DRAM rate, 43 times a
step (40 layers + the drafter's 3). exl3_router streams the rows at DRAM rate with one warp an expert, starts reading
the weights before the PDL wait, and sums in a fixed order (deterministic; fp32 accumulation like cuBLAS, not its
order). Up to 8 rows (decode with DSpark verify rows); larger batches and anything else keep vLLM's tiers.

Installed as an out-of-tree replacement of GateLinear (vLLM instantiates the registered class for every gate), only
when EXL3_ROUTER=1 (default off until measured on the serve).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("EXL3_ROUTER", "0") == "1"
MAX_ROWS = 8
SMEM_BYTES = 96 * 1024             # x is staged whole: rows * K * 2 bytes

_ext = None
_installed = False


def ext():
    global _ext
    if _ext is None:
        from .jit import load
        _ext = load("exl3_router", "EXL3_ROUTER_FLAGS")
    return _ext


def install() -> None:
    """Register the replacement GateLinear (idempotent; vLLM may load plugins several times per process)."""
    global _installed
    if _installed or not ENABLED:
        return
    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    if "GateLinear" in op_registry_oot:
        _installed = True
        return

    class Exl3GateLinear(GateLinear):
        """GateLinear with the decode-size router GEMM on sm_12x; every other case is GateLinear's own forward."""

        def _exl3_ok(self, x: torch.Tensor) -> bool:
            w = self.weight
            if getattr(self, "_exl3_static", None) is None:
                self._exl3_static = (
                    w.is_cuda and torch.cuda.get_device_capability(w.device)[0] == 12
                    and self.is_unquantized and self.bias is None and w.dtype == torch.bfloat16
                    and w.dim() == 2 and w.is_contiguous() and w.shape[1] % 8 == 0 and w.data_ptr() % 16 == 0)
                if self._exl3_static:
                    logger.info_once("EXL3 router: gate GEMMs of up to %d rows on exl3_router (sm_12x)", MAX_ROWS)
            return (self._exl3_static and self.out_dtype == torch.float32 and x.dtype == torch.bfloat16
                    and x.dim() == 2 and 0 < x.shape[0] <= MAX_ROWS and x.shape[1] == w.shape[1]
                    and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
                    and x.shape[0] * x.shape[1] * 2 <= SMEM_BYTES)

        def forward(self, x: torch.Tensor):
            if self._exl3_ok(x):
                out = torch.empty((x.shape[0], self.weight.shape[0]), dtype=torch.float32, device=x.device)
                ext().router_logits(x, self.weight, out)
                return self._return(out)
            return super().forward(x)

    PluggableLayer.register_oot(name="GateLinear")(Exl3GateLinear)
    _installed = True
    logger.info("EXL3 router: GateLinear replaced (EXL3_ROUTER=1)")
