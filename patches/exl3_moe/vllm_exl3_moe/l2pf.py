# SPDX-License-Identifier: Apache-2.0
"""Paced L2 prefetch of the next dense EXL3 weights on a side stream, at decode sizes (csrc/exl3_l2pf.cu).

Three points a layer, each pulling the weights of the dense modules that run after the next DRAM-idle stretch into L2,
so their exl3_mrow launches stream from L2 instead of DRAM:
  attn  after wq_b       -> this layer's wo_a and wo_b      (during the indexer and attention)
  moe   after wo_b       -> this layer's shared expert       (during the attention all-reduce, the router, mHC)
  next  after the MoE    -> the next layer's wq_a+wkv, wq_b  (during the MoE all-reduce and mHC)
A fork waits for the current stream, then a one-block kernel issues the ranges in their reader's order in waves paced
under DRAM's rate; at most one prefetch is in flight (each fork first joins the previous one), and the last layer's MoE
joins without forking, so a captured forward ends with no side-stream work. A prefetch writes nothing: outputs keep
their bits. The design and the kernel are TensorFold's (csrc/NOTICE.tensorfold); the points are ours.

EXL3_L2PF: comma list of points to run (attn,moe,next) or 0 (default: off until measured on the serve);
EXL3_L2PF_MB: bytes a fork (16); EXL3_L2PF_RATE_GBPS: pacing (230, 0 = issue all at once);
EXL3_L2PF_DELAY_NS: wait before issuing (3000).
EXL3_L2PF_HC=1 also pulls each sublayer boundary's fp32 mHC mix weights (24 x 20480, 1.9 MiB) and the router gate
(bf16 384 x 5120, 3.75 MiB), first in their fork: `moe` takes this layer's hc_ffn_fn + gate (read right after the
attention all-reduce), `next` the next layer's hc_attn_fn. TensorFold prefetches its mix weights and router the same
way. They are registered by patch 0017 (register_decoder_layers, after the model's weights load).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

CHUNK = 8 << 10
MAX_ROWS = 16                      # decode sizes only (the dense modules' exl3_mrow range)
POINTS = {p for p in os.environ.get("EXL3_L2PF", "0").replace(" ", "").split(",") if p and p != "0"}
BUDGET = int(float(os.environ.get("EXL3_L2PF_MB") or 16) * (1 << 20))
RATE_GBPS = float(os.environ.get("EXL3_L2PF_RATE_GBPS") or 230)
DELAY_NS = int(os.environ.get("EXL3_L2PF_DELAY_NS") or 3000)

HC = os.environ.get("EXL3_L2PF_HC", "0") == "1"

# What each point prefetches: (layer offset, module) in the order the forward reads them.
PLAN = {
    "attn": ((0, "attn.wo_a"), (0, "attn.wo_b")),
    "moe": ((0, "hc_ffn_fn"), (0, "ffn.gate")) * HC + ((0, "ffn.shared_experts.gate_up_proj"),
                                                         (0, "ffn.shared_experts.down_proj")),
    "next": ((1, "hc_attn_fn"),) * HC + ((1, "attn.fused_wqa_wkv"), (1, "attn.wq_b")),
}

_ext = None
_reg: dict[tuple[int, str], list[torch.Tensor]] = {}
_last_layer = -1
_stream: torch.cuda.Stream | None = None
_pending = False
_announced = False


def enabled() -> bool:
    return bool(POINTS)


def ext():
    global _ext
    if _ext is None:
        from .jit import load
        _ext = load("exl3_l2pf", "EXL3_L2PF_FLAGS")
    return _ext


def register(layer: int, module: str, tensors) -> None:
    """The weight tensors a dense module streams, in the order its kernel reads them."""
    global _last_layer
    if not enabled():
        return
    _reg[(layer, module)] = [t for t in tensors if t is not None and t.is_cuda and t.is_contiguous()]
    _last_layer = max(_last_layer, layer)


def register_decoder_layers(model) -> None:
    """The mHC mix weights and router gate of every backbone layer of a DeepseekV4Model (EXL3_L2PF_HC=1; called by
    patch 0017 once the weights are loaded)."""
    if not enabled() or not HC:
        return
    n = 0
    for i in range(getattr(model, "start_layer", 0), getattr(model, "end_layer", len(model.layers))):
        layer = model.layers[i]
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            t = getattr(layer, name, None)
            if t is not None:
                register(i, name, [t.data])
                n += 1
        gate = getattr(getattr(layer, "ffn", None), "gate", None)
        w = getattr(gate, "weight", None)
        if w is not None:
            register(i, "ffn.gate", [w.data])
            n += 1
    logger.info("EXL3 L2 prefetch: %d mHC mix / router gate tensors registered", n)


def _ranges(layer: int, point: str) -> tuple[list[int], list[int]]:
    ptrs, sizes, left = [], [], BUDGET
    for off, module in PLAN[point]:
        for t in _reg.get((layer + off, module), ()):
            n = min(t.numel() * t.element_size(), left) // 16 * 16
            p = t.data_ptr()
            if n <= 0 or p % 16:
                continue
            ptrs.append(p)
            sizes.append(n)
            left -= n
            if left < 16:
                break
    cap = 24                                   # PMAX in csrc/exl3_l2pf.cu
    return ptrs[:cap], sizes[:cap]


def join() -> None:
    global _pending
    if _pending:
        torch.cuda.current_stream().wait_stream(_stream)
        _pending = False


def point(name: str, layer: int, rows: int) -> None:
    """Prefetch point `name` of `layer` (a decode forward with `rows` rows)."""
    global _stream, _pending, _announced
    if not POINTS or not 0 < rows <= MAX_ROWS:
        return
    last = name == "next" and layer >= _last_layer
    if name not in POINTS and not last:
        return
    join()
    if last or name not in POINTS:
        return                                 # the last layer's MoE: the forward ends with nothing in flight
    ptrs, sizes = _ranges(layer, name)
    if not ptrs:
        return
    cur = torch.cuda.current_stream()
    if _stream is None:
        _stream = torch.cuda.Stream()
    _stream.wait_stream(cur)
    with torch.cuda.stream(_stream):
        wave_ns = int(128 * CHUNK / RATE_GBPS) if RATE_GBPS > 0 else 0
        ext().l2pf(ptrs, sizes, CHUNK, 4, DELAY_NS, wave_ns)
    _pending = True
    if not _announced:
        _announced = True
        logger.info("EXL3 L2 prefetch on: points %s, %d MiB a fork, %.0f GB/s pacing, %d modules registered "
                    "(layers 0-%d)", ",".join(sorted(POINTS)), BUDGET >> 20, RATE_GBPS, len(_reg), _last_layer)
