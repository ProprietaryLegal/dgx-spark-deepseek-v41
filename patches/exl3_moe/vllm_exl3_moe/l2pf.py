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

EXL3_L2PF_DRAFTER=1 (default off until measured on the serve): the DSpark draft layers (40-42) get the same three points
as a chain of their own: their EXL3 modules register (prefetch="drafter" in dense.py, config.py), `attn` pulls a draft
layer's wo_b (its wo_a is FP8, not registered), `moe` its shared expert, `next` the next draft layer's wq_a + wkv and
wq_b, and the last draft layer only joins, so the draft forward also ends with nothing in flight. The backbone's chain
and its last layer (the LM head's prefetch) are unchanged: draft layers never move `_last_layer`. On the Oct 21 prose
trace the three draft layers ran ~2 ms a round with no prefetch, their weight kernels (wq_b, wo_b, wo_a, the shared
expert) behind mHC, the router and the all-reduces.

EXL3_L2PF_ORDER=1 (default 0 until measured on the serve; the order TensorFold's rounds use):
  - the critical kernel's weights first: `moe` pulls the router gate before hc_ffn_fn and `next` wq_a + wkv before
    hc_attn_fn. Under patch 0014 (VLLM_DSV41_MHC_OVERLAP=1) a sublayer's mix statistics run on a side stream and are
    needed only at its post, while the gate GEMV and wq_a + wkv start the sublayer on the main stream: with the mix
    first, the stretch before them ran out before their bytes landed (router 3.75 MiB at ~190 GB/s, wq_a + wkv 28 us
    against TensorFold's 18.8 us from L2). Same bytes per fork, another order.
  - the `moe` fork capped at EXL3_L2PF_MOE_CAP_MB (8 with the flag, else no cap; an explicit value applies either
    way): gate, mix and the shared expert's first ~2 MiB. TensorFold measured 16 MB there 0.2-0.4 ms slower a forward
    than 8 (the routed experts' stream evicted the shared expert before its kernels read it; model.py L2_MOE_MB).
  - the LM head's first EXL3_L2PF_HEAD_MB (4 with the flag, else 0; an explicit value applies either way) after the
    last layer's MoE, during its all-reduce, mHC post, collapse and final norm, as TensorFold's `next` fork does after
    its last layer (rounds.py _l2_next). The head runs in compute_logits, outside the captured forward, so the fork is
    joined by a forward hook on the model's final norm that register_decoder_layers installs (patch 0017); without
    that hook, or without an EXL3 head (head.py registers its trellis; a bf16 or MXFP8 head has none), the last
    layer only joins. The bytes are the head trellis's first in memory, its first k-rows of every column: exl3_mrow's
    head reads them in its first k-slice, exllamav3's (the shipping head) walks one column block at a time from ~one
    start per SM, so most of them are read late there.
"""

from __future__ import annotations

import os

import torch

try:
    from vllm.logger import init_logger
except ImportError:                    # the kernel tests run without vLLM
    from logging import getLogger as init_logger

logger = init_logger(__name__)

CHUNK = 8 << 10
MAX_ROWS = 16                      # decode sizes only (the dense modules' exl3_mrow range)
POINTS = {p for p in os.environ.get("EXL3_L2PF", "0").replace(" ", "").split(",") if p and p != "0"}
BUDGET = int(float(os.environ.get("EXL3_L2PF_MB") or 16) * (1 << 20))
RATE_GBPS = float(os.environ.get("EXL3_L2PF_RATE_GBPS") or 230)
DELAY_NS = int(os.environ.get("EXL3_L2PF_DELAY_NS") or 3000)

HC = os.environ.get("EXL3_L2PF_HC", "0") == "1"
ORDER = os.environ.get("EXL3_L2PF_ORDER", "0") == "1"
DRAFTER = os.environ.get("EXL3_L2PF_DRAFTER", "0") == "1"


def _mib(env: str, default: float) -> int:
    v = os.environ.get(env)
    return int(float(v if v else default) * (1 << 20))


MOE_CAP = _mib("EXL3_L2PF_MOE_CAP_MB", 8 if ORDER else 0)        # 0: the `moe` fork takes BUDGET like the others
HEAD_BUDGET = _mib("EXL3_L2PF_HEAD_MB", 4 if ORDER else 0)       # 0: no head prefetch

# What each point prefetches: (layer offset, module) in the order the forward reads them.
_SHARED = ((0, "ffn.shared_experts.gate_up_proj"), (0, "ffn.shared_experts.down_proj"))
PLAN = {
    "attn": ((0, "attn.wo_a"), (0, "attn.wo_b")),
    "moe": ((0, "hc_ffn_fn"), (0, "ffn.gate")) * HC + _SHARED,
    "next": ((1, "hc_attn_fn"),) * HC + ((1, "attn.fused_wqa_wkv"), (1, "attn.wq_b")),
}
if ORDER:            # the main stream's first reader before the mix weights the side stream reads (patch 0014)
    PLAN["moe"] = ((0, "ffn.gate"), (0, "hc_ffn_fn")) * HC + _SHARED
    PLAN["next"] = ((1, "attn.fused_wqa_wkv"),) + ((1, "hc_attn_fn"),) * HC + ((1, "attn.wq_b"),)

_ext = None
_reg: dict[tuple[int, str], list[torch.Tensor]] = {}
_head: list[torch.Tensor] = []         # the LM head's weights (register_head)
_head_hooked = False                   # the final norm joins (register_decoder_layers): the head may be forked
_last_layer = -1                       # the backbone's last registered layer
_drafter: set[int] = set()             # the draft layers' chain (EXL3_L2PF_DRAFTER)
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


def register(layer: int, module: str, tensors, drafter: bool = False) -> None:
    """The weight tensors a dense module streams, in the order its kernel reads them. `drafter`: a DSpark draft layer's
    module, in the draft chain (EXL3_L2PF_DRAFTER), never the backbone's last layer."""
    global _last_layer
    if not enabled() or (drafter and not DRAFTER):
        return
    _reg[(layer, module)] = [t for t in tensors if t is not None and t.is_cuda and t.is_contiguous()]
    if drafter:
        _drafter.add(layer)
    else:
        _last_layer = max(_last_layer, layer)


def register_head(tensors) -> None:
    """The LM head's weights, in the order of their bytes (head.py, an EXL3 head: its trellis)."""
    global _head
    if not enabled() or HEAD_BUDGET <= 0:
        return
    _head = [t for t in tensors if t is not None and t.is_cuda and t.is_contiguous()]


def _join_hook(module, args, output) -> None:
    join()


def register_decoder_layers(model) -> None:
    """Called by patch 0017 once a DeepseekV4Model's weights are loaded. EXL3_L2PF_HC=1: the mHC mix weights and
    router gate of every backbone layer. EXL3_L2PF_HEAD_MB > 0: a forward hook on the final norm that joins the head
    prefetch the last layer's MoE forks, so a captured forward still ends with no side-stream work."""
    global _head_hooked
    if not enabled():
        return
    if HC:
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
    norm = getattr(model, "norm", None)
    if (HEAD_BUDGET > 0 and isinstance(norm, torch.nn.Module) and type(norm).__name__ != "PPMissingLayer"
            and "next" in POINTS):
        if not getattr(norm, "_exl3_l2pf_join", False):
            norm.register_forward_hook(_join_hook)
            norm._exl3_l2pf_join = True
        _head_hooked = True
        logger.info("EXL3 L2 prefetch: the final norm joins the LM head prefetch (%d MiB)", HEAD_BUDGET >> 20)


def _cut(tensors, budget: int) -> tuple[list[int], list[int]]:
    """(pointers, sizes) of the tensors' bytes in order, `budget` bytes in all."""
    ptrs, sizes, left = [], [], budget
    for t in tensors:
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


def _ranges(layer: int, point: str) -> tuple[list[int], list[int]]:
    budget = min(BUDGET, MOE_CAP) if point == "moe" and MOE_CAP > 0 else BUDGET
    return _cut([t for off, module in PLAN[point] for t in _reg.get((layer + off, module), ())], budget)


def join() -> None:
    global _pending
    if _pending:
        torch.cuda.current_stream().wait_stream(_stream)
        _pending = False


def _fork(ptrs: list[int], sizes: list[int]) -> None:
    """The ranges into L2 on the side stream, once the current stream's work so far is done."""
    global _stream, _pending, _announced
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
        logger.info("EXL3 L2 prefetch on: points %s, %d MiB a fork (moe %d), %.0f GB/s pacing, %d modules registered "
                    "(layers 0-%d), %s order, head %d MiB", ",".join(sorted(POINTS)), BUDGET >> 20,
                    (min(BUDGET, MOE_CAP) if MOE_CAP > 0 else BUDGET) >> 20, RATE_GBPS, len(_reg), _last_layer,
                    "critical-first" if ORDER else "default", HEAD_BUDGET >> 20 if _head_hooked and _head else 0)


def point(name: str, layer: int, rows: int) -> None:
    """Prefetch point `name` of `layer` (a decode forward with `rows` rows)."""
    if not POINTS or not 0 < rows <= MAX_ROWS:
        return
    if layer in _drafter:
        # the draft chain: `next` forks only into a following draft layer; the last one joins and ends the chain
        if name not in POINTS:
            if name == "next":
                join()
            return
        join()
        if name != "next" or (layer + 1) in _drafter:
            _fork(*_ranges(layer, name))
        return
    last = name == "next" and layer >= _last_layer
    if name not in POINTS and not last:
        return
    join()
    if last:
        # the last layer's MoE: the forward ends with nothing in flight, except the LM head's first bytes when the
        # final norm's hook joins them (EXL3_L2PF_HEAD_MB); a drafter layer past the backbone never forks
        if name in POINTS and layer == _last_layer and _head_hooked and _head:
            _fork(*_cut(_head, HEAD_BUDGET))
        return
    _fork(*_ranges(layer, name))
