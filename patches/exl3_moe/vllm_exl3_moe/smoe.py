# SPDX-License-Identifier: Apache-2.0
"""exl3_smoe: our decode-size routed-expert kernel (csrc/exl3_smoe.cu) for one layer's EXL3 bank, every K in one launch.

Three launches per call (rot + run table, gate/up + activation, down + ordered per-token sum), weights streamed with TMA
bulk copies, bf16 in and out, deterministic, graph-safe (static grids from m, no host sync). Scratch is shared by every
layer of the process (layers run in stream order; the per-token counters reset themselves).

attach_shared(): the layer's shared expert (its EXL3 dense gate_up / down, same tile layout and TP cut as an expert
slot) becomes physical slot E; a call with shared=True gives every live row that slot as one more pick with weight 1,
summed with the routed picks in pick order: the output is then routed + shared, this rank's partial sum.

fused(): a decode step's whole MoE phase in one persistent launch: router GEMV + DeepSeek-V4
top-k + routed experts + the shared expert, whose stream starts beside the router instead of after routing.
Bit-identical to exl3_router.router_topk followed by __call__(..., shared=True).

EXL3_L2_DISCARD=1 (default 0 until measured on the serve): the build with -DSMOE_L2_DISCARD=1, where the warp that sums
a token's picks drops their per-slot fp32 outputs (ys) from L2 (discard.global.L2) instead of leaving them to be
written back to DRAM as the next weights stream through; the outputs keep their bits (tests/test_l2_discard_gpu.py).
"""

from __future__ import annotations

import os

import torch

from .bank import Exl3Bank

_ext: dict[bool, object] = {}
_SCRATCH: dict[tuple, tuple[torch.Tensor, ...]] = {}
_FUSED: dict[tuple, dict[str, torch.Tensor]] = {}
MAX_SLOTS = 256                 # m * top_k per call (exl3_smoe.cu)
FUSED_MAX_ROWS = 8              # tokens a fused launch (exl3_smoe.cu R_MAXM)


def l2_discard() -> bool:
    """EXL3_SMOE_L2_DISCARD (0 / 1) overrides EXL3_L2_DISCARD for this kernel alone (isolating one kernel's discard in an A/B)."""
    v = os.environ.get("EXL3_SMOE_L2_DISCARD")
    return (v if v is not None else os.environ.get("EXL3_L2_DISCARD", "0")) == "1"


def ext(discard: bool | None = None):
    """The compiled kernels; `discard` picks the L2-discard build (None: as EXL3_L2_DISCARD says). Each build is a
    module of its own (jit.py hashes the flag into its name), so both can live in one process."""
    d = l2_discard() if discard is None else bool(discard)
    e = _ext.get(d)
    if e is None:
        from .jit import load
        # e.g. EXL3_SMOE_FLAGS="-DSMOE_OBI=4" (tuning)
        e = _ext[d] = load("exl3_smoe", "EXL3_SMOE_FLAGS", cflags=("-DSMOE_L2_DISCARD=1",) if d else ())
    return e


def supported(bank: Exl3Bank) -> bool:
    """Integer K in 2..6 for every expert, 128-aligned shapes (the plan's codebook must be mul1: the caller checks)."""
    ks = [k for b in bank.lp.bits for k in (b.k_gu, b.k_d)]
    return (all(float(k).is_integer() and 2 <= k <= 6 for k in ks) and bank.hidden % 256 == 0
            and bank.inter_local % 128 == 0)


def _scratch(device: torch.device, H: int, I: int, e=None):
    key = (str(device), H, I)
    sc = _SCRATCH.get(key)
    if sc is None:
        n_hs, n_act, n_ys, n_rt, n_ctr = (e or ext()).sizes(H, I)
        sc = (torch.empty(n_hs, dtype=torch.float16, device=device),
              torch.empty(n_act, dtype=torch.float16, device=device),
              torch.empty(n_ys, dtype=torch.float32, device=device),
              torch.zeros(n_rt, dtype=torch.int32, device=device),
              torch.zeros(n_ctr, dtype=torch.int32, device=device))
        _SCRATCH[key] = sc
    return sc


def _fused_scratch(device: torch.device, n: int, top_k: int, e=None) -> dict[str, torch.Tensor]:
    """Routing workspace and sync words of the fused launch, shared by every layer (layers run in stream order; the
    launch leaves its sync words zero)."""
    key = (str(device), n, top_k)
    sc = _FUSED.get(key)
    if sc is None:
        sc = {"ws": torch.empty(2 * FUSED_MAX_ROWS * n, dtype=torch.float32, device=device),
              "ids": torch.zeros(FUSED_MAX_ROWS * top_k, dtype=torch.int32, device=device),
              "w": torch.zeros(FUSED_MAX_ROWS * top_k, dtype=torch.float32, device=device),
              "sync": torch.zeros((e or ext()).sync_words(), dtype=torch.int32, device=device),
              "f32": torch.empty(0, dtype=torch.float32, device=device),
              "i64": torch.empty(0, dtype=torch.int64, device=device)}
        _FUSED[key] = sc
    return sc


class SmoeLayer:
    """One layer's banks registered with the kernel (raw pointers: rebuild if the bank tensors ever move)."""

    def __init__(self, bank: Exl3Bank, top_k: int, discard: bool | None = None):
        if not supported(bank):
            raise ValueError("exl3_smoe: needs integer K in 2..6 and 128-aligned shapes")
        t, bl, lp = bank.t, bank.bl, bank.lp
        dev = t["w13_trellis"].device
        E = bank.num_experts

        def tab(vals, dtype):
            return torch.tensor(vals, dtype=dtype, device=dev)

        bits = [lp.bits[lp.inv_perm[p]] for p in range(E)]          # physical slot order
        self.top_k = top_k
        self.H, self.I = bank.hidden, bank.inter_local
        self.tables = (tab([2 * o for o in bl.off_w1], torch.int64), tab([2 * o for o in bl.off_w3], torch.int64),
                       tab([2 * o for o in bl.off_w2], torch.int64),
                       tab([int(b.k_gu) for b in bits], torch.int32), tab([int(b.k_d) for b in bits], torch.int32))
        for n in ("w13_trellis", "w2_trellis", "w13_suh", "w13_svh", "w2_suh", "w2_svh"):
            if not t[n].is_contiguous():
                raise ValueError(f"exl3_smoe: {n} must be contiguous")
        self.ext = ext(discard)                                     # the handle belongs to this build's registry
        self.handle = self.ext.register(t["w13_trellis"], t["w2_trellis"], *self.tables, t["w13_suh"], t["w13_svh"],
                                        t["w2_suh"], t["w2_svh"], self.H, self.I)
        self.scratch = _scratch(dev, self.H, self.I, self.ext)
        self.has_shared = False

    def attach_shared(self, gate, up, down) -> None:
        """The shared expert as physical slot E. gate / up / down: EXL3 linears of this rank (exllamav3 LinearEXL3 or
        anything with trellis int16 [in/16, out/16, 16K], suh [in], svh [out] fp16): gate / up [H -> I], down [I -> H]."""
        def c(v):
            return v if v.is_contiguous() else v.contiguous()
        self.ext.attach_shared(self.handle, c(gate.trellis), c(up.trellis), c(down.trellis), c(gate.suh), c(up.suh),
                               c(gate.svh), c(up.svh), c(down.suh), c(down.svh))
        self._shared_keep = (gate, up, down)
        self.has_shared = True

    def rows_ok(self, m: int, shared: bool = False) -> bool:
        return 1 <= m and m * (self.top_k + int(shared)) <= MAX_SLOTS

    def fused_ok(self, m: int, n: int) -> bool:
        return 1 <= m <= FUSED_MAX_ROWS and n % 32 == 0 and n <= 384 and self.top_k <= 8

    def fused(self, x: torch.Tensor, gate_weight: torch.Tensor, bias: torch.Tensor, perm: torch.Tensor,
              act_limit: float, scale: float, pad: torch.Tensor | None = None, shared: bool = False,
              bias_vl: torch.Tensor | None = None, token_ids: torch.Tensor | None = None, sentinel_lo: int = 0):
        """x bf16 [m, H] (1 <= m <= 8) -> (this rank's partial MoE output bf16 [m, H], topk_ids int32 [m, top_k],
        topk_weights fp32 [m, top_k]): the router (gate_weight bf16 [n, H], fp32 correction bias [n]; bias_vl [n] with
        token_ids int64 [m] for image-sentinel routing), the top-k, the routed experts and with shared=True the shared
        expert, in one launch. The routing tensors are views of shared scratch: valid until the next fused call."""
        m = x.shape[0]
        if x.stride(-1) != 1 or x.data_ptr() % 16 or x.stride(0) % 8:
            x = x.contiguous().clone()
        sc = _fused_scratch(x.device, gate_weight.shape[0], self.top_k, self.ext)
        out = torch.empty((m, self.H), dtype=torch.bfloat16, device=x.device)
        self.ext.run_fused(self.handle, x, gate_weight, bias, sc["f32"] if bias_vl is None else bias_vl,
                           sc["i64"] if token_ids is None else token_ids, int(sentinel_lo), float(scale), self.top_k,
                           perm, out, float(act_limit), *self.scratch, pad, int(shared), sc["ws"], sc["ids"], sc["w"],
                           sc["sync"])
        k = self.top_k
        return out, sc["ids"][: m * k].view(m, k), sc["w"][: m * k].view(m, k)

    def __call__(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, perm: torch.Tensor,
                 act_limit: float, pad: torch.Tensor | None = None, shared: bool = False) -> torch.Tensor:
        """x bf16 [m, H] -> this rank's routed partial sum, bf16 [m, H] (+ the shared expert's with shared=True, after
        attach_shared). `pad` (bool, >= m rows): rows that take no experts (vLLM's is_padding), whatever their top-k
        ids say."""
        m = x.shape[0]
        if x.stride(-1) != 1 or x.data_ptr() % 16 or x.stride(0) % 8:
            x = x.contiguous().clone()
        w = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
        ids = topk_ids if topk_ids.dtype in (torch.int32, torch.int64) else topk_ids.to(torch.int32)
        if w.stride(-1) != 1:
            w = w.contiguous()
        if ids.stride(-1) != 1:
            ids = ids.contiguous()
        out = torch.empty((m, self.H), dtype=torch.bfloat16, device=x.device)
        self.ext.run(self.handle, x, ids, w, perm, out, float(act_limit), *self.scratch, pad, int(shared))
        return out
