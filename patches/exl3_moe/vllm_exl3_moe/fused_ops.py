# SPDX-License-Identifier: Apache-2.0
"""Two small Triton kernels around the decode-size routed MoE, replacing ~13 tiny torch launches per layer (casts,
mask, clamp, gather, where, clone, adds) with 2. Same arithmetic as the torch ops they replace, bit for bit:

  moe_prep(x, topk_ids, topk_weights, perm) -> x16 (fp16), sel (int64 physical slots), rw16 (fp16, 0 for unrouted)
      = x.to(fp16), perm[ids.clamp(min=0)], where(ids >= 0, w, 0).to(fp16)
  sum_groups_bf16(outs, m) -> bf16 [m, hidden]
      = (outs[0][:m] + outs[1][:m] + ...).to(bf16), added in list order in fp32 like the clone + add_ chain
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_GROUPS = 8


@triton.jit
def _moe_prep_kernel(x_ptr, ids_ptr, w_ptr, perm_ptr, x16_ptr, sel_ptr, rw_ptr,
                     x_stride, ids_stride, w_stride,
                     H: tl.constexpr, TOPK: tl.constexpr, TOPK_P2: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offs_k = tl.arange(0, TOPK_P2)
    mk = offs_k < TOPK
    ids = tl.load(ids_ptr + row * ids_stride + offs_k, mask=mk, other=0).to(tl.int64)
    valid = ids >= 0
    phys = tl.load(perm_ptr + tl.where(valid, ids, 0), mask=mk, other=0)
    w = tl.load(w_ptr + row * w_stride + offs_k, mask=mk, other=0.0)
    w = tl.where(valid, w, 0.0)
    tl.store(sel_ptr + row * TOPK + offs_k, phys, mask=mk)
    tl.store(rw_ptr + row * TOPK + offs_k, w.to(tl.float16), mask=mk)
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mh = offs < H
        v = tl.load(x_ptr + row * x_stride + offs, mask=mh, other=0.0)
        tl.store(x16_ptr + row * H + offs, v.to(tl.float16), mask=mh)


def moe_prep(x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, perm: torch.Tensor):
    m, h = x.shape
    top_k = topk_ids.shape[-1]
    x16 = torch.empty((m, h), dtype=torch.float16, device=x.device)
    sel = torch.empty((m, top_k), dtype=torch.int64, device=x.device)
    rw16 = torch.empty((m, top_k), dtype=torch.float16, device=x.device)
    _moe_prep_kernel[(m,)](x, topk_ids, topk_weights, perm, x16, sel, rw16,
                           x.stride(0), topk_ids.stride(0), topk_weights.stride(0),
                           H=h, TOPK=top_k, TOPK_P2=triton.next_power_of_2(top_k), BLOCK_H=1024)
    return x16, sel, rw16


@triton.jit
def _sum_groups_kernel(o0, o1, o2, o3, o4, o5, o6, o7, out_ptr, numel, N_GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.load(o0 + offs, mask=mask, other=0.0)
    if N_GROUPS > 1:
        acc += tl.load(o1 + offs, mask=mask, other=0.0)
    if N_GROUPS > 2:
        acc += tl.load(o2 + offs, mask=mask, other=0.0)
    if N_GROUPS > 3:
        acc += tl.load(o3 + offs, mask=mask, other=0.0)
    if N_GROUPS > 4:
        acc += tl.load(o4 + offs, mask=mask, other=0.0)
    if N_GROUPS > 5:
        acc += tl.load(o5 + offs, mask=mask, other=0.0)
    if N_GROUPS > 6:
        acc += tl.load(o6 + offs, mask=mask, other=0.0)
    if N_GROUPS > 7:
        acc += tl.load(o7 + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


def sum_groups_bf16(outs: list[torch.Tensor], m: int) -> torch.Tensor:
    """outs: per-group fp32 [rows_max, hidden] scratch (rows past m unused); returns bf16 [m, hidden]."""
    n = len(outs)
    assert 1 <= n <= MAX_GROUPS
    hidden = outs[0].shape[1]
    out = torch.empty((m, hidden), dtype=torch.bfloat16, device=outs[0].device)
    ptrs = list(outs) + [outs[0]] * (MAX_GROUPS - n)
    numel = m * hidden
    BLOCK = 1024
    _sum_groups_kernel[(triton.cdiv(numel, BLOCK),)](*ptrs, out, numel, N_GROUPS=n, BLOCK=BLOCK)
    return out
