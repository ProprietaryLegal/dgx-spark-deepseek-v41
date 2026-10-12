# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 sparse attention for verify-sized calls on SM12x (GB10), Triton.

What vLLM's SM120 attention hands FlashInfer for its short (verify) queries, computed in two launches:
  part:  grid (token, head group of HB heads, split). Each program walks every SPLITS-th key block of its token: the
         sliding-window keys (SWA paged cache, per-token row indices and lengths) then the selected compressed keys
         (compressed paged cache, global rows, lengths). It dequantizes them, keeps an online softmax in fp32 and stores
         its partial (max, sum, accumulator).
  merge: grid (token, head group). It combines the splits in split order with the attention sink, so a row's bits never
         depend on how many rows share the call.
The keys are their own values (DSV4's MQA: one 512-wide vector a key, 448 nope + 64 rope dims); the output keeps the rope
part rotated (the o-projection applies the inverse RoPE), as FlashInfer's does.

KV record on SM12x (vLLM's fp8_ds_mla, 584 B, V4.1 attention.py: not the MXFP8 528 B record, which is family 100 only):
448 float8_e4m3 nope bytes, 64 bf16 rope values (128 B), 7 UE8M0 scale bytes (one per 64 nope values) + 1 pad. A page of
P tokens holds its P data rows (576 B each) first and its P scale rows (8 B each) after; pages sit `page_stride` bytes
apart. Rows are addressed by absolute index r: page r // P, slot r % P.

The split-over-keys + ordered-merge structure follows TensorFold's DeepSeek-V4.1 sparse attention (_sparse_attn_part /
_sparse_attn_merge, github.com/bertholomus/TensorFold, branch deepseek-v41-tp2, Apache-2.0; csrc/NOTICE.tensorfold);
the record decoding, paging and interface here are ours, for vLLM's cache.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

ROW = 576          # data bytes a row: 448 fp8 nope + 128 bf16 rope
NOPE = 448
ROPE = 64
SCALES = 8         # bytes a scale row (7 ue8m0 + pad)


@triton.jit
def _keys(CACHE, rows, ok, page_stride, PAGE: tl.constexpr, BN: tl.constexpr):
    """BN records -> (nope [BN, 512] bf16 with columns 448.. zero, rope [BN, 64] bf16)."""
    pg = rows // PAGE
    sl = rows % PAGE
    base = pg.to(tl.int64) * page_stride + sl.to(tl.int64) * 576
    sbase = pg.to(tl.int64) * page_stride + PAGE * 576 + sl.to(tl.int64) * 8
    c = tl.arange(0, 512)
    cm = c < 448
    raw = tl.load(CACHE + base[:, None] + c[None, :], mask=ok[:, None] & cm[None, :], other=0)
    v = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)                       # [BN, 512]
    g = tl.arange(0, 8)
    sb = tl.load(CACHE + sbase[:, None] + g[None, :], mask=ok[:, None] & (g < 7)[None, :], other=127)
    sc = tl.exp2(sb.to(tl.float32) - 127.0)                                      # [BN, 8]
    nope = tl.reshape(tl.reshape(v, (BN, 8, 64)) * sc[:, :, None], (BN, 512)).to(tl.bfloat16)
    r = tl.arange(0, 64)
    rp = (CACHE + base[:, None] + 448 + 2 * r[None, :]).to(tl.pointer_type(tl.bfloat16))
    rope = tl.load(rp, mask=ok[:, None], other=0.0)
    return nope, rope


@triton.jit
def _dsv4_sparse_part(Q, SWA, SIDX, SLEN, CMP, CIDX, CLEN, PM, PL, PO, PAD, pad_off, scale, ns, nc, swa_ps, cmp_ps,
                      q_st, q_sh, H: tl.constexpr, HB: tl.constexpr, BN: tl.constexpr, SPLITS: tl.constexpr,
                      SPAGE: tl.constexpr, CPAGE: tl.constexpr, HAS_CMP: tl.constexpr, NBS: tl.constexpr,
                      NBC: tl.constexpr, HAS_PAD: tl.constexpr = False, STAGES: tl.constexpr = 2):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    sp = tl.program_id(2)
    if HAS_PAD:
        if tl.load(PAD + pad_off + t) != 0:
            return                     # a dead verify row (past its request's depth): the merge writes zeros
    h = hb * HB + tl.arange(0, HB)
    c = tl.arange(0, 512)
    qb = Q + t.to(tl.int64) * q_st + h[:, None] * q_sh
    q_nope = tl.load(qb + c[None, :], mask=(c < 448)[None, :], other=0.0)            # [HB, 512] bf16, 448.. zero
    q_rope = tl.load(qb + 448 + tl.arange(0, 64)[None, :])                          # [HB, 64]
    m_i = tl.full((HB,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((HB,), dtype=tl.float32)
    acc_n = tl.zeros((HB, 512), dtype=tl.float32)
    acc_r = tl.zeros((HB, 64), dtype=tl.float32)
    n = tl.arange(0, BN)
    slen = tl.load(SLEN + t)
    clen = 0
    if HAS_CMP:
        clen = tl.load(CLEN + t)
    # two static loops (the window, then the picks), so Triton can pipeline the next block's gathers under this
    # block's math; split sp takes blocks sp, sp + SPLITS, ... of each segment
    for b in tl.range(sp, NBS, SPLITS, num_stages=STAGES):
        j = b * BN + n
        ok = j < slen
        rows = tl.load(SIDX + t.to(tl.int64) * ns + j, mask=ok, other=0)
        ok = ok & (rows >= 0)
        rows = tl.where(ok, rows, 0)
        k_n, k_r = _keys(SWA, rows, ok, swa_ps, SPAGE, BN)
        s = (tl.dot(q_nope, tl.trans(k_n)) + tl.dot(q_rope, tl.trans(k_r))) * scale  # [HB, BN]
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p16 = p.to(tl.bfloat16)
        acc_n = acc_n * alpha[:, None] + tl.dot(p16, k_n)
        acc_r = acc_r * alpha[:, None] + tl.dot(p16, k_r)
        m_i = m_new
    if HAS_CMP:
        for b in tl.range(sp, NBC, SPLITS, num_stages=STAGES):
            j = b * BN + n
            ok = j < clen
            rows = tl.load(CIDX + t.to(tl.int64) * nc + j, mask=ok, other=0)
            ok = ok & (rows >= 0)
            rows = tl.where(ok, rows, 0)
            k_n, k_r = _keys(CMP, rows, ok, cmp_ps, CPAGE, BN)
            s = (tl.dot(q_nope, tl.trans(k_n)) + tl.dot(q_rope, tl.trans(k_r))) * scale
            s = tl.where(ok[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            p16 = p.to(tl.bfloat16)
            acc_n = acc_n * alpha[:, None] + tl.dot(p16, k_n)
            acc_r = acc_r * alpha[:, None] + tl.dot(p16, k_r)
            m_i = m_new
    base = (t * (H // HB) + hb) * SPLITS + sp
    tl.store(PM + base * HB + tl.arange(0, HB), m_i)
    tl.store(PL + base * HB + tl.arange(0, HB), l_i)
    ob = PO + base.to(tl.int64) * HB * 512 + tl.arange(0, HB)[:, None] * 512
    tl.store(ob + c[None, :], acc_n, mask=(c < 448)[None, :])
    tl.store(ob + 448 + tl.arange(0, 64)[None, :], acc_r)


@triton.jit
def _dsv4_sparse_merge(PM, PL, PO, SINK, OUT, PAD, pad_off, o_st, o_sh, H: tl.constexpr, HB: tl.constexpr,
                       SPLITS: tl.constexpr, HAS_PAD: tl.constexpr = False):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    hh = tl.arange(0, HB)
    d = tl.arange(0, 512)
    if HAS_PAD:
        if tl.load(PAD + pad_off + t) != 0:
            ob = OUT + t.to(tl.int64) * o_st + (hb * HB + hh)[:, None] * o_sh
            tl.store(ob + d[None, :], tl.zeros((HB, 512), dtype=tl.bfloat16))
            return
    base0 = (t * (H // HB) + hb) * SPLITS
    m = tl.full((HB,), -1e30, dtype=tl.float32)
    for sp in range(SPLITS):
        m = tl.maximum(m, tl.load(PM + (base0 + sp) * HB + hh))
    l = tl.zeros((HB,), dtype=tl.float32)
    acc = tl.zeros((HB, 512), dtype=tl.float32)
    for sp in range(SPLITS):
        a = tl.exp(tl.load(PM + (base0 + sp) * HB + hh) - m)
        l += tl.load(PL + (base0 + sp) * HB + hh) * a
        acc += tl.load(PO + (base0 + sp).to(tl.int64) * HB * 512 + hh[:, None] * 512 + d[None, :]) * a[:, None]
    h = hb * HB + hh
    l += tl.exp(tl.load(SINK + h) - m)
    ob = OUT + t.to(tl.int64) * o_st + h[:, None] * o_sh
    tl.store(ob + d[None, :], (acc / l[:, None]).to(tl.bfloat16))


_scratch: dict = {}


def sparse_attention(q: torch.Tensor, out: torch.Tensor, scale: float, sink: torch.Tensor,
                     swa_cache: torch.Tensor, swa_idx: torch.Tensor, swa_len: torch.Tensor, swa_page: int,
                     cmp_cache: torch.Tensor | None = None, cmp_idx: torch.Tensor | None = None,
                     cmp_len: torch.Tensor | None = None, cmp_page: int = 0,
                     hb: int = 16, bn: int = 64, splits: int = 4, stages: int = 1,
                     pad: torch.Tensor | None = None, pad_off: int = 0) -> torch.Tensor:
    """q [T, H, 512] bf16 -> out [T, H, 512] bf16 (rope part still rotated). Caches: the paged uint8 records (any
    shape whose leading dim is pages; page stride = its stride(0)), indices [T, n] int32 absolute rows, lengths [T]."""
    T, H, D = q.shape
    assert D == 512 and q.dtype == torch.bfloat16 and q.stride(2) == 1 and out.stride(2) == 1
    assert H % hb == 0
    if sink.dtype != torch.float32:
        sink = sink.float()
    has_cmp = cmp_cache is not None and cmp_idx is not None and cmp_idx.shape[-1] > 0
    swa_b = swa_cache.view(torch.uint8) if swa_cache.dtype != torch.uint8 else swa_cache
    cmp_b = (cmp_cache.view(torch.uint8) if cmp_cache.dtype != torch.uint8 else cmp_cache) if has_cmp else swa_b
    ns, nc = swa_idx.shape[-1], (cmp_idx.shape[-1] if has_cmp else 0)
    key = (q.device, T, H, hb, splits)
    sc = _scratch.get(key)
    if sc is None:
        n = T * (H // hb) * splits * hb
        sc = (torch.empty(n, dtype=torch.float32, device=q.device), torch.empty(n, dtype=torch.float32, device=q.device),
              torch.empty(n * 512, dtype=torch.float32, device=q.device))
        _scratch[key] = sc
    pm, pl, po = sc
    has_pad = pad is not None
    padv = (pad.view(torch.uint8) if pad.dtype == torch.bool else pad) if has_pad else pm
    _dsv4_sparse_part[(T, H // hb, splits)](
        q, swa_b, swa_idx, swa_len, cmp_b, cmp_idx if has_cmp else swa_idx, cmp_len if has_cmp else swa_len,
        pm, pl, po, padv, pad_off, scale, ns, max(nc, 1), swa_b.stride(0), cmp_b.stride(0), q.stride(0), q.stride(1),
        H=H, HB=hb, BN=bn, SPLITS=splits, SPAGE=swa_page, CPAGE=(cmp_page or swa_page), HAS_CMP=has_cmp,
        NBS=triton.cdiv(ns, bn), NBC=(triton.cdiv(nc, bn) if has_cmp else 0), HAS_PAD=has_pad, STAGES=stages,
        num_warps=8)
    _dsv4_sparse_merge[(T, H // hb)](pm, pl, po, sink, out, padv, pad_off, out.stride(0), out.stride(1), H=H, HB=hb,
                                     SPLITS=splits, HAS_PAD=has_pad, num_warps=8)
    return out


def reference(q, scale, sink, swa_cache, swa_idx, swa_len, swa_page, cmp_cache=None, cmp_idx=None, cmp_len=None,
              cmp_page=0):
    """The same attention in fp32 PyTorch, from the same bytes (for tests)."""
    def records(cache, rows, page):
        b = cache.view(torch.uint8).reshape(cache.shape[0], -1)
        pg, sl = rows // page, rows % page
        data = torch.stack([b[p, s * ROW:(s + 1) * ROW] for p, s in zip(pg.tolist(), sl.tolist())])
        sb = torch.stack([b[p, page * ROW + s * SCALES: page * ROW + s * SCALES + 7] for p, s in zip(pg.tolist(),
                                                                                                         sl.tolist())])
        nope = data[:, :NOPE].view(torch.float8_e4m3fn).float().reshape(-1, 7, 64)
        nope = (nope * torch.exp2(sb.float() - 127.0)[:, :, None]).reshape(-1, NOPE)
        rope = data[:, NOPE:].contiguous().view(torch.bfloat16).float()
        return torch.cat([nope.bfloat16().float(), rope], 1)                           # [n, 512]
    T, H, _ = q.shape
    out = torch.empty(T, H, 512, dtype=torch.float32, device=q.device)
    for t in range(T):
        ks = [records(swa_cache, swa_idx[t, :int(swa_len[t])].long(), swa_page)]
        if cmp_cache is not None and cmp_idx is not None and int(cmp_len[t]) > 0:
            ks.append(records(cmp_cache, cmp_idx[t, :int(cmp_len[t])].long(), cmp_page))
        k = torch.cat(ks, 0)                                                            # [n, 512]
        s = (q[t].float() @ k.t()) * scale                                              # [H, n]
        s = torch.cat([s, sink.float()[:, None]], 1)
        p = torch.softmax(s, 1)[:, :-1]
        out[t] = p @ k
    return out
