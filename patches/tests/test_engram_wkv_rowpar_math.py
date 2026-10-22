# SPDX-License-Identifier: Apache-2.0
"""Patch 0028 (VLLM_ENGRAM_WKV_ROWPAR=1), the math only: plain torch on the CPU, runs anywhere.

(1) Head bookkeeping. With TP-only head shards (ParallelEngramEmbedding._get_shard_info at Engram DP size 1: part =
cdiv(heads, tp), head_start = rank * part, padded heads looked up as zeros), a rank's lookup rows [T, part, head_dim],
flattened, are exactly columns [rank * in_per, (rank + 1) * in_per) of the rows embed() gathers today (all-gather on
dim 1, then [:, :n_hash_cols], flattened), in_per = 6144 / tp = the RowParallelLinear input partition. At TP=5 the 24
heads pad to 25 and 6144 does not split into 5: the fast path refuses that.
(2) The FP8 cut. The checkpoint's wkv is e4m3 [25600, 6144] with ue8m0 scales [800, 192] (weight_block_size [32, 32]).
vLLM's MXFP8 loaders for a RowParallelLinear (KMxfp8Static's scale loader repeats each block row 32 times, then
RowvLLMParameter.load_row_parallel_weight narrows input_dim = 1 at rank * shard) give each rank a weight and a scale
shard whose dequantized product is bit-exact the input-column slice of the full dequantized matrix; same for the
128 x 128 FP8 block layout (fp32 scales, BlockQuantScaleParameter narrowed on input_dim = 1).
(3) Numerics at the real shapes (6 rows, 6144 -> 25600, two ranks). Today each output is one exact dot product rounded
to bf16 once. Row-parallel rounds each rank's partial to bf16, then the RDMA ring sums the two in fp32 in rank order and
rounds once (NCCL's two-rank bf16 sum rounds the same way, checked). Asserted elementwise: |y_row - y| <=
u * (|y| + |p0| + |p1|), u = 2^-8, plus the fp32 accumulation error measured against fp64. Printed: the max abs / rel
difference to today's bits and to the exact product. The partials come from an fp32 GEMM like the kernels' fp32
accumulators.
(4) EXL3 (the plugin's engram.wkv when exl3_moe.dense.engram.wkv is set): dense.Cut's input-row cut (whole 16-row
trellis tiles, suh cut, svh whole) at 6144 inputs; the two ranks' partial products, from exl3_ref (the plain-torch EXL3
linear the plugin's kernels are checked against), sum to the full product.

  ~/.venvs/dsv41-kdev/bin/python3 -m pytest -q -s test_engram_wkv_rowpar_math.py
"""
import os
import sys

import pytest
import torch

HEADS, HEAD_DIM = 24, 256          # (engram_max_ngram_size - 1) * engram_n_heads, engram_head_dim
IN, OUT = HEADS * HEAD_DIM, 5120 * 5   # wkv: 6144 -> hidden_size * (hc_mult + 1) = 25600
TP, T = 2, 6                       # ranks; rows of a k=5 DSpark verify step
U = 2.0 ** -8                      # bf16 unit roundoff (8 significant bits, round to nearest)


def cdiv(a, b):
    return -(-a // b)


def rank_heads(tp, rank):
    """(part, head_start) of a rank's head shard, as ParallelEngramEmbedding at Engram DP size 1."""
    part = cdiv(HEADS, tp)
    return part, rank * part


def lookup_rows(rows, tp, rank):
    """What the lookup kernel writes for one rank: its heads in order, zeros for padded heads."""
    part, start = rank_heads(tp, rank)
    out = torch.zeros(rows.shape[0], part, HEAD_DIM, dtype=rows.dtype)
    n = max(0, min(part, HEADS - start))
    out[:, :n] = rows[:, start:start + n]
    return out


@pytest.mark.parametrize("tp", [2, 3, 4, 6])
def test_rank_heads_are_wkv_input_rows(tp):
    rows = torch.randn(T, HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(tp)).bfloat16()
    local = [lookup_rows(rows, tp, r) for r in range(tp)]
    gathered = torch.cat(local, dim=1)[:, :HEADS].flatten(-2)       # today's embed(), flattened for wkv
    assert torch.equal(gathered, rows.flatten(-2))
    part, _ = rank_heads(tp, 0)
    assert part * tp == HEADS and IN % tp == 0                      # equal, contiguous shares: the fast path's rule
    in_per = IN // tp                                               # RowParallelLinear.input_size_per_partition
    for r in range(tp):
        assert local[r].flatten(-2).shape[1] == in_per
        assert torch.equal(local[r].flatten(-2), gathered[:, r * in_per:(r + 1) * in_per])


def test_padded_heads_do_not_tile_the_input():
    part, start = rank_heads(5, 4)
    assert part * 5 != HEADS and IN % 5 != 0                        # 25 head slots for 24 heads; 6144 / 5 is no integer
    rows = torch.randn(T, HEADS, HEAD_DIM).bfloat16()
    assert torch.count_nonzero(lookup_rows(rows, 5, 4)[:, HEADS - start:]) == 0   # the last rank's padded head


# ---------------------------------------------------------------------------------------------------- the FP8 cut
def checkpoint_wkv(block, seed=0):
    """wkv as the checkpoint stores it: e4m3 values, one scale a block x block tile (ue8m0 at 32, fp32 at 128)."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(OUT, IN, generator=g) * 96).clamp(-448, 448).to(torch.float8_e4m3fn)
    exps = torch.randint(127 - 13, 127 - 5, (OUT // block, IN // block), generator=g, dtype=torch.uint8)
    if block == 32:
        return w, exps.view(torch.float8_e8m0fnu)
    return w, torch.exp2(exps.float() - 127)


def dequant(w, scale_elems):
    """e4m3 times the per-element scale (exact in fp32: a power of two or an fp32 times an 8-bit significand)."""
    return w.float() * scale_elems


def block_scales(scale, block):
    """[rows/b, cols/b] block scales -> per-element fp32 values."""
    s = torch.exp2(scale.view(torch.uint8).float() - 127) if scale.dtype == torch.float8_e8m0fnu else scale
    return s.repeat_interleave(block, 0).repeat_interleave(block, 1)


def load_row_parallel(w, scale, block, rank, tp=TP):
    """What a RowParallelLinear's parameters hold on `rank` after vLLM's loaders ran on the checkpoint tensors."""
    in_per = IN // tp
    w_param = w.narrow(1, rank * in_per, in_per)                    # ModelWeightParameter, input_dim 1
    if block == 32:
        # ModelOptLinearMethod (MXFP8, CkptCtx scale_block_size (32, 32)): KMxfp8Static's scale loader
        # repeat_interleave(32, dim=0), then load_row_parallel_weight on [OUT, in_per / 32], input_dim 1
        rows = scale.view(torch.uint8).repeat_interleave(block, dim=0)
        s_param = rows.narrow(1, rank * (in_per // block), in_per // block)
        s_elems = torch.exp2(s_param.float() - 127).repeat_interleave(block, 1)
    else:
        # Fp8LinearMethod block quant: BlockQuantScaleParameter [OUT / 128, in_per / 128], input_dim 1
        s_param = scale.narrow(1, rank * (in_per // block), in_per // block)
        s_elems = s_param.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return w_param, s_param, s_elems


@pytest.mark.parametrize("block", [32, 128])
def test_fp8_block_scales_cut_by_input_rows(block):
    assert (IN // TP) % block == 0                                  # no scale block straddles the cut
    w, scale = checkpoint_wkv(block)
    full = dequant(w, block_scales(scale, block))
    in_per = IN // TP
    for r in range(TP):
        w_param, s_param, s_elems = load_row_parallel(w, scale, block, r)
        s_rows = OUT if block == 32 else OUT // block                 # MXFP8 scales are repeated to weight rows
        assert w_param.shape == (OUT, in_per) and s_param.shape == (s_rows, in_per // block)
        mine = dequant(w_param, s_elems)
        assert torch.equal(mine, full[:, r * in_per:(r + 1) * in_per]), f"rank {r}: shard is not the input-column slice"
        # today's column cut, for contrast: output rows [r * 12800, ...) of the same matrix
        out_per = OUT // TP
        if block == 32:
            s_col = scale.view(torch.uint8).repeat_interleave(block, 0).narrow(0, r * out_per, out_per)
            s_col = torch.exp2(s_col.float() - 127).repeat_interleave(block, 1)
            col = dequant(w.narrow(0, r * out_per, out_per), s_col)
            assert torch.equal(col, full[r * out_per:(r + 1) * out_per])


# ---------------------------------------------------------------------------------------------------- numerics
@pytest.fixture(scope="module")
def problem():
    w, scale = checkpoint_wkv(32, seed=1)
    W = dequant(w, block_scales(scale, 32))                         # [25600, 6144] fp32, exact
    g = torch.Generator().manual_seed(2)
    # Engram rows as the lookup kernel writes them: e4m3 table values times a ue8m0 scale per 32 values, bf16 (exact)
    vals = (torch.randn(T, IN, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    exps = torch.randint(-11, -4, (T, IN // 32), generator=g).float()
    x = (vals * torch.exp2(exps).repeat_interleave(32, 1)).bfloat16()
    assert torch.equal(x.float(), vals * torch.exp2(exps).repeat_interleave(32, 1))
    return x, W


def test_row_parallel_sum_is_rounding_level(problem):
    x, W = problem
    in_per = IN // TP
    cols = [slice(r * in_per, (r + 1) * in_per) for r in range(TP)]
    xd, Wd = x.double(), W.double()
    y = xd @ Wd.T                                                   # exact product (8-bit significands, fp64 sums)
    p = [xd[:, c] @ Wd[:, c].T for c in cols]                       # each rank's exact partial
    # the kernels: fp32 accumulation, one bf16 rounding of what they store
    y32 = x.float() @ W.T
    p32 = [x[:, c].float() @ W[:, c].T for c in cols]
    acc = [(y32.double() - y).abs()] + [(q32.double() - q).abs() for q32, q in zip(p32, p)]
    y_col = y32.bfloat16()                                          # today: column shards of the full-K product
    parts = [q.bfloat16() for q in p32]                             # row-parallel: each rank's partial, stored as bf16
    y_row = (parts[0].float() + parts[1].float()).bfloat16()        # the RDMA ring: rank-order fp32 sum, rounded once
    assert torch.equal(parts[0] + parts[1], y_row)                  # NCCL's bf16 sum of two values: the same bits

    bound_col = U * y.abs() * (1 + 2 * U) + 2 * acc[0]
    bound_row = U * (y.abs() + p[0].abs() + p[1].abs()) * (1 + 4 * U) + 2 * (acc[1] + acc[2])
    err_col, err_row = (y_col.double() - y).abs(), (y_row.double() - y).abs()
    assert bool((err_col <= bound_col).all()), "today's output outside one bf16 rounding"
    assert bool((err_row <= bound_row).all()), "row-parallel output outside its two-rounding bound"

    diff = (y_row.double() - y_col.double()).abs()
    scale = y.abs().max().item()
    big = y.abs() >= 1e-2 * scale
    # the extra rounding is that of the partials: measure it in ulps of the larger one (bf16: 7 stored bits)
    p_max = torch.maximum(parts[0].double().abs(), parts[1].double().abs()).clamp_min(1e-30)
    ulps = diff / torch.exp2(torch.floor(torch.log2(p_max)) - 7)
    n_rel_row = (err_row.norm() / y.norm()).item()
    n_rel_col = (err_col.norm() / y.norm()).item()
    print(f"\nwkv {T} x {IN} -> {OUT}, TP={TP}, max |y| {scale:.4g}. Row-parallel vs today's bits: "
          f"{(diff > 0).double().mean():.1%} of elements differ, max abs diff {diff.max():.3g} "
          f"({diff.max() / scale:.2e} of max |y|), norm-wise {(diff.norm() / y.norm()).item():.2e}, at most "
          f"{ulps.max():.2f} ulps of the larger partial; max rel diff {(diff[big] / y.abs()[big]).max():.2e} where "
          f"|y| >= 1% of max (partials cancel there: (|p0| + |p1|) / |y| up to "
          f"{((p[0].abs() + p[1].abs())[big] / y.abs()[big]).max():.0f})")
    print(f"vs the exact product: norm-wise rel error row {n_rel_row:.2e}, today {n_rel_col:.2e} "
          f"(x{n_rel_row / n_rel_col:.2f}); max abs error row {err_row.max():.3g}, today {err_col.max():.3g}")
    assert n_rel_row < 2.5 * n_rel_col                              # one extra rounding of each partial, not more
    # half an ulp for each partial, at most one (of the larger partial, |y| <= 2 x it) for the sum's and for today's
    # rounding, plus fp32 accumulation noise
    assert ulps.max() <= 3.01


# ---------------------------------------------------------------------------------------------------- EXL3
def exl3_ref():
    here = os.path.dirname(os.path.abspath(__file__))
    plugin = os.path.join(here, "..", "exl3_moe")
    if plugin not in sys.path:
        sys.path.insert(0, plugin)
    from vllm_exl3_moe import exl3_ref as ref                        # torch only: the package imports no vLLM

    return ref


@pytest.mark.parametrize("k", [4, 5])
def test_exl3_input_row_cut_sums_to_full(k):
    ref = exl3_ref()
    out_f = 2048                                                    # a narrower output than 25,600: the same cut
    g = torch.Generator().manual_seed(10 + k)
    trellis = torch.randint(-32768, 32768, (IN // 16, out_f // 16, 16 * k), dtype=torch.int16, generator=g)
    suh = ((torch.randint(0, 2, (IN,), generator=g) * 2 - 1) * 0.05).half()
    svh = ((torch.randint(0, 2, (out_f,), generator=g) * 2 - 1) * 0.05).half()
    x = torch.randn(T, IN, generator=g).bfloat16()
    full = ref.linear(x, trellis, suh, svh, k)
    parts = []
    for r in range(TP):
        lo, hi = r * IN // TP, (r + 1) * IN // TP                   # dense.rank_cut(6144, 2, r, row_parallel=True)
        assert lo % 128 == 0 and hi % 128 == 0                      # whole Hadamard blocks, whole 16-row tiles
        tr = trellis.narrow(0, lo // 16, (hi - lo) // 16)           # dense.Cut.trellis
        parts.append(ref.linear(x[:, lo:hi], tr, suh[lo:hi], svh, k))   # Cut.suh cuts, Cut.svh keeps
    rel = ((parts[0] + parts[1] - full).norm() / full.norm()).item()
    sum_bf16 = (parts[0].bfloat16().float() + parts[1].bfloat16().float()).bfloat16()
    rel_bf16 = ((sum_bf16.float() - full).norm() / full.norm()).item()
    print(f"\nEXL3 K{k} engram.wkv input-row cut: fp32 partials rel {rel:.2e}, bf16 partials (mrow) rel {rel_bf16:.2e}")
    assert rel < 1e-5 and rel_bf16 < 1e-2
