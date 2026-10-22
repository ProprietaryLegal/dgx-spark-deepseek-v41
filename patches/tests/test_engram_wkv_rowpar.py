# SPDX-License-Identifier: Apache-2.0
"""Patch 0028: Engram's wkv row-parallel over the hash heads (VLLM_ENGRAM_WKV_ROWPAR=1), in vLLM at an emulated TP=2
(each rank's modules built in one process with the TP rank patched in, the collectives emulated from both ranks'
tensors).

(1) engram_wkv_row_parallel: off without the flag and at TP=1; on at TP=2 over 24 heads; refused (column layout kept)
with sequence parallelism, DP-sharded tables and a head count TP does not divide.
(2) Engram._create_wkv with the checkpoint's own quant config (V4.1 FP8: [32, 32] ue8m0 blocks, so ModelOpt's MXFP8
method on Marlin, as the serve runs it with FlashInferCutlassMxfp8LinearKernel disabled). Flag on: a RowParallelLinear
over a 3,072-wide input shard that all-reduces; off: today's gathered ColumnParallelLinear. The checkpoint tensors
(layers.1.engram.wkv.weight e4m3 [25600, 6144], .scale e8m0 [800, 192]) go through the parameters' own weight loaders as
load_weights calls them: each rank holds bit-exact the input-column slice of the weight and of the block scales (the
output-row slice with the flag off). After process_weights_after_loading, the ranks' partial products summed in rank
order in fp32 and rounded once (the RDMA ring; NCCL rounds a two-rank bf16 sum the same way) match today's gathered
output to rounding, against the exact product of the dequantized weights.
(3) Engram.embed and Engram.forward with the real lookup kernel (GPU-resident tables, prepare_embeddings inline) and the
collectives emulated: flag on, embed returns the rank's own staged rows (no gather, the persistent buffer itself), which
are its columns of today's gathered rows; forward hands wkv exactly those 3,072 columns (its partial is what the
all-reduce receives) and runs today's post kernel on the all-reduced kv (bit-identical to today's module fed that kv);
both ranks' outputs are identical, kv matches today's to rounding.
(4) The plugin's EXL3 engram.wkv (exl3_moe.dense.engram.wkv = K) on the same RowParallelLinear: its input-row cut (whole
16-row trellis tiles, suh cut, svh whole) loads the full checkpoint tensors, the FP8 wkv tensors still sink, and the
partials sum to the full reconstructed product, at decode (exl3_mrow) and prompt (exllamav3) row counts.

Run inside the dsv41 image on a GPU:  PYTHONPATH=/p/exl3_moe python3 -m pytest -q -s test_engram_wkv_rowpar.py
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pytest.importorskip("vllm")   # runs in the serve image

import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.parameter as param_mod
import vllm.models.deepseek_v41.common.engram as engram_ops
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.models.deepseek_v41.common.engram import Engram, ParallelEngramEmbedding, engram_wkv_row_parallel

DEV = "cuda"
TP, T = 2, 6                                    # ranks; rows of a k=5 DSpark verify step
HEAD_DIM, HIDDEN, HC = 256, 5120, 4
HEAD_SIZES = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,      # 24 bucket ranges (prime sizes),
              163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227)      # 3 n-gram orders x 8 heads
IN, OUT = len(HEAD_SIZES) * HEAD_DIM, HIDDEN * (HC + 1)                       # 6144 -> 25600
PREFIX = "model.layers.1.engram"
U = 2.0 ** -8                                   # bf16 unit roundoff
FLAG = "VLLM_ENGRAM_WKV_ROWPAR"


@pytest.fixture(autouse=True)
def serve_kernels(monkeypatch):
    # dsv41_tp2.sh: MXFP8 linears on Marlin (W8A16), not FlashInfer's CUTLASS mm_mxfp8
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "FlashInferCutlassMxfp8LinearKernel")


def at_rank(monkeypatch, rank, tp=TP):
    """Make the modules built next think they are TP rank `rank` of `tp`."""
    for mod in (engram_ops, linear_mod, param_mod):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: tp)


def v41_fp8_config():
    """config.json's quantization_config of the V4.1 checkpoint."""
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    return DeepseekV4FP8Config.from_config({"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3",
                                            "scale_fmt": "ue8m0", "weight_block_size": [32, 32]})


def embedding(monkeypatch, rank, tp=TP, head_sizes=HEAD_SIZES, table=None):
    at_rank(monkeypatch, rank, tp)
    with torch.device(DEV):
        emb = ParallelEngramEmbedding(sum(head_sizes), HEAD_DIM, head_sizes)
    if table is not None:
        for p, t in zip((emb.weight, emb.weight_scale_inv), table):
            p.weight_loader(p, t)
    return emb


def bare_engram(emb):
    m = Engram.__new__(Engram)
    nn.Module.__init__(m)
    m.dim, m.hc_mult = HIDDEN, HC
    m.embed_tokens = emb
    return m


@pytest.fixture(scope="module")
def wkv_ckpt():
    """layers.1.engram.wkv as the checkpoint stores it, and its exact dequantized matrix (CPU)."""
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(OUT, IN, generator=g) * 96).clamp(-448, 448).to(torch.float8_e4m3fn)
    exps = torch.randint(127 - 13, 127 - 5, (OUT // 32, IN // 32), generator=g, dtype=torch.uint8)
    W = w.float() * torch.exp2(exps.float() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    return w, exps.view(torch.float8_e8m0fnu), W


def load_wkv(wkv, w, scale):
    """V4.1's load_weights for engram.wkv: the generic branch, param.weight_loader(param, tensor)."""
    params = dict(wkv.named_parameters())
    for name, t in (("weight", w), ("weight_scale", scale)):
        params[name].weight_loader(params[name], t)


def build_wkv(monkeypatch, rank, rowpar, emb=None, quant_config=None):
    """A bare Engram of TP rank `rank` with the flag set or not, and its wkv from Engram._create_wkv."""
    if rowpar:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    emb = emb if emb is not None else embedding(monkeypatch, rank)
    m = bare_engram(emb)
    at_rank(monkeypatch, rank)
    with torch.device(DEV):
        m.wkv = m._create_wkv(IN, quant_config or v41_fp8_config(), False, PREFIX)
    return m


# ---------------------------------------------------------------------------------------------------- (1) decision
def test_flag_engages_only_on_equal_tp_head_shares(monkeypatch):
    emb = embedding(monkeypatch, 0)
    monkeypatch.delenv(FLAG, raising=False)
    assert not engram_wkv_row_parallel(emb, False)                  # off by default
    monkeypatch.setenv(FLAG, "1")
    assert engram_wkv_row_parallel(emb, False)
    assert (emb.part_n_hash_cols, emb.head_start) == (12, 0)
    assert engram_wkv_row_parallel(embedding(monkeypatch, 1), False)
    assert not engram_wkv_row_parallel(emb, True)                   # sequence parallel: untouched
    assert not engram_wkv_row_parallel(embedding(monkeypatch, 0, tp=1), False)
    assert not engram_wkv_row_parallel(embedding(monkeypatch, 0, head_sizes=HEAD_SIZES[:15]), False)  # 15 heads / 2
    # DP-sharded tables (a node-local Engram DP group of 2, no shared memory): heads over TP x DP shards
    monkeypatch.setattr(engram_ops, "get_engram_dp_size", lambda: 2)
    monkeypatch.setattr(engram_ops, "get_engram_dp_group", lambda: SimpleNamespace(world_size=2, rank_in_group=0))
    dp_emb = embedding(monkeypatch, 0)
    assert dp_emb.dp_size == 2 and dp_emb.part_n_hash_cols == 6
    assert not engram_wkv_row_parallel(dp_emb, False)


# ---------------------------------------------------------------------------------------------------- (2) wkv
@pytest.mark.parametrize("rowpar", [False, True])
def test_create_wkv_layout_and_loaded_shards(monkeypatch, wkv_ckpt, rowpar):
    w, scale, _ = wkv_ckpt
    rows = scale.view(torch.uint8).repeat_interleave(32, 0)         # KMxfp8Static's scale loader: [25600, 192]
    for r in range(TP):
        m = build_wkv(monkeypatch, r, rowpar)
        wkv = m.wkv
        assert m.wkv_row_parallel is rowpar
        load_wkv(wkv, w, scale)
        if rowpar:
            assert isinstance(wkv, RowParallelLinear) and wkv.reduce_results and wkv.input_is_parallel
            assert (wkv.input_size_per_partition, wkv.output_size_per_partition) == (IN // TP, OUT)
            c = slice(r * IN // TP, (r + 1) * IN // TP)
            want_w, want_s = w[:, c], rows[:, r * IN // TP // 32:(r + 1) * IN // TP // 32]
        else:
            assert isinstance(wkv, ColumnParallelLinear) and wkv.gather_output
            assert (wkv.input_size_per_partition, wkv.output_size_per_partition) == (IN, OUT // TP)
            o = slice(r * OUT // TP, (r + 1) * OUT // TP)
            want_w, want_s = w[o], rows[o]
        assert type(wkv.quant_method).__name__ == "ModelOptLinearMethod"
        assert torch.equal(wkv.weight.data.cpu().view(torch.uint8), want_w.contiguous().view(torch.uint8))
        assert torch.equal(wkv.weight_scale.data.cpu(), want_s.contiguous())


def test_row_parallel_partials_match_gathered_output(monkeypatch, wkv_ckpt):
    w, scale, W = wkv_ckpt
    g = torch.Generator().manual_seed(3)
    x = ((torch.randn(T, IN, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn).float()
         * torch.exp2(torch.randint(-11, -4, (T, IN // 32), generator=g).float()).repeat_interleave(32, 1)).bfloat16()
    xd = x.to(DEV)
    parts, cols = [], []
    for r in range(TP):
        for rowpar in (True, False):
            m = build_wkv(monkeypatch, r, rowpar)
            load_wkv(m.wkv, w, scale)
            m.wkv.quant_method.process_weights_after_loading(m.wkv)
            assert type(m.wkv.quant_method.kernel).__name__ == "MarlinMxfp8LinearKernel"
            if rowpar:
                parts.append(m.wkv.quant_method.apply(m.wkv, xd[:, r * IN // TP:(r + 1) * IN // TP].contiguous()))
            else:
                cols.append(m.wkv.quant_method.apply(m.wkv, xd))
    today = torch.cat(cols, -1).cpu()
    row = (parts[0].float() + parts[1].float()).bfloat16().cpu()
    assert torch.equal((parts[0] + parts[1]).cpu(), row)           # a bf16 add of the two parts: the same bits
    y = x.double() @ W.double().T
    p = [x[:, r * IN // TP:(r + 1) * IN // TP].double() @ W[:, r * IN // TP:(r + 1) * IN // TP].double().T
         for r in range(TP)]
    terms = x.double().abs() @ W.double().abs().T                   # sum of |x_i w_i|: the fp32 accumulators' scale
    err_today, err_row = (today.double() - y).abs(), (row.double() - y).abs()
    assert bool((err_today <= U * y.abs() * (1 + 2 * U) + 2.0 ** -14 * terms).all())
    assert bool((err_row <= U * (y.abs() + p[0].abs() + p[1].abs()) * (1 + 4 * U) + 2.0 ** -14 * terms).all())
    rel_today, rel_row = (err_today.norm() / y.norm()).item(), (err_row.norm() / y.norm()).item()
    diff = (row.double() - today.double()).abs()
    print(f"\nMarlin MXFP8 wkv {T} x {IN} -> {OUT}: row-parallel vs today {(diff > 0).double().mean():.1%} of elements "
          f"differ, max abs {diff.max():.3g} of max |y| {y.abs().max():.4g}; norm-wise error vs the exact product "
          f"row {rel_row:.2e}, today {rel_today:.2e}")
    assert rel_row < 2.5 * rel_today and rel_today < 1e-2


# ---------------------------------------------------------------------------------------------------- (3) Engram
@pytest.fixture(scope="module")
def table():
    rows = sum(HEAD_SIZES)
    g = torch.Generator().manual_seed(5)
    weight = (torch.randn(rows, HEAD_DIM, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    scales = torch.randint(127 - 9, 127 - 4, (rows, HEAD_DIM // 32), generator=g, dtype=torch.uint8)
    return weight, scales.view(torch.float8_e8m0fnu)               # the checkpoint stores the scale as F8_E8M0


def hash_ids(num_tokens, seed=13):
    """Row ids per head inside its own bucket range, the first and last row of each range included."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ids = torch.empty(num_tokens, len(HEAD_SIZES), dtype=torch.int32, device=DEV)
    start = 0
    for head, size in enumerate(HEAD_SIZES):
        ids[:, head].random_(start, start + size, generator=g)
        ids[0, head], ids[-1, head] = start, start + size - 1
        start += size
    return ids


def engram_rank(monkeypatch, rank, rowpar, table, wkv_ckpt, qk):
    w, scale, _ = wkv_ckpt
    emb = embedding(monkeypatch, rank, table=table)
    m = build_wkv(monkeypatch, rank, rowpar, emb=emb)
    load_wkv(m.wkv, w, scale)
    m.wkv.quant_method.process_weights_after_loading(m.wkv)
    m.layer_hash_index, m.eps, m.clamp_value, m.use_sequence_parallel = 0, 1e-6, 1e-6, False
    m.q_weight = nn.Parameter(qk[0].clone(), requires_grad=False)
    m.k_weight = nn.Parameter(qk[1].clone(), requires_grad=False)
    m.staged_rows = torch.empty(64, emb.part_n_hash_cols, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    return m


def run_engram(monkeypatch, rowpar, table, wkv_ckpt, ids, hidden, mask, qk, kv_override=None):
    """Both ranks' Engram.forward; kv_override (column layout only) replaces what the output all-gather returns."""
    ms = [engram_rank(monkeypatch, r, rowpar, table, wkv_ckpt, qk) for r in range(TP)]
    assert all(m.wkv_row_parallel is rowpar for m in ms)
    for m in ms:
        m.prepare_embeddings(ids)                                  # GPU-resident tables: the lookup runs inline
    staged = [m.staged_rows[: ids.shape[0]] for m in ms]
    gathered = torch.cat(staged, dim=1)                            # today's all-gather of the rows on dim 1
    if rowpar:
        partial = [m.wkv.quant_method.apply(m.wkv, s.flatten(-2)) for m, s in zip(ms, staged)]
        kv = (partial[0].float() + partial[1].float()).bfloat16()
    else:
        col = [m.wkv.quant_method.apply(m.wkv, gathered.flatten(-2)) for m in ms]
        kv = torch.cat(col, -1) if kv_override is None else kv_override
    outs = []
    for r, m in enumerate(ms):
        at_rank(monkeypatch, r)
        calls = []

        def all_gather(t, dim=-1, r=r):
            assert not rowpar, "a row-parallel wkv gathers nothing"
            calls.append("all_gather")
            if t.dim() == 3:                                       # embed()'s rows
                assert dim == 1 and torch.equal(t, staged[r])
                return gathered
            assert torch.equal(t, col[r])                          # wkv's output columns
            return kv

        def all_reduce(t, r=r):
            assert rowpar
            calls.append("all_reduce")
            assert torch.equal(t, partial[r]), "wkv's input was not this rank's own rows"
            return kv

        monkeypatch.setattr(engram_ops, "tensor_model_parallel_all_gather", all_gather)
        monkeypatch.setattr(linear_mod, "tensor_model_parallel_all_gather", all_gather)
        monkeypatch.setattr(linear_mod, "tensor_model_parallel_all_reduce", all_reduce)
        rows = m.embed(ids)
        if rowpar:
            assert calls == [] and rows.data_ptr() == m.staged_rows.data_ptr()   # the staging buffer, ungathered
            c = slice(r * IN // TP, (r + 1) * IN // TP)
            assert torch.equal(rows.flatten(-2), gathered.flatten(-2)[:, c])
        else:
            assert calls == ["all_gather"] and torch.equal(rows, gathered)
        calls.clear()
        outs.append(m(hidden, ids, mask))
        assert calls == (["all_reduce"] if rowpar else ["all_gather", "all_gather"])
    assert torch.equal(outs[0], outs[1])                           # both ranks hold the same stream
    return outs[0], kv


def test_engram_forward_rowpar_matches_gathered(monkeypatch, table, wkv_ckpt):
    g = torch.Generator(device=DEV).manual_seed(7)
    ids = hash_ids(T)
    hidden = torch.randn(T, HC, HIDDEN, device=DEV, generator=g).bfloat16()
    mask = torch.ones(T, dtype=torch.bool, device=DEV)
    mask[2] = False                                                # a dead position passes through untouched
    qk = [torch.randn(HC, HIDDEN, device=DEV, generator=g).bfloat16() for _ in range(2)]
    out_today, kv_today = run_engram(monkeypatch, False, table, wkv_ckpt, ids, hidden, mask, qk)
    out_row, kv_row = run_engram(monkeypatch, True, table, wkv_ckpt, ids, hidden, mask, qk)
    # the row path is today's post kernel on the all-reduced kv: today's module fed that kv gives the same bits
    out_cross, _ = run_engram(monkeypatch, False, table, wkv_ckpt, ids, hidden, mask, qk, kv_override=kv_row)
    assert torch.equal(out_row, out_cross)
    rel_kv = ((kv_row.float() - kv_today.float()).norm() / kv_today.float().norm()).item()
    kv_max = kv_today.float().abs().max().item()
    d = (out_row.float() - out_today.float()).abs()
    print(f"\nEngram forward, {T} tokens: kv rel diff {rel_kv:.2e} (|kv| up to {kv_max:.3g}); output max abs diff "
          f"{d.max():.3g}, {(d > 0).double().mean():.1%} of elements differ")
    assert rel_kv < 1e-2
    assert torch.equal(out_row[2], hidden[2]) and torch.equal(out_today[2], hidden[2])   # gate shut: untouched
    # value's extra rounding (about 2 u of the partials), the gate's shift and the output's own rounding
    assert d.max().item() <= 8 * U * kv_max


# ---------------------------------------------------------------------------------------------------- (4) EXL3
def exl3_config(k):
    from vllm_exl3_moe.config import DeepseekV41Exl3Config

    return DeepseekV41Exl3Config.from_config({"quant_method": "deepseek_v41_exl3", "activation_scheme": "dynamic",
                                              "fmt": "e4m3", "scale_fmt": "ue8m0", "weight_block_size": [32, 32],
                                              "exl3_moe": {"dense": {"engram": {"wkv": k}}}})


@pytest.mark.parametrize("rows", [6, 24])                          # exl3_mrow (<= 16 rows), exllamav3 above
def test_exl3_engram_wkv_row_parallel(monkeypatch, rows):
    pytest.importorskip("exllamav3")
    pytest.importorskip("vllm_exl3_moe")
    from exllamav3.modules.quant.exl3 import LinearEXL3

    k = 4
    g = torch.Generator(device=DEV).manual_seed(11)
    full = {"trellis": torch.randint(-32768, 32768, (IN // 16, OUT // 16, 16 * k), dtype=torch.int16, device=DEV,
                                     generator=g),
            "suh": ((torch.randint(0, 2, (IN,), device=DEV, generator=g) * 2 - 1) * 0.05).half(),
            "svh": ((torch.randint(0, 2, (OUT,), device=DEV, generator=g) * 2 - 1) * 0.05).half(),
            "mul1": torch.tensor(0x83DCD12D - 2 ** 32, dtype=torch.int32, device=DEV)}
    W = LinearEXL3(None, IN, OUT, suh=full["suh"], svh=full["svh"], trellis=full["trellis"],
                   mul1=full["mul1"]).get_weight_tensor().t().float()                   # [OUT, IN]
    x = torch.randn(rows, IN, device=DEV, generator=g).bfloat16()
    y = 0
    for r in range(TP):
        m = build_wkv(monkeypatch, r, True, quant_config=exl3_config(k))
        wkv = m.wkv
        assert isinstance(wkv, RowParallelLinear) and type(wkv.quant_method).__name__ == "Exl3DenseLinearMethod"
        params = dict(wkv.named_parameters())
        for n in ("weight", "weight_scale"):                       # the variant's FP8 wkv tensors still stream in
            assert params[n].weight_loader(params[n], torch.zeros(8, 8, dtype=torch.uint8)) and params[n].numel() == 0
        for n, t in full.items():
            assert params[n].weight_loader(params[n], t)
        assert wkv.trellis.shape == (IN // TP // 16, OUT // 16, 16 * k) and wkv.svh.shape == (OUT,)
        wkv.quant_method.process_weights_after_loading(wkv)
        y = y + wkv.quant_method.apply(wkv, x[:, r * IN // TP:(r + 1) * IN // TP].contiguous()).float()
    ref = torch.nn.functional.linear(x.float(), W)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"\nEXL3 K{k} engram.wkv row-parallel, {rows} rows: partial sum vs the full product rel {rel:.2e}")
    assert rel < 1e-2
