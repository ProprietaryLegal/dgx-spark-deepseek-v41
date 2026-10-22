# SPDX-License-Identifier: Apache-2.0
"""Patches 0024 / 0025 and the launcher's INDEXER_LOGITS_MB: their sizing at our serve settings, on the CPU.

0024 builds DeepSeek-V4.1's two fp32 RoPE tables to max(max_model_len, max_num_batched_tokens) + 2048 rows instead of
1,048,576; 0025 sizes the sparse indexer's prefill gather buffer at 8 x max_model_len rows instead of 40 x. Checked at
the launcher's settings (MAX_LEN 262144, MAX_BATCH 2048, MAX_SEQS 2, DSpark k = 5): the row counts and the bytes they
save per rank, that a capped table keeps the full table's bits, that every prefill chunk the indexer's chunker forms
fits the smaller buffer and is the chunk factor 40 forms, and what a 128 MB logits cap costs in query sub-chunks.

The patches' helpers are exec'd from the patch files (their added lines), so no vLLM is needed:
    ~/venvs/exl3/bin/python3 -m pytest -q patches/tests/test_dsv41_memory_caps.py
Where vLLM imports (the serve image), the chunker under test is vLLM's own; elsewhere the copy below.
"""
import math
import pathlib
import random
import sys
import types

import pytest
import torch

PATCHES = pathlib.Path(__file__).resolve().parents[1]
MAX_LEN, MAX_BATCH, MAX_SEQS, SPEC_K = 262_144, 2_048, 2, 5   # scripts/launch/dsv41_tp2.sh defaults; DSpark k
V41_ROWS = 65_536 * 16      # max_position_embeddings x factor: 1,048,576 rows in both of V4.1's tables
ROPE_COLS = 64              # qk_rope_head_dim 64: 32 cos + 32 sin columns a row, fp32
KEY_ROW = 128 + 4           # an indexer key row in the gather buffer: 128 fp8 values + an fp32 scale
RADIX_TOPK = 1 << 20        # indexer_topk.RADIX_TOPK_WORKSPACE_SIZE, reserved beside the gather in the profile run
MIB = 1 << 20

try:   # the serve image: test vLLM's own chunker
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

    split_chunks = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks
except Exception:  # noqa: BLE001  (no vLLM here: the copy below)
    split_chunks = None


def _split_indexer_prefill_chunks_copy(
    compressed_seq_lens_cpu, prefill_query_lens_cpu, workspace_size, max_logits_bytes, request_offset=0
):
    """vLLM main 6e517b15, vllm/v1/attention/backends/mla/indexer.py, DeepseekV32IndexerMetadataBuilder.
    _split_indexer_prefill_chunks (Apache-2.0), unchanged by patches 0001-0025; docstring and comments trimmed."""
    chunks = []
    n = len(compressed_seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4
    end = 0
    while end < n:
        start, chunk_m, chunk_n = end, 0, 0
        while end < n:
            q, s = (prefill_query_lens_cpu[end].item(), compressed_seq_lens_cpu[end].item())
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break
        if end == start:
            chunk_m, chunk_n = (prefill_query_lens_cpu[end].item(), compressed_seq_lens_cpu[end].item())
            end += 1
        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else max(1, chunk_m)
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))
    return chunks


split_chunks = split_chunks or _split_indexer_prefill_chunks_copy


def patch_lines(patch_glob):
    return next(PATCHES.glob(patch_glob)).read_text().splitlines()


def added_lines(patch_glob):
    return [ln[1:] for ln in patch_lines(patch_glob) if ln.startswith("+") and not ln.startswith("+++")]


def added_source(patch_glob, names):
    """The module-level definitions `names` (a function or a constant) as the patch adds them: the run of added lines
    from `def name(` / `name =` up to the next top-level line or the end of the run."""
    lines, out = patch_lines(patch_glob), []
    for name in names:
        start = next(i for i, ln in enumerate(lines) if ln.startswith((f"+def {name}(", f"+{name} =")))
        out.append(lines[start][1:])
        for ln in lines[start + 1:]:
            if not ln.startswith("+") or ln[1:2] not in ("", " "):
                break
            out.append(ln[1:])
    return "\n".join(out) + "\n"


def fake_config(max_len, batch):
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(max_model_len=max_len),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=batch),
    )


@pytest.fixture
def rope_rows(monkeypatch):
    """0024's _dsv41_rope_rows against a stub vllm.config / vllm.logger; rope_rows(full, max_len, batch)."""
    state = {"config": None}
    logged = []
    config_mod, logger_mod = types.ModuleType("vllm.config"), types.ModuleType("vllm.logger")
    config_mod.get_current_vllm_config_or_none = lambda: state["config"]
    logger_mod.init_logger = lambda name: types.SimpleNamespace(info_once=lambda msg, *a: logged.append(msg % a))
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", config_mod)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.delenv("VLLM_DSV41_ROPE_CAP", raising=False)
    ns = {"__name__": "deepseek_scaling_rope"}
    exec(added_source("0024-*.patch", ["_DSV41_ROPE_MARGIN", "_dsv41_rope_rows"]), ns)

    def rows(full, max_len=MAX_LEN, batch=MAX_BATCH):
        state["config"] = None if max_len is None else fake_config(max_len, batch)
        return ns["_dsv41_rope_rows"](full)

    rows.logged = logged
    return rows


@pytest.fixture
def buffer_factor(monkeypatch):
    """0025's _indexer_prefill_buffer_factor (reads VLLM_INDEXER_PREFILL_BUFFER_FACTOR)."""
    monkeypatch.delenv("VLLM_INDEXER_PREFILL_BUFFER_FACTOR", raising=False)
    ns = {}
    exec(added_source("0025-*.patch", ["_indexer_prefill_buffer_factor"]), ns)
    return ns["_indexer_prefill_buffer_factor"]


# ---- 0024: RoPE tables ----------------------------------------------------------------------------------------------

def test_rope_rows_at_the_launcher_settings(rope_rows):
    assert rope_rows(V41_ROWS) == MAX_LEN + 2_048 == 264_192
    assert rope_rows.logged == ["DeepSeek-V4 RoPE tables: 264192 of 1048576 rows (patch 0024; "
                                "VLLM_DSV41_ROPE_CAP=0 builds them whole)"]
    assert rope_rows(float(V41_ROWS)) == 264_192                # the sliding-window table's length arrives as a float
    assert rope_rows(V41_ROWS, 327_680) == 329_728              # HeNryous's 320K serve
    assert rope_rows(V41_ROWS, 4_096, 16_384) == 18_432         # a token budget above max_model_len reaches further
    assert rope_rows(V41_ROWS, V41_ROWS) == V41_ROWS            # never more rows than the checkpoint's table
    assert rope_rows(V41_ROWS, None) == V41_ROWS                # no vLLM config (the layer built alone): full table


def test_rope_cap_off_switch(rope_rows, monkeypatch):
    monkeypatch.setenv("VLLM_DSV41_ROPE_CAP", "0")
    assert rope_rows(V41_ROWS) == V41_ROWS


def test_every_position_the_serve_reaches_is_in_the_table(rope_rows):
    """The furthest positions the engine feeds the RoPE kernels, which read the table without a bounds check."""
    rows = rope_rows(V41_ROWS)
    furthest = {
        "a request's last token (the scheduler keeps spec tokens under max_model_len)": MAX_LEN - 1,
        "DSpark's draft block (dflash/speculator.py clamps it to max_model_len - 1)": MAX_LEN - 1,
        "a draft block even if unclamped": MAX_LEN - 1 + SPEC_K,
        "a dummy or capture run (MAX_BATCH tokens, context clamped to max_model_len)": max(MAX_LEN, MAX_BATCH) - 1,
    }
    assert all(pos < rows for pos in furthest.values()), furthest


def _yarn_inv_freq(theta, factor, original, rot=ROPE_COLS, beta_fast=32, beta_slow=1):
    """DeepseekScalingRotaryEmbedding._compute_inv_freq with vLLM's yarn helpers (rotary_embedding/common.py,
    vLLM main 6e517b15, Apache-2.0), for V4.1's parameters (build_deepseek_v4_rope)."""
    def corr_dim(rotations):
        return (rot * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(theta))

    low, high = max(math.floor(corr_dim(beta_fast)), 0), min(math.ceil(corr_dim(beta_slow)), rot - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(rot // 2, dtype=torch.float) - low) / (high - low), 0, 1)
    pos_freqs = theta ** (torch.arange(0, rot, 2, dtype=torch.float) / rot)
    mask = 1 - ramp
    return (1.0 / (factor * pos_freqs)) * (1 - mask) + (1.0 / pos_freqs) * mask


def _cos_sin_table(rows, inv_freq, mscale=1.0):
    """DeepseekV4ScalingRotaryEmbedding._compute_cos_sin_cache with t = arange(rows) (V4.1: mscale 1)."""
    t = torch.arange(rows, dtype=torch.float32, device=inv_freq.device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos() * mscale, freqs.sin() * mscale), dim=-1)


@pytest.mark.parametrize("theta, factor, original", [(160_000, 16, 65_536), (10_000, 1.0, V41_ROWS)],
                         ids=["compressed-yarn", "sliding-window"])
def test_a_capped_table_keeps_the_full_tables_bits(theta, factor, original):
    """The serve builds these on the GPU, one element a thread, so a shorter table holds the full one's bits: checked
    bit for bit on a GPU when there is one. A CPU's vectorized cos / sin may take another path for a tensor's tail
    (torch's aarch64 build, on a GB10's Grace cores, differs in the last bit with the length), so on a CPU to 1 ulp."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inv_freq = _yarn_inv_freq(theta, factor, original).to(dev)
    full = _cos_sin_table(original * factor, inv_freq)
    assert full.shape == (V41_ROWS, ROPE_COLS)
    for rows in (264_192, 329_728, 6_145):
        part = _cos_sin_table(rows, inv_freq)
        if dev == "cuda":
            assert torch.equal(part, full[:rows])
        else:
            torch.testing.assert_close(part, full[:rows], rtol=0, atol=2.0 ** -23)


# ---- 0025: the indexer's prefill gather buffer ----------------------------------------------------------------------

def test_buffer_factor_default_env_and_refusals(buffer_factor, monkeypatch):
    assert buffer_factor() == 8
    for raw, want in (("40", 40), ("2", 2), ("", 8)):
        monkeypatch.setenv("VLLM_INDEXER_PREFILL_BUFFER_FACTOR", raw)
        assert buffer_factor() == want
    for raw in ("1", "0", "-8", "eight", "8.0"):
        monkeypatch.setenv("VLLM_INDEXER_PREFILL_BUFFER_FACTOR", raw)
        with pytest.raises(ValueError):
            buffer_factor()


def test_the_buffer_size_goes_through_the_factor():
    added = added_lines("0025-*.patch")
    assert "    factor = _indexer_prefill_buffer_factor()" in added
    assert "    return max_model_len * factor" in added


def _prefill_steps(rng, seqs=MAX_SEQS, n=1500):
    """Prefill steps the scheduler can form at the launcher's settings: 1..seqs requests prefilling, MAX_BATCH query
    rows in all, each request's KV rows (its context so far, these queries included) at most MAX_LEN."""
    yield [MAX_LEN], [MAX_BATCH]                                           # the last chunk of a full-length prompt
    yield [MAX_LEN] * seqs, [MAX_BATCH // seqs] * seqs                      # every seat at full length
    yield [MAX_BATCH] * seqs, [MAX_BATCH // seqs] * seqs                    # first chunks only
    for _ in range(n):
        k = rng.randint(1, seqs)
        cuts = sorted(rng.sample(range(1, MAX_BATCH), k - 1)) if k > 1 else []
        q = [b - a for a, b in zip([0] + cuts, cuts + [MAX_BATCH])]
        yield [rng.randint(qi, MAX_LEN) for qi in q], q


def _plan(seq, q, ratio, factor, logits_mb):
    rows = [s // ratio if ratio > 1 else s for s in seq]                   # indexer.py: seq_lens // compress_ratio
    buffer_rows = factor * MAX_LEN // ratio                               # get_max_prefill_buffer_size // ratio
    chunks = split_chunks(torch.tensor(rows, dtype=torch.int32), torch.tensor(q, dtype=torch.int32), buffer_rows,
                          logits_mb * MIB)
    return rows, buffer_rows, [(c[0].start, c[0].stop, c[1].start, c[1].stop) for c in chunks]


@pytest.mark.parametrize("ratio", [1, 2], ids=["ratio-1-layers-20-36", "ratio-2-layers-2-14"])
@pytest.mark.parametrize("logits_mb", [512, 128])
def test_every_prefill_chunk_fits_the_smaller_buffer(ratio, logits_mb):
    rng = random.Random(1020 + ratio + logits_mb)
    for seq, q in _prefill_steps(rng):
        rows, buffer_rows, plan = _plan(seq, q, ratio, 8, logits_mb)
        for r0, r1, q0, q1 in plan:
            gathered = sum(rows[r0:r1])
            assert gathered <= buffer_rows                                 # the gather never writes past the buffer
            assert (q1 - q0) * gathered * 4 <= logits_mb * MIB             # nor the logits past the cap
        # each request lands in exactly one request-level chunk, its query rows covered once, in order
        spans = sorted({(r0, r1) for r0, r1, _, _ in plan})
        assert spans[0][0] == 0 and spans[-1][1] == len(seq)
        assert all(a1 == b0 for (_, a1), (b0, _) in zip(spans, spans[1:]))
        for r0, r1 in spans:
            subs = [(q0, q1) for a, b, q0, q1 in plan if (a, b) == (r0, r1)]
            assert subs[0][0] == 0 and subs[-1][1] == sum(q[r0:r1])
            assert all(x1 == y0 for (_, x1), (y0, _) in zip(subs, subs[1:]))
        # at MAX_SEQS 2 the buffer never binds: the chunks are the ones factor 40 makes, so results cannot move
        assert plan == _plan(seq, q, ratio, 40, logits_mb)[2]


def test_one_request_fits_at_any_allowed_factor():
    """The chunker takes a single request over the buffer whole (its gather would run past the end): with the
    smallest allowed factor a full-length request plus its draft tokens still fits."""
    for ratio in (1, 2):
        assert (MAX_LEN + SPEC_K) // ratio <= 2 * MAX_LEN // ratio


def test_where_the_buffer_binds_the_chunker_splits_by_request():
    """At 8 the logits cap binds long before the buffer: 16 full-length prefills of 128 query rows pack two to a chunk
    (2 x 128 rows x 524,288 keys x 4 B = 512 MiB). The buffer binds only for many long requests with few query rows:
    at factor 2, three full-length requests of 50 rows fit the logits cap but not 2 x MAX_LEN keys, so the chunker
    splits them by request, every chunk inside the buffer; at 8 the same step is one chunk."""
    rows, buffer_rows, plan = _plan([MAX_LEN] * 16, [128] * 16, 1, 8, 512)
    assert sorted({(r0, r1) for r0, r1, _, _ in plan}) == [(i, i + 2) for i in range(0, 16, 2)]
    assert max(sum(rows[r0:r1]) for r0, r1, _, _ in plan) == 2 * MAX_LEN < buffer_rows
    rows, buffer_rows, plan = _plan([MAX_LEN] * 3, [50] * 3, 1, 2, 512)
    assert sorted({(r0, r1) for r0, r1, _, _ in plan}) == [(0, 2), (2, 3)]
    assert all(sum(rows[r0:r1]) <= buffer_rows for r0, r1, _, _ in plan)
    assert sorted({(r0, r1) for r0, r1, _, _ in _plan([MAX_LEN] * 3, [50] * 3, 1, 8, 512)[2]}) == [(0, 3)]


def test_logits_cap_query_sub_chunks_at_256k():
    """A 2,048-token prefill chunk at a full 262,144-row context: 4 logits / top-k launches at vLLM's 512 MB, 16 at
    128 MB on the ratio-1 layers (2 and 8 on the ratio-2 layers), each launch's logits block at most the cap."""
    for ratio, want in ((1, {512: 4, 128: 16}), (2, {512: 2, 128: 8})):
        for logits_mb, n in want.items():
            rows, _, plan = _plan([MAX_LEN], [MAX_BATCH], ratio, 8, logits_mb)
            assert len(plan) == n
            assert max((q1 - q0) * rows[0] * 4 for _, _, q0, q1 in plan) == logits_mb * MIB


# ---- per-rank bytes at the launcher's settings (the patch headers quote these) ----------------------------------------

def test_bytes_saved_per_rank(rope_rows, buffer_factor):
    rope_saved = 2 * (V41_ROWS - rope_rows(V41_ROWS)) * ROPE_COLS * 4
    assert rope_saved == 401_604_608 and round(rope_saved / MIB) == 383          # 0.37 GiB

    def lane_bytes(factor):   # the profile run's reservation by a ratio-1 index layer: values, scales, top-k scratch
        rows = factor * MAX_LEN
        return sum(-(-b // 256) * 256 for b in (rows * 128, rows * 4, RADIX_TOPK))

    assert lane_bytes(40) == 1_321 * MIB and lane_bytes(buffer_factor()) == 265 * MIB
    assert lane_bytes(40) - lane_bytes(8) == 1_107_296_256 == 1_056 * MIB      # 1.03 GiB
    assert 8 * MAX_LEN * KEY_ROW == 265 * MIB - RADIX_TOPK
