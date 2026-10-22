# SPDX-License-Identifier: Apache-2.0
"""Patch 0027: greedy steps sampled from a vocab-parallel argmax (VLLM_GREEDY_LOCAL_ARGMAX=1).

(1) vLLM's LogitsProcessor.get_top_tokens (what the target's compute_logits_argmax runs) at an emulated TP=2, each rank's
call run on its own head shard with the all-gather replaced by the stacked pairs of both ranks, returns on every rank the
token the stock greedy sampler (gumbel_sample at temperature 0) picks from the gathered logits: exact ties inside and
across shards and across the sampler's vocab blocks, padding columns that would win unmasked, bf16 and fp32, 1-16 rows.
(2) rejection_sampler_utils.greedy_verify on the argmax ids returns what the stock rejection_sample returns for all-greedy
batches (sampled tokens up to num_sampled, and num_sampled): accepted, rejected, placeholder (-1) drafts, drafts equal to
a tied maximum that is not the first, requests with fewer drafts or none.
(3) GreedyLocalArgmax.sample against the stock RejectionSampler (and the stock plain greedy sampler) end to end, with
depth-policy dead rows, prefill placeholders and a chunked prefill; GreedyLocalArgmax.applies and the factory gate on every
sampling feature and configuration that reads the logits; both DeepSeek-V4.1 model classes route the hook to the head.
get_top_tokens is called unbound with a stand-in `self` (scale 1, no soft cap, the head's quant_method as the GEMM): the
same code as a constructed LogitsProcessor runs, without a vLLM config context.

Run inside the dsv41 image on a GPU:  python3 -m pytest -q test_greedy_local_argmax.py
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pytest.importorskip("vllm")   # runs in the serve image
DEV = "cuda"
V, TP, K = 129_280, 2, 5      # vocabulary, tensor parallel, DSpark drafts per round in the serve
PAD_TO = 64                   # vLLM's DEFAULT_VOCAB_PADDING_SIZE


def shard_layout(vocab, tp):
    """Per-rank width and [(org_vocab_start_index, num_org_vocab_padding)] of a ParallelLMHead's shards."""
    padded = -(-vocab // PAD_TO) * PAD_TO
    per = padded // tp
    layout = []
    for r in range(tp):
        start, end = min(r * per, vocab), min((r + 1) * per, vocab)
        layout.append((start, per - (end - start)))
    return per, layout


def padded_table(full, tp, pad_value=float("inf")):
    """The concatenated head output of all ranks: real columns, then padding columns (here +inf, which would win
    unmasked). full == padded_table(...)[:, :vocab] is what compute_logits returns after the all-gather."""
    rows, vocab = full.shape
    per, layout = shard_layout(vocab, tp)
    out = torch.full((rows, per * tp), pad_value, dtype=full.dtype, device=full.device)
    out[:, :vocab] = full
    return out, per, layout


class HeadShard:
    """What get_top_tokens reads from one rank's ParallelLMHead. The "hidden states" are row ids into a table of head
    outputs, and the head GEMM returns this rank's columns of those rows (padding columns included)."""

    def __init__(self, table_cols, tp_size, start, num_pad):
        self.tp_size = tp_size
        self.shard_indices = SimpleNamespace(org_vocab_start_index=start, num_org_vocab_padding=num_pad)
        self.quant_method = SimpleNamespace(apply=lambda layer, rows, bias=None: table_cols[rows].clone())


LP_SELF = SimpleNamespace(
    scale=1.0,
    soft_cap=None,
    _apply_head=lambda head, h, bias: head.quant_method.apply(head, h, bias=bias),
)


def tp_top_tokens(monkeypatch, table, tp, vocab, rows):
    """get_top_tokens run as each of the tp ranks on its head shard: a first pass collects every rank's (value, index)
    pairs, a second hands each rank all of them stacked, as tensor_model_parallel_all_gather(dim=-1) does. Returns the
    ids of every rank."""
    from vllm.model_executor.layers import logits_processor as lp_mod

    per, layout = shard_layout(vocab, tp)
    heads = [HeadShard(table[:, r * per:(r + 1) * per], tp, s, p) for r, (s, p) in enumerate(layout)]
    pairs = []

    def collect(t, dim=-1):
        assert dim == -1 and t.shape == (rows.numel(), 2) and t.dtype == torch.float32
        pairs.append(t.clone())
        return torch.cat([t] * tp, dim=-1)

    monkeypatch.setattr(lp_mod, "tensor_model_parallel_all_gather", collect)
    for head in heads:
        lp_mod.LogitsProcessor.get_top_tokens(LP_SELF, head, rows)
    gathered = torch.cat(pairs, dim=-1)
    monkeypatch.setattr(lp_mod, "tensor_model_parallel_all_gather", lambda t, dim=-1: gathered.clone())
    return [lp_mod.LogitsProcessor.get_top_tokens(LP_SELF, head, rows) for head in heads]


def stock_greedy_sample(logits):
    """The stock sampler at temperature 0 for every row (Sampler._sample_random with no top-k / top-p)."""
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    n = logits.shape[0]
    return gumbel_sample(
        logits,
        torch.arange(n, dtype=torch.int32, device=DEV),
        torch.zeros(n, dtype=torch.float32, device=DEV),
        torch.zeros(n, dtype=torch.int64, device=DEV),
        torch.zeros(n, dtype=torch.int64, device=DEV),
        apply_temperature=False,
        is_drafting=False,
    )


def planted_logits(rows, vocab, dtype, seed):
    """Random logits (bf16 brings exact duplicates by itself) with a planted case per row: ties inside shard 0, across
    the shards, inside shard 1, across 1024- and 8192-column blocks, at the shard boundary, the last column, a constant
    row, an all -inf row, signed zeros, +inf."""
    g = torch.Generator().manual_seed(seed)
    full = (torch.randn((rows, vocab), generator=g) * 3).to(dtype)
    per, _ = shard_layout(vocab, TP)
    m = float(torch.tensor(float(full.float().max()) + 4.0).to(dtype))
    for r in range(rows):
        row, case = full[r], r % 11
        if case == 1:
            row[10] = row[20] = m
        elif case == 2:
            row[per + 30] = row[30] = m
        elif case == 3:
            row[per + 9] = row[per + 5] = m
        elif case == 4:
            row[8192] = row[1024] = row[1023] = m
        elif case == 5:
            row[per] = row[per - 1] = m
        elif case == 6:
            row[vocab - 1] = m
        elif case == 7:
            row.fill_(1.5)
        elif case == 8:
            row.fill_(-float("inf"))
        elif case == 9:
            row.fill_(-1.0)
            row[per + 1] = row[7] = 0.0
            row[3] = -0.0
        elif case == 10:
            row[per + 50] = row[50] = float("inf")
    return full.to(DEV)


# --- (1) the reduction against the stock greedy sampler ---------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab", [V, V - 3])          # no padding; 3 padding columns on rank 1
def test_get_top_tokens_tp2_equals_stock_greedy_sampler(monkeypatch, dtype, vocab):
    for rows in range(1, 17):
        full = planted_logits(rows, vocab, dtype, seed=rows)
        table, _, _ = padded_table(full, TP)
        ids = torch.arange(rows, device=DEV)
        per_rank = tp_top_tokens(monkeypatch, table, TP, vocab, ids)
        # what compute_logits hands the sampler: the gathered row, padding sliced off (a strided view)
        stock = stock_greedy_sample(table[:, :vocab])
        for got in per_rank:
            assert got.dtype == torch.int64 and got.shape == (rows,)
            assert torch.equal(got, stock), (rows, (got != stock).nonzero().flatten().tolist())
        assert torch.equal(stock, full.argmax(dim=-1))


# --- (2) greedy_verify against rejection_sample -----------------------------------------------------------------------

def make_verify_batch(seed, num_drafts, dtype=torch.bfloat16, vocab=V):
    """An all-greedy verify batch: per request one row per draft plus the bonus row. Drafts (in the row after the one
    they are verified against) are the target argmax, a tied maximum that is not the first, another token, or -1."""
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    num_reqs = len(num_drafts)
    cu = np.zeros(num_reqs + 1, dtype=np.int32)
    cu[1:] = np.cumsum(np.asarray(num_drafts) + 1)
    n = int(cu[-1])
    logits = (torch.randn((n, vocab), generator=g) * 3).to(dtype)
    m = float(torch.tensor(float(logits.float().max()) + 4.0).to(dtype))
    tied = {}
    for row in rng.choice(n, size=max(1, n // 3), replace=False):
        a, b = sorted(rng.choice(vocab, size=2, replace=False))
        logits[row, b] = logits[row, a] = m
        tied[int(row)] = (int(a), int(b))
    argmax = logits.float().argmax(dim=-1)
    drafts = torch.from_numpy(rng.integers(0, vocab, size=n).astype(np.int32))
    for r in range(num_reqs):
        s = int(cu[r])
        for i in range(num_drafts[r]):
            pick = rng.random()
            if pick < 0.6:
                drafts[s + i + 1] = argmax[s + i]
            elif pick < 0.75 and (s + i) in tied:
                drafts[s + i + 1] = tied[s + i][1]           # equal value, higher index: rejected
            elif pick < 0.85:
                drafts[s + i + 1] = -1                       # placeholder / invalid draft
    idx_mapping = torch.from_numpy(rng.permutation(16)[:num_reqs].astype(np.int32))
    expanded = torch.repeat_interleave(idx_mapping, torch.from_numpy(np.diff(cu)).long())
    local_pos = torch.cat([torch.arange(d + 1, dtype=torch.int32) for d in num_drafts])
    return dict(
        logits=logits.to(DEV),
        argmax=argmax.to(DEV),
        drafts=drafts.to(DEV),
        cu=torch.from_numpy(cu).to(DEV),
        idx_mapping=idx_mapping.to(DEV),
        expanded=expanded.to(DEV),
        local_pos=local_pos.to(DEV),
        pos=torch.arange(n, dtype=torch.int64, device=DEV) + 1000,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_greedy_verify_equals_rejection_sample(dtype):
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import greedy_verify, rejection_sample

    shapes = [[K], [K, K], [K, 3, 0, K], [0], [1, K, 2], [K] * 7, [0, 0, K]]
    temperature = torch.zeros(16, dtype=torch.float32, device=DEV)
    seeds = torch.arange(16, dtype=torch.int64, device=DEV)
    for seed in range(24):
        num_drafts = shapes[seed % len(shapes)]
        b = make_verify_batch(seed, num_drafts, dtype)
        want, want_n = rejection_sample(
            b["logits"], None, b["drafts"], b["cu"], b["pos"], b["idx_mapping"], b["expanded"], b["local_pos"],
            temperature, seeds, K,
        )
        got, got_n = greedy_verify(b["argmax"], b["drafts"], b["cu"], K)
        assert got.shape == want.shape and got.dtype == want.dtype == torch.int64
        assert got_n.dtype == want_n.dtype == torch.int32
        assert torch.equal(got_n, want_n), (seed, got_n.tolist(), want_n.tolist())
        for r, k in enumerate(want_n.tolist()):
            assert torch.equal(got[r, :k], want[r, :k]), (seed, r)


# --- (3) the runner's fast path end to end ----------------------------------------------------------------------------

def _uva(values, np_dtype, torch_dtype):
    return SimpleNamespace(np=np.array(values, dtype=np_dtype), gpu=torch.tensor(values, dtype=torch_dtype, device=DEV))


class FakeSampler:
    """The Sampler state GreedyLocalArgmax and RejectionSampler read. apply_sampling_params is the identity, as the
    real one is when no request of the batch needs logits processing (the fast path's condition)."""

    def __init__(self, max_num_reqs, vocab, prefill_len):
        n = max_num_reqs
        self.sampling_states = SimpleNamespace(
            vocab_size=vocab,
            temperature=_uva([0.0] * n, np.float32, torch.float32),
            top_k=_uva([vocab] * n, np.int32, torch.int32),
            top_p=_uva([1.0] * n, np.float32, torch.float32),
            min_p=_uva([0.0] * n, np.float32, torch.float32),
            seeds=_uva(list(range(n)), np.int64, torch.int64),
            num_logprobs=np.full(n, -1, dtype=np.int32),
        )
        self.sampling_states.max_num_logprobs = lambda idx: int(np.max(self.sampling_states.num_logprobs[idx]))
        self.num_token_ids = np.zeros(n, dtype=np.int32)
        self.logprob_token_ids_state = SimpleNamespace(
            max_num_token_ids=lambda idx: int(self.num_token_ids[idx].max(initial=0))
        )
        self.needs_logits_processing = np.zeros(n, dtype=bool)
        self.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=prefill_len))
        self.compute_nans = False
        self.return_sampling_mask = False
        self.logprobs_mode = "raw_logprobs"
        self.use_fp64_gumbel = False

    def apply_sampling_params(self, logits, *args, **kwargs):
        return logits


class FakeModel:
    """compute_logits / compute_logits_argmax over a table of logits, one row per token (the hidden state of a token
    is its row id); the argmax goes through get_top_tokens at an emulated TP=2."""

    def __init__(self, full, monkeypatch):
        self.full = full
        self.table, _, _ = padded_table(full, TP)
        self.monkeypatch = monkeypatch

    def compute_logits(self, hidden_states):
        return self.table[hidden_states[:, 0].long(), : self.full.shape[1]]

    def compute_logits_argmax(self, hidden_states):
        rows = hidden_states[:, 0].long()
        ids = tp_top_tokens(self.monkeypatch, self.table[rows], TP, self.full.shape[1], torch.arange(rows.numel(), device=DEV))
        assert torch.equal(ids[0], ids[1])
        return ids[0]


def make_step(spec, max_num_reqs=8, seed=0):
    """A runner step: per request (slot, query tokens, logits rows, first position, prefill length). Logits rows sit
    at the end of each request's query; the draft rows hold drafts against the target argmax."""
    rng = np.random.default_rng(seed)
    num_tokens = sum(q for _, q, _, _, _ in spec)
    g = torch.Generator().manual_seed(seed)
    full = (torch.randn((num_tokens, V), generator=g) * 3).to(torch.bfloat16)
    m = float(torch.tensor(float(full.float().max()) + 4.0).to(torch.bfloat16))
    full[0, 77] = full[0, 66] = m                                      # a tie in the first token's row
    argmax = full.float().argmax(dim=-1)
    input_ids = torch.from_numpy(rng.integers(0, V, size=num_tokens).astype(np.int32))
    positions, logits_indices, cu, local_pos, expanded = [], [], [0], [], []
    prefill_len = torch.zeros(max_num_reqs, dtype=torch.int32)
    seq_lens, idx_mapping = [], []
    t = 0
    for slot, q, nl, p0, pl in spec:
        positions += list(range(p0, p0 + q))
        rows = list(range(t + q - nl, t + q))
        logits_indices += rows
        for i, row in enumerate(rows[:-1]):                             # draft i sits in the row after its target row
            input_ids[row + 1] = int(argmax[row]) if rng.random() < 0.7 else int(rng.integers(0, V))
        cu.append(cu[-1] + nl)
        local_pos += list(range(nl))
        expanded += [slot] * nl
        prefill_len[slot] = pl
        seq_lens.append(p0 + q)
        idx_mapping.append(slot)
        t += q
    num_reqs = len(spec)
    num_logits = len(logits_indices)
    hidden = torch.arange(num_tokens, dtype=torch.float32).view(-1, 1)
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        idx_mapping=torch.tensor(idx_mapping, dtype=torch.int32, device=DEV),
        idx_mapping_np=np.array(idx_mapping, dtype=np.int32),
        expanded_idx_mapping=torch.tensor(expanded, dtype=torch.int32, device=DEV),
        expanded_local_pos=torch.tensor(local_pos, dtype=torch.int32, device=DEV),
        input_ids=input_ids.to(DEV),
        positions=torch.tensor(positions, dtype=torch.int64, device=DEV),
        logits_indices=torch.tensor(logits_indices, dtype=torch.int64, device=DEV),
        cu_num_logits=torch.tensor(cu, dtype=torch.int32, device=DEV),
        cu_num_logits_np=np.array(cu, dtype=np.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        num_draft_tokens=num_logits - num_reqs,
        has_structured_output_reqs=False,
    )
    return full.to(DEV), hidden.to(DEV), batch, prefill_len.to(DEV)


VERIFY_STEP = [
    # slot, query tokens, logits rows, first position, prefill length
    (5, K + 1, K + 1, 300, 40),     # decode, full window
    (2, K + 1, K + 1, 90, 91),      # first window after a remote prefill: its drafts are placeholders
    (7, 4, 4, 512, 100),            # three drafts
    (0, 9, 1, 0, 64),               # chunked prefill, no drafts: nothing sampled
    (3, K + 1, K + 1, 1000, 10),    # decode, depth-policy dead rows below
]


def test_fast_path_equals_stock_rejection_sampler(monkeypatch):
    from vllm.v1.worker.gpu.sample.greedy_local_argmax import GreedyLocalArgmax
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    for seed in range(6):
        full, hidden, batch, prefill_len = make_step(VERIFY_STEP, seed=seed)
        sampler = FakeSampler(8, V, prefill_len)
        spec_config = SimpleNamespace(
            num_speculative_tokens=K,
            enable_adaptive_verification=False,
            rejection_sample_method="standard",
            synthetic_acceptance_rates=None,
        )
        stock = RejectionSampler(sampler, spec_config, torch.device(DEV))
        model = FakeModel(full, monkeypatch)
        fast = GreedyLocalArgmax(model, sampler, stock, SimpleNamespace(), torch.device(DEV))
        # depth-policy dead rows: the last two rows of the last request, and seed-dependent extras
        dead = torch.zeros(batch.logits_indices.shape[0], dtype=torch.bool, device=DEV)
        dead[-2:] = True
        if seed % 2:
            dead[3] = True
        for invalid in (None, dead):
            want = stock(model.compute_logits(hidden[batch.logits_indices]), batch, None, invalid)
            got = fast.sample(hidden, batch, invalid)
            assert torch.equal(got.num_sampled, want.num_sampled), (seed, got.num_sampled.tolist())
            assert torch.equal(got.num_rejected, want.num_rejected)
            assert got.sampled_token_ids.shape == want.sampled_token_ids.shape
            for r, k in enumerate(want.num_sampled.tolist()):
                assert torch.equal(got.sampled_token_ids[r, :k], want.sampled_token_ids[r, :k]), (seed, r)
            assert got.logprobs_tensors is None and got.num_nans is None
        assert want.num_sampled[3] == 0                                 # the chunked prefill


def test_fast_path_equals_stock_plain_sampler(monkeypatch):
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.greedy_local_argmax import GreedyLocalArgmax

    spec = [(5, 1, 1, 300, 40), (2, 1, 1, 90, 10), (0, 9, 1, 0, 64), (6, 1, 1, 7, 3)]
    full, hidden, batch, prefill_len = make_step(spec, seed=11)
    sampler = FakeSampler(8, V, prefill_len)
    model = FakeModel(full, monkeypatch)
    fast = GreedyLocalArgmax(model, sampler, None, SimpleNamespace(), torch.device(DEV))
    got = fast.sample(hidden, batch, None)
    tokens = stock_greedy_sample(model.compute_logits(hidden[batch.logits_indices]))
    want_n, want_rej = get_num_sampled_and_rejected(
        batch.seq_lens.new_ones(batch.num_reqs), batch.seq_lens, batch.cu_num_logits, batch.idx_mapping, prefill_len
    )
    assert torch.equal(got.sampled_token_ids, tokens.view(-1, 1))
    assert torch.equal(got.num_sampled, want_n) and torch.equal(got.num_rejected, want_rej)
    assert got.num_sampled.tolist() == [1, 1, 0, 1]


def test_applies_only_to_plain_greedy_steps():
    from vllm.v1.worker.gpu.sample.greedy_local_argmax import GreedyLocalArgmax

    sampler = FakeSampler(8, V, torch.zeros(8, dtype=torch.int32, device=DEV))
    prompt = SimpleNamespace(uses_prompt_logprobs=np.zeros(8, dtype=bool))
    fast = GreedyLocalArgmax(None, sampler, None, prompt, torch.device(DEV))
    batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([3, 1], dtype=np.int32),
        logits_indices=torch.zeros(2, dtype=torch.int64),
        has_structured_output_reqs=False,
    )
    st = sampler.sampling_states
    assert fast.applies(batch, None)
    st.temperature.np[5] = 0.8                                         # outside the batch: no effect
    assert fast.applies(batch, None)
    mutations = [
        (st.temperature.np, 1, 0.7),
        (st.top_k.np, 3, 50),
        (st.top_p.np, 1, 0.9),
        (st.min_p.np, 3, 0.05),
        (sampler.needs_logits_processing, 1, True),                    # penalties, bias, bad words, budget, ...
        (st.num_logprobs, 3, 0),                                        # logprobs=0 still returns the sampled one
        (sampler.num_token_ids, 1, 2),                                  # logprob token ids
        (prompt.uses_prompt_logprobs, 3, True),
    ]
    for arr, i, value in mutations:
        old = arr[i]
        arr[i] = value
        assert not fast.applies(batch, None), (i, value)
        arr[i] = old
        assert fast.applies(batch, None)
    assert not fast.applies(batch, SimpleNamespace())                  # a grammar bitmask this step
    batch.has_structured_output_reqs = True
    assert not fast.applies(batch, None)
    batch.has_structured_output_reqs = False
    fast.log_every = 3                                                 # VLLM_GREEDY_LOCAL_ARGMAX_LOG: same answers
    assert [fast.applies(batch, g) for g in (None, SimpleNamespace(), None, None)] == [True, False, True, True]
    assert fast._steps == 4 and fast._hits == 1
    batch.logits_indices = torch.zeros(0, dtype=torch.int64)
    assert not fast.applies(batch, None)


def test_factory_gates_on_configuration(monkeypatch):
    from vllm.v1.worker.gpu.sample.greedy_local_argmax import (
        GreedyLocalArgmax,
        maybe_create_greedy_local_argmax,
    )
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    def sampler(**kw):
        s = Sampler.__new__(Sampler)
        s.compute_nans, s.return_sampling_mask, s.trace_replay_state = False, False, None
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    def verifier(cls=RejectionSampler, **kw):
        r = cls.__new__(cls)
        r.synthetic_conditional_rates, r.enable_adaptive_verification, r.watermark_key = None, False, None
        r.num_speculative_steps = K
        for k, v in kw.items():
            setattr(r, k, v)
        return r

    model = SimpleNamespace(compute_logits_argmax=lambda h: None)
    spec = SimpleNamespace(draft_logits=None, acceptance_estimator=None)
    prompt = SimpleNamespace(uses_prompt_logprobs=np.zeros(4, dtype=bool))

    def make(**kw):
        return maybe_create_greedy_local_argmax(
            kw.get("model", model), kw.get("sampler", sampler()), kw.get("verifier", verifier()), prompt,
            kw.get("spec", spec), lora=kw.get("lora", False), batch_sharded=kw.get("sharded", False),
            device=torch.device(DEV),
        )

    monkeypatch.delenv("VLLM_GREEDY_LOCAL_ARGMAX", raising=False)
    assert make() is None                                               # off by default
    monkeypatch.setenv("VLLM_GREEDY_LOCAL_ARGMAX", "1")
    assert isinstance(make(), GreedyLocalArgmax)
    assert isinstance(make(verifier=None, spec=None), GreedyLocalArgmax)

    class WatermarkLike(Sampler):
        pass

    class CustomVerifier(RejectionSampler):
        pass

    blocked = [
        dict(model=SimpleNamespace()),
        dict(sampler=sampler(compute_nans=True)),
        dict(sampler=sampler(return_sampling_mask=True)),
        dict(sampler=sampler(trace_replay_state=object())),
        dict(sampler=WatermarkLike.__new__(WatermarkLike)),
        dict(verifier=verifier(synthetic_conditional_rates=torch.ones(K))),
        dict(verifier=verifier(enable_adaptive_verification=True)),
        dict(verifier=verifier(watermark_key=7)),
        dict(verifier=verifier(CustomVerifier)),
        dict(spec=SimpleNamespace(draft_logits=torch.zeros(1), acceptance_estimator=None)),
        dict(spec=SimpleNamespace(draft_logits=None, acceptance_estimator=object())),
        dict(lora=True),
        dict(sharded=True),
    ]
    for kw in blocked:
        assert make(**kw) is None, kw


def test_model_classes_route_to_get_top_tokens():
    model_mod = pytest.importorskip("vllm.models.deepseek_v41.nvidia.model")
    vl_mod = pytest.importorskip("vllm.models.deepseek_v41.nvidia.vl_model")
    lm = SimpleNamespace(
        lm_head="head", logits_processor=SimpleNamespace(get_top_tokens=lambda head, h: ("top", head, h))
    )
    assert model_mod.DeepseekV41LLMForCausalLM.compute_logits_argmax(lm, "h") == ("top", "head", "h")
    vl = SimpleNamespace(language_model=SimpleNamespace(compute_logits_argmax=lambda h: ("lm", h)))
    assert vl_mod.DeepseekV41ForCausalLM.compute_logits_argmax(vl, "h") == ("lm", "h")
