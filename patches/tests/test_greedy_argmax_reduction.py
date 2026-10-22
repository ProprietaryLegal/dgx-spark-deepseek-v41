# SPDX-License-Identifier: Apache-2.0
"""Patch 0027: the vocab-parallel (value, index) argmax reduction, in plain torch (no vLLM; CPU, and the GPU when present).

The target's compute_logits_argmax is vLLM's LogitsProcessor.get_top_tokens: each TP rank sets its padding columns to -inf,
takes the max and argmax of its shard, and all-gathers one (value as fp32, global index as fp32) pair per row; the rank
with the larger value wins, the lower rank on equal values. `pair_argmax` is that reduction line for line, the all-gather
emulated by concatenating the per-rank pairs. For every row it must return the first index of the maximum over the real
vocabulary columns of the gathered logits (torch.argmax), which is what the stock greedy sampler and greedy verifier
compute in two stages (per vocab block the max and its lowest index, then the first block among equal block maxima),
emulated by `blockwise_first_argmax` at both of their block widths. Covered: 2 and 4 shards with and without padding
columns (whose values, +inf here, must never win), exact ties inside a shard, across shards, across vocab-block
boundaries and between the last column of one shard and the first of the next, a constant row, an all -inf row, signed
zeros, +inf, bf16 and fp32, 1 to 16 rows. The stock Triton kernels themselves are compared in test_greedy_local_argmax.py
(serve image).

Runs anywhere:  python3 -m pytest -q test_greedy_argmax_reduction.py
"""
import pytest
import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
PAD_TO = 64                                   # vLLM's DEFAULT_VOCAB_PADDING_SIZE
LAYOUTS = [(129_280, 2), (129_277, 2), (1_000, 2), (129_277, 4)]   # (vocab, tp): no padding, 3 / 24 / 3 padding columns


def shard_layout(vocab, tp):
    """Per-rank width and [(org_vocab_start_index, num_org_vocab_padding)] of a ParallelLMHead's shards."""
    padded = -(-vocab // PAD_TO) * PAD_TO
    assert padded % tp == 0
    per = padded // tp
    layout = []
    for r in range(tp):
        start, end = min(r * per, vocab), min((r + 1) * per, vocab)
        layout.append((start, per - (end - start)))
    return per, layout


def shard(full, tp, pad_value=float("inf")):
    """The TP shards a sharded lm_head produces for these logits: real columns, then padding columns (here +inf, so an
    unmasked padding column would win)."""
    rows, vocab = full.shape
    per, layout = shard_layout(vocab, tp)
    padded = torch.full((rows, per * tp), pad_value, dtype=full.dtype, device=full.device)
    padded[:, :vocab] = full
    return [padded[:, r * per:(r + 1) * per] for r in range(tp)], layout


def pair_argmax(shards, layout):
    """LogitsProcessor.get_top_tokens after the head GEMM (scale 1, no soft cap), the all-gather emulated."""
    tp = len(shards)
    pairs = []
    for logits, (start, num_pad) in zip(shards, layout):
        logits = logits.clone()
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        local_max_vals, local_max_indices = logits.max(dim=-1)
        global_indices = local_max_indices + start
        pairs.append(torch.stack([local_max_vals.float(), global_indices.float()], dim=-1))
    gathered = torch.cat(pairs, dim=-1).view(shards[0].shape[0], tp, 2)   # [rows, 2 * tp] -> [rows, tp, 2]
    max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(dim=-1, index=max_rank_idx).squeeze(-1).to(torch.int64)


def blockwise_first_argmax(full, block):
    """The stock greedy argmax: logits read as fp32, out-of-range columns -inf, per block the max and its lowest index,
    then the first block among equal block maxima (gumbel_sample: 1024-column blocks; rejection_sample: 8192)."""
    rows, vocab = full.shape
    nblk = -(-vocab // block)
    x = torch.full((rows, nblk * block), -float("inf"), dtype=torch.float32, device=full.device)
    x[:, :vocab] = full.float()
    vals, idx = x.view(rows, nblk, block).max(dim=-1)
    best = vals.argmax(dim=-1, keepdim=True)
    return best.squeeze(-1) * block + idx.gather(-1, best).squeeze(-1)


def planted_logits(rows, vocab, tp, dtype, device, seed):
    """Random logits (bf16 has many exact duplicates by itself) with one planted case per row (row r gets case r % 13).
    Returns the logits and {row: expected token}."""
    g = torch.Generator().manual_seed(seed)
    full = (torch.randn((rows, vocab), generator=g) * 3).to(dtype)
    per, _ = shard_layout(vocab, tp)
    m = float(full.float().max()) + 4.0              # above every random value, exact in bf16 (a small integer + 4)
    m = float(torch.tensor(m).to(dtype))
    want = {}
    for r in range(rows):
        case = r % 13
        row = full[r]
        if case == 1:                                  # tie inside shard 0: the lower index
            row[10] = row[20] = m
            want[r] = 10
        elif case == 2 and per + 30 < vocab:           # tie across shards: shard 0 (the lower index)
            row[per + 30] = row[30] = m
            want[r] = 30
        elif case == 3 and per + 9 < vocab:            # tie inside shard 1
            row[per + 9] = row[per + 5] = m
            want[r] = per + 5
        elif case == 4:                                # the last real column
            row[vocab - 1] = m
            want[r] = vocab - 1
        elif case == 5:                                # first and last column tie
            row[vocab - 1] = row[0] = m
            want[r] = 0
        elif case == 6:                                # constant row
            row.fill_(1.5)
            want[r] = 0
        elif case == 7:                                # all -inf
            row.fill_(-float("inf"))
            want[r] = 0
        elif case == 8 and per + 1 < vocab:            # signed zeros are equal: the first one
            row.fill_(-1.0)
            row[per + 1] = 0.0
            row[7] = 0.0
            row[3] = -0.0
            want[r] = 3
        elif case == 9 and per + 50 < vocab:           # +inf ties
            row[per + 50] = row[50] = float("inf")
            want[r] = 50
        elif case == 10 and vocab > 8193:              # ties across the stock kernels' block boundaries
            row[8192] = row[1024] = row[1023] = m
            want[r] = 1023
        elif case == 11 and per < vocab:               # last column of shard 0 ties first column of shard 1
            row[per] = row[per - 1] = m
            want[r] = per - 1
        elif case == 12:                               # many-way tie at the max, lowest in the middle of shard 0
            cols = torch.tensor([per - 2, 777, vocab - 2, 400]) % vocab
            row[cols] = m
            want[r] = int(cols.min())
    return full.to(device), want


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab,tp", LAYOUTS)
def test_pair_reduction_equals_argmax_over_real_columns(device, dtype, vocab, tp):
    for rows in range(1, 17):
        full, want = planted_logits(rows, vocab, tp, dtype, device, seed=rows * 7 + tp)
        shards, layout = shard(full, tp)
        got = pair_argmax(shards, layout)
        ref = full.argmax(dim=-1)                      # first index of the maximum over the real columns
        assert torch.equal(got, ref), (rows, (got != ref).nonzero().flatten().tolist())
        assert torch.equal(ref, blockwise_first_argmax(full, 1024))
        assert torch.equal(ref, blockwise_first_argmax(full, 8192))
        for r, tok in want.items():
            assert int(got[r]) == tok, (rows, r, int(got[r]), tok)


@pytest.mark.parametrize("device", DEVICES)
def test_padding_never_wins(device):
    """Every real column below a padding value that would win unmasked: the real maximum still comes out."""
    vocab, tp = 129_277, 2
    full = torch.full((3, vocab), -5.0, device=device)
    full[0, 17] = -1.0
    full[1, vocab - 1] = -1.0
    full[2] = -float("inf")                            # nothing real beats -inf: the first column
    shards, layout = shard(full, tp, pad_value=float("inf"))
    assert layout[-1][1] == 3
    assert pair_argmax(shards, layout).tolist() == [17, vocab - 1, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_large_index_exact_in_fp32(device):
    """Global indices travel as fp32: every index below 2**24 is exact (the vocabulary is 129,280)."""
    idx = torch.arange(0, 1 << 24, 4099, device=device)
    assert torch.equal(idx.float().to(torch.int64), idx)
    vocab, tp = 129_280, 2
    full = torch.zeros((1, vocab), device=device)
    full[0, vocab - 1] = 1.0
    shards, layout = shard(full, tp)
    assert pair_argmax(shards, layout).tolist() == [vocab - 1]
