# SPDX-License-Identifier: Apache-2.0
"""Patch 0012: DSpark greedy drafts argmaxed per vocabulary shard.

(1) MarkovBiasCache over a contiguous row range of markov_w2 (one TP rank's columns) returns exactly those columns of the
full-vocab cache's rows, for cached and uncached tokens (same kernel, the reduction over the rank is per column).
(2) The per-shard reduction (local max + argmax, then the rank with the larger value, the lower rank on equal values)
picks the same token as torch.argmax over the concatenated logits, including exact ties inside a shard and across
shards. The all-gather itself is vLLM's (tensor_model_parallel_all_gather), emulated here by stacking the shards.

Run inside the dsv41 image on a GPU:  python3 -m pytest -q test_dspark_local_argmax.py
"""
import json

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pytest.importorskip("vllm")   # runs in the serve image
DEV = "cuda"
V, R, TP = 129_280, 256, 2


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device=DEV).manual_seed(1)
    w1 = (torch.randn((V, R), device=DEV, generator=g) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn((V, R), device=DEV, generator=g) * 0.05).to(torch.bfloat16)
    return w1, w2


def make(w1, w2, monkeypatch, tmp_path, rows):
    from vllm.models.deepseek_v41.nvidia.ops.markov_cache import MarkovBiasCache

    f = tmp_path / "tokens.json"
    f.write_text(json.dumps([5, 7, 11, 100, 129_279]))
    monkeypatch.setenv("VLLM_DSPARK_MARKOV_CACHE", str(f))
    monkeypatch.setenv("VLLM_DSPARK_MARKOV_CACHE_ROWS", str(rows))
    return MarkovBiasCache(w1, w2, torch.bfloat16)


def test_shard_cache_is_a_column_slice_of_the_full_one(weights, monkeypatch, tmp_path):
    w1, w2 = weights
    full = make(w1, w2, monkeypatch, tmp_path, 4)
    ids = torch.tensor([5, 3, 11, 129_279, 100, 0], device=DEV)   # cached, not, cached, not (5th id cut by ROWS), ...
    emb = w1[ids].contiguous()
    ref = full(ids, emb)
    per = V // TP
    for rank in range(TP):
        lo, hi = rank * per, (rank + 1) * per
        shard = make(w1, w2[lo:hi], monkeypatch, tmp_path, 4)
        assert shard.cache.shape == (4, hi - lo) and shard.slot_of_token.shape == (V,)
        assert torch.equal(shard.slot_of_token, full.slot_of_token)
        got = shard(ids, emb)
        assert torch.equal(got, ref[:, lo:hi]), f"rank {rank}: shard bias differs from the full rows' columns"


def reduce_shards(shards, starts):
    """The draft_argmax_local reduction with the all-gather emulated: [rows, V/TP] per rank -> token ids."""
    pairs = []
    for logits, lo in zip(shards, starts):
        vals, idx = logits.max(dim=-1)
        pairs.append(torch.stack([vals.float(), (idx + lo).float()], dim=-1))
    gathered = torch.cat(pairs, dim=-1).view(-1, len(shards), 2)
    best = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(dim=-1, index=best).squeeze(-1).to(torch.int64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shard_reduction_equals_full_argmax(dtype):
    g = torch.Generator(device=DEV).manual_seed(2)
    rows, per = 64, V // TP
    full = torch.randn((rows, V), device=DEV, generator=g).to(dtype)
    m = full.max().item() + 1
    full[0, 10] = full[0, 20] = m                   # tie inside shard 0: the lower index wins
    full[1, 30] = full[1, per + 30] = m             # tie across shards: shard 0 (lower index) wins
    full[2, per + 5] = full[2, per + 9] = m         # tie inside shard 1
    full[3, V - 1] = m                              # the last column
    shards = [full[:, r * per:(r + 1) * per] for r in range(TP)]
    got = reduce_shards(shards, [r * per for r in range(TP)])
    want = full.argmax(dim=-1)
    assert torch.equal(got, want), (got[:4].tolist(), want[:4].tolist())
    assert got[:4].tolist() == [10, 30, per + 5, V - 1]


def test_bias_add_then_reduce_matches_full_path(weights, monkeypatch, tmp_path):
    """The speculator's full path (gathered base logits + full bias, argmax) and the local path (shard base + shard
    bias, local reduction) pick the same draft tokens."""
    w1, w2 = weights
    full_cache = make(w1, w2, monkeypatch, tmp_path, 4)
    g = torch.Generator(device=DEV).manual_seed(3)
    ids = torch.tensor([5, 3, 11, 0, 77, 100], device=DEV)
    emb = w1[ids].contiguous()
    base = torch.randn((ids.numel(), V), device=DEV, generator=g).to(torch.bfloat16) * 4
    want = (base + full_cache(ids, emb)).argmax(dim=-1)
    per = V // TP
    shards = []
    for rank in range(TP):
        lo, hi = rank * per, (rank + 1) * per
        shard = make(w1, w2[lo:hi], monkeypatch, tmp_path, 4)
        shards.append(base[:, lo:hi] + shard(ids, emb))
    got = reduce_shards(shards, [r * per for r in range(TP)])
    assert torch.equal(got, want)
