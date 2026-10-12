# SPDX-License-Identifier: Apache-2.0
"""Patch 0011: DSpark Markov bias rows cached for frequent tokens.

The kernel against a float32 reference (markov_w2 @ markov_w1[t]), cached rows bit-identical to computed ones (the same
kernel fills the cache), a batch mixing cached and uncached tokens, and CUDA-graph replay with tokens that move in and
out of the cache between replays (the branch is on the device).

Run inside the dsv41 image on a GPU:  python3 -m pytest -q test_dspark_markov_cache.py
"""
import json

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pytest.importorskip("vllm")   # runs in the serve image
DEV = "cuda"
V, R = 129_280, 256          # V4.1: full vocabulary, Markov rank 256


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device=DEV).manual_seed(0)
    w1 = (torch.randn((V, R), device=DEV, generator=g) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn((V, R), device=DEV, generator=g) * 0.05).to(torch.bfloat16)
    return w1, w2


@pytest.fixture()
def cache(weights, monkeypatch, tmp_path):
    from vllm.models.deepseek_v41.nvidia.ops.markov_cache import MarkovBiasCache

    f = tmp_path / "tokens.json"
    f.write_text(json.dumps([5, 7, 7, 11, 2**20, 100, 129_279]))  # a duplicate and an out-of-vocab id are dropped
    monkeypatch.setenv("VLLM_DSPARK_MARKOV_CACHE", str(f))
    monkeypatch.setenv("VLLM_DSPARK_MARKOV_CACHE_ROWS", "4")
    w1, w2 = weights
    return MarkovBiasCache(w1, w2, torch.bfloat16)


def test_cache_contents(cache):
    assert cache.cache.shape == (4, V)
    slots = cache.slot_of_token
    assert slots[[5, 7, 11, 100]].tolist() == [0, 1, 2, 3]
    assert slots[129_279].item() == -1 and int((slots >= 0).sum()) == 4


def test_matches_reference_and_cached_rows_are_bit_identical(cache, weights):
    from vllm.models.deepseek_v41.nvidia.ops.markov_cache import markov_bias_rows

    w1, w2 = weights
    ids = torch.tensor([5, 3, 11, 129_279, 100, 0], device=DEV)      # cached, not, cached, not, cached, not
    emb = w1[ids]
    out = cache(ids, emb)
    ref = (emb.float() @ w2.float().t())
    rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    assert rel < 1e-2, rel
    # the same tokens with the cache bypassed (every slot -1): the kernel's computed rows
    none = torch.full_like(cache.slot_of_token, -1)
    plain = markov_bias_rows(ids, none, cache.cache, emb.contiguous(), w2, torch.empty_like(out))
    assert torch.equal(out, plain)


def test_graph_replay_follows_tokens(cache, weights):
    w1, w2 = weights
    ids = torch.tensor([5, 3], device=DEV)
    emb = w1[ids].contiguous()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        cache(ids, emb)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = cache(ids, emb)
    for toks in ([5, 3], [3, 5], [100, 11], [129_279, 0]):
        ids.copy_(torch.tensor(toks, device=DEV))
        emb.copy_(w1[ids])
        g.replay()
        torch.cuda.synchronize()
        ref = (w1[ids].float() @ w2.float().t())
        assert ((out.float() - ref).abs().max() / ref.abs().max()).item() < 1e-2, toks
