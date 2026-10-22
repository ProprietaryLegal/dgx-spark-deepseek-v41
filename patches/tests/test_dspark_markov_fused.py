# SPDX-License-Identifier: Apache-2.0
"""Patch 0031: the local draft argmax with the Markov bias fused in (MarkovBiasCache.argmax_pair).

Against the unfused path the speculator runs without VLLM_DSPARK_MARKOV_FUSED (the cache's bias rows, a bf16 add,
torch.max, the stacked pair), bit for bit: cached, uncached and mixed rows at 1-6 rows, a padded logits row stride,
forced ties inside one 128-column block and across blocks (the lowest index wins, as torch.max), and CUDA-graph replay
with tokens that move in and out of the cache between replays.

Run inside the dsv41 image on a GPU:  python3 -m pytest -q test_dspark_markov_fused.py
(On a machine without vLLM, the kernels were checked the same way against a copy of the unfused kernel, Oct 21.)
"""
import json

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pytest.importorskip("vllm")   # runs in the serve image
DEV = "cuda"
VOCAB, R = 129_280, 256       # V4.1: the previous-token vocabulary, Markov rank 256
V, LO = 64_640, 64_640        # rank 1's vocabulary shard at TP=2


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    import os

    from vllm.models.deepseek_v41.nvidia.ops.markov_cache import MarkovBiasCache

    g = torch.Generator(device=DEV).manual_seed(0)
    w1 = (torch.randn((VOCAB, R), device=DEV, generator=g) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn((V, R), device=DEV, generator=g) * 0.05).to(torch.bfloat16)
    toks = torch.randperm(VOCAB, device=DEV, generator=g)[:512].tolist()
    f = tmp_path_factory.mktemp("markov") / "tokens.json"
    f.write_text(json.dumps(toks))
    os.environ["VLLM_DSPARK_MARKOV_CACHE"] = str(f)
    os.environ["VLLM_DSPARK_MARKOV_CACHE_ROWS"] = "512"
    return MarkovBiasCache(w1, w2, torch.bfloat16), w1, toks


def unfused(c, ids, emb, base, vocab, lo):
    bias = c(ids, emb)
    logits = base[:, :vocab] + bias
    vals, idx = logits.max(dim=-1)
    return torch.stack([vals.float(), (idx + lo).float()], dim=-1)


def ids_for(c, toks, rows, kind, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    hit = torch.tensor(toks, device=DEV)[torch.randint(0, len(toks), (rows,), device=DEV, generator=g)]
    miss = torch.randint(0, VOCAB, (4 * rows,), device=DEV, generator=g)
    miss = miss[c.slot_of_token[miss] < 0][:rows]
    return {"hit": hit, "miss": miss, "mixed": torch.where(torch.arange(rows, device=DEV) % 2 == 0, hit, miss)}[kind]


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 6])
@pytest.mark.parametrize("kind", ["hit", "miss", "mixed"])
def test_matches_unfused(cache, rows, kind):
    c, w1, toks = cache
    ids = ids_for(c, toks, rows, kind, 100 * rows + len(kind))
    emb = w1[ids].contiguous()
    base = (torch.randn((rows, V + 64), device=DEV) * 3).to(torch.bfloat16)   # padded shard columns
    got = c.argmax_pair(ids, emb, base, V, LO)
    assert torch.equal(got, unfused(c, ids, emb, base, V, LO))


@pytest.mark.parametrize("a,b", [(100, 120), (1500, 40000), (0, V - 1)])
def test_ties_lowest_index(cache, a, b):
    c, w1, toks = cache
    ids = torch.tensor(toks[:3], device=DEV)
    emb = w1[ids].contiguous()
    bias = c(ids, emb)
    base = torch.full((3, V), -20.0, device=DEV).to(torch.bfloat16)
    for r in range(3):          # the same bf16 sum at columns a < b, above every other column
        base[r, a] = (30.0 - bias[r, a].float()).to(torch.bfloat16)
        target = (base[r, a].float() + bias[r, a].float()).to(torch.bfloat16).float()
        x = (30.0 - bias[r, b].float()).to(torch.bfloat16)
        for _ in range(64):
            s = (x.float() + bias[r, b].float()).to(torch.bfloat16).float()
            if s == target:
                break
            x = (x.float() + (0.125 if s < target else -0.125)).to(torch.bfloat16)
        base[r, b] = x
    got = c.argmax_pair(ids, emb, base, V, LO)
    assert torch.equal(got, unfused(c, ids, emb, base, V, LO))
    assert (got[:, 1] == LO + a).all()


def test_graph_replay(cache):
    c, w1, toks = cache
    rows = 4
    ids = ids_for(c, toks, rows, "mixed", 7)
    emb = w1[ids].contiguous()
    base = torch.randn((rows, V), device=DEV).to(torch.bfloat16)
    c.argmax_pair(ids, emb, base, V, LO)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g):
        out = c.argmax_pair(ids, emb, base, V, LO)
    torch.cuda.current_stream().wait_stream(s)
    for k in range(3):          # tokens move in and out of the cache between replays
        nid = ids_for(c, toks, rows, ("hit", "miss", "mixed")[k], 50 + k)
        ids.copy_(nid)
        emb.copy_(w1[nid])
        base.copy_(torch.randn((rows, V), device=DEV).to(torch.bfloat16))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, unfused(c, ids, emb, base, V, LO))
