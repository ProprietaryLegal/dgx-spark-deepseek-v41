# SPDX-License-Identifier: Apache-2.0
"""Patch 0036: VLLM_DSPARK_CONF_FEAT_HEAD, the confidence from the head's logit plus each draft's logit margin and
top-1 probability, the features read off the fused Markov argmax (MarkovBiasCache.argmax_record).

- argmax_record's first two columns are argmax_pair's bit for bit (cached, uncached and mixed rows, a padded logits row
  stride), its second-largest value is torch.topk's (a tie at the max counts twice), its logsumexp torch's to fp32
  rounding, and it replays in a CUDA graph;
- draft_features_from_records over two vocabulary shards gives the margin and top-1 probability of the whole
  vocabulary (what patch 0034's draft_logit_features logs);
- conf_feat_head is the logistic scripts/dspark/fit_draft_features.py fits and exports.

Run inside the dsv41 image on a GPU:  python3 -m pytest -q test_dspark_conf_feat_head.py
(The kernel tests also run on a machine without vLLM against a stand-in package holding markov_cache.py, Oct 21.)
"""
import json

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
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


def ids_for(c, toks, rows, kind, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    hit = torch.tensor(toks, device=DEV)[torch.randint(0, len(toks), (rows,), device=DEV, generator=g)]
    miss = torch.randint(0, VOCAB, (4 * rows,), device=DEV, generator=g)
    miss = miss[c.slot_of_token[miss] < 0][:rows]
    return {"hit": hit, "miss": miss, "mixed": torch.where(torch.arange(rows, device=DEV) % 2 == 0, hit, miss)}[kind]


def reference(c, ids, emb, base, vocab):
    """The unfused logits of the speculator's path, as fp32: (top-1, top-2, logsumexp) per row."""
    lf = (base[:, :vocab] + c(ids, emb)).float()
    top = lf.topk(2, dim=-1).values
    return top[:, 0], top[:, 1], torch.logsumexp(lf, dim=-1)


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 6])
@pytest.mark.parametrize("kind", ["hit", "miss", "mixed"])
def test_record_matches_pair_and_torch(cache, rows, kind):
    c, w1, toks = cache
    ids = ids_for(c, toks, rows, kind, 100 * rows + len(kind))
    emb = w1[ids].contiguous()
    base = (torch.randn((rows, V + 64), device=DEV) * 3).to(torch.bfloat16)   # padded shard columns
    rec = c.argmax_record(ids, emb, base, V, LO)
    assert rec.shape == (rows, 4) and rec.dtype == torch.float32
    assert torch.equal(rec[:, :2], c.argmax_pair(ids, emb, base, V, LO))
    t1, t2, lse = reference(c, ids, emb, base, V)
    assert torch.equal(rec[:, 0], t1)
    assert torch.equal(rec[:, 2], t2)
    torch.testing.assert_close(rec[:, 3], lse, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("cols", [(100, 120), (1500, 40000), (0, V - 1)])
def test_record_tie_at_max_counts_twice(cache, cols):
    c, w1, toks = cache
    ids = torch.tensor(toks[:2], device=DEV)
    emb = w1[ids].contiguous()
    bias = c(ids, emb)
    base = torch.full((2, V), -20.0, device=DEV).to(torch.bfloat16)
    for r in range(2):          # the same bf16 sum at both columns, above every other column
        for col in cols:
            base[r, col] = (30.0 - bias[r, col].float()).to(torch.bfloat16)
    lf = (base + bias).float()
    for r in range(2):
        if lf[r, cols[0]] != lf[r, cols[1]]:
            pytest.skip("bf16 rounding gave the two columns different sums")
    rec = c.argmax_record(ids, emb, base, V, LO)
    assert torch.equal(rec[:, 2], rec[:, 0])          # margin 0, as torch.topk
    assert torch.equal(rec[:, 1], torch.full((2,), float(cols[0] + LO), device=DEV))
    t1, t2, lse = reference(c, ids, emb, base, V)
    assert torch.equal(rec[:, 2], t2)
    torch.testing.assert_close(rec[:, 3], lse, rtol=2e-6, atol=2e-6)


def test_record_short_last_block(cache):
    c, w1, toks = cache
    vocab = 300                 # 2 full 128-column blocks and a 44-column tail
    ids = torch.tensor(toks[:3], device=DEV)
    emb = w1[ids].contiguous()
    base = (torch.randn((3, vocab), device=DEV) * 2).to(torch.bfloat16)
    rec = c.argmax_record(ids, emb, base, vocab, 0)
    assert torch.equal(rec[:, :2], c.argmax_pair(ids, emb, base, vocab, 0))
    lf = (base + c(ids, emb)[:, :vocab]).float()
    assert torch.equal(rec[:, 2], lf.topk(2, dim=-1).values[:, 1])
    torch.testing.assert_close(rec[:, 3], torch.logsumexp(lf, -1), rtol=2e-6, atol=2e-6)


def test_record_graph_replay(cache):
    c, w1, toks = cache
    rows = 3
    ids = torch.tensor(toks[:rows], device=DEV)
    emb = w1[ids].contiguous()
    base = (torch.randn((rows, V), device=DEV) * 3).to(torch.bfloat16)
    c.argmax_record(ids, emb, base, V, LO)            # compile outside the capture
    out = torch.empty((rows, 4), device=DEV)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g):
        out.copy_(c.argmax_record(ids, emb, base, V, LO))
    torch.cuda.current_stream().wait_stream(s)
    for seed in range(3):
        new = ids_for(c, toks, rows, "mixed", seed)
        ids.copy_(new)
        emb.copy_(w1[new])
        base.copy_((torch.randn((rows, V), device=DEV) * 3).to(torch.bfloat16))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[:, :2], c.argmax_pair(ids, emb, base, V, LO))
        t1, t2, lse = reference(c, ids, emb, base, V)
        assert torch.equal(out[:, 2], t2)
        torch.testing.assert_close(out[:, 3], lse, rtol=2e-6, atol=2e-6)


def test_features_from_two_shards():
    pytest.importorskip("vllm.models.deepseek_v41.nvidia.dspark")
    from vllm.models.deepseek_v41.nvidia.dspark import draft_features_from_records

    g = torch.Generator(device=DEV).manual_seed(5)
    reqs, k, half = 3, 5, 1000
    full = (torch.randn((reqs, k, 2 * half), device=DEV, generator=g) * 4).to(torch.bfloat16).float()
    full[0, 0, 7] = full[0, 0, 1500] = full[0, 0].max() + 1.0       # a tie across the two shards: margin 0
    recs = []
    for r in range(2):
        sh = full[..., r * half:(r + 1) * half]
        top = sh.topk(2, dim=-1).values
        vals, idx = sh.max(dim=-1)
        recs.append(torch.stack([vals, (idx + r * half).float(), top[..., 1], torch.logsumexp(sh, -1)], -1))
    rec = torch.stack(recs, dim=-2)                                   # [reqs, k, tp, 4]
    got = draft_features_from_records(rec)
    top = full.topk(2, dim=-1).values
    want = torch.stack([top[..., 0] - top[..., 1], torch.exp(top[..., 0] - torch.logsumexp(full, -1))], -1)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert got[0, 0, 0].item() == 0.0


def test_conf_feat_head_is_the_fitted_logistic(tmp_path):
    pytest.importorskip("vllm.v1.worker.gpu.spec_decode.dspark.speculator")
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import conf_feat_head

    rng = np.random.default_rng(3)
    k, reqs, eps = 5, 4, 1e-6
    w = rng.normal(size=(k, 4))
    np.savez(tmp_path / "feat.npz", w=w)
    wt = torch.tensor(np.load(tmp_path / "feat.npz")["w"], dtype=torch.float32, device=DEV)
    c = rng.uniform(0, 1, size=(reqs, k)); c[0, 0] = 0.0; c[1, 1] = 1.0           # clamped ends
    f = np.stack([rng.exponential(2.0, size=(reqs, k)), rng.uniform(0, 1, size=(reqs, k))], -1)
    f[2, 2, 1] = 1.0
    got = conf_feat_head(torch.tensor(c, dtype=torch.float32, device=DEV),
                         torch.tensor(f, dtype=torch.float32, device=DEV), wt).cpu().numpy()

    def logit(p):               # torch.logit's clamp: eps and 1 - eps in the tensor's dtype (fp32)
        p = np.clip(p.astype(np.float32), np.float32(eps), np.float32(1 - eps)).astype(np.float64)
        return np.log(p) - np.log1p(-p)

    u = w[:, 0] + w[:, 1] * logit(c) + w[:, 2] * f[..., 0] + w[:, 3] * logit(f[..., 1])
    np.testing.assert_allclose(got, 1 / (1 + np.exp(-u)), rtol=2e-5, atol=2e-6)


def test_policy_conf_feat_switch(monkeypatch, tmp_path):
    pytest.importorskip("vllm.v1.worker.gpu.spec_decode.depth_policy")
    from vllm.v1.worker.gpu.spec_decode.depth_policy import DSparkDepthPolicy, _parse_policy, _parse_table

    assert _parse_policy("conf+feat") == ("conf", 0) == _parse_policy("conf+feat:10,20")
    assert _parse_table("conf+feat:10,20,30") == [10.0, 20.0, 30.0] and _parse_table("conf+feat") is None
    table = "10,10.5,11,11.5,12,12.5"         # flat enough that confident drafts are all worth verifying
    f = tmp_path / "depth"
    f.write_text(f"conf:{table}")
    monkeypatch.setenv("VLLM_DSPARK_DEPTH_POLICY", "conf")
    monkeypatch.setenv("VLLM_DSPARK_DEPTH_FILE", str(f))
    p = DSparkDepthPolicy(5, 4, torch.device(DEV), has_confidence=True)
    idx = torch.arange(2, dtype=torch.int32, device=DEV)
    raw = torch.full((2, 5), 0.05, device=DEV)        # the head's own: verify nothing
    fitted = torch.full((2, 5), 0.999, device=DEV)    # the fitted ones: verify every draft
    p.record(raw, idx, 2, fitted=fitted)              # record 0 reads the file
    assert p.describe() == "conf" and p.depth[:2].tolist() == [0, 0]
    f.write_text(f"conf+feat:{table}")
    p.record(raw, idx, 2, fitted=fitted)              # mid-request: the file waits for step 64 or a new request
    assert p.describe() == "conf" and p.depth[:2].tolist() == [0, 0]
    p.reset(0)                                        # a request starts: its first verify step reads the file
    p.record(raw, idx, 2, fitted=fitted)
    assert p.describe() == "conf+feat" and p.depth[:2].tolist() == [5, 5]
    p.record(raw, idx, 2, fitted=None)                # conf+feat without fitted confidences: the raw ones
    assert p.depth[:2].tolist() == [0, 0]
    f.write_text(f"conf:{table}")
    p.reset(1)
    p.record(raw, idx, 2, fitted=fitted)
    assert p.describe() == "conf" and p.depth[:2].tolist() == [0, 0]
