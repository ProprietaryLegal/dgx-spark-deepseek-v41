# SPDX-License-Identifier: Apache-2.0
"""Patch 0008: DSpark top-k Markov bias for DeepSeek-V4.1. The shared DSparkMarkovHead.apply_bias_gathered, at V4.1's
shapes (129,280 x 256, BF16 markov_w2 as V4.1 builds it: unquantized), equals base + full Markov bias on the top-K
base-logit candidates, leaves every other token at -inf, and so picks the same draft token whenever the full-vocab
argmax is a candidate; DSparkDeepseekV4ForCausalLM.apply_markov_bias_gathered delegates to it.

Run inside the dsv41 image (GPU if present, else CPU):  python3 -m pytest -q test_dspark_topk_markov.py
"""
import types

import pytest
import torch

pytest.importorskip("vllm")   # runs in the serve image

from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead

V, R = 129280, 256
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def head_with(weight):
    h = DSparkMarkovHead.__new__(DSparkMarkovHead)
    torch.nn.Module.__init__(h)
    # V4.1 builds markov_w2 without a quant config: a plain BF16 weight, no NVFP4 copy retained for gathering
    object.__setattr__(h, "markov_w2", types.SimpleNamespace(weight=weight, _retain_weight_for_gather=False,
                                                             is_w4a16_nvfp4=False))
    return h


@pytest.mark.parametrize("k", [16, 64])
def test_gathered_bias_matches_full_on_candidates(k):
    g = torch.Generator(device=dev).manual_seed(k)
    w2 = (torch.randn(V, R, device=dev, generator=g) * 0.02).bfloat16()
    emb = torch.randn(3, R, device=dev, generator=g).bfloat16()
    base = (torch.randn(3, V, device=dev, generator=g) * 4).bfloat16()
    full = base.float() + emb.float() @ w2.float().t()                    # what markov_bias + add gives
    vals, idx = base.topk(k, dim=-1)
    out = head_with(w2).apply_bias_gathered(emb, torch.full_like(base, float("-inf")), vals, idx)
    got, want = out.gather(1, idx).float(), full.gather(1, idx)
    assert (got - want).abs().max().item() < 0.1                          # bf16 rounding of the corrected values
    rest = torch.ones_like(out, dtype=torch.bool).scatter_(1, idx, False)
    assert torch.isinf(out[rest]).all() and (out[rest] < 0).all()
    top2 = want.topk(2, dim=-1).values
    clear = (top2[:, 0] - top2[:, 1]) > 0.1                               # skip near-ties bf16 could flip
    inside = (idx == full.argmax(-1, keepdim=True)).any(-1)
    pick = idx.gather(1, got.argmax(-1, keepdim=True)).squeeze(-1)
    assert (pick[clear & inside] == full.argmax(-1)[clear & inside]).all()


def test_v41_model_delegates():
    from vllm.models.deepseek_v41.nvidia.dspark import DSparkDeepseekV4ForCausalLM
    m = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    calls = []

    class Head:
        def apply_bias_gathered(self, *args):
            calls.append(args)
            return "gathered"
    object.__setattr__(m, "model", types.SimpleNamespace(markov_head=Head()))
    assert m.apply_markov_bias_gathered("embed", "logits", "values", "index") == "gathered"
    assert calls == [("embed", "logits", "values", "index")]
