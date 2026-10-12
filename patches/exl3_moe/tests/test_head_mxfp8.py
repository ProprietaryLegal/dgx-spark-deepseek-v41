# SPDX-License-Identifier: Apache-2.0
"""MXFP8 Marlin lm_head (vllm_exl3_moe/head.py) == F.linear with the MXFP8-rounded weight (W8A16: only accumulation
order differs), and its decode-size timing vs the BF16 head, at DeepSeek-V4.1's per-rank head at TP=2 (64,640 x 5,120).

Run inside the serve image on a Spark, with this checkout first on the path:
  PYTHONPATH=/plugin python3 -m pytest -q -s test_head_mxfp8.py
"""
import time

import pytest
import torch
from torch.nn.parameter import Parameter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pytest.importorskip("vllm")

N, K = 64640, 5120
dev = torch.device("cuda")


def make_layer(seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    layer = torch.nn.Module()
    layer.weight = Parameter(w.clone(), requires_grad=False)
    return layer, w


@pytest.fixture(scope="module")
def head():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Marlin needs sm_80+")
    from vllm_exl3_moe.head import Mxfp8MarlinHeadMethod
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16, mxfp8_e4m3_quantize

    layer, w = make_layer()
    q, s = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    wq = dequant_mxfp8_to_bf16(q, s)
    m = Mxfp8MarlinHeadMethod()
    m.process_weights_after_loading(layer)
    return m, layer, w, wq


def test_bf16_copy_dropped(head):
    m, layer, w, wq = head
    assert layer.weight.numel() == 0
    assert layer.mxfp8_nk == (N, K)


@pytest.mark.parametrize("tokens", [1, 6, 8, 64])
def test_matches_rounded_weight(head, tokens):
    m, layer, w, wq = head
    x = torch.randn(tokens, K, device=dev, dtype=torch.bfloat16)
    out = m.apply(layer, x).float()
    ref_q = torch.nn.functional.linear(x.float(), wq.float())
    ref_b = torch.nn.functional.linear(x.float(), w.float())
    rel_q = ((out - ref_q).norm() / ref_q.norm()).item()
    rel_b = ((out - ref_b).norm() / ref_b.norm()).item()
    print(f"tokens {tokens}: vs MXFP8-rounded weight {rel_q:.2e}, vs BF16 head {rel_b:.2e}")
    assert out.shape == (tokens, N)
    assert rel_q < 5e-3                     # same weights, BF16 output rounding
    assert rel_b < 0.05                     # the MXFP8 rounding itself (~2.6% on Gaussian weights)


def test_decode_timing(head):
    m, layer, w, wq = head
    for tokens in (1, 6, 8):
        x = torch.randn(tokens, K, device=dev, dtype=torch.bfloat16)
        res = {}
        for name, fn in (("bf16 F.linear", lambda: torch.nn.functional.linear(x, w)),
                         ("mxfp8 marlin", lambda: m.apply(layer, x))):
            for _ in range(5):
                fn()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                g.replay()
            torch.cuda.synchronize()
            res[name] = (time.perf_counter() - t0) / 100 * 1e6
        print(f"lm_head tokens={tokens}: " + ", ".join(f"{k} {v:.0f} us" for k, v in res.items()))
