# SPDX-License-Identifier: Apache-2.0
"""Patch 0004: Sm12xMarlinMxfp8BmmLinearKernel (wo_a on GB10) == the BF16 emulation it replaces (Marlin is W8A16: BF16
activations, so only accumulation-order differences), at DeepSeek-V4.1 TP=2 shapes (4 groups of 1024 x 4096), plus a decode-size timing of both.

Run inside the dsv41 image on a Spark:  python3 -m pytest -q -s test_sm12x_mxfp8_bmm.py
"""
import time

import pytest
import torch

pytest.importorskip("vllm")   # runs in the serve image

from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp8.sm12x_bmm import Sm12xMarlinMxfp8BmmLinearKernel
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16

G, N, K = 4, 1024, 4096
dev = torch.device("cuda")


def make_layer(seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(G * N, K, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(118, 126, (G * N, K // 32), device=dev, generator=g, dtype=torch.uint8)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)      # vLLM keeps MXFP8 scales as raw uint8
    ref_w = dequant_mxfp8_to_bf16(w, s)                                  # [G*N, K] bf16, what emulation keeps
    return layer, ref_w


def test_supported_on_this_gpu():
    ok, why = Sm12xMarlinMxfp8BmmLinearKernel.is_supported()
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM12x only")
    assert ok, why
    ok, why = Sm12xMarlinMxfp8BmmLinearKernel.can_implement(
        Mxfp8LinearLayerConfig(weight_shape=(G * N, K), bmm_batch_size=G))
    assert ok, why


@pytest.mark.parametrize("tokens", [1, 2, 6, 8, 64, 512])
def test_matches_bf16_emulation(tokens):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM12x only")
    layer, ref_w = make_layer()
    k = Sm12xMarlinMxfp8BmmLinearKernel(Mxfp8LinearLayerConfig(weight_shape=(G * N, K), bmm_batch_size=G))
    k.process_weights_after_loading(layer)
    x = torch.randn(tokens, G, K, device=dev, dtype=torch.bfloat16)
    out = k.apply_weights(layer, x)
    ref = torch.bmm(x.transpose(0, 1).float(), ref_w.view(G, N, K).transpose(1, 2).float()).transpose(0, 1)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert out.shape == (tokens, G, N)
    assert rel < 0.01, rel                        # W8A16: same FP8 weights, BF16 activations


def test_decode_timing():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM12x only")
    layer, ref_w = make_layer()
    k = Sm12xMarlinMxfp8BmmLinearKernel(Mxfp8LinearLayerConfig(weight_shape=(G * N, K), bmm_batch_size=G))
    k.process_weights_after_loading(layer)
    wb = ref_w.view(G, N, K).transpose(1, 2).contiguous()
    for tokens in (1, 6, 8):
        x = torch.randn(tokens, G, K, device=dev, dtype=torch.bfloat16)
        z = torch.empty(tokens, G, N, device=dev, dtype=torch.bfloat16)

        def emu():
            torch.bmm(x.transpose(0, 1), wb, out=z.transpose(0, 1))

        def fp8():
            k.apply_weights(layer, x)

        res = {}
        for name, fn in (("bf16 emulation", emu), ("fp8 marlin per group", fp8)):
            for _ in range(10):
                fn()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                g.replay()
            torch.cuda.synchronize()
            res[name] = (time.perf_counter() - t0) / 200 * 1e6
        print(f"wo_a tokens={tokens}: " + ", ".join(f"{k_} {v:.1f} us" for k_, v in res.items())
              + f" (x40 layers: {res['bf16 emulation'] * 40 / 1e3:.2f} -> {res["fp8 marlin per group"] * 40 / 1e3:.2f} ms)")
