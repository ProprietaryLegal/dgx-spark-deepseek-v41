# SPDX-License-Identifier: Apache-2.0
"""Patch 0032: sm12x_marlin_bmm's groups on concurrent streams (VLLM_SM12X_BMM_STREAMS=1).

The concurrent path against the group loop, bit for bit, on the DSpark drafter's wo_a shape (4 groups of [1024, 4096]
MXFP8, Marlin-packed as Sm12xMarlinMxfp8BmmLinearKernel packs them) at 1-16 tokens, eager and replayed from a captured
CUDA graph with new inputs.

Run inside the dsv41 image on an SM12x GPU:  python3 -m pytest -q test_sm12x_bmm_streams.py
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12, reason="needs an SM12x GPU")
pytest.importorskip("vllm")   # runs in the serve image
DEV = "cuda"
G, N, K = 4, 1024, 4096


@pytest.fixture(scope="module")
def packed():
    from torch.nn.parameter import Parameter

    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_mxfp8_layer_for_marlin

    g = torch.Generator(device=DEV).manual_seed(0)
    w = (torch.randn((G * N, K), device=DEV, generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(120, 130, (G * N, K // 32), device=DEV, generator=g, dtype=torch.uint8)
    weights, scales = [], []
    for i in range(G):          # as Sm12xMarlinMxfp8BmmLinearKernel.process_weights_after_loading
        h = torch.nn.Module()
        h.weight = Parameter(w[i * N:(i + 1) * N].contiguous(), requires_grad=False)
        h.weight_scale = Parameter(s[i * N:(i + 1) * N].contiguous().view(torch.float8_e8m0fnu), requires_grad=False)
        h.output_size_per_partition, h.input_size_per_partition = N, K
        prepare_mxfp8_layer_for_marlin(h)
        weights.append(h.weight.data)
        scales.append(h.weight_scale.data)
    return torch.stack(weights), torch.stack(scales)


def run(mod, x, weight, scales, streams):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_mxfp8_marlin_linear

    if streams:
        return mod._sm12x_marlin_bmm_streams(x, weight, scales, G, N, K, apply_mxfp8_marlin_linear)
    old, mod._STREAMS = mod._STREAMS, False
    try:
        return mod.sm12x_marlin_bmm(x, weight, scales, G, N, K)
    finally:
        mod._STREAMS = old


@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 6, 8, 16])
def test_eager_bit_equal(packed, tokens):
    from vllm.model_executor.kernels.linear.mxfp8 import sm12x_bmm as mod

    weight, scales = packed
    x = torch.randn((tokens, G, K), device=DEV).to(torch.bfloat16)
    assert torch.equal(run(mod, x, weight, scales, True), run(mod, x, weight, scales, False))


def test_graph_replay_bit_equal(packed):
    from vllm.model_executor.kernels.linear.mxfp8 import sm12x_bmm as mod

    weight, scales = packed
    x = torch.randn((6, G, K), device=DEV).to(torch.bfloat16)
    run(mod, x, weight, scales, True)
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(graph):
        out = run(mod, x, weight, scales, True)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        x.copy_(torch.randn((6, G, K), device=DEV).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, run(mod, x, weight, scales, False))
