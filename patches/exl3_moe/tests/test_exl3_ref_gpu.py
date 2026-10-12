# SPDX-License-Identifier: Apache-2.0
"""exl3_ref (the plain-torch EXL3 decode the custom kernel is checked against) vs exllamav3 itself: the decoded
rotated-basis weight bit-exact against ext.reconstruct, the Hadamard against ext.had_r_128, the whole linear against
LinearEXL3.forward.   EXL3_INT8_GEMV=0 python3 -m pytest -q test_exl3_ref_gpu.py   (GPU, inside the serve image;
exllamav3's default int8 activations at m=1 are ~0.7% off the exact product, the serve runs with them off)
"""
import pytest
import torch

from vllm_exl3_moe import exl3_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pytest.importorskip("exllamav3")
dev = torch.device("cuda")
MUL1 = torch.tensor(0x83DCD12D - 2 ** 32, dtype=torch.int32)


def rand_linear(k, n, K, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    trellis = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev, generator=g)
    suh = ((torch.randint(0, 2, (k,), device=dev, generator=g) * 2 - 1) * 0.05).half()
    svh = ((torch.randint(0, 2, (n,), device=dev, generator=g) * 2 - 1) * 0.05).half()
    return trellis, suh, svh


@pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
def test_decode_bit_exact_vs_reconstruct(K):
    from exllamav3.ext import exllamav3_ext as ext
    trellis, _, _ = rand_linear(256, 384, K, K)
    w = torch.empty((256, 384), dtype=torch.half, device=dev)
    ext.reconstruct(w, trellis, K, False, True)
    mine = exl3_ref.decode(trellis, K)
    assert torch.equal(mine.view(torch.int16), w.view(torch.int16)), \
        f"K={K}: {(mine != w).sum().item()} of {w.numel()} weights differ"


def test_hadamard_matches_had_r_128():
    from exllamav3.ext import exllamav3_ext as ext
    x = torch.randn(4, 512, device=dev).half()
    s = ((torch.randint(0, 2, (512,), device=dev) * 2 - 1) * 1.0).half()
    out = torch.empty_like(x)
    ext.had_r_128(x, out, s, None, 1.0)
    ref = exl3_ref.had_r(x.float() * s.float(), exl3_ref.hadamard128(dev))
    assert (out.float() - ref).abs().max().item() < 2e-3 * ref.abs().max().item()


@pytest.mark.parametrize("m", [1, 4, 8])
def test_linear_matches_linear_exl3(m):
    import os
    if m == 1 and os.environ.get("EXL3_INT8_GEMV", "2") != "0":
        pytest.skip("exllamav3 runs m=1 with int8 activations unless EXL3_INT8_GEMV=0")
    from exllamav3.modules.quant.exl3 import LinearEXL3
    k, n, K = 1024, 512, 5
    trellis, suh, svh = rand_linear(k, n, K, 100 + m)
    lin = LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis, mul1=MUL1.to(dev))
    x = torch.randn(m, k, device=dev).half()
    y = lin.forward(x, {}, out_dtype=torch.float32)
    ref = exl3_ref.linear(x, trellis, suh, svh, K)
    err = ((y.float() - ref).norm() / ref.norm()).item()
    assert err < 2e-3, err
