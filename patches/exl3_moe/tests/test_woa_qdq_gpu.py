# SPDX-License-Identifier: Apache-2.0
"""VLLM_DSV41_WOA_QDQ (woa_qdq.py): DeepSeek's FP8 round trip of the attention output o before wo_a, bit for bit.

1. act_quant_ref (the torch transcription of inference/kernel.py's act_quant) against an independent build of the same
   math: the scale from frexp in float64, e4m3 by a nearest-value search over all 256 codes with ties to the even code;
2. woa_qdq.qdq (Triton: the prompt sizes and the FP8 wo_a) == act_quant_ref on random and adversarial input (every
   finite bf16 value, signed zeros, subnormals, the 448 * 2^k scale boundaries, e4m3 ties, values that round to inf);
3. exl3_mrow's wo_a build (-DMROW_QDQ=1, the decode sizes) == woa_qdq.qdq then the plain exl3_mrow, at K 2-6, 1-16 rows,
   2-D input and the 3-D [T, G, D] view (EXL3_MROW_GROUPED);
4. install_fp8 / hook_fp8 (an FP8 wo_a: the DSpark drafter's) hand the Marlin bmm the rounded o.

    python3 -m pytest -q test_woa_qdq_gpu.py
"""
import functools
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_exl3_moe import woa_qdq  # noqa: E402
from vllm_exl3_moe.mrow import MrowGroup  # noqa: E402

dev = torch.device("cuda")


def bits(t):
    return t.contiguous().view(torch.int16)


def e4m3_values():
    """Every finite e4m3fn value (float64) and its code."""
    v = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).double()
    keep = ~torch.isnan(v)
    return v[keep], torch.arange(256)[keep]


def independent_qdq(x):
    """The same semantics built another way: s = 2^ceil(log2(max(amax, 1e-4f) / 448)) from frexp in float64, x / s to
    the nearest e4m3 value (ties to the even code, saturating at 448), times s, to bf16. CPU, finite input."""
    xd = x.double().cpu().unflatten(-1, (-1, 32))
    amax = xd.abs().amax(-1, keepdim=True).clamp_min(float(torch.tensor(1e-4, dtype=torch.float32)))
    m, ex = torch.frexp(amax / 448.0)
    s = torch.ldexp(torch.ones_like(amax), torch.where(m == 0.5, ex - 1, ex))
    v = xd / s
    vals, codes = e4m3_values()
    pos = vals >= 0
    order = torch.argsort(vals[pos])
    mag, mcode = vals[pos][order], codes[pos][order]           # 0 .. 448 ascending
    a = v.abs().flatten()
    i = torch.searchsorted(mag, a).clamp(1, len(mag) - 1)
    lo, hi = mag[i - 1], mag[i]
    up = (hi - a < a - lo) | ((hi - a == a - lo) & (mcode[i] % 2 == 0))
    q = torch.where(a >= mag[-1], mag[-1], torch.where(up, hi, lo))
    q = torch.copysign(q.view_as(v), v)                          # keeps -0
    return (q * s).flatten(-2).to(torch.bfloat16).to(x.device)


CASE_NAMES = ["randn*1e-30", "randn*1e-08", "randn*1e-05", "randn*0.0001", "randn*0.001", "randn*0.1", "randn*1",
              "randn*37", "randn*448", "randn*100000", "randn*1e+20", "randn*1e+36", "every_finite_bf16",
              "every_finite_bf16_shuffled", "signed_zeros", "subnormals", "scale_boundaries", "e4m3_ties",
              "near_bf16_max"]


@functools.lru_cache(maxsize=1)
def adversarial():
    g = torch.Generator(device=dev).manual_seed(0)
    c = {}
    for sc in (1e-30, 1e-8, 1e-5, 1e-4, 1e-3, 0.1, 1.0, 37.0, 448.0, 1e5, 1e20, 1e36):
        c[f"randn*{sc:g}"] = (torch.randn(16, 4096, device=dev, generator=g) * sc).bfloat16()
    allb = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    allb = allb[torch.isfinite(allb.float())]
    n = allb.numel() // 32 * 32
    c["every_finite_bf16"] = allb[:n].reshape(-1, 256).to(dev)
    perm = torch.randperm(allb.numel(), generator=torch.Generator().manual_seed(1))
    c["every_finite_bf16_shuffled"] = allb[perm][:n].reshape(-1, 256).to(dev)
    z = torch.zeros(8, 256, dtype=torch.bfloat16)
    z[1], z[2, ::2], z[3, 5] = -0.0, -0.0, 1e-3
    c["signed_zeros"] = z.to(dev)
    sub = torch.arange(1, 128, dtype=torch.int16).view(torch.bfloat16)
    c["subnormals"] = torch.cat([sub, -sub, torch.zeros(2, dtype=torch.bfloat16)]).reshape(1, 256).to(dev)
    rows = []                                                    # amax exactly 448 * 2^k and one bf16 step either side
    for k in range(-30, 120):
        for f in (1.0, 1.0 - 2 ** -8, 1.0 + 2 ** -7):
            a = 448.0 * 2.0 ** k * f
            r = torch.full((32,), a * 0.37, dtype=torch.float64)
            r[5], r[7] = a, -a / 3
            rows.append(r)
    c["scale_boundaries"] = torch.stack(rows).bfloat16().to(dev)
    vals, _ = e4m3_values()
    mag = torch.sort(vals[vals > 0]).values
    mids = torch.cat([(mag[:-1] + mag[1:]) / 2, torch.tensor([2.0 ** -10, 3 * 2.0 ** -10])])
    rows = []                                                    # every e4m3 midpoint (a tie) at s = 2^k
    for k in (-20, -3, 0, 5, 100):
        m = mids * 2.0 ** k
        for i in range(0, len(m), 31):
            r = torch.zeros(32, dtype=torch.float64)
            ch = m[i:i + 31]
            r[:len(ch)] = ch * torch.where(torch.arange(len(ch)) % 2 == 0, 1.0, -1.0)
            r[31] = 448.0 * 2.0 ** k
            rows.append(r)
    ties = torch.stack(rows)
    assert torch.equal(ties.bfloat16().double(), ties)           # ties are exact bf16 values
    c["e4m3_ties"] = ties.bfloat16().to(dev)
    big = torch.tensor([3.3895e38, -3.3895e38, 3.2e38, 3.0e38, 1e38, 2.5e38], dtype=torch.float32).bfloat16()
    c["near_bf16_max"] = torch.cat([big, torch.ones(26, dtype=torch.bfloat16)]).reshape(1, 32).to(dev)
    assert list(c) == CASE_NAMES, list(c)
    return c


@pytest.mark.parametrize("name", CASE_NAMES)
def test_reference_matches_independent_semantics(name):
    x = adversarial()[name]
    assert torch.equal(bits(woa_qdq.act_quant_ref(x)), bits(independent_qdq(x))), name


@pytest.mark.parametrize("name", CASE_NAMES)
def test_triton_matches_reference(name):
    x = adversarial()[name]
    y = woa_qdq.qdq(x)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    assert torch.equal(bits(y), bits(woa_qdq.act_quant_ref(x))), name


def test_cases_cover_the_edges():
    """The adversarial set reaches the corners: values that round to inf, ties that move, groups under the 1e-4 floor."""
    c = adversarial()
    r = woa_qdq.act_quant_ref(c["every_finite_bf16"])
    assert torch.isinf(r.float()).sum() == 16                    # |x| >= 1.9375 * 2^127 -> 256 * 2^120
    t = woa_qdq.act_quant_ref(c["e4m3_ties"])
    assert not torch.equal(bits(t), bits(c["e4m3_ties"]))        # the ties moved
    tiny = c["randn*1e-08"]
    assert tiny.float().abs().amax() < 1e-4                      # the floor sets s = 2^-22 for every group there
    floor = ((tiny.float() * 2.0 ** 22).to(torch.float8_e4m3fn).float() * 2.0 ** -22).bfloat16()
    assert torch.equal(bits(woa_qdq.act_quant_ref(tiny)), bits(floor))


def test_inf_input_gives_the_reference_nan_group():
    x = (torch.randn(8, 4096, device=dev) * 2e38).bfloat16()   # +-inf where |x| > bf16 max
    x[3, :32] = float("inf")
    r, y = woa_qdq.act_quant_ref(x), woa_qdq.qdq(x)
    assert torch.isinf(x.float()).any()
    assert torch.equal(torch.isnan(r.float()), torch.isnan(y.float()))
    keep = ~torch.isnan(r.float())
    assert torch.equal(bits(r)[keep], bits(y)[keep])


@pytest.mark.parametrize("shape", [(1, 32), (3, 7, 96), (5, 4, 4096), (0, 64), (2, 1, 1, 2048)])
def test_shapes_and_fp16_out(shape):
    x = (torch.randn(shape, device=dev) * 3).bfloat16()
    r = woa_qdq.act_quant_ref(x)
    assert torch.equal(bits(woa_qdq.qdq(x)), bits(r))
    h = woa_qdq.qdq(x, torch.float16)                            # the exllamav3 prompt path's input
    assert h.dtype == torch.float16 and torch.equal(bits(h), bits(r.to(torch.float16)))


def test_layouts():
    """o_rot is the [T, G, D] view of the inverse-RoPE kernel's [G, T, D] buffer: the output keeps that layout, so each
    group's [T, D] slice stays contiguous; a padded (non-dense) input is made contiguous first."""
    buf = (torch.randn(4, 7, 4096, device=dev) * 5).bfloat16()
    o = buf.transpose(0, 1)
    y = woa_qdq.qdq(o)
    assert y.stride() == o.stride() and y[:, 2].is_contiguous()
    assert torch.equal(bits(y), bits(woa_qdq.act_quant_ref(o)))
    pad = torch.randn(6, 4096 + 64, device=dev).bfloat16()[:, :4096]
    assert torch.equal(bits(woa_qdq.qdq(pad)), bits(woa_qdq.act_quant_ref(pad)))
    with pytest.raises(ValueError):
        woa_qdq.qdq(torch.randn(2, 64, device=dev))              # fp32: not what act_quant takes
    with pytest.raises(ValueError):
        woa_qdq.qdq(torch.randn(2, 48, device=dev).bfloat16())   # 48 values: not whole groups


# ------------------------------------------------------------------------------------------------- exl3_mrow fold
def rand_linear(k, n, K, seed, scale=0.05):
    g = torch.Generator(device=dev).manual_seed(seed)
    trellis = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev, generator=g)
    suh = ((torch.randint(0, 2, (k,), device=dev, generator=g) * 2 - 1) * scale).half()
    svh = ((torch.randint(0, 2, (n,), device=dev, generator=g) * 2 - 1) * scale).half()
    return trellis, suh, svh


def wo_a_groups(K, k, n, G, seed):
    """wo_a's head groups as one mrow group: linear i reads input columns [i k, (i + 1) k) and writes [i n, (i + 1) n)."""
    lins = [rand_linear(k, n, K, seed + i) for i in range(G)]
    parts = [(t, su, sv, i * k, i * n) for i, (t, su, sv) in enumerate(lins)]
    return MrowGroup(parts, K, qdq=True), MrowGroup(parts, K)


def o_rows(m, width, seed):
    """Attention-output-like rows at several magnitudes, plus zero / tiny / mixed rows."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(m, width, device=dev, generator=g)
    x *= torch.logspace(-3, 1, m, device=dev)[:, None]
    if m >= 4:
        x[1] = 0.0
        x[2] *= 1e-6                                             # below the 1e-4 amax floor
        x[3, ::7] *= 300.0                                       # a few large values per group
    return x.bfloat16()


@pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("m", [1, 5, 8, 9, 16])
def test_mrow_fold_equals_qdq_then_mrow(K, m):
    k, n, G = 512, 256, 4
    folded, plain = wo_a_groups(K, k, n, G, 100 * K + m)
    assert folded.ext is not plain.ext and len(set(folded.ks)) >= 1
    x = o_rows(m, G * k, K + 7 * m)
    y_fold = folded(x, torch.empty(m, G * n, device=dev, dtype=torch.bfloat16))
    y_ref = plain(woa_qdq.qdq(x), torch.empty(m, G * n, device=dev, dtype=torch.bfloat16))
    y_raw = plain(x, torch.empty(m, G * n, device=dev, dtype=torch.bfloat16))
    assert torch.equal(bits(y_fold), bits(y_ref)), (K, m)
    assert not torch.equal(bits(y_fold), bits(y_raw))           # the fold does round


def test_mrow_fold_serve_shape_k_split_and_replay():
    """wo_a at TP=2 (4 head groups of 4096 -> 1024, K5): several k-slices a problem; counters reset across calls."""
    K, k, n, G = 5, 4096, 1024, 4
    folded, plain = wo_a_groups(K, k, n, G, 7)
    assert all(k // ks > 1 for ks in folded.ks), folded.ks
    for m in (1, 6, 16):
        x = o_rows(m, G * k, m)
        ref = plain(woa_qdq.qdq(x), torch.empty(m, G * n, device=dev, dtype=torch.bfloat16))
        for _ in range(3):
            y = folded(x, torch.empty(m, G * n, device=dev, dtype=torch.bfloat16))
            assert torch.equal(bits(y), bits(ref)), m


def test_mrow_fold_grouped_3d_input():
    """EXL3_MROW_GROUPED=1: each problem reads its head group straight from the [G, T, D] buffer seen as [T, G, D]."""
    K, k, n, G, T = 5, 1024, 256, 4, 6
    folded, plain = wo_a_groups(K, k, n, G, 11)
    buf = (torch.randn(G, T, k, device=dev) * 2).bfloat16()
    o = buf.transpose(0, 1)
    oq = woa_qdq.qdq(o)
    assert oq.stride() == o.stride()
    y = folded(o, torch.empty(T, G * n, device=dev, dtype=torch.bfloat16))
    ref = plain(oq, torch.empty(T, G * n, device=dev, dtype=torch.bfloat16))
    assert torch.equal(bits(y), bits(ref))
    assert torch.equal(bits(y), bits(folded(o.reshape(T, G * k), torch.empty_like(y))))   # 3-D == 2-D copy


@pytest.mark.parametrize("name", ["every_finite_bf16_shuffled", "scale_boundaries", "e4m3_ties", "signed_zeros",
                                  "subnormals", "near_bf16_max", "randn*1e-08", "randn*1e+36"])
def test_mrow_fold_adversarial(name):
    """The fold on the adversarial rows (as wo_a input rows of 4 x 512): same bits as qdq then mrow, NaN / inf included
    (the kernel's fp16 staging overflows on the huge rows the same way on both sides)."""
    K, k, n, G = 4, 512, 256, 4
    folded, plain = wo_a_groups(K, k, n, G, 21)
    x = adversarial()[name].reshape(-1)
    x = torch.cat([x, torch.zeros((-x.numel()) % (G * k), dtype=x.dtype, device=dev)]).reshape(-1, G * k)
    for r0 in range(0, x.shape[0], 16):
        xs = x[r0:r0 + 16].contiguous()
        m = xs.shape[0]
        y = folded(xs, torch.empty(m, G * n, device=dev, dtype=torch.bfloat16))
        ref = plain(woa_qdq.qdq(xs), torch.empty(m, G * n, device=dev, dtype=torch.bfloat16))
        assert torch.equal(bits(y), bits(ref)), (name, r0)


def test_mrow_fold_cuda_graph_replay():
    K, k, n, G, m = 5, 1024, 256, 4, 4
    folded, plain = wo_a_groups(K, k, n, G, 31)
    x = o_rows(m, G * k, 3)
    out = torch.empty(m, G * n, device=dev, dtype=torch.bfloat16)
    ref = plain(woa_qdq.qdq(x), torch.empty_like(out)).clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        folded(x, out)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        folded(x, out)
    out.zero_()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), bits(ref))


# ------------------------------------------------------------------------------------------------- FP8 wo_a (drafter)
def test_install_fp8_rounds_before_the_bmm():
    seen = {}

    def bmm(x, w, s, groups, n, k):
        seen.update(x=x, w=w, s=s, groups=groups, n=n, k=k)
        return torch.zeros(x.shape[0], groups, n, device=x.device, dtype=x.dtype)

    layer = types.SimpleNamespace(sm12x_bmm_groups=4, sm12x_bmm_nk=(1024, 4096), weight="w0", weight_scale="s0")
    assert woa_qdq.install_fp8(layer, bmm=bmm)
    layer.weight = "w1"                                          # read at call time, like o_proj.py's own branch
    o = (torch.randn(4, 3, 4096, device=dev) * 2).bfloat16().transpose(0, 1)
    z = layer.grouped_bmm(o)
    assert z.shape == (3, 4, 1024)
    assert torch.equal(bits(seen["x"]), bits(woa_qdq.act_quant_ref(o)))
    assert (seen["w"], seen["s"], seen["groups"], seen["n"], seen["k"]) == ("w1", "s0", 4, 1024, 4096)
    other = types.SimpleNamespace()
    assert not woa_qdq.install_fp8(other, bmm=bmm) and not hasattr(other, "grouped_bmm")


def test_hook_fp8_keeps_the_method_class(monkeypatch):
    class FakeFp8Method:                                         # stands in for vLLM's ModelOptLinearMethod
        def process_weights_after_loading(self, layer):
            layer.sm12x_bmm_groups, layer.sm12x_bmm_nk = 4, (1024, 4096)
            layer.weight, layer.weight_scale = "w", "s"

    calls = []
    monkeypatch.setattr(woa_qdq, "install_fp8", lambda layer: calls.append(layer) or True)
    m = FakeFp8Method()
    assert woa_qdq.hook_fp8(m, "model.layers.43.attn.wo_a") is m and type(m) is FakeFp8Method
    layer = types.SimpleNamespace()
    m.process_weights_after_loading(layer)
    assert calls == [layer] and layer.sm12x_bmm_groups == 4


def test_flag_and_names(monkeypatch):
    monkeypatch.delenv("VLLM_DSV41_WOA_QDQ", raising=False)
    assert not woa_qdq.enabled()
    monkeypatch.setenv("VLLM_DSV41_WOA_QDQ", "1")
    assert woa_qdq.enabled()
    assert woa_qdq.is_wo_a("model.layers.43.attn.wo_a") and woa_qdq.is_wo_a("language_model.model.layers.0.attn.wo_a")
    assert not woa_qdq.is_wo_a("model.layers.3.attn.wo_b") and not woa_qdq.is_wo_a("model.layers.3.attn.wo_a.slice.0")
