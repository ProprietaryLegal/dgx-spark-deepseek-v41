# SPDX-License-Identifier: Apache-2.0
"""exl3_mrow (our decode-size EXL3 kernel) against exl3_ref (proven bit-exact against exllamav3's decode), then its speed
against exllamav3's LinearEXL3 at the serve's per-rank TP=2 shapes, weights cycled from DRAM inside a CUDA graph.

    python3 -m pytest -q -s test_mrow_gpu.py                       (correctness)
    EXL3_BENCH=1 python3 -m pytest -q -s test_mrow_gpu.py -k bench  (speed, idle GPU)
"""
import os
import time

import pytest
import torch

from vllm_exl3_moe import exl3_ref
from vllm_exl3_moe.mrow import MrowGroup

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
dev = torch.device("cuda")


def rand_linear(k, n, K, seed, scale=0.05):
    g = torch.Generator(device=dev).manual_seed(seed)
    trellis = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev, generator=g)
    suh = ((torch.randint(0, 2, (k,), device=dev, generator=g) * 2 - 1) * scale).half()
    svh = ((torch.randint(0, 2, (n,), device=dev, generator=g) * 2 - 1) * scale).half()
    return trellis, suh, svh


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("m", [1, 4, 6, 8, 16])
def test_single_linear_matches_reference(K, m):
    k, n = 1024, 512
    t, su, sv = rand_linear(k, n, K, 10 * K + m)
    grp = MrowGroup([(t, su, sv, 0, 0)], K)
    x = torch.randn(m, k, device=dev).bfloat16()
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    grp(x, out)
    ref = exl3_ref.linear(x.float(), t, su, sv, K)
    assert rel(out, ref) < 6e-3, (K, m, rel(out, ref), grp.ks)


def test_group_with_offsets_and_k_split_matches_reference():
    """Three linears in one launch: two read the same input (gate / up style), one reads a column slice; outputs land at
    their column offsets; a 4096-wide input forces several k-slices (the counter + ordered sum path)."""
    K, m = 5, 6
    specs = [(4096, 1024, 0, 0), (4096, 1024, 0, 1024), (1024, 512, 4096, 2048)]
    lins = [rand_linear(k, n, K, 7 + i) for i, (k, n, _, _) in enumerate(specs)]
    grp = MrowGroup([(t, su, sv, xc, oc) for (t, su, sv), (_, _, xc, oc) in zip(lins, specs)], K)
    assert any(4096 // ks > 1 for ks in grp.ks[:2]), grp.ks
    x = torch.randn(m, 5120, device=dev).bfloat16()
    out = torch.zeros(m, 2560, device=dev, dtype=torch.bfloat16)
    for _ in range(3):                                   # counters must self-reset between calls
        grp(x, out)
    for (t, su, sv), (k, n, xc, oc) in zip(lins, specs):
        ref = exl3_ref.linear(x[:, xc:xc + k].float(), t, su, sv, K)
        assert rel(out[:, oc:oc + n], ref) < 6e-3


def test_deterministic_and_graph_replay():
    K, m = 5, 4
    t, su, sv = rand_linear(4096, 1024, K, 3)
    grp = MrowGroup([(t, su, sv, 0, 0)], K)
    x = torch.randn(m, 4096, device=dev).bfloat16()
    out = torch.empty(m, 1024, device=dev, dtype=torch.bfloat16)
    grp(x, out)
    first = out.clone()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        grp(x, out)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        grp(x, out)
    for _ in range(5):
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, first)


SHAPES = {   # name: list of (k, n) linears in one group, per rank at TP=2 (the serve's dense set, K5)
    "wq_a + wkv": [(5120, 1280), (5120, 512)],
    "wq_b": [(1280, 16384)],
    "wo_a (4 groups)": [(4096, 1024)] * 4,
    "wo_b": [(4096, 5120)],
    "shared gate + up": [(5120, 1152), (5120, 1152)],
    "shared down": [(1152, 5120)],
}


def _graph_us(fns, reps=30):
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / len(fns) * 1e6


@pytest.mark.skipif(not os.environ.get("EXL3_BENCH"), reason="timing: EXL3_BENCH=1 on an idle GPU")
@pytest.mark.parametrize("m", [4, 6])
def test_bench_vs_exllamav3(m):
    from exllamav3.modules.quant.exl3 import LinearEXL3
    K = 5
    mul1 = torch.tensor(0x83DCD12D - 2 ** 32, dtype=torch.int32, device=dev)
    print(f"\nm={m}  {'group':18s} {'exllamav3 (+casts)':>20s} {'mrow':>14s} {'speedup':>8s}  (us per group call, GB/s)")
    tot_e = tot_m = 0.0
    for name, lins in SHAPES.items():
        k_in = max(xc + k for xc, (k, _) in zip(_xcols(name, lins), lins))
        n_out = sum(n for _, n in lins)
        bytes_ = sum(k * n * K / 8 for k, n in lins)
        copies = max(2, int((256 << 20) // bytes_) + 1)
        x = torch.randn(m, k_in, device=dev).bfloat16()
        out = torch.empty(m, n_out, device=dev, dtype=torch.bfloat16)
        e_fns, m_fns = [], []
        for c in range(copies):
            ws = [rand_linear(k, n, K, 1000 * c + i) for i, (k, n) in enumerate(lins)]
            xcs = _xcols(name, lins)
            ocs = [sum(n for _, n in lins[:i]) for i in range(len(lins))]
            grp = MrowGroup([(t, su, sv, xc, oc) for (t, su, sv), xc, oc in zip(ws, xcs, ocs)], K)
            m_fns.append(lambda grp=grp: grp(x, out))
            exl = [LinearEXL3(None, k, n, suh=su, svh=sv, trellis=t, mul1=mul1) for (t, su, sv), (k, n) in zip(ws, lins)]

            def e(exl=exl, xcs=xcs, ocs=ocs):
                for lin, xc, oc, (k, n) in zip(exl, xcs, ocs, lins):
                    y = lin.forward(x[:, xc:xc + k].half().contiguous(), {})     # what dense.py does today
                    out[:, oc:oc + n].copy_(y)
            e_fns.append(e)
        te, tm = _graph_us(e_fns), _graph_us(m_fns)
        tot_e += te
        tot_m += tm
        print(f"      {name:18s} {te:10.1f} {bytes_ / te / 1e3:6.0f} GB/s {tm:7.1f} {bytes_ / tm / 1e3:4.0f} GB/s {te / tm:7.2f}x")
    print(f"      {'per layer':18s} {tot_e:10.1f} {'':>11s} {tot_m:7.1f} {'':>9s} {tot_e / tot_m:7.2f}x")


def _xcols(name, lins):
    if name.startswith("wo_a"):
        return [i * 4096 for i in range(len(lins))]     # each head group reads its own slice of the attention output
    return [0] * len(lins)


@pytest.mark.parametrize("m", [1, 2, 6, 8, 16])
def test_grouped_3d_input_bit_exact(m):
    """x as [m, problems, D] of any strides (wo_a's head groups straight from the inverse-RoPE kernel's [G, m, D] buffer,
    EXL3_MROW_GROUPED) gives the bits of the row-major copy [m, G * D]: same inputs, same kernel, other addresses."""
    K, G, D, N = 5, 4, 4096, 1024
    lins = [rand_linear(D, N, K, 40 + i) for i in range(G)]
    grp = MrowGroup([(t, su, sv, i * D, i * N) for i, (t, su, sv) in enumerate(lins)], K)
    buf = torch.randn(G, m, D, device=dev).bfloat16()             # the inverse-RoPE kernel's out_buf
    x3 = buf.transpose(0, 1)                                      # [m, G, D], strides (D, m * D, 1)
    out2, out3, out3c = (torch.empty(m, G * N, device=dev, dtype=torch.bfloat16) for _ in range(3))
    grp(x3.reshape(m, G * D).contiguous(), out2)
    grp(x3, out3)
    grp(x3.contiguous(), out3c)                                   # [m, G, D] row-major, as 3-D
    assert torch.equal(out3, out2) and torch.equal(out3c, out2)
    for i, (t, su, sv) in enumerate(lins):
        assert rel(out3[:, i * N:(i + 1) * N], exl3_ref.linear(buf[i].float(), t, su, sv, K)) < 6e-3


def test_grouped_3d_graph_replay_and_checks():
    K, G, D, N, m = 5, 4, 4096, 1024, 6
    lins = [rand_linear(D, N, K, 50 + i) for i in range(G)]
    grp = MrowGroup([(t, su, sv, i * D, i * N) for i, (t, su, sv) in enumerate(lins)], K)
    buf = torch.randn(G, m, D, device=dev).bfloat16()
    out = torch.empty(m, G * N, device=dev, dtype=torch.bfloat16)
    grp(buf.transpose(0, 1), out)
    first = out.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        grp(buf.transpose(0, 1), out)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        grp(buf.transpose(0, 1), out)
    for _ in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, first)
    with pytest.raises(RuntimeError):                              # a 3-D x needs one slice per problem
        grp(buf[:3].transpose(0, 1), out)
    off = MrowGroup([(t, su, sv, 0, i * N) for i, (t, su, sv) in enumerate(lins[:2])], K)   # both read column 0
    with pytest.raises(RuntimeError):
        off(buf[:2].transpose(0, 1), out[:, :2 * N])


@pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("want_items", [0, 4096])      # 4096 items forces k-slices: strips read inside an item's run
@pytest.mark.parametrize("m", [1, 6, 16])
def test_strips_bit_exact(K, want_items, m):
    """EXL3_HEAD_STRIPS: the same trellis in strips (to_strips) streams each item's K range as one run; every stage holds
    the same bytes, so the output is the standard layout's bit for bit, with and without k-splits."""
    from vllm_exl3_moe.mrow import to_strips

    k, n = 2048, 1024
    t, su, sv = rand_linear(k, n, K, 100 * K + m + want_items)
    ref_grp = MrowGroup([(t, su, sv, 0, 0)], K, want_items=want_items)
    ts = to_strips(t)
    assert not torch.equal(ts, t)
    grp = MrowGroup([(ts, su, sv, 0, 0)], K, want_items=want_items, strips=True)
    assert grp.ks == ref_grp.ks
    if want_items:
        assert k // grp.ks[0] > 1, grp.ks
    x = torch.randn(m, k, device=dev).bfloat16()
    a = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    b = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    ref_grp(x, a)
    for _ in range(2):
        grp(x, b)
        assert torch.equal(a, b), (K, want_items, m)


def test_strips_graph_replay():
    from vllm_exl3_moe.mrow import to_strips

    K, k, n, m = 5, 5120, 2048, 6
    t, su, sv = rand_linear(k, n, K, 77)
    ref_grp = MrowGroup([(t, su, sv, 0, 0)], K)
    grp = MrowGroup([(to_strips(t), su, sv, 0, 0)], K, strips=True)
    x = torch.randn(m, k, device=dev).bfloat16()
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    grp(x, out)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g):
        grp(x, out)
    torch.cuda.current_stream().wait_stream(s)
    ref = torch.empty_like(out)
    for seed in range(3):
        x.copy_(torch.randn(m, k, device=dev, generator=torch.Generator(device=dev).manual_seed(seed)).bfloat16())
        g.replay()
        ref_grp(x, ref)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
