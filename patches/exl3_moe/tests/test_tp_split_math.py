# SPDX-License-Identifier: Apache-2.0
"""Why the TP=2 channel split of EXL3 experts is exact (numpy, runs anywhere).

exllamav3 reconstructs a weight as (modules/quant/exl3.py:239-249, quantize.py:410-434):
    W = had_r( suh[:, None] * had_l(W_rot) ) * svh[None, :]
with had_l / had_r block-diagonal 128x128 Hadamards (1/sqrt(128)) on the in / out axis, and W_rot
decoded tile by tile from the trellis (16x16 tiles, each coded on its own). So block (a, b) of W only
depends on W_rot block (a, b), suh block a and svh block b: cutting (W_rot, suh, svh) at multiples of
128 gives exactly the matching block of W. This test checks that property and the resulting
"sum of per-rank partial MoE outputs == full expert output" on random data with the real
DeepSeek-V4.1 expert shapes (hidden 5120, intermediate 2304, TP=2 -> 1152 = 9 x 128 per rank).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_exl3_moe.plan import shard_slices  # noqa: E402

HB = 128
H, I = 5120, 2304
rng = np.random.default_rng(0)


def sylvester(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


HAD = sylvester(HB)


def had_l(x):
    k, n = x.shape
    return (HAD @ x.reshape(k // HB, HB, n)).reshape(k, n)


def had_r(x):
    k, n = x.shape
    return (x.reshape(k, n // HB, HB) @ HAD).reshape(k, n)


def weight(w_rot, suh, svh):
    return had_r(suh[:, None] * had_l(w_rot)) * svh[None, :]


def tiles_of(w_rot):
    """(in/16, out/16, 256) 'trellis' stand-in: the tile grid exllamav3 indexes, contents verbatim."""
    k, n = w_rot.shape
    return w_rot.reshape(k // 16, 16, n // 16, 16).transpose(0, 2, 1, 3).reshape(k // 16, n // 16, 256)


def untile(t):
    a, b, _ = t.shape
    return t.reshape(a, b, 16, 16).transpose(0, 2, 1, 3).reshape(a * 16, b * 16)


def make_linear(k, n):
    return (rng.standard_normal((k, n)).astype(np.float64),
            np.sign(rng.standard_normal(k)) * rng.uniform(0.5, 1.5, k),
            np.sign(rng.standard_normal(n)) * rng.uniform(0.5, 1.5, n))


def cut(lin, proj, rank, tp=2):
    """Apply the plugin's shard_slices to (trellis-tiles, suh, svh) and rebuild that rank's W."""
    w_rot, suh, svh = lin
    s = shard_slices(proj, I, tp, rank)
    t = tiles_of(w_rot)
    a, z = s.trellis_range
    t = t[:, a:z] if s.trellis_dim == 1 else t[a:z]
    suh_r = suh if s.suh_range is None else suh[s.suh_range[0]:s.suh_range[1]]
    svh_r = svh if s.svh_range is None else svh[s.svh_range[0]:s.svh_range[1]]
    return weight(untile(t), suh_r, svh_r)


def test_tile_roundtrip():
    w = rng.standard_normal((64, 48))
    assert np.array_equal(untile(tiles_of(w)), w)


def test_column_and_row_cuts_are_exact_blocks():
    w1 = make_linear(H, I)        # gate: in = hidden, out = intermediate
    w2 = make_linear(I, H)        # down: in = intermediate, out = hidden
    W1, W2 = weight(*w1), weight(*w2)
    for r in range(2):
        i0, i1 = r * 1152, (r + 1) * 1152
        np.testing.assert_allclose(cut(w1, "w1", r), W1[:, i0:i1], rtol=0, atol=1e-10)
        np.testing.assert_allclose(cut(w2, "w2", r), W2[i0:i1, :], rtol=0, atol=1e-10)


def test_a_cut_off_the_128_grid_is_not_exact():
    # a 128-wide window starting at column 64 is tile-aligned (4 x 16) but straddles two Hadamard
    # blocks: the rebuilt slice no longer matches. This is why TP=4 (576 = 4.5 x 128) cannot use
    # the channel split for intermediate 2304.
    w_rot, suh, svh = make_linear(256, 384)
    full = weight(w_rot, suh, svh)
    bad = weight(w_rot[:, 64:192], suh, svh[64:192])
    assert np.abs(bad - full[:, 64:192]).max() > 1e-3


def test_sum_of_rank_partials_equals_full_expert():
    limit = 10.0
    g, u, d = make_linear(H, I), make_linear(H, I), make_linear(I, H)
    scale = 1.0 / np.sqrt(H)
    x = rng.standard_normal((4, H)) * scale

    def swiglu(a, b):
        a = np.minimum(a, limit)
        b = np.clip(b, -limit, limit)
        return a / (1.0 + np.exp(-a)) * b

    full = swiglu(x @ weight(*g), x @ weight(*u)) @ weight(*d)
    parts = [swiglu(x @ cut(g, "w1", r), x @ cut(u, "w3", r)) @ cut(d, "w2", r) for r in range(2)]
    np.testing.assert_allclose(parts[0] + parts[1], full, rtol=1e-9, atol=1e-9)
