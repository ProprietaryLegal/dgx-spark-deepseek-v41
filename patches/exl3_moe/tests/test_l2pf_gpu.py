# SPDX-License-Identifier: Apache-2.0
"""l2pf.py + csrc/exl3_l2pf.cu: the paced L2 prefetch of the next dense weights on a side stream.

The kernel runs (paced and all-at-once), a prefetch changes no output, the points fork and join so that a CUDA-graph
capture of a forward-like sequence ends with no side-stream work (the last layer's "next" point joins), and with
EXL3_L2PF_BENCH=1 a read of 16 MiB is timed cold vs right after a prefetch (idle GPU only). EXL3_L2PF_ORDER=1: the
forks' ranges in the critical kernels' order (gate before hc_ffn_fn, wq_a + wkv before hc_attn_fn), the `moe` fork
capped at 8 MiB, and the LM head's first bytes forked after the last layer and joined by the final norm's hook (captured
forwards still end joined; a drafter layer, or a model without the hook, never forks the head).

    EXL3_L2PF=attn,moe,next python3 -m pytest -q -s test_l2pf_gpu.py
"""
import importlib
import os
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEV = "cuda"
ENVS = ("EXL3_L2PF", "EXL3_L2PF_HC", "EXL3_L2PF_ORDER", "EXL3_L2PF_MOE_CAP_MB", "EXL3_L2PF_HEAD_MB", "EXL3_L2PF_DRAFTER")


def _reloaded(monkeypatch, **env):
    for k in ENVS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import vllm_exl3_moe.l2pf as m
    return importlib.reload(m)


@pytest.fixture()
def l2pf(monkeypatch):
    yield _reloaded(monkeypatch, EXL3_L2PF="attn,moe,next")
    _reloaded(monkeypatch)


@pytest.fixture()
def reload(monkeypatch):
    """l2pf reloaded with the given env (any number of times in a test), back to the defaults after it."""
    yield lambda **env: _reloaded(monkeypatch, **env)
    _reloaded(monkeypatch)


def test_kernel_runs(l2pf):
    t = torch.randn(4 << 20, device=DEV)                     # 16 MiB
    for wave_ns in (0, int(128 * l2pf.CHUNK / 230)):
        l2pf.ext().l2pf([t.data_ptr()], [t.numel() * 4], l2pf.CHUNK, 4, 0, wave_ns)
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError):
        l2pf.ext().l2pf([t.data_ptr() + 8], [1024], l2pf.CHUNK, 4, 0, 0)   # misaligned start


def fake_forward(l2pf, w, x, layers):
    """wq_b -> attn point, wo_b -> moe point, routed MoE -> next point, per layer (the serve's order)."""
    y = x
    for layer in range(layers):
        y = y @ w[(layer, "attn.wq_b")][0][: y.shape[1], : y.shape[1]]
        l2pf.point("attn", layer, y.shape[0])
        y = torch.tanh(y)
        l2pf.point("moe", layer, y.shape[0])
        y = y * 0.5 + 1.0
        l2pf.point("next", layer, y.shape[0])
    return y


def test_points_capture_and_change_nothing(l2pf):
    layers = 3
    w = {}
    for layer in range(layers):
        for module in ("attn.fused_wqa_wkv", "attn.wq_b", "attn.wo_a", "attn.wo_b",
                       "ffn.shared_experts.gate_up_proj", "ffn.shared_experts.down_proj"):
            w[(layer, module)] = [torch.randn(1024, 1024, device=DEV) / 32]
            l2pf.register(layer, module, w[(layer, module)])
    x = torch.randn(6, 512, device=DEV)
    ref = fake_forward(l2pf, w, x, layers)
    assert not l2pf._pending                                  # the last layer's MoE joined
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fake_forward(l2pf, w, x, layers)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):                                 # raises on unjoined side-stream work
        out = fake_forward(l2pf, w, x, layers)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    # prefill sizes never prefetch
    l2pf.point("attn", 0, 64)
    assert not l2pf._pending


@pytest.mark.skipif(not os.environ.get("EXL3_L2PF_BENCH"), reason="timing: EXL3_L2PF_BENCH=1 on an idle GPU")
def test_bench_read_after_prefetch(l2pf):
    pool = [torch.randn(4 << 20, device=DEV) for _ in range(16)]   # 16 x 16 MiB: each read starts cold
    out = torch.empty((), device=DEV)

    def read(t):
        torch.sum(t, dim=0, out=out)

    for i in range(2):
        read(pool[i])
    torch.cuda.synchronize()
    cold, warm = [], []
    for i in range(2, 16):
        t = pool[i]
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if i % 2:
            l2pf.ext().l2pf([t.data_ptr()], [t.numel() * 4], l2pf.CHUNK, 4, 0, 0)
            torch.cuda._sleep(200_000)                       # ~DRAM-idle time for the prefetch to land
        a.record()
        read(t)
        b.record()
        torch.cuda.synchronize()
        (warm if i % 2 else cold).append(a.elapsed_time(b) * 1000)
    print(f"16 MiB read: cold {sorted(cold)[len(cold) // 2]:.1f} us, after a prefetch {sorted(warm)[len(warm) // 2]:.1f} us")


# ---- EXL3_L2PF_ORDER ---------------------------------------------------------------------------------------------

MIB = 1 << 20


def _t(*shape, dtype=torch.int16):
    return torch.zeros(*shape, dtype=dtype, device=DEV)


def _layer_weights(layers):
    """Per-rank shapes at TP=2 (dense K4, the shared expert K5): (layer, module) -> tensors in their kernel's order."""
    w = {}
    for i in range(layers):
        w[(i, "hc_attn_fn")] = [_t(24, 20480, dtype=torch.float32)]          # 1.875 MiB
        w[(i, "hc_ffn_fn")] = [_t(24, 20480, dtype=torch.float32)]
        w[(i, "ffn.gate")] = [_t(384, 5120, dtype=torch.bfloat16)]          # 3.75 MiB
        w[(i, "attn.fused_wqa_wkv")] = [_t(320, 80, 64), _t(320, 32, 64)]   # wq_a 3.125, wkv 1.25 MiB
        w[(i, "attn.wq_b")] = [_t(80, 1024, 64)]                            # 10 MiB
        w[(i, "attn.wo_a")] = [_t(256, 64, 64) for _ in range(4)]           # 2 MiB each
        w[(i, "attn.wo_b")] = [_t(256, 320, 64)]                            # 10 MiB
        w[(i, "ffn.shared_experts.gate_up_proj")] = [_t(320, 72, 80), _t(320, 72, 80)]   # 3.52 MiB each
        w[(i, "ffn.shared_experts.down_proj")] = [_t(72, 320, 80)]
    return w


def _register(m, w):
    for (layer, module), ts in w.items():
        if module.startswith(("hc_", "ffn.gate")):
            continue                                  # register_decoder_layers' (EXL3_L2PF_HC=1)
        m.register(layer, module, ts)


class _Norm(torch.nn.Module):
    def forward(self, x):
        return x * 2.0


def _model(w, layers):
    """What register_decoder_layers reads of a DeepseekV4Model: layers[i].hc_attn_fn / hc_ffn_fn / ffn.gate.weight and
    the final norm."""
    ls = [SimpleNamespace(hc_attn_fn=w[(i, "hc_attn_fn")][0], hc_ffn_fn=w[(i, "hc_ffn_fn")][0],
                          ffn=SimpleNamespace(gate=SimpleNamespace(weight=w[(i, "ffn.gate")][0])))
          for i in range(layers)]
    return SimpleNamespace(layers=ls, start_layer=0, end_layer=layers, norm=_Norm())


def _order(m, layer, point, w):
    """The (layer, module, tensor index, bytes) of each range of a fork, from the registered tensors' pointers."""
    where = {t.data_ptr(): (key, j) for key, ts in w.items() for j, t in enumerate(ts)}
    ptrs, sizes = m._ranges(layer, point)
    return [(*where[p][0], where[p][1], n) for p, n in zip(ptrs, sizes)]


def test_order_and_moe_cap(reload):
    w = _layer_weights(2)
    seen = {}
    for name, env in (("default", {}), ("order", {"EXL3_L2PF_ORDER": "1"}),
                      ("cap only", {"EXL3_L2PF_MOE_CAP_MB": "8"})):
        m = reload(EXL3_L2PF="attn,moe,next", EXL3_L2PF_HC="1", **env)
        _register(m, w)
        m.register_decoder_layers(_model(w, 2))
        seen[name] = {p: _order(m, 0, p, w) for p in ("attn", "moe", "next")}
    d, o, c = seen["default"], seen["order"], seen["cap only"]
    mods = lambda rs: [r[1] for r in rs]                                    # noqa: E731
    total = lambda rs: sum(r[3] for r in rs)                                # noqa: E731
    gu = "ffn.shared_experts.gate_up_proj"
    assert mods(d["moe"])[:3] == ["hc_ffn_fn", "ffn.gate", gu] and total(d["moe"]) == 16 * MIB
    assert mods(d["next"])[:3] == ["hc_attn_fn", "attn.fused_wqa_wkv", "attn.fused_wqa_wkv"]
    assert mods(o["moe"]) == ["ffn.gate", "hc_ffn_fn", gu] and total(o["moe"]) == 8 * MIB
    assert mods(o["next"]) == ["attn.fused_wqa_wkv", "attn.fused_wqa_wkv", "hc_attn_fn", "attn.wq_b"]
    assert all(r[0] == 1 for r in o["next"]) and total(o["next"]) == total(d["next"]) == 16 * MIB
    assert sorted(o["next"]) == sorted(d["next"])                           # the same bytes, another order
    assert o["attn"] == d["attn"]
    assert mods(c["moe"]) == mods(d["moe"])[:3] and total(c["moe"]) == 8 * MIB  # the cap alone keeps the order


def _forward(m, layers, x, w, norm=None, extra=()):
    """The serve's point sequence per layer (wq_b -> attn, wo_b -> moe, MoE -> next), then the final norm; `extra`:
    "next" points of layers past the backbone (a drafter's MoE) after it."""
    y = x
    for layer in range(layers):
        y = y @ w[(layer, "mat")]
        m.point("attn", layer, y.shape[0])
        y = torch.tanh(y)
        m.point("moe", layer, y.shape[0])
        y = y * 0.5 + 1.0
        m.point("next", layer, y.shape[0])
    if norm is not None:
        y = norm(y)
    for layer in extra:
        m.point("next", layer, y.shape[0])
    return y


def _recorder(m, monkeypatch):
    calls = []
    real = m.ext()

    class Rec:
        def l2pf(self, ptrs, sizes, *a):
            calls.append((list(ptrs), list(sizes)))
            return real.l2pf(ptrs, sizes, *a)
    monkeypatch.setattr(m, "_ext", Rec())
    return calls


def test_head_fork_joined_by_final_norm(reload, monkeypatch):
    layers = 3
    w = _layer_weights(layers)
    for i in range(layers):
        w[(i, "mat")] = torch.randn(256, 256, device=DEV) / 16
    head = _t(320, 1024, 64)                                                # a K4 head trellis (a quarter of one)
    x = torch.randn(6, 256, device=DEV)
    plain = reload()
    ref = _forward(plain, layers, x, w, _Norm())

    m = reload(EXL3_L2PF="attn,moe,next", EXL3_L2PF_HC="1", EXL3_L2PF_ORDER="1")
    assert m.HEAD_BUDGET == 4 * MIB and m.MOE_CAP == 8 * MIB
    _register(m, w)
    model = _model(w, layers)
    m.register_decoder_layers(model)
    m.register_head([head])
    calls = _recorder(m, monkeypatch)
    out = _forward(m, layers, x, w)                                         # no norm yet: the head is in flight
    assert m._pending and calls[-1] == ([head.data_ptr()], [4 * MIB])
    model.norm(out)                                                         # the hook joins it
    assert not m._pending
    out = _forward(m, layers, x, w, model.norm, extra=(layers, layers + 1))  # drafter layers past the backbone
    assert not m._pending and calls[-1] == ([head.data_ptr()], [4 * MIB])   # ... join, never fork
    assert torch.equal(out, ref)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _forward(m, layers, x, w, model.norm)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    n0 = len(calls)
    with torch.cuda.graph(g):                                               # raises on unjoined side-stream work
        out = _forward(m, layers, x, w, model.norm)
    assert ([head.data_ptr()], [4 * MIB]) in calls[n0:]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref) and not m._pending


def test_head_needs_the_hook(reload, monkeypatch):
    """No final-norm hook (patch 0017 absent, or a model without a norm module): the last layer only joins, so a
    captured forward still ends with nothing in flight."""
    layers = 2
    w = _layer_weights(layers)
    for i in range(layers):
        w[(i, "mat")] = torch.randn(256, 256, device=DEV) / 16
    m = reload(EXL3_L2PF="attn,moe,next", EXL3_L2PF_ORDER="1")
    _register(m, w)
    m.register_head([_t(320, 1024, 64)])
    calls = _recorder(m, monkeypatch)
    x = torch.randn(4, 256, device=DEV)
    _forward(m, layers, x, w)
    assert not m._pending and len(calls) == 3 * layers - 1                 # every point but the last layer's "next"
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _forward(m, layers, x, w)
    g.replay()
    torch.cuda.synchronize()
    m.register_decoder_layers(SimpleNamespace(layers=[], start_layer=0, end_layer=0, norm=None))
    assert not m._head_hooked


def _draft_forward(m, first, n, x, w):
    """The draft layers' point sequence (wq_b -> attn, wo_b -> moe, MoE -> next) for layers first .. first + n - 1."""
    y = x
    for layer in range(first, first + n):
        y = y @ w[(layer, "mat")]
        m.point("attn", layer, y.shape[0])
        y = torch.tanh(y)
        m.point("moe", layer, y.shape[0])
        y = y * 0.5 + 1.0
        m.point("next", layer, y.shape[0])
    return y


def test_draft_chain(reload, monkeypatch):
    """EXL3_L2PF_DRAFTER=1: draft layers (registered with drafter=True) fork their own attn / moe / next points, the
    last draft layer's `next` only joins, the backbone's last layer still forks the LM head, and a captured draft
    forward ends joined; without the flag a draft module registers nothing and its points only join."""
    backbone, drafts = 2, 3
    w = _layer_weights(backbone + drafts)
    for i in range(backbone + drafts):
        w[(i, "mat")] = torch.randn(256, 256, device=DEV) / 16
    x = torch.randn(5, 256, device=DEV)
    ref = _draft_forward(reload(), backbone, drafts, x, w)

    m = reload(EXL3_L2PF="attn,moe,next", EXL3_L2PF_ORDER="1", EXL3_L2PF_DRAFTER="1")
    for (layer, module), ts in w.items():
        if module.startswith(("hc_", "ffn.gate", "attn.wo_a")) or module == "mat":
            continue                                  # the drafter's wo_a is FP8: not registered
        m.register(layer, module, ts, drafter=layer >= backbone)
    assert m._last_layer == backbone - 1 and m._drafter == set(range(backbone, backbone + drafts))
    m.register_head([_t(320, 1024, 64)])
    m.register_decoder_layers(_model(w, backbone))
    calls = _recorder(m, monkeypatch)
    _forward(m, backbone, x, w)                        # backbone: its last layer forks the head (joined by the norm hook)
    assert calls[-1][1] == [4 * MIB]
    m.join()
    n0 = len(calls)
    out = _draft_forward(m, backbone, drafts, x, w)
    got = calls[n0:]
    assert not m._pending and len(got) == 3 * drafts - 1                    # every draft point but the last `next`
    first_attn = _order(m, backbone, "attn", w)
    assert [k[:2] for k in first_attn] == [(backbone, "attn.wo_b")]         # wo_b only (wo_a FP8)
    nxt = _order(m, backbone, "next", w)
    assert {k[:2] for k in nxt} == {(backbone + 1, "attn.fused_wqa_wkv"), (backbone + 1, "attn.wq_b")}
    assert torch.equal(out, ref)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):                                               # raises on unjoined side-stream work
        out = _draft_forward(m, backbone, drafts, x, w)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref) and not m._pending

    off = reload(EXL3_L2PF="attn,moe,next", EXL3_L2PF_ORDER="1")
    off.register(backbone, "attn.wo_b", w[(backbone, "attn.wo_b")], drafter=True)
    assert (backbone, "attn.wo_b") not in off._reg and not off._drafter
    calls = _recorder(off, monkeypatch)
    _draft_forward(off, backbone, drafts, x, w)
    assert calls == [] and not off._pending

