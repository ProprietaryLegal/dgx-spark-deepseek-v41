# SPDX-License-Identifier: Apache-2.0
"""l2pf.py + csrc/exl3_l2pf.cu: the paced L2 prefetch of the next dense weights on a side stream.

The kernel runs (paced and all-at-once), a prefetch changes no output, the points fork and join so that a CUDA-graph
capture of a forward-like sequence ends with no side-stream work (the last layer's "next" point joins), and with
EXL3_L2PF_BENCH=1 a read of 16 MiB is timed cold vs right after a prefetch (idle GPU only).

    EXL3_L2PF=attn,moe,next python3 -m pytest -q -s test_l2pf_gpu.py
"""
import importlib
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
pytest.importorskip("vllm")   # every test takes the l2pf fixture, and l2pf.py imports vllm.logger

DEV = "cuda"


@pytest.fixture()
def l2pf(monkeypatch):
    monkeypatch.setenv("EXL3_L2PF", "attn,moe,next")
    import vllm_exl3_moe.l2pf as m
    m = importlib.reload(m)
    yield m
    monkeypatch.delenv("EXL3_L2PF")
    importlib.reload(m)


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
