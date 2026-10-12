# SPDX-License-Identifier: Apache-2.0
"""Patch 0009: FULL cudagraphs with eager tensor-parallel all-reduces. A capture through BreakableCUDAGraphCapture
marked all-reduce-only breaks exactly at eager_all_reduce_break (ops decorated with eager_break_during_capture, like
attention, stay in the graph), the all-reduce runs eagerly into a graph-pool buffer the next segment reads, and the
replay equals an eager forward on new inputs. One GPU: a fake PyNccl stands in for the communicator (it
doubles its input, as a 2-rank all-reduce of identical shards would). Off by default (env unset: no break), and only
the plain PyNccl path breaks.

Run inside the dsv41 image on a GPU:  python3 -m pytest -q test_full_graph_eager_allreduce.py
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pytest.importorskip("vllm")   # runs in the serve image


class FakeNccl:
    disabled = False

    def all_reduce(self, in_tensor, out_tensor=None):
        out_tensor = torch.empty_like(in_tensor) if out_tensor is None else out_tensor
        return torch.mul(in_tensor, 2.0, out=out_tensor)


class FakeComm:
    def __init__(self, **extra):
        self.pynccl_comm = FakeNccl()
        self.ca_comm = None
        for k, v in extra.items():
            setattr(self, k, v)

    def all_reduce(self, input_):               # the communicator's own path (inside the graph)
        return self.pynccl_comm.all_reduce(input_)


def all_reduce(comm, t):
    from vllm.compilation.breakable_cudagraph import eager_all_reduce_break
    out = eager_all_reduce_break(comm, t)
    return out if out is not None else comm.all_reduce(t)


def forward(x, w1, w2, comm):
    h = all_reduce(comm, x @ w1)                 # attention-output style all-reduce
    y = all_reduce(comm, torch.relu(h) @ w2)      # MoE-output style all-reduce
    return y + 1.0


def capture(fn, allreduce_only=True):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        cap = BreakableCUDAGraphCapture(allreduce_only=allreduce_only)
        with cap:
            out = fn()
    torch.cuda.current_stream().wait_stream(s)
    return cap, out


def inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(4, 64, device="cuda", generator=g)
    w1 = torch.randn(64, 128, device="cuda", generator=g) / 8
    w2 = torch.randn(128, 32, device="cuda", generator=g) / 8
    return x, w1, w2


def test_breaks_at_all_reduces_and_replays_like_eager(monkeypatch):
    monkeypatch.setenv("VLLM_FULL_GRAPH_EAGER_ALLREDUCE", "1")
    x, w1, w2 = inputs()
    comm = FakeComm()
    forward(x, w1, w2, comm)                     # warm-up (allocator, cuBLAS handles) outside capture
    cap, out = capture(lambda: forward(x, w1, w2, comm))
    assert cap.num_eager_breaks == 2 and cap.num_graphs == 3, cap
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        ref = forward(x, w1, w2, comm)           # eager (no capture open: no break)
        cap.replay()
        torch.cuda.synchronize()
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_FULL_GRAPH_EAGER_ALLREDUCE", raising=False)
    x, w1, w2 = inputs()
    comm = FakeComm()
    forward(x, w1, w2, comm)
    cap, _ = capture(lambda: forward(x, w1, w2, comm))
    assert cap.num_eager_breaks == 0 and cap.num_graphs == 1


def test_other_all_reduce_paths_stay_in_graph(monkeypatch):
    monkeypatch.setenv("VLLM_FULL_GRAPH_EAGER_ALLREDUCE", "1")

    class Custom:
        disabled = False
    x, w1, w2 = inputs()
    comm = FakeComm(ca_comm=Custom())            # an active custom all-reduce owns the call: no break
    forward(x, w1, w2, comm)
    cap, _ = capture(lambda: forward(x, w1, w2, comm))
    assert cap.num_eager_breaks == 0


def test_attention_style_breaks_stay_in_graph(monkeypatch):
    """The serve build has eager_break_during_capture active (attention, the sparse indexer): inside an all-reduce-only
    capture those ops must be captured, or a FULL replay would run them eagerly without a forward context."""
    monkeypatch.setenv("VLLM_FULL_GRAPH_EAGER_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    @eager_break_during_capture
    def attn_like(t, out):
        torch.sin(t, out=out)
        return out

    x, w1, w2 = inputs()
    comm = FakeComm()
    buf = torch.empty(4, 128, device="cuda")

    def fwd():
        h = all_reduce(comm, x @ w1)
        attn_like(h, buf)
        return all_reduce(comm, torch.relu(buf) @ w2)
    fwd()
    cap, out = capture(fwd)
    assert cap.num_eager_breaks == 2, cap                 # the two all-reduces only
    x.copy_(torch.randn_like(x))
    ref = fwd()
    cap.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
    cap_pw, _ = capture(fwd, allreduce_only=False)          # a breakable PIECEWISE-style capture: unchanged
    assert cap_pw.num_eager_breaks == 1, cap_pw             # the attention-style op only, no all-reduce break


def test_eager_in_full_graph_ops_still_break(monkeypatch):
    """Ops that fork / join a side stream across segments (V4.1 Engram's prefetch start / finish) are marked
    eager_in_full_graph and break in an all-reduce-only capture too, so no segment ends with unjoined work."""
    monkeypatch.setenv("VLLM_FULL_GRAPH_EAGER_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture, eager_in_full_graph

    side = torch.cuda.Stream()
    done = torch.cuda.Event()
    x, w1, w2 = inputs()
    comm = FakeComm()
    staged = torch.empty(4, 64, device="cuda")

    @eager_break_during_capture
    @eager_in_full_graph
    def start_prefetch(src, dst):                         # side stream forked before the first all-reduce
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            torch.mul(src, 3.0, out=dst)
        done.record(side)

    @eager_break_during_capture
    @eager_in_full_graph
    def finish_prefetch():                                # joined after it
        torch.cuda.current_stream().wait_event(done)

    def fwd():
        start_prefetch(x, staged)
        h = all_reduce(comm, x @ w1)
        finish_prefetch()
        return all_reduce(comm, torch.relu(h) @ w2) + staged.sum()
    fwd()
    torch.cuda.synchronize()
    cap, out = capture(fwd)                               # would raise "unjoined work" without the marker
    assert cap.num_eager_breaks == 4, cap
    x.copy_(torch.randn_like(x))
    ref = fwd()
    torch.cuda.synchronize()
    cap.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_parallel_region_keeps_overlap_without_breaks(monkeypatch):
    """maybe_execute_in_parallel keeps its aux stream in an all-reduce-only capture (vLLM turns it off for breakable
    PIECEWISE captures), and an all-reduce issued inside the forked region stays in the graph."""
    monkeypatch.setenv("VLLM_FULL_GRAPH_EAGER_ALLREDUCE", "1")
    from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

    aux = torch.cuda.Stream()
    e0, e1 = torch.cuda.Event(), torch.cuda.Event()
    x, w1, w2 = inputs()
    comm = FakeComm()
    seen = []

    def on_aux():
        seen.append(torch.cuda.current_stream() == aux)
        return (x @ w1) * 0.5

    def fwd():
        a, b = maybe_execute_in_parallel(lambda: all_reduce(comm, x @ w1), on_aux, e0, e1, aux)
        return all_reduce(comm, torch.relu(a + b) @ w2)
    fwd()
    seen.clear()
    cap, out = capture(fwd)
    assert seen == [True], seen                           # fn1 ran on the aux stream during the capture
    assert cap.num_eager_breaks == 1, cap                 # only the all-reduce outside the parallel region
    x.copy_(torch.randn_like(x))
    ref = fwd()
    torch.cuda.synchronize()
    cap.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)
