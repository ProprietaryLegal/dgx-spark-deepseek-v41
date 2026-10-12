# SPDX-License-Identifier: Apache-2.0
"""EXL3_ROUTER=1 against the pinned vLLM (runs inside the runtime image on a GPU; skipped without vLLM).

The plugin replaces GateLinear: a gate vLLM builds is our subclass, decode-size bf16 inputs go through exl3_router
(fp32 logits equal to x @ W^T within fp32-accumulation error) and everything else (more rows than the kernel takes,
fp32 inputs) falls back to GateLinear's own tiers with the same result.
    EXL3_KERNEL_ARCH=121 python3 -m pytest -q test_router_vllm_gpu.py
"""

import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
os.environ["EXL3_ROUTER"] = "1"            # read at import of vllm_exl3_moe.router
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

import vllm_exl3_moe  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def gate():
    vllm_exl3_moe.register()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    from vllm.distributed.parallel_state import (init_distributed_environment, initialize_model_parallel,
                                                 model_parallel_is_initialized)

    with set_current_vllm_config(VllmConfig()):
        if not model_parallel_is_initialized():          # GateLinear (a ReplicatedLinear) asks the TP group its rank
            import socket
            with socket.socket() as sk:
                sk.bind(("127.0.0.1", 0))
                port = sk.getsockname()[1]
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                         distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
            initialize_model_parallel(1, 1)
        # the serve builds the gate under the model dtype (bf16, DeepseekV4MoE passes no params_dtype); a bare test's
        # default dtype is fp32, which would leave the weight fp32 and every call on GateLinear's own tiers
        g = GateLinear(5120, 384, bias=False, out_dtype=torch.float32, params_dtype=torch.bfloat16,
                       prefix="model.layers.3.ffn.gate")
    g = g.to("cuda")
    with torch.no_grad():
        g.weight.copy_((torch.randn(384, 5120, device="cuda") * 0.02).to(g.weight.dtype))
    return g


def logits(out):
    return out[0] if isinstance(out, tuple) else out


def test_gate_is_ours(gate):
    assert type(gate).__name__ == "Exl3GateLinear"
    assert gate.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("rows", [1, 4, 6, 8])
def test_decode_rows_use_router_kernel(gate, rows, monkeypatch):
    from vllm_exl3_moe import router

    calls = []
    real = router.ext().router_logits
    monkeypatch.setattr(router, "_ext", type("E", (), {"router_logits": staticmethod(
        lambda *a: (calls.append(1), real(*a))[1])})())
    x = torch.randn(rows, 5120, device="cuda").to(torch.bfloat16)
    y = logits(gate(x))
    assert calls, "exl3_router was not called"
    assert y.dtype == torch.float32 and y.shape == (rows, 384)
    ref = x.double() @ gate.weight.double().T
    assert (y.double() - ref).abs().max().item() < 2e-4 * ref.abs().max().item()


def test_larger_batches_fall_back(gate):
    x = torch.randn(40, 5120, device="cuda").to(torch.bfloat16)
    y = logits(gate(x))
    ref = x.double() @ gate.weight.double().T
    assert y.shape == (40, 384)
    assert (y.double() - ref).abs().max().item() < 1e-2 * ref.abs().max().item()
