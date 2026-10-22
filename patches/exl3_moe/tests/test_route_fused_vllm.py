# SPDX-License-Identifier: Apache-2.0
"""Exl3RoutedMoEMethod.fused_moe_forward with VLLM_EXL3_MOE_ROUTE_FUSED=1 (called by patch 0016 from DeepseekV4MoE):
router_topk in one exl3_router launch, the routed experts on exl3_smoe, the shared expert (any callable module) on
vLLM's aux stream, their sum. Against the launches it replaces, bit for bit: exl3_router.router_topk's ids and weights
into smoe, plus the shared output; 1-8 tokens are routed this way, 9 tokens and hash-routed layers keep the runner.
Runs in the runtime image (vLLM installed), on a GPU; DeepseekV4MoE is stood in by its attributes.
    python3 -m pytest -q test_route_fused_vllm.py
"""
import os
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_kernels_gpu import DEV, H, LIMIT, TOPK, make_bank, random_experts  # noqa: E402

N, SCALE = 32, 1.5
KS = [2, 3, 4, 5] * 8


def test_route_fused_matches_router_then_smoe(monkeypatch):
    from vllm_exl3_moe import moe_method as MM
    from vllm_exl3_moe.jit import load
    from vllm_exl3_moe.smoe import SmoeLayer

    monkeypatch.setattr(MM, "_ROUTE_FUSED", 1)
    monkeypatch.setattr(MM, "_SMOE_FUSED", 0)
    experts = random_experts(KS, seed=41)
    b = make_bank(experts, KS, 2, 0)
    g = torch.Generator().manual_seed(9)
    gw = (torch.randn(N, H, generator=g) * 0.02).to(DEV, torch.bfloat16)
    bias = (torch.randn(N, generator=g) * 0.05).to(DEV)
    wsh = (torch.randn(H, H, generator=g) * 0.01).to(DEV, torch.bfloat16)
    moe = SimpleNamespace(
        gate=SimpleNamespace(weight=gw, e_score_correction_bias=bias, tid2eid=None, is_unquantized=True, bias_vl=None),
        shared_experts=lambda x: x @ wsh,
        routed_scaling_factor=SCALE, reduce_results=True, image_sentinel_lo=0, use_sequence_parallel=False)
    me = SimpleNamespace(smoe=SmoeLayer(b["bank"], TOPK), perm=b["perm"], act_limit=LIMIT, tp_size=1, layer_idx=3,
                         settings=SimpleNamespace(decode_kernel="coop"))
    me._route_fused_forward = types.MethodType(MM.Exl3RoutedMoEMethod._route_fused_forward, me)
    rx = load("exl3_router", "EXL3_ROUTER_FLAGS")
    ref_sl = SmoeLayer(b["bank"], TOPK)
    for m in (1, 2, 3, 6, 8):
        x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
        out = MM.Exl3RoutedMoEMethod.fused_moe_forward(me, moe, x, None)
        torch.cuda.synchronize()
        assert out is not None, f"route-fused path declined a {m}-token step"
        logits = torch.empty(m, N, dtype=torch.float32, device=DEV)
        tw = torch.empty(m, TOPK, dtype=torch.float32, device=DEV)
        ti = torch.empty(m, TOPK, dtype=torch.int32, device=DEV)
        rx.router_topk(x, gw, logits, bias, torch.empty(0, device=DEV), torch.empty(0, dtype=torch.int64, device=DEV),
                       0, SCALE, tw, ti, torch.zeros(1, dtype=torch.int32, device=DEV),
                       torch.empty(2 * m * N, device=DEV))
        ref = ref_sl(x, ti, tw, b["perm"], LIMIT, None) + x @ wsh
        assert torch.equal(out, ref), (m, (out.float() - ref.float()).abs().max().item())
    assert MM.Exl3RoutedMoEMethod.fused_moe_forward(me, moe, torch.randn((9, H), device=DEV, dtype=torch.bfloat16),
                                                    None) is None
    moe.gate.tid2eid = torch.zeros(4, dtype=torch.int64, device=DEV)          # a hash-routed layer: the runner runs
    assert MM.Exl3RoutedMoEMethod.fused_moe_forward(me, moe, torch.randn((4, H), device=DEV, dtype=torch.bfloat16),
                                                    None) is None
