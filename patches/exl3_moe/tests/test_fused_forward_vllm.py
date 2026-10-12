# SPDX-License-Identifier: Apache-2.0
"""Exl3RoutedMoEMethod.fused_moe_forward (VLLM_EXL3_MOE_FUSED=1, called by patch 0016 from DeepseekV4MoE) against
the launches the runner would make with our router: exl3_router.router_topk, then smoe with the shared expert as a
slot. The glue must hand the fused launch the gate weight, the correction bias, routed_scaling_factor, the shared
MLP's EXL3 linears and the padding rows exactly: the outputs are compared bit for bit. Runs in the runtime image
(vLLM installed), on a GPU; DeepseekV4MoE is stood in by its attributes.
    python3 -m pytest -q test_fused_forward_vllm.py
"""
import os
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
os.environ["VLLM_EXL3_MOE_FUSED"] = "1"             # read at import of moe_method
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_kernels_gpu import DEV, H, LIMIT, TOPK, make_bank, random_experts  # noqa: E402
from test_smoe_shared_gpu import cut  # noqa: E402

N, SCALE = 32, 1.5
KS = [2, 3, 4, 5] * 8


def test_fused_moe_forward_matches_router_then_smoe():
    from vllm_exl3_moe import moe_method as MM
    from vllm_exl3_moe.jit import load
    from vllm_exl3_moe.plan import CODEBOOK_MUL1
    from vllm_exl3_moe.smoe import SmoeLayer

    experts = random_experts(KS + [5], seed=31)
    b = make_bank(experts[:-1], KS, 2, 0)
    g = torch.Generator().manual_seed(7)
    gw = (torch.randn(N, H, generator=g) * 0.02).to(DEV, torch.bfloat16)
    bias = (torch.randn(N, generator=g) * 0.05).to(DEV)
    gl, ul, dl = cut(experts[-1])
    mul1 = torch.tensor([CODEBOOK_MUL1 - 2 ** 32 if CODEBOOK_MUL1 >= 2 ** 31 else CODEBOOK_MUL1], dtype=torch.int32,
                        device=DEV)
    moe = SimpleNamespace(
        gate=SimpleNamespace(weight=gw, e_score_correction_bias=bias, tid2eid=None, is_unquantized=True, bias_vl=None),
        shared_experts=SimpleNamespace(gate_up_proj=SimpleNamespace(_exl3=[gl, ul], mul1=mul1.repeat(2)),
                                       down_proj=SimpleNamespace(_exl3=[dl], mul1=mul1)),
        routed_scaling_factor=SCALE, reduce_results=True, image_sentinel_lo=0, use_sequence_parallel=False)
    me = SimpleNamespace(smoe=SmoeLayer(b["bank"], TOPK), perm=b["perm"], act_limit=LIMIT, tp_size=1, layer_idx=3,
                         settings=SimpleNamespace(decode_kernel="coop"))
    me._fused_setup = types.MethodType(MM.Exl3RoutedMoEMethod._fused_setup, me)
    rx = load("exl3_router", "EXL3_ROUTER_FLAGS")
    ref_sl = SmoeLayer(b["bank"], TOPK)
    ref_sl.attach_shared(gl, ul, dl)
    for m in (2, 4, 6, 8):
        x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
        out = MM.Exl3RoutedMoEMethod.fused_moe_forward(me, moe, x, None)
        assert out is not None, "fused path declined a qualifying step"
        logits = torch.empty(m, N, dtype=torch.float32, device=DEV)
        tw = torch.empty(m, TOPK, dtype=torch.float32, device=DEV)
        ti = torch.empty(m, TOPK, dtype=torch.int32, device=DEV)
        rx.router_topk(x, gw, logits, bias, torch.empty(0, device=DEV), torch.empty(0, dtype=torch.int64, device=DEV),
                       0, SCALE, tw, ti, torch.zeros(1, dtype=torch.int32, device=DEV),
                       torch.empty(2 * m * N, device=DEV))
        ref = ref_sl(x, ti, tw, b["perm"], LIMIT, None, shared=True)
        assert torch.equal(out, ref), (m, (out.float() - ref.float()).abs().max().item())
    # 1 token (below VLLM_EXL3_MOE_FUSED_MIN_ROWS, default 2) and 9 tokens: not fused steps (the runner runs)
    assert MM.Exl3RoutedMoEMethod.fused_moe_forward(me, moe, torch.randn((1, H), device=DEV, dtype=torch.bfloat16),
                                                    None) is None
    assert MM.Exl3RoutedMoEMethod.fused_moe_forward(me, moe, torch.randn((9, H), device=DEV, dtype=torch.bfloat16),
                                                    None) is None
