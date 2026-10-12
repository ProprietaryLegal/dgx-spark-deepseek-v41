# SPDX-License-Identifier: Apache-2.0
"""fused_ops (one prep launch + one group-sum launch on the bf16 decode path) == the torch ops it replaces, bit for bit,
for chained and concurrent K groups, with unrouted (-1) slots.   python3 -m pytest -q test_fused_ops_gpu.py
"""
import pytest
import torch

from test_kernels_gpu import DEV, H, KS, LIMIT, TOPK, make_bank, random_experts, routing
from vllm_exl3_moe import forward as F
from vllm_exl3_moe import kernels as K

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def bank():
    return make_bank(random_experts(KS, seed=11), KS)


@pytest.fixture(scope="module")
def par(bank):
    return K.get_parallel_coop(DEV, 32, TOPK, H, bank["bank"].inter_local, K.MAX_K_GROUPS)


def run(b, x, w, ids, par, fused, monkeypatch):
    monkeypatch.setattr(F, "_FUSED", fused)
    return F.routed_moe_forward(x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"],
                                LIMIT, False, True, decode_kernel="coop", par=par)


@pytest.mark.parametrize("m", [1, 4, 6, 8, 16])
@pytest.mark.parametrize("parallel", [False, True])
def test_fused_equals_torch_ops(bank, par, m, parallel, monkeypatch):
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=300 + m)
    ids = ids.clone()
    ids[0, -1] = -1                                     # an unrouted slot
    p = par if parallel else None
    ref = run(bank, x, w, ids, p, False, monkeypatch)
    out = run(bank, x, w, ids, p, True, monkeypatch)
    assert out.dtype == ref.dtype == torch.bfloat16
    assert torch.equal(out, ref), (out.float() - ref.float()).abs().max().item()
