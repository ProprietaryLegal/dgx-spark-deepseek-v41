# SPDX-License-Identifier: Apache-2.0
"""TensorFold's prompt-chunk path for the routed experts (tf_prompt.py, VLLM_EXL3_MOE_TF_PROMPT=1) against ours, at a
TP=2 rank's DeepSeek-V4.1 shapes (H 5120, I 1152) on random EXL3 banks (test_kernels_gpu.py's helpers): the same values
within rounding, the same bits on every run, a row's bits independent of the chunk around it, unrouted picks skipped.
Runs where exllamav3's extension and the JIT toolchain exist (the serve image on a Spark):
    CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_tf_prompt_gpu.py
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm_exl3_moe import kernels as K  # noqa: E402

if not K.have_ext():
    pytest.skip("exllamav3_ext (v1.6.0) not importable", allow_module_level=True)
from test_kernels_gpu import DEV, H, KS, LIMIT, make_bank, random_experts, rel_err, routing  # noqa: E402

import vllm_exl3_moe.forward as F  # noqa: E402
from vllm_exl3_moe import tf_prompt  # noqa: E402
from vllm_exl3_moe.forward import reference_moe  # noqa: E402


@pytest.fixture(scope="module")
def rank0():
    return make_bank(random_experts(KS, seed=3), KS, tp_size=2, tp_rank=0)


def tf(b, x, w, ids):
    return tf_prompt.routed(x, w, ids, b["perm"], b["bank"], LIMIT, False, True)


def ours(b, x, w, ids, monkeypatch):
    monkeypatch.setattr(F, "_DET_PREFILL", True)
    monkeypatch.setattr(F, "_TF_PROMPT", False)
    return F.routed_moe_forward(x, w, ids, b["perm"], b["groups"], b["bank"], b["coop"], b["fused"], LIMIT, False,
                                True).float()


@pytest.mark.parametrize("m", [64, 600, 2048])
def test_matches_our_prefill(rank0, m, monkeypatch):
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=20 + m)
    a = tf(rank0, x, w, ids)
    b = ours(rank0, x, w, ids, monkeypatch)
    print(f"m={m}: rel {rel_err(a, b):.2e}")
    assert a.shape == (m, H) and a.dtype == torch.float32
    assert rel_err(a, b) < 1e-2, rel_err(a, b)


def test_matches_reference(rank0):
    m = 600
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=31)
    a = tf(rank0, x, w, ids)
    sub = slice(0, 24)
    ref = reference_moe(x[sub], w[sub], ids[sub], rank0["bank"], LIMIT, False, True)
    assert rel_err(a[sub], ref) < 2e-2, rel_err(a[sub], ref)


def test_same_bits_every_run_and_any_chunk(rank0):
    m = 2048
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=41)
    runs = [tf(rank0, x, w, ids) for _ in range(3)]
    assert all(torch.equal(r, runs[0]) for r in runs[1:]), "differs run to run"
    part = tf(rank0, x[:600].contiguous(), w[:600].contiguous(), ids[:600].contiguous())
    assert torch.equal(part, runs[0][:600]), "a row's bits depend on the chunk around it"


def test_unrouted_picks(rank0, monkeypatch):
    m = 256
    x = torch.randn((m, H), device=DEV, dtype=torch.bfloat16)
    w, ids = routing(m, len(KS), seed=51)
    ids[3, :] = -1                       # a row with no experts
    ids[7, 2] = -1
    a = tf(rank0, x, w, ids)
    assert torch.count_nonzero(a[3]) == 0
    b = ours(rank0, x, w, ids, monkeypatch)
    assert rel_err(a, b) < 1e-2
