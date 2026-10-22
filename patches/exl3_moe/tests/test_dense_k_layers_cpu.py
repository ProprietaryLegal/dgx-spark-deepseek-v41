# SPDX-License-Identifier: Apache-2.0
"""exl3_moe.dense.k_layers: a dense module's K per layer (scripts/quantize/p4_dense_mix.py --layers): the listed layers
take the override, every other layer the module's "k", modules without an override keep "k", layers outside "layers"
stay FP8 (None), and the vLLM module names map to the config's checkpoint names (fused_wqa_wkv -> attn.wq_a,
gate_up_proj -> ffn.shared_experts.w1). CPU.

  PYTHONPATH=/plugin python3 -m pytest -q test_dense_k_layers_cpu.py      (serve image)
"""
import pytest

pytest.importorskip("vllm")

from vllm_exl3_moe.config import Exl3Settings  # noqa: E402


def settings(dense):
    s = Exl3Settings()
    s.dense = dense
    return s


def test_per_layer_override():
    s = settings({"k": {"attn.wq_a": 5, "attn.wq_b": 4, "attn.wo_a": 5, "attn.wo_b": 5, "ffn.shared_experts.w1": 5,
                        "ffn.shared_experts.w2": 5}, "layers": [0, 39],
                  "k_layers": {"attn.wo_b": {"30": 4, "39": 4}, "ffn.shared_experts.w1": {"38": 4},
                               "attn.wq_a": {"39": 4}}})
    assert s.dense_k("attn.wo_b", 30) == 4 and s.dense_k("attn.wo_b", 39) == 4
    assert s.dense_k("attn.wo_b", 29) == 5 and s.dense_k("attn.wo_b", 0) == 5
    assert s.dense_k("ffn.shared_experts.gate_up_proj", 38) == 4
    assert s.dense_k("ffn.shared_experts.gate_up_proj", 37) == 5
    assert s.dense_k("ffn.shared_experts.down_proj", 38) == 5       # w2 has no override
    assert s.dense_k("attn.fused_wqa_wkv", 39) == 4 and s.dense_k("attn.fused_wqa_wkv", 38) == 5
    assert s.dense_k("attn.wq_b", 39) == 4 and s.dense_k("attn.wo_a", 39) == 5


def test_no_override_and_range():
    s = settings({"k": {"attn.wo_b": 5}, "layers": [0, 9]})
    assert s.dense_k("attn.wo_b", 5) == 5
    assert s.dense_k("attn.wo_b", 10) is None                       # outside "layers": the checkpoint's FP8
    assert s.dense_k("attn.wq_b", 5) is None                        # no K for the module: FP8
    assert settings(None).dense_k("attn.wo_b", 5) is None
