# SPDX-License-Identifier: Apache-2.0
"""Registration against the pinned vLLM (runs inside the runtime image; skipped without vLLM).

No GPU work: checks that the plugin resolves `deepseek_v41_exl3`, keeps the V4.1 FP8 description of
the non-expert tensors, does not steal official checkpoints, and routes layers as designed.
"""

import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vllm_exl3_moe  # noqa: E402
from vllm_exl3_moe.plan import synthetic_plan  # noqa: E402

OFFICIAL_QCFG = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
                 "scale_fmt": "ue8m0", "expert_dtype": "fp4"}


def ours(plan_dict=None):
    q = dict(OFFICIAL_QCFG)
    q["quant_method"] = vllm_exl3_moe.QUANT_METHOD
    q["exl3_moe"] = {"plan": plan_dict or synthetic_plan(num_layers=40, uniform_k=3).to_dict()}
    return q


def test_register_and_resolve():
    vllm_exl3_moe.register()
    vllm_exl3_moe.register()                       # idempotent
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config

    from vllm_exl3_moe.config import DeepseekV41Exl3Config

    assert vllm_exl3_moe.QUANT_METHOD in QUANTIZATION_METHODS
    assert get_quantization_config(vllm_exl3_moe.QUANT_METHOD) is DeepseekV41Exl3Config


def test_from_config_keeps_fp8_description():
    vllm_exl3_moe.register()
    from vllm_exl3_moe.config import DeepseekV41Exl3Config

    cfg = DeepseekV41Exl3Config.from_config(ours())
    assert cfg.get_name() == vllm_exl3_moe.QUANT_METHOD
    assert cfg.is_checkpoint_fp8_serialized            # fp8.py:159-160 needs "fp8" in quant_method
    assert cfg.weight_block_size == [32, 32]
    assert cfg.exl3_plan.has_layer(0) and not cfg.exl3_plan.has_layer(40)


def test_override_claims_only_our_checkpoints():
    vllm_exl3_moe.register()
    from vllm_exl3_moe.config import DeepseekV41Exl3Config

    assert DeepseekV41Exl3Config.override_quantization_method(ours(), None) == vllm_exl3_moe.QUANT_METHOD
    assert DeepseekV41Exl3Config.override_quantization_method(OFFICIAL_QCFG, None) is None


def test_routed_layer_index():
    from vllm_exl3_moe.config import routed_layer_index

    assert routed_layer_index("language_model.model.layers.17.ffn.experts") == 17
    assert routed_layer_index("model.layers.41.ffn.experts") == 41          # DSpark drafter layer
    assert routed_layer_index("model.layers.3.attn.wq_a") is None


@pytest.mark.parametrize("head", ["mxfp8", "exl3"])
def test_head_method_only_for_lm_head(head):
    """The head setting applies to the lm_head only: another vocab projection given this config (a DSpark Markov head,
    markov_w2) must not get a head method that expects head.* pieces it does not have."""
    vllm_exl3_moe.register()
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm_exl3_moe.config import DeepseekV41Exl3Config
    from vllm_exl3_moe.head import Exl3HeadMethod, Mxfp8MarlinHeadMethod

    q = ours()
    q["exl3_moe"]["head"] = head
    if head == "exl3":
        q["exl3_moe"]["head_k"] = 6
    cfg = DeepseekV41Exl3Config.from_config(q)
    lm_head = ParallelLMHead.__new__(ParallelLMHead)
    want = Exl3HeadMethod if head == "exl3" else Mxfp8MarlinHeadMethod
    for prefix in ("lm_head", "language_model.lm_head"):
        assert isinstance(cfg.get_quant_method(lm_head, prefix), want), prefix
    other = cfg.get_quant_method(lm_head, "model.markov_head.markov_w2")
    assert not isinstance(other, (Exl3HeadMethod, Mxfp8MarlinHeadMethod)), type(other)


def test_drafter_dense_routing(monkeypatch):
    """exl3_moe.dense.drafter: the DSpark drafter's dense linears (layers >= num_hidden_layers) and main_proj get the EXL3
    method at their K, kept out of the L2 prefetch chain; wo_a and unlisted modules keep FP8; main_proj only when listed;
    backbone routing is unchanged."""
    vllm_exl3_moe.register()
    from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
    import vllm_exl3_moe.config as C
    from vllm_exl3_moe.dense import Exl3DenseLinearMethod

    monkeypatch.setattr(C, "_num_hidden_layers", lambda: 40)
    q = ours()
    q["exl3_moe"]["dense"] = {"k": {"attn.wq_b": 5}, "layers": [0, 39],
                              "drafter": {"attn.wq_b": 4, "attn.wo_b": 4}}
    cfg = C.DeepseekV41Exl3Config.from_config(q)
    col, row, rep = (ColumnParallelLinear.__new__(ColumnParallelLinear), RowParallelLinear.__new__(RowParallelLinear),
                     ReplicatedLinear.__new__(ReplicatedLinear))
    m = cfg.get_quant_method(col, "model.layers.40.attn.wq_b")
    assert isinstance(m, Exl3DenseLinearMethod) and m.k == 4 and m.prefetch is False
    m = cfg.get_quant_method(row, "model.layers.42.attn.wo_b")
    assert isinstance(m, Exl3DenseLinearMethod) and m.k == 4
    assert not isinstance(cfg.get_quant_method(rep, "model.main_proj"), Exl3DenseLinearMethod)   # not listed: FP8
    assert not isinstance(cfg.get_quant_method(col, "model.layers.40.attn.wo_a"), Exl3DenseLinearMethod)
    assert not isinstance(cfg.get_quant_method(col, "model.layers.41.ffn.shared_experts.gate_up_proj"),
                          Exl3DenseLinearMethod)                       # not listed: FP8
    m = cfg.get_quant_method(col, "model.layers.7.attn.wq_b")          # backbone: dense.k, prefetch chain on
    assert isinstance(m, Exl3DenseLinearMethod) and m.k == 5 and m.prefetch is True
    # main_proj may be EXL3 (its input is scaled into fp16 range above 16 rows, dense.py); wo_a may not
    q["exl3_moe"]["dense"]["drafter"] = {"main_proj": 5}
    cfg = C.DeepseekV41Exl3Config.from_config(q)
    m = cfg.get_quant_method(rep, "model.main_proj")
    assert isinstance(m, Exl3DenseLinearMethod) and m.k == 5 and m.prefetch is False
    q["exl3_moe"]["dense"]["drafter"] = {"attn.wo_a": 5}
    with pytest.raises(ValueError, match="unsupported"):
        C.DeepseekV41Exl3Config.from_config(q)


def test_woa_qdq_hooks_only_fp8_wo_a(monkeypatch):
    """VLLM_DSV41_WOA_QDQ=1: a wo_a that stays FP8 (the DSpark drafter's) keeps the checkpoint's method object and class
    (vLLM picks the weight loader by the class name) with the round trip added to its loading (woa_qdq.hook_fp8); other
    FP8 linears are untouched and the backbone's EXL3 wo_a does it itself (dense.py); nothing changes with the flag off."""
    vllm_exl3_moe.register()
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    import vllm_exl3_moe.config as C
    from vllm_exl3_moe.dense import Exl3DenseLinearMethod

    monkeypatch.setattr(C, "_num_hidden_layers", lambda: 40)
    q = ours()
    q["exl3_moe"]["dense"] = {"k": {"attn.wo_a": 5}, "layers": [0, 39]}
    cfg = C.DeepseekV41Exl3Config.from_config(q)
    col = ColumnParallelLinear.__new__(ColumnParallelLinear)
    monkeypatch.delenv("VLLM_DSV41_WOA_QDQ", raising=False)
    plain = cfg.get_quant_method(col, "model.layers.40.attn.wo_a")
    assert "process_weights_after_loading" not in vars(plain)
    monkeypatch.setenv("VLLM_DSV41_WOA_QDQ", "1")
    m = cfg.get_quant_method(col, "model.layers.40.attn.wo_a")
    assert type(m) is type(plain) and "process_weights_after_loading" in vars(m)
    assert "process_weights_after_loading" not in vars(cfg.get_quant_method(col, "model.layers.40.attn.wq_b"))
    assert isinstance(cfg.get_quant_method(col, "model.layers.3.attn.wo_a"), Exl3DenseLinearMethod)
