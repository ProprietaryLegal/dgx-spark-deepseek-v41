# SPDX-License-Identifier: Apache-2.0
"""Patch 0026: the DSpark drafter's own lm_head (VLLM_DSPARK_OWN_HEAD=1).

(1) Flag off: _own_lm_head builds nothing, so __init__ builds today's placeholder and load_dspark_model aliases the
    target's head; the name map skips mtp.head.*. (2) Flag on: the head is built under OWN_HEAD_PREFIX with the drafter's
    quant config and kept (keeps_own_lm_head) only when the config gives that prefix a method (the recipe plugin's
    Exl3HeadMethod at exl3_moe.drafter_head_k); without one, or without a quant config, the head stays shared.
(3) load_weights routes mtp.head.{trellis,suh,svh,mul1} into lm_head through the head method's loaders (this rank's
    vocab columns of TP=2), skips them for a shared head, and refuses to start when a declared piece is missing.
(4) load_dspark_model keeps a drafter's own head and still aliases the target's otherwise. (5) Every other checkpoint
    name maps as before.
ParallelLMHead is a recording subclass in (2) (it resolves the method as the real one does, without the TP group); TP
size and rank are patched in (3); get_model and the config plumbing are stand-ins in (4). CPU only.

Run inside the dsv41 image:  python3 -m pytest -q test_dspark_own_head.py
"""
import types

import pytest
import torch

pytest.importorskip("vllm")   # runs in the serve image

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod
from vllm.models.deepseek_v41.nvidia import dspark as D

V, H, TP = 1024, 256, 2               # vocab, hidden: each rank's range is whole 128-wide blocks
OFFICIAL_QCFG = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
                 "scale_fmt": "ue8m0", "expert_dtype": "fp4"}


def bare(own=False):
    """A DSparkDeepseekV4ForCausalLM without its layers: enough for the name map, _own_lm_head and load_weights."""
    m = D.DSparkDeepseekV4ForCausalLM.__new__(D.DSparkDeepseekV4ForCausalLM)
    torch.nn.Module.__init__(m)
    object.__setattr__(m, "model", types.SimpleNamespace(confidence_head=object()))
    if own:
        m.keeps_own_lm_head = True
    return m


@pytest.mark.parametrize("own", [False, True])
def test_name_map(own):
    m = bare(own)
    for p in ("trellis", "suh", "svh", "mul1"):
        assert m._remap_dspark_name(f"mtp.head.{p}") == (f"lm_head.{p}" if own else None)
    assert m._remap_dspark_name("head.trellis") is None                 # the target's head
    assert m._remap_dspark_name("mtp.head.weight") is None              # not a piece the drafter loads
    assert m._remap_dspark_name("mtp.0.attn.wq_b.weight") == "model.layers.0.attn.wq_b.weight"
    assert m._remap_dspark_name("mtp.1.ffn.experts.3.w1.weight") == "model.layers.1.ffn.experts.3.w1.weight"
    assert m._remap_dspark_name("mtp.2.markov_head.head.weight") == "model.markov_head.markov_w2.weight"
    assert m._remap_dspark_name("mtp.2.norm.weight") == "model.norm.weight"
    assert m._remap_dspark_name("mtp.2.confidence_head.weight") == "model.confidence_head.weight"
    assert m._remap_dspark_name("mtp.0.main_proj.weight") == "model.main_proj.weight"


# --- (1) / (2): building the head ------------------------------------------------------------------------------------

class RecordingHead(ParallelLMHead):
    """Records its construction; resolves the quant method as ParallelLMHead does, without the TP group."""
    made: list = []

    def __init__(self, num_embeddings, embedding_dim, quant_config=None, prefix=""):
        torch.nn.Module.__init__(self)
        RecordingHead.made.append((num_embeddings, embedding_dim, quant_config, prefix))
        method = quant_config.get_quant_method(self, prefix) if quant_config is not None else None
        self.quant_method = method if method is not None else UnquantizedEmbeddingMethod()
        self.prefix = prefix


class HeadMethod(UnquantizedEmbeddingMethod):
    """A quantized head method (vllm_exl3_moe's Exl3HeadMethod subclasses UnquantizedEmbeddingMethod the same way)."""


class Claims:
    """A quant config that gives the method to one prefix (default: the drafter head's) and nothing to any other."""

    def __init__(self, method, prefix=None):
        self.method, self.prefix = method, prefix

    def get_quant_method(self, layer, prefix):
        return self.method if prefix == (self.prefix or D.OWN_HEAD_PREFIX) else None


@pytest.fixture
def made(monkeypatch):
    RecordingHead.made = []
    monkeypatch.setattr(D, "ParallelLMHead", RecordingHead)
    return RecordingHead.made


def drafter(quant_config):
    m = bare()
    m.config = types.SimpleNamespace(vocab_size=V, hidden_size=H)
    m.quant_config = quant_config
    return m


@pytest.mark.parametrize("flag", [None, "0", ""])
def test_flag_off_builds_nothing(made, monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("VLLM_DSPARK_OWN_HEAD", raising=False)
    else:
        monkeypatch.setenv("VLLM_DSPARK_OWN_HEAD", flag)
    m = drafter(Claims(HeadMethod()))
    assert m._own_lm_head("") is None and made == [] and m.keeps_own_lm_head is False


def test_flag_on_keeps_a_claimed_head(made, monkeypatch):
    monkeypatch.setenv("VLLM_DSPARK_OWN_HEAD", "1")
    method = HeadMethod()
    qc = Claims(method)
    m = drafter(qc)
    head = m._own_lm_head("")
    assert isinstance(head, RecordingHead) and head.quant_method is method
    assert made == [(V, H, qc, "mtp_head")] and head.prefix == D.OWN_HEAD_PREFIX == "mtp_head"
    assert m.keeps_own_lm_head is True and D.DSparkDeepseekV4ForCausalLM.keeps_own_lm_head is False


@pytest.mark.parametrize("qc", [None, Claims(HeadMethod(), prefix="lm_head"), Claims(UnquantizedEmbeddingMethod()),
                                Claims(UnquantizedLinearMethod())],
                         ids=["no-quant-config", "claims-lm_head-only", "unquantized-embedding", "unquantized-linear"])
def test_flag_on_without_a_method_shares(made, monkeypatch, qc):
    monkeypatch.setenv("VLLM_DSPARK_OWN_HEAD", "1")
    m = drafter(qc)
    assert m._own_lm_head("") is None and m.keeps_own_lm_head is False
    assert [x[3] for x in made] == ["mtp_head"]                         # one probe, dropped


def test_flag_on_with_the_recipe_plugin(made, monkeypatch):
    pytest.importorskip("vllm_exl3_moe")
    import vllm_exl3_moe
    vllm_exl3_moe.register()
    from vllm_exl3_moe.config import DeepseekV41Exl3Config
    from vllm_exl3_moe.head import Exl3HeadMethod

    monkeypatch.setenv("VLLM_DSPARK_OWN_HEAD", "1")
    q = dict(OFFICIAL_QCFG, quant_method="deepseek_v41_exl3")
    q["exl3_moe"] = {"plan_file": "exl3_moe_plan.json", "head": "exl3", "head_k": 6, "drafter_head_k": 4}
    m = drafter(DeepseekV41Exl3Config.from_config(q))
    hm = m._own_lm_head("").quant_method
    assert type(hm) is Exl3HeadMethod and hm.k == 4 and hm.prefetch is False and m.keeps_own_lm_head is True
    del q["exl3_moe"]["drafter_head_k"]                                 # a checkpoint without a drafter head
    m = drafter(DeepseekV41Exl3Config.from_config(q))
    assert m._own_lm_head("") is None and m.keeps_own_lm_head is False


# --- (3): load_weights -----------------------------------------------------------------------------------------------

def full_head(k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"trellis": torch.randint(-32768, 32768, (H // 16, V // 16, 16 * k), dtype=torch.int16, generator=g),
            "suh": torch.randn(H, generator=g).half(), "svh": torch.randn(V, generator=g).half(),
            "mul1": torch.tensor(-2083270355, dtype=torch.int32)}


def loading_drafter(monkeypatch, rank=1):
    """bare(own=True) with an EXL3 K4 head module (the plugin's method, rank `rank` of TP=2) and the attributes
    load_weights reads; TP size and rank patched in dspark's namespace."""
    E = pytest.importorskip("vllm_exl3_moe.head")
    m = bare(own=True)
    m.config = types.SimpleNamespace(num_attention_heads=64, n_routed_experts=8, expert_dtype="fp4")
    m.linear_scale_name, m.pad_shared_expert = "weight_scale", False
    layer = types.SimpleNamespace(ffn=types.SimpleNamespace(use_native_mega_moe=True,
                                                            finalize_mega_moe_weights=lambda: None),
                                  attn=types.SimpleNamespace())
    object.__setattr__(m, "model", types.SimpleNamespace(layers=[layer], confidence_head=None))
    head = torch.nn.Module()
    head.tp_rank = rank
    head.shard_indices = types.SimpleNamespace(org_vocab_start_index=rank * V // TP,
                                               org_vocab_end_index=(rank + 1) * V // TP)
    E.Exl3HeadMethod(4, prefetch=False).create_weights(head, H, [V // TP], H, V, torch.bfloat16, weight_loader=None)
    m.lm_head = head
    monkeypatch.setattr(D, "get_tensor_model_parallel_world_size", lambda: TP)
    monkeypatch.setattr(D, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.delenv("VLLM_DSPARK_MARKOV_CACHE", raising=False)
    return m, head


def test_load_weights_fills_the_own_head(monkeypatch):
    m, head = loading_drafter(monkeypatch, rank=1)
    t = full_head(4)
    weights = [(f"mtp.head.{p}", v) for p, v in t.items()]
    weights += [("head.trellis", full_head(6, 1)["trellis"]), ("embed.weight", torch.zeros(4, 4))]   # the target's
    assert m.load_weights(iter(weights)) == {f"lm_head.{p}" for p in t}
    lo, hi = V // TP, V
    assert torch.equal(head.trellis, t["trellis"][:, lo // 16:hi // 16]) and torch.equal(head.svh, t["svh"][lo:hi])
    assert torch.equal(head.suh, t["suh"]) and int(head.mul1) == int(t["mul1"])


def test_load_weights_skips_mtp_head_for_a_shared_head(monkeypatch):
    m, head = loading_drafter(monkeypatch)
    m.keeps_own_lm_head = False
    before = head.trellis.clone()
    assert m.load_weights(iter([(f"mtp.head.{p}", v) for p, v in full_head(4).items()])) == set()
    assert torch.equal(head.trellis, before)


def test_load_weights_refuses_a_missing_piece(monkeypatch):
    m, _ = loading_drafter(monkeypatch)
    with pytest.raises(ValueError, match=r"mtp\.head.*lm_head\.svh"):
        m.load_weights(iter([(f"mtp.head.{p}", v) for p, v in full_head(4).items() if p != "svh"]))


# --- (4): load_dspark_model ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("own", [False, True])
def test_load_dspark_model_keeps_an_own_head(monkeypatch, own):
    import vllm.model_executor.model_loader as model_loader
    import vllm.model_executor.models.qwen3_dflash as qwen3_dflash
    import vllm.model_executor.models.utils as model_utils
    import vllm.v1.worker.gpu.spec_decode.dspark.utils as U
    import vllm.v1.worker.gpu.spec_decode.eagle.utils as eagle_utils

    draft = bare(own)
    draft.lm_head = own_head = torch.nn.Module()
    target = torch.nn.Module()
    target.model = types.SimpleNamespace()
    target.lm_head = target_head = torch.nn.Module()
    dmc = types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="deepseek_v41"), get_vocab_size=lambda: V)
    vc = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(draft_model_config=dmc, attention_backend=None, kv_cache_dtype=None,
                                                 draft_parallel_config=types.SimpleNamespace(tensor_parallel_size=TP)),
        attention_config=types.SimpleNamespace(backend=None), parallel_config=types.SimpleNamespace(),
        cache_config=types.SimpleNamespace(), model_config=types.SimpleNamespace(get_vocab_size=lambda: V))
    monkeypatch.setattr(U, "replace", lambda obj, **kw: types.SimpleNamespace(**{**vars(obj), **kw}))
    monkeypatch.setattr(U, "_get_dspark_parallel_config", lambda pc, tp: pc)
    monkeypatch.setattr(U, "get_draft_load_config", lambda c: None)
    monkeypatch.setattr(U, "get_pp_safe_draft_load_config", lambda c: c)
    monkeypatch.setattr(model_loader, "get_model", lambda vllm_config, model_config: draft)
    monkeypatch.setattr(qwen3_dflash, "dflash_has_any_non_causal", lambda cfg: False)
    monkeypatch.setattr(model_utils, "get_draft_quant_config", lambda c: None)
    monkeypatch.setattr(eagle_utils, "maybe_share_target_embed", lambda *a: None)
    got = U.load_dspark_model(target, vc)
    assert got is draft and got.lm_head is (own_head if own else target_head)
