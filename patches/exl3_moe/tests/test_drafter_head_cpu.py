# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter's own EXL3 head in the plugin (exl3_moe.drafter_head_k; patch 0026 asks for it under
VLLM_DSPARK_OWN_HEAD=1): the config key, the routing of the drafter head's prefix (mtp_head) to Exl3HeadMethod at
drafter_head_k outside l2pf's head prefetch while the target's lm_head keeps head / head_k, and the head's TP cut of
the mtp.head.* tensors, which is the target head's cut (each rank holds the same vocab columns).

CPU only. Runs against the real vLLM where it imports (the serve image) and otherwise against minimal stand-ins for the
few vLLM names config.py / head.py / dense.py import, installed for each test and removed after it, so it also runs on
a box without vLLM. exllamav3's LinearEXL3 is always a stand-in (the kernel path is test_head_exl3_gpu.py's).

  python3 -m pytest -q tests/test_drafter_head_cpu.py
"""
import importlib
import importlib.util
import logging
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OFFICIAL_QCFG = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
                 "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
D, V, TP = 256, 1024, 2                       # hidden, vocab: each rank's range is whole 128-wide blocks
PLUGIN_MODULES = ("config", "head", "dense", "woa_qdq")    # the plugin modules that import vLLM at import time


def _stand_ins() -> dict[str, dict]:
    """module name -> attributes: just what config.py, head.py and dense.py import from vLLM."""
    class Module(torch.nn.Module):
        pass

    class DeepseekV4FP8Config:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_config(cls, config):
            return cls()

        @classmethod
        def get_min_capability(cls):
            return 80

        def get_quant_method(self, layer, prefix):
            return None

    def set_weight_attrs(weight, attrs):
        for k, v in (attrs or {}).items():
            setattr(weight, k, v)

    linear = type("LinearBase", (Module,), {})
    return {
        "vllm": {}, "vllm.model_executor": {}, "vllm.model_executor.layers": {}, "vllm.models": {},
        "vllm.models.deepseek_v41": {},
        "vllm.logger": {"init_logger": logging.getLogger},
        "vllm.model_executor.utils": {"set_weight_attrs": set_weight_attrs},
        "vllm.model_executor.layers.fused_moe": {"RoutedExperts": type("RoutedExperts", (Module,), {})},
        "vllm.model_executor.layers.quantization": {"register_quantization_config": lambda name: (lambda c: c)},
        "vllm.model_executor.layers.linear": {
            "LinearBase": linear, "LinearMethodBase": type("LinearMethodBase", (), {}),
            "ReplicatedLinear": type("ReplicatedLinear", (linear,), {}),
            "RowParallelLinear": type("RowParallelLinear", (linear,), {})},
        "vllm.model_executor.layers.vocab_parallel_embedding": {
            "ParallelLMHead": type("ParallelLMHead", (Module,), {}),
            "UnquantizedEmbeddingMethod": type("UnquantizedEmbeddingMethod", (), {})},
        "vllm.models.deepseek_v41.quant_config": {"DeepseekV4FP8Config": DeepseekV4FP8Config},
    }


def _fresh(monkeypatch, name: str) -> None:
    """sys.modules without `name` for this test (imported anew), restored or removed again at teardown."""
    monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    del sys.modules[name]


@pytest.fixture
def plugin(monkeypatch):
    """(config module, head module, ParallelLMHead class) of the plugin, on the real vLLM or the stand-ins."""
    if importlib.util.find_spec("vllm") is None:
        for name, attrs in _stand_ins().items():
            mod = types.ModuleType(name)
            mod.__path__ = []
            for k, v in attrs.items():
                setattr(mod, k, v)
            monkeypatch.setitem(sys.modules, name, mod)
        import vllm_exl3_moe
        for sub in PLUGIN_MODULES:              # built against the stand-ins, gone again after the test
            _fresh(monkeypatch, f"vllm_exl3_moe.{sub}")
            monkeypatch.setattr(vllm_exl3_moe, sub, None, raising=False)
    C = importlib.import_module("vllm_exl3_moe.config")
    H = importlib.import_module("vllm_exl3_moe.head")
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    return C, H, ParallelLMHead


def qcfg(**exl3):
    q = dict(OFFICIAL_QCFG, quant_method="deepseek_v41_exl3")
    q["exl3_moe"] = {"plan_file": "exl3_moe_plan.json", **exl3}
    return q


def full_head(k, seed):
    g = torch.Generator().manual_seed(seed)
    return {"trellis": torch.randint(-32768, 32768, (D // 16, V // 16, 16 * k), dtype=torch.int16, generator=g),
            "suh": torch.randn(D, generator=g).half(), "svh": torch.randn(V, generator=g).half(),
            "mul1": torch.tensor(-2083270355, dtype=torch.int32)}


def rank_layer(method, rank):
    """A head module of TP rank `rank` with the method's parameters (what ParallelLMHead.__init__ hands it)."""
    layer = torch.nn.Module()
    layer.tp_rank = rank
    layer.shard_indices = types.SimpleNamespace(org_vocab_start_index=rank * V // TP,
                                                org_vocab_end_index=(rank + 1) * V // TP)
    method.create_weights(layer, D, [V // TP], D, V, torch.bfloat16, weight_loader=None)
    return layer


def test_drafter_head_k_setting(plugin):
    C, H, _ = plugin
    assert C.Exl3Settings.from_dict({}).drafter_head_k is None
    assert C.Exl3Settings.from_dict({"drafter_head_k": 4}).drafter_head_k == 4
    assert C.Exl3Settings.from_dict({"drafter_head_k": 3.0}).drafter_head_k == 3
    for bad in (0, 9, 4.5, "4", True, -1, float("nan")):
        with pytest.raises(ValueError, match="drafter_head_k"):
            C.Exl3Settings.from_dict({"drafter_head_k": bad})


@pytest.mark.parametrize("target", ["exl3", "bf16"])
def test_drafter_head_routing(plugin, target):
    """mtp_head -> Exl3HeadMethod(drafter_head_k, prefetch=False) whatever the target head is; the target's lm_head keeps
    its own setting and the prefetch; the Markov head (markov_w2) gets no head method."""
    C, H, ParallelLMHead = plugin
    lm = ParallelLMHead.__new__(ParallelLMHead)
    extra = {"head": "exl3", "head_k": 6} if target == "exl3" else {}
    cfg = C.DeepseekV41Exl3Config.from_config(qcfg(drafter_head_k=4, **extra))
    m = cfg.get_quant_method(lm, C.DRAFTER_HEAD_PREFIX)
    assert type(m) is H.Exl3HeadMethod and m.k == 4 and m.prefetch is False
    t = cfg.get_quant_method(lm, "lm_head")
    if target == "exl3":
        assert type(t) is H.Exl3HeadMethod and t.k == 6 and t.prefetch is True
    else:
        assert not isinstance(t, H.Exl3HeadMethod)
    assert not isinstance(cfg.get_quant_method(lm, "model.markov_head.markov_w2"), H.Exl3HeadMethod)
    assert not isinstance(cfg.get_quant_method(lm, "model.mtp_head_x"), H.Exl3HeadMethod)


def test_no_drafter_head_without_the_key(plugin):
    """A checkpoint without drafter_head_k gives the drafter's head no method: patch 0026 then shares the target's."""
    C, H, ParallelLMHead = plugin
    lm = ParallelLMHead.__new__(ParallelLMHead)
    cfg = C.DeepseekV41Exl3Config.from_config(qcfg(head="exl3", head_k=6))
    assert cfg.get_quant_method(lm, "mtp_head") is None
    assert type(cfg.get_quant_method(lm, "lm_head")) is H.Exl3HeadMethod


def test_prefix_matches_patch_0026(plugin):
    C, _, _ = plugin
    assert C.DRAFTER_HEAD_PREFIX == "mtp_head"
    try:
        from vllm.models.deepseek_v41.nvidia.dspark import OWN_HEAD_PREFIX
    except ImportError:
        pytest.skip("needs the serve image with patch 0026")
    assert OWN_HEAD_PREFIX == C.DRAFTER_HEAD_PREFIX


def test_drafter_cut_is_the_target_cut(plugin):
    """Each rank's drafter head holds the same vocab columns as its target head: the trellis tiles, svh and the output
    width of [lo, hi), from full mtp.head.* tensors through the parameter loaders."""
    _, H, _ = plugin
    dt, tt = full_head(4, 0), full_head(6, 1)
    for r in range(TP):
        lo, hi = r * V // TP, (r + 1) * V // TP
        dl, tl = rank_layer(H.Exl3HeadMethod(4, prefetch=False), r), rank_layer(H.Exl3HeadMethod(6), r)
        for layer, t in ((dl, dt), (tl, tt)):
            for name in ("trellis", "suh", "svh", "mul1"):
                p = getattr(layer, name)
                assert p.weight_loader(p, t[name])
        assert torch.equal(dl.trellis, dt["trellis"][:, lo // 16:hi // 16])
        assert torch.equal(dl.svh, dt["svh"][lo:hi]) and torch.equal(dl.suh, dt["suh"])
        assert int(dl.mul1) == int(dt["mul1"])
        assert dl.exl3_shape == tl.exl3_shape == (D, hi - lo)
        assert dl.trellis.shape[:2] == tl.trellis.shape[:2] and dl.trellis.shape[2] == 64 and tl.trellis.shape[2] == 96


def test_drafter_head_rejects_another_k(plugin):
    _, H, _ = plugin
    layer = rank_layer(H.Exl3HeadMethod(4, prefetch=False), 0)
    with pytest.raises(ValueError, match="head_k=4"):
        layer.trellis.weight_loader(layer.trellis, full_head(6, 0)["trellis"])


def test_only_the_target_head_feeds_the_l2_prefetch(plugin, monkeypatch):
    _, H, _ = plugin
    calls = []
    monkeypatch.setattr(H.l2pf, "register_head", lambda tensors: calls.append(tensors))
    monkeypatch.setenv("VLLM_EXL3_HEAD_MROW", "0")
    stand_in = types.ModuleType("exllamav3.modules.quant.exl3")
    stand_in.LinearEXL3 = lambda *args, **kwargs: types.SimpleNamespace(args=args, kwargs=kwargs)
    monkeypatch.setitem(sys.modules, "exllamav3.modules.quant.exl3", stand_in)
    target, drafter = H.Exl3HeadMethod(6), H.Exl3HeadMethod(4, prefetch=False)
    tl, dl = rank_layer(target, 1), rank_layer(drafter, 1)
    target.process_weights_after_loading(tl)
    drafter.process_weights_after_loading(dl)
    assert len(calls) == 1 and calls[0][0].data_ptr() == tl.trellis.data_ptr()
    assert dl._exl3.args == (None, D, V // TP) and dl._exl3.kwargs["trellis"] is dl.trellis and dl._mrow is None
