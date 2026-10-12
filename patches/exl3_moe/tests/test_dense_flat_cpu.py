# SPDX-License-Identifier: Apache-2.0
"""dense.py's flat layout for a replicated merged linear with unequal shards (fused_wqa_wkv = wq_a 1280 + wkv 512): each
shard's pieces land at their offsets and the per-shard views read them back. CPU, no exllamav3 (no forward).

  PYTHONPATH=/plugin python3 -m pytest -q test_dense_flat_cpu.py      (serve image)
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.model_executor.layers.linear import MergedColumnParallelLinear  # noqa: E402
from vllm_exl3_moe.dense import Exl3DenseLinearMethod  # noqa: E402


def fake_layer(tp_size=1):
    layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.tp_rank, layer.tp_size = 0, tp_size
    return layer


def pieces(in_f, out_f, k, seed):
    g = torch.Generator().manual_seed(seed)
    return {"trellis": torch.randint(-32768, 32768, (in_f // 16, out_f // 16, 16 * k), dtype=torch.int16, generator=g),
            "suh": torch.randn(in_f, generator=g).half(), "svh": torch.randn(out_f, generator=g).half(),
            "mul1": torch.tensor(seed, dtype=torch.int32)}


@pytest.mark.parametrize("k", [5, 6])
def test_flat_offsets_round_trip(k):
    in_f, outs = 5120, [1280, 512]                       # the model's shapes
    m = Exl3DenseLinearMethod("attn.fused_wqa_wkv", k)
    layer = fake_layer()
    m.create_weights(layer, in_f, outs, in_f, sum(outs), torch.bfloat16, weight_loader=None)
    src = [pieces(in_f, o, k, 30 + i) for i, o in enumerate(outs)]
    params = dict(layer.named_parameters())
    for i in (1, 0):                                     # load order must not matter
        for n in ("trellis", "suh", "svh", "mul1"):
            assert params[n].weight_loader(params[n], src[i][n], i)
    assert layer.exl3_n == 2 and layer.exl3_shape == (in_f, sum(outs))
    for i, (o, t_off, s_off) in enumerate(layer.exl3_flat):
        tr = layer.trellis[t_off: t_off + (in_f // 16) * (o // 16) * 16 * k].view(in_f // 16, o // 16, 16 * k)
        assert torch.equal(tr, src[i]["trellis"])
        assert torch.equal(layer.suh[i], src[i]["suh"]) and torch.equal(layer.svh[s_off: s_off + o], src[i]["svh"])
        assert int(layer.mul1[i]) == 30 + i


def test_flat_loader_checks():
    m = Exl3DenseLinearMethod("attn.fused_wqa_wkv", 5)
    layer = fake_layer()
    m.create_weights(layer, 5120, [1280, 512], 5120, 1792, torch.bfloat16, weight_loader=None)
    p = dict(layer.named_parameters())["trellis"]
    with pytest.raises(ValueError, match="K=5"):
        p.weight_loader(p, pieces(5120, 512, 4, 1)["trellis"], 1)      # K4 words for a K5 config
    with pytest.raises(ValueError, match="shard id"):
        p.weight_loader(p, pieces(5120, 1280, 5, 1)["trellis"], None)
    with pytest.raises(ValueError, match="replicated"):
        m.create_weights(fake_layer(tp_size=2), 5120, [640, 256], 5120, 1792, torch.bfloat16, weight_loader=None)
