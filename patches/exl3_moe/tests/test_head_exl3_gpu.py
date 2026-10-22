# SPDX-License-Identifier: Apache-2.0
"""EXL3 lm_head (vllm_exl3_moe/head.py Exl3HeadMethod) at TP=2: each rank cuts its vocab range from the full head
tensors through the parameter loaders; the ranks' logits side by side equal F.linear with the full reconstructed head.

  PYTHONPATH=/plugin python3 -m pytest -q -s test_head_exl3_gpu.py      (serve image, any CUDA GPU)
"""
import types

import pytest
import torch

pytest.importorskip("vllm")
from vllm_exl3_moe.head import Exl3HeadMethod

dev = torch.device("cuda")
MUL1 = 0x83DCD12D - 2 ** 32


def full_head(dim, vocab, k, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    return {"trellis": torch.randint(-32768, 32768, (dim // 16, vocab // 16, 16 * k), dtype=torch.int16, device=dev,
                                     generator=g),
            "suh": ((torch.randint(0, 2, (dim,), device=dev, generator=g) * 2 - 1) * 0.05).half(),
            "svh": ((torch.randint(0, 2, (vocab,), device=dev, generator=g) * 2 - 1) * 0.05).half(),
            "mul1": torch.tensor(MUL1, dtype=torch.int32, device=dev)}


@pytest.mark.parametrize("mrow", ["0", "1"])
@pytest.mark.parametrize("k", [3, 4, 5, 6])          # 3 / 4: the DSpark drafter's own head (exl3_moe.drafter_head_k)
def test_vocab_parallel_head(k, mrow, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_HEAD_MROW", mrow)            # 1: decode sizes through exl3_mrow
    dim, vocab, T = 1024, 2560, 6
    t = full_head(dim, vocab, k)
    from exllamav3.modules.quant.exl3 import LinearEXL3
    W = LinearEXL3(None, dim, vocab, suh=t["suh"], svh=t["svh"], trellis=t["trellis"],
                   mul1=t["mul1"]).get_weight_tensor().t().float()                      # [vocab, dim]
    x = torch.randn(T, dim, device=dev, dtype=torch.bfloat16)
    outs = []
    for r in range(2):
        layer = torch.nn.Module()
        layer.tp_rank = r
        layer.shard_indices = types.SimpleNamespace(org_vocab_start_index=r * vocab // 2,
                                                    org_vocab_end_index=(r + 1) * vocab // 2)
        m = Exl3HeadMethod(k)
        with torch.device(dev):
            m.create_weights(layer, dim, [vocab // 2], dim, vocab, torch.bfloat16, weight_loader=None)
        for name in ("trellis", "suh", "svh", "mul1"):
            p = getattr(layer, name)
            assert p.weight_loader(p, t[name])
        m.process_weights_after_loading(layer)
        assert (layer._mrow is not None) == (mrow == "1")
        outs.append(m.apply(layer, x))
    y = torch.cat(outs, -1).float()
    ref = torch.nn.functional.linear(x.float(), W)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"EXL3 head K{k} mrow={mrow}: rel {rel:.2e}")
    assert y.shape == (T, vocab) and rel < 1e-2


def test_uneven_vocab_rejected():
    layer = torch.nn.Module()
    layer.shard_indices = types.SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=1000)
    with pytest.raises(ValueError, match="128-wide"):
        Exl3HeadMethod(6).create_weights(layer, 1024, [1000], 1024, 2000, torch.bfloat16)
