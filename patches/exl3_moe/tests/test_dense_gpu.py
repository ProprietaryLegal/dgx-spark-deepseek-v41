# SPDX-License-Identifier: Apache-2.0
"""EXL3 dense linears (vllm_exl3_moe/dense.py) at TP=2: each rank loads its cut of the FULL checkpoint tensors through
the parameter loaders, and the ranks' outputs (concatenated for column-parallel, summed for row-parallel, per group for
wo_a) equal F.linear with the full reconstructed matrix. Random trellis words: the kernels decode any bit pattern.

  PYTHONPATH=/plugin python3 -m pytest -q -s test_dense_gpu.py      (serve image, any CUDA GPU)
"""
import pytest
import torch

pytest.importorskip("vllm")
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from vllm_exl3_moe.dense import Exl3DenseLinearMethod, dense_module

dev = torch.device("cuda")
MUL1 = 0x83DCD12D - 2 ** 32


def full_tensors(in_f, out_f, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return {"trellis": torch.randint(-32768, 32768, (in_f // 16, out_f // 16, 16 * k), dtype=torch.int16, device=dev,
                                     generator=g),
            "suh": ((torch.randint(0, 2, (in_f,), device=dev, generator=g) * 2 - 1) * 0.05).half(),
            "svh": ((torch.randint(0, 2, (out_f,), device=dev, generator=g) * 2 - 1) * 0.05).half(),
            "mul1": torch.tensor(MUL1, dtype=torch.int32, device=dev)}


def dense_weight(t, in_f, out_f):
    from exllamav3.modules.quant.exl3 import LinearEXL3
    return LinearEXL3(None, in_f, out_f, suh=t["suh"], svh=t["svh"], trellis=t["trellis"],
                      mul1=t["mul1"]).get_weight_tensor().t().float()          # [out, in]


def fake_layer(cls, tp_rank, tp_size=2, **attrs):
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.tp_rank, layer.tp_size = tp_rank, tp_size
    for k, v in attrs.items():
        setattr(layer, k, v)
    return layer


def build(method, layer, in_loc, outs, in_full, out_full, loads):
    with torch.device(dev):
        method.create_weights(layer, in_loc, outs, in_full, out_full, torch.bfloat16, weight_loader=None)
    for name, w, shard in loads:
        p = dict(layer.named_parameters())[name]
        assert p.weight_loader(p, w, shard)
    method.process_weights_after_loading(layer)
    return layer


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_dense_module_names():
    assert dense_module("model.layers.12.attn.wq_b") == (12, "attn.wq_b")
    assert dense_module("language_model.model.layers.3.ffn.shared_experts.gate_up_proj") == (3, "ffn.shared_experts.gate_up_proj")
    assert dense_module("model.layers.3.attn.fused_wqa_wkv") == (3, "attn.fused_wqa_wkv")
    assert dense_module("model.layers.3.attn.compressor.fused_wkv_wgate") is None
    assert dense_module("model.layers.3.ffn.experts") is None


@pytest.mark.parametrize("k", [4, 5])
def test_column_parallel(k):
    in_f, out_f, T = 1280, 4096, 6
    t = full_tensors(in_f, out_f, k, 1)
    W = dense_weight(t, in_f, out_f)
    x = torch.randn(T, in_f, device=dev, dtype=torch.bfloat16)
    ys = []
    for r in range(2):
        m = Exl3DenseLinearMethod("attn.wq_b", k)
        layer = build(m, fake_layer(ColumnParallelLinear, r), in_f, [out_f // 2], in_f, out_f,
                      [(n, t[n], None) for n in ("trellis", "suh", "svh", "mul1")])
        ys.append(m.apply(layer, x))
    y = torch.cat(ys, -1)
    ref = torch.nn.functional.linear(x.float(), W)
    print(f"column K{k}: rel {rel(y, ref):.2e}")
    assert y.shape == (T, out_f) and rel(y, ref) < 1e-2


def test_engram_wkv_sinks_fp8_tensors():
    """engram.wkv on EXL3 (column-parallel, gathered): a variant keeps the FP8 wkv tensors in V4.1's 101 GB shards, so
    they still stream in; the method registers empty sinks for weight / weight_scale / weight_scale_inv that take any
    tensor and keep nothing, and the EXL3 output is unchanged by them."""
    in_f, out_f, T, k = 6144, 25600 // 4, 6, 4           # a narrower output than the real 25,600: same code path
    t = full_tensors(in_f, out_f, k, 7)
    W = dense_weight(t, in_f, out_f)
    x = torch.randn(T, in_f, device=dev, dtype=torch.bfloat16)
    ys = []
    for r in range(2):
        m = Exl3DenseLinearMethod("engram.wkv", k, layer_idx=1, prefetch=False)
        layer = fake_layer(ColumnParallelLinear, r)
        with torch.device(dev):
            m.create_weights(layer, in_f, [out_f // 2], in_f, out_f, torch.bfloat16, weight_loader=None)
        params = dict(layer.named_parameters())
        for n in ("weight", "weight_scale", "weight_scale_inv"):
            assert params[n].numel() == 0
            assert params[n].weight_loader(params[n], torch.randn(out_f, in_f).to(torch.float8_e4m3fn))
            assert params[n].numel() == 0                     # nothing kept
        for n in ("trellis", "suh", "svh", "mul1"):
            assert params[n].weight_loader(params[n], t[n])
        m.process_weights_after_loading(layer)
        ys.append(m.apply(layer, x))
    y = torch.cat(ys, -1)
    ref = torch.nn.functional.linear(x.float(), W)
    print(f"engram.wkv K{k} with FP8 sinks: rel {rel(y, ref):.2e}")
    assert y.shape == (T, out_f) and rel(y, ref) < 1e-2


def test_merged_gate_up():
    in_f, inter, T, k = 1024, 512, 5, 5
    t1, t3 = full_tensors(in_f, inter, k, 2), full_tensors(in_f, inter, k, 3)
    W1, W3 = dense_weight(t1, in_f, inter), dense_weight(t3, in_f, inter)
    x = torch.randn(T, in_f, device=dev, dtype=torch.bfloat16)
    for r in range(2):
        m = Exl3DenseLinearMethod("ffn.shared_experts.gate_up_proj", k)
        loads = [(n, t1[n], 0) for n in ("trellis", "suh", "svh", "mul1")] + \
                [(n, t3[n], 1) for n in ("trellis", "suh", "svh", "mul1")]
        layer = build(m, fake_layer(MergedColumnParallelLinear, r), in_f, [inter // 2] * 2, in_f, 2 * inter, loads)
        y = m.apply(layer, x)
        c = slice(r * inter // 2, (r + 1) * inter // 2)
        ref = torch.cat([torch.nn.functional.linear(x.float(), W1)[:, c], torch.nn.functional.linear(x.float(), W3)[:, c]], -1)
        print(f"gate_up rank {r}: rel {rel(y, ref):.2e}")
        assert y.shape == (T, inter) and rel(y, ref) < 1e-2


@pytest.mark.parametrize("k", [4, 5])
def test_row_parallel(k):
    in_f, out_f, T = 1024, 512, 7
    t = full_tensors(in_f, out_f, k, 4)
    W = dense_weight(t, in_f, out_f)
    x = torch.randn(T, in_f, device=dev, dtype=torch.bfloat16)
    y = 0
    for r in range(2):
        m = Exl3DenseLinearMethod("attn.wo_b", k)
        layer = build(m, fake_layer(RowParallelLinear, r), in_f // 2, [out_f], in_f, out_f,
                      [(n, t[n], None) for n in ("trellis", "suh", "svh", "mul1")])
        y = y + m.apply(layer, x[:, r * in_f // 2:(r + 1) * in_f // 2].contiguous()).float()
    ref = torch.nn.functional.linear(x.float(), W)
    print(f"row K{k}: rel {rel(y, ref):.2e}")
    assert rel(y, ref) < 1e-2


def test_wo_a_grouped():
    in_g, out_g, G, T, k = 512, 256, 8, 6, 5
    ts = [full_tensors(in_g, out_g, k, 10 + g) for g in range(G)]
    Ws = [dense_weight(t, in_g, out_g) for t in ts]
    for r in range(2):
        m = Exl3DenseLinearMethod("attn.wo_a", k)
        layer = fake_layer(ColumnParallelLinear, r)
        with torch.device(dev):
            m.create_weights(layer, in_g, [G * out_g // 2], in_g, G * out_g, torch.bfloat16, weight_loader=None)
        params = dict(layer.named_parameters())
        for g in range(G):
            for n in ("trellis", "suh", "svh", "mul1"):
                p = params[f"slice.{g}.{n}"]
                assert p.weight_loader(p, ts[g][n])
        m.process_weights_after_loading(layer)
        assert layer.weight.dtype == torch.bfloat16 and layer.exl3_groups == list(range(4 * r, 4 * r + 4))
        o = torch.randn(T, G // 2, in_g, device=dev, dtype=torch.bfloat16)
        z = layer.grouped_bmm(o)
        ref = torch.stack([torch.nn.functional.linear(o[:, i].float(), Ws[g]) for i, g in enumerate(layer.exl3_groups)], 1)
        print(f"wo_a rank {r}: rel {rel(z, ref):.2e}")
        assert z.shape == (T, G // 2, out_g) and rel(z, ref) < 1e-2


@pytest.mark.parametrize("T", [1, 6, 16, 24, 40])
def test_wo_a_grouped_qdq(monkeypatch, T):
    """VLLM_DSV41_WOA_QDQ=1 (woa_qdq.py): grouped_bmm(o) == the flag-off grouped_bmm of DeepSeek's FP8 round trip of o,
    at decode sizes (exl3_mrow's wo_a build folds it into its input load) and prompt sizes (woa_qdq.qdq writes the fp16
    the exllamav3 path reads), on o_rot as patch 0006 hands it over: the [T, G, D] view of a [G, T, D] buffer."""
    from vllm_exl3_moe import woa_qdq
    in_g, out_g, G, k = 512, 256, 8, 5
    ts = [full_tensors(in_g, out_g, k, 40 + g) for g in range(G)]

    def wo_a(flag):
        if flag:
            monkeypatch.setenv("VLLM_DSV41_WOA_QDQ", "1")
        else:
            monkeypatch.delenv("VLLM_DSV41_WOA_QDQ", raising=False)
        m = Exl3DenseLinearMethod("attn.wo_a", k)
        layer = fake_layer(ColumnParallelLinear, 0)
        with torch.device(dev):
            m.create_weights(layer, in_g, [G * out_g // 2], in_g, G * out_g, torch.bfloat16, weight_loader=None)
        params = dict(layer.named_parameters())
        for g in range(G):
            for n in ("trellis", "suh", "svh", "mul1"):
                p = params[f"slice.{g}.{n}"]
                assert p.weight_loader(p, ts[g][n])
        m.process_weights_after_loading(layer)
        return layer

    on, off = wo_a(True), wo_a(False)
    o = (torch.randn(G // 2, T, in_g, device=dev) * 3).bfloat16().transpose(0, 1)
    z = on.grouped_bmm(o)
    assert torch.equal(z.view(torch.int16), off.grouped_bmm(woa_qdq.qdq(o)).view(torch.int16)), T
    assert not torch.equal(z.view(torch.int16), off.grouped_bmm(o).view(torch.int16))


@pytest.mark.parametrize("k", [5, 6])
def test_fused_wqa_wkv_replicated(k):
    """fused_wqa_wkv: replicated (disable_tp), two shards of unequal width (wq_a + wkv; 1280 + 512 in the model)."""
    in_f, outs, T = 1024, [384, 128], 6
    tq, tkv = full_tensors(in_f, outs[0], k, 20), full_tensors(in_f, outs[1], k, 21)
    Wq, Wkv = dense_weight(tq, in_f, outs[0]), dense_weight(tkv, in_f, outs[1])
    x = torch.randn(T, in_f, device=dev, dtype=torch.bfloat16)
    m = Exl3DenseLinearMethod("attn.fused_wqa_wkv", k)
    loads = [(n, tq[n], 0) for n in ("trellis", "suh", "svh", "mul1")] + \
            [(n, tkv[n], 1) for n in ("trellis", "suh", "svh", "mul1")]
    layer = build(m, fake_layer(MergedColumnParallelLinear, 0, tp_size=1), in_f, outs, in_f, sum(outs), loads)
    y = m.apply(layer, x)
    ref = torch.cat([torch.nn.functional.linear(x.float(), Wq), torch.nn.functional.linear(x.float(), Wkv)], -1)
    print(f"fused_wqa_wkv K{k}: rel {rel(y, ref):.2e}")
    assert y.shape == (T, sum(outs)) and rel(y, ref) < 1e-2


def test_context_kv_group():
    """context_kv_group: the wkv part of three fused wq_a + wkv modules (DSpark's draft layers) in one launch on one
    input, each module's columns equal to that module's own wkv output (F.linear with its reconstructed matrix), run to
    run identical; None for a module without flat parts or a different K."""
    from vllm_exl3_moe.dense import context_kv_group
    in_f, outs, T, k = 1024, [384, 128], 6, 5
    layers, refs = [], []
    x = torch.randn(T, in_f, device=dev, dtype=torch.bfloat16)
    for j in range(3):
        tq, tkv = full_tensors(in_f, outs[0], k, 40 + 2 * j), full_tensors(in_f, outs[1], k, 41 + 2 * j)
        m = Exl3DenseLinearMethod("attn.fused_wqa_wkv", k, layer_idx=40 + j, prefetch=False)
        loads = [(n, tq[n], 0) for n in ("trellis", "suh", "svh", "mul1")] + \
                [(n, tkv[n], 1) for n in ("trellis", "suh", "svh", "mul1")]
        layers.append(build(m, fake_layer(MergedColumnParallelLinear, 0, tp_size=1), in_f, outs, in_f, sum(outs), loads))
        refs.append(torch.nn.functional.linear(x.float(), dense_weight(tkv, in_f, outs[1])))
    run, widths = context_kv_group(layers)
    assert widths == [outs[1]] * 3
    y = run(x)
    assert y.shape == (T, 3 * outs[1])
    for j in range(3):
        r = rel(y[:, j * outs[1]:(j + 1) * outs[1]], refs[j])
        print(f"context kv group, module {j}: rel {r:.2e}")
        assert r < 1e-2
    assert torch.equal(y, run(x)), "not deterministic"
    assert context_kv_group(layers)[0] is run                          # cached on the first module
    m6 = Exl3DenseLinearMethod("attn.fused_wqa_wkv", 6, layer_idx=43, prefetch=False)
    t6q, t6kv = full_tensors(in_f, outs[0], 6, 50), full_tensors(in_f, outs[1], 6, 51)
    l6 = build(m6, fake_layer(MergedColumnParallelLinear, 0, tp_size=1), in_f, outs, in_f, sum(outs),
               [(n, t6q[n], 0) for n in ("trellis", "suh", "svh", "mul1")] +
               [(n, t6kv[n], 1) for n in ("trellis", "suh", "svh", "mul1")])
    assert context_kv_group([layers[1], l6]) is None                   # K differ
    assert context_kv_group([fake_layer(MergedColumnParallelLinear, 0, tp_size=1)]) is None


def test_unequal_shards_need_replicated():
    m = Exl3DenseLinearMethod("attn.fused_wqa_wkv", 5)
    with torch.device(dev), pytest.raises(ValueError, match="replicated"):
        m.create_weights(fake_layer(MergedColumnParallelLinear, 0, tp_size=2), 1024, [192, 64], 1024, 512,
                         torch.bfloat16, weight_loader=None)


def test_shape_mismatch_raises():
    m = Exl3DenseLinearMethod("attn.wq_b", 5)
    layer = fake_layer(ColumnParallelLinear, 0)
    with torch.device(dev):
        m.create_weights(layer, 1280, [2048], 1280, 4096, torch.bfloat16, weight_loader=None)
    bad = full_tensors(1280, 4096, 4, 0)["trellis"]                    # K4 words into a K5 layer
    with pytest.raises(ValueError, match="K=5"):
        layer.trellis.weight_loader(layer.trellis, bad, None)


@pytest.mark.parametrize("T", [4, 20])
def test_main_proj_large_inputs(T):
    """main_proj (the drafter's context combiner, replicated) reads the target's raw residual stream: |x| up to ~1.6e5.
    At 4 rows it runs on mrow (fp32 scaling before its fp16 cast), at 20 on exllamav3, whose fp16 input cast overflowed
    to inf / NaN before dense.py scaled it by a power of two (found Oct 10)."""
    from vllm.model_executor.layers.linear import ReplicatedLinear
    in_f, out_f, k = 1024, 512, 5
    t = full_tensors(in_f, out_f, k, 30)
    W = dense_weight(t, in_f, out_f)
    g = torch.Generator(device=dev).manual_seed(31)
    x = torch.randn(T, in_f, device=dev, generator=g) * 4200
    x[:, 7] = 1.6e5
    x[:, 100] = -1.2e5
    x = x.bfloat16()
    m = Exl3DenseLinearMethod("main_proj", k, prefetch=False)
    layer = build(m, fake_layer(ReplicatedLinear, 0, tp_size=1), in_f, [out_f], in_f, out_f,
                  [(n, t[n], None) for n in ("trellis", "suh", "svh", "mul1")])
    y = m.apply(layer, x)
    ref = torch.nn.functional.linear(x.float(), W)
    print(f"main_proj T={T}: rel {rel(y, ref):.2e}, finite {bool(torch.isfinite(y).all())}")
    assert torch.isfinite(y).all() and rel(y, ref) < 1e-2
