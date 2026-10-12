# SPDX-License-Identifier: Apache-2.0
"""Engram disk offload (patch 0002) must return exactly what the pinned-host and
GPU-resident paths return, for every TP shard, through lookup(), forward() and
the Engram prefetch path. Modeled on upstream's
tests/kernels/test_engram.py::test_engram_head_shards_reconstruct_checkpoint.

Run inside the dsv41-recipe image:  python3 -m pytest -q test_engram_disk_vllm.py
"""
import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("vllm")   # runs in the serve image

import vllm.models.deepseek_v41.common.engram as engram_ops
from vllm.models.deepseek_v41.common.engram import (
    Engram,
    ParallelEngramEmbedding,
    _safetensors_tensor_offset,
)


def _reference_lookup(weight, scale_inv, ids, start, end, block=32):
    mask = (ids < start) | (ids >= end)
    local = (ids - start).masked_fill(mask, 0).long()
    values = torch.nn.functional.embedding(local, weight)
    scales = torch.nn.functional.embedding(local, scale_inv)
    scales = (scales.to(torch.int32) << 23).view(torch.float32)
    values = values.float().unflatten(-1, (-1, block))
    values = (values * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    return values.masked_fill(mask.unsqueeze(-1), 0)


HEAD_SIZES = (17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73)


@pytest.fixture
def table(tmp_path):
    num_rows, dim = sum(HEAD_SIZES), 64
    torch.manual_seed(0)
    weight = torch.randn(num_rows + 7, dim).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 134, (num_rows + 7, dim // 32), dtype=torch.uint8)
    path = str(tmp_path / "model-00001-of-00001.safetensors")
    # Same names and dtypes as the checkpoint (scale stored as F8_E8M0).
    save_file(
        {
            "layers.1.engram.embed.weight": weight,
            "layers.1.engram.embed.scale": scales.view(torch.float8_e8m0fnu),
        },
        path,
    )
    source = (
        path,
        _safetensors_tensor_offset(path, "layers.1.engram.embed.weight"),
        path,
        _safetensors_tensor_offset(path, "layers.1.engram.embed.scale"),
    )
    return weight, scales, num_rows, dim, source


def _hashes(num_tokens=7):
    hashes = torch.empty(num_tokens, len(HEAD_SIZES), dtype=torch.int32, device="cuda")
    start = 0
    for head, size in enumerate(HEAD_SIZES):
        hashes[:, head].random_(start, start + size)
        hashes[0, head] = start
        hashes[-1, head] = start + size - 1
        start += size
    return hashes


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("cache_rows", [0, 32])
def test_engram_disk_matches_reference(table, tp_size, cache_rows, monkeypatch):
    weight, scales, num_rows, dim, source = table
    ids = _hashes()
    expected = _reference_lookup(weight.cuda(), scales.cuda(), ids, 0, num_rows)
    monkeypatch.setattr(
        engram_ops, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    shards = []
    for rank in range(tp_size):
        monkeypatch.setattr(
            engram_ops, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        with torch.device("cuda"):
            layer = ParallelEngramEmbedding(
                num_rows + 7,
                dim,
                HEAD_SIZES,
                cpu_offload=True,
                disk_source=source,
                disk_threads=4,
                disk_cache_rows=cache_rows,
            )
            layer.init_disk_staging(64 * layer.part_n_hash_cols)
        # The loader must not allocate or copy the table in disk mode.
        layer.weight.weight_loader(layer.weight, weight)
        assert layer.weight.shape[0] == 1
        out = torch.empty(
            len(ids), layer.part_n_hash_cols, dim, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(3):  # repeated lookups reuse the pinned staging and the cache
            layer.lookup(ids, out)
            torch.cuda.synchronize()
        shards.append(out.clone())
    gathered = torch.cat(shards, dim=1)
    torch.testing.assert_close(gathered[:, : len(HEAD_SIZES)], expected, rtol=0, atol=0)
    assert torch.count_nonzero(gathered[:, len(HEAD_SIZES) :]) == 0


def test_engram_disk_prefetch_path(table, monkeypatch):
    weight, scales, num_rows, dim, source = table
    ids = _hashes(num_tokens=33)
    expected = _reference_lookup(weight.cuda(), scales.cuda(), ids, 0, num_rows)
    monkeypatch.setattr(engram_ops, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(engram_ops, "get_tensor_model_parallel_rank", lambda: 0)
    with torch.device("cuda"):
        layer = ParallelEngramEmbedding(
            num_rows + 7, dim, HEAD_SIZES, cpu_offload=True, disk_source=source
        )
        layer.init_disk_staging(64 * layer.part_n_hash_cols)
    module = Engram.__new__(Engram)
    torch.nn.Module.__init__(module)
    module.embed_tokens = layer
    module.use_sequence_parallel = False
    module.staged_rows = torch.empty(
        64, layer.part_n_hash_cols, dim, dtype=torch.bfloat16, device="cuda"
    )
    module._prefetch_stream = torch.cuda.Stream()
    module._prefetch_done = torch.cuda.Event()
    for _ in range(3):
        module.prepare_embeddings(ids)
        got = module.embed(ids)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
