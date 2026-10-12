# SPDX-License-Identifier: Apache-2.0
"""Patch 0007: an EXL3 head's pieces (head.trellis / suh / svh / mul1) load into lm_head exactly where head.weight does,
for the text model and the VL wrapper, and no other checkpoint name moves. CPU only (name mapping, no weights).

Run inside the dsv41 image:  python3 -m pytest -q test_head_pieces_mapper.py
"""
import pytest

pytest.importorskip("vllm")   # runs in the serve image

from vllm.models.deepseek_v41.nvidia.model import _make_deepseek_v4_weights_mapper
from vllm.models.deepseek_v41.nvidia.vl_model import _make_deepseek_v4_vl_weights_mapper

PIECES = ("trellis", "suh", "svh", "mul1")


def mapper(vl):
    make = _make_deepseek_v4_vl_weights_mapper if vl else _make_deepseek_v4_weights_mapper
    return make("fp4", "weight_scale")


@pytest.mark.parametrize("vl", [False, True])
def test_head_pieces_follow_head_weight(vl):
    m = mapper(vl)
    w, shard = m._map_name_with_shard("head.weight")
    assert shard is None and w == ("language_model.lm_head.weight" if vl else "lm_head.weight")
    for p in PIECES:
        assert m._map_name_with_shard(f"head.{p}") == (w[: -len("weight")] + p, None)


@pytest.mark.parametrize("vl", [False, True])
def test_other_names_unchanged(vl):
    m = mapper(vl)
    pre = "language_model.model." if vl else "model."
    assert m._map_name_with_shard("layers.3.attn.wq_b.trellis") == (pre + "layers.3.attn.wq_b.trellis", None)
    assert m._map_name_with_shard("layers.3.attn.wq_b.scale") == (pre + "layers.3.attn.wq_b.weight_scale", None)
    assert m._map_name_with_shard("layers.3.ffn.shared_experts.w2.trellis") == (
        pre + "layers.3.ffn.shared_experts.down_proj.trellis", None)
    assert m._map_name_with_shard("mtp.0.attn.wq_b.weight") is None      # the drafter's weights load elsewhere
