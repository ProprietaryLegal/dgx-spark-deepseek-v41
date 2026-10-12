# SPDX-License-Identifier: Apache-2.0
"""EXL3 tensors from a sharded checkpoint, applied to the layer-streamed reference.

Packs that quantize more than the routed experts (Mia-AiLab's EXL3 2.9 bpw: attention, shared expert, compressor,
indexer, Engram wkv and head too; our dense pass) store each linear as `<path>.{trellis,suh,svh,mul1|mcg}` under the
official module path, wo_a as eight `attn.wo_a.slice.G` group matrices. `overlay_block` swaps every such dense linear of
a reference Block for its reconstructed weight (the reconstruction the serving kernels compute), so p4_kld.py scores the
whole pack with the same implementation and panel as the native model. Routed experts stay with `exl3.quant_moe`.
"""
from __future__ import annotations

import json
import os
import re

import torch
from safetensors import safe_open

from .exl3 import reconstruct
from .refmodel import drop_page_cache

PIECES = ("trellis", "suh", "svh", "mul1", "mcg")


class Exl3Checkpoint:
    """Index over a sharded safetensors checkpoint; reads only EXL3 pieces (and nothing else) on demand."""

    def __init__(self, path: str):
        self.dir = path
        self.weight_map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]

    def has_experts(self, layer: int) -> bool:
        return f"layers.{layer}.ffn.experts.0.w1.trellis" in self.weight_map

    def tensors(self, prefix: str, device="cuda") -> dict[str, torch.Tensor]:
        keys = [k for k in self.weight_map if k.startswith(prefix) and k.rsplit(".", 1)[-1] in PIECES]
        by_file: dict[str, list[str]] = {}
        for k in keys:
            by_file.setdefault(self.weight_map[k], []).append(k)
        out = {}
        for f, ks in by_file.items():
            p = os.path.join(self.dir, f)
            with safe_open(p, framework="pt", device=str(device)) as sf:
                for k in ks:
                    out[k] = sf.get_tensor(k)
            drop_page_cache(p)
        return out


def _modules(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, dict[str, torch.Tensor]]:
    """{module path relative to prefix: {piece: tensor}}"""
    mods: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in tensors.items():
        if not k.startswith(prefix):
            continue
        path, piece = k[len(prefix):].rsplit(".", 1)
        if piece in PIECES:
            mods.setdefault(path, {})[piece] = v
    return mods


def _dims(t: dict[str, torch.Tensor]) -> tuple[int, int]:
    tr = t["trellis"]                                           # [in / 16, out / 16, 16 K]
    return tr.shape[0] * 16, tr.shape[1] * 16


def overlay_block(blk: torch.nn.Module, layer: int, tensors: dict[str, torch.Tensor], prefix: str | None = None) -> list[str]:
    """Replace the dense linears of reference block `layer` that have EXL3 tensors. Returns the module paths replaced."""
    prefix = f"layers.{layer}." if prefix is None else prefix
    mods = _modules(tensors, prefix)
    done = []
    slices = sorted((int(m.group(1)), p) for p in mods if (m := re.fullmatch(r"attn\.wo_a\.slice\.(\d+)", p)))
    if slices:
        ws = [reconstruct(mods[p], *_dims(mods[p])) for _, p in slices]        # [1024, 4096] each
        w = torch.cat(ws, 0)                                                    # [n_groups * o_lora_rank, 4096]
        mod = blk.attn.wo_a
        assert tuple(w.shape) == tuple(mod.weight.shape), (w.shape, mod.weight.shape)
        _set_weight(mod, w)
        done.append("attn.wo_a")
    for path, t in sorted(mods.items()):
        if ".experts." in f".{path}." and "shared_experts" not in path or path.startswith("attn.wo_a.slice"):
            continue
        try:
            mod = blk.get_submodule(path)
        except AttributeError:
            raise KeyError(f"layer {layer}: EXL3 tensors for {path!r}, but the reference block has no such module")
        k_in, n_out = _dims(t)
        assert (mod.in_features, mod.out_features) == (k_in, n_out), (path, (mod.in_features, mod.out_features), (k_in, n_out))
        _set_weight(mod, reconstruct(t, k_in, n_out))
        done.append(path)
    return done


def _set_weight(mod: torch.nn.Module, w16: torch.Tensor) -> None:
    """Dense weight in the module's float dtype (bf16 where the checkpoint had FP8: the reference's linear() then runs
    F.linear, as for its other unquantized linears)."""
    dt = mod.weight.dtype if mod.weight.dtype in (torch.float32, torch.bfloat16, torch.float16) else torch.bfloat16
    mod.weight = torch.nn.Parameter(w16.to(dt).contiguous(), requires_grad=False)
    if getattr(mod, "scale", None) is not None:
        mod.scale = None


def head_weight(ckpt: Exl3Checkpoint, device="cuda") -> torch.Tensor | None:
    """The head [vocab, dim] reconstructed from `head.*` EXL3 tensors, or None if the pack keeps the official head."""
    t = _modules(ckpt.tensors("head.", device), "")
    if "head" not in t:
        return None
    return reconstruct(t["head"], *_dims(t["head"]))
