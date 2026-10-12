# SPDX-License-Identifier: Apache-2.0
"""Input-Hessian capture for linears that are still FP8 on the serve (the DSpark drafter's), for their EXL3 encode.

VLLM_EXL3_HCAP=<dir> (an eager serve: CUDA-graph replays run no Python) wraps the quant method of every matching linear
so each apply adds X^T X of its input rows (fp32, on the GPU) to that module's sum; the sums are written to
<dir>/<module>.rank<r>.pt (H, rows) when <dir>/SAVE appears (checked every 64 calls; each rank then writes
<dir>/saved.rank<r>) and at exit. main_proj's H alone is ~0.9 GB, so saves happen on request, not on a timer. A row-parallel module sees this rank's
input channels only, so its file holds the diagonal block of the full H for this rank's slice, which is what an encode
of that slice needs (EXL3 rotates inputs in 128-wide blocks, so slices cut at 128 channels encode independently).

VLLM_EXL3_HCAP_MATCH: regex over the vLLM module prefix (default: the drafter layers' dense linears and main_proj).
"""

from __future__ import annotations

import atexit
import os
import re

import torch
from vllm.model_executor.layers.linear import LinearMethodBase

DIR = os.environ.get("VLLM_EXL3_HCAP") or ""
MATCH = re.compile(os.environ.get("VLLM_EXL3_HCAP_MATCH") or
                   r"(layers\.(4[0-9])\.(attn\.(fused_wqa_wkv|wq_b|wo_b)|ffn\.shared_experts\.(gate_up_proj|down_proj))"
                   r"|main_proj)$")
_saved_flag = [0.0]                  # mtime of the SAVE file this rank last answered

_sums: dict[str, list] = {}          # name -> [H fp32 [in, in], rows]


def enabled(prefix: str) -> bool:
    return bool(DIR) and MATCH.search(prefix) is not None


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001
        return 0


def save() -> None:
    if not DIR or not _sums:
        return
    os.makedirs(DIR, exist_ok=True)
    r = _rank()
    for name, (h, n) in _sums.items():
        path = os.path.join(DIR, f"{name}.rank{r}.pt")
        torch.save({"H": h.cpu(), "rows": n, "module": name}, path + ".tmp")
        os.replace(path + ".tmp", path)
    with open(os.path.join(DIR, f"saved.rank{r}"), "w") as f:
        f.write(" ".join(f"{k}:{v[1]}" for k, v in _sums.items()) + "\n")


def _maybe_save() -> None:
    flag = os.path.join(DIR, "SAVE")
    try:
        t = os.path.getmtime(flag)
    except OSError:
        return
    if t != _saved_flag[0]:
        _saved_flag[0] = t
        save()


atexit.register(save)


class HCaptureMethod(LinearMethodBase):
    """The FP8 method of one linear, unchanged, plus X^T X of what it is applied to (eager calls only)."""

    def __init__(self, inner: LinearMethodBase, name: str):
        self.inner, self.name, self.calls = inner, name, 0

    def create_weights(self, *args, **kwargs):
        return self.inner.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer):
        return self.inner.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        if not torch.cuda.is_current_stream_capturing():
            x2 = x.reshape(-1, x.shape[-1]).float()
            s = _sums.get(self.name)
            if s is None:
                s = _sums[self.name] = [torch.zeros((x2.shape[1], x2.shape[1]), dtype=torch.float32,
                                                    device=x2.device), 0]
            s[0].addmm_(x2.t(), x2)
            s[1] += x2.shape[0]
            self.calls += 1
            if self.calls % 64 == 0:
                _maybe_save()
        return self.inner.apply(layer, x, bias)

    def __getattr__(self, item):          # anything else the layer asks of its method
        return getattr(self.inner, item)
