# SPDX-License-Identifier: Apache-2.0
"""exl3_mrow: our decode-size EXL3 linear kernel (csrc/exl3_mrow.cu), 1..16 rows, K = 2..6, mul1 codebook.

A group = linears that run back to back on one input row block (wq_a + wkv, gate + up, the wo_a head groups), each with
its own column offset in the input and the output; one launch serves the whole group, bf16 in and out (no casts around
it), deterministic (k-slices summed in order by the last block of each output block). Built on first use with the
image's nvcc for sm_121 and cached under TORCH_EXTENSIONS_DIR (the serve container mounts ~/.cache, so it persists).
"""

from __future__ import annotations

import os

import torch

_ext = None


def ext():
    global _ext
    if _ext is None:
        from .jit import load
        _ext = load("exl3_mrow", "EXL3_MROW_FLAGS")             # e.g. EXL3_MROW_FLAGS="-DMROW_PIPE=4" (tuning)
    return _ext


class MrowGroup:
    """One registered group: linears given as (trellis, suh, svh, x_col, out_col), all with the same K."""

    def __init__(self, linears, K: int, want_items: int = 0):
        if not want_items:
            want_items = int(os.environ.get("EXL3_MROW_ITEMS", "0"))             # tuning override
        if not want_items:   # 1.5 items per SM: fewer, longer items beat finer splits (tune_mrow.py sweep, Oct 9)
            want_items = 3 * torch.cuda.get_device_properties(linears[0][0].device).multi_processor_count // 2
        tr, su, sv, xc, oc = zip(*linears)
        self.trellis = list(tr)          # the bytes the kernel streams (kept alive here; l2pf.py prefetches them)
        self.K = K
        self.handle = ext().register(list(tr), list(su), list(sv), list(xc), list(oc), K, want_items)
        info = ext().info(self.handle)
        self.items, self.ks = info[0], info[1:]

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        ext().run(self.handle, x, out)
        return out
