# SPDX-License-Identifier: Apache-2.0
"""exl3_mrow: our decode-size EXL3 linear kernel (csrc/exl3_mrow.cu), 1..16 rows, K = 2..6, mul1 codebook.

A group = linears that run back to back on one input row block (wq_a + wkv, gate + up, the wo_a head groups), each with
its own column offset in the input and the output; one launch serves the whole group, bf16 in and out (no casts around
it), deterministic (k-slices summed in order by the last block of each output block). Built on first use with the
image's nvcc for sm_121 and cached under TORCH_EXTENSIONS_DIR (the serve container mounts ~/.cache, so it persists).

EXL3_L2_DISCARD=1 (default 0 until measured on the serve): the build with -DMROW_L2_DISCARD=1, whose last block of an
output block drops the k-split partials it has summed from L2 (discard.global.L2) instead of leaving them to be written
back to DRAM as the weights stream through; the outputs keep their bits (tests/test_l2_discard_gpu.py).

VLLM_DSV41_WOA_QDQ=1 (default 0): wo_a's group runs the build with -DMROW_QDQ=1, which gives its bf16 input DeepSeek's
FP8 round trip (woa_qdq.py) as it loads it: the same bits as woa_qdq.qdq then this kernel, no extra launch
(tests/test_woa_qdq_gpu.py).
"""

from __future__ import annotations

import os

import torch

_ext: dict[tuple[bool, bool], object] = {}


def l2_discard() -> bool:
    """EXL3_MROW_L2_DISCARD (0 / 1) overrides EXL3_L2_DISCARD for this kernel alone (isolating one kernel's discard in an A/B)."""
    v = os.environ.get("EXL3_MROW_L2_DISCARD")
    return (v if v is not None else os.environ.get("EXL3_L2_DISCARD", "0")) == "1"


def ext(discard: bool | None = None, qdq: bool = False):
    """The compiled kernel; `discard` picks the L2-discard build (None: as EXL3_L2_DISCARD says), `qdq` the build that
    rounds its input like DeepSeek's act_quant (wo_a's). Each build is a module of its own (jit.py hashes the flags into
    its name), so several can live in one process."""
    d = l2_discard() if discard is None else bool(discard)
    key = (d, bool(qdq))
    e = _ext.get(key)
    if e is None:
        from .jit import load
        # e.g. EXL3_MROW_FLAGS="-DMROW_PIPE=4" (tuning)
        flags = (("-DMROW_L2_DISCARD=1",) if d else ()) + (("-DMROW_QDQ=1",) if qdq else ())
        e = _ext[key] = load("exl3_mrow", "EXL3_MROW_FLAGS", cflags=flags)
    return e


def to_strips(t: torch.Tensor) -> torch.Tensor:
    """trellis [k/16, n/16, 16K] -> the same bytes in strips, [n/128][k/16][8 tiles][16K] (returned with the original
    shape): each 128-column item's k-rows contiguous, so a strips group streams an item's whole K range as one run.
    A k-row's 8 tiles keep their order, so every stage the kernel loads holds the same bytes as before."""
    k16, n16, w = t.shape
    assert n16 % 8 == 0, "n must be a multiple of 128"
    return t.view(k16, n16 // 8, 8, w).permute(1, 0, 2, 3).contiguous().view(k16, n16, w)


class MrowGroup:
    """One registered group: linears given as (trellis, suh, svh, x_col, out_col), all with the same K. `qdq`: the input
    gets DeepSeek's FP8 round trip on load, 32-value groups counted from each linear's own first input column (wo_a's
    head groups)."""

    def __init__(self, linears, K: int, want_items: int = 0, discard: bool | None = None, qdq: bool = False,
                 strips: bool = False):
        if not want_items:
            want_items = int(os.environ.get("EXL3_MROW_ITEMS", "0"))             # tuning override
        if not want_items:   # 1.5 items per SM: fewer, longer items beat finer splits (tune_mrow.py sweep, Oct 9)
            want_items = 3 * torch.cuda.get_device_properties(linears[0][0].device).multi_processor_count // 2
        tr, su, sv, xc, oc = zip(*linears)
        self.trellis = list(tr)          # the bytes the kernel streams (kept alive here; l2pf.py prefetches them)
        self.K = K
        self.qdq = bool(qdq)
        self.ext = ext(discard, qdq)     # the handle belongs to this build's registry
        self.handle = self.ext.register(list(tr), list(su), list(sv), list(xc), list(oc), K, want_items)
        if strips:                       # every trellis given here is in strips (to_strips)
            self.ext.set_strips(self.handle, [1] * len(tr))
        info = self.ext.info(self.handle)
        self.items, self.ks = info[0], info[1:]

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        self.ext.run(self.handle, x, out)
        return out
