# SPDX-License-Identifier: Apache-2.0
"""copy_from_host: a host-to-device load copy from a safetensors mmap view (through the pinned staging buffer) leaves
the mapped pages file-backed (no copy-on-write growth of this process's anonymous memory) and lands the exact bytes,
contiguous and strided sources alike.

    python3 -m pytest -q test_hostcopy_gpu.py      (GPU)
"""
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_exl3_moe.hostcopy import copy_from_host

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
MiB = 2**20


def rss_anon() -> int:
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) * 1024 for line in f if line.startswith("RssAnon:"))


def test_mmap_view_copies_without_anonymous_growth(tmp_path):
    g = torch.Generator().manual_seed(0)
    full = torch.randint(-32768, 32767, (64, 1024, 1024), dtype=torch.int16, generator=g)   # 128 MiB
    path = str(tmp_path / "w.safetensors")
    save_file({"w": full}, path)
    want_rows, want_cols = full[32:].clone(), full[:, :, 512:].clone()
    del full
    dst_rows = torch.empty((32, 1024, 1024), dtype=torch.int16, device="cuda")
    dst_cols = torch.empty((64, 1024, 512), dtype=torch.int16, device="cuda")
    with safe_open(path, framework="pt") as f:
        w = f.get_tensor("w")
    copy_from_host(dst_rows[:1], w.narrow(0, 0, 1))     # first call allocates the pinned staging buffer
    torch.cuda.synchronize()
    a0 = rss_anon()
    copy_from_host(dst_rows, w.narrow(0, 32, 32))          # contiguous row cut (the w2 / svh case)
    copy_from_host(dst_cols, w.narrow(2, 512, 512))        # strided column cut (the w1 / w3 case)
    torch.cuda.synchronize()
    grown = rss_anon() - a0
    assert grown < 16 * MiB, f"anonymous memory grew {grown / MiB:.0f} MiB while copying 128 MiB from the mmap"
    assert torch.equal(dst_rows.cpu(), want_rows)
    assert torch.equal(dst_cols.cpu(), want_cols)


def test_cpu_destination_is_a_plain_copy():
    src = torch.arange(10, dtype=torch.float16)
    dst = torch.empty(10, dtype=torch.float16)
    assert copy_from_host(dst, src) is dst
    assert torch.equal(dst, src)
