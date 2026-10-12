# SPDX-License-Identifier: Apache-2.0
"""The *_gpu.py tests JIT-build the plugin's CUDA kernels on first use (vllm_exl3_moe/jit.py: torch's cpp_extension,
which needs ninja and an nvcc of the CUDA major torch was built with). The serve image has both. On a GPU machine
without them the GPU tests skip with the reason instead of failing in the build; without a GPU each file's own skipif
applies. The kernels build for the GPU in this machine unless EXL3_KERNEL_ARCH says otherwise (the plugin's default,
121, is GB10)."""

import os
import re
import shutil
import subprocess

import pytest


def _toolchain_missing():
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("EXL3_KERNEL_ARCH", f"{major}{minor}")
    missing = [] if shutil.which("ninja") else ["ninja"]
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc") if CUDA_HOME else None
    if not nvcc or not os.path.exists(nvcc):
        missing.append("nvcc")
    else:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+)\.", out)
        want = (torch.version.cuda or "").split(".")[0]
        if m and want and m.group(1) != want:
            missing.append(f"an nvcc for CUDA {want} (found {m.group(1)})")
    return missing


def pytest_collection_modifyitems(config, items):
    missing = _toolchain_missing()
    if not missing:
        return
    skip = pytest.mark.skip(reason="JIT-builds the plugin's CUDA kernels: needs " + " and ".join(missing))
    for item in items:
        path = str(item.fspath)
        if path.endswith("_gpu.py") and os.sep + os.path.join("exl3_moe", "tests") + os.sep in path:
            item.add_marker(skip)
