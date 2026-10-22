# SPDX-License-Identifier: Apache-2.0
"""JIT build of our CUDA kernels (csrc/*.cu) with the image's nvcc, cached under TORCH_EXTENSIONS_DIR (the serve
container mounts ~/.cache, so a build persists across boots). The module name carries a hash of the source, the shared
header and the flags: torch only tracks the listed sources, so a header edit would otherwise reuse a stale build."""

from __future__ import annotations

import hashlib
import os

CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
HEADERS = ("exl3_common.cuh", "exl3_launch.h")


def load(base: str, flags_env: str, ldflags=(), cflags=()):
    """`cflags`: build flags the caller picks (a variant, e.g. mrow.py's L2 discard), after the env's; hashed like them,
    so each variant is a module of its own and several can be loaded in one process."""
    from torch.utils.cpp_extension import load as _load
    src = os.path.join(CSRC, base + ".cu")
    arch = os.environ.get("EXL3_KERNEL_ARCH", os.environ.get("EXL3_MROW_ARCH", "121"))
    # tuning, e.g. EXL3_MROW_FLAGS="-DMROW_PIPE=4"; commas separate flags too (the launcher's env list splits on spaces)
    flags = os.environ.get(flags_env, "").replace(",", " ").split() + list(cflags)
    h = hashlib.md5()
    for f in (src, *(os.path.join(CSRC, n) for n in HEADERS)):
        with open(f, "rb") as fh:
            h.update(fh.read())
    h.update(" ".join([arch, *flags, *ldflags]).encode())
    return _load(name=f"{base}_{h.hexdigest()[:10]}", sources=[src], verbose=bool(os.environ.get("EXL3_KERNEL_VERBOSE")),
                 extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo", *flags],
                 extra_ldflags=list(ldflags))


def load_dir(sub: str, sources: tuple[str, ...], flags_env: str, include: tuple[str, ...] = ()):
    """A multi-file extension from csrc/<sub>/ (a vendored kernel set with its own headers): every file under it and
    under the ``include`` dirs (csrc-relative, searched first) goes into the module name's hash; the listed sources
    compile as one nvcc or c++ job each, in parallel under ninja."""
    from torch.utils.cpp_extension import load as _load
    d = os.path.join(CSRC, sub)
    inc = [os.path.join(CSRC, i) for i in include]
    arch = os.environ.get("EXL3_KERNEL_ARCH", os.environ.get("EXL3_MROW_ARCH", "121"))
    flags = os.environ.get(flags_env, "").replace(",", " ").split()
    h = hashlib.md5()
    for top in (d, *inc):
        for root, _, files in sorted(os.walk(top)):
            for n in sorted(files):
                with open(os.path.join(root, n), "rb") as fh:
                    h.update(os.path.relpath(os.path.join(root, n), CSRC).encode() + fh.read())
    h.update(" ".join([arch, *flags]).encode())
    return _load(name=f"{sub}_{h.hexdigest()[:10]}", sources=[os.path.join(d, s) for s in sources],
                 verbose=bool(os.environ.get("EXL3_KERNEL_VERBOSE")), extra_include_paths=inc,
                 extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo", *flags])
