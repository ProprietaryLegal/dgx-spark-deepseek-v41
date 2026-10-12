// SPDX-License-Identifier: Apache-2.0
// Host launch helper for our kernels: cudaLaunchKernelEx with programmatic stream serialization (PDL, see
// exl3_common.cuh) unless EXL3_PDL=0. Captured into a CUDA graph it becomes a programmatic edge.
#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <utility>

namespace exl3c {

inline bool pdl_enabled()
{
    static const bool on = [] { const char* v = std::getenv("EXL3_PDL"); return !(v && std::strcmp(v, "0") == 0); }();
    return on;
}

template <typename... KArgs, typename... Args>
void launch_k(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args&&... args)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl_enabled() ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

}  // namespace exl3c
