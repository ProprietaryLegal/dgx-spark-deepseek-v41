// SPDX-License-Identifier: Apache-2.0
// vllm_exl3_moe: stands in for ATen/cuda/CUDAContext.h when csrc/tf_experts/ builds (jit.load_dir puts csrc/tf_shim
// first on the include path). The real header includes cuSPARSE and cuSOLVER headers that the serve image's CUDA
// toolkit lacks; TensorFold's files take only getCurrentCUDAStream from it, which c10 provides.
#pragma once
#include <c10/cuda/CUDAStream.h>

namespace at {
namespace cuda {
using c10::cuda::getCurrentCUDAStream;
}  // namespace cuda
}  // namespace at
