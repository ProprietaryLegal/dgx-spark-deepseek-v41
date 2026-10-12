// Paced L2 prefetch of byte ranges: the weights the next kernels of a decode step read, pulled into L2 while the GPU
// runs kernels that leave DRAM idle (the TP all-reduces, attention, norms). Threads issue bulk L2 prefetches
// (cp.async.bulk.prefetch.L2, sm_90+: no registers, no shared memory, nothing waits for them) in chunks, the ranges in
// the order given (the order their reader takes them), after an optional delay: all at once, or one block's waves of
// blockDim chunks paced by the global timer, so DRAM's queue holds about a wave and the latency-bound kernels running
// beside keep their load latency. Nothing is written: no output changes, only where the next kernels find their bytes.
//
// The kernel is TensorFold's (github.com/bertholomus/TensorFold, branch deepseek-v41-tp2, commit f464b09e,
// src/tensorfold/cuda/exl3/l2_prefetch.cu; Copyright 2026 TensorFold contributors, Apache License 2.0, see
// NOTICE.tensorfold), with our launcher and binding.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>         // c10, not ATen/cuda: the image lacks cusparse.h
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <vector>

namespace {

constexpr int PMAX = 24;               // ranges a launch

struct PArgs {
    const char* p[PMAX];               // range starts (16-byte aligned)
    long long n[PMAX];                 // range bytes (multiples of 16)
    long long end[PMAX];               // chunk index one past range i (each range starts on a chunk of its own)
    int count;
    int chunk;                         // bytes a prefetch (a multiple of 16)
    long long delay_ns;                // issue this long after the kernel starts
    long long wave_ns;                 // > 0 (one block): a wave of blockDim chunks every wave_ns, in range order
};

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void __launch_bounds__(128) l2pf_kernel(const __grid_constant__ PArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (a.delay_ns > 0) {
        const unsigned long long t0 = gtime();
        while (gtime() - t0 < (unsigned long long)a.delay_ns) __nanosleep(500);
    }
    const long long total = a.end[a.count - 1];
    const long long stride = (long long)gridDim.x * blockDim.x;
    int r = 0;
    const unsigned long long t0 = gtime();
    long long wave = 0;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c - threadIdx.x < total; c += stride) {
        if (a.wave_ns > 0) {
            __syncthreads();
            if (threadIdx.x == 0)
                while (gtime() - t0 < (unsigned long long)(wave * a.wave_ns)) __nanosleep(256);
            __syncthreads();
            ++wave;
        }
        if (c >= total) continue;
        while (c >= a.end[r]) ++r;
        const long long off = (c - (r ? a.end[r - 1] : 0)) * a.chunk;
        const long long left = a.n[r] - off;
        const unsigned size = (unsigned)(left < a.chunk ? left : a.chunk);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a.p[r] + off), "r"(size) : "memory");
    }
#endif
}

}  // namespace

int64_t l2pf_max() { return PMAX; }

// Bulk L2 prefetches of the byte ranges (ptrs[i], sizes[i]) in order, `chunk` bytes each, on the current stream:
// wave_ns > 0: one block, a wave of 128 chunks every wave_ns; else up to `blocks` blocks issuing at once.
void l2pf(const std::vector<int64_t>& ptrs, const std::vector<int64_t>& sizes, int64_t chunk, int64_t blocks,
          int64_t delay_ns, int64_t wave_ns) {
    TORCH_CHECK(ptrs.size() == sizes.size() && (int)ptrs.size() <= PMAX, "at most ", PMAX, " ranges, a size each");
    TORCH_CHECK(chunk >= 16 && chunk % 16 == 0 && chunk <= (1 << 20) && blocks >= 1, "chunk: 16 B .. 1 MiB, x16");
    PArgs a;
    std::memset(&a, 0, sizeof(a));
    long long c = 0;
    for (size_t i = 0; i < ptrs.size(); ++i) {
        TORCH_CHECK(ptrs[i] % 16 == 0 && sizes[i] % 16 == 0 && sizes[i] > 0, "ranges: 16-byte aligned, x16 bytes");
        a.p[a.count] = reinterpret_cast<const char*>(ptrs[i]);
        a.n[a.count] = sizes[i];
        c += (sizes[i] + chunk - 1) / chunk;
        a.end[a.count] = c;
        ++a.count;
    }
    a.chunk = (int)chunk;
    a.delay_ns = delay_ns;
    a.wave_ns = wave_ns;
    if (a.count == 0 || c == 0) return;
    const long long threads = 128, need = (c + threads - 1) / threads;
    const unsigned grid = wave_ns > 0 ? 1u : (unsigned)(need < blocks ? need : blocks);
    l2pf_kernel<<<grid, (unsigned)threads, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l2pf", &l2pf, "paced bulk L2 prefetch of byte ranges on the current stream");
    m.def("l2pf_max", &l2pf_max, "ranges a launch");
}
