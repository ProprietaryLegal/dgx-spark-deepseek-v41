// SPDX-License-Identifier: Apache-2.0
// Device helpers shared by our EXL3 kernels (exl3_mrow.cu, exl3_smoe.cu): mbarrier / TMA bulk-copy wrappers, the mul1
// trellis tile decode straight into m16n8k16 B fragments, mma.sync and the 128-wide Hadamard on a warp.
//
// The decode device functions (fshift, bfe16, mul1_2 and the bit math in LaneDq / dq_words) are adapted from exllamav3
// v1.6.0, MIT License, Copyright (c) 2025 Turboderp (exllamav3_ext/quant/exl3_dq.cuh, quant/codebook.cuh); the notice
// is in LICENSE.exllamav3 next to this file.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace exl3c {

constexpr float RSQRT128 = 0.08838834764831845f;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n"
                 :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar)
{
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity)
{
    asm volatile("{\n.reg .pred p;\nWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT;\n}\n"
                 :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
// the same with the mbarrier's 32-bit shared-window address computed once outside the loop
__device__ __forceinline__ void mbar_arrive_s(uint32_t bar)
{
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait_s(uint32_t bar, uint32_t parity)
{
    asm volatile("{\n.reg .pred p;\nWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}
// 16-byte cp.async (the LSU path), and an mbarrier arrival that fires once every earlier cp.async of this thread has
// landed (.noinc: the arrival counts against the barrier's init count)
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_arrive_noinc(uint32_t bar)
{
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
}
// TMA bulk copy global -> shared, completion counted in bytes on the mbarrier
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

// ---- adapted from exllamav3 (see header) ----

__device__ __forceinline__ uint32_t fshift(uint32_t b, uint32_t a, int shift)
{
    uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (merged >> shift);
}

__device__ __forceinline__ uint32_t bfe16(uint32_t src, int pos) { return (src >> pos) & 0xffff; }

__device__ __forceinline__ half2 mul1_2(uint32_t w0, uint32_t w1)
{
    uint32_t x0 = w0 * 0x83DCD12Du;
    uint32_t x1 = w1 * 0x83DCD12Du;
    uint32_t s0 = __dp4a(x0, 0x01010101u, 0x6400u);
    uint32_t s1 = __dp4a(x1, 0x01010101u, 0x6400u);
    const half2 k_inv = __half2half2(__ushort_as_half(0x1eee));
    const half2 k_bias = __half2half2(__ushort_as_half(0xc931));
    half2 h = __halves2half2(__ushort_as_half((unsigned short) s0), __ushort_as_half((unsigned short) s1));
    return __hfma2(h, k_inv, k_bias);
}

// Per-lane word offsets inside a tile (fixed per lane: t_offset = lane * 8) and the shift amounts of exllamav3's
// dq8_aligned_2bits (K=2), dq8<3, cb, 4> (K=3), dq8_aligned_4bits (K=4) and two dq4 (K=5, 6), split from the decode so
// the words of later tiles can be loaded ahead of it. Same bit math as those functions.
template <int K>
struct LaneDq
{
    static constexpr int NW = K >= 5 ? 4 : 2;
    int idx[NW];
    int sh[2];

    __device__ __forceinline__ explicit LaneDq(int lane)
    {
        const int t = lane * 8;
        if constexpr (K == 2)
        {
            const int i1 = t >> 4;
            idx[0] = (i1 + 15) & 15; idx[1] = i1;
            sh[0] = ((~t) & 8) << 1; sh[1] = 0;
        }
        else if constexpr (K == 3)
        {
            const int b1 = (t + 257) * K, b2 = b1 + K * 7;
            const int i0 = (b1 - 16) / 32, i2 = (b2 - 1) / 32;
            idx[0] = i0 % (K * 8); idx[1] = i2 % (K * 8);
            sh[0] = (i2 + 1) * 32 - b2; sh[1] = 0;
        }
        else if constexpr (K == 4)
        {
            const int i1 = t >> 3;
            idx[0] = (i1 + 31) & 31; idx[1] = i1;
            sh[0] = 0; sh[1] = 0;
        }
        else
        {
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int b0 = (t + 4 * h + 257) * K - 16;
                const int b2 = b0 + 3 * K + 16;
                const int i0 = b0 / 32, i2 = (b2 - 1) / 32;
                idx[2 * h] = i0 % (K * 8); idx[2 * h + 1] = i2 % (K * 8);
                sh[h] = (i2 + 1) * 32 - b2;
            }
        }
    }
};

template <int K>
__device__ __forceinline__ void dq_words(const uint32_t (&w)[LaneDq<K>::NW], const LaneDq<K>& c,
                                         half2 (&f0)[2], half2 (&f1)[2])
{
    if constexpr (K == 2)
    {
        const uint32_t b = fshift(w[1], w[0], c.sh[0]);
        f0[0] = mul1_2(bfe16(b, 14), bfe16(b, 12));
        f0[1] = mul1_2(bfe16(b, 10), bfe16(b, 8));
        f1[0] = mul1_2(bfe16(b, 6), bfe16(b, 4));
        f1[1] = mul1_2(bfe16(b, 2), b & 0xffff);
    }
    else if constexpr (K == 3)
    {
        const uint32_t a = w[0], b = w[1];
        const uint32_t w7 = fshift(b, a, c.sh[0]);
        const uint32_t w6 = w7 >> K, w5 = w6 >> K, w4 = w5 >> K;
        const uint32_t w3 = fshift(b, a, c.sh[0] + K * 4);
        const uint32_t w2 = w3 >> K, w1 = w2 >> K, w0 = w1 >> K;
        f0[0] = mul1_2(w0 & 0xffff, w1 & 0xffff);
        f0[1] = mul1_2(w2 & 0xffff, w3 & 0xffff);
        f1[0] = mul1_2(w4 & 0xffff, w5 & 0xffff);
        f1[1] = mul1_2(w6 & 0xffff, w7 & 0xffff);
    }
    else if constexpr (K == 4)
    {
        const uint32_t a = w[0], b = w[1];
        const uint32_t s = __funnelshift_r(b, a, 20);
        f0[0] = mul1_2(bfe16(s, 8), bfe16(s, 4));
        f0[1] = mul1_2(s & 0xffff, bfe16(b, 16));
        f1[0] = mul1_2(bfe16(b, 12), bfe16(b, 8));
        f1[1] = mul1_2(bfe16(b, 4), b & 0xffff);
    }
    else
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const uint32_t a = w[2 * h], b = w[2 * h + 1];
            const int s2 = c.sh[h];
            const uint32_t w3 = fshift(b, a, s2) & 0xffff;
            const uint32_t w2 = fshift(b, a, s2 + K) & 0xffff;
            const uint32_t w1 = fshift(b, a, s2 + K * 2) & 0xffff;
            const uint32_t w0 = fshift(b, a, s2 + K * 3) & 0xffff;
            if (h == 0) { f0[0] = mul1_2(w0, w1); f0[1] = mul1_2(w2, w3); }
            else        { f1[0] = mul1_2(w0, w1); f1[1] = mul1_2(w2, w3); }
        }
    }
}

// ---- end of the exllamav3 adaptations ----

template <int K>
__device__ __forceinline__ void decode_tile(const uint32_t* ptr, const LaneDq<K>& c, half2 (&f0)[2], half2 (&f1)[2])
{
    uint32_t w[LaneDq<K>::NW];
    #pragma unroll
    for (int j = 0; j < LaneDq<K>::NW; ++j) w[j] = ptr[c.idx[j]];
    dq_words<K>(w, c, f0, f1);
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], const half2 (&b)[2])
{
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]));
}

// x * H_128 (Sylvester, unnormalised) for one 128-wide row block held 4 per lane (element = lane * 4 + j)
__device__ __forceinline__ void had128(float (&v)[4], int lane)
{
    float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2; v[1] = a1 + a3; v[2] = a0 - a2; v[3] = a1 - a3;
    #pragma unroll
    for (int s = 1; s < 32; s <<= 1)
    {
        const bool upper = lane & s;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float o = __shfl_xor_sync(0xffffffffu, v[j], s);
            v[j] = upper ? (o - v[j]) : (v[j] + o);
        }
    }
}

// Asynchronous prefetch of [src, src + bytes) into L2 (TMA engine, no shared memory, no completion to wait for);
// bytes a multiple of 16. For weights a kernel will stream, issued before griddep_wait.
__device__ __forceinline__ void l2_prefetch(const void* src, uint32_t bytes)
{
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(src), "r"(bytes) : "memory");
}

// Programmatic dependent launch (PDL): a kernel launched with programmatic stream serialization may start while the
// previous kernel in the stream is still running. griddep_wait blocks until that kernel has completed and its writes
// are visible (a no-op without PDL); griddep_launch lets this grid's own dependents start once every block has issued
// it or exited. Our rule: before griddep_wait, read only static data and write nothing global; issue griddep_launch
// only after griddep_wait, so a dependent that starts early can rely on everything before this kernel being complete.
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

}  // namespace exl3c
