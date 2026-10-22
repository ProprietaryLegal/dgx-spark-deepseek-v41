// SPDX-License-Identifier: Apache-2.0
// exl3_mrow: EXL3 linears (mul1 codebook, K = 2..6) at decode sizes (1..16 rows), several linears per launch.
//
// One work item = one problem (linear) x one 128-column output block x one k-slice. Eight consumer warps (one 16-column
// tile each) transform the item's slice of the input (bf16 -> * suh -> 128-wide Hadamard -> fp16 in shared memory; the
// -DMROW_QDQ=1 build, wo_a's, first gives the bf16 input DeepSeek's FP8 round trip, see qdq_ue8m0_x4),
// decode the trellis tiles straight into tensor-core B fragments (the decode's register layout is the m16n8k16 B
// layout), accumulate in fp32 with mma.sync, apply the output Hadamard and svh to their partial (both linear, so per
// k-slice is exact), and the last block of an output block (one atomic counter, no waiting) sums the k-slices in slice
// order and writes bf16. Deterministic, graph-safe (no spin across blocks, no co-residency assumption), counters
// self-reset.
// Weights stream through a ring of MROW_PIPE shared-memory stages. A stage is one k-row of the item's 8 tiles, which is
// contiguous in the trellis (8 x 32K bytes), fetched with one TMA bulk copy that a producer warp issues as soon as the
// stage is free (mbarriers; no block-wide barrier in the loop). Why: GB10 reads 1280-byte chunks per k-row at 230-237
// GB/s but 160-byte per-warp tiles at 202-213 GB/s (read-pattern microbench, Oct 9), and per-warp direct loads left the
// warps waiting on DRAM latency (ncu: long-scoreboard stalls, 12 of 16.6 cycles per issued instruction at wo_b).
//
// The trellis decode it uses (exl3_common.cuh) is adapted from exllamav3 v1.6.0, MIT License, Copyright (c) 2025
// Turboderp; the notice is in LICENSE.exllamav3 next to this file.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if MROW_QDQ
#include <cuda_fp8.h>
#endif

#include "exl3_common.cuh"
#include "exl3_launch.h"

#include <algorithm>
#include <vector>

namespace {

using namespace exl3c;

constexpr int NB = 128;          // output columns per work item (one output Hadamard block)
constexpr int WARPS = 8;         // consumer warps, one 16-column tile each
constexpr int CONSUMER_THREADS = WARPS * 32;
constexpr int THREADS = CONSUMER_THREADS + 32;   // + the producer warp
constexpr int MAX_M = 16;
constexpr int MAX_P = 8;
#ifndef MROW_MAX_KS
#define MROW_MAX_KS 2048
#endif
#ifndef MROW_PIPE
#define MROW_PIPE 8
#endif
#ifndef MROW_MINB
#define MROW_MINB 1
#endif
#ifndef MROW_L2PF
#define MROW_L2PF 0      // k-rows of the item's weights past the ring prefetched into L2 before griddep_wait (0 = off)
#endif
#ifndef MROW_CPASYNC
#define MROW_CPASYNC 0   // 1: the producer's 32 lanes copy each stage with 16-byte cp.async (smoe's path) instead of one TMA bulk copy
#endif
#ifndef MROW_L2HINT
#define MROW_L2HINT 0    // with MROW_CPASYNC: 128 / 256 = the copies ask L2 for 128- / 256-byte DRAM fetches
#endif
#ifndef MROW_L2_DISCARD
#define MROW_L2_DISCARD 0   // 1 (EXL3_L2_DISCARD=1, mrow.py): the k-split partials leave L2 unwritten once summed (step 4)
#endif
#ifndef MROW_QDQ
#define MROW_QDQ 0          // 1 (VLLM_DSV41_WOA_QDQ=1, wo_a's group only, mrow.py): x gets DeepSeek's FP8 round trip on load
#endif
constexpr int MAX_KS = MROW_MAX_KS;   // k-slice cap: the transformed input slice lives in shared memory
constexpr int PIPE = MROW_PIPE;       // trellis stages (k-rows of the item) in flight
static_assert(PIPE >= 2 && PIPE % 2 == 0, "MROW_PIPE: even, >= 2 (two k-rows per loop step)");
constexpr int SX_PAD = 8;        // halves of row padding: rows land on different banks for the A-fragment loads

struct Problem
{
    const uint16_t* trellis;     // [k/16][n/16][16K]
    const half* suh;             // [k]
    const half* svh;             // [n]
    int k, n;
    int x_col, out_col;          // column offsets of this problem's input / output in x / out
    int ks, n_slices, n_blocks;
    int item_start;              // first work item of this problem
    long long ws_off;            // float offset of this problem's k-slice partials ([n_slices][MAX_M][n])
    int ctr_off;                 // offset of this problem's per-output-block counters
    int strips;                  // 1: trellis in strips, [n/128][k/16][8 tiles][16K] (set_strips): an item's k-rows
                                 // contiguous, so its whole K range streams as one run (the heads, EXL3_HEAD_STRIPS)
};

struct Group
{
    Problem p[MAX_P];
    int n_problems;
    int n_items;
    int ks_max;                  // widest k-slice in the group: the shared-memory input rows are this wide
};

// shared memory per block: the transformed input [rows][ks_max + SX_PAD] halves, then the trellis ring (PIPE stages of
// 8 tiles), which the epilogue's [rows][128] fp32 partial reuses
__host__ __device__ constexpr int smem_sx_bytes(int rows, int ks_max) { return ((rows * (ks_max + SX_PAD) * 2 + 15) / 16) * 16; }
__host__ __device__ constexpr int smem_ring_bytes(int K) { return WARPS * PIPE * 32 * K; }
__host__ __device__ constexpr int smem_bytes(int K, int rows, int ks_max)
{
    return smem_sx_bytes(rows, ks_max) + (smem_ring_bytes(K) > rows * NB * 4 ? smem_ring_bytes(K) : rows * NB * 4);
}

// mbarrier / TMA helpers, tile decode, mma and Hadamard: exl3_common.cuh
__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;\n" :: "n"(CONSUMER_THREADS) : "memory"); }

#if MROW_QDQ
// DeepSeek-V4.1-Flash-1017's act_quant(o, 32, "ue8m0", e8m0, inplace=True) before wo_a (inference/model.py:787, 1073;
// the semantics and line references are in woa_qdq.py), on the four values a lane loads in step 1: a 32-value group is
// 8 lanes of a 128-wide block (lanes 8j .. 8j + 7 hold its columns 32j .. 32j + 31, problem-local). amax over the
// group (xor shuffles inside the 8 lanes), at least 1e-4; s = 2^ceil(log2(amax * fl32(1/448))) from the IEEE bits like
// the reference's fast_round_scale; x / s (exact, as x * 2^-e) to e4m3 round-to-nearest-even (satfinite: |x / s| <= 448,
// so it never clamps), back times s. The reference stores that product as bf16; here it already is a bf16 value (4
// significant bits at most, 2^-31 <= |q s| <= 240 * 2^120 when not zero) or inf (256 * 2^120 overflows fp32 as it does
// bf16), so that store would change nothing and is left out. Same bits as woa_qdq.py's kernel, which the prompt sizes
// use (tests/test_woa_qdq_gpu.py).
__device__ __forceinline__ float2 qdq_e4m3x2(float2 v, float s, float inv)
{
    const __half2_raw q = __nv_cvt_fp8x2_to_halfraw2(
        __nv_cvt_float2_to_fp8x2(make_float2(v.x * inv, v.y * inv), __NV_SATFINITE, __NV_E4M3), __NV_E4M3);
    const float2 f = __half22float2(__half2(q));
    return make_float2(f.x * s, f.y * s);
}

__device__ __forceinline__ void qdq_ue8m0_x4(float2& x01, float2& x23)
{
    float a = fmaxf(fmaxf(fabsf(x01.x), fabsf(x01.y)), fmaxf(fabsf(x23.x), fabsf(x23.y)));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4));
    const uint32_t t = __float_as_uint(fmaxf(a, 1e-4f) * (1.0f / 448.0f));
    const int e = (int) ((t >> 23) & 0xFFu) - 127 + ((t & 0x7FFFFFu) != 0u);     // -22 .. 120 for finite bf16
    const float s = __uint_as_float((uint32_t) (e + 127) << 23);
    const float inv = e < 128 ? __uint_as_float((uint32_t) (127 - e) << 23) : 0.f;   // inf amax: the reference's NaN group
    x01 = qdq_e4m3x2(x01, s, inv);
    x23 = qdq_e4m3x2(x23, s, inv);
}
#endif

// one stage (SB bytes, contiguous) by the producer warp's 32 lanes in 16-byte cp.async pieces; the stage's mbarrier
// counts one arrival per lane, each fired when that lane's copies have landed (MROW_CPASYNC)
template <int SB>
__device__ __forceinline__ void stage_cp_async(unsigned char* dst, const unsigned char* src, uint64_t* bar, int lane)
{
    const uint32_t d = smem_u32(dst);
    for (int i = lane; i < SB / 16; i += 32)
    {
#if MROW_L2HINT == 256
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(d + i * 16), "l"(src + i * 16) : "memory");
#elif MROW_L2HINT == 128
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(d + i * 16), "l"(src + i * 16) : "memory");
#else
        cp_async16(d + i * 16, src + i * 16);
#endif
    }
    cp_async_arrive_noinc(smem_u32(bar));
}

// M16: 9..16 rows (two 8-row halves of the m16 fragment); otherwise 1..8 rows, the upper half of A is zero registers
template <int K, bool M16>
__global__ void __launch_bounds__(THREADS, MROW_MINB) mrow_kernel
(
    const __nv_bfloat16* __restrict__ x, int x_stride,
    __nv_bfloat16* __restrict__ out, int out_stride,
    int m, float* __restrict__ ws, int* __restrict__ ctr, const __grid_constant__ Group g
)
{
    constexpr int RX = M16 ? 16 : 8;     // input rows held in shared memory
    constexpr int TB = 32 * K;           // bytes per 16x16 trellis tile
    constexpr int SB = WARPS * TB;       // bytes per stage: the item's 8 tiles of one k-row, contiguous in the trellis
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ __align__(8) uint64_t full[PIPE];
    __shared__ __align__(8) uint64_t empty[PIPE];
    __shared__ int s_last;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    int pi = 0;
    #pragma unroll 1
    while (pi + 1 < g.n_problems && (int) blockIdx.x >= g.p[pi + 1].item_start) ++pi;
    const Problem& P = g.p[pi];
    const int local = blockIdx.x - P.item_start;
    const int nb = local % P.n_blocks;
    const int sl = local / P.n_blocks;
    const int k0 = sl * P.ks;
    const int n0 = nb * NB;
    const int kts = P.ks / 16;
    half* sx = reinterpret_cast<half*>(smem);
    unsigned char* ring = smem + smem_sx_bytes(RX, g.ks_max);
    float* sy = reinterpret_cast<float*>(ring);                              // [RX][NB], once every stage is consumed

    if (threadIdx.x == 0)
    {
        for (int s = 0; s < PIPE; ++s) { mbar_init(&full[s], MROW_CPASYNC ? 32 : 1); mbar_init(&empty[s], WARPS); }
        mbar_fence_init();
    }
    __syncthreads();

    // producer warp: one bulk copy per k-row of the slice, issued as soon as its stage is free. The first PIPE stages
    // are weights only (static), so they go out before griddep_wait: under PDL they stream while the previous kernel
    // finishes.
    if (warp == WARPS)
    {
        const int tiles_n = P.n / 16;
        const size_t row_bytes = P.strips ? (size_t) SB : (size_t) tiles_n * TB;
        const unsigned char* src = reinterpret_cast<const unsigned char*>(P.trellis)
                                   + (P.strips ? ((size_t) nb * (P.k / 16) + k0 / 16) * SB
                                               : ((size_t) (k0 / 16) * tiles_n + n0 / 16) * TB);
        const int pre = kts < PIPE ? kts : PIPE;
#if MROW_CPASYNC
        for (int kk = 0; kk < pre; ++kk) stage_cp_async<SB>(ring + kk * SB, src + kk * row_bytes, &full[kk], lane);
        griddep_wait();
        griddep_launch();
        #pragma unroll 1
        for (int kk = pre; kk < kts; ++kk)
        {
            const int s = kk % PIPE;
            mbar_wait(&empty[s], ((kk / PIPE) - 1) & 1);
            stage_cp_async<SB>(ring + s * SB, src + kk * row_bytes, &full[s], lane);
        }
        return;
#endif
        if (lane == 0)
        {
            for (int kk = 0; kk < pre; ++kk)
            {
                mbar_expect_tx(&full[kk], SB);
                bulk_load(ring + kk * SB, src + kk * row_bytes, SB, &full[kk]);
            }
#if MROW_L2PF > 0
            // the next k-rows into L2 as well, so the stream after the wait starts from L2 (TensorFold's kernels
            // prefetch their weights to L2 before griddepcontrol.wait)
            const int pf_end = kts < pre + MROW_L2PF ? kts : pre + MROW_L2PF;
            for (int kk = pre; kk < pf_end; ++kk) l2_prefetch(src + kk * row_bytes, SB);
#endif
        }
        griddep_wait();
        griddep_launch();
        if (lane == 0)
        {
            #pragma unroll 1
            for (int kk = pre; kk < kts; ++kk)
            {
                const int s = kk % PIPE;
                mbar_wait(&empty[s], ((kk / PIPE) - 1) & 1);
                mbar_expect_tx(&full[s], SB);
                bulk_load(ring + s * SB, src + kk * row_bytes, SB, &full[s]);
            }
        }
        return;
    }
    griddep_wait();                      // x comes from the previous kernel
    griddep_launch();

    // 1. input slice: x[:, x_col + k0 : +ks] * suh, Hadamard per 128 block, fp16 into shared memory; rows >= m are zero
    const int kblocks = P.ks / 128;
    const int sxw = g.ks_max + SX_PAD;                                       // halves per shared input row
    const int ntask = m * kblocks;                                           // (row, 128 block) pairs
    constexpr int XB = 4;                                                    // tasks per warp with loads in flight
    for (int u0 = warp; u0 < ntask; u0 += XB * WARPS)
    {
        uint2 xv[XB], hv[XB];
        #pragma unroll
        for (int i = 0; i < XB; ++i)
        {
            const int u = u0 + i * WARPS;
            if (u >= ntask) break;
            const int r = u / kblocks;
            const int c = k0 + (u % kblocks) * 128 + lane * 4;
            xv[i] = *reinterpret_cast<const uint2*>(x + (size_t) r * x_stride + P.x_col + c);
            hv[i] = *reinterpret_cast<const uint2*>(P.suh + c);
        }
        #pragma unroll
        for (int i = 0; i < XB; ++i)
        {
            const int u = u0 + i * WARPS;
            if (u >= ntask) break;
            const int r = u / kblocks;
            const int bb = u % kblocks;
            float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv[i].x));
            float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv[i].y));
#if MROW_QDQ
            qdq_ue8m0_x4(x01, x23);          // the whole warp is here (u is warp-uniform), so the shuffles see all 32 lanes
#endif
            const float2 h01 = __half22float2(*reinterpret_cast<const half2*>(&hv[i].x));
            const float2 h23 = __half22float2(*reinterpret_cast<const half2*>(&hv[i].y));
            float v[4] = {x01.x * h01.x, x01.y * h01.y, x23.x * h23.x, x23.y * h23.y};
            had128(v, lane);
            const half2 o01 = __floats2half2_rn(v[0] * RSQRT128, v[1] * RSQRT128);
            const half2 o23 = __floats2half2_rn(v[2] * RSQRT128, v[3] * RSQRT128);
            uint2 ov;
            ov.x = *reinterpret_cast<const uint32_t*>(&o01);
            ov.y = *reinterpret_cast<const uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(sx + r * sxw + bb * 128 + lane * 4) = ov;
        }
    }
    for (int i = threadIdx.x; i < (RX - m) * P.ks; i += CONSUMER_THREADS)
        sx[(m + i / P.ks) * sxw + i % P.ks] = __float2half_rn(0.f);
    consumer_sync();

    // 2. this warp's 16 output columns over the k-slice: decode + mma, two k-rows per step
    const int gq = lane / 4;
    const int tq = lane % 4;
    const LaneDq<K> cfg(lane);
    float c0[4] = {0.f, 0.f, 0.f, 0.f};
    float c1[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t* xr0 = reinterpret_cast<const uint32_t*>(sx + gq * sxw);
    const uint32_t* xr1 = reinterpret_cast<const uint32_t*>(sx + (gq + 8) * sxw);
    const unsigned char* wtile = ring + warp * TB;
    #pragma unroll 1
    for (int kk = 0; kk < kts; kk += 2)
    {
        const int s0 = kk % PIPE, s1 = s0 + 1;
        const uint32_t par = (kk / PIPE) & 1;
        mbar_wait(&full[s0], par);
        mbar_wait(&full[s1], par);
        half2 f0[2], f1[2], e0[2], e1[2];
        decode_tile<K>(reinterpret_cast<const uint32_t*>(wtile + s0 * SB), cfg, f0, f1);
        decode_tile<K>(reinterpret_cast<const uint32_t*>(wtile + s1 * SB), cfg, e0, e1);
        uint32_t a[4], b[4];
        a[0] = xr0[kk * 8 + tq];     a[2] = xr0[kk * 8 + tq + 4];
        b[0] = xr0[kk * 8 + 8 + tq]; b[2] = xr0[kk * 8 + 8 + tq + 4];
        if constexpr (M16)
        {
            a[1] = xr1[kk * 8 + tq];     a[3] = xr1[kk * 8 + tq + 4];
            b[1] = xr1[kk * 8 + 8 + tq]; b[3] = xr1[kk * 8 + 8 + tq + 4];
        }
        else { a[1] = 0u; a[3] = 0u; b[1] = 0u; b[3] = 0u; }
        mma16816(c0, a, f0);
        mma16816(c1, a, f1);
        mma16816(c0, b, e0);
        mma16816(c1, b, e1);
        __syncwarp();                    // every lane has read both stages before they are handed back
        if (lane == 0) { mbar_arrive(&empty[s0]); mbar_arrive(&empty[s1]); }
    }

    // 3. output Hadamard + svh on the partial, then write (one slice) or keep for the k-split sum
    consumer_sync();                     // sy reuses the ring: every warp is past its last stage
    const int col = warp * 16 + 2 * tq;
    sy[gq * NB + col] = c0[0];     sy[gq * NB + col + 1] = c0[1];
    sy[gq * NB + col + 8] = c1[0]; sy[gq * NB + col + 9] = c1[1];
    if constexpr (M16)
    {
        sy[(gq + 8) * NB + col] = c0[2];     sy[(gq + 8) * NB + col + 1] = c0[3];
        sy[(gq + 8) * NB + col + 8] = c1[2]; sy[(gq + 8) * NB + col + 9] = c1[3];
    }
    consumer_sync();

    const int oc = n0 + lane * 4;
    for (int r = warp; r < m; r += WARPS)
    {
        float v[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = sy[r * NB + lane * 4 + j];
        had128(v, lane);
        const uint2 sv = *reinterpret_cast<const uint2*>(P.svh + oc);
        const float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sv.x));
        const float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sv.y));
        const float y0 = v[0] * RSQRT128 * s01.x, y1 = v[1] * RSQRT128 * s01.y;
        const float y2 = v[2] * RSQRT128 * s23.x, y3 = v[3] * RSQRT128 * s23.y;
        if (P.n_slices == 1)
        {
            const __nv_bfloat162 o01 = __floats2bfloat162_rn(y0, y1), o23 = __floats2bfloat162_rn(y2, y3);
            uint2 ov;
            ov.x = *reinterpret_cast<const uint32_t*>(&o01);
            ov.y = *reinterpret_cast<const uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(out + (size_t) r * out_stride + P.out_col + oc) = ov;
        }
        else
            *reinterpret_cast<float4*>(ws + P.ws_off + ((size_t) sl * m + r) * P.n + oc) = make_float4(y0, y1, y2, y3);
    }
    if (P.n_slices == 1) return;

    // 4. k-split: the last block of this output block sums the slices in order (no waiting on anyone)
    __threadfence();
    consumer_sync();
    if (threadIdx.x == 0)
    {
        const int prev = atomicAdd(&ctr[P.ctr_off + nb], 1);
        s_last = prev == P.n_slices - 1;
    }
    consumer_sync();
    if (!s_last) return;
    __threadfence();
    for (int r = warp; r < m; r += WARPS)
    {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int q = 0; q < P.n_slices; ++q)
        {
            const float4 t = __ldcg(reinterpret_cast<const float4*>(ws + P.ws_off + ((size_t) q * m + r) * P.n + oc));
            acc.x += t.x; acc.y += t.y; acc.z += t.z; acc.w += t.w;
        }
        const __nv_bfloat162 o01 = __floats2bfloat162_rn(acc.x, acc.y), o23 = __floats2bfloat162_rn(acc.z, acc.w);
        uint2 ov;
        ov.x = *reinterpret_cast<const uint32_t*>(&o01);
        ov.y = *reinterpret_cast<const uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(out + (size_t) r * out_stride + P.out_col + oc) = ov;
#if MROW_L2_DISCARD
        // Row r's partials of this output block, every k-slice (n_slices x 512 bytes, four 128-byte lines each: ws is
        // 128-byte aligned and ws_off, the (slice, row) offsets and n0 are multiples of 128 floats), are dead here, so
        // their dirty L2 lines are dropped instead of written back to DRAM. Why nothing can read a dropped line:
        //  - writers: each k-slice block wrote its rows in step 3 and ran __threadfence() before its arrival on this
        //    output block's counter; this block took the last arrival and fenced, so every partial was in L2 before the
        //    __ldcg loads above;
        //  - readers: only this block reads them, and row r only in this warp (rows r = warp, warp + 8). Each lane's
        //    loads feed acc and the bf16 store just above, so they have returned before the lane reaches __syncwarp(),
        //    which orders all 32 lanes' loads before any lane's discard;
        //  - the next writer is this group's next launch, whose consumers write ws only after griddepcontrol.wait, i.e.
        //    after this grid has completed (the ordering the loads and the counter reset already rely on), and a launch
        //    writes every (slice, row, column) partial it sums before reading it, so a dropped line is never read.
        __syncwarp();
        const float* zr = ws + P.ws_off + (size_t) r * P.n + n0;
        for (int i = lane; i < P.n_slices * 4; i += 32)
            l2_discard(zr + (size_t) (i >> 2) * m * P.n + (i & 3) * 32);
#endif
    }
    if (threadIdx.x == 0) ctr[P.ctr_off + nb] = 0;
}

struct Handle
{
    Group g;
    int K;
    at::Tensor ws, ctr;
    std::vector<at::Tensor> keep;   // keep the weights alive while registered
};

std::vector<Handle>& registry()
{
    static std::vector<Handle> r;
    return r;
}

int pick_ks(int k, int n_blocks, int want_items)
{
    int best = 128;
    for (int ks = 128; ks <= std::min(k, MAX_KS); ks += 128)
    {
        if (k % ks) continue;
        if ((long long) n_blocks * (k / ks) >= want_items) best = ks;   // largest slice that still fills the GPU
    }
    if ((long long) n_blocks * (k / best) < want_items)                  // nothing fills it: the most slices we can
        for (int ks = 128; ks <= std::min(k, MAX_KS); ks += 128)
            if (k % ks == 0) { best = ks; break; }
    return best;
}

template <int K, bool M16>
void launch(const Group& g, const __nv_bfloat16* xp, int xs, __nv_bfloat16* op, int os, int m, float* ws, int* ctr,
            cudaStream_t stream)
{
    const int smem = smem_bytes(K, M16 ? 16 : 8, g.ks_max);
    static int smem_set = 48 * 1024;      // dynamic shared memory past 48 KB needs the opt-in, once per size
    if (smem > smem_set)
    {
        C10_CUDA_CHECK(cudaFuncSetAttribute(mrow_kernel<K, M16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        smem_set = smem;
    }
    launch_k(mrow_kernel<K, M16>, dim3(g.n_items), dim3(THREADS), smem, stream, xp, xs, op, os, m, ws, ctr, g);
}

}  // namespace

int64_t mrow_register(std::vector<at::Tensor> trellis, std::vector<at::Tensor> suh, std::vector<at::Tensor> svh,
                      std::vector<int64_t> x_col, std::vector<int64_t> out_col, int64_t K, int64_t want_items)
{
    const int np = (int) trellis.size();
    TORCH_CHECK(np >= 1 && np <= MAX_P, "exl3_mrow: 1..", MAX_P, " linears per group");
    TORCH_CHECK(suh.size() == (size_t) np && svh.size() == (size_t) np && x_col.size() == (size_t) np &&
                out_col.size() == (size_t) np, "exl3_mrow: per-linear lists differ in length");
    TORCH_CHECK(K >= 2 && K <= 6, "exl3_mrow: K ", K, " (2..6 supported)");
    Handle h;
    h.K = (int) K;
    h.g.n_problems = np;
    const int per = (int) ((want_items + np - 1) / np);
    long long ws_floats = 0;
    int ctrs = 0, items = 0;
    for (int i = 0; i < np; ++i)
    {
        const at::Tensor& t = trellis[i];
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kShort && t.dim() == 3 && t.size(2) == 16 * K &&
                    t.is_contiguous(), "exl3_mrow: trellis ", i, " must be contiguous int16 [k/16, n/16, 16K]");
        const int k = (int) t.size(0) * 16, n = (int) t.size(1) * 16;
        TORCH_CHECK(k % 128 == 0 && n % 128 == 0, "exl3_mrow: k and n must be multiples of 128");
        TORCH_CHECK(suh[i].scalar_type() == at::kHalf && suh[i].numel() == k && suh[i].is_contiguous(), "suh ", i);
        TORCH_CHECK(svh[i].scalar_type() == at::kHalf && svh[i].numel() == n && svh[i].is_contiguous(), "svh ", i);
        TORCH_CHECK(x_col[i] % 4 == 0 && out_col[i] % 4 == 0, "exl3_mrow: column offsets must be multiples of 4");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "exl3_mrow: trellis ", i, " not 16-byte aligned");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(suh[i].data_ptr()) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(svh[i].data_ptr()) % 8 == 0, "exl3_mrow: suh / svh ", i, " not 8-byte aligned");
        Problem& p = h.g.p[i];
        p.trellis = reinterpret_cast<const uint16_t*>(t.data_ptr<int16_t>());
        p.suh = reinterpret_cast<const half*>(suh[i].data_ptr<at::Half>());
        p.svh = reinterpret_cast<const half*>(svh[i].data_ptr<at::Half>());
        p.k = k; p.n = n;
        p.x_col = (int) x_col[i]; p.out_col = (int) out_col[i];
        p.n_blocks = n / NB;
        p.ks = pick_ks(k, p.n_blocks, per);
        p.n_slices = k / p.ks;
        p.item_start = items;
        items += p.n_blocks * p.n_slices;
        p.ws_off = ws_floats;
        if (p.n_slices > 1) ws_floats += (long long) p.n_slices * MAX_M * n;
        p.ctr_off = ctrs;
        p.strips = 0;
        ctrs += p.n_blocks;
        h.keep.push_back(t); h.keep.push_back(suh[i]); h.keep.push_back(svh[i]);
    }
    h.g.n_items = items;
    h.g.ks_max = 0;
    for (int i = 0; i < np; ++i) h.g.ks_max = std::max(h.g.ks_max, h.g.p[i].ks);
    auto opts = trellis[0].options();
    h.ws = at::empty({std::max<long long>(ws_floats, 1)}, opts.dtype(at::kFloat));
    h.ctr = at::zeros({std::max(ctrs, 1)}, opts.dtype(at::kInt));
#if MROW_L2_DISCARD
    TORCH_CHECK(reinterpret_cast<uintptr_t>(h.ws.data_ptr()) % 128 == 0, "exl3_mrow: ws not 128-byte aligned (L2 discard)");
#endif
    registry().push_back(h);
    return (int64_t) registry().size() - 1;
}

void mrow_set_strips(int64_t handle, std::vector<int64_t> strips)
{
    Handle& h = registry().at(handle);
    TORCH_CHECK(strips.size() == (size_t) h.g.n_problems, "exl3_mrow: one strips flag per linear");
    for (int i = 0; i < h.g.n_problems; ++i) h.g.p[i].strips = strips[i] ? 1 : 0;
}

std::vector<int64_t> mrow_info(int64_t handle)
{
    const Handle& h = registry().at(handle);
    std::vector<int64_t> r = {h.g.n_items};
    for (int i = 0; i < h.g.n_problems; ++i) r.push_back(h.g.p[i].ks);
    return r;
}

void mrow_run(int64_t handle, at::Tensor x, at::Tensor out)
{
    Handle& h = registry().at(handle);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "exl3_mrow: bf16 in and out");
    TORCH_CHECK((x.dim() == 2 || x.dim() == 3) && out.dim() == 2 && x.stride(-1) == 1 && out.stride(1) == 1,
                "exl3_mrow: x [m, *] or [m, problems, *] with unit last stride, out row-major [m, *]");
    const int m = (int) x.size(0);
    TORCH_CHECK(m >= 1 && m <= MAX_M && out.size(0) == m, "exl3_mrow: 1..16 rows");
    TORCH_CHECK(x.stride(0) % 4 == 0 && out.stride(0) % 4 == 0, "exl3_mrow: row strides must be multiples of 4");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
                "exl3_mrow: x and out must be 8-byte aligned (4-element vector loads and stores)");
    // 3-D x: problem i reads x[:, i, :] wherever that sits (any row / problem strides), e.g. wo_a's groups straight from
    // the inverse-RoPE kernel's [G, T, D] buffer seen as [T, G, D], instead of a copy to [T, G * D] first. The
    // descriptor goes to the kernel by value, so a per-call copy with these offsets also holds under graph capture.
    Group g = h.g;
    if (x.dim() == 3)
    {
        TORCH_CHECK(x.size(1) == h.g.n_problems && x.stride(1) % 4 == 0 && x.stride(1) * (h.g.n_problems - 1) < (1LL << 31),
                    "exl3_mrow: 3-D x needs one [m, *] slice per problem, problem stride a multiple of 4");
        for (int i = 0; i < h.g.n_problems; ++i)
        {
            TORCH_CHECK(h.g.p[i].x_col == i * h.g.p[0].k, "exl3_mrow: 3-D x needs problem i registered at column i * k");
            TORCH_CHECK(h.g.p[i].k <= x.size(2), "exl3_mrow: input narrower than problem ", i);
            g.p[i].x_col = (int) (i * x.stride(1));
        }
    }
    for (int i = 0; i < h.g.n_problems; ++i)
    {
        if (x.dim() == 2) TORCH_CHECK(h.g.p[i].x_col + h.g.p[i].k <= x.size(1), "exl3_mrow: input narrower than problem ", i);
        TORCH_CHECK(h.g.p[i].out_col + h.g.p[i].n <= out.size(1), "exl3_mrow: output narrower than problem ", i);
    }
    const c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();   // c10, not ATen/cuda: the image lacks cusparse.h
    auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    float* ws = h.ws.data_ptr<float>();
    int* ctr = h.ctr.data_ptr<int>();
    const int xs = (int) x.stride(0), os = (int) out.stride(0);
    const bool m16 = m > 8;
    switch (h.K)
    {
        case 2: m16 ? launch<2, true>(g, xp, xs, op, os, m, ws, ctr, stream) : launch<2, false>(g, xp, xs, op, os, m, ws, ctr, stream); break;
        case 3: m16 ? launch<3, true>(g, xp, xs, op, os, m, ws, ctr, stream) : launch<3, false>(g, xp, xs, op, os, m, ws, ctr, stream); break;
        case 4: m16 ? launch<4, true>(g, xp, xs, op, os, m, ws, ctr, stream) : launch<4, false>(g, xp, xs, op, os, m, ws, ctr, stream); break;
        case 5: m16 ? launch<5, true>(g, xp, xs, op, os, m, ws, ctr, stream) : launch<5, false>(g, xp, xs, op, os, m, ws, ctr, stream); break;
        case 6: m16 ? launch<6, true>(g, xp, xs, op, os, m, ws, ctr, stream) : launch<6, false>(g, xp, xs, op, os, m, ws, ctr, stream); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod)
{
    mod.def("register", &mrow_register, "register a group of EXL3 linears; returns a handle");
    mod.def("info", &mrow_info, "[work items, k-slice per linear...]");
    mod.def("set_strips", &mrow_set_strips, "per linear: 1 = its trellis is in strips ([n/128][k/16][8][16K], to_strips)");
    mod.def("workspace", [](int64_t handle) { return registry().at(handle).ws; },
            "the group's fp32 k-split partials buffer (tests: tests/test_l2_discard_gpu.py fills it with a canary)");
    mod.def("run", &mrow_run, "run a registered group: x bf16 [m, *] (or [m, problems, *]) -> out bf16 [m, *]");
}
