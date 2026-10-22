// SPDX-License-Identifier: Apache-2.0
// exl3_smoe: the routed experts of one MoE layer at decode sizes (m tokens x top_k picks <= 256 slots), EXL3 mul1
// experts with K = 2..6 bits each, every K in one launch; deterministic, graph-safe (no host sync, static grids),
// three launches per layer:
//   1. rot:  per routed slot (token, pick) and projection: x * suh (gate | up) -> 128-wide Hadamard -> fp16, stored per
//            16-wide k-tile ([slot][k-tile][gate 16 | up 16]) so one row of one k-tile is one 64-byte copy. Block 0
//            also builds the run table: the distinct experts picked, in descending physical slot (= descending K, the
//            heavy items start first), each run with up to 8 rows (slots).
//   2. gu:   one block per (run, 128 intermediate columns), all of k: a producer warp streams each k-row's 8 gate and 8
//            up tiles (two contiguous chunks) plus the run's input rows into an mbarrier ring (32 lanes of 16-byte
//            cp.async, completion counted on the stage's mbarrier), 8 decode warps (one 16-column tile of gate and of up each) accumulate in fp32 with mma.sync, then per row:
//            output Hadamard * svh, silu(min(gate, limit)) * clamp(up, +-limit), * down suh, Hadamard -> fp16 down input
//            [run][k-tile][row][16].
//   3. down: one block per (run, OBI x 128 output columns), all of the intermediate dim: the same stream over the down
//            tiles and the run's down input; per row: output Hadamard * svh * routing weight -> fp32 per slot; the last
//            arrival of a (token, 128 block) sums the token's picks in pick order and writes bf16.
// Weights are read once per (expert, launch) whatever the number of rows; blocks past the run count exit at once.
// Shared expert as a slot (attach_shared + run(..., shared=1)): the layer's shared expert (an EXL3 dense MLP with the
// same tile layout and TP cut as an expert slot) is physical slot E, read from its own tensors, and every live token
// gets it as pick top_k with weight 1; the per-token sum adds it in pick order with the routed picks and rounds once, so
// the shared MLP's own launches, the aux-stream join and its separate bf16 add go away (TensorFold's fused MoE decode
// runs the shared expert as a 7th slot the same way: github.com/bertholomus/TensorFold, branch deepseek-v41-tp2,
// 662b395; this is our implementation in smoe).
// SMOE_RUNFLAGS=1 (build flag, EXL3_SMOE_FLAGS="-DSMOE_RUNFLAGS=1"): down waits per run, not per grid. Each gate / up
// block publishes its act rows and counts itself on its run's counter (rot zeroes them); a down block's producer waits
// for its own run's count instead of griddepcontrol.wait, so down blocks of finished runs work through gate / up's tail
// (PDL schedules down blocks only after every gate / up block has started: no deadlock). Same math, same sums.
// Why a block-level stream: GB10 reads block-contiguous chunks per k-row at 230-240 GB/s against ~205 for per-warp
// 160-byte tiles; why cp.async, not TMA bulk copies: with one small copy per input row per k-row the TMA path capped
// the stream near 200-210 GB/s, 32 lanes of cp.async move the same stages at 207-221 (LOG, Oct 9).
// The trellis decode (exl3_common.cuh) is adapted from exllamav3 v1.6.0, MIT License, Copyright (c) 2025 Turboderp;
// the notice is in LICENSE.exllamav3 next to this file.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_common.cuh"
#include "exl3_launch.h"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

using namespace exl3c;

constexpr int WARPS = 8;                    // decode warps
constexpr int CT = WARPS * 32;              // consumer threads
constexpr int THREADS = CT + 32;            // + the producer warp
constexpr int RX = 8;                       // rows per run (the m16 fragment's upper half is zero registers)
constexpr int MAX_SLOTS = 256;
#ifndef SMOE_PIPE
#define SMOE_PIPE 8
#endif
#ifndef SMOE_OBI
#define SMOE_OBI 2
#endif
#ifndef SMOE_RUNFLAGS
#define SMOE_RUNFLAGS 0
#endif
#ifndef SMOE_RUNFLAGS_SLEEP
#define SMOE_RUNFLAGS_SLEEP 256                     // ns between polls of a run's counter
#endif
#ifndef SMOE_L2HINT
#define SMOE_L2HINT 0                               // 128 / 256: weight copies ask L2 for 128- / 256-byte DRAM fetches
#endif
#ifndef SMOE_EVICT_FIRST
#define SMOE_EVICT_FIRST 1                          // weight copies carry an L2 evict-first policy (read once), so they
                                                    // do not push out what the L2 prefetch pulled in for the next dense
                                                    // kernels: +0.8-1.0% a serve round (Oct 11); 0 = off
#endif
#ifndef SMOE_L2_DISCARD
#define SMOE_L2_DISCARD 0                           // 1 (EXL3_L2_DISCARD=1, smoe.py): the per-slot outputs ys leave L2
                                                    // unwritten once a token's sum has read them (down_item)
#endif
constexpr int PIPE = SMOE_PIPE;             // k-rows in flight per block
constexpr int OBI = SMOE_OBI;               // 128-column output blocks per down item
constexpr int ROT_WARPS = 8;

// run table, an int32 scratch
constexpr int RT_RC = 0;                           // number of runs
constexpr int RT_P = 16;                           // [MAX_SLOTS] physical expert slot of each run
constexpr int RT_N = RT_P + MAX_SLOTS;             // [MAX_SLOTS] rows of each run, 1..8
constexpr int RT_SLOT = RT_N + MAX_SLOTS;          // [MAX_SLOTS][RX] routed slot (token * top_k + pick) per row
constexpr int RT_NV = RT_SLOT + MAX_SLOTS * RX;    // [MAX_SLOTS] valid picks per token
constexpr int RT_DONE = RT_NV + MAX_SLOTS;         // [MAX_SLOTS] gate / up blocks done per run (SMOE_RUNFLAGS)
constexpr int RT_LEN = RT_DONE + MAX_SLOTS;

struct Layer
{
    const unsigned char* w13;   // trellis bank: [w1 | w3] per physical slot
    const unsigned char* w2;
    const int64_t* off1;      // [E] byte offsets of w1 / w3 / w2 of each physical slot in its bank
    const int64_t* off3;
    const int64_t* off2;
    const int* kgu;             // [E] bits of gate / up and of down
    const int* kd;
    const half* gu_suh;         // [E][2][H]
    const half* gu_svh;         // [E][2][I]
    const half* d_suh;          // [E][I]
    const half* d_svh;          // [E][H]
    int E, H, I;
    // the shared expert as physical slot E (sh != 0): its own trellis tensors and scales, K bits each
    int sh, sh_kgu, sh_kd;
    const unsigned char* sh_w1;
    const unsigned char* sh_w3;
    const unsigned char* sh_w2;
    const half* sh_suh[2];      // [H] gate, up input scales
    const half* sh_svh[2];      // [I] gate, up output scales
    const half* sh_d_suh;       // [I]
    const half* sh_d_svh;       // [H]
};

// a physical slot's tensors: a routed expert's from the banks, the shared expert's (slot E) from its own tensors
__device__ __forceinline__ const unsigned char* w1_of(const Layer& L, int p) { return p < L.E ? L.w13 + L.off1[p] : L.sh_w1; }
__device__ __forceinline__ const unsigned char* w3_of(const Layer& L, int p) { return p < L.E ? L.w13 + L.off3[p] : L.sh_w3; }
__device__ __forceinline__ const unsigned char* w2_of(const Layer& L, int p) { return p < L.E ? L.w2 + L.off2[p] : L.sh_w2; }
__device__ __forceinline__ int kgu_of(const Layer& L, int p) { return p < L.E ? L.kgu[p] : L.sh_kgu; }
__device__ __forceinline__ int kd_of(const Layer& L, int p) { return p < L.E ? L.kd[p] : L.sh_kd; }
__device__ __forceinline__ const half* gu_suh_of(const Layer& L, int p, int g)
{
    return p < L.E ? L.gu_suh + ((size_t) p * 2 + g) * L.H : L.sh_suh[g];
}
__device__ __forceinline__ const half* gu_svh_of(const Layer& L, int p, int g)
{
    return p < L.E ? L.gu_svh + ((size_t) p * 2 + g) * L.I : L.sh_svh[g];
}
__device__ __forceinline__ const half* d_suh_of(const Layer& L, int p) { return p < L.E ? L.d_suh + (size_t) p * L.I : L.sh_d_suh; }
__device__ __forceinline__ const half* d_svh_of(const Layer& L, int p) { return p < L.E ? L.d_svh + (size_t) p * L.H : L.sh_d_svh; }

struct Call
{
    const __nv_bfloat16* x;
    int x_stride;
    const void* ids;
    int ids_i64, ids_stride;
    const float* w;
    int w_stride;
    const int64_t* perm;
    __nv_bfloat16* out;
    int out_stride;
    int m, top_k;
    float limit;
    half* hs;                   // [MAX_SLOTS][H / 16][2][16] rotated inputs
    half* act;                  // [MAX_SLOTS][I / 16][RX][16] rotated down inputs per run
    float* ys;                  // [MAX_SLOTS][H] weighted down outputs per slot
    int* rt;                    // run table
    int* ctr;                   // [MAX_SLOTS][H / 128] arrivals per (token, 128 block): zero, self-resetting
    const bool* pad;            // [m] or null: padding / dead rows (vLLM's is_padding) take no experts
    int shared;                 // 1: every live row also takes the shared expert, as pick top_k (weight 1)
    const int* ready;           // fused launch: nonzero once routing and the run table are written (else null)
};

__device__ __forceinline__ int picks(const Call& c) { return c.top_k + c.shared; }   // slots per token


__device__ __forceinline__ int64_t read_id(const Call& c, int t, int j)
{
    // vLLM's DSv4 top-k kernel ignores is_padding on the non-hash layers, so the rows it marks (CUDA-graph padding,
    // patch 0010's verify rows past the chosen depth) are dropped here: no run, no weight bytes, a zero output row
    if (c.pad && c.pad[t]) return -1;
    return c.ids_i64 ? __ldcg(reinterpret_cast<const int64_t*>(c.ids) + (size_t) t * c.ids_stride + j)
                     : (int64_t) __ldcg(reinterpret_cast<const int*>(c.ids) + (size_t) t * c.ids_stride + j);
}

// physical slot of pick j of token t, -1 for none
__device__ __forceinline__ int pick_slot(const Layer& L, const Call& c, int t, int j)
{
    if (j == c.top_k) return (c.pad && c.pad[t]) ? -1 : L.E;
    const int64_t id = read_id(c, t, j);
    return id >= 0 ? (int) c.perm[id] : -1;
}

__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;\n" :: "n"(CT) : "memory"); }

// run-table word: the fused launch's tables come from another block of the same launch, so it reads through L2
template <bool FUSED>
__device__ __forceinline__ int rtv(const Call& c, int i) { return FUSED ? __ldcg(c.rt + i) : c.rt[i]; }

__device__ __forceinline__ int ld_acquire_gpu(const int* p)
{
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint2 pack_h4(float a, float b, float c, float d)
{
    const half2 lo = __floats2half2_rn(a, b), hi = __floats2half2_rn(c, d);
    uint2 r;
    r.x = *reinterpret_cast<const uint32_t*>(&lo);
    r.y = *reinterpret_cast<const uint32_t*>(&hi);
    return r;
}

__device__ __forceinline__ void unpack_h4(uint2 v, float (&f)[4])
{
    const float2 a = __half22float2(*reinterpret_cast<const half2*>(&v.x));
    const float2 b = __half22float2(*reinterpret_cast<const half2*>(&v.y));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. rot + run table

__device__ void build_runs(const Layer& L, const Call& c, bool zero_done)
{
    __shared__ int ph[MAX_SLOTS], lead[MAX_SLOTS], rowi[MAX_SLOTS], cnt[MAX_SLOTS], start[MAX_SLOTS];
    const int kp = picks(c);
    const int n = c.m * kp;
    for (int s = threadIdx.x; s < n; s += blockDim.x) ph[s] = pick_slot(L, c, s / kp, s % kp);
    __syncthreads();
    for (int s = threadIdx.x; s < n; s += blockDim.x)
    {
        int ld = -1, ri = 0, ct = 0;
        if (ph[s] >= 0)
            for (int q = 0; q < n; ++q)
                if (ph[q] == ph[s]) { if (ld < 0) ld = q; if (q < s) ++ri; ++ct; }
        lead[s] = ld; rowi[s] = ri; cnt[s] = ct;
    }
    __syncthreads();
    for (int s = threadIdx.x; s < n; s += blockDim.x)          // runs of an expert start after every higher slot's
    {
        int st = 0;
        if (lead[s] == s)
            for (int q = 0; q < n; ++q)
                if (lead[q] == q && ph[q] > ph[s]) st += (cnt[q] + RX - 1) / RX;
        start[s] = st;
    }
    __syncthreads();
    int* rt = c.rt;
    for (int s = threadIdx.x; s < n; s += blockDim.x)
    {
        if (ph[s] < 0) continue;
        const int ld = lead[s];
        const int r = start[ld] + rowi[s] / RX;
        const int i = rowi[s] % RX;
        rt[RT_SLOT + r * RX + i] = s;
        if (i == 0)
        {
            rt[RT_P + r] = ph[s];
            rt[RT_N + r] = min(RX, cnt[ld] - (rowi[s] / RX) * RX);
        }
    }
    if (threadIdx.x == 0)
    {
        int rc = 0;
        for (int q = 0; q < n; ++q)
            if (lead[q] == q) rc += (cnt[q] + RX - 1) / RX;
        rt[RT_RC] = rc;
    }
    if (zero_done)
        for (int r = threadIdx.x; r < MAX_SLOTS; r += blockDim.x) rt[RT_DONE + r] = 0;
    for (int t = threadIdx.x; t < c.m; t += blockDim.x)
    {
        int nv = 0;
        for (int j = 0; j < kp; ++j) nv += ph[t * kp + j] >= 0;
        rt[RT_NV + t] = nv;
        if (nv == 0)                                            // no pick: the routed sum is zero
            for (int h = 0; h < L.H; h += 4)
                *reinterpret_cast<uint2*>(c.out + (size_t) t * c.out_stride + h) = make_uint2(0u, 0u);
    }
}

// one warp: slot s (physical expert p, -1 = none), projection proj, 128-wide block hb of x -> c.hs
__device__ __forceinline__ void rot_unit(const Layer& L, const Call& c, int s, int p, int proj, int hb, int lane)
{
    if (p < 0) return;
    const int t = s / picks(c);
    const int col = hb * 128 + lane * 4;
    const uint2 xv = *reinterpret_cast<const uint2*>(c.x + (size_t) t * c.x_stride + col);
    float hv[4];
    unpack_h4(*reinterpret_cast<const uint2*>(gu_suh_of(L, p, proj) + col), hv);
    const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.x));
    const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.y));
    float v[4] = {x01.x * hv[0], x01.y * hv[1], x23.x * hv[2], x23.y * hv[3]};
    had128(v, lane);
    const int kt = hb * 8 + lane / 4;
    half* dst = c.hs + (((size_t) s * (L.H / 16) + kt) * 2 + proj) * 16 + (lane % 4) * 4;
    *reinterpret_cast<uint2*>(dst) = pack_h4(v[0] * RSQRT128, v[1] * RSQRT128, v[2] * RSQRT128, v[3] * RSQRT128);
}

__global__ void __launch_bounds__(ROT_WARPS * 32) smoe_rot(const __grid_constant__ Layer L, const __grid_constant__ Call c)
{
    griddep_wait();                             // routing and x come from earlier kernels
    griddep_launch();
    if (blockIdx.x == 0) build_runs(L, c, SMOE_RUNFLAGS != 0);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int hbn = L.H / 128;
    const int kp = picks(c);
    const int tasks = c.m * kp * 2 * hbn;                       // (slot, projection, 128 block)
    for (int u = blockIdx.x * ROT_WARPS + warp; u < tasks; u += gridDim.x * ROT_WARPS)
    {
        const int s = u / (2 * hbn);
        rot_unit(L, c, s, pick_slot(L, c, s / kp, s % kp), (u / hbn) % 2, u % hbn, lane);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 16-byte cp.async of weight bytes with the build's L2 hints (SMOE_L2HINT, SMOE_EVICT_FIRST); inputs use cp_async16
__device__ __forceinline__ void cp_async16w(uint32_t dst, const void* src, uint64_t pol)
{
#if SMOE_EVICT_FIRST && SMOE_L2HINT == 256
    asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
#elif SMOE_EVICT_FIRST && SMOE_L2HINT == 128
    asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
#elif SMOE_EVICT_FIRST
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
#elif SMOE_L2HINT == 256
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
#elif SMOE_L2HINT == 128
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
#else
    cp_async16(dst, src);
#endif
}
__device__ __forceinline__ uint64_t weight_policy()
{
    uint64_t pol = 0;
#if SMOE_EVICT_FIRST
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
#endif
    return pol;
}

// 2. gate + up + activation

__host__ __device__ constexpr int gu_stage_bytes(int K) { return 2 * WARPS * 32 * K + RX * 64; }
__host__ __device__ constexpr int gu_smem_bytes(int K)
{
    return PIPE * gu_stage_bytes(K) > 2 * RX * 128 * 4 ? PIPE * gu_stage_bytes(K) : 2 * RX * 128 * 4;
}

template <int K, bool FUSED = false>
__device__ __forceinline__ void gu_item(const Layer& L, const Call& c, int rc, int p, int nr, int cb,
                                        unsigned char* ring, uint64_t* full, uint64_t* empty)
{
    constexpr int TB = 32 * K;                  // bytes per 16x16 tile
    constexpr int CH = WARPS * TB;              // one projection's 8 tiles of a k-row, contiguous
    constexpr int SB = gu_stage_bytes(K);       // [gate chunk][up chunk][rows x (gate 16 | up 16) fp16]
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int kts = L.H / 16;

    if (warp == WARPS)
    {
        // all 32 lanes copy 16-byte pieces of the stage: gate chunk, up chunk, then the run's input rows
        const size_t row_bytes = (size_t) (L.I / 16) * TB;
        const unsigned char* g1 = w1_of(L, p) + (size_t) cb * CH;
        const unsigned char* g3 = w3_of(L, p) + (size_t) cb * CH;
        const int my_row = lane / 4;                            // input-row pieces: 4 per row
        const int my_slot = my_row < nr ? rtv<FUSED>(c, RT_SLOT + rc * RX + my_row) : 0;
        const uint32_t ring_s = smem_u32(ring), full_s = smem_u32(full), empty_s = smem_u32(empty);
        const uint64_t pol = weight_policy();
        #pragma unroll 1
        for (int kk = 0; kk < kts; ++kk)
        {
            const int s = kk % PIPE;
            if (kk >= PIPE) mbar_wait_s(empty_s + s * 8, ((kk / PIPE) - 1) & 1);
            const uint32_t st = ring_s + s * SB;
            const unsigned char* a1 = g1 + kk * row_bytes;
            const unsigned char* a3 = g3 + kk * row_bytes;
            for (int i = lane; i < CH / 16; i += 32)
            {
                cp_async16w(st + i * 16, a1 + i * 16, pol);
                cp_async16w(st + CH + i * 16, a3 + i * 16, pol);
            }
            if (my_row < nr)
                cp_async16(st + 2 * CH + lane * 16, c.hs + ((size_t) my_slot * kts + kk) * 32 + (lane % 4) * 8);
            cp_async_arrive_noinc(full_s + s * 8);
        }
        return;
    }

    const int gq = lane / 4, tq = lane % 4;
    const LaneDq<K> cfg(lane);
    float g0[4] = {0.f, 0.f, 0.f, 0.f}, g1[4] = {0.f, 0.f, 0.f, 0.f};
    float u0[4] = {0.f, 0.f, 0.f, 0.f}, u1[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t full_s = smem_u32(full), empty_s = smem_u32(empty);
    const unsigned char* wt = ring + warp * TB;
    const unsigned char* xr = ring + 2 * CH + gq * 64;     // this lane's input row (rows >= nr: stale, unused)
    #pragma unroll 1
    for (int kk = 0; kk < kts; kk += 2)                    // two k-rows per step (kts even, PIPE even)
    {
        const int s0 = kk % PIPE;
        const uint32_t par = (kk / PIPE) & 1;
        mbar_wait_s(full_s + s0 * 8, par);
        mbar_wait_s(full_s + s0 * 8 + 8, par);
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const int off = (s0 + h) * SB;
            half2 fg0[2], fg1[2], fu0[2], fu1[2];
            decode_tile<K>(reinterpret_cast<const uint32_t*>(wt + off), cfg, fg0, fg1);
            decode_tile<K>(reinterpret_cast<const uint32_t*>(wt + off + CH), cfg, fu0, fu1);
            const uint32_t* xa = reinterpret_cast<const uint32_t*>(xr + off);
            const uint32_t ag[4] = {xa[tq], 0u, xa[tq + 4], 0u};
            const uint32_t au[4] = {xa[8 + tq], 0u, xa[12 + tq], 0u};
            mma16816(g0, ag, fg0);
            mma16816(g1, ag, fg1);
            mma16816(u0, au, fu0);
            mma16816(u1, au, fu1);
        }
        __syncwarp();
        if (lane == 0) { mbar_arrive_s(empty_s + s0 * 8); mbar_arrive_s(empty_s + s0 * 8 + 8); }
    }

    consumer_sync();                            // the ring is drained: reuse it for the partials
    float* syg = reinterpret_cast<float*>(ring);
    float* syu = syg + RX * 128;
    const int col = warp * 16 + 2 * tq;
    syg[gq * 128 + col] = g0[0]; syg[gq * 128 + col + 1] = g0[1];
    syg[gq * 128 + col + 8] = g1[0]; syg[gq * 128 + col + 9] = g1[1];
    syu[gq * 128 + col] = u0[0]; syu[gq * 128 + col + 1] = u0[1];
    syu[gq * 128 + col + 8] = u1[0]; syu[gq * 128 + col + 9] = u1[1];
    consumer_sync();

    if (warp < nr)
    {
        const int r = warp;
        float gv[4], uv[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) { gv[j] = syg[r * 128 + lane * 4 + j]; uv[j] = syu[r * 128 + lane * 4 + j]; }
        had128(gv, lane);
        had128(uv, lane);
        const int ic = cb * 128 + lane * 4;
        float sg[4], su[4], sd[4];
        unpack_h4(*reinterpret_cast<const uint2*>(gu_svh_of(L, p, 0) + ic), sg);
        unpack_h4(*reinterpret_cast<const uint2*>(gu_svh_of(L, p, 1) + ic), su);
        unpack_h4(*reinterpret_cast<const uint2*>(d_suh_of(L, p) + ic), sd);
        float a[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float g = gv[j] * RSQRT128 * sg[j];
            float u = uv[j] * RSQRT128 * su[j];
            if (c.limit != 0.f) { g = fminf(g, c.limit); u = fminf(fmaxf(u, -c.limit), c.limit); }
            a[j] = g / (1.f + __expf(-g)) * u * sd[j];
        }
        had128(a, lane);
        half* dst = c.act + (((size_t) rc * (L.I / 16) + cb * 8 + lane / 4) * RX + r) * 16 + (lane % 4) * 4;
        *reinterpret_cast<uint2*>(dst) = pack_h4(a[0] * RSQRT128, a[1] * RSQRT128, a[2] * RSQRT128, a[3] * RSQRT128);
    }
    if constexpr (FUSED || SMOE_RUNFLAGS)
    {
        __threadfence();                        // this block's act rows before its count on the run
        consumer_sync();
        if (threadIdx.x == 0) atomicAdd(c.rt + RT_DONE + rc, 1);
    }
}

__global__ void __launch_bounds__(THREADS, 1) smoe_gu(const __grid_constant__ Layer L, const __grid_constant__ Call c)
{
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ __align__(8) uint64_t full[PIPE];
    __shared__ __align__(8) uint64_t empty[PIPE];
    const int nI = L.I / 128;
    const int rc = blockIdx.x / nI, cb = blockIdx.x % nI;
    griddep_wait();                             // the run table and the rotated inputs come from rot
    griddep_launch();                           // (after the wait: down reads rot's run table before its own wait)
    if (rc >= c.rt[RT_RC]) return;
    const int p = c.rt[RT_P + rc];
    const int nr = c.rt[RT_N + rc];
    if (threadIdx.x == 0)
    {
        for (int s = 0; s < PIPE; ++s) { mbar_init(&full[s], 32); mbar_init(&empty[s], WARPS); }
        mbar_fence_init();
    }
    __syncthreads();
    switch (kgu_of(L, p))
    {
        case 2: gu_item<2>(L, c, rc, p, nr, cb, smem, full, empty); break;
        case 3: gu_item<3>(L, c, rc, p, nr, cb, smem, full, empty); break;
        case 4: gu_item<4>(L, c, rc, p, nr, cb, smem, full, empty); break;
        case 5: gu_item<5>(L, c, rc, p, nr, cb, smem, full, empty); break;
        case 6: gu_item<6>(L, c, rc, p, nr, cb, smem, full, empty); break;
        default: break;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. down + per-token sum

__host__ __device__ constexpr int dn_stage_bytes(int K) { return OBI * WARPS * 32 * K + RX * 32; }
__host__ __device__ constexpr int dn_smem_bytes(int K)
{
    return PIPE * dn_stage_bytes(K) > RX * OBI * 128 * 4 ? PIPE * dn_stage_bytes(K) : RX * OBI * 128 * 4;
}

template <int K, bool FUSED = false>
__device__ __forceinline__ void down_item(const Layer& L, const Call& c, int rc, int p, int nr, int ob0,
                                          unsigned char* ring, uint64_t* full, uint64_t* empty, int* last)
{
    constexpr int TB = 32 * K;
    constexpr int CH = OBI * WARPS * TB;        // OBI x 8 tiles of a k-row, contiguous
    constexpr int SB = dn_stage_bytes(K);       // [chunk][rows x 16 fp16]
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int kts = L.I / 16;

    if (warp == WARPS)
    {
        const size_t row_bytes = (size_t) (L.H / 16) * TB;
        const unsigned char* g2 = w2_of(L, p) + (size_t) ob0 * WARPS * TB;
        const unsigned char* a0 = reinterpret_cast<const unsigned char*>(c.act + (size_t) rc * kts * RX * 16);
        const uint32_t ring_s = smem_u32(ring), full_s = smem_u32(full), empty_s = smem_u32(empty);
        const uint64_t pol = weight_policy();
        const int pre = kts < PIPE ? kts : PIPE;
        for (int kk = 0; kk < pre; ++kk)            // static weights: stream them while gu finishes (PDL)
        {
            const uint32_t st = ring_s + kk * SB;
            const unsigned char* src = g2 + kk * row_bytes;
            for (int i = lane; i < CH / 16; i += 32) cp_async16w(st + i * 16, src + i * 16, pol);
        }
        if constexpr (FUSED || SMOE_RUNFLAGS)
        {
            while (ld_acquire_gpu(c.rt + RT_DONE + rc) < L.I / 128) __nanosleep(SMOE_RUNFLAGS_SLEEP);   // act rows in
            if constexpr (!FUSED) griddep_launch();
        }
        else
        {
            griddep_wait();                         // the down inputs come from gu
            griddep_launch();
        }
        for (int kk = 0; kk < pre; ++kk)
        {
            if (lane < nr * 2) cp_async16(ring_s + kk * SB + CH + lane * 16, a0 + (size_t) kk * RX * 32 + lane * 16);
            cp_async_arrive_noinc(full_s + kk * 8);
        }
        #pragma unroll 1
        for (int kk = pre; kk < kts; ++kk)
        {
            const int s = kk % PIPE;
            mbar_wait_s(empty_s + s * 8, ((kk / PIPE) - 1) & 1);
            const uint32_t st = ring_s + s * SB;
            const unsigned char* src = g2 + kk * row_bytes;
            for (int i = lane; i < CH / 16; i += 32) cp_async16w(st + i * 16, src + i * 16, pol);
            if (lane < nr * 2) cp_async16(st + CH + lane * 16, a0 + (size_t) kk * RX * 32 + lane * 16);
            cp_async_arrive_noinc(full_s + s * 8);
        }
        return;
    }
    if constexpr (FUSED || SMOE_RUNFLAGS)
    {
        // park until this run's gate / up is done (a sleeping poll, not the ring's spinning try_wait: down blocks
        // share SMs with gate / up blocks still running)
        if (lane == 0)
            while (ld_acquire_gpu(c.rt + RT_DONE + rc) < L.I / 128) __nanosleep(SMOE_RUNFLAGS_SLEEP);
        __syncwarp();
    }
    else
    {
        griddep_wait();                             // consumers write ys / out and count arrivals: after gu is done
        griddep_launch();
    }

    const int gq = lane / 4, tq = lane % 4;
    const LaneDq<K> cfg(lane);
    float acc[OBI][2][4];
    #pragma unroll
    for (int o = 0; o < OBI; ++o)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[o][h][j] = 0.f;
    const uint32_t full_s = smem_u32(full), empty_s = smem_u32(empty);
    const unsigned char* wt = ring + warp * TB;
    const unsigned char* xr = ring + CH + gq * 32;
    #pragma unroll 1
    for (int kk = 0; kk < kts; kk += 2)                    // two k-rows per step (kts even, PIPE even)
    {
        const int s0 = kk % PIPE;
        const uint32_t par = (kk / PIPE) & 1;
        mbar_wait_s(full_s + s0 * 8, par);
        mbar_wait_s(full_s + s0 * 8 + 8, par);
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const int off = (s0 + h) * SB;
            const uint32_t* xa = reinterpret_cast<const uint32_t*>(xr + off);
            const uint32_t a[4] = {xa[tq], 0u, xa[tq + 4], 0u};
            #pragma unroll
            for (int o = 0; o < OBI; ++o)
            {
                half2 f0[2], f1[2];
                decode_tile<K>(reinterpret_cast<const uint32_t*>(wt + off + o * WARPS * TB), cfg, f0, f1);
                mma16816(acc[o][0], a, f0);
                mma16816(acc[o][1], a, f1);
            }
        }
        __syncwarp();
        if (lane == 0) { mbar_arrive_s(empty_s + s0 * 8); mbar_arrive_s(empty_s + s0 * 8 + 8); }
    }

    consumer_sync();
    float* sy = reinterpret_cast<float*>(ring);  // [RX][OBI * 128]
    const int col = warp * 16 + 2 * tq;
    #pragma unroll
    for (int o = 0; o < OBI; ++o)
    {
        float* row = sy + gq * (OBI * 128) + o * 128;
        row[col] = acc[o][0][0]; row[col + 1] = acc[o][0][1];
        row[col + 8] = acc[o][1][0]; row[col + 9] = acc[o][1][1];
    }
    consumer_sync();

    for (int u = warp; u < nr * OBI; u += WARPS)
    {
        const int r = u / OBI, o = u % OBI;
        const int s = rtv<FUSED>(c, RT_SLOT + rc * RX + r);
        const int kp = picks(c), j = s % kp;
        // routed runs exist only once routing is in (the fused launch's shared runs take weight 1)
        const float wgt = j < c.top_k ? __ldcg(c.w + (size_t) (s / kp) * c.w_stride + j) : 1.f;   // the shared expert: 1
        float v[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = sy[r * (OBI * 128) + o * 128 + lane * 4 + j];
        had128(v, lane);
        const int hc = (ob0 + o) * 128 + lane * 4;
        float sv[4];
        unpack_h4(*reinterpret_cast<const uint2*>(d_svh_of(L, p) + hc), sv);
        *reinterpret_cast<float4*>(c.ys + (size_t) s * L.H + hc) =
            make_float4(v[0] * RSQRT128 * sv[0] * wgt, v[1] * RSQRT128 * sv[1] * wgt,
                        v[2] * RSQRT128 * sv[2] * wgt, v[3] * RSQRT128 * sv[3] * wgt);
    }
    __threadfence();
    consumer_sync();
    if constexpr (FUSED)                        // the arrival counts compare with RT_NV: the run table must be in
    {
        if (threadIdx.x == 0)
            while (ld_acquire_gpu(c.ready) == 0) __nanosleep(SMOE_RUNFLAGS_SLEEP);
        consumer_sync();
    }
    if (threadIdx.x < nr * OBI)                 // one arrival per (row, block); the last one of a token sums it
    {
        const int r = threadIdx.x / OBI, o = threadIdx.x % OBI;
        const int t = rtv<FUSED>(c, RT_SLOT + rc * RX + r) / picks(c);
        const int prev = atomicAdd(&c.ctr[t * (L.H / 128) + ob0 + o], 1);
        last[threadIdx.x] = prev == rtv<FUSED>(c, RT_NV + t) - 1;
    }
    consumer_sync();
    __threadfence();
    for (int u = warp; u < nr * OBI; u += WARPS)
    {
        if (!last[u]) continue;
        const int r = u / OBI, o = u % OBI;
        const int t = rtv<FUSED>(c, RT_SLOT + rc * RX + r) / picks(c);
        const int hc = (ob0 + o) * 128 + lane * 4;
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        const int kp = picks(c);
#if SMOE_L2_DISCARD
        unsigned live = 0;                       // the picks summed (the same for every lane)
#endif
        for (int j = 0; j < kp; ++j)             // pick order: the same sum whichever block arrives last
        {
            if (j < c.top_k ? read_id(c, t, j) < 0 : (c.pad && c.pad[t])) continue;
#if SMOE_L2_DISCARD
            live |= 1u << j;
#endif
            const float4 y = __ldcg(reinterpret_cast<const float4*>(c.ys + ((size_t) t * kp + j) * L.H + hc));
            sum.x += y.x; sum.y += y.y; sum.z += y.z; sum.w += y.w;
        }
        const __nv_bfloat162 lo = __floats2bfloat162_rn(sum.x, sum.y), hi = __floats2bfloat162_rn(sum.z, sum.w);
        uint2 ov;
        ov.x = *reinterpret_cast<const uint32_t*>(&lo);
        ov.y = *reinterpret_cast<const uint32_t*>(&hi);
        *reinterpret_cast<uint2*>(c.out + (size_t) t * c.out_stride + hc) = ov;
        if (lane == 0) c.ctr[t * (L.H / 128) + ob0 + o] = 0;
#if SMOE_L2_DISCARD
        // The token's per-slot outputs of this 128-column block (one 512-byte piece per summed pick, four 128-byte
        // lines: ys is 128-byte aligned and its rows are H floats, H a multiple of 128) are dead here, so their dirty L2
        // lines are dropped instead of written back to DRAM. Why nothing can read a dropped line:
        //  - writers: each pick's piece was written above by the down block of the run holding that slot, which ran
        //    __threadfence() before its arrival on this (token, block) counter; this warp's block took the last arrival
        //    and fenced, so every piece was in L2 before the __ldcg loads above;
        //  - readers: the sum is the only reader of ys, and a (token, block) is summed once, by this warp (last[u] is
        //    set for one arrival). Each lane's loads feed the bf16 store above, so they have returned before the lane
        //    reaches __syncwarp(), which orders all 32 lanes' loads before any lane's discard;
        //  - the next writer is the next call's down work, which writes only after that call's gate / up rows, and gate /
        //    up (rot before it, a fused launch at its start) begin with griddepcontrol.wait: after this grid has
        //    completed (the order the loads and the counter reset above already rely on). A call writes every piece it
        //    sums (all H columns of each live slot) before reading it, so a dropped line is never read.
        // Picks not summed (unrouted, padded rows) were not written by this call and are left alone.
        __syncwarp();
        const float* yb = c.ys + (size_t) t * kp * L.H + (ob0 + o) * 128;
        for (int i = lane; i < kp * 4; i += 32)
            if ((live >> (i >> 2)) & 1u) l2_discard(yb + (size_t) (i >> 2) * L.H + (i & 3) * 32);
#endif
    }
}

__global__ void __launch_bounds__(THREADS, 1) smoe_down(const __grid_constant__ Layer L, const __grid_constant__ Call c)
{
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ __align__(8) uint64_t full[PIPE];
    __shared__ __align__(8) uint64_t empty[PIPE];
    __shared__ int last[RX * OBI];
    const int nob = L.H / 128 / OBI;
    const int rc = blockIdx.x / nob, ob0 = (blockIdx.x % nob) * OBI;
    // read before griddep_wait (rot finished before any gu block released us), through L2: no stale L1 line
    if (rc >= __ldcg(c.rt + RT_RC)) return;
    const int p = __ldcg(c.rt + RT_P + rc);
    const int nr = __ldcg(c.rt + RT_N + rc);
    if (threadIdx.x == 0)
    {
        for (int s = 0; s < PIPE; ++s) { mbar_init(&full[s], 32); mbar_init(&empty[s], WARPS); }
        mbar_fence_init();
    }
    __syncthreads();
    switch (kd_of(L, p))
    {
        case 2: down_item<2>(L, c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 3: down_item<3>(L, c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 4: down_item<4>(L, c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 5: down_item<5>(L, c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 6: down_item<6>(L, c, rc, p, nr, ob0, smem, full, empty, last); break;
        default: break;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. fused: a decode layer's whole MoE phase in one persistent launch
//
// One block per resident slot (THREADS threads, as gu / down), each pulling tasks from one queue, in this order:
//   setup | shared rot | router rows | shared gate / up | shared down | routed rot || routed gate / up | routed down
// (the part after || is sized once routing is in). The shared expert depends only on x and the padding rows, so its
// ~11 MB stream from the first microsecond beside the router GEMV instead of after routing; setup writes the shared
// runs' table rows (physical slot E sorts first in build_runs, so they are runs 0 .. ns-1 whatever the routing). The
// block that finishes the last router task selects the top k and builds the run table, then sets the ready word.
// A task waits only on tasks earlier in the queue, and every block is resident and finishes each task before it takes
// the next, so the queue cannot deadlock (a single block runs it too). The arithmetic is exl3_router's router_topk
// followed by smoe(shared=1) step for step (router rows, selection, rotations, gate / up and down items in the same
// order), so the output is bit-identical to that pair of launches (tests/test_smoe_fused_gpu.py). The last block out
// zeroes the sync words and the run counters: the next launch, or a graph replay, starts clean.

#ifndef SMOE_FUSED_TRACE
#define SMOE_FUSED_TRACE 0                  // 1: record every task's claim / start / end time (tests/trace_smoe_fused.py)
#endif
#ifndef SMOE_FUSED_MINB
#define SMOE_FUSED_MINB 2                   // resident blocks an SM: 96 registers (3 caps at 72 and spills in the items)
#endif
#ifndef SMOE_FUSED_RPF
#define SMOE_FUSED_RPF 2                    // router rows: 16-byte loads in flight a lane (exl3_router: 4; the depth
#endif                                      // only prefetches, the sum order is the same, so the logits are too)
#ifndef SMOE_FUSED_SHARED_AFTER_ROUTER
#define SMOE_FUSED_SHARED_AFTER_ROUTER 0    // 1: the shared expert waits for the router rows (not for the selection),
#endif                                      // so it does not share DRAM with them; it still overlaps top-k and rot
constexpr int R_PF = SMOE_FUSED_RPF;
constexpr int R_MAXM = 8;                   // tokens a fused launch (exl3_router's MAXM)
constexpr int R_MAXN = 384;                 // experts the top-k handles, a multiple of 32 (exl3_router's MAXN)
constexpr int R_MAXK = 8;                   // top-k (exl3_router's MAXK)
constexpr int ROT_U = 4;                    // rotation units a warp takes per rot task
// sync words, 32 ints apart (one 128-byte line each): zero at launch, zeroed again by the last block out
constexpr int S_Q = 0, S_SETUP = 32, S_RDONE = 64, S_READY = 96, S_ROTSH = 128, S_ROTRT = 160, S_EXIT = 192;
constexpr int S_LEN = 224;

struct Fused
{
    const __nv_bfloat16* gw;    // router weight [n][H] bf16
    const float* bias;          // [n] correction bias
    const float* bias_vl;       // [n] or null: the bias of image-sentinel tokens
    const int64_t* tok;         // [m] token ids (with bias_vl)
    long long sentinel_lo;      // image sentinels: ids sentinel_lo .. sentinel_lo + 4
    float scale;                // routed_scaling_factor
    int n;                      // experts the router scores
    float* ws;                  // [2][m][n] fp32: scores, then weights
    int* ids;                   // [m][top_k] int32: the routing (Call.ids reads it)
    float* w;                   // [m][top_k] fp32 (Call.w reads it)
    int* sync;                  // [S_LEN]
    long long* trace;           // SMOE_FUSED_TRACE: [4096][4] (block << 32 | task, kind, t_claim, t_end) or null
};

__device__ __forceinline__ long long gtimer()
{
    long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint4 ld_stream(const uint4* p)
{
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ float dot8(const uint4& w, const uint4& x)
{
    const __nv_bfloat162* a = reinterpret_cast<const __nv_bfloat162*>(&w);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&x);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const float2 fa = __bfloat1622float2(a[i]);
        const float2 fb = __bfloat1622float2(b[i]);
        s = fmaf(fa.x, fb.x, s);
        s = fmaf(fa.y, fb.y, s);
    }
    return s;
}

__device__ __forceinline__ void st_release_gpu(int* p, int v)
{
    asm volatile("st.release.gpu.global.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void wait_ge(const int* p, int v)
{
    while (ld_acquire_gpu(p) < v) __nanosleep(SMOE_RUNFLAGS_SLEEP);
}

__device__ __forceinline__ void mbar_inval(uint64_t* bar)
{
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(smem_u32(bar)) : "memory");
}

// router rows e = 8 task .. 8 task + 7, one warp each, for M tokens: scores and weights into f.ws. exl3_router's
// router_kernel<M, true> step for step; x comes from global memory there via shared memory, here directly: the same
// values summed in the same order.
template <int M>
__device__ void router_rows(const Layer& L, const Call& c, const Fused& f, int task)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = task * WARPS + warp;
    if (warp >= WARPS || e >= f.n) return;
    const int K8 = L.H >> 3;
    const int nit = (K8 + 31) >> 5;
    const uint4* wr = reinterpret_cast<const uint4*>(f.gw) + (size_t) e * K8;
    uint4 pre[R_PF];
#pragma unroll
    for (int i = 0; i < R_PF; ++i)
    {
        const int j = i * 32 + lane;
        pre[i] = j < K8 ? ld_stream(wr + j) : make_uint4(0, 0, 0, 0);
    }
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int base = 0; base < nit; base += R_PF)
    {
        uint4 cur[R_PF];
#pragma unroll
        for (int i = 0; i < R_PF; ++i) cur[i] = pre[i];
#pragma unroll
        for (int i = 0; i < R_PF; ++i)
        {
            const int j = (base + R_PF + i) * 32 + lane;
            pre[i] = j < K8 ? ld_stream(wr + j) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < R_PF; ++i)
        {
            const int j = (base + i) * 32 + lane;
            if (j < K8)
            {
#pragma unroll
                for (int m = 0; m < M; ++m)
                    acc[m] += dot8(cur[i], __ldg(reinterpret_cast<const uint4*>(c.x + (size_t) m * c.x_stride) + j));
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
    }
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
        if (lane != m) continue;                    // lane m writes token m (every lane holds every token's sum)
        const float l = acc[m];
        bool img = false;
        if (f.bias_vl != nullptr)
        {
            const long long id = f.tok[m];
            img = id >= f.sentinel_lo && id < f.sentinel_lo + 5;
        }
        const float wt = sqrtf(l > 20.f ? l : logf(1.f + expf(l)));
        const float sc = wt + (img ? f.bias_vl[e] : f.bias[e]);
        f.ws[(size_t) m * f.n + e] = sc == sc ? sc : -1e30f;
        f.ws[(size_t) (M + m) * f.n + e] = wt;
        __threadfence();
    }
}

// the top k of every token from f.ws: exl3_router's last-block selection step for step (score descending, the lower
// expert on equal scores, the selected weights renormalized to sum 1 and scaled)
template <int M>
__device__ void select_topk(const Call& c, const Fused& f)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= WARPS) return;
    const int N = f.n, k = c.top_k;
    const int npl = N >> 5;                         // experts a lane: lane, lane + 32, ...
    for (int m = warp; m < M; m += WARPS)
    {
        float ls[R_MAXK];                           // this lane's best k: score descending, index ascending
        int li[R_MAXK];
#pragma unroll
        for (int s = 0; s < R_MAXK; ++s) { ls[s] = -INFINITY; li[s] = INT_MAX; }
#pragma unroll
        for (int q = 0; q < R_MAXN / 32; ++q)
        {
            if (q >= npl) break;
            float v = __ldcg(f.ws + (size_t) m * N + q * 32 + lane);
            int vi = q * 32 + lane;
#pragma unroll
            for (int s = 0; s < R_MAXK; ++s)
            {
                const bool better = v > ls[s];
                const float ts = ls[s];
                const int ti = li[s];
                ls[s] = better ? v : ls[s];
                li[s] = better ? vi : li[s];
                v = better ? ts : v;
                vi = better ? ti : vi;
            }
        }
        float selw[R_MAXK];
        int seli[R_MAXK];
#pragma unroll
        for (int s = 0; s < R_MAXK; ++s)
        {
            float bv = ls[0];
            int bi = li[0];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            seli[s] = bi;
            selw[s] = __ldcg(f.ws + (size_t) (M + m) * N + bi);
            if (li[0] == bi)                        // the owner pops its head
            {
#pragma unroll
                for (int r = 0; r < R_MAXK - 1; ++r) { ls[r] = ls[r + 1]; li[r] = li[r + 1]; }
                ls[R_MAXK - 1] = -INFINITY;
                li[R_MAXK - 1] = INT_MAX;
            }
        }
        if (lane == 0)
        {
            float sum = 0.f;
#pragma unroll
            for (int s = 0; s < R_MAXK; ++s) sum += s < k ? selw[s] : 0.f;
            const float fct = f.scale / (sum > 0.f ? sum : 1.f);
#pragma unroll
            for (int s = 0; s < R_MAXK; ++s)
            {
                if (s < k)
                {
                    f.w[m * k + s] = selw[s] * fct;
                    f.ids[m * k + s] = seli[s];
                }
            }
        }
    }
}

template <int M>
__device__ void router_task(const Layer& L, const Call& c, const Fused& f, int task, int n_rt, int* flag)
{
    router_rows<M>(L, c, f, task);
    __syncthreads();
    if (threadIdx.x == 0) *flag = atomicAdd(f.sync + S_RDONE, 1) == n_rt - 1;
    __syncthreads();
    if (!*flag) return;
    __threadfence();                                // the last router task: every row's scores are in
    select_topk<M>(c, f);
    __threadfence();
    __syncthreads();
    build_runs(L, c, false);                        // the shared runs may already count gate / up blocks: no zeroing
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) st_release_gpu(f.sync + S_READY, 1);
}

// The run counters zeroed (a three-launch smoe call with run flags may have left counts in them), then the shared
// runs' table rows from the padding rows alone: run r holds the live tokens 8r .. 8r + 7, in order, as build_runs
// writes them once routing is in (same values). Every gate / up task waits for the setup word, so no count is lost.
__device__ void setup_task(const Layer& L, const Call& c, const Fused& f)
{
    for (int r = threadIdx.x; r < MAX_SLOTS; r += blockDim.x) c.rt[RT_DONE + r] = 0;
    __threadfence();
    __syncthreads();
    if (threadIdx.x != 0) return;
    int nl = 0;
    if (c.shared)
    {
        const int kp = picks(c);
        for (int t = 0; t < c.m; ++t)
        {
            if (c.pad && c.pad[t]) continue;
            const int r = nl / RX, i = nl % RX;
            c.rt[RT_SLOT + r * RX + i] = t * kp + c.top_k;
            if (i == 0) c.rt[RT_P + r] = L.E;
            ++nl;
        }
        for (int r = 0; r * RX < nl; ++r) c.rt[RT_N + r] = min(RX, nl - r * RX);
    }
    __threadfence();
    st_release_gpu(f.sync + S_SETUP, 1 + (nl + RX - 1) / RX);   // 1 + the number of shared runs
}

// rotation units of one task (8 warps x ROT_U): shared (unit -> token, projection, block; slot t * kp + top_k) or
// routed (unit -> routed slot, projection, block)
template <bool SHARED>
__device__ void rot_task(const Layer& L, const Call& c, int task, int units)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= WARPS) return;
    const int hbn = L.H / 128, kp = picks(c);
    for (int q = 0; q < ROT_U; ++q)
    {
        const int u = (task * WARPS + warp) * ROT_U + q;
        if (u >= units) break;
        const int a = u / (2 * hbn), proj = (u / hbn) % 2, hb = u % hbn;
        if (SHARED)
            rot_unit(L, c, a * kp + c.top_k, (c.pad && c.pad[a]) ? -1 : L.E, proj, hb, lane);
        else
            rot_unit(L, c, (a / c.top_k) * kp + a % c.top_k, pick_slot(L, c, a / c.top_k, a % c.top_k), proj, hb, lane);
    }
}

// (re)initialize the block's stage barriers for the next gate / up or down item (no copy or arrival is pending: the
// previous item consumed every stage before the block-wide barrier at the top of the task loop)
__device__ void ring_init(uint64_t* full, uint64_t* empty, bool inited)
{
    if (threadIdx.x == 0)
    {
        for (int s = 0; s < PIPE; ++s)
        {
            if (inited) { mbar_inval(&full[s]); mbar_inval(&empty[s]); }
            mbar_init(&full[s], 32);
            mbar_init(&empty[s], WARPS);
        }
        mbar_fence_init();
    }
    __syncthreads();
}

__device__ void gu_task(const Layer& L, const Call& c, int rc, int cb, unsigned char* smem, uint64_t* full,
                        uint64_t* empty, bool inited)
{
    const int p = __ldcg(c.rt + RT_P + rc), nr = __ldcg(c.rt + RT_N + rc);
    ring_init(full, empty, inited);
    switch (kgu_of(L, p))
    {
        case 2: gu_item<2, true>(L, c, rc, p, nr, cb, smem, full, empty); break;
        case 3: gu_item<3, true>(L, c, rc, p, nr, cb, smem, full, empty); break;
        case 4: gu_item<4, true>(L, c, rc, p, nr, cb, smem, full, empty); break;
        case 5: gu_item<5, true>(L, c, rc, p, nr, cb, smem, full, empty); break;
        case 6: gu_item<6, true>(L, c, rc, p, nr, cb, smem, full, empty); break;
        default: break;
    }
}

__device__ void down_task(const Layer& L, const Call& c, int rc, int ob0, unsigned char* smem, uint64_t* full,
                          uint64_t* empty, int* last, bool inited)
{
    const int p = __ldcg(c.rt + RT_P + rc), nr = __ldcg(c.rt + RT_N + rc);
    ring_init(full, empty, inited);
    switch (kd_of(L, p))
    {
        case 2: down_item<2, true>(L, c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 3: down_item<3, true>(L, c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 4: down_item<4, true>(L, c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 5: down_item<5, true>(L, c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 6: down_item<6, true>(L, c, rc, p, nr, ob0, smem, full, empty, last); break;
        default: break;
    }
}

__global__ void __launch_bounds__(THREADS, SMOE_FUSED_MINB) smoe_fused(const __grid_constant__ Layer L, const __grid_constant__ Call c,
                                                         const __grid_constant__ Fused f)
{
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ __align__(8) uint64_t full[PIPE];
    __shared__ __align__(8) uint64_t empty[PIPE];
    __shared__ int last[RX * OBI];
    __shared__ int task_s, flag_s;
#if SMOE_FUSED_TRACE
    __shared__ long long t_ready_s;
#define TRACE_READY() do { if (threadIdx.x == 0) t_ready_s = gtimer(); } while (0)
#else
#define TRACE_READY() do { } while (0)
#endif
    griddep_wait();                                 // x and the padding rows come from earlier kernels
    griddep_launch();
    int* S = f.sync;
    const int m = c.m, kp = picks(c), hbn = L.H / 128, nI = L.I / 128, nob = L.H / 128 / OBI;
    const int ns_max = c.shared ? (m + RX - 1) / RX : 0;
    const int per = WARPS * ROT_U;
    const int n_rsh = c.shared ? (m * 2 * hbn + per - 1) / per : 0;
    const int n_rt = (f.n + WARPS - 1) / WARPS;
    const int n_rrt = (m * c.top_k * 2 * hbn + per - 1) / per;
    const int b_rsh = 1, b_rt = b_rsh + n_rsh, b_gush = b_rt + n_rt, b_dnsh = b_gush + ns_max * nI;
    const int b_rrt = b_dnsh + ns_max * nob, b_dyn = b_rrt + n_rrt;
    bool inited = false;
#if SMOE_FUSED_TRACE
    long long t_claim = 0;
    int prev_task = -1;
#endif
    for (;;)
    {
        __syncthreads();                            // the previous task is done with shared memory and task_s
#if SMOE_FUSED_TRACE
        if (threadIdx.x == 0 && prev_task >= 0 && prev_task < 4096 && f.trace)
        {
            long long* r = f.trace + (size_t) prev_task * 4;
            r[0] = ((long long) blockIdx.x << 32) | prev_task;
            r[1] = t_ready_s;
            r[2] = t_claim;
            r[3] = gtimer();
        }
#endif
        if (threadIdx.x == 0) task_s = atomicAdd(S + S_Q, 1);
        __syncthreads();
        const int i = task_s;
#if SMOE_FUSED_TRACE
        prev_task = i;
        t_claim = gtimer();
        if (threadIdx.x == 0) t_ready_s = t_claim;
#endif
        if (i == 0) { setup_task(L, c, f); continue; }
        if (i < b_rt)                               // shared rot
        {
            rot_task<true>(L, c, i - b_rsh, m * 2 * hbn);
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) atomicAdd(S + S_ROTSH, 1);
            continue;
        }
        if (i < b_gush)                             // router rows (the last one selects and builds the runs)
        {
            switch (m)
            {
                case 1: router_task<1>(L, c, f, i - b_rt, n_rt, &flag_s); break;
                case 2: router_task<2>(L, c, f, i - b_rt, n_rt, &flag_s); break;
                case 3: router_task<3>(L, c, f, i - b_rt, n_rt, &flag_s); break;
                case 4: router_task<4>(L, c, f, i - b_rt, n_rt, &flag_s); break;
                case 5: router_task<5>(L, c, f, i - b_rt, n_rt, &flag_s); break;
                case 6: router_task<6>(L, c, f, i - b_rt, n_rt, &flag_s); break;
                case 7: router_task<7>(L, c, f, i - b_rt, n_rt, &flag_s); break;
                default: router_task<8>(L, c, f, i - b_rt, n_rt, &flag_s); break;
            }
            continue;
        }
        if (i < b_dnsh)                             // shared gate / up: needs the shared runs and their rotations
        {
            if (threadIdx.x == 0)
            {
                wait_ge(S + S_SETUP, 1);
                wait_ge(S + S_ROTSH, n_rsh);
                if (SMOE_FUSED_SHARED_AFTER_ROUTER) wait_ge(S + S_RDONE, n_rt);
            }
            TRACE_READY();
            __syncthreads();
            const int rc = (i - b_gush) / nI;
            if (rc < __ldcg(S + S_SETUP) - 1) { gu_task(L, c, rc, (i - b_gush) % nI, smem, full, empty, inited); inited = true; }
            continue;
        }
        if (i < b_rrt)                              // shared down (waits for its run's gate / up inside the item)
        {
            if (threadIdx.x == 0) wait_ge(S + S_SETUP, 1);
            __syncthreads();
            const int rc = (i - b_dnsh) / nob;
            if (rc < __ldcg(S + S_SETUP) - 1)
            {
                down_task(L, c, rc, ((i - b_dnsh) % nob) * OBI, smem, full, empty, last, inited);
                inited = true;
            }
            continue;
        }
        if (threadIdx.x == 0) { wait_ge(S + S_SETUP, 1); wait_ge(S + S_READY, 1); }   // routing and the setup
        TRACE_READY();
        __syncthreads();
        if (i < b_dyn)                              // routed rot
        {
            rot_task<false>(L, c, i - b_rrt, m * c.top_k * 2 * hbn);
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) atomicAdd(S + S_ROTRT, 1);
            continue;
        }
        const int ns = __ldcg(S + S_SETUP) - 1, rc_all = __ldcg(c.rt + RT_RC);
        const int n_gu = (rc_all - ns) * nI, n_dn = (rc_all - ns) * nob;
        const int d = i - b_dyn;
        if (d < n_gu)                               // routed gate / up: needs every routed rotation
        {
            if (threadIdx.x == 0) wait_ge(S + S_ROTRT, n_rrt);
            TRACE_READY();
            __syncthreads();
            gu_task(L, c, ns + d / nI, d % nI, smem, full, empty, inited);
            inited = true;
            continue;
        }
        if (d < n_gu + n_dn)                        // routed down
        {
            const int e = d - n_gu;
            down_task(L, c, ns + e / nob, (e % nob) * OBI, smem, full, empty, last, inited);
            inited = true;
            continue;
        }
        break;                                      // the queue is empty
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence();
        flag_s = atomicAdd(S + S_EXIT, 1) == (int) gridDim.x - 1;
    }
    __syncthreads();
    if (flag_s)                                     // the last block out: a clean slate for the next launch
    {
        __threadfence();
        for (int k = threadIdx.x; k < S_LEN; k += blockDim.x) S[k] = 0;
        __threadfence();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// host

struct Handle
{
    Layer L;
    int k_max;
    std::vector<at::Tensor> keep;
};

std::vector<Handle>& registry()
{
    static std::vector<Handle> r;
    return r;
}

void set_smem(const void* fn, int bytes)
{
    C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

}  // namespace

int64_t smoe_register(at::Tensor w13, at::Tensor w2, at::Tensor off1, at::Tensor off3, at::Tensor off2,
                      at::Tensor kgu, at::Tensor kd, at::Tensor gu_suh, at::Tensor gu_svh, at::Tensor d_suh,
                      at::Tensor d_svh, int64_t H, int64_t I)
{
    const int E = (int) off1.numel();
    TORCH_CHECK(H % (128 * OBI) == 0 && I % 128 == 0, "exl3_smoe: H must be a multiple of ", 128 * OBI, ", I of 128");
    static_assert(PIPE % 2 == 0, "SMOE_PIPE must be even (two k-rows per consumer step)");
    TORCH_CHECK(w13.is_cuda() && w2.is_cuda() && w13.is_contiguous() && w2.is_contiguous(), "exl3_smoe: banks");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(w13.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(w2.data_ptr()) % 16 == 0, "exl3_smoe: banks not 16-byte aligned");
    for (const at::Tensor* t : {&off1, &off3, &off2})
        TORCH_CHECK(t->scalar_type() == at::kLong && t->numel() == E && t->is_contiguous() && t->is_cuda(), "offsets");
    for (const at::Tensor* t : {&kgu, &kd})
        TORCH_CHECK(t->scalar_type() == at::kInt && t->numel() == E && t->is_contiguous() && t->is_cuda(), "bits");
    TORCH_CHECK(gu_suh.scalar_type() == at::kHalf && gu_suh.numel() == (int64_t) E * 2 * H && gu_suh.is_contiguous(), "gu_suh");
    TORCH_CHECK(gu_svh.scalar_type() == at::kHalf && gu_svh.numel() == (int64_t) E * 2 * I && gu_svh.is_contiguous(), "gu_svh");
    TORCH_CHECK(d_suh.scalar_type() == at::kHalf && d_suh.numel() == (int64_t) E * I && d_suh.is_contiguous(), "d_suh");
    TORCH_CHECK(d_svh.scalar_type() == at::kHalf && d_svh.numel() == (int64_t) E * H && d_svh.is_contiguous(), "d_svh");
    const auto kc = at::cat({kgu, kd}).cpu();
    const int kmin = kc.min().item<int>(), kmax = kc.max().item<int>();
    TORCH_CHECK(kmin >= 2 && kmax <= 6, "exl3_smoe: K ", kmin, "..", kmax, " (2..6 supported)");
    Handle h;
    h.L.w13 = reinterpret_cast<const unsigned char*>(w13.data_ptr());
    h.L.w2 = reinterpret_cast<const unsigned char*>(w2.data_ptr());
    h.L.off1 = off1.data_ptr<int64_t>();
    h.L.off3 = off3.data_ptr<int64_t>();
    h.L.off2 = off2.data_ptr<int64_t>();
    h.L.kgu = kgu.data_ptr<int>();
    h.L.kd = kd.data_ptr<int>();
    h.L.gu_suh = reinterpret_cast<const half*>(gu_suh.data_ptr<at::Half>());
    h.L.gu_svh = reinterpret_cast<const half*>(gu_svh.data_ptr<at::Half>());
    h.L.d_suh = reinterpret_cast<const half*>(d_suh.data_ptr<at::Half>());
    h.L.d_svh = reinterpret_cast<const half*>(d_svh.data_ptr<at::Half>());
    h.L.E = E; h.L.H = (int) H; h.L.I = (int) I;
    h.L.sh = 0; h.L.sh_kgu = h.L.sh_kd = 0;
    h.L.sh_w1 = h.L.sh_w3 = h.L.sh_w2 = nullptr;
    h.L.sh_suh[0] = h.L.sh_suh[1] = h.L.sh_svh[0] = h.L.sh_svh[1] = h.L.sh_d_suh = h.L.sh_d_svh = nullptr;
    h.k_max = kmax;
    h.keep = {w13, w2, off1, off3, off2, kgu, kd, gu_suh, gu_svh, d_suh, d_svh};
    registry().push_back(h);
    return (int64_t) registry().size() - 1;
}

// The layer's shared expert as physical slot E: EXL3 trellis int16 w1 / w3 [H/16][I/16][16K] (this rank's columns),
// w2 [I/16][H/16][16K] (this rank's rows), fp16 scales; K from the trellis shapes (2..6, mul1 codebook as the banks).
void smoe_attach_shared(int64_t handle, at::Tensor w1, at::Tensor w3, at::Tensor w2, at::Tensor suh_g, at::Tensor suh_u,
                        at::Tensor svh_g, at::Tensor svh_u, at::Tensor d_suh, at::Tensor d_svh)
{
    Handle& h = registry().at(handle);
    const int H = h.L.H, I = h.L.I;
    for (const at::Tensor* t : {&w1, &w3, &w2})
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kShort && t->dim() == 3 &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "exl3_smoe shared: contiguous int16 trellis");
    TORCH_CHECK(w1.size(0) == H / 16 && w1.size(1) == I / 16 && w3.sizes() == w1.sizes(), "exl3_smoe shared: w1 / w3 [H/16][I/16][16K]");
    TORCH_CHECK(w2.size(0) == I / 16 && w2.size(1) == H / 16, "exl3_smoe shared: w2 [I/16][H/16][16K]");
    const int kgu = (int) (w1.size(2) / 16), kd = (int) (w2.size(2) / 16);
    TORCH_CHECK(w1.size(2) % 16 == 0 && w2.size(2) % 16 == 0 && kgu >= 2 && kgu <= 6 && kd >= 2 && kd <= 6,
                "exl3_smoe shared: K 2..6");
    for (const at::Tensor* t : {&suh_g, &suh_u, &d_svh})
        TORCH_CHECK(t->scalar_type() == at::kHalf && t->numel() == H && t->is_contiguous() && t->is_cuda(), "exl3_smoe shared: [H] scales");
    for (const at::Tensor* t : {&svh_g, &svh_u, &d_suh})
        TORCH_CHECK(t->scalar_type() == at::kHalf && t->numel() == I && t->is_contiguous() && t->is_cuda(), "exl3_smoe shared: [I] scales");
    h.L.sh_w1 = reinterpret_cast<const unsigned char*>(w1.data_ptr());
    h.L.sh_w3 = reinterpret_cast<const unsigned char*>(w3.data_ptr());
    h.L.sh_w2 = reinterpret_cast<const unsigned char*>(w2.data_ptr());
    h.L.sh_suh[0] = reinterpret_cast<const half*>(suh_g.data_ptr<at::Half>());
    h.L.sh_suh[1] = reinterpret_cast<const half*>(suh_u.data_ptr<at::Half>());
    h.L.sh_svh[0] = reinterpret_cast<const half*>(svh_g.data_ptr<at::Half>());
    h.L.sh_svh[1] = reinterpret_cast<const half*>(svh_u.data_ptr<at::Half>());
    h.L.sh_d_suh = reinterpret_cast<const half*>(d_suh.data_ptr<at::Half>());
    h.L.sh_d_svh = reinterpret_cast<const half*>(d_svh.data_ptr<at::Half>());
    h.L.sh_kgu = kgu;
    h.L.sh_kd = kd;
    h.L.sh = 1;
    h.k_max = std::max(h.k_max, std::max(kgu, kd));
    for (const at::Tensor* t : {&w1, &w3, &w2, &suh_g, &suh_u, &svh_g, &svh_u, &d_suh, &d_svh}) h.keep.push_back(*t);
}

std::vector<int64_t> smoe_sizes(int64_t H, int64_t I)
{
    // scratch element counts: hs (fp16), act (fp16), ys (fp32), rt (int32), ctr (int32)
    return {(int64_t) MAX_SLOTS * H * 2, (int64_t) MAX_SLOTS * I * RX, (int64_t) MAX_SLOTS * H, RT_LEN,
            (int64_t) MAX_SLOTS * (H / 128)};
}

void smoe_run(int64_t handle, at::Tensor x, at::Tensor ids, at::Tensor w, at::Tensor perm, at::Tensor out,
              double limit, at::Tensor hs, at::Tensor act, at::Tensor ys, at::Tensor rt, at::Tensor ctr,
              c10::optional<at::Tensor> pad, int64_t shared)
{
    const Handle& h = registry().at(handle);
    const int m = (int) x.size(0), top_k = (int) ids.size(1);
    TORCH_CHECK(!shared || h.L.sh, "exl3_smoe: shared=1 without attach_shared");
    const int kp = top_k + (shared ? 1 : 0);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "exl3_smoe: bf16 x and out");
    TORCH_CHECK(x.size(1) == h.L.H && out.size(1) == h.L.H && out.size(0) == m && x.stride(1) == 1 && out.stride(1) == 1,
                "exl3_smoe: x / out shape");
    TORCH_CHECK(x.stride(0) % 4 == 0 && out.stride(0) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
                "exl3_smoe: x / out rows must be 8-byte aligned");
    TORCH_CHECK(m >= 1 && (int64_t) m * kp <= MAX_SLOTS, "exl3_smoe: ", m, " x ", kp, " slots (max ", MAX_SLOTS, ")");
    TORCH_CHECK(ids.size(0) == m && ids.stride(1) == 1 && (ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong),
                "exl3_smoe: topk ids int32 / int64 [m, top_k]");
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.size(0) == m && w.size(1) == top_k && w.stride(1) == 1,
                "exl3_smoe: topk weights fp32 [m, top_k]");
    TORCH_CHECK(perm.scalar_type() == at::kLong && perm.numel() == h.L.E, "exl3_smoe: perm int64 [E]");
    Call c;
    c.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    c.x_stride = (int) x.stride(0);
    c.ids = ids.data_ptr();
    c.ids_i64 = ids.scalar_type() == at::kLong;
    c.ids_stride = (int) ids.stride(0);
    c.w = w.data_ptr<float>();
    c.w_stride = (int) w.stride(0);
    c.perm = perm.data_ptr<int64_t>();
    c.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    c.out_stride = (int) out.stride(0);
    c.m = m; c.top_k = top_k;
    c.shared = shared ? 1 : 0;
    c.limit = (float) limit;
    c.hs = reinterpret_cast<half*>(hs.data_ptr<at::Half>());
    c.act = reinterpret_cast<half*>(act.data_ptr<at::Half>());
    c.ys = ys.data_ptr<float>();
#if SMOE_L2_DISCARD
    TORCH_CHECK(reinterpret_cast<uintptr_t>(c.ys) % 128 == 0, "exl3_smoe: ys not 128-byte aligned (L2 discard)");
#endif
    c.rt = rt.data_ptr<int>();
    c.ctr = ctr.data_ptr<int>();
    c.pad = nullptr;
    if (pad.has_value() && pad->defined()) {
        TORCH_CHECK(pad->scalar_type() == at::kBool && pad->numel() >= m && pad->is_contiguous(),
                    "exl3_smoe: pad must be a contiguous bool tensor of at least m rows");
        c.pad = pad->data_ptr<bool>();
    }

    const c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int slots = m * kp;
    const int rot_tasks = slots * 2 * (h.L.H / 128);
    const int rot_blocks = std::max(1, std::min((rot_tasks + ROT_WARPS - 1) / ROT_WARPS, 1024));
    launch_k(smoe_rot, dim3(rot_blocks), dim3(ROT_WARPS * 32), 0, stream, h.L, c);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    static int gu_set = 0, dn_set = 0;
    const int gs = gu_smem_bytes(h.k_max), ds = dn_smem_bytes(h.k_max);
    if (gs > gu_set) { set_smem((const void*) smoe_gu, gs); gu_set = gs; }
    if (ds > dn_set) { set_smem((const void*) smoe_down, ds); dn_set = ds; }
    launch_k(smoe_gu, dim3(slots * (h.L.I / 128)), dim3(THREADS), gs, stream, h.L, c);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    launch_k(smoe_down, dim3(slots * (h.L.H / 128 / OBI)), dim3(THREADS), ds, stream, h.L, c);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

long long* g_fused_trace = nullptr;              // set_fused_trace: SMOE_FUSED_TRACE builds write here

// The whole MoE phase of a decode step in one launch: router GEMV (gate_w bf16 [n, H]) + DeepSeek-V4 top-k
// (sqrt-softplus + bias, bias_vl for image-sentinel tokens when given, renormalized x scale) + routed experts + the
// shared expert (with shared=1, after attach_shared). Bit-identical to exl3_router.router_topk then run(..., shared).
// ws fp32 >= 2 * m * n, ids int32 / w fp32 >= m * top_k (the routing, readable after the call), sync int32 >= S_LEN
// zeroed once by the caller (the launch leaves it zero). 1 <= m <= 8.
void smoe_run_fused(int64_t handle, at::Tensor x, at::Tensor gate_w, at::Tensor bias, at::Tensor bias_vl, at::Tensor tok,
                    int64_t sentinel_lo, double scale, int64_t top_k, at::Tensor perm, at::Tensor out, double limit,
                    at::Tensor hs, at::Tensor act, at::Tensor ys, at::Tensor rt, at::Tensor ctr,
                    c10::optional<at::Tensor> pad, int64_t shared, at::Tensor ws, at::Tensor ids, at::Tensor w,
                    at::Tensor sync)
{
    const Handle& h = registry().at(handle);
    const int m = (int) x.size(0), n = (int) gate_w.size(0), k = (int) top_k;
    TORCH_CHECK(!shared || h.L.sh, "exl3_smoe fused: shared=1 without attach_shared");
    const int kp = k + (shared ? 1 : 0);
    TORCH_CHECK(m >= 1 && m <= R_MAXM, "exl3_smoe fused: 1..", R_MAXM, " tokens");
    TORCH_CHECK(k >= 1 && k <= R_MAXK && (int64_t) m * kp <= MAX_SLOTS, "exl3_smoe fused: top_k 1..", R_MAXK);
    TORCH_CHECK(n % 32 == 0 && n <= R_MAXN && n <= h.L.E, "exl3_smoe fused: router experts a multiple of 32, <= ", R_MAXN,
                " and <= the bank's ", h.L.E);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "exl3_smoe fused: bf16 x and out");
    TORCH_CHECK(x.size(1) == h.L.H && out.size(1) == h.L.H && out.size(0) == m && x.stride(1) == 1 && out.stride(1) == 1,
                "exl3_smoe fused: x / out shape");
    TORCH_CHECK(x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && out.stride(0) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0, "exl3_smoe fused: 16-byte aligned x rows, 8-byte out");
    TORCH_CHECK(gate_w.scalar_type() == at::kBFloat16 && gate_w.is_contiguous() && gate_w.size(1) == h.L.H &&
                reinterpret_cast<uintptr_t>(gate_w.data_ptr()) % 16 == 0, "exl3_smoe fused: gate weight bf16 [n, H]");
    TORCH_CHECK(bias.scalar_type() == at::kFloat && bias.is_contiguous() && bias.numel() == n, "exl3_smoe fused: fp32 bias [n]");
    const bool vl = bias_vl.numel() > 0;
    if (vl)
    {
        TORCH_CHECK(bias_vl.scalar_type() == at::kFloat && bias_vl.is_contiguous() && bias_vl.numel() == n,
                    "exl3_smoe fused: fp32 bias_vl [n]");
        TORCH_CHECK(tok.scalar_type() == at::kLong && tok.is_contiguous() && tok.numel() >= m, "exl3_smoe fused: int64 ids [m]");
    }
    TORCH_CHECK(perm.scalar_type() == at::kLong && perm.numel() == h.L.E, "exl3_smoe fused: perm int64 [E]");
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= 2 * m * n, "exl3_smoe fused: fp32 ws [2, m, n]");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.numel() >= m * k && w.scalar_type() == at::kFloat && w.numel() >= m * k,
                "exl3_smoe fused: int32 ids / fp32 w [m, top_k]");
    TORCH_CHECK(sync.scalar_type() == at::kInt && sync.numel() >= S_LEN, "exl3_smoe fused: int32 sync [", S_LEN, "]");
    Call c;
    c.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    c.x_stride = (int) x.stride(0);
    c.ids = ids.data_ptr();
    c.ids_i64 = 0;
    c.ids_stride = k;
    c.w = w.data_ptr<float>();
    c.w_stride = k;
    c.perm = perm.data_ptr<int64_t>();
    c.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    c.out_stride = (int) out.stride(0);
    c.m = m; c.top_k = k;
    c.shared = shared ? 1 : 0;
    c.limit = (float) limit;
    c.hs = reinterpret_cast<half*>(hs.data_ptr<at::Half>());
    c.act = reinterpret_cast<half*>(act.data_ptr<at::Half>());
    c.ys = ys.data_ptr<float>();
#if SMOE_L2_DISCARD
    TORCH_CHECK(reinterpret_cast<uintptr_t>(c.ys) % 128 == 0, "exl3_smoe: ys not 128-byte aligned (L2 discard)");
#endif
    c.rt = rt.data_ptr<int>();
    c.ctr = ctr.data_ptr<int>();
    c.pad = nullptr;
    if (pad.has_value() && pad->defined())
    {
        TORCH_CHECK(pad->scalar_type() == at::kBool && pad->numel() >= m && pad->is_contiguous(),
                    "exl3_smoe fused: pad must be a contiguous bool tensor of at least m rows");
        c.pad = pad->data_ptr<bool>();
    }
    c.ready = sync.data_ptr<int>() + S_READY;
    Fused f;
    f.gw = reinterpret_cast<const __nv_bfloat16*>(gate_w.data_ptr<at::BFloat16>());
    f.bias = bias.data_ptr<float>();
    f.bias_vl = vl ? bias_vl.data_ptr<float>() : nullptr;
    f.tok = vl ? tok.data_ptr<int64_t>() : nullptr;
    f.sentinel_lo = (long long) sentinel_lo;
    f.scale = (float) scale;
    f.n = n;
    f.ws = ws.data_ptr<float>();
    f.ids = ids.data_ptr<int>();
    f.w = w.data_ptr<float>();
    f.sync = sync.data_ptr<int>();
    f.trace = g_fused_trace;

    const c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int smem = std::max(gu_smem_bytes(h.k_max), dn_smem_bytes(h.k_max));
    static int fs_set = 0, blocks = 0, blocks_smem = -1;
    if (smem > fs_set) { set_smem((const void*) smoe_fused, smem); fs_set = smem; }
    if (blocks_smem != smem)                        // persistent: exactly as many blocks as can be resident at once
    {
        int dev = 0, sms = 0, per_sm = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, smoe_fused, THREADS, smem));
        TORCH_CHECK(per_sm >= 1, "exl3_smoe fused: no block fits an SM");
        const char* cap = std::getenv("EXL3_SMOE_FUSED_BLOCKS");            // tuning: fewer blocks than resident
        blocks = sms * per_sm;
        if (cap && std::atoi(cap) > 0) blocks = std::min(blocks, std::atoi(cap));
        blocks_smem = smem;
    }
    launch_k(smoe_fused, dim3(blocks), dim3(THREADS), smem, stream, h.L, c, f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void smoe_set_fused_trace(at::Tensor t)
{
    TORCH_CHECK(!t.defined() || t.numel() == 0 || (t.scalar_type() == at::kLong && t.numel() >= 4096 * 4 && t.is_cuda()),
                "exl3_smoe: trace int64 [4096 * 4] on the GPU (empty to stop)");
    g_fused_trace = (t.defined() && t.numel()) ? reinterpret_cast<long long*>(t.data_ptr<int64_t>()) : nullptr;
}

int64_t smoe_fused_blocks(int64_t handle)
{
    const Handle& h = registry().at(handle);
    const int smem = std::max(gu_smem_bytes(h.k_max), dn_smem_bytes(h.k_max));
    int dev = 0, sms = 0, per_sm = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaFuncSetAttribute((const void*) smoe_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, smoe_fused, THREADS, smem));
    return (int64_t) sms * per_sm;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod)
{
    mod.def("register", &smoe_register, "register one layer's expert banks and tables; returns a handle");
    mod.def("sizes", &smoe_sizes, "scratch element counts for H, I: hs, act, ys, rt, ctr");
    mod.def("run", &smoe_run, "routed experts of one layer: x bf16 [m, H] -> out bf16 [m, H]");
    mod.def("attach_shared", &smoe_attach_shared, "the layer's shared expert as physical slot E (run with shared=1)");
    mod.def("run_fused", &smoe_run_fused, "router + top-k + routed (+ shared) experts of one decode step in one launch");
    mod.def("fused_blocks", &smoe_fused_blocks, "resident blocks of the fused launch (its grid)");
    mod.def("sync_words", []() { return (int64_t) S_LEN; }, "int32 sync words the fused launch needs");
    mod.def("set_fused_trace", &smoe_set_fused_trace, "SMOE_FUSED_TRACE builds: per-task times into this int64 tensor");
}
