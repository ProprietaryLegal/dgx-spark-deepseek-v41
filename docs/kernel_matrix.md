# Kernel matrix: DeepSeek-V4.1-Flash on GB10 (sm_121), upstream vLLM main 6e517b15 + our patches

Living document (P2 / P5). What each op runs on GB10 in our image, how we know, and what it costs. Sources: G0 boots of
the truncated official model (1 / 2 / 3 layers, Oct 8, `runs/g0/*.log` on spark2), vLLM source at the pin, kernel
probes. "runs" = executes and gives sane numbers; it says nothing yet about speed unless a number is given.

| op | kernel on GB10 | status | evidence | cost / note |
|---|---|---|---|---|
| routed experts, native MXFP4 (G0 only; our serve uses EXL3) | `DEEPGEMM_MXFP4` / `DeepGemmFP4Experts` | runs | G0 log `mxfp4.py:746` | the drafter keeps this path |
| routed experts, our EXL3 | exllamav3 1.6.0 `exl3_moe_coop` (decode), `exl3_moe` + reconstruct (prefill) via `vllm_exl3_moe` | runs, 15/15 kernel tests | `patches/exl3_moe/tests/test_kernels_gpu.py`, bridge test on a real expert (rel 2e-3 vs dense) | graph m=1: 129-147 GB/s (47-54% of 273), m=16: 179-192 GB/s (`data/decode_byte_budget.md`) |
| MXFP8 linears (fused wq_a+wkv, wq_b, wo_b, shared expert, indexer wq_b, ...) | `FlashInferCutlassMxfp8LinearKernel` (vLLM's default pick on SM12x) | runs, **slow at decode: 26-33 GB/s** | G0 log `__init__.py:929`; `scripts/bench/linear_decode_bench.py` (Oct 8 10:25, spark2, GPU shared with pass 1) | per rank at m=1: wq_b 737 us, wo_b 723 us, shared gate/up 376 us, shared down 222 us: ~2 ms per layer, **~80 ms per decode step** over 40 layers. **Marlin MXFP8 (W8A16), the next candidate in vLLM's list, does the same at m=1 in 57 / 44 / 30 / 26 us (12-16x faster)** and stays flat to m=8. Switch: `VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel` (prefill at large M to re-check) |
| **wo_a (grouped bmm over 8 head groups)** | stock: **`EmulationMxfp8LinearKernel`**; ours: `Sm12xMarlinMxfp8BmmLinearKernel` (patch 0004, 8/8 tests): DeepGEMM's MXFP8 bmm is datacenter-only, emulation dequantizes FP8 -> BF16 **once at load** (`VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD`) and runs a BF16 bmm | runs | G0 log; `kernels/linear/__init__.py:895-897`, `kernels/linear/mxfp8/emulation.py` | **2x bytes and 2x memory**, and cuBLAS BF16 at m=1 is slow on GB10 (one group 137 us at m=1 vs 43 us at m=6). Measured per group (4 per rank): BF16 137 us, CUTLASS MXFP8 151 us, **Marlin MXFP8 26 us**. Fix = patch 0004 rewritten to run Marlin per group (the first 0004 draft looped CUTLASS: slower than what it replaced) |
| sparse MLA attention (SWA + compressed KV + indexer top-k) | FlashInfer 0.7.0.post1 SM120 sparse MLA (DSv4 decode + prefill), fp8_ds_mla KV | runs **with patch 0001** (64-token SWA pages; FlashInfer 0.7.0 instantiates SM120 decode for 64-token pages only) **and patch 0003** (compressor ring kept out of the null block, else NaN from layer 0) | G0 1 / 2 / 3 layers | FlashInfer >= 0.7.1 (#5197) makes page sizes runtime; not in our image |
| indexer top-k | not DeepSelect (`vllm._deepselect_C` is built for sm_100 only, upstream too) | runs (fallback path) | G0 log "Failed to import the DeepSelect extension" warning, generation fine | which fallback (cooperative / persistent / flashinfer top-k) and its cost: to measure |
| mHC (hyper-connection mixes, Sinkhorn, pre / post) | vLLM's TileLang kernels `mhc_fused_tilelang`, `mhc_pre_big_fuse_with_norm_tilelang` | runs | G0 log (TileLang 0.1.12 JIT on sm_121) | |
| Engram lookup | our `engram_disk` (patch 0002): pread pool -> pinned staging -> upstream `_engram_lookup_kernel` | runs, bit-exact | `patches/tests/test_engram_disk_vllm.py` 7/7 in the image; harness stats identical to a memmap reference | decode cost at TP=2: to measure in G0 |
| vision tower attention | `_vllm_fa2_C` (sm_80 cubins + sm_80 PTX, driver-JIT on sm_121) | not exercised yet | cuobjdump of our wheel | |
| DSpark drafter (3 MTP layers, 128 experts top-3, MXFP4) | DeepGEMM MXFP4 experts + SWA attention | not exercised yet | | |
| lm_head (BF16, vocab-parallel, shared with the DSpark drafter) | plain BF16 GEMM (UnquantizedEmbeddingMethod); opt-in **MXFP8 Marlin** via patch 0005 + the plugin's `Mxfp8MarlinHeadMethod` | BF16 runs; MXFP8 tested 6/6 (`patches/exl3_moe/tests/test_head_mxfp8.py`) | | 0.66 GB per forward per rank, read by the target and the drafter each step: **2,831-2,897 us -> 1,481-1,495 us per call** at 1-8 tokens (spark1, GPU shared). On only after the panel KLD of the MXFP8 head (`p4_kld.py --head-mxfp8`) |

## Decode linear costs per rank (TP=2), us per call, CUDA-graph replay (spark2, Oct 8 10:25, GPU shared with pass 1)

| shape (in, out) | m | FP8 CUTLASS | FP8 Marlin | BF16 F.linear | EXL3 K3 | EXL3 K4 | EXL3 K5 |
|---|---|---|---|---|---|---|---|
| wq_b (1280, 16384) | 1 | 737 | 57.5 | 484 | 42.3 | 39.3 | 56.8 |
| wq_b | 6 | 715 | 57.5 | 403 | 73.8 | 134.7 | 72.8 |
| wo_b (4096, 5120) | 1 | 723 | 44.1 | 559 | 45.0 | 42.7 | 58.9 |
| wo_b | 6 | 708 | 54.4 | 449 | 73.3 | 73.7 | 85.0 |
| wo_a group (4096, 1024) | 1 | 151 | 26.1 | 137 | 27.3 | 26.4 | 27.2 |
| wo_a group | 6 | 149 | 26.9 | 43.4 | 42.4 | 40.5 | 42.9 |
| shared gate+up (5120, 2304) | 1 | 376 | 29.8 | 245 | 35.3 | 36.8 | 38.6 |
| shared gate+up | 6 | 357 | 35.6 | 65.3 | 58.5 | 28.0 | 33.6 |
| shared down (1152, 5120) | 1 | 222 | 26.3 | 136 | 26.2 | 25.3 | 27.4 |
| shared down | 6 | 211 | 26.7 | 29.0 | 42.9 | 44.6 | 41.6 |

Marlin figures above 273 GB/s include L2 reuse (the same weight replayed); the ranking is the point. EXL3 at K4 beats Marlin FP8 only at m=1 and loses at the m=6-8 sizes of a DSpark verify step, so quantizing the attention projections to EXL3 (the "v2" lever) is not worth it once Marlin is selected.

## Decode and prefill sizes on an idle GPU (spark1, Oct 8 10:28, `linear_decode_bench.py 1,6,8,64,512,2048`), us per call

The table above was measured with pass 1 sharing the GPU: every kernel is ~2x faster idle, the ranking holds. Marlin
(W8A16) leads up to m=64; CUTLASS MXFP8 (W8A8, FP8 tensor cores) leads from m=512, where BF16 F.linear matches it.

| shape | m | FP8 CUTLASS | FP8 Marlin | BF16 |
|---|---|---|---|---|
| wq_b (1280, 16384) | 1 / 8 / 64 | 316.5 / 299.8 / 190.8 | **29.9 / 30.9 / 62.3** | 227.8 / 177.8 / 190.1 |
| wq_b | 512 / 2048 | **241.3 / 949.3** | 448.3 / 1,616.1 | 303.0 / 1,018.7 |
| wo_b (4096, 5120) | 1 / 8 / 64 | 312.7 / 298.5 / 72.1 | **31.2 / 29.1 / 50.7** | 238.1 / 178.5 / 188.2 |
| wo_b | 512 / 2048 | **250.4 / 1,009.9** | 375.4 / 1,447.7 | 307.5 / 988.1 |
| wo_a group (4096, 1024) | 1 / 8 / 64 | 67.6 / 63.8 / 20.4 | **9.6 / 10.2 / 19.6** | 13.0 / 19.0 / 17.5 |
| wo_a group | 512 / 2048 | **54.6** / 284.8 | 70.7 / 270.5 | 60.2 / **192.9** |
| shared gate+up (5120, 2304) | 1 / 8 / 64 | 164.0 / 161.0 / 41.4 | **13.3 / 14.3 / 32.1** | 50.7 / 56.9 / 72.8 |
| shared gate+up | 512 / 2048 | **142.5** / 647.1 | 178.5 / 729.5 | 143.0 / **614.1** |
| shared down (1152, 5120) | 1 / 8 / 64 | 95.9 / 81.1 / 58.5 | **10.6 / 10.8 / 19.5** | 16.4 / 12.7 / 16.4 |
| shared down | 512 / 2048 | **68.1 / 287.9** | 107.1 / 411.4 | 76.5 / 322.0 |

All-Marlin costs ~1.3 ms per layer per 2,048-token prefill chunk vs CUTLASS (~52 ms over 40 layers, ~5% of a chunk at
~2K tok/s). A size-dispatch kernel needs both weight layouts in memory (+1.7 GiB per rank for wq_b + wo_b alone).

## In the model (dummy-weight TP=2 boot, Oct 8 ~11:25, torch profiler, 25 decode steps, verify m=6)

GPU busy 52.0 ms per step, wall 60.7: EXL3 routed experts 20.9 ms, **Marlin MXFP8 dense linears 17.7 ms** (~4.3 GB per
rank per step: backbone wq_b / wo_b / wo_a / shared expert / wq_a / wkv / indexer, the drafter's, the MXFP8 head; ~240
GB/s, 88% of peak), NCCL all-reduce 4.3 ms (81 x 52 us), BF16 GEMMs 2.7, sparse MLA 2.0, mHC 1.7, other 2.4.

## DRAM-bound linears (`linear_dram_bench.py`, distinct weights cycled so nothing stays in L2; spark1 with pass 1 sharing the GPU, so both kernels run at ~90 GB/s: read the ratios)

| shape (per rank) | m | Marlin MXFP8 | EXL3 K4 | EXL3 K5 | EXL3 K6 |
|---|---|---|---|---|---|
| wq_b (1280, 16384) | 1 / 6 / 8 | 233 / 240 / 238 | 114 / 122 / 132 | 173 / 142 / 148 | 165 / 167 / 149 |
| wo_b (4096, 5120) | 1 / 6 / 8 | 208 / 218 / 218 | 105 / 121 / 128 | 169 / 146 / 149 | 162 / 175 / 173 |
| wo_a group (4096, 1024) | 1 / 6 / 8 | 49 / 51 / 51 | 37 / 42 / 41 | 42 / 48 / 47 | 43 / 40 / 50 |
| shared gate+up (5120, 2304) | 1 / 6 / 8 | 123 / 123 / 56 | 71 / 81 / 74 | 92 / 93 / 93 | 96 / 106 / 106 |
| shared down (1152, 5120) | 1 / 6 / 8 | 35 / 36 / 65 | 22 / 34 / 54 | 28 / 36 / 55 | 28 / 59 / 61 |

Both kernels are bandwidth-bound in the model, so time follows bytes: MXFP8 is 8.25 bits per weight, EXL3 K4 / K5 / K6 are
4 / 5 / 6 (~1.9x / ~1.5x / ~1.35x faster on the big shapes). EXL3's error vs the FP8 weights (per-K figures from the
experts): K4 ~2.4%, K5 ~1.0%, K6 ~0.4% relative output error. Lever: EXL3 dense linears (plugin linear method + harness
pass for their Hessians), measured on the KLD panel before use.

## Reference implementation (DeepSeek's inference/kernel.py, used by the P4 harness) on GB10

All run in our image (TileLang 0.1.12 + apache-tvm-ffi 0.1.11) with correct numerics (`scripts/quantize/tilelang_gb10_probe.py`):
act_quant, fp8_gemm, fp4_gemm (MXFP4 packing low nibble first), fp4_act_quant, hc_split_sinkhorn. `sparse_attn` needs
141,312 B of shared memory at 64 heads x 512; GB10 allows ~99 KB, so the harness calls it per 16-head group (exact).
With TileLang 0.1.8 the pin matters: apache-tvm-ffi 0.1.4-0.1.6 works, 0.1.2 / 0.1.3 give wrong results, 0.1.11 does not
compile.

## Still to fill (G0 at TP=2 with `LOAD_FORMAT=dummy`, then the real weights)

- per-op time at decode m = 1, 6, 8 (nsys): attention, the MXFP8 linears, wo_a emulation, EXL3 experts, all-reduces over
  RoCE (80 per step), Engram, lm_head;
- memory per rank after load (T3 needs >= 4 GiB MemAvailable);
- CUDA graph capture (FULL_DECODE_ONLY) with the EXL3 plugin and DSpark.
