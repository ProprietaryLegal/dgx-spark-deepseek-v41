# EXL3 MoE quantization method for upstream vLLM: design (DeepSeek-V4.1-Flash, 2x GB10, TP=2)

> Pre-build record. The shipped plugin grew past this design: decode runs on our own kernels (`exl3_smoe` for the routed experts, `exl3_mrow` for the dense linears) rather than exllamav3's `exl3_moe_coop`; the dense layers, the head and the DSpark drafter are on EXL3 too; the numbers here are the plan's models, the measured ones are in the README. Step labels from the build plan: P4 quantization, P5 kernels, P6 the DSpark depth sweep; T2 the vLLM pin, T3 the memory fit (>= 4 GiB free per node after load), T4 the speed targets, T5 the quality panel, T9 the release; G0-G4 the plugin's bring-up stages (the list near the end).

Written Oct 8 2026 for the recipe's quantization (P4) and kernel (P5) steps, route A: an out-of-tree vLLM quant method. Read-only study plus a skeleton: nothing was run on a GPU, nothing on the Sparks was modified. Skeleton: `patches/exl3_moe/` (package `vllm_exl3_moe`, README there).

Skeleton status (Oct 8): every file passes `py_compile`; the package root and `plan.py` import without torch / vLLM / exllamav3 (`register()` logs and no-ops); pytest: 17 pass without torch (plan, TP-split math), 23 pass with a CPU torch (adds the bank layout / TP cut / pointer-table tests); the GPU kernel tests and the vLLM registration tests skip. Every GPU path was untested until the bring-up ran on a Spark (Oct 9).

**Sources and citation keys**

- `x3:` = turboderp-org/exllamav3 tag **v1.6.0** = `151539c77abc7ab7425d30da7a4e8e3c5c154e7b` (Oct 7 2026, MIT), from a local clone. Paths from the repo root.
- `vl:` = vllm-project/vllm **main @ `6e517b15c1833cf72a7f557ee32524d98682e617`**, shallow fetch of that commit.
- `ref:` = spark2 `~/Projects/local-ai-images/deepseek-v4.1-flash-spark/` @ `e22505f8` (0xSero's image repo, no LICENSE): read over one read-only ssh, cited for ideas, **no code copied**.
- `cfg:` = official `config.json` @ `2cba9e42aa` (hidden 5120, `moe_intermediate_size` 2304, 384 routed experts top-6 + 1 shared, 40 layers, `swiglu_limit` 10.0, `routed_scaling_factor` 1.5, `scoring_func` sqrtsoftplus, DSpark 3 layers x 128 experts top-3; quantization_config `{"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"}`).
- `pkg:` = `patches/exl3_moe/` (this design's skeleton).

---

## 0. Summary

1. **exllamav3 v1.6.0 has two MoE kernels we can call as is.** `exl3_moe_coop` (fused decode: rot + gate/up + down in 2-3 launches, any `bsz x top_k <= 256` slots, no grid barrier, deterministic, no host sync) for decode and DSpark verify, and `exl3_moe` (grouped over expert-sorted rows, 16/32/64-row tiles) plus `reconstruct` + `hgemm` for prefill. Both take **one bitrate pair per launch** (`K_gate == K_up`, `K_down`), and exllamav3's own MoE module asserts uniform K per layer (`x3: exllamav3/modules/multilinear.py:25-26`). Per-expert K mixing is ours: experts sorted by K into contiguous physical slots, **one launch per K group**, groups chained through the coop kernel's additive `sh_out` input. 0xSero's fork did the per-K split too (buckets, `ref: patch/st_exl3_moe.py:141-153`), but with `exl3_moe` at decode (`ref: patch/st_exl3_moe.py:12`, `:263-272`), where, by reading, each bucket launch under graphs (`num_active = -1`) runs 6 groups of 8 SMs and only the groups that win an expert ticket work: with ~1.5-2 experts per bucket at bsz 1, 8-16 of the 48 SMs.
2. **The kernels are sm_121-clean by reading:** mma.sync m16n8k16, cp.async, ldmatrix, L2 cache hints, no wgmma/TMA/tcgen05; the only `__CUDA_ARCH__ >= 1000` branch is a PTX 8.6 mixed add in `hgemm_f16acc.cu:55` (not on our path). GB10 is classed `CC_BLACKWELL` on the host (`x3: quant/exl3_devctx.cu:46`), so tuning heuristics assume B200/5090-class parts. Build with `TORCH_CUDA_ARCH_LIST=12.1`.
3. **vLLM needs no source patch.** A `vllm.general_plugins` entry point registers quant_method `deepseek_v41_exl3`, whose config subclasses the V4.1 `DeepseekV4FP8Config` and redirects only backbone `RoutedExperts` named in the plan. The V4.1 loader already routes `experts.{e}.w1.<suffix>` to a param `experts.routed_experts.w13_<suffix>` with `(shard_id, expert_id)`, so registering `w13_trellis / w13_suh / w13_svh / w13_mul1 / w13_mcg` (and `w2_*`) is enough.
4. **TP layout decision: vLLM's own MoE channel split, cut at load time from the canonical exllamav3 tensors.** EXL3 tensors CAN be split along the intermediate dim at multiples of 128: tiles are coded independently and both Hadamard rotations are 128-block-diagonal. exllamav3 itself splits experts this way (`x3: modules/block_sparse_mlp.py:1517-1554, 1640-1648`, `modules/quant/exl3.py:296-342`). At TP=2 each rank holds 1,152 = 9 x 128 channels of all 384 experts. No per-shard encoding, checkpoint stays exllamav3-native and engine-portable, per-rank memory is exactly half whatever the K mix, and decode work is balanced (expert-parallel-in-TP pays E[max of 2 ranks] = 3.94 of 6 experts per token per layer, about 31% more than 3, and memory imbalance under a mixed-K plan).
5. **Decode cost model (routed experts only, uniform routing, 2.7 bpw, 273 GB/s):** 1 token = 6 experts = 35.8 MB per rank per layer = 1.43 GB per rank per token = **5.25 ms** at roofline. A DSpark k=7 verify step (8 tokens) touches ~45 unique experts per layer: 10.85 GB per rank, **39.8 ms per step**, 7.6x the bytes of one token. Routed-expert bytes scale with unique experts, so spec decode only pays through the fixed per-step costs (attention, head, ~80 all-reduces over RoCE, Engram). Measure acceptance and unique-expert counts before choosing k.
6. **Biggest risks:** per-K-group decode launches each activate ~1.5 of the 6 experts (grid under-filled; the fix is a small MIT patch giving `exl3_moe_coop` a per-expert K table); exllamav3 build/runtime on sm_121 not yet observed; prefill speed of stock `exl3_moe` against T4's 2,050 tok/s at 8K; UMA page cache while each rank reads 94-187 GB of EXL3 shards at load; atomics in the prefill path (G1 exactness).
7. **First on a Spark (after the vLLM build, one build at a time):** build the exllamav3 v1.6.0 wheel for 12.1, run `pkg: tests/test_kernels_gpu.py` (kernels vs reconstruct reference at every K, TP=2 split sums to the unsplit expert, CUDA graph replay bit-exact, decode GB/s bench), then a `--load-format dummy` TP=2 G0 with a synthetic 2.76-bpw plan for memory fit and the decode ceiling before any quantized weights exist.

---

## 1. exllamav3 v1.6.0 kernel API

### 1.1 Entry points (pybind, `x3: exllamav3/exllamav3_ext/bindings.cpp`)

| entry point | bind line | role for us | key arguments |
|---|---|---|---|
| `exl3_moe_coop` | :292 | **decode / DSpark verify / every CUDA-graph size** | `x` fp16 (bsz, H); `sel` int64 (bsz, top_k); `rw` fp16 (bsz, top_k); `min_expert, max_expert`; `Hi`; 9 per-expert pointer tables (g/u/d x trellis/suh/svh, int64 contiguous); 3 optional bias tables; `Kg, Ku, Kd` (float); `mcg, mul1`; `act` (0 = SiLU); `act_limit`; `gated`; scratch `had_g, had_u` fp16 (slots, Hi), `gu_g, gu_u, act_out` fp16 (slots, I), `d_out` fp32 (slots, Ho), `ctr` int32; `out` fp32 (rows, H); optional `sh_out` fp32, `sh_gate_w` fp16 (`x3: quant/exl3_moe_coop.cuh:122-158`, `quant/exl3_moe_coop.cu:315-364`) |
| `exl3_moe` | :290 | prefill tiers; graph-safe fallback | `hidden_state` fp16 (bsz, H); `output_state` fp32 (bsz, H) zeroed; `expert_count` int64 (E+1) (last = sentinel); `token_sorted` int64, `weight_sorted` fp16 (bsz x top_k, sorted by expert); temp `(C, R, H)` x2 + `(C, R, I)` x2 fp16; `act_function`; `K_gate, K_up, K_down`; 9 pointer tables (E); 6 codebook flags; `act_limit`; `num_active` (-1 = unknown); optional `output_scratch` fp32 + `fused_base` int64 (deterministic slots); `count_lo, count_hi`; `m_tile` 16/32/64 (`x3: quant/exl3_moe.cuh:8-48`, doc block `quant/exl3_moe.cu:189-258`) |
| `exl3_moe_gather` | :291 | deterministic sum of slot outputs | `x3: quant/exl3_moe.cu:575-697` |
| `exl3_moe_max_concurrency` | :289 | C for the `exl3_moe` temp buffers | `num_sms / 8` on CUDA (`quant/exl3_moe.cu:147-151`); 6 on GB10's 48 SMs |
| `reconstruct`, `reconstruct_slice`, `reconstruct_had_slice`, `reconstruct_had_batch` | :170-174 | hot-expert prefill tier | trellis -> fp16 W, rotated basis (exact in fp16) or folded original basis (`x3: quant/reconstruct.cuh:5-54`) |
| `had_r_128` | :175 | activation-side rotations for the reconstruct tier | `(input, output, pre_scale=suh, post_scale=svh, scale)`, fp16 or fp32, dim % 128 (`x3: quant/hadamard.cuh:5-12`) |
| `hgemm` | :209 | fp16 GEMM, fp16/fp32 out | `x3: hgemm.cuh:14-19` |
| `exl3_mgemm` | :203-208 | multi-matrix GEMV/GEMM (exllamav3's graph paths) | not used by the plugin |
| `BC_BlockSparseMLP` (bound class) | `libtorch/blocksparse_mlp_bc.h:1-113` | exllamav3's own MoE runtime (`run_bszN`, `run_single_expert[_dq]`) | not used: it owns its routing/shared expert and asserts uniform K |

**`exl3_moe_coop` behaviour that the design relies on**
- Slots with routing weight 0 or an expert outside `[min_expert, max_expert)` are inactive; table index = `sel - min_expert` (`x3: quant/exl3_moe_coop_kernel.cuh:207-223`).
- Kernel A (gate/up) groups the slots of one expert into runs of up to 8 MMA rows, so an expert's weights are read once per run (`:3-21`). Kernel B writes each token row as the fixed-order sum of its active slots plus `sh_out` (`:929-947`). A row with no active slot gets `sh_out` alone (`:572-610`, called at `:659-661`, `:715-717`). **Overwrites `out`, adds `sh_out`.** That is what lets K groups chain: `out_g = group_g + out_{g-1}`.
- Counters: A zeroes the down-stage counters for this call, B zeroes the gate/up counters for the next (`:695-698`, `:833-835`); the buffer is zeroed once at allocation. One scratch set serves every layer in stream order.
- `rows_max = out.size(-2)` sets the counter layout (`quant/exl3_moe_coop.cu:228-238`), so `out` is always passed whole.
- Limits: `slots_max <= 256` (`:226`, kernel `MAX_SLOTS` `quant/exl3_moe_coop_kernel.cuh:46`), `Hi, I, Ho % 128 == 0` (`quant/exl3_moe_coop.cu:202`), gate and up share K (`:194`). Tile geometry: always wide below Blackwell, picked by `kslices` and slots on Blackwell-class devices (`:108-114`); split-k by SM count (`:81-93`).
- First launch per instance sets the >48 KB smem attribute (`:116-125`): prime before capture.

**`exl3_moe` behaviour**
- Walks experts in index order with a prefix sum of `expert_count`; experts with `count > R` (temp rows) or outside `[count_lo, count_hi]` are skipped (`x3: quant/exl3_moe_kernel.cuh:183-197`). Rows beyond the fused kernel go to a reconstruct path the caller runs.
- `K = 0` instance switches K at runtime per projection in half-bit units (`quant/exl3_moe_kernel.cuh:31-70`); integer K uses compile-time instances (`quant/exl3_moe.cu:35-47`, `:367-378`).
- 32/64-row tiles exist for mul1 only (`quant/exl3_moe.cu:436-448`).
- Group spin barriers assume all blocks co-resident (`quant/exl3_moe.cu:451-492`); launches `num_groups x 8` blocks.
- Lock/scheduler buffer is `cudaMalloc`ed lazily on first use (`quant/exl3_devctx.cu:89-106`): illegal during stream capture, so prime eagerly (0xSero hit the same: `ref: patch/st_exl3_moe.py:126-138`).
- Without `output_scratch`, outputs are atomically added (not bit-reproducible); with it plus `exl3_moe_gather`, deterministic (`quant/exl3_moe_kernel.cuh:342-386`, `quant/exl3_moe.cu:575-697`).

### 1.2 How exllamav3's own MoE module drives them (`x3: exllamav3/modules/block_sparse_mlp.py`)

- `load_local` (`:508-754`): builds `MultiLinear` per projection (pointer tables, one K, one codebook, `x3: modules/multilinear.py:5-43`), scratch for the fused-decode path sized `MAX_BSZN (8) x top_k` slots, the bound class `BC_BlockSparseMLP`, and `exl3_moe` temp buffers `(C, R, H/I)` with `R = 256` when mul1 allows wide tiles (`:738-753`).
- `forward` (`:1007-1454`): bsz <= 8 goes to `bc.run_bszN` = `exl3_moe_coop_run` (`x3: libtorch/blocksparse_mlp.cpp:70-111`); larger batches: flatten, map out-of-slice experts to the sentinel, `argsort`, `bincount`, one host read of the counts when rows could exceed `R`, then `exl3_moe` in 64/32/16-row tiers, a batched reconstruct tier (`modules/moe_batch_recon.py`) and a per-expert reconstruct/graph tier, deterministic gather by default (`:1105-1372`).
- TP (`:1494-1724`): `moe_tensor_split` = **channel split** of every expert in 128-channel units (`channel_width = 128`, `channels_to_split = ups[0].out_features // 128`, `:1536-1548`), `gu_split = (True, first, last)` and `d_split = (False, first, last)` (`:1640-1648`) through `LinearEXL3.tp_import_split` (`modules/quant/exl3.py:296-342`: out-split slices trellis dim 1 and svh, in-split slices trellis dim 0 and suh); otherwise expert parallel by ranges.
- DeepSeek-V4 MoE config (`x3: exllamav3/architecture/deepseek_v4.py:175-210`): keys `layers.{i}.ffn.experts.{e}.w1/w3/w2`, `activation_fn="silu"`, `act_limit = swiglu_limit`, `interm_dtype = torch.half`, router `sqrtsp` / `sqrtsp_hash`. **No V4.1 architecture** in v1.6.0 (no Engram / CSA2 / encoder-decoder), so conversion of V4.1 is per-matrix with our own Hessians (P4).

### 1.3 Per-expert K

- exllamav3 does not mix K inside a layer's experts: `MultiLinear` asserts one K and one codebook (`x3: modules/multilinear.py:25-37`); the converter allocates per module group per layer (`block_sparse_mlp.py:493-506`).
- Kernels accept one `(K_gate = K_up, K_down)` per launch (`quant/exl3_moe.cu:360-378`, `quant/exl3_moe_coop.cu:193-194`, `:274-276`).
- So a K2/K3/K4/K5 mix costs one launch set per distinct pair per layer. With our K-sorted slots a group is a range `[lo, hi)` and the coop kernel's own expert-range filter masks other groups: no extra masking kernel, no re-sorting.
- Half-integer K (1.5/2.5/3.5) exists for mul1 (`quant/exl3_moe.cu:367-378`, `quant/exl3_moe_coop.cu:20-45`): a useful extra allocation step between 2 and 3 bits.

### 1.4 sm_121 (GB10) build and arch guards

| item | finding | where |
|---|---|---|
| instructions | mma.sync m16n8k16 f16 (sm_80+), cp.async (+ `createpolicy` L2 hints), ldmatrix, acquire/release ld/st; nothing Hopper-only (wgmma, TMA) or datacenter-Blackwell-only (tcgen05/TMEM) | `x3: exllamav3_ext/ptx.cuh:42-493` |
| compile-time gates | `EXL3_SM75` fallbacks only below sm_80 | `arch.cuh:22-26` |
| | `__CUDA_ARCH__ == 860` tuning branches (sm_86 only) | `quant/exl3_gemm_inner.cuh:14`, `quant/codebook.cuh:20`, `quant/exl3_moe_coop_kernel.cuh:54` |
| | `> 890`: group-barrier path in `exl3_mgemm` (taken on sm_121) | `quant/exl3_gemm_kernel.cuh:110, 222, 260` |
| | `>= 1000`: `add.rn.f32.f16` (PTX ISA 8.6, "sm_100 or higher"); sm_121 qualifies, **check ptxas accepts it at build** (file is compiled even though we do not call it) | `hgemm_f16acc.cu:55-63` |
| host classes | major >= 10 -> `CC_BLACKWELL`: GB10 gets B200/5090-tuned host heuristics (coop wide/narrow, gemv-int8 K cap) | `quant/exl3_devctx.cu:44-50`, `quant/exl3_moe_coop.cu:108-114`, `quant/exl3_gemv_int8.cu:57` |
| shared memory | requests `min(opt-in max, 90 KB)`; GB10 (sm_12x class) opt-in is about 99 KB, fits | `quant/exl3_devctx.cu:56-75`, `arch.cuh:28-37` |
| build | `setup.py` uses torch `CUDAExtension("exllamav3_ext")` (top-level module), honours `TORCH_CUDA_ARCH_LIST`; `--compress-mode=size` needs CUDA >= 12.8 (13.0 ok); the JIT path derives `12.1+PTX` clamped to torch's arch list | `setup.py`, `exllamav3/util/cuda_flags.py:221-256`, `exllamav3/util/arch_list.py:8-41` |
| UMA | exllamav3 knows GB10 shares host memory | `exllamav3/util/memory.py:27` |
| reference | 0xSero built exllamav3 1.5.3 and `st_moe_ext` with `12.1` and served on GB10 | our reading of 0xSero's `local-ai-images` (`ref:`) |

### 1.5 On-disk EXL3 format (per linear, safetensors)

| tensor | dtype | shape | notes |
|---|---|---|---|
| `<key>.trellis` | int16 | `(in/16, out/16, 16*K)` (`16*K` = 40 for K=2.5) | 16x16 tiles, each coded on its own; K = `shape[-1] / 16` (`x3: modules/quant/exl3.py:46-68`, `quantize.py:93-98`, `:121-128`, `:1094-1110`) |
| `<key>.suh` | fp16 | `(in,)` | input sign x channel scale (`quantize.py:1831`, scales folded by `regularize`, `:1371-1470`) |
| `<key>.svh` | fp16 | `(out,)` | output sign x channel scale |
| `<key>.mul1` or `<key>.mcg` | int32 scalar | `()` | codebook marker; value 0x83DCD12D / 0xCBAC1FED, never read beyond presence (`quantize.py:18-19`, `:1846-1856`) |
| `<key>.bias` | fp16 | `(out,)` | optional (none in DeepSeek) |
| `<key>.su`, `<key>.sv` | int16 | packed sign bits | legacy format, unpacked at load (`exl3.py:39-64`); our checkpoints never write them |

- Orientation: `(in_features, out_features)` row-major, the transpose of `nn.Linear` (`quantize.py:1636-1640`).
- Reconstruction: `W = had_r(suh * had_l(W_rot)) * svh`, 128-block Hadamards scaled 1/sqrt(128) (`exl3.py:239-249`, `quantize.py:15`, `:410-434`).
- Model-level `config.json`: `{"quant_method": "exl3", "version", "bits", "head_bits", "calibration", "out_scales", "codebook"}` (`x3: exllamav3/conversion/compile.py:194-240`). Ours uses its own quant_method (section 3.2) and keeps the tensor format.
- DeepSeek-V4.1 expert shapes: w1, w3 `(5120, 2304)` -> trellis `(320, 144, 16K)`; w2 `(2304, 5120)` -> `(144, 320, 16K)`.

---

## 2. vLLM main @ 6e517b15

### 2.1 Registration path

1. Plugin: entry-point group `vllm.general_plugins` (`vl: vllm/plugins/__init__.py:18`), executed by `load_general_plugins()` (`:77-91`) in `EngineArgs.__post_init__` (`vl: vllm/engine/arg_utils.py:891-895`), the engine core (`vl: vllm/v1/engine/core.py:125`) and every worker (`vl: vllm/v1/worker/worker_base.py:274-276`). Must be idempotent.
2. `register_quantization_config(name)(cls)` appends to `QUANTIZATION_METHODS` and the platform list and stores the class (`vl: vllm/model_executor/layers/quantization/__init__.py:61-109`); `get_quantization_config` merges customs last, so they win (`:112-197`). On CUDA, `"deepseek_v4_fp8"` resolves to the V4.1 class (`:120-128`).
3. `ModelConfig._verify_quantization` probes custom methods first, then the ordered built-in overrides (`vl: vllm/config/model.py:1362-1431`). A custom class inheriting the V4.1 `override_quantization_method` would claim every official fp8 V4.1 checkpoint, so ours overrides it to claim only `deepseek_v41_exl3` (`pkg: vllm_exl3_moe/config.py`).
4. `get_quant_config` calls `quant_cls.from_config(hf_quantization_config)` (`vl: vllm/model_executor/model_loader/weight_utils.py:192-265`). The capability gate is `vl: vllm/config/vllm.py:925-929`.
5. `DeepseekV4ForCausalLMConfig.verify_and_update_model_config` rewrites only `quant_method == "fp8"` to `deepseek_v4_fp8` (`vl: vllm/model_executor/models/config.py:344-365`, registered for `DeepseekV41ForCausalLM` at `:1013`); ours passes untouched.

### 2.2 `QuantizationConfig` (`vl: vllm/model_executor/layers/quantization/base_config.py:99-207`)

Abstract: `get_name`, `get_supported_act_dtypes`, `get_min_capability` (classmethod), `get_config_filenames`, `from_config` (classmethod), `get_quant_method(layer, prefix)` (`:120-151`, `:192-207`); optional `override_quantization_method` (`:153-172`). The V4.1 class (`vl: vllm/models/deepseek_v41/quant_config.py:33-231`, exported by `vllm/models/deepseek_v41/__init__.py:7`) implements all of them on top of `Fp8Config`. Gotcha: `Fp8Config.from_config` sets `is_checkpoint_fp8_serialized = "fp8" in quant_method` (`vl: vllm/model_executor/layers/quantization/fp8.py:158-175`), so our `from_config` hands the parent a copy with `quant_method: "fp8"` or every MXFP8 linear would be re-quantized online.

### 2.3 `FusedMoEMethodBase` and the call chain

- Base: `vl: vllm/model_executor/layers/fused_moe/fused_moe_method_base.py:28-187`. Abstract `create_weights(layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra)` (`:49-59`) and `get_fused_moe_quant_config` (`:102-106`); `apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)` (`:143-166`); `maybe_roundup_sizes` (`:70-100`); `topk_indices_dtype` (`:108-112`); `is_monolithic` false without a modular kernel (`:134-141`). `process_weights_after_loading` is inherited from `QuantizeMethodBase` (`base_config.py:80-85`) and run by `vl: vllm/model_executor/model_loader/utils.py:146-170`.
- `FusedMoEFactory` builds `FusedMoEConfig` (`vl: vllm/model_executor/layers/fused_moe/layer.py:347-374`; `intermediate_size_per_partition = intermediate_size // tp_size`, `config.py:1339-1341`), the `RoutedExperts` (`layer.py:384-413`) and the `MoERunner` (`:418-436`).
- `RoutedExperts.__init__` resolves the method via `quant_config.get_quant_method(layer, prefix)` and calls `create_weights(layer=self, num_experts=num_local_experts, hidden_size, unpadded_hidden_size, intermediate_size_per_partition, params_dtype, weight_loader=self.weight_loader, global_num_experts)` (`vl: vllm/model_executor/layers/fused_moe/routed_experts.py:150-198`).
- Forward: the runner is reached through the opaque custom ops `torch.ops.vllm.moe_forward[_shared]` (`vl: .../fused_moe/runner/moe_runner.py:117-213`; torch.compile does not trace inside, CUDA graphs capture whatever launches inside). `_forward_impl` (`:865-935`) -> `_apply_quant_method` (`:595-648`): router `select_experts(..., topk_indices_dtype=quant_method.topk_indices_dtype)` (int32 by default, `router/fused_topk_bias_router.py:146-215`) then `RoutedExperts.forward_modular` -> `quant_method.apply(...)` (`routed_experts.py:1258-1293`).
- Reduction: with `moe_kernel is None` the fused output counts as un-reduced (`moe_runner.py:420-424`); shared + routed are summed and all-reduced once (`:480-514`). So `apply()` returns this rank's partial sum and never touches the shared expert.

### 2.4 How V4.1 picks its MoE method today

`config.json` says `fp8` -> rewritten/claimed as `deepseek_v4_fp8` (2.1 step 5; `quant_config.py:138-167`) -> `DeepseekV4FP8Config.get_quant_method` (`vl: vllm/models/deepseek_v41/quant_config.py:188-231`):
- `LinearBase` with `[32, 32]` blocks and e8m0 scales -> `ModelOptLinearMethod(kMxfp8Static, kMxfp8Dynamic)` (`:189-210`);
- `RoutedExperts`, `expert_dtype == "fp4"` -> `Mxfp4MoEMethod(layer.moe_config)` (`:211-228`), whose backend comes from `select_deepseek_v4_mxfp4_moe_backend` (`vl: vllm/model_executor/layers/quantization/mxfp4.py:480-489`, `fused_moe/oracle/mxfp4.py:682`; on sm_12x DeepGEMM MXFP4, Marlin fallback in vLLM's MXFP4 backend oracle); `moe_quant_algo == "NVFP4"` -> ModelOpt NVFP4.
- MoE block: `DeepseekV4MoE` (V4 base `vl: vllm/models/deepseek_v4/nvidia/model.py:784-1117`, `_init_fused_moe_experts` `:972-1018` passes `swiglu_limit`, `scoring_func`, `routed_scaling_factor`, `e_score_correction_bias`); the V4.1 subclass gives layers with index >= `num_hidden_layers` the DSpark expert count (128, top-3) (`vl: vllm/models/deepseek_v41/nvidia/model.py:124-151`). The mHC deferred-finalize fusion applies to TRTLLM MXFP4 only (`:153-172`); ours finalizes normally.

### 2.5 Per-expert loading and TP sharding

- `DeepseekV4Model.load_weights` (`vl: vllm/models/deepseek_v41/nvidia/model.py:1185-1311`): names go through the V4 weights mapper (`:1398-1452`: `.scale` -> `weight_scale` rules, `layers.` -> `model.layers.`, `mtp.` dropped for the target); expert tensors (`:1246-1282`) loop over `get_expert_mapping()` (`:1339-1351` -> `fused_moe_make_expert_params_mapping`, `vl: .../fused_moe/layer.py:441-459`), replace `experts.{e}.w1.` by `experts.routed_experts.w13_` (`routed_experts.py:1124-1127`, `:1164-1180`), look up `params_dict[name_mapped]` (KeyError if the param does not exist) and call `param.weight_loader(param, w, name_mapped, shard_id, expert_id, return_success=True)`.
- Native TP split (`RoutedExperts.weight_loader`, `routed_experts.py:648-916`): non-local experts return False (`:678-680`); shard dim `{"w1": 0, "w2": 1, "w3": 0}` on the `[E, N, K]` layout (`:710`), slices of `loaded_per_rank * tp_rank` (`_load_w13` `:514-563`, `_load_w2` `:565-614`). At TP=2: w13 keeps output rows `[r*1152, (r+1)*1152)` of gate and up, w2 keeps input columns of the same range.
- vLLM's strict "all weights loaded" check runs only for unquantized models (`vl: vllm/model_executor/model_loader/default_loader.py:487-497`): the plugin tracks every (expert, projection, tensor) itself.
- `--load-format dummy` leaves integer params uninitialised and fills float ones with small random values (`weight_utils.py:1480-1499`): enough for fit and speed probes.

### 2.6 What the plugin subclasses

- `DeepseekV41Exl3Config(vllm.models.deepseek_v41.quant_config.DeepseekV4FP8Config)`: overrides `get_name`, `get_min_capability` (80), `override_quantization_method` (own name only), `from_config` (split off the `exl3_moe` block, hand the rest to the parent as fp8), `get_quant_method` (backbone `RoutedExperts` in the plan -> ours; everything else -> `super()`).
- `Exl3RoutedMoEMethod(FusedMoEMethodBase)`: `create_weights`, per-param weight loaders, `process_weights_after_loading`, `apply`, `maybe_roundup_sizes` (no padding), `get_fused_moe_quant_config`.
- Untouched, through `super()`: attention / compressor / indexer / shared-expert MXFP8 linears, router, Engram (our `0002` disk patch), vision, the DSpark drafter (MXFP4), any layer the plan leaves native. No model file is patched.

---

## 3. Plugin architecture (`pkg:`)

| module | role |
|---|---|
| `__init__.py` | `register()` entry point; no-op with a warning when vLLM is missing; package importable without torch |
| `config.py` | `DeepseekV41Exl3Config`, `Exl3Settings` (config.json `exl3_moe` block + env overrides), plan resolution (model dir or HF hub), layer index from the prefix |
| `moe_method.py` | `Exl3RoutedMoEMethod`: registers `w13_trellis` (int16 flat), `w2_trellis` (int16 flat), `w13_suh [E,2,5120]`, `w13_svh [E,2,1152]`, `w2_suh [E,1152]`, `w2_svh [E,5120]`, codebook markers; loader with load bookkeeping; post-load pointer tables, scratch, priming; `apply` -> `forward.routed_moe_forward` |
| `bank.py` | `Exl3Bank`: K-sorted slots, flat banks, the TP cut of full exllamav3 tensors (`load_piece`), views, pointer tables; torch only, shared with the GPU tests |
| `forward.py` | `routed_moe_forward` (the three paths of section 4), `run_prefill`, `reference_moe` (tests) |
| `kernels.py` | wrappers over `exllamav3_ext`, process-wide scratch, `prime()` |
| `plan.py` | plan schema, groups, `BankLayout`, `shard_slices`, bpw and bytes accounting; pure Python |

**Checkpoint we publish (P4 output):** the official tensors except the backbone MXFP4 routed experts, which are replaced by `layers.{L}.ffn.experts.{E}.w{1,2,3}.{trellis,suh,svh,mul1}` (exllamav3's own DeepSeek-V4 keys, `x3: architecture/deepseek_v4.py:144, 177, 182-184`). `mtp.*` (DSpark, MXFP4) and the Engram shards stay byte-identical. Shards that mix experts with other tensors are rewritten (vLLM yields every tensor of every listed shard, and a stray MXFP4 expert tensor has no param to land in). `config.json` gets `quant_method: "deepseek_v41_exl3"` plus the original FP8 keys and an `exl3_moe` block; the per-expert plan ships as `exl3_moe_plan.json` (schema in `pkg: README.md`; one character per expert per layer, optional separate down K).

**Loader:** each EXL3 tensor arrives with `(shard_id, expert_id)`; the bank maps the logical expert to its physical slot (`perm`), checks the trellis shape against the plan's K (a plan/checkpoint mismatch fails loudly instead of decoding garbage), cuts this rank's channels (section 5) and copies into the slot. Codebook markers are validated and must be uniform. After loading, every (slot, projection, tensor) must be present unless the load format is dummy.

**Post-load:** device pointer tables (int64, physical order) for gate/up/down x trellis/suh/svh, one `GroupTables` view per K group, the `perm` tensor, process-wide coop scratch (`rows_max = 32`, 192 slots: ~10 MB) and `exl3_moe` temp buffers (`C = 6, R = 256`: ~38 MB), one eager priming launch per group and kernel, `malloc_trim` (UMA).

**Layers left native:** a plan may omit layers; they keep `Mxfp4MoEMethod` (their MXFP4 tensors must then stay in the checkpoint). Useful for the quality ladder ("which layer breaks", PLAN section 5).

---

## 4. How routing results feed the kernels

vLLM's router (unchanged) gives `topk_weights` fp32 `[m, 6]` (routed_scaling_factor 1.5 applied) and `topk_ids` int32 `[m, 6]`. Per call, `pkg: forward.routed_moe_forward`:

1. `x16 = x.to(fp16)` (MoE input is post-RMSNorm, far inside fp16 range; check max |x| on the calibration set), `phys = perm[topk_ids]` (logical id -> K-sorted slot), `rw16 = topk_weights.to(fp16)` with unrouted ids (< 0) given weight 0.
2. **Decode, DSpark verify, every CUDA-graph size (`m <= 32`):** for each K group `g` (static list, 3-4 groups): `exl3_moe_coop(x16, phys, rw16, lo_g, hi_g, 5120, tables[lo_g:hi_g], K_g..., out=buf[g % 2], sh_out=buf[(g-1) % 2])`. No sort, no histogram, no host sync. Bsz 1: 2 kernels per group (rotation in kernel A); bsz 2-32: 3 kernels per group. Output = last buffer, cast to bf16.
3. **Graph capture above 32 rows, or `decode_kernel=fused`:** per group, bucket-local ids (other groups -> sentinel bin), `argsort`, counts by `scatter_add` (bincount reads the max back to the host), `exl3_moe(..., num_active=-1)`. Graph-safe while `m <= R` (an expert takes at most `m` rows since top-k picks are distinct per token).
4. **Eager prefill:** per group, one host read of the counts; `exl3_moe` tiers (`m_tile` 64 for >32 rows, 32 for 17-32, 16 for <=16, mul1) for experts up to `R = 256` rows; hotter experts through `had_r_128(suh)` -> `reconstruct` (rotated basis, exact in fp16) -> `hgemm` fp32 -> `had_r_128(svh)` for gate/up, clamped SwiGLU in torch, same for down, `index_add_` with the routing weight. Atomic accumulation for now (TODO: deterministic slot mode for G1).

Activation: SiLU-gated with `act_limit = swiglu_limit = 10.0`, gate clamped above, up clamped both sides: exllamav3 (`x3: activation_kernels.cuh:178-182`, `quant/hadamard_inner.cuh:389-394`) and vLLM (`vl: .../fused_moe/activation.py:296-310`) agree.

Per-layer launch count at bsz 1, G groups: ~5 small torch kernels (casts, gather, where) + 2G coop kernels, i.e. 11-13 for G = 3-4. Under graph replay that is roughly 25-40 us per layer, 1-1.6 ms per token over 40 layers: worth a fused prep kernel (or bf16/int32/fp32 inputs in the coop launcher) once the GPU tests pass.

---

## 5. TP=2 sharding of EXL3 tensors

**Can the trellis be split along the intermediate dim? Yes, at multiples of 128 channels.**
- Every 16x16 tile is coded on its own at fixed K (`x3: quantize.py:121-128`, `:1094-1110`; `exl3.py:66-68`), so slicing the tile grid is exact.
- `suh`/`svh` are per-channel.
- Both rotations are 128-block-diagonal (`quantize.py:15`, `:410-434`; kernels rotate per 128 chunk with 1/sqrt(128), `quant/exl3_moe_kernel.cuh:213-240`). Hence block (a, b) of W depends only on trellis block (a, b), `suh` block a and `svh` block b.
- Gate/up are cut on their output axis (trellis dim 1, `svh`); down on its input axis (trellis dim 0, `suh`). The down output rotation is linear, so per-rank partial sums add up after the all-reduce.
- exllamav3 does exactly this for its own MoE TP (`x3: modules/block_sparse_mlp.py:1536-1548, 1640-1648`; `modules/quant/exl3.py:296-342`).
- `pkg: tests/test_tp_split_math.py` checks the property numerically at the real shapes: rank 0 + rank 1 == the unsplit expert to 1e-9. It also shows a tile-aligned but not 128-aligned cut breaks it.

**Decision: vLLM's channel split (`tp_layout: channels`), cut at load time from the canonical full tensors.**

| option | per-rank memory | decode work per token per layer | checkpoint | verdict |
|---|---|---|---|---|
| **channel split** (each rank: 1,152 channels of all 384 experts) | exactly half, any K mix | 6 half-experts on both ranks, balanced | canonical exllamav3, engine-portable (TensorFold TP2 / exllamav3 can read it) | **chosen** |
| expert parallel inside TP (rank r: experts `[192r, 192r+192)`; what the fork did, `ref: patch/st_exl3_moe.py:9-11`) | depends on how many high-K experts land per half (needs a balancing permutation) | slowest rank holds E[max] = 3.94 of 6 experts (+31% vs 3); at a k=7 verify step 25.3 vs 22.7 unique experts | canonical | rejected for TP=2 |
| encode per TP shard at quantization time | half | as channel split | non-standard tensors, tied to TP=2 | unnecessary: the cut is exact |

Constraints: `I_loc % 128 == 0`, so TP in {1, 2, 3, 6, 9, 18} for 2304; **TP=4 (576 = 4.5 x 128) and TP=8 do not work** with this layout (expert parallel or re-encoding then). Load I/O: each rank keeps half of every expert (~94 GB of the ~187 GB of EXL3 tensors at 2.76 bpw), but the gate/up halves are strided (each tile row splits into two 4.6-11.5 KB runs at K=2-5), so with mmap readahead a rank reads most of w1/w3 anyway: expect 94-187 GB read per rank. A checkpoint variant with per-rank pre-split tensors would make each rank's reads contiguous and exactly half; a load-time optimization, not needed for correctness.

Memory per rank at TP=2 (`pkg: tools/make_plan.py`; 543.6B routed params; trellis + 0.54 GiB fp16 `suh/svh`):

| static bpw | GiB per rank |
|---|---|
| 2.50 | 79.64 |
| 2.60 | 82.80 |
| 2.70 | 85.97 |
| 2.758 (K2 7,279 / K3 6,299 / K5 1,782) | 87.81 |
| 2.80 | 89.13 |

The memory budget from the official safetensors headers leaves ~82 GiB per node for experts (worst case, 17 GiB native replicated): **~2.6 bpw** unless the native weights shard better. Every GiB of scratch, graphs or headroom saved buys ~0.03 bpw.

---

## 6. CUDA graph compatibility

- vLLM captures whatever the MoE custom op launches (FULL_DECODE_ONLY on GB10: in the reference build FULL_AND_PIECEWISE drove a rank to 2 GiB MemAvailable). Capture sizes up to 16 rows (2 streams x (k+1) = 16) all take the coop path: no host sync, static scratch, pointer tables fixed after load, deterministic output; per-call host decisions (wide/narrow tile, split-k) depend only on `m`, which is static per captured graph.
- Inputs created inside the op (`x16`, `phys`, `rw16`, outputs) come from the graph's private pool; `MoeCoopParams` is a by-value kernel argument baked at capture.
- **Prime before capture:** first launch of each coop/exl3_moe instance sets its smem attribute, and `exl3_moe`'s lock buffer is `cudaMalloc`ed on first use. `prime()` runs one eager single-token call per K group at load and refuses to run while capturing.
- Above 32 rows under capture: graph-safe `exl3_moe` path (`m <= 256`), else a clear error asking for a smaller `--max-cudagraph-capture-size`.
- Shared coop counters assume one stream: fine for vLLM's single forward stream; DBO/ubatching is already off for V4.1 with Engram (vLLM's V4.1 model code).
- `exl3_moe` (eager prefill only) spins on group barriers that assume co-resident blocks: a side-stream kernel (our Engram prefetch) can delay but not deadlock it.
- Open upstream issue #57156 (NaN with CUDA graphs on SM12x from dummy capture batches) is attention-side; for us dummy batches only mean random but valid expert ids. Unrouted ids are masked.
- `test_cuda_graph_replay_is_bit_exact` captures and replays the decode path at 1 and 8 rows.

---

## 7. DSpark drafter at source precision

- The drafter's MoE layers sit at indices `num_hidden_layers + i` = 40-42 (`vl: vllm/models/deepseek_v41/nvidia/dspark.py:126-128`), 128 experts top-3 (`nvidia/model.py:133-141`). `get_quant_method` sends any `RoutedExperts` with index >= `num_hidden_layers` (read from the live config, so `--hf-overrides num_hidden_layers` truncation stays correct) or absent from the plan to `super()` -> `Mxfp4MoEMethod`.
- The checkpoint keeps `mtp.*` exactly as shipped (MXFP4 experts, MXFP8 linears, ~7.4 GiB). The warning in vLLM's V4.1 code (a re-quantized low-bit drafter halves acceptance) does not apply: nothing in the drafter is re-quantized. (Shipped later: the drafter's dense linears at EXL3 K5 and its experts at K3, each gated on acceptance.)
- The drafter does no Engram reads (the reference build's profile). Its per-step cost (k sequential passes, each reading its active MXFP4 experts and attention) is part of the fixed cost below and must be measured.

---

## 8. Decode cost model on GB10 (routed experts only)

Assumptions: 2.7 bpw trellis (sign vectors add 0.4%), whole expert = 3 x 5120 x 2304 x 2.7 / 8 = **11.94 MB**, half per rank at TP=2; routing uniform and independent across tokens (DeepSeek's bias-balanced routing is close to uniform over a window; real token-to-token correlation lowers the unique counts, measure them); 273 GB/s per box (measured figure, vault `project_spark_bandwidth_contradiction`); assumed: EXL3 trellis decode is ALU-light at 2-5 bits, so bandwidth-bound if the grid is full (the bench in `test_kernels_gpu.py` checks it). Unique experts per layer for n tokens: E[U] = 384 x (1 - (1 - 6/384)^n), Monte Carlo in `pkg: tools/cost_model.py` (rerun with other bpw / bandwidth).

| tokens per step n (DSpark k = n-1) | unique experts per layer | per rank per step, channel split | time @ 273 GB/s | time @ 80% (218 GB/s) | expert-parallel slowest rank @ 273 GB/s |
|---|---|---|---|---|---|
| 1 (spec off) | 6.0 | 1.43 GB | 5.25 ms | 6.56 ms | 6.86 ms |
| 2 (k=1) | 11.9 | 2.84 GB | 10.42 ms | 13.03 ms | 12.72 ms |
| 4 (k=3) | 23.5 | 5.61 GB | 20.54 ms | 25.68 ms | 23.84 ms |
| 8 (k=7) | 45.4 | 10.85 GB | 39.76 ms | 49.70 ms | 44.21 ms |
| 16 (2 streams, k=7) | 85.4 | 20.40 GB | 74.72 ms | 93.39 ms | 80.38 ms |

Read-outs:
- **Routed-expert floor with spec decode off: 5.25 ms per token per rank, i.e. at most ~190 tok/s from expert bytes alone (spec off, model, not a measurement).** The fixed per-step costs (attention and indexer weights, shared expert ~0.71 GB per rank, head ~0.66 GB per rank, about two NCCL all-reduces per layer over RoCE at ~65 us each for a 48 KB message (the reference build's NCCL numbers), Engram reads, kernel launches, the drafter) decide the real number; they are not in this table. G0 with dummy weights measures them.
- **A k=7 verify step reads 7.6x the expert bytes of a single token.** At an acceptance length of 4 that is 9.9 ms of expert reads per accepted token versus 5.25 ms with spec off. On this model and box, DSpark only wins through the fixed per-step costs, so the k sweep (PLAN P6: 3 / 5 / 7) must be measured with unique-expert counts logged (the P4 calibration pass can log U(n) from real routing traces for free).
- **Usage-weighted bpw is what decode pays.** Bytes per token = sum over experts of p(e) x bytes(e); giving K5 to the hottest experts raises decode bytes far more than the 2.76 static average suggests (`plan.usage_weighted_bpw`). The P4 allocator needs two budgets: static bpw for memory (T3) and usage-weighted bpw for bandwidth (T4).
- **Expert-parallel-in-TP's slower rank reads 16-31% more expert bytes than the channel split at n <= 4** (the decode-critical range), 8-11% more at n = 8-16.
- **Grid fill matters as much as bytes:** one coop launch per K group at bsz 1 activates ~1.5 experts x 2 projections x 9 column groups (~27 blocks for gate/up) on 48 SMs. If one block cannot pull ~10 GB/s, the per-group launches fall short of roofline; the bench in `test_kernels_gpu.py` reports GB/s per layer to decide whether the per-expert-K kernel patch (section 9 item 1) comes first.

---

## 9. Kernel work for P5 (ordered; each with a test against the reconstruct reference and a roofline bench)

1. **Per-expert K table in `exl3_moe_coop` (patch to exllamav3, MIT, upstreamable to turboderp).** Kernel A's only K-dependent call is `gemv_tile<bits, cb, WIDE, HALF>(...)` for the run's expert (`x3: quant/exl3_moe_coop_kernel.cuh:676-805`, call at `:747-753`), kernel B's likewise (`:813-959`). Add `const uint8_t* k_gu_table, k_d_table` to `MoeCoopParams`, switch on the run's K inside the block (block-uniform, no divergence), size shared memory for the largest K in the set, instantiate a "mixed" kernel for {2, 3, 4, 5} x mul1. One launch set per layer instead of one per K group; all 6 experts in one grid at bsz 1.
2. **Input plumbing:** bf16 `x`, int32 ids, fp32 weights and bf16 output inside the coop launcher (or one fused Triton prep kernel), removing ~5 kernels per layer (~1 ms per token under graphs).
3. **Prefill grouped GEMM for sm_121, our own code:** decode each trellis tile once per 64-128-row block into shared memory, mma.sync m16n8k16, cp.async pipeline, GPU-built tile lists (no host sync), down-projection epilogue with output rotation x svh x routing weight. 0xSero reports 1.3-1.65x over `exl3_moe` on GB10 routed prefill with this shape of kernel (`ref: README.md:54-55`, design described at `ref: patch/st_moe_ext/st_moe_ext.cu:2-14`); we re-derive it from exllamav3's public `exl3_dq.cuh` / `hadamard_inner.cuh` primitives (MIT). Target: T4 prefill >= 2,050 tok/s at 8K.
4. **Deterministic prefill accumulation** (`output_scratch` + `fused_base` + `exl3_moe_gather`) for G1 exactness.
5. **GB10 tuning sweeps:** `EXL3_MOE_COOP_WIDE`, `EXL3_MOE_COOP_KSPLIT`, `EXL3_MOE_TILE_N` (exllamav3's own knobs) on the 48-SM part.

---

## 10. Open risks

| # | risk | why it matters | mitigation / test |
|---|---|---|---|
| 1 | per-K-group decode launches under-fill the GPU | each launch carries ~1.5 of 6 experts at bsz 1 | GB/s bench first; section 9 item 1 kernel patch; cap plans at 3 K groups per layer if needed |
| 2 | exllamav3 v1.6.0 on sm_121 not observed (build: PTX 8.6 add in `hgemm_f16acc.cu`, 90 KB smem; runtime: Blackwell-class host heuristics on a 48-SM part) | blocks route A | build the wheel with `12.1`, `cuobjdump` for sm_121, `test_kernels_gpu.py`; fall back to 1.5.4 if 1.6.0 regresses |
| 3 | prefill speed of stock `exl3_moe` (0xSero: ~35 TFLOP/s of ~95 on GB10, `ref: patch/st_moe_ext/st_moe_ext.cu:2-14`) | T4 prefill >= 2,050 tok/s at 8K | measure in G4; section 9 item 3 |
| 4 | fp16 kernel activations from a bf16 model | overflow would be silent | MoE input is post-RMSNorm; log max abs on the calibration set; swiglu clamp at 10 bounds the intermediate |
| 5 | UMA page cache during load (each rank reads 94-187 GB of EXL3 shards on a 128 GB box shared with the GPU) | load-time memory pressure, free-memory accounting counts reclaimable cache as free | watch MemAvailable during G0 load; `posix_fadvise(DONTNEED)` / O_DIRECT loader or pre-split shards if it bites |
| 6 | atomics in the prefill path | G1 exactness, run-to-run drift | deterministic slot mode (section 9 item 4); decode (coop) is already deterministic |
| 7 | plan/checkpoint mismatch | garbage experts, no crash | trellis shape checked against the plan's K per tensor; every piece tracked; codebook markers validated |
| 8 | TP != {1,2,3,6,9,18} | channel split impossible (TP=4: 576 channels) | expert parallel path (TODO) or re-encode |
| 9 | usage-weighted bpw ignored by the allocator | decode bytes above the static-bpw promise | second budget in P4 |
| 10 | vLLM MoE internals are mid-migration (`moe_runner.py` "TODO(bnell)... MK migration") | a newer main may change `apply` / loader contracts | pinned to 6e517b15; `test_vllm_registration.py` + G0 on every rebase |
| 11 | spec decode multiplies expert bytes (section 8) | DSpark k=7 may lose to k=3 or to spec off | k sweep with U(n) logged; decide by measurement |
| 12 | exllamav3 moves fast (1.5.3 -> 1.6.0 in a week) | API drift | pin `151539c7`; wrappers check required entry points at import |
| 13 | pointer tables are raw addresses | anything that moves params after load (sleep mode, weight reload, CPU offload) breaks them | rebuild tables in `process_weights_after_loading`; do not enable those features with this method |
| 14 | quality is measured only end to end | the EXL3 proxy error is not KLD | KLD panel through the plugin (T5), layer by layer when it fails |

---

## 11. What to test first on a Spark

Order, each step gating the next (none needs the 510 GB download or a 2-node launch until step 4):

1. **Build** the exllamav3 v1.6.0 wheel for `TORCH_CUDA_ARCH_LIST=12.1` in the vLLM build env after the vLLM wheel finishes (one build at a time), `cuobjdump --list-elf` for sm_121 only.
2. **Kernel tests, one GPU, no vLLM:** `python -m pytest -q tests/test_kernels_gpu.py` (coop vs reconstruct reference at K = 2/3/4/5 and 1..16 rows, fused and prefill paths, TP=2 rank sums equal the unsplit expert, empty K groups, unrouted slots, CUDA graph replay bit-exact), then `EXL3_BENCH=1 ... -k bench` for decode GB/s per layer vs 273 GB/s with 384 experts at a 2.76-bpw mix. This answers risk 1 and 2 within an hour.
3. **Registration in the runtime image:** `tests/test_vllm_registration.py`.
4. **Dummy-load G0 at TP=2:** synthetic 2.76-bpw plan, `tools/make_dummy_model_dir.py`, `--load-format dummy`, FULL_DECODE_ONLY, capture sizes <= 16. Read per-rank MemAvailable after load (T3 >= 4 GiB) and decode tok/s with DSpark off and on: the ceiling of this route before any weights exist.
5. **Two real layers:** EXL3-encode layers 0-1 from the official MXFP4 experts (P4 harness), truncated checkpoint, `--hf-overrides '{"num_hidden_layers": 2}'`, logits vs the native run.

---

## 12. Licenses and attribution

- The plugin: Apache-2.0 (repo license, T9). It calls exllamav3 (MIT, turboderp) through its pybind API; credit turboderp-org/exllamav3 in NOTICE/README. A kernel patch to exllamav3 (section 9 item 1) stays MIT and goes upstream to turboderp.
- vLLM: Apache-2.0, used as is.
- `ref:` (0xSero/local-ai-images, no license): ideas cited (per-K buckets, priming before capture, `scatter_add` instead of `bincount`, prefill grouped-GEMM shape), no code copied. The local-inference-lab fork r38 was not needed for this design.
