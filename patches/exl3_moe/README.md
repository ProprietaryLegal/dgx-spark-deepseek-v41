# vllm_exl3_moe: EXL3 routed experts for DeepSeek-V4.1-Flash on upstream vLLM

A vLLM **general plugin** (no source patch to vLLM) that serves DeepSeek-V4.1-Flash with its 15,360 backbone routed experts stored as exllamav3 EXL3 trellis tensors at a per-expert bit mix (K = 2/3/4/5, about 2.6-2.8 bpw), while attention, the shared expert, router, Engram, vision and the DSpark drafter keep the official checkpoint's own FP8 / MXFP8 / MXFP4 handling.

- Engine: upstream vLLM `main @ 6e517b15c1833cf72a7f557ee32524d98682e617`.
- Kernels: exllamav3 **v1.6.0** (`turboderp-org/exllamav3 @ 151539c77abc7ab7425d30da7a4e8e3c5c154e7b`, MIT), called through its compiled extension `exllamav3_ext`. No exllamav3 code is copied here.
- TP layout: vLLM's own MoE tensor-parallel split. At TP=2 each rank keeps intermediate channels `[r*1152, (r+1)*1152)` of **every** expert, cut from the full EXL3 tensors at load time (exact: the cut is 128-aligned, see `docs/exl3_moe_design.md` section 5).
- Status (Oct 8 2026, GB10): GPU tests pass in the serve image: `test_kernels_gpu.py` 15/15 (coop / fused / prefill vs the reconstruct reference at K 2-5, TP halves == unsplit, graph replay bit-exact), `test_dense_gpu.py` 8/8, `test_head_mxfp8.py` 6/6, `test_head_exl3_gpu.py` 3/3, `test_parallel_groups_gpu.py` 6/6 (concurrent K groups == chained, bit for bit). A 2-layer real-weight checkpoint (EXL3 experts + K5 dense) served by vLLM matches the reference running the same weights: mean |d logprob| 0.04-0.12, 20/20 top-20 shared (the native engine's own 2-layer gap is 0.06-0.14), CUDA-graph greedy tokens identical to eager. A dummy-weight TP=2 boot of the full plan fits and decodes. The full model has served since Oct 9; build v3 decodes 103.0 / 58.3 / 141.3 tok/s (kit_bench set b, code / prose / structured) on two Sparks.

Design, citations and the cost model: `../../docs/exl3_moe_design.md`.

## Layout

| file | what |
|---|---|
| `vllm_exl3_moe/__init__.py` | `register()`, the `vllm.general_plugins` entry point (no-op with a warning when vLLM is absent) |
| `vllm_exl3_moe/config.py` | `DeepseekV41Exl3Config(DeepseekV4FP8Config)`, quant_method `deepseek_v41_exl3` |
| `vllm_exl3_moe/moe_method.py` | `Exl3RoutedMoEMethod(FusedMoEMethodBase)`: `create_weights`, weight loader, `process_weights_after_loading`, `apply` |
| `vllm_exl3_moe/bank.py` | one layer's bank on one rank: K-sorted slots, flat trellis banks, TP cut, pointer tables (torch only) |
| `vllm_exl3_moe/forward.py` | `routed_moe_forward()`: decode (coop), graph-safe fused, eager prefill; `reference_moe()` for tests |
| `vllm_exl3_moe/kernels.py` | thin wrappers over `exllamav3_ext` (`exl3_moe_coop`, `exl3_moe`, `reconstruct`, `had_r_128`, `hgemm`), scratch, priming |
| `vllm_exl3_moe/plan.py` | per-expert K plan, bank layout, TP slice math (pure Python) |
| `vllm_exl3_moe/router.py`, `csrc/exl3_router.cu` | MoE gate GEMM at decode sizes on sm_12x (`EXL3_ROUTER=1`): an out-of-tree GateLinear (vLLM's own tiers skip GB10 and leave cuBLAS + a split-K reduce), plus an optional fused DeepSeek-V4 top-k |
| `tests/test_router_gpu.py`, `tests/bench_router.py` | any sm_12x GPU: router kernel vs fp64 and the dsv4 routing math (ties, NaN, image rows), graph replay; timing vs cuBLAS |
| `tests/test_router_vllm_gpu.py` | inside the runtime image: the GateLinear replacement and its fallbacks |
| `tests/test_plan.py`, `tests/test_tp_split_math.py` | run anywhere (pytest, numpy) |
| `tests/test_bank_cpu.py` | bank layout, TP cut and pointer tables on CPU tensors (needs torch, CPU build is enough) |
| `tests/test_kernels_gpu.py` | Spark: kernels vs reconstruct reference, TP=2 split, K groups, CUDA graph replay, bandwidth bench |
| `tests/test_vllm_registration.py` | inside the runtime image: registration and config resolution |
| `tools/make_plan.py` | synthetic plans (tests only) and plan reports (bpw, GiB per rank, K groups) |
| `tools/make_dummy_model_dir.py` | weightless model dir for `--load-format dummy` fit / speed probes |
| `tools/cost_model.py` | routed-expert decode bytes / ms per step vs tokens per step (design doc section 8) |

## Checkpoint format the plugin reads

Same keys exllamav3 writes for DeepSeek-V4 experts, one EXL3 linear per projection:

```
layers.{L}.ffn.experts.{E}.w1.trellis  int16 [320, 144, 16*K]   gate, stored (in=5120, out=2304)
layers.{L}.ffn.experts.{E}.w1.suh      fp16  [5120]
layers.{L}.ffn.experts.{E}.w1.svh      fp16  [2304]
layers.{L}.ffn.experts.{E}.w1.mul1     int32 []                  codebook marker (or .mcg)
layers.{L}.ffn.experts.{E}.w3.*        same as w1                up
layers.{L}.ffn.experts.{E}.w2.trellis  int16 [144, 320, 16*K]   down, stored (in=2304, out=5120)
layers.{L}.ffn.experts.{E}.w2.suh      fp16  [2304]
layers.{L}.ffn.experts.{E}.w2.svh      fp16  [5120]
layers.{L}.ffn.experts.{E}.w2.mul1     int32 []
```

Everything else is the official `deepseek-ai/DeepSeek-V4.1-Flash @ 2cba9e42` tensors, unchanged, including `mtp.*` (DSpark drafter, MXFP4 experts) and the Engram tables. The MXFP4 routed-expert tensors of planned layers must **not** be in the shards (vLLM yields every tensor of every listed shard; there is no param for them).

`config.json`:

```json
"quantization_config": {
  "quant_method": "deepseek_v41_exl3",
  "activation_scheme": "dynamic", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4",
  "exl3_moe": {"plan_file": "exl3_moe_plan.json", "decode_kernel": "coop", "coop_max_tokens": 32,
               "fused_rows": 256, "tp_layout": "channels", "exllamav3_version": "1.6.0"}
}
```

`exl3_moe_plan.json` (`plan.py`): `format`, `version: 1`, `codebook` (`mul1`), `num_layers` 40, `num_experts` 384, `hidden_size` 5120, `moe_intermediate_size` 2304, and `layers: {"0": {"k": "2352...", "k_down": optional}, ...}` with one character per expert (`1`-`8`, or `a`/`b`/`c` = 1.5/2.5/3.5). Layers missing from the plan stay on the native MXFP4 method. The loader checks every trellis shape against the plan and refuses a half-loaded layer.

## Install into the runtime image

Build order (one build at a time on a Spark, after the vLLM wheel):

```bash
# 1. exllamav3 v1.6.0 extension for GB10 (sm_121), in the same build env as vLLM (torch 2.13.0+cu130, nvcc 13.0)
git clone --branch v1.6.0 https://github.com/turboderp-org/exllamav3 && cd exllamav3
git rev-parse HEAD   # must print 151539c77abc7ab7425d30da7a4e8e3c5c154e7b
TORCH_CUDA_ARCH_LIST="12.1" MAX_JOBS=8 pip wheel --no-build-isolation --no-deps -w ../wheels .
# after installing the wheel (below): the compiled module must carry sm_121 SASS only
cuobjdump --list-elf "$(python -c 'import torch, exllamav3_ext; print(exllamav3_ext.__file__)')" | sort -u

# 2. this plugin (pure Python)
pip wheel --no-deps -w ../wheels /path/to/dgx-spark-deepseek-v41/patches/exl3_moe
```

Runtime image (`FROM` the vLLM image we built):

```dockerfile
COPY wheels/exllamav3-1.6.0-*.whl wheels/vllm_exl3_moe-*.whl /tmp/
RUN pip install --no-deps /tmp/exllamav3-1.6.0-*.whl /tmp/vllm_exl3_moe-*.whl
```

`--no-deps` on both: the plugin only imports `exllamav3_ext` (the top-level compiled module), not the `exllamav3` Python package, so none of exllamav3's runtime dependencies are needed for serving, and pip must not touch torch / NCCL. For development, `pip install --no-deps -e /path/to/patches/exl3_moe` inside the container works the same way.

The entry point (`[project.entry-points."vllm.general_plugins"] exl3_moe = "vllm_exl3_moe:register"`) is picked up by every vLLM process automatically. `VLLM_PLUGINS=exl3_moe` restricts loading to this plugin if others are installed.

## Serve flags that matter

- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` and `--max-cudagraph-capture-size` at or below `coop_max_tokens` (32 by default; 2 streams x DSpark k=7 = 16 rows). Bigger captured shapes fall back to the graph-safe fused path (needs rows <= `fused_rows`) or raise.
- No `--enable-expert-parallel` (v0 implements the TP channel split only).
- `EXL3_ROUTER=1`: decode-size gate GEMMs (up to 8 rows) on `exl3_router`; off by default until measured on the serve.
- `VLLM_EXL3_MOE_SMOE_SHARED=1`: the layer's shared expert runs inside exl3_smoe as one more pick per token (weight 1), instead of its own EXL3 dense launches on vLLM's aux stream (the quant method reports `mk_can_overlap_shared_experts`, so the runner hands it over); off by default until measured. Build flag `-DSMOE_RUNFLAGS=1` (`EXL3_SMOE_FLAGS`): down waits per run.
- Runtime knobs (identical on every rank): `VLLM_EXL3_MOE_DECODE_KERNEL=coop|fused`, `VLLM_EXL3_MOE_COOP_MAX_TOKENS`, `VLLM_EXL3_MOE_FUSED_ROWS`; exllamav3's own tuning envs `EXL3_MOE_COOP_WIDE`, `EXL3_MOE_COOP_KSPLIT`, `EXL3_MOE_TILE_N`.

## How to test

Without a GPU or vLLM:

```bash
cd patches/exl3_moe
python3 -m py_compile vllm_exl3_moe/*.py tests/*.py tools/*.py
python3 -m pytest -q tests/                  # plan + TP-split math; torch / GPU / vLLM tests skip
<venv with torch>/bin/python -m pytest -q tests/   # + bank layout on CPU (GPU and vLLM tests skip)
python3 tools/make_plan.py --counts 2:6157,3:7688,4:1481,5:34   # plan v2's K mix (the real plan comes from p4_allocate.py)
```

On a Spark, in order (each step gates the next):

1. **Kernels, no vLLM** (one GPU, nothing else heavy running):
   `python -m pytest -q tests/test_kernels_gpu.py` then `EXL3_BENCH=1 python -m pytest -q -s tests/test_kernels_gpu.py -k bench`.
   Pass: coop / fused / prefill vs the reconstruct reference within 2e-2 relative error at every K; TP rank-0 + rank-1 == unsplit; CUDA graph replay bit-exact; bench prints decode GB/s per layer for 1..16 rows against the 273 GB/s roofline.
2. **Registration** in the runtime image: `python -m pytest -q tests/test_vllm_registration.py`.
3. **Dummy-load G0 at TP=2** (memory fit and decode ceiling without any quantized weights):
   `tools/make_plan.py --counts 2:6157,3:7688,4:1481,5:34 -o /tmp/plan.json`,
   `tools/make_dummy_model_dir.py --src <official snapshot> --plan /tmp/plan.json --out ~/models/dsv41-exl3-dummy`,
   then the launch layer with `--load-format dummy`; record per-rank MemAvailable after load (the fit needs >= 4 GiB) and decode tok/s with DSpark off and on (text is garbage by construction).
4. **Real weights, 2 layers** (passed Oct 8): EXL3-encode layers 0-1 from the official MXFP4 experts (P4 tooling), assemble a truncated checkpoint (`p4_assemble.py --test-layers 2`), serve it and compare logits against the reference running the same EXL3 weights: mean |d logprob| 0.04-0.12, 20/20 top-20 shared.

## Attribution and licenses

This package: Apache-2.0. It calls exllamav3 (MIT, Copyright turboderp) kernels through their public pybind entry points; credit turboderp-org/exllamav3 wherever this ships. The reference image `0xSero/local-ai-images` (no license) was read for ideas only and is cited, never copied.
