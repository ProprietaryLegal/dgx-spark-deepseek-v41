# dgx-spark-deepseek-v41

> **PLI Labs fork, branch `pli/dgx-spark-offload`** (on top of upstream `cc59c36`, v1.1). Two small launcher changes,
> both inert by default: `EXTRA_DOCKER_ARGS` (extra `docker run` arguments on both ranks, used to bind-mount offload
> overlays) and `SERVE_DIR` (keep the generated launch file out of `/tmp`). They support PLI Labs' x86-host offloads for
> this recipe — Engram tables from host RAM over RDMA (TTFT −5.4 % @32K, −5.7 % @150K) and the DSpark drafter on an
> RTX 3090 with the freed memory spent on KV (989,460 → 1,604,926 tokens; 4 × 300K concurrent). Measured with our own
> suite, not yet with this repo's `kit_bench`/oracle. Entries:
> [combined](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) ·
> [findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings) ·
> [GitHub](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload). All credit for the recipe
> to sudoingX; changes here are Apache-2.0 like the rest of the repo.

<!-- pli:blurb:start -->

> This fork is maintained by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai); the upstream work and its credit belong to sudoingX. PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work.

**PLI Labs standalone repositories for these offloads:** [stack overview](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · [async hand-off](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff) · [remote RTX 3090 drafter](https://github.com/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) · [remote Engram](https://github.com/ProprietaryLegal/dsv41-remote-engram-rdma) · [Hugging Face collection](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e)

<!-- pli:blurb:end -->

<!-- pli:update:2026-10-27:start -->

**Update 2026-10-27:** the full DSV41 source is now public — [dsv41-offload](https://github.com/ProprietaryLegal/dsv41-offload) (every component; `production`, `main` and per-experiment `lane/*` branches), with focused repos for the [measurement harness](https://github.com/ProprietaryLegal/dsv41-window-harness) and [sampled drafting](https://github.com/ProprietaryLegal/dsv41-sampled-drafting). Status and lessons: [2026-10-27 post](https://github.com/ProprietaryLegal/pli-labs/blob/main/research/dgx-spark-offload/2026-10-27-status-and-code.md).

<!-- pli:update:2026-10-27:end -->



[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

DeepSeek-V4.1-Flash on two NVIDIA DGX Sparks, from weights we quantized out of the official checkpoint, served by upstream vLLM with our patches and one plugin (our own decode kernels), tensor parallel 2 over the ConnectX link. On the same two boxes and client it decodes faster than TensorFold v0.6.0, its newest engine, on code and structured output, with weights closer to the native model than any other pack we measured.

The official checkpoint (`deepseek-ai/DeepSeek-V4.1-Flash` @ `2cba9e42`) is 510 GB: 289 GB of MXFP4 routed experts, 203 GB of Engram tables, 18.5 GB of everything else. Two Sparks have 256 GB between them. This repo makes it fit and run:

- the 15,360 routed experts re-encoded in EXL3 at 2, 3, 4 or 5 bits per weight, chosen per expert from measured sensitivity, 2.70 bpw on average (289 GB down to about 184 GB),
- the Engram tables read from the checkpoint files on NVMe at lookup time, CUDA-graph safe, so they never sit in memory,
- the dense linears and the LM head on EXL3 too (K4, `wo_a` K5, head K4), and the DSpark drafter's dense layers (K5) and routed experts (K3), all through our decode kernels `exl3_mrow` and `exl3_smoe`,
- per-round DSpark draft depth from a measured cost table, TensorFold's RDMA all-reduce ring, paced L2 prefetch and, since v1.1, its grouped EXL3 prompt kernels for the routed experts (Apache-2.0, credited), FlashInfer autotune off.

Weights: [`sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw`](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw) (196 GB; the two Engram shards come from DeepSeek's own repo, quickstart step 3).

Build: v3 (`dsv41-recipe-exl3-v3b-dk5-h4-dx3` on our boxes; see [How it works](#how-it-works) for what v2, v2.1 and v3 contain).

**v1.1 (Oct 13 2026), same weights:**
- **Prefill:** TensorFold's grouped EXL3 prompt kernels for the routed experts are on by default (`VLLM_EXL3_MOE_TF_PROMPT=1`). Prefill is 2,090 / 2,092 tok/s at 8K / 32K (v1: 1,763 / 1,773, so +18.5% / +18.0%). A 255,709-token prompt's first token arrives in 159.6 s (v1: 183.3 s).
- **Decode is unchanged:** set b 102.7 / 57.9 / 141.0 tok/s on a fresh v1.1 boot, and greedy output is identical to v1 on all 24 oracle prompts.
- **Boots:** the JIT kernel builds now stay in the mounted cache, so a boot no longer recompiles them.
- **To update:** `scripts/launch/dsv41_tp2.sh down`, then `git pull` and `./install.sh` in `~/dsv41_recipe` (images only, the wheels are kept), then `up` again.

> **Target hardware:** 2x NVIDIA DGX Spark. GB10 Grace-Blackwell, compute `sm_121`, 128 GB unified LPDDR5X each, about 273 GB/s, a ConnectX-7 cable between the two boxes (RoCE v2), DGX OS (Ubuntu 24.04), CUDA 13.0, driver 580.x (tested on 580.142 and 580.173.02), aarch64. Tensor parallel 2, one rank per box.

## Results

Every number below was measured on the same two Sparks. Comparison builds were run by us on these boxes with the same client; no number is copied from another README.

### Speed

`kit_bench.py` set b: single stream, greedy, thinking off, 384-token replies, median of 9 reps a prompt (Mia-AiLab's kit: 3). DSpark speculative decoding on (k = 5 with per-round depth) unless the row says off. TensorFold is bertholomus's engine (v0.6.0, TensorFold 1.0's native Zig engine, built from source at `ce20a54`) serving Mia-AiLab's 2.9 bpw pack, run by us on these boxes with the same client; two network constants changed for our fabric (path MTU 1024, RoCE GID index 3; his are hardcoded for jumbo frames and index 5).

| | this build | TensorFold v0.6.0 + Mia 2.9 bpw | Mia-AiLab vLLM kit (2.9 bpw pack) | 0xSero build |
|---|---|---|---|---|
| decode, code, DSpark on | **103.0 tok/s** | 100.5 tok/s | 41.6 tok/s (k = 3) | not measured |
| decode, prose, DSpark on | 58.3 tok/s | **62.1 tok/s** | 25.1 tok/s (k = 3) | not measured |
| decode, structured, DSpark on | **141.3 tok/s** | 139.0 tok/s | 40.4 tok/s (k = 3) | not measured |
| DSpark acceptance (accepted / drafted) | 49.3%, 2.46 accepted + 1 a verify step | not exposed by the engine | 49.2%, 1.47 + 1 (k = 3) | not measured |
| decode, code / prose / structured, DSpark off | 44.6 / 44.4 / 44.5 tok/s | not measured | not measured | not measured |
| decode, set a, code / prose / structured | 86.6 / 54.2 / 103.7 tok/s | not measured | not measured | not measured |
| 2 streams at once (code + prose, 256 tokens each), aggregate | 81.6 / 80.2 tok/s (2 reps) | not measured | not measured | not measured |
| prefill, 8K / 32K prompt (`kit_bench.py prefill`, median of 3) | 2,090 / 2,092 tok/s (v1.1; v1: 1,763 / 1,773) | **2,163 / 2,276 tok/s** | not measured | not measured |
| decode after a 32K prompt | 81.7 tok/s (a needle-ladder document, 512 tokens); 128.2 tok/s on `kit_bench.py depth` (random-token prompt, 256 tokens; this one swings 58-128 tok/s run to run with the continuation's DSpark acceptance, 25-68%) | not measured | not measured | not measured |

Raw files and method for every row: [BENCHMARKS.md](BENCHMARKS.md). Greedy output is identical run to run on all three prompts for this build and for TensorFold. The rows from set a down come from one run of `scripts/bench/serve_speed_suite.sh` on the recipe defaults (Oct 12 02:14-02:37; its set b measured 102.2 / 58.1 / 142.6), DSpark off from a boot with `SPEC=` (empty). The prefill row is its own run, both engines on fresh boots with the same client, 3 reps each (Oct 12 13:21-13:36); the first 8K rep of each reads Engram's rows cold (ours 1,407, TensorFold 2,111 tok/s). Only 2 sequences are served at once (`MAX_SEQS=2`, for the 262K context); TensorFold serves 4. The decode rows are generated from the raw result files by `scripts/bench/results_table.py` (this build: the recipe defaults, median of 9 reps a prompt, Oct 12 00:23; every rep within 102.4-103.4 / 58.1-58.6 / 140.8-142.1). TensorFold's column is its 9-rep median (Oct 12, same session), with slow reps on all three prompts (ours: none in 27); its 3-rep run measured 101.0 / 62.6 / 140.3. Keep the page-cache guard off while serving (`guards.sh serve`): its drops evict Engram's hot rows and gave structured an intermittent ~5% slow mode.

### Quality

KL divergence and top-1 agreement against the native model on our panel: 199 conversations, 254,846 scored assistant positions, every build scored by DeepSeek's own reference implementation (`inference/model.py`). Method in [Quality method](#quality-method).

| | this build | Mia-AiLab 2.9 bpw | 0xSero J268 |
|---|---|---|---|
| routed-expert bits, memory / per decoded token | 2.70 / 2.90 | 2.875 / 2.875 | 2.43 / 2.77 |
| KLD vs native (mean over positions, nats) | **0.0243** | 0.0295 | 0.0327 |
| KLD vs native, the 8 domains weighted equally | **0.0398** | 0.0443 | 0.0516 |
| KLD p99 | **0.350** | 0.392 | 0.492 |
| top-1 agreement | **95.57%** | 94.99% | 95.00% |
| perplexity on the panel (native 1.4978) | **1.5167** | 1.5221 | 1.5277 |

By domain, KLD (lower is closer to native):

| domain | this build | Mia-AiLab 2.9 bpw | 0xSero J268 |
|---|---|---|---|
| code (65% of positions) | **0.0153** | 0.0204 | 0.0214 |
| agentic (7% of positions) | **0.0201** | 0.0309 | 0.0221 |
| math (4% of positions) | **0.0189** | 0.0215 | 0.0312 |
| long (6% of positions) | **0.0306** | 0.0348 | 0.0369 |
| knowledge (6% of positions) | **0.0450** | 0.0515 | 0.0522 |
| writing (2% of positions) | **0.0578** | 0.0676 | 0.0697 |
| multilingual (3% of positions) | 0.0692 | **0.0579** | 0.0894 |
| chat (7% of positions) | **0.0613** | 0.0695 | 0.0901 |

This build is closest to native over all positions, with every domain weighted equally, and on 7 of the 8 domains; Mia-AiLab's pack is closer on multilingual. On a second panel of native IFEval-style replies (300 sequences) the KLD is 0.1266 for this build against 0.1302 for Mia-AiLab's pack. Generated from the raw result files by `scripts/bench/results_table.py`.

Bits per decoded token weight each expert by how often it is routed (pass-0 routing counts on our calibration set). Mia-AiLab's and 0xSero's bit widths are read from the headers of their published files.

Suites, greedy, thinking off, 2,048 max tokens, paired per item against the native model served on the official weights. A suite fails if the drop is larger than 1 point at p < 0.05.

| suite | native | this build | TensorFold v0.5.1 + Mia 2.9 bpw |
|---|---|---|---|
| HumanEval+ (164), base / plus | 97.6 / 92.1 | 97.0 / 90.9 (within noise) | 95.7 / 90.9 |
| MBPP+ (378), base / plus | 91.0 / 77.5 | 91.3 / 78.6 (within noise) | 92.1 / 78.8 |
| IFEval (541), prompt strict / prompt loose | 89.5 / 91.5 | 87.6 / 90.4 (paired vs native p = 0.13, vs TensorFold p = 0.29: within noise) | 89.1 / 91.1 |

### Fit and stability

| | this build |
|---|---|
| context per request | 262,144 tokens, needle ladder 8K / 32K / 64K / 128K / 256K: 3/3 at every depth |
| the 256K rung (a 255,709-token prompt) | v1.1: cold prefill 159.6 s (1,602 tok/s), needles 3/3, decode after it 72.3 tok/s (512 tokens), lowest MemAvailable during it 6.5 GiB. v1: 183 s (1,395 tok/s), 76.4 tok/s, 7.5 GiB |
| MemAvailable after load, head / worker | 8 / 9 GiB (v1.1; v1: 9 / 9) |
| OpenAI API check (tools, streaming, reasoning) | 14 of 14 |
| 2-hour agent soak | not run yet |

## Quickstart

You need two DGX Sparks cabled with ConnectX-7, an IPv4 address on `enp1s0f0np0` on both, Docker on both, and a control machine (one of the Sparks works) that can `ssh` to both without a password. About 430 GB free on each Spark at the same path: the model folder is 399 GB (196 GB of our weights + DeepSeek's 203 GB Engram tables, which stay in the official files) and the serving image about 25 GB.

```bash
# 1. this repo, on both Sparks (the scripts expect ~/dsv41_recipe)
git clone https://github.com/sudoingX/dgx-spark-deepseek-v41 ~/dsv41_recipe

# 2. engine images, on one Spark with nothing else heavy running on either: the vLLM and exllamav3 wheels for sm_121
#    (memory-capped CPU compiles), the three images, then the serving image copied to the second Spark
cd ~/dsv41_recipe && PEER=<second-spark> ./install.sh   # every step it runs: GUIDE.md, section 1
HEAD=<ssh-host-of-rank-0> WORKER=<ssh-host-of-rank-1> ./check.sh   # read-only, PASS / WARN / FAIL per check

# 3. weights, on both Sparks at the same absolute path
hf download sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw --local-dir ~/models/dsv41-exl3
hf download deepseek-ai/DeepSeek-V4.1-Flash model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027 --local-dir ~/models/dsv41-exl3   # Engram tables, 203 GB
(cd ~/models/dsv41-exl3 && sha256sum -c MANIFEST.sha256)

# 4. serve, from the control machine
HEAD=<ssh-host-of-rank-0> WORKER=<ssh-host-of-rank-1> MODEL_DIR=/absolute/path/to/dsv41-exl3 \
  ./scripts/launch/dsv41_tp2.sh up      # starts both ranks, waits for /v1/models, prints MemAvailable per node
./scripts/launch/dsv41_tp2.sh status    # logs [head|worker] | down
```

The server speaks the OpenAI API on the head node, port 8000, model name `dsv41`, with V4.1's tool-call and reasoning parsers on. The second Spark can copy the weights from the first over the ConnectX link instead of downloading them again (`rsync` ran at about 550 MB/s in our runs).

Measure it the way the table above was measured, with bertholomus's `kit_bench.py` (standard library only; not vendored here):

```bash
curl -LO https://raw.githubusercontent.com/bertholomus/TensorFold/fe6f9f7b373db76422ddfa121577238474c0a1b7/tools/dsv41/kit_bench.py
sha256sum kit_bench.py   # 83a4fbeb987032f906cf00d3f6c9dd2fad47dd675e75dd498c4c4188d63a6c0d, the file every number here used
python3 kit_bench.py --base http://<head>:8000 --model dsv41 decode --set b --tokens 384 --reps 3
SERVE_ID=mine KIT=/path/to/kit_bench.py HEAD=<head> ./scripts/bench/serve_speed_suite.sh   # the whole speed table
```

### Launcher settings

Environment variables read by `scripts/launch/dsv41_tp2.sh`, same values on both ranks:

| variable | default | what |
|---|---|---|
| `MODEL_DIR` | required | assembled checkpoint, same absolute path on both Sparks |
| `HEAD`, `WORKER` | `spark`, `spark2` | ssh hosts of rank 0 and rank 1 |
| `FAB_IF`, `FAB_HCA` | `enp1s0f0np0`, `rocep1s0f0` | fabric interface and its RoCE device; NCCL is pinned to it with the GID index of its IPv4 |
| `IMAGE` | `dsv41-serve:v4` | serving image (quickstart step 2 builds it) |
| `MAX_LEN`, `MAX_SEQS`, `MAX_BATCH` | 262144, 2, 2048 | context per request, concurrent sequences, prefill chunk |
| `UTIL` | 0.88 | `--gpu-memory-utilization` |
| `KV_BYTES` | 2684354560 | KV pool per rank pinned at 2.5 GiB (empty: sized from `UTIL`) |
| `RECIPE_ENV` | the measured-best knobs | our kernel, prefetch, all-reduce and DSpark settings with the round-cost table, listed in the launcher (v1.1 adds `VLLM_EXL3_MOE_TF_PROMPT=1`, the prompt kernels, and `TORCH_EXTENSIONS_DIR` in the mounted cache); empty runs without them |
| `SPEC` | DSpark, 5 tokens | `--speculative-config` JSON, empty = off |
| `GRAPH` | `FULL_DECODE_ONLY` | CUDA graphs for decode only |
| `ENGRAM` | disk offload, 256 reader threads | `--engram-config` JSON |
| `KERNEL_CONFIG` | FlashInfer autotune off | `--kernel-config` JSON |
| `TEXT_ONLY` | 1 | `--limit-mm-per-prompt '{"image":0}'`: skips the vision tower (about 0.9 GiB per rank); 0 serves images |
| `VLLM_DISABLED_KERNELS` | `FlashInferCutlassMxfp8LinearKernel` | sends every MXFP8 linear to Marlin |
| `DOCKER_ENV` | empty | extra container env, after `RECIPE_ENV` so it overrides a knob there, e.g. `VLLM_EXL3_HEAD=mxfp8` |
| `EXTRA` | empty | extra `vllm serve` flags |

## How it works

### Bits per expert from measured sensitivity

Each of the 15,360 routed experts (40 layers x 384) gets K = 2, 3, 4 or 5 bits per weight. Importance of an expert = the measured sensitivity of its layer x the expert's share of the routed output, per token.

- **Layer sensitivity** (`p4_sensitivity.py`): KL at the model output per unit of relative noise injected into one layer's routed-expert output. It falls from 4.71 at layer 0 to 0.019 at layer 39; layers 0-19 carry about 88% of it.
- **Per-token share.** Every V4.1 layer has a "super expert" that writes very large outputs on a handful of tokens. In layer 1 one expert holds 92.3% of the routed output energy and 1.4% of the per-token share. Our first plan priced experts by energy: it gave that expert K5, pushed most of layer 1 to K2, and the layer's routed-output error came out at 6.4-7.4% (layer 0: 1.2-1.8%). The allocator now prices the per-token share and keeps the 34 super experts (20% or more of their layer's routed energy) at K5 anyway.
- **DSpark target layers.** The drafter reads layers 37, 38 and 39; their experts stay at K3 or more.
- **Measured error per K** (layer 0, full calibration set, median relative output error): K2 16.2%, K3 6.5%, K4 2.4%, K5 1.0%.

Result for v3 (`data/plans/plan_v3_270.json`, importance mixed over the calibration domains): K2 6,157, K3 7,689, K4 1,479, K5 35 experts. 2.70 bpw in memory (85.75 GiB per rank), 2.90 bpw per decoded token. (v2's plan, `plan_v2_270.json`, had the same budget: K2 6,157, K3 7,688, K4 1,481, K5 34.)

### Sequential encode

`p4_pass1.py` encodes one layer at a time with exllamav3's EXL3 trellis quantizer (v1.6.0, mul1 codebook). Each layer is calibrated on the hidden states that come out of the already-quantized layers before it. Each expert gets its own Hessian from the tokens routed to it, weighted by their routing weights, plus an all-token prior so rarely routed experts stay well posed. The two Sparks split every layer's experts and exchange partial MoE outputs over the ConnectX link.

### The engine on GB10

| change | measured on GB10 | where |
|---|---|---|
| Marlin for every MXFP8 linear | at decode vLLM's default MXFP8 GEMM (FlashInfer CUTLASS) reads weights at 66-78 GB/s, Marlin at 200-251 GB/s (distinct weights cycled from DRAM). At a 2,048-token prefill chunk CUTLASS is faster (wq_b 953 us vs 1,563 us). | `VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel` |
| FP8 `wo_a` as one Marlin GEMM per head group (v2; in v3 only the drafter's `wo_a` is still FP8) | the stock path makes a BF16 copy at load (0.67 GiB more per rank); Marlin takes 21.4 us per group at one token, the CUTLASS MXFP8 GEMM 65.8 us | patch 0004 |
| Engram tables on disk | rows read from the official shards at lookup time by a pread pool into pinned staging, instead of about 100 GB of pinned host memory per rank; 4,096 cold rows in 5.8 ms at 256 reader threads (80.2 ms at 32) | patch 0002 |
| FlashInfer autotune off | with autotune on, MemAvailable fell from 4-5 GiB to 0.8 GiB in about 5 minutes of mixed requests | launcher default |
| page cache dropped before launch | CUDA allocations on GB10 do not reclaim page cache: a 67 MB allocation failed with 45 GB "available" and 8 GB free | launcher, needs `sudo -n` for one command |
| optional: MXFP8 head through Marlin | 2.83 ms to 1.48 ms per head call; per-weight error 2.65% | `VLLM_EXL3_HEAD=mxfp8`, patch 0005 |
| EXL3 dense linears and head (v3: K4, `wo_a` K5, head K4) | on `exl3_mrow` (our decode-size EXL3 linear: TMA k-rows into an mbarrier ring) at 186-229 GB/s; K5 was already 1.2-1.45x Marlin per dense linear (wq_b 67 us vs 98 us); K4 dense + K5 head measured 1.7 ms a round faster than K5 dense + K6 head; the K4 head shipped after a KLD check (0.0243 vs 0.0236), its own speed share not measured separately; activations in BF16 (`EXL3_INT8_GEMV=0`) | patches 0006, 0007, plugin `dense.py` / `mrow.py` |
| DSpark drafter on EXL3 (dense K5, routed experts K3) | the drafter runs its 3 layers once a round over a 5-token block; on deep_gemm MXFP4 its experts cost ~1.5 ms a round, on `exl3_smoe` at K3 rounds got 1.1-1.3 ms faster with drafter acceptance unchanged (48.5% vs 48.3% per drafted token) | plugin, plan `drafter` section |
| concurrent K groups and wide tiles | a layer has up to four K groups; they run on side streams, bit-exact to the chained launches. Measured together with the wide coop tiles: 1.09-1.27x a layer at 4-6 rows (graph replay, one TP=2 shard; the K2-heavy layers gain most) | launcher default (`VLLM_EXL3_MOE_PARALLEL_GROUPS=1`, `EXL3_MOE_COOP_WIDE=1`) |

Builds: **v2** = EXL3 routed experts, everything else at the checkpoint's own precision (FP8 dense, BF16 head). **v2.1** = v2 + EXL3 K5 for `wq_b`, `wo_a`, `wo_b` and the shared expert, and an EXL3 K6 head. **v3** (this release) = routed experts re-encoded on a broader calibration (instruction, chat, writing and multilingual text added), dense K4 with `wo_a` K5, an EXL3 K4 head, and the DSpark drafter on EXL3 (dense K5, routed experts K3).

### Patches

Upstream vLLM main @ `6e517b15c1833cf72a7f557ee32524d98682e617`. Every patch file starts with what it does, why, and where it came from. "Recipe" says whether the launcher's defaults turn it on (most later patches are opt-in by an environment variable and were kept only where an A/B on these two Sparks picked them).

| patch | origin | what | recipe |
|---|---|---|---|
| 0000 | ours | CMake note: add compute 12.1 to the arch list (applied by the build script) | build |
| 0001 | upstream PR #56509 | SWA and indexer page geometry that FlashInfer's SM120 sparse MLA needs on GB10 | on |
| 0002 | ours | Engram tables read from the checkpoint shards at lookup time, CUDA-graph safe | on |
| 0003 | upstream PR #58560 | compressor ring kept out of the null KV block (NaN from layer 0 on GB10 without it) | on |
| 0004 | ours | `wo_a` on SM12x: FP8 kept, one Marlin MXFP8 GEMM per head group | on |
| 0005 | ours | the lm_head gets the model's quant config, so a plugin can serve an MXFP8 or EXL3 head | on |
| 0006 | ours | hooks for non-FP8 dense linears: `wo_a` grouped bmm, shared-expert padding that leaves 1-D tensors alone | on |
| 0007 | ours | `head.{trellis,suh,svh,mul1}` map to the lm_head the way `head.weight` does | on |
| 0008 | ours | DSpark top-k Markov bias for the V4.1 drafter | off (the local argmax replaces it) |
| 0010 | ours | per-round DSpark verify depth from the confidence head and a measured round-cost table | on (`conf`) |
| 0011 | ours | DSpark Markov bias rows cached for the most frequent tokens | on |
| 0012 | ours | greedy drafts argmaxed per vocabulary shard: one (value, index) gather instead of the full draft logits | on |
| 0013 | ours | remote MessageQueue reader busy-polls before sleeping | off (flat) |
| 0014 | ours | mHC mix statistics overlapped on a side stream in FULL decode graphs | on |
| 0015 | ours | Engram disk reads in spin mode | on |
| 0016 | ours | hook for out-of-tree MoE methods that run a whole decode MoE phase | off (slower) |
| 0017 | ours | hands the loaded model to the plugin's L2 prefetch (mix weights and router gates) | on |
| 0018 | ours | router gate before the shared-expert fork | off (flat) |
| 0019 | ours | short-context CUDA graph twins that skip the sparse indexer's scoring while every entry is selected | on |
| 0020 | ours | verify rows through FlashInfer's one-row decode kernel | off (its hunk sits in the non-SM12x class) |
| 0021 | ours | mHC post + input collapse + RMSNorm in one plugin launch | off (flat) |
| 0022 | ours, `patches/experimental/` | verify rows through the plugin's Triton sparse attention instead of FlashInfer | not in the image (flat: 2.8x faster alone, the window is bytes-bound) |

0009 (eager all-reduces inside FULL graphs) and 0022 are measured experiments in [`patches/experimental/`](patches/experimental/), outside the release image.

The plugin (`patches/exl3_moe`, a vLLM general plugin, quant method `deepseek_v41_exl3`) carries our GB10 kernels: `exl3_smoe` (the routed experts of a decode step, every K group in one launch set, run flags and L2 evict-first weight copies), `exl3_mrow` (EXL3 dense linears at decode sizes, TMA weight streaming; wo_a reads its head groups in place), a router GEMV with its gate prefetched to L2, TensorFold's paced L2 prefetch (Apache-2.0, `csrc/NOTICE.tensorfold`), an RDMA all-reduce ring over the ConnectX link, the fused mHC post kernel, and the Triton verify attention (`sparse_attn.py`, split-over-keys + ordered merge after TensorFold's design). Prefill sizes stay on exllamav3's kernels. At TP=2 each rank keeps intermediate channels `[r*1152, (r+1)*1152)` of every expert, cut from the full EXL3 tensors at load.

## Reproduce the weights

[RECIPE.md](RECIPE.md) has every step with its script. In short:

| step | script | time |
|---|---|---|
| calibration and panel data: 500 seeded prompts, native answers from DeepSeek-V4.1-Flash on the official weights | `scripts/quantize/caldata/` | about 35 min, $0.59 on OpenRouter |
| routing stats and per-token shares | `p4_pass0.py` | about 70 min (100 s per layer), one Spark |
| layer sensitivity | `p4_sensitivity.py` | 27 min, one Spark |
| allocation | `p4_allocate.py` | seconds |
| pass 1, EXL3 encode of the routed experts | `p4_pass1.py` via `run_pass1_2box.sh` | about 16 h on two Sparks for v3 (~24 min a layer) |
| dense linears and head | `p4_dense.py` via `run_dense_2box.sh` (K4 set: `dense_v3_k4`); the K4 head from `p4_pass1.py --head 4` + `p4_head_variant.py` | about 80 min on two Sparks |
| DSpark drafter dense on EXL3 | `p4_drafter.py` on Hessians captured in an eager serve (`VLLM_EXL3_HCAP`) | about 2 min, one GPU |
| DSpark drafter routed experts on EXL3 | `p4_drafter_experts.py` (layer-level Hessian from the drafter's captured shared-expert input) + `p4_drafter_experts_variant.py` | about 23 min, one GPU |
| assemble | `p4_link_variant.py` (v3: pass-1 part files hardlinked into the model dir; `p4_assemble.py` via `assemble_2box.sh` copies instead) | seconds |
| KLD on the panel | `p4_kld.py` | about 25 min per build, one Spark |
| per-round DSpark depth table | `scripts/bench/depth_sweep.sh` at fixed depths 0-5 on the final build; it prints the `VLLM_DSPARK_ROUND_MS` table | about 16 min |

Calibration (v3, `cal_v3`): 766 sequences, 853,106 tokens = the base set, 374 sequences and 548,231 tokens (native conversations 398,727, exllamav3 standard text 133,120, random 16,384), followed by a supplement of 392 native conversations, 304,875 tokens (instruction following with format / case / length / language constraints, chat, writing, 30 languages; no prompt shares an 8-gram with IFEval). Panel: 199 conversations, 316,944 tokens, 254,846 scored positions. Calibration and panel come from disjoint prompts. sha256 of the token files: base calibration `83f5b3b78a9e220b468d5ab68c979aaedd4743c014d13a86532c781009cfe41f`, panel `39e0db2f0b58aa432a512aed7b425e3feb1b424c9e02cc315be39bc9b7b71cf2`.

## Quality method

- **Panel**: native DeepSeek-V4.1-Flash answers (thinking and chat modes, tool loops with canned tool results, long-context reading, code, math, knowledge, multilingual, writing), rendered with the checkpoint's own encoder. Only assistant-generated positions are scored.
- **Reference**: DeepSeek's `inference/model.py`, streamed layer by layer on a Spark, MXFP4 experts dequantized, FP8 activation quantization emulated. The same implementation scores the native weights and every quantized build: each build's own tensors (its routed experts, plus dense linears and head where a pack quantizes them) are loaded into it, everything else stays official. Differences in the table come from the weights, not from the engine.
- **KLD**: mean KL(native || quantized) over scored positions, in nats. **Top-1**: share of positions where both argmax tokens match.
- **Suites**: HumanEval+ and MBPP+ (evalplus), IFEval (lm-eval checker). The native baseline came through OpenRouter, restricted to providers serving the official weights, no fallback.

## Tests

| test | covers |
|---|---|
| `patches/engram_disk/test_engram_disk.py`, `patches/tests/test_engram_disk_vllm.py` | patch 0002: disk reads == in-memory tables, bit-exact at TP 1 / 2 / 4, with and without the hot-row cache |
| `patches/tests/test_sm12x_mxfp8_bmm.py` | patch 0004: Marlin `wo_a` == BF16 emulation within 1% at 1 to 512 tokens |
| `patches/exl3_moe/tests/test_kernels_gpu.py` | coop / fused / prefill vs the EXL3 reconstruction at K 2-5, TP=2 split == unsplit, CUDA graph replay bit-exact |
| `patches/exl3_moe/tests/test_parallel_groups_gpu.py` | concurrent K groups == chained, eager and under graph capture |
| `patches/exl3_moe/tests/test_dense_gpu.py` | EXL3 dense linears at TP=2 (patch 0006) |
| `patches/exl3_moe/tests/test_head_mxfp8.py`, `test_head_exl3_gpu.py` | MXFP8 and EXL3 heads (patches 0005, 0007) |
| `patches/exl3_moe/tests/test_plan.py`, `test_tp_split_math.py`, `test_bank_cpu.py`, `test_vllm_registration.py` | plan format, TP slice math, bank layout, plugin registration |

## Known limits

- **Two Sparks, TP=2 only.** No single-box path; the experts alone need about 184 GB.
- **Text-only by default.** `TEXT_ONLY=0` loads the vision tower; image input on this build has not been tested.
- **This engine only.** The checkpoint's quant method (`deepseek_v41_exl3`) needs the plugin and patches 0001-0007 on vLLM `6e517b15`. Stock vLLM will not load it. Other engines: not tested.
- **Two sequences at a time** (`MAX_SEQS=2`), up to 262,144 tokens each.
- **GB10 memory traps.** CUDA allocations do not reclaim page cache (the launcher drops it on both nodes and needs passwordless `sudo -n` for that one command, otherwise it prints a warning). Keep 4 GiB or more MemAvailable per node after load; raise `UTIL` only with a measurement. FlashInfer autotune grows driver memory per new kernel shape; leave it off.
- **No large file I/O beside a running GPU job.** A Spark went silent twice under a long GPU job: once during a 25 GB image copy, once at a layer exchange of the two-box quantization pass (~8 GB of files landing per layer). Likely mechanism: on GB10, CUDA allocations cannot reclaim page cache, so a burst of file I/O starves the GPU job's next allocation. The quantization pass now moves its data in chunks and evicts each file from the page cache; copy images and weights while the boxes are idle.
- **If one rank dies, restart both** (`dsv41_tp2.sh down` then `up`). The containers run with `--restart no` on purpose: a restarted single rank cannot rejoin its peer.

## Repository map

| path | what |
|---|---|
| [`GUIDE.md`](GUIDE.md) | install, serve, verify and troubleshoot, step by step |
| [`install.sh`](install.sh) | the one-command install: the wheels, the three images, the copy to the second Spark |
| [`check.sh`](check.sh) | read-only checks of both boxes (and of a live serve with `./check.sh serve`) |
| [`BENCHMARKS.md`](BENCHMARKS.md) | every number with its method and raw file |
| [`FINDINGS.md`](FINDINGS.md) | what GB10 taught us, each with its measurement |
| [`NOTES.md`](NOTES.md) | open items: not measured yet, weaker than the best, still on the table |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | how to measure a change and when it counts |
| [`RECIPE.md`](RECIPE.md) | the pipeline step by step, with the script for each step |
| [`patches/`](patches/) | the patch series 0000-0021 (0009 and 0022 in `experimental/`), `engram_disk/` (patch 0002's reader), `exl3_moe/` (the plugin), `tests/` |
| [`scripts/build/`](scripts/build/) | vLLM and exllamav3 wheels for `sm_121`, built in the pinned build-environment image |
| [`scripts/image/`](scripts/image/) | the runtime, quantization and serving images |
| [`scripts/launch/`](scripts/launch/) | the two-node launcher |
| [`scripts/quantize/`](scripts/quantize/) | calibration data, pass 0, sensitivity, allocation, pass 1, dense pass, assembly, KLD |
| [`scripts/bench/`](scripts/bench/) | speed suite, kernel benches, NCCL and Engram reader benches |
| [`scripts/verify/`](scripts/verify/) | sha256 of every official shard against the Hub's metadata |
| [`data/`](data/) | the allocation plan, routing stats, layer sensitivity, the official manifest, the raw speed, KLD and eval files |
| [`docs/`](docs/) | the plugin's design notes and the GB10 kernel matrix |

## Credits and licenses

- **DeepSeek-V4.1-Flash**: DeepSeek, MIT. The weights published with this repo are a derivative and keep that license.
- **vLLM**: Apache-2.0. Patches 0001 (PR #56509 by zeenat28-ui) and 0003 (PR #58560 by bluemelov1) are upstream work carried unchanged with credit.
- **exllamav3** (turboderp, MIT): the EXL3 quantizer and kernels, called through its public functions; nothing vendored.
- **Marlin** (IST-DASLab, as extended in vLLM), **CUTLASS** (NVIDIA) and **FlashInfer**: the kernels this engine runs on.
- **TensorFold** (Apache-2.0; ashhart/TensorFold and bertholomus's DeepSeek-V4.1 fork): the RDMA all-reduce ring and the paced L2 prefetch kernel are adapted from it, and its grouped EXL3 prompt kernels for the routed experts are vendored unchanged (`csrc/tf_experts/`, on by default since v1.1); the sparse verify attention's structure, the shared expert as one more slot of the routed-expert launch, the per-round depth rule (0010) and the cached Markov rows (0011) follow its designs. Each adapted file is listed in `NOTICE` and `patches/exl3_moe/vllm_exl3_moe/csrc/NOTICE.tensorfold`.
- **bertholomus**: `kit_bench.py`, the client behind the speed table.
- **Prior art**: 0xSero's DeepSeek-V4.1-Flash build for two Sparks and 0xSero's `local-ai-images` repo (read for GB10 build lessons, no code used), Mia-AiLab's EXL3 2.9 bpw pack and vLLM kit. Their builds were measured here as baselines; nothing of theirs is in these weights or this engine.
- This repo (scripts, patches 0000, 0002 and 0004-0022, the plugin): Apache-2.0.

---

Independent measurements on two DGX Sparks. Not affiliated with DeepSeek, NVIDIA or the vLLM project. Pinned to vLLM main `6e517b15` and exllamav3 v1.6.0; expect drift as both move.

<!-- pli:footer:start -->

---

## About PLI Labs

“Your proprietary work product is your proprietary intelligence.” PLI Labs is the open research arm of Proprietary Legal Intelligence. We publish legal model prunes, serving recipes for owned hardware, and open-source tools so readers can inspect our methods and test our claims before hiring us.

Founded by a practicing trial lawyer, PLI builds AI systems around a legal organization's own work product: retrieval, staged workflows, validation, and attorney review. The approach grew from systems running on live matters in a litigation practice. Our consulting work supports deployment through approved frontier APIs, private cloud, or open-weight models on hardware you own. Explore the research or contact us about your organization's work.

[Website](https://proprietarylegal.ai) · [PLI Labs](https://proprietarylegal.ai/labs.html) · [Research](https://proprietarylegal.ai/research/index.html) · [GitHub](https://github.com/ProprietaryLegal) · [Hugging Face](https://huggingface.co/ProprietaryLegal) · [Contact](https://proprietarylegal.ai/contact.html)

Proprietary Legal Intelligence LLC is a technology consultancy, not a law firm, and does not provide legal services or legal advice.

<!-- pli:footer:end -->
