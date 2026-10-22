# dgx-spark-deepseek-v41

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

DeepSeek-V4.1-Flash on two NVIDIA DGX Sparks, from weights we quantized out of the official checkpoints, served by upstream vLLM with our patches and one plugin (our own decode kernels), tensor parallel 2 over the ConnectX link. The v2.0 build of DeepSeek-V4.1-Flash-1017 is the closest to native 1017 of the five 1017 packs we measured, on two panels of native replies, and on the same two boxes it decodes faster than TensorFold on code (+5.3% to +6.1%) and structured output (+2.0% to +2.8%) in three sessions, with prose level (-0.6% to +0.5%), against both `8e0966e2` (the commit lucataco's 1017 recipe pins) and TensorFold's newest DeepSeek TP2 commit (`af26d42b`, its v0.6 release plus two fixes). On the same engine, v1.2's lighter 2.7 bpw build runs ahead of TensorFold on all three prompts (+5.0% to +8.9%). Every number comes with its raw file in [`data/`](data/) and its method in [BENCHMARKS.md](BENCHMARKS.md).

The official checkpoints (`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` and the launch `deepseek-ai/DeepSeek-V4.1-Flash` @ `2cba9e42`, the same layout and sizes) are 510 GB each: 289 GB of MXFP4 routed experts, 203 GB of Engram tables, 18.5 GB of everything else. Two Sparks have 256 GB between them. This repo makes it fit and run:

- the 15,360 routed experts re-encoded in EXL3 at 2, 3, 4 or 5 bits per weight, chosen per expert from measured sensitivity, 2.80 bpw on average with a cap of 3.10 bpw per decoded token (289 GB down to about 191 GB),
- the Engram tables read from the checkpoint files on NVMe at lookup time, CUDA-graph safe, so they never sit in memory,
- the dense linears and the LM head on EXL3 too (K5, `wq_b` K4, head K5), and the DSpark drafter's dense layers (K5), routed experts (K3) and its own lighter K4 head, all through our decode kernels `exl3_mrow` and `exl3_smoe`,
- per-round DSpark draft depth from a model fitted to the pack's own acceptance and a measured cost table, TensorFold's RDMA all-reduce ring, paced L2 prefetch and grouped EXL3 prompt kernels for the routed experts (Apache-2.0, credited), FlashInfer autotune off.

Weights: the v2.0 build, [`sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw) (204 GB). Earlier builds on the same engine: v1.2's [`sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw`](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw) and the launch checkpoint's [`sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw`](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw) (196 GB each). The two Engram shards come from DeepSeek's own repo of the same checkpoint (quickstart step 3).

Builds: v2.0 (`dsv41-recipe-exl3-1017q310r-dk5-dx3-wqb4h5-dh4` on our boxes), v1.2 (`dsv41-recipe-exl3-1017b-dk5-h4-dx3`) and v3 on the launch checkpoint (`dsv41-recipe-exl3-v3b-dk5-h4-dx3`); see [How it works](#how-it-works) for what each contains.

**v2.0 (Oct 2026), a new 1017 build and a new engine:**
- **New weights:** experts at 2.80 bpw with a cap of 3.10 bpw per decoded token (v1.2: 2.70 / 2.90), dense linears at K5 with `wq_b` at K4, the head at K5, the drafter with its own K4 head. KLD vs native 1017: 0.0198 on our panel and 0.0213 on a second panel of native 1017 replies, the lowest of the five 1017 packs we measured on both (HeNryous's 2.75 bpw build is next at 0.0201 / 0.0216, LibertAIDAI's 2.9 bpw at 0.0304 / 0.0322). HumanEval+, MBPP+ and IFEval within noise of native 1017 (paired p 0.61 to 1.00).
- **New engine** (`dsv41-serve:v5`: patches 0023-0038 and the plugin's new kernels): the draft depth from the confidence head plus each draft's logit margin and top-1 probability, through a small model fitted to the pack (0036); the drafter's own head (0026); greedy verify from a vocabulary-shard argmax (0027); Engram's projection row-parallel on the RDMA ring (0028); a row GEMM for the attention inputs (0029); five drafter-phase changes (0030-0033, 0038); every SWA metadata build of a decode step in one launch (0037); the critical-first L2 prefetch order and the heads on `exl3_mrow`. Patches 0024 and 0025 (after HeNryous's) build the RoPE tables and the indexer's prefill buffer to the serve's size, 1.4 GiB a rank by the shapes, which the heavier weights use.
- **Speed:** against TensorFold on the same boxes, in three sessions: code 103.8 vs 98.5, 103.7 vs 97.7 and 103.9 vs 98.4 tok/s, structured 144.1 vs 141.1, 144.4 vs 140.4 and 143.4 vs 140.5, prose 57.8 vs 58.2, 57.8 vs 57.5 and 57.8 vs 57.9 (`8e0966e2` in the first two, TensorFold's newest DeepSeek TP2 commit `af26d42b` in the third). v1.2's weights on the new engine: 106.2-106.4 / 61.8-62.0 / 147.6-149.1 tok/s (v1.2's own engine: 100.7 / 59.2 / 141.3), ahead of TensorFold on all three prompts.
- **To switch:** `scripts/launch/dsv41_tp2.sh down`, then `git pull` and `./install.sh` in `~/dsv41_recipe` (images only, the wheels are kept; the serving image is now `dsv41-serve:v5`), download the 2.8 bpw weights (the same 1017 Engram shards as v1.2), and `up` with `MODEL_DIR` on the new folder. The launcher's defaults are this build's, including its round-cost table; the depth model ships in the model folder. The v1.2 weights run on v5 with their own table, `DOCKER_ENV="VLLM_DSPARK_ROUND_MS=27.3,31.3,34.7,38.1,41.1,43.8"` (no depth model for that pack, so the launcher falls back to the confidence head alone).

**v1.2 (Oct 19 2026), DeepSeek-V4.1-Flash-1017, same engine as v1.1:** the whole pipeline re-run on 1017 with the launch build's plan settings and composition. Panel KLD vs native 1017 0.0243, suites within noise of native 1017, set b 100.7 / 59.2 / 141.3 tok/s against the launch build's 103.0 / 58.0 / 142.6 in the same session.

**v1.1 (Oct 13 2026), same weights as v1:** TensorFold's grouped EXL3 prompt kernels for the routed experts on by default: prefill 2,090 / 2,092 tok/s at 8K / 32K (v1: 1,763 / 1,773), a 255,709-token prompt's first token in 159.6 s (v1: 183.3 s); decode and greedy output unchanged; the JIT kernel builds kept in the mounted cache.

> **Target hardware:** 2x NVIDIA DGX Spark. GB10 Grace-Blackwell, compute `sm_121`, 128 GB unified LPDDR5X each, about 273 GB/s, a ConnectX-7 cable between the two boxes (RoCE v2), DGX OS (Ubuntu 24.04), CUDA 13.0, driver 580.x (tested on 580.142 and 580.173.02), aarch64. Tensor parallel 2, one rank per box.

## Results

Every number below was measured on the same two Sparks. Comparison builds were run by us on these boxes with the same client; no number is copied from another README.

### Speed

`kit_bench.py` set b: single stream, greedy, thinking off, 384-token replies, a 64-token warm-up, then 9 reps a prompt, the median. DSpark speculative decoding on (k = 5 with per-round depth).

The v2.0 build against TensorFold, one session (Oct 22, 08:28-09:07 ICT), the two engines on alternating fresh boots with the same memory preparation before each (raw files and setup: [`data/speed/samebox_v20`](data/speed/samebox_v20/)). TensorFold `8e0966e2` (bertholomus's DeepSeek-V4.1 TP2 branch) serves object-oriented-architects' 1017 EXL3 2.7 bpw pack with lucataco's recipe settings; one network constant changed for our fabric (path MTU 1024).

| tok/s, median of 9 | code | prose | structured |
|---|---|---|---|
| TensorFold, boot 1 | 98.4 | 58.0 | 141.3 |
| TensorFold, boot 2 | 98.6 | **58.4** | 140.9 |
| v2.0 build, boot 1 | **103.9** | 57.9 | 144.0 |
| v2.0 build, boot 2 | 103.6 | 57.7 | **144.1** |
| mean, v2.0 vs TensorFold | +5.3% | -0.6% | +2.1% |

Every rep's text was identical within each run. On prose TensorFold verifies 1.99 tokens a round in 34.1-34.3 ms, this build 2.10 tokens in 36.3 ms: its rounds are shorter, ours accept more. After load TensorFold has 12-13 GiB free a box against our 5-7 GiB (vision on), and it serves a 1M window and 4 sequences at once against our 262,144 tokens and 2.

The release gates were a second session (Oct 22, 12:21-13:00 ICT): each engine booted once after the same memory preparation, ours on the release image with the launcher's defaults exactly as a user runs them (raw: `data/speed/samebox_v20`, `sb2_*`).

| tok/s, median of 9 | code | prose | structured |
|---|---|---|---|
| TensorFold `8e0966e2`, the same setup | 97.7 | 57.5 | 140.4 |
| v2.0 build | 103.7 (+6.1%) | 57.8 (+0.5%) | 144.4 (+2.8%) |
| v1.2 build on the v2.0 engine, its own round table | **106.4** (+8.9%) | **61.8** (+7.5%) | **149.1** (+6.2%) |

The v2.0 build read the same tokens a round as in the first session (4.47 / 2.10 / 5.91) and the same first 400 characters of every reply. On v1.2's weights the v2.0 engine is +5.7% / +4.4% / +5.5% over v1.2's own (100.7 / 59.2 / 141.3, Oct 19); the code and structured replies match v1.2's, and the prose reply departs at character 387 of the 400 compared (a near tie between two words: Engram's row-parallel projection sums in another order).

TensorFold's newest DeepSeek TP2 commit, a third session (Oct 22, 13:18-14:08 ICT): `af26d42b` (its v0.6 release of Oct 18, "leaner prompt chunks, wider rounds", plus two fixes), built with lucataco's Dockerfile at that commit, the same settings, alternating fresh boots with the v2.0 build (raw: `sb3_*`).

| tok/s, median of 9 | code | prose | structured |
|---|---|---|---|
| TensorFold `af26d42b`, boot 1 | 98.6 | **58.5** | 140.7 |
| TensorFold `af26d42b`, boot 2 | 98.2 | 57.4 | 140.3 |
| v2.0 build, boot 1 | **104.2** | 57.8 | **143.6** |
| v2.0 build, boot 2 | 103.6 | 57.7 | 143.1 |
| mean, v2.0 vs TensorFold | +5.6% | -0.3% | +2.0% |
| v1.2 build on the v2.0 engine, one boot | **106.2** (+7.9%) | **62.0** (+7.0%) | **147.6** (+5.0%) |

The newer TensorFold verifies the same tokens a round as `8e0966e2` (4.27 / 1.99 / 5.91) at about the same round times (prose 34.0-34.7 ms).

Earlier builds: v1.2 measured 100.7 / 59.2 / 141.3 tok/s on its own engine, and in the Oct 19 same-box session 101.0 / 59.2 / 142.5 against TensorFold's 99.0 / 58.7 / 142.1 (two-boot means, [`data/speed/samebox_1017`](data/speed/samebox_1017/)). The launch build's tables (v1 and v1.1, against TensorFold v0.6.0 and Mia-AiLab's kit, prefill, DSpark off, two streams) are in [BENCHMARKS.md](BENCHMARKS.md#the-launch-build).

### Quality

Every 1017 pack we measured, on three panels, scored by DeepSeek's own reference implementation with each pack's own tensors loaded (method in [Quality method](#quality-method)):

| pack | routed experts, bpw | our panel, KLD / top-1 | second panel (native 1017 replies), KLD / top-1 | IFEval-reply panel KLD |
|---|---|---|---|---|
| **v2.0 build** | 2.80 (3.10 per decoded token) | **0.0198** / 95.94% | **0.0213** / 95.97% | 0.0980 |
| HeNryous 2.75 bpw (dense K5, head K6) | 2.75 (3.10 per decoded token) | 0.0201 / **95.96%** | 0.0216 / **95.97%** | **0.0974** |
| object-oriented-architects 2.7 bpw | 2.7 | 0.0232 / 95.65% | 0.0249 / 95.65% | 0.1201 |
| v1.2 build | 2.70 (2.90 per decoded token) | 0.0243 / 95.47% | 0.0262 / 95.49% | 0.1199 |
| LibertAIDAI 2.9 bpw | 2.9 | 0.0304 / 94.76% | 0.0322 / 94.70% | 0.1135 |

By domain, KLD (lower is closer to native), our panel / second panel:

| domain | v2.0 build | HeNryous 2.75 bpw | v1.2 build | LibertAIDAI 2.9 bpw |
|---|---|---|---|---|
| code (65% / 68% of positions) | **0.0122** / **0.0124** | 0.0125 / 0.0126 | 0.0150 / 0.0153 | 0.0211 / 0.0210 |
| agentic | **0.0175** / **0.0241** | 0.0183 / 0.0242 | 0.0221 / 0.0280 | 0.0329 / 0.0393 |
| math | 0.0159 / 0.0147 | **0.0152** / **0.0144** | 0.0202 / 0.0194 | 0.0252 / 0.0253 |
| long | **0.0256** / 0.0266 | 0.0258 / **0.0249** | 0.0304 / 0.0316 | 0.0346 / 0.0385 |
| knowledge | 0.0396 / **0.0495** | **0.0382** / 0.0502 | 0.0479 / 0.0628 | 0.0550 / 0.0706 |
| writing | **0.0480** / **0.0748** | 0.0514 / 0.0801 | 0.0614 / 0.1008 | 0.0655 / 0.1104 |
| multilingual | 0.0585 / 0.0693 | 0.0601 / 0.0734 | 0.0697 / 0.0814 | **0.0571** / **0.0637** |
| chat | **0.0475** / **0.0594** | 0.0482 / 0.0608 | 0.0575 / 0.0739 | 0.0716 / 0.0881 |
| the 8 domains weighted equally | **0.0331** / **0.0413** | 0.0337 / 0.0426 | 0.0405 / 0.0517 | 0.0454 / 0.0571 |

The v2.0 build has the lowest KLD on both panels, over all positions and with the domains weighted equally. Top-1 agreement is level with HeNryous's pack (95.94% vs 95.96%, 95.967% vs 95.969%), which is also a little closer on the IFEval-reply panel and on math; LibertAIDAI's pack is closest on multilingual. HeNryous's and object-oriented-architects' packs are made with this recipe's pipeline. Perplexity on our panel 1.6117 and on the second 1.5869 (native 1.6005 / 1.5720).

Suites, greedy, thinking off, 2,048 max tokens, paired per item against native 1017 served on the official weights. A suite fails if the drop is larger than 1 point at p < 0.05.

| suite | native 1017 | v2.0 build | v1.2 build |
|---|---|---|---|
| HumanEval+ (164), base / plus | 97.0 / 90.2 | 97.0 / 89.6 (p 1.00) | 96.3 / 90.2 |
| MBPP+ (378), base / plus | 91.5 / 77.2 | 90.7 / 77.8 (p 0.61 / 0.80) | 89.7 / 76.2 |
| IFEval (541), prompt strict / prompt loose | 90.4 / 92.1 | 90.8 / 92.4 (p 0.86 / 0.85) | 89.1 / 91.1 |

Raw files: [`data/kld/`](data/kld/), [`data/evals/`](data/evals/). The launch build's quality tables (against Mia-AiLab's 2.9 bpw pack and 0xSero's J268 on the launch checkpoint) are in [BENCHMARKS.md](BENCHMARKS.md#the-launch-build).

### Fit and stability

| | v2.0 build | v1.2 build |
|---|---|---|
| context per request | 262,144 tokens; needles 3/3 at 8K / 32K / 64K / 128K, decode after the prompt 83.9 / 85.7 / 81.7 / 81.3 tok/s | 262,144; needles 3/3 at 8K-128K, decode after it 79.0 / 80.9 / 78.1 / 84.8 tok/s |
| the 256K rung (a 255,709-token prompt) | not run yet | not run on 1017 (launch build v1.1: cold prefill 159.6 s, needles 3/3, decode after it 72.3 tok/s, lowest MemAvailable 6.5 GiB) |
| MemAvailable after load, head / worker | 6 / 6 GiB with vision on; 4 / 4 after the 128K rung | 8 / 9 GiB; 7 / 7 after the 128K rung |
| prefill, 8K / 32K prompt (`kit_bench.py prefill`, median of 3) | 2,055 / 2,061 tok/s | 2,090 / 2,092 tok/s (launch build, v1.1) |
| OpenAI API check (tools, streaming, reasoning) | 14 of 14 | 14 of 14 |
| 2-hour agent soak | not run yet | not run yet |

## Quickstart

You need two DGX Sparks cabled with ConnectX-7, an IPv4 address on `enp1s0f0np0` on both, Docker on both, and a control machine (one of the Sparks works) that can `ssh` to both without a password. About 440 GB free on each Spark at the same path: the model folder is 407 GB (204 GB of our weights + DeepSeek's 203 GB Engram tables, which stay in the official files) and the serving image about 25 GB.

```bash
# 1. this repo, on both Sparks (the scripts expect ~/dsv41_recipe)
git clone https://github.com/sudoingX/dgx-spark-deepseek-v41 ~/dsv41_recipe

# 2. engine images, on one Spark with nothing else heavy running on either: the vLLM and exllamav3 wheels for sm_121
#    (memory-capped CPU compiles), the three images, then the serving image copied to the second Spark
cd ~/dsv41_recipe && PEER=<second-spark> ./install.sh   # every step it runs: GUIDE.md, section 1
HEAD=<ssh-host-of-rank-0> WORKER=<ssh-host-of-rank-1> ./check.sh   # read-only, PASS / WARN / FAIL per check

# 3. weights, on both Sparks at the same absolute path (the v2.0 build; v1.2's is sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw
#    with the same Engram shards; the launch build's sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw takes
#    deepseek-ai/DeepSeek-V4.1-Flash @ 2cba9e42aa026125f3ed06c6d98c1db82f7ca027)
hf download sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw --local-dir ~/models/dsv41-exl3
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir ~/models/dsv41-exl3   # Engram tables, 203 GB
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
| `IMAGE` | `dsv41-serve:v5` | serving image (quickstart step 2 builds it; v1.x: `dsv41-serve:v4`) |
| `MAX_LEN`, `MAX_SEQS`, `MAX_BATCH` | 262144, 2, 2048 | context per request, concurrent sequences, prefill chunk |
| `UTIL` | 0.88 | `--gpu-memory-utilization` |
| `KV_BYTES` | 2684354560 | KV pool per rank pinned at 2.5 GiB (empty: sized from `UTIL`) |
| `RECIPE_ENV` | the measured-best knobs | our kernel, prefetch, all-reduce and DSpark settings with the v2.0 build's round-cost table (`27.6,31.9,35.6,38.9,42.2,44.8` ms by depth 0-5), listed and explained in the launcher; empty runs without them. Another pack's table goes in `DOCKER_ENV` |
| depth model | `dspark_conf_feat_head.npz` in `MODEL_DIR` | the launcher hands the pack's fitted depth model to the `conf+feat` policy (patch 0036) when the file is in the model folder on both boxes; without it the policy is `conf`, the confidence head alone |
| `SERVE_CPUS` | `0-18` | the serve's processes start on these cores (`taskset`), so the RDMA ring's proxy thread, pinned to the last core (19, `EXL3_RDMA_CPU`), keeps that core to itself; empty runs unpinned |
| `SPEC` | DSpark, 5 tokens | `--speculative-config` JSON, empty = off |
| `GRAPH` | `FULL_DECODE_ONLY` | CUDA graphs for decode only |
| `ENGRAM` | disk offload, 256 reader threads | `--engram-config` JSON |
| `KERNEL_CONFIG` | FlashInfer autotune off | `--kernel-config` JSON |
| `TEXT_ONLY` | 1 | `--limit-mm-per-prompt '{"image":0}'`: skips the vision tower (about 0.9 GiB per rank); 0 serves images |
| `VLLM_DISABLED_KERNELS` | `FlashInferCutlassMxfp8LinearKernel` | sends every MXFP8 linear to Marlin |
| `INDEXER_LOGITS_MB` | empty (vLLM's 512) | opt-in: sets `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, the largest block of sparse-indexer logits one prefill launch takes; 128 holds less memory after long prompts and runs a long-context prefill chunk in more launches |
| `EXPANDABLE_SEGMENTS` | 0 | opt-in: 1 sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, PyTorch's CUDA allocator grows mapped segments instead of caching separate blocks; on a box up for days, compact memory before the boot |
| `DOCKER_ENV` | empty | extra container env, after `RECIPE_ENV` and the two memory knobs so it overrides a knob there, e.g. `VLLM_EXL3_HEAD=mxfp8` |
| `EXTRA` | empty | extra `vllm serve` flags |

## How it works

### Bits per expert from measured sensitivity

Each of the 15,360 routed experts (40 layers x 384) gets K = 2, 3, 4 or 5 bits per weight. Importance of an expert = the measured sensitivity of its layer x the expert's share of the routed output, per token.

- **Layer sensitivity** (`p4_sensitivity.py`): KL at the model output per unit of relative noise injected into one layer's routed-expert output. It falls from 4.71 at layer 0 to 0.019 at layer 39; layers 0-19 carry about 88% of it.
- **Per-token share.** Every V4.1 layer has a "super expert" that writes very large outputs on a handful of tokens. In layer 1 one expert holds 92.3% of the routed output energy and 1.4% of the per-token share. Our first plan priced experts by energy: it gave that expert K5, pushed most of layer 1 to K2, and the layer's routed-output error came out at 6.4-7.4% (layer 0: 1.2-1.8%). The allocator now prices the per-token share and keeps the 34 super experts (20% or more of their layer's routed energy) at K5 anyway.
- **DSpark target layers.** The drafter reads layers 37, 38 and 39; their experts stay at K3 or more.
- **Measured error per K** (layer 0, full calibration set, median relative output error): K2 16.2%, K3 6.5%, K4 2.4%, K5 1.0%.

Result for v3 (`data/plans/plan_v3_270.json`, importance mixed over the calibration domains): K2 6,157, K3 7,689, K4 1,479, K5 35 experts. 2.70 bpw in memory (85.75 GiB per rank), 2.90 bpw per decoded token. (v2's plan, `plan_v2_270.json`, had the same budget: K2 6,157, K3 7,688, K4 1,481, K5 34.)

On 1017 the statistics and the sensitivity were measured again. v1.2 (`plan_1017_270.json`, the same settings): K2 6,183, K3 7,637, K4 1,505, K5 35. v2.0 (`plan_1017_280_310.json`): a 2.80 bpw memory budget and a cap of 3.10 bpw per decoded token (bits weighted by how often the router picks each expert), so the extra bits can go to the experts the router picks most: K2 5,709, K3 7,056, K4 2,553, K5 42, 88.91 GiB per rank. The allocator reproduces it exactly from the shipped statistics (RECIPE.md).

### Sequential encode

`p4_pass1.py` encodes one layer at a time with exllamav3's EXL3 trellis quantizer (v1.6.0, mul1 codebook). Each layer is calibrated on the hidden states that come out of the already-quantized layers before it. Each expert gets its own Hessian from the tokens routed to it, weighted by their routing weights, plus an all-token prior so rarely routed experts stay well posed. The two Sparks split every layer's experts and exchange partial MoE outputs over the ConnectX link. v2.0's experts came from three runs of this encode: v1.2's full one, a run at 2.80 / 3.00 that encoded the 1,536 experts gaining a bit, and the 2.80 / 3.10 run, which encoded 893 experts and took every other from the first earlier build that holds it at its new K (`p4_pass1.py --reuse a,b --old-plan a,b`).

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
| v2.0: per-round depth from a fitted model | the confidence head's logit plus each draft's logit margin and top-1 probability, 4 weights a draft position, fitted to the pack's own acceptance (11,357 fixed-depth rounds); against the head alone, request by request in one boot: code +1.1%, prose +1.8%, structured level; the drafts and the text do not change | `VLLM_DSPARK_DEPTH_POLICY=conf+feat`, `dspark_conf_feat_head.npz` in the model folder, patch 0036 |
| v2.0: the drafter's own head | the drafter shared the target's head, so a heavier head was read twice a round (a K6 head ~1.0 ms a round over K4); the drafter now reads a K4 copy and the target keeps K5 | `VLLM_DSPARK_OWN_HEAD=1`, patch 0026 |
| v2.0: Engram's projection row-parallel | each rank multiplies its own hash heads' rows and one all-reduce replaces two all-gathers, on the RDMA ring up to 1 MB: +1.2 / +3.3 / +1.9% (code / prose / structured) in one A/B. The partial sums add in another order, so the outputs can differ from v1.x's in the last bits (suites within noise of native) | `VLLM_ENGRAM_WKV_ROWPAR=1`, `EXL3_RDMA_AR_MAX_KB=1024`, patch 0028 |
| v2.0: critical-first L2 prefetch | the prefetch fork loads what the next kernels read first: +1.5-1.8%, bit-identical to v1.x | `EXL3_L2PF_ORDER=1` |
| v2.0: a row GEMM for the attention inputs | the compressor's and the indexer's small bf16 projections stream at DRAM rate instead of cuBLAS split-K (87-97 us for 10.5 MB): code / structured rounds -0.61 / -0.38 ms | `EXL3_ROWMM=1`, patch 0029 |
| v2.0: the round's small launches | greedy verify from a vocabulary-shard argmax (no full-logits all-gather, 163-187 us a round, bit-exact); every SWA metadata build of a decode step in one launch (57 of the 109 launches between rounds, ~0.2 ms a round, bit-exact); the depth rule in one launch | `VLLM_GREEDY_LOCAL_ARGMAX=1`, `VLLM_DSV41_SWA_BATCH=1`, `VLLM_DSPARK_DEPTH_FUSED=1`, patches 0027, 0037, 0038 |
| v2.0: the drafter phase | `main_proj` row-parallel (39 MB a rank instead of 79), the Markov step's bias, add and argmax in two launches, `wo_a`'s FP8 groups on concurrent streams, the context KV in one EXL3 launch, a prefetch chain for the draft layers, the heads on `exl3_mrow`: 0.05-0.2 ms a round each, the target's outputs untouched | patches 0030-0033 and the plugin, flags in `RECIPE_ENV` |
| v2.0: memory | the RoPE tables built to the serve's length and the indexer's prefill buffer at 8x the context instead of 40x (after HeNryous's patches): 0.37 + 1.03 GiB a rank less from load on, from the shapes | patches 0024, 0025, on in the image |

Builds: **v2** = EXL3 routed experts, everything else at the checkpoint's own precision (FP8 dense, BF16 head). **v2.1** = v2 + EXL3 K5 for `wq_b`, `wo_a`, `wo_b` and the shared expert, and an EXL3 K6 head. **v3** (the launch checkpoint's release) = routed experts re-encoded on a broader calibration (instruction, chat, writing and multilingual text added), dense K4 with `wo_a` K5, an EXL3 K4 head, and the DSpark drafter on EXL3 (dense K5, routed experts K3). **v1.2's 1017 build** = v3's pipeline and composition on 1017. **v2.0's 1017 build** = experts at 2.80 / 3.10, dense K5 with `wq_b` K4, an EXL3 K5 head, the drafter at dense K5 and experts K3 with its own K4 head, and the fitted depth model.

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
| 0010 | ours | per-round DSpark verify depth from the confidence head and a measured round-cost table | on (`conf+feat` with 0036) |
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
| 0023 | ours | per-position temperatures for the DSpark confidence head, and a per-round log to fit them from | off (the fitted model of 0036 replaced them) |
| 0024 | ours, after HeNryous's 0033a | RoPE tables built to the serve's length + 2,048 rows instead of 1M | on (`VLLM_DSV41_ROPE_CAP=0` restores) |
| 0025 | ours, after HeNryous's 0033b | the sparse indexer's prefill buffer at 8x the context instead of 40x | on (`VLLM_INDEXER_PREFILL_BUFFER_FACTOR=40` restores) |
| 0026 | ours | the DSpark drafter's own lm_head (`mtp.head.*`), lighter than the target's | on, when the pack has one |
| 0027 | ours | greedy steps from a vocabulary-shard argmax instead of the full-logits all-gather | on |
| 0028 | ours, TensorFold's layout | Engram's `wkv` row-parallel over the hash heads, one all-reduce instead of two all-gathers | on |
| 0029 | ours | the attention inputs' small bf16 projections on the plugin's row GEMM | on |
| 0030 | ours | the drafter's `main_proj` row-parallel over the target taps | on |
| 0031 | ours | the Markov step of the local draft argmax in two launches | on |
| 0032 | ours | Marlin MXFP8 bmm groups on concurrent streams at decode sizes (the drafter's `wo_a`) | on |
| 0033 | ours | the drafter's context KV in one EXL3 launch | on |
| 0034 | ours | the round log gains each draft's logit margin and top-1 probability, and the confidence head's inputs | off (logging, for the fit) |
| 0035 | ours | a fitted confidence head read from an `.npz` | off (0036 is the one shipped) |
| 0036 | ours | the `conf+feat` depth policy: the head's logit and the draft features through a fitted model | on (`dspark_conf_feat_head.npz` from the model folder) |
| 0037 | ours | every SWA metadata build of a decode step in one launch | on |
| 0038 | ours | the depth rule in one Triton launch | on |

0009 (eager all-reduces inside FULL graphs) and 0022 are measured experiments in [`patches/experimental/`](patches/experimental/), outside the release image. Every v2.0 patch is opt-in by its own environment variable except 0024 and 0025 (on in the image, each with a switch back); `RECIPE_ENV` turns on the ones the A/Bs picked.

The plugin (`patches/exl3_moe`, a vLLM general plugin, quant method `deepseek_v41_exl3`) carries our GB10 kernels: `exl3_smoe` (the routed experts of a decode step, every K group in one launch set, run flags and L2 evict-first weight copies), `exl3_mrow` (EXL3 dense linears at decode sizes, TMA weight streaming; wo_a reads its head groups in place), a router GEMV with its gate prefetched to L2, TensorFold's paced L2 prefetch (Apache-2.0, `csrc/NOTICE.tensorfold`), an RDMA all-reduce ring over the ConnectX link, the fused mHC post kernel, the Triton verify attention (`sparse_attn.py`, split-over-keys + ordered merge after TensorFold's design), and since v2.0 a row GEMM for small bf16 projections (`rowmm`), the heads and the drafter's context KV on `exl3_mrow`, and a prefetch chain for the draft layers. Opt-in experiments that stayed off: the reference's FP8 round trip before `wo_a` (`VLLM_DSV41_WOA_QDQ`) and bf16 mHC mixing weights (`VLLM_DSV41_HC_BF16`, no round time gained). Prefill sizes stay on exllamav3's kernels. At TP=2 each rank keeps intermediate channels `[r*1152, (r+1)*1152)` of every expert, cut from the full EXL3 tensors at load.

## Reproduce the weights

[RECIPE.md](RECIPE.md) has every step with its script. In short:

| step | script | time |
|---|---|---|
| calibration and panel data: 500 seeded prompts, native answers from DeepSeek-V4.1-Flash on the official weights | `scripts/quantize/caldata/` | about 35 min, $0.59 on OpenRouter |
| routing stats and per-token shares | `p4_pass0.py` | about 70 min (100 s per layer), one Spark |
| layer sensitivity | `p4_sensitivity.py` | 27 min, one Spark |
| allocation | `p4_allocate.py` (v2.0: `--bpw 2.80 --decode-bpw 3.10`) | seconds |
| pass 1, EXL3 encode of the routed experts | `p4_pass1.py` via `run_pass1_2box.sh`; `--reuse` / `--old-plan` take the experts already encoded at their K from earlier builds | about 16 h on two Sparks for a full encode (~24 min a layer); a run with `--reuse` encodes only the experts it cannot take from an earlier build |
| dense linears and head | `p4_dense.py` via `run_dense_2box.sh` (K4 set: `dense_v3_k4`); the head from `p4_pass1.py --head K` + `p4_head_variant.py`; v2.0's mix of the K5 set with `wq_b` from the K4 set: `p4_dense_mix.py` | about 80 min on two Sparks a K set |
| DSpark drafter dense on EXL3 | `p4_drafter.py` on Hessians captured in an eager serve (`VLLM_EXL3_HCAP`) | about 2 min, one GPU |
| DSpark drafter routed experts on EXL3 | `p4_drafter_experts.py` (layer-level Hessian from the drafter's captured shared-expert input) + `p4_drafter_experts_variant.py` | about 23 min, one GPU |
| the drafter's own head | `p4_drafter_head_variant.py` (another build's K4 head as `mtp.head.*`) | seconds |
| assemble | `p4_link_variant.py` (v3: pass-1 part files hardlinked into the model dir; `p4_assemble.py` via `assemble_2box.sh` copies instead) | seconds |
| KLD on the panel | `p4_kld.py` | about 25 min per build, one Spark |
| per-round DSpark depth table | `scripts/bench/depth_sweep.sh` at fixed depths 0-5 on the final build; it prints the `VLLM_DSPARK_ROUND_MS` table | about 16 min |
| the depth model | a `fixed:5` serve with `VLLM_DSPARK_CONF_LOG` and `VLLM_DSPARK_CONF_LOG_FEATURES=1` (patch 0034), then `scripts/dspark/fit_draft_features.py --export` | about 30 min of serving, seconds to fit |

Calibration (v3, `cal_v3`): 766 sequences, 853,106 tokens = the base set, 374 sequences and 548,231 tokens (native conversations 398,727, exllamav3 standard text 133,120, random 16,384), followed by a supplement of 392 native conversations, 304,875 tokens (instruction following with format / case / length / language constraints, chat, writing, 30 languages; no prompt shares an 8-gram with IFEval). Panel: 199 conversations, 316,944 tokens, 254,846 scored positions. Calibration and panel come from disjoint prompts. sha256 of the token files: base calibration `83f5b3b78a9e220b468d5ab68c979aaedd4743c014d13a86532c781009cfe41f`, panel `39e0db2f0b58aa432a512aed7b425e3feb1b424c9e02cc315be39bc9b7b71cf2`.

## Quality method

- **Panels**: our panel, 199 conversations of native DeepSeek-V4.1-Flash answers (thinking and chat modes, tool loops with canned tool results, long-context reading, code, math, knowledge, multilingual, writing), rendered with the checkpoint's own encoder, 254,846 scored positions. Since v2.0 a second panel: the same 199 prompts answered by native 1017 (`data/caldata/conversations_1017_panel.jsonl`, rendered by `scripts/quantize/caldata/build_panel.py`), 237,457 scored positions, so a pack calibrated on the launch model's replies is not favoured. The IFEval-reply panel: 300 IFEval prompts with native replies, 116,318 positions. Only assistant-generated positions are scored.
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
| `patches/tests/test_dspark_conf_feat_head.py`, `test_dspark_conf_head.py`, `test_dspark_conf_features.py`, `test_dspark_conf_calibration.py`, `test_dspark_depth_fused.py` | patches 0023 and 0034-0036, 0038: the depth policies against a plain reference, the fitted model's file format, the fused rule |
| `patches/tests/test_dspark_own_head.py`, `patches/exl3_moe/tests/test_drafter_head_cpu.py` | patch 0026: the drafter's own head loads, the shared head without it |
| `patches/tests/test_greedy_local_argmax.py`, `test_greedy_argmax_reduction.py` | patch 0027: the shard argmax == the stock greedy paths, ties and padding columns included |
| `patches/tests/test_engram_wkv_rowpar_math.py`, `test_engram_wkv_rowpar.py` | patch 0028: row-parallel == column-parallel within rounding, the loaders' cuts |
| `patches/exl3_moe/tests/test_rowmm_gpu.py`, `patches/tests/test_dspark_markov_fused.py`, `test_sm12x_bmm_streams.py`, `test_dsv41_swa_batch.py` | patches 0029, 0031, 0032, 0037: the row GEMM vs fp64, the fused paths bit-equal to the unfused, eager and in graphs |
| `patches/tests/test_dsv41_memory_caps.py` | patches 0024, 0025: a capped RoPE table keeps the full table's bits (on a GPU), every prefill chunk fits the smaller buffer |

In the v2.0 image on a GB10 (`dsv41-serve:v5`): 702 passed, 9 skipped.

## Known limits

- **Two Sparks, TP=2 only.** No single-box path; the experts alone need about 191 GB (v1.x builds: 184 GB).
- **Text-only by default.** `TEXT_ONLY=0` loads the vision tower (about 0.9 GiB per rank); the v2.0 numbers above were measured with it on. Image input has no measurement beyond a basic test.
- **This engine only.** The checkpoints' quant method (`deepseek_v41_exl3`) needs the plugin and patches 0001-0007 on vLLM `6e517b15`, and the v2.0 build's drafter head needs patch 0026 (the v2.0 image). Stock vLLM will not load them. Other engines: not tested.
- **Two sequences at a time** (`MAX_SEQS=2`), up to 262,144 tokens each.
- **GB10 memory traps.** CUDA allocations do not reclaim page cache (the launcher drops it on both nodes and needs passwordless `sudo -n` for that one command, otherwise it prints a warning). Keep 4 GiB or more MemAvailable per node after load (the v2.0 build leaves 6 GiB with vision on and 4 after a 128K prompt, text-only about 0.9 GiB more; the v1.x builds 8-9); raise `UTIL` only with a measurement. FlashInfer autotune grows driver memory per new kernel shape; leave it off.
- **Compact memory on a box that has been up for days.** Free memory splits into small blocks over days of uptime, and a GPU job started on such a box ran ~2x slower than on its twin (a quantization pass: 9.5 against 5.0 s an expert, `kcompactd` at 100% of a core). With the GPUs idle, `sync; echo 1 | sudo tee /proc/sys/vm/drop_caches; echo 1 | sudo tee /proc/sys/vm/compact_memory` takes under a second; then start the job.
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
| [`patches/`](patches/) | the patch series 0000-0038 (0009 and 0022 in `experimental/`), `engram_disk/` (patch 0002's reader), `exl3_moe/` (the plugin), `tests/` |
| [`scripts/build/`](scripts/build/) | vLLM and exllamav3 wheels for `sm_121`, built in the pinned build-environment image |
| [`scripts/image/`](scripts/image/) | the runtime, quantization and serving images |
| [`scripts/launch/`](scripts/launch/) | the two-node launcher |
| [`scripts/quantize/`](scripts/quantize/) | calibration data and panels, pass 0, sensitivity, allocation, pass 1, dense pass and mixes, the drafter's head, assembly, KLD |
| [`scripts/dspark/`](scripts/dspark/) | the depth model's fitter (`fit_draft_features.py`), the confidence temperatures' fitter, the acceptance-predictor study |
| [`scripts/bench/`](scripts/bench/) | speed suite, kernel benches, NCCL and Engram reader benches |
| [`scripts/verify/`](scripts/verify/) | sha256 of every official shard against the Hub's metadata |
| [`data/`](data/) | the allocation plans, routing stats, layer sensitivity, the official manifests, the panels, the depth model's log, the raw speed, KLD and eval files |
| [`docs/`](docs/) | the plugin's design notes and the GB10 kernel matrix |

## Credits and licenses

- **DeepSeek-V4.1-Flash**: DeepSeek, MIT. The weights published with this repo are a derivative and keep that license.
- **vLLM**: Apache-2.0. Patches 0001 (PR #56509 by zeenat28-ui) and 0003 (PR #58560 by bluemelov1) are upstream work carried unchanged with credit.
- **exllamav3** (turboderp, MIT): the EXL3 quantizer and kernels, called through its public functions; nothing vendored.
- **Marlin** (IST-DASLab, as extended in vLLM), **CUTLASS** (NVIDIA) and **FlashInfer**: the kernels this engine runs on.
- **TensorFold** (Apache-2.0; ashhart/TensorFold and bertholomus's DeepSeek-V4.1 fork): the RDMA all-reduce ring and the paced L2 prefetch kernel are adapted from it, and its grouped EXL3 prompt kernels for the routed experts are vendored unchanged (`csrc/tf_experts/`, on by default since v1.1); the sparse verify attention's structure, the shared expert as one more slot of the routed-expert launch, the per-round depth rule (0010), the cached Markov rows (0011), the vocabulary-shard argmax pair (0012, 0027), the row-parallel Engram projection (0028) and the critical-first prefetch order follow its designs. Each adapted file is listed in `NOTICE` and `patches/exl3_moe/vllm_exl3_moe/csrc/NOTICE.tensorfold`.
- **HeNryous** (Apache-2.0, `HeNryous/deepseek-v41-1017-dgx-spark`): patches 0024 and 0025 rework HeNryous's memory patches 0033a / 0033b.
- **bertholomus**: `kit_bench.py`, the client behind the speed table. **lucataco**: the TensorFold recipe (Dockerfile and settings) our same-box runs use.
- **Prior art**: 0xSero's DeepSeek-V4.1-Flash build for two Sparks and 0xSero's `local-ai-images` repo (read for GB10 build lessons, no code used), Mia-AiLab's EXL3 2.9 bpw pack and vLLM kit. Their builds, and the 1017 packs of HeNryous, object-oriented-architects and LibertAIDAI, were measured here as baselines; nothing of theirs is in these weights.
- This repo (scripts, patches 0000, 0002 and 0004-0038, the plugin): Apache-2.0.

---

Independent measurements on two DGX Sparks. Not affiliated with DeepSeek, NVIDIA or the vLLM project. Pinned to vLLM main `6e517b15` and exllamav3 v1.6.0; expect drift as both move.
