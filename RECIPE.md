# DeepSeek-V4.1-Flash on 2x DGX Spark: the recipe, step by step

How the published builds were made: v3 on the launch checkpoint (v1, Oct 12 2026), the same pipeline on
DeepSeek-V4.1-Flash-1017 (v1.2, Oct 19), and the v2.0 build on 1017 (Oct 22). Every step names the script that does it and what it writes. Paths assume both Sparks mirror
`~/dsv41_recipe/` (this repo's `scripts/` and `patches/`) and hold the official checkpoint at the same path.

## 0. What you get

- Upstream vLLM (main @ 6e517b15) + our patches (the release image applies 0001-0008, 0010-0021 and 0023-0038; 0009 and 0022 are measured experiments in `patches/experimental/`) + one plugin
  (`vllm_exl3_moe`) carrying our own CUDA kernels (`exl3_mrow`, `exl3_smoe`, the router GEMV) and TensorFold's RDMA ring
  and paced L2 prefetch (Apache-2.0, credited), no fork.
- Our own weights from the official `deepseek-ai/DeepSeek-V4.1-Flash` @ 2cba9e42 (build v3): routed experts in EXL3 at a
  per-expert bit width chosen from measured sensitivity (2.70 bpw average, broad calibration), the dense linears in EXL3
  K4 (`wo_a` K5), the head in EXL3 K4, and the DSpark drafter's dense layers (K5) and routed experts (K3) in EXL3.
- v2.0: our weights from `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ 1bae0e10: routed experts at 2.80 bpw with a cap of
  3.10 bpw per decoded token, the dense linears in EXL3 K5 (`wq_b` K4), the head in EXL3 K5, the drafter's dense layers
  (K5), routed experts (K3) and its own K4 head, and a depth model fitted to the pair (section 3, "Build v2.0").
- A calibration set of native V4.1 conversations and separate KLD panels: the conversations themselves
  (`data/caldata/`), their generator, and the sha256 of the token sets.

## 1. Engine image

| step | command | notes |
|---|---|---|
| vLLM wheel at the pin | `scripts/build/` (wheel build on a Spark, CUDA 13, arch 12.1) | see `patches/0000-cmake-sm121-README.txt` |
| runtime image | `scripts/image/build_image.sh` | applies every numbered patch in `patches/` in order, then proves the series is in site-packages by undoing it in reverse on a scratch copy |
| + exllamav3 | `scripts/image/build_quant_image.sh` | exllamav3 1.6.0 wheel built for sm_121 (`scripts/build/build_exllamav3_in_container.sh`) |
| + plugin | `scripts/image/build_serve_image.sh` | `patches/exl3_moe` -> `vllm_exl3_moe` wheel, `--no-deps` |

The patches (each file starts with what it does and why):

| patch | origin | what |
|---|---|---|
| 0001 | upstream PR #56509 | SWA / indexer page geometry FlashInfer's SM120 sparse MLA needs on GB10 |
| 0002 | ours | Engram tables read from the checkpoint shards at lookup time (pread pool -> pinned staging -> upstream kernel) instead of ~100 GB of pinned host tables per rank; CUDA-graph safe |
| 0003 | upstream PR #58560 | compressor ring kept out of the null KV block (else NaN from layer 0 on GB10) |
| 0004 | ours | wo_a on SM12x: one Marlin MXFP8 GEMM per head group instead of a BF16 emulation copy |
| 0005 | ours | V4.1's lm_head gets the quant config (lets the plugin serve an MXFP8 or EXL3 head) |
| 0006 | ours | hooks for non-FP8 dense linears: wo_a `grouped_bmm`, shared-expert padding that leaves 1-D tensors alone |
| 0007 | ours | `head.{trellis,suh,svh,mul1}` map to lm_head like `head.weight` |
| 0008 | ours | DSpark top-k Markov bias for V4.1 (`dspark_draft_topk`, opt-in; measured: lower acceptance, off by default) |
| 0009 | ours, `patches/experimental/`, not in the image | FULL CUDA graphs split only at the TP all-reduces, which run eagerly (`VLLM_FULL_GRAPH_EAGER_ALLREDUCE=1`); measured slower on 2x GB10 (no in-graph NCCL penalty in the serve), kept with its test as a record |
| 0010 | ours (TensorFold's depth rule, Apache-2.0) | per-round DSpark verify depth from the confidence head (`VLLM_DSPARK_DEPTH_POLICY=conf`): rows past the depth are MoE padding (no expert bytes; exl3_smoe reads `is_padding` itself, since vLLM's `dsv4_topk` ignores it) and invalid drafts, the window keeps its shape; greedy text unchanged at every depth; prose +18% vs k=5, code +5% |
| 0011 | ours (TensorFold's idea, Apache-2.0) | Markov bias rows of the 4096 most frequent tokens cached at load (`VLLM_DSPARK_MARKOV_CACHE=<token list>`, `data/dspark/markov_top_tokens.json`): ~+2% on code |
| 0012-0038 | ours, 0024 / 0025 after HeNryous's, 0028 after TensorFold's layout | the README's patch table: one line each with whether the defaults turn it on; v2.0 adds 0023-0038 |

The plugin's own kernels (JIT-built for sm_121 on first use, cached under `~/.cache`):

| kernel | file | what it replaces | measured |
|---|---|---|---|
| `exl3_mrow` v2 | `csrc/exl3_mrow.cu` | exllamav3's GEMM for every dense EXL3 linear at <= 16 rows (wq_a + wkv, wq_b, wo_a heads, wo_b, shared gate + up, shared down) | 255 us per layer vs 378 (m=4), 186-229 GB/s; a producer warp streams each k-row's tiles with one TMA bulk copy into an mbarrier ring |
| `exl3_smoe` | `csrc/exl3_smoe.cu` | exllamav3's coop MoE (one launch set per bit-width group, side streams) for the routed experts at decode sizes | 1.02-1.15x the coop path in isolation, +4-5% end to end; every bit width in one launch set (rot, gate + up, down), 32-lane cp.async producer |
| fused MoE ops | `fused_ops.py` (Triton) | ~13 small torch ops around the MoE | bit-identical |
| PDL launches | `csrc/exl3_launch.h` | plain launches | +1.2-3.4%: weights prefetch under the previous kernel's tail |
| paced L2 prefetch | `csrc/exl3_l2pf.cu`, `l2pf.py` (TensorFold's kernel, Apache-2.0, `csrc/NOTICE.tensorfold`) | nothing: the next dense weights stream into L2 on a side stream while an all-reduce or attention leaves DRAM idle (`EXL3_L2PF=attn,moe,next`) | a 16 MiB read 81 -> 39 us after a prefetch; +2% end to end with the ring |
| RDMA all-reduce ring | `csrc/exl3_rdma.cu`, `rdma.py` (TensorFold's one-hop ring, Apache-2.0) | NCCL for decode-size bf16 TP all-reduces (`EXL3_RDMA_AR=1`): staging kernel, pinned CPU proxy posting RDMA writes with immediate, rank-order sum kernel | 14.6 vs 36.5 us an all-reduce in a graph, bit-identical to NCCL; +5% end to end |

Shared device code (trellis decode adapted from exllamav3, MIT, notice in `csrc/LICENSE.exllamav3`): `csrc/exl3_common.cuh`.

Serve-time choices that matter on GB10: `VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel` (vLLM's default
MXFP8 GEMM runs about 3x slower than Marlin at decode sizes on SM12x: 66-78 vs 200-251 GB/s with the weights read from
DRAM, `scripts/bench/linear_dram_bench.py`), FlashInfer autotune off, FULL_DECODE_ONLY graphs, `EXL3_INT8_GEMV=0`
(exllamav3's int8 activations are a serving-path error the BF16 harness cannot see).

Plugin switches (defaults are the measured best; each has an off value for A/B runs): `VLLM_EXL3_DENSE_MROW=1`
(`groups` = multi-matrix modules only, `0` = exllamav3), `VLLM_EXL3_MOE_SMOE=1` (`0` = the exllamav3 coop path, which
then wants `VLLM_EXL3_MOE_PARALLEL_GROUPS=1 EXL3_MOE_COOP_WIDE=1`), `VLLM_EXL3_MOE_FUSED_OPS=1`, `EXL3_PDL=1`.
Measured and on in the launcher's defaults: `EXL3_L2PF=attn,moe,next` (+2% with the ring, -2% without it) and
`EXL3_RDMA_AR=1` (decode TP all-reduces over RDMA writes, `rdma.py` + `csrc/exl3_rdma.cu`, TensorFold's ring with a
bf16 rank-order sum: 14.6 vs NCCL's 36.5 us an all-reduce, bit-identical to NCCL; +5% end to end). Measured neutral,
opt-in until v2.0: `VLLM_EXL3_HEAD_MROW=1` (bit-exact, within noise; on since v2.0, with the drafter's own head). Since
v2.0 also on: `EXL3_L2PF_ORDER=1` (the prefetch fork loads what the next kernels read first: +1.5-1.8%, bit-identical),
`EXL3_L2PF_DRAFTER=1` (a prefetch chain for the draft layers), `EXL3_ROWMM=1` (`rowmm.py` + `csrc/exl3_router.cu`: the
attention inputs' small bf16 projections at DRAM rate, patch 0029) and `EXL3_RDMA_AR_MAX_KB=1024` (the ring carries
all-reduces up to 1 MB: patch 0028's row-parallel Engram reduces 307 KB at a 6-row verify step).
`VLLM_EXL3_MOE_DET_PREFILL=1` (default): the eager prefill MoE sums each token's expert outputs in k order from
per-assignment slots (exllamav3's slot mode) instead of fp32 atomics, so identical prompts give identical bits.

## 2. Calibration data and the KLD panel

`scripts/quantize/caldata/{prompts,generate,build}.py`: 500 seeded prompts (code, agentic tool loops with canned tool
results, long-context reading, chat, math, knowledge, multilingual, writing; thinking and chat modes), native answers
from DeepSeek-V4.1-Flash through OpenRouter (DeepInfra, FP8), encoded with the checkpoint's own encoder. The prompts split
299 / 201 into two disjoint sets. Calibration: 374 sequences, 548,231 tokens (native conversations 398,727, exllamav3's
standard text 133,120, random 16,384), sha256 `83f5b3b78a9e220b468d5ab68c979aaedd4743c014d13a86532c781009cfe41f`.
Panel: 199 conversations, 316,944 tokens, 254,846 scored assistant positions, sha256
`39e0db2f0b58aa432a512aed7b425e3feb1b424c9e02cc315be39bc9b7b71cf2`. The generator requests no sampling settings, so
providers use their defaults: rerunning it gives new conversations, not these hashes. `build.py`'s `std_dir` is exllamav3's own standard calibration text in its v1.6.0 tree (`exllamav3/conversion/standard_cal_data`, MIT, turboderp-org @ 151539c7): read from a clone, not copied here.

**Calibration v3 (build v3).** The first set is 42% code tokens (chat 4.5%, writing 1.2%, multilingual 5%), and an EXL3
encode fits the input statistics of its Hessians: build v2.1's KLD gap to Mia's pack grew as a domain thinned in this
set (code 0.58x Mia, multilingual 1.34x). `caldata/prompts_instruct.py` + `generate.py --set instruct` add 392 held-out
conversations (236 instruction-following with format / case / length / language constraints on 76 topics written for
it, 48 chat, 48 writing, 60 multilingual in 30 languages; no prompt shares an 8-gram with IFEval's 541, and IFEval's
distinctive conventions are avoided; native answers, same providers, $0.15). `caldata/build_supplement.py` renders them
(304,875 tokens) and writes `cal_v3` = the calibration set followed by the supplement: 766 sequences, 853,106 tokens.
The KLD panel is unchanged.

**The native-1017 panel (v2.0).** `generate.py --split panel` on the 1017 checkpoint answers the panel's 199 prompts
again (`data/caldata/conversations_1017_panel.jsonl`, 277 calls through OpenRouter, 276 served by DeepInfra, $0.20),
and `caldata/build_panel.py` renders them the way `build.py` renders the published panel: 237,457 scored positions,
sha256 `f22a47c238f0926ffa22c2b5f42f9276eabdbec2358949ac6bc3d50420fe0da0` (`data/caldata/panel_1017/`). A pack
calibrated on the launch model's replies gets no edge on it; `p4_kld.py` takes it as its panel argument.

## 3. Weights

| pass | script | what it measures or writes | time on 2 Sparks |
|---|---|---|---|
| reference check | `ref_truncated_logits.py`, `compare_topk.py` | DeepSeek's own `inference/model.py` runs layer-streamed on GB10 (sparse_attn per 16-head group) and matches vLLM | minutes |
| pass 0 | `p4_pass0.py` | per expert: routed slots, routing weights, output energy, **per-token share** of the routed output; per layer energies | ~35 min, one box |
| sensitivity | `p4_sensitivity.py` | per layer: KL at the output per unit of per-token relative noise in the routed output | ~27 min, one box |
| allocation | `p4_allocate.py` | K in {2, 3, 4, 5} per expert: importance = sensitivity x token share, measured error per K, memory budget + decode-bandwidth cap; super experts (>= 20% of a layer's routed energy) and the DSpark target layers 37-39 protected | seconds |
| pass 1 | `p4_pass1.py` via `run_pass1_2box.sh` | sequential EXL3 encode of the routed experts (routing-weighted Hessians with an all-token prior, mul1 codebook), each layer calibrated on the stream through the already-quantized layers; the two boxes split each layer's experts and swap partial MoE outputs | ~20 min per layer on 2 boxes (K2-heavy layers slowest), ~13 h for 40 |
| dense pass (optional) | `p4_dense.py` | input Hessians of wq_a + wkv (one, shared), wq_b, wo_a (per head group), wo_b, shared expert, and the head; EXL3 encode | ~1 h 23 min on 2 boxes (v3 1 h 22 min, 1017 1 h 24 min) |
| assemble | `p4_assemble.py` via `assemble_2box.sh` | official non-expert shards (minus what EXL3 replaces) + EXL3 files + the official Engram shards hardlinked + config with the plugin's quantization block | ~14 min (1017) |
| fidelity | `p4_kld.py` | KL vs the native model, top-1 agreement, NLL, per domain, on the panel, with the same reference implementation for every build (also scores other EXL3 packs: `--exl3-ckpt`) | ~25 min a build on the panel, ~13 min on the IFEval-reply panel, one box each |

Build v3 (shipped Oct 11) runs the same passes on `cal_v3`: pass 0 with `--index` (per-domain statistics; their
token-weighted merge re-allocates plan v2 exactly), `p4_merge_stats.py` (importance as a domain mix: the eight panel
domains weight 1, instruction 1.5, exllamav3's standard text 0.5, random 0.1), `p4_allocate.py` at plan v2's budget
(`data/plans/plan_v3_270.json`: 2.70 bpw, 2.90 decode cap), pass 1 on `cal_v3` with token-weighted Hessians (a probe at
layer 10 found re-weighting domains in the Hessians adds nothing over simply including the supplement:
`hessian_domain_probe.py`), and `p4_link_variant.py` instead of the assembly (the pass-1 part files hardlinked into the
model dir, the dense EXL3 tensors in one `model-exl3-dense.safetensors`). `p4_pass1.py --reuse --old-plan` re-encodes
only the experts whose K changed (unchanged ones come from an assembled checkpoint) and `p4_overlay_variant.py` builds
that variant; `--index --domain-weights` weights the Hessians by domain.

On top of the v3 experts, the shipping dir (`dsv41-recipe-exl3-v3b-dk5-h4-dx3`) is built by hardlink variants, each
gated:

| step | script | gate |
|---|---|---|
| dense K4 set (`wo_a` K5, head K5) | `run_dense_2box.sh` (K5 + K4 sets from one stream) + `p4_dense_variant.py` | panel KLD within 10% of the K5 set and under Mia's (0.0236 vs 0.0216) |
| head K4 | `p4_pass1.py --head 4` (the stream's final states) + `p4_head_variant.py` | panel KLD < 0.0275 (0.0243) |
| drafter dense K5 | `p4_drafter.py` on an HCAP capture + `p4_drafter_variant.py` | target output identical, acceptance held |
| drafter routed experts K3 | `p4_drafter_experts.py` (layer-level Hessian = the drafter's captured shared-expert input; Gaussian-sample prior for the down projection) + `p4_drafter_experts_variant.py` | set b tok/s up on code and prose, target output identical (24/24) |
| depth table | `scripts/bench/depth_sweep.sh` at fixed depths 0-5 on the final dir (prints the `VLLM_DSPARK_ROUND_MS` table) | re-measured whenever kernels or weights change |

Build 1017 (Oct 18) rebuilds every weight from `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ 1bae0e10 with v3's pipeline,
step for step: the passes above in order, then the same gated variants. 1017 keeps 2cba9e42's
config, tokenizer and tensor layout, so the image, patches, kernels, launcher, `cal_v3` and both panels carry over.
Every weight changed, the Engram shards 47 / 48 included, so pass 1 is a full encode (`--reuse` has nothing to reuse)
and the Engram shards come from the 1017 repo. Pass 0 and the sensitivity re-ran on 1017: its experts read ~1.35x more
sensitive to the same relative noise (mean alpha 1.855 against 1.400, `data/alpha_sensitivity_1017.json`; only layer 4
reads lower). The allocation with plan v3's exact settings (`data/plans/plan_1017_270.json`: 2.70 bpw, 2.90 decode
cap, 85.75 GiB a rank) gives K2 / K3 / K4 / K5 = 6,183 / 7,637 / 1,505 / 35 experts, 762 of 15,360 on a different K
than plan v3. Pass 1 encodes 6.83 s per K2 expert and 4.43 s per K3 / K4 expert, so 17.5-18 min a layer on two boxes
where K3 dominates and ~23.5 min on the K2-heavy layers 25-36. At equal bits 1017's KLD runs higher, so v3's absolute
caps (taken from Mia's 2cba9e42 pack) don't carry over: the gates take theirs from a Mia-layout 1017 pack
(`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` @ c178a012) scored on the same panel. The dense K4 set must stay
within 10% of the K5 set and under that pack's KLD, and the K4 head under 0.932x that pack's (v3's ratio to Mia's 0.0295).

What the 1017 build took and measured (Oct 18-19): pass 1 16.2 h of wall time (06:53 -> 23:05, ~2 h of it lost to a
frozen node; 18-24 min a layer on two boxes, with 15-18 encoded solo on one), the dense pass 1 h 24 min, assembly
14 min. The variant tools swap dense-only files, so the assembled dir is turned into v3's layout first
(`p4_link_variant.py`: pass 1's part files hardlinked, the dense set in its own file) before `p4_dense_variant.py`,
`p4_head_variant.py` and the drafter variants run. Panel KLD vs native 1017: 1017b-h4 0.0243, 1017b 0.0237, 1017r
0.0219, LibertAIDAI's pack 0.0304, so the rule picks dense K4 + head K4 again, with the same numbers as v3 on 2cba9e42 even
though 1017's experts read more sensitive. The speed side of that pick, measured afterwards with each composition's own
depth table in one session: dense K5 + head K4 (KLD 0.0227) 94.8 / 54.4 / 138.6 tok/s, dense K5 + head K6 (0.0219)
95.1 / 56.7 / 135.3, the shipping dir 100.4 / 59.0 / 142.1 (`data/speed/variants_1017`). Drafter capture (36 prompts on an eager serve) 10 min, its encode 30 min.
The shipping dir `dsv41-recipe-exl3-1017b-dk5-h4-dx3` decodes at v3's speed (set b 100.7 / 59.2 / 141.3 vs v3's
103.0 / 58.0 / 142.6 the same hour), and every suite (HumanEval+, MBPP+, IFEval) sits within noise of native 1017 (BENCHMARKS.md).

**Build v2.0 (Oct 20-22).** The 1017 statistics, sensitivity, calibration and panels carry over; the budget, the
composition and the depth model change. Each step was gated on the three panels and the suites:

| step | script | what it wrote or measured |
|---|---|---|
| allocation | `p4_allocate.py data/p0_stats_1017/merged_bal --bpw 2.80 --decode-bpw 3.10 --alpha data/alpha_sensitivity_1017.json --eps 2:0.026,3:0.0042,4:0.00058,5:0.000096 --importance tshare --protect-share 0.2 --floor-layers 37,38,39:3` | `data/plans/plan_1017_280_310.json` (its `drafter` section, all K3, from v1.2's plan): K2 5,709, K3 7,056, K4 2,553, K5 42, 88.91 GiB a rank, predicted cost 0.1614 against v1.2's 0.2206; the command reproduces the plan exactly |
| pass 1 | `p4_pass1.py --reuse <2.80/3.00 build>,<v1.2 dir> --old-plan <their plans>` | 893 experts encoded; the rest from a 2.80 / 3.00 build (whose run encoded the 1,536 experts that gained a bit over v1.2's plan) and from v1.2's full encode, each expert from the first build holding it at its new K |
| dense and head | `p4_dense_mix.py` (the K5 set with `attn.wq_b` from the K4 set, both from the 1017 rebuild's one capture) + `p4_dense_variant.py`; the K5 head from the 1017 rebuild + `p4_head_variant.py` | KLD 0.0198 on our panel; the K5 set with the K6 head measured 0.0194, `wq_b` K4 0.0196, the K5 head 0.0197 |
| drafter | v1.2's drafter dense (K5) and experts (K3); `p4_drafter_head_variant.py` with v1.2's K4 head as `mtp.head.*` | the drafter reads a 331 MB head instead of the target's 414 MB; the target's outputs do not change |
| depth table | `scripts/bench/depth_sweep.sh` at fixed depths 0-5 | 27.6 / 31.9 / 35.6 / 38.9 / 42.2 / 44.8 ms (`data/speed/depth_v20`), the launcher's default |
| depth model | a `fixed:5` serve with `VLLM_DSPARK_CONF_LOG` and `VLLM_DSPARK_CONF_LOG_FEATURES=1` (30 prompts a domain, 11,357 rounds: `data/dspark/conflog_1017_28bpw_fixed5.jsonl`), then `scripts/dspark/fit_draft_features.py --export` | `dspark_conf_feat_head.npz` (`data/dspark/conf_feat_head_1017_28bpw.npz`); the fitter reproduces it from the log, max difference 0 |
| gates | `p4_kld.py` on three panels, the suites against native 1017, set b against TensorFold in one session | 0.0198 / 0.0213 / 0.0980; suites p 0.61-1.00; 103.8 / 57.8 / 144.1 against 98.5 / 58.2 / 141.1 tok/s |

A full pass 1 at `plan_1017_280_310` encodes every expert on one stream; ours came from the chained runs, so a rebuild
differs from it in bits, not in method. Gate for the build: KLD at or under 0.0200 on our panel and the lowest on the
native-1017 panel, every suite within noise of native 1017, set b ahead of TensorFold in the same session outside noise
(code and structured cleared it; prose is 0.6% short), and 262,144 tokens a request with 5 GiB or more MemAvailable.

## 4. Serve

`scripts/launch/dsv41_tp2.sh up` with `MODEL_DIR=<assembled dir>`: two containers, vLLM's multi-node TP=2 over RoCE on the
ConnectX link (NCCL pinned to the HCA that carries the fabric IPv4), Engram on disk, DSpark drafter, V4.1 tool and
reasoning parsers, text-only by default. Knobs at the top of the script. The measured configuration is the default
(`RECIPE_ENV` in the script: our MoE knobs, DSpark k=5 with per-round depth `conf` and the measured round-cost table,
the Markov bias cache with its token list copied to each node, the paced L2 prefetch and the RDMA all-reduce ring, and
since v1.1 TensorFold's grouped EXL3 prompt kernels for the routed experts with the JIT builds in the mounted cache):

```
MODEL_DIR=/home/spark/models/dsv41-recipe-exl3-1017q310r-dk5-dx3-wqb4h5-dh4 scripts/launch/dsv41_tp2.sh up   # v2.0: image dsv41-serve:v5, the launcher's default
```

Since v2.0 the defaults are the v2.0 build's (`RECIPE_ENV`: 33 settings with the conf+feat policy and that build's
round table; the depth model read from `MODEL_DIR`; `SERVE_CPUS=0-18`). The rest of this section is the launch build
on v1.1's engine, whose launcher and defaults are the `v1.2` tag's.

with the guards in serve mode on both boxes first (`bash ~/dsv41_recipe/scripts/quantize/guards.sh serve`: page-cache drops
during a serve evict Engram's rows and slow random reps ~5%). kit_bench set b on it (Oct 12 00:23, the recipe defaults, 9
reps a prompt): **code 103.0 / prose 58.3 / structured 141.3 tok/s**, every rep within 1%, greedy text identical run to
run; TensorFold v0.6.0 on the same boxes 100.5 / 62.1 / 139.0 (9 reps). The round-cost table is a property of the
serve: re-measure it after changing kernels or weights (`scripts/bench/depth_sweep.sh` at fixed depths 0-5 on a serve
booted with `VLLM_DSPARK_DEPTH_FILE`; it prints the new table).

DSpark k: 5 is faster on code and structured output (77.1 / 113.2 tok/s), 3 on prose (41.4 tok/s) (set b, Oct 10).
The serve leaves ~9-11 GiB MemAvailable per node with the 2.5 GiB KV pool pinned (`KV_BYTES`).

## 5. Measuring

Speed: `kit_bench.py` (the client behind the published TensorFold and Mia numbers, byte-identical to bertholomus's
`tools/dsv41/kit_bench.py`; set b = greedy, thinking off, 384 tokens, median of 3) through
`scripts/bench/serve_speed_suite.sh` (`QUICK=1`: set b + DSpark acceptance + the oracle), our streaming ruler, and
`scripts/bench/profile_decode.sh` for a torch-profiler decode step (`trace_union.py` style breakdowns). Report every
number per prompt with its DSpark k; judge engine changes on the structured prompt, whose greedy text does not move. Gates: the engines harness (`api_check.py`, `ctx_ladder.py`, `evals.py` +
`paired_stats.py` against the native model through OpenRouter on FP8 providers).
