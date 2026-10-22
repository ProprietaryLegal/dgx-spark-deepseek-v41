# Benchmarks

Every number with its method and its raw file. Two DGX Sparks (GB10), tensor parallel 2 over a ConnectX-7 cable at MTU
1500; every comparison build was run by us on the same two boxes with the same client.

## Method

- **Client:** bertholomus's `kit_bench.py` (`tools/dsv41/kit_bench.py` in github.com/bertholomus/TensorFold @
  `fe6f9f7b`, sha256 `83a4fbeb987032f9...`), over HTTP with streaming. Decode tok/s = (completion tokens - 1) / (time
  of the last streamed piece - time of the first), so prefill is excluded.
- **Set b:** three chat prompts (code, prose, structured output), single stream, greedy (temperature 0), thinking off,
  384-token replies, median of 3 reps (9 where a row says so). Set a: the client's other prompt set, same settings.
- **Speculative decoding:** DSpark on unless a row says off. Ours: k = 5 with per-round depth from the measured cost
  table; TensorFold: its exact drafting (`--mtp-drafts 5`); Mia-AiLab's kit: k = 3 as shipped.
- **Our build:** the recipe defaults (`dsv41_tp2.sh up`), the page-cache guard in serve mode (`guards.sh serve`). v2.0:
  image `dsv41-serve:v5`, the v2.0 build, vision on (`TEXT_ONLY=0`); the sections from "Earlier builds" down: image
  `dsv41-serve:v4`, builds v3 (launch checkpoint) and v1.2 (1017).
- **Fidelity:** KL(native || quantized) over scored assistant positions, every pack scored by DeepSeek's own
  `inference/model.py` with its own tensors loaded (README, "Quality method").

## v2.0: decode, same boxes and client

One session, Oct 22 08:28-09:07 ICT: the two engines on alternating fresh boots, each boot after the same memory
preparation (page cache dropped, memory compacted; 82-88 GB of free 2 MiB-and-up blocks on one box and 62-68 GB on the
other), set b with a 64-token warm-up and 9 reps a prompt. Raw files and setup: `data/speed/samebox_v20`.

| build | code | prose | structured | raw |
|---|---|---|---|---|
| **v2.0 build**, boot 1 | **103.9** (103.3-104.2) | 57.9 (57.6-58.1) | **144.0** (143.2-144.7) | `sb_ours_b1.json` |
| **v2.0 build**, boot 2 | **103.6** (103.4-104.1) | 57.7 (57.6-58.1) | **144.1** (143.3-144.7) | `sb_ours_b2.json` |
| TensorFold `8e0966e2` + object-oriented-architects 1017 2.7 bpw (lucataco's settings), boot 1 | 98.4 (91.1-99.1) | 58.0 (56.2-58.4) | 141.3 (126.2-142.5) | `sb_tf_b1.json` |
| same, boot 2 | 98.6 (90.8-99.1) | **58.4** (52.9-58.6) | 140.9 (138.1-141.4) | `sb_tf_b2.json` |

- On the two-boot means: code +5.3%, structured +2.1%, prose -0.6%.
- Tokens per verify round (streamed chunks) and milliseconds per round, v2.0 vs TensorFold: code 4.47 vs 4.27 at 43.0
  vs 43.3-43.4 ms, prose 2.10 vs 1.99 at 36.3 vs 34.1-34.3 ms, structured 5.91 for both at 41.0 vs 41.8-41.9 ms. On
  prose TensorFold's rounds are 2 ms shorter and ours accept more.
- Every rep's text was identical within each run. TensorFold's reps include slow ones on all three prompts; ours stay
  within 1% of their median.
- After load TensorFold has 12-13 GiB free a box, the v2.0 build 5-7 GiB (both with vision). TensorFold serves a 1M
  window and 4 sequences, ours 262,144 tokens and 2.
- TensorFold ran as in v1.2's same-box test (`data/speed/samebox_1017/README.md`): lucataco's recipe settings and its
  `TF_DS_*` environment, the path MTU set to 1024 for our fabric; the full launch is in `data/speed/samebox_v20/README.md`.
- A second session, the release gates (Oct 22 12:21-13:00, one fresh boot each with the same memory preparation, the
  release image `dsv41-serve:v5` and the launcher's defaults): TensorFold 97.7 (94.3-98.1) / 57.5 (54.6-58.2) / 140.4
  (123.4-140.6), the v2.0 build 103.7 (103.0-104.5) / 57.8 (57.5-58.0) / 144.4 (143.6-144.7): +6.1%, +0.5%, +2.8%. Raw:
  `sb2_tf.json`, `sb2_ours.json`.
- v1.2's 2.7 bpw build on the v2.0 engine, the same session, its own round table (`27.3,31.3,34.7,38.1,41.1,43.8`, depth
  policy `conf`: no depth model for that pack): **106.4** (105.8-107.4) / **61.8** (60.0-62.1) / **149.1** (148.5-149.4),
  +8.9% / +7.5% / +6.2% over TensorFold and +5.7% / +4.4% / +5.5% over v1.2's own engine; 4.36 / 2.11 / 5.91 tokens a
  round at 41.0 / 34.1 / 39.6 ms (`sb2_v12.json`). Its code and structured replies match v1.2's first 400 characters;
  prose departs at character 387 (Engram's row-parallel sums, patch 0028).
- A third session, TensorFold's newest DeepSeek TP2 commit (Oct 22 13:18-14:08): `af26d42b` (its v0.6 release of Oct 18,
  "leaner prompt chunks, wider rounds", plus two fixes), built with lucataco's Dockerfile at that commit with its own
  rank cache, the same settings; alternating fresh boots with the v2.0 build, the same memory preparation before each:

  | build | code | prose | structured | raw |
  |---|---|---|---|---|
  | TensorFold `af26d42b`, boot 1 | 98.6 (84.8-99.3) | **58.5** (52.5-58.7) | 140.7 (134.6-141.7) | `sb3_tf06_b1.json` |
  | TensorFold `af26d42b`, boot 2 | 98.2 (90.2-98.7) | 57.4 (56.0-58.4) | 140.3 (129.2-140.8) | `sb3_tf06_b2.json` |
  | **v2.0 build**, boot 1 | **104.2** (103.2-104.5) | 57.8 (56.1-58.1) | **143.6** (141.7-144.2) | `sb3_ours_b1.json` |
  | **v2.0 build**, boot 2 | 103.6 (103.3-104.3) | 57.7 (57.5-58.0) | 143.1 (142.1-143.6) | `sb3_ours_b2.json` |
  | v1.2's build on the v2.0 engine, one boot | **106.2** (105.3-107.2) | **62.0** (61.6-62.2) | **147.6** (145.9-149.0) | `sb3_v12.json` |

  Two-boot means: code +5.6%, structured +2.0%, prose -0.3% for v2.0; v1.2's build +7.9% / +7.0% / +5.0%. The newer
  TensorFold verifies the same tokens a round (4.27 / 1.99 / 5.91) at about the same round times (prose 34.0-34.7 ms) and
  loads with 11-13 GiB free a box.

## v2.0: how the speed came (tok/s, code / prose / structured)

The quality build's heavier experts, dense set and head cost 2.1-2.7 ms a verify round against v1.2's on the same
engine (Oct 21); the engine work below won that back on code and structured, and prose stays 2.4% under v1.2's. Each
row is the build of that day on fresh boots, set b.

| when (Oct 2026) | code / prose / structured | what changed |
|---|---|---|
| Oct 19 | 100.7 / 59.2 / 141.3 | v1.2 on its own engine (image v4), the starting point |
| Oct 21 02:00 | 99.1 / 54.1 / 136.8 | the 2.80 / 3.10 experts with dense K5 and a K6 head, the critical-first prefetch order, greedy verify from the shard argmax (0027), the drafter's own K4 head (0026) |
| Oct 21 09:12 | 98.6 / 56.2 / 141.6 | `wq_b` K4 and the head K5 (KLD 0.0198), the heads on `exl3_mrow`, Engram's projection row-parallel on the ring (0028), the serve off the ring proxy's core |
| Oct 21 17:50 | 100.7-100.8 / 57.0-57.1 / 141.2-141.6 | the row GEMM for the attention inputs (0029), the drafter phase (0030-0033, its prefetch chain), the fitted depth model (0036) |
| Oct 21 21:31 | 102.8-102.9 / 58.0 / 144.2-144.3 | every SWA metadata build in one launch (0037), the depth rule in one launch (0038) |
| Oct 21 23:53 | 103.9-104.0 / 57.7-57.8 / 143.0-143.2 | the round-cost table re-measured on that build (the v2.0 default) |
| Oct 22 09:07 | **103.8 / 57.8 / 144.1** | the same-session head to head above (two-boot means) |

## v2.0: fidelity on three panels

Our panel: 199 native conversations, 254,846 scored positions. The second panel: the same 199 prompts answered by
native 1017 through OpenRouter (277 calls, 276 served by DeepInfra and 1 by Together), 237,457 scored positions
(`data/caldata/conversations_1017_panel.jsonl`, `data/caldata/panel_1017/`). The IFEval-reply panel: 300 IFEval prompts
with native replies, 116,318 positions. Every 1017 pack below was scored here with every EXL3 tensor it ships loaded.

| pack | routed experts, bpw | our panel KLD / p99 / top-1 | second panel KLD / p99 / top-1 | IFEval-reply KLD / top-1 |
|---|---|---|---|---|
| **v2.0 build** | 2.80 (3.10 per decoded token) | **0.0198** / **0.276** / 95.94% | **0.0213** / **0.317** / 95.967% | 0.0980 / 92.18% |
| [HeNryous 2.75 bpw](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark) @ `ede209db` (dense K5, head K6) | 2.75 (3.10 per decoded token) | 0.0201 / 0.285 / **95.96%** | 0.0216 / 0.330 / **95.969%** | **0.0974** / **92.27%** |
| [object-oriented-architects 2.7 bpw](https://huggingface.co/object-oriented-architects/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-TensorFold) @ `cdb167c4` | 2.7 | 0.0232 / 0.325 / 95.65% | 0.0249 / 0.381 / 95.649% | 0.1201 / 91.31% |
| v1.2 build | 2.70 (2.90 per decoded token) | 0.0243 / 0.339 / 95.47% | 0.0262 / 0.386 / 95.492% | 0.1199 / 91.37% |
| [LibertAIDAI 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) @ `c178a012` | 2.9 | 0.0304 / 0.383 / 94.76% | 0.0322 / 0.433 / 94.698% | 0.1135 / 91.44% |

Perplexity, our panel / second panel (native 1.6005 / 1.5720): v2.0 1.6117 / 1.5869, HeNryous 1.6119 / 1.5871,
object-oriented-architects 1.6146 / 1.5931, v1.2 1.6159 / 1.5946, LibertAIDAI 1.6155 / 1.6040.

By domain, KLD on our panel / the second panel:

| domain (share of positions) | v2.0 build | HeNryous 2.75 bpw | object-oriented-architects 2.7 bpw | v1.2 build | LibertAIDAI 2.9 bpw |
|---|---|---|---|---|---|
| code (65% / 68%) | **0.0122** / **0.0124** | 0.0125 / 0.0126 | 0.0140 / 0.0142 | 0.0150 / 0.0153 | 0.0211 / 0.0210 |
| agentic (7% / 7%) | **0.0175** / **0.0241** | 0.0183 / 0.0242 | 0.0208 / 0.0269 | 0.0221 / 0.0280 | 0.0329 / 0.0393 |
| math (4% / 4%) | 0.0159 / 0.0147 | **0.0152** / **0.0144** | 0.0188 / 0.0179 | 0.0202 / 0.0194 | 0.0252 / 0.0253 |
| long (6% / 7%) | **0.0256** / 0.0266 | 0.0258 / **0.0249** | 0.0285 / 0.0302 | 0.0304 / 0.0316 | 0.0346 / 0.0385 |
| knowledge (6% / 5%) | 0.0396 / **0.0495** | **0.0382** / 0.0502 | 0.0474 / 0.0592 | 0.0479 / 0.0628 | 0.0550 / 0.0706 |
| writing (2% / 1%) | **0.0480** / **0.0748** | 0.0514 / 0.0801 | 0.0561 / 0.0980 | 0.0614 / 0.1008 | 0.0655 / 0.1104 |
| multilingual (3% / 3%) | 0.0585 / 0.0693 | 0.0601 / 0.0734 | 0.0698 / 0.0790 | 0.0697 / 0.0814 | **0.0571** / **0.0637** |
| chat (7% / 6%) | **0.0475** / **0.0594** | 0.0482 / 0.0608 | 0.0568 / 0.0747 | 0.0575 / 0.0739 | 0.0716 / 0.0881 |
| 8 domains weighted equally | **0.0331** / **0.0413** | 0.0337 / 0.0426 | 0.0390 / 0.0500 | 0.0405 / 0.0517 | 0.0454 / 0.0571 |

Raw: `data/kld/ours_1017_28bpw_on_{our,n1017,ifeval}_panel.json`, `henryous_1017_275bpw_on_*`, `ooa_1017_27bpw_on_*`,
`ours_1017_ship_on_*`, `libertai_1017_29bpw_on_*`. The v2.0 build is closest over all positions and with the domains
weighted equally on both panels, and on 6 of the 8 domains of each against HeNryous's pack, which is level on top-1 and
a little closer on the IFEval-reply panel. A drafter never moves these numbers: they score the target's weights.

Before v2.0's composition was picked (Oct 20-21, our panel): the same experts with dense K5 and a K6 head 0.0194 /
96.00%, with `wq_b` K4 0.0196, with the head at K5 0.0197, with both (shipped) 0.0198. `wq_b` K4 and the head K5 bought
the shorter rounds for 0.0004 of KLD.

## v2.0: suites

Greedy, thinking off, 2,048 max tokens, paired per item (McNemar) against native 1017 on OpenRouter's FP8 providers
(`data/evals/native_1017_openrouter/`, `data/evals/ours_1017_28bpw/`).

| suite | native 1017 | v2.0 build | paired p | v1.2 build |
|---|---|---|---|---|
| HumanEval+ (164), base / plus | 96.95 / 90.24 | 96.95 / 89.63 | 1.000 / 1.000 | 96.34 / 90.24 |
| MBPP+ (378), base / plus | 91.53 / 77.25 | 90.74 / 77.78 | 0.607 / 0.804 | 89.68 / 76.19 |
| IFEval (541), prompt strict / loose | 90.39 / 92.05 | 90.76 / 92.42 | 0.856 / 0.845 | 89.09 / 91.13 |
| IFEval, instruction strict | 93.65 | 93.76 | | |

No suite differs significantly from native 1017.

## v2.0: release gates

The release image (`dsv41-serve:v5`) with the launcher's defaults exactly as a user runs them, vision on, on the
release folder (the files staged for Hugging Face plus the Engram shards), Oct 22 12:21-13:00 ICT. Raw: `data/gates/v20/`.

| gate | result |
|---|---|
| the image's tests on GB10 (patch tests + plugin tests) | 702 passed, 9 skipped |
| OpenAI API check (models, chat, streaming, usage, stop, max_tokens, seed, thinking on / off, tool calls) | 14 of 14 |
| set b against the head to head's boots | 103.7 / 57.8 / 144.4 tok/s (103.8 / 57.8 / 144.1), the same tokens a round, the same first 400 characters of every reply |
| needles 8K / 32K / 64K / 128K (three needles at 10 / 50 / 90%) | 3/3 at each; prefill 1,699 / 1,638 / 1,678 / 1,631 tok/s on the ladder's documents, decode after the prompt 83.9 / 85.7 / 81.7 / 81.3 tok/s |
| prefill, `kit_bench.py prefill --lengths 8192,32768 --reps 3` | 2,055 / 2,061 tok/s (reps 2,042-2,060 / 2,037-2,070) |
| MemAvailable, head / worker | 6 / 6 GiB after load, 4 / 4 after the 128K rung |
| oracle file (24 prompts: greedy 64 tokens + top-20 logprobs at every position) | `kit_oracle.jsonl`, the reference for `scripts/bench/oracle_compare.py` against another install |

Not run on the v2.0 build: the 256K rung and the 2-hour soak.


## Earlier builds: decode, same boxes and client

In the sections from here down, "this build" is v3 on the launch checkpoint (`2cba9e42`) and "the 1017 build" is v1.2.

| build | code | prose | structured | raw |
|---|---|---|---|---|
| **1017 build** (DeepSeek-V4.1-Flash-1017), 9 reps, same session as the next row | **100.7** (100.4-101.1) | **59.2** (59.1-59.5) | **141.3** (140.5-141.7) | `data/speed/r1017_setb9` |
| v1.1 build (2cba9e42) rerun in the same session, 9 reps | 103.0 (102.4-103.6) | 58.0 (57.7-58.2) | 142.6 (140.5-143.2) | `data/speed/r1017_setb9` |
| **1017 build**, Oct 19 same-box session, two fresh boots, 9 reps each | **101.1 / 100.9** | **59.3 / 59.1** | **142.5 / 142.5** | `data/speed/samebox_1017` |
| TensorFold `8e0966e2` + object-oriented-architects 1017 2.7 bpw pack (lucataco's settings), same session, two boots, 9 reps each | 99.5 / 98.6 | 59.1 / 58.2 | 143.4 / 140.8 | `data/speed/samebox_1017` |
| **1017 build**, Oct 19 variant session (control), fresh boot, 9 reps | **100.4** (100.1-100.9) | **59.0** (58.7-59.3) | **142.1** (140.5-143.4) | `data/speed/variants_1017` |
| same experts, dense K5 + head K4 (KLD 0.0227, not shipped), its own depth table, same session | 94.8 (94.0-95.3) | 54.4 (54.2-54.6) | 138.6 (137.9-139.0) | `data/speed/variants_1017` |
| same experts, dense K5 + head K6 (KLD 0.0219, not shipped), its own depth table, same session | 95.1 (94.6-95.5) | 56.7 (56.6-56.8) | 135.3 (134.9-136.0) | `data/speed/variants_1017` |
| **this build**, 9 reps | **103.0** (102.4-103.4) | 58.3 (58.1-58.6) | **141.3** (140.8-142.1) | `data/speed/slowmode_servemode` |
| this build, the speed-suite run | 102.2 | 58.1 | 142.6 | `data/speed/ours_v3_full` |
| this build, image rebuilt from this repo by `install.sh`, fresh boot | 102.2 | 58.2 | 141.5 | `data/speed/release_v4` |
| v1.1 (TensorFold's prompt kernels on), image rebuilt by `install.sh`, fresh boot, 3 reps | 102.7 | 57.9 | 141.0 | `data/speed/release_v11` |
| TensorFold v0.6.0 (Zig engine `ce20a54`) + Mia-AiLab 2.9 bpw | 101.0 | **62.6** | 140.3 | `data/speed/tf_v060_samebox` |
| same, 9 reps | 100.5 | 62.1 | 139.0 | `data/speed/tf_v060_samebox_b9` |
| TensorFold v0.5.1 (Python engine `90b68e0`) + Mia-AiLab 2.9 bpw | 100.4 | 60.8 | 140.7 | `data/speed/tf_v051_samebox` |
| Mia-AiLab vLLM kit + her 2.9 bpw pack, DSpark k = 3 | 41.6 | 25.1 | 40.4 | `data/speed/mia_kit_stock` |

- Against TensorFold v0.6.0 (9-rep medians): code +2.5%, structured +1.7%, prose 93.9%.
- Greedy output is identical across reps for this build and for TensorFold. The rebuilt image's run also gave the
  same greedy continuations as the measured image's on all 24 oracle prompts, with the same DSpark counts (2,533 of
  5,140 drafts accepted).
- TensorFold's 9 reps include slow ones on all three prompts (structured 124.7 and 126.0); ours, none in 27.
- TensorFold on these boxes needed two network constants changed (path MTU 1024 instead of 4096, RoCE GID index 3
  instead of 5; `scripts/bench/tensorfold_zig_samebox.sh`). It ran at 98% of its author's published 103.1 / 64.1 /
  143.2 on all three prompts.
- Mia-AiLab's kit ran as shipped (`scripts/bench/mia_kit_samebox.sh`).

## The 1017 build (v1.2): fidelity and suites

Same pipeline, plan settings and composition as the build below, on `deepseek-ai/DeepSeek-V4.1-Flash-1017` @
`1bae0e10`; scored against native 1017 on the same panel, and against LibertAIDAI's Mia-layout 1017 pack, the first 1017 EXL3 pack on Hugging Face
(`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` @ `c178a012`, every EXL3 tensor it ships overlaid; the weights are the
same at `9a85740a`, Oct 19). The
IFEval-reply panel is 300 IFEval prompts with native replies (116,318 scored positions).

| | 1017 build | LibertAIDAI 1017 2.9 bpw |
|---|---|---|
| KLD, mean | **0.0243** | 0.0304 |
| KLD, 8 domains weighted equally | **0.0405** | 0.0454 |
| top-1 agreement | **95.47%** | 94.76% |
| perplexity (native 1.6005) | 1.6159 | 1.6155 |
| domains closer to native | 7 of 8 | multilingual (0.0571 vs 0.0697) |
| IFEval-reply panel KLD / top-1 | 0.1199 / 91.37% | **0.1135 / 91.44%** |
| raw | `data/kld/ours_1017_ship_on_our_panel.json`, `..._on_ifeval_panel.json` | `data/kld/libertai_1017_29bpw_on_*.json` |

| suite | native 1017 | 1017 build | paired p |
|---|---|---|---|
| HumanEval+ (164), base / plus | 96.95 / 90.24 | 96.34 / 90.24 | 1.000 / 1.000 |
| MBPP+ (378), base / plus | 91.53 / 77.25 | 89.68 / 76.19 | 0.143 / 0.523 |
| IFEval (541), prompt strict / loose | 90.39 / 92.05 | 89.09 / 91.13 | 0.281 / 0.442 |

No suite differs significantly from native 1017; IFEval prompt strict sits 1.5 points above the 2cba9e42 build's
(87.6). Native 1017 ran on OpenRouter's FP8 providers (`data/evals/native_1017_openrouter/`). Release gates on the
1017 build: API checks 14 of 14, needles 3 of 3 at 8K / 32K / 64K / 128K with decode 79.0 / 80.9 / 78.1 / 84.8 tok/s
after the prompt, 8 / 9 GiB free per box after load (`data/gates/r1017/`).

We scored one more 1017 pack on both panels with the same harness, every EXL3 tensor it ships overlaid: the
[object-oriented-architects 2.7 bpw pack](https://huggingface.co/object-oriented-architects/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-TensorFold)
@ `cdb167c4` (routed experts made with this recipe on replies written by 1017, every other tensor LibertAIDAI's; the
TensorFold side of the same-box speed test): KLD 0.0232 (domains weighted equally 0.0390), top-1 95.65%, perplexity
1.6146; IFEval-reply panel 0.1201 / 91.31% (`data/kld/ooa_1017_27bpw_on_*.json`). It is closer to native than this
build on 7 of the 8 domains (multilingual even); this build's experts with the dense set at K5 and the head at K6
(1017r, below) are closer than it on all 8.

Two more 1017 packs report KLD on this panel, both made with this recipe:
[HeNryous's 2.75 bpw build](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark), with
the dense set at K5 and the head at K6, 0.0201 / 95.96% top-1 (rerun here for v2.0: the same), and
[0xSero's 2.7 bpw build](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw), 0.0304 / 94.95%. The
dense and head bits are most of the gap to the first: our experts with the dense set at K5 and the head at K6 (1017r)
measure 0.0219 / 95.75%, with the dense set at K5 and the head at K4 0.0227 / 95.64%, and HeNryous reports 0.0225 for
theirs with both at K4. This build keeps K4 / K4 for decode speed. Measured in one session on these boxes, each with
its own depth table, dense K5 adds ~1 ms a verify round and the K6 head ~1 ms more: 94.8 / 54.4 / 138.6 and 95.1 /
56.7 / 135.3 tok/s against this build's 100.4 / 59.0 / 142.1, so both heavier compositions run slower than the
TensorFold setup above while scoring closer to native (`data/speed/variants_1017`).

## The launch build

v3 on `deepseek-ai/DeepSeek-V4.1-Flash` @ `2cba9e42`, image `dsv41-serve:v4` (v1.1).

### The rest of the speed suite (this build)

`scripts/bench/serve_speed_suite.sh`, one run on the recipe defaults; DSpark off from a boot with `SPEC=` (empty).

| measure | result | raw |
|---|---|---|
| DSpark acceptance, set b | 49.3% of drafted tokens; 2.46 accepted per verify step + 1 | `data/speed/ours_v3_full/acceptance.txt` |
| decode, set a | 86.6 / 54.2 / 103.7 tok/s | `data/speed/ours_v3_full/kit_decode_a.json` |
| decode, DSpark off | 44.6 / 44.4 / 44.5 tok/s | `data/speed/ours_v3_nospec` |
| 2 streams at once (code + prose, 256 tokens each) | 81.6 / 80.2 tok/s aggregate (2 reps) | `data/speed/ours_v3_full/kit_conc.json` |
| prefill, 8K / 32K prompt, 3 reps (same-box run below) | 2,090 / 2,092 tok/s since v1.1 (v1: 1,763 / 1,773) | `data/speed/ours_v3_prefill/kit_prefill.json` |
| prefill in the suite run, 1 rep (cold Engram rows) | 1,565 / 1,764 tok/s | `data/speed/ours_v3_full/kit_prefill.json` |
| decode after a 32K prompt | 128.2 tok/s (`kit_bench.py depth`: random-token prompt, 256 tokens); 81.7 on a real 32K document (needle ladder, 512 tokens) | `data/speed/ours_v3_full/kit_depth.json` |

### Prefill, same boxes and client

Both engines on fresh boots (each drops the page cache), `kit_bench.py prefill --lengths 8192,32768 --reps 3`: a new
random-word prompt every rep, so no prefix cache; time to the first token, prompt tokens over it. Oct 12, 13:21-13:36.

| build | 8K | 32K | raw |
|---|---|---|---|
| this build, v1 | 1,763 (1,407 / 1,763 / 1,781) | 1,773 (1,768 / 1,776 / 1,773) | `data/speed/ours_v3_prefill` |
| this build, v1.1 (Oct 13, the serve warm) | 2,090 (2,090 / 2,097 / 2,090) | 2,092 (2,101 / 2,092 / 2,092) | `data/speed/release_v11/kit_prefill_3rep.json` |
| TensorFold v0.6.0 + Mia-AiLab 2.9 bpw | **2,163** (2,111 / 2,193 / 2,163) | **2,276** (2,295 / 2,276 / 2,252) | `data/speed/tf_v060_samebox_prefill` |

TensorFold prefilled 1.23x / 1.28x faster than v1 here, and 1.03x / 1.09x faster than v1.1 (its author publishes 2,433 / 2,495 on his boxes). Each engine's first 8K
rep finds Engram's rows cold.

### How the speed came (code, tok/s)

| when (Oct 2026) | code / prose / structured | what changed |
|---|---|---|
| Oct 9 | 59.3 / 38.7 / 73.7 | first full serve of our EXL3 weights (build v2.1, DSpark k = 3) |
| Oct 10 | 85.6 / 47.1 / 121.4 | our routed-expert decode kernel (`exl3_smoe`), k = 5 with per-round depth (patch 0010), Markov bias cache (0011), the paced L2 prefetch and the RDMA ring, the drafter's dense on EXL3 |
| Oct 11 05:11 | 89.6 / 50.6 / 127.0 | build v3 experts, local draft argmax (0012), smoe run flags, mHC overlap (0014), Engram spin reader (0015) |
| Oct 11 08:18 | 95.3 / 52.9 / 132.2 | smoe evict-first, short-context graph twins (0019), our router with its gate in L2 |
| Oct 11 12:07 | 98.2 / 53.9 / 136.4 | dense linears at EXL3 K4 (`wo_a` K5) |
| Oct 11 14:56 | 99.9 / 54.4 / 138.5 | `wo_a` reads its head groups in place (grouped `exl3_mrow`) |
| Oct 11 20:40 | 101.0 / 55.7 / 139.7 | head K4, the depth table re-measured |
| Oct 11 23:12 | 102.5 / 57.5 / 141.3 | the drafter's routed experts on EXL3 K3 |
| Oct 12 00:23 | **103.0 / 58.3 / 141.3** | the depth table re-measured on that build; the page-cache guard off while serving |

### Fidelity on our panel

KL(native || quantized) over 254,846 scored assistant positions of 199 native conversations, every build scored by
DeepSeek's own `inference/model.py`. Method: README, "Quality method".

| | this build | Mia-AiLab 2.9 bpw | 0xSero J268 |
|---|---|---|---|
| KLD, mean | **0.0243** | 0.0295 | 0.0327 |
| KLD, 8 domains weighted equally | **0.0398** | 0.0443 | 0.0516 |
| top-1 agreement | **95.57%** | 94.99% | 95.00% |
| perplexity (native 1.4978) | **1.5167** | 1.5221 | 1.5277 |
| raw | `data/kld/ours_v3_ship_on_our_panel.json` | `data/kld/mia_29bpw_on_our_panel.json` | `data/kld/sero_J268_on_our_panel.json` |

### Suites

Greedy, thinking off, 2,048 max tokens, paired per item against the native model on the official weights (the scored
files for native, this build and TensorFold are in `data/evals/`).

| suite | native | this build | TensorFold v0.5.1 + Mia-AiLab 2.9 bpw |
|---|---|---|---|
| HumanEval+ (164), base / plus | 97.6 / 92.1 | 97.0 / 90.9 | 95.7 / 90.9 |
| MBPP+ (378), base / plus | 91.0 / 77.5 | 91.3 / 78.6 | 92.1 / 78.8 |
| IFEval (541), prompt strict / loose | 89.5 / 91.5 | 87.6 / 90.4 | 89.1 / 91.1 |

No suite differs significantly from native or from TensorFold (IFEval prompt strict: paired p = 0.13 against native,
0.29 against TensorFold).
