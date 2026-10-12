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
- **Our build:** the recipe defaults (`dsv41_tp2.sh up`, image `dsv41-serve:v4`, build v3), the page-cache guard in
  serve mode (`guards.sh serve`).

## Decode, same boxes and client

| build | code | prose | structured | raw |
|---|---|---|---|---|
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

## The rest of the speed suite (this build)

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

## Prefill, same boxes and client

Both engines on fresh boots (each drops the page cache), `kit_bench.py prefill --lengths 8192,32768 --reps 3`: a new
random-word prompt every rep, so no prefix cache; time to the first token, prompt tokens over it. Oct 12, 13:21-13:36.

| build | 8K | 32K | raw |
|---|---|---|---|
| this build, v1 | 1,763 (1,407 / 1,763 / 1,781) | 1,773 (1,768 / 1,776 / 1,773) | `data/speed/ours_v3_prefill` |
| this build, v1.1 (Oct 13, the serve warm) | 2,090 (2,090 / 2,097 / 2,090) | 2,092 (2,101 / 2,092 / 2,092) | `data/speed/release_v11/kit_prefill_3rep.json` |
| TensorFold v0.6.0 + Mia-AiLab 2.9 bpw | **2,163** (2,111 / 2,193 / 2,163) | **2,276** (2,295 / 2,276 / 2,252) | `data/speed/tf_v060_samebox_prefill` |

TensorFold prefilled 1.23x / 1.28x faster than v1 here, and 1.03x / 1.09x faster than v1.1 (its author publishes 2,433 / 2,495 on his boxes). Each engine's first 8K
rep finds Engram's rows cold.

## How the speed came (code, tok/s)

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

## Fidelity on our panel

KL(native || quantized) over 254,846 scored assistant positions of 199 native conversations, every build scored by
DeepSeek's own `inference/model.py`. Method: README, "Quality method".

| | this build | Mia-AiLab 2.9 bpw | 0xSero J268 |
|---|---|---|---|
| KLD, mean | **0.0243** | 0.0295 | 0.0327 |
| KLD, 8 domains weighted equally | **0.0398** | 0.0443 | 0.0516 |
| top-1 agreement | **95.57%** | 94.99% | 95.00% |
| perplexity (native 1.4978) | **1.5167** | 1.5221 | 1.5277 |
| raw | `data/kld/ours_v3_ship_on_our_panel.json` | `data/kld/mia_29bpw_on_our_panel.json` | `data/kld/sero_J268_on_our_panel.json` |

## Suites

Greedy, thinking off, 2,048 max tokens, paired per item against the native model on the official weights (the scored
files for native, this build and TensorFold are in `data/evals/`).

| suite | native | this build | TensorFold v0.5.1 + Mia-AiLab 2.9 bpw |
|---|---|---|---|
| HumanEval+ (164), base / plus | 97.6 / 92.1 | 97.0 / 90.9 | 95.7 / 90.9 |
| MBPP+ (378), base / plus | 91.0 / 77.5 | 91.3 / 78.6 | 92.1 / 78.8 |
| IFEval (541), prompt strict / loose | 89.5 / 91.5 | 87.6 / 90.4 | 89.1 / 91.1 |

No suite differs significantly from native or from TensorFold (IFEval prompt strict: paired p = 0.13 against native,
0.29 against TensorFold).
