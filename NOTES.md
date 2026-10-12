# Notes: open items

What is not done, not measured, or known to be weaker, as of the release. Numbers are `kit_bench.py` set b on two
DGX Sparks unless a line says otherwise.

## Not measured yet

- **A long agent soak.** A 2-hour agent run with restarts is the last stability gate.
- **Image input.** `TEXT_ONLY=0` loads the vision tower; no image test has run on this build.
- **Power and efficiency.** Not measured.
- **The wheel compiles inside `install.sh`.** `install.sh` rebuilt the three images from this repo and copied the
  serving image to the second Spark; its patch list, vLLM sources and plugin hash the same as the image behind the
  numbers, and the recipe defaults on it gave set b 102.2 / 58.2 / 141.5 tok/s with greedy output identical to the
  measured run (24 of 24 oracle continuations). The vLLM and exllamav3 wheels were already built by the same scripts,
  so that run skipped them; a first install compiles them first.

## Weaker than the best build we measured

- **Prose decode:** 58.3 tok/s against 62.1 for TensorFold v0.6.0 on the same boxes (9-rep medians, 94%). Prose drafts less well, so
  a round's fixed cost weighs more; the fixed cost is the lever.
- **Prefill:** 2,090 / 2,092 tok/s at 8K / 32K since v1.1 (TensorFold's grouped EXL3 prompt kernels for the routed
  experts; v1: 1,763 / 1,773) against TensorFold v0.6.0's 2,163 / 2,276 on the same boxes (3-rep medians): 1.03-1.09x
  behind, from 1.23-1.28x. What follows describes v1's gap: Each EXL3 linear in a prompt chunk runs exllamav3's general path (a full weight
  reconstruct, two Hadamard passes and dtype casts around the GEMM); a path that decodes the trellis once per chunk and
  folds the rotations into the GEMM (TensorFold's prompt kernels do) is the lever.
- **Context:** 262,144 tokens a request (the 256K needle rung passes 3/3 with >= 7.5 GiB free on both boxes). A 1M
  window does not fit with this build: with DSpark on, a boot with a 1M window and a 4.5 GiB KV pool a rank leaves 3 GiB
  free; with DSpark off, one box ran out of GPU memory during a 512K prefill. TensorFold serves 1M on the same boxes.
- **Concurrency:** 2 sequences at a time (the 262K context per request takes the memory); 2 streams together give
  81 tok/s. TensorFold serves 4.

## Still on the table for speed

- Per-expert Hessians for the drafter's routed experts (they share one layer-level Hessian today).
- The 1-row latency floor: a verify round with one live row still costs 28.9 ms.

Measured and left out: the dense `wo_a` at K4 (it is K5). Verify rounds got 0.1-0.3 ms faster, but DSpark acceptance
fell a point (48.3% against 49.3%) and the panel KLD rose to 0.0250 from 0.0243, so set b got slower on code and prose
(102.3 / 57.4 / 142.1 against 102.8 / 58.1 / 141.6 in the same session).

## Repo

- **Paths on the Sparks.** The quantization scripts default to `/home/spark/dsv41_recipe` and `/home/spark/models` on
  both Sparks. Each path is an override (`RUN`, `CALDIR`, `OFFICIAL`, `PLAN`, `P1`, `MODEL_DIR`: see the script's
  header), or `sudo ln -s "$HOME" /home/spark` covers them all; one `SPARK_HOME` variable would be cleaner. The
  serving path only needs `MODEL_DIR` and `~/dsv41_recipe`.
- **Upstream.** Patches 0001 and 0003 carry upstream vLLM PRs #56509 and #58560 (the latter merged after our pin). The
  rest are ours; none is proposed upstream yet.
- **Calibration data.** The exact conversations are in `data/caldata/`: `conversations.jsonl` (500 seeded prompts with
  the native model's answers: 301 for calibration, the 199 of the KLD panel) and `instruct_v3.jsonl` (the 392 held-out
  instruction, chat, writing and multilingual prompts of build v3). The generator samples, so rerunning it gives new
  conversations; these files rebuild the exact token sets.
