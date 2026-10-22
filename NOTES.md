# Notes: open items

What is not done, not measured, or known to be weaker, as of the release. Numbers are `kit_bench.py` set b on two
DGX Sparks unless a line says otherwise.

## Not measured yet

- **The 256K rung on the v2.0 build.** Needles pass 3/3 at 8K-128K; the 255,709-token prompt has not run on it (the
  launch build passed it at v1.1 with 6.5 GiB at the lowest). It starts with 6 / 6 GiB free after load with vision on,
  against v1.2's 8 / 9, and had 4 / 4 after the 128K rung: for long prompts serve text-only (the default) and try the
  opt-in `INDEXER_LOGITS_MB=128`, neither measured at 256K on this build yet.
- **A long agent soak.** A 2-hour agent run with restarts is the last stability gate.
- **Image input.** `TEXT_ONLY=0` loads the vision tower, and v2.0's numbers were measured with it loaded. One basic
  test on the 1017 build (Oct 19): a generated PNG, left half red and right half blue, answered "Red, blue." No image
  benchmark or broader test yet.
- **Power and efficiency.** Not measured.
- **A from-scratch `install.sh` of v2.0.** The measured image (`dsv41-serve:v5`) is patches 0023-0038 and the plugin
  layered on v1.1's image. `install.sh` applies the same patch files in order to a fresh wheel: in v5 every patch
  reverses cleanly in order, each patch's code hunks match this repo's files, and the 42 plugin source files are
  byte-identical to this repo's except `csrc/NOTICE.tensorfold` (credit lines added after the build). A full
  `install.sh` run of v2.0 has not been timed or benched; for v1.1 it gave the measured numbers (set b 102.2 / 58.2 /
  141.5 tok/s, greedy output identical on 24 of 24 oracle prompts).

## Weaker than the best build we measured

- **Prose decode:** 57.8 tok/s against TensorFold's 58.2 in the same session (-0.6%). TensorFold's prose rounds are
  about 2 ms shorter (34.1-34.3 against 36.3 ms) and ours accept more drafts (2.10 against 1.99 tokens a round). The
  v2.0 build carries more bytes a round than TensorFold's pack for its KLD. The other two sessions read +0.5% (57.8
  against 57.5) and -0.3% against TensorFold's newest commit: level within the ~1% between boots. v1.2's lighter weights
  on the v2.0 engine read 61.8-62.0.
- **Instruction-reply text:** on the IFEval-reply panel HeNryous's 2.75 bpw pack sits a little closer to native,
  0.0974 against 0.0980 KLD, with top-1 level on both main panels (95.96% against 95.94%). IFEval itself is within noise
  of native 1017 (90.76 against 90.39).
- **Multilingual:** LibertAIDAI's 2.9 bpw pack is closer on the multilingual domain of both panels (0.0571 / 0.0637
  against 0.0585 / 0.0693); v2.0 is closer on the other seven.
- **Memory after load:** 6 / 6 GiB free with vision on against TensorFold's 12-13 GiB, and v1.2's 8 / 9. Keep other
  GPU work off both boxes while serving.
- **Prefill:** 2,055 / 2,061 tok/s at 8K / 32K on v2.0. The launch build since v1.1: 2,090 / 2,092 tok/s against TensorFold
  v0.6.0's 2,163 / 2,276 on the same boxes (3-rep medians).
- **Context:** 262,144 tokens a request. A 1M window does not fit: with DSpark on, a boot with a 1M window and a 4.5 GiB
  KV pool a rank left 3 GiB free on the launch build; with DSpark off, one box ran out of GPU memory during a 512K
  prefill. TensorFold serves 1M on the same boxes.
- **Concurrency:** 2 sequences at a time (the 262K context per request takes the memory); 2 streams together gave
  81 tok/s on the launch build. TensorFold serves 4.

## Still on the table for speed

- The dense and expert kernels' last ~15% to the read ceiling: `exl3_mrow` streams ~200 GB/s where 230-237 GB/s was
  measured, and the experts phase ~158 GB/s with the shared expert and the prefetch beside it. Those phases 10% faster
  would be about 5% on prose.
- Exact engine items in a prose round (the stage copy in the producers' epilogue, the head strips, the compressor's
  FP8 linear on the row GEMM, the next round's prep beside the drafter, Markov cache misses): about 0.6 ms a round on
  paper, less in practice, since programmatic dependent launch already hides most small kernels (FINDINGS 28).
- More quality per byte: KLD freed by a better encode would let one dense group go back to K4, about 0.38 ms a round
  each.
- Per-expert Hessians for the drafter's routed experts (they share one layer-level Hessian today).

Measured and left out: the dense `wo_a` at K4 (it is K5). Verify rounds got 0.1-0.3 ms faster, but DSpark acceptance
fell a point (48.3% against 49.3%) and the panel KLD rose to 0.0250 from 0.0243, so set b got slower on code and prose
(102.3 / 57.4 / 142.1 against 102.8 / 58.1 / 141.6 in the same session; launch build). The drafter's dense at K4 or K3
cost prose acceptance (-0.6%).

## Repo

- **Paths on the Sparks.** The quantization scripts default to `/home/spark/dsv41_recipe` and `/home/spark/models` on
  both Sparks. Each path is an override (`RUN`, `CALDIR`, `OFFICIAL`, `PLAN`, `P1`, `MODEL_DIR`: see the script's
  header), or `sudo ln -s "$HOME" /home/spark` covers them all; one `SPARK_HOME` variable would be cleaner. The
  serving path only needs `MODEL_DIR` and `~/dsv41_recipe`.
- **Upstream.** Patches 0001 and 0003 carry upstream vLLM PRs #56509 and #58560 (the latter merged after our pin). The
  rest are ours; none is proposed upstream yet.
- **The KLD harness loads a layer, then computes it.** `p4_kld.py` streams the model layer by layer, and each layer's
  weights and Engram rows load serially with its compute: during the 1017 reference runs the GPUs averaged 56-60%
  busy, a third of the 1 s samples read under 20%, at 44 W against 80-90 W under pass 1 (Oct 19). Prefetching the
  next layer during the current one would cut a KLD stage by roughly a quarter. It changes the quality gate's
  harness, so it goes in with an A/B against these numbers (same panel, bit-identical KLD), not mid-build.
- **`DOCKER_ENV` values reach the remote shell unquoted.** `dsv41_tp2.sh` adds each entry as `-e KEY=VALUE` to a
  `docker run` that runs through ssh, so a value with glob characters (`[0-9]`, parentheses, `|`) breaks on a box
  whose login shell is zsh ("no matches found"). Quote such a value inside DOCKER_ENV (`KEY='...'`), as
  `hcap_drafter.sh` does; a launcher-side fix (`printf %q` per entry) is open.
- **DeepSeek's encoder fix of Oct 16-17 is not in the image.** DeepSeek changed the reference `encoding/` in both repos
  (launch `314113f`, the 1017 release): a tool result no longer opens a new user turn, so the reasoning before a tool
  call is kept, and a literal image token in text is rejected. vLLM PRs #62196 / #62199 / #62200 port it (open on
  Oct 18). Without it, a multi-turn tool conversation in thinking mode sent with no `tools` field loses its earlier
  reasoning; single-turn prompts (the evals, set b) and requests that declare tools encode the same.
- **Calibration data.** The exact conversations are in `data/caldata/`: `conversations.jsonl` (500 seeded prompts with
  the native model's answers: 301 for calibration, the 199 of the KLD panel) and `instruct_v3.jsonl` (the 392 held-out
  instruction, chat, writing and multilingual prompts of build v3). The generator samples, so rerunning it gives new
  conversations; these files rebuild the exact token sets.
