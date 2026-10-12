# Contributing

Speed, quality and fit changes are welcome. A change lands when it is measured the way this repo measures, against the
current recipe on two DGX Sparks, with the raw files attached.

## Measure the baseline first

On your boxes, with nothing else on either GPU and the guards in serve mode (`guards.sh serve`, GUIDE.md section 6):

```bash
MODEL_DIR=... ./scripts/launch/dsv41_tp2.sh up
SERVE_ID=base QUICK=1 KIT=$PWD/kit_bench.py HEAD=<rank0> ./scripts/bench/serve_speed_suite.sh
```

`QUICK=1` runs `kit_bench.py` set b (3 reps, greedy, 384 tokens) with DSpark's acceptance counters and the oracle: 24
greedy continuations plus teacher-forced top-20 logprobs, the check that a change kept the model's output.

## Measure your change the same way

- **A knob or a kernel behind an environment variable:** `scripts/bench/p_levers_ab.sh` boots the baseline, then each
  lever alone, ~11 minutes a configuration; `scripts/bench/levers_pick.py` applies the pick rule below. Add your lever
  to the `LEVER` table there.
- **A patch or plugin change:** build an overlay image (`scripts/image/build_serve_overlay.sh`), then the same A/B with
  `IMAGE=` pointing at it.
- **New weights:** the panel KLD first (`scripts/quantize/p4_kld.py`, about 25 minutes a build on one Spark), then the
  speed A/B.

## When a change counts

- **Speed:** verify rounds at least 0.5% faster on structured (ms per round, `scripts/bench/round_ms.py`) and the mean
  over the three prompts not worse. Rounds rather than tok/s, because a change in numerics changes the greedy text and
  with it the acceptance; changes that move acceptance on purpose (drafter, depth table) are judged on tok/s.
- **Output:** a change that keeps the math gives all 24 oracle continuations identical; one that rounds differently
  needs top-1 agreement >= 97% and truncated KL <= 0.025 against the baseline (routing-flip noise; `scripts/bench/levers_pick.py`
  lists the measured spread).
- **Quality:** new weights or anything that changes the model's arithmetic beyond rounding: panel KLD at most 0.0270
  (8% under the 0.0295 of the pack TensorFold serves; the current build is 0.0243), and no suite among HumanEval+,
  MBPP+ and IFEval dropping more than 1 point at p < 0.05, paired against native.
- **Fit:** MemAvailable after load >= 4 GiB on both nodes at `MAX_LEN=262144`.

## What to send

The numbers with their raw files (the `results/speed/<id>/` folders the scripts write), the image id and weights you
measured, and one line on what changed. Pull requests: one change each, with its test when it touches code that has
one (`patches/exl3_moe/tests/`, `patches/tests/`). Code is Apache-2.0; third-party code needs a compatible license and
its notice, as in `NOTICE`.
