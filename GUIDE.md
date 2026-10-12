# Guide: install, serve, verify, troubleshoot

The README's quickstart in full, with the checks between the steps and what to do when one fails. Two DGX Sparks,
called rank 0 (the head, it serves the API) and rank 1 below, and a control machine that reaches both over ssh (one of
the Sparks works).

## 0. Before you start

You need, on both Sparks:

- DGX OS with driver 580.x (tested on 580.142 and 580.173.02), Docker with the NVIDIA container toolkit;
- the ConnectX-7 cable between them, an IPv4 address on the fabric port (`enp1s0f0np0` by default) and its RoCE device
  (`rocep1s0f0`);
- passwordless ssh from the control machine, and `sudo -n` allowed for the launcher's one root command
  (`tee /proc/sys/vm/drop_caches`, see Troubleshooting);
- about 430 GB free at the same absolute path: the model folder is 399 GB (196 GB of our weights + the official 203 GB
  Engram tables), the serving image about 25 GB.

`check.sh` looks at both from the control machine, read-only, one line per check (PASS / WARN / FAIL):

```bash
HEAD=<rank0> WORKER=<rank1> ./check.sh
```

It checks ssh, the GB10 and its driver, Docker and the NVIDIA container toolkit, the fabric port's IPv4 and its RoCE v2
GID, `sudo -n`, the serving image, free disk and memory, and ssh between the boxes over the fabric. Before the install
only the image line fails; with `MODEL_DIR=/absolute/path` after the weights (section 2) it also checks the model
folder on both boxes. DGX OS registers the GPU with Docker through CDI, so `docker info` lists no `nvidia` runtime
there: that is expected.

## 1. Install

Steps 1 and 2 of the README quickstart: clone this repo to `~/dsv41_recipe` on both Sparks, then on one Spark, with
nothing else heavy running on either:

```bash
cd ~/dsv41_recipe && PEER=<rank1> ./install.sh
```

It runs, in order (each script also runs alone):

1. the vLLM source at the pin `6e517b15` and exllamav3 v1.6.0, cloned into `~/src` if missing;
2. the vLLM wheel for `sm_121` (`scripts/build/run_build_gb10.sh`, a memory-capped CPU compile in the pinned
   build-environment image) and the exllamav3 1.6.0 wheel (`scripts/build/run_build_exllamav3.sh`), each skipped when
   its wheel is already in `wheels/` (`FORCE=1` rebuilds);
3. `dsv41-recipe:dev`: the wheel + every patch in `patches/` (0001-0008, 0010-0021), each verified in the image by a
   reverse apply (`scripts/image/build_image.sh`); `dsv41-quant:dev`: + exllamav3 (`build_quant_image.sh`);
   `dsv41-serve:v4`: + the `vllm_exl3_moe` plugin (`build_serve_image.sh`), the image the launcher runs by default;
4. the serving image to `PEER` (`docker save | ssh docker load`); it stops if the two image ids differ.

It refuses to start while a GPU container runs on either box: heavy builds and image copies beside GPU work have hung a
Spark (FINDINGS.md 6). Afterwards `./check.sh` shows the same image id on both boxes.

The patches in `patches/experimental/` are measured experiments outside the image; their README shows how to layer
one on.

## 2. Weights

Step 3 of the quickstart: our files from Hugging Face and the two Engram shards from DeepSeek's repo into the same
folder, on both Sparks at the same path, then `sha256sum -c MANIFEST.sha256` there. The second Spark can copy the
folder from the first over the ConnectX link instead (about 550 MB/s in our runs), with the GPUs idle.

## 3. Serve

```bash
HEAD=<rank0> WORKER=<rank1> MODEL_DIR=/absolute/path/to/dsv41-exl3 ./scripts/launch/dsv41_tp2.sh up
./scripts/launch/dsv41_tp2.sh status            # logs [head|worker] | down
```

`up` drops the page cache on both boxes, starts rank 1 then rank 0, waits for `/v1/models` and prints MemAvailable per
node. Expect about 9 minutes to ready and about 9 GiB MemAvailable on each node. Every setting is an environment
variable (README, "Launcher settings"); the defaults are the measured-best configuration.

## 4. Verify

```bash
H=http://<rank0>:8000/v1
curl -s $H/models
curl -s $H/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dsv41", "max_tokens": 64, "temperature": 0,
  "messages": [{"role": "user", "content": "Say hello in five words."}],
  "chat_template_kwargs": {"enable_thinking": false}}'
```

`"enable_thinking": true` turns the reasoning on; tool calls use the standard `tools` field (V4.1's tool-call and
reasoning parsers are on). Greedy output is identical run to run, so a repeated request is a quick health check.

## 5. Measure

```bash
curl -LO https://raw.githubusercontent.com/bertholomus/TensorFold/fe6f9f7b373db76422ddfa121577238474c0a1b7/tools/dsv41/kit_bench.py
python3 kit_bench.py --base http://<rank0>:8000 --model dsv41 decode --set b --tokens 384 --reps 3
SERVE_ID=mine KIT=$PWD/kit_bench.py HEAD=<rank0> ./scripts/bench/serve_speed_suite.sh   # the README's speed table
```

Results land in `./results/speed/<SERVE_ID>` (or `$DSV41_RESULTS`). Run benchmarks with nothing else on either GPU.
Our numbers, for comparison: set b 103.0 / 58.3 / 141.3 tok/s (code / prose / structured), 49% of drafts accepted.

## 6. Memory guards

Two modes, one script (`scripts/quantize/guards.sh`, on each Spark):

- `guards.sh start` for encodes and passes: drops the page cache every 5 s and watches memory. Without it a pass can
  run out of memory with plenty "available", or thrash unified memory (FINDINGS 1, 3).
- `guards.sh serve` for serving: the memory watch only. Dropping caches during a serve evicts Engram's hot rows and
  slows decode at random (FINDINGS 2).

## 7. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `fabric not ready` from the launcher | no IPv4 on `FAB_IF`, or no RoCE v2 GID for it | bring the port up with an address; check `FAB_IF` / `FAB_HCA` |
| NCCL fails at init, "no IP" or wrong device | NCCL picked a port without an IPv4 | the launcher pins `NCCL_IB_HCA` and the GID index of the fabric IPv4; keep `FAB_HCA` matching `FAB_IF` |
| `CUDA error: out of memory` at boot or in a pass, with lots "available" | GB10 CUDA allocations do not reclaim the page cache | the launcher drops caches before boot (needs `sudo -n`); `guards.sh start` during passes |
| MemAvailable shrinks while serving | FlashInfer autotune on, or another job on the box | keep `KERNEL_CONFIG` with autotune off; one serve per box |
| one rank exited, the other hangs | a TP pair cannot heal one rank | `dsv41_tp2.sh down`, then `up`; the containers run with `--restart no` on purpose |
| decode slower on some runs, at random | page-cache drops during the serve | `guards.sh serve` on both boxes |
| a Spark stops answering during a copy | multi-GB file I/O beside a GPU job (FINDINGS 6) | power-cycle it; copy images and weights with the GPUs idle |
| the quantization scripts cannot find files | their paths default to `/home/spark/...` on the Sparks | set `RUN` / `CALDIR` / `OFFICIAL` / `PLAN` / `P1`, or `sudo ln -s "$HOME" /home/spark` (NOTES.md) |

## 8. Reproduce the weights

[RECIPE.md](RECIPE.md): calibration data, pass 0, sensitivity, allocation, the two-box encode, the dense pass, the
drafter, assembly and the KLD panel, each with its script, inputs and time. Start the encode guards first.
