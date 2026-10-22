# Same-box speed, v2.0 (Oct 22 2026)

Two engines on the same two DGX Sparks in one session (08:28-09:07 ICT), alternating fresh boots, TensorFold first,
measured with the same client: `kit_bench.py` set b (sha256 `83a4fbeb...`, the file TensorFold's DeepSeek branch ships
and lucataco's recipe pins), 384-token replies, a 64-token warm-up, then 9 reps a prompt. Greedy, thinking off, one
stream. Before every boot, on both boxes: page cache dropped, memory compacted, the free 2 MiB-and-up blocks logged
(82-88 GB on the head, 62-68 GB on the worker), so both engines start from the same memory state (FINDINGS 24).

- `sb_tf_b1.json`, `sb_tf_b2.json`: TensorFold's DeepSeek-V4.1 TP2 engine at `8e0966e2` (bertholomus/TensorFold,
  branch `deepseek-v41-tp2`), built with lucataco's recipe Dockerfile (NVIDIA PyTorch 26.07 + TensorFold at that commit),
  serving `object-oriented-architects/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-TensorFold` @ `cdb167c4` with DeepSeek's 1017
  Engram shards and the recipe's patched chat template (sha256 `13f33e7f...`) in the pack's folder. One container per
  box: `tensorfold serve /model --tp 2 --rank <r> --context 1048576 --vision --parallel 4 --mtp-drafts 5`, with the
  recipe's environment (`TF_DS_REPLAY=1 TF_DS_REPLAY_FLOOR=1
  TF_DS_PREFILL_CHUNK=2048 TF_DS_RANK_CACHE_READERS=32 TF_DS_WARM_LENGTHS=1,17,33,131,514,1024,2113`, Engram from the
  official shards) and NCCL / RDMA pinned to the ConnectX port. One change for our network: its RDMA gather hard-codes a
  4096-byte path MTU, and our link runs RoCE MTU 1024, where 4096 drops every packet, so the container sets 1024 before
  starting (as in `data/speed/samebox_1017`).
- `sb_ours_b1.json`, `sb_ours_b2.json`: this repo's v2.0 build (`dsv41-recipe-exl3-1017q310r-dk5-dx3-wqb4h5-dh4`, the
  files of `sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`) on the v2.0 engine's code, the v2.0 launcher defaults with
  vision on (`TEXT_ONLY=0`).

| tok/s, median of 9 (range) | code | prose | structured |
|---|---|---|---|
| TensorFold, boot 1 | 98.4 (91.1-99.1) | 58.0 (56.2-58.4) | 141.3 (126.2-142.5) |
| TensorFold, boot 2 | 98.6 (90.8-99.1) | **58.4** (52.9-58.6) | 140.9 (138.1-141.4) |
| v2.0 build, boot 1 | **103.9** (103.3-104.2) | 57.9 (57.6-58.1) | 144.0 (143.2-144.7) |
| v2.0 build, boot 2 | 103.6 (103.4-104.1) | 57.7 (57.6-58.1) | **144.1** (143.3-144.7) |

Two-boot means: TensorFold 98.5 / 58.2 / 141.1, v2.0 103.8 / 57.8 / 144.1: code +5.3%, prose -0.6%, structured +2.1%.
Tokens per verify round (streamed chunks) and milliseconds per round, v2.0 against TensorFold: code 4.47 vs 4.27 at
43.0 vs 43.3-43.4 ms, prose 2.10 vs 1.99 at 36.3 vs 34.1-34.3 ms, structured 5.91 for both at 41.0 vs 41.8-41.9 ms.
Every rep's text was identical within each run. After load: TensorFold 12-13 GiB free per box, v2.0 5-7 GiB.

A second session, the release gates (12:21-13:00 ICT): one fresh boot each with the same memory preparation, the
release image `dsv41-serve:v5` with the launcher's defaults as a user runs them, vision on, in this order:

- `sb2_ours.json`: the v2.0 build, launcher defaults: 103.7 (103.0-104.5) / 57.8 (57.5-58.0) / 144.4 (143.6-144.7).
- `sb2_v12.json`: v1.2's build (`sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw`) on the same image, its own round table
  (`VLLM_DSPARK_ROUND_MS=27.3,31.3,34.7,38.1,41.1,43.8`, depth policy `conf`): 106.4 (105.8-107.4) / 61.8 (60.0-62.1) /
  149.1 (148.5-149.4).
- `sb2_tf.json`: TensorFold as above: 97.7 (94.3-98.1) / 57.5 (54.6-58.2) / 140.4 (123.4-140.6).

Against TensorFold in this session: the v2.0 build +6.1% / +0.5% / +2.8%, v1.2's build on the v2.0 engine +8.9% /
+7.5% / +6.2%. After load: TensorFold 13 / 13 GiB free, the v2.0 build 6 / 6, v1.2's build 9 / 10.

A third session (13:18-14:08 ICT), TensorFold's newest DeepSeek TP2 commit: `af26d42b` (its v0.6 release of Oct 18 plus
two fixes), built with lucataco's Dockerfile at that commit (`TF_COMMIT=af26d42b...`) with a rank cache of its own, the
same launch and pack as above; alternating fresh boots, the same memory preparation before each:

- `sb3_tf06_b1.json`, `sb3_tf06_b2.json`: TensorFold `af26d42b`: 98.6 / 58.5 / 140.7 and 98.2 / 57.4 / 140.3 (the first
  boot wrote its rank cache, 561 s; the second came up in 155 s).
- `sb3_ours_b1.json`, `sb3_ours_b2.json`: the v2.0 build, launcher defaults: 104.2 / 57.8 / 143.6 and 103.6 / 57.7 /
  143.1.
- `sb3_v12.json`: v1.2's build on the v2.0 engine, its own table: 106.2 / 62.0 / 147.6.

Two-boot means: v2.0 +5.6% / -0.3% / +2.0% (code / prose / structured), v1.2's build +7.9% / +7.0% / +5.0%.
