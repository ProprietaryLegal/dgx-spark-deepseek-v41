# Same-box speed, DeepSeek-V4.1-Flash-1017 (Oct 19 2026)

Two engines on the same two DGX Sparks in one session (16:01-16:36 ICT), each booted twice, measured with the same
client: `kit_bench.py` set b (sha256 `83a4fbeb...`, the file TensorFold's DeepSeek branch ships and lucataco's recipe
pins), 384-token replies, a 64-token warm-up, then 9 reps a prompt. Greedy, thinking off, one stream.

- `sb_tf_b1.json`, `sb_tf_b2.json`: TensorFold's DeepSeek-V4.1 TP2 engine at `8e0966e2` (bertholomus/TensorFold,
  branch `deepseek-v41-tp2`), built with lucataco's recipe Dockerfile (NVIDIA PyTorch 26.07 + TensorFold at that commit)
  and served with that recipe's settings (`--context 1048576 --parallel 4 --mtp-drafts 5`, its `TF_DS_*` environment),
  on `object-oriented-architects/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-TensorFold` @ `cdb167c4` (sha256 of every file
  checked against the Hub) with DeepSeek's 1017 Engram shards and the recipe's patched chat template (sha256
  `13f33e7f...`). One change for our network: its RDMA gather hard-codes a 4096-byte path MTU, and our ConnectX link
  runs MTU 1500 (RoCE active MTU 1024), where 4096 drops every packet, so the container set 1024 before starting
  (the same change as the Oct 12 same-box runs).
- `sb_ours_b1.json`, `sb_ours_b2.json`: this repo's 1017 build (`dsv41-recipe-exl3-1017b-dk5-h4-dx3`) on the release
  image, launcher defaults plus vision on and 1017's depth table (`VLLM_DSPARK_ROUND_MS=28.99,...,45.06`).

| tok/s, median of 9 | code | prose | structured |
|---|---|---|---|
| TensorFold 8e0966e2 + 2.7 bpw pack, boot 1 | 99.5 | 59.1 | 143.4 |
| TensorFold 8e0966e2 + 2.7 bpw pack, boot 2 | 98.6 | 58.2 | 140.8 |
| this build, boot 1 | **101.1** | **59.3** | 142.5 |
| this build, boot 2 | **100.9** | 59.1 | 142.5 |

Every rep's text was identical within each run. Spread over the 9 reps: TensorFold code 91.0-99.9 and 93.7-99.0,
structured 142.3-143.6 and 130.2-141.3; this build code 101.0-101.7 and 100.1-101.5, structured 141.5-143.5 and
141.3-143.5. Tokens per verify round (streamed chunks): code 4.27 vs 4.31, prose 1.99 vs 2.15, structured 5.91 for
both; milliseconds per round: code 42.9-43.3 vs 42.7, prose 33.7-34.2 vs 36.2-36.3, structured 41.2-42.0 vs 41.5.
After load: TensorFold 11-13 GiB free per box, this build 7-9 GiB.
