# Heavier dense and head bits on the 1017 build: speed (Oct 19 2026)

Can the 1017 build's experts with heavier dense and head encodes beat both the speed and the panel KLD of the
TensorFold setup measured the same day on these boxes (`../samebox_1017`: 99.0 / 58.7 / 142.1 tok/s on the two-boot
means, KLD 0.0232)? Two compositions, both hardlink variants of the shipping dir: its routed experts and drafter
files, with only the dense set and the head swapped.

- `setb9_r_dk5_dx3.json`: dense K5 (every module) + the K6 head from the dense pass (`p4_dense_variant.py` with the K5
  dense pass).
- `setb9_r_dk5_h4_dx3.json`: dense K5 + pass 1's K4 head (then `p4_head_variant.py`).
- `setb9_ship_control.json`: the shipping build (dense K4 with `wo_a` K5, head K4), the same session's control.

Method, as `../samebox_1017`: `kit_bench.py` set b (sha256 `83a4fbeb...`), 384-token replies, a 64-token warm-up, then
9 reps a prompt; greedy, thinking off, one stream; a fresh boot for each build (page cache dropped, memory compacted),
vision on, launcher defaults. Each build ran with its own depth table, measured first on its own text-only boot
(`scripts/bench/depth_sweep.sh` at fixed depths): round ms by verify rows 1-6

| build | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| shipping (dense K4, head K4) | 28.99 | 32.76 | 36.11 | 39.34 | 42.43 | 45.06 |
| dense K5, head K4 | 29.68 | 33.55 | 37.07 | 40.69 | 43.55 | 46.26 |
| dense K5, head K6 | 30.65 | 34.49 | 38.15 | 41.62 | 44.50 | 47.23 |

Set b, Oct 19 19:43-20:17, one session:

| tok/s, median of 9 [range] | code | prose | structured | panel KLD / top-1 | IFEval-reply KLD |
|---|---|---|---|---|---|
| shipping, control | **100.4** [100.1-100.9] | **59.0** [58.7-59.3] | **142.1** [140.5-143.4] | 0.0243 / 95.47% | 0.1199 |
| dense K5, head K4 | 94.8 [94.0-95.3] | 54.4 [54.2-54.6] | 138.6 [137.9-139.0] | 0.0227 / 95.64% | 0.1162 |
| dense K5, head K6 | 95.1 [94.6-95.5] | 56.7 [56.6-56.8] | 135.3 [134.9-136.0] | **0.0219 / 95.75%** | **0.1147** |

Every rep's text was identical within each run. Structured writes the same text in all three builds (5.91 tokens a
verify round), so its time a round prices the bits alone: 41.58 ms shipping, 42.62 ms with dense K5 (+2.5%), 43.67
ms with the K6 head on top (+5.0%). On code and prose the dense K5 + head K4 build writes other continuations that
accept fewer drafts (4.17 vs 4.31 and 1.99 vs 2.15 tokens a round), while the K6 head build accepts like the shipping
one (4.27, 2.17). The KLD files are `data/kld/ours_1017r_h4_on_*.json` (head K4) and
`data/kld/ours_1017r_on_*.json` (head K6), both with dense K5.

Neither composition beats that setup on both counts: each is closer to native and 2.5-7.3% slower than it on every
prompt. The shipping build keeps the dense set and head at K4.
