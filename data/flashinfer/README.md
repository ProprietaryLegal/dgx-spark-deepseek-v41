# FlashInfer SM120 sparse-MLA calibration (GB10)

`sparse_mla_sm120_cpb.gb10_oct8_spark.json` / `_spark2.json`: the caches FlashInfer 0.7.0.post1's autotune wrote on
each Spark on Oct 8 2026 (`~/.cache/dsv41-serve/flashinfer/0.7.0.post1/121a/autotune/`). Every serve measured since read
them, each rank its own box's. They differ slightly: the 32-head top-k 256 crossover is 4 on spark, 8 on spark2, and the
chunk-model constants and overrides vary a little. A release ships one file to both nodes, so the ranks route alike. It holds the decode chunk model (`dsv4`) and the kernel crossovers (`decode_max_tokens`): a decode-form call
of N tokens takes the decode kernel only while N <= the crossover for its (heads, top-k), else `prefill_mg_dual`. At 32
heads (V4.1 at TP=2) the crossover is 0 at top-k 128, so the 6-token verify calls go to prefill.

`sparse_mla_sm120_cpb.gb10_dec16.json`: spark's file with every 32- and 64-head dsv4 crossover at 16 (each box's
`~/.cache/dsv41-serve/fi_autotune_dec16/` holds its own file raised the same way) (lever `fidec`,
Oct 11). FlashInfer reads the file under `FLASHINFER_AUTOTUNE_DIR`; without one (and with autotune off, as
our launcher runs) it routes decode-first. Device key "0:NVIDIA GB10", so either file applies to any DGX Spark.
