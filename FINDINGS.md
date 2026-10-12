# GB10 findings

What two DGX Sparks taught us while quantizing and serving DeepSeek-V4.1-Flash, October 2026. Each finding carries the
measurement behind it and what the recipe does about it. Speeds are `kit_bench.py` set b unless a line says otherwise.

## Memory

1. **CUDA allocations do not reclaim the page cache.** GB10's memory is unified: file pages and CUDA allocations draw
   from the same 128 GB. A 67 MB allocation failed with `CUDA error: out of memory` while `free -g` showed 45 GB
   available but only 8 GB free (48 GB of buff/cache from checkpoint I/O); after `drop_caches` the same run passed.
   The launcher drops the page cache before boot (`sudo -n` for that one command), and long encodes run
   `scripts/quantize/guards.sh start`: a page-cache drop every 5 s plus a memory watch.
2. **Drop caches while encoding, never while serving.** Engram's hot rows live in the page cache. The encode guard left
   running during a serve gave structured output an intermittent ~5% slow mode (a random rep at 132 against
   141 tok/s). `guards.sh serve` keeps the memory watch and stops the drops: 27 of 27 reps clean since.
3. **Without the drop, an encode can thrash.** After a reboot that lost the guard, a pass-1 layer sat at 0% GPU with
   a busy `UVM GPU` kernel thread: unified memory paging instead of computing. Start the guards before any pass.
4. **FlashInfer autotune grows driver memory with every new kernel shape.** With autotune on, a dummy-weight boot went
   from 4-5 GiB to 0.8 GiB MemAvailable in about 5 minutes of mixed requests. The launcher passes
   `{"enable_flashinfer_autotune": false}`.
5. **earlyoom does not fire while swap is free.** A Spark went to 240 MiB available (0.19%) with 94% of swap free,
   hung under memory pressure and was rebooted by the hardware watchdog. Watch MemAvailable yourself (the guards do).
6. **No large file I/O beside a long GPU job.** A Spark stopped answering (no ssh, no ping on either link, not caught by
   the watchdog) while it streamed a 25 GB image to the other box and an encode held ~80 GB of CUDA memory; the same
   copy had run fine with the GPU idle. Probable cause: the read filled the page cache faster than the 5 s guard while
   the allocator needed memory (finding 1). Copy images and weights with the GPUs idle.
7. **The vision tower costs ~0.9 GiB a rank.** The launcher serves text only by default (`TEXT_ONLY=1`), which lets the
   262K KV cache fit with 9 GiB MemAvailable left on each node.

## Bandwidth and kernels

8. **vLLM's default MXFP8 GEMM is slow at decode on GB10.** FlashInfer's CUTLASS MXFP8 path streams weights at
   66-78 GB/s at decode sizes; Marlin reads the same weights at 200-251 GB/s. At a 2,048-token prefill chunk CUTLASS
   wins (wq_b 953 us against 1,563 us). The launcher disables `FlashInferCutlassMxfp8LinearKernel`. Benchmark with
   distinct weights cycled from DRAM: a weight that stays in the 24 MiB L2 reports rates GB10's ~273 GB/s cannot give.
9. **Decode is weight bytes.** Our decode-size EXL3 linear (`exl3_mrow`: TMA k-rows into an mbarrier ring) reaches
   186-229 GB/s. One DSpark verify round costs 28.9 ms with 1 row and 45.7 ms with 6 (`VLLM_DSPARK_ROUND_MS`): each
   live row adds ~3.4 ms, nearly all of it the routed experts it touches.
10. **The verify window is bound by bytes, not attention.** Our Triton sparse attention for verify rows is 2.8x faster
    than FlashInfer's path on its own and flat end to end (patch 0022 in `patches/experimental/`).
11. **NCCL pays a fixed cost in CUDA graphs that no setting removes.** 11 + 7 NCCL configurations (protocols,
    algorithms, channels, GPUDirect RDMA, registration, launch mode) left it; PIECEWISE graphs, which run attention and
    NCCL eagerly between graph segments, cost 19% end to end. The RDMA all-reduce ring (TensorFold's design, credited) does an all-reduce of
    [6, 5120] in 14.6 us inside a graph against NCCL's 35.9, bit-exact with the rank-order fp32 sum.
12. **MoE batching buys little at 2 rows.** Two streams at once give 81 tok/s together against ~74 back to back (+10%):
    two rows touch nearly twice the expert weights.

## Quantization

13. **Energy share misprices experts.** Every V4.1 layer has a "super expert" that writes very large outputs on a few
    tokens: in layer 1 one expert holds 92.3% of the routed output energy and 1.4% of the per-token share. Our first
    plan priced experts by energy and starved the rest. The allocator prices an expert by its per-token share times its
    layer's measured sensitivity; super experts stay at K5.
14. **Sensitivity falls ~250x through the stack.** KL at the output per unit of relative noise in one layer's routed
    output: 4.71 at layer 0, 0.019 at layer 39; layers 0-19 carry about 88% of it.
15. **Measured EXL3 error per K** (layer 0, median relative output error): K2 16.2%, K3 6.5%, K4 2.4%, K5 1.0%.
16. **The calibration mix shows up in the evals.** A calibration set that was 42% code (chat 4.5%, writing 1.2%)
    lost IFEval prompt-strict against native (-2.6 points, paired p = 0.034) and trailed Mia-AiLab's pack where a domain
    was thin (multilingual KLD 1.34x hers, code 0.58x). A held-out supplement of 392 native conversations (instruction
    following, chat, writing, 30 languages; no 8-gram shared with IFEval) closed it: within noise of native on IFEval,
    closer than her pack on 7 of 8 domains. Re-weighting domains inside the Hessians added nothing over including the
    data (a layer-10 probe, `scripts/quantize/hessian_domain_probe.py`).
17. **The DSpark drafter tolerates EXL3.** Its dense linears at K5 and its routed experts at K3 kept acceptance and made
    each round 1.1-1.3 ms faster than its MXFP4 experts on deep_gemm. Dense at K3 or K4 cost prose acceptance.

## Serving

18. **A null-block NaN on V4.1** (upstream #57156): dummy, warm-up and capture batches carry zeroed block tables, so the
    compressor ring writes fp32 rows into block 0, the null block every KV group shares; every logit came out NaN on a
    3-layer test model. Fixed upstream by PR #58560 after our pin; carried as patch 0003.
19. **Engram disk reads need threads.** With 32 reader threads a 512-token prefill step stalled ~74 ms on Engram; cold
    rows: 4,096 in 80.2 ms at 32 threads, 5.8 ms at 256 (16,384: 318 ms to 20.4 ms). The launcher uses 256.
20. **`--restart no` for a TP pair.** Docker restarting one rank of a tensor-parallel pair leaves a head that can never
    rejoin; restart both or neither.
21. **Long-context decode numbers depend on the prompt.** After a 32K random-token prompt decode reads 128.2 tok/s
    (`kit_bench.py depth`); after a real 32K document, 81.7. Random tokens draft easily. Quote the real-document
    number for real use.

Measured and dropped (each A/B in `BENCHMARKS.md`): PIECEWISE graphs, eager all-reduces in FULL graphs (patch 0009),
FlashInfer's decode kernel for verify rows, top-64 Markov drafting, int8 activations, Engram wkv at K4, a speed-aware
expert budget, the drafter's dense at K3 / K4, pinning vLLM to the fast CPU cores.
