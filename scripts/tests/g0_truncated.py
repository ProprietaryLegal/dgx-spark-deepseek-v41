# SPDX-License-Identifier: Apache-2.0
"""G0 on a layer-truncated official DeepSeek-V4.1-Flash: boot our image's vLLM on one GB10 with Engram on disk, greedy
decode a few prompts, save token ids + top-5 logprobs (for comparison against the reference harness and against later
builds), and print prefill / decode timings. The text is not meaningful (N of 40 layers); the plumbing is the test.

  DSV41_TRUNCATE_LAYERS=3 PYTHONPATH=/tests/plugins VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python3 g0_truncated.py /model /out/g0_L3.json [--eager] [--engram disk|pinned]
"""
import argparse
import json
import os
import time

N = int(os.environ["DSV41_TRUNCATE_LAYERS"])


def truncate(cfg):
    tc = getattr(cfg, "text_config", cfg)
    tc.num_hidden_layers = N
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--decode-graphs", action="store_true",
                    help="cudagraph_mode FULL_DECODE_ONLY (what the TP=2 serve uses; the EXL3 MoE plugin refuses larger "
                         "captures), instead of vLLM's default piecewise + full graphs")
    ap.add_argument("--engram", default="disk", choices=["disk", "pinned"])
    ap.add_argument("--gpu-util", type=float, default=0.35)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--logprobs", type=int, default=20)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    engram = {"cpu_offload": True, "dp_shared_memory": False}
    if a.engram == "disk":
        engram.update(disk_offload=True, disk_threads=32)
    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=1, max_model_len=8192, gpu_memory_utilization=a.gpu_util,
              hf_overrides=truncate, enforce_eager=a.eager, engram_config=engram,
              **({"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}} if a.decode_graphs else {}),
              limit_mm_per_prompt={"image": 0}, seed=0, max_logprobs=a.logprobs)
    boot = time.time() - t0
    prompts = [
        "The capital of France is",
        "def fibonacci(n):\n    ",
        "Engram is a conditional memory that",
        "1, 2, 3, 4, 5, 6, 7,",
    ]
    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens, logprobs=a.logprobs, ignore_eos=True)
    llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=4))   # warm-up
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    gen = time.time() - t1
    one = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    t2 = time.time()
    llm.generate(prompts[:1], one)
    single = time.time() - t2
    res = {"layers": N, "eager": a.eager, "engram": a.engram, "boot_s": round(boot, 1),
           "batch4_32tok_s": round(gen, 3), "single_128tok_s": round(single, 3),
           "single_tok_per_s": round(128 / single, 2), "outputs": []}
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        res["outputs"].append({"prompt": p, "token_ids": list(c.token_ids), "text": c.text,
                               "topk": [{str(k): round(v.logprob, 5) for k, v in lp.items()} for lp in (c.logprobs or [])]})
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "outputs"}))
    for r in res["outputs"]:
        print(repr(r["prompt"]), "->", repr(r["text"][:80]), r["token_ids"][:8])


if __name__ == "__main__":
    main()
