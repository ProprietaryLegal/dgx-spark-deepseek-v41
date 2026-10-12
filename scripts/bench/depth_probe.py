#!/usr/bin/env python3
"""Per-prompt DSpark round cost on a running vLLM serve: kit_bench's set b requests (greedy, thinking off, 384 tokens),
one at a time, with vLLM's spec-decode counters read around each, so every reply gives its verify steps, tokens per
step and ms per step (decode time / steps). Run on the head node with kit_bench.py beside it (standard library only):

  python3 depth_probe.py --kit /tmp/kit_bench.py --base http://localhost:8000 --model dsv41 --reps 2 --tag fixed3 \
      --out /tmp/probe_fixed3.json

Use it to measure the verify-cost table of patch 0010 (VLLM_DSPARK_ROUND_MS: one fixed depth per run) and to check that
greedy texts do not depend on the depth (the full texts are kept).
"""
import argparse
import importlib.util
import json
import re
import urllib.request

COUNTERS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total")


def metrics(base: str) -> dict:
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for name in COUNTERS:
        vals = [float(m.group(1)) for m in re.finditer(r"^" + re.escape(name) + r"(?:\{[^}]*\})? ([0-9.e+]+)$",
                                                          text, re.M)]
        out[name] = sum(vals)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kit", required=True)
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--set", default="b")
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("kit_bench", a.kit)
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    c = kit.Client(a.base, a.model, None)
    rows = []
    for name, prompt in kit.PROMPT_SETS[a.set]:
        c.stream(kit.decode_body(prompt, 32), "/v1/chat/completions")          # warm-up, as kit_bench does
        for rep in range(a.reps):
            m0 = metrics(a.base)
            r = c.stream(kit.decode_body(prompt, a.tokens), "/v1/chat/completions")
            m1 = metrics(a.base)
            steps = m1[COUNTERS[0]] - m0[COUNTERS[0]]
            drafted = m1[COUNTERS[1]] - m0[COUNTERS[1]]
            accepted = m1[COUNTERS[2]] - m0[COUNTERS[2]]
            row = {"tag": a.tag, "prompt": name, "rep": rep, "tokens": r["tokens"], "decode_s": r["decode_s"],
                   "decode_tps": r["decode_tps"], "steps": steps, "drafted": drafted, "accepted": accepted,
                   "tokens_per_step": (r["tokens"] - 1) / steps if steps else None,
                   "ms_per_step": 1000 * r["decode_s"] / steps if steps else None, "text": r["text"]}
            rows.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "text"}),
                  flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
