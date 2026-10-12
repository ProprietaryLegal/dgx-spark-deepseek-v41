#!/usr/bin/env python3
"""Same greedy request N times against a live serve: are the outputs token-identical, and if not, where do they first
part and how close were the top two candidates there? A near-tie (top-1 vs top-2 logprob gap of a few hundredths) at
the first split is ordinary non-batch-invariant arithmetic (a DSpark verify step's row count changes which kernel
shapes run); a wide gap means something reads different data between runs. Run on the head (stdlib only):

    python3 greedy_repeat_check.py --base http://localhost:8000 --model dsv41 --n 5
    python3 greedy_repeat_check.py --kit /tmp/kit_bench.py --prompt code      (kit_bench set b's prompt by name, read at
                                                                                runtime from that file, not copied here)
"""
import argparse
import importlib.util
import json
import urllib.request


def chat(base, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "logprobs": True, "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    steps = d["choices"][0]["logprobs"]["content"]
    return [(s["token"], [(t["token"], t["logprob"]) for t in s["top_logprobs"]]) for s in steps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--kit", help="kit_bench.py path: take the prompt from its set b by --prompt name")
    ap.add_argument("--prompt", default="code")
    ap.add_argument("--text", help="prompt text (instead of --kit)")
    a = ap.parse_args()
    if a.text:
        prompt = a.text
    else:
        spec = importlib.util.spec_from_file_location("kit_bench", a.kit)
        kb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kb)
        prompt = dict(kb.DECODE_PROMPTS_B)[a.prompt]
    runs = [chat(a.base, a.model, prompt, a.max_tokens) for _ in range(a.n)]
    toks = [[t for t, _ in r] for r in runs]
    ref = toks[0]
    report = {"n": a.n, "lengths": [len(t) for t in toks], "identical": all(t == ref for t in toks), "splits": []}
    for i, t in enumerate(toks[1:], 1):
        j = next((k for k in range(min(len(ref), len(t))) if ref[k] != t[k]), None)
        if j is None:
            continue
        def gap(run):
            top = run[j][1]
            return round(top[0][1] - top[1][1], 4) if len(top) > 1 else None
        report["splits"].append({"run": i, "first_diff_at": j, "ref_token": ref[j], "run_token": t[j],
                                 "ref_top2_gap": gap(runs[0]), "run_top2_gap": gap(runs[i])})
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
