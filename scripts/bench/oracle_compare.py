# SPDX-License-Identifier: Apache-2.0
"""Serving-path fidelity between two configurations of the same weights, from kit_bench.py oracle files (greedy 64-token
continuations + teacher-forced top-20 logprobs at every position of the 24-prompt oracle set): e.g. EXL3_INT8_GEMV=0
(fp16 activations, the reference) vs 2 (plain int8, the CUDA default), or parallel vs chained K groups (should be exact).

Per position where both runs saw the same tokens (prompt + the shared prefix of the two continuations): top-1 agreement
and a truncated KL(ref || cand) over ref's top-20 (cand's missing entries floored at its own 20th logprob, so the KL is
a lower-bound-style estimate, comparable between candidates scored against the same reference).
  python3 oracle_compare.py ref.jsonl cand.jsonl [cand2.jsonl ...]
"""
import json
import math
import statistics
import sys


def load(p):
    return {r["index"]: r for r in map(json.loads, open(p))}


def compare(ref, cand):
    kls, top1, same_gen, div_at = [], [], 0, []
    for i, a in ref.items():
        b = cand.get(i)
        if b is None or a["prompt_ids"] != b["prompt_ids"]:
            continue
        ga, gb = a["gen_ids"], b["gen_ids"]
        n = next((j for j, (x, y) in enumerate(zip(ga, gb)) if x != y), min(len(ga), len(gb)))
        if ga == gb:
            same_gen += 1
        else:
            div_at.append(n)
        shared = len(a["prompt_ids"]) + n                  # positions whose context is identical in both runs
        for pa, pb in zip(a["tf_top"][:shared], b["tf_top"][:shared]):
            if not pa or not pb:
                continue
            floor = min(pb.values())
            ka = max(pa, key=pa.get)
            top1.append(ka == max(pb, key=pb.get))
            kls.append(sum(math.exp(la) * (la - pb.get(t, floor)) for t, la in pa.items()))
    return {"prompts": len(ref), "identical_greedy": same_gen, "first_divergence_tokens": sorted(div_at),
            "positions": len(kls), "top1_agree": sum(top1) / max(len(top1), 1),
            "kl_mean": statistics.fmean(kls) if kls else float("nan"),
            "kl_p99": sorted(kls)[int(0.99 * (len(kls) - 1))] if kls else float("nan")}


def main():
    ref = load(sys.argv[1])
    for p in sys.argv[2:]:
        r = compare(ref, load(p))
        print(f"{p}: {r['identical_greedy']}/{r['prompts']} identical greedy continuations "
              f"(diverge at {r['first_divergence_tokens']}), {r['positions']} shared positions, "
              f"top-1 agree {100 * r['top1_agree']:.2f}%, truncated KL mean {r['kl_mean']:.5f} p99 {r['kl_p99']:.4f}")


if __name__ == "__main__":
    main()
