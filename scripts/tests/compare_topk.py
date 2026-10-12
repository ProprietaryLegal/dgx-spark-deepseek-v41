# SPDX-License-Identifier: Apache-2.0
"""Compare vLLM (g0_truncated.py JSON) against the reference (ref_truncated_logits.py JSON) at the first generated
position: top-1 agreement, |logprob delta| over vLLM's top-k, and a KL estimate on the union support.
  python3 compare_topk.py vllm.json ref.json"""
import json
import math
import sys

v, r = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for ov, orf in zip(v["outputs"], r["outputs"]):
    a = {int(k): x for k, x in ov["topk"][0].items()}
    b = {int(k): x for k, x in orf["topk"].items()}
    top1 = max(a, key=a.get) == max(b, key=b.get)
    common = [t for t in a if t in b]
    d = [abs(a[t] - b[t]) for t in common]
    # KL(ref || vllm) over the reference's top-64 (mass mostly there); tokens vLLM did not list get its k-th logprob
    floor = min(a.values())
    kl = sum(math.exp(b[t]) * (b[t] - a.get(t, floor)) for t in b)
    print(f"{ov['prompt'][:28]!r:32s} top1 {'match' if top1 else 'DIFF '}  common {len(common):2d}/{len(a)}  "
          f"mean|d| {sum(d) / max(len(d), 1):.4f}  max|d| {max(d) if d else float('nan'):.4f}  KL~ {kl:.4f}")
