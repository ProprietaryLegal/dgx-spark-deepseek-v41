# SPDX-License-Identifier: Apache-2.0
"""Reference (DeepSeek's inference/ code, layer-streamed) logits of the first N layers for the G0 prompts: the ground
truth a vLLM boot with text_config.num_hidden_layers = N must reproduce. Prints and saves the top-5 logprobs at the
last prompt position (= the first generated token's distribution in g0_truncated.py).

  python3 ref_truncated_logits.py /model N /out/ref_LN.json
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41q.stream import run_truncated  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Engram is a conditional memory that",
    "1, 2, 3, 4, 5, 6, 7,",
]


def main():
    ckpt, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    tensors_dir = sys.argv[4] if len(sys.argv) > 4 else ""      # optional: EXL3 experts (pass-1 part files)
    dense_dir = sys.argv[5] if len(sys.argv) > 5 else ""        # optional: EXL3 dense linears (p4_dense.py output)
    import tilelang
    print("tilelang", tilelang.__version__, "torch", torch.__version__, flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    seqs = [tok.encode(p) for p in PROMPTS]
    torch.set_default_dtype(torch.bfloat16)
    t0 = time.time()
    logits = run_truncated(ckpt, seqs, n, tensors_dir=tensors_dir, dense_dir=dense_dir)
    res = {"layers": n, "seconds": round(time.time() - t0, 1), "outputs": []}
    for p, ids, lg in zip(PROMPTS, seqs, logits):
        lp = torch.log_softmax(lg[-1].float(), -1)
        top = torch.topk(lp, 64)
        d = {str(i): round(v, 5) for i, v in zip(top.indices.tolist(), top.values.tolist())}
        res["outputs"].append({"prompt": p, "ids": ids, "topk": d, "nan": bool(torch.isnan(lg).any())})
        print(repr(p), ids[:6], "->", dict(list(d.items())[:5]), "nan" if torch.isnan(lg).any() else "")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
