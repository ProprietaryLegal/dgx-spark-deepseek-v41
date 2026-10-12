# SPDX-License-Identifier: Apache-2.0
"""A KLD panel from an eval suite's prompts and native's replies (e.g. IFEval through OpenRouter, thinking off), in
p4_kld.py's layout: tokens int32 [T], offsets int64 [n+1], score uint8 [T] (1 = assistant-generated position), and an
index with a domain per sequence (the caller's category, e.g. IFEval's instruction group), so KLD is reported per
category. Rendered like build.py: the checkpoint's own encoder, chat mode (thinking off), no extra special tokens.

This is an evaluation set, never calibration: what it measures is how far a build's next-token distribution is from
native's on replies to these prompts.

  python3 build_eval_panel.py /model prompts.json replies.jsonl categories.json /out/name [--max-len 2048]
    prompts.json     {id: prompt text}
    replies.jsonl    {"id", "content"} per line (rows with an error or no content are skipped)
    categories.json  {id: domain label}
Writes /out/name.safetensors and /out/name_index.json.
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompts")
    ap.add_argument("replies")
    ap.add_argument("categories")
    ap.add_argument("out")
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.model, "encoding"))
    from encoding import encode_messages
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    sp = {n: tok.token_to_id(n) for n in ("<｜Assistant｜>", "<｜User｜>", "<｜System｜>", "<｜end▁of▁sentence｜>")}
    assert all(v is not None for v in sp.values()), sp
    prompts = json.load(open(a.prompts))
    cats = json.load(open(a.categories))
    seqs, index = [], []
    for line in open(a.replies):
        r = json.loads(line)
        if r.get("error") or not r.get("content"):
            continue
        msgs = [{"role": "user", "content": prompts[r["id"]]}, {"role": "assistant", "content": r["content"]}]
        ids = tok.encode(encode_messages(msgs, thinking_mode="chat"), add_special_tokens=False).ids[: a.max_len]
        score, inside = [], False
        for t in ids:                                 # build.py's rule: positions inside an assistant turn
            if t in (sp["<｜User｜>"], sp["<｜System｜>"]):
                inside = False
            score.append(1 if inside else 0)
            if t == sp["<｜Assistant｜>"]:
                inside = True
            elif t == sp["<｜end▁of▁sentence｜>"]:
                inside = False
        seqs.append((ids, score))
        index.append({"id": r["id"], "domain": cats[r["id"]], "length": len(ids), "scored": sum(score)})
    flat = [t for s, _ in seqs for t in s]
    offs = [0]
    for s, _ in seqs:
        offs.append(offs[-1] + len(s))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file({"tokens": torch.tensor(flat, dtype=torch.int32), "offsets": torch.tensor(offs, dtype=torch.int64),
               "score": torch.tensor([x for _, sc in seqs for x in sc], dtype=torch.uint8)}, a.out + ".safetensors")
    json.dump(index, open(a.out + "_index.json", "w"), indent=0)
    print(f"{len(seqs)} sequences, {len(flat):,} tokens, {sum(x['scored'] for x in index):,} scored positions")


if __name__ == "__main__":
    main()
