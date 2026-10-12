# SPDX-License-Identifier: Apache-2.0
"""Token sets for P4, from generate.py's conversations + exllamav3's standard calibration text (MIT, turboderp-org):

  cal.safetensors    tokens int32 [T], offsets int64 [n+1]            calibration sequences (variable length)
  cal_index.json     per sequence: source, id, domain, length
  panel.safetensors  tokens, offsets, score uint8 [T] (1 = assistant-generated position)   the published KLD panel
  panel_index.json

Conversations are rendered with the checkpoint's own encoder (encoding/encoding.py: thinking mode + numeric
reasoning effort, DSML tool calls, tool results merged into user turns) and tokenized without extra special tokens
(the encoder writes BOS itself).

  python3 build.py /model conversations.jsonl /path/to/standard_cal_data /out [--cal-max 1536 --panel-max 2048]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys

import torch
from safetensors.torch import save_file

SEED = 20261008
STD_FILES = [("c4.utf8", 20), ("code.utf8", 20), ("multilingual.utf8", 10), ("technical.utf8", 10), ("wiki.utf8", 50),
             ("tiny.utf8", 5)]


def make_encoder(model):
    """(tokenizer, encode(conversation) -> (token ids, score mask)) with the checkpoint's own encoder."""
    sys.path.insert(0, os.path.join(model, "encoding"))
    from encoding import encode_messages
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
    sp = {name: tok.token_to_id(name) for name in ("<｜Assistant｜>", "<｜User｜>", "<｜System｜>", "<｜end▁of▁sentence｜>",
                                                    "<｜begin▁of▁sentence｜>")}
    assert all(v is not None for v in sp.values()), sp

    def encode(r):
        msgs = copy.deepcopy(r["messages"])
        if r.get("tools"):
            if msgs[0]["role"] != "system":
                msgs.insert(0, {"role": "system", "content": ""})
            msgs[0]["tools"] = r["tools"]            # the encoder renders tools only on a system message
        mode = "thinking" if r["thinking"] else "chat"
        effort = {"low": 50, "high": 75}.get(r["effort"]) if r["thinking"] else None
        text = encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort)
        ids = tok.encode(text, add_special_tokens=False).ids
        score, inside = [], False
        for t in ids:
            if t in (sp["<｜User｜>"], sp["<｜System｜>"]):
                inside = False
            score.append(1 if inside else 0)
            if t == sp["<｜Assistant｜>"]:
                inside = True                        # the header itself is a prompt token
            elif t == sp["<｜end▁of▁sentence｜>"]:
                inside = False
        return ids, score

    return tok, encode


def pack(seqs):
    flat = [t for s in seqs for t in s]
    offs = [0]
    for s in seqs:
        offs.append(offs[-1] + len(s))
    return torch.tensor(flat, dtype=torch.int32), torch.tensor(offs, dtype=torch.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("conversations")
    ap.add_argument("std_dir")
    ap.add_argument("out")
    ap.add_argument("--cal-max", type=int, default=1536)
    ap.add_argument("--panel-max", type=int, default=2048)
    ap.add_argument("--std-rows", type=int, default=64)
    ap.add_argument("--std-len", type=int, default=2048)
    ap.add_argument("--random-rows", type=int, default=8)
    a = ap.parse_args()
    tok, encode = make_encoder(a.model)
    rng = random.Random(SEED)

    cal, cal_idx, panel, panel_idx = [], [], [], []
    n_conv = {"cal": 0, "panel": 0}
    with open(a.conversations) as f:
        for line in f:
            r = json.loads(line)
            ids, score = encode(r)
            if r["split"] == "panel":
                ids, score = ids[: a.panel_max], score[: a.panel_max]
                panel.append((ids, score))
                panel_idx.append({"id": r["id"], "domain": r["domain"], "length": len(ids), "scored": sum(score),
                                  "thinking": r["thinking"], "effort": r["effort"], "tools": bool(r.get("tools"))})
            else:
                ids = ids[: a.cal_max]
                cal.append(ids)
                cal_idx.append({"source": "chat", "id": r["id"], "domain": r["domain"], "length": len(ids)})
            n_conv[r["split"]] += 1

    # exllamav3's standard text, sampled as fixed-width windows in its source weights
    wsum = sum(w for _, w in STD_FILES)
    for fname, w in STD_FILES:
        n_rows = max(1, round(a.std_rows * w / wsum))
        with open(os.path.join(a.std_dir, fname), encoding="utf8") as f:
            ids = tok.encode(f.read(), add_special_tokens=False).ids
        for _ in range(n_rows):
            s = rng.randrange(0, max(1, len(ids) - a.std_len))
            cal.append(ids[s: s + a.std_len])
            cal_idx.append({"source": "exllamav3_std", "id": f"{fname}@{s}", "domain": fname.split(".")[0],
                            "length": len(cal[-1])})
    vocab = tok.get_vocab_size()
    g = torch.Generator().manual_seed(SEED)
    for i in range(a.random_rows):
        cal.append(torch.randint(0, vocab, (a.std_len,), generator=g).tolist())
        cal_idx.append({"source": "random", "id": f"random{i}", "domain": "random", "length": a.std_len})

    os.makedirs(a.out, exist_ok=True)
    t, o = pack(cal)
    save_file({"tokens": t, "offsets": o}, os.path.join(a.out, "cal.safetensors"))
    json.dump(cal_idx, open(os.path.join(a.out, "cal_index.json"), "w"), indent=0)
    t, o = pack([p[0] for p in panel])
    sc = torch.tensor([x for p in panel for x in p[1]], dtype=torch.uint8)
    save_file({"tokens": t, "offsets": o, "score": sc}, os.path.join(a.out, "panel.safetensors"))
    json.dump(panel_idx, open(os.path.join(a.out, "panel_index.json"), "w"), indent=0)
    by = {}
    for x in cal_idx:
        by[x["source"]] = by.get(x["source"], 0) + x["length"]
    print(f"cal: {len(cal)} sequences ({n_conv['cal']} conversations), {sum(by.values()):,} tokens {by}")
    print(f"panel: {len(panel)} conversations, {sum(x['length'] for x in panel_idx):,} tokens, "
          f"{sum(x['scored'] for x in panel_idx):,} scored positions")


if __name__ == "__main__":
    main()
