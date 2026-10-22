# SPDX-License-Identifier: Apache-2.0
"""The KLD panel alone, rendered exactly as build.py renders the published one: the checkpoint's own encoder
(make_encoder), only the rows with split == "panel", each cut to --panel-max tokens, score = 1 on assistant-generated
positions. No calibration rows and no exllamav3 text, so no std_dir. For a second panel of native replies to the same
prompts (generate.py --split panel, e.g. the 1017 checkpoint's), and for rendering the published conversations the
same way to compare. --split cal (or any) renders other rows the same way, from one or more files (a drafter
training set: the calibration conversations, which the KLD panel never saw).

  python3 build_panel.py /model conversations_1017_panel.jsonl /out/name [--panel-max 2048]
  python3 build_panel.py /model conversations.jsonl instruct_v3.jsonl /out/drafter_train --split cal --panel-max 3072
Writes /out/name.safetensors (tokens int32, offsets int64, score uint8 [T]) + /out/name_index.json (build.py's
panel_index fields: id, domain, length, scored, thinking, effort, tools), in the file's row order.
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file

from build import make_encoder, pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("conversations", nargs="+")
    ap.add_argument("out")
    ap.add_argument("--panel-max", type=int, default=2048)
    ap.add_argument("--split", default="panel", help="the rows' split to render (panel, cal), or any")
    a = ap.parse_args()
    _, encode = make_encoder(a.model)
    panel, panel_idx = [], []
    rows = [json.loads(line) for f in a.conversations for line in open(f)]
    for r in rows:
        if a.split != "any" and r["split"] != a.split:
            continue
        ids, score = encode(r)
        ids, score = ids[: a.panel_max], score[: a.panel_max]
        panel.append((ids, score))
        panel_idx.append({"id": r["id"], "domain": r["domain"], "length": len(ids), "scored": sum(score),
                          "thinking": r["thinking"], "effort": r["effort"], "tools": bool(r.get("tools"))})
    t, o = pack([p[0] for p in panel])
    sc = torch.tensor([x for p in panel for x in p[1]], dtype=torch.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file({"tokens": t, "offsets": o, "score": sc}, a.out + ".safetensors")
    json.dump(panel_idx, open(a.out + "_index.json", "w"), indent=0)
    by = {}
    for x in panel_idx:
        d = by.setdefault(x["domain"].split("_")[0], [0, 0, 0])
        d[0] += 1
        d[1] += x["length"]
        d[2] += x["scored"]
    print(f"panel: {len(panel)} conversations, {sum(x['length'] for x in panel_idx):,} tokens, "
          f"{sum(x['scored'] for x in panel_idx):,} scored positions")
    for k in sorted(by):
        print(f"  {k:13s} {by[k][0]:4d} conversations {by[k][1]:9,} tokens {by[k][2]:9,} scored")


if __name__ == "__main__":
    main()
