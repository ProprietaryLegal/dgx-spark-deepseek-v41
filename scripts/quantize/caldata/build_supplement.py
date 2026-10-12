# SPDX-License-Identifier: Apache-2.0
"""Token set for the v3 calibration supplement (generate.py --set instruct): every conversation rendered like build.py
(the checkpoint's own encoder), all in the calibration split.

  python3 build_supplement.py /model instruct_v3.jsonl /out/instruct [--max-len 2048]
                              [--concat cal.safetensors cal_index.json /out/cal_v3]
Writes /out/instruct.safetensors (tokens int32, offsets int64) + /out/instruct_index.json (source, id, domain, length),
and with --concat the calibration set followed by the supplement (pass 1's token set for the v3 re-encode).
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file

from build import make_encoder, pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("conversations")
    ap.add_argument("out")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--concat", nargs=3, metavar=("CAL", "CAL_INDEX", "OUT"))
    a = ap.parse_args()
    _, encode = make_encoder(a.model)
    seqs, index = [], []
    for line in open(a.conversations):
        r = json.loads(line)
        ids, _ = encode(r)
        ids = ids[: a.max_len]
        seqs.append(ids)
        index.append({"source": "instruct_v3", "id": r["id"], "domain": r["domain"], "length": len(ids)})
    t, o = pack(seqs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file({"tokens": t, "offsets": o}, a.out + ".safetensors")
    json.dump(index, open(a.out + "_index.json", "w"), indent=0)
    print(f"supplement: {len(seqs)} sequences, {int(o[-1]):,} tokens")
    if a.concat:
        cal, cal_index, out = a.concat
        d = load_file(cal)
        ct, co = d["tokens"].tolist(), d["offsets"].tolist()
        cseqs = [ct[co[i]: co[i + 1]] for i in range(len(co) - 1)]
        cidx = json.load(open(cal_index))
        assert len(cidx) == len(cseqs) and all(x["length"] == len(s) for x, s in zip(cidx, cseqs))
        t2, o2 = pack(cseqs + seqs)
        save_file({"tokens": t2, "offsets": o2}, out + ".safetensors")
        json.dump(cidx + index, open(out + "_index.json", "w"), indent=0)
        print(f"concat: {len(cseqs)} + {len(seqs)} sequences, {int(o2[-1]):,} tokens -> {out}.safetensors")


if __name__ == "__main__":
    main()
