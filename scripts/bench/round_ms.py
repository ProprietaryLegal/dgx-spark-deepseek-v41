# SPDX-License-Identifier: Apache-2.0
"""Milliseconds per verify round and tokens per round, per prompt, from kit_bench set b result folders (each streamed
chunk is one verify round's accepted tokens). tok/s = tokens per round / ms per round: a numerics change that alters
the greedy text moves the first (acceptance on a different text), a kernel change the second. Rounding-class levers
are judged on ms per round.
    python3 round_ms.py <speed dir>/<run> [<run> ...]
"""
import json
import os
import sys


def rows(d):
    r = json.load(open(os.path.join(d, "kit_decode_b.json")))["result"]
    out = {}
    for p in r:
        n, c, t = p["tokens_all"][0], p["chunks_all"][0], p["decode_tps_median"]
        out[p["prompt"]] = (t, n / c, 1000.0 * n / t / c)
    return out


def main():
    base = None
    for d in sys.argv[1:]:
        r = rows(d)
        name = os.path.basename(d.rstrip("/"))
        cells = []
        for p in ("code", "prose", "structured"):
            t, tpr, ms = r[p]
            rel = f" ({base[p][2] / ms:.3f}x)" if base else ""
            cells.append(f"{p} {t:6.1f} tok/s = {tpr:4.2f} tok/round x {ms:5.1f} ms{rel}")
        print(f"{name:18s} " + " | ".join(cells))
        base = base or r


if __name__ == "__main__":
    main()
