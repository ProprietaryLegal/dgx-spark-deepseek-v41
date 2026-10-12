# SPDX-License-Identifier: Apache-2.0
"""Check pass 1's output holds all 384 routed experts for layers 0..LAST (safetensors headers only, no tensor reads).
Exit 0 when complete, else 1 with the incomplete layers listed. Host python, no dependencies.

  python3 p1_complete.py /path/to/runs/p1 LAST
  python3 p1_complete.py /path/to/runs/p1 --prefix     prints the highest L with layers 0..L complete (-1 if none)
"""
import glob
import json
import os
import struct
import sys


def keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).keys()


def n_experts(d, l):
    files = sorted(glob.glob(os.path.join(d, f"layer_{l:02d}.part*.safetensors")))
    return len({k.split(".")[4] for f in files for k in keys(f) if ".ffn.experts." in k}), len(files)


def main():
    d = sys.argv[1]
    if sys.argv[2] == "--prefix":
        l = 0
        while l < 40 and n_experts(d, l)[0] == 384:
            l += 1
        print(l - 1)
        return
    last = int(sys.argv[2])
    bad = []
    for l in range(last + 1):
        n, nf = n_experts(d, l)
        if n != 384:
            bad.append(f"layer {l}: {n} experts in {nf} file(s)")
    print("\n".join(bad) if bad else f"layers 0-{last}: all 384 experts present")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
