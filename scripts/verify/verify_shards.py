#!/usr/bin/env python3
"""sha256 every file of a local HF snapshot against the repo's LFS metadata at a pinned revision.
  python3 verify_shards.py deepseek-ai/DeepSeek-V4.1-Flash 2cba9e42aa026125f3ed06c6d98c1db82f7ca027 ~/models/DeepSeek-V4.1-Flash-official-2cba9e42 [--jobs 4]
"""
import argparse, hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor


def sha256(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("rev"); ap.add_argument("dir")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--manifest", help="write {file: {size, sha256, result}} here (the sha256 of every shard on record)")
    a = ap.parse_args()
    url = f"https://huggingface.co/api/models/{a.repo}/revision/{a.rev}?blobs=true"
    meta = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "dsv41-verify/1"}), timeout=120))
    files = [(s["rfilename"], (s.get("lfs") or {}).get("sha256"), s.get("size")) for s in meta["siblings"]]
    todo = [(n, h, sz) for n, h, sz in files if h]  # LFS files carry a sha256; small git files are checked by size only
    bad, missing = [], []
    for n, h, sz in files:
        p = os.path.join(a.dir, n)
        if not os.path.exists(p):
            missing.append(n)
        elif sz is not None and os.path.getsize(p) != sz:
            bad.append((n, f"size {os.path.getsize(p)} != {sz}"))

    def check(item):
        n, h, _ = item
        p = os.path.join(a.dir, n)
        if not os.path.exists(p):
            return n, "missing"
        got = sha256(p)
        return n, "ok" if got == h else f"sha256 {got[:12]} != {h[:12]}"

    results = {}
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, (n, r) in enumerate(ex.map(check, todo), 1):
            results[n] = r
            if r != "ok" and r != "missing":
                bad.append((n, r))
            print(f"[{i}/{len(todo)}] {n}: {r}", flush=True)
    if a.manifest:
        man = {"repo": a.repo, "revision": a.rev, "files": {
            n: {"size": sz, "sha256": h, "result": results.get(n, "missing" if n in missing else "size ok")}
            for n, h, sz in files}}
        with open(a.manifest, "w") as f:
            json.dump(man, f, indent=1)
    print(json.dumps({"files": len(files), "lfs_checked": len(todo), "missing": missing, "bad": bad}, indent=1))
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
