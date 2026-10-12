#!/usr/bin/env python3
"""Exact per-tensor-group size of a Hugging Face checkpoint from its safetensors headers only (range requests, no
shard download). Groups tensors by name (routed experts, shared expert, attention, Engram, embeddings / head, draft,
vision, norms, other) and prints params and bytes per group and dtype, as JSON + a markdown table.

  python3 hf_safetensors_budget.py deepseek-ai/DeepSeek-V4.1-Flash 2cba9e42aa026125f3ed06c6d98c1db82f7ca027 \
      --json budget_headers.json --md budget_headers.md
"""
import argparse, json, re, struct, sys, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "E8M0": 1,
               "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "U16": 2, "U32": 4, "BOOL": 1, "F4": 0.5, "F8_E4M3FN": 1}


def get(url, rng=None, tries=6):
    import time
    h = {"User-Agent": "dsv41-budget/1"}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120).read()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"retry {i+1} for {url.rsplit('/', 1)[-1]} ({type(e).__name__})", file=sys.stderr)
            time.sleep(2 * (i + 1))


def header(repo, rev, fname):
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"
    n = struct.unpack("<Q", get(url, (0, 7)))[0]
    return fname, json.loads(get(url, (8, 8 + n - 1)))


def group(name):
    n = name.lower()
    if "engram" in n:
        return "engram"
    if re.search(r"(visual|vision|vit|mm_projector|merger)", n):
        return "vision"
    if re.search(r"(^|\.)(mtp|dspark|draft|nextn)", n):
        return "draft (DSpark/MTP)"
    if ".experts." in n and "shared" not in n:
        return "routed experts"
    if "shared_expert" in n:
        return "shared expert"
    if re.search(r"(gate\.weight|\.gate\.|router|e_score_correction)", n) and ".experts." not in n:
        return "router / gate"
    if re.search(r"(attn|attention|q_proj|kv_proj|o_proj|wq|wkv|wo|indexer)", n):
        return "attention"
    if re.search(r"(embed_tokens|lm_head|\.head\.)", n):
        return "embeddings / head"
    if re.search(r"(norm|layernorm)", n):
        return "norms"
    if re.search(r"(mlp|ffn)", n):
        return "dense mlp"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("rev")
    ap.add_argument("--json"); ap.add_argument("--md")
    a = ap.parse_args()
    idx = json.loads(get(f"https://huggingface.co/{a.repo}/resolve/{a.rev}/model.safetensors.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
    with ThreadPoolExecutor(3) as ex:
        heads = list(ex.map(lambda f: header(a.repo, a.rev, f), shards))
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # group -> dtype -> [tensors, elements, bytes]
    examples = defaultdict(list)
    for fname, h in heads:
        for name, meta in h.items():
            if name == "__metadata__":
                continue
            els = 1
            for d in meta["shape"]:
                els *= d
            nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
            g = group(name)
            t = agg[g][meta["dtype"]]
            t[0] += 1; t[1] += els; t[2] += nbytes
            if len(examples[g]) < 3:
                examples[g].append(f"{name} {meta['dtype']} {meta['shape']}")
    total = sum(v[2] for g in agg.values() for v in g.values())
    out = {"repo": a.repo, "rev": a.rev, "shards": len(shards), "total_bytes": total,
           "groups": {g: {dt: {"tensors": v[0], "elements": v[1], "bytes": v[2]} for dt, v in d.items()} for g, d in agg.items()},
           "examples": examples}
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    lines = [f"# Checkpoint budget from safetensors headers: `{a.repo}` @ `{a.rev[:10]}`", "",
             f"{len(shards)} shards, **{total/2**30:.1f} GiB** stored. Elements are stored elements (packed 4-bit = 2 per byte shows as U8 bytes).", "",
             "| group | dtype | tensors | stored elements | GiB |", "|---|---|---|---|---|"]
    for g in sorted(agg, key=lambda g: -sum(v[2] for v in agg[g].values())):
        for dt, v in sorted(agg[g].items(), key=lambda kv: -kv[1][2]):
            lines.append(f"| {g} | {dt} | {v[0]:,} | {v[1]:,} | {v[2]/2**30:.2f} |")
    lines += ["", "Examples per group:"] + [f"- **{g}**: " + "; ".join(e) for g, e in examples.items()]
    md = "\n".join(lines) + "\n"
    if a.md:
        open(a.md, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
