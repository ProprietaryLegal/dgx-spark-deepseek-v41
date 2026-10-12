# SPDX-License-Identifier: Apache-2.0
"""Assemble our DeepSeek-V4.1-Flash checkpoint from the official snapshot + pass-1 EXL3 experts:

  model-NNNNN-of-MMMMM.safetensors  every official tensor except the backbone routed experts (layers < 40), copied
                                    shard by shard (the DSpark drafter keeps its MXFP4 experts: source precision)
  model-exl3-LXX.safetensors        layer XX's EXL3 experts (all pass-1 parts merged), plugin key names
  model-0004{7,8}-of-00048.safetensors  the official Engram shards, hardlinked (they hold only Engram tensors)
  model.safetensors.index.json, config.json (+ quantization_config for the vllm_exl3_moe plugin), exl3_moe_plan.json,
  tokenizer / chat template / encoding files, MANIFEST.sha256

  python3 p4_assemble.py /official /pass1_dir plan.json /out [--no-hash]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil

from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROUTED = re.compile(r"^layers\.(\d+)\.ffn\.experts\.\d+\.")
COPY_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "LICENSE"]
COPY_DIRS = ["encoding"]


def sha256(path, buf=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("official")
    ap.add_argument("pass1")
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--dense", default="",
                    help="EXL3 dense linears from p4_dense.py (dense_LXX.safetensors + index): they replace the FP8 "
                         "wq_b / wo_a / wo_b / shared-expert tensors, and the config gets exl3_moe.dense")
    ap.add_argument("--test-layers", type=int, default=0,
                    help="truncated test checkpoint: only layers < N (+ embed / head / norm / vision), no mtp.*; "
                         "serve with text_config.num_hidden_layers = N")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plan = json.load(open(a.plan))
    n_layers = a.test_layers or plan.get("num_layers", 40)
    LAYER = re.compile(r"^layers\.(\d+)\.")

    def wanted(k):
        if not a.test_layers:
            return True
        if k.startswith("mtp."):
            return False
        m = LAYER.match(k)
        return m is None or int(m.group(1)) < a.test_layers
    dense_t, dense_mods = {}, set()                    # {key: (file, None)}, {"layers.L.attn.wq_b", ...}
    if a.dense:
        dwm = json.load(open(os.path.join(a.dense, "model.safetensors.index.json")))["weight_map"]
        for k, f in dwm.items():
            dense_t[k] = f
            if k.startswith("head."):                  # EXL3 head (p4_dense.py --head): replaces head.weight
                dense_mods.add("head")
                continue
            m = re.match(r"^(layers\.\d+\.(?:attn\.wq_a|attn\.wkv|attn\.wq_b|attn\.wo_a|attn\.wo_b|ffn\.shared_experts\.w[123]))\.", k)
            if not m:
                raise SystemExit(f"--dense: unexpected tensor {k}")
            dense_mods.add(m.group(1))

    def replaced_by_dense(k):                          # the FP8 weight / scale of a module that has EXL3 tensors
        mod, _, leaf = k.rpartition(".")
        return leaf in ("weight", "scale") and mod in dense_mods
    idx = json.load(open(os.path.join(a.official, "model.safetensors.index.json")))["weight_map"]
    by_file = {}
    for k, f in idx.items():
        by_file.setdefault(f, []).append(k)
    engram_files = sorted({f for k, f in idx.items() if ".engram.embed." in k})
    for f in engram_files:                             # Engram shards must hold nothing we replace
        assert not any(ROUTED.match(k) for k in by_file[f]), f
    weight_map = {}
    dense_files = [f for f in sorted(by_file) if f not in engram_files]
    total = len(dense_files)
    for i, f in enumerate(dense_files, 1):
        keep = [k for k in by_file[f] if wanted(k) and not replaced_by_dense(k)
                and not (ROUTED.match(k) and int(ROUTED.match(k).group(1)) < n_layers)]
        if not keep:
            continue
        name = f"model-{i:05d}-of-{total:05d}.safetensors"
        dst = os.path.join(a.out, name)
        if not os.path.exists(dst):
            with safe_open(os.path.join(a.official, f), framework="pt") as sf:
                tensors = {k: sf.get_tensor(k) for k in keep}
                meta = sf.metadata() or {}
            save_file(tensors, dst + ".tmp", metadata={"format": "pt", **meta})
            os.replace(dst + ".tmp", dst)
        for k in keep:
            weight_map[k] = name
        print(f"[{i}/{total}] {f}: kept {len(keep)} of {len(by_file[f])} tensors -> {name}", flush=True)
    for f in engram_files:
        if a.test_layers and not any(wanted(k) for k in by_file[f]):
            continue
        dst = os.path.join(a.out, f)
        if not os.path.exists(dst):
            try:
                os.link(os.path.join(a.official, f), dst)
            except OSError:
                shutil.copyfile(os.path.join(a.official, f), dst)
        for k in by_file[f]:
            weight_map[k] = f
    missing_layers = []
    for l in range(n_layers):
        parts = sorted(glob.glob(os.path.join(a.pass1, f"layer_{l:02d}.part*.safetensors")))
        if not parts:
            missing_layers.append(l)
            continue
        name = f"model-exl3-L{l:02d}.safetensors"
        dst = os.path.join(a.out, name)
        t = {}
        for p in parts:
            t.update(load_file(p))
        n_exp = len({k.split(".")[4] for k in t if ".ffn.experts." in k})
        dkeys = [k for k in dense_t if k.startswith(f"layers.{l}.")]
        if a.dense and not dkeys:
            raise SystemExit(f"--dense: no dense EXL3 tensors for layer {l}")
        for k in dkeys:
            with safe_open(os.path.join(a.dense, dense_t[k]), framework="pt") as sf:
                t[k] = sf.get_tensor(k)
        assert n_exp == plan["num_experts"], f"layer {l}: {n_exp} experts in {len(parts)} part file(s)"
        if not os.path.exists(dst):
            save_file(t, dst + ".tmp", metadata={"format": "pt", "quant": "exl3", "encoder": "exllamav3 1.6.0"})
            os.replace(dst + ".tmp", dst)
        for k in t:
            weight_map[k] = name
    if missing_layers:
        raise SystemExit(f"layers without EXL3 tensors: {missing_layers} (they would have no routed experts)")
    head_k = None
    hkeys = [k for k in dense_t if k.startswith("head.")]
    if hkeys:
        name = "model-exl3-head.safetensors"
        th = {}
        for k in hkeys:
            with safe_open(os.path.join(a.dense, dense_t[k]), framework="pt") as sf:
                th[k] = sf.get_tensor(k)
        head_k = th["head.trellis"].shape[-1] // 16
        dst = os.path.join(a.out, name)
        if not os.path.exists(dst):
            save_file(th, dst + ".tmp", metadata={"format": "pt", "quant": "exl3", "encoder": "exllamav3 1.6.0"})
            os.replace(dst + ".tmp", dst)
        for k in th:
            weight_map[k] = name
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"assembled_from": "deepseek-ai/DeepSeek-V4.1-Flash@2cba9e42"},
                   "weight_map": dict(sorted(weight_map.items()))}, f, indent=1)
    cfg = json.load(open(os.path.join(a.official, "config.json")))
    if a.test_layers:
        (cfg["text_config"] if "text_config" in cfg else cfg)["num_hidden_layers"] = a.test_layers   # no self-reference
    q = dict(cfg["quantization_config"])
    q["quant_method"] = "deepseek_v41_exl3"
    q["exl3_moe"] = {"plan_file": "exl3_moe_plan.json", "decode_kernel": "coop", "coop_max_tokens": 32,
                     "fused_rows": 256, "tp_layout": "channels", "exllamav3_version": "1.6.0"}
    if a.dense:
        ks = {}
        for k in dense_t:
            if not k.endswith(".trellis") or k.startswith("head."):
                continue
            mod = re.sub(r"^layers\.\d+\.", "", k[: -len(".trellis")])
            mod = re.sub(r"\.slice\.\d+$", "", mod)
            mod = "ffn.shared_experts.w1" if mod == "ffn.shared_experts.w3" else mod
            mod = "attn.wq_a" if mod == "attn.wkv" else mod              # one K per fused module (fused_wqa_wkv)
            with safe_open(os.path.join(a.dense, dense_t[k]), framework="pt") as sf:
                kk = sf.get_slice(k).get_shape()[-1] // 16
            ks.setdefault(mod, set()).add(kk)
        mixed = {m: v for m, v in ks.items() if len(v) != 1}
        if mixed:
            raise SystemExit(f"--dense: one K per module across layers is what the plugin reads, got {mixed}")
        layers_d = sorted({int(k.split(".")[1]) for k in dense_t if k.startswith("layers.")})
        if head_k:
            q["exl3_moe"]["head"], q["exl3_moe"]["head_k"] = "exl3", head_k
        q["exl3_moe"]["dense"] = {"k": {m: v.pop() for m, v in sorted(ks.items())}, "layers": [layers_d[0], layers_d[-1]]}
        if layers_d != list(range(layers_d[0], layers_d[-1] + 1)):
            raise SystemExit(f"--dense: layers must be contiguous, got {layers_d}")
    cfg["quantization_config"] = q
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    shutil.copyfile(a.plan, os.path.join(a.out, "exl3_moe_plan.json"))
    for f in COPY_FILES:
        if os.path.exists(os.path.join(a.official, f)):
            shutil.copyfile(os.path.join(a.official, f), os.path.join(a.out, f))
    for d in COPY_DIRS:
        if os.path.isdir(os.path.join(a.official, d)) and not os.path.exists(os.path.join(a.out, d)):
            shutil.copytree(os.path.join(a.official, d), os.path.join(a.out, d))
    if not a.no_hash:
        with open(os.path.join(a.out, "MANIFEST.sha256"), "w") as f:
            for name in sorted(os.listdir(a.out)):
                p = os.path.join(a.out, name)
                if os.path.isfile(p) and name != "MANIFEST.sha256":
                    f.write(f"{sha256(p)}  {name}\n")
    print(f"assembled {a.out}: {len(weight_map)} tensors")


if __name__ == "__main__":
    main()
