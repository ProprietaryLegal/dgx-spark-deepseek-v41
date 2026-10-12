# SPDX-License-Identifier: Apache-2.0
"""p4_assemble.py --dense on a synthetic one-layer checkpoint: the EXL3 dense set (wq_a + wkv, wq_b) replaces exactly
those modules' FP8 weight / scale, every other tensor stays (the compressor's wkv, the drafter's wq_a, the norms), the
routed experts come from pass 1, and the config's dense block folds wkv into wq_a's K. CPU, tiny files.

  python3 test_assemble_dense_cpu.py /path/to/scripts/quantize        (any image with torch + safetensors)
"""
import json
import os
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file


def exl3(in_f, out_f, k):
    return {"trellis": torch.zeros(in_f // 16, out_f // 16, 16 * k, dtype=torch.int16),
            "suh": torch.ones(in_f, dtype=torch.float16), "svh": torch.ones(out_f, dtype=torch.float16),
            "mul1": torch.tensor(1, dtype=torch.int32)}


def main():
    q = sys.argv[1]
    d = tempfile.mkdtemp()
    off, p1, dense, out = (os.path.join(d, n) for n in ("official", "p1", "dense", "out"))
    for x in (off, p1, dense):
        os.makedirs(x)
    fp8 = lambda *s: torch.zeros(*s, dtype=torch.float8_e4m3fn)           # noqa: E731
    official = {"embed.weight": torch.zeros(8, 128), "head.weight": torch.zeros(8, 128), "norm.weight": torch.ones(128),
                "layers.0.attn.wq_a.weight": fp8(128, 128), "layers.0.attn.wq_a.scale": torch.ones(1, 1),
                "layers.0.attn.wkv.weight": fp8(128, 128), "layers.0.attn.wkv.scale": torch.ones(1, 1),
                "layers.0.attn.wq_b.weight": fp8(256, 128), "layers.0.attn.wq_b.scale": torch.ones(2, 1),
                "layers.0.attn.compressor.wkv.weight": torch.zeros(128, 128),
                "layers.0.attn_norm.weight": torch.ones(128),
                "layers.0.ffn.experts.0.w1.weight": fp8(128, 128), "layers.0.ffn.experts.1.w1.weight": fp8(128, 128),
                "mtp.0.attn.wq_a.weight": fp8(128, 128)}
    save_file(official, os.path.join(off, "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in official}},
              open(os.path.join(off, "model.safetensors.index.json"), "w"))
    json.dump({"quantization_config": {"quant_method": "fp8"}, "num_hidden_layers": 40},
              open(os.path.join(off, "config.json"), "w"))
    plan = os.path.join(d, "plan.json")
    json.dump({"num_layers": 40, "num_experts": 2}, open(plan, "w"))
    save_file({f"layers.0.ffn.experts.{e}.w{j}.{p}": v for e in range(2) for j in (1, 2, 3)
               for p, v in exl3(128, 128, 2).items()}, os.path.join(p1, "layer_00.part0.safetensors"))
    dt = {}
    for mod, (i, o) in (("attn.wq_a", (128, 128)), ("attn.wkv", (128, 128)), ("attn.wq_b", (128, 256))):
        dt.update({f"layers.0.{mod}.{p}": v for p, v in exl3(i, o, 5).items()})
    save_file(dt, os.path.join(dense, "dense_L00.safetensors"))
    json.dump({"weight_map": {k: "dense_L00.safetensors" for k in dt}},
              open(os.path.join(dense, "model.safetensors.index.json"), "w"))
    subprocess.run([sys.executable, os.path.join(q, "p4_assemble.py"), off, p1, plan, out, "--no-hash",
                    "--test-layers", "1", "--dense", dense], check=True, stdout=subprocess.DEVNULL)
    wm = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    gone = {"layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale", "layers.0.attn.wkv.weight",
            "layers.0.attn.wkv.scale", "layers.0.attn.wq_b.weight", "layers.0.attn.wq_b.scale",
            "layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.1.w1.weight", "mtp.0.attn.wq_a.weight"}
    assert not gone & set(wm), sorted(gone & set(wm))                  # mtp.* drops because of --test-layers
    for k in ("layers.0.attn.compressor.wkv.weight", "layers.0.attn_norm.weight", "embed.weight", "head.weight",
              "layers.0.attn.wq_a.trellis", "layers.0.attn.wkv.svh", "layers.0.attn.wq_b.mul1",
              "layers.0.ffn.experts.1.w3.trellis"):
        assert k in wm, k
    t = load_file(os.path.join(out, wm["layers.0.attn.wkv.trellis"]))
    assert t["layers.0.attn.wkv.trellis"].shape == (8, 8, 80)
    cfg = json.load(open(os.path.join(out, "config.json")))
    dense_cfg = cfg["quantization_config"]["exl3_moe"]["dense"]
    assert dense_cfg == {"k": {"attn.wq_a": 5, "attn.wq_b": 5}, "layers": [0, 0]}, dense_cfg
    print(f"assemble --dense: {len(wm)} tensors, wq_a / wkv / wq_b FP8 replaced, compressor.wkv kept, dense {dense_cfg}")
    print("OK")


if __name__ == "__main__":
    main()
