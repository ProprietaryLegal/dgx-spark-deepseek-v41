#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a weightless model dir for `vllm serve <dir> --load-format dummy` (G0 fit + speed probe).

Copies config.json from an official DeepSeek-V4.1-Flash snapshot with quantization_config switched to
`deepseek_v41_exl3` + an `exl3_moe` block, writes the plan, and symlinks the tokenizer / generation /
chat-template files. No safetensors: with --load-format dummy vLLM allocates every parameter (our EXL3
banks included, sized by the plan) and never reads weights. Integer params stay uninitialised and
fp16 ones get small random values (model_loader/weight_utils.py:1480-1499), which is fine for memory
and bandwidth numbers and meaningless for text.

  tools/make_dummy_model_dir.py --src ~/models/DeepSeek-V4.1-Flash-official-2cba9e42 \
      --plan /tmp/plan_mix.json --out ~/models/dsv41-exl3-dummy
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_exl3_moe import QUANT_METHOD  # noqa: E402
from vllm_exl3_moe.plan import Exl3MoEPlan  # noqa: E402

LINK = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
        "special_tokens_map.json", "preprocessor_config.json", "processor_config.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="official snapshot dir (config.json + tokenizer files)")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode-kernel", default="coop", choices=("coop", "fused"))
    a = ap.parse_args()

    plan = Exl3MoEPlan.from_json_file(a.plan)
    with open(os.path.join(a.src, "config.json")) as f:
        cfg = json.load(f)
    q = dict(cfg.get("quantization_config") or {})
    if q.get("quant_method") not in ("fp8", "deepseek_v4_fp8", QUANT_METHOD):
        sys.exit(f"unexpected source quant_method {q.get('quant_method')!r}")
    q["quant_method"] = QUANT_METHOD
    q["exl3_moe"] = {"plan_file": "exl3_moe_plan.json", "decode_kernel": a.decode_kernel,
                     "tp_layout": "channels", "exllamav3_version": "1.6.0"}
    cfg["quantization_config"] = q
    text = cfg.get("text_config", cfg)
    if plan.num_layers != text.get("num_hidden_layers") or plan.num_experts != text.get("n_routed_experts"):
        sys.exit("plan geometry does not match config.json")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    shutil.copyfile(a.plan, os.path.join(a.out, "exl3_moe_plan.json"))
    for name in LINK:
        src = os.path.join(a.src, name)
        dst = os.path.join(a.out, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)
    print(f"wrote {a.out}: plan {plan.bpw():.3f} bpw, {plan.bytes_per_rank(2) / 2**30:.2f} GiB experts per rank at TP=2")


if __name__ == "__main__":
    main()
