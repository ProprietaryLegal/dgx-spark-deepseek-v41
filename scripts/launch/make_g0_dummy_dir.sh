#!/usr/bin/env bash
# G0 dummy model dir for `LOAD_FORMAT=dummy dsv41_tp2.sh up`: our config (quant_method deepseek_v41_exl3) + the plan +
# tokenizer files (patches/exl3_moe/tools/make_dummy_model_dir.py), plus the official index and the two Engram shards
# hardlinked, because Engram disk offload resolves its tables through the index even when no weights load.
# Run on each Spark (inside the serve image, which has the plugin):
#   make_g0_dummy_dir.sh /home/spark/models/DeepSeek-V4.1-Flash-official-2cba9e42 plan.json /home/spark/models/dsv41-g0-dummy
set -euo pipefail
SRC=$1 PLAN=$2 OUT=$3
mkdir -p "$OUT"
docker run --rm -v "$SRC":"$SRC":ro -v "$(dirname "$(readlink -f "$PLAN")")":/plan:ro -v "$OUT":/out \
  -v "$HOME/dsv41_recipe/patches/exl3_moe":/p:ro --entrypoint bash dsv41-serve:dev -c \
  "python3 /p/tools/make_dummy_model_dir.py --src '$SRC' --plan /plan/$(basename "$PLAN") --out /out && chmod -R a+rX /out"
# the tool symlinks tokenizer files into $SRC; the serve container mounts only $OUT, so make them real files
for f in "$OUT"/*; do [ -L "$f" ] && cp --remove-destination "$(readlink -f "$f")" "$f"; done
cp -f "$SRC/model.safetensors.index.json" "$OUT/"
for f in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  [ -e "$OUT/$f" ] || ln "$SRC/$f" "$OUT/$f"
done
ls -la "$OUT"
