#!/usr/bin/env bash
# Overlay build on a Spark: BASE image + the patches listed (from patches/ or patches/experimental/) + the plugin from
# source.
#   BASE=dsv41-serve:dev TAG=dsv41-serve:v2 PATCHES="0007 0008" ./build_serve_overlay.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-dsv41-serve:dev}; TAG=${TAG:-dsv41-serve:v3}; PATCHES=${PATCHES:-"0007 0008 0010 0011"}   # never retag the base; experimental patches: patches/experimental/README.md
STAGE=$(mktemp -d ~/dsv41_recipe/image_stage.XXXX)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/patches"
for n in $PATCHES; do
  f=$(ls ~/dsv41_recipe/patches/$n-*.patch ~/dsv41_recipe/patches/experimental/$n-*.patch 2>/dev/null | head -1 || true)
  [ -n "$f" ] || { echo "no patch $n"; exit 1; }
  cp "$f" "$STAGE/patches/"
done
cp -r ~/dsv41_recipe/patches/exl3_moe "$STAGE/plugin"
find "$STAGE/plugin" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$HERE/Dockerfile.serve-overlay" "$STAGE/Dockerfile"
LOG=$STAGE.log
if ! docker build --progress=plain --build-arg BASE="$BASE" -t "$TAG" "$STAGE" > "$LOG" 2>&1; then
  grep -n -E "Traceback|Error|error|ERROR" "$LOG" | head -5; tail -25 "$LOG"; echo "build failed, full log: $LOG"; exit 1
fi
grep -E "overlay ok|Applied patch" "$LOG"; rm -f "$LOG"
docker image inspect "$TAG" --format 'built {{.Id}} {{.Size}}'
