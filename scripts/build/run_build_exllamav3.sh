#!/usr/bin/env bash
# Build the exllamav3 v1.6.0 wheel for GB10 on a Spark, in the vLLM build-environment image. One build at a time.
#   SRC=~/src/exllamav3-v1.6.0 ./run_build_exllamav3.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/src/exllamav3-v1.6.0}
OUT=${OUT:-$HOME/dsv41_recipe/wheels}
LOGS=${LOGS:-$HOME/dsv41_recipe/logs}
IMG=${IMG:-vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3}
MEM=${MEM:-64g}
PIN=151539c77abc7ab7425d30da7a4e8e3c5c154e7b
[ "$(git -C "$SRC" rev-parse HEAD)" = "$PIN" ] || { echo "exllamav3 source is not v1.6.0 ($PIN)"; exit 1; }
docker ps --format '{{.Names}}' | grep -qE '^(dsv41-build|x3-build)$' && { echo "a build is already running"; exit 1; }
mkdir -p "$OUT" "$LOGS"
echo "x3 build start $(date -Is) img=$IMG src=$PIN mem=$MEM" | tee -a "$LOGS/build_runs.log"
docker run --rm --name x3-build --memory "$MEM" --memory-swap "$MEM" -e MAX_JOBS="${MAX_JOBS:-10}" \
  -v "$SRC":/src/x3 -v "$OUT":/out -v "$LOGS":/logs -v "$HERE":/build:ro \
  --entrypoint bash "$IMG" /build/build_exllamav3_in_container.sh
echo "x3 build end $(date -Is) rc=$?" | tee -a "$LOGS/build_runs.log"
