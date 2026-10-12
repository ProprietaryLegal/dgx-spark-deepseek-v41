#!/usr/bin/env bash
# Build upstream vLLM (pinned commit) for GB10 / DGX Spark (sm_121) as a wheel. Runs on a Spark.
# The official vLLM arm64 image is used ONLY as the build environment (its torch + CUDA 13 toolchain); every vLLM
# .so in the wheel is compiled here from the pinned source. CPU-only compile, memory-capped, one build at a time.
# GB10 build lessons (arch list, CUDA headers, FA3) learned by reading 0xSero/local-ai-images
# deepseek-v4.1-flash-spark/build_wheels.sh (reference only, not vendored).
#   git clone https://github.com/vllm-project/vllm ~/src/vllm-main-6e517b15 && git -C ~/src/vllm-main-6e517b15 checkout 6e517b15c1833cf72a7f557ee32524d98682e617
#   ./run_build_gb10.sh          (PIN= with another commit, or PIN=any, to build something else)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/src/vllm-main-6e517b15}
PIN=${PIN:-6e517b15c1833cf72a7f557ee32524d98682e617}
OUT=${OUT:-$HOME/dsv41_recipe/wheels}
LOGS=${LOGS:-$HOME/dsv41_recipe/logs}
IMG=${IMG:-vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3}   # v0.31.0 arm64: toolchain only
MEM=${MEM:-88g}
MAX_JOBS=${MAX_JOBS:-10}
VLLM_VERSION=${VLLM_VERSION:-0.31.1.dev0+g6e517b15.dsv41gb10}
[ -f "$SRC/CMakeLists.txt" ] || { echo "no vLLM source at $SRC"; exit 1; }
[ "$PIN" = any ] || [ "$(git -C "$SRC" rev-parse HEAD)" = "$PIN" ] || { echo "$SRC is not at $PIN"; exit 1; }
docker ps --format '{{.Names}}' | grep -qx dsv41-build && { echo "a build is already running"; exit 1; }
mkdir -p "$OUT" "$LOGS"
echo "build start $(date -Is) img=$IMG src=$(git -C "$SRC" rev-parse HEAD) mem=$MEM jobs=$MAX_JOBS" | tee -a "$LOGS/build_runs.log"
docker run --rm --name dsv41-build --memory "$MEM" --memory-swap "$MEM" \
  -e MAX_JOBS="$MAX_JOBS" -e NVCC_THREADS=1 -e VLLM_VERSION="$VLLM_VERSION" \
  -v "$SRC":/src/vllm -v "$OUT":/out -v "$LOGS":/logs -v "$HERE":/build:ro \
  --entrypoint bash "$IMG" /build/build_vllm_in_container.sh
echo "build end $(date -Is) rc=$?" | tee -a "$LOGS/build_runs.log"
ls -la "$OUT"
