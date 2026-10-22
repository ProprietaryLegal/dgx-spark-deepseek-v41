#!/usr/bin/env bash
# One-command install (README quickstart, step 2), run on one Spark from this repo cloned to ~/dsv41_recipe:
#   1. the vLLM source at the pin and exllamav3 v1.6.0 (cloned if missing);
#   2. the vLLM wheel for sm_121 (a memory-capped CPU compile in the pinned build-environment image) and the exllamav3
#      wheel, each skipped when its wheel is already in ~/dsv41_recipe/wheels (FORCE=1 rebuilds);
#   3. the images: dsv41-recipe:dev (every patch in patches/, verified), dsv41-quant:dev, then the serving image $TAG;
#   4. the serving image to the other Spark (docker save | ssh docker load), and both ids printed.
# Refuses to start while a GPU container runs on either box: heavy builds and image copies beside GPU work have hung
# a Spark (FINDINGS.md 6).
#   PEER=<ssh name of the other Spark> ./install.sh
set -euo pipefail
cd "$(dirname "$0")"
[ "$(pwd -P)" = "$(cd ~/dsv41_recipe 2>/dev/null && pwd -P)" ] || { echo "clone this repo to ~/dsv41_recipe first (the image scripts read from there)"; exit 1; }
PEER=${PEER:?ssh name of the other Spark}
TAG=${TAG:-dsv41-serve:v5}   # v2.0: patches 0023-0038 and the plugin as of v2.0 (v1.x: dsv41-serve:v4)
VLLM_PIN=6e517b15c1833cf72a7f557ee32524d98682e617
VLLM_SRC=${VLLM_SRC:-$HOME/src/vllm-main-6e517b15}
X3_SRC=${X3_SRC:-$HOME/src/exllamav3-v1.6.0}
BUILD_ENV=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
step() { echo; echo "== $(date +%T) $*"; }

GPU_CTRS='docker ps -q | xargs -r docker inspect --format "{{.Name}} {{json .HostConfig.DeviceRequests}}" | grep -i gpu | cut -d" " -f1'
g=$(bash -c "$GPU_CTRS"); [ -z "$g" ] || { echo "GPU containers running here, stop them first: $g"; exit 1; }
g=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" "$GPU_CTRS") || { echo "$PEER unreachable"; exit 1; }
[ -z "$g" ] || { echo "GPU containers running on $PEER, stop them first: $g"; exit 1; }

step "sources"
[ -d "$VLLM_SRC/.git" ] || git clone https://github.com/vllm-project/vllm "$VLLM_SRC"
git -C "$VLLM_SRC" checkout -q "$VLLM_PIN"
[ -d "$X3_SRC/.git" ] || git clone --branch v1.6.0 https://github.com/turboderp-org/exllamav3 "$X3_SRC"
echo "vLLM $(git -C "$VLLM_SRC" rev-parse --short HEAD), exllamav3 $(git -C "$X3_SRC" describe --tags 2>/dev/null || git -C "$X3_SRC" rev-parse --short HEAD)"

step "vLLM wheel"
if [ "${FORCE:-0}" != 1 ] && ls wheels/vllm-*.whl >/dev/null 2>&1; then echo "have $(ls wheels/vllm-*.whl)"
else SRC=$VLLM_SRC IMG=$BUILD_ENV scripts/build/run_build_gb10.sh; fi

step "exllamav3 wheel"
if [ "${FORCE:-0}" != 1 ] && ls wheels/exllamav3-1.6.0-*.whl >/dev/null 2>&1; then echo "have $(ls wheels/exllamav3-1.6.0-*.whl)"
else SRC=$X3_SRC IMG=$BUILD_ENV scripts/build/run_build_exllamav3.sh; fi

step "images"
scripts/image/build_image.sh | tail -25
scripts/image/build_quant_image.sh | tail -3
TAG=$TAG scripts/image/build_serve_image.sh | tail -3

step "serving image to $PEER"
docker save "$TAG" | ssh -o BatchMode=yes "$PEER" docker load
here=$(docker image inspect "$TAG" --format '{{.Id}}')
there=$(ssh -o BatchMode=yes "$PEER" "docker image inspect $TAG --format '{{.Id}}'")
echo "$TAG: here ${here:7:12}, $PEER ${there:7:12}"
[ "$here" = "$there" ] || { echo "the ids differ"; exit 1; }
echo "done: next, the weights (README quickstart, step 3) and ./check.sh"
