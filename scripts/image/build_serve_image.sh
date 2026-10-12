#!/usr/bin/env bash
# Builds the plugin wheel (inside the quant image, no network) and dsv41-serve:dev on a Spark.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-dsv41-serve:dev}
PLUGIN=${PLUGIN:-$HOME/dsv41_recipe/patches/exl3_moe}
STAGE=$(mktemp -d ~/dsv41_recipe/image_stage.XXXX)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/wheels"
docker run --rm -v "$PLUGIN":/src:ro -v "$STAGE/wheels":/out --entrypoint bash dsv41-quant:dev -c \
  "cp -r /src /tmp/p && cd /tmp/p && python3 -m pip wheel --no-deps --no-build-isolation -w /out . >/dev/null && chmod 644 /out/*.whl"
ls "$STAGE/wheels"
cp "$HERE/Dockerfile.serve" "$STAGE/Dockerfile"
docker build --progress=plain -t "$TAG" "$STAGE" 2>&1
docker image inspect "$TAG" --format 'built {{.Id}} {{.Size}}'
