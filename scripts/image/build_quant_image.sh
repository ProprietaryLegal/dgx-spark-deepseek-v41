#!/usr/bin/env bash
# Builds dsv41-quant:dev on a Spark from dsv41-recipe:dev + the exllamav3 wheel in ~/dsv41_recipe/wheels.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-dsv41-quant:dev}
STAGE=$(mktemp -d ~/dsv41_recipe/image_stage.XXXX)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/wheels"
cp ~/dsv41_recipe/wheels/exllamav3-1.6.0-*.whl "$STAGE/wheels/"
cp "$HERE/Dockerfile.quant" "$STAGE/Dockerfile"
docker build --progress=plain -t "$TAG" "$STAGE" 2>&1
docker image inspect "$TAG" --format 'built {{.Id}} {{.Size}}'
