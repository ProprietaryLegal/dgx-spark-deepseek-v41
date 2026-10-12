#!/usr/bin/env bash
# Builds the dsv41-recipe runtime image on a Spark from the wheel in ~/dsv41_recipe/wheels and the repo's patches.
#   TAG=dsv41-recipe:dev ./build_image.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-dsv41-recipe:dev}
STAGE=$(mktemp -d ~/dsv41_recipe/image_stage.XXXX)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/wheels" "$STAGE/patches"
cp ~/dsv41_recipe/wheels/vllm-*.whl "$STAGE/wheels/"
cp ~/dsv41_recipe/patches/[0-9][0-9][0-9][0-9]-*.patch "$STAGE/patches/"   # every numbered patch, applied in order
cp "$HERE/Dockerfile" "$STAGE/Dockerfile"
docker build --progress=plain -t "$TAG" "$STAGE" 2>&1
docker image inspect "$TAG" --format 'built {{.Id}} {{.Size}}'
# Prove the patch series is in the image: on a scratch copy of the installed vllm package, undo the patches in reverse
# order (each must reverse-apply cleanly on the tree the later ones left), the check for a stacked series whose patches
# touch neighbouring lines.
docker run --rm --entrypoint bash "$TAG" -c 'SITE="$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))" 2>/dev/null)" &&
  mkdir -p /tmp/chk && cp -r "$SITE/vllm" /tmp/chk/ && cd /tmp/chk &&
  for p in $(ls /opt/dsv41/patches/*.patch | sort -r); do git apply -R --include="vllm/*" -p1 "$p" && echo "in image: $(basename "$p")" || exit 1; done' 2>&1 | grep -v -i warn
