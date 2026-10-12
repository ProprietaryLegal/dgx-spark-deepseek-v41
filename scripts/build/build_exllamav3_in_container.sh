#!/usr/bin/env bash
# Runs INSIDE the build-environment image (vllm/vllm-openai@sha256:c1c9f6fd..., as for the vLLM wheel: torch 2.13.0+cu130,
# nvcc 13.0, py3.12). Builds the exllamav3 v1.6.0 wheel (MIT, turboderp-org) for GB10 = sm_121.
set -euo pipefail
: "${MAX_JOBS:=10}"
export MAX_JOBS CUDA_HOME=/usr/local/cuda CUDACXX=/usr/local/cuda/bin/nvcc
TS=$(date +%Y%m%d_%H%M%S)
python3 -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'py', sys.version.split()[0])"
# CUDA math headers (cusparse.h, ...) only in the cu13 pip wheel: same fix as build_vllm_in_container.sh step 1.
PIP_INC=""
for c in $(python3 -c "import nvidia; print(' '.join(p + '/cu13/include' for p in nvidia.__path__))" 2>/dev/null) \
         /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include; do
  [ -e "$c/cusparse.h" ] && { PIP_INC=$c; break; }
done
if [ -n "$PIP_INC" ]; then
  EXTRA=/opt/cu13-extra-include; mkdir -p "$EXTRA"
  for f in "$PIP_INC"/*; do b=$(basename "$f"); [ "$b" = crt ] && continue; [ -e "/usr/local/cuda/include/$b" ] || ln -sfn "$f" "$EXTRA/$b"; done
  export CPATH=$EXTRA${CPATH:+:$CPATH}; echo "CPATH extra headers: $(ls "$EXTRA" | wc -l) entries"
fi
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq --no-install-recommends git >/dev/null; }
git config --global --add safe.directory '*'
python3 -m pip install -q "setuptools>=77" wheel ninja
cd /src/x3
echo "exllamav3 $(git rev-parse HEAD)"
t0=$(date +%s)
TORCH_CUDA_ARCH_LIST="12.1" python3 -m pip wheel -v --no-build-isolation --no-deps -w /out . 2>&1 | tee "/logs/exllamav3_wheel_$TS.log" > /dev/null
ls -la /out/exllamav3-*.whl
echo "exllamav3 wheel built in $(( $(date +%s) - t0 ))s"
