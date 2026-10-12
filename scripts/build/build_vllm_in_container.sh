#!/usr/bin/env bash
# Runs INSIDE the build-environment image (arm64, torch + cu13, nvcc 13, py3.12). Builds the vLLM wheel for sm_121.
set -euo pipefail
: "${MAX_JOBS:=10}" "${NVCC_THREADS:=1}" "${VLLM_VERSION:=0.31.0+dsv41.gb10}"
export MAX_JOBS NVCC_THREADS CUDA_HOME=/usr/local/cuda CUDACXX=/usr/local/cuda/bin/nvcc
TS=$(date +%Y%m%d_%H%M%S)
python3 -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'py', sys.version.split()[0])"
"$CUDACXX" --version | tail -2
# 1. CUDA math headers the toolkit lacks (cusparse.h, ...) live only in the cu13 pip wheel. Expose just those, never
#    the pip crt/cccl headers (CPATH beats -isystem, and nvcc 13.0 stubs break on newer crt headers).
# `nvidia` is a namespace package (no __file__): search its __path__, then the usual dist-packages spot.
PIP_INC=""
for c in $(python3 -c "import nvidia; print(' '.join(p + '/cu13/include' for p in nvidia.__path__))" 2>/dev/null) \
         /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include; do
  [ -e "$c/cusparse.h" ] && { PIP_INC=$c; break; }
done
if [ -z "$PIP_INC" ] && [ ! -e /usr/local/cuda/include/cusparse.h ]; then
  echo "FATAL: cusparse.h found neither in /usr/local/cuda/include nor in the cu13 pip wheel"; find / -name cusparse.h 2>/dev/null | head; exit 1
fi
echo "cu13 pip headers: ${PIP_INC:-none needed}"
if [ -n "$PIP_INC" ] && [ -d "$PIP_INC" ]; then
  EXTRA=/opt/cu13-extra-include; mkdir -p "$EXTRA"
  for f in "$PIP_INC"/*; do b=$(basename "$f"); [ "$b" = crt ] && continue; [ -e "/usr/local/cuda/include/$b" ] || ln -sfn "$f" "$EXTRA/$b"; done
  export CPATH=$EXTRA${CPATH:+:$CPATH}; echo "CPATH extra headers: $(ls "$EXTRA" | wc -l) entries"
fi
# 2. tools (CMake FetchContent needs git)
# git: CMake FetchContent; libdw-dev: DeepGEMM's deep_jit includes elfutils/libdwfl.h
NEED=""; command -v git >/dev/null || NEED="$NEED git"; [ -e /usr/include/elfutils/libdwfl.h ] || NEED="$NEED libdw-dev"
if [ -n "$NEED" ]; then apt-get update -qq && apt-get install -y -qq --no-install-recommends $NEED >/dev/null && echo "installed:$NEED"; fi
git config --global --add safe.directory '*'
python3 -m pip install -q "cmake>=3.26.1,<4" "setuptools-rust>=1.9.0" "setuptools>=77.0.3,<81" "setuptools-scm>=8" build wheel ninja
cd /src/vllm
# 3. idempotent source fixes, each logged, the diff saved
# vLLM's CUDA>=13.0 (and >=13.4) supported-arch lists end at 12.0. With TORCH_CUDA_ARCH_LIST=12.1a the loose
# intersection narrows to 12.0 and kernels come out as sm_120a cubins, which do not load on GB10 (sm_121). Append 12.1
# to every CUDA_SUPPORTED_ARCHS list that ends at 12.0 (the CUDA 12.8 list already has it). Idempotent.
n=$(grep -cE 'set\(CUDA_SUPPORTED_ARCHS "[^"]*;12\.0"\)' CMakeLists.txt || true)
if [ "${n:-0}" -gt 0 ]; then
  sed -i -E 's/(set\(CUDA_SUPPORTED_ARCHS "[^"]*;12\.0)("\))/\1;12.1\2/' CMakeLists.txt
  echo "FIX: CUDA_SUPPORTED_ARCHS += 12.1 on $n list(s)"
fi
grep -n 'set(CUDA_SUPPORTED_ARCHS' CMakeLists.txt
FA3='ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))'
if grep -qF "$FA3" setup.py; then
  python3 - <<'PY'
from pathlib import Path
p = Path("setup.py"); s = p.read_text()
old = 'ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))'
p.write_text(s.replace(old, 'pass  # GB10 build: FA3 is Hopper-only (sm_90), skipped', 1)); print("FIX: FA3 extension skipped")
PY
fi
git diff > "/logs/build_source_fixes_$TS.diff" || true
# 4. the wheel
t0=$(date +%s)
SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION" VLLM_TARGET_DEVICE=cuda TORCH_CUDA_ARCH_LIST=12.1a \
VLLM_DISABLE_SCCACHE=1 CMAKE_BUILD_TYPE=Release \
  python3 -m pip wheel -v --no-build-isolation --no-deps -w /out . 2>&1 | tee "/logs/vllm_wheel_$TS.log" > /dev/null
ls /out/vllm-*.whl
echo "vllm wheel built in $(( $(date +%s) - t0 ))s"
