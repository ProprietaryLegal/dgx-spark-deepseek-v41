#!/usr/bin/env bash
# Everything that needs an idle Spark GPU before the real serve, in one go (~20 min): the plugin's GPU tests, the MoE
# decode benches (chained vs parallel K groups; coop tile / split-k knobs), dense EXL3 linear time and output error per
# int8 activation mode (EXL3_INT8_GEMV 0 / 1 / 2). Logs into ./results. Run from the control machine, nothing else on
# that box's GPU.
#   BOX=spark2 ./gpu_checks.sh            (IMAGE=dsv41-serve:v2)
set -uo pipefail
BOX=${BOX:-spark2}; IMAGE=${IMAGE:-dsv41-serve:v2}
OUT=${OUT:-${DSV41_RESULTS:-$(cd "$(dirname "$0")/../.." && pwd)/results}/kernels/gpu_checks_$(date +%Y-%m-%d_%H%M)_$BOX}
mkdir -p "$OUT"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
rsync -a --exclude __pycache__ "$REPO/patches/exl3_moe/" "$BOX:/tmp/gc_plugin/"
rsync -a --exclude __pycache__ "$REPO/scripts/bench/" "$BOX:/tmp/gc_bench/"
[ "$(ssh -o BatchMode=yes "$BOX" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")" = 0 ] \
  || { echo "$BOX has GPU processes running"; exit 1; }
run() {   # $1 log name, $2 env (space-separated KEY=VALUE), rest: command inside the image
  local name=$1 envs=$2; shift 2
  local e=""; for kv in $envs; do e="$e -e $kv"; done
  echo "== $name [$envs] $(date +%H:%M:%S)" | tee -a "$OUT/summary.txt"
  ssh -o BatchMode=yes "$BOX" "docker run --rm --gpus all --ipc host $e -e DSV41_CUDA_MEM_FRACTION=0.5 \
    -v /tmp/gc_plugin/tests:/t:ro -v /tmp/gc_bench:/b:ro --entrypoint bash $IMAGE -c '$*'" > "$OUT/$name.log" 2>&1
  grep -E "passed|failed|error|x\)|us/layer|GB/s|rel |gaussian" "$OUT/$name.log" | grep -v -i warn | tail -40 | tee -a "$OUT/summary.txt"
}
# 1. correctness (the installed plugin in the image, tests from the repo)
run tests "" "cd /t && python3 -m pytest -q -s -p no:cacheprovider test_kernels_gpu.py test_dense_gpu.py test_head_exl3_gpu.py test_head_mxfp8.py test_parallel_groups_gpu.py"
# 2. MoE decode: chained vs parallel K groups (plan v2 layer 0 mix, TP=2 shard), default kernel heuristics then forced knobs
run moe_par_default "EXL3_BENCH=1" "cd /t && python3 -m pytest -q -s -p no:cacheprovider test_parallel_groups_gpu.py -k bench"
for knobs in "EXL3_MOE_COOP_WIDE=1" "EXL3_MOE_COOP_WIDE=0" "EXL3_MOE_COOP_KSPLIT=2" "EXL3_MOE_COOP_KSPLIT=4"; do
  run "moe_par_${knobs//=/_}" "EXL3_BENCH=1 $knobs" "cd /t && python3 -m pytest -q -s -p no:cacheprovider test_parallel_groups_gpu.py -k bench"
done
run moe_graph_default "EXL3_BENCH=1" "cd /t && python3 -m pytest -q -s -p no:cacheprovider test_kernels_gpu.py -k bench"
# 3. dense EXL3 linears: time (DRAM-bound, m = 1 / 6) and output error, per int8 activation mode
for mode in 2 1 0; do
  run "dense_time_int8_$mode" "EXL3_INT8_GEMV=$mode" "python3 /b/linear_dram_bench.py 1,6"
  run "dense_err_int8_$mode" "EXL3_INT8_GEMV=$mode" "python3 /b/exl3_int8_accuracy.py --m 1,6 --k 5,6"
done
ssh -o BatchMode=yes "$BOX" "rm -rf /tmp/gc_plugin /tmp/gc_bench"
echo "logs: $OUT"
