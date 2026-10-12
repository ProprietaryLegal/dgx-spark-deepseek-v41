#!/usr/bin/env bash
# Served decode / prefill logits vs the reference harness on one Spark (scripts/bench/served_vs_harness.py): panel from a
# kit_bench oracle file, the native reference (p4_kld.py --mode ref), our weights through the same harness (--mode quant
# with the pass 1 experts + the dense pass, the build v2.1 weights the published KLD was measured on, --dump-top 20),
# then the table. ~30 min of GPU time on an idle box. Run from the control machine:
#   BOX=spark2 ./served_vs_harness_run.sh <assets>/speed/sweep_v21_k3_pw/kit_oracle.jsonl
set -uo pipefail
ORACLE=${1:?kit_bench oracle jsonl}
BOX=${BOX:-spark2}; IMAGE=${IMAGE:-dsv41-serve:dev}
R=${RECIPE_DIR:-/home/spark/dsv41_recipe}
MODEL=${OFFICIAL:-/home/spark/models/DeepSeek-V4.1-Flash-official-2cba9e42}   # OFFICIAL= the official checkpoint dir
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TAG=$(basename "$(dirname "$ORACLE")")
W=$R/runs/served_vs_harness/$TAG
OUT=${OUT:-${DSV41_RESULTS:-$(cd "$(dirname "$0")/../.." && pwd)/results}/quality/served_vs_harness_$TAG}
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$BOX" "$@"; }
die() { say "STOP: $*"; exit 1; }

[ "$(sh_ "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")" = 0 ] || die "$BOX has a GPU job"
rsync -a --exclude __pycache__ --exclude caldata -e "ssh -o BatchMode=yes" "$REPO/scripts/" "$BOX:$R/scripts/" || die "sync"
sh_ "mkdir -p $W" && scp -q "$ORACLE" "$BOX:$W/kit_oracle.jsonl" || die "oracle copy"
M="-v $MODEL:/model:ro -v $R/scripts:/s:ro -v $R/runs/p1:/p1:ro -v $R/runs/dense:/dense:ro -v $W:/w"
run() {   # $1 step name, rest: command in the image (stdout + stderr to $W/<step>.log)
  local name=$1; shift
  say "$name"
  sh_ "sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; docker rm -f dsv41-svh >/dev/null 2>&1; \
    docker run --rm --name dsv41-svh --gpus all --ipc=host -e DSV41_CUDA_MEM_FRACTION=0.70 $M --entrypoint bash $IMAGE \
    -c '$* > /w/$name.log 2>&1'" || { sh_ "tail -20 $W/$name.log" | tee -a "$OUT/run.log"; die "$name failed"; }
}
run panel "python3 /s/bench/served_vs_harness.py panel /w/kit_oracle.jsonl /w"
run ref "python3 -u /s/quantize/p4_kld.py /model /w/panel.safetensors /w/panel_index.json /w --mode ref"
run quant "python3 -u /s/quantize/p4_kld.py /model /w/panel.safetensors /w/panel_index.json /w --mode quant --tensors /p1 --dense /dense --dump-top 20"
run compare "python3 /s/bench/served_vs_harness.py compare /w/kit_oracle.jsonl /w"
for f in compare.log quant.log ref.log kld.json; do scp -q "$BOX:$W/$f" "$OUT/" 2>/dev/null; done
cat "$OUT/compare.log" | tee -a "$OUT/run.log"
say "done: $OUT"
