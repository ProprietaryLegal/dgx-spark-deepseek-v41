#!/usr/bin/env bash
# Pass 1 on both Sparks: each encodes a strided half of every layer's experts (--part i/2) and they swap partial MoE
# outputs per layer through exchange_sync.sh (host side, ConnectX link). Run from the control machine (or either Spark).
#   PLAN=~/dsv41_recipe/plans/plan.json ./run_pass1_2box.sh [first_layer-last_layer]
# Partial re-encode (only experts whose K changed; the rest from an assembled checkpoint), own run dir and token set:
#   PLAN=.../plan_v3.json OLD_PLAN=.../plan_v2_270.json REUSE=/home/spark/models/dsv41-recipe-exl3-v2.1 \
#   RUN=/home/spark/dsv41_recipe/runs/p1_v3 CALF=cal_v3.safetensors ./run_pass1_2box.sh
# Domain-weighted Hessians: P1_EXTRA="--index /cal/cal_v3_index.json --domain-weights code=1,chat=1,..." (any extra
# p4_pass1.py arguments; paths as the container sees them: /cal, /plan, /out).
set -euo pipefail
A=${A:-spark}; B=${B:-spark2}
FAB_IF=${FAB_IF:-enp1s0f0np0}   # the ConnectX port both boxes share (the launcher's default)
fab_ip() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1" || true; }   # || true: under set -e a failed ssh here would exit with no message
A_IP=${A_IP:-$(fab_ip "$A")}; B_IP=${B_IP:-$(fab_ip "$B")}
[ -n "$A_IP" ] && [ -n "$B_IP" ] || { echo "no IPv4 on $FAB_IF: $A '$A_IP', $B '$B_IP'"; exit 1; }
PLAN=${PLAN:?path of the plan json, same path on both Sparks}
LAYERS=${1:-0-39}
RUN=${RUN:-/home/spark/dsv41_recipe/runs/p1}
CAL=${CALDIR:-/home/spark/dsv41_recipe/caldata}
CALF=${CALF:-cal.safetensors}
REUSE=${REUSE:-}
OLD_PLAN=${OLD_PLAN:-}
P1_EXTRA=${P1_EXTRA:-}
if [ -n "$REUSE" ]; then
  [ -n "$OLD_PLAN" ] && [ "$(dirname "$OLD_PLAN")" = "$(dirname "$PLAN")" ] || { echo "REUSE needs OLD_PLAN next to PLAN"; exit 1; }
  RMOUNT="-v $REUSE:/reuse:ro"; RARGS="--reuse /reuse --old-plan /plan/$(basename "$OLD_PLAN")"
else
  RMOUNT=""; RARGS=""
fi
MODEL=${OFFICIAL:-/home/spark/models/DeepSeek-V4.1-Flash-official-2cba9e42}   # OFFICIAL= the official checkpoint dir
IMAGE=${IMAGE:-dsv41-quant:dev}   # any image with exllamav3 + the harness deps; must be the same image ID on both
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }
ha=$(sh_ "$A" "sha256sum $PLAN $OLD_PLAN $CAL/$CALF $REUSE/model.safetensors.index.json 2>&1 | cut -c1-16 | tr '\n' ' '")
hb=$(sh_ "$B" "sha256sum $PLAN $OLD_PLAN $CAL/$CALF $REUSE/model.safetensors.index.json 2>&1 | cut -c1-16 | tr '\n' ' '")
[ "$ha" = "$hb" ] || { echo "plan / calibration differ between boxes: $ha vs $hb"; exit 1; }
echo "plan + calibration identical on both: $ha"
ia=$(sh_ "$A" "docker image inspect -f {{.Id}} $IMAGE"); ib=$(sh_ "$B" "docker image inspect -f {{.Id}} $IMAGE")
[ "$ia" = "$ib" ] || { echo "$IMAGE differs between boxes: $ia vs $ib"; exit 1; }
echo "image $IMAGE identical on both: $ia"
for spec in "$A:0:$B_IP" "$B:1:$A_IP"; do
  IFS=: read -r n part peer <<<"$spec"
  sh_ "$n" "mkdir -p $RUN/exchange && tmux kill-session -t p1sync 2>/dev/null; \
    tmux new-session -d -s p1sync 'bash ~/dsv41_recipe/scripts/quantize/exchange_sync.sh $part $peer $RUN >> $RUN/exchange_sync.p$part.log 2>&1'"
  sh_ "$n" "docker rm -f dsv41-p1 >/dev/null 2>&1; docker run -d --name dsv41-p1 --gpus all --ipc=host --restart no \
    -e DSV41_CUDA_MEM_FRACTION=0.70 \
    -v $MODEL:/model:ro -v ~/dsv41_recipe/scripts/quantize:/q:ro -v $CAL:/cal:ro -v $(dirname "$PLAN"):/plan:ro \
    -v $RUN:/out $RMOUNT --entrypoint bash $IMAGE -c \
    'python3 -u /q/p4_pass1.py /model /cal/$CALF /plan/$(basename "$PLAN") /out --layers $LAYERS \
       --part $part/2 --exchange /out/exchange $RARGS $P1_EXTRA > /out/pass1.p$part.L$LAYERS.stdout 2>&1; \
       echo P1_EXIT=\$? >> /out/pass1.p$part.L$LAYERS.stdout' \
    >/dev/null && echo '$n part $part/2 started (stdout $RUN/pass1.p$part.L$LAYERS.stdout)'"
done
echo "watch: OUT0=pass1.p0.L$LAYERS.stdout OUT1=pass1.p1.L$LAYERS.stdout ./watch_pass1.sh"
