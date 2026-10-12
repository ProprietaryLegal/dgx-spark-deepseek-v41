#!/usr/bin/env bash
# Continue a 2-box pass 1 on one Spark when the other stopped and cannot be powered on soon. This box finishes its half of
# the first incomplete layer N (waits for it), then its waiting part is stopped and one process replays 0..N-1 from the
# saved part files, encodes the dead box's half of layer N against this box's half, and every expert of layers N+1..39
# (--solo-from): ~37 min a layer instead of ~19. Refuses while the other box answers on the fabric (two writers of one
# half). rejoin_2box.sh switches back to two boxes once the other Spark is up. Run from the control machine. DRY=1: checks only.
#   PLAN=/home/spark/dsv41_recipe/plans/plan_v3_270.json RUN=/home/spark/dsv41_recipe/runs/p1_v3 CALF=cal_v3.safetensors \
#   IMAGE=dsv41-quant:p1 DEAD_IP=<its ConnectX address> ./solo_from.sh      (ON=spark2, the dead box's part DEAD_PART=0
#   by default)
#   then: SOLO=spark2 OUT=pass1.solo.L<N>-39.stdout WAKE_ON_UP=spark ./watch_pass1.sh   (the run logs as part 0)
set -uo pipefail
ON=${ON:-spark2}; DEAD_PART=${DEAD_PART:-0}; DEAD_IP=${DEAD_IP:?the ConnectX address of the stopped box}
PLAN=${PLAN:?plan json}; RUN=${RUN:?run dir}; CAL=${CALDIR:-/home/spark/dsv41_recipe/caldata}; CALF=${CALF:-cal.safetensors}
IMAGE=${IMAGE:-dsv41-quant:dev}; REUSE=${REUSE:-}; OLD_PLAN=${OLD_PLAN:-}; P1_EXTRA=${P1_EXTRA:-}; DRY=${DRY:-0}
MODEL=${OFFICIAL:-/home/spark/models/DeepSeek-V4.1-Flash-official-2cba9e42}   # OFFICIAL= the official checkpoint dir
if [ -n "$REUSE" ]; then RMOUNT="-v $REUSE:/reuse:ro"; RARGS="--reuse /reuse --old-plan /plan/$(basename "$OLD_PLAN")"
else RMOUNT=""; RARGS=""; fi
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }
[ "$DEAD_PART" = 0 ] || { echo "the watcher's SOLO mode reads part 0's log: DEAD_PART=1 needs it extended first"; exit 1; }
if sh_ "$ON" "ping -c 2 -W 2 $DEAD_IP >/dev/null"; then
  echo "$DEAD_IP answers on the fabric: the other box is up (resume its part with resume_part.sh instead)"
  [ "$DRY" = 1 ] || exit 1
fi
N=$(( $(sh_ "$ON" "python3 ~/dsv41_recipe/scripts/quantize/p1_complete.py $RUN --prefix") + 1 ))
[ "$N" -le 39 ] || { echo "pass 1 already complete"; exit 0; }
live=$(printf 'layer_%02d.part%s' "$N" $(( 1 - DEAD_PART ))); dead=$(printf 'layer_%02d.part%s' "$N" "$DEAD_PART")
echo "layers 0-$((N - 1)) complete on $ON; layer $N: waiting for $live.json (this box's half)"
until sh_ "$ON" "test -f $RUN/$live.json"; do [ "$DRY" = 1 ] && { echo "DRY: not there yet"; break; }; sleep 20; done
st=$(sh_ "$ON" "cd $RUN; if [ -f $dead.json ]; then echo done; elif [ -e $dead.safetensors ]; then echo partial; else echo none; fi")
[ "$st" = none ] || { echo "$ON already holds $dead ($st): look at it first"; exit 1; }
START="docker rm -f dsv41-p1 >/dev/null 2>&1; docker run -d --name dsv41-p1 --gpus all --ipc=host --restart no \
  -e DSV41_CUDA_MEM_FRACTION=0.70 \
  -v $MODEL:/model:ro -v ~/dsv41_recipe/scripts/quantize:/q:ro -v $CAL:/cal:ro -v $(dirname "$PLAN"):/plan:ro \
  -v $RUN:/out $RMOUNT --entrypoint bash $IMAGE -c \
  'python3 -u /q/p4_pass1.py /model /cal/$CALF /plan/$(basename "$PLAN") /out --layers $N-39 \
     --part $DEAD_PART/2 --exchange /out/exchange --solo-from $((N + 1)) $RARGS $P1_EXTRA > /out/pass1.solo.L$N-39.stdout 2>&1; \
     echo P1_EXIT=\$? >> /out/pass1.solo.L$N-39.stdout' >/dev/null && date +%T"
if [ "$DRY" = 1 ]; then echo "== DRY: on $ON would run:"; echo "$START"; exit 0; fi
echo "== stopping $ON's waiting part, starting the solo run from layer $N (stdout $RUN/pass1.solo.L$N-39.stdout)"
sh_ "$ON" "$START"
echo "watch: SOLO=$ON OUT=pass1.solo.L$N-39.stdout WAKE_ON_UP=spark RUN=${RUN#/home/spark/} ./watch_pass1.sh"
