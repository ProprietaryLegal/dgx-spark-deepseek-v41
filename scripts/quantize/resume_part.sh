#!/usr/bin/env bash
# Resume one part of a 2-box pass 1 after its Spark stopped while the other part waits in wait_parts (a part that never
# publishes stalls its peer, it does not corrupt it): check the box's saved part files against the peer's, restart its
# guards / gpuwatch / exchange sync (they die with a reboot), start the part from layer N. It replays layers 0..N-1 from
# the saved part files (~105 s a layer: the same stream the waiting peer holds), and the peer continues once it publishes
# layer N. Same arguments as the run being resumed. Run from the control machine. DRY=1: the checks only, the commands printed.
#   PLAN=/home/spark/dsv41_recipe/plans/plan_v3_270.json RUN=/home/spark/dsv41_recipe/runs/p1_v3 CALF=cal_v3.safetensors \
#   IMAGE=dsv41-quant:p1 ./resume_part.sh 0 18                part 0 (spark) from layer 18
set -uo pipefail
PART=${1:?part to resume: 0 (spark) or 1 (spark2)}; L=${2:?first layer to encode}
A=${A:-spark}; B=${B:-spark2}
FAB_IF=${FAB_IF:-enp1s0f0np0}   # the ConnectX port both boxes share (the launcher's default)
fab_ip() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1" || true; }   # || true: under set -e a failed ssh here would exit with no message
A_IP=${A_IP:-$(fab_ip "$A")}; B_IP=${B_IP:-$(fab_ip "$B")}
[ -n "$A_IP" ] && [ -n "$B_IP" ] || { echo "no IPv4 on $FAB_IF: $A '$A_IP', $B '$B_IP'"; exit 1; }
PLAN=${PLAN:?plan json, same path on both Sparks}; RUN=${RUN:?run dir of the run being resumed}
CAL=${CALDIR:-/home/spark/dsv41_recipe/caldata}; CALF=${CALF:-cal.safetensors}; IMAGE=${IMAGE:-dsv41-quant:dev}
REUSE=${REUSE:-}; OLD_PLAN=${OLD_PLAN:-}; P1_EXTRA=${P1_EXTRA:-}; DRY=${DRY:-0}
MODEL=${OFFICIAL:-/home/spark/models/DeepSeek-V4.1-Flash-official-2cba9e42}   # OFFICIAL= the official checkpoint dir
if [ "$PART" = 0 ]; then H=$A; P=$B; PEER_IP=$B_IP; NAME=spark1; else H=$B; P=$A; PEER_IP=$A_IP; NAME=spark2; fi
if [ -n "$REUSE" ]; then RMOUNT="-v $REUSE:/reuse:ro"; RARGS="--reuse /reuse --old-plan /plan/$(basename "$OLD_PLAN")"
else RMOUNT=""; RARGS=""; fi
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }
lf() { printf 'layer_%02d.part%s' "$1" "$2"; }

echo "== $H: uptime, containers"
sh_ "$H" "uptime; docker ps --format '{{.Names}} {{.Status}}'" || { echo "$H unreachable"; exit 1; }
sh_ "$H" "docker ps --format '{{.Names}}' | grep -qx dsv41-p1" && { echo "dsv41-p1 is running on $H: nothing to resume"; [ "$DRY" = 1 ] || exit 1; }
echo "== part files of layers 0..$((L-1)) (names + sizes), $H vs $P"
listing="cd $RUN && ls -l layer_*.part*.safetensors layer_*.part*.json | awk '{print \$5, \$9}' | sort -k2"
a=$(sh_ "$H" "$listing"); b=$(sh_ "$P" "$listing")
below() { awk -v L="$L" '{split($2, x, "[_.]"); if (x[2] + 0 < L) print}'; }
if diff <(echo "$a" | below) <(echo "$b" | below) >/dev/null; then echo "identical below layer $L"
else echo "differ below layer $L:"; diff <(echo "$a" | below) <(echo "$b" | below) | head; exit 1; fi
n=$(echo "$a" | below | wc -l); [ "$n" -eq $(( L * 4 )) ] || { echo "$H has $n of $(( L * 4 )) files below layer $L"; exit 1; }
echo "$n files (2 parts x tensors + json per layer)"
echo "== md5 of layer $((L-1)), both parts (the last files written before the stop)"
m1=$(sh_ "$H" "cd $RUN && md5sum $(lf $((L-1)) 0).safetensors $(lf $((L-1)) 1).safetensors")
m2=$(sh_ "$P" "cd $RUN && md5sum $(lf $((L-1)) 0).safetensors $(lf $((L-1)) 1).safetensors")
[ "$m1" = "$m2" ] && echo "identical" || { echo "md5 differ:"; echo "$m1"; echo "$m2"; exit 1; }
echo "== layer $L part $PART on $H (the json is written after the tensors: json = complete, tensors alone = partial)"
f=$(lf "$L" "$PART")
st=$(sh_ "$H" "cd $RUN; if [ -f $f.json ]; then echo done; elif [ -e $f.safetensors ]; then echo partial; else echo none; fi")
case $st in
  done) echo "layer $L part $PART is complete: resume from layer $((L + 1))"; exit 1 ;;
  partial) echo "$f.safetensors without its json (a write cut by the stop): move it aside first"; exit 1 ;;
  *) echo "none" ;;
esac
GUARDS="bash ~/dsv41_recipe/scripts/quantize/guards.sh start $NAME >/dev/null; \
  tmux has-session -t gpuwatch 2>/dev/null || tmux new-session -d -s gpuwatch 'bash ~/dsv41_recipe/scripts/gpuwatch.sh'; \
  tmux kill-session -t p1sync 2>/dev/null; tmux new-session -d -s p1sync \
  'bash ~/dsv41_recipe/scripts/quantize/exchange_sync.sh $PART $PEER_IP $RUN >> $RUN/exchange_sync.p$PART.log 2>&1'; tmux ls"
START="docker rm -f dsv41-p1 >/dev/null 2>&1; docker run -d --name dsv41-p1 --gpus all --ipc=host --restart no \
  -e DSV41_CUDA_MEM_FRACTION=0.70 \
  -v $MODEL:/model:ro -v ~/dsv41_recipe/scripts/quantize:/q:ro -v $CAL:/cal:ro -v $(dirname "$PLAN"):/plan:ro \
  -v $RUN:/out $RMOUNT --entrypoint bash $IMAGE -c \
  'python3 -u /q/p4_pass1.py /model /cal/$CALF /plan/$(basename "$PLAN") /out --layers $L-39 \
     --part $PART/2 --exchange /out/exchange $RARGS $P1_EXTRA > /out/pass1.p$PART.L$L-39.stdout 2>&1; \
     echo P1_EXIT=\$? >> /out/pass1.p$PART.L$L-39.stdout' >/dev/null && date +%T"
if [ "$DRY" = 1 ]; then echo "== DRY: on $H would run:"; echo "$GUARDS"; echo "$START"; exit 0; fi
echo "== guards, gpuwatch, exchange sync on $H"
sh_ "$H" "$GUARDS"
echo "== part $PART from layer $L (stdout $RUN/pass1.p$PART.L$L-39.stdout)"
sh_ "$H" "$START"
