#!/usr/bin/env bash
# Switch a solo pass 1 (--solo-from on one Spark) back to two boxes once the other Spark is up again: wait for the solo
# run to finish its current layer (NOW=1: stop it at once and redo that layer on two boxes), stop it, mirror its tensor
# files to the returning box over the ConnectX link, and start both parts from the next layer with run_pass1_2box.sh
# (each part replays the finished layers from the saved tensors, ~65 s per layer). Run from the control machine.
#   PLAN=/home/spark/dsv41_recipe/plans/plan_v2_270.json ./rejoin_2box.sh          (SOLO=spark2 JOIN=spark by default)
# Any other run: its RUN / CALF / IMAGE (and REUSE / OLD_PLAN / P1_EXTRA) in the environment, as for run_pass1_2box.sh,
# e.g. after solo_from.sh on the v3 run: PLAN=.../plan_v3_270.json RUN=/home/spark/dsv41_recipe/runs/p1_v3 \
#   CALF=cal_v3.safetensors IMAGE=dsv41-quant:p1 ./rejoin_2box.sh
set -euo pipefail
PLAN=${PLAN:?plan json, same path on both Sparks}
SOLO=${SOLO:-spark2}; JOIN=${JOIN:-spark}
FAB_IF=${FAB_IF:-enp1s0f0np0}   # the ConnectX port both boxes share (the launcher's default)
fab_ip() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1" || true; }   # || true: under set -e a failed ssh here would exit with no message
JOIN_IP=${JOIN_IP:-$(fab_ip "$JOIN")}   # JOIN's ConnectX address
[ -n "$JOIN_IP" ] || { echo "$JOIN: no IPv4 on $FAB_IF (link down?)"; exit 1; }
IMAGE=${IMAGE:-dsv41-serve:dev}; NOW=${NOW:-0}
RUN=${RUN:-/home/spark/dsv41_recipe/runs/p1}; export RUN   # run_pass1_2box.sh reads RUN (and CALF, REUSE, ...) from the environment
HERE=$(cd "$(dirname "$0")" && pwd)
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }
last() { sh_ "$SOLO" "python3 ~/dsv41_recipe/scripts/quantize/p1_complete.py $RUN --prefix"; }   # layers 0..N complete

# 1. the fabric is back (a powered-off peer drops the direct link: no carrier, no 10.10.10.x address), the returning box is
#    idle, same image, current scripts, guards running
for h in "$SOLO" "$JOIN"; do
  sh_ "$h" "ip -4 -o addr show ${FAB_IF:-enp1s0f0np0} | grep -q 'inet '" \
    || { echo "$h: no fabric address on ${FAB_IF:-enp1s0f0np0} (link down?)"; exit 1; }
done
sh_ "$SOLO" "ping -c 2 -W 2 $JOIN_IP >/dev/null" || { echo "fabric ping $SOLO -> $JOIN_IP failed"; exit 1; }
[ "$(sh_ "$JOIN" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")" = 0 ] \
  || { echo "$JOIN has GPU processes running"; exit 1; }
[ "$(sh_ "$JOIN" "docker image inspect -f {{.Id}} $IMAGE")" = "$(sh_ "$SOLO" "docker image inspect -f {{.Id}} $IMAGE")" ] \
  || { echo "$IMAGE differs between $SOLO and $JOIN"; exit 1; }
rsync -a --exclude __pycache__ --exclude caldata "$HERE/../" "$JOIN:~/dsv41_recipe/scripts/"
rsync -a --exclude __pycache__ --exclude caldata "$HERE/../" "$SOLO:~/dsv41_recipe/scripts/"
sh_ "$JOIN" "bash ~/dsv41_recipe/scripts/quantize/guards.sh start"

# 2. stop the solo run at a layer boundary
n0=$(last)
if [ "$NOW" != 1 ]; then
  echo "$(date +%H:%M) layers 0-$n0 complete on $SOLO; waiting for the next one"
  while [ "$(last)" = "$n0" ]; do sleep 20; done
fi
N=$(last)
sh_ "$SOLO" "docker rm -f dsv41-p1 >/dev/null"
echo "$(date +%H:%M) stopped the solo run; layers 0-$N are done"

# 3. mirror the tensor files (the solo box's copy wins: its stream is the one later layers were encoded on), clear stale
#    exchange files and leftovers, drop page caches (no GPU job on either box now), check every layer 0-N is complete
sh_ "$SOLO" "rsync -a --exclude '*.tmp' --include 'layer_*.part*.safetensors' --include 'layer_*.part*.json' --exclude '*' \
  -e 'ssh -T -o Compression=no -c aes128-gcm@openssh.com' $RUN/ $JOIN_IP:$RUN/"
for h in "$SOLO" "$JOIN"; do
  sh_ "$h" "rm -f $RUN/exchange/Y_L* $RUN/*.tmp; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; \
    python3 ~/dsv41_recipe/scripts/quantize/p1_complete.py $RUN $N" || { echo "$h: pass 1 files incomplete"; exit 1; }
done

# 4. both parts from layer N+1
[ "$N" -lt 39 ] || { echo "pass 1 already complete"; exit 0; }
PLAN=$PLAN IMAGE=$IMAGE "$HERE/run_pass1_2box.sh" "$((N + 1))-39"     # part 0 on spark, part 1 on spark2, as before
