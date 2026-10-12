#!/usr/bin/env bash
# Dense pass on the Sparks: each box streams the calibration set through pass 1's EXL3 experts and encodes its share
# of the dense linears (p4_dense.py --part i/n, layer % n == i); part 0 also encodes the head (--head K) when the range
# ends at the last layer. Then each box pulls the other's files and writes the index, so both hold the full dense set
# for assembly. Run from the control machine.
#   K=5 HEAD_K=6 ./run_dense_2box.sh            (watch: ssh spark 'tail -f ~/dsv41_recipe/runs/dense/dense.part0.log')
#   ./run_dense_2box.sh finish                  (after both report DONE: exchange files + index on both)
# A range only needs pass 1's tensors up to its last layer, so the early layers can run while pass 1 finishes elsewhere:
#   PARTS="spark:0" LAYERS=0-31 ./run_dense_2box.sh                 (one box, every layer 0-31, no head)
#   LAYERS=32-39 HEAD_K=6 ./run_dense_2box.sh                       (later: both boxes split 32-39, part 0 adds the head)
# Build v3 (experts from pass 1 v3, calibration v3):
#   P1=/home/spark/dsv41_recipe/runs/p1_v3 CALF=cal_v3.safetensors OUT=/home/spark/dsv41_recipe/runs/dense_v3 \
#   IMAGE=dsv41-quant:p1 K=5 HEAD_K=6 ./run_dense_2box.sh
# A second K set from the same Hessians: OUT_ALT=.../runs/dense_v3_k4 K_ALT=4 KMAP_ALT=attn.wo_a=5 (finish does both);
# HEAD_K_ALT=5 also encodes the head at K5 into OUT_ALT (same head Hessian)
set -euo pipefail
A=${A:-spark}; B=${B:-spark2}
FAB_IF=${FAB_IF:-enp1s0f0np0}   # the ConnectX port both boxes share (the launcher's default); finish copies over it
K=${K:-5}; HEAD_K=${HEAD_K:-6}; KMAP=${KMAP:-}
OUT_ALT=${OUT_ALT:-}; K_ALT=${K_ALT:-4}; KMAP_ALT=${KMAP_ALT:-}; HEAD_K_ALT=${HEAD_K_ALT:-}
LAYERS=${LAYERS:-0-39}
PARTS=${PARTS:-"$A:0 $B:1"}         # box:part pairs; the number of pairs is the split
IMAGE=${IMAGE:-dsv41-serve:dev}
OUT=${OUT:-/home/spark/dsv41_recipe/runs/dense}   # a second encode (e.g. K6) goes to its own dir: OUT=.../runs/dense_k6
P1=${P1:-/home/spark/dsv41_recipe/runs/p1}
CAL=${CALDIR:-/home/spark/dsv41_recipe/caldata}
CALF=${CALF:-cal.safetensors}
MODEL=${OFFICIAL:-/home/spark/models/DeepSeek-V4.1-Flash-official-2cba9e42}
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }
if [ "${1:-}" = finish ]; then
  # the two boxes exchange their dense_*.safetensors over the ConnectX link (run with no GPU job on either box)
  fab_ip() { sh_ "$1" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1" || true; }
  A_IP=${A_IP:-$(fab_ip "$A")}; B_IP=${B_IP:-$(fab_ip "$B")}
  [ -n "$A_IP" ] && [ -n "$B_IP" ] || { echo "no IPv4 on $FAB_IF: $A '$A_IP', $B '$B_IP'"; exit 1; }
  for D in $OUT $OUT_ALT; do
    sh_ "$A" "rsync -rlt $D/dense_*.safetensors $D/dense_*.json $B_IP:$D/"   # -rlt: the files are root-owned (container); -a tried chgrp and set -e stopped the finish (Oct 11)
    sh_ "$B" "rsync -rlt $D/dense_*.safetensors $D/dense_*.json $A_IP:$D/"
    for n in "$A" "$B"; do
      sh_ "$n" "docker run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -v /home/spark:/home/spark --entrypoint python3 $IMAGE \
        /home/spark/dsv41_recipe/scripts/quantize/p4_dense.py --index $D 2>&1 | tail -1"
    done
  done
  exit 0
fi
# An operator hold: while any $HOLD* file exists (on the control machine), wait before starting. Lets jobs that need both GPUs (a
# same-box profile, a lever A/B) run between the post chain's profile and dense stages without editing the running chain.
HOLD=${HOLD:-$HOME/.dsv41_dense_hold}
while compgen -G "$HOLD*" >/dev/null; do sleep 30; done
L1=${LAYERS#*-}
NPARTS=$(echo $PARTS | wc -w)
boxes=$(for s in $PARTS; do echo "${s%%:*}"; done)
for n in $boxes; do   # all 384 experts of every layer the stream runs through (0..L1), from the file headers
  sh_ "$n" "python3 ~/dsv41_recipe/scripts/quantize/p1_complete.py $P1 $L1" || { echo "$n: pass 1 is incomplete"; exit 1; }
done
ids=$(for n in $boxes; do sh_ "$n" "docker image inspect -f {{.Id}} $IMAGE"; done | sort -u | wc -l)
[ "$ids" = 1 ] || { echo "$IMAGE differs between boxes"; exit 1; }
for spec in $PARTS; do
  IFS=: read -r n part <<<"$spec"
  head_arg=$([ "$part" = 0 ] && [ "$L1" = 39 ] && echo "--head $HEAD_K${OUT_ALT:+${HEAD_K_ALT:+ --head-alt $HEAD_K_ALT}}" || echo "")
  kmap_arg=$([ -n "$KMAP" ] && echo "--k-map $KMAP" || echo "")
  alt_arg=$([ -n "$OUT_ALT" ] && echo "--out-alt /out_alt --k-alt $K_ALT${KMAP_ALT:+ --k-map-alt $KMAP_ALT}" || echo "")
  alt_mount=$([ -n "$OUT_ALT" ] && echo "-v $OUT_ALT:/out_alt" || echo "")
  sh_ "$n" "mkdir -p $OUT ${OUT_ALT} && sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; docker rm -f dsv41-dense >/dev/null 2>&1; \
    docker run -d --name dsv41-dense --gpus all --ipc=host --restart no -e DSV41_CUDA_MEM_FRACTION=0.70 \
      -v $MODEL:/model:ro -v ~/dsv41_recipe/scripts/quantize:/q:ro -v $CAL:/cal:ro -v $P1:/p1:ro -v $OUT:/out $alt_mount \
      --entrypoint bash $IMAGE -c 'python3 -u /q/p4_dense.py /model /cal/$CALF /p1 /out --k $K $kmap_arg $alt_arg \
        --part $part/$NPARTS --layers $LAYERS $head_arg > /out/dense.p$part.L$LAYERS.stdout 2>&1; \
        echo DENSE_EXIT=\$? >> /out/dense.p$part.L$LAYERS.stdout; chmod -R a+rX /out ${OUT_ALT:+/out_alt}' >/dev/null \
      && echo '$n dense part $part/$NPARTS layers $LAYERS started (K $K${head_arg:+, head K$HEAD_K})'"
done
