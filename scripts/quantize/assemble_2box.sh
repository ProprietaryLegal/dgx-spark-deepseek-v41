#!/usr/bin/env bash
# Assemble the served checkpoint on both Sparks after pass 1 (each box holds every part's tensors: exchange_sync.sh
# mirrors them). /home/spark is mounted once at the same path, so the official Engram shards are hardlinked rather than
# copied (two bind mounts = two mount points = EXDEV, a 203 GB copy), and the container runs as spark so the files are not
# root-owned. Run from the control machine.
#   PLAN=/home/spark/dsv41_recipe/plans/plan_v2_270.json OUT=/home/spark/models/dsv41-recipe-exl3-v2 [DENSE=.../runs/dense] ./assemble_2box.sh
set -euo pipefail
PLAN=${PLAN:?plan json, same path on both Sparks}
OUT=${OUT:?output dir, same path on both Sparks}
P1=${P1:-/home/spark/dsv41_recipe/runs/p1}
MODEL=${MODEL:-/home/spark/models/DeepSeek-V4.1-Flash-official-2cba9e42}
IMAGE=${IMAGE:-dsv41-quant:dev}
DENSE=${DENSE:-}            # p4_dense.py output dir (EXL3 dense linears + head), same path on both Sparks, or empty
for n in ${BOXES:-spark spark2}; do
  ssh -o BatchMode=yes "$n" "set -e; test ! -e $OUT || { echo '$n: $OUT exists, not overwriting'; exit 1; }; \
    [ \"\$(docker inspect -f {{.State.Running}} dsv41-assemble 2>/dev/null)\" = true ] && { echo '$n: an assembly is running'; exit 1; }; \
    docker rm dsv41-assemble >/dev/null 2>&1 || true; \
    n_parts=\$(ls $P1/layer_*.part*.safetensors | wc -l); echo \"$n: \$n_parts part files\"; \
    docker run -d --name dsv41-assemble --restart no --user \$(id -u):\$(id -g) -e HOME=/tmp \
      -v /home/spark:/home/spark --entrypoint bash $IMAGE -c \
      'python3 -u /home/spark/dsv41_recipe/scripts/quantize/p4_assemble.py $MODEL $P1 $PLAN $OUT ${DENSE:+--dense $DENSE} \
         > $OUT.assemble.log 2>&1; echo ASM_EXIT=\$? >> $OUT.assemble.log' >/dev/null && echo '$n: assembling into $OUT'"
done
echo "watch: ssh spark 'tail -f $OUT.assemble.log'"
