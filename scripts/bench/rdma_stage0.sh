#!/usr/bin/env bash
# P2 stage 0 on both Sparks: tests/rdma_ar_2node.py (ring vs rank-order sum and NCCL, 96 all-reduces in a graph) with
# the RDMA ring's flag-write mode off and on, plus STRESS all-reduces with every result checked (the ordering check
# for NIC-written flags). Run from the control machine with no serve up; prints rank 0's lines. Each run has a timeout.
#   IMAGE=dsv41-serve:v4 STRESS=1000000 scripts/bench/rdma_stage0.sh
set -uo pipefail
IMAGE=${IMAGE:-dsv41-serve:v4}; STRESS=${STRESS:-1000000}; GID=${GID:-3}; TMO=${TMO:-900}
MASTER=${MASTER:-$(ssh -o BatchMode=yes spark "ip -4 -o addr show enp1s0f0np0 | awk '{print \$4}' | cut -d/ -f1")}   # rank 0's ConnectX address
[ -n "$MASTER" ] || { echo "spark: no IPv4 on enp1s0f0np0"; exit 1; }
run() {   # $1 flag-write mode, $2 stress count
  local port=$((29610 + $1)) h r
  for spec in "spark:0" "spark2:1"; do
    h=${spec%%:*}; r=${spec#*:}
    ssh -o BatchMode=yes $h "timeout -k 30 $TMO docker run --rm --gpus all --network host --ipc host --device /dev/infiniband \
      --ulimit memlock=-1 -e RANK=$r -e MASTER_ADDR=$MASTER -e MASTER_PORT=$port -e NCCL_IB_HCA=rocep1s0f0 \
      -e NCCL_IB_GID_INDEX=$GID -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
      -e EXL3_RDMA_AR=1 -e EXL3_RDMA_FLAGW=$1 -e STRESS=$2 \
      -v ~/dsv41_recipe/patches:/opt/p:ro -v ~/.cache/dsv41-serve:/root/.cache --entrypoint python3 $IMAGE \
      /opt/p/exl3_moe/tests/rdma_ar_2node.py 2>&1 | grep -v -i -E 'warn|nccl info' | tail -14" > /tmp/stage0_$1_$r.log 2>&1 &
  done
  wait
  echo "== flag-write $1, stress $2"
  cat /tmp/stage0_$1_0.log
  grep -h -E "wrong|Error|Traceback" /tmp/stage0_$1_1.log | head -3
}
run 0 $((STRESS / 10))
run 1 "$STRESS"
