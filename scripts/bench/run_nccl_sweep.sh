#!/usr/bin/env bash
# NCCL all-reduce sweep across the two Sparks (both GPUs must be free of long jobs: run between passes, not during).
# Same fabric settings as scripts/launch/dsv41_tp2.sh (GID index derived per node from its fabric IPv4); each case
# varies NCCL_PROTO / NCCL_ALGO / channels / GPUDirect RDMA (GB10 memory is system memory: if NCCL stages through host
# bounce buffers, forcing GDR may cut the ~52 us decode all-reduce).
#   scripts/bench/run_nccl_sweep.sh [rows]        -> logs/nccl_sweep_<case>.log on spark
set -uo pipefail
HEAD=${HEAD:-spark}; WORKER=${WORKER:-spark2}
FAB_IF=${FAB_IF:-enp1s0f0np0}; FAB_HCA=${FAB_HCA:-rocep1s0f0}
IMAGE=${IMAGE:-dsv41-serve:dev}
ROWS=${1:-6,512,2048}
CASES=${CASES:-"default PROTO=LL PROTO=LL128 PROTO=Simple PROTO=Simple,ALGO=Tree PROTO=LL,MAX_NCHANNELS=1 PROTO=LL,MAX_NCHANNELS=2 PROTO=Simple,MIN_NCHANNELS=4 NET_GDR_LEVEL=SYS PROTO=LL,NET_GDR_LEVEL=SYS,NET_GDR_READ=1 PROTO=LL,NET_GDR_LEVEL=SYS,MAX_NCHANNELS=1"}
gid_of() {   # RoCE v2 GID index whose IPv4-mapped GID is this node's fabric address (as dsv41_tp2.sh)
  ssh -o BatchMode=yes "$1" "want=\$(ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1); P=/sys/class/infiniband/$FAB_HCA/ports/1
    hex=\$(printf '%02x%02x%02x%02x' \$(echo \$want | tr '.' ' '))
    for i in \$(seq 0 15); do t=\$(cat \$P/gid_attrs/types/\$i 2>/dev/null); g=\$(cat \$P/gids/\$i 2>/dev/null)
      case \"\$t\" in *v2*) ;; *) continue;; esac
      case \"\$g\" in *ffff:\${hex:0:4}:\${hex:4:4}*) echo \$i; break;; esac; done"
}
HG=$(gid_of "$HEAD"); WG=$(gid_of "$WORKER")
[ -n "$HG" ] && [ -n "$WG" ] || { echo "no RoCE v2 GID for the fabric address: head '$HG' worker '$WG'"; exit 1; }
echo "GID index: $HEAD $HG, $WORKER $WG"
MIP=$(ssh -o BatchMode=yes "$HEAD" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1")   # rank 0's fabric address
[ -n "$MIP" ] || { echo "$HEAD: no IPv4 on $FAB_IF"; exit 1; }
run() {   # $1 node $2 rank $3 extra env (NCCL_ prefixed already) $4 GID index
  ssh -o BatchMode=yes "$1" "docker rm -f nccl-bench >/dev/null 2>&1; docker run --rm --name nccl-bench --network host --ipc host \
    --gpus all --device /dev/infiniband --ulimit memlock=-1 -v ~/dsv41_recipe/scripts/bench:/b:ro \
    -e NCCL_IB_HCA=$FAB_HCA -e NCCL_IB_GID_INDEX=$4 -e NCCL_IB_DISABLE=0 -e NCCL_NET=IB -e NCCL_SOCKET_IFNAME=$FAB_IF \
    -e GLOO_SOCKET_IFNAME=$FAB_IF -e RANK=$2 -e WORLD_SIZE=2 -e MASTER_ADDR=$MIP -e MASTER_PORT=29511 $3 \
    --entrypoint python3 $IMAGE /b/nccl_allreduce_bench.py --rows $ROWS 2>&1 | grep -v -i warn"
}
for c in $CASES; do
  envs=""
  [ "$c" != default ] && for kv in ${c//,/ }; do envs="$envs -e NCCL_$kv"; done
  echo "== $c"
  run "$WORKER" 1 "$envs" "$WG" >/dev/null &
  sleep 2
  run "$HEAD" 0 "$envs" "$HG"
  wait
done
