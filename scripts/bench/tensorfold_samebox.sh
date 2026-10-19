#!/usr/bin/env bash
# Same-box baseline: bertholomus's TensorFold DeepSeek-V4.1 TP2 engine (github.com/bertholomus/TensorFold, branch
# deepseek-v41-tp2, v0.5.1 = 90b68e0, Apache-2.0) on Mia-AiLab's EXL3 2.9 bpw pack, on our two Sparks, with his
# published serve settings (deepseek-v4.1-tensorfold-tp2-2xgb10 README, "Run it (v0.1.0-v0.5.1)"), measured with the same
# kit_bench.py we use (byte-identical to his tools/dsv41/kit_bench.py). We run his engine to measure it; none of its
# code goes into our recipe. Run from the control machine with nothing else on either GPU:
#   ./tensorfold_samebox.sh image     # copy the base image to the worker if missing (docker save | ssh docker load)
#   ./tensorfold_samebox.sh up        # start rank 0 (head), wait for its store port, start rank 1, wait for the API
#   ./tensorfold_samebox.sh status | logs [head|worker] | down
#   NSYS=1 ./tensorfold_samebox.sh up        # under Nsight Systems (the host's, mounted): nothing captured yet
#   ./tensorfold_samebox.sh profile [tag]    # per prompt: both ranks around one streamed request -> $NSYS_OUT
# Knobs: HEAD WORKER IMAGE FORK MODEL ENGRAM CACHE PORT CONTEXT PARALLEL DRAFTS HCAS GID PATH_MTU
# TF_ENV="TF_DS_CONF_LOG=1 ...": extra engine env (e.g. its DSpark acceptance report every 200 rounds)
set -euo pipefail
HEAD=${HEAD:-spark}; WORKER=${WORKER:-spark2}
FAB_IF=${FAB_IF:-enp1s0f0np0}; HCAS=${HCAS:-rocep1s0f0}; GID_HCA=${GID_HCA:-rocep1s0f0}   # one port: the second PCIe function has no IPv4 GID at index 3 (measured Oct 10 and 11)
IMAGE=${IMAGE:-tensorfold:0.6.6-cb2ebf0}             # nvcr.io/nvidia/pytorch:26.07-py3 + TensorFold 0.6.6 (his v0.5 base)
FORK=${FORK:-/home/spark/dsv41_recipe/tensorfold_fork}
MODEL=${MODEL:-/home/spark/models/Mia-DeepSeek-V4.1-Flash-EXL3-2.9bpw-64ba41b6}
ENGRAM=${ENGRAM:-/home/spark/models/DeepSeek-V4.1-Flash-official-2cba9e42}   # DeepSeek's shards 47 / 48
CACHE=${CACHE:-/home/spark/dsv41_recipe/tf_rank_cache}  # ~106 GB per rank, written on the first start
PORT=${PORT:-18891}; CONTEXT=${CONTEXT:-1048576}; PARALLEL=${PARALLEL:-4}; DRAFTS=${DRAFTS:-5}
NAME=tf-dsv41
NSYS_DIR=${NSYS_DIR:-/opt/nvidia/nsight-systems/2025.3.2}       # the Sparks' host install, mounted read-only
NSYS_OUT=${NSYS_OUT:-/home/spark/dsv41_recipe/tf_nsys}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cmd=${1:-status}

sh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }
fab_ip() { sh_ "$1" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1"; }
gid_of() {   # RoCE v2 GID index whose IPv4-mapped GID is this node's fabric address (same rule as dsv41_tp2.sh)
  sh_ "$1" "want=\$(ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1); P=/sys/class/infiniband/$GID_HCA/ports/1
    hex=\$(printf '%02x%02x%02x%02x' \$(echo \$want | tr '.' ' '))
    for i in \$(seq 0 15); do t=\$(cat \$P/gid_attrs/types/\$i 2>/dev/null); g=\$(cat \$P/gids/\$i 2>/dev/null)
      case \"\$t\" in *v2*) ;; *) continue;; esac
      case \"\$g\" in *ffff:\${hex:0:4}:\${hex:4:4}*) echo \$i; break;; esac; done"
}

launch() {   # $1 node, $2 rank, $3 master (head fabric ip), $4 gid index
  local n=$1 rank=$2 master=$3 gid=$4
  local mnt="" pre=""
  if [ "${NSYS:-0}" = 1 ]; then      # nsys launch: profiling starts only on 'profile' (nsys start / stop in the container)
    mnt="-v $NSYS_DIR:$NSYS_DIR:ro -v $NSYS_OUT:$NSYS_OUT"
    pre="$NSYS_DIR/target-linux-sbsa-armv8/nsys launch --session-new=tf --trace=cuda,nvtx --cuda-graph-trace=node"   # sampling / context switches: set on start
  fi
  sh_ "$n" "docker rm -f $NAME >/dev/null 2>&1; mkdir -p $CACHE $NSYS_OUT
    docker run -d --name $NAME --network host --ipc host --shm-size 64g --gpus all --device /dev/infiniband $mnt \
      --ulimit memlock=-1 --ulimit stack=67108864 --restart no \
      -v $FORK:/opt/tf_fork -v $MODEL:$MODEL:ro -v $ENGRAM:$ENGRAM:ro -v $CACHE:$CACHE \
      -e TF_DS_REPLAY=1 -e TF_DS_PREFILL_CHUNK=2048 -e TF_DS_RANK_CACHE=$CACHE -e TF_DS_RANK_CACHE_READERS=32 \
      -e TF_DS_WARM_LENGTHS=1,17,33,131,514,1024,2113 -e TF_DS_ENGRAM=$ENGRAM -e TF_RDMA_DEVICES=$HCAS \
      -e NCCL_IB_HCA=$HCAS -e NCCL_IB_GID_INDEX=$gid -e NCCL_SOCKET_IFNAME=$FAB_IF -e GLOO_SOCKET_IFNAME=$FAB_IF \
      $(for kv in ${TF_ENV:-}; do printf -- '-e %s ' "$kv"; done) \
      --entrypoint bash $IMAGE -c 'pip install --no-deps --no-build-isolation -q -e /opt/tf_fork && \
        exec $pre tensorfold serve $MODEL --tp 2 --rank $rank --master $master --host 0.0.0.0 --port $PORT \
          --context $CONTEXT --vision --parallel $PARALLEL --mtp-drafts $DRAFTS' >/dev/null && echo '$n rank $rank started'"
}

case "$cmd" in
  image)
    if sh_ "$WORKER" "docker image inspect $IMAGE >/dev/null 2>&1"; then echo "$WORKER has $IMAGE"; exit 0; fi
    echo "copying $IMAGE $HEAD -> $WORKER $(date +%H:%M:%S)"
    # node to node (the head's ssh alias for the worker), not through this machine
    sh_ "$HEAD" "docker save $IMAGE | ssh -o BatchMode=yes $WORKER docker load" | tail -1
    echo "done $(date +%H:%M:%S)"
    ;;
  up)
    for n in "$HEAD" "$WORKER"; do
      [ "$(sh_ "$n" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")" = 0 ] || { echo "$n has GPU processes"; exit 1; }
      sh_ "$n" "docker image inspect $IMAGE >/dev/null" || { echo "$n lacks $IMAGE (run: $0 image)"; exit 1; }
    done
    for n in "$HEAD" "$WORKER"; do
      rsync -a --delete --exclude .git --exclude __pycache__ -e "ssh -o BatchMode=yes" "$REPO/../benchmarks/tensorfold_v41/tensorfold_fork/" "$n:$FORK/" 2>/dev/null \
        || rsync -a --delete --exclude .git --exclude __pycache__ -e "ssh -o BatchMode=yes" "$HOME/Projects/benchmarks/tensorfold_v41/tensorfold_fork/" "$n:$FORK/"
      # Network adaptation, measurement copy only: his RDMA gather sets a 4096-byte path MTU (jumbo frames on his link);
      # our fabric NICs run MTU 1500 (RoCE active MTU 1024), where 4096 drops every packet (transport retry exceeded,
      # seen with our port of the same ring, Oct 10). PATH_MTU=4096 keeps his value (after an MTU-9000 setup).
      sh_ "$n" "sed -i 's/rtr.path_mtu = IBV_MTU_4096;/rtr.path_mtu = IBV_MTU_${PATH_MTU:-1024};/' $FORK/src/tensorfold/cuda/rdma_gather.cu && grep -c 'IBV_MTU_${PATH_MTU:-1024};' $FORK/src/tensorfold/cuda/rdma_gather.cu"
    done
    master=$(fab_ip "$HEAD"); g0=${GID:-$(gid_of "$HEAD")}; g1=${GID:-$(gid_of "$WORKER")}
    # GID=1 (RoCE v2 link-local) for both PCIe functions of the port: only rocep1s0f0 has the IPv4 GID (index 3) here
    echo "master $master gid $g0 / $g1 devices $HCAS"
    [ -n "$master" ] && [ -n "$g0" ] && [ -n "$g1" ] || { echo "fabric ip / gid lookup failed"; exit 1; }
    launch "$HEAD" 0 "$master" "$g0"
    echo "waiting for rank 0's store port 29551"
    for i in $(seq 1 120); do sh_ "$HEAD" "ss -ltn | grep -q ':29551 '" && break; sleep 5; done
    launch "$WORKER" 1 "$master" "$g1"
    t0=$(date +%s)
    until sh_ "$HEAD" "curl -sf localhost:$PORT/v1/models >/dev/null"; do
      for n in "$HEAD" "$WORKER"; do
        [ "$(sh_ "$n" "docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null")" = true ] \
          || { echo "$n: $NAME stopped"; sh_ "$n" "docker logs --tail 30 $NAME 2>&1"; exit 1; }
      done
      sleep 15
    done
    echo "ready after $(( $(date +%s) - t0 )) s"
    for n in "$HEAD" "$WORKER"; do sh_ "$n" "echo \$(hostname): \$(free -g | awk '/Mem:/{print \"used \"\$3\" GiB, available \"\$7\" GiB\"}')"; done
    ;;
  profile)   # one capture per prompt (PROMPTS, default code prose structured) around one streamed TOKENS request after
              # a warm-up, both ranks; the client line (chunks = rounds, ttft, decode seconds) goes to <tag>_<prompt>.json
              # and each rank's kernel trace to CSV (nsys stats cuda_gpu_trace) next to its report
    tag=${2:-samebox}
    N=$NSYS_DIR/target-linux-sbsa-armv8/nsys
    scp -q "$REPO/scripts/bench/prof_stream.py" "$HEAD:/tmp/prof_stream.py"
    sh_ "$HEAD" "test -f /tmp/kit_bench.py" || scp -q "${KIT:-$HOME/Projects/benchmarks/kit_bench/kit_bench.py}" "$HEAD:/tmp/kit_bench.py"
    cl="python3 /tmp/prof_stream.py --base http://localhost:$PORT --model dsv41"
    for p in ${PROMPTS:-code prose structured}; do
      sh_ "$HEAD" "$cl --prompt $p --tokens 32" >/dev/null
      for n in "$HEAD" "$WORKER"; do
        sh_ "$n" "docker exec $NAME $N start --session=tf --sample=none --cpuctxsw=none \
          -o $NSYS_OUT/tf_\$(hostname)_${tag}_$p --force-overwrite=true" &
      done
      wait
      sh_ "$HEAD" "$cl --prompt $p --tokens ${TOKENS:-384} | tee $NSYS_OUT/${tag}_$p.json"
      for n in "$HEAD" "$WORKER"; do sh_ "$n" "docker exec $NAME $N stop --session=tf >/dev/null" & done
      wait
    done
    for n in "$HEAD" "$WORKER"; do
      sh_ "$n" "cd $NSYS_OUT && for f in tf_*_${tag}_*.nsys-rep; do $N stats --report cuda_gpu_trace --format csv \
        --force-export=true -o \${f%.nsys-rep} \$f >/dev/null 2>&1 || echo \"\$f: stats failed\"; done; ls -la | grep $tag" &
    done
    wait
    ;;
  status) for n in "$HEAD" "$WORKER"; do sh_ "$n" "docker ps --filter name=$NAME --format '{{.Names}} {{.Status}}'" | sed "s/^/$n: /"; done ;;
  logs) n=${2:-head}; [ "$n" = worker ] && n=$WORKER || n=$HEAD; sh_ "$n" "docker logs --tail 60 $NAME 2>&1" ;;
  down) for n in "$WORKER" "$HEAD"; do sh_ "$n" "docker rm -f $NAME >/dev/null 2>&1; echo \$(hostname) down"; done ;;
  *) echo "usage: $0 image|up|status|logs|down|profile"; exit 1 ;;
esac
