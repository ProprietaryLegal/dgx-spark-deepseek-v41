#!/usr/bin/env bash
# Two-node TP=2 serve of the DeepSeek-V4.1-Flash recipe on 2x DGX Spark (GB10), over RoCE v2 on the ConnectX link.
# Modeled on sparkctl's launch: one container per node, vLLM's MP backend across nodes (--nnodes 2,
# the worker headless), NCCL pinned to the RoCE HCA that carries the fabric IPv4 with the GID index derived from that
# IPv4 (auto-detect hands NCCL the port without an IP and dies), --device /dev/infiniband instead of --privileged,
# --restart no (docker restarting one rank of a TP pair yields a head that can never rejoin).
#
#   MODEL_DIR=/home/spark/models/<ours> ./dsv41_tp2.sh up        start both ranks, wait for /v1/models
#   LOAD_FORMAT=dummy MODEL_DIR=... ./dsv41_tp2.sh up          G0 fit / speed probe without weights
#   ./dsv41_tp2.sh status | logs [head|worker] | down
# Knobs (env): IMAGE SERVED PORT MAX_LEN MAX_SEQS MAX_BATCH UTIL KV_BYTES SPEC (JSON or empty = off) GRAPH KERNEL_CONFIG TEXT_ONLY EXTRA HEAD WORKER
#   DOCKER_ENV="VLLM_EXL3_HEAD=mxfp8 ..." (extra container env, same on both ranks)
#   INDEXER_LOGITS_MB=128 EXPANDABLE_SEGMENTS=1 (opt-in memory knobs, both off by default; see "Memory" below)
set -euo pipefail
HEAD=${HEAD:-spark}; WORKER=${WORKER:-spark2}
FAB_IF=${FAB_IF:-enp1s0f0np0}; FAB_HCA=${FAB_HCA:-rocep1s0f0}
IMAGE=${IMAGE:-dsv41-serve:v5}   # the release image: patches 0001-0008, 0010-0021 and 0023-0038 + the plugin (README quickstart)
SERVED=${SERVED:-dsv41}; PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-262144}; MAX_SEQS=${MAX_SEQS:-2}; MAX_BATCH=${MAX_BATCH:-2048}; UTIL=${UTIL:-0.88}
# KV pool pinned per rank instead of sized from UTIL: 2.5 GiB, as Mia-AiLab's kit ships (KV_CACHE_MEMORY_BYTES, her README:
# ~774K tokens of fp8_ds_mla per rank), so both serves get the same pool; KV_BYTES= (empty) sizes it from UTIL again
KV_BYTES=${KV_BYTES-2684354560}
SPEC=${SPEC-'{"method":"dspark","num_speculative_tokens":5}'}
GRAPH=${GRAPH:-'{"cudagraph_mode":"FULL_DECODE_ONLY"}'}
# 256 reader threads: with 32, Engram disk reads stalled a 512-token prefill step ~74 ms (dummy-boot profile, Oct 8);
# scripts/bench/engram_reader_bench.py, cold rows: 4,096 rows 80.2 ms at 32 threads, 5.8 ms at 256 (16,384: 318 -> 20.4 ms).
ENGRAM=${ENGRAM:-'{"cpu_offload":true,"dp_shared_memory":false,"disk_offload":true,"disk_threads":256}'}
LOAD_FORMAT=${LOAD_FORMAT:-auto}
# FlashInfer autotune off: on GB10 every new kernel shape grows driver-side memory; with autotune on, a dummy-weight
# boot went from 4-5 to 0.8 GiB MemAvailable in about 5 minutes of mixed requests
KERNEL_CONFIG=${KERNEL_CONFIG:-'{"enable_flashinfer_autotune":false}'}
# Text-only by default: an image limit of 0 makes vLLM stub out the V4.1 vision tower and skip its weights (~0.9 GiB
# per rank, which the 262K KV cache needs more). TEXT_ONLY=0 serves images.
TEXT_ONLY=${TEXT_ONLY:-1}
# SERVE_CPUS=0-18 (default): the serve's processes start on those cores (taskset), so the RDMA ring's proxy thread, which
# pins itself to EXL3_RDMA_CPU (default the last core, 19: a Cortex-X925), keeps its core to itself. SERVE_CPUS= (empty):
# no taskset.
SERVE_CPUS=${SERVE_CPUS-0-18}
EXTRA=${EXTRA:-}
# The measured-best serve settings (BENCHMARKS.md, RECIPE.md): our EXL3 MoE knobs, DSpark per-round depth from the
# confidence head with the measured round-cost table (patch 0010; ms by depth 0-5, drafter included, measured with the
# ring and the prefetch on), Markov bias rows cached (patch 0011; the token list is copied to each node's cache below),
# the paced L2 prefetch and the RDMA all-reduce ring (plugin), and the levers picked since: the local draft argmax
# (0012), smoe run flags (evict-first is compiled in), mHC overlap (0014), the Engram spin reader (0015), short-context
# graph twins (0019), our router with its gate in L2, exl3_mrow's grouped launch, and since v1.1 TensorFold's grouped
# EXL3 prompt kernels for the routed experts (Apache-2.0, plugin csrc/tf_experts/: prefill +18-20%, decode unchanged)
# with the JIT builds kept in the mounted cache (no recompile on every boot). Since v2.0 also: the critical-first L2
# prefetch order, the greedy verify's local argmax (0027), the drafter's own head (0026, when the pack has one; else the
# shared head), the heads on exl3_mrow, Engram's wkv row-parallel (0028), the row GEMM for the attention inputs (0029),
# the drafter's main_proj row-parallel (0030), the fused Markov step (0031), the drafter's FP8 bmm groups on streams
# (0032), its context KV in one launch (0033), the draft layers' prefetch chain, the fused depth rule (0038), every SWA
# metadata build in one launch (0037), all-reduces up to 1 MB on the ring, and the conf+feat depth policy (0036: the
# confidence head's logit plus each draft's margin and top-1 probability, through a head fitted to the pack's own
# target, `dspark_conf_feat_head.npz` in the model dir, see below). The round-cost table is the one measured on the
# v2.0 weights with all of this on (scripts/bench/depth_sweep.sh at fixed depths, Oct 21); re-measure it after changing
# weights, image or knobs. RECIPE_ENV= (empty) runs without them; DOCKER_ENV comes after it, so a knob there overrides
# the same knob here.
RECIPE_ENV=${RECIPE_ENV-"VLLM_EXL3_MOE_PARALLEL_GROUPS=1 EXL3_MOE_COOP_WIDE=1 EXL3_INT8_GEMV=0 VLLM_DSPARK_DEPTH_POLICY=conf+feat VLLM_DSPARK_ROUND_MS=27.6,31.9,35.6,38.9,42.2,44.8 VLLM_DSPARK_DRAFT_MS=0 VLLM_DSPARK_MARKOV_CACHE=/root/.cache/markov_top_tokens.json EXL3_L2PF=attn,moe,next EXL3_RDMA_AR=1 VLLM_SHORT_CTX_BOUNDS=512,1025 EXL3_ROUTER=1 EXL3_L2PF_HC=1 VLLM_DSPARK_LOCAL_ARGMAX=1 EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1 VLLM_DSV41_MHC_OVERLAP=1 VLLM_ENGRAM_DISK_SPIN_MS=1000 EXL3_MROW_GROUPED=1 VLLM_EXL3_MOE_TF_PROMPT=1 TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions EXL3_L2PF_ORDER=1 VLLM_GREEDY_LOCAL_ARGMAX=1 VLLM_DSPARK_OWN_HEAD=1 VLLM_EXL3_HEAD_MROW=1 VLLM_ENGRAM_WKV_ROWPAR=1 EXL3_RDMA_AR_MAX_KB=1024 EXL3_ROWMM=1 VLLM_DSPARK_MAIN_PROJ_ROWPAR=1 VLLM_DSPARK_MARKOV_FUSED=1 VLLM_SM12X_BMM_STREAMS=1 VLLM_DSPARK_CONTEXT_KV_GROUPED=1 EXL3_L2PF_DRAFTER=1 VLLM_DSPARK_DEPTH_FUSED=1 VLLM_DSV41_SWA_BATCH=1"}
# Memory, opt-in: both knobs off by default, effects per rank at MAX_LEN 262144 / MAX_BATCH 2048 estimated from the
# code, not measured. INDEXER_LOGITS_MB=128 sets VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (vLLM's default 512), the largest
# logits block the sparse indexer takes in a prefill: a 2,048-token chunk reaches 512 MiB at 64K of context and the
# allocator keeps that block, so up to 384 MiB less held after long prompts, for 4x the logits / top-k launches at 256K
# (16 a chunk instead of 4). EXPANDABLE_SEGMENTS=1 sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: PyTorch grows
# one mapped segment instead of caching separate cudaMalloc blocks, so long prompts leave less split slack reserved
# (HeNryous measured +1.3 GiB free at a 314K-token peak); it maps 2 MiB pages as it grows, also while serving, which can
# stall in compaction on a box with fragmented free memory (FINDINGS 22: compact before boot). Our pinned buffers (the
# plugin's load staging, the RDMA rings, Engram's row buffers) are host memory and do not move with it. Both land before
# DOCKER_ENV, which still overrides them.
# Patches 0024 / 0025 (RoPE tables to MAX_LEN + 2048 rows, indexer prefill buffer 8 x MAX_LEN) work in an image built
# with them; DOCKER_ENV="VLLM_DSV41_ROPE_CAP=0 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=40" restores upstream's sizes. Not a
# knob: each Spark's kdump reservation (crashkernel= on the kernel command line; HeNryous: +2.1 GiB a node without it)
# is a boot-parameter change and the operator's call; without it a kernel panic leaves no crash dump.
INDEXER_LOGITS_MB=${INDEXER_LOGITS_MB:-}; EXPANDABLE_SEGMENTS=${EXPANDABLE_SEGMENTS:-0}
MEM_ENV=""
[ -z "$INDEXER_LOGITS_MB" ] || MEM_ENV="VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=$INDEXER_LOGITS_MB"
[ "$EXPANDABLE_SEGMENTS" != 1 ] || MEM_ENV="${MEM_ENV:+$MEM_ENV }PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
DOCKER_ENV=${DOCKER_ENV:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
NAME=dsv41-serve
cmd=${1:-status}

sh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }
fab_ip() { sh_ "$1" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1"; }
gid_of() {   # RoCE v2 GID index whose IPv4-mapped GID is this node's fabric address
  sh_ "$1" "want=\$(ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1); P=/sys/class/infiniband/$FAB_HCA/ports/1
    hex=\$(printf '%02x%02x%02x%02x' \$(echo \$want | tr '.' ' '))
    for i in \$(seq 0 15); do t=\$(cat \$P/gid_attrs/types/\$i 2>/dev/null); g=\$(cat \$P/gids/\$i 2>/dev/null)
      case \"\$t\" in *v2*) ;; *) continue;; esac
      case \"\$g\" in *ffff:\${hex:0:4}:\${hex:4:4}*) echo \$i; break;; esac; done"
}

launch() {   # $1 node  $2 rank  $3 master ip  $4 self ip  $5 gid
  local n=$1 rank=$2 master=$3 self=$4 gid=$5 tmp envs="" kv
  for kv in $RECIPE_ENV $MEM_ENV $DOCKER_ENV; do envs="$envs -e $kv"; done
  tmp=$(mktemp)
  {
    echo '#!/usr/bin/env bash'
    echo 'set -e'
    echo "exec ${SERVE_CPUS:+taskset -c $SERVE_CPUS }vllm serve '$MODEL_DIR' --served-model-name '$SERVED' --host 0.0.0.0 --port $PORT \\"
    echo "  --tensor-parallel-size 2 --nnodes 2 --node-rank $rank --master-addr $master --master-port 25000 \\"
    echo "  --gpu-memory-utilization $UTIL --max-model-len $MAX_LEN --max-num-seqs $MAX_SEQS \\"
    echo "  --max-num-batched-tokens $MAX_BATCH --load-format $LOAD_FORMAT \\"
    echo "  --engram-config '$ENGRAM' --compilation-config '$GRAPH' --kernel-config '$KERNEL_CONFIG' \\"
    echo "  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \\"
    [ -n "$SPEC" ] && echo "  --speculative-config '$SPEC' \\"
    [ -n "$KV_BYTES" ] && echo "  --kv-cache-memory-bytes $KV_BYTES \\"
    [ "$TEXT_ONLY" = 1 ] && echo "  --limit-mm-per-prompt '{\"image\":0}' \\"
    [ "$rank" != 0 ] && echo "  --headless \\"
    echo "  $EXTRA"
  } > "$tmp"
  scp -q -o BatchMode=yes "$tmp" "$n:/tmp/$NAME.sh"
  rm -f "$tmp"
  sh_ "$n" "chmod +x /tmp/$NAME.sh; docker rm -f $NAME >/dev/null 2>&1; mkdir -p \$HOME/.cache/dsv41-serve
    docker run -d --name $NAME --network host --ipc host --shm-size 64g --gpus all --device /dev/infiniband \
      --ulimit memlock=-1 --ulimit stack=67108864 --restart no --entrypoint /bin/bash \
      -v '$MODEL_DIR':'$MODEL_DIR':ro -v /tmp/$NAME.sh:/serve.sh:ro -v \$HOME/.cache/dsv41-serve:/root/.cache \
      -e NCCL_IB_HCA=$FAB_HCA -e NCCL_IB_GID_INDEX=$gid -e NCCL_IB_DISABLE=0 -e NCCL_NET=IB \
      -e NCCL_SOCKET_IFNAME=$FAB_IF -e GLOO_SOCKET_IFNAME=$FAB_IF -e TP_SOCKET_IFNAME=$FAB_IF \
      -e VLLM_HOST_IP=$self -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
      -e VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-FlashInferCutlassMxfp8LinearKernel} \
      -e HF_HUB_OFFLINE=1 $envs $IMAGE /serve.sh >/dev/null && echo '$n rank $rank started'"
}

case "$cmd" in
  up)
    # operator hold: a chain stage waits here while any ~/.dsv41_serve_hold* exists (lever A/Bs slot in between stages
    # without editing the running chain); the A/B scripts pass SKIP_SERVE_HOLD=1. TERM is ignored while holding: chain
    # stages wrap this in `timeout 1500`, a hold outlives that, and a killed boot would send the chain on to its next
    # `down`, which stops whatever lever serve is up. Released after the timer fired, that boot still
    # returns timeout's 124 (a pipefail chain skips what follows it), and kill -9 is what stops a held instance.
    if compgen -G "$HOME/.dsv41_serve_hold*" >/dev/null && [ -z "${SKIP_SERVE_HOLD:-}" ]; then
      trap '' TERM
      while compgen -G "$HOME/.dsv41_serve_hold*" >/dev/null; do sleep 30; done
      trap - TERM
    fi
    : "${MODEL_DIR:?set MODEL_DIR (same path on both nodes)}"
    [ -z "$INDEXER_LOGITS_MB" ] || [[ "$INDEXER_LOGITS_MB" =~ ^[1-9][0-9]*$ ]] \
      || { echo "INDEXER_LOGITS_MB=$INDEXER_LOGITS_MB: want a whole number of MB (vLLM's default: 512)"; exit 2; }
    case "$EXPANDABLE_SEGMENTS" in 0|1) ;; *) echo "EXPANDABLE_SEGMENTS=$EXPANDABLE_SEGMENTS: 0 or 1"; exit 2 ;; esac
    hip=$(fab_ip "$HEAD"); wip=$(fab_ip "$WORKER")
    hg=$(gid_of "$HEAD"); wg=$(gid_of "$WORKER")
    [ -n "$hip" ] && [ -n "$wip" ] && [ -n "$hg" ] && [ -n "$wg" ] || { echo "fabric not ready: $hip/$wip gid $hg/$wg"; exit 1; }
    echo "head $HEAD $hip gid $hg | worker $WORKER $wip gid $wg | image $IMAGE | model $MODEL_DIR | load $LOAD_FORMAT${MEM_ENV:+ | $MEM_ENV}"
    for n in "$HEAD" "$WORKER"; do sh_ "$n" "test -d '$MODEL_DIR'" || { echo "$MODEL_DIR missing on $n"; exit 1; }; done
    for n in "$HEAD" "$WORKER"; do sh_ "$n" "docker image inspect $IMAGE >/dev/null 2>&1" \
      || { echo "$IMAGE missing on $n: build it (README quickstart, step 2) or docker save | ssh $n docker load"; exit 1; }; done
    # patch 0011's token list (most frequent first) into each node's cache, which the containers mount at /root/.cache
    case "$RECIPE_ENV $DOCKER_ENV" in *VLLM_DSPARK_MARKOV_CACHE=*)
      for n in "$HEAD" "$WORKER"; do
        sh_ "$n" "mkdir -p \$HOME/.cache/dsv41-serve" && scp -q "$REPO/data/dspark/markov_top_tokens.json" "$n:.cache/dsv41-serve/"
      done ;;
    esac
    # patch 0036's draft-feature head, fitted to the pack's own target and drafter (scripts/dspark/fit_draft_features.py):
    # read from the model dir when the pack ships one (the container mounts it at the same path); without one, or on
    # an older pack, conf+feat falls back to the confidence head alone (conf)
    case "$RECIPE_ENV $DOCKER_ENV" in *VLLM_DSPARK_DEPTH_POLICY=conf+feat*)
      case "$DOCKER_ENV" in *VLLM_DSPARK_CONF_FEAT_HEAD=*) ;; *)
        if sh_ "$HEAD" "test -f '$MODEL_DIR/dspark_conf_feat_head.npz'" && sh_ "$WORKER" "test -f '$MODEL_DIR/dspark_conf_feat_head.npz'"; then
          DOCKER_ENV="VLLM_DSPARK_CONF_FEAT_HEAD=$MODEL_DIR/dspark_conf_feat_head.npz${DOCKER_ENV:+ $DOCKER_ENV}"
        else
          echo "no dspark_conf_feat_head.npz in $MODEL_DIR: depth policy conf"
          DOCKER_ENV="${DOCKER_ENV:+$DOCKER_ENV }VLLM_DSPARK_DEPTH_POLICY=conf"
        fi ;;
      esac ;;
    esac
    # GB10: CUDA allocations do not reclaim page cache; vLLM would size its pools to "free", not "available"
    for n in "$HEAD" "$WORKER"; do sh_ "$n" "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || echo 'drop_caches needs sudo -n on $n'"; done
    launch "$WORKER" 1 "$hip" "$wip" "$wg"
    launch "$HEAD" 0 "$hip" "$hip" "$hg"
    t0=$(date +%s)
    until sh_ "$HEAD" "curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1"; do
      for n in "$HEAD" "$WORKER"; do
        st=$(sh_ "$n" "docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null" || echo false)
        [ "$st" = true ] || { echo "$n: container stopped"; sh_ "$n" "docker logs --tail 40 $NAME 2>&1"; exit 1; }
      done
      sleep 15
    done
    echo "ready after $(( $(date +%s) - t0 )) s"
    for n in "$HEAD" "$WORKER"; do echo "$n: $(sh_ "$n" "free -g | awk '/Mem:/{print \"used\", \$3, \"GiB, available\", \$7, \"GiB\"}'")"; done ;;
  down)
    for n in "$HEAD" "$WORKER"; do sh_ "$n" "docker rm -f $NAME >/dev/null 2>&1; echo '$n down'"; done ;;
  logs)
    n=$([ "${2:-head}" = worker ] && echo "$WORKER" || echo "$HEAD"); sh_ "$n" "docker logs --tail ${3:-60} $NAME 2>&1" ;;
  status)
    for n in "$HEAD" "$WORKER"; do echo "$n: $(sh_ "$n" "docker ps --filter name=$NAME --format '{{.Status}}'; free -g | awk '/Mem:/{print \"avail\", \$7, \"GiB\"}'" | tr '\n' ' ')"; done
    sh_ "$HEAD" "curl -s http://localhost:$PORT/v1/models | head -c 300"; echo ;;
  *) echo "usage: $0 up|down|logs [head|worker]|status"; exit 2 ;;
esac
