#!/usr/bin/env bash
# Same-box baseline, v0.6.0: bertholomus's TensorFold DeepSeek-V4.1 TP2 on TensorFold 1.0's native Zig engine (branch
# deepseek-v41-zig, engine ce20a54, Apache-2.0) on Mia-AiLab's EXL3 2.9 bpw pack, on our two Sparks, with his served
# settings (deepseek-v4.1-tensorfold-tp2-2xgb10 README, "Run it (v0.6.0, the Zig engine)": --context 1048576 --parallel 4),
# measured with the same kit_bench.py. We build and run his engine to measure it; none of its code goes into our recipe.
# Two network adaptations to our fabric (his values are hardcoded for his boxes), nothing else changed:
#   - path MTU 1024 instead of 4096 (our link is MTU 1500; rdma.zig, one line), as for v0.5.1;
#   - RoCE GID index 3 instead of 5 (our IPv4 RoCE v2 GID sits at index 3; model.zig x2, layer_main.zig x1).
# One RoCE port (rocep1s0f0), as for v0.5.1.
# Inputs on both boxes under /home/spark/dsv41_recipe/tfzig: out/native/bin/tensorfold-native (rank 0, the server) and
# out/bin/tf-dsv41-lanes (rank 1), from `zig build install native` (LOG Oct 12), kit/ (his kit/ +
# the generated files, verify_kit.sh: 221 files OK), dsv41_token_map.json; the v0.5.1 engine's rank caches in
# tf_rank_cache. Run from the control machine with nothing else on either GPU:
#   ./tensorfold_zig_samebox.sh up | bench [OUT_ID] | logs [head|worker] | down
set -euo pipefail
HEAD=${HEAD:-spark}; WORKER=${WORKER:-spark2}
FAB_IF=${FAB_IF:-enp1s0f0np0}; HCA=${HCA:-rocep1s0f0}
IMAGE=${IMAGE:-tensorfold:0.6.6-cb2ebf0}               # nvcr.io/nvidia/pytorch:26.07-py3 based (his build image)
W=${W:-/home/spark/dsv41_recipe/tfzig}
MODEL=${MODEL:-/home/spark/models/Mia-DeepSeek-V4.1-Flash-EXL3-2.9bpw-64ba41b6}
ENGRAM=${ENGRAM:-/home/spark/models/DeepSeek-V4.1-Flash-official-2cba9e42}
CACHE=${CACHE:-/home/spark/dsv41_recipe/tf_rank_cache}
PORT=${PORT:-18891}; TPPORT=${TPPORT:-29620}
NSYS_DIR=${NSYS_DIR:-/opt/nvidia/nsight-systems/2025.3.2}   # the Sparks' host install, mounted read-only (NSYS=1 up)
NSYS_OUT=${NSYS_OUT:-/home/spark/dsv41_recipe/tf_nsys}
NAME=tf-zig
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cmd=${1:-status}
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }
fab_ip() { sh_ "$1" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1"; }
gid_of() {   # RoCE v2 GID index whose IPv4-mapped GID is this node's fabric address (same rule as dsv41_tp2.sh)
  sh_ "$1" "want=\$(ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1); P=/sys/class/infiniband/$HCA/ports/1
    hex=\$(printf '%02x%02x%02x%02x' \$(echo \$want | tr '.' ' '))
    for i in \$(seq 0 15); do t=\$(cat \$P/gid_attrs/types/\$i 2>/dev/null); g=\$(cat \$P/gids/\$i 2>/dev/null)
      case \"\$t\" in *v2*) ;; *) continue;; esac
      case \"\$g\" in *ffff:\${hex:0:4}:\${hex:4:4}*) echo \$i; break;; esac; done"
}
run_flags="--network host --ipc host --shm-size 64g --gpus all --device /dev/infiniband --ulimit memlock=-1 --ulimit stack=67108864 --restart no -v $W:$W -v $MODEL:$MODEL:ro -v $ENGRAM:$ENGRAM:ro -v $CACHE:$CACHE"
case "$cmd" in
  up)
    hip=$(fab_ip $HEAD); g0=$(gid_of $HEAD); g1=$(gid_of $WORKER)
    ent0="--entrypoint $W/out/native/bin/tensorfold-native $IMAGE"; ent1="--entrypoint $W/out/bin/tf-dsv41-lanes $IMAGE"; mnt=""
    if [ "${NSYS:-0}" = 1 ]; then   # under nsys launch: nothing is captured until 'profile' runs nsys start / stop
      N=$NSYS_DIR/target-linux-sbsa-armv8/nsys; mnt="-v $NSYS_DIR:$NSYS_DIR:ro -v $NSYS_OUT:$NSYS_OUT"
      ent0="--entrypoint $N $IMAGE launch --session-new=tf --trace=cuda,nvtx $W/out/native/bin/tensorfold-native"
      ent1="--entrypoint $N $IMAGE launch --session-new=tf --trace=cuda,nvtx $W/out/bin/tf-dsv41-lanes"
      for n in $HEAD $WORKER; do sh_ $n "mkdir -p $NSYS_OUT"; done
    fi
    echo "head $HEAD $hip gid $g0 | worker $WORKER gid $g1 | image $IMAGE | model $MODEL"
    for n in $HEAD $WORKER; do sh_ $n "docker rm -f $NAME >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || true"; done
    sh_ $WORKER "docker run -d --name $NAME $run_flags -e TF_DS_REPLAY_FLOOR=1 -e NCCL_IB_HCA=$HCA -e NCCL_IB_GID_INDEX=$g1 \
      -e NCCL_SOCKET_IFNAME=$FAB_IF $mnt $ent1 $MODEL $CACHE 1 2 $hip $TPPORT $W/kit \
      --engram $ENGRAM --token-map $W/dsv41_token_map.json --pool 1048576 --rdma $HCA --graphs 1 --side 1 --prefetch 1 \
      --aio 1 >/dev/null && echo 'rank 1 started'"
    sleep 5
    sh_ $HEAD "docker run -d --name $NAME $run_flags -e TF_DS_REPLAY_FLOOR=1 -e TF_TP_WORLD=2 -e TF_TP_MASTER=$hip \
      -e TF_TP_PORT=$TPPORT -e TF_DS_RANK_CACHE=$CACHE -e TF_DS_KIT=$W/kit -e TENSORFOLD_CUDA_KERNELS=$W/kit/aot \
      -e TF_DS_ENGRAM=$ENGRAM -e TF_DS_TOKEN_MAP=$W/dsv41_token_map.json -e TF_RDMA_DEVICES=$HCA -e TF_DS_GRAPHS=1 \
      -e TF_DS_HC_SIDE=1 -e TF_DS_L2_PREFETCH=1 -e TF_DS_ENGRAM_AIO=1 -e NCCL_IB_HCA=$HCA -e NCCL_IB_GID_INDEX=$g0 \
      -e NCCL_SOCKET_IFNAME=$FAB_IF $mnt $ent0 serve $MODEL --host 127.0.0.1 --port $PORT \
      --name DeepSeek-V4.1-Flash-TF --api-key-file $W/apikey --context 1048576 --parallel 4 --thinking \
      --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 20 --backend cuda >/dev/null && echo 'rank 0 started'"
    scp -q "$HOME/Projects/benchmarks/kit_bench/kit_bench.py" $HEAD:/tmp/kit_bench.py
    t0=$(date +%s)
    until sh_ $HEAD "cd /tmp && timeout 20 python3 kit_bench.py --base http://127.0.0.1:$PORT --model DeepSeek-V4.1-Flash-TF \
        --key-file $W/apikey ready --timeout 15 >/dev/null 2>&1"; do
      for n in $HEAD $WORKER; do
        sh_ $n "docker ps --format '{{.Names}}' | grep -qx $NAME" || { echo "$n: $NAME exited"; sh_ $n "docker logs --tail 30 $NAME 2>&1"; exit 1; }
      done
      [ $(( $(date +%s) - t0 )) -gt 2400 ] && { echo "not ready in 40 min"; exit 1; }
      sleep 20
    done
    echo "ready after $(( $(date +%s) - t0 )) s" ;;
  bench)
    ID=${2:-tf_v060_samebox}
    OUT=${DSV41_RESULTS:-$(cd "$(dirname "$0")/../.." && pwd)/results}/speed/$ID; mkdir -p "$OUT"
    sh_ $HEAD "cd /tmp && python3 kit_bench.py --base http://127.0.0.1:$PORT --model DeepSeek-V4.1-Flash-TF --key-file $W/apikey \
      decode --set b --tokens 384 --reps ${REPS:-3} --out /tmp/$ID.json" > "$OUT/run.log" 2>&1
    scp -q $HEAD:/tmp/$ID.json "$OUT/kit_decode_b.json"
    python3 - "$OUT/kit_decode_b.json" <<'EOF'
import json, sys
for r in json.load(open(sys.argv[1]))["result"]:
    print(f"{r['prompt']:10s} median {r['decode_tps_median']:.1f} tok/s, reps {[round(x, 1) for x in r['decode_tps_all']]}, "
          f"identical {r.get('identical_texts')}")
EOF
    ;;
  profile)   # after NSYS=1 up: one PROMPT_TOKENS (8192) prefill under nsys on both ranks, after a warm-up prefill of the
             # same length (a different random-word prompt); each rank's kernel trace to CSV (nsys stats cuda_gpu_trace)
    tag=${2:-prefill8k}; N=$NSYS_DIR/target-linux-sbsa-armv8/nsys
    pf() {   # $1 seed: kit_bench's filler prompt of PROMPT_TOKENS tokens, 1 token out; prints its TTFT
      sh_ $HEAD "cd /tmp && python3 - $1 ${PROMPT_TOKENS:-8192} $PORT $W/apikey" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("kb", "/tmp/kit_bench.py"); kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kb)
seed, n, port, keyf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
c = kb.Client(f"http://127.0.0.1:{port}", "DeepSeek-V4.1-Flash-TF", kb.load_key(keyf))
r = c.stream({"prompt": kb.filler_ids(c, n, seed=seed), "max_tokens": 1, "temperature": 0}, "/v1/completions")
print(f"prompt {r['prompt_tokens']} tokens, ttft {r['ttft_s']:.3f} s, {r['prompt_tokens'] / r['ttft_s']:.1f} tok/s")
PY
    }
    echo "warm-up: $(pf $((RANDOM + 1)))"
    for n in $HEAD $WORKER; do
      sh_ $n "docker exec $NAME $N start --session=tf --sample=none --cpuctxsw=none -o $NSYS_OUT/tfz_\$(hostname)_$tag --force-overwrite=true" &
    done
    wait
    echo "profiled: $(pf $((RANDOM + 40000)))"
    for n in $HEAD $WORKER; do sh_ $n "docker exec $NAME $N stop --session=tf >/dev/null" & done
    wait
    for n in $HEAD $WORKER; do
      sh_ $n "cd $NSYS_OUT && $N stats --report cuda_gpu_trace --format csv --force-export=true -o tfz_\$(hostname)_$tag \
        tfz_\$(hostname)_$tag.nsys-rep >/dev/null 2>&1 && ls -la tfz_\$(hostname)_${tag}*" &
    done
    wait ;;
  logs) n=$HEAD; [ "${2:-head}" = worker ] && n=$WORKER; sh_ $n "docker logs --tail ${TAIL:-60} $NAME 2>&1" ;;
  down) for n in $HEAD $WORKER; do sh_ $n "docker rm -f $NAME >/dev/null 2>&1 && echo '$n: down' || true"; done ;;
  status) for n in $HEAD $WORKER; do sh_ $n "docker ps --format '{{.Names}} {{.Status}}' | grep $NAME || echo '$n: not running'"; done ;;
  *) echo "usage: $0 up | bench [ID] | profile [TAG] | logs [head|worker] | down | status"; exit 2 ;;
esac
