#!/usr/bin/env bash
# Input Hessians of V4.1's Engram projections (layers.<1|14>.engram.wkv) for their EXL3 encode (p4_engram.py): an
# eager serve (no CUDA graphs: a graph replay runs no Python, so the capture hook would see nothing) with
# VLLM_EXL3_HCAP, 16 calibration sequences prefilled (cal_v3, up to 2,048 tokens each, ~30K rows: a 6,144-wide H needs
# well over 6,144) and set b's three decodes, then SAVE on both nodes (each mounts its own cache) and a few more calls
# so the hook sees it. Run from the control machine with nothing else on either GPU. Out: ~/.cache/dsv41-serve/hcap_engram on spark
# (rank 0's file is the module's H: column-parallel, every rank sees the whole input).
#   scripts/quantize/hcap_engram.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/home/spark/models/dsv41-recipe-exl3-v3-dk5}
export SKIP_SERVE_HOLD=1
say() { echo "$(date +%T) $*"; }
for h in spark spark2; do ssh $h "rm -rf ~/.cache/dsv41-serve/hcap_engram; mkdir -p ~/.cache/dsv41-serve/hcap_engram"; done
say "eager serve with the Engram capture"
IMAGE=${IMAGE:-dsv41-serve:v4} MODEL_DIR=$MODEL_DIR GRAPH='{"cudagraph_mode":"NONE"}' \
  DOCKER_ENV="VLLM_EXL3_HCAP=/root/.cache/hcap_engram VLLM_EXL3_HCAP_MATCH=engram.wkv" \
  timeout 1800 "$REPO/scripts/launch/dsv41_tp2.sh" up 2>&1 | tail -2
send() {   # $1 sequences, $2 stride offset: prefill-only completions of calibration token sequences
  ssh spark "docker run -i --rm --network host -v /home/spark:/home/spark --entrypoint python3 dsv41-quant:p1 - $1 $2" <<'EOF'   # -i: stdin carries the script (without it nothing ran, Oct 11)
import json, sys, urllib.request
from safetensors import safe_open
want, shift = int(sys.argv[1]), int(sys.argv[2])
with safe_open("/home/spark/dsv41_recipe/caldata/cal_v3.safetensors", "pt") as f:
    tok, off = f.get_tensor("tokens"), f.get_tensor("offsets")
seqs = len(off) - 1
n = 0
for j in range(want):
    i = (j * (seqs // want) + shift) % seqs
    ids = tok[off[i]:off[i + 1]].tolist()[:2048]
    body = json.dumps({"model": "dsv41", "prompt": ids, "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request("http://localhost:8000/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=900).read()
    n += len(ids)
print(f"prefilled {want} sequences, {n} tokens", flush=True)
EOF
}
send 16 0
ssh spark "test -f /tmp/kit_bench.py" || scp -q "${KIT:-$HOME/Projects/benchmarks/kit_bench/kit_bench.py}" spark:/tmp/kit_bench.py
ssh spark "python3 /tmp/kit_bench.py --base http://localhost:8000 --model dsv41 decode --set b --reps 1 --tokens 256 2>&1 | tail -3"
for h in spark spark2; do ssh $h "touch ~/.cache/dsv41-serve/hcap_engram/SAVE"; done
# The hook looks for SAVE every 64 calls of its module, and engram.wkv runs once a forward: a set b decode after SAVE
# (~300 verify forwards) crosses a check on both ranks. Prefill-only calls (~50 in the wait below) did not (Oct 11 11:35:
# no file, and the serve's SIGKILL skips the hook's atexit save).
ssh spark "python3 /tmp/kit_bench.py --base http://localhost:8000 --model dsv41 decode --set b --reps 1 --tokens 256 2>&1 | tail -1"
for hr in spark:0 spark2:1; do
  h=${hr%%:*}; r=${hr#*:}; t0=$(date +%s)
  until ssh $h "test -f ~/.cache/dsv41-serve/hcap_engram/saved.rank$r"; do
    [ $(( $(date +%s) - t0 )) -gt 300 ] && { say "$h: no saved.rank$r after 5 min"; break; }
    send 1 $((RANDOM % 700)) >/dev/null; sleep 5
  done
done
for h in spark spark2; do ssh $h "ls -la ~/.cache/dsv41-serve/hcap_engram/ | tail -4"; done
"$REPO/scripts/launch/dsv41_tp2.sh" down >/dev/null 2>&1
say "capture done"
