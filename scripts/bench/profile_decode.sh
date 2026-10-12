#!/usr/bin/env bash
# Real-weight decode profile of one serve configuration: boots dsv41_tp2.sh with vLLM's torch profiler, warms up, then
# profiles one greedy single-stream decode (kit_bench set b's code prompt by default) between /start_profile and
# /stop_profile, copies both ranks' traces to the assets folder and runs trace_breakdown.py on rank 0 with the number of
# verify steps the DSpark counters saw in the window (one step per decode token when drafting is off). Run from the control machine
# with nothing else on either GPU:
#   NAME=k3 SPEC='{"method":"dspark","num_speculative_tokens":3}' ./profile_decode.sh
#   NAME=k0 SPEC= ./profile_decode.sh
set -uo pipefail
NAME=${NAME:?profile name}
HERE=$(cd "$(dirname "$0")" && pwd)
L="$HERE/../launch/dsv41_tp2.sh"
HEAD=${HEAD:-spark}; WORKER=${WORKER:-spark2}; PORT=${PORT:-8000}; MODEL=${MODEL:-dsv41}
TOKENS=${TOKENS:-128}
PROMPT=${PROMPT:-code}                            # kit_bench set b prompt: code, prose or structured
KIT=${KIT:-$HOME/Projects/benchmarks/kit_bench/kit_bench.py}
OUT=${OUT:-${DSV41_RESULTS:-$(cd "$(dirname "$0")/../.." && pwd)/results}/profiles/$NAME}
export MODEL_DIR=${MODEL_DIR:-/home/spark/models/dsv41-recipe-exl3-v2.1}
export DOCKER_ENV=${DOCKER_ENV-"VLLM_EXL3_MOE_PARALLEL_GROUPS=1 EXL3_MOE_COOP_WIDE=1 EXL3_INT8_GEMV=0"}
export MAX_LEN=${MAX_LEN:-131072} SPEC=${SPEC-'{"method":"dspark","num_speculative_tokens":3}'}
PROF_DIR=/root/.cache/prof_$NAME                  # the launcher mounts ~/.cache/dsv41-serve as /root/.cache
export EXTRA="--profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROF_DIR\",\"torch_profiler_with_stack\":false}'"
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" "$2"; }
die() { say "STOP: $*"; exit 1; }
post() { sh_ "$HEAD" "curl -sf -X POST localhost:$PORT/$1" >/dev/null; }
drafts() { sh_ "$HEAD" "curl -s localhost:$PORT/metrics" | awk '/^vllm:spec_decode_num_drafts_total/{s+=$2} END{print s+0}'; }
gen() { sh_ "$HEAD" "curl -s localhost:$PORT/metrics" | awk '/^vllm:generation_tokens_total/{s+=$2} END{print s+0}'; }
chat() {   # $1 max tokens: kit_bench set b's $PROMPT prompt, greedy, thinking off
  local body
  body=$(python3 - "$KIT" "$1" "$MODEL" "$PROMPT" <<'PY'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("kb", sys.argv[1]); kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kb)
print(json.dumps({"model": sys.argv[3], "messages": [{"role": "user", "content": dict(kb.DECODE_PROMPTS_B)[sys.argv[4]]}],
                  "max_tokens": int(sys.argv[2]), "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}))
PY
)
  echo "$body" | sh_ "$HEAD" "curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @-" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["usage"]["completion_tokens"])'
}

say "boot $NAME: SPEC=[$SPEC] env [$DOCKER_ENV]"
"$L" down >/dev/null 2>&1
for n in "$HEAD" "$WORKER"; do sh_ "$n" "rm -rf \$HOME/.cache/dsv41-serve/prof_$NAME"; done
"$L" up >> "$OUT/run.log" 2>&1 || die "serve did not come up"
say "warm-up: $(chat 64) tokens"
d0=$(drafts); g0=$(gen)
post start_profile || die "start_profile (profiler config not active?)"
n=$(chat "$TOKENS")
post stop_profile || die "stop_profile"
d1=$(drafts); g1=$(gen)
steps=$(( d1 - d0 )); [ "$steps" -gt 0 ] || steps=$(( g1 - g0 ))
say "profiled: $n tokens, $steps decode steps (drafts $d0 -> $d1, generated $g0 -> $g1)"
sleep 20                                          # the profiler writes its traces after stop
for n2 in "$HEAD" "$WORKER"; do
  mkdir -p "$OUT/$n2"
  scp -q -r "$n2:.cache/dsv41-serve/prof_$NAME/*" "$OUT/$n2/" 2>>"$OUT/run.log" || say "no trace from $n2 yet"
done
"$L" down >/dev/null 2>&1
t=$(find "$OUT/$HEAD" -name '*.json*' -size +1M | head -1)
[ -n "$t" ] || die "no rank-0 trace under $OUT/$HEAD"
python3 "$HERE/trace_breakdown.py" "$t" --steps "$steps" --top 30 2>&1 | tee "$OUT/breakdown.txt"
say "done: $OUT"
