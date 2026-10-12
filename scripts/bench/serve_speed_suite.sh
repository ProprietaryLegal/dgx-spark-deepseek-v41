#!/usr/bin/env bash
# Speed suite against a running TP=2 serve (scripts/launch/dsv41_tp2.sh up), run from the control machine:
# bertholomus's kit_bench.py (the client behind the published TensorFold and Mia-kit numbers: set b = greedy, thinking
# off, 384 tokens, median of 3), 2 concurrent streams, prefill at 8K / 32K, decode after a 32K prompt, and the DSpark
# draft counters from vLLM's /metrics around the decode set (acceptance = accepted / drafted). Writes to
# $DSV41_RESULTS/speed/<SERVE_ID> (default: results/ in this repo).
#   SERVE_ID=v2_marlin_mxfp8head ./serve_speed_suite.sh            (KIT=path/to/kit_bench.py, HEAD=spark, PORT=8000)
#   QUICK=1 ...: decode set b + DSpark acceptance + the oracle only (the speed sweep's per-configuration run)
set -uo pipefail
SERVE_ID=${SERVE_ID:?name for this serve configuration}
HEAD=${HEAD:-spark}; PORT=${PORT:-8000}; MODEL=${MODEL:-dsv41}
KIT=${KIT:-$HOME/Projects/benchmarks/kit_bench/kit_bench.py}   # not vendored here: no license header
QUICK=${QUICK:-0}
OUT=${OUT:-${DSV41_RESULTS:-$(cd "$(dirname "$0")/../.." && pwd)/results}/speed/$SERVE_ID}
mkdir -p "$OUT"
scp -q "$KIT" "$HEAD:/tmp/kit_bench.py"
ssh -o BatchMode=yes "$HEAD" "rm -f /tmp/ss_*.json /tmp/ss_*.jsonl"     # no earlier run's files under this run's name
run() { ssh -o BatchMode=yes "$HEAD" "cd /tmp && python3 kit_bench.py --base http://localhost:$PORT --model $MODEL $*" ; }
metrics() { ssh -o BatchMode=yes "$HEAD" "curl -s localhost:$PORT/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total|^vllm:spec_decode_num_drafts_total|^vllm:generation_tokens_total'" ; }
echo "== $SERVE_ID  $(date +%F\ %T)" | tee "$OUT/run.log"
ssh -o BatchMode=yes "$HEAD" "curl -s localhost:$PORT/v1/models" > "$OUT/models.json"
run ready --timeout 900 --out /tmp/ss_ready.json 2>&1 | tee -a "$OUT/run.log"
metrics > "$OUT/metrics_before.txt"
run decode --set b --tokens 384 --reps 3 --out /tmp/ss_decode_b.json 2>&1 | tee -a "$OUT/run.log"
metrics > "$OUT/metrics_after_decode_b.txt"
# greedy continuations + teacher-forced top-20 logprobs on the fixed 24-prompt set: oracle_compare.py scores the
# serving-path fidelity between configurations of the same weights (int8 activation modes, kernel paths)
if [ "${NO_ORACLE:-0}" != 1 ]; then      # NO_ORACLE=1: another build's weights, where the oracle compares nothing
  run oracle --out /tmp/ss_oracle.jsonl 2>&1 | tail -2 | tee -a "$OUT/run.log"
  scp -q "$HEAD:/tmp/ss_oracle.jsonl" "$OUT/kit_oracle.jsonl" 2>/dev/null
fi
if [ "$QUICK" != 1 ]; then
  run decode --set a --tokens 384 --reps 3 --out /tmp/ss_decode_a.json 2>&1 | tee -a "$OUT/run.log"
  run concurrent --streams 2 --tokens 256 --set b --out /tmp/ss_conc.json 2>&1 | tee -a "$OUT/run.log"
  run prefill --lengths 8192,32768 --out /tmp/ss_prefill.json 2>&1 | tee -a "$OUT/run.log"
  run depth --lengths 32768 --tokens 256 --out /tmp/ss_depth.json 2>&1 | tee -a "$OUT/run.log"
fi
for f in decode_b decode_a conc prefill depth; do scp -q "$HEAD:/tmp/ss_$f.json" "$OUT/kit_$f.json" 2>/dev/null; done
python3 - "$OUT" <<'EOF'
import re, sys
out = sys.argv[1]
def read(p):
    d = {}
    for line in open(p):
        m = re.match(r'^(vllm:\w+)\{[^}]*\}\s+([\d.e+]+)', line)
        if m:
            d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(2))
    return d
a, b = read(f'{out}/metrics_before.txt'), read(f'{out}/metrics_after_decode_b.txt')
acc = b.get('vllm:spec_decode_num_accepted_tokens_total', 0) - a.get('vllm:spec_decode_num_accepted_tokens_total', 0)
dra = b.get('vllm:spec_decode_num_draft_tokens_total', 0) - a.get('vllm:spec_decode_num_draft_tokens_total', 0)
nd = b.get('vllm:spec_decode_num_drafts_total', 0) - a.get('vllm:spec_decode_num_drafts_total', 0)
msg = (f'set b DSpark: accepted {acc:.0f} of {dra:.0f} drafted ({100 * acc / max(dra, 1):.1f}%), '
       f'mean accepted per verify step {acc / max(nd, 1):.2f} (+1 bonus token)')
print(msg)
open(f'{out}/acceptance.txt', 'w').write(msg + '\n')
EOF
echo "results: $OUT"
