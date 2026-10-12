#!/usr/bin/env bash
# Patch 0010 sweep on one running serve: for each DSpark depth policy, write it to the serve's depth file (rank 0 reads
# it every 64 verify steps), let a warm-up request cross a reload, then run depth_probe.py (set b, per-prompt steps,
# tokens per step, ms per step, full texts). Run from the control machine against a serve booted with
#   DOCKER_ENV="... VLLM_DSPARK_DEPTH_POLICY=fixed:5 VLLM_DSPARK_DEPTH_FILE=/root/.cache/dspark_depth VLLM_DSPARK_DEPTH_LOG=200"
# (the container's /root/.cache is the head's ~/.cache/dsv41-serve).
#   SWEEP_ID=v3_k5 POLICIES="fixed:5 fixed:4 fixed:3 fixed:2 fixed:1 fixed:0 conf" ./depth_sweep.sh
set -uo pipefail
SWEEP_ID=${SWEEP_ID:?name for this sweep}
HEAD=${HEAD:-spark}; PORT=${PORT:-8000}; MODEL=${MODEL:-dsv41}; REPS=${REPS:-2}
POLICIES=${POLICIES:-"fixed:5 fixed:4 fixed:3 fixed:2 fixed:1 fixed:0 conf"}
KIT=${KIT:-$HOME/Projects/benchmarks/kit_bench/kit_bench.py}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-${DSV41_RESULTS:-$(cd "$(dirname "$0")/../.." && pwd)/results}/speed/depth_$SWEEP_ID}
mkdir -p "$OUT"
scp -q "$KIT" "$HEAD:/tmp/kit_bench.py"; scp -q "$HERE/depth_probe.py" "$HEAD:/tmp/depth_probe.py"
table() {   # round ms by depth from the fixed-depth probes so far (median over prompts), "conf:<ms,...>"
  python3 - "$OUT" <<'PY'
import glob, json, statistics, sys
cost = {}
for f in glob.glob(f"{sys.argv[1]}/probe_fixed*.json"):
    d = int(f.rsplit("fixed", 1)[1].split(".")[0])
    per = {}
    for r in json.load(open(f)):
        per.setdefault(r["prompt"], []).append(r["ms_per_step"])
    cost[d] = statistics.median(statistics.median(v) for v in per.values())
print("conf:" + ",".join(f"{cost[d]:.2f}" for d in sorted(cost)) if cost and sorted(cost) == list(range(len(cost))) else "conf")
PY
}
for pol in $POLICIES; do
  tag=${pol/:/}
  [ "$pol" = conf ] && pol=$(table)       # conf runs with the table the fixed depths just measured
  echo "== $pol $(date +%T)" | tee -a "$OUT/run.log"
  ssh -o BatchMode=yes "$HEAD" "echo '$pol' > ~/.cache/dsv41-serve/dspark_depth"
  # >= 64 verify steps so the serve re-reads the file before the measured requests
  ssh -o BatchMode=yes "$HEAD" "curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 300, one number a line.\"}],\"max_tokens\":600,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}' >/dev/null"
  ssh -o BatchMode=yes "$HEAD" "cd /tmp && python3 depth_probe.py --kit /tmp/kit_bench.py --base http://localhost:$PORT \
    --model $MODEL --reps $REPS --tag $tag --out /tmp/probe_$tag.json" 2>&1 | tee -a "$OUT/run.log"
  scp -q "$HEAD:/tmp/probe_$tag.json" "$OUT/probe_$tag.json"
done
python3 - "$OUT" $POLICIES <<'EOF'
import json, statistics, sys
out, pols = sys.argv[1], [p.split(":")[0] if p.startswith("conf") else p for p in sys.argv[2:]]
texts, cost = {}, {}
print(f"{'policy':10} {'prompt':11} {'tok/s':>7} {'tok/step':>8} {'ms/step':>8}  same text as first policy")
for pol in pols:
    rows = json.load(open(f"{out}/probe_{pol.replace(':', '')}.json"))
    for name in ("code", "prose", "structured"):
        rs = [r for r in rows if r["prompt"] == name]
        tps = statistics.median(r["decode_tps"] for r in rs)
        tpstep = statistics.median(r["tokens_per_step"] for r in rs)
        ms = statistics.median(r["ms_per_step"] for r in rs)
        same = texts.setdefault(name, rs[0]["text"]) == rs[0]["text"] and len({r["text"] for r in rs}) == 1
        print(f"{pol:10} {name:11} {tps:7.1f} {tpstep:8.2f} {ms:8.2f}  {same}")
        if pol.startswith("fixed:"):
            cost.setdefault(int(pol.split(":")[1]), []).append(ms)
if cost:
    # round cost by depth d (= d + 1 live rows), median over the prompts, for patch 0010's table (drafter included)
    ds = sorted(cost)
    print("VLLM_DSPARK_ROUND_MS=" + ",".join(f"{statistics.median(cost[d]):.1f}" for d in ds) +
          " VLLM_DSPARK_DRAFT_MS=0   (depths " + ",".join(map(str, ds)) + ")")
EOF
