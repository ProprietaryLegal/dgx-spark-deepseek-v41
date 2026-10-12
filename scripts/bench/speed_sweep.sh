#!/usr/bin/env bash
# Unattended speed sweep: one TP=2 serve per configuration (scripts/launch/dsv41_tp2.sh), kit_bench set b decode + DSpark
# acceptance for each (serve_speed_suite.sh QUICK=1, FULL=1 for the whole suite), MemAvailable after boot, one summary
# table at the end. Run from the control machine with no other GPU job on either Spark.
#   ./speed_sweep.sh sweep.conf            (DRY=1: print what each configuration would launch, launch nothing)
# sweep.conf, one configuration per line, '#' comments:   name | model dir | DSpark k (0 = off, or a speculative-config
# JSON) | container env
#   v2_k5_chained  | /home/spark/models/dsv41-recipe-exl3-v2 | 5 |
#   v2_k5_par      | /home/spark/models/dsv41-recipe-exl3-v2 | 5 | VLLM_EXL3_MOE_PARALLEL_GROUPS=1
#   v2_k3_par      | /home/spark/models/dsv41-recipe-exl3-v2 | 3 | VLLM_EXL3_MOE_PARALLEL_GROUPS=1
set -uo pipefail
CONF=${1:?sweep config file}
HERE=$(cd "$(dirname "$0")" && pwd)
L="$HERE/../launch/dsv41_tp2.sh"
ROOT=${ROOT:-${DSV41_RESULTS:-$(cd "$(dirname "$0")/../.." && pwd)/results}/speed}
MAX_LEN=${MAX_LEN:-131072}            # spec decode at 256K has wedged GB10 boxes; the sweep needs 32K at most
MIN_AVAIL=${MIN_AVAIL:-2}             # GiB per node after boot, below it the configuration is skipped
trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; echo "${s%"${s##*[![:space:]]}"}"; }
names=()
while IFS='|' read -r -u 3 name dir k envs; do          # fd 3: ssh inside the loop must not eat the config
  name=$(trim "$name"); dir=$(trim "$dir"); k=$(trim "$k"); envs=$(trim "${envs:-}")
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  out="$ROOT/sweep_$name"
  case "$k" in                                       # a number = DSpark depth, or a whole speculative-config JSON
    0) spec="" ;;
    "{"*) spec="$k" ;;
    *) spec="{\"method\":\"dspark\",\"num_speculative_tokens\":$k}" ;;
  esac
  if [ "${DRY:-0}" = 1 ]; then echo "$name: MODEL_DIR=$dir SPEC='$spec' DOCKER_ENV='$envs' MAX_LEN=$MAX_LEN"; continue; fi
  mkdir -p "$out"
  echo "== $(date +%H:%M) $name: $dir, DSpark k=$k, env [$envs]" | tee "$out/sweep.log"
  "$L" down >/dev/null 2>&1
  if ! MODEL_DIR="$dir" SPEC="$spec" DOCKER_ENV="$envs" MAX_LEN="$MAX_LEN" "$L" up 2>&1 | tee -a "$out/sweep.log"; then
    echo "$name: serve did not come up" | tee -a "$out/sweep.log"
    "$L" logs head 80 > "$out/head_logs.txt" 2>&1; "$L" logs worker 80 > "$out/worker_logs.txt" 2>&1
    continue
  fi
  low=$(grep -o -E 'available [0-9]+' "$out/sweep.log" | awk -v m="$MIN_AVAIL" '$2 < m' | wc -l)
  if [ "$low" -gt 0 ]; then echo "$name: MemAvailable under $MIN_AVAIL GiB after boot, skipped" | tee -a "$out/sweep.log"; continue; fi
  if [ "${FULL:-0}" = 1 ]; then q=0; else q=1; fi
  SERVE_ID="sweep_$name" OUT="$out" QUICK=$q "$HERE/serve_speed_suite.sh" 2>&1 | tee -a "$out/sweep.log"
  names+=("$name")
done 3< "$CONF"
[ "${DRY:-0}" = 1 ] && exit 0
"$L" down >/dev/null 2>&1
python3 - "$ROOT" "${names[@]}" <<'PY'
import json, sys
root, names = sys.argv[1], sys.argv[2:]
print(f"{'configuration':28s} {'code':>8s} {'prose':>8s} {'struct':>8s}  acceptance")
for n in names:
    d = f"{root}/sweep_{n}"
    try:
        rows = {r["prompt"]: r["decode_tps_median"] for r in json.load(open(f"{d}/kit_decode_b.json"))["result"]}
    except (OSError, ValueError, KeyError):
        rows = {}
    try:
        acc = open(f"{d}/acceptance.txt").read().strip().split(": ", 1)[-1]
    except OSError:
        acc = "-"
    f = lambda k: f"{rows[k]:8.1f}" if k in rows else f"{'-':>8s}"
    print(f"{n:28s} {f('code')} {f('prose')} {f('structured')}  {acc}")
print("tok/s, kit_bench set b, median of 3, single stream")
PY
