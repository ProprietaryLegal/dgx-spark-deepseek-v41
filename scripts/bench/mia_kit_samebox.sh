#!/usr/bin/env bash
# Same-box speed check: Mia-AiLab's DeepSeek-V4.1-Flash kit (AGPL, run as shipped from ~/Projects/benchmarks/mia_kit,
# nothing of it copied into this repo) on our two Sparks, measured with our client: kit_bench set b (greedy, thinking
# off, 384 tokens, median of 3) + vLLM's DSpark counters, through serve_speed_suite.sh. Her kit's own serving settings
# (DSpark k=3, text-only, her stream contract); our boxes' wiring from ~/Projects/benchmarks/mia_kit_ours.env plus the
# fabric GIDs. Run from the control machine with our serve down and no other GPU job on either Spark: a box without her image pulls
# it here (22.5 GB), which is big I/O, so never beside a GPU job.
#   ./mia_kit_samebox.sh                (OUT=.../data/speed/mia_kit_stock; KEEP=1 leaves her serve up)
set -uo pipefail
HEAD=${HEAD:-spark}; WORKER=${WORKER:-spark2}
FAB_IF=${FAB_IF:-enp1s0f0np0}; FAB_HCA=${FAB_HCA:-rocep1s0f0}
KIT=${KIT:-$HOME/Projects/benchmarks/mia_kit}
ENVF=${ENVF:-$HOME/Projects/benchmarks/mia_kit_ours.env}
MODEL=${MODEL:-DeepSeek-v4.1-Flash-EXL3}          # her served model id (README)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-${DSV41_RESULTS:-$(cd "$(dirname "$0")/../.." && pwd)/results}/speed/mia_kit_stock}
IMAGE=$(sed -n 's/^IMAGE=//p' "$ENVF" | tail -1); PORT=$(sed -n 's/^PORT=//p' "$ENVF" | tail -1)
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" "$2"; }
die() { say "STOP: $*"; exit 1; }
gid_of() {   # RoCE v2 GID index whose IPv4-mapped GID is this node's fabric address (scripts/launch/dsv41_tp2.sh)
  sh_ "$1" "want=\$(ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1); P=/sys/class/infiniband/$FAB_HCA/ports/1
    hex=\$(printf '%02x%02x%02x%02x' \$(echo \$want | tr '.' ' '))
    for i in \$(seq 0 15); do t=\$(cat \$P/gid_attrs/types/\$i 2>/dev/null); g=\$(cat \$P/gids/\$i 2>/dev/null)
      case \"\$t\" in *v2*) ;; *) continue;; esac
      case \"\$g\" in *ffff:\${hex:0:4}:\${hex:4:4}*) echo \$i; break;; esac; done"
}
[ -n "$IMAGE" ] && [ -n "$PORT" ] || die "IMAGE / PORT missing from $ENVF"

say "1. both boxes idle, our serve down"
for n in "$HEAD" "$WORKER"; do
  [ "$(sh_ "$n" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")" = 0 ] || die "$n has a GPU job"
  ! sh_ "$n" "docker ps --format '{{.Names}}'" | grep -q -x dsv41-serve || die "$n: our serve is up (dsv41_tp2.sh down)"
done

say "2. her image on both boxes ($IMAGE)"
for n in "$HEAD" "$WORKER"; do
  sh_ "$n" "docker image inspect '$IMAGE' >/dev/null 2>&1" && continue
  say "   pulling on $n"
  sh_ "$n" "docker pull -q '$IMAGE'" >> "$OUT/run.log" 2>&1 || die "docker pull on $n"
done

say "3. her kit to $HEAD:~/mia_kit, .env = her .env.example + our wiring + fabric GIDs"
hg=$(gid_of "$HEAD"); wg=$(gid_of "$WORKER")
[ -n "$hg" ] && [ -n "$wg" ] || die "fabric GID not found ($hg / $wg)"
rsync -a --delete --exclude .git --exclude logs --exclude .env -e "ssh -o BatchMode=yes" "$KIT/" "$HEAD:mia_kit/" \
  || die "kit copy"
# her .env.example first (her shipped defaults: her start.sh sources .env, later lines win), then our wiring + GIDs
{ cat "$KIT/.env.example"; echo; cat "$ENVF"; echo "HEAD_GID=$hg"; echo "WORKER_GID=$wg"; } \
  | sh_ "$HEAD" "umask 077; cat > ~/mia_kit/.env"
say "   kit $(git -C "$KIT" rev-parse --short HEAD 2>/dev/null), GIDs $hg / $wg"

say "4. start (weights already on both boxes: no sync, no build, no image ship)"
sh_ "$HEAD" "cd ~/mia_kit && SKIP_SYNC=1 SKIP_BUILD=1 SKIP_SHIP=1 ./start.sh" > "$OUT/start.log" 2>&1 \
  || { tail -30 "$OUT/start.log" | tee -a "$OUT/run.log"; die "her start.sh failed (full log: $OUT/start.log)"; }
sh_ "$HEAD" "curl -sf localhost:$PORT/v1/models" > "$OUT/models.json" || die "no /v1/models on :$PORT"
for n in "$HEAD" "$WORKER"; do say "   $n: $(sh_ "$n" "free -g | awk '/Mem:/{print \"available\", \$7, \"GiB\"}'")"; done

say "5. measure: kit_bench set b + DSpark counters (no oracle: different weights)"
HEAD="$HEAD" PORT="$PORT" MODEL="$MODEL" SERVE_ID=mia_kit_stock OUT="$OUT" QUICK=1 NO_ORACLE=1 \
  "$HERE/serve_speed_suite.sh" 2>&1 | tee -a "$OUT/run.log"

if [ "${KEEP:-0}" != 1 ]; then
  say "6. stop her serve"
  sh_ "$HEAD" "cd ~/mia_kit && ./start.sh stop" >> "$OUT/run.log" 2>&1
fi
say "done: $OUT"
