#!/usr/bin/env bash
# Host-side mirror for a 2-box pass 1 (run on each Spark, in tmux, as the normal user):
#   exchange_sync.sh <my_part> <peer> <run_dir>          e.g. exchange_sync.sh 0 <peer ConnectX IP> ~/dsv41_recipe/runs/p1
# Pushes this box's newest partial MoE output (exchange/Y_LXX_p<part>.bf16, or .pt from runs before Oct 8 23:00) and its
# encoded tensor files (layer_XX.part<part>.*) to the same path on the peer over the ConnectX link. Only the newest Y
# file is pushed, so a Y file deleted after use is never re-sent; tensor files are pushed once (--ignore-existing) and
# kept for assembly. Every pushed file is then evicted from the page cache on both boxes (rsync's read here, its write
# there): GB10 CUDA allocations cannot reclaim page cache, and ~8 GB of it landing per layer during the peer's exchange
# step is the likely cause of spark1 going silent on Oct 8 (12:18 and 22:25).
set -u
PART=$1 PEER=$2 RUN=${3%/}
SSH="ssh -T -o Compression=no -c aes128-gcm@openssh.com"
mkdir -p "$RUN/exchange"
evict() {   # $1 dir, then file names relative to it: flush + drop each file's pages here and on the peer
  local d=$1 f
  shift
  [ $# -gt 0 ] || return 0
  for f in "$@"; do sync -d "$d/$f" 2>/dev/null; dd if="$d/$f" iflag=nocache count=0 2>/dev/null; done
  $SSH "$PEER" "cd '$d' && for f in $*; do sync -d \"\$f\" 2>/dev/null; dd if=\"\$f\" iflag=nocache count=0 2>/dev/null; done"
}
while true; do
  y=$(ls -t "$RUN"/exchange/Y_L*_p"$PART".bf16 "$RUN"/exchange/Y_L*_p"$PART".pt 2>/dev/null | head -1)
  if [ -n "$y" ]; then
    sent=$(rsync -a --ignore-existing --out-format='%n' -e "$SSH" "$y" "$PEER:$RUN/exchange/" | grep -v '/$')
    [ -n "$sent" ] && evict "$RUN/exchange" $sent
  fi
  sent=$(rsync -a --ignore-existing --out-format='%n' --include "layer_*.part$PART.safetensors" \
    --include "layer_*.part$PART.json" --exclude "*" -e "$SSH" "$RUN/" "$PEER:$RUN/" 2>/dev/null | grep -v '/$')
  [ -n "$sent" ] && evict "$RUN" $sent
  sleep 3
done
