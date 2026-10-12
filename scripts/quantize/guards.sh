#!/usr/bin/env bash
# Memory guards for long GPU jobs on a Spark, in tmux (they die with a reboot, so start them again after one):
#   dropcache  drop the page cache when free memory is under 20 GiB AND at least 4 GiB of it is reclaimable (Buffers +
#              Cached - Shmem; GB10 CUDA allocations cannot reclaim it), logged in ~/dsv41_recipe/logs/dropcache.log.
#              The reclaimable test matters: a GPU job's shared memory counts as "cache" but cannot be dropped, and a
#              guard keyed on `free` alone then syncs and drops every 5 s for nothing (898 futile drops on spark2,
#              Oct 8-9). Page cache only (echo 1): dentries / inodes are not what starves CUDA.
#   memwatch   used / available GiB every 10 s into ~/dsv41_recipe/logs/memwatch_<name>.log
#   servewatch kill the dsv41-serve container when MemAvailable < 1.5 GiB (GB10 serves creep driver-side memory until
#              the SoC wedges; a dead serve costs a restart, a wedged box a power cycle), logged in logs/servewatch.log
#   bash guards.sh start [name]        start dropcache + memwatch if not running (name defaults to the hostname)
#   bash guards.sh serve [name]        for a serve: stop dropcache (it would evict Engram's hot rows), start memwatch + servewatch
set -u
L=~/dsv41_recipe/logs
case "${1:-start}" in
  start)
    mkdir -p "$L"
    me=$(readlink -f "$0")
    tmux has-session -t dropcache 2>/dev/null || tmux new-session -d -s dropcache "bash $me loop-dropcache"
    tmux has-session -t memwatch 2>/dev/null || tmux new-session -d -s memwatch "bash $me loop-memwatch ${2:-$(hostname)}"
    tmux ls 2>/dev/null | grep -E '^(dropcache|memwatch):' ;;
  loop-dropcache)
    while true; do
      read -r f r < <(awk '/^MemFree:/{m=$2} /^Buffers:/{b=$2} /^Cached:/{c=$2} /^Shmem:/{s=$2}
                            END{printf "%d %d\n", m / 1048576, (b + c - s) / 1048576}' /proc/meminfo)
      if [ "$f" -lt 20 ] && [ "$r" -ge 4 ]; then
        sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
        echo "$(date +%T) dropped page cache (free was $f GiB, $r GiB reclaimable)" >> "$L/dropcache.log"
      fi
      sleep 5
    done ;;
  serve)
    mkdir -p "$L"
    me=$(readlink -f "$0")
    tmux kill-session -t dropcache 2>/dev/null
    tmux has-session -t memwatch 2>/dev/null || tmux new-session -d -s memwatch "bash $me loop-memwatch ${2:-$(hostname)}"
    tmux has-session -t servewatch 2>/dev/null || tmux new-session -d -s servewatch "bash $me loop-servewatch"
    tmux ls 2>/dev/null | grep -E '^(dropcache|memwatch|servewatch):' ;;
  loop-servewatch)
    while true; do
      a=$(awk '/^MemAvailable:/{printf "%d", $2 / 1024}' /proc/meminfo)     # MiB
      if [ "$a" -lt 1536 ] && docker ps -q --filter name=^dsv41-serve$ | grep -q .; then
        docker rm -f dsv41-serve >/dev/null 2>&1
        echo "$(date +%T) MemAvailable $a MiB: killed dsv41-serve" >> "$L/servewatch.log"
      fi
      sleep 2
    done ;;
  loop-memwatch)
    while true; do
      echo "$(date +%H:%M:%S) $(free -g | awk '/Mem:/{print "used", $3, "avail", $7}')" >> "$L/memwatch_$2.log"
      sleep 10
    done ;;
  *) echo "usage: $0 start [name]"; exit 2 ;;
esac
