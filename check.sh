#!/usr/bin/env bash
# Doctor for the two Sparks, run from the control machine; read-only. Prints one line per check, PASS / WARN / FAIL, and
# exits 1 on any FAIL.
#   HEAD=<rank0> WORKER=<rank1> MODEL_DIR=/abs/path ./check.sh          the boxes (MODEL_DIR optional)
#   HEAD=<rank0> ./check.sh serve                                      + a live serve: /v1/models, a chat, a tool call
# Same knobs as the launcher: HEAD WORKER FAB_IF FAB_HCA IMAGE PORT SERVED.
set -uo pipefail
HEAD=${HEAD:-spark}; WORKER=${WORKER:-spark2}
FAB_IF=${FAB_IF:-enp1s0f0np0}; FAB_HCA=${FAB_HCA:-rocep1s0f0}
IMAGE=${IMAGE:-dsv41-serve:v5}; PORT=${PORT:-8000}; SERVED=${SERVED:-dsv41}
MODEL_DIR=${MODEL_DIR:-}
fails=0
pass() { printf 'PASS  %-8s %s\n' "$1" "$2"; }
warn() { printf 'WARN  %-8s %s\n' "$1" "$2"; }
fail() { printf 'FAIL  %-8s %s\n' "$1" "$2"; fails=$((fails + 1)); }
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2" 2>/dev/null; }

box() {   # $1 host
  local h=$1 v ip gid
  sh_ "$h" true || { fail "$h" "ssh without a password"; return; }
  pass "$h" "ssh"
  v=$(sh_ "$h" "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
  case "$v" in *GB10*) pass "$h" "GPU: $v";; *) fail "$h" "no GB10 found (nvidia-smi: '${v:-nothing}')";; esac
  case "$v" in *", 580."*) ;; *GB10*) warn "$h" "driver not 580.x (tested: 580.142, 580.173.02)";; esac
  sh_ "$h" "docker info >/dev/null && { command -v nvidia-container-cli || command -v nvidia-ctk; }" >/dev/null \
    && pass "$h" "docker and the NVIDIA container toolkit" || fail "$h" "docker or the NVIDIA container toolkit missing"
  ip=$(sh_ "$h" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1")
  [ -n "$ip" ] && pass "$h" "IPv4 on $FAB_IF" || fail "$h" "no IPv4 on $FAB_IF"
  gid=$(sh_ "$h" "want=$ip; P=/sys/class/infiniband/$FAB_HCA/ports/1
    hex=\$(printf '%02x%02x%02x%02x' \$(echo \$want | tr '.' ' '))
    for i in \$(seq 0 15); do t=\$(cat \$P/gid_attrs/types/\$i 2>/dev/null); g=\$(cat \$P/gids/\$i 2>/dev/null)
      case \"\$t\" in *v2*) ;; *) continue;; esac
      case \"\$g\" in *ffff:\${hex:0:4}:\${hex:4:4}*) echo \$i; break;; esac; done")
  [ -n "$gid" ] && pass "$h" "RoCE v2 GID $gid on $FAB_HCA for that IPv4" || fail "$h" "no RoCE v2 GID on $FAB_HCA for $FAB_IF's IPv4"
  sh_ "$h" "sudo -n true" && pass "$h" "sudo -n (the launcher's page-cache drop)" \
    || warn "$h" "no sudo -n: the launcher cannot drop the page cache before boot (CUDA may run out of memory)"
  v=$(sh_ "$h" "docker image inspect $IMAGE --format '{{.Id}}'")
  [ -n "$v" ] && pass "$h" "image $IMAGE ${v:7:12}" || fail "$h" "image $IMAGE missing"
  eval "img_$2=\$v"
  if [ -n "$MODEL_DIR" ]; then
    v=$(sh_ "$h" "cd '$MODEL_DIR' 2>/dev/null && ls config.json exl3_moe_plan.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors 2>/dev/null | wc -l")
    [ "$v" = 4 ] && pass "$h" "MODEL_DIR has the config, the plan and both Engram shards" \
      || fail "$h" "MODEL_DIR incomplete on this box (config, plan, Engram shards: $v of 4)"
  fi
  v=$(sh_ "$h" "df -BG --output=avail ~ | tail -1 | tr -dc 0-9")
  [ "${v:-0}" -ge 30 ] && pass "$h" "free disk ${v} GiB in ~" || warn "$h" "free disk ${v:-?} GiB in ~"
  v=$(sh_ "$h" "free -g | awk '/Mem:/{print \$7}'")
  if docker_up=$(sh_ "$h" "docker ps --format '{{.Names}}' | grep -x dsv41-serve"); [ -n "$docker_up" ]; then
    [ "${v:-0}" -ge 4 ] && pass "$h" "serving, MemAvailable $v GiB" || warn "$h" "serving, MemAvailable $v GiB (want >= 4)"
  else
    [ "${v:-0}" -ge 100 ] && pass "$h" "idle, MemAvailable $v GiB" || warn "$h" "idle, MemAvailable $v GiB: something else holds memory"
  fi
}

serve() {
  local base="http://$(ssh -G "$HEAD" | awk '/^hostname /{print $2}'):$PORT/v1" r
  r=$(curl -s -m 10 "$base/models") && echo "$r" | grep -q "\"$SERVED\"" && pass serve "/v1/models lists $SERVED" \
    || { fail serve "/v1/models at $base"; return; }
  r=$(curl -s -m 120 "$base/chat/completions" -H 'Content-Type: application/json' -d "{\"model\": \"$SERVED\",
    \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false},
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: ready\"}]}")
  echo "$r" | grep -qi 'ready' && pass serve "chat completion" || fail serve "chat completion: ${r:0:200}"
  r=$(curl -s -m 120 "$base/chat/completions" -H 'Content-Type: application/json' -d "{\"model\": \"$SERVED\",
    \"max_tokens\": 256, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false},
    \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Weather for a city\",
      \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}],
    \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in Bangkok? Use the tool.\"}]}")
  echo "$r" | grep -q '"tool_calls"' && echo "$r" | grep -q 'get_weather' && pass serve "tool call" \
    || fail serve "tool call: ${r:0:200}"
}

if [ "${1:-}" = serve ]; then
  serve
else
  box "$HEAD" 0
  box "$WORKER" 1
  [ -n "${img_0:-}" ] && [ "${img_0:-}" = "${img_1:-}" ] && pass both "same image id on both boxes" \
    || { [ -n "${img_0:-}${img_1:-}" ] && warn both "image ids differ: two separate builds; docker save | ssh <rank1> docker load gives one"; }
  sh_ "$HEAD" "ssh -o BatchMode=yes -o ConnectTimeout=5 $(sh_ "$WORKER" "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1") true" \
    && pass both "ssh $HEAD -> $WORKER over the fabric" || warn both "no ssh from $HEAD to $WORKER over the fabric (weight copies, encodes)"
fi
[ "$fails" = 0 ] || { echo "$fails check(s) failed"; exit 1; }
