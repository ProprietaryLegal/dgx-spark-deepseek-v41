#!/usr/bin/env bash
# Input Hessians of the DSpark drafter's dense linears (layers 40-42: fused wq_a + wkv, wq_b, wo_b, shared gate_up /
# down) for their EXL3 encode (p4_drafter.py) and the drafter experts' prior (p4_drafter_experts.py reads the
# shared expert's gate_up input). An eager serve (a CUDA-graph replay runs no Python, so the hook would see nothing)
# with VLLM_EXL3_HCAP drafts on 36 calibration prompts (split "cal", no tools, no long-context rows, spread over the
# domains, each in its own thinking mode, 320 tokens each: the Oct 10 capture's size, 36 prompts / ~10K tokens), then
# SAVE on both nodes (each mounts its own cache) and a few more requests so each rank's hook sees it. rank 1's files
# are copied next to rank 0's on the head node, where p4_drafter.py reads both (wo_b and down are row-parallel).
# main_proj is left out of the match: the shipping composition keeps it FP8, and its H alone is ~0.9 GB.
# Run from the control machine with nothing else on either GPU.
#   MODEL_DIR=/home/spark/models/<target with the official FP8 drafter> scripts/quantize/hcap_drafter.sh
# Out: ~/.cache/dsv41-serve/hcap_drafter on the head node (every module's .rank0.pt and .rank1.pt).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_DIR=${MODEL_DIR:?the target model dir, its drafter still FP8}
HEAD=${HEAD:-spark}; WORKER=${WORKER:-spark2}; FAB_IF=${FAB_IF:-enp1s0f0np0}
N=${N:-36}; TOKENS=${TOKENS:-320}
CONV=${CONV:-$REPO/data/caldata/conversations.jsonl}
MATCH='layers\.(4[0-9])\.(attn\.(fused_wqa_wkv|wq_b|wo_b)|ffn\.shared_experts\.(gate_up_proj|down_proj))$'
D=.cache/dsv41-serve/hcap_drafter
export SKIP_SERVE_HOLD=1
# The launcher passes DOCKER_ENV to docker through the remote login shell unquoted, and the Sparks log in to zsh, which
# globs MATCH's [0-9] and parentheses ("no matches found", Oct 19 02:41): MATCH goes in single quotes, the remote
# shell strips them and docker gets the plain regex.
say() { echo "$(date +%T) $*"; }
for h in $HEAD $WORKER; do ssh $h "rm -rf ~/$D; mkdir -p ~/$D"; done
say "eager serve with the drafter capture on $MODEL_DIR"
IMAGE=${IMAGE:-dsv41-serve:v4} MODEL_DIR=$MODEL_DIR GRAPH='{"cudagraph_mode":"NONE"}' \
  DOCKER_ENV="VLLM_EXL3_HCAP=/root/.cache/hcap_drafter VLLM_EXL3_HCAP_MATCH='$MATCH'" \
  timeout 1800 "$REPO/scripts/launch/dsv41_tp2.sh" up 2>&1 | tail -2
host=$(ssh -G $HEAD | awk '/^hostname /{print $2}')
gen() {   # $1 = how many prompts, $2 = offset into the picked list
  python3 - "$CONV" "$host" "$1" "$2" "$TOKENS" <<'EOF'
import collections, json, sys, threading, urllib.request
conv, host, want, off, tokens = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rows = [json.loads(l) for l in open(conv)]
rows = [r for r in rows if r["split"] == "cal" and not r.get("tools") and r["domain"] != "long_context"]
by = collections.defaultdict(list)
for r in sorted(rows, key=lambda r: r["id"]):
    by[r["domain"].split("_")[0]].append(r)
pick, i = [], 0
while len(pick) < len(rows):                       # round robin over the domains, every row once
    for d in sorted(by):
        if i < len(by[d]):
            pick.append(by[d][i])
    i += 1
pick = [pick[(off + j) % len(pick)] for j in range(want)]
done, lock = [0], threading.Lock()
def one(r):
    body = {"model": "dsv41", "messages": [r["messages"][0]], "max_tokens": tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": bool(r["thinking"])}}
    req = urllib.request.Request(f"http://{host}:8000/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(req, timeout=1800)).get("usage", {})
    with lock:
        done[0] += u.get("completion_tokens", 0)
q = list(pick)
def worker():
    while True:
        with lock:
            if not q:
                return
            r = q.pop(0)
        one(r)
ts = [threading.Thread(target=worker) for _ in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
print(f"{want} prompts, {done[0]} generated tokens, domains {sorted({r['domain'] for r in pick})}", flush=True)
EOF
}
gen "$N" 0
for h in $HEAD $WORKER; do ssh $h "touch ~/$D/SAVE"; done
# The hook looks for SAVE every 64 calls of a module; the drafter's linears run several times a forward, so a couple
# of requests cross a check on both ranks.
for hr in $HEAD:0 $WORKER:1; do
  h=${hr%%:*}; r=${hr#*:}; t0=$(date +%s)
  until ssh $h "test -f ~/$D/saved.rank$r"; do
    [ $(( $(date +%s) - t0 )) -gt 600 ] && { say "$h: no saved.rank$r after 10 min"; exit 1; }
    gen 2 $((RANDOM % 30)) >/dev/null
  done
done
"$REPO/scripts/launch/dsv41_tp2.sh" down >/dev/null 2>&1
hip=$(ssh $HEAD "ip -4 -o addr show $FAB_IF | awk '{print \$4}' | cut -d/ -f1")
ssh $WORKER "rsync -rlt ~/$D/*.rank1.pt $hip:$D/"     # root-owned files: -rlt (no chown)
ssh $HEAD "ls ~/$D | sed -n '1,40p'; du -sh ~/$D"
say "capture done"
