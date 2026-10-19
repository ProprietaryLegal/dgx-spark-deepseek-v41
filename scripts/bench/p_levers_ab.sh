#!/usr/bin/env bash
# Decode-speed levers on the live serve: a baseline of one image and one weight set, then each lever alone,
# then any combination asked for. Per configuration: set b with DSpark acceptance and the oracle (serve_speed_suite.sh
# QUICK=1), and oracle_compare.py against the baseline: top-1 agreement and truncated KL of the serving path (exactly
# 0 / 100% for the levers that keep the math: run flags, remote spin, flag-write, Engram spin reader; small for those
# that round differently: router, shared slot, mHC overlap). Run from the control machine with nothing else on the
# Sparks; ~11 min a config.
#   MODEL_DIR=/home/spark/models/dsv41-recipe-exl3-v3-dk5 IMAGE=dsv41-serve:v4 scripts/bench/p_levers_ab.sh
#   ... p_levers_ab.sh router shared                 (only these levers)
#   COMBO="router runflags spin" ... p_levers_ab.sh -   (the baseline, then just this combination)
#   SKIP_BASE=1 reuses speed/<TAG>_base from an earlier run (same image, weights and BASE_ENV) instead of booting it
#   again; scripts/bench/levers_pick.py picks the combination from the one-at-a-time results.
# cstate (a system setting, the operator's call: runs only with CSTATE_OK=1): the deep CPU idle states LPI-2 / LPI-3 (231 /
# 433 us exit latency on GB10) disabled on both Sparks for that configuration's run, re-enabled after it.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL_DIR=${MODEL_DIR:?model dir on both Sparks}; IMAGE=${IMAGE:-dsv41-serve:v4}; TAG=${TAG:-p}
BASE_ENV=${BASE_ENV:-"VLLM_DSPARK_LOCAL_ARGMAX=1"}        # on top of the launcher's recipe defaults, in every config
COMBO=${COMBO:-}
ASSETS=${DSV41_RESULTS:-$(pwd)/results}   # the cd above put us at the repo root
declare -A LEVER=(
  [router]="EXL3_ROUTER=1"
  [shared]="VLLM_EXL3_MOE_SMOE_SHARED=1"
  [runflags]="EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1"
  [spin]="VLLM_MQ_REMOTE_SPIN_MS=200"
  [mhc]="VLLM_DSV41_MHC_OVERLAP=1"
  [flagw]="EXL3_RDMA_FLAGW=1"
  [engram]="VLLM_ENGRAM_DISK_SPIN_MS=1000"
  [fused]="VLLM_EXL3_MOE_FUSED=1"
  [cstate]=""
  # image v5 (patch 0017 + plugin of Oct 11 05:50): on top of a BASE_ENV that holds the picked levers
  [l2pfhc]="EXL3_L2PF_HC=1"
  [l2pfhc20]="EXL3_L2PF_HC=1 EXL3_L2PF_MB=20"
  [mrowcp]="EXL3_MROW_FLAGS=-DMROW_CPASYNC=1"
  [mrowp16]="EXL3_MROW_FLAGS=-DMROW_PIPE=16"
  [smoel2]="EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1,-DSMOE_L2HINT=256"
  [smoeef]="EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1,-DSMOE_EVICT_FIRST=1"
  [mainproj]=""        # no env: the drafter's main_proj on EXL3 K5 comes from the model dir (MODEL_DIR=...-v3-dk5mp)
  [gatefirst]="VLLM_MOE_GATE_FIRST=1"       # patch 0018: router GEMM before the shared-expert fork
  [gatefirst_l2pfhc]="VLLM_MOE_GATE_FIRST=1 EXL3_L2PF_HC=1"
  [l2pf24]="EXL3_L2PF_MB=24"                # the attn fork then covers wo_a and all of wo_b (23.6 MiB)
  [v3s]=""             # no env: the speed-aware expert budget comes from the model dir (MODEL_DIR=...-v3s<cap>-dk5)
  [engram4]=""         # no env: Engram wkv on EXL3 K4 comes from the model dir (MODEL_DIR=...-eng4, p4_engram_variant.py)
  [v3r]=""             # no env: dense re-encoded on the v3 experts, K5 + head K6 (MODEL_DIR=...-v3r-dk5, the chain's)
  [v3b]=""             # no env: the bytes set, dense K4 (wo_a K5) + head K5 (MODEL_DIR=...-v3b-dk5, the chain's)
  [mrowgrp]="EXL3_MROW_GROUPED=1"          # wo_a reads its head groups in place (no [G, T, D] -> [T, G * D] copy a layer)
  [newtbl]="VLLM_DSPARK_ROUND_MS=$(cat "${NEWTBL_FILE:-/dev/null}" 2>/dev/null)"   # a re-measured depth-policy cost table (NEWTBL_FILE)
  [fidec]="FLASHINFER_AUTOTUNE_DIR=/root/.cache/fi_autotune_dec16"   # FlashInfer SM120 sparse MLA: decode kernel for verify rows
                       # (its Oct 8 GB10 calibration sends > 4-8 tokens to prefill_mg_dual; this copy raises 32/64-head crossovers to 16)
  [tverify]="VLLM_DSV41_TRITON_VERIFY=1"     # patch 0022: verify rows through our Triton sparse attention, not FlashInfer
  [mhcfuse]="VLLM_DSV41_MHC_FUSED_POST=1"     # patch 0021: mHC post + input collapse + RMSNorm in one launch
  [shortctx]="VLLM_SHORT_CTX_BOUNDS=512,1025"   # patch 0019: decode graph twins that skip the indexer's scoring
  [rtrhc]="EXL3_ROUTER=1 EXL3_L2PF_HC=1"     # our router GEMV, its gate prefetched to L2
  [l2pfr0]="EXL3_L2PF_RATE_GBPS=0"          # prefetch forks issue at once (the next wq_a lands before its kernel)
  [flatv]="VLLM_DSV41_FLAT_VERIFY=1"        # patch 0020: verify rows through the one-row sparse-MLA decode kernel
  [dk3]=""            # no env: the drafter's dense on EXL3 K3 comes from the model dir (MODEL_DIR=...-dk3, p4_drafter_variant.py)
  [dk4]=""            # the same at K4
  [dx3]=""            # no env: the drafter's routed experts on EXL3 K3 come from the model dir (...-dx3, p4_drafter_experts_variant.py)
  [dx4]=""            # the same at K4
  [pcores]="EXL3_CPU_PIN=5-9,15-19"   # every vLLM process on GB10's Cortex-X925 cores (plugin __init__, measured Oct 11)
  [woa4]=""           # no env: the dense wo_a at K4 comes from the model dir (...-v3c-dk5-h4-dx3, session 9)
)
cstate() {   # $1: 1 = disable LPI-2 / LPI-3 on every core of both Sparks, 0 = enable them again
  for h in spark spark2; do
    ssh -o BatchMode=yes $h "for c in /sys/devices/system/cpu/cpu*/cpuidle/state2 /sys/devices/system/cpu/cpu*/cpuidle/state3; do
      echo $1 | sudo -n tee \$c/disable >/dev/null; done; cat /sys/devices/system/cpu/cpu0/cpuidle/state3/disable" | sed "s/^/   $h LPI-3 disabled: /"
  done
}
# flagw only when named: it needs P2 stage 0 (scripts/bench/rdma_stage0.sh: every sum checked) to have passed first
if [ "${1:-}" = "-" ]; then names=""; else names=${*:-"router shared runflags spin mhc engram fused"}; fi
SUM=$ASSETS/speed/${TAG}_levers_summary.txt
throttle() {   # cumulative GPU slowdown counters (us) per box: SW power cap, SW thermal, HW thermal
  for h in spark spark2; do
    ssh -o BatchMode=yes $h "nvidia-smi -q -d PERFORMANCE" 2>/dev/null \
      | awk -v h=$h '/SW Power Capping/{p=$(NF-1)} /SW Thermal Slowdown/{t=$(NF-1)} /HW Thermal Slowdown/{x=$(NF-1)} END{print h, p+0, t+0, x+0}'
  done
}
run() {   # $1 id, $2 extra env
  scripts/launch/dsv41_tp2.sh down >/dev/null 2>&1
  echo "== $1: ${2:-(baseline)} ($(date +%H:%M))" | tee -a "$SUM"
  # timeouts: a lever that hangs the serve (boot ~7 min, set b QUICK ~5) fails its configuration instead of the run
  if ! IMAGE=$IMAGE MODEL_DIR=$MODEL_DIR DOCKER_ENV="$BASE_ENV $2" timeout 1500 scripts/launch/dsv41_tp2.sh up 2>&1 | tail -1; then
    echo "   boot failed" | tee -a "$SUM"; return 1
  fi
  local t0; t0=$(throttle)
  SERVE_ID=${TAG}_$1 QUICK=1 timeout 2400 scripts/bench/serve_speed_suite.sh 2>&1 | grep -E '"prompt"|DSpark' | tee -a "$SUM"
  # GPU time spent power-capped / thermally slowed during this configuration's run (a capped rank makes the other wait)
  paste <(echo "$t0") <(throttle) | awk '{printf "   slowdown during the run: %s power-cap %.1f s, SW thermal %.1f s, HW thermal %.1f s\n", $1, ($6-$2)/1e6, ($7-$3)/1e6, ($8-$4)/1e6}' | tee -a "$SUM"
  if [ "$1" != base ] && [ -f "$ASSETS/speed/${TAG}_base/kit_oracle.jsonl" ]; then
    python3 scripts/bench/oracle_compare.py "$ASSETS/speed/${TAG}_base/kit_oracle.jsonl" \
      "$ASSETS/speed/${TAG}_$1/kit_oracle.jsonl" 2>&1 | tail -3 | sed 's/^/   oracle vs base: /' | tee -a "$SUM"
  fi
}
if [ "${SKIP_BASE:-0}" = 1 ] && [ -f "$ASSETS/speed/${TAG}_base/kit_decode_b.json" ]; then
  echo "== base: reused from $ASSETS/speed/${TAG}_base" | tee -a "$SUM"
else
  run base "" || exit 1
fi
for n in $names; do
  [ -n "${LEVER[$n]+x}" ] || { echo "unknown lever $n (${!LEVER[*]})"; continue; }
  if [ "$n" = cstate ]; then
    [ "${CSTATE_OK:-0}" = 1 ] || { echo "cstate skipped: a system setting, needs CSTATE_OK=1 (the operator's OK)" | tee -a "$SUM"; continue; }
    cstate 1; run cstate ""; cstate 0
    continue
  fi
  run "$n" "${LEVER[$n]}"
done
if [ -n "$COMBO" ]; then
  env=""; id="combo"
  for n in $COMBO; do env="$env ${LEVER[$n]}"; id="${id}_$n"; done
  run "$id" "$env"
fi
scripts/launch/dsv41_tp2.sh down >/dev/null 2>&1
echo "summary: $SUM"
