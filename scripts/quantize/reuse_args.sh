# Sourced by run_pass1_2box.sh, resume_part.sh and solo_from.sh: p4_pass1.py's partial re-encode arguments from REUSE
# and OLD_PLAN, each one path or a comma list of the same length (an expert comes from the first checkpoint whose plan
# gives it the new plan's K). Every OLD_PLAN sits next to PLAN, whose dir the container sees as /plan. Sets RMOUNT (the
# docker -v flags, checkpoint i at /reuse<i>), RARGS (--reuse / --old-plan) and REUSE_HASH_FILES (what both boxes must
# hold identically: each old plan and each checkpoint's index). Returns 1 with a message on a bad pairing.
reuse_args() {
  RMOUNT=""; RARGS=""; REUSE_HASH_FILES=""
  [ -n "${REUSE:-}" ] || return 0
  local -a ru op
  local i rl="" ol=""
  IFS=, read -ra ru <<< "$REUSE"
  IFS=, read -ra op <<< "${OLD_PLAN:-}"
  [ "${#ru[@]}" = "${#op[@]}" ] || { echo "REUSE lists ${#ru[@]} checkpoints, OLD_PLAN ${#op[@]} plans: they pair up"; return 1; }
  for i in "${!ru[@]}"; do
    [ "$(dirname "${op[$i]}")" = "$(dirname "$PLAN")" ] || { echo "REUSE needs OLD_PLAN next to PLAN: ${op[$i]}"; return 1; }
    RMOUNT="$RMOUNT -v ${ru[$i]}:/reuse$i:ro"
    rl="$rl${rl:+,}/reuse$i"
    ol="$ol${ol:+,}/plan/$(basename "${op[$i]}")"
    REUSE_HASH_FILES="$REUSE_HASH_FILES ${op[$i]} ${ru[$i]}/model.safetensors.index.json"
  done
  RARGS="--reuse $rl --old-plan $ol"
}
