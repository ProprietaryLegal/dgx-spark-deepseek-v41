# SPDX-License-Identifier: Apache-2.0
"""Pick the decode-speed levers that won their one-at-a-time A/B (p_levers_ab.sh) for a combined run.

Per lever, from speed/<tag>_<lever>/ against speed/<tag>_base/: kit_bench set b tok/s per prompt, and the oracle
(24 greedy continuations + teacher-forced top-20 logprobs) compared as oracle_compare.py does. A lever is picked when:
  - its verify rounds are faster: ms per round (kit_bench chunks = rounds, scripts/bench/round_ms.py) on structured
    >= MIN_GAIN better and the geometric mean over the three prompts not worse. Rounds, not tok/s: a lever that
    changes numerics changes the greedy text and with it the acceptance, which is noise here (the shared-slot lever on
    v3: prose 2.11 -> 1.94 tokens a round), while ms per round is the kernel's own effect;
  - levers that leave the target's text alone and change tokens a round on purpose (DRAFTER: a drafter-only change,
    whose acceptance change is its own effect, not text noise, a cheaper drafter that proposes worse can lose, mainproj
    did; or a depth-policy cost table, which moves depths) are judged on tok/s instead, with the same gains, and must
    give all 24 greedy continuations identical;
  - quality holds: levers that keep the math (EXACT) must give all 24 greedy continuations identical; the others need
    top-1 agreement >= MIN_TOP1 and truncated KL <= MAX_KL against the baseline. Those bars are set at routing-flip
    noise, not at bit-closeness: any change upstream of a router, even 1e-6 in its logits, flips near-tied 6th / 7th
    experts and the flips cascade (the router lever on v3: top-1 97.66%, KL 0.013 with fp32-identical math), the size
    of the prefill path's run-to-run spread before deterministic prefill (97.9%, 0.0137, measured Oct 9); mHC overlap
    (vLLM's own TF32 prenorm path on family-100 Blackwell) lands at 0.020. A real error shows far past it; quality
    against native stays the panel KLD's and the eval suites' job.
Details go to stderr; the last stdout line is the picked levers, space-separated (empty if none).
  python3 levers_pick.py <speed dir> <tag> <lever> [<lever> ...]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_compare import compare, load  # noqa: E402

EXACT = {"runflags", "spin", "flagw", "engram", "l2pfhc", "l2pfhc20", "mrowcp", "mrowp16", "smoel2", "smoeef", "gatefirst", "gatefirst_l2pfhc", "l2pf24", "l2pfr0", "mrowgrp"}
DRAFTER = {"mainproj", "drafterk4", "newtbl"}   # judged on tok/s: drafter-only, or a depth table
MIN_GAIN = float(os.environ.get("MIN_GAIN", "0.005"))
MIN_TOP1 = float(os.environ.get("MIN_TOP1", "0.97"))
MAX_KL = float(os.environ.get("MAX_KL", "0.025"))


def tps(d, tokens=False):
    """rounds a second per prompt (1000 / ms per round), so a higher number is a faster round; tokens=True: tok/s"""
    p = os.path.join(d, "kit_decode_b.json")
    if not os.path.exists(p):
        return None
    out = {}
    for r in json.load(open(p))["result"]:
        n, c, t = r["tokens_all"][0], r["chunks_all"][0], r["decode_tps_median"]
        out[r["prompt"]] = t if tokens else t * c / n
    return out


def main():
    speed, tag, levers = sys.argv[1], sys.argv[2], sys.argv[3:]
    base_dir = os.path.join(speed, f"{tag}_base")
    base = tps(base_dir)
    if not base:
        print(f"no baseline in {base_dir}", file=sys.stderr)
        print("")
        return
    ref_path = os.path.join(base_dir, "kit_oracle.jsonl")
    ref = load(ref_path) if os.path.exists(ref_path) else None
    print("base: " + ", ".join(f"{k} {1000 / v:.1f}" for k, v in base.items()) + " ms a round", file=sys.stderr)
    picked = []
    for lv in levers:
        d = os.path.join(speed, f"{tag}_{lv}")
        drafter = lv in DRAFTER
        t, b = (tps(d, True), tps(base_dir, True)) if drafter else (tps(d), base)
        if not t or set(t) != set(b):
            print(f"{lv:9s}: no result", file=sys.stderr)
            continue
        ratio = {k: t[k] / b[k] for k in b}
        gm = math.exp(sum(math.log(r) for r in ratio.values()) / len(ratio))
        cand = os.path.join(d, "kit_oracle.jsonl")
        if ref is None or not os.path.exists(cand):
            print(f"{lv:9s}: no oracle file, not picked", file=sys.stderr)
            continue
        q = compare(ref, load(cand))
        same, total, top1, kl = q["identical_greedy"], q["prompts"], q["top1_agree"], q["kl_mean"]
        quality = same == total if (lv in EXACT or drafter) else (top1 >= MIN_TOP1 and kl <= MAX_KL)
        speed_ok = ratio.get("structured", 0) >= 1 + MIN_GAIN and gm >= 1.0
        verdict = "PICK" if (quality and speed_ok) else ("slower or flat" if quality else "quality changed")
        print(f"{lv:9s}: {'tok/s' if drafter else 'rounds'} " + " ".join(f"{k} x{r:.3f}" for k, r in ratio.items()) + f" (geomean x{gm:.3f}); "
              f"oracle {same}/{total} identical, top-1 {top1:.4f}, KL {kl:.5f} -> {verdict}", file=sys.stderr)
        if verdict == "PICK":
            picked.append(lv)
    print(" ".join(picked))


if __name__ == "__main__":
    main()
