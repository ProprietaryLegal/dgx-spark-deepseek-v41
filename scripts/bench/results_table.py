# SPDX-License-Identifier: Apache-2.0
"""Markdown result tables straight from the raw files, so no number in the README is typed by hand.

  python3 results_table.py --kld ours=data/kld/ours_v3_ship_on_our_panel.json --kld mia=<its kld.json> --kld 0xsero=<its kld.json> \
      --speed ours=<assets>/speed/sweep_v21_k3_pw --speed mia=<assets>/speed/mia_kit_k3

--kld NAME=kld.json (p4_kld.py output; head variants inside it, if any, become their own rows), --speed NAME=dir (a
serve_speed_suite.sh output dir: kit_decode_b.json, acceptance.txt, kit_prefill.json, kit_depth.json, kit_conc.json when
present). A missing file prints as "missing", never as a guess.
"""
import argparse
import json
import os


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def kld_rows(name, d):
    """The build's own row, then each head variant p4_kld.py scored in the same pass (top-level head_* entries)."""
    rows = [(name, d)]
    for k, v in d.items():
        if k.startswith("head_") and isinstance(v, dict) and "kld_mean" in v:
            rows.append((f"{name}, {k.replace('_', ' ')}", v))
    return rows


def fmt(v, spec):
    return "missing" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kld", action="append", default=[], help="NAME=path/to/kld.json")
    ap.add_argument("--speed", action="append", default=[], help="NAME=serve_speed_suite output dir")
    a = ap.parse_args()
    if a.kld:
        print("| build | KLD vs native (mean, nats) | KLD, domains weighted equally | KLD p99 | top-1 agreement | PPL on the panel |")
        print("|---|---|---|---|---|---|")
        native_ppl, domains = None, {}
        for spec in a.kld:
            name, path = spec.split("=", 1)
            d = load(path)
            if d is None:
                print(f"| {name} | missing | missing | missing | missing | missing |")
                continue
            native_ppl = native_ppl or d.get("ppl_native")
            bd = d.get("by_domain") or {}
            balanced = sum(v["kld"] for v in bd.values()) / len(bd) if bd else None   # the panel is ~65% code positions
            for rn, r in kld_rows(name, d):
                main = r is d                                      # variants carry only KLD mean and top-1
                g = lambda k, spec: fmt(r.get(k), spec) if (main or k in r) else "-"      # noqa: E731
                print(f"| {rn} | {g('kld_mean', '.4f')} | {fmt(balanced, '.4f') if main and balanced is not None else '-'} | "
                      f"{g('kld_p99', '.3f')} | "
                      f"{fmt(r.get('top1_agreement') and 100 * r['top1_agreement'], '.2f')}% | {g('ppl_ours', '.4f')} |")
            domains[name] = d.get("by_domain") or {}
        if native_ppl:
            print(f"\nNative PPL on the same panel: {native_ppl:.4f}.\n")
        keys = sorted({k for v in domains.values() for k in v})
        if keys:
            print("| KLD by domain | " + " | ".join(domains) + " |")
            print("|---|" + "---|" * len(domains))
            for k in keys:
                print(f"| {k} | " + " | ".join(fmt((domains[n].get(k) or {}).get("kld"), ".4f") for n in domains) + " |")
            print()
    if a.speed:
        print("| configuration | code tok/s | prose tok/s | structured tok/s | reps | DSpark acceptance |")
        print("|---|---|---|---|---|---|")
        for spec in a.speed:
            name, d = spec.split("=", 1)
            dec = load(os.path.join(d, "kit_decode_b.json"))
            rows = {r["prompt"]: r["decode_tps_median"] for r in (dec or {}).get("result", [])}
            reps = sorted({len(r.get("decode_tps_all", [])) for r in (dec or {}).get("result", [])})
            try:
                acc = open(os.path.join(d, "acceptance.txt")).read().strip().split(": ", 1)[-1]
            except OSError:
                acc = "missing"
            f = lambda k: fmt(rows.get(k), ".1f")                  # noqa: E731
            print(f"| {name} | {f('code')} | {f('prose')} | {f('structured')} | {'/'.join(map(str, reps)) or 'missing'} | {acc} |")
        print("\nkit_bench set b (greedy, thinking off, 384 tokens), median over the reps a prompt, single stream.")


if __name__ == "__main__":
    main()
