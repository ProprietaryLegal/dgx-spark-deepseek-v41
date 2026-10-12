# SPDX-License-Identifier: Apache-2.0
"""Generate native DeepSeek-V4.1-Flash conversations for our calibration set and the KLD panel, through OpenRouter,
pinned to providers serving the official FP8 / MXFP4 weights (no FP4 re-quantizations, no fallback outside the list). Resumable: done ids in the output JSONL are
skipped. Hard budget stop. The API key comes from OPENROUTER_API_KEY or an env file (an OPENROUTER_API_KEY= line) and
is never printed. --wiki: exllamav3 v1.6.0's exllamav3/conversion/standard_cal_data/wiki.utf8 (the long-context prompts).

  python3 generate.py out.jsonl [--wiki wiki.utf8] [--workers 8] [--budget-usd 5] [--key-file <env file>]
  python3 generate.py instruct.jsonl --set instruct          the v3 calibration supplement (prompts_instruct.py)
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import prompts
import prompts_instruct
from prompts import TOOL_RESULTS

SETS = {"base": prompts, "instruct": prompts_instruct}
FOLLOWUPS = prompts.FOLLOWUPS                   # replaced by the chosen set's in main()

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4.1-flash"
PROVIDERS = ["DeepInfra", "Together", "Parasail", "BaseTen", "SiliconFlow", "Novita", "CoreWeave", "GMICloud",
             "AtlasCloud", "StreamLake"]
lock = threading.Lock()
spent = [0.0]


def read_key(path):
    if not path:
        if os.environ.get("OPENROUTER_API_KEY"):
            return os.environ["OPENROUTER_API_KEY"].strip()
        raise SystemExit("set OPENROUTER_API_KEY or pass --key-file <env file with an OPENROUTER_API_KEY= line>")
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"no OPENROUTER_API_KEY in {path}")


def call(key, messages, tools, thinking, effort, max_tokens=6000, tries=5):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "usage": {"include": True},
            # DeepSeek's own endpoint is excluded by the account's no-training guardrail; these serve the official
            # FP8 / MXFP4 checkpoint ("fp8" on OpenRouter) and pass it. Only these, never an FP4 re-quantization.
            "provider": {"order": PROVIDERS, "allow_fallbacks": False}}
    if tools:
        body["tools"] = tools
    body["reasoning"] = {"effort": effort} if thinking else {"enabled": False}
    data = json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {key}",
                                                              "Content-Type": "application/json",
                                                              "X-Title": "dsv41-recipe calibration"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.load(r)
            if "choices" not in out:
                raise RuntimeError(str(out)[:300])
            return out
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as e:
            msg = e.read().decode()[:300] if isinstance(e, urllib.error.HTTPError) else str(e)[:300]
            if attempt == tries - 1:
                raise RuntimeError(f"failed after {tries}: {msg}")
            time.sleep(4 * 2 ** attempt)


def assistant_message(resp):
    m = resp["choices"][0]["message"]
    msg = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning"):
        msg["reasoning_content"] = m["reasoning"]
    if m.get("tool_calls"):
        msg["tool_calls"] = [{"id": t.get("id"), "type": "function",
                              "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}}
                             for t in m["tool_calls"]]
    return msg, resp["choices"][0].get("finish_reason")


def run_item(key, it, budget):
    if spent[0] > budget:
        return None
    msgs = list(it["messages"])
    meta = {"calls": 0, "cost": 0.0, "providers": [], "finish": []}

    def step():
        resp = call(key, msgs, it["tools"], it["thinking"], it["effort"])
        u = resp.get("usage", {})
        meta["calls"] += 1
        meta["cost"] += float(u.get("cost") or 0.0)
        meta["providers"].append(resp.get("provider"))
        a, fin = assistant_message(resp)
        meta["finish"].append(fin)
        msgs.append(a)
        return a

    a = step()
    rounds = 0
    while a.get("tool_calls") and rounds < 3:              # answer every call with a canned, plausible result
        for tc in a["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = TOOL_RESULTS.get(tc["function"]["name"], lambda _: {"error": "unknown tool"})
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(fn(args))})
        a = step()
        rounds += 1
    if it.get("followup") and not a.get("tool_calls"):
        msgs.append({"role": "user", "content": FOLLOWUPS[int(it["id"][1:]) % len(FOLLOWUPS)]})
        step()
    with lock:
        spent[0] += meta["cost"]
    return {**{k: it[k] for k in ("id", "domain", "split", "thinking", "effort", "tools")}, "messages": msgs, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--wiki")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--key-file", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", choices=sorted(SETS), default="base")
    a = ap.parse_args()
    key = read_key(a.key_file)
    global FOLLOWUPS
    FOLLOWUPS = SETS[a.set].FOLLOWUPS
    items = SETS[a.set].build_prompt_set(open(a.wiki).read() if a.wiki else None)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                r = json.loads(line)
                done.add(r["id"])
                spent[0] += r["meta"]["cost"]
    todo = [it for it in items if it["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(items)} prompts, {len(done)} done, {len(todo)} to go, spent so far ${spent[0]:.4f}", flush=True)
    t0, n_ok, n_err = time.time(), 0, 0
    with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_item, key, it, a.budget_usd): it for it in todo}
        for fut in futs:
            it = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"[err] {it['id']} {it['domain']}: {str(e)[:200]}", flush=True)
                continue
            if r is None:
                print(f"[stop] budget ${a.budget_usd} reached", flush=True)
                continue
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"{n_ok} ok, {n_err} err, ${spent[0]:.4f}, {time.time() - t0:.0f}s", flush=True)
    print(f"DONE {n_ok} ok, {n_err} err, total spent ${spent[0]:.4f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
