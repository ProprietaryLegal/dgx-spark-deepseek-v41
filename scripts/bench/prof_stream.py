# SPDX-License-Identifier: Apache-2.0
"""One streamed greedy request with a kit_bench set b prompt, for profile captures of any OpenAI-compatible engine
(TensorFold, ours): prints one JSON line with the completion tokens, the content chunks (= verify rounds on engines
that stream a round's accepted tokens as one chunk), the time to first token and the decode seconds (first to last
chunk). The profile scripts trim a capture to its decode window with these: [first kernel + ttft, last kernel], over
chunks - 1 rounds.
    python3 prof_stream.py --base http://localhost:18891 --model dsv41 --prompt code --tokens 384 [--kit /tmp/kit_bench.py]
"""
import argparse
import importlib.util
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="code")
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--set", default="b")
    ap.add_argument("--kit", default="/tmp/kit_bench.py")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("kit_bench", a.kit)
    kb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kb)
    prompt = dict(kb.PROMPT_SETS[a.set])[a.prompt]
    r = kb.Client(a.base, a.model, None).stream(kb.decode_body(prompt, a.tokens), "/v1/chat/completions")
    print(json.dumps({"prompt": a.prompt, "tokens": r["tokens"], "chunks": r["chunks"], "ttft_s": r["ttft_s"],
                      "decode_s": r["decode_s"], "decode_tps": r["decode_tps"]}), flush=True)


if __name__ == "__main__":
    main()
