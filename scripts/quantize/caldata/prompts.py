# SPDX-License-Identifier: Apache-2.0
"""The fixed prompt set for our calibration data and the published KLD panel: seeded, deterministic, built from
templates, with no third-party prompt dataset. `build_prompt_set()` returns a list of dicts:

  {"id", "domain", "messages" (OpenAI format, user/system turns only), "tools" (OpenAI tools or None),
   "thinking" (bool), "effort" ("low" | "high" | None), "split" ("cal" | "panel")}

Splits are disjoint by construction (each prompt lands in exactly one), stratified per domain.
"""
from __future__ import annotations

import random

SEED = 20261008

LANGS = ["Python", "TypeScript", "Rust", "Go", "C++", "C", "Java", "SQL", "Bash", "CUDA C++", "Kotlin", "Swift"]

CODE_TASKS = [
    "a function that merges overlapping intervals and returns them sorted",
    "an LRU cache with O(1) get and put",
    "a thread-safe bounded queue with blocking put and get",
    "a function that parses an ISO 8601 duration like P3DT4H5M into seconds",
    "a trie that supports insert, exact search and prefix count",
    "Dijkstra's shortest path on an adjacency list with a binary heap",
    "a rate limiter using the token bucket algorithm",
    "a function that validates a Sudoku board (9x9, partially filled)",
    "an iterative in-order traversal of a binary tree without recursion",
    "a function that computes the longest common subsequence of two strings and returns it",
    "a CSV parser that handles quoted fields with embedded commas and escaped quotes",
    "a function that detects a cycle in a directed graph and returns one cycle",
    "a ring buffer for audio samples with overwrite-on-full semantics",
    "a simple JSON pretty-printer without using a JSON library",
    "a function that converts a Roman numeral to an integer and back",
    "matrix multiplication tiled for cache locality",
    "a debounce helper that coalesces calls within a time window",
    "an exponential backoff retry wrapper with jitter",
    "a consistent hashing ring with virtual nodes",
    "a function that tokenizes arithmetic expressions and evaluates them with operator precedence",
    "a min-heap from scratch with push, pop and heapify",
    "a function that finds the k most frequent words in a text",
    "a union-find structure with path compression and union by rank",
    "a bloom filter with configurable false-positive rate",
    "a function that rotates an NxN matrix 90 degrees in place",
    "a reservoir sampler that picks k items from a stream of unknown length",
    "a topological sort that reports when the graph has a cycle",
    "a function that computes the edit distance and the edit script",
    "a parallel prefix sum (scan) over an array",
    "a skiplist with insert, delete and search",
]

BUGGY_SNIPPETS = [
    ("Python", "def mean(xs):\n    total = 0\n    for i in range(1, len(xs)):\n        total += xs[i]\n    return total / len(xs)\n"),
    ("Python", "def dedupe(items):\n    seen = []\n    for x in items:\n        if x not in seen:\n            seen.append(x)\n            items.remove(x)\n    return seen\n"),
    ("JavaScript", "async function fetchAll(urls) {\n  const out = [];\n  urls.forEach(async (u) => {\n    const r = await fetch(u);\n    out.push(await r.json());\n  });\n  return out;\n}\n"),
    ("C", "char *copy(const char *s) {\n    char *d = malloc(strlen(s));\n    strcpy(d, s);\n    return d;\n}\n"),
    ("Go", "func sum(ch chan int) int {\n    total := 0\n    for {\n        v := <-ch\n        total += v\n    }\n    return total\n}\n"),
    ("Rust", "fn largest(v: &Vec<i32>) -> &i32 {\n    let mut m = &v[0];\n    for x in v { if x > m { m = x; } }\n    m\n}\n// called with an empty vec in production\n"),
    ("SQL", "SELECT customer_id, COUNT(*) AS orders\nFROM orders\nWHERE status = 'shipped'\nGROUP BY customer_id\nHAVING orders > 5 AND created_at > '2026-01-01';\n"),
    ("Python", "import threading\ncounter = 0\ndef work():\n    global counter\n    for _ in range(100000):\n        counter += 1\nts = [threading.Thread(target=work) for _ in range(8)]\n[t.start() for t in ts]\nprint(counter)\n"),
    ("C++", "std::vector<int> v = {1, 2, 3, 4};\nfor (auto it = v.begin(); it != v.end(); ++it) {\n    if (*it % 2 == 0) v.erase(it);\n}\n"),
    ("Bash", "for f in $(ls *.log); do\n  if [ $f -nt backup ]; then cp $f backup/; fi\ndone\n"),
    ("CUDA C++", "__global__ void add(float *a, float *b, float *c, int n) {\n    int i = threadIdx.x;\n    c[i] = a[i] + b[i];\n}\n// launched as add<<<n/256, 256>>>(a, b, c, n);\n"),
    ("TypeScript", "function groupBy<T>(xs: T[], key: keyof T) {\n  return xs.reduce((acc, x) => {\n    (acc[x[key]] ||= []).push(x);\n    return acc;\n  }, {});\n}\n"),
]

CODE_VERBS = [
    "Write {lang} code for {task}. Include a short usage example.",
    "Implement {task} in {lang}. Explain the time and space complexity.",
    "In {lang}, write {task}, then write unit tests that cover the edge cases.",
    "I need {task} in {lang} for a production service. Keep it idiomatic and document the tricky parts.",
    "Give me a minimal {lang} implementation of {task}, then list two ways it could fail under load.",
]

MATH = [
    "A train leaves at 9:40 at 72 km/h; a second leaves the same station at 10:05 at 90 km/h on the same track. When and where does the second catch the first?",
    "How many 5-card poker hands contain exactly two pairs? Show the counting.",
    "Prove that the square root of 3 is irrational.",
    "Find all real x with |2x - 5| + |x + 1| = 9.",
    "A bag has 4 red, 5 blue and 6 green balls. Draw 3 without replacement. What is the probability all three colors appear?",
    "Solve the recurrence T(n) = 3T(n/2) + n^2 and justify with the master theorem.",
    "What is the expected number of rolls of a fair die until two consecutive sixes appear?",
    "Compute the sum of the first 200 terms of 1/(k(k+1)).",
    "A rectangle has perimeter 46 and diagonal 17. Find its area.",
    "Show that for any integer n, n^5 - n is divisible by 30.",
    "Three friends split a bill. A pays 40% more than B, C pays 25 less than A, total is 410. How much did each pay?",
    "Find the eigenvalues and eigenvectors of [[2, 1], [1, 2]] and explain what they mean geometrically.",
    "How many integers between 1 and 10,000 are divisible by 3 or 7 but not by 21?",
    "A loan of 250,000 at 6% annual interest is repaid monthly over 30 years. Derive the payment and compute it.",
    "Five people sit at a round table. In how many seatings do two specific people not sit next to each other?",
    "Integrate x^2 e^(-x) from 0 to infinity and explain the method.",
    "A clock's hands overlap at noon. At what exact times do they next overlap twice?",
    "Prove by induction that 1^3 + 2^3 + ... + n^3 = (n(n+1)/2)^2.",
    "A 10% salt solution is mixed with a 35% solution to get 40 liters at 20%. How much of each?",
    "What is the probability that a random 3-digit number has strictly increasing digits?",
    "Find the remainder when 7^2026 is divided by 13.",
    "Two dice are rolled until the sum is 7 or 8. What is the probability the last roll sums to 7?",
    "A cone has volume 96 pi and height equal to twice its radius. Find the slant height.",
    "Show that the harmonic series diverges, in two different ways.",
    "There are 12 coins, one counterfeit of unknown weight. With 3 weighings on a balance, find it and tell heavier or lighter.",
    "Minimize x^2 + y^2 subject to x + 2y = 10 using Lagrange multipliers, then check with substitution.",
    "Explain why 0.999... = 1 with a rigorous argument.",
    "How many ways can 8 rooks be placed on a chessboard so none attack each other, and how many if two specific squares are forbidden?",
]

KNOWLEDGE = [
    "Explain how a GPU executes a warp, what divergence costs, and how memory coalescing works.",
    "Why does unified memory on a single SoC change how you size a model's KV cache versus a discrete GPU?",
    "Explain TCP congestion control from slow start to BBR in plain terms.",
    "What is the difference between a process, a thread and a coroutine? When would you pick each?",
    "How does mixture-of-experts routing work in large language models, and why does load balancing matter?",
    "Explain quantization of neural network weights: scales, groups, and why outliers hurt.",
    "What causes the seasons on Earth? Correct the common misconception.",
    "Explain how mRNA vaccines work, step by step, without hype.",
    "How do RAID 5 and RAID 10 differ in performance and failure behavior?",
    "What were the main causes of the 2008 financial crisis?",
    "Explain how a transformer attention layer computes its output, with the matrix shapes.",
    "Why is the sky blue at noon and red at sunset?",
    "What does a compiler's register allocator do, and why is it hard?",
    "Explain the CAP theorem with a concrete example of each trade-off.",
    "How does public-key cryptography let two strangers agree on a secret?",
    "What is speculative decoding and when does it speed up generation?",
    "How do vaccines, antibiotics and antivirals differ in what they target?",
    "Explain how NVMe differs from SATA and why queue depth matters.",
    "What is the difference between latency-bound and bandwidth-bound workloads? Give examples.",
    "How does a lithium-ion battery degrade over time?",
    "Explain the B-tree and why databases use it instead of a binary search tree.",
    "What problem does RDMA solve and what are its failure modes?",
    "How did the printing press change European society?",
    "Explain eventual consistency to a product manager.",
    "What is the Hessian of a loss function and how do second-order quantization methods use it?",
    "Why does a Kalman filter work, intuitively?",
]

WRITING = [
    "Write a concise incident report for a 40-minute API outage caused by an expired TLS certificate.",
    "Draft a polite email declining a meeting and proposing two alternative times next week.",
    "Write a 200-word short story about a lighthouse keeper who receives a message from the future.",
    "Write release notes for version 2.4 of a CLI tool that adds parallel downloads and fixes a resume bug.",
    "Turn these notes into a clear paragraph: shipped feature late, root cause scope creep, fix: weekly scope review, owner per milestone.",
    "Write a haiku sequence (three haiku) about debugging at night.",
    "Write a product description for a mechanical keyboard aimed at programmers, under 120 words.",
    "Write a README introduction for an open-source benchmarking tool that measures tokens per second.",
    "Rewrite this to be more direct: 'We would like to perhaps consider the possibility of moving the deadline.'",
    "Write a cover letter opening paragraph for a systems engineer applying to a GPU startup.",
    "Produce a JSON object describing a fictional city with name, population, districts (array) and transit lines.",
    "Write a markdown table comparing three databases on latency, consistency and operational cost.",
    "Write a limerick about a cache miss.",
    "Summarize the plot of a heist movie you invent, in five bullet points.",
    "Write a short dialogue between a junior and senior engineer about why tests flake.",
    "Write a toast for a friend's startup launch, warm but under 100 words.",
]

MULTI = [
    ("zh", "请解释一下什么是哈希表，以及冲突是怎么解决的。"),
    ("zh", "写一首关于秋天的五言绝句，并解释意象。"),
    ("zh", "用Python写一个函数，统计一段中文文本中每个字出现的次数。"),
    ("ja", "量子コンピュータと古典コンピュータの違いを中学生にも分かるように説明してください。"),
    ("ja", "リモートワークのメリットとデメリットを箇条書きでまとめてください。"),
    ("ko", "블록체인이 무엇인지 간단하게 설명해 주세요."),
    ("es", "Explica la diferencia entre aprendizaje supervisado y no supervisado con ejemplos."),
    ("es", "Escribe un correo formal solicitando una prórroga para entregar un informe."),
    ("fr", "Explique pourquoi la Révolution française a commencé en 1789."),
    ("fr", "Rédige un court poème sur la mer en hiver."),
    ("de", "Erkläre, wie ein Verbrennungsmotor funktioniert, in einfachen Worten."),
    ("de", "Schreibe eine kurze Anleitung zum Wechseln eines Fahrradreifens."),
    ("ru", "Объясни, что такое рекурсия, на примере вычисления факториала."),
    ("ru", "Напиши короткий рассказ о кошке, которая научилась открывать двери."),
    ("ar", "اشرح أهمية الطاقة الشمسية في المستقبل."),
    ("hi", "कृत्रिम बुद्धिमत्ता के फायदे और नुकसान समझाइए।"),
    ("th", "อธิบายวิธีทำต้มยำกุ้งแบบง่ายๆ"),
    ("th", "ช่วยเขียนอีเมลขอลาป่วยถึงหัวหน้าให้หน่อย"),
    ("vi", "Hãy giải thích cách hoạt động của mạng nơ-ron nhân tạo."),
    ("pt", "Explique como funciona a fotossíntese de forma simples."),
    ("it", "Scrivi una breve recensione di un ristorante immaginario a Napoli."),
    ("id", "Jelaskan perbedaan antara cuaca dan iklim."),
    ("tr", "Bir web sitesinin hızını artırmak için beş öneri ver."),
    ("ne", "हिमालयको पर्यावरणीय महत्त्वबारे छोटो निबन्ध लेख्नुहोस्।"),
    ("zh", "比较一下Rust和Go在并发编程上的设计思路。"),
    ("ja", "このSQLの問題点を指摘してください: SELECT * FROM users WHERE name LIKE '%a%' ORDER BY RAND();"),
]

CHAT = [
    "I have two job offers: one pays 20% more, the other has a better team. How should I decide?",
    "Plan a 3-day trip to Kyoto in November for someone who likes temples and food, on a mid budget.",
    "My sourdough comes out dense. What am I probably doing wrong?",
    "How do I tell a coworker their code reviews feel harsh without making it awkward?",
    "Give me a 4-week plan to go from no running to a 5K.",
    "What should I check before buying a used car?",
    "I keep procrastinating on a big project. Give me a concrete method, not motivation.",
    "Recommend a reading order for getting into science fiction, with one line on each book.",
    "How do I set up a home network so my work laptop is isolated from smart devices?",
    "My landlord wants to raise rent 15%. How do I negotiate?",
    "Suggest five dinner ideas with chicken, rice and whatever vegetables are cheap.",
    "Is it worth learning Rust if I already know C++ well?",
    "How do I explain a two-year career gap in an interview?",
    "What is a sensible way to split chores fairly with roommates?",
    "Help me name a small coffee shop that also sells used books.",
    "I want to start lifting weights at 35. What should the first month look like?",
    "What are the trade-offs between renting and buying a home right now?",
    "Write me a short morning routine that fits in 20 minutes.",
    "How can I make a slow laptop usable again without buying a new one?",
    "Give me talking points for asking my manager for a raise.",
    "My plant's leaves are turning yellow. What could it be?",
    "How do I get better at estimating how long software tasks take?",
    "I'm nervous about a conference talk next week. How should I prepare?",
    "What's a good system for organizing notes across work and personal life?",
    "Explain how to read a nutrition label quickly.",
    "How do I back up a family photo collection properly?",
    "What questions should I ask at the end of a job interview?",
    "Help me write a birthday message for my grandmother turning 90.",
    "How do I stay focused when working from home with kids around?",
    "Suggest a weekend project to learn electronics with an Arduino.",
]

TOOLS = {
    "get_weather": {"description": "Current weather for a city", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "units": {"type": "string", "enum": ["metric", "imperial"]}}, "required": ["city"]}},
    "web_search": {"description": "Search the web, returns top results with titles and snippets", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
        "required": ["query"]}},
    "read_file": {"description": "Read a text file from the workspace", "parameters": {"type": "object",
        "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    "write_file": {"description": "Write a text file in the workspace", "parameters": {"type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    "run_shell": {"description": "Run a shell command in the workspace and return stdout, stderr and exit code",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}},
                       "required": ["command"]}},
    "run_python": {"description": "Execute Python code and return what it prints", "parameters": {"type": "object",
        "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    "query_db": {"description": "Run a read-only SQL query on the analytics database", "parameters": {"type": "object",
        "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
    "calendar_create": {"description": "Create a calendar event", "parameters": {"type": "object", "properties": {
        "title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601"},
        "duration_min": {"type": "integer"}, "attendees": {"type": "array", "items": {"type": "string"}}},
        "required": ["title", "start"]}},
    "http_get": {"description": "HTTP GET a URL and return status and body (truncated)", "parameters": {"type": "object",
        "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    "gpu_status": {"description": "List GPUs on a host with memory used, utilization and temperature", "parameters": {
        "type": "object", "properties": {"host": {"type": "string"}}, "required": ["host"]}},
}

AGENT_TASKS = [
    (["get_weather"], "Should I bike to work in Lisbon tomorrow morning? Check the weather first."),
    (["get_weather"], "Compare the weather in Tokyo and Osaka right now and tell me where it is warmer."),
    (["web_search"], "Find the latest stable release of PostgreSQL and summarize what changed."),
    (["web_search"], "What are people saying about the new RISC-V laptops? Search and summarize."),
    (["read_file", "write_file"], "Read config/app.yaml, bump the version patch number, and write it back."),
    (["read_file"], "Open src/server.py and tell me why requests might hang under load."),
    (["run_shell"], "Check how much disk space is left on this machine and which directories are largest."),
    (["run_shell"], "Run the test suite with pytest and tell me which tests fail."),
    (["run_shell", "read_file"], "The build is failing. Run make, read the error, and propose a fix."),
    (["run_python"], "Compute the 30-day rolling average of [12, 15, 11, 19, 22, 17, 25] repeated to 60 values and report the last 5."),
    (["run_python"], "Use Python to check whether 2^61 - 1 is prime."),
    (["query_db"], "How many active users did we have last week compared to the week before? Query the database."),
    (["query_db"], "Find the top 5 products by revenue this quarter."),
    (["calendar_create"], "Schedule a 45-minute design review with ana@example.com and raj@example.com next Tuesday at 14:00 UTC."),
    (["http_get"], "Is https://status.example.com reporting any incidents? Fetch it and summarize."),
    (["gpu_status"], "Check the GPUs on host spark-a and tell me if any are close to full memory."),
    (["gpu_status", "run_shell"], "One of the GPUs on host spark-b looks stuck. Check its status and restart the inference service if needed."),
    (["web_search", "write_file"], "Research three open-source vector databases and write a comparison to notes/vector_dbs.md."),
    (["read_file", "run_python"], "Read data/latency.csv and compute the p50, p90 and p99 latency."),
    (["get_weather", "calendar_create"], "If it will be sunny in Bangkok on Saturday, book a 2-hour picnic at 10:00 local time."),
]

TOOL_RESULTS = {
    "get_weather": lambda a: {"city": a.get("city", "?"), "temp_c": 23, "condition": "light rain", "wind_kmh": 18,
                              "forecast_tomorrow": {"temp_c": 21, "condition": "showers until 10:00, then clearing"}},
    "web_search": lambda a: {"results": [
        {"title": f"{a.get('query', '')[:40]} - overview", "snippet": "The latest release adds incremental sorting, faster vacuum and better JSON support."},
        {"title": "Release notes", "snippet": "Bug fixes for replication, new monitoring views, deprecations listed."},
        {"title": "Community discussion", "snippet": "Users report smoother upgrades but note one regression in planner estimates."}]},
    "read_file": lambda a: {"path": a.get("path", "?"), "content":
        "version: 1.8.3\nserver:\n  workers: 4\n  timeout_s: 30\n# TODO: pool size hardcoded\ndb:\n  pool_size: 5\n"},
    "write_file": lambda a: {"ok": True, "bytes_written": len(a.get("content", ""))},
    "run_shell": lambda a: {"exit_code": 1 if "make" in a.get("command", "") or "pytest" in a.get("command", "") else 0,
                            "stdout": "Filesystem  Size  Used Avail Use%\n/dev/nvme0n1p2  3.7T  2.2T  1.4T  63%\n",
                            "stderr": "src/io.c:42: error: 'buf_len' undeclared (first use in this function)\n"
                            if "make" in a.get("command", "") else ("FAILED tests/test_api.py::test_timeout - AssertionError\n"
                                                                     if "pytest" in a.get("command", "") else "")},
    "run_python": lambda a: {"stdout": "result computed; see values: [18.6, 18.9, 19.1, 18.7, 19.0]\n"},
    "query_db": lambda a: {"columns": ["week", "active_users"], "rows": [["2026-W39", 41230], ["2026-W40", 43815]]},
    "calendar_create": lambda a: {"ok": True, "event_id": "evt_7f3a", "start": a.get("start")},
    "http_get": lambda a: {"status": 200, "body": "All systems operational. Scheduled maintenance: DB failover test Oct 12 02:00 UTC."},
    "gpu_status": lambda a: {"host": a.get("host"), "gpus": [{"id": 0, "mem_used_gb": 117.8, "mem_total_gb": 121.7,
                                                            "util": 0, "temp_c": 71, "note": "no progress for 14 min"}]},
}


def _passages(seed_text: str, n: int, rng: random.Random, min_chars=2500, max_chars=6000):
    paras = [p.strip() for p in seed_text.split("\n\n") if len(p.strip()) > 400]
    out = []
    for _ in range(n):
        start = rng.randrange(max(1, len(paras) - 8))
        text, i = "", start
        while len(text) < rng.randint(min_chars, max_chars) and i < len(paras):
            text += paras[i] + "\n\n"
            i += 1
        out.append(text.strip())
    return out


def build_prompt_set(long_text: str | None = None, n_panel_frac: float = 0.40):
    rng = random.Random(SEED)
    items = []

    def add(domain, messages, tools=None):
        items.append({"domain": domain, "messages": messages, "tools": tools})

    # code: write (150 combos sampled), debug (all snippets x 2 phrasings)
    combos = [(l, t, v) for l in LANGS for t in CODE_TASKS for v in CODE_VERBS]
    for lang, task, verb in rng.sample(combos, 220):
        add("code", [{"role": "user", "content": verb.format(lang=lang, task=task)}])
    for lang, snip in BUGGY_SNIPPETS:
        add("code_debug", [{"role": "user", "content": f"This {lang} code has a bug. Find it, explain it, and fix it:\n\n```\n{snip}```"}])
        add("code_debug", [{"role": "user", "content": f"Review this {lang} snippet like a senior engineer. What breaks and when?\n\n```\n{snip}```"}])
    # agentic: every task twice, once with a system prompt
    for names, task in AGENT_TASKS:
        tools = [{"type": "function", "function": {"name": n, **TOOLS[n]}} for n in names]
        extra = [n for n in rng.sample(list(TOOLS), 3) if n not in names]
        tools_wide = tools + [{"type": "function", "function": {"name": n, **TOOLS[n]}} for n in extra]
        add("agentic", [{"role": "user", "content": task}], tools)
        add("agentic", [{"role": "system", "content": "You are an operations agent. Use tools when they help; be brief."},
                        {"role": "user", "content": task}], tools_wide)
        add("agentic", [{"role": "system", "content": "You are a careful assistant with tool access. Plan first, call the "
                                                      "tools you need, verify results, then answer with a short summary."},
                        {"role": "user", "content": task + " Explain what you did at the end."}], tools_wide)
    for q in MATH:
        add("math", [{"role": "user", "content": q}])
    for q in KNOWLEDGE:
        add("knowledge", [{"role": "user", "content": q}])
    for q in WRITING:
        add("writing", [{"role": "user", "content": q}])
    for q in CHAT:
        add("chat", [{"role": "user", "content": q}])
    for lang, q in MULTI:
        add(f"multilingual_{lang}", [{"role": "user", "content": q}])
    if long_text:
        asks = ["Summarize the passage below in five bullet points, then name its weakest claim.",
                "Read the passage below and write three questions a careful reader should ask, with answers from the text.",
                "What is the main argument of the passage below? Quote the sentence that supports it best.",
                "Rewrite the passage below for a 12-year-old reader, keeping every fact."]
        for p in _passages(long_text, 70, rng):
            add("long_context", [{"role": "user", "content": f"{rng.choice(asks)}\n\n---\n{p}"}])
    # multi-turn: a follow-up turn on a subset of single-turn items is created at generation time (needs the answer)

    for i, it in enumerate(items):
        it["id"] = f"p{i:04d}"
        r = rng.random()
        if it["domain"] == "math":
            it["thinking"], it["effort"] = True, "high"
        elif r < 0.30:
            it["thinking"], it["effort"] = False, None
        elif r < 0.65:
            it["thinking"], it["effort"] = True, "high"
        else:
            it["thinking"], it["effort"] = True, "low"
        it["followup"] = it["domain"] in ("code", "knowledge", "writing", "chat") and rng.random() < 0.25

    # stratified split: within each domain, a fixed fraction goes to the published panel
    by_dom = {}
    for it in items:
        by_dom.setdefault(it["domain"].split("_")[0], []).append(it)
    for dom, its in sorted(by_dom.items()):
        rng.shuffle(its)
        k = max(1, round(len(its) * n_panel_frac))
        for j, it in enumerate(its):
            it["split"] = "panel" if j < k else "cal"
    return items


FOLLOWUPS = ["Can you make it shorter and keep only the essential part?",
             "What would change if the input were 100x larger?",
             "Now explain the trickiest part as if to a beginner.",
             "Give one alternative approach and compare the two."]


if __name__ == "__main__":
    import collections
    import sys
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else None
    s = build_prompt_set(text)
    c = collections.Counter((x["domain"].split("_")[0], x["split"]) for x in s)
    print(len(s), "prompts")
    for k in sorted(c):
        print(k, c[k])
    print("thinking", sum(x["thinking"] for x in s), "followups", sum(x["followup"] for x in s))
