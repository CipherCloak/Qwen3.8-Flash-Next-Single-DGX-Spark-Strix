#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strix-shaped benchmark: decode, cold prefill, and a multi-turn tool session.

Run on the Spark against an idle server; every prompt is seeded, so two lanes
(or two configs) see byte-identical requests and the JSON lines compare
directly.

    python3 bench/agent_bench.py --tag v034 [--only decode prefill session]

Reads API_KEY / PORT / SERVED_MODEL_NAME from .env like scripts/smoke-test.sh.

  decode   S=1/2/4 streams, code and prose, thinking off, 512 tokens, temp 0:
           per-stream tok/s after the first token, and aggregate tok/s.
  prefill  cold prompts of ~16k/64k/128k tokens (unique per run), max_tokens 1:
           TTFT and prompt tok/s.
  session  12 agent turns over a ~10k-token system+tools prefix, each turn
           adding a ~3k-token tool result, strict tools, reasoning on with
           thinking_token_budget: per-turn TTFT, cached prompt tokens, decode
           tok/s, and whether the reply was a parseable tool call.
"""
import argparse
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = ("request handler session token query user admin route config value buffer parse "
         "index cache module import return string object error input output file path "
         "server client header cookie payload filter render template secret hash").split()


def dotenv():
    env = {}
    try:
        for line in open(os.path.join(REPO, ".env")):
            m = re.match(r'^([A-Z_][A-Z0-9_]*)="?([^"#\n]*)"?', line.strip())
            if m:
                env[m.group(1)] = m.group(2).strip()
    except OSError:
        pass
    return env


def words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def code_blob(rng, lines):
    out = []
    for i in range(lines):
        a, b = rng.choice(WORDS), rng.choice(WORDS)
        out.append(f"    {a}_{i} = {b}.get('{rng.choice(WORDS)}', {rng.randint(0, 999)})  # {words(rng, 6)}")
    return "def handler(request):\n" + "\n".join(out)


class Client:
    def __init__(self, base, key, model):
        self.base, self.key, self.model = base.rstrip("/"), key, model

    def stream(self, body, timeout=1800):
        body = {"model": self.model, "stream": True, "stream_options": {"include_usage": True}, **body}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        t0 = time.time()
        first = last = None
        text, reasoning, tools, usage, finish = [], [], {}, {}, None
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    d = ch.get("delta", {})
                    piece = (d.get("content") or "") + (d.get("reasoning") or d.get("reasoning_content") or "")
                    if piece or d.get("tool_calls"):
                        now = time.time()
                        first = first or now
                        last = now
                    if d.get("content"):
                        text.append(d["content"])
                    if d.get("reasoning") or d.get("reasoning_content"):
                        reasoning.append(d.get("reasoning") or d.get("reasoning_content"))
                    for tc in d.get("tool_calls") or []:
                        slot = tools.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""
                    finish = ch.get("finish_reason") or finish
        gen = usage.get("completion_tokens", 0)
        return {
            "ttft": (first or time.time()) - t0,
            "total": time.time() - t0,
            "decode_tps": (gen - 1) / (last - first) if first and last and last > first and gen > 1 else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": gen,
            "finish": finish,
            "text": "".join(text),
            "reasoning_chars": sum(map(len, reasoning)),
            "tools": [tools[k] for k in sorted(tools)],
        }


def bench_decode(c, emit):
    prompts = {
        "code": "Write a complete Python module that implements an LRU cache with TTL expiry, "
                "thread safety, and a decorator interface. Include docstrings and type hints.",
        "prose": "Write a detailed essay on the history of cryptography from the Caesar cipher "
                 "to public-key systems, with concrete examples for each era.",
    }
    for kind, prompt in prompts.items():
        for streams in (1, 2, 4):
            results = [None] * streams
            body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 512, "temperature": 0,
                    "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}

            def run(i):
                results[i] = c.stream(body)
            t0 = time.time()
            threads = [threading.Thread(target=run, args=(i,)) for i in range(streams)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            wall = time.time() - t0
            per = [r["decode_tps"] for r in results if r["decode_tps"]]
            emit({"test": "decode", "prompt": kind, "streams": streams,
                  "per_stream_tps": round(statistics.mean(per), 1) if per else None,
                  "aggregate_tps": round(sum(r["completion_tokens"] for r in results) / wall, 1)})


def bench_prefill(c, emit):
    rng = random.Random(time.time_ns())  # cold on purpose: never a prefix-cache hit
    for target in (16_000, 64_000, 128_000):
        doc = words(rng, int(target / 1.05))
        r = c.stream({"messages": [{"role": "user", "content": doc + "\n\nReply with the single word OK."}],
                      "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        emit({"test": "prefill", "prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["ttft"], 2),
              "prefill_tps": round(r["prompt_tokens"] / r["ttft"], 0) if r["prompt_tokens"] else None})


TOOLS = [
    {"type": "function", "function": {
        "name": name, "description": desc + " " + words(random.Random(name), 60), "strict": True,
        "parameters": {"type": "object", "additionalProperties": False, "required": list(props),
                       "properties": {p: {"type": "string", "description": d} for p, d in props.items()}}}}
    for name, desc, props in (
        ("terminal_execute", "Run a shell command in the sandbox.", {"command": "shell command", "timeout": "seconds"}),
        ("read_file", "Read part of a file.", {"path": "file path", "range": "line range like 1-80"}),
        ("send_request", "Send an HTTP request to the target.", {"method": "HTTP method", "url": "full URL", "body": "request body"}),
        ("record_note", "Save a note to the workspace.", {"title": "short title", "content": "note text"}),
        ("report_finding", "Report a confirmed vulnerability.", {"title": "title", "severity": "severity", "evidence": "proof"}),
    )
]


def bench_session(c, emit, turns=12, budget=1024, run=1):
    rng = random.Random(1234)  # identical session on every lane
    system = ("You are a security testing agent working on an authorized assessment. Use the tools; "
              "call exactly one tool per turn.\n\n" + words(rng, 9000))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Assess the target application at http://app.local and its source in /workspace/app."}]
    for turn in range(1, turns + 1):
        r = c.stream({"messages": messages, "tools": TOOLS, "tool_choice": "auto", "max_tokens": 4096,
                      "thinking_token_budget": budget})
        call = r["tools"][0] if r["tools"] else None
        valid = False
        if call:
            try:
                json.loads(call["args"] or "{}")
                valid = True
            except ValueError:
                pass
        emit({"test": "session", "run": run, "turn": turn, "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"],
              "ttft_s": round(r["ttft"], 2), "decode_tps": round(r["decode_tps"], 1) if r["decode_tps"] else None,
              "completion_tokens": r["completion_tokens"], "reasoning_chars": r["reasoning_chars"],
              "tool": call["name"] if call else None, "valid_args": valid, "finish": r["finish"],
              **({} if call else {"text_head": r["text"][:240]})})
        call_id = f"call_{turn}"
        messages.append({"role": "assistant", "content": r["text"] or None,
                         "tool_calls": [{"id": call_id, "type": "function",
                                         "function": {"name": call["name"] if call else "read_file",
                                                      "arguments": call["args"] if valid else "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": code_blob(rng, 120)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--only", nargs="+", choices=["decode", "prefill", "session"], default=["decode", "prefill", "session"])
    ap.add_argument("--out", default=os.path.join(REPO, "logs", "agent-bench.jsonl"))
    ap.add_argument("--session-runs", type=int, default=1, help="repeat the session (sampling varies per run)")
    a = ap.parse_args()
    env = {**dotenv(), **os.environ}
    c = Client(f"http://localhost:{env.get('PORT', '8888')}", env.get("API_KEY", ""),
               env.get("SERVED_MODEL_NAME", "qwen3.8-flash-next"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    def emit(row):
        row = {"tag": a.tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    for name in a.only:
        if name == "session":
            for run in range(1, a.session_runs + 1):
                bench_session(c, emit, run=run)
        else:
            {"decode": bench_decode, "prefill": bench_prefill}[name](c, emit)


if __name__ == "__main__":
    main()
