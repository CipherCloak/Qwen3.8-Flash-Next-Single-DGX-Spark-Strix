#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What one agent run cost the server, from two Prometheus snapshots.

    strix-metrics.py snapshot URL            > before.prom   (URL: http://host:8888/metrics)
    strix-metrics.py diff before.prom after.prom [--wall SECONDS]

/metrics needs no API key. Counters and histograms are differenced, so the
report covers only what happened between the two snapshots (restart in
between = negative deltas; the report says so). Standard library only, so it
runs on the machine that runs Strix as well as on the Spark.
"""
import argparse
import math
import re
import sys
import time
import urllib.request
from collections import defaultdict

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+(\S+)')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
# Labels every vLLM series carries; dropped so series from one engine sum cleanly.
NOISE = {"engine", "model_name"}


def parse(text):
    series = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(3) or "", m.group(4)
        try:
            v = float(value)
        except ValueError:
            continue
        key = (name, tuple(sorted((k, x) for k, x in LABEL.findall(labels) if k not in NOISE)))
        series[key] = series.get(key, 0.0) + v
    return series


def delta(before, after):
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(before) | set(after)}


def total(d, name, **match):
    return sum(v for (n, labels), v in d.items()
               if n == name and all(dict(labels).get(k) == x for k, x in match.items()))


def by_label(d, name, label):
    out = defaultdict(float)
    for (n, labels), v in d.items():
        if n == name:
            out[dict(labels).get(label, "")] += v
    return dict(out)


def histogram(d, base):
    """(count, sum, [(le, cumulative)]) of a histogram delta, labels summed."""
    buckets = defaultdict(float)
    for (n, labels), v in d.items():
        if n == base + "_bucket":
            le = dict(labels).get("le")
            buckets[math.inf if le == "+Inf" else float(le)] += v
    return total(d, base + "_count"), total(d, base + "_sum"), sorted(buckets.items())


def quantile(buckets, q):
    """(value, bound) where bound is "<=" or ">=" when the bucket edges are all
    we know (first bucket, or past the last finite one), else ""."""
    if not buckets or buckets[-1][1] <= 0:
        return None, ""
    target = q * buckets[-1][1]
    prev_le, prev_c = 0.0, 0.0
    for i, (le, c) in enumerate(buckets):
        if c >= target:
            if math.isinf(le):
                return prev_le, ">="
            if i == 0:
                return le, "<="  # interpolating from 0 would invent a value
            span = c - prev_c
            return prev_le + (le - prev_le) * ((target - prev_c) / span if span else 1.0), ""
        prev_le, prev_c = le, c
    return prev_le, ">="


def fmt(x, unit="", digits=1):
    if x is None:
        return "-"
    if unit == "s" and x >= 120:
        return f"{x / 60:.1f} min"
    return f"{x:,.{digits}f}{unit}"


def hist_line(d, base, label, unit="s", digits=2):
    count, s, buckets = histogram(d, base)
    if count <= 0:
        return None
    (p50, b50), (p95, b95) = quantile(buckets, 0.5), quantile(buckets, 0.95)
    return (f"  {label:<28} n={int(count):<6} mean {fmt(s / count, unit, digits):>10}"
            f"   p50 {b50 or '~':>2}{fmt(p50, unit, digits):>10}   p95 {b95 or '~':>2}{fmt(p95, unit, digits):>10}")


def report(before, after, wall):
    d = delta(before, after)
    out = []
    if any(v < 0 for (n, _), v in d.items() if n.endswith("_total")):
        out.append("WARNING: some counters went down; the server restarted between the snapshots.")
    reasons = by_label(d, "vllm:request_success_total", "finished_reason")
    done = sum(reasons.values())
    prompt = total(d, "vllm:prompt_tokens_total")
    gen = total(d, "vllm:generation_tokens_total")
    out.append("== requests")
    out.append(f"  finished {int(done)}: " + ", ".join(f"{k}={int(v)}" for k, v in sorted(reasons.items()) if v))
    if reasons.get("length"):
        out.append(f"  {int(reasons['length'])} hit max_tokens (finish_reason=length): raise max_tokens in LLM_EXTRA_BODY"
                   " or lower thinking_token_budget")
    out.append(f"  preemptions {int(total(d, 'vllm:num_preemptions_total'))}")
    out.append("== tokens")
    out.append(f"  prompt {prompt:,.0f}   generated {gen:,.0f}"
               + (f"   generated/request {gen / done:,.0f}   prompt/request {prompt / done:,.0f}" if done else ""))
    q, h = total(d, "vllm:prefix_cache_queries_total"), total(d, "vllm:prefix_cache_hits_total")
    if q:
        out.append(f"  prefix cache: {h:,.0f} of {q:,.0f} prompt tokens hit ({100 * h / q:.1f}%), "
                   f"{q - h:,.0f} prefilled")
    drafts = total(d, "vllm:spec_decode_num_draft_tokens_total")
    acc = total(d, "vllm:spec_decode_num_accepted_tokens_total")
    nd = total(d, "vllm:spec_decode_num_drafts_total")
    if drafts:
        out.append(f"  MTP: {100 * acc / drafts:.1f}% of draft tokens accepted, "
                   f"{1 + acc / nd:.2f} tokens per verify step" if nd else "")
        pos = by_label(d, "vllm:spec_decode_num_accepted_tokens_per_pos_total", "position")
        if pos and nd:
            out.append("  MTP acceptance by position: " + "  ".join(
                f"{p}:{100 * v / nd:.0f}%" for p, v in sorted(pos.items(), key=lambda kv: int(kv[0] or 0))))
    out.append("== latency (per request)")
    for base, label in (("vllm:e2e_request_latency_seconds", "end to end"),
                        ("vllm:request_queue_time_seconds", "queued (waiting for a slot)"),
                        ("vllm:time_to_first_token_seconds", "time to first token"),
                        ("vllm:request_prefill_time_seconds", "prefill"),
                        ("vllm:request_decode_time_seconds", "decode")):
        line = hist_line(d, base, label)
        if line:
            out.append(line)
    for base, label in (("vllm:inter_token_latency_seconds", "inter-token latency"),
                        ("vllm:request_time_per_output_token_seconds", "time per output token")):
        line = hist_line(d, base, label, unit="s", digits=4)
        if line:
            out.append(line)
            break
    out.append("== request shape")
    for base, label in (("vllm:request_prompt_tokens", "prompt tokens"),
                        ("vllm:request_generation_tokens", "generated tokens"),
                        ("vllm:request_params_max_tokens", "max_tokens requested")):
        line = hist_line(d, base, label, unit="", digits=0)
        if line:
            out.append(line)
    evict = [hist_line(d, base, label) for base, label in (
        ("vllm:kv_block_lifetime_seconds", "KV block lifetime"),
        ("vllm:kv_block_idle_before_evict_seconds", "KV idle before eviction"),
        ("vllm:kv_block_reuse_gap_seconds", "KV reuse gap"))]
    if any(evict):
        out.append("== KV cache residency (--kv-cache-metrics, sampled)")
        out.extend(x for x in evict if x)
    usage = after.get(("vllm:kv_cache_usage_perc", ()), None)
    if usage is not None:
        out.append(f"  KV cache in use at the end: {100 * usage:.1f}%")
    if wall:
        busy = histogram(d, "vllm:e2e_request_latency_seconds")[1]
        out.append("== wall clock")
        out.append(f"  {fmt(wall, 's')} elapsed; generated {gen / wall:,.1f} tok/s on average"
                   + (f"; {busy / wall:.2f} requests in flight on average" if busy else ""))
    return "\n".join(x for x in out if x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("url")
    s.add_argument("--timeout", type=float, default=15)
    df = sub.add_parser("diff")
    df.add_argument("before")
    df.add_argument("after")
    df.add_argument("--wall", type=float, default=0, help="elapsed seconds, for the rate lines")
    a = ap.parse_args()
    if a.cmd == "snapshot":
        with urllib.request.urlopen(a.url, timeout=a.timeout) as r:
            body = r.read().decode()
        sys.stdout.write(f"# strix-metrics snapshot {a.url} at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n{body}")
    else:
        before, after = parse(open(a.before).read()), parse(open(a.after).read())
        print(report(before, after, a.wall))


if __name__ == "__main__":
    main()
