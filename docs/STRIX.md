# Serving Strix from one DGX Spark

This fork tunes the single-Spark Qwen3.8-Flash-Next recipe for
[Strix](https://github.com/usestrix/strix) (tested with 1.7.1), the autonomous
security agent, in combined SAST + DAST scans:

```bash
strix --target ./my-app-repo --target https://staging.example.com \
      --instruction-file strix/instructions/sast-dast-combined.md
```

Every number here was measured on one GB10 (121.7 GiB unified memory) on
2026-11-20 with the Keys ablit checkpoint unless it says otherwise.

## What changes against upstream

| Area | Change | Why |
|---|---|---|
| Profiles | `PROFILE=strix-v034` (default), `strix-agent`, `strix-agent-262k`, `strix-v034-262k` in `.env.sample` | One switch for the Strix settings below |
| Reasoning | `--default-chat-template-kwargs '{"reasoning_effort":"medium"}'` | The stock template defaults to `xhigh`; Strix sends no effort for a model LiteLLM does not know |
| Sampling | `--override-generation-config` temperature 0.6, top_p 0.95, top_k 20 | Strix sends no temperature; vLLM's default is 1.0 |
| Tool calls | `TOOL_CALL_PARSER=qwen3_xml` knob (no duplicate `--tool-call-parser`) | The name Strix's guide lists; same parser class as `qwen3_coder` |
| Auth | `API_KEY` reaches vLLM as `VLLM_API_KEY`, not `--api-key` | vLLM printed the key in its startup line, so it ended up in every log archive |
| Observability | `--kv-cache-metrics`, `scripts/strix-metrics.py`, `strix/run-strix.sh` | Per-scan server report: tokens, cache hits, queueing, evictions, MTP acceptance |
| Image | Opt-in stock vLLM 0.34 lane, `start-v034.sh` | Cold prefill +30-55% and per-turn latency halved on the same model; see below |
| Client | `strix/strix.env.sample` | Timeouts, `max_tokens`, `thinking_token_budget`, compaction settings |
| Prompting | `strix/instructions/sast-dast-combined.md` | Method plus context discipline for a local model |
| Benchmark | `bench/agent_bench.py` | Decode, cold prefill and a 12-turn strict-tool session, seeded |

## Quick start

On the Spark:

```bash
cp .env.sample .env            # PROFILE="strix-v034" is the default
# set HF_TOKEN (gated ablit checkpoint) and API_KEY (openssl rand -hex 32)
ABLIT=1 ./download.sh
./start.sh                     # or ./start-v034.sh for the 0.34 lane
EXPECT_LEN=524288 ./scripts/smoke-test.sh   # 7/8: vision fails by design (--language-model-only)
```

On the machine that runs Strix (it needs this repo checked out, or just the
`strix/` and `scripts/strix-metrics.py` files):

```bash
cp strix/strix.env.sample strix/strix.env   # set LLM_API_BASE and LLM_API_KEY
strix/run-strix.sh --target ./my-app-repo --target https://staging.example.com \
    --instruction-file strix/instructions/sast-dast-combined.md
```

`run-strix.sh` checks `/health`, the key, the served name and the context
window first, snapshots `/metrics` before and after, and writes
`strix-server-reports/<timestamp>/server-report.txt`.

## Server profiles

| Profile | Image | Context | Seqs | KV (FP8) | Notes |
|---|---|---|---|---|---|
| `strix-agent` | pinned `qwen38-flash-next` | 524,288 (YaRN 2.0) | 4 | 1,526,329 tok | Pinned-image fallback; proven over a 7.6 h Strix run. Roll back here if a 0.34.x image regresses |
| `strix-agent-262k` | pinned | 262,144 native | 6 | ~1.5M tok | Better short-context quality (no YaRN); needs working compaction |
| `strix-v034` | `vllm/vllm-openai:v0.34.0` | 524,288 (YaRN 2.0) | 4 | 1,382,356 tok (20 GiB pinned) | **Default in `.env.sample`.** Faster (see below); validated by a 5 h `-m deep` soak (2026-11-22) |
| `strix-v034-262k` | `vllm/vllm-openai:v0.34.0` | 262,144 native | 6 | 20 GiB pinned | Next A/B: shorter per-agent contexts, no YaRN (see "When Strix actually compacts") |

All of them: Keys ablit checkpoint, MTP 4 with the 47k draft vocab,
`disable_eagle_block_drop`, BF16 SSM state, 4,096-token prefill chunks, no
vision tower, reasoning `medium`, temperature 0.6, `qwen3_xml`.

**Why not more sequences at 512k.** Strix creates sub-agents with no global
cap (3.6 requests in flight on average over the 7.6 h run, more in bursts). At
512k the KV pool holds 2.6-2.9 full contexts; extra running sequences only
evict each other's prefix cache and turn every turn into a fresh prefill.
Queueing (20% of request time in that run) is cheaper than re-prefilling 150k
tokens. The instruction template also asks Strix for at most 4 concurrent
sub-agents, which it follows as an instruction.

**YaRN or native 262k.** YaRN factor 2 costs some short-context accuracy and is
only needed because Strix's compaction used to fail (below). With the client
settings in place every agent stays under ~200k, so `strix-agent-262k` is the
profile to A/B next.

## Client settings that matter (`strix/strix.env`)

| Setting | Value | Failure it fixes |
|---|---|---|
| `LLM_STREAM_FIRST_EVENT_TIMEOUT` | 1200 | Strix's 120 s default abandoned queued or long-prefill requests; each retry re-prefilled from scratch (84 abandoned requests in the 7.6 h run) |
| `LLM_STREAM_TOTAL_TIMEOUT` | 0 | The 600 s default cut long reasoning turns mid-stream |
| `LLM_TIMEOUT` / `LLM_STREAM_IDLE_TIMEOUT` | 1800 / 600 | Same, for the non-stream and idle paths |
| `LLM_EXTRA_BODY` `max_tokens` | 16384 | Strix sends no `max_tokens`; `extra_body` is merged last, so it also lifts the 4,096-token compaction summary |
| `LLM_EXTRA_BODY` `thinking_token_budget` | 8192 | vLLM forces the end of reasoning at the budget, so a runaway think cannot consume `max_tokens` and return an empty answer |
| `STRIX_CONTEXT_FALLBACK_TOKENS` | 200000 | Compaction threshold for an unmapped model (minus a 20k buffer). Strix counts its own system prompt and tool schemas, ~64k tokens per agent, so this leaves ~116k of history |
| `STRIX_CONTEXT_SUMMARY_TOKENS` | 8192 | Strix's own cap for unmapped models; the request gets 16,384 anyway |
| `STRIX_CACHE_BLOCK_TOKENS` | 1680 | Reporting only: the hybrid model caches in 1,680-token blocks |
| `STRIX_TELEMETRY` | false | Private targets |

**Reasoning effort: what one hard one-shot prompt showed.** A strict C11
"spinning donut" spec (exact constants, no warnings allowed), 3 samples per
setting on the 0.34 lane, each compiled with the prompt's own `gcc` line and
run for 1.2 s:

| Setting | Reasoning tokens | Time | Compiles / no warnings | Correct animation |
|---|---|---|---|---|
| `medium` (server default) | 712 / 1,404 / 4,286 | 36-100 s | 3 / 0 | 2 (one broken by an out-of-bounds loop) |
| `xhigh` | 16,760 / 17,780 / 32,768 (cut) | 6.5-12 min | 2 / 2 | 2 (the third returned no code at all) |
| `xhigh` + `thinking_token_budget` 8192 | 8,191 each | ~3.5 min | 3 / 3 | 1 (one never homes the cursor, so frames stack; one off-centre) |

Depth buys spec compliance (zero warnings) at 5-15x the tokens; correctness
was about the same, the unbounded run is the morning's failure in miniature
(32k tokens of thinking, no answer), and a budget well below what the task
wants can hurt (the model is made to answer mid-thought). Strix turns are
short and get tool feedback, so the server keeps `medium`; to try depth on a
real target, use the commented `xhigh` line in `strix/strix.env.sample` (a
generous budget, a larger cap) and compare the two `run-strix.sh` reports.

**When Strix actually compacts.** Only at the start of an agent's run cycle
(`strix/core/execution.py`: before `Runner.run_streamed`, whose cycle lasts up
to `--max-turns` turns) and when a request fails. Within one long cycle the
context keeps growing past `STRIX_CONTEXT_FALLBACK_TOKENS` (the validation scan
below peaked at 125k against an 80k threshold). The hard bound is the server:
when a prompt no longer fits, vLLM answers 400, Strix's native OpenAI route
treats it as a transient error (its overflow detector only recognises LiteLLM
exception types), retries with backoff, and each retry's proactive compaction
shrinks the history. Measured on 0.34: it accepts prompt + `max_tokens` above
`max_model_len` and shortens the reply to the room left; the 400 ("maximum
context length") comes only when the prompt alone is longer than
`max_model_len` (the pinned image rejects as soon as prompt + `max_tokens`
exceeds it). So the per-agent ceiling is about `max_model_len`, and the last
turns before it get short replies.

Two consequences, both seen in forced tests at 80k and 98k:

- A retry cannot help when the history is too small to summarise (Strix's
  ~64k base prompt plus one large tool output): five 400s in a row and Strix
  marks the agent failed. Keep `max_model_len` far above 64k; at 262k or 512k
  the history is what fills the window and compaction frees most of it.
- Of 8 summary calls in those tests, 5 returned a full summary and 3 came back
  empty after ~150 tokens; Strix then keeps the history and tries again on the
  next retry. Replaying the same request shape (Strix's instructions, 0.5k to
  38k tokens of conversation, streaming and not) gave a full summary 12 times
  out of 12, so the empty ones are not explained yet. `run-strix.sh` users can
  watch for "compaction summary returned no content" in the Strix log.

**The compaction bug in one paragraph.** Strix summarises an agent's history
when it nears the window, with `max_tokens=4096`. With reasoning at `xhigh`
the model spent those tokens thinking, the summary came back empty
(`finish_reason=length`), Strix logged "compaction summary returned no content"
and kept the full history. Contexts grew to 200-500k, the KV pool could hold
fewer of them, evictions turned into prefill storms. Medium effort, a 16k cap
and an 8k thinking budget together leave the summary at least 8k tokens.

## The vLLM 0.34 lane

Upstream vLLM now ships natively most of what the pinned image (an August
2026 dev build) needs patches for:

| Pinned image needs | vLLM 0.34 |
|---|---|
| `patch_ple_offload.py` + packed 27 GiB PLE table | `--engram-config '{"disk_offload":true}'` reads the rows from the checkpoint shards (vllm#59958) |
| `patch_qsa_fp8_kv.py` | FP8 KV on the QSA path (vllm#55557) |
| `patch_block_drop.py` (vllm#53388 backport) | `disable_eagle_block_drop` native |
| `patch_modelopt_mxfp8.py` | MXFP8 shape fallback (vllm#54223), mixed-precision NVFP4 detection (vllm#56050) |
| - | SM121 (GB10) skinny-GEMM plans for this model (vllm#59753), int32 offset fixes for long GDN prefills (vllm#63761), UMA-correct KV sizing (vllm#65413), `--tool-strict-level`, `--max-num-active-seqs` |

What the lane still does itself:

- **Config view** (`files/v034_config_shim.py`). The Mia/Keys config predates
  the upstream rename (`qwen3_8_flash_next` → `qwen4_exp`) and 0.34 cannot
  parse it. start.sh writes `~/.cache/vllm/v034_model/<repo>/<snapshot>/`:
  every file links to the original snapshot, only `config.json` is rewritten.
  The Hugging Face cache is never modified.
- **YaRN in that config.** 0.34 follows the transformers convention:
  `max_position_embeddings` is already the scaled length and the factor is not
  applied again, and it drops a nested `--hf-overrides`. The view sets
  `max_position_embeddings=524288` plus the YaRN `rope_parameters`.
- **Draft vocab.** `files/patch_mtp_draft_vocab_v030.py` applies unchanged to
  0.34's `mtp.py` (same five anchors); the log reports "MTP draft vocab: 47172
  of 248320 tokens".
- **Pinned KV.** `V034_KV_GIB` (default 20) goes to `--kv-cache-memory-bytes`,
  which also turns off the extensible KV cache that would otherwise size itself
  from free memory. The host-reserve check still applies.

Switch and roll back:

```bash
./stop.sh && ./start.sh          # default: strix-v034 (stock vLLM 0.34), PROFILE in .env
./stop.sh && ./start-v034.sh     # same lane explicitly (forces V034=true regardless of PROFILE)

# roll back to the pinned image: set PROFILE="strix-agent" in .env, then
./stop.sh && ./start.sh
```

`PROFILE` is read only from `.env` (it is not an environment override like
`V034`), so rolling back means editing `.env`. The supervisor and maintenance
relaunch follow whichever lane started last (`logs/launch-lane`).

### Measured: pinned image against 0.34 (same model, same settings)

`bench/agent_bench.py`, idle server, MTP 4, 47k draft vocab, thinking off for
decode. Decode is per-stream tok/s after the first token (aggregate in
brackets).

| | pinned (`strix-agent`) | 0.34 (`strix-v034`) |
|---|---|---|
| Boot to `/health` (warm page cache) | 662 s | **241 s** |
| Cold prefill, 15k / 61k / 122k tokens | 1,849 / 2,309 / 2,237 tok/s | **2,870 / 2,991 / 2,969 tok/s** |
| Decode code, S=1 / 2 / 4 | 58.1 / 51.8 / 40.8 (50.1 / 100.6 / 148.6) | **62.1 / 54.8 / 44.2 (59.5 / 106.4 / 172.4)** |
| Decode prose, S=1 / 2 / 4 | 41.7 / **43.4** / 25.1 (38.0 / **80.4** / 97.3) | **45.8** / 34.5 / **26.7** (**45.1** / 66.0 / **98.6**) |
| Session TTFT, turns 2-12 (~3.4k new tokens each), 4 runs | 1.55-4.67 s | **1.27-1.98 s** |
| Prompt tokens recomputed on turn 12 (of ~44.5k) | 7,118 | **3,534** |
| Prefix-cache hit, whole bench | 45.3% | **52.0%** |
| Session turns with a tool call (4 runs x 12) / calls with valid JSON args | 46 / 46 | 42 / 42 |
| MTP acceptance by position | 76 / 56 / 42 / 31% (3.05 tokens per step) | 78 / 59 / 43 / 32% (3.13 tokens per step) |
| KV pool | 1,526,329 tok | 1,382,356 tok |
| MemAvailable idle / minimum during the run | 15.5 / 11.8 GiB | **19.3 / 17.3 GiB** |

Prefill, which dominates an agent's turn (every turn re-reads tens of
thousands of context tokens, mostly from the prefix cache, plus the new tool
output), is 30-55% faster cold and the per-turn latency of a tool session
halves: 0.34 lands prefix-cache hits ~3.5k tokens closer to the end of the
prompt on this hybrid model, so each turn recomputes half as much. Decode is
equal or better except prose at two streams. Boot is 2.7x faster because 0.34
reads the PLE rows straight from the checkpoint instead of building and mapping
the packed table. The cost is a smaller KV pool at the same host margin
(V034_KV_GIB=20; there is ~4 GiB more MemAvailable to spend on it after a soak).
The turns answered in text instead of with a tool (6 of 48 against 2 of 48) are
the model closing out a synthetic target with nothing in it; the difference is
not significant (Fisher p = 0.27) and every call either lane made had valid
arguments.

`thinking_token_budget` verified on 0.34 with `reasoning_effort=xhigh` and
`max_tokens=6000` on a hard maths prompt: no budget, 6,000 reasoning tokens and
an empty answer (`finish_reason=length`, exactly the compaction failure);
budget 256, reasoning stopped at 255 tokens and the answer followed; budget
1024, 1,023 reasoning tokens, full answer, `finish_reason=stop`.

Caveats:

- Decode and prefill rows are one run each; the session rows are four runs.
- No vision on either lane (`--language-model-only`).
- Not ported to 0.34: the determinism knobs and `MTP_K_SCHEDULE`.
- start.sh's quant_algo pre-flight is skipped on both lanes (its introspection
  container has no GPU, so the vLLM import fails); coherent answers, valid tool
  calls and the smoke test are the evidence the mixed-precision map loads.
- 0.34 logs two harmless warnings: `cudagraph_capture_sizes` is deprecated
  (still honoured) and `VLLM_USE_V2_MODEL_RUNNER=1` from `EXTRA_DOCKER_ARGS` is
  redundant.

## Measuring a scan

`strix/run-strix.sh` prints and saves this (the validation scan below, 0.34
lane):

```
== requests
  finished 159: stop=159
  preemptions 0
== tokens
  prompt 12,342,025   generated 159,105   generated/request 1,001   prompt/request 77,623
  prefix cache: 11,902,464 of 12,342,025 prompt tokens hit (96.4%), 439,561 prefilled
  MTP: 52.7% of draft tokens accepted, 3.11 tokens per verify step
  MTP acceptance by position: 0:78%  1:58%  2:43%  3:32%
== latency (per request)
  end to end                   n=159    mean     35.22s   p50  ~    14.87s   p95  ~   2.7 min
  queued (waiting for a slot)  n=159    mean      0.25s   p50 <=     0.30s   p95  ~     0.51s
  time to first token          n=159    mean      1.86s   p50  ~     1.60s   p95  ~     4.70s
  prefill                      n=159    mean      1.29s   p50  ~     0.80s   p95  ~     3.62s
  decode                       n=159    mean     33.41s   p50  ~    12.50s   p95  ~   2.7 min
  inter-token latency          n=51253  mean    0.1036s   p50  ~   0.1017s   p95  ~   0.1459s
== request shape
  prompt tokens                n=159    mean     77,623   p50  ~    76,527   p95  ~   155,833
  generated tokens             n=159    mean      1,001   p50  ~       420   p95  ~     4,506
  max_tokens requested         n=159    mean     16,384   p50  ~    15,000   p95  ~    19,500
== KV cache residency (--kv-cache-metrics, sampled)
  KV block lifetime            n=40     mean     31.33s   p50  ~    24.29s   p95  ~   108.00s
  KV idle before eviction      n=40     mean     31.33s   p50  ~    24.29s   p95  ~   108.00s
== wall clock
  36.8 min elapsed; generated 72.2 tok/s on average; 2.54 requests in flight on average
```

Percentiles come from Prometheus buckets, so they are bucket-edge estimates
(`<=` marks a value inside the first bucket).

What to look at, in order:

1. `length` finishes: answers cut at `max_tokens`. A few are normal; many mean
   the thinking budget is too high for the cap.
2. Queue time against prefill time: long queues with short prefills mean the
   agents are waiting on each other (more `MAX_NUM_SEQS` would help only if the
   KV pool has room); long prefills with low cache hit mean evictions.
3. `KV idle before eviction`: seconds a block sat unused before eviction. Short
   values mean the pool is too small for the number of live contexts.
4. Generated per request: the reasoning budget at work (1,600 at `xhigh` on
   2026-11-20).

## Fine-tuning for Strix

**Weights: not now.** The checkpoint is a hybrid (Gated DeltaNet + QSA sparse
attention + PLE + MTP) stored as ModelOpt NVFP4/MXFP8. There is no QLoRA path
for that format in the common trainers, the BF16 weights (~250 GB) do not fit a
Spark, and a LoRA adapter served with MTP would lower draft acceptance (the
drafter was trained on the base distribution) on a decode-bound workload. The
refusal problem that motivates most security fine-tunes is already handled by
the Keys ablit.

**What does pay off, without training:**

1. The serving defaults above (reasoning effort, thinking budget, sampling,
   strict tool grammar: vLLM constrains a tool call's arguments to its JSON
   schema when the tool is `strict`, which Strix sets on every tool).
2. A Strix-specific MTP draft vocabulary. The draft head proposes from the 47k
   `en_code` subset; Strix emits tool-call XML, shell, HTTP and code. Collect
   Strix outputs, run `files/build_draft_vocab_extend.py` (the recipe already
   used it for Spanish), and compare per-position acceptance in the
   `run-strix.sh` report before and after.
3. The instruction file: scope, method and context discipline
   (`strix/instructions/sast-dast-combined.md`). Short targeted tool calls keep
   contexts small, which on one Spark is what saves time.
4. Keep the data: every Strix run leaves its full trajectory in
   `strix_runs/<run>/`. If a trainable checkpoint of this family appears,
   successful trajectories are the SFT set.

## Validation scan

Strix 1.7.1 from a second machine, `-n -m standard --max-turns 150`, against
`vuln-shop`: a 150-line standard-library web shop with eight planted
weaknesses, passed both as source directory and running URL, with an
instruction file built from `strix/instructions/sast-dast-combined.md`. Server:
`strix-v034`; client: `strix/strix.env.sample` (test window 100k instead of
200k to see compaction sooner).

| | |
|---|---|
| Duration | 36.8 min, 5 agents (root, SAST bundle, injection, access control, file/SSRF) |
| Findings | 9 confirmed dynamically, 5 high / 4 medium: SQL injection on login, path traversal reading `/etc/passwd` and `config.py`, IDOR on orders, forgeable session cookie, `host` injection on `/admin/ping`, reflected XSS, unsalted MD5, static admin token, `/health` secret leak. All eight planted classes found; the path-traversal report chains the leaked admin token to `/admin/ping` |
| Requests | 159, all `finish_reason=stop`; 0 errors, 0 timeouts, 0 cut at `max_tokens`, 0 preemptions |
| Tokens | 12.3M prompt (96.4% from the prefix cache), 159k generated (1,001 per request, 45% of it reasoning) |
| Latency | TTFT mean 1.86 s (p95 ~4.7 s), queue mean 0.25 s, 2.5 requests in flight on average |
| Context | Strix's own system prompt + tools ~64k tokens per agent; largest request 125k |
| Host | MemAvailable never below 17.3 GiB |

For scale: the 7.6 h production scan of 2026-11-20 morning (pinned image,
`xhigh` default, Strix defaults) generated 1,600 tokens per request and
abandoned 84 requests on Strix's 120 s first-event timeout. Different target,
so durations do not compare; the per-request numbers do.

Two more scans of the same target, same client settings:

| Server | Duration | Findings | Notes |
|---|---|---|---|
| 0.34, native 131k, no YaRN | 47.5 min | 8 confirmed (2 high, 6 medium), all planted classes; command injection proven via `shell=True` | largest request 124k; 1 reply shortened near the limit; 0 errors |
| 0.34, native 98k / 80k (forced compaction) | stopped after the checks | - | overflow 400 → retry → compaction path confirmed; see the caveats above |
