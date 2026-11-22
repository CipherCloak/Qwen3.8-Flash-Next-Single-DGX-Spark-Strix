<h1 align="center">Qwen3.8-Flash-Next on ONE DGX Spark (TP=1)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a> · remastered by <b>Claude (Anthropic)</b></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve the 99 GiB **Qwen3.8-Flash-Next** vision-language model from a **single
DGX Spark** (121 GiB unified memory, TP=1) with vLLM — tuned to drive the
[Strix](https://github.com/usestrix/strix) autonomous security agent for
SAST + DAST scans.

> 🛠️ **Remastered work.** Original recipe by **Mia'a AI Lab**; this copy was
> remastered by **Claude (Anthropic)**: default engine moved to stock vLLM 0.34,
> the Strix serving profile made the default, faster and leaner (see below).

## Quick start

```bash
cp .env.sample .env        # defaults to the vLLM 0.34 lane (profile strix-v034)
./download.sh              # fetch the ~99 GiB checkpoint (resumable, sha256-verified)
./start.sh                 # ~4 min to /health; serves on :8888
./stop.sh                  # container + watchdog, graceful
```

## What changed, and why vLLM 0.34

The default engine is now **stock `vllm/vllm-openai:v0.34.0`**, up from a pinned
August dev build. Upstream 0.34 ships natively what the old image needed local
patches for — PLE read straight from the checkpoint (disk offload), FP8 KV on
QSA, `disable_eagle_block_drop`, NVFP4 / MXFP8 detection and GB10 (SM121) GEMM
plans — so the **only overlay left is the draft vocab**. The payoff: faster boot
and prefill, better cache hits and ~5 GiB more free host memory. The Strix
profile (`strix-v034`) is the default, validated by an A/B (`reasoning_effort`
`medium` beat `xhigh`) and a 5 h deep soak.

## Speed & gains

Pinned dev image → vLLM 0.34.0, same model and settings (`bench/agent_bench.py`):

| Metric | Old · pinned image | New · vLLM 0.34.0 | Gain |
|---|---|---|---|
| Boot to `/health` | 662 s | **241 s** | **2.7× faster** |
| Cold prefill (15k / 61k / 122k) | 1,849 / 2,309 / 2,237 tok/s | **2,870 / 2,991 / 2,969 tok/s** | **+30–55 %** |
| Per-turn latency, 12-turn tool session | 1.55–4.67 s | **1.27–1.98 s** | **~halved** |
| Context recomputed at turn 12 (of ~44.5k) | 7,118 tok | **3,534 tok** | **−50 %** |
| Prefix-cache hit (whole bench) | 45.3 % | **52.0 %** | **+6.7 pts** |
| Code decode, 4 streams (aggregate) | 148.6 tok/s | **172.4 tok/s** | **+16 %** |
| Free host memory, minimum under load | 11.8 GiB | **17.3 GiB** | **+5.5 GiB** |

## Documentation

- **[docs/STRIX.md](docs/STRIX.md)** — serving the Strix security agent: profiles,
  client settings, tuning and measurements. **Start here for Strix.**
- **[docs/REFERENCE.md](docs/REFERENCE.md)** — full technical reference: every
  config knob, benchmark, long-context / YaRN, checkpoints, multimodal,
  unattended operation, and what is patched and why.

## Credits & license

Original recipe and checkpoint by **Mia'a AI Lab**; splice base
**local-inference-lab**; abliteration by **Keys (drowzeys)**; Spanish draft vocab
by **oscarmenendezgarcia**. Base model by **Qwen / Alibaba**, NVFP4 quant by
**NVIDIA**. Full credits in [docs/REFERENCE.md](docs/REFERENCE.md#credits).
Remastered (vLLM 0.34 lane, Strix tuning, 2026-11) by **Claude (Anthropic)**.

Copyright © 2026 MiaAI Lab. **AGPL-3.0-or-later** — see [LICENSE](LICENSE). This
repository runs a network server: under AGPL § 13, if you modify it and offer the
service over a network, you must offer your users the corresponding modified
source.
