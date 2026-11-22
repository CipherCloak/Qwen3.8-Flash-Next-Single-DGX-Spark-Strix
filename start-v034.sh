#!/usr/bin/env bash
# start-v034.sh - same single-Spark launch as start.sh on stock vLLM 0.34
# (V034_IMAGE, default vllm/vllm-openai:v0.34.0), serving the Mia/ablit
# checkpoint selected by .env. Knobs stay in .env (V034_KV_GIB, V034_IMAGE);
# setting V034=true there does the same as this wrapper. See docs/STRIX.md.
# Usage: ./start-v034.sh [--no-launch]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export V034=true

exec "$SCRIPT_DIR/start.sh" "$@"
