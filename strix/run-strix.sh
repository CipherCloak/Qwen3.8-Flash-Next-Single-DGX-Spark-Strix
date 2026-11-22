#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# run-strix.sh — run Strix against this server with the tuned client settings
# (strix/strix.env), check the server first, and write what the scan cost the
# server next to Strix's own report.
#
# Usage (all arguments go to strix unchanged):
#   strix/run-strix.sh --target ./my-app-repo --target https://staging.example.com \
#       --instruction-file strix/instructions/sast-dast-combined.md
#
#   STRIX_ENV_FILE=/path/to/other.env strix/run-strix.sh ...
set -euo pipefail

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*" >&2; exit 1; }

# Everything runs inside main(), which bash parses whole before running it: a
# scan lasts hours, and editing or git-pulling this file meanwhile would
# otherwise make bash resume reading at a stale offset (seen 2026-11-20: it
# re-ran the strix command after the scan finished).
main() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
    ENV_FILE="${STRIX_ENV_FILE:-$SCRIPT_DIR/strix.env}"

    [[ -f "$ENV_FILE" ]] || err "$ENV_FILE not found. cp strix/strix.env.sample strix/strix.env and edit it."
    set -a
    # shellcheck source=strix.env.sample
    source "$ENV_FILE"
    set +a
    [[ -n "${LLM_API_BASE:-}" && "$LLM_API_BASE" != *SPARK_HOST* ]] || err "Set LLM_API_BASE in $ENV_FILE."
    [[ -n "${STRIX_LLM:-}" ]] || err "Set STRIX_LLM in $ENV_FILE."
    STRIX_BIN="${STRIX_BIN:-strix}"
    command -v "$STRIX_BIN" >/dev/null || err "strix not found (STRIX_BIN=$STRIX_BIN)."

    BASE="${LLM_API_BASE%/}"
    BASE="${BASE%/v1}"
    METRICS_URL="${STRIX_METRICS_URL:-$BASE/metrics}"
    SERVED="${STRIX_LLM#*/}"
    AUTH=()
    [[ -n "${LLM_API_KEY:-}" ]] && AUTH=(-H "Authorization: Bearer $LLM_API_KEY")

    # --- preflight: the server is up, the key works, the name and window match ---
    code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/health" || echo 000)
    [[ "$code" == 200 ]] || err "$BASE/health answered $code. Is the server up (./start.sh on the Spark)?"
    models=$(curl -s -m 10 "${AUTH[@]}" -w '\n%{http_code}' "$BASE/v1/models" || true)
    code="${models##*$'\n'}"
    [[ "$code" == 401 ]] && err "401 from $BASE/v1/models: LLM_API_KEY does not match the server's API_KEY."
    [[ "$code" == 200 ]] || err "$BASE/v1/models answered $code."
    read -r name maxlen <<<"$(printf '%s' "${models%$'\n'*}" | python3 -c \
        'import json, sys; m = json.load(sys.stdin)["data"][0]; print(m["id"], m.get("max_model_len", 0))')"
    [[ "$name" == "$SERVED" ]] || err "server serves '$name' but STRIX_LLM asks for '$SERVED'."
    window="${STRIX_CONTEXT_FALLBACK_TOKENS:-200000}"
    if (( maxlen > 0 && window > maxlen )); then
        err "STRIX_CONTEXT_FALLBACK_TOKENS=$window is above the server's max_model_len=$maxlen: compaction would start too late."
    fi
    compact_at=$((window - ${STRIX_CONTEXT_BUFFER_TOKENS:-20000}))
    info "server OK: $name, max_model_len $maxlen, Strix compacts at $compact_at tokens"
    # Strix's own system prompt and tool schemas are ~64k tokens per agent (1.7.1,
    # measured), and its compaction count includes them: below ~90k it would
    # summarise on almost every cycle.
    (( compact_at < 90000 )) && warn "compaction at $compact_at tokens is close to Strix's ~64k base prompt; expect a summary call nearly every turn."
    busy=$(curl -s -m 10 "$METRICS_URL" | awk '/^vllm:num_requests_(running|waiting)\{/ {s += $2} END {print s + 0}')
    [[ "${busy%.*}" -gt 0 ]] && warn "the server already has ${busy%.*} request(s) running or queued; this scan shares it."

    # --- run, bracketed by metrics snapshots ---
    RUN_DIR="${STRIX_REPORT_DIR:-./strix-server-reports}/$(date +%Y%m%dT%H%M%S)"
    mkdir -p "$RUN_DIR"
    python3 "$REPO_DIR/scripts/strix-metrics.py" snapshot "$METRICS_URL" > "$RUN_DIR/metrics-before.prom" \
        || warn "could not read $METRICS_URL; the server report will be missing."
    START=$(date +%s)
    set +e
    "$STRIX_BIN" "$@"
    RC=$?
    set -e
    ELAPSED=$(( $(date +%s) - START ))
    if python3 "$REPO_DIR/scripts/strix-metrics.py" snapshot "$METRICS_URL" > "$RUN_DIR/metrics-after.prom" \
            && [[ -s "$RUN_DIR/metrics-before.prom" ]]; then
        python3 "$REPO_DIR/scripts/strix-metrics.py" diff "$RUN_DIR/metrics-before.prom" "$RUN_DIR/metrics-after.prom" \
            --wall "$ELAPSED" | tee "$RUN_DIR/server-report.txt"
        info "server report: $RUN_DIR/server-report.txt"
    fi
    return "$RC"
}

main "$@"; exit $?
