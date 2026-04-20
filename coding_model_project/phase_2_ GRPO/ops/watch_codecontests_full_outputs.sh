#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs/codecontests_full_outputs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs"}
STEPS=${STEPS:-"100 120 160 180 200"}
POLL_SECONDS=${POLL_SECONDS:-30}
MIN_LINES=${MIN_LINES:-117}

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG_DIR/watch_copy.log"
}

src_for_step() {
    local step="$1"
    if [[ "$step" == "100" ]]; then
        echo "$OUTPUT_BASE/phase0_fullval_step100_same_protocol/per_problem/codecontests_valid.jsonl"
    else
        echo "$OUTPUT_BASE/phase0_fullval_step${step}_lb24_multi2/per_problem/codecontests_valid.jsonl"
    fi
}

main() {
    log "START watch_codecontests_full_outputs steps=${STEPS}"
    local step
    for step in $STEPS; do
        local src
        local dst
        src="$(src_for_step "$step")"
        dst="$LOG_DIR/step${step}_codecontests_valid_full.jsonl"
        while [[ ! -f "$src" ]]; do
            sleep "$POLL_SECONDS"
        done
        while true; do
            local line_count
            line_count="$(wc -l < "$src" 2>/dev/null || echo 0)"
            if [[ "${line_count}" -ge "${MIN_LINES}" ]]; then
                break
            fi
            sleep "$POLL_SECONDS"
        done
        cp -f "$src" "$dst"
        log "COPIED step=${step} src=${src} dst=${dst} lines=$(wc -l < "$dst")"
    done
    log "DONE watch_codecontests_full_outputs"
}

main "$@"
