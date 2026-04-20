#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/sandbox_backend_common.sh"

BACKEND_START_SCRIPT=${BACKEND_START_SCRIPT:-"$SCRIPT_DIR/sandbox_backend_start.sh"}
RENDER_SCRIPT=${RENDER_SCRIPT:-"$SCRIPT_DIR/render_nginx_sandbox_lb.sh"}
APPLY_SCRIPT=${APPLY_SCRIPT:-"$SCRIPT_DIR/apply_nginx_sandbox_lb.sh"}
PROBE_SCRIPT=${PROBE_SCRIPT:-"$SCRIPT_DIR/lb_validate_probe.py"}

BASE_PORT=${BASE_PORT:-8081}
BACKEND_COUNT=${BACKEND_COUNT:-8}
LB_BASE_PORT=${LB_BASE_PORT:-8090}
LB_COUNT=${LB_COUNT:-4}
BACKENDS_PER_LB=${BACKENDS_PER_LB:-2}
MAX_BACKENDS=${MAX_BACKENDS:-8}
STATE_ROOT=${STATE_ROOT:-"$(default_state_root)"}
SANDBOX_VENV=${SANDBOX_VENV:-"$(default_sandbox_venv)"}
PID_ROOT=${PID_ROOT:-"$STATE_ROOT/pids"}
PROBE_REQUESTS=${PROBE_REQUESTS:-8}
PROBE_WORKERS=${PROBE_WORKERS:-4}

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

if (( LB_COUNT * BACKENDS_PER_LB != BACKEND_COUNT )); then
    echo "LB_COUNT * BACKENDS_PER_LB must equal BACKEND_COUNT" >&2
    exit 1
fi

log "START backends=${BACKEND_COUNT} base_port=${BASE_PORT}"
BACKEND_COUNT_TARGET="$BACKEND_COUNT" \
BASE_PORT="$BASE_PORT" \
MAX_BACKENDS="$MAX_BACKENDS" \
STATE_ROOT="$STATE_ROOT" \
PID_ROOT="$PID_ROOT" \
SANDBOX_VENV="$SANDBOX_VENV" \
bash "$BACKEND_START_SCRIPT"

active_count="$(count_active_backends "$PID_ROOT" "$BASE_PORT" "$MAX_BACKENDS")"
if (( active_count != BACKEND_COUNT )); then
    echo "Expected ${BACKEND_COUNT} active backends after setup, found ${active_count}. Refusing to configure ${LB_COUNT} load balancers." >&2
    exit 1
fi

for (( slot=0; slot<LB_COUNT; slot++ )); do
    lb_port=$((LB_BASE_PORT + slot))
    backend_start=$((BASE_PORT + slot * BACKENDS_PER_LB))
    backend_end=$((backend_start + BACKENDS_PER_LB - 1))
    backend_ports_csv=""
    for (( port=backend_start; port<=backend_end; port++ )); do
        if [[ -n "$backend_ports_csv" ]]; then
            backend_ports_csv+=","
        fi
        backend_ports_csv+="$port"
    done

    config_path="$SCRIPT_DIR/generated/nginx_sandbox_lb_${lb_port}.conf"
    access_log="$STATE_ROOT/nginx/sandbox_lb_${lb_port}.access.log"
    error_log="$STATE_ROOT/nginx/sandbox_lb_${lb_port}.error.log"
    pid_file="$STATE_ROOT/nginx/nginx_${lb_port}.pid"
    summary_file="$STATE_ROOT/state/lb_${lb_port}_probe_summary.json"
    jsonl_file="$STATE_ROOT/state/lb_${lb_port}_probe.jsonl"

    log "RENDER lb_port=${lb_port} backends=${backend_ports_csv}"
    LISTEN_PORT="$lb_port" \
    BASE_PORT="$BASE_PORT" \
    MAX_BACKENDS="$MAX_BACKENDS" \
    STATE_ROOT="$STATE_ROOT" \
    PID_ROOT="$PID_ROOT" \
    BACKEND_PORTS_CSV="$backend_ports_csv" \
    OUTPUT_PATH="$config_path" \
    ACCESS_LOG="$access_log" \
    ERROR_LOG="$error_log" \
    PID_FILE="$pid_file" \
    bash "$RENDER_SCRIPT"

    log "APPLY lb_port=${lb_port}"
    STATE_ROOT="$STATE_ROOT" \
    PID_ROOT="$PID_ROOT" \
    CONFIG_PATH="$config_path" \
    PID_FILE="$pid_file" \
    bash "$APPLY_SCRIPT"

    log "PROBE lb_port=${lb_port}"
    python3 "$PROBE_SCRIPT" \
        --endpoint "http://127.0.0.1:${lb_port}" \
        --requests "$PROBE_REQUESTS" \
        --workers "$PROBE_WORKERS" \
        --jsonl-output "$jsonl_file" \
        --summary-output "$summary_file" \
        --require-all-success >/dev/null
done

log "DONE setup lb_ports=$(seq -s, "$LB_BASE_PORT" $((LB_BASE_PORT + LB_COUNT - 1))) backends=${BACKEND_COUNT}"
