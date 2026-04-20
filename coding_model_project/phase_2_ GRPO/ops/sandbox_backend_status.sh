#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/sandbox_backend_common.sh"

BASE_PORT=${BASE_PORT:-8081}
MAX_BACKENDS=${MAX_BACKENDS:-4}
PID_ROOT=${PID_ROOT:-"$(default_pid_root)"}
LOG_ROOT=${LOG_ROOT:-"$(default_log_root)"}
STATE_ROOT=${STATE_ROOT:-"$(default_state_root)"}
BASELINE_FILE=${BASELINE_FILE:-"$(default_baseline_file)"}

echo "Host metrics:"
echo "  mem_available_kb: $(read_mem_available_kb)"
echo "  tmp_available_kb: $(read_tmp_available_kb)"
echo "  swap_used_kb: $(read_swap_used_kb)"
echo

if [[ -f "$BASELINE_FILE" ]]; then
    echo "Baseline file: $BASELINE_FILE"
    cat "$BASELINE_FILE"
    echo
fi

echo "Backends:"
for (( offset=0; offset<MAX_BACKENDS; offset++ )); do
    port=$((BASE_PORT + offset))
    pid_file="$(backend_pid_file "$PID_ROOT" "$port")"
    log_file="$(backend_log_file "$LOG_ROOT" "$port")"
    metrics_file="$(backend_metrics_file "$STATE_ROOT" "$port")"
    pid=""
    rss_kb=""
    health="down"

    if [[ -f "$pid_file" ]]; then
        pid="$(<"$pid_file")"
        if pid_is_running "$pid"; then
            rss_kb="$(ps -o rss= -p "$pid" | awk '{print $1}')"
        else
            pid=""
        fi
    fi

    if curl -sf "http://127.0.0.1:${port}/v1/ping" >/dev/null; then
        health="up"
    fi

    echo "  port $port:"
    echo "    pid: ${pid:-}"
    echo "    health: $health"
    echo "    rss_kb: ${rss_kb:-}"
    echo "    pid_file: $pid_file"
    echo "    log_file: $log_file"
    if [[ -f "$metrics_file" ]]; then
        echo "    metrics_file: $metrics_file"
    fi
done
