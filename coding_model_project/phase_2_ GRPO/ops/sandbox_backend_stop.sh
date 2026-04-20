#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/sandbox_backend_common.sh"

BACKEND_COUNT_TARGET=${BACKEND_COUNT_TARGET:-4}
BASE_PORT=${BASE_PORT:-8081}
MAX_BACKENDS=${MAX_BACKENDS:-4}
PID_ROOT=${PID_ROOT:-"$(default_pid_root)"}
STATE_ROOT=${STATE_ROOT:-"$(default_state_root)"}
STAGE_STATE_FILE=${STAGE_STATE_FILE:-"$(default_stage_state_file)"}
STOP_LEGACY_8080=${STOP_LEGACY_8080:-false}

if ! [[ "$BACKEND_COUNT_TARGET" =~ ^[0-9]+$ ]]; then
    echo "BACKEND_COUNT_TARGET must be an integer" >&2
    exit 1
fi

for (( offset=0; offset<BACKEND_COUNT_TARGET && offset<MAX_BACKENDS; offset++ )); do
    port=$((BASE_PORT + offset))
    pid_file="$(backend_pid_file "$PID_ROOT" "$port")"
    pid=""
    if [[ -f "$pid_file" ]]; then
        pid="$(<"$pid_file")"
    else
        pid="$(find_backend_pid_by_port "$port" || true)"
    fi

    if [[ -n "$pid" ]]; then
        echo "Stopping backend on port $port with pid $pid"
        kill_backend_pid "$pid"
    else
        echo "No running backend found on port $port"
    fi
    rm -f "$pid_file"
    rm -f "$(backend_metrics_file "$STATE_ROOT" "$port")"
    rm -f "$(backend_probe_summary_file "$STATE_ROOT" "$port")"
    rm -f "$(backend_probe_jsonl_file "$STATE_ROOT" "$port")"
done

if [[ "$STOP_LEGACY_8080" == "true" ]]; then
    legacy_pid="$(find_backend_pid_by_port 8080 || true)"
    if [[ -n "$legacy_pid" ]]; then
        echo "Stopping legacy backend on port 8080 with pid $legacy_pid"
        kill_backend_pid "$legacy_pid"
    else
        echo "No legacy backend found on port 8080"
    fi
fi

rm -f "$STAGE_STATE_FILE"
