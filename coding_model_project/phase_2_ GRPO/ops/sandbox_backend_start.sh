#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/sandbox_backend_common.sh"

PROJECT_ROOT=${PROJECT_ROOT:-"$(repo_root_from_ops_dir "$SCRIPT_DIR")"}
BACKEND_COUNT_TARGET=${BACKEND_COUNT_TARGET:-2}
BASE_PORT=${BASE_PORT:-8081}
MAX_BACKENDS=${MAX_BACKENDS:-4}
STATE_ROOT=${STATE_ROOT:-"$(default_state_root)"}
LOG_ROOT=${LOG_ROOT:-"$(default_log_root)"}
PID_ROOT=${PID_ROOT:-"$(default_pid_root)"}
BASELINE_FILE=${BASELINE_FILE:-"$(default_baseline_file)"}
STAGE_STATE_FILE=${STAGE_STATE_FILE:-"$(default_stage_state_file)"}
SANDBOX_CONFIG=${SANDBOX_CONFIG:-local}
START_TIMEOUT_S=${START_TIMEOUT_S:-60}
WARMUP_S=${WARMUP_S:-3}
SMOKE_REQUESTS=${SMOKE_REQUESTS:-6}
SMOKE_WORKERS=${SMOKE_WORKERS:-2}
ABS_MEM_LIMIT_KB=${ABS_MEM_LIMIT_KB:-$((128 * 1024 * 1024))}
REL_MEM_DROP_LIMIT_KB=${REL_MEM_DROP_LIMIT_KB:-$((16 * 1024 * 1024))}
TMP_LIMIT_KB=${TMP_LIMIT_KB:-$((40 * 1024 * 1024))}
RSS_LIMIT_KB=${RSS_LIMIT_KB:-$((4 * 1024 * 1024))}

if ! [[ "$BACKEND_COUNT_TARGET" =~ ^[0-9]+$ ]]; then
    echo "BACKEND_COUNT_TARGET must be an integer" >&2
    exit 1
fi
if (( BACKEND_COUNT_TARGET < 1 || BACKEND_COUNT_TARGET > MAX_BACKENDS )); then
    echo "BACKEND_COUNT_TARGET must be between 1 and $MAX_BACKENDS" >&2
    exit 1
fi

ensure_dir "$STATE_ROOT/state"
ensure_dir "$LOG_ROOT"
ensure_dir "$PID_ROOT"

if [[ -f "$BASELINE_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$BASELINE_FILE"
else
    echo "Baseline file not found at $BASELINE_FILE; capturing current host state as baseline." >&2
    BASELINE_MEM_AVAILABLE_KB="$(read_mem_available_kb)"
    BASELINE_TMP_AVAILABLE_KB="$(read_tmp_available_kb)"
    BASELINE_SWAP_USED_KB="$(read_swap_used_kb)"
fi

active_count_before_start="$(count_active_backends "$PID_ROOT" "$BASE_PORT" "$MAX_BACKENDS")"
if (( active_count_before_start == 0 )); then
    rm -f "$STAGE_STATE_FILE"
fi

baseline_swap_used_kb="${BASELINE_SWAP_USED_KB:-0}"
previous_stage_mem_available_kb="${BASELINE_MEM_AVAILABLE_KB:-$(read_mem_available_kb)}"
previous_stage_smoke_p95_ms=""
if (( active_count_before_start > 0 )) && [[ -f "$STAGE_STATE_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$STAGE_STATE_FILE"
    previous_stage_mem_available_kb="${PREVIOUS_STAGE_MEM_AVAILABLE_KB:-$previous_stage_mem_available_kb}"
    previous_stage_smoke_p95_ms="${PREVIOUS_STAGE_SMOKE_P95_MS:-}"
fi

stop_reason=""
for (( offset=0; offset<BACKEND_COUNT_TARGET; offset++ )); do
    port=$((BASE_PORT + offset))
    pid_file="$(backend_pid_file "$PID_ROOT" "$port")"
    log_file="$(backend_log_file "$LOG_ROOT" "$port")"
    metrics_file="$(backend_metrics_file "$STATE_ROOT" "$port")"
    summary_file="$(backend_probe_summary_file "$STATE_ROOT" "$port")"
    jsonl_file="$(backend_probe_jsonl_file "$STATE_ROOT" "$port")"

    if [[ -f "$pid_file" ]]; then
        pid="$(<"$pid_file")"
        if pid_is_running "$pid"; then
            echo "Backend on port $port is already running with pid $pid; skipping start."
            continue
        fi
        rm -f "$pid_file"
    fi

    pod_name="sandbox-${port}"
    echo "Starting backend on port $port with pod name $pod_name"
    start_backend_via_make "$PROJECT_ROOT" "$port" "$log_file" "$pod_name" "$SANDBOX_CONFIG"

    if ! wait_for_backend_ping "$port" "$START_TIMEOUT_S"; then
        stop_reason="backend ${port} did not become healthy within ${START_TIMEOUT_S}s"
        break
    fi

    sleep "$WARMUP_S"

    pid="$(find_backend_pid_by_port "$port")"
    if [[ -z "$pid" ]]; then
        stop_reason="could not resolve uvicorn pid for backend ${port}"
        break
    fi
    printf '%s\n' "$pid" >"$pid_file"

    if ! python3 "$SCRIPT_DIR/lb_validate_probe.py" \
        --endpoint "http://127.0.0.1:${port}" \
        --requests "$SMOKE_REQUESTS" \
        --workers "$SMOKE_WORKERS" \
        --jsonl-output "$jsonl_file" \
        --summary-output "$summary_file" \
        --require-all-success >/dev/null; then
        stop_reason="direct smoke failed for backend ${port}"
        kill_backend_pid "$pid"
        rm -f "$pid_file"
        break
    fi

    rss_kb="$(ps -o rss= -p "$pid" | awk '{print $1}')"
    mem_available_kb="$(read_mem_available_kb)"
    tmp_available_kb="$(read_tmp_available_kb)"
    swap_used_kb="$(read_swap_used_kb)"
    smoke_p95_ms="$(read_json_field "$summary_file" "latency_ms.p95")"

    write_env_file "$metrics_file" \
        PORT "$port" \
        PID "$pid" \
        RSS_KB "$rss_kb" \
        MEM_AVAILABLE_KB "$mem_available_kb" \
        TMP_AVAILABLE_KB "$tmp_available_kb" \
        SWAP_USED_KB "$swap_used_kb" \
        SMOKE_P95_MS "$smoke_p95_ms" \
        LOG_FILE "$log_file" \
        SUMMARY_FILE "$summary_file"

    if (( rss_kb > RSS_LIMIT_KB )); then
        stop_reason="backend ${port} exceeded rss guardrail (${rss_kb}KB > ${RSS_LIMIT_KB}KB)"
    elif [[ -n "${BASELINE_MEM_AVAILABLE_KB:-}" ]] \
        && (( BASELINE_MEM_AVAILABLE_KB >= ABS_MEM_LIMIT_KB )) \
        && (( mem_available_kb < ABS_MEM_LIMIT_KB )); then
        stop_reason="backend ${port} pushed MemAvailable below 128GiB"
    elif (( previous_stage_mem_available_kb - mem_available_kb > REL_MEM_DROP_LIMIT_KB )); then
        stop_reason="backend ${port} reduced MemAvailable by more than 16GiB versus previous stage"
    elif (( tmp_available_kb < TMP_LIMIT_KB )); then
        stop_reason="backend ${port} pushed /tmp free space below 40GiB"
    elif (( swap_used_kb > baseline_swap_used_kb )); then
        stop_reason="backend ${port} increased swap usage above clean-host baseline (${swap_used_kb}KB > ${baseline_swap_used_kb}KB)"
    elif [[ -n "$previous_stage_smoke_p95_ms" ]]; then
        if ! python3 - "$previous_stage_smoke_p95_ms" "$smoke_p95_ms" <<'PY'
import sys

previous = float(sys.argv[1])
current = float(sys.argv[2])
sys.exit(0 if current <= previous * 2 else 1)
PY
        then
            stop_reason="backend ${port} smoke p95 (${smoke_p95_ms}ms) exceeded 2x previous stage (${previous_stage_smoke_p95_ms}ms)"
        fi
    fi

    if [[ -n "$stop_reason" ]]; then
        echo "Stopping expansion: $stop_reason" >&2
        kill_backend_pid "$pid"
        rm -f "$pid_file"
        break
    fi

    previous_stage_mem_available_kb="$mem_available_kb"
    previous_stage_smoke_p95_ms="$smoke_p95_ms"
    write_env_file "$STAGE_STATE_FILE" \
        PREVIOUS_STAGE_MEM_AVAILABLE_KB "$previous_stage_mem_available_kb" \
        PREVIOUS_STAGE_SMOKE_P95_MS "$previous_stage_smoke_p95_ms" \
        LAST_ACCEPTED_PORT "$port"

    echo "Accepted backend $port:"
    echo "  pid: $pid"
    echo "  rss_kb: $rss_kb"
    echo "  mem_available_kb: $mem_available_kb"
    echo "  tmp_available_kb: $tmp_available_kb"
    echo "  swap_used_kb: $swap_used_kb"
    echo "  smoke_p95_ms: $smoke_p95_ms"
done

active_count="$(count_active_backends "$PID_ROOT" "$BASE_PORT" "$MAX_BACKENDS")"
echo "Active staged backends: $active_count"

if [[ -n "$stop_reason" ]]; then
    echo "Expansion stopped: $stop_reason" >&2
fi

if (( active_count < 2 )); then
    echo "Fewer than 2 backends passed guardrails; aborting multi-sandbox staging on this host." >&2
    exit 1
fi
