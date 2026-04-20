#!/bin/bash

repo_root_from_ops_dir() {
    local ops_dir="$1"
    (cd "$ops_dir/../../.." && pwd)
}

default_state_root() {
    if [[ -n "${STATE_ROOT:-}" ]]; then
        echo "$STATE_ROOT"
        return 0
    fi

    local candidate pid_file pid
    for candidate in /root/sandboxfusion-multi /workspace/sandboxfusion-multi; do
        if [[ ! -d "$candidate/pids" ]]; then
            continue
        fi
        for pid_file in "$candidate"/pids/*.pid; do
            [[ -f "$pid_file" ]] || continue
            pid="$(<"$pid_file")"
            if pid_is_running "$pid"; then
                echo "$candidate"
                return 0
            fi
        done
    done

    for candidate in /root/sandboxfusion-multi /workspace/sandboxfusion-multi; do
        if [[ -d "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    echo "/root/sandboxfusion-multi"
}

default_sandbox_venv() {
    if [[ -n "${SANDBOX_VENV:-}" ]]; then
        echo "$SANDBOX_VENV"
        return 0
    fi

    local python_path
    python_path="$(pgrep -af 'uvicorn .*sandbox\.server\.server:app' 2>/dev/null | awk 'NR==1 {print $2}')"
    if [[ -n "$python_path" ]] && [[ "$python_path" == */bin/python* ]]; then
        echo "${python_path%/bin/*}"
        return 0
    fi

    local candidate
    for candidate in /root/sandboxfusion-venv /workspace/sandboxfusion-venv; do
        if [[ -x "$candidate/bin/python3" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    echo "/root/sandboxfusion-venv"
}

default_log_root() {
    echo "${LOG_ROOT:-$(default_state_root)/logs}"
}

default_pid_root() {
    echo "${PID_ROOT:-$(default_state_root)/pids}"
}

default_stage_state_file() {
    echo "${STAGE_STATE_FILE:-$(default_state_root)/state/stage_state.env}"
}

default_baseline_file() {
    echo "${BASELINE_FILE:-$(default_state_root)/state/host_baseline.env}"
}

ensure_dir() {
    mkdir -p "$1"
}

read_mem_available_kb() {
    if [[ -r /proc/meminfo ]]; then
        awk '/MemAvailable:/ {print $2}' /proc/meminfo
    else
        echo "0"
    fi
}

read_swap_used_kb() {
    if [[ -r /proc/meminfo ]]; then
        awk '
            /SwapTotal:/ {total=$2}
            /SwapFree:/ {free=$2}
            END {print total - free}
        ' /proc/meminfo
    else
        echo "0"
    fi
}

read_tmp_available_kb() {
    df -Pk /tmp | awk 'NR==2 {print $4}'
}

backend_log_file() {
    local log_root="$1"
    local port="$2"
    echo "$log_root/sandbox_${port}.log"
}

backend_pid_file() {
    local pid_root="$1"
    local port="$2"
    echo "$pid_root/sandbox_${port}.pid"
}

backend_metrics_file() {
    local state_root="$1"
    local port="$2"
    echo "$state_root/state/backend_${port}.env"
}

backend_probe_summary_file() {
    local state_root="$1"
    local port="$2"
    echo "$state_root/state/backend_${port}_probe_summary.json"
}

backend_probe_jsonl_file() {
    local state_root="$1"
    local port="$2"
    echo "$state_root/state/backend_${port}_probe.jsonl"
}

uvicorn_pattern_for_port() {
    local port="$1"
    echo "uvicorn sandbox.server.server:app --host 0.0.0.0 --port ${port}"
}

find_listener_pid_by_port() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1
        return 0
    fi

    if command -v ss >/dev/null 2>&1; then
        ss -ltnp "( sport = :${port} )" 2>/dev/null \
            | awk -F 'pid=' 'NR>1 && NF>1 {split($2, tail, ","); print tail[1]; exit}'
        return 0
    fi

    return 0
}

find_backend_pid_by_port() {
    local port="$1"
    local pattern pid command
    pattern="$(uvicorn_pattern_for_port "$port")"
    pid="$(pgrep -f "$pattern" | head -n 1 || true)"
    if [[ -n "$pid" ]]; then
        echo "$pid"
        return 0
    fi

    pid="$(find_listener_pid_by_port "$port" || true)"
    if [[ -n "$pid" ]]; then
        command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
        if [[ "$command" == *"sandbox.server.server:app"* ]]; then
            echo "$pid"
            return 0
        fi
    fi

    pgrep -af 'uvicorn .*sandbox\.server\.server:app' 2>/dev/null \
        | awk -v port="$port" 'index($0, "--port " port) {print $1; exit}'
}

pid_is_running() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

pid_matches_sandbox_backend() {
    local pid="$1"
    local port="$2"
    local command
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [[ -n "$command" ]] || return 1
    [[ "$command" == *"sandbox.server.server:app"* ]] || return 1
    [[ "$command" == *"--port ${port}"* ]]
}

wait_for_backend_ping() {
    local port="$1"
    local timeout_s="$2"
    local deadline=$((SECONDS + timeout_s))
    while (( SECONDS < deadline )); do
        if curl -sf "http://127.0.0.1:${port}/v1/ping" >/dev/null; then
            return 0
        fi
        sleep 2
    done
    return 1
}

start_backend_via_make() {
    local project_root="$1"
    local port="$2"
    local log_file="$3"
    local pod_name="$4"
    local sandbox_config="$5"
    local sandbox_venv="${SANDBOX_VENV:-$(default_sandbox_venv)}"

    ensure_dir "$(dirname "$log_file")"
    (
        cd "$project_root/SandboxFusion"
        if [[ -d "$sandbox_venv/bin" ]]; then
            export PATH="$sandbox_venv/bin:$PATH"
            export VIRTUAL_ENV="$sandbox_venv"
        fi
        MY_POD_NAME="$pod_name" SANDBOX_CONFIG="$sandbox_config" PYTHONUNBUFFERED=1 \
            nohup make run-online HOST=0.0.0.0 PORT="$port" >"$log_file" 2>&1 < /dev/null &
    )
}

write_env_file() {
    local output_file="$1"
    shift

    ensure_dir "$(dirname "$output_file")"
    : >"$output_file"
    while [[ $# -gt 0 ]]; do
        printf '%s=%q\n' "$1" "$2" >>"$output_file"
        shift 2
    done
}

read_json_field() {
    local json_file="$1"
    local field_path="$2"
    python3 - "$json_file" "$field_path" <<'PY'
import json
import sys

json_file = sys.argv[1]
field_path = sys.argv[2].split(".")

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

value = data
for part in field_path:
    if not part:
        continue
    value = value[part]

if value is None:
    print("")
else:
    print(value)
PY
}

active_backend_ports() {
    local pid_root="$1"
    local base_port="$2"
    local max_backends="$3"
    local offset port pid_file pid
    for (( offset=0; offset<max_backends; offset++ )); do
        port=$((base_port + offset))
        pid_file="$(backend_pid_file "$pid_root" "$port")"
        if [[ -f "$pid_file" ]]; then
            pid="$(<"$pid_file")"
            if pid_is_running "$pid" && pid_matches_sandbox_backend "$pid" "$port"; then
                echo "$port"
            else
                rm -f "$pid_file"
            fi
        fi
    done
}

count_active_backends() {
    local pid_root="$1"
    local base_port="$2"
    local max_backends="$3"
    active_backend_ports "$pid_root" "$base_port" "$max_backends" | wc -l | awk '{print $1}'
}

kill_backend_pid() {
    local pid="$1"
    if ! pid_is_running "$pid"; then
        return 0
    fi

    kill "$pid" >/dev/null 2>&1 || true
    for _ in $(seq 1 10); do
        if ! pid_is_running "$pid"; then
            return 0
        fi
        sleep 1
    done

    kill -9 "$pid" >/dev/null 2>&1 || true
}
