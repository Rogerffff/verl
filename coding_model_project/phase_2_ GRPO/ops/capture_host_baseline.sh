#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/sandbox_backend_common.sh"

PROJECT_ROOT=${PROJECT_ROOT:-"$(repo_root_from_ops_dir "$SCRIPT_DIR")"}
STATE_ROOT=${STATE_ROOT:-"$(default_state_root)"}
BASELINE_FILE=${BASELINE_FILE:-"$(default_baseline_file)"}
LOG_ROOT=${LOG_ROOT:-"$(default_log_root)"}
PID_ROOT=${PID_ROOT:-"$(default_pid_root)"}
BASE_PORT=${BASE_PORT:-8081}
MAX_BACKENDS=${MAX_BACKENDS:-4}

ensure_dir "$STATE_ROOT/state"
ensure_dir "$LOG_ROOT"
ensure_dir "$PID_ROOT"

captured_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"
baseline_mem_available_kb="$(read_mem_available_kb)"
baseline_tmp_available_kb="$(read_tmp_available_kb)"
baseline_swap_used_kb="$(read_swap_used_kb)"
baseline_active_backends="$(count_active_backends "$PID_ROOT" "$BASE_PORT" "$MAX_BACKENDS")"

write_env_file "$BASELINE_FILE" \
    CAPTURED_AT "$captured_at" \
    BASELINE_MEM_AVAILABLE_KB "$baseline_mem_available_kb" \
    BASELINE_TMP_AVAILABLE_KB "$baseline_tmp_available_kb" \
    BASELINE_SWAP_USED_KB "$baseline_swap_used_kb" \
    BASELINE_ACTIVE_BACKENDS "$baseline_active_backends" \
    PROJECT_ROOT "$PROJECT_ROOT" \
    BASE_PORT "$BASE_PORT"

echo "Captured host baseline:"
echo "  file: $BASELINE_FILE"
echo "  captured_at: $captured_at"
echo "  mem_available_kb: $baseline_mem_available_kb"
echo "  tmp_available_kb: $baseline_tmp_available_kb"
echo "  swap_used_kb: $baseline_swap_used_kb"
echo "  active_backends: $baseline_active_backends"
