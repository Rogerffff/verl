#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BASE_PORT=${BASE_PORT:-8081}
export BACKEND_COUNT=${BACKEND_COUNT:-8}
export LB_BASE_PORT=${LB_BASE_PORT:-8090}
export LB_COUNT=${LB_COUNT:-1}
export BACKENDS_PER_LB=${BACKENDS_PER_LB:-8}
export MAX_BACKENDS=${MAX_BACKENDS:-8}
export PROBE_REQUESTS=${PROBE_REQUESTS:-32}
export PROBE_WORKERS=${PROBE_WORKERS:-16}
export REL_MEM_DROP_LIMIT_KB=${REL_MEM_DROP_LIMIT_KB:-$((64 * 1024 * 1024))}
export BASELINE_SWAP_USED_KB=${BASELINE_SWAP_USED_KB:-$(awk '/SwapTotal:/ {t=$2} /SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)}

bash "$SCRIPT_DIR/setup_eval_sandbox_4x2.sh"
