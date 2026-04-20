#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BASE_PORT=${BASE_PORT:-8081}
export BACKEND_COUNT=${BACKEND_COUNT:-6}
export LB_BASE_PORT=${LB_BASE_PORT:-8090}
export LB_COUNT=${LB_COUNT:-1}
export BACKENDS_PER_LB=${BACKENDS_PER_LB:-6}
export MAX_BACKENDS=${MAX_BACKENDS:-8}
export PROBE_REQUESTS=${PROBE_REQUESTS:-24}
export PROBE_WORKERS=${PROBE_WORKERS:-12}

bash "$SCRIPT_DIR/setup_eval_sandbox_4x2.sh"
