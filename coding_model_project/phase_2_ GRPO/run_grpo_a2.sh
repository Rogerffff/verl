#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_VARIANT=A2 \
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_formal_a2} \
bash "$SCRIPT_DIR/run_grpo_formal.sh" "$@"
