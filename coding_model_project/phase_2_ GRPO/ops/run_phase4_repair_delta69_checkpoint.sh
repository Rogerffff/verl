#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
DELTA69_MANIFEST_DIR=${DELTA69_MANIFEST_DIR:-"$ROOT/coding_model_project/phase_2_ GRPO/review_assets/step640_v1/delta69_eval_slice_raw/manifests"}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_delta69_repair"}

export ROOT
export MANIFEST_DIR="$DELTA69_MANIFEST_DIR"
export OUTPUT_BASE

exec bash "$ROOT/coding_model_project/phase_2_ GRPO/ops/run_phase4_repair_eval_checkpoint.sh"
