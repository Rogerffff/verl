#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
DELTA69_MANIFEST_DIR=${DELTA69_MANIFEST_DIR:-"$ROOT/coding_model_project/phase_2_ GRPO/review_assets/step640_v1/delta69_eval_slice_raw/manifests"}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_delta69"}

export ROOT
export MANIFEST_DIR="$DELTA69_MANIFEST_DIR"
export OUTPUT_BASE
export MAX_CONCURRENT=${MAX_CONCURRENT:-180}
export MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-180}
export VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-180}
export BATCH_SIZE=${BATCH_SIZE:-180}

exec bash "$ROOT/coding_model_project/phase_2_ GRPO/ops/run_grpo_codecontests_validbig_checkpoint.sh"
