#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-/workspace/verl/coding_model_project}

SETUP_SANDBOX=${SETUP_SANDBOX:-true}
if [[ "$SETUP_SANDBOX" == "true" ]]; then
  bash "$SCRIPT_DIR/setup_repair_v2b_step400_isolated_sandbox.sh"
fi

exec env \
  REPAIR_DATA_DIR="${REPAIR_DATA_DIR:-${PROJECT_ROOT}/phase_2_ GRPO/sft_repair_data/v2b_step400}" \
  SPEC_PATH="${SPEC_PATH:-${PROJECT_ROOT}/phase_2_ GRPO/sft_repair_data/v2b_step400/repair_v2b_step400_spec.json}" \
  SANDBOX_URL="${SANDBOX_URL:-http://localhost:8096}" \
  VERIFIER_LIMITER_BUDGET="${VERIFIER_LIMITER_BUDGET:-64}" \
  MAX_CONCURRENT="${MAX_CONCURRENT:-64}" \
  MAX_CONCURRENT_JUDGES="${MAX_CONCURRENT_JUDGES:-64}" \
  BATCH_SIZE="${BATCH_SIZE:-64}" \
  GPU_DEVICE="${GPU_DEVICE:-2}" \
  VLLM_PORT="${VLLM_PORT:-8004}" \
  CKPT_DIR="${CKPT_DIR:-/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0/global_step_400}" \
  EVAL_NAME="${EVAL_NAME:-global_step_400_stepref_v2b_step400}" \
  PREPARE_STATIC_ASSETS="${PREPARE_STATIC_ASSETS:-true}" \
  BUILD_TEACHER_REQUESTS="${BUILD_TEACHER_REQUESTS:-false}" \
  bash "$SCRIPT_DIR/run_repair_v2a_student_reference_eval.sh"
