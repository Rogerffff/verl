#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_a1_curriculum_step620_v3q3_to900_seed0}
CKPT_ROOT=${CKPT_ROOT:-"$ROOT/checkpoints/rlvr_coding_model/$EXPERIMENT_NAME"}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_validbig500"}
QUARANTINE_JSON=${QUARANTINE_JSON:-"$ROOT/coding_model_project/data/problem_quarantine_v3.json"}

GPU_DEVICE=${GPU_DEVICE:-3}
VLLM_PORT=${VLLM_PORT:-8019}
VLLM_INTERNAL_PORT_BASE=${VLLM_INTERNAL_PORT_BASE:-8219}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}

MAX_CONCURRENT=${MAX_CONCURRENT:-200}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-180}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-180}
BATCH_SIZE=${BATCH_SIZE:-200}

run_one() {
    local step="$1"
    local eval_name="${EXPERIMENT_NAME}_step${step}_codecontests_validbig500_raw"
    local out_dir="${OUTPUT_BASE}/${eval_name}"
    local raw_log="${LOG_DIR}/${eval_name}.launch.log"
    local clean_log="${LOG_DIR}/${EXPERIMENT_NAME}_step${step}_codecontests_validbig500_clean.launch.log"

    env \
        ROOT="$ROOT" \
        GPU_DEVICE="$GPU_DEVICE" \
        VLLM_PORT="$VLLM_PORT" \
        VLLM_INTERNAL_PORT_BASE="$VLLM_INTERNAL_PORT_BASE" \
        SANDBOX_URL="$SANDBOX_URL" \
        MAX_CONCURRENT="$MAX_CONCURRENT" \
        MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
        VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
        BATCH_SIZE="$BATCH_SIZE" \
        CKPT_DIR="$CKPT_ROOT/global_step_${step}" \
        EVAL_NAME="$eval_name" \
        LOG_DIR="$LOG_DIR" \
        OUTPUT_BASE="$OUTPUT_BASE" \
        bash "$ROOT/coding_model_project/phase_2_ GRPO/ops/run_grpo_codecontests_validbig_checkpoint.sh" \
        >"$raw_log" 2>&1

    python3 "$ROOT/coding_model_project/src/compute_eval_clean_overlay.py" \
        --per-problem-jsonl "$out_dir/per_problem/codecontests_valid_big.jsonl" \
        --quarantine-json "$QUARANTINE_JSON" \
        --output-json "$out_dir/clean_overlay_problem_quarantine_v3.json" \
        >"$clean_log" 2>&1
}

run_one 800
run_one 700
