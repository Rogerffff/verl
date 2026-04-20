#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0}
CKPT_ROOT=${CKPT_ROOT:-"$ROOT/checkpoints/rlvr_coding_model/$EXPERIMENT_NAME"}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_validbig500"}
QUARANTINE_JSON=${QUARANTINE_JSON:-"$ROOT/coding_model_project/data/problem_quarantine_v3.json"}

GPU_DEVICE=${GPU_DEVICE:-0}
VLLM_PORT=${VLLM_PORT:-8125}
VLLM_INTERNAL_PORT_BASE=${VLLM_INTERNAL_PORT_BASE:-8325}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}

MAX_CONCURRENT=${MAX_CONCURRENT:-200}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-180}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-180}
BATCH_SIZE=${BATCH_SIZE:-200}

STEP=${STEP:-${1:-}}
if [[ -z "$STEP" ]]; then
    echo "Need STEP=<checkpoint_step> or first positional arg" >&2
    exit 1
fi

wait_checkpoint_ready() {
    local ckpt_dir="$1"
    local max_tries="${2:-720}"
    local i
    for ((i=1; i<=max_tries; i++)); do
        if [[ -f "$ckpt_dir/data.pt" && -f "$ckpt_dir/actor/fsdp_config.json" && -f "$ckpt_dir/actor/huggingface/config.json" ]]; then
            return 0
        fi
        sleep 5
    done
    echo "checkpoint not ready: $ckpt_dir" >&2
    return 1
}

CKPT_DIR="${CKPT_ROOT}/global_step_${STEP}"
EVAL_NAME="${EXPERIMENT_NAME}_step${STEP}_codecontests_validbig500_raw"
OUT_DIR="${OUTPUT_BASE}/${EVAL_NAME}"
CLEAN_LOG="${LOG_DIR}/${EXPERIMENT_NAME}_step${STEP}_codecontests_validbig500_clean.launch.log"

wait_checkpoint_ready "$CKPT_DIR"

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
    CKPT_DIR="$CKPT_DIR" \
    EVAL_NAME="$EVAL_NAME" \
    LOG_DIR="$LOG_DIR" \
    OUTPUT_BASE="$OUTPUT_BASE" \
    bash "$ROOT/coding_model_project/phase_2_ GRPO/ops/run_grpo_codecontests_validbig_checkpoint.sh"

python3 "$ROOT/coding_model_project/src/compute_eval_clean_overlay.py" \
    --per-problem-jsonl "$OUT_DIR/per_problem/codecontests_valid_big.jsonl" \
    --quarantine-json "$QUARANTINE_JSON" \
    --output-json "$OUT_DIR/clean_overlay_problem_quarantine_v3.json" \
    >"$CLEAN_LOG" 2>&1
