#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT=${ROOT:-/workspace/verl}
WAIT_RUN_NAME=${WAIT_RUN_NAME:-grpo_a1_formal_observe_lb64_multi4_resume400_to460_seed0}
TARGET_STEP=${TARGET_STEP:-460}
WAIT_CKPT_ROOT=${WAIT_CKPT_ROOT:-"$ROOT/checkpoints/rlvr_coding_model/$WAIT_RUN_NAME"}
WAIT_CKPT_DIR=${WAIT_CKPT_DIR:-"$WAIT_CKPT_ROOT/global_step_${TARGET_STEP}"}
WAIT_VALIDBIG_OUTPUT_BASE=${WAIT_VALIDBIG_OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_validbig500"}
WAIT_VALIDBIG_EVAL_NAME=${WAIT_VALIDBIG_EVAL_NAME:-"${WAIT_RUN_NAME}_step${TARGET_STEP}_codecontests_validbig500"}
WAIT_VALIDBIG_SUMMARY=${WAIT_VALIDBIG_SUMMARY:-"$WAIT_VALIDBIG_OUTPUT_BASE/$WAIT_VALIDBIG_EVAL_NAME/summary.json"}

STEP200_CKPT_DIR=${STEP200_CKPT_DIR:-"$ROOT/checkpoints/rlvr_coding_model/global_step_200"}
WAIT_TIMEOUT_S=${WAIT_TIMEOUT_S:-21600}
POLL_INTERVAL_S=${POLL_INTERVAL_S:-30}

GPU_DEVICE=${GPU_DEVICE:-1}
VLLM_PORT=${VLLM_PORT:-8002}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8094}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-24}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-24}
BATCH_SIZE=${BATCH_SIZE:-24}
RUN_TIMEOUT=${RUN_TIMEOUT:-30}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-20000}
MAX_RESPONSE_CHARS=${MAX_RESPONSE_CHARS:-20000}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/repair_v2a_student_reference_eval"}
EVAL_NAME=${EVAL_NAME:-global_step_200_stepref_after_step460_v2a}

PREPARE_STATIC_ASSETS=${PREPARE_STATIC_ASSETS:-false}
PREPARE_ONLY=${PREPARE_ONLY:-false}
FORCE_EVAL=${FORCE_EVAL:-false}
KEEP_VLLM=${KEEP_VLLM:-false}
REMOVE_MERGED_AFTER_EVAL=${REMOVE_MERGED_AFTER_EVAL:-false}
BUILD_TEACHER_REQUESTS=${BUILD_TEACHER_REQUESTS:-false}

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

checkpoint_ready() {
    [[ -f "$WAIT_CKPT_DIR/data.pt" ]] || return 1
    [[ -f "$WAIT_CKPT_DIR/actor/fsdp_config.json" ]] || return 1
    [[ -f "$WAIT_CKPT_DIR/actor/huggingface/config.json" ]] || return 1
    local rank
    for rank in 0 1 2 3; do
        [[ -f "$WAIT_CKPT_DIR/actor/model_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$WAIT_CKPT_DIR/actor/optim_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$WAIT_CKPT_DIR/actor/extra_state_world_size_4_rank_${rank}.pt" ]] || return 1
    done
}

wait_for_checkpoint() {
    local waited=0
    while ! checkpoint_ready; do
        if (( waited >= WAIT_TIMEOUT_S )); then
            echo "Timed out waiting for checkpoint readiness: $WAIT_CKPT_DIR" >&2
            return 1
        fi
        log "WAIT_CHECKPOINT ckpt=${WAIT_CKPT_DIR} waited_s=${waited}"
        sleep "$POLL_INTERVAL_S"
        waited=$((waited + POLL_INTERVAL_S))
    done
    log "CHECKPOINT_READY ckpt=${WAIT_CKPT_DIR}"
}

wait_for_validbig_completion() {
    local waited=0
    while [[ ! -f "$WAIT_VALIDBIG_SUMMARY" ]]; do
        if (( waited >= WAIT_TIMEOUT_S )); then
            echo "Timed out waiting for valid_big summary: $WAIT_VALIDBIG_SUMMARY" >&2
            return 1
        fi
        log "WAIT_VALIDBIG summary=${WAIT_VALIDBIG_SUMMARY} waited_s=${waited}"
        sleep "$POLL_INTERVAL_S"
        waited=$((waited + POLL_INTERVAL_S))
    done
    log "VALIDBIG_READY summary=${WAIT_VALIDBIG_SUMMARY}"
}

main() {
    mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

    [[ -d "$STEP200_CKPT_DIR" ]] || {
        echo "Missing step200 checkpoint directory: $STEP200_CKPT_DIR" >&2
        return 1
    }

    wait_for_checkpoint
    wait_for_validbig_completion

    GPU_DEVICE="$GPU_DEVICE" \
    VLLM_PORT="$VLLM_PORT" \
    SANDBOX_URL="$SANDBOX_URL" \
    VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
    MAX_CONCURRENT="$MAX_CONCURRENT" \
    MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
    BATCH_SIZE="$BATCH_SIZE" \
    RUN_TIMEOUT="$RUN_TIMEOUT" \
    MAX_TOKENS="$MAX_TOKENS" \
    MAX_PROMPT_CHARS="$MAX_PROMPT_CHARS" \
    MAX_RESPONSE_CHARS="$MAX_RESPONSE_CHARS" \
    LOG_DIR="$LOG_DIR" \
    OUTPUT_BASE="$OUTPUT_BASE" \
    EVAL_NAME="$EVAL_NAME" \
    PREPARE_STATIC_ASSETS="$PREPARE_STATIC_ASSETS" \
    PREPARE_ONLY="$PREPARE_ONLY" \
    FORCE_EVAL="$FORCE_EVAL" \
    KEEP_VLLM="$KEEP_VLLM" \
    REMOVE_MERGED_AFTER_EVAL="$REMOVE_MERGED_AFTER_EVAL" \
    BUILD_TEACHER_REQUESTS="$BUILD_TEACHER_REQUESTS" \
    CKPT_DIR="$STEP200_CKPT_DIR" \
    bash "$SCRIPT_DIR/run_repair_v2a_student_reference_eval.sh"
}

main "$@"
