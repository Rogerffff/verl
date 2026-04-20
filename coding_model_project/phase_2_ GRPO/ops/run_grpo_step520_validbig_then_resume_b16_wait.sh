#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT=${ROOT:-/workspace/verl}
RUN_NAME=${RUN_NAME:-grpo_a1_formal_observe_lb96_multi6_resume460_to520_seed0}
TARGET_STEP=${TARGET_STEP:-520}
CKPT_ROOT=${CKPT_ROOT:-"$ROOT/checkpoints/rlvr_coding_model/$RUN_NAME"}
CKPT_DIR=${CKPT_DIR:-"$CKPT_ROOT/global_step_${TARGET_STEP}"}

WAIT_TIMEOUT_S=${WAIT_TIMEOUT_S:-21600}
POLL_INTERVAL_S=${POLL_INTERVAL_S:-30}

GPU_DEVICE=${GPU_DEVICE:-0}
VLLM_PORT=${VLLM_PORT:-8016}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
MAX_CONCURRENT=${MAX_CONCURRENT:-96}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-96}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-96}
BATCH_SIZE=${BATCH_SIZE:-64}
RUN_TIMEOUT=${RUN_TIMEOUT:-30}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-20000}
MAX_RESPONSE_CHARS=${MAX_RESPONSE_CHARS:-20000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}

LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_validbig500"}
TMPDIR=${TMPDIR:-/workspace/tmp_step520_validbig}
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/workspace/vllm_cache_step520_validbig}
REMOVE_MERGED_AFTER_EVAL=${REMOVE_MERGED_AFTER_EVAL:-true}

EVAL_NAME=${EVAL_NAME:-"${RUN_NAME}_step${TARGET_STEP}_codecontests_validbig500"}
MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-"$CKPT_DIR/actor_merged_hf_valid_big_step${TARGET_STEP}"}

NEXT_TRAIN_LOG=${NEXT_TRAIN_LOG:-"$LOG_DIR/grpo_a1_formal_observe_lb96_multi6_b16_onpolicy_resume520_to580_seed0.out"}

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

checkpoint_ready() {
    [[ -f "$CKPT_DIR/data.pt" ]] || return 1
    [[ -f "$CKPT_DIR/actor/fsdp_config.json" ]] || return 1
    [[ -f "$CKPT_DIR/actor/huggingface/config.json" ]] || return 1
    local rank
    for rank in 0 1 2 3; do
        [[ -f "$CKPT_DIR/actor/model_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$CKPT_DIR/actor/optim_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$CKPT_DIR/actor/extra_state_world_size_4_rank_${rank}.pt" ]] || return 1
    done
}

wait_for_checkpoint() {
    local waited=0
    while ! checkpoint_ready; do
        if (( waited >= WAIT_TIMEOUT_S )); then
            echo "Timed out waiting for checkpoint readiness: $CKPT_DIR" >&2
            return 1
        fi
        log "WAIT_CHECKPOINT ckpt=${CKPT_DIR} waited_s=${waited}"
        sleep "$POLL_INTERVAL_S"
        waited=$((waited + POLL_INTERVAL_S))
    done
    log "CHECKPOINT_READY ckpt=${CKPT_DIR}"
}

run_valid_big() {
    GPU_DEVICE="$GPU_DEVICE" \
    VLLM_PORT="$VLLM_PORT" \
    SANDBOX_URL="$SANDBOX_URL" \
    MAX_CONCURRENT="$MAX_CONCURRENT" \
    MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
    VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
    BATCH_SIZE="$BATCH_SIZE" \
    RUN_TIMEOUT="$RUN_TIMEOUT" \
    MAX_TOKENS="$MAX_TOKENS" \
    MAX_PROMPT_CHARS="$MAX_PROMPT_CHARS" \
    MAX_RESPONSE_CHARS="$MAX_RESPONSE_CHARS" \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    LOG_DIR="$LOG_DIR" \
    OUTPUT_BASE="$OUTPUT_BASE" \
    TMPDIR="$TMPDIR" \
    VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
    REMOVE_MERGED_AFTER_EVAL="$REMOVE_MERGED_AFTER_EVAL" \
    CKPT_DIR="$CKPT_DIR" \
    MERGED_MODEL_DIR="$MERGED_MODEL_DIR" \
    EVAL_NAME="$EVAL_NAME" \
    bash "$SCRIPT_DIR/run_grpo_codecontests_validbig_checkpoint.sh"
}

launch_next_train() {
    log "START_NEXT_TRAIN script=run_grpo_a1_resume520_to580_b16.sh log=${NEXT_TRAIN_LOG}"
    nohup bash "$SCRIPT_DIR/run_grpo_a1_resume520_to580_b16.sh" >"$NEXT_TRAIN_LOG" 2>&1 &
    log "DONE_NEXT_TRAIN pid=$!"
}

main() {
    mkdir -p "$LOG_DIR" "$OUTPUT_BASE" "$TMPDIR" "$VLLM_CACHE_ROOT"
    wait_for_checkpoint
    run_valid_big
    launch_next_train
}

main "$@"
