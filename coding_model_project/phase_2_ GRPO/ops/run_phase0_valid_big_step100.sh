#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
CKPT_ROOT=${CKPT_ROOT:-"$ROOT/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0/global_step_100"}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-24}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-24}
BATCH_SIZE=${BATCH_SIZE:-50}
VLLM_PORT=${VLLM_PORT:-8001}
GPU_DEVICE=${GPU_DEVICE:-0}
REMOVE_MERGED_AFTER_EVAL=${REMOVE_MERGED_AFTER_EVAL:-true}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-20000}
MAX_RESPONSE_CHARS=${MAX_RESPONSE_CHARS:-20000}

LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/coding_model_project/outputs/phase0_fullval_step100_valid_big_lb24_multi2"}
RUN_LOG=${RUN_LOG:-"$LOG_DIR/step100_valid_big_lb24_multi2.out"}
MERGE_LOG=${MERGE_LOG:-"$LOG_DIR/step100_valid_big_lb24_multi2_merge.log"}
VLLM_LOG=${VLLM_LOG:-"$LOG_DIR/step100_valid_big_lb24_multi2_vllm.log"}
EVAL_LOG=${EVAL_LOG:-"$LOG_DIR/step100_valid_big_lb24_multi2_eval.log"}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

VLLM_PID=""

log() {
    echo "[$(date --iso-8601=seconds)] $*" | tee -a "$RUN_LOG"
}

cleanup_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true
        wait "$VLLM_PID" || true
    fi
    VLLM_PID=""
}

trap cleanup_vllm EXIT

checkpoint_ready() {
    [[ -f "$CKPT_ROOT/data.pt" ]] || return 1
    [[ -f "$CKPT_ROOT/actor/fsdp_config.json" ]] || return 1
    [[ -f "$CKPT_ROOT/actor/huggingface/config.json" ]] || return 1
    local rank
    for rank in 0 1 2 3; do
        [[ -f "$CKPT_ROOT/actor/model_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$CKPT_ROOT/actor/optim_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$CKPT_ROOT/actor/extra_state_world_size_4_rank_${rank}.pt" ]] || return 1
    done
}

merged_has_weights() {
    local merged_dir="$1"
    [[ -d "$merged_dir" ]] || return 1
    find "$merged_dir" -maxdepth 1 -type f \
        \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'model*.safetensors' \) \
        -print -quit | grep -q .
}

start_vllm() {
    local model_dir="$1"

    cleanup_vllm

    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        log "ERROR port ${VLLM_PORT} already occupied"
        return 1
    fi

    log "START_VLLM model_dir=${model_dir}"
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$model_dir" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 6144 \
        --dtype bfloat16 \
        --trust-remote-code \
        >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!

    local ok=false
    for _ in $(seq 1 120); do
        if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            ok=true
            break
        fi
        sleep 5
    done

    if [[ "$ok" != "true" ]]; then
        log "ERROR vLLM failed to become healthy on port ${VLLM_PORT}"
        return 1
    fi

    log "DONE_VLLM_HEALTH pid=${VLLM_PID}"
}

main() {
    local actor_dir="$CKPT_ROOT/actor"
    local merged_dir="$CKPT_ROOT/actor_merged_hf"

    : >"$RUN_LOG"
    log "START step100 valid_big eval sandbox=${SANDBOX_URL} limiter=${VERIFIER_LIMITER_BUDGET}"

    if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
        log "SKIP existing summary at $OUTPUT_DIR/summary.json"
        return 0
    fi

    if ! checkpoint_ready; then
        log "ERROR checkpoint not complete: $CKPT_ROOT"
        return 1
    fi

    if ! merged_has_weights "$merged_dir"; then
        log "START_MERGE ckpt_root=${CKPT_ROOT}"
        cd "$ROOT"
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$merged_dir" \
            >"$MERGE_LOG" 2>&1
        log "DONE_MERGE"
    else
        log "SKIP_MERGE existing merged weights"
    fi

    start_vllm "$merged_dir"

    log "START_EVAL output_dir=${OUTPUT_DIR}"
    python3 "$ROOT/coding_model_project/src/phase0_eval.py" \
        --mode simple \
        --model "$merged_dir" \
        --vllm_url "http://localhost:${VLLM_PORT}" \
        --sandbox_url "$SANDBOX_URL" \
        --manifest_dir "$ROOT/coding_model_project/data/manifests" \
        --datasets codecontests_valid_big \
        --temperature 0.0 \
        --max_tokens 2048 \
        --run_timeout 30 \
        --max_concurrent "$MAX_CONCURRENT" \
        --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
        --verifier_limiter_budget "$VERIFIER_LIMITER_BUDGET" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        --save_full_results \
        --max_prompt_chars "$MAX_PROMPT_CHARS" \
        --max_response_chars "$MAX_RESPONSE_CHARS" \
        >"$EVAL_LOG" 2>&1
    log "DONE_EVAL"

    cleanup_vllm

    if [[ "$REMOVE_MERGED_AFTER_EVAL" == "true" ]]; then
        rm -rf "$merged_dir"
        log "REMOVED_MERGED"
    fi

    log "DONE step100 valid_big eval"
}

main "$@"
