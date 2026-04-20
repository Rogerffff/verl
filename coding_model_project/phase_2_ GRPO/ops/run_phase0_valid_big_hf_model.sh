#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-24}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-24}
BATCH_SIZE=${BATCH_SIZE:-50}
VLLM_PORT=${VLLM_PORT:-8001}
GPU_DEVICE=${GPU_DEVICE:-0}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-20000}
MAX_RESPONSE_CHARS=${MAX_RESPONSE_CHARS:-20000}

LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/coding_model_project/outputs/phase0_fullval_baseline_valid_big_lb24_multi2"}
RUN_LOG=${RUN_LOG:-"$LOG_DIR/baseline_valid_big_lb24_multi2.out"}
VLLM_LOG=${VLLM_LOG:-"$LOG_DIR/baseline_valid_big_lb24_multi2_vllm.log"}
EVAL_LOG=${EVAL_LOG:-"$LOG_DIR/baseline_valid_big_lb24_multi2_eval.log"}

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

start_vllm() {
    cleanup_vllm

    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        log "ERROR port ${VLLM_PORT} already occupied"
        return 1
    fi

    log "START_VLLM model=${MODEL_PATH}"
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 6144 \
        --dtype bfloat16 \
        --trust-remote-code \
        >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!

    local ok=false
    for _ in $(seq 1 180); do
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
    : >"$RUN_LOG"
    log "START baseline valid_big eval model=${MODEL_PATH} sandbox=${SANDBOX_URL} limiter=${VERIFIER_LIMITER_BUDGET}"

    if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
        log "SKIP existing summary at $OUTPUT_DIR/summary.json"
        return 0
    fi

    start_vllm

    log "START_EVAL output_dir=${OUTPUT_DIR}"
    python3 "$ROOT/coding_model_project/src/phase0_eval.py" \
        --mode simple \
        --model "$MODEL_PATH" \
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
    log "DONE baseline valid_big eval"
}

main "$@"
