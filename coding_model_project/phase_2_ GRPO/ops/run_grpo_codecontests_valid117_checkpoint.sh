#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
RUN_TIMEOUT=${RUN_TIMEOUT:-30}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-20000}
MAX_RESPONSE_CHARS=${MAX_RESPONSE_CHARS:-20000}
GPU_DEVICE=${GPU_DEVICE:-0}
VLLM_PORT=${VLLM_PORT:-8013}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
MAX_CONCURRENT=${MAX_CONCURRENT:-24}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-16}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-16}
BATCH_SIZE=${BATCH_SIZE:-16}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_valid117"}
TMPDIR=${TMPDIR:-/tmp/t117}
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/dev/vc117}

MODEL_PATH=${MODEL_PATH:-}
CKPT_DIR=${CKPT_DIR:-}
EVAL_NAME=${EVAL_NAME:-}
VLLM_PID=""

mkdir -p "$LOG_DIR" "$OUTPUT_BASE" "$TMPDIR" "$VLLM_CACHE_ROOT"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

cleanup_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true
        wait "$VLLM_PID" || true
    fi
    VLLM_PID=""
}

trap 'cleanup_vllm' EXIT

resolve_model_path() {
    if [[ -n "$MODEL_PATH" ]]; then
        echo "$MODEL_PATH"
        return 0
    fi

    if [[ -z "$CKPT_DIR" ]]; then
        echo "Need MODEL_PATH or CKPT_DIR" >&2
        return 1
    fi

    if [[ -d "$CKPT_DIR/actor_merged_hf_rl_canary" ]]; then
        echo "$CKPT_DIR/actor_merged_hf_rl_canary"
        return 0
    fi

    if [[ -d "$CKPT_DIR/huggingface" ]]; then
        echo "$CKPT_DIR/huggingface"
        return 0
    fi

    echo "Could not resolve model path from $CKPT_DIR" >&2
    return 1
}

start_vllm() {
    local resolved_model_path="$1"
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "Port ${VLLM_PORT} already occupied" >&2
        return 1
    fi

    log "START_VLLM model=${resolved_model_path} port=${VLLM_PORT}"
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
    VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
    TMPDIR="$TMPDIR" \
    TMP="$TMPDIR" \
    TEMP="$TMPDIR" \
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$resolved_model_path" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 6144 \
        --dtype bfloat16 \
        --trust-remote-code \
        >"$LOG_DIR/${EVAL_NAME}_vllm.log" 2>&1 &
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
        echo "vLLM failed to become healthy on port ${VLLM_PORT}" >&2
        return 1
    fi
    log "DONE_VLLM_HEALTH pid=${VLLM_PID}"
}

run_eval() {
    local resolved_model_path="$1"
    local output_dir="$OUTPUT_BASE/$EVAL_NAME"
    mkdir -p "$output_dir"
    log "START_EVAL dataset=codecontests_valid output=${output_dir}"
    python3 "$ROOT/coding_model_project/src/phase0_eval.py" \
        --mode simple \
        --model "$resolved_model_path" \
        --vllm_url "http://localhost:${VLLM_PORT}" \
        --sandbox_url "$SANDBOX_URL" \
        --manifest_dir "$ROOT/coding_model_project/data/manifests" \
        --datasets codecontests_valid \
        --temperature 0.0 \
        --max_tokens "$MAX_TOKENS" \
        --run_timeout "$RUN_TIMEOUT" \
        --max_concurrent "$MAX_CONCURRENT" \
        --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
        --verifier_limiter_budget "$VERIFIER_LIMITER_BUDGET" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$output_dir" \
        --save_full_results \
        --max_prompt_chars "$MAX_PROMPT_CHARS" \
        --max_response_chars "$MAX_RESPONSE_CHARS"
    log "DONE_EVAL dataset=codecontests_valid output=${output_dir}"
}

main() {
    if [[ -z "$EVAL_NAME" ]]; then
        if [[ -n "$CKPT_DIR" ]]; then
            EVAL_NAME="$(basename "$CKPT_DIR")_codecontests_valid117"
        else
            EVAL_NAME="codecontests_valid117_eval"
        fi
    fi

    RESOLVED_MODEL_PATH="$(resolve_model_path)"
    start_vllm "$RESOLVED_MODEL_PATH"
    run_eval "$RESOLVED_MODEL_PATH"
}

main "$@"
