#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_JSON=${SUITE_JSON:-"$ROOT/coding_model_project/phase_2_ GRPO/sft_repair_data/final_v1/phase2_sft_eval_suite_v1.json"}
SUITE_ROOT=${SUITE_ROOT:-"$ROOT/coding_model_project/phase_2_ GRPO/sft_repair_data/final_v1/eval_suite_materialized_v1"}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-24}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-24}
BATCH_SIZE=${BATCH_SIZE:-24}
RUN_TIMEOUT=${RUN_TIMEOUT:-30}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-20000}
MAX_RESPONSE_CHARS=${MAX_RESPONSE_CHARS:-20000}
GPU_DEVICE=${GPU_DEVICE:-0}
VLLM_PORT=${VLLM_PORT:-8001}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/phase2_sft_eval_suite_v1"}
SLICES=${SLICES:-repair_val retention_canary_codecontests timeout_canary_codecontests structural_hard_watchlist humaneval_mini mbpp_reg_mini}

MODEL_PATH=${MODEL_PATH:-}
CKPT_DIR=${CKPT_DIR:-}
EVAL_NAME=${EVAL_NAME:-}
VLLM_PID=""

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

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

trap cleanup_vllm EXIT

resolve_model_path() {
    if [[ -n "$MODEL_PATH" ]]; then
        echo "$MODEL_PATH"
        return 0
    fi
    if [[ -n "$CKPT_DIR" ]] && [[ -d "$CKPT_DIR/huggingface" ]]; then
        echo "$CKPT_DIR/huggingface"
        return 0
    fi
    echo "Need MODEL_PATH or CKPT_DIR with huggingface/ subdir" >&2
    return 1
}

materialize_suite_if_needed() {
    if [[ -f "$SUITE_ROOT/suite_materialization_summary.json" ]]; then
        return 0
    fi
    python3 "$ROOT/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/materialize_phase2_eval_suite.py" \
        --suite_json "$SUITE_JSON" \
        --raw_root "$ROOT/coding_model_project/data/raw" \
        --output_root "$SUITE_ROOT"
}

dataset_key_for_slice() {
    case "$1" in
        repair_val) echo "codecontests_train" ;;
        retention_canary_codecontests) echo "codecontests_valid" ;;
        timeout_canary_codecontests) echo "codecontests_valid" ;;
        structural_hard_watchlist) echo "codecontests_valid" ;;
        humaneval_mini) echo "humaneval" ;;
        mbpp_reg_mini) echo "mbpp_reg" ;;
        *) echo "Unknown slice: $1" >&2; return 1 ;;
    esac
}

start_vllm() {
    local resolved_model_path="$1"
    cleanup_vllm
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "Port ${VLLM_PORT} already occupied" >&2
        return 1
    fi
    log "START_VLLM model=${resolved_model_path} port=${VLLM_PORT}"
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" nohup python3 -m vllm.entrypoints.openai.api_server \
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

run_slice() {
    local slice="$1"
    local dataset_key="$2"
    local manifest_dir="$SUITE_ROOT/$slice/manifests"
    local output_dir="$OUTPUT_BASE/$EVAL_NAME/$slice"
    local run_log="$LOG_DIR/${EVAL_NAME}_${slice}.log"
    if [[ -f "$output_dir/summary.json" ]]; then
        log "SKIP slice=${slice} existing summary=${output_dir}/summary.json"
        return 0
    fi

    mkdir -p "$output_dir"
    log "START_EVAL slice=${slice} dataset=${dataset_key} output=${output_dir}"
    python3 "$ROOT/coding_model_project/src/phase0_eval.py" \
        --mode simple \
        --model "$RESOLVED_MODEL_PATH" \
        --vllm_url "http://localhost:${VLLM_PORT}" \
        --sandbox_url "$SANDBOX_URL" \
        --manifest_dir "$manifest_dir" \
        --datasets "$dataset_key" \
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
        --max_response_chars "$MAX_RESPONSE_CHARS" \
        >"$run_log" 2>&1
    log "DONE_EVAL slice=${slice}"
}

main() {
    RESOLVED_MODEL_PATH="$(resolve_model_path)"
    if [[ -z "$EVAL_NAME" ]]; then
        if [[ -n "$CKPT_DIR" ]]; then
            EVAL_NAME="$(basename "$CKPT_DIR")"
        else
            EVAL_NAME="$(basename "$RESOLVED_MODEL_PATH")"
        fi
    fi

    materialize_suite_if_needed
    start_vllm "$RESOLVED_MODEL_PATH"

    for slice in $SLICES; do
        dataset_key="$(dataset_key_for_slice "$slice")"
        run_slice "$slice" "$dataset_key"
    done

    cleanup_vllm
    log "DONE eval_name=${EVAL_NAME}"
}

main "$@"
