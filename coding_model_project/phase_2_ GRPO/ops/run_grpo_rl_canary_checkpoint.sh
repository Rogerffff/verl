#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_JSON=${SUITE_JSON:-"$ROOT/coding_model_project/phase_2_ GRPO/rl_canary_suite_v1.json"}
SUITE_ROOT=${SUITE_ROOT:-"$ROOT/coding_model_project/phase_2_ GRPO/rl_canary_materialized_v1"}
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
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_canary_v1"}
SLICES=${SLICES:-retention_canary_codecontests timeout_canary_codecontests structural_hard_watchlist mid_val_codecontests_32}
REMOVE_MERGED_AFTER_EVAL=${REMOVE_MERGED_AFTER_EVAL:-false}
MATERIALIZE_LOCK_WAIT_S=${MATERIALIZE_LOCK_WAIT_S:-600}

MODEL_PATH=${MODEL_PATH:-}
CKPT_DIR=${CKPT_DIR:-}
EVAL_NAME=${EVAL_NAME:-}
VLLM_PID=""
MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-}

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

default_exec_cache_root() {
    local shm_opts=""
    shm_opts=$(findmnt -T /dev/shm -no OPTIONS 2>/dev/null || true)
    if [[ "$shm_opts" == *noexec* ]]; then
        echo "/dev/vllm_cache_${EVAL_NAME}"
    else
        echo "/dev/shm/vllm_cache_${EVAL_NAME}"
    fi
}

cleanup_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true
        wait "$VLLM_PID" || true
    fi
    VLLM_PID=""
}

cleanup_merged() {
    if [[ "$REMOVE_MERGED_AFTER_EVAL" == "true" ]] && [[ -n "$MERGED_MODEL_DIR" ]] && [[ -d "$MERGED_MODEL_DIR" ]]; then
        rm -rf "$MERGED_MODEL_DIR"
    fi
}

trap 'cleanup_vllm; cleanup_merged' EXIT

merged_has_weights() {
    local merged_dir="$1"
    [[ -d "$merged_dir" ]] || return 1
    find "$merged_dir" -maxdepth 1 -type f \
        \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'model*.safetensors' \) \
        -print -quit | grep -q .
}

resolve_model_path() {
    if [[ -n "$MODEL_PATH" ]]; then
        echo "$MODEL_PATH"
        return 0
    fi

    if [[ -z "$CKPT_DIR" ]]; then
        echo "Need MODEL_PATH or CKPT_DIR" >&2
        return 1
    fi

    if [[ -d "$CKPT_DIR/huggingface" ]]; then
        echo "$CKPT_DIR/huggingface"
        return 0
    fi

    if [[ -d "$CKPT_DIR/actor" ]]; then
        MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-"${CKPT_DIR}/actor_merged_hf_rl_canary"}"
        if ! merged_has_weights "$MERGED_MODEL_DIR"; then
            log "START_MERGE ckpt=${CKPT_DIR}" >&2
            cd "$ROOT"
            python3 -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "$CKPT_DIR/actor" \
                --target_dir "$MERGED_MODEL_DIR" \
                >"$LOG_DIR/${EVAL_NAME:-$(basename "$CKPT_DIR")}_merge.log" 2>&1
            log "DONE_MERGE ckpt=${CKPT_DIR}" >&2
        fi
        echo "$MERGED_MODEL_DIR"
        return 0
    fi

    echo "Need MODEL_PATH or CKPT_DIR with actor/ or huggingface/ subdir" >&2
    return 1
}

materialize_suite_if_needed() {
    local summary_path="$SUITE_ROOT/suite_materialization_summary.json"
    local lock_dir="$SUITE_ROOT/.materialize.lock"
    local waited=0

    if [[ -f "$summary_path" ]]; then
        return 0
    fi

    mkdir -p "$SUITE_ROOT"
    while true; do
        if [[ -f "$summary_path" ]]; then
            return 0
        fi
        if mkdir "$lock_dir" 2>/dev/null; then
            break
        fi
        sleep 2
        waited=$((waited + 2))
        if (( waited >= MATERIALIZE_LOCK_WAIT_S )); then
            echo "Timed out waiting for suite materialization lock at $lock_dir" >&2
            return 1
        fi
    done

    local status=0
    local -a slice_args=()
    read -r -a slice_args <<< "$SLICES"
    python3 "$ROOT/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/materialize_phase2_eval_suite.py" \
        --suite_json "$SUITE_JSON" \
        --raw_root "$ROOT/coding_model_project/data/raw" \
        --output_root "$SUITE_ROOT" \
        --slices "${slice_args[@]}" || status=$?
    rmdir "$lock_dir" 2>/dev/null || true
    return "$status"
}

dataset_key_for_slice() {
    case "$1" in
        retention_canary_codecontests|timeout_canary_codecontests|structural_hard_watchlist|mid_val_codecontests_32)
            echo "codecontests_valid"
            ;;
        *)
            echo "Unknown slice: $1" >&2
            return 1
            ;;
    esac
}

start_vllm() {
    local resolved_model_path="$1"
    local cache_root="${VLLM_CACHE_ROOT:-$(default_exec_cache_root)}"
    local torchinductor_cache="${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/torchinductor}"
    local triton_cache="${TRITON_CACHE_DIR:-${cache_root}/triton}"
    local temp_root="${TMPDIR:-/dev/shm/tmp_${EVAL_NAME}}"
    cleanup_vllm
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "Port ${VLLM_PORT} already occupied" >&2
        return 1
    fi
    mkdir -p "$cache_root" "$torchinductor_cache" "$triton_cache" "$temp_root"
    log "START_VLLM model=${resolved_model_path} port=${VLLM_PORT}"
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
    VLLM_CACHE_ROOT="$cache_root" \
    TORCHINDUCTOR_CACHE_DIR="$torchinductor_cache" \
    TRITON_CACHE_DIR="$triton_cache" \
    TMPDIR="$temp_root" \
    TMP="$temp_root" \
    TEMP="$temp_root" \
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
