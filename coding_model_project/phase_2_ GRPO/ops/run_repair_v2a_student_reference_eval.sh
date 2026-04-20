#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
PROJECT_ROOT="${ROOT}/coding_model_project"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${PROJECT_ROOT}/phase_2_ GRPO/sft_repair_data/scripts"
REPAIR_DATA_DIR=${REPAIR_DATA_DIR:-"${PROJECT_ROOT}/phase_2_ GRPO/sft_repair_data/v2a"}
RUN_TAG=${RUN_TAG:-"$(basename "$REPAIR_DATA_DIR")"}
DEFAULT_SPEC_PATH="${REPAIR_DATA_DIR}/repair_${RUN_TAG}_spec.json"
if [[ -z "${SPEC_PATH:-}" ]]; then
    if [[ -f "$DEFAULT_SPEC_PATH" ]]; then
        SPEC_PATH="$DEFAULT_SPEC_PATH"
    else
        SPEC_PATH="${REPAIR_DATA_DIR}/repair_v2a_spec.json"
    fi
fi
SUBSET_ROOT=${SUBSET_ROOT:-"${REPAIR_DATA_DIR}/student_reference_eval_subset_${RUN_TAG}"}
SUBSET_MANIFEST_DIR=${SUBSET_MANIFEST_DIR:-"${SUBSET_ROOT}/manifests"}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"${PROJECT_ROOT}/outputs/repair_${RUN_TAG}_student_reference_eval"}
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
PYTHON_BIN=${PYTHON_BIN:-python}
PREPARE_STATIC_ASSETS=${PREPARE_STATIC_ASSETS:-true}
PREPARE_ONLY=${PREPARE_ONLY:-false}
FORCE_EVAL=${FORCE_EVAL:-false}
KEEP_VLLM=${KEEP_VLLM:-false}
REMOVE_MERGED_AFTER_EVAL=${REMOVE_MERGED_AFTER_EVAL:-false}
BUILD_TEACHER_REQUESTS=${BUILD_TEACHER_REQUESTS:-false}
MODEL_PATH=${MODEL_PATH:-}
CKPT_DIR=${CKPT_DIR:-}
EVAL_NAME=${EVAL_NAME:-}
VLLM_PID=""
MERGED_MODEL_DIR=""

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

cleanup_vllm() {
    if [[ "$KEEP_VLLM" == "true" ]]; then
        return 0
    fi
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
        MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-"${CKPT_DIR}/actor_merged_hf_${RUN_TAG}"}"
        if ! merged_has_weights "$MERGED_MODEL_DIR"; then
            log "START_MERGE ckpt=${CKPT_DIR}" >&2
            cd "$ROOT"
            "$PYTHON_BIN" -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "$CKPT_DIR/actor" \
                --target_dir "$MERGED_MODEL_DIR" \
                >"$LOG_DIR/${EVAL_NAME:-$(basename "$CKPT_DIR")}_${RUN_TAG}_merge.log" 2>&1
            log "DONE_MERGE ckpt=${CKPT_DIR}" >&2
        fi
        printf '%s\n' "$MERGED_MODEL_DIR"
        return 0
    fi

    echo "Need MODEL_PATH or CKPT_DIR with actor/ or huggingface/ subdir" >&2
    return 1
}

prepare_v2a_assets() {
    if [[ ! -f "$SPEC_PATH" ]]; then
        echo "Missing repair spec: $SPEC_PATH" >&2
        return 1
    fi

    if [[ "$PREPARE_STATIC_ASSETS" != "true" ]]; then
        [[ -f "${SUBSET_MANIFEST_DIR}/codecontests_train_manifest.jsonl" ]] || {
            echo "Missing subset manifest and PREPARE_STATIC_ASSETS=false: ${SUBSET_MANIFEST_DIR}/codecontests_train_manifest.jsonl" >&2
            return 1
        }
        return 0
    fi

    log "PREPARE_REPAIR_ASSETS output_dir=${REPAIR_DATA_DIR}"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/build_seed_manifest.py" \
        --spec "$SPEC_PATH" \
        --output_dir "$REPAIR_DATA_DIR"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/expand_phase2_retrieval_v2.py" \
        --spec "$SPEC_PATH" \
        --output_dir "$REPAIR_DATA_DIR"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/build_student_reference_shortlist.py" \
        --spec "$SPEC_PATH" \
        --output_dir "$REPAIR_DATA_DIR"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/materialize_student_reference_eval_subset.py" \
        --spec "$SPEC_PATH" \
        --output_dir "$REPAIR_DATA_DIR"

    [[ -f "${SUBSET_MANIFEST_DIR}/codecontests_train_manifest.jsonl" ]] || {
        echo "Failed to materialize student-reference eval subset manifest." >&2
        return 1
    }
    [[ -f "${SUBSET_ROOT}/raw/codecontests_train_raw.jsonl" ]] || {
        echo "Failed to materialize student-reference eval subset raw file." >&2
        return 1
    }
    log "DONE_PREPARE_REPAIR_ASSETS subset_root=${SUBSET_ROOT}"
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
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
        --model "$resolved_model_path" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 6144 \
        --dtype bfloat16 \
        --trust-remote-code \
        >"$LOG_DIR/${EVAL_NAME}_${RUN_TAG}_vllm.log" 2>&1 &
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
    local output_dir="$1"
    local run_log="$LOG_DIR/${EVAL_NAME}_${RUN_TAG}_student_ref_eval.log"

    if [[ "$FORCE_EVAL" != "true" ]] && [[ -f "$output_dir/summary.json" ]]; then
        log "SKIP_EVAL existing summary=${output_dir}/summary.json"
        return 0
    fi

    mkdir -p "$output_dir"
    log "START_EVAL output=${output_dir}"
    "$PYTHON_BIN" "$ROOT/coding_model_project/src/phase0_eval.py" \
        --mode simple \
        --model "$RESOLVED_MODEL_PATH" \
        --vllm_url "http://localhost:${VLLM_PORT}" \
        --sandbox_url "$SANDBOX_URL" \
        --manifest_dir "$SUBSET_MANIFEST_DIR" \
        --datasets codecontests_train \
        --temperature 0.0 \
        --max_tokens "$MAX_TOKENS" \
        --run_timeout "$RUN_TIMEOUT" \
        --use_external_tests \
        --max_concurrent "$MAX_CONCURRENT" \
        --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
        --verifier_limiter_budget "$VERIFIER_LIMITER_BUDGET" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$output_dir" \
        --save_full_results \
        --max_prompt_chars "$MAX_PROMPT_CHARS" \
        --max_response_chars "$MAX_RESPONSE_CHARS" \
        >"$run_log" 2>&1
    log "DONE_EVAL output=${output_dir}"
}

build_student_references() {
    local eval_output_dir="$1"
    local per_problem_jsonl="${eval_output_dir}/per_problem/codecontests_train.jsonl"
    local run_info_json="${eval_output_dir}/run_info.json"

    [[ -f "$per_problem_jsonl" ]] || {
        echo "Missing eval results JSONL: $per_problem_jsonl" >&2
        return 1
    }
    [[ -f "$run_info_json" ]] || {
        echo "Missing eval run_info.json: $run_info_json" >&2
        return 1
    }

    log "START_BUILD_STUDENT_REFS eval_dir=${eval_output_dir}"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/build_student_references.py" \
        --spec "$SPEC_PATH" \
        --eval_results "$per_problem_jsonl" \
        --eval_run_info "$run_info_json" \
        --output_dir "$REPAIR_DATA_DIR"
    log "DONE_BUILD_STUDENT_REFS output_dir=${REPAIR_DATA_DIR}"
}

maybe_build_teacher_requests() {
    if [[ "$BUILD_TEACHER_REQUESTS" != "true" ]]; then
        return 0
    fi
    log "START_BUILD_TEACHER_REQUESTS"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/build_teacher_generation_requests.py" \
        --spec "$SPEC_PATH" \
        --output_dir "$REPAIR_DATA_DIR" \
        --student_references "${REPAIR_DATA_DIR}/student_references_$(python - <<'PY'
import json, os, re
spec = json.load(open(os.environ["SPEC_PATH"]))
anchor = str(spec.get("student_anchor_checkpoint", "")).lower()
match = re.search(r"step[_-]?(\d+)", anchor)
print(f"step{match.group(1)}" if match else "student_anchor")
PY
).jsonl"
    log "DONE_BUILD_TEACHER_REQUESTS output_dir=${REPAIR_DATA_DIR}"
}

main() {
    prepare_v2a_assets
    if [[ "$PREPARE_ONLY" == "true" ]]; then
        log "DONE_PREPARE_ONLY subset_root=${SUBSET_ROOT}"
        return 0
    fi

    RESOLVED_MODEL_PATH="$(resolve_model_path)"
    if [[ -z "$EVAL_NAME" ]]; then
        if [[ -n "$CKPT_DIR" ]]; then
            EVAL_NAME="$(basename "$CKPT_DIR")"
        else
            EVAL_NAME="$(basename "$RESOLVED_MODEL_PATH")"
        fi
    fi

    EVAL_OUTPUT_DIR="${OUTPUT_BASE}/${EVAL_NAME}"

    start_vllm "$RESOLVED_MODEL_PATH"
    run_eval "$EVAL_OUTPUT_DIR"
    build_student_references "$EVAL_OUTPUT_DIR"
    maybe_build_teacher_requests
    cleanup_vllm
    log "DONE repair student-reference eval eval_name=${EVAL_NAME} run_tag=${RUN_TAG}"
}

main "$@"
