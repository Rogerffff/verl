#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
PROJECT_ROOT="${ROOT}/coding_model_project"
PHASE2_DIR="${PROJECT_ROOT}/phase_2_ GRPO"
SCRIPTS_DIR="${PHASE2_DIR}/sft_repair_data/scripts"
ARTIFACT_TAG=${ARTIFACT_TAG:-step900_candidate_v2_backend_rr}
DATA_ROOT=${DATA_ROOT:-"${PHASE2_DIR}/sft_repair_data/provisional_repair_conditioned/${ARTIFACT_TAG}"}
SUBSET_ROOT=${SUBSET_ROOT:-"${DATA_ROOT}/student_reference_eval_subset"}
SUBSET_MANIFEST_DIR=${SUBSET_MANIFEST_DIR:-"${SUBSET_ROOT}/manifests"}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"${PROJECT_ROOT}/outputs/repair_conditioned_student_reference_eval"}

CANDIDATE_POOL_JSONL="${DATA_ROOT}/candidate_pool_${ARTIFACT_TAG}.jsonl"
CANDIDATE_POOL_SUMMARY_JSON="${DATA_ROOT}/candidate_pool_${ARTIFACT_TAG}.summary.json"
INSTABILITY_BLACKLIST_JSON=${INSTABILITY_BLACKLIST_JSON:-"${PHASE2_DIR}/sft_repair_data/provisional_repair_conditioned/step900_candidate_v1/instability_blacklist_v1.json"}
RETAINED_C_DECISIONS_JSONL="${DATA_ROOT}/retained_c_decisions_${ARTIFACT_TAG}.jsonl"
RETAINED_C_SUMMARY_JSON="${DATA_ROOT}/retained_c_decisions_${ARTIFACT_TAG}.summary.json"
STUDENT_REFS_JSONL="${DATA_ROOT}/student_references_${ARTIFACT_TAG}.jsonl"
STUDENT_REFS_SUMMARY_JSON="${DATA_ROOT}/student_references_${ARTIFACT_TAG}.summary.json"
TEACHER_REQUESTS_JSONL="${DATA_ROOT}/teacher_generation_requests_${ARTIFACT_TAG}.jsonl"
TEACHER_REQUESTS_SUMMARY_JSON="${DATA_ROOT}/teacher_generation_requests_${ARTIFACT_TAG}.summary.json"

CURRICULUM_STATE=${CURRICULUM_STATE:-"${PROJECT_ROOT}/curriculum_states/grpo_a1_curriculum_step620_v3q3_to900_seed0/curriculum_state_step_900.json"}
TRAIN_RAW=${TRAIN_RAW:-"${PROJECT_ROOT}/data/raw/codecontests_train_wo_valid_big_raw.jsonl"}
QUARANTINE_PATH=${QUARANTINE_PATH:-"${PROJECT_ROOT}/data/problem_quarantine_v3.json"}
VALID_BIG_MANIFEST=${VALID_BIG_MANIFEST:-"${PROJECT_ROOT}/data/manifests/codecontests_valid_big_manifest.jsonl"}
VALID_MANIFEST=${VALID_MANIFEST:-"${PROJECT_ROOT}/data/manifests/codecontests_valid_manifest.jsonl"}
TEST_MANIFEST=${TEST_MANIFEST:-"${PROJECT_ROOT}/data/manifests/codecontests_test_manifest.jsonl"}
DELTA69_MANIFEST=${DELTA69_MANIFEST:-"${PHASE2_DIR}/review_assets/step640_v1/delta69_eval_slice_raw/manifests/codecontests_valid_big_manifest.jsonl"}
CANARY_SUITE=${CANARY_SUITE:-"${PHASE2_DIR}/rl_canary_suite_v1.json"}
KEEP_C_DECISIONS=${KEEP_C_DECISIONS:-"${PHASE2_DIR}/curriculum_assets/step600_v3_review_local/c_hard_partial_review_decisions_step600_v3.jsonl"}
KEEP_C_DECISIONS_SECONDARY=${KEEP_C_DECISIONS_SECONDARY:-"${PHASE2_DIR}/curriculum_assets/step600_v2_review_local/c_hard_partial_review_decisions_step600_v2_FILLED.jsonl"}
LATER_C_DECISIONS=${LATER_C_DECISIONS:-"${PHASE2_DIR}/review_assets/step640_v1/c_bucket_actions_decisions_step640_v1.jsonl"}

CKPT_DIR=${CKPT_DIR:-/workspace/verl/checkpoints/global_step_900}
MODEL_PATH=${MODEL_PATH:-}
MERGED_MODEL_DIR=""
EVAL_NAME=${EVAL_NAME:-"${ARTIFACT_TAG}_student_ref"}

SANDBOX_URLS=${SANDBOX_URLS:-"http://localhost:8081,http://localhost:8082,http://localhost:8083,http://localhost:8084,http://localhost:8085,http://localhost:8086,http://localhost:8087,http://localhost:8088"}
SETUP_SANDBOX_IF_NEEDED=${SETUP_SANDBOX_IF_NEEDED:-true}
SETUP_SANDBOX_SCRIPT=${SETUP_SANDBOX_SCRIPT:-"${PHASE2_DIR}/ops/setup_eval_sandbox_4x2.sh"}
SANDBOX_PROBE_SCRIPT=${SANDBOX_PROBE_SCRIPT:-"${PHASE2_DIR}/ops/lb_validate_probe.py"}

VLLM_PORT=${VLLM_PORT:-8011}
GPU_DEVICE=${GPU_DEVICE:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
MAX_CONCURRENT=${MAX_CONCURRENT:-160}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-160}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-160}
BATCH_SIZE=${BATCH_SIZE:-160}
RUN_TIMEOUT=${RUN_TIMEOUT:-30}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-20000}
MAX_RESPONSE_CHARS=${MAX_RESPONSE_CHARS:-50000}
EXPANDED_C_MIN_EMA_PASS_RATIO=${EXPANDED_C_MIN_EMA_PASS_RATIO:-0.2}
EXPANDED_C_MAX_EMA_TIMEOUT_RATE=${EXPANDED_C_MAX_EMA_TIMEOUT_RATE:-0.25}
EXPANDED_C_MAX_NEW_KEPT=${EXPANDED_C_MAX_NEW_KEPT:-40}
FORCE_EVAL=${FORCE_EVAL:-false}
KEEP_VLLM=${KEEP_VLLM:-false}
VLLM_PID=""

mkdir -p "$DATA_ROOT" "$SUBSET_MANIFEST_DIR" "${SUBSET_ROOT}/raw" "$LOG_DIR" "$OUTPUT_BASE"

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

trap 'cleanup_vllm' EXIT

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
    if [[ -d "$CKPT_DIR/huggingface" ]]; then
        echo "$CKPT_DIR/huggingface"
        return 0
    fi
    if [[ -d "$CKPT_DIR/actor" ]]; then
        MERGED_MODEL_DIR="${CKPT_DIR}/actor_merged_hf_${ARTIFACT_TAG}"
        if ! merged_has_weights "$MERGED_MODEL_DIR"; then
            log "START_MERGE ckpt=${CKPT_DIR}" >&2
            cd "$ROOT"
            "$PYTHON_BIN" -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "$CKPT_DIR/actor" \
                --target_dir "$MERGED_MODEL_DIR" \
                >"$LOG_DIR/${EVAL_NAME}_merge.log" 2>&1
            log "DONE_MERGE ckpt=${CKPT_DIR}" >&2
        fi
        echo "$MERGED_MODEL_DIR"
        return 0
    fi
    echo "Need MODEL_PATH or CKPT_DIR with actor/ or huggingface/ subdir" >&2
    return 1
}

ensure_sandbox() {
    local probe_endpoint="${SANDBOX_URLS%%,*}"
    if [[ -f "$SANDBOX_PROBE_SCRIPT" ]] && \
        "$PYTHON_BIN" "$SANDBOX_PROBE_SCRIPT" \
            --endpoint "$probe_endpoint" \
            --requests 1 \
            --workers 1 \
            --require-all-success >/dev/null 2>&1; then
        return 0
    fi
    if curl -sf "${probe_endpoint}/" >/dev/null 2>&1; then
        return 0
    fi
    if [[ "$SETUP_SANDBOX_IF_NEEDED" != "true" ]]; then
        echo "sandbox not healthy and SETUP_SANDBOX_IF_NEEDED=false" >&2
        return 1
    fi
    log "SETUP_SANDBOX start"
    BACKEND_COUNT=8 LB_COUNT=4 BACKENDS_PER_LB=2 BASE_PORT=8081 LB_BASE_PORT=8090 \
        bash "$SETUP_SANDBOX_SCRIPT" >"$LOG_DIR/${EVAL_NAME}_sandbox_setup.log" 2>&1
    log "SETUP_SANDBOX done"
}

build_retained_c_expansion() {
    log "BUILD_RETAINED_C"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/build_step900_retained_c_expansion.py" \
        --curriculum_state "$CURRICULUM_STATE" \
        --base_keep_c_decisions "$KEEP_C_DECISIONS" \
        --base_keep_c_decisions "$KEEP_C_DECISIONS_SECONDARY" \
        --later_c_decisions "$LATER_C_DECISIONS" \
        --min_ema_pass_ratio "$EXPANDED_C_MIN_EMA_PASS_RATIO" \
        --max_ema_timeout_rate "$EXPANDED_C_MAX_EMA_TIMEOUT_RATE" \
        --max_new_kept "$EXPANDED_C_MAX_NEW_KEPT" \
        --output "$RETAINED_C_DECISIONS_JSONL" \
        --summary_out "$RETAINED_C_SUMMARY_JSON"
}

build_candidate_pool() {
    log "BUILD_CANDIDATE_POOL"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/build_repair_conditioned_candidate_pool.py" \
        --curriculum_state "$CURRICULUM_STATE" \
        --train_raw "$TRAIN_RAW" \
        --quarantine "$QUARANTINE_PATH" \
        --valid_big_manifest "$VALID_BIG_MANIFEST" \
        --valid_manifest "$VALID_MANIFEST" \
        --test_manifest "$TEST_MANIFEST" \
        --delta69_manifest "$DELTA69_MANIFEST" \
        --canary_suite "$CANARY_SUITE" \
        --keep_c_decisions "$RETAINED_C_DECISIONS_JSONL" \
        --later_c_decisions "$LATER_C_DECISIONS" \
        --instability_blacklist "$INSTABILITY_BLACKLIST_JSON" \
        --output "$CANDIDATE_POOL_JSONL" \
        --summary_out "$CANDIDATE_POOL_SUMMARY_JSON"
}

materialize_subset() {
    log "MATERIALIZE_SUBSET"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/materialize_repair_candidate_subset.py" \
        --candidate_pool "$CANDIDATE_POOL_JSONL" \
        --train_raw "$TRAIN_RAW" \
        --manifest_out "${SUBSET_MANIFEST_DIR}/codecontests_train_manifest.jsonl" \
        --raw_out "${SUBSET_ROOT}/raw/codecontests_train_raw.jsonl" \
        --meta_out "${SUBSET_ROOT}/slice_meta.json"
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
        --max-model-len 8192 \
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
    local output_dir="${OUTPUT_BASE}/${EVAL_NAME}"
    if [[ "$FORCE_EVAL" != "true" ]] && [[ -f "$output_dir/summary.json" ]]; then
        log "SKIP_EVAL existing summary=${output_dir}/summary.json"
        return 0
    fi
    mkdir -p "$output_dir"
    log "START_EVAL output=${output_dir}"
    "$PYTHON_BIN" "${PROJECT_ROOT}/src/phase0_eval.py" \
        --mode simple \
        --model "$resolved_model_path" \
        --vllm_url "http://localhost:${VLLM_PORT}" \
        --sandbox_url "$SANDBOX_URLS" \
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
        >"$LOG_DIR/${EVAL_NAME}_eval.log" 2>&1
    log "DONE_EVAL output=${output_dir}"
}

build_student_references() {
    local output_dir="${OUTPUT_BASE}/${EVAL_NAME}"
    log "BUILD_STUDENT_REFERENCES"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/build_repair_conditioned_student_references.py" \
        --candidate_pool "$CANDIDATE_POOL_JSONL" \
        --eval_results "${output_dir}/per_problem/codecontests_train.jsonl" \
        --eval_run_info "${output_dir}/run_info.json" \
        --output "$STUDENT_REFS_JSONL" \
        --summary_out "$STUDENT_REFS_SUMMARY_JSON"
}

build_teacher_requests() {
    log "BUILD_TEACHER_REQUESTS"
    "$PYTHON_BIN" "${SCRIPTS_DIR}/build_repair_conditioned_teacher_requests.py" \
        --student_references "$STUDENT_REFS_JSONL" \
        --output "$TEACHER_REQUESTS_JSONL" \
        --summary_out "$TEACHER_REQUESTS_SUMMARY_JSON" \
        --prompt_mode short_diagnosis_code
}

main() {
    ensure_sandbox
    build_retained_c_expansion
    build_candidate_pool
    materialize_subset
    local resolved_model_path
    resolved_model_path="$(resolve_model_path)"
    start_vllm "$resolved_model_path"
    run_eval "$resolved_model_path"
    build_student_references
    build_teacher_requests
    log "DONE step900 repair-conditioned student-reference eval"
}

main "$@"
