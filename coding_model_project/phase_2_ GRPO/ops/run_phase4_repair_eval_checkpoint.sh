#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
RUN_TIMEOUT=${RUN_TIMEOUT:-30}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-20000}
MAX_RESPONSE_CHARS=${MAX_RESPONSE_CHARS:-20000}
GPU_DEVICE=${GPU_DEVICE:-0}
VLLM_PORT=${VLLM_PORT:-8016}
VLLM_INTERNAL_PORT_BASE=${VLLM_INTERNAL_PORT_BASE:-}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
MAX_CONCURRENT=${MAX_CONCURRENT:-200}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-150}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-150}
BATCH_SIZE=${BATCH_SIZE:-200}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_validbig500_repair"}
MANIFEST_DIR=${MANIFEST_DIR:-"$ROOT/coding_model_project/data/manifests"}
QUARANTINE_JSON=${QUARANTINE_JSON:-"$ROOT/coding_model_project/data/problem_quarantine_v3.json"}
TMPDIR=${TMPDIR:-/tmp/tbig_repair}
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/dev/vcbig_repair}
REMOVE_MERGED_AFTER_EVAL=${REMOVE_MERGED_AFTER_EVAL:-true}

REPAIR_MIN_PASS_RATIO=${REPAIR_MIN_PASS_RATIO:-0.2}
REPAIR_MAX_FAILURE_CASES=${REPAIR_MAX_FAILURE_CASES:-1}
REPAIR_MAX_FEEDBACK_CHARS=${REPAIR_MAX_FEEDBACK_CHARS:-1200}
REPAIR_MAX_PROMPT_CHARS=${REPAIR_MAX_PROMPT_CHARS:-12000}
REPAIR_PROMPT_MODE=${REPAIR_PROMPT_MODE:-code_only}
REPAIR_ERROR_TYPES=${REPAIR_ERROR_TYPES:-"wrong_answer runtime_error timeout"}
FIRST_PASS_PER_PROBLEM=${FIRST_PASS_PER_PROBLEM:-}

MODEL_PATH=${MODEL_PATH:-}
CKPT_DIR=${CKPT_DIR:-}
MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-}
EVAL_NAME=${EVAL_NAME:-}
VLLM_PID=""

mkdir -p "$LOG_DIR" "$OUTPUT_BASE" "$TMPDIR" "$VLLM_CACHE_ROOT"

log() {
    echo "[$(date --iso-8601=seconds)] $*" >&2
}

cleanup_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true
        wait "$VLLM_PID" || true
    fi
    VLLM_PID=""
}

trap 'cleanup_vllm' EXIT

checkpoint_ready() {
    if [[ -f "$CKPT_DIR/actor/fsdp_config.json" && -f "$CKPT_DIR/actor/huggingface/config.json" ]]; then
        [[ -f "$CKPT_DIR/data.pt" ]] || return 1
        local rank
        for rank in 0 1 2 3; do
            [[ -f "$CKPT_DIR/actor/model_world_size_4_rank_${rank}.pt" ]] || return 1
            [[ -f "$CKPT_DIR/actor/optim_world_size_4_rank_${rank}.pt" ]] || return 1
            [[ -f "$CKPT_DIR/actor/extra_state_world_size_4_rank_${rank}.pt" ]] || return 1
        done
        return 0
    fi

    [[ -f "$CKPT_DIR/data.pt" ]] || return 1
    [[ -f "$CKPT_DIR/fsdp_config.json" ]] || return 1
    [[ -f "$CKPT_DIR/huggingface/config.json" ]] || return 1
    local rank
    for rank in 0 1 2 3; do
        [[ -f "$CKPT_DIR/model_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$CKPT_DIR/optim_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$CKPT_DIR/extra_state_world_size_4_rank_${rank}.pt" ]] || return 1
    done
}

merge_local_dir() {
    if [[ -f "$CKPT_DIR/actor/fsdp_config.json" && -f "$CKPT_DIR/actor/huggingface/config.json" ]]; then
        echo "$CKPT_DIR/actor"
    else
        echo "$CKPT_DIR"
    fi
}

default_merged_model_dir() {
    if [[ -f "$CKPT_DIR/actor/fsdp_config.json" && -f "$CKPT_DIR/actor/huggingface/config.json" ]]; then
        echo "$CKPT_DIR/actor_merged_hf_phase4_repair"
    else
        echo "$CKPT_DIR/merged_hf_phase4_repair"
    fi
}

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

    if [[ -n "$MERGED_MODEL_DIR" ]] && merged_has_weights "$MERGED_MODEL_DIR"; then
        echo "$MERGED_MODEL_DIR"
        return 0
    fi

    if [[ -n "$MERGED_MODEL_DIR" ]]; then
        if ! checkpoint_ready; then
            echo "Checkpoint not complete: $CKPT_DIR" >&2
            return 1
        fi

        log "START_MERGE ckpt_root=${CKPT_DIR}"
        cd "$ROOT"
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$(merge_local_dir)" \
            --target_dir "$MERGED_MODEL_DIR" \
            >"$LOG_DIR/${EVAL_NAME}_merge.log" 2>&1
        log "DONE_MERGE target=${MERGED_MODEL_DIR}"
        echo "$MERGED_MODEL_DIR"
        return 0
    fi

    local candidate
    for candidate in \
        "$CKPT_DIR/actor_merged_hf_valid_big" \
        "$CKPT_DIR/actor_merged_hf_phase4_repair" \
        "$CKPT_DIR/actor_merged_hf_rl_canary" \
        "$CKPT_DIR/huggingface"; do
        if merged_has_weights "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done

    if ! checkpoint_ready; then
        echo "Checkpoint not complete: $CKPT_DIR" >&2
        return 1
    fi

    local target_dir="${MERGED_MODEL_DIR:-"$(default_merged_model_dir)"}"
    log "START_MERGE ckpt_root=${CKPT_DIR}"
    cd "$ROOT"
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$(merge_local_dir)" \
        --target_dir "$target_dir" \
        >"$LOG_DIR/${EVAL_NAME}_merge.log" 2>&1
    log "DONE_MERGE target=${target_dir}"
    echo "$target_dir"
}

start_vllm() {
    local resolved_model_path="$1"
    if curl --max-time 5 -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "Port ${VLLM_PORT} already occupied" >&2
        return 1
    fi

    local internal_port_base="${VLLM_INTERNAL_PORT_BASE:-}"
    log "START_VLLM model=${resolved_model_path} port=${VLLM_PORT} internal_port_base=${internal_port_base:-auto}"
    cd "$ROOT"
    if [[ -n "$internal_port_base" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
        VLLM_PORT="$internal_port_base" \
        VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
        TMPDIR="$TMPDIR" \
        TMP="$TMPDIR" \
        TEMP="$TMPDIR" \
        nohup python3 -m vllm.entrypoints.openai.api_server \
            --model "$resolved_model_path" \
            --port "$VLLM_PORT" \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len 6144 \
            --dtype bfloat16 \
            --trust-remote-code \
            >"$LOG_DIR/${EVAL_NAME}_vllm.log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
        VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
        TMPDIR="$TMPDIR" \
        TMP="$TMPDIR" \
        TEMP="$TMPDIR" \
        nohup python3 -m vllm.entrypoints.openai.api_server \
            --model "$resolved_model_path" \
            --port "$VLLM_PORT" \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len 6144 \
            --dtype bfloat16 \
            --trust-remote-code \
            >"$LOG_DIR/${EVAL_NAME}_vllm.log" 2>&1 &
    fi
    VLLM_PID=$!

    local ok=false
    for _ in $(seq 1 180); do
        if curl --max-time 5 -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
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
    local first_pass_args=()
    if [[ -n "$FIRST_PASS_PER_PROBLEM" ]]; then
        first_pass_args=(--first_pass_per_problem "$FIRST_PASS_PER_PROBLEM")
    fi
    mkdir -p "$output_dir"
    log "START_REPAIR_EVAL datasets=codecontests_valid_big output=${output_dir}"
    # shellcheck disable=SC2086
    python3 "$ROOT/coding_model_project/src/phase4_repair_eval.py" \
        --mode simple \
        --model "$resolved_model_path" \
        --vllm_url "http://localhost:${VLLM_PORT}" \
        --sandbox_url "$SANDBOX_URL" \
        --manifest_dir "$MANIFEST_DIR" \
        --datasets codecontests_valid_big \
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
        --repair_min_pass_ratio "$REPAIR_MIN_PASS_RATIO" \
        --repair_max_failure_cases "$REPAIR_MAX_FAILURE_CASES" \
        --repair_max_feedback_chars "$REPAIR_MAX_FEEDBACK_CHARS" \
        --repair_max_prompt_chars "$REPAIR_MAX_PROMPT_CHARS" \
        --repair_prompt_mode "$REPAIR_PROMPT_MODE" \
        "${first_pass_args[@]}" \
        --repair_error_types $REPAIR_ERROR_TYPES \
        >"$LOG_DIR/${EVAL_NAME}_eval.log" 2>&1
    log "DONE_REPAIR_EVAL datasets=codecontests_valid_big output=${output_dir}"
}

clean_overlay_artifacts_ready() {
    local output_dir="$1"
    [[ -f "$output_dir/first_pass_clean_overlay_problem_quarantine_v3.json" ]] || return 1
    [[ -f "$output_dir/clean_overlay_problem_quarantine_v3.json" ]] || return 1
}

per_problem_artifacts_ready() {
    local output_dir="$1"
    [[ -f "$output_dir/per_problem_first_pass/codecontests_valid_big.jsonl" ]] || return 1
    [[ -f "$output_dir/per_problem/codecontests_valid_big.jsonl" ]] || return 1
}

run_clean_overlays() {
    local output_dir="$OUTPUT_BASE/$EVAL_NAME"
    log "START_CLEAN_OVERLAYS output=${output_dir}"
    python3 "$ROOT/coding_model_project/src/compute_eval_clean_overlay.py" \
        --per-problem-jsonl "$output_dir/per_problem_first_pass/codecontests_valid_big.jsonl" \
        --quarantine-json "$QUARANTINE_JSON" \
        --output-json "$output_dir/first_pass_clean_overlay_problem_quarantine_v3.json" \
        >"$LOG_DIR/${EVAL_NAME}_first_pass_clean.launch.log" 2>&1

    python3 "$ROOT/coding_model_project/src/compute_eval_clean_overlay.py" \
        --per-problem-jsonl "$output_dir/per_problem/codecontests_valid_big.jsonl" \
        --quarantine-json "$QUARANTINE_JSON" \
        --output-json "$output_dir/clean_overlay_problem_quarantine_v3.json" \
        >"$LOG_DIR/${EVAL_NAME}_clean.launch.log" 2>&1
    log "DONE_CLEAN_OVERLAYS output=${output_dir}"
}

main() {
    if [[ -z "$EVAL_NAME" ]]; then
        if [[ -n "$CKPT_DIR" ]]; then
            EVAL_NAME="$(basename "$CKPT_DIR")_codecontests_validbig500_repair"
        else
            EVAL_NAME="codecontests_validbig500_repair_eval"
        fi
    fi

    local output_dir="$OUTPUT_BASE/$EVAL_NAME"
    if [[ -f "$output_dir/summary.json" ]]; then
        if clean_overlay_artifacts_ready "$output_dir"; then
            log "SKIP existing summary and clean overlays at $output_dir"
            exit 0
        fi
        if per_problem_artifacts_ready "$output_dir"; then
            log "SUMMARY exists but clean overlays are missing; regenerating overlays only"
            run_clean_overlays
            exit 0
        fi
        log "SUMMARY exists but required per-problem artifacts are incomplete; rerunning eval"
    fi

    RESOLVED_MODEL_PATH="$(resolve_model_path)"
    start_vllm "$RESOLVED_MODEL_PATH"
    run_eval "$RESOLVED_MODEL_PATH"
    run_clean_overlays

    cleanup_vllm

    if [[ -n "$CKPT_DIR" && "$REMOVE_MERGED_AFTER_EVAL" == "true" ]]; then
        local merged_to_remove="${MERGED_MODEL_DIR:-"$(default_merged_model_dir)"}"
        if [[ "$merged_to_remove" != "$RESOLVED_MODEL_PATH" ]]; then
            return 0
        fi
        rm -rf "$merged_to_remove"
        log "REMOVED_MERGED dir=${merged_to_remove}"
    fi
}

main "$@"
