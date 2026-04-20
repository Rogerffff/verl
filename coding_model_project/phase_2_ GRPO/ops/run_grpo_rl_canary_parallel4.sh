#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER=${RUNNER:-"$SCRIPT_DIR/run_grpo_rl_canary_checkpoint.sh"}
MATERIALIZER=${MATERIALIZER:-"$ROOT/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/materialize_phase2_eval_suite.py"}
SUITE_JSON=${SUITE_JSON:-"$ROOT/coding_model_project/phase_2_ GRPO/rl_canary_suite_v1.json"}
SUITE_ROOT=${SUITE_ROOT:-"$ROOT/coding_model_project/phase_2_ GRPO/rl_canary_materialized_v1"}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
RUN_NAME=${RUN_NAME:-grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0}
RUN_BASE=${RUN_BASE:-"$ROOT/checkpoints/rlvr_coding_model/$RUN_NAME"}
TARGETS=${TARGETS:-"pre_step200 step280 step360 step400"}

GPU_DEVICES_CSV=${GPU_DEVICES_CSV:-0,1,2,3}
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8001}
LB_BASE_PORT=${LB_BASE_PORT:-8090}
MAX_PARALLEL=${MAX_PARALLEL:-4}

VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-16}
MAX_CONCURRENT=${MAX_CONCURRENT:-20}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-16}
BATCH_SIZE=${BATCH_SIZE:-16}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_canary_v1"}
SLICES=${SLICES:-retention_canary_codecontests timeout_canary_codecontests structural_hard_watchlist mid_val_codecontests_32}
REMOVE_MERGED_AFTER_EVAL=${REMOVE_MERGED_AFTER_EVAL:-true}

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

checkpoint_ready() {
    local ckpt_root="$1"
    [[ -d "$ckpt_root" ]] || return 1
    [[ -f "$ckpt_root/data.pt" ]] || return 1
    [[ -d "$ckpt_root/actor" ]] || return 1
}

resolve_pre_step200_ckpt() {
    if [[ -n "${PRE_STEP200_CKPT:-}" ]]; then
        if ! checkpoint_ready "$PRE_STEP200_CKPT"; then
            echo "ERROR: PRE_STEP200_CKPT is set but incomplete: $PRE_STEP200_CKPT" >&2
            exit 1
        fi
        echo "$PRE_STEP200_CKPT"
        return 0
    fi

    local candidate
    local -a candidates=(
        "$ROOT/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0/global_step_200"
        "$ROOT/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0_resume120_keep3_to200/global_step_200"
    )
    for candidate in "${candidates[@]}"; do
        if checkpoint_ready "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done

    echo "ERROR: Could not resolve a valid pre-SFT step200 checkpoint. Set PRE_STEP200_CKPT explicitly." >&2
    exit 1
}

resolve_target_ckpt() {
    local target="$1"
    case "$target" in
        pre_step200)
            echo "$PRE_STEP200_CKPT"
            ;;
        step[0-9]*)
            local step="${target#step}"
            local ckpt="${RUN_BASE}/global_step_${step}"
            if ! checkpoint_ready "$ckpt"; then
                echo "ERROR: checkpoint not ready for target ${target}: ${ckpt}" >&2
                return 1
            fi
            echo "$ckpt"
            ;;
        *)
            echo "ERROR: Unknown target ${target}" >&2
            return 1
            ;;
    esac
}

gpu_for_slot() {
    local slot="$1"
    echo "${GPU_DEVICES[$slot]}"
}

ensure_suite_materialized_once() {
    if [[ -f "$SUITE_ROOT/suite_materialization_summary.json" ]]; then
        return 0
    fi

    local -a slice_args=()
    read -r -a slice_args <<<"$SLICES"
    python3 "$MATERIALIZER" \
        --suite_json "$SUITE_JSON" \
        --raw_root "$ROOT/coding_model_project/data/raw" \
        --output_root "$SUITE_ROOT" \
        --slices "${slice_args[@]}"
}

run_one() {
    local target="$1"
    local slot="$2"
    local ckpt_dir
    ckpt_dir="$(resolve_target_ckpt "$target")"
    local gpu_device
    gpu_device="$(gpu_for_slot "$slot")"
    local sandbox_port=$((LB_BASE_PORT + slot))
    local vllm_port=$((VLLM_BASE_PORT + slot))
    local eval_name
    if [[ "$target" == "pre_step200" ]]; then
        eval_name="${RUN_NAME}_pre_step200_rl_canary_v1"
    else
        eval_name="${RUN_NAME}_${target}_rl_canary_v1"
    fi

    log "START slot=${slot} target=${target} gpu=${gpu_device} vllm_port=${vllm_port} sandbox_port=${sandbox_port}"
    GPU_DEVICE="$gpu_device" \
    VLLM_PORT="$vllm_port" \
    SANDBOX_URL="http://localhost:${sandbox_port}" \
    CKPT_DIR="$ckpt_dir" \
    EVAL_NAME="$eval_name" \
    VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
    MAX_CONCURRENT="$MAX_CONCURRENT" \
    MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
    BATCH_SIZE="$BATCH_SIZE" \
    OUTPUT_BASE="$OUTPUT_BASE" \
    SLICES="$SLICES" \
    SUITE_JSON="$SUITE_JSON" \
    SUITE_ROOT="$SUITE_ROOT" \
    REMOVE_MERGED_AFTER_EVAL="$REMOVE_MERGED_AFTER_EVAL" \
    bash "$RUNNER"
    log "DONE slot=${slot} target=${target}"
}

main() {
    mkdir -p "$LOG_DIR"
    PRE_STEP200_CKPT="$(resolve_pre_step200_ckpt)"
    IFS=',' read -r -a GPU_DEVICES <<<"$GPU_DEVICES_CSV"
    if (( ${#GPU_DEVICES[@]} < MAX_PARALLEL )); then
        echo "Need at least ${MAX_PARALLEL} GPU ids in GPU_DEVICES_CSV" >&2
        exit 1
    fi
    ensure_suite_materialized_once

    read -r -a target_array <<<"$TARGETS"
    local total="${#target_array[@]}"
    local batch_start
    for (( batch_start=0; batch_start<total; batch_start+=MAX_PARALLEL )); do
        pids=()
        names=()
        local slot
        for (( slot=0; slot<MAX_PARALLEL && batch_start + slot < total; slot++ )); do
            local target="${target_array[batch_start + slot]}"
            (
                run_one "$target" "$slot"
            ) >"$LOG_DIR/${RUN_NAME}_${target}_parallel4.log" 2>&1 &
            pids+=("$!")
            names+=("$target")
        done

        local status=0
        local idx
        for idx in "${!pids[@]}"; do
            if ! wait "${pids[$idx]}"; then
                echo "Parallel eval failed for target ${names[$idx]}" >&2
                status=1
            fi
        done
        if (( status != 0 )); then
            exit "$status"
        fi
    done

    log "DONE parallel targets=${TARGETS}"
}

main "$@"
