#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER=${RUNNER:-"$SCRIPT_DIR/run_grpo_rl_canary_checkpoint.sh"}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
RUN_NAME=${RUN_NAME:-grpo_a1_formal_observe_lb24_multi2_resume200_to260_seed0}
RUN_BASE=${RUN_BASE:-"$ROOT/checkpoints/rlvr_coding_model/$RUN_NAME"}
TARGETS=${TARGETS:-"pre_step200 step220 step240 step260"}

SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-24}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-24}
BATCH_SIZE=${BATCH_SIZE:-24}
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

run_one() {
    local target="$1"
    case "$target" in
        pre_step200)
            log "QUEUE target=pre_step200 ckpt=${PRE_STEP200_CKPT}"
            CKPT_DIR="$PRE_STEP200_CKPT" \
            EVAL_NAME="${RUN_NAME}_pre_step200_rl_canary_v1" \
            SANDBOX_URL="$SANDBOX_URL" \
            VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
            MAX_CONCURRENT="$MAX_CONCURRENT" \
            MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
            BATCH_SIZE="$BATCH_SIZE" \
            OUTPUT_BASE="$OUTPUT_BASE" \
            SLICES="$SLICES" \
            REMOVE_MERGED_AFTER_EVAL="$REMOVE_MERGED_AFTER_EVAL" \
            bash "$RUNNER"
            ;;
        step[0-9]*)
            local step="${target#step}"
            if ! checkpoint_ready "${RUN_BASE}/global_step_${step}"; then
                echo "ERROR: checkpoint not ready for target ${target}: ${RUN_BASE}/global_step_${step}" >&2
                return 1
            fi
            log "QUEUE target=${target} ckpt=${RUN_BASE}/global_step_${step}"
            CKPT_DIR="${RUN_BASE}/global_step_${step}" \
            EVAL_NAME="${RUN_NAME}_step${step}_rl_canary_v1" \
            SANDBOX_URL="$SANDBOX_URL" \
            VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
            MAX_CONCURRENT="$MAX_CONCURRENT" \
            MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
            BATCH_SIZE="$BATCH_SIZE" \
            OUTPUT_BASE="$OUTPUT_BASE" \
            SLICES="$SLICES" \
            REMOVE_MERGED_AFTER_EVAL="$REMOVE_MERGED_AFTER_EVAL" \
            bash "$RUNNER"
            ;;
        *)
            echo "Unknown target: $target" >&2
            return 1
            ;;
    esac
}

main() {
    mkdir -p "$LOG_DIR"
    PRE_STEP200_CKPT="$(resolve_pre_step200_ckpt)"
    for target in $TARGETS; do
        run_one "$target"
    done
    log "DONE batch targets=${TARGETS}"
}

main "$@"
