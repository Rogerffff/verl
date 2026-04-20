#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER=${RUNNER:-"$SCRIPT_DIR/run_phase2_sft_eval_checkpoint.sh"}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}

SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-24}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-24}
BATCH_SIZE=${BATCH_SIZE:-24}

TARGETS=${TARGETS:-"step8 step12 step16 pre_sft_step200"}

STEP8_CKPT=${STEP8_CKPT:-"$ROOT/checkpoints/rlvr_coding_model/phase2_repair_anchor_sft_step200_r1a1_v1/global_step_8"}
STEP12_CKPT=${STEP12_CKPT:-"$ROOT/checkpoints/rlvr_coding_model/phase2_repair_anchor_sft_step200_r1a1_v1/global_step_12"}
STEP16_CKPT=${STEP16_CKPT:-"$ROOT/checkpoints/rlvr_coding_model/phase2_repair_anchor_sft_step200_r1a1_v1/global_step_16"}
PRE_SFT_STEP200_MODEL=${PRE_SFT_STEP200_MODEL:-"$ROOT/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0/global_step_200/actor_merged_hf_phase2_sft"}
BASELINE_MODEL=${BASELINE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

run_one() {
    local target="$1"

    case "$target" in
        step8)
            log "QUEUE target=step8 ckpt=${STEP8_CKPT}"
            CKPT_DIR="$STEP8_CKPT" \
            EVAL_NAME="phase2_sft_step8_fullsuite_v1" \
            SANDBOX_URL="$SANDBOX_URL" \
            VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
            MAX_CONCURRENT="$MAX_CONCURRENT" \
            MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
            BATCH_SIZE="$BATCH_SIZE" \
            bash "$RUNNER"
            ;;
        step12)
            log "QUEUE target=step12 ckpt=${STEP12_CKPT}"
            CKPT_DIR="$STEP12_CKPT" \
            EVAL_NAME="phase2_sft_step12_fullsuite_v1" \
            SANDBOX_URL="$SANDBOX_URL" \
            VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
            MAX_CONCURRENT="$MAX_CONCURRENT" \
            MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
            BATCH_SIZE="$BATCH_SIZE" \
            bash "$RUNNER"
            ;;
        step16)
            log "QUEUE target=step16 ckpt=${STEP16_CKPT}"
            CKPT_DIR="$STEP16_CKPT" \
            EVAL_NAME="phase2_sft_step16_fullsuite_v1" \
            SANDBOX_URL="$SANDBOX_URL" \
            VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
            MAX_CONCURRENT="$MAX_CONCURRENT" \
            MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
            BATCH_SIZE="$BATCH_SIZE" \
            bash "$RUNNER"
            ;;
        pre_sft_step200)
            log "QUEUE target=pre_sft_step200 model=${PRE_SFT_STEP200_MODEL}"
            MODEL_PATH="$PRE_SFT_STEP200_MODEL" \
            EVAL_NAME="phase2_sft_pre_sft_step200_fullsuite_v1" \
            SANDBOX_URL="$SANDBOX_URL" \
            VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
            MAX_CONCURRENT="$MAX_CONCURRENT" \
            MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
            BATCH_SIZE="$BATCH_SIZE" \
            bash "$RUNNER"
            ;;
        baseline)
            log "QUEUE target=baseline model=${BASELINE_MODEL}"
            MODEL_PATH="$BASELINE_MODEL" \
            EVAL_NAME="phase2_sft_baseline_fullsuite_v1" \
            SANDBOX_URL="$SANDBOX_URL" \
            VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
            MAX_CONCURRENT="$MAX_CONCURRENT" \
            MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
            BATCH_SIZE="$BATCH_SIZE" \
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

    if [[ ! -x "$RUNNER" && ! -f "$RUNNER" ]]; then
        echo "Runner not found: $RUNNER" >&2
        exit 1
    fi

    for target in $TARGETS; do
        run_one "$target"
    done

    log "DONE batch targets=${TARGETS}"
}

main "$@"
