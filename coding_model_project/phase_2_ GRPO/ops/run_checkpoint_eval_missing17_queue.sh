#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-"$(cd "$PROJECT_ROOT/.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/coding_model_project/data/grpo_parquet"}
EVAL_LOG_DIR=${EVAL_LOG_DIR:-"$WORKSPACE_ROOT/eval_logs"}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-"$PROJECT_ROOT/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0"}
FILTERED_QUEUE_LOG=${FILTERED_QUEUE_LOG:-"$EVAL_LOG_DIR/grpo_eval_queue_step60_step80.out"}
WAIT_FOR_MARKER=${WAIT_FOR_MARKER:-"DONE grpo_a1_eval_step80_humaneval"}
MISSING_PARQUET=${MISSING_PARQUET:-"$DATA_DIR/codecontests_valid_missing17.parquet"}
MAX_PROMPT_LENGTH_FULL=${MAX_PROMPT_LENGTH_FULL:-16384}
SUMMARY_SCRIPT=${SUMMARY_SCRIPT:-"$SCRIPT_DIR/summarize_checkpoint_eval.py"}

mkdir -p "$EVAL_LOG_DIR"

wait_for_filtered_queue() {
    while true; do
        if [[ -f "$FILTERED_QUEUE_LOG" ]] && grep -q "$WAIT_FOR_MARKER" "$FILTERED_QUEUE_LOG"; then
            echo "FILTERED_QUEUE_DONE marker='$WAIT_FOR_MARKER' at $(date --iso-8601=seconds)"
            return 0
        fi
        echo "WAITING_FOR_FILTERED_QUEUE marker='$WAIT_FOR_MARKER' at $(date --iso-8601=seconds)"
        sleep 120
    done
}

run_missing_eval() {
    local step="$1"
    local ckpt="$CHECKPOINT_BASE_DIR/global_step_${step}"
    local exp="grpo_a1_eval_step${step}_codecontests_missing17_full"
    local log_path="$EVAL_LOG_DIR/${exp}.log"
    local dump_dir="$PROJECT_ROOT/validation_dumps/${exp}"
    local local_dir="$PROJECT_ROOT/checkpoints/rlvr_coding_model/${exp}"

    if [[ ! -d "$ckpt" ]]; then
        echo "Checkpoint missing for step=$step: $ckpt" >&2
        return 1
    fi

    if [[ -f "$dump_dir/${step}.jsonl" ]]; then
        echo "SKIP $exp because $dump_dir/${step}.jsonl already exists"
        return 0
    fi

    echo "START $exp at $(date --iso-8601=seconds)"
    SANDBOX_URL="${SANDBOX_URL:-http://localhost:8090}" \
    LIMITER_BUDGET="${LIMITER_BUDGET:-24}" \
    RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-30}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}" \
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}" \
    PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}" \
    ROLLOUT_N="${ROLLOUT_N:-8}" \
    ACTOR_OFFLOAD_POLICY="${ACTOR_OFFLOAD_POLICY:-True}" \
    EXPERIMENT_NAME="$exp" \
    RESUME_FROM_PATH="$ckpt" \
    VAL_FILES="$MISSING_PARQUET" \
    VALIDATION_DATA_DIR="$dump_dir" \
    CKPT_DIR="$local_dir" \
    TRAINER_LOGGER='["console"]' \
    bash "$SCRIPT_DIR/run_grpo_a1_eval_only.sh" \
        data.max_prompt_length="$MAX_PROMPT_LENGTH_FULL" \
        data.filter_overlong_prompts=True \
        >"$log_path" 2>&1
    echo "DONE $exp at $(date --iso-8601=seconds)"
}

write_summary() {
    local step="$1"
    local output_json="$EVAL_LOG_DIR/grpo_a1_eval_step${step}_combined_summary.json"
    local filtered_dump="$PROJECT_ROOT/validation_dumps/grpo_a1_eval_step${step}_valtier1/${step}.jsonl"
    local missing_dump="$PROJECT_ROOT/validation_dumps/grpo_a1_eval_step${step}_codecontests_missing17_full/${step}.jsonl"
    local humaneval_dump="$PROJECT_ROOT/validation_dumps/grpo_a1_eval_step${step}_humaneval/${step}.jsonl"

    if [[ ! -f "$filtered_dump" ]]; then
        echo "Filtered dump missing for step=$step: $filtered_dump" >&2
        return 1
    fi

    if [[ ! -f "$humaneval_dump" ]]; then
        echo "Humaneval dump missing for step=$step: $humaneval_dump" >&2
        return 1
    fi

    python3 "$SUMMARY_SCRIPT" "$output_json" "$filtered_dump" "$missing_dump" "$humaneval_dump"
}

main() {
    if [[ ! -f "$MISSING_PARQUET" ]]; then
        echo "Missing subset parquet not found: $MISSING_PARQUET" >&2
        exit 1
    fi

    wait_for_filtered_queue
    run_missing_eval 60
    write_summary 60
    run_missing_eval 80
    write_summary 80
}

main "$@"
