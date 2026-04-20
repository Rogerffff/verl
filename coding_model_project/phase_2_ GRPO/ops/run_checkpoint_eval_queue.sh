#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/coding_model_project/data/grpo_parquet"}
EVAL_LOG_DIR=${EVAL_LOG_DIR:-"$PROJECT_ROOT/eval_logs"}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-"$PROJECT_ROOT/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0"}
WAIT_FOR_STEP=${WAIT_FOR_STEP:-80}
START_STEP=${START_STEP:-60}

mkdir -p "$EVAL_LOG_DIR"

HUMANEVAL_PARQUET=${HUMANEVAL_PARQUET:-"$DATA_DIR/humaneval_only.parquet"}

ensure_humaneval_subset() {
    if [[ -f "$HUMANEVAL_PARQUET" ]]; then
        return
    fi

    python3 - "$DATA_DIR/final_eval.parquet" "$HUMANEVAL_PARQUET" <<'PY'
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)
subset = df[df["data_source"] == "humaneval"].reset_index(drop=True)
if subset.empty:
    raise SystemExit("No humaneval rows found in final_eval.parquet")
subset.to_parquet(dst, index=False)
print(f"Wrote {len(subset)} humaneval rows to {dst}")
PY
}

checkpoint_complete() {
    local ckpt_dir="$1"
    [[ -f "$ckpt_dir/data.pt" ]] || return 1
    [[ -f "$ckpt_dir/actor/fsdp_config.json" ]] || return 1
    [[ -f "$ckpt_dir/actor/huggingface/config.json" ]] || return 1

    local rank
    for rank in 0 1 2 3; do
        [[ -f "$ckpt_dir/actor/model_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$ckpt_dir/actor/optim_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$ckpt_dir/actor/extra_state_world_size_4_rank_${rank}.pt" ]] || return 1
    done
}

checkpoint_signature() {
    local ckpt_dir="$1"
    find "$ckpt_dir" \
        \( -path "*/actor/model_world_size_4_rank_*.pt" \
        -o -path "*/actor/optim_world_size_4_rank_*.pt" \
        -o -path "*/actor/extra_state_world_size_4_rank_*.pt" \
        -o -path "*/actor/huggingface/config.json" \
        -o -path "*/actor/fsdp_config.json" \
        -o -name "data.pt" \) \
        -type f -printf "%P %s\n" | sort
}

checkpoint_stable() {
    local ckpt_dir="$1"
    local first
    local second

    first="$(checkpoint_signature "$ckpt_dir")"
    sleep 120
    checkpoint_complete "$ckpt_dir" || return 1
    second="$(checkpoint_signature "$ckpt_dir")"
    [[ "$first" == "$second" ]]
}

run_eval() {
    local step="$1"
    local dataset_tag="$2"
    local val_files="$3"
    local ckpt="$CHECKPOINT_BASE_DIR/global_step_${step}"
    local exp="grpo_a1_eval_step${step}_${dataset_tag}"
    local log_path="$EVAL_LOG_DIR/${exp}.log"
    local dump_dir="$PROJECT_ROOT/validation_dumps/${exp}"
    local local_dir="$PROJECT_ROOT/checkpoints/rlvr_coding_model/${exp}"

    if [[ -f "$dump_dir/${step}.jsonl" ]]; then
        echo "SKIP $exp because $dump_dir/${step}.jsonl already exists"
        return
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
    VAL_FILES="$val_files" \
    VALIDATION_DATA_DIR="$dump_dir" \
    CKPT_DIR="$local_dir" \
    TRAINER_LOGGER='["console"]' \
    bash "$SCRIPT_DIR/run_grpo_a1_eval_only.sh" >"$log_path" 2>&1

    echo "DONE $exp at $(date --iso-8601=seconds)"
}

wait_for_step() {
    local step="$1"
    local ckpt="$CHECKPOINT_BASE_DIR/global_step_${step}"

    while true; do
        if checkpoint_complete "$ckpt" && checkpoint_stable "$ckpt"; then
            echo "CHECKPOINT_READY step=$step at $(date --iso-8601=seconds)"
            return 0
        fi
        echo "WAITING step=$step at $(date --iso-8601=seconds)"
        sleep 120
    done
}

main() {
    ensure_humaneval_subset

    run_eval "$START_STEP" "valtier1" "$DATA_DIR/val_tier1.parquet"
    run_eval "$START_STEP" "humaneval" "$HUMANEVAL_PARQUET"

    if [[ "$WAIT_FOR_STEP" != "$START_STEP" ]]; then
        wait_for_step "$WAIT_FOR_STEP"
        run_eval "$WAIT_FOR_STEP" "valtier1" "$DATA_DIR/val_tier1.parquet"
        run_eval "$WAIT_FOR_STEP" "humaneval" "$HUMANEVAL_PARQUET"
    fi
}

main "$@"
