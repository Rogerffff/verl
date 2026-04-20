#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
export LIMITER_BUDGET=${LIMITER_BUDGET:-24}
export RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-30}
export SEED=${SEED:-0}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export ROLLOUT_N=${ROLLOUT_N:-8}
export ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-True}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
export VAL_FILES=${VAL_FILES:-"/workspace/verl/coding_model_project/data/grpo_parquet/fast_val_codecontests.parquet"}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-260}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-20}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
DATA_VAL_BATCH_SIZE=${DATA_VAL_BATCH_SIZE:-16}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-4}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-4}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_a1_formal_observe_lb24_multi2_resume200_to260_seed${SEED}"}
CKPT_DIR=${CKPT_DIR:-"/workspace/verl/checkpoints/rlvr_coding_model/${EXPERIMENT_NAME}"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"/workspace/verl/validation_dumps/${EXPERIMENT_NAME}"}

checkpoint_ready() {
    local ckpt_root="$1"
    [[ -d "$ckpt_root" ]] || return 1
    [[ -f "$ckpt_root/data.pt" ]] || return 1
    [[ -d "$ckpt_root/actor" ]] || return 1
}

resolve_resume_from_path() {
    if [[ -n "${RESUME_FROM_PATH:-}" ]]; then
        if ! checkpoint_ready "$RESUME_FROM_PATH"; then
            echo "ERROR: RESUME_FROM_PATH is set but incomplete: $RESUME_FROM_PATH" >&2
            exit 1
        fi
        echo "$RESUME_FROM_PATH"
        return 0
    fi

    local candidate
    local -a candidates=(
        "/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0/global_step_200"
        "/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0_resume120_keep3_to200/global_step_200"
        "${PHASE2_DIR%/coding_model_project/phase_2_ GRPO}/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0/global_step_200"
        "${PHASE2_DIR%/coding_model_project/phase_2_ GRPO}/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0_resume120_keep3_to200/global_step_200"
    )

    for candidate in "${candidates[@]}"; do
        if checkpoint_ready "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done

    echo "ERROR: Could not resolve a valid pre-SFT step200 checkpoint. Set RESUME_FROM_PATH explicitly." >&2
    exit 1
}

RESUME_FROM_PATH="$(resolve_resume_from_path)"

bash "$PHASE2_DIR/run_grpo_a1.sh" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    data.val_batch_size="$DATA_VAL_BATCH_SIZE" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$RESUME_FROM_PATH" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
    trainer.max_critic_ckpt_to_keep="$MAX_CRITIC_CKPT_TO_KEEP" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy="$ACTOR_OFFLOAD_POLICY" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    "$@"
