#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$PHASE2_DIR/../.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/coding_model_project/data/grpo_parquet"}

export VAL_FILES=${VAL_FILES:-"$DATA_DIR/fast_val_codecontests.parquet"}
export SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
export LIMITER_BUDGET=${LIMITER_BUDGET:-12}
export RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-30}
export SEED=${SEED:-0}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-4}
TEST_FREQ=${TEST_FREQ:-0}
SAVE_FREQ=${SAVE_FREQ:-0}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
DATA_VAL_BATCH_SIZE=${DATA_VAL_BATCH_SIZE:-16}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
export ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-True}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export ROLLOUT_N=${ROLLOUT_N:-8}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_a1_reward_probe_lb${LIMITER_BUDGET}_seed${SEED}"}

bash "$PHASE2_DIR/run_grpo_a1.sh" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    data.val_batch_size="$DATA_VAL_BATCH_SIZE" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    "$@"
