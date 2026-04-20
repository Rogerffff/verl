#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$PHASE2_DIR/../.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/coding_model_project/data/grpo_parquet"}

export SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
export VAL_FILES=${VAL_FILES:-"$DATA_DIR/fast_val_codecontests.parquet"}
export LIMITER_BUDGET=${LIMITER_BUDGET:-24}
export RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-30}
export SEED=${SEED:-0}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-20}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
DATA_VAL_BATCH_SIZE=${DATA_VAL_BATCH_SIZE:-16}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-1}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
export ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-True}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export ROLLOUT_N=${ROLLOUT_N:-8}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_a1_formal_observe_lb24_multi2_resume10_seed${SEED}"}

BASE_CKPT_DIR=${BASE_CKPT_DIR:-"/root/verl/checkpoints/rlvr_coding_model/grpo_a1_final_readiness_lb12_micro4_minib8_seed0"}
CKPT_DIR=${CKPT_DIR:-"/root/verl/checkpoints/rlvr_coding_model/${EXPERIMENT_NAME}"}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-"$BASE_CKPT_DIR/global_step_10"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"/root/verl/validation_dumps/${EXPERIMENT_NAME}"}

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
