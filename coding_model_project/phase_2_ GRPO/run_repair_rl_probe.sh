#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}

DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/coding_model_project/data/repair_rl_parquet/step1300_probe_v0}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/smoke_train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/smoke_val.parquet}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
MODEL_PATH=${MODEL_PATH:-}
REPAIR_PROMPT_MODE=${REPAIR_PROMPT_MODE:-code_only}
REWARD_MODE=${REWARD_MODE:-repair_delta_v0}
LIMITER_BUDGET=${LIMITER_BUDGET:-8}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
ROLLOUT_N=${ROLLOUT_N:-1}
RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-30}
ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-True}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-4}
TEST_FREQ=${TEST_FREQ:-2}
SAVE_FREQ=${SAVE_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
SEED=${SEED:-0}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-repair_rl_probe_step1300_${REPAIR_PROMPT_MODE}_smoke}

if [[ -z "$MODEL_PATH" ]]; then
    echo "MODEL_PATH must be set explicitly for repair RL runs." >&2
    exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.seed=$SEED \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=$ACTOR_OFFLOAD_POLICY \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_manager.source=register \
    reward_manager.name=batch \
    reward_model.use_reward_loop=False \
    reward_model.launch_reward_fn_async=False \
    custom_reward_function.path=$PROJECT_ROOT/coding_model_project/src/repair_grpo_batch_reward.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.sandbox_endpoint=$SANDBOX_URL \
    +custom_reward_function.reward_kwargs.reward_mode=$REWARD_MODE \
    +custom_reward_function.reward_kwargs.expected_prompt_mode=$REPAIR_PROMPT_MODE \
    +custom_reward_function.reward_kwargs.limiter_budget=$LIMITER_BUDGET \
    +custom_reward_function.reward_kwargs.run_timeout_s=$RUN_TIMEOUT_S \
    +custom_reward_function.reward_kwargs.memory_limit_mb=1024 \
    trainer.project_name=rlvr_coding_model \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=$TRAINER_LOGGER \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.test_freq=$TEST_FREQ \
    trainer.save_freq=$SAVE_FREQ \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    +algorithm.filter_groups.enable=false \
    +algorithm.filter_groups.metric=null \
    +algorithm.filter_groups.max_num_gen_batches=0 \
    "$@"
