#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}
DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/coding_model_project/data/grpo_parquet}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILES=${VAL_FILES:-"[$DATA_DIR/val_tier1.parquet,$DATA_DIR/val_tier2.parquet]"}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8080}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}
ALGO_VARIANT=${ALGO_VARIANT:-A1}
REWARD_MODE=${REWARD_MODE:-anchored_dense_v1}
LIMITER_BUDGET=${LIMITER_BUDGET:-8}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
ROLLOUT_N=${ROLLOUT_N:-8}
RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-30}
ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-True}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
SEED=${SEED:-0}
USE_RESCUE_KL=${USE_RESCUE_KL:-False}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_formal_${ALGO_VARIANT,,}}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files=$TRAIN_FILE
    data.val_files=$VAL_FILES
    data.train_batch_size=$TRAIN_BATCH_SIZE
    data.max_prompt_length=1024
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path=$MODEL_PATH
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.fsdp_config.offload_policy=$ACTOR_OFFLOAD_POLICY
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=$ROLLOUT_N
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    reward_manager.source=register
    reward_manager.name=batch
    reward_model.use_reward_loop=False
    reward_model.launch_reward_fn_async=False
    custom_reward_function.path=$PROJECT_ROOT/coding_model_project/src/grpo_batch_reward.py
    custom_reward_function.name=compute_score
    +custom_reward_function.reward_kwargs.sandbox_endpoint=$SANDBOX_URL
    +custom_reward_function.reward_kwargs.reward_mode=$REWARD_MODE
    +custom_reward_function.reward_kwargs.limiter_budget=$LIMITER_BUDGET
    +custom_reward_function.reward_kwargs.run_timeout_s=$RUN_TIMEOUT_S
    +custom_reward_function.reward_kwargs.memory_limit_mb=1024
    trainer.project_name=rlvr_coding_model
    trainer.experiment_name=$EXPERIMENT_NAME
    trainer.logger=$TRAINER_LOGGER
    trainer.val_before_train=$VAL_BEFORE_TRAIN
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS
    trainer.test_freq=$TEST_FREQ
    trainer.save_freq=$SAVE_FREQ
    data.seed=$SEED
    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    algorithm.filter_groups.enable=false
)

case "$ALGO_VARIANT" in
    A0)
        ALGO_ARGS=(
            algorithm.norm_adv_by_std_in_grpo=True
            actor_rollout_ref.actor.loss_agg_mode=token-mean
            actor_rollout_ref.actor.clip_ratio_low=0.2
            actor_rollout_ref.actor.clip_ratio_high=0.2
            actor_rollout_ref.actor.use_kl_loss=True
            actor_rollout_ref.actor.kl_loss_coef=0.001
        )
        ;;
    A1)
        ALGO_ARGS=(
            algorithm.norm_adv_by_std_in_grpo=True
            actor_rollout_ref.actor.loss_agg_mode=token-mean
            actor_rollout_ref.actor.clip_ratio_low=0.2
            actor_rollout_ref.actor.clip_ratio_high=0.28
            actor_rollout_ref.actor.use_kl_loss=False
        )
        ;;
    A2)
        ALGO_ARGS=(
            algorithm.norm_adv_by_std_in_grpo=False
            actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
            actor_rollout_ref.actor.loss_scale_factor=2048
            actor_rollout_ref.actor.clip_ratio_low=0.2
            actor_rollout_ref.actor.clip_ratio_high=0.28
            actor_rollout_ref.actor.use_kl_loss=False
        )
        ;;
    *)
        echo "Unsupported ALGO_VARIANT=$ALGO_VARIANT" >&2
        exit 1
        ;;
esac

RESCUE_ARGS=()
if [[ "$ALGO_VARIANT" == "A1" && "$USE_RESCUE_KL" == "True" ]]; then
    RESCUE_ARGS=(
        actor_rollout_ref.actor.use_kl_loss=True
        actor_rollout_ref.actor.kl_loss_coef=0.001
        actor_rollout_ref.actor.kl_loss_type=low_var_kl
    )
fi

python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${RESCUE_ARGS[@]}" \
    "$@"
