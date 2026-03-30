#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}
DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/coding_model_project/data/grpo_parquet}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8080}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}
REWARD_MODE=${REWARD_MODE:-dense_pass_ratio}
LIMITER_BUDGET=${LIMITER_BUDGET:-8}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files=$DATA_DIR/smoke_train.parquet \
    data.val_files=$DATA_DIR/smoke_val.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_manager.source=register \
    reward_manager.name=batch \
    reward_model.launch_reward_fn_async=False \
    custom_reward_function.path=$PROJECT_ROOT/coding_model_project/src/grpo_batch_reward.py \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.sandbox_endpoint=$SANDBOX_URL \
    custom_reward_function.reward_kwargs.reward_mode=$REWARD_MODE \
    custom_reward_function.reward_kwargs.limiter_budget=$LIMITER_BUDGET \
    custom_reward_function.reward_kwargs.run_timeout_s=30 \
    custom_reward_function.reward_kwargs.memory_limit_mb=1024 \
    trainer.project_name=rlvr_coding_model \
    trainer.experiment_name=grpo_smoke_shared_verifier \
    trainer.logger='["console"]' \
    trainer.val_before_train=True \
    trainer.total_epochs=1 \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    $@
