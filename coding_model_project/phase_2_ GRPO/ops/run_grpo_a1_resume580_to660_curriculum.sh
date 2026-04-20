#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$PHASE2_DIR/../.." && pwd)"}

export SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
export LIMITER_BUDGET=${LIMITER_BUDGET:-128}
export RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-30}
export SEED=${SEED:-0}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
export ROLLOUT_N=${ROLLOUT_N:-8}
export ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-True}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
export VAL_FILES=${VAL_FILES:-"/workspace/verl/coding_model_project/data/grpo_parquet/fast_val_codecontests.parquet"}
export TRAIN_FILE=${TRAIN_FILE:-"/workspace/verl/coding_model_project/data/grpo_parquet/train.parquet"}

# vLLM's CuMemAllocator is incompatible with PyTorch expandable segments.
# Unset it here so resume/launch behavior is stable across shells/hosts.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "INFO: Unsetting PYTORCH_CUDA_ALLOC_CONF for vLLM compatibility"
    unset PYTORCH_CUDA_ALLOC_CONF
fi

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-660}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-20}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
SKIP_DATALOADER_STATE_LOAD=${SKIP_DATALOADER_STATE_LOAD:-False}
DATA_VAL_BATCH_SIZE=${DATA_VAL_BATCH_SIZE:-16}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-3}

CURRICULUM_START_GLOBAL_STEP=${CURRICULUM_START_GLOBAL_STEP:-580}
CURRICULUM_PHASE_BOUNDARIES=${CURRICULUM_PHASE_BOUNDARIES:-"[20,60,80]"}
CURRICULUM_PHASE0_QUOTAS=${CURRICULUM_PHASE0_QUOTAS:-'{"U_unseen":8,"A_retention":3,"B_near_miss":3,"C_hard_partial":1,"D_dead_hard":1}'}
CURRICULUM_PHASE1_QUOTAS=${CURRICULUM_PHASE1_QUOTAS:-'{"U_unseen":3,"A_retention":4,"B_near_miss":6,"C_hard_partial":2,"D_dead_hard":1}'}
CURRICULUM_PHASE2_QUOTAS=${CURRICULUM_PHASE2_QUOTAS:-'{"U_unseen":2,"A_retention":5,"B_near_miss":6,"C_hard_partial":2,"D_dead_hard":1}'}
CURRICULUM_PHASE0_U_REVISIT_QUOTA=${CURRICULUM_PHASE0_U_REVISIT_QUOTA:-0}
CURRICULUM_PHASE1_U_REVISIT_QUOTA=${CURRICULUM_PHASE1_U_REVISIT_QUOTA:-0}
CURRICULUM_PHASE2_U_REVISIT_QUOTA=${CURRICULUM_PHASE2_U_REVISIT_QUOTA:-0}
CURRICULUM_PREFER_LOW_VISITS=${CURRICULUM_PREFER_LOW_VISITS:-False}
CURRICULUM_EMA_ALPHA=${CURRICULUM_EMA_ALPHA:-0.4}
CURRICULUM_MIN_VISITS=${CURRICULUM_MIN_VISITS:-2}
CURRICULUM_RECENT_EXCLUSION_WINDOW=${CURRICULUM_RECENT_EXCLUSION_WINDOW:-2}
CURRICULUM_SNAPSHOT_STEPS=${CURRICULUM_SNAPSHOT_STEPS:-"[600,620,640,660]"}
CURRICULUM_RESET_STATE_ON_DATASET_MISMATCH=${CURRICULUM_RESET_STATE_ON_DATASET_MISMATCH:-False}
CURRICULUM_MIN_RUNTIME_A=${CURRICULUM_MIN_RUNTIME_A:-154}
CURRICULUM_MIN_RUNTIME_B=${CURRICULUM_MIN_RUNTIME_B:-308}
CURRICULUM_MIN_RUNTIME_C=${CURRICULUM_MIN_RUNTIME_C:-154}
CURRICULUM_ASSET_DIR=${CURRICULUM_ASSET_DIR:-"/workspace/verl/coding_model_project/phase_2_ GRPO/curriculum_assets/step580_v1"}
CURRICULUM_MANIFEST_PATH=${CURRICULUM_MANIFEST_PATH:-"$CURRICULUM_ASSET_DIR/curriculum_train_manifest_step580_v1.jsonl"}
CURRICULUM_DATASET_PATH=${CURRICULUM_DATASET_PATH:-"/workspace/verl/coding_model_project/src/step580_curriculum_dataset.py"}
CURRICULUM_SAMPLER_PATH=${CURRICULUM_SAMPLER_PATH:-"/workspace/verl/coding_model_project/src/step580_curriculum_sampler.py"}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_a1_curriculum_step580_v1_seed${SEED}"}
CKPT_DIR=${CKPT_DIR:-"/workspace/verl/checkpoints/rlvr_coding_model/${EXPERIMENT_NAME}"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"/workspace/verl/validation_dumps/${EXPERIMENT_NAME}"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"/workspace/verl/rollout_dumps/${EXPERIMENT_NAME}"}
CURRICULUM_STATE_DIR=${CURRICULUM_STATE_DIR:-"/workspace/verl/coding_model_project/curriculum_states/${EXPERIMENT_NAME}"}

checkpoint_ready() {
    local ckpt_root="$1"
    [[ -d "$ckpt_root" ]] || return 1
    [[ -f "$ckpt_root/data.pt" ]] || return 1
    [[ -d "$ckpt_root/actor" ]] || return 1
}

create_actor_only_resume_shim() {
    local source_ckpt="$1"
    local resume_step="$2"
    local shim_dir="$CURRICULUM_STATE_DIR/resume_shim_global_step_${resume_step}"

    rm -rf "$shim_dir"
    mkdir -p "$shim_dir"
    ln -s "$source_ckpt/actor" "$shim_dir/actor"

    if [[ -d "$source_ckpt/critic" ]]; then
        ln -s "$source_ckpt/critic" "$shim_dir/critic"
    fi
    if [[ -d "$source_ckpt/Critic" ]]; then
        ln -s "$source_ckpt/Critic" "$shim_dir/Critic"
    fi

    echo "$shim_dir"
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
        "/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb128_multi8_b16_micro8_resume550_to580_seed${SEED}/global_step_580"
        "${PROJECT_ROOT}/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb128_multi8_b16_micro8_resume550_to580_seed${SEED}/global_step_580"
    )

    for candidate in "${candidates[@]}"; do
        if checkpoint_ready "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done

    echo "ERROR: Could not resolve a valid step580 checkpoint. Set RESUME_FROM_PATH explicitly." >&2
    exit 1
}

extract_step() {
    local ckpt_path="$1"
    local base
    base="$(basename "$ckpt_path")"
    if [[ "$base" =~ ^global_step_([0-9]+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    echo "ERROR: Cannot parse checkpoint step from path: $ckpt_path" >&2
    exit 1
}

RESUME_FROM_PATH="$(resolve_resume_from_path)"
RESUME_STEP="$(extract_step "$RESUME_FROM_PATH")"
RESUME_STATE_PATH=${RESUME_STATE_PATH:-}

if [[ "$RESUME_STEP" -gt "$CURRICULUM_START_GLOBAL_STEP" && -z "$RESUME_STATE_PATH" ]]; then
    echo "ERROR: curriculum training resume from step ${RESUME_STEP} requires RESUME_STATE_PATH." >&2
    exit 1
fi

if [[ -n "$RESUME_STATE_PATH" && ! -f "$RESUME_STATE_PATH" ]]; then
    echo "ERROR: RESUME_STATE_PATH does not exist: $RESUME_STATE_PATH" >&2
    exit 1
fi

if [[ -n "$RESUME_STATE_PATH" ]]; then
    python3 - "$RESUME_STATE_PATH" "$RESUME_STEP" "$CURRICULUM_START_GLOBAL_STEP" <<'PY'
import json
import sys
from pathlib import Path

snapshot_path = Path(sys.argv[1])
resume_step = int(sys.argv[2])
start_global_step = int(sys.argv[3])
payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

expected_local_update_step = resume_step - start_global_step
if int(payload.get("global_step", -1)) != resume_step:
    raise SystemExit(
        f"ERROR: snapshot global_step {payload.get('global_step')} does not match resume step {resume_step}"
    )
if int(payload.get("local_update_step", -1)) != expected_local_update_step:
    raise SystemExit(
        "ERROR: snapshot local_update_step "
        f"{payload.get('local_update_step')} does not match expected {expected_local_update_step}"
    )
PY
fi

mkdir -p "$CURRICULUM_STATE_DIR" "$VALIDATION_DATA_DIR" "$ROLLOUT_DATA_DIR"

if [[ -z "$RESUME_STATE_PATH" && "$RESUME_STEP" -eq "$CURRICULUM_START_GLOBAL_STEP" ]]; then
    RESUME_FROM_PATH="$(create_actor_only_resume_shim "$RESUME_FROM_PATH" "$RESUME_STEP")"
    echo "INFO: Using actor-only resume shim without dataloader state: $RESUME_FROM_PATH"
fi

EXTRA_ARGS=(
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS"
    trainer.test_freq="$TEST_FREQ"
    trainer.save_freq="$SAVE_FREQ"
    trainer.val_before_train="$VAL_BEFORE_TRAIN"
    +trainer.skip_dataloader_state_load="$SKIP_DATALOADER_STATE_LOAD"
    data.val_batch_size="$DATA_VAL_BATCH_SIZE"
    trainer.resume_mode=resume_path
    trainer.resume_from_path="$RESUME_FROM_PATH"
    trainer.default_local_dir="$CKPT_DIR"
    trainer.validation_data_dir="$VALIDATION_DATA_DIR"
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR"
    trainer.log_val_generations="$LOG_VAL_GENERATIONS"
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP"
    trainer.max_critic_ckpt_to_keep="$MAX_CRITIC_CKPT_TO_KEEP"
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILES"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.dataloader_num_workers=0
    data.custom_cls.path="$CURRICULUM_DATASET_PATH"
    data.custom_cls.name=Step580CurriculumDataset
    data.sampler.class_path="$CURRICULUM_SAMPLER_PATH"
    data.sampler.class_name=Step580CurriculumSampler
    +data.curriculum.manifest_path="$CURRICULUM_MANIFEST_PATH"
    +data.curriculum.state_dir="$CURRICULUM_STATE_DIR"
    +data.curriculum.start_global_step="$CURRICULUM_START_GLOBAL_STEP"
    +data.curriculum.resume_checkpoint_step="$RESUME_STEP"
    +data.curriculum.phase_boundaries="$CURRICULUM_PHASE_BOUNDARIES"
    +data.curriculum.phase0_quotas="'$CURRICULUM_PHASE0_QUOTAS'"
    +data.curriculum.phase1_quotas="'$CURRICULUM_PHASE1_QUOTAS'"
    +data.curriculum.phase2_quotas="'$CURRICULUM_PHASE2_QUOTAS'"
    +data.curriculum.phase0_u_revisit_quota="$CURRICULUM_PHASE0_U_REVISIT_QUOTA"
    +data.curriculum.phase1_u_revisit_quota="$CURRICULUM_PHASE1_U_REVISIT_QUOTA"
    +data.curriculum.phase2_u_revisit_quota="$CURRICULUM_PHASE2_U_REVISIT_QUOTA"
    +data.curriculum.prefer_low_visits="$CURRICULUM_PREFER_LOW_VISITS"
    +data.curriculum.ema_alpha="$CURRICULUM_EMA_ALPHA"
    +data.curriculum.min_visits_for_online="$CURRICULUM_MIN_VISITS"
    +data.curriculum.recent_exclusion_window="$CURRICULUM_RECENT_EXCLUSION_WINDOW"
    +data.curriculum.snapshot_steps="$CURRICULUM_SNAPSHOT_STEPS"
    +data.curriculum.reset_state_on_dataset_mismatch="$CURRICULUM_RESET_STATE_ON_DATASET_MISMATCH"
    +data.curriculum.min_runtime_a="$CURRICULUM_MIN_RUNTIME_A"
    +data.curriculum.min_runtime_b="$CURRICULUM_MIN_RUNTIME_B"
    +data.curriculum.min_runtime_c="$CURRICULUM_MIN_RUNTIME_C"
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.actor.fsdp_config.offload_policy="$ACTOR_OFFLOAD_POLICY"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
)

if [[ -n "$RESUME_STATE_PATH" ]]; then
    EXTRA_ARGS+=(+data.curriculum.resume_state_path="$RESUME_STATE_PATH")
else
    EXTRA_ARGS+=(+data.curriculum.resume_state_path=null)
fi

bash "$PHASE2_DIR/run_grpo_a1.sh" "${EXTRA_ARGS[@]}" "$@"
