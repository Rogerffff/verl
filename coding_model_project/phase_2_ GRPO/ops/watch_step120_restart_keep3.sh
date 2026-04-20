#!/usr/bin/env bash

set -euo pipefail

RUN_NAME=${RUN_NAME:-grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0}
RUN_DIR=${RUN_DIR:-/workspace/verl/checkpoints/rlvr_coding_model/$RUN_NAME}
BASE_DIR=${BASE_DIR:-/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0}
VAL_DIR=${VAL_DIR:-/workspace/verl/validation_dumps/$RUN_NAME}
WATCH_LOG=${WATCH_LOG:-/workspace/eval_logs/${RUN_NAME}_watch_step120.log}
RESTART_LOG=${RESTART_LOG:-/workspace/${RUN_NAME}_resume120_keep3_to200.log}
TARGET_STEP=${TARGET_STEP:-120}
KEEP_COUNT=${KEEP_COUNT:-3}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}

mkdir -p "$(dirname "$WATCH_LOG")"

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$WATCH_LOG"
}

actor_sig() {
    local actor_dir="$1"
    if [[ ! -d "$actor_dir" ]]; then
        return 1
    fi
    find "$actor_dir" -type f -printf '%P %s\n' 2>/dev/null | sort | sha256sum | awk '{print $1}'
}

wait_for_checkpoint() {
    local ckpt_dir="$RUN_DIR/global_step_${TARGET_STEP}"
    local actor_dir="$ckpt_dir/actor"
    log "waiting for ${ckpt_dir}"
    while true; do
        if [[ -d "$actor_dir" ]]; then
            local sig1 sig2 latest
            latest="$(cat "$RUN_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
            sig1="$(actor_sig "$actor_dir" || true)"
            sleep 30
            sig2="$(actor_sig "$actor_dir" || true)"
            if [[ -n "$sig1" && "$sig1" == "$sig2" && "$latest" == "$TARGET_STEP" ]]; then
                log "checkpoint global_step_${TARGET_STEP} looks stable"
                du -sh "$ckpt_dir" | tee -a "$WATCH_LOG"
                return 0
            fi
        fi
        sleep 30
    done
}

stop_current_run() {
    log "stopping current training run for ${RUN_NAME}"
    pkill -TERM -f "python3 -m verl.trainer.main_ppo.*${RUN_NAME}" || true
    sleep 20
    pkill -TERM -f "/workspace/verl/coding_model_project/phase_2_ GRPO/run_grpo_formal.sh.*${RUN_NAME}" || true
    pkill -TERM -f "/workspace/verl/coding_model_project/phase_2_ GRPO/run_grpo_a1.sh.*${RUN_NAME}" || true
    sleep 10
    pkill -KILL -f "python3 -m verl.trainer.main_ppo.*${RUN_NAME}" || true
}

skip_old_checkpoint_cleanup() {
    log "skipping old checkpoint cleanup; step100 is preserved and older checkpoints were already handled manually"
    df -h /workspace | tee -a "$WATCH_LOG"
}

restart_with_keep3() {
    log "restarting from global_step_${TARGET_STEP} with keep=${KEEP_COUNT} to total_steps=${TOTAL_TRAINING_STEPS}"
    cd /workspace/verl
    nohup env \
        EXPERIMENT_NAME="$RUN_NAME" \
        PROJECT_ROOT=/workspace/verl \
        SANDBOX_URL=http://localhost:8090 \
        LIMITER_BUDGET=24 \
        TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
        TEST_FREQ=10 \
        SAVE_FREQ=20 \
        VAL_BEFORE_TRAIN=False \
        bash "/workspace/verl/coding_model_project/phase_2_ GRPO/run_grpo_a1.sh" \
        trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
        trainer.test_freq=10 \
        trainer.save_freq=20 \
        trainer.val_before_train=False \
        data.val_files=/workspace/verl/coding_model_project/data/grpo_parquet/fast_val_codecontests.parquet \
        data.val_batch_size=16 \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$RUN_DIR/global_step_${TARGET_STEP}" \
        trainer.default_local_dir="$RUN_DIR" \
        trainer.validation_data_dir="$VAL_DIR" \
        trainer.log_val_generations=0 \
        trainer.max_actor_ckpt_to_keep="$KEEP_COUNT" \
        trainer.max_critic_ckpt_to_keep="$KEEP_COUNT" \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.ref.use_torch_compile=False \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.actor.fsdp_config.offload_policy=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        >"$RESTART_LOG" 2>&1 &
    local new_pid=$!
    log "relaunch pid=${new_pid}"
    sleep 20
    ps -p "$new_pid" -o pid,etime,cmd | tee -a "$WATCH_LOG" || true
}

main() {
    log "watcher started for ${RUN_NAME}"
    wait_for_checkpoint
    stop_current_run
    skip_old_checkpoint_cleanup
    restart_with_keep3
    log "watcher finished"
}

main "$@"
