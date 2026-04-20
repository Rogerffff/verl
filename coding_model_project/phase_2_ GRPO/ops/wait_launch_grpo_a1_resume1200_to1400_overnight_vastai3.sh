#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED=${SEED:-0}
SOURCE_EXPERIMENT=${SOURCE_EXPERIMENT:-"grpo_a1_curriculum_step1000_overnight_u5a4b4c3_to1200_lb240_seed${SEED}_vastai3"}
TARGET_EXPERIMENT=${TARGET_EXPERIMENT:-"grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed${SEED}_vastai3"}
TARGET_STEP=${TARGET_STEP:-1200}
SOURCE_CKPT_DIR=${SOURCE_CKPT_DIR:-"/workspace/verl/checkpoints/rlvr_coding_model/${SOURCE_EXPERIMENT}"}
SOURCE_STATE_DIR=${SOURCE_STATE_DIR:-"/workspace/verl/coding_model_project/curriculum_states/${SOURCE_EXPERIMENT}"}
TARGET_CKPT="${SOURCE_CKPT_DIR}/global_step_${TARGET_STEP}"
TARGET_STATE="${SOURCE_STATE_DIR}/curriculum_state_step_${TARGET_STEP}.json"
WAIT_LOG=${WAIT_LOG:-"/workspace/eval_logs/${SOURCE_EXPERIMENT}_wait_resume1200_to1400.log"}
LAUNCH_LOG=${LAUNCH_LOG:-"/workspace/eval_logs/grpo_a1_resume1200_to1400_overnight_vastai3.log"}
LOCK_FILE=${LOCK_FILE:-"/tmp/${TARGET_EXPERIMENT}.wait.lock"}
LAUNCH_MARKER=${LAUNCH_MARKER:-"/workspace/eval_logs/${TARGET_EXPERIMENT}.launch_requested"}
TARGET_CKPT_DIR=${TARGET_CKPT_DIR:-"/workspace/verl/checkpoints/rlvr_coding_model/${TARGET_EXPERIMENT}"}
GPU_FREE_THRESHOLD_MB=${GPU_FREE_THRESHOLD_MB:-2000}
LAUNCH_CONFIRM_TIMEOUT_S=${LAUNCH_CONFIRM_TIMEOUT_S:-120}
LAUNCH_RETRY_INTERVAL_S=${LAUNCH_RETRY_INTERVAL_S:-60}

mkdir -p "$(dirname "$WAIT_LOG")"

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$WAIT_LOG"
}

checkpoint_ready() {
    local ckpt_root="$1"
    [[ -d "$ckpt_root" ]] || return 1
    [[ -f "$ckpt_root/data.pt" ]] || return 1
    [[ -d "$ckpt_root/actor" ]] || return 1
}

acquire_lock() {
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "another watcher is already active; exiting"
        exit 0
    fi
}

continuation_already_started() {
    if pgrep -f "python3 -m verl.trainer.main_ppo.*${TARGET_EXPERIMENT}" >/dev/null 2>&1; then
        return 0
    fi
    if [[ -f "${TARGET_CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
        return 0
    fi
    if [[ -f "$LAUNCH_MARKER" ]]; then
        local marker_pid=""
        marker_pid="$(grep '^launch_shell_pid=' "$LAUNCH_MARKER" 2>/dev/null | head -n1 | cut -d= -f2 || true)"
        if [[ -n "$marker_pid" ]] && kill -0 "$marker_pid" 2>/dev/null; then
            return 0
        fi
        log "stale launch marker detected; removing $LAUNCH_MARKER"
        rm -f "$LAUNCH_MARKER"
    fi
    return 1
}

wait_for_step1200() {
    log "waiting for checkpoint ${TARGET_CKPT} and state ${TARGET_STATE}"
    while true; do
        local latest=""
        latest="$(cat "${SOURCE_CKPT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
        if checkpoint_ready "$TARGET_CKPT" && [[ -f "$TARGET_STATE" && "$latest" == "$TARGET_STEP" ]]; then
            log "step ${TARGET_STEP} artifacts detected"
            du -sh "$TARGET_CKPT" | tee -a "$WAIT_LOG"
            ls -lh "$TARGET_STATE" | tee -a "$WAIT_LOG"
            return 0
        fi
        sleep 60
    done
}

wait_for_source_exit() {
    log "waiting for source training process to exit: ${SOURCE_EXPERIMENT}"
    while pgrep -f "python3 -m verl.trainer.main_ppo.*${SOURCE_EXPERIMENT}" >/dev/null 2>&1; do
        sleep 30
    done
    log "source training process has exited"
}

wait_for_gpus_free() {
    log "waiting for GPUs to become available"
    while true; do
        local busy=0
        while IFS= read -r used_mb; do
            [[ -z "$used_mb" ]] && continue
            if (( used_mb > GPU_FREE_THRESHOLD_MB )); then
                busy=1
                break
            fi
        done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true)
        if (( busy == 0 )); then
            log "GPU memory usage is below threshold ${GPU_FREE_THRESHOLD_MB}MB on all visible GPUs"
            return 0
        fi
        sleep 30
    done
}

launch_and_confirm() {
    local launch_pid=""
    local deadline=0
    local now=0

    cd /workspace/verl
    nohup bash "/workspace/verl/coding_model_project/phase_2_ GRPO/ops/run_grpo_a1_resume1200_to1400_overnight_vastai3.sh" \
        >"$LAUNCH_LOG" 2>&1 &
    launch_pid=$!
    log "launch attempt started shell_pid=${launch_pid}"

    deadline=$(( $(date +%s) + LAUNCH_CONFIRM_TIMEOUT_S ))
    while true; do
        if pgrep -f "python3 -m verl.trainer.main_ppo.*${TARGET_EXPERIMENT}" >/dev/null 2>&1; then
            {
                echo "launch_shell_pid=${launch_pid}"
                echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
                echo "launch_log=${LAUNCH_LOG}"
            } > "$LAUNCH_MARKER"
            log "launch confirmed for ${TARGET_EXPERIMENT}; marker written to ${LAUNCH_MARKER}"
            return 0
        fi
        now=$(date +%s)
        if (( now >= deadline )); then
            log "launch confirmation timed out after ${LAUNCH_CONFIRM_TIMEOUT_S}s; will retry"
            tail -n 40 "$LAUNCH_LOG" 2>/dev/null | tee -a "$WAIT_LOG" || true
            return 1
        fi
        sleep 10
    done
}

main() {
    acquire_lock
    if continuation_already_started; then
        log "continuation already launched or checkpointed; exiting"
        exit 0
    fi
    log "waiter started"
    wait_for_step1200
    if continuation_already_started; then
        log "continuation already launched after artifact readiness; exiting"
        exit 0
    fi
    wait_for_source_exit
    wait_for_gpus_free
    while true; do
        if continuation_already_started; then
            log "continuation already launched before final handoff; exiting"
            exit 0
        fi
        log "launching 1200->1400 continuation"
        if launch_and_confirm; then
            exit 0
        fi
        sleep "$LAUNCH_RETRY_INTERVAL_S"
    done
}

main "$@"
