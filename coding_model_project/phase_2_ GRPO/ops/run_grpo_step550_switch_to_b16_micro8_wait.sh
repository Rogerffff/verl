#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT=${ROOT:-/workspace/verl}
RUN_NAME=${RUN_NAME:-grpo_a1_formal_observe_lb96_multi6_b16_onpolicy_resume520_to580_seed0}
TARGET_STEP=${TARGET_STEP:-550}
CKPT_ROOT=${CKPT_ROOT:-"$ROOT/checkpoints/rlvr_coding_model/$RUN_NAME"}
CKPT_DIR=${CKPT_DIR:-"$CKPT_ROOT/global_step_${TARGET_STEP}"}

WAIT_TIMEOUT_S=${WAIT_TIMEOUT_S:-21600}
POLL_INTERVAL_S=${POLL_INTERVAL_S:-20}

SWITCH_TRAIN_LOG=${SWITCH_TRAIN_LOG:-/workspace/eval_logs/grpo_a1_formal_observe_lb128_multi8_b16_micro8_resume550_to580_seed0.out}
SETUP_SANDBOX_SCRIPT=${SETUP_SANDBOX_SCRIPT:-"$SCRIPT_DIR/setup_reward_sandbox_lb8.sh"}
NEXT_TRAIN_SCRIPT=${NEXT_TRAIN_SCRIPT:-"$SCRIPT_DIR/run_grpo_a1_resume550_to580_b16_micro8.sh"}
STATE_ROOT=${STATE_ROOT:-/root/sandboxfusion-multi}
SANDBOX_VENV=${SANDBOX_VENV:-/root/sandboxfusion-venv}

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

checkpoint_ready() {
    [[ -f "$CKPT_DIR/data.pt" ]] || return 1
    [[ -f "$CKPT_DIR/actor/fsdp_config.json" ]] || return 1
    [[ -f "$CKPT_DIR/actor/huggingface/config.json" ]] || return 1
    local rank
    for rank in 0 1 2 3; do
        [[ -f "$CKPT_DIR/actor/model_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$CKPT_DIR/actor/optim_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$CKPT_DIR/actor/extra_state_world_size_4_rank_${rank}.pt" ]] || return 1
    done
}

wait_for_checkpoint() {
    local waited=0
    while ! checkpoint_ready; do
        if (( waited >= WAIT_TIMEOUT_S )); then
            echo "Timed out waiting for checkpoint readiness: $CKPT_DIR" >&2
            return 1
        fi
        log "WAIT_CHECKPOINT ckpt=${CKPT_DIR} waited_s=${waited}"
        sleep "$POLL_INTERVAL_S"
        waited=$((waited + POLL_INTERVAL_S))
    done
    log "CHECKPOINT_READY ckpt=${CKPT_DIR}"
}

stop_current_train() {
    local pids
    pids="$(pgrep -f "$RUN_NAME" || true)"
    if [[ -z "$pids" ]]; then
        log "NO_RUNNING_TRAIN run=${RUN_NAME}"
        return 0
    fi
    log "STOP_TRAIN run=${RUN_NAME} pids=$(echo "$pids" | tr '\n' ',' | sed 's/,$//')"
    while IFS= read -r pid; do
        [[ -n "$pid" ]] || continue
        kill "$pid" >/dev/null 2>&1 || true
    done <<<"$pids"
    sleep 10
    pids="$(pgrep -f "$RUN_NAME" || true)"
    if [[ -n "$pids" ]]; then
        while IFS= read -r pid; do
            [[ -n "$pid" ]] || continue
            kill -9 "$pid" >/dev/null 2>&1 || true
        done <<<"$pids"
    fi
    log "STOP_TRAIN_DONE run=${RUN_NAME}"
}

setup_reward_pool() {
    log "SETUP_REWARD_POOL script=$(basename "$SETUP_SANDBOX_SCRIPT")"
    STATE_ROOT="$STATE_ROOT" SANDBOX_VENV="$SANDBOX_VENV" bash "$SETUP_SANDBOX_SCRIPT"
    log "SETUP_REWARD_POOL_DONE"
}

launch_next_train() {
    log "START_NEXT_TRAIN script=$(basename "$NEXT_TRAIN_SCRIPT") log=${SWITCH_TRAIN_LOG}"
    nohup env STATE_ROOT="$STATE_ROOT" SANDBOX_VENV="$SANDBOX_VENV" bash "$NEXT_TRAIN_SCRIPT" >"$SWITCH_TRAIN_LOG" 2>&1 &
    log "DONE_NEXT_TRAIN pid=$!"
}

main() {
    mkdir -p /workspace/eval_logs
    wait_for_checkpoint
    stop_current_train
    setup_reward_pool
    launch_next_train
}

main "$@"
