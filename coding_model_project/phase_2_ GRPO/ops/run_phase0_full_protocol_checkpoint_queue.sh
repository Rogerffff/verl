#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
OLD_CKPT_BASE=${OLD_CKPT_BASE:-"$ROOT/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0"}
NEW_CKPT_BASE=${NEW_CKPT_BASE:-"$ROOT/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0"}
STEPS=${STEPS:-"120 160 180 200"}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8090}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-24}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-24}
BATCH_SIZE=${BATCH_SIZE:-50}
VLLM_PORT=${VLLM_PORT:-8001}
GPU_DEVICE=${GPU_DEVICE:-0}
REMOVE_MERGED_AFTER_EVAL=${REMOVE_MERGED_AFTER_EVAL:-true}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs"}
QUEUE_LOG=${QUEUE_LOG:-"$LOG_DIR/phase0_full_protocol_eval_queue_lb24_multi2.out"}

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

VLLM_PID=""

cleanup_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true
        wait "$VLLM_PID" || true
    fi
    VLLM_PID=""
}

trap cleanup_vllm EXIT

log() {
    echo "[$(date --iso-8601=seconds)] $*" | tee -a "$QUEUE_LOG"
}

checkpoint_root_for_step() {
    local step="$1"
    if [[ "$step" == "100" ]]; then
        echo "$OLD_CKPT_BASE/global_step_${step}"
    else
        echo "$NEW_CKPT_BASE/global_step_${step}"
    fi
}

checkpoint_ready() {
    local ckpt_root="$1"
    [[ -f "$ckpt_root/data.pt" ]] || return 1
    [[ -f "$ckpt_root/actor/fsdp_config.json" ]] || return 1
    [[ -f "$ckpt_root/actor/huggingface/config.json" ]] || return 1
    local rank
    for rank in 0 1 2 3; do
        [[ -f "$ckpt_root/actor/model_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$ckpt_root/actor/optim_world_size_4_rank_${rank}.pt" ]] || return 1
        [[ -f "$ckpt_root/actor/extra_state_world_size_4_rank_${rank}.pt" ]] || return 1
    done
}

merged_has_weights() {
    local merged_dir="$1"
    [[ -d "$merged_dir" ]] || return 1
    find "$merged_dir" -maxdepth 1 -type f \
        \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'model*.safetensors' \) \
        -print -quit | grep -q .
}

start_vllm() {
    local model_dir="$1"
    local vllm_log="$2"

    cleanup_vllm

    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        log "port ${VLLM_PORT} already occupied before start"
        return 1
    fi

    log "START_VLLM model_dir=${model_dir}"
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$model_dir" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 6144 \
        --dtype bfloat16 \
        --trust-remote-code \
        >"$vllm_log" 2>&1 &
    VLLM_PID=$!

    local ok=false
    for _ in $(seq 1 120); do
        if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            ok=true
            break
        fi
        sleep 5
    done

    if [[ "$ok" != "true" ]]; then
        log "ERROR vLLM failed to become healthy on port ${VLLM_PORT}"
        return 1
    fi

    log "DONE_VLLM_HEALTH pid=${VLLM_PID}"
}

run_step_eval() {
    local step="$1"
    local ckpt_root
    ckpt_root="$(checkpoint_root_for_step "$step")"
    local actor_dir="$ckpt_root/actor"
    local merged_dir="$ckpt_root/actor_merged_hf"
    local out_dir="$OUTPUT_BASE/phase0_fullval_step${step}_lb24_multi2"
    local merge_log="$LOG_DIR/step${step}_phase0_full_protocol_lb24_multi2_merge.log"
    local vllm_log="$LOG_DIR/step${step}_phase0_full_protocol_lb24_multi2_vllm.log"
    local eval_log="$LOG_DIR/step${step}_phase0_full_protocol_lb24_multi2_eval.log"

    log "CHECK_STEP step=${step} ckpt_root=${ckpt_root}"

    if [[ -f "$out_dir/summary.json" ]]; then
        log "SKIP step=${step} because ${out_dir}/summary.json already exists"
        return 0
    fi

    if ! checkpoint_ready "$ckpt_root"; then
        log "ERROR checkpoint not complete for step=${step}: ${ckpt_root}"
        return 1
    fi

    if ! merged_has_weights "$merged_dir"; then
        log "START_MERGE step=${step}"
        cd "$ROOT"
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$merged_dir" \
            >"$merge_log" 2>&1
        log "DONE_MERGE step=${step}"
    else
        log "SKIP_MERGE step=${step} existing merged weights"
    fi

    start_vllm "$merged_dir" "$vllm_log"

    log "START_EVAL step=${step}"
    python3 "$ROOT/coding_model_project/src/phase0_eval.py" \
        --mode simple \
        --model "$merged_dir" \
        --vllm_url "http://localhost:${VLLM_PORT}" \
        --sandbox_url "$SANDBOX_URL" \
        --manifest_dir "$ROOT/coding_model_project/data/manifests" \
        --datasets humaneval mbpp_reg codecontests_valid \
        --temperature 0.0 \
        --max_tokens 2048 \
        --run_timeout 30 \
        --max_concurrent "$MAX_CONCURRENT" \
        --max_concurrent_judges "$MAX_CONCURRENT_JUDGES" \
        --verifier_limiter_budget "$VERIFIER_LIMITER_BUDGET" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$out_dir" \
        --save_full_results \
        >"$eval_log" 2>&1
    log "DONE_EVAL step=${step}"

    cleanup_vllm

    if [[ "$REMOVE_MERGED_AFTER_EVAL" == "true" ]]; then
        rm -rf "$merged_dir"
        log "REMOVED_MERGED step=${step}"
    fi
}

main() {
    log "START phase0 full-protocol queue with steps=${STEPS} sandbox=${SANDBOX_URL} limiter=${VERIFIER_LIMITER_BUDGET}"
    local step
    for step in $STEPS; do
        run_step_eval "$step"
    done
    log "DONE phase0 full-protocol queue"
}

main "$@"
