#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_a1_curriculum_step620_v3q3_to900_seed0}
CKPT_ROOT=${CKPT_ROOT:-"$ROOT/checkpoints/rlvr_coding_model/$EXPERIMENT_NAME"}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_delta69"}
DELTA69_MANIFEST_DIR=${DELTA69_MANIFEST_DIR:-"$ROOT/coding_model_project/phase_2_ GRPO/review_assets/step640_v1/delta69_eval_slice_raw/manifests"}

# delta69 只有 69 题；外层并发开到 69 即可全量铺开。
MAX_CONCURRENT=${MAX_CONCURRENT:-69}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-72}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-72}
BATCH_SIZE=${BATCH_SIZE:-69}

launch_one() {
    local step="$1"
    local gpu="$2"
    local vllm_port="$3"
    local internal_port="$4"
    local sandbox_url="$5"
    local eval_name="${EXPERIMENT_NAME}_step${step}_delta69_raw"
    local launch_log="$LOG_DIR/${eval_name}.launch.log"

    mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

    nohup env \
        ROOT="$ROOT" \
        GPU_DEVICE="$gpu" \
        VLLM_PORT="$vllm_port" \
        VLLM_INTERNAL_PORT_BASE="$internal_port" \
        SANDBOX_URL="$sandbox_url" \
        MAX_CONCURRENT="$MAX_CONCURRENT" \
        MAX_CONCURRENT_JUDGES="$MAX_CONCURRENT_JUDGES" \
        VERIFIER_LIMITER_BUDGET="$VERIFIER_LIMITER_BUDGET" \
        BATCH_SIZE="$BATCH_SIZE" \
        CKPT_DIR="$CKPT_ROOT/global_step_${step}" \
        EVAL_NAME="$eval_name" \
        LOG_DIR="$LOG_DIR" \
        OUTPUT_BASE="$OUTPUT_BASE" \
        DELTA69_MANIFEST_DIR="$DELTA69_MANIFEST_DIR" \
        "$ROOT/coding_model_project/phase_2_ GRPO/ops/run_grpo_codecontests_delta69_checkpoint.sh" \
        >"$launch_log" 2>&1 < /dev/null &

    echo "launched step=${step} gpu=${gpu} sandbox=${sandbox_url} log=${launch_log}"
}

launch_one 700 0 8116 8316 http://localhost:8090
launch_one 800 1 8117 8317 http://localhost:8091
launch_one 900 2 8118 8318 http://localhost:8092
