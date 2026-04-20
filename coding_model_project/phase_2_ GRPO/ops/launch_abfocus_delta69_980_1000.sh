#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/workspace/verl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0}
CKPT_ROOT=${CKPT_ROOT:-"$ROOT/checkpoints/rlvr_coding_model/$EXPERIMENT_NAME"}
LOG_DIR=${LOG_DIR:-/workspace/eval_logs}
OUTPUT_BASE=${OUTPUT_BASE:-"$ROOT/coding_model_project/outputs/rl_codecontests_delta69"}
DELTA69_MANIFEST_DIR=${DELTA69_MANIFEST_DIR:-"$ROOT/coding_model_project/phase_2_ GRPO/review_assets/step640_v1/delta69_eval_slice_raw/manifests"}
QUARANTINE_JSON=${QUARANTINE_JSON:-"$ROOT/coding_model_project/data/problem_quarantine_v3.json"}

# delta69 只有 69 题；外层并发开到 69 即可铺满。
MAX_CONCURRENT=${MAX_CONCURRENT:-69}
MAX_CONCURRENT_JUDGES=${MAX_CONCURRENT_JUDGES:-72}
VERIFIER_LIMITER_BUDGET=${VERIFIER_LIMITER_BUDGET:-72}
BATCH_SIZE=${BATCH_SIZE:-69}

wait_checkpoint_ready() {
    local ckpt_dir="$1"
    local max_tries="${2:-720}"
    local i
    for ((i=1; i<=max_tries; i++)); do
        if [[ -f "$ckpt_dir/data.pt" && -f "$ckpt_dir/actor/fsdp_config.json" && -f "$ckpt_dir/actor/huggingface/config.json" ]]; then
            return 0
        fi
        sleep 5
    done
    echo "checkpoint not ready: $ckpt_dir" >&2
    return 1
}

launch_one() {
    local step="$1"
    local gpu="$2"
    local vllm_port="$3"
    local internal_port="$4"
    local sandbox_url="$5"
    local eval_name="${EXPERIMENT_NAME}_step${step}_delta69_raw"
    local out_dir="${OUTPUT_BASE}/${eval_name}"
    local launch_log="${LOG_DIR}/${eval_name}.launch.log"
    local clean_log="${LOG_DIR}/${EXPERIMENT_NAME}_step${step}_delta69_clean.launch.log"
    local ckpt_dir="${CKPT_ROOT}/global_step_${step}"

    mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

    nohup bash -lc "
        set -euo pipefail
        wait_checkpoint_ready() {
            local ckpt_dir=\"\$1\"
            local max_tries=\"\${2:-720}\"
            local i
            for ((i=1; i<=max_tries; i++)); do
                if [[ -f \"\$ckpt_dir/data.pt\" && -f \"\$ckpt_dir/actor/fsdp_config.json\" && -f \"\$ckpt_dir/actor/huggingface/config.json\" ]]; then
                    return 0
                fi
                sleep 5
            done
            echo \"checkpoint not ready: \$ckpt_dir\" >&2
            return 1
        }

        wait_checkpoint_ready '$ckpt_dir'

        env \
            ROOT='$ROOT' \
            GPU_DEVICE='$gpu' \
            VLLM_PORT='$vllm_port' \
            VLLM_INTERNAL_PORT_BASE='$internal_port' \
            SANDBOX_URL='$sandbox_url' \
            MAX_CONCURRENT='$MAX_CONCURRENT' \
            MAX_CONCURRENT_JUDGES='$MAX_CONCURRENT_JUDGES' \
            VERIFIER_LIMITER_BUDGET='$VERIFIER_LIMITER_BUDGET' \
            BATCH_SIZE='$BATCH_SIZE' \
            CKPT_DIR='$ckpt_dir' \
            EVAL_NAME='$eval_name' \
            LOG_DIR='$LOG_DIR' \
            OUTPUT_BASE='$OUTPUT_BASE' \
            DELTA69_MANIFEST_DIR='$DELTA69_MANIFEST_DIR' \
            '$ROOT/coding_model_project/phase_2_ GRPO/ops/run_grpo_codecontests_delta69_checkpoint.sh' \
            >'$launch_log' 2>&1

        python3 '$ROOT/coding_model_project/src/compute_eval_clean_overlay.py' \
            --per-problem-jsonl '$out_dir/per_problem/codecontests_valid_big.jsonl' \
            --quarantine-json '$QUARANTINE_JSON' \
            --output-json '$out_dir/clean_overlay_problem_quarantine_v3.json' \
            >'$clean_log' 2>&1
    " >/dev/null 2>&1 < /dev/null &

    echo "launched step=${step} gpu=${gpu} sandbox=${sandbox_url} log=${launch_log}"
}

launch_one 980 0 8120 8320 http://localhost:8091
launch_one 1000 1 8121 8321 http://localhost:8092
