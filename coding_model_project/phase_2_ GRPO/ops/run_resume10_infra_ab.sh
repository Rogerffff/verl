#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}

WAIT_FOR_EXPERIMENT=${WAIT_FOR_EXPERIMENT:-""}
BASE_CKPT_DIR=${BASE_CKPT_DIR:-"/root/verl/checkpoints/rlvr_coding_model/grpo_a1_final_readiness_lb12_micro4_minib8_seed0"}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-"$BASE_CKPT_DIR/global_step_10"}
RESULT_JSON=${RESULT_JSON:-"/root/resume10_infra_ab_results.json"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-30}

PRELIGHT_TOTAL_STEPS=${PRELIGHT_TOTAL_STEPS:-12}
CONFIRM_TOTAL_STEPS=${CONFIRM_TOTAL_STEPS:-14}

run_case() {
    local name="$1"
    local sandbox_url="$2"
    local limiter_budget="$3"
    local total_training_steps="$4"
    local log_path="/root/${name}.log"
    local ckpt_dir="/root/verl/checkpoints/rlvr_coding_model/${name}"
    local validation_dir="/root/verl/validation_dumps/${name}"

    rm -rf "$ckpt_dir" "$validation_dir"

    (
        cd "$PROJECT_ROOT"
        SANDBOX_URL="$sandbox_url" \
        LIMITER_BUDGET="$limiter_budget" \
        TOTAL_TRAINING_STEPS="$total_training_steps" \
        TEST_FREQ=0 \
        SAVE_FREQ=0 \
        VAL_BEFORE_TRAIN=False \
        LOG_VAL_GENERATIONS=0 \
        ACTOR_OFFLOAD_POLICY=True \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
        PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
        PPO_MICRO_BATCH_SIZE_PER_GPU="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
        ROLLOUT_N="$ROLLOUT_N" \
        RUN_TIMEOUT_S="$RUN_TIMEOUT_S" \
        EXPERIMENT_NAME="$name" \
        BASE_CKPT_DIR="$BASE_CKPT_DIR" \
        RESUME_FROM_PATH="$RESUME_FROM_PATH" \
        CKPT_DIR="$ckpt_dir" \
        VALIDATION_DATA_DIR="$validation_dir" \
        bash "$SCRIPT_DIR/run_grpo_a1_integrated_final_pilot.sh" >"$log_path" 2>&1
    )
}

parse_runs() {
    python3 - <<'PY'
import json
import os
import re
import statistics

runs = {
    "ab_single_lb12_s2": "/root/grpo_a1_resume10_ab_single_lb12_s2.log",
    "ab_single_lb20_s2": "/root/grpo_a1_resume10_ab_single_lb20_s2.log",
    "ab_multi2_lb20_s2": "/root/grpo_a1_resume10_ab_multi2_lb20_s2.log",
    "ab_multi2_lb24_s2": "/root/grpo_a1_resume10_ab_multi2_lb24_s2.log",
    "ab_confirm": "/root/grpo_a1_resume10_ab_confirm.log",
}
keys = [
    "timing_s/reward",
    "timing_s/update_actor",
    "timing_s/step",
    "verifier/judge_time_s_p95",
    "verifier/timeout_rate",
    "verifier/reward_raw_valid_rate",
    "verifier/invalid_for_rl_rate",
    "verifier/truncated_by_max_tokens_rate",
    "response_length/mean",
]
result = {}
for name, path in runs.items():
    if not os.path.exists(path):
        continue
    rows = []
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            match = re.search(r"step:(\d+)", line)
            if not match:
                continue
            row = {"step": int(match.group(1))}
            for key in keys:
                value_match = re.search(re.escape(key) + r":([^\s]+)", line)
                row[key] = float(value_match.group(1)) if value_match else None
            rows.append(row)
    if not rows:
        continue
    result[name] = {
        "rows": rows,
        "reward_median": statistics.median(
            [row["timing_s/reward"] for row in rows if row["timing_s/reward"] is not None]
        ),
        "step_median": statistics.median(
            [row["timing_s/step"] for row in rows if row["timing_s/step"] is not None]
        ),
        "update_actor_median": statistics.median(
            [row["timing_s/update_actor"] for row in rows if row["timing_s/update_actor"] is not None]
        ),
        "timeout_median": statistics.median(
            [row["verifier/timeout_rate"] for row in rows if row["verifier/timeout_rate"] is not None]
        ),
    }

with open(os.environ["RESULT_JSON"], "w") as handle:
    json.dump(result, handle, indent=2)

print(json.dumps(result, indent=2))
PY
}

choose_best() {
    python3 - <<'PY'
import json
import os

path = os.environ["RESULT_JSON"]
if not os.path.exists(path):
    raise SystemExit(0)

result = json.load(open(path))
candidates = []
for name in ("ab_single_lb12_s2", "ab_single_lb20_s2", "ab_multi2_lb20_s2", "ab_multi2_lb24_s2"):
    summary = result.get(name)
    if not summary:
        continue
    rows = summary["rows"]
    valid = True
    for row in rows:
        if row["verifier/reward_raw_valid_rate"] is not None and row["verifier/reward_raw_valid_rate"] < 0.98:
            valid = False
        if row["verifier/invalid_for_rl_rate"] is not None and row["verifier/invalid_for_rl_rate"] > 0.02:
            valid = False
        if (
            row["verifier/truncated_by_max_tokens_rate"] is not None
            and row["verifier/truncated_by_max_tokens_rate"] > 0.02
        ):
            valid = False
    if valid:
        candidates.append((summary["reward_median"], name))

if candidates:
    candidates.sort()
    print(candidates[0][1])
PY
}

if [[ -n "$WAIT_FOR_EXPERIMENT" ]]; then
    while pgrep -f "trainer.experiment_name=${WAIT_FOR_EXPERIMENT}" >/dev/null; do
        sleep 30
    done
fi

run_case "grpo_a1_resume10_ab_single_lb12_s2" "http://localhost:8081" 12 "$PRELIGHT_TOTAL_STEPS"
run_case "grpo_a1_resume10_ab_single_lb20_s2" "http://localhost:8081" 20 "$PRELIGHT_TOTAL_STEPS"
run_case "grpo_a1_resume10_ab_multi2_lb20_s2" "http://localhost:8090" 20 "$PRELIGHT_TOTAL_STEPS"
run_case "grpo_a1_resume10_ab_multi2_lb24_s2" "http://localhost:8090" 24 "$PRELIGHT_TOTAL_STEPS"

parse_runs

BEST="$(choose_best || true)"
echo "BEST_2STEP=${BEST}"

if [[ -n "$BEST" && "$BEST" != "ab_multi2_lb24_s2" ]]; then
    case "$BEST" in
        ab_single_lb12_s2)
            run_case "grpo_a1_resume10_ab_confirm" "http://localhost:8081" 12 "$CONFIRM_TOTAL_STEPS"
            ;;
        ab_single_lb20_s2)
            run_case "grpo_a1_resume10_ab_confirm" "http://localhost:8081" 20 "$CONFIRM_TOTAL_STEPS"
            ;;
        ab_multi2_lb20_s2)
            run_case "grpo_a1_resume10_ab_confirm" "http://localhost:8090" 20 "$CONFIRM_TOTAL_STEPS"
            ;;
    esac
    parse_runs
fi
