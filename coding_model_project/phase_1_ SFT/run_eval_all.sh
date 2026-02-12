#!/bin/bash
# =============================================================================
# Phase 1 SFT - Batch Evaluation of All Checkpoints
# =============================================================================
#
# Evaluate all saved checkpoints sequentially.
# For each checkpoint:
#   1. Start vLLM server with the HF model
#   2. Run phase1_eval.py with appropriate tier
#   3. Shut down vLLM server
#   4. Move to next checkpoint
#
# After all evaluations, print a comparison table.
#
# Usage:
#   bash run_eval_all.sh [CHECKPOINT_BASE_DIR] [SANDBOX_URL]
#
# Example:
#   bash run_eval_all.sh \
#       checkpoints/rlvr_coding_model/phase1_sft_qwen7b_coder \
#       http://localhost:8080
#
# The script auto-detects global_step_* directories and evaluates them
# in ascending order. The final step gets Tier 3 (full), others get Tier 1.
#
# =============================================================================
set -euo pipefail

# --- Arguments ---
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CHECKPOINT_BASE=${1:-"${SCRIPT_DIR}/checkpoints/rlvr_coding_model/phase1_sft_qwen7b_coder"}
SANDBOX_URL=${2:-"http://localhost:8080"}
OUTPUT_BASE="${SCRIPT_DIR}/outputs/phase1"
VLLM_PORT=8000
TP_SIZE=1

# --- Validate ---
if [ ! -d "${CHECKPOINT_BASE}" ]; then
    echo "Error: Checkpoint base directory not found: ${CHECKPOINT_BASE}"
    echo "Usage: bash run_eval_all.sh <checkpoint_base_dir> [sandbox_url]"
    exit 1
fi

# --- Find all checkpoint directories, sorted by step number ---
CKPT_DIRS=()
while IFS=$'\t' read -r step dir; do
    CKPT_DIRS+=("$dir")
done < <(
    find "${CHECKPOINT_BASE}" -maxdepth 1 -type d -name "global_step_*" \
        | awk -F'global_step_' 'NF > 1 {step=$NF; if (step ~ /^[0-9]+$/) printf "%s\t%s\n", step, $0}' \
        | sort -n -k1,1
)

if [ ${#CKPT_DIRS[@]} -eq 0 ]; then
    echo "Error: No global_step_* directories found in ${CHECKPOINT_BASE}"
    exit 1
fi

echo "============================================"
echo "Phase 1 SFT - Batch Evaluation"
echo "============================================"
echo "  Checkpoint base: ${CHECKPOINT_BASE}"
echo "  Found ${#CKPT_DIRS[@]} checkpoints:"
for dir in "${CKPT_DIRS[@]}"; do
    echo "    - $(basename "$dir")"
done
echo "  Sandbox: ${SANDBOX_URL}"
echo "  Output:  ${OUTPUT_BASE}"
echo "============================================"
echo ""

# --- Determine the final step (for Tier 3 evaluation) ---
LAST_IDX=$((${#CKPT_DIRS[@]} - 1))
LAST_CKPT="${CKPT_DIRS[$LAST_IDX]}"
LAST_STEP=$(basename "$LAST_CKPT" | sed 's/global_step_//')

mkdir -p "${OUTPUT_BASE}"

# --- Evaluate each checkpoint ---
FAILED=()
for ckpt_dir in "${CKPT_DIRS[@]}"; do
    step_name=$(basename "$ckpt_dir")
    step_num=$(echo "$step_name" | sed 's/global_step_//')

    # Determine tier: final step gets Tier 3, others get Tier 1
    if [ "$step_num" == "$LAST_STEP" ]; then
        TIER=3
    else
        TIER=1
    fi

    echo ""
    echo "============================================"
    echo "  Evaluating: ${step_name} (Tier ${TIER})"
    echo "============================================"

    if python "${SCRIPT_DIR}/phase1_eval.py" \
        --checkpoint_dir "${ckpt_dir}" \
        --tier ${TIER} \
        --output_base "${OUTPUT_BASE}" \
        --sandbox_url "${SANDBOX_URL}" \
        --vllm_port ${VLLM_PORT} \
        --tensor_parallel_size ${TP_SIZE}; then
        echo "  [PASS] ${step_name} evaluation complete"
    else
        echo "  [FAIL] ${step_name} evaluation failed"
        FAILED+=("${step_name}")
    fi
done

# --- Print comparison table ---
echo ""
echo "============================================"
echo "  Evaluation Results Comparison"
echo "============================================"
echo ""

# Read eval_history.jsonl and format as a table
HISTORY_FILE="${OUTPUT_BASE}/eval_history.jsonl"
if [ -f "${HISTORY_FILE}" ]; then
    echo "Step | codecontests_valid (exec_success) | codecontests_valid (accepted@1) | mbpp_reg (accepted@1)"
    echo "-----|----------------------------------|-------------------------------|---------------------"
    while IFS= read -r line; do
        step=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('step','?'))" 2>/dev/null || echo "?")
        cc_exec=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('scores',{}).get('codecontests_valid',{}); v=s.get('exec_success_rate'); print(f'{v:.2%}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        cc_score=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('scores',{}).get('codecontests_valid',{}); v=s.get('accepted_at_1'); print(f'{v:.2%}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        mbpp_score=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('scores',{}).get('mbpp_reg',{}); v=s.get('accepted_at_1'); print(f'{v:.2%}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        printf "%-5s| %-34s| %-32s| %s\n" "$step" "$cc_exec" "$cc_score" "$mbpp_score"
    done < "${HISTORY_FILE}"
else
    echo "  (eval_history.jsonl not found)"
fi

BEST_FILE="${OUTPUT_BASE}/best_checkpoint.json"
if [ -f "${BEST_FILE}" ]; then
    echo ""
    echo "Best checkpoint (by codecontests_valid exec_success_rate):"
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); e=d.get('best_exec_success_rate'); a=d.get('best_accepted_at_1'); ef=f'{e:.4f}' if isinstance(e,(int,float)) else 'N/A'; af=f'{a:.4f}' if isinstance(a,(int,float)) else 'N/A'; print(f\"  step={d.get('best_step')} exec_success_rate={ef} accepted@1={af}\")" "${BEST_FILE}" \
        || true
fi

echo ""

# --- Summary ---
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "WARNING: ${#FAILED[@]} checkpoint(s) failed evaluation:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}"
    done
    exit 1
else
    echo "All ${#CKPT_DIRS[@]} checkpoints evaluated successfully!"
    echo "Results saved to: ${OUTPUT_BASE}"
    echo ""
    echo "To compare in detail:"
    echo "  cat ${OUTPUT_BASE}/eval_history.jsonl | python3 -m json.tool"
fi
