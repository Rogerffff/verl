#!/bin/bash
# =============================================================================
# Phase 1 SFT - Smoke Test (10 steps, 1 GPU)
# =============================================================================
#
# Quick validation that the entire pipeline works:
#   - Data loading + tokenization
#   - Model loading + FSDP
#   - Forward + backward pass
#   - Checkpoint saving (with HF model export)
#
# Usage:
#   bash run_smoke_test.sh
#
# Expected:
#   - Loss decreases over 10 steps
#   - Checkpoint saved at step 5 and 10
#   - huggingface/ directory created in checkpoint
#
# =============================================================================
set -euxo pipefail

# --- Paths ---
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
VERL_DIR=$(cd "${PROJECT_DIR}/.." && pwd)
DATA_DIR="${SCRIPT_DIR}/data"
SAVE_DIR="${SCRIPT_DIR}/checkpoints/smoke_test"

# --- Validate data files exist ---
if [ ! -f "${DATA_DIR}/sft_train.parquet" ]; then
    echo "Error: ${DATA_DIR}/sft_train.parquet not found."
    echo "Run: python prepare_sft_data.py --input_jsonl ... --output_dir data/"
    exit 1
fi

# --- Clean previous smoke test ---
rm -rf "${SAVE_DIR}"

echo "============================================"
echo "Phase 1 SFT - Smoke Test"
echo "  1 GPU, 10 steps"
echo "  Data: ${DATA_DIR}"
echo "  Save: ${SAVE_DIR}"
echo "============================================"

cd "${VERL_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    "data.train_files=${DATA_DIR}/sft_train.parquet" \
    "data.val_files=${DATA_DIR}/sft_val.parquet" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=4096 \
    data.truncation=right \
    data.train_batch_size=4 \
    data.micro_batch_size_per_gpu=2 \
    data.balance_dp_token=False \
    model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.strategy=fsdp2 \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=2e-5 \
    'optim.betas=[0.9,0.95]' \
    optim.weight_decay=0.01 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    "trainer.default_local_dir=${SAVE_DIR}" \
    trainer.project_name=rlvr_coding_model \
    trainer.experiment_name=smoke_test \
    trainer.total_training_steps=10 \
    trainer.total_epochs=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.n_gpus_per_node=1 \
    'trainer.logger=["console"]' \
    trainer.seed=42 \
    'trainer.checkpoint.save_contents=["model","optimizer","extra","hf_model"]'

# --- Verification ---
echo ""
echo "============================================"
echo "Smoke Test Verification"
echo "============================================"

# Check checkpoint exists
if ls "${SAVE_DIR}"/rlvr_coding_model/smoke_test/global_step_*/huggingface/config.json 1>/dev/null 2>&1; then
    echo "  [PASS] HF model checkpoint found"
    ls -la "${SAVE_DIR}"/rlvr_coding_model/smoke_test/global_step_*/huggingface/config.json
else
    echo "  [FAIL] HF model checkpoint NOT found"
    echo "  Listing checkpoint directory:"
    find "${SAVE_DIR}" -maxdepth 4 -type d 2>/dev/null || true
    exit 1
fi

echo ""
echo "Smoke test passed! Ready for full training."
