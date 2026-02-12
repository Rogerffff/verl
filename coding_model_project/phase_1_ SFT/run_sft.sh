#!/bin/bash
# =============================================================================
# Phase 1 SFT Training Script
# =============================================================================
#
# Usage:
#   bash run_sft.sh [NPROC_PER_NODE]
#
# Example:
#   bash run_sft.sh 4        # 4x 5090 (recommended)
#   bash run_sft.sh 2        # 2x 6000 Pro
#   bash run_sft.sh 8        # 8x 4090
#
# Prerequisites:
#   1. Run prepare_sft_data.py to generate sft_train.parquet and sft_val.parquet
#   2. Ensure verl is installed (pip install -e .)
#   3. wandb login (if using wandb logging)
#
# =============================================================================
set -euxo pipefail

NPROC_PER_NODE=${1:-4}

# --- Paths ---
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
VERL_DIR=$(cd "${PROJECT_DIR}/.." && pwd)
DATA_DIR="${SCRIPT_DIR}/data"
SAVE_DIR="${SCRIPT_DIR}/checkpoints"

# --- Validate data files exist ---
if [ ! -f "${DATA_DIR}/sft_train.parquet" ]; then
    echo "Error: ${DATA_DIR}/sft_train.parquet not found."
    echo "Run: python prepare_sft_data.py --input_jsonl ... --output_dir data/"
    exit 1
fi

# --- Training ---
echo "============================================"
echo "Phase 1 SFT Training"
echo "  GPUs: ${NPROC_PER_NODE}"
echo "  Data: ${DATA_DIR}"
echo "  Save: ${SAVE_DIR}"
echo "============================================"

cd "${VERL_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
    -m verl.trainer.fsdp_sft_trainer \
    "data.train_files=${DATA_DIR}/sft_train.parquet" \
    "data.val_files=${DATA_DIR}/sft_val.parquet" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=4096 \
    data.truncation=right \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    data.balance_dp_token=True \
    model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.strategy=fsdp2 \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=2e-5 \
    'optim.betas=[0.9,0.95]' \
    optim.weight_decay=0.01 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    "trainer.default_local_dir=${SAVE_DIR}" \
    trainer.project_name=rlvr_coding_model \
    trainer.experiment_name=phase1_sft_qwen7b_coder \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.n_gpus_per_node=${NPROC_PER_NODE} \
    trainer.max_ckpt_to_keep=5 \
    'trainer.logger=["console","wandb"]' \
    trainer.seed=42 \
    trainer.resume_mode=auto \
    'trainer.checkpoint.save_contents=["model","optimizer","extra","hf_model"]'

echo "Training complete!"
echo "Checkpoints saved to: ${SAVE_DIR}"
