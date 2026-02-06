#!/bin/bash
# =============================================================================
# Phase 0 Baseline 评测脚本
# =============================================================================
# 用途：运行 Phase 0 基线评测（假设 vLLM 和 SandboxFusion 已经启动）
#
# 前置条件：
#   1. vLLM 服务已启动在 localhost:8000
#   2. SandboxFusion 服务已启动在 localhost:8080
#
# 使用方式：
#   chmod +x scripts/run_phase0.sh
#   ./scripts/run_phase0.sh
# =============================================================================

set -e

# 项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# =============================================================================
# 评测参数配置（来自 eval_config.py，所有 Phase 保持一致）
# =============================================================================

# 模型配置
MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"

# 服务地址
VLLM_URL="http://localhost:8001"
SANDBOX_URL="http://localhost:8080"

# 解码参数（EVAL@1 协议）
TEMPERATURE=0.0
MAX_TOKENS=2048

# SandboxFusion 配置
RUN_TIMEOUT=30
# memory_limit_mb=1024 在脚本中不需要传，代码里有默认值

# 并发配置（单卡 4090）
MAX_CONCURRENT=32
BATCH_SIZE=50

# 数据集
DATASETS="humaneval mbpp_reg codecontests_valid"

# 输出目录
OUTPUT_DIR="outputs/phase0_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# 运行评测
# =============================================================================

echo "============================================================"
echo "   Phase 0 Baseline Evaluation"
echo "============================================================"
echo ""
echo "Model:        $MODEL_NAME"
echo "Datasets:     $DATASETS"
echo "Output:       $OUTPUT_DIR"
echo ""
echo "Parameters (from eval_config.py):"
echo "  temperature:    $TEMPERATURE"
echo "  max_tokens:     $MAX_TOKENS"
echo "  run_timeout:    ${RUN_TIMEOUT}s"
echo "  max_concurrent: $MAX_CONCURRENT"
echo ""

# 检查服务
echo "Checking services..."
if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM service not available at $VLLM_URL"
    echo "Please start vLLM first."
    exit 1
fi
echo "  vLLM: OK"

if ! curl -s "${SANDBOX_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: SandboxFusion service not available at $SANDBOX_URL"
    echo "Please start SandboxFusion first."
    exit 1
fi
echo "  SandboxFusion: OK"
echo ""

# 检查 manifest 目录
MANIFEST_ARG=""
if [ -d "data/manifests" ] && [ "$(ls -A data/manifests/*.jsonl 2>/dev/null)" ]; then
    MANIFEST_ARG="--manifest_dir data/manifests"
    echo "Using manifest files from data/manifests/"
else
    echo "No manifest files found, will use SandboxFusion built-in data"
fi
echo ""

# 运行评测
echo "Starting evaluation..."
echo ""

python src/phase0_eval.py \
    --mode simple \
    --model "$MODEL_NAME" \
    --vllm_url "$VLLM_URL" \
    --sandbox_url "$SANDBOX_URL" \
    $MANIFEST_ARG \
    --datasets $DATASETS \
    --temperature $TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --run_timeout $RUN_TIMEOUT \
    --max_concurrent $MAX_CONCURRENT \
    --batch_size $BATCH_SIZE \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "   Evaluation Complete"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
