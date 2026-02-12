# Phase 1 SFT - Step 4: Checkpoint Evaluation Pipeline

> 三层评测架构、指标定义、QA Log 规范、评测自动化脚本

---

## 1. 为什么需要独立的评测流水线

verl `FSDPSFTTrainer` 只计算 **validation loss**（交叉熵），不会：
- 生成代码（greedy decoding）
- 运行代码（SandboxFusion 执行）
- 计算通过率（exec_success_rate, accepted@1, pass_ratio）
- 分析错误类型（syntax_error, runtime_error, timeout, wrong_answer）

Phase 1 的核心目标是降低 syntax/runtime/timeout 错误。这些指标必须通过**实际生成代码 + 沙箱执行**来测量。因此需要独立的 Checkpoint Evaluation Pipeline。

---

## 2. 三层评测架构

### 2.1 层级定义

| Tier | 数据集 | 大小 | 频率 | WandB Key 模式 | 目的 |
|------|--------|------|------|----------------|------|
| **Tier 1** | codecontests_valid | 117 题 | 每 500 步（每个 checkpoint） | `eval/codecontests_valid/*` | 快速反馈，主要选模指标 |
| **Tier 1** | MBPP_reg | 200 题 | 每 500 步（每个 checkpoint） | `eval/mbpp_reg/*` | 回归检测 |
| **Tier 2** | codecontests_valid_big | 500 题 | 每 2000 步 | `eval/codecontests_valid_big/*` | 更可靠的信号，低频节省算力 |
| **Tier 3** | codecontests_test | 165 题 | 仅 Phase 结束 | `eval/codecontests_test/*` | 最终测试，一次性 |
| **Tier 3** | HumanEval | 164 题 | 仅 Phase 结束 | `eval/humaneval/*` | 行业基准回归检测 |

### 2.2 层级执行时机

假设总训练步数约 1500，save_freq=500:

```
Step 500:   Tier 1 (valid=117 + MBPP=200)
Step 1000:  Tier 1 (valid=117 + MBPP=200)
Step 1500:  Tier 1 (valid=117 + MBPP=200)
Step 2000:  Tier 1 + Tier 2 (valid_big=500)  ← 如果训练超过 2000 步
Final:      Tier 1 + Tier 2 + Tier 3 (test=165 + HumanEval=164)
```

> **Tier 2 频率**: 每 2000 步（即每 4 个 Tier 1 eval 才跑一次 Tier 2）。500 题的评测耗时是 117 题的约 4 倍，降低频率可以显著节省 GPU 时间。

### 2.3 数据集来源

| 数据集 | Manifest 文件 | 说明 |
|--------|-------------|------|
| codecontests_valid | `data/manifests/codecontests_valid_manifest.jsonl` | 117 个 problem_id |
| codecontests_valid_big | `data/manifests/codecontests_valid_big_manifest.jsonl` | 500 个从训练集拆分的 problem_id |
| codecontests_test | `data/manifests/codecontests_test_manifest.jsonl` | 165 个 test-only problem_id |
| MBPP_reg | 内置于 `src/eval_config.py` DATASET_CONFIGS | id_range=(11, 210)，200 题 |
| HumanEval | 内置于 `src/eval_config.py` DATASET_CONFIGS | 164 题 |

---

## 3. 评测流程（每个 Checkpoint）

### 3.1 端到端流程

```
phase1_eval.py --checkpoint_dir <path> --step <N> --datasets <dataset...>

  ├─ [1] 定位 HuggingFace 模型
  │     checkpoint_dir/global_step_N/huggingface/
  │     → 验证 config.json, model.safetensors 存在
  │
  ├─ [2] 启动 vLLM 推理服务
  │     vllm serve <model_path> \
  │       --dtype bfloat16 \
  │       --max-model-len 6144 \
  │       --gpu-memory-utilization 0.85 \
  │       --tensor-parallel-size 1 \
  │       --port 8000
  │     → 等待服务就绪 (健康检查)
  │
  ├─ [3] 加载评测问题
  │     根据 tier 选择数据集:
  │     - Tier 1: codecontests_valid (117) + MBPP_reg (200)
  │     - Tier 2: codecontests_valid_big (500)
  │     - Tier 3: codecontests_test (165) + HumanEval (164)
  │
  ├─ [4] 生成代码 (EVAL@1 协议)
  │     参数来源: src/eval_config.py → EVAL_CONSTANTS
  │     - temperature: 0.0 (greedy decoding)
  │     - top_p: 1.0
  │     - max_new_tokens: 2048
  │     使用 aiohttp 调用 vLLM OpenAI-compatible API
  │
  ├─ [5] 沙箱执行 (SandboxFusion)
  │     对每个问题的生成代码:
  │     - 复用 `phase0_eval.py::evaluate_with_run_code()`
  │     - 通过 `run_code(RunCodeRequest(...))` 手动传入测试用例
  │       * HumanEval: `test_code + check(entry_point)`
  │       * MBPP: `test_setup_code + assert 列表`
  │       * CodeContests: 逐 testcase 传入 `stdin`，比较 `stdout`
  │     - run_timeout: 30s (from EVAL_CONSTANTS)
  │     - memory_limit_mb: 1024 (from EVAL_CONSTANTS)
  │     - 解析字段: `result.status` / `result.run_result.status` / `return_code`
  │
  ├─ [6] 收集指标
  │     使用 MetricsCollector (src/utils/metrics.py)
  │     → 见 Section 4 完整指标列表
  │
  ├─ [7] 记录到 WandB
  │     使用 wandb.log()
  │     - 质量/成本指标: `eval/{dataset}/{metric}`
  │     - step=global_step
  │
  ├─ [8] 保存 QA 日志
  │     使用 QALogger (src/utils/qa_logger.py)
  │     → 见 Section 5 QA 日志规范
  │
  └─ [9] 更新最佳 checkpoint 记录
        if summary.datasets.codecontests_valid.exec_success_rate is best:
            update best_checkpoint.json
```

### 3.2 Phase 0 复用组件

| 组件 | 文件路径 | 类/函数 | Phase 1 适配 |
|------|---------|---------|------------|
| 指标计算 | `src/utils/metrics.py` | `MetricsCollector`, `EvalResult`, `DatasetMetrics` | 直接复用 |
| QA 日志 | `src/utils/qa_logger.py` | `QALogger`, `QALogEntry` | 直接复用 |
| 配置常量 | `src/eval_config.py` | `EVAL_CONSTANTS`, `DATASET_CONFIGS`, `get_sampling_params()` | 直接复用 |
| 代码生成 | `src/phase0_eval.py` | `generate_code()` 异步函数 | 适配 checkpoint 路径 |
| 沙箱评测 | `src/phase0_eval.py` | `evaluate_with_run_code()` + `run_code()` | 直接复用 |
| Prompt 模板 | `src/phase0_eval.py` | `SYSTEM_PROMPT`, `PROMPT_TEMPLATES` | 直接复用 |

**主要修改**:
1. `phase0_eval.py` 中 vLLM 服务器连接固定模型路径，Phase 1 需要改为 checkpoint 路径参数。
2. `phase1_eval.py` 评测时强制传 `--manifest_dir`，确保走外部 `test_cases + run_code` 路径。

---

## 4. 指标定义（完整列表）

### 4.1 `metrics.json` + WandB（按数据集打平）

`phase0_eval.py` 当前按数据集分别输出，并在 WandB 记录为 `eval/{dataset}/{metric}`。

| 指标名 | WandB Key 示例（codecontests_valid） | 来源 |
|--------|--------------------------------------|------|
| accepted_at_1 | `eval/codecontests_valid/accepted_at_1` | `evaluate_dataset()` |
| pass_ratio_mean | `eval/codecontests_valid/pass_ratio_mean` | `evaluate_dataset()` |
| pass_ratio_p50 | `eval/codecontests_valid/pass_ratio_p50` | `evaluate_dataset()` |
| pass_ratio_p90 | `eval/codecontests_valid/pass_ratio_p90` | `evaluate_dataset()` |
| avg_gen_tokens | `eval/codecontests_valid/avg_gen_tokens` | `evaluate_dataset()` |
| avg_judge_time | `eval/codecontests_valid/avg_judge_time` | `evaluate_dataset()` |
| throughput | `eval/codecontests_valid/throughput` | `evaluate_dataset()` |
| truncation_rate | `eval/codecontests_valid/truncation_rate` | `evaluate_dataset()` |
| timeout_rate | `eval/codecontests_valid/timeout_rate` | `evaluate_dataset()` |

### 4.2 `summary.json`（用于选模的错误分布与可执行率）

`MetricsCollector.get_summary()` 提供每个数据集的聚合字段，包含 `exec_success_rate` 与错误分布。

| 字段路径（summary.json） | 定义 |
|-------------------------|------|
| `datasets.<dataset>.exec_success_rate` | `error_type ∈ {success, wrong_answer}` 比例 |
| `datasets.<dataset>.syntax_error_rate` | 语法错误占比 |
| `datasets.<dataset>.runtime_error_rate` | 运行时错误占比 |
| `datasets.<dataset>.timeout_rate` | 超时占比 |
| `datasets.<dataset>.wrong_answer_rate` | WA 占比 |
| `datasets.<dataset>.api_error_rate` | API 失败占比 |

### 4.3 主选模指标（固定）

主指标使用：

`summary["datasets"]["codecontests_valid"]["exec_success_rate"]`

这是 Phase 1 的目标指标（先提高可运行性，再追求最终正确率）。

### 4.4 `run_code` 结果映射约定

`evaluate_with_run_code()` 的实现不是读取 `final_status/pass_ratio/judge_time` 顶层字段，而是：

1. 读取 `result.status`（RunStatus，`Success/Failed/...`）。
2. 读取 `result.run_result.status` 与 `return_code`。
3. CodeContests 按 testcase 比较 `stdout` 得到 `pass_ratio`。
4. HumanEval/MBPP 通过断言失败、SyntaxError、超时等推导 `error_type`。

### 4.5 MetricsCollector 使用方式

```python
from utils.metrics import MetricsCollector, EvalResult

collector = MetricsCollector()

for item in per_problem_results:
    collector.add_result(
        dataset=item["dataset"],
        result=EvalResult(
            problem_id=item["problem_id"],
            accepted=item["accepted"],
            pass_ratio=item["pass_ratio"],
            error_type=item["error_type"],
            judge_time=item["judge_time"],
            gen_tokens=item["gen_tokens"],
            gen_time=item["gen_time"],
        ),
    )

collector.set_wall_clock_time("codecontests_valid", wall_clock_seconds)
summary = collector.get_summary()
exec_rate = summary["datasets"]["codecontests_valid"]["exec_success_rate"]
```

---

## 5. QA Log 规范

### 5.1 采样策略

按照 `final_experiment_design.md` 的要求，使用分层抽样：

| 评测时机 | 数据集 | 采样数 | 说明 |
|---------|--------|--------|------|
| 每 500 步 (Tier 1) | codecontests_valid | 30 | 5 success + 5 WA + 5 syntax + 5 runtime + 5 timeout + 5 random |
| 每 500 步 (Tier 1) | MBPP_reg | 20 | 按错误类型分层 |
| 每 2000 步 (Tier 2) | codecontests_valid_big | 50 | 按错误类型分层 |
| Phase 结束 (Tier 3) | codecontests_test | 50 | 按错误类型分层 |
| Phase 结束 (Tier 3) | HumanEval | 30 | 按错误类型分层 |

### 5.2 QA Log 格式

```json
{
    "dataset": "codecontests_valid",
    "problem_id": "Codeforces/1575/G",
    "prompt": "In Python3, your task is to solve...",
    "response": "import sys\nfrom collections import deque\n...",
    "accepted": false,
    "pass_ratio": 0.3,
    "error_type": "wrong_answer",
    "judge_time": 2.5,
    "gen_time": 1.2,
    "gen_tokens": 350,
    "extra": {
        "step": 500,
        "checkpoint": "global_step_500",
        "model": "phase1_sft_qwen7b_coder"
    }
}
```

### 5.3 QA Log 存储路径

```text
coding_model_project/phase_1_ SFT/outputs/
  ├── eval_step_500/
  │   ├── metrics.json
  │   ├── summary.json
  │   └── qa_logs/
  │       ├── codecontests_valid_qa.jsonl
  │       ├── mbpp_reg_qa.jsonl
  │       └── qa_summary.json
  ├── eval_step_1000/
  │   └── ...
  └── eval_final/
      ├── metrics.json
      ├── summary.json
      ├── qa_logs/
      │   ├── codecontests_valid_qa.jsonl
      │   ├── codecontests_valid_big_qa.jsonl
      │   ├── codecontests_test_qa.jsonl
      │   ├── humaneval_qa.jsonl
      │   └── qa_summary.json
      └── best_checkpoint.json
```

### 5.4 QALogger 使用方式

```python
from pathlib import Path
from utils.qa_logger import QALogger

logger = QALogger(
    output_dir=Path("outputs/eval_step_500/qa_logs"),
    sample_size=30,
    min_per_error_type=5,
)

# 记录每个问题的 QA 日志
for problem, gen_result, eval_result in results:
    logger.log(
        dataset="codecontests_valid",
        problem_id=problem.id,
        prompt=problem.prompt,
        response=gen_result.code,
        eval_result=eval_result,
        gen_metadata={
            "gen_time": gen_result.elapsed_time,
            "completion_tokens": gen_result.token_count,
            "finish_reason": gen_result.finish_reason,
        },
    )

# 保存（自动分层抽样）
logger.save()
```

---

## 6. 评测自动化

### 6.1 `phase1_eval.py` — 单 Checkpoint 评测脚本

```bash
PHASE1_DIR="coding_model_project/phase_1_ SFT"
CKPT_BASE="$PHASE1_DIR/checkpoints/rlvr_coding_model/phase1_sft_qwen7b_coder"
MANIFEST_DIR="coding_model_project/data/manifests"

# Tier 1 评测（每个 checkpoint 都跑）
python "$PHASE1_DIR/phase1_eval.py" \
    --checkpoint_dir "$CKPT_BASE" \
    --step 500 \
    --datasets codecontests_valid mbpp_reg \
    --manifest_dir "$MANIFEST_DIR" \
    --use_external_tests \
    --no_submit_api \
    --output_dir "$PHASE1_DIR/outputs/eval_step_500" \
    --wandb_project rlvr_coding_model \
    --wandb_run_name phase1_eval_step_500

# Tier 2 评测（每 2000 步跑）
python "$PHASE1_DIR/phase1_eval.py" \
    --checkpoint_dir "$CKPT_BASE" \
    --step 2000 \
    --datasets codecontests_valid mbpp_reg codecontests_valid_big \
    --manifest_dir "$MANIFEST_DIR" \
    --use_external_tests \
    --no_submit_api \
    --output_dir "$PHASE1_DIR/outputs/eval_step_2000"

# Tier 3 评测（Phase 结束时跑）
python "$PHASE1_DIR/phase1_eval.py" \
    --checkpoint_dir "$CKPT_BASE" \
    --step best \
    --datasets codecontests_valid codecontests_valid_big codecontests_test humaneval mbpp_reg \
    --manifest_dir "$MANIFEST_DIR" \
    --use_external_tests \
    --no_submit_api \
    --output_dir "$PHASE1_DIR/outputs/eval_final"
```

### 6.2 `run_eval_checkpoints.sh` — 批量评测脚本

```bash
#!/bin/bash
set -e

# ============================================================
# Phase 1 SFT Checkpoint 批量评测脚本
# ============================================================

PHASE1_DIR="coding_model_project/phase_1_ SFT"
CKPT_BASE="$PHASE1_DIR/checkpoints/rlvr_coding_model/phase1_sft_qwen7b_coder"
OUTPUT_BASE="$PHASE1_DIR/outputs"
MANIFEST_DIR="coding_model_project/data/manifests"
SANDBOX_URL="http://localhost:8080"

# 遍历所有 checkpoint
for ckpt_dir in "$CKPT_BASE"/global_step_*; do
    [ -d "$ckpt_dir" ] || continue
    step=$(basename "$ckpt_dir" | sed 's/global_step_//')
    model_path="${ckpt_dir}/huggingface"

    # 检查 HF 模型是否存在
    if [ ! -f "${model_path}/config.json" ]; then
        echo "SKIP: ${ckpt_dir} - no huggingface model"
        continue
    fi

    echo "=========================================="
    echo "Evaluating checkpoint: global_step_${step}"
    echo "=========================================="

    # 启动 vLLM 服务
    python -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" \
        --dtype bfloat16 \
        --max-model-len 6144 \
        --gpu-memory-utilization 0.85 \
        --port 8000 &
    VLLM_PID=$!

    # 等待服务就绪
    echo "Waiting for vLLM server..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM server ready!"
            break
        fi
        sleep 5
    done

    # 决定评测层级
    DATASETS="codecontests_valid mbpp_reg"
    if [ $((step % 2000)) -eq 0 ] && [ $step -gt 0 ]; then
        DATASETS="${DATASETS} codecontests_valid_big"
        echo "  → Including Tier 2 (valid_big)"
    fi

    # 运行评测
    python "$PHASE1_DIR/phase1_eval.py" \
        --checkpoint_dir "${CKPT_BASE}" \
        --step "${step}" \
        --datasets ${DATASETS} \
        --manifest_dir "${MANIFEST_DIR}" \
        --use_external_tests \
        --no_submit_api \
        --output_dir "${OUTPUT_BASE}/eval_step_${step}" \
        --vllm_url http://localhost:8000 \
        --sandbox_url "${SANDBOX_URL}"

    # 停止 vLLM 服务
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
    echo "vLLM server stopped."

    sleep 5  # 等待 GPU 释放
done

echo "All checkpoints evaluated!"
```

### 6.3 vLLM 服务配置

配置来源: `src/eval_config.py` → `SERVICE_CONFIGS["vllm"]`

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dtype` | `bfloat16` | 与训练精度一致 |
| `--max-model-len` | `6144` | 4096 prompt + 2048 generation |
| `--gpu-memory-utilization` | `0.85` | GPU 内存利用率 |
| `--tensor-parallel-size` | `1` | 单卡推理（评测时） |

---

## 7. 最佳 Checkpoint 选择

### 7.1 选择标准

**主指标**: `summary.datasets.codecontests_valid.exec_success_rate` (Tier 1)

```python
best_checkpoint = {
    "step": None,
    "exec_success_rate": 0.0,
    "accepted_at_1": 0.0,
    "val_loss": float('inf'),
}

for step, metrics in all_eval_results.items():
    exec_rate = metrics["summary"]["datasets"]["codecontests_valid"]["exec_success_rate"]
    acc1 = metrics["summary"]["datasets"]["codecontests_valid"]["accepted_at_1"]
    if exec_rate > best_checkpoint["exec_success_rate"]:
        best_checkpoint = {
            "step": step,
            "exec_success_rate": exec_rate,
            "accepted_at_1": acc1,
            "val_loss": metrics.get("val/loss", None),
        }
```

### 7.2 输出

```json
// coding_model_project/phase_1_ SFT/outputs/best_checkpoint.json
{
    "best_step": 1000,
    "best_exec_success_rate": 0.65,
    "best_accepted_at_1": 0.08,
    "model_path": "coding_model_project/phase_1_ SFT/checkpoints/.../global_step_1000/huggingface",
    "selection_metric": "summary.datasets.codecontests_valid.exec_success_rate",
    "all_checkpoints": {
        "500": {"exec_success_rate": 0.55, "accepted_at_1": 0.05},
        "1000": {"exec_success_rate": 0.65, "accepted_at_1": 0.08},
        "1500": {"exec_success_rate": 0.62, "accepted_at_1": 0.07}
    }
}
```

---

## 8. 评测脚本实现规划

### 8.1 `phase1_eval.py` 核心结构

```python
#!/usr/bin/env python3
"""Phase 1 SFT Checkpoint Evaluation Pipeline"""

import argparse, asyncio, json, os, sys, time
from pathlib import Path

# 复用 Phase 0 组件
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.metrics import MetricsCollector, EvalResult
from utils.qa_logger import QALogger
from eval_config import EVAL_CONSTANTS, DATASET_CONFIGS, get_sampling_params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--use_external_tests", dest="use_external_tests", action="store_true", default=True)
    parser.add_argument("--no_external_tests", dest="use_external_tests", action="store_false")
    parser.add_argument("--use_submit_api", dest="use_submit_api", action="store_true", default=False)
    parser.add_argument("--no_submit_api", dest="use_submit_api", action="store_false")
    parser.add_argument("--vllm_url", default="http://localhost:8000")
    parser.add_argument("--sandbox_url", default="http://localhost:8080")
    parser.add_argument("--wandb_project", default="rlvr_coding_model")
    parser.add_argument("--wandb_run_name", default=None)
    return parser.parse_args()

async def evaluate_dataset(dataset_name, problems, vllm_url, sandbox_url, sampling_params):
    """评测单个数据集的所有问题"""
    # 复用 phase0_eval.py 的生成和评测逻辑
    ...

def main():
    args = parse_args()
    # 1. 验证 checkpoint 存在
    # 2. 对每个 dataset 运行评测
    # 3. 收集指标
    # 4. 记录 WandB
    # 5. 保存 QA 日志
    # 6. 更新 best_checkpoint
    ...
```

### 8.2 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| vLLM 管理方式 | 外部启动/停止 | 评测脚本不负责 GPU 资源管理 |
| 并发评测 | asyncio + aiohttp | 复用 Phase 0 的异步架构 |
| WandB 集成 | 独立 run，同一 project | 训练和评测分开但可对比 |
| 错误处理 | 记录但继续 | 单题评测失败不影响整体 |
