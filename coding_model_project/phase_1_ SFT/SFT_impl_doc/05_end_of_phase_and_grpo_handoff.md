# Phase 1 SFT - Step 5: End-of-Phase Evaluation & GRPO Handoff

> 最终评测、Phase 0 vs Phase 1 对比报告、GRPO Phase 3 检查点交接

---

## 1. Overview

Phase 1 SFT 训练完成后的收尾工作：
1. 在所有测试数据集上运行最终评测 (Tier 3)
2. 生成 Phase 0 (Base) vs Phase 1 (SFT) 的对比报告
3. 准备最佳 checkpoint 供 Phase 3 GRPO 使用

---

## 2. 最终评测 (Tier 3)

### 2.1 最佳 Checkpoint 选择

从 `coding_model_project/phase_1_ SFT/outputs/best_checkpoint.json` 读取最佳 step（按 `summary.datasets.codecontests_valid.exec_success_rate` 选择）。

### 2.2 评测数据集

| 数据集 | 大小 | 角色 | 规则 |
|--------|------|------|------|
| CodeContests_test | 165 题 | **Test-only** | 只在 Phase 结束时运行，只报告不用于选模 |
| HumanEval | 164 题 | **Test-only** | 行业基准回归检测 |
| CodeContests_valid | 117 题 | Dev/Val | 最终综合评测（与训练中的 Tier 1 eval 对比） |
| CodeContests_valid_big | 500 题 | Dev/Val | 最终综合评测（与 Tier 2 对比） |
| MBPP_reg | 200 题 | Dev/Val | 回归检测 |

### 2.3 执行命令

```bash
PHASE1_DIR="coding_model_project/phase_1_ SFT"
CKPT_BASE="$PHASE1_DIR/checkpoints/rlvr_coding_model/phase1_sft_qwen7b_coder"
MANIFEST_DIR="coding_model_project/data/manifests"

# 使用最佳 checkpoint
BEST_STEP=$(python -c "import json; print(json.load(open('$PHASE1_DIR/outputs/best_checkpoint.json'))['best_step'])")

python "$PHASE1_DIR/phase1_eval.py" \
    --checkpoint_dir "$CKPT_BASE" \
    --step ${BEST_STEP} \
    --datasets codecontests_valid codecontests_valid_big codecontests_test humaneval mbpp_reg \
    --manifest_dir "$MANIFEST_DIR" \
    --use_external_tests \
    --no_submit_api \
    --output_dir "$PHASE1_DIR/outputs/eval_final" \
    --wandb_project rlvr_coding_model \
    --wandb_run_name phase1_final_eval
```

### 2.4 QA Log 要求（Tier 3）

| 数据集 | 采样数 | 分层策略 |
|--------|--------|---------|
| CodeContests_test | 50 | success/WA/syntax/runtime/timeout 各 10 |
| HumanEval | 30 | success/WA/syntax/runtime/timeout 各 6 |
| CodeContests_valid | 30 | 同 Tier 1 |
| CodeContests_valid_big | 50 | 同 Tier 2 |
| MBPP_reg | 20 | 同 Tier 1 |

---

## 3. Phase 0 vs Phase 1 对比报告

### 3.1 对比表格模板

#### 质量指标对比

| 指标 | Phase 0 (Base) | Phase 1 (SFT) | 变化 | 预期方向 |
|------|---------------|---------------|------|---------|
| **exec_success_rate** (valid) | X% | Y% | +Z pp | ↑ +20-40pp |
| **exec_success_rate** (valid_big) | - | Y% | - | 新增 |
| **exec_success_rate** (test) | X% | Y% | +Z pp | ↑ |
| accepted_at_1 (valid) | X% | Y% | +Z pp | ↑ small |
| accepted_at_1 (test) | X% | Y% | +Z pp | ↑ small |
| pass_ratio_mean (valid) | X | Y | +Z | ↑ |
| pass_ratio_p50 (valid) | X | Y | +Z | ↑ |
| pass_ratio_p90 (valid) | X | Y | +Z | ↑ |
| HumanEval pass@1 | X% | Y% | ΔZ pp | ≈ (无显著退化) |
| MBPP_reg pass@1 | X% | Y% | ΔZ pp | ≈ (无显著退化) |

#### 错误分布对比

| 错误类型 | Phase 0 (valid) | Phase 1 (valid) | 变化 | 预期方向 |
|---------|----------------|----------------|------|---------|
| **syntax_error** | X% | Y% | -Z% relative | ↓ -30-60% |
| **runtime_error** | X% | Y% | -Z% relative | ↓ |
| **timeout** | X% | Y% | -Z% relative | ↓ |
| wrong_answer | X% | Y% | ΔZ% | ≈ 或 ↑ |

#### 成本指标对比

| 指标 | Phase 0 | Phase 1 | 变化 |
|------|---------|---------|------|
| avg_gen_tokens | X | Y | ΔZ |
| avg_judge_time | X s | Y s | ΔZ s |
| throughput | X prob/s | Y prob/s | ΔZ |
| cost_per_solved_tokens | X | Y | ΔZ |
| cost_per_solved_judge_time | X s | Y s | ΔZ s |

### 3.2 预期结果（from final_experiment_design.md）

| 指标 | 预期变化方向 | 预期幅度 | 解释 |
|------|------------|---------|------|
| exec_success_rate | ↑ 显著 | +20-40 pp | SFT 让模型学会正确的 I/O 格式和 Python 语法 |
| syntax_error_rate | ↓ 显著 | -30-60% relative | SFT 最直接的效果 |
| runtime_error_rate | ↓ | -10-30% relative | 代码质量整体提升 |
| timeout_rate | ↓ | -10-30% relative | 学会更高效的算法模式 |
| accepted@1 | ↑ small or stable | +1-5 pp | 更多代码能运行 → 更多可能通过 |
| pass_ratio | ↑ small | +5-15 pp | 错误减少 → 通过更多测试用例 |
| HumanEval pass@1 | ≈ | ±2 pp | 不应显著退化 |
| MBPP_reg pass@1 | ≈ | ±2 pp | 不应显著退化 |

### 3.3 异常情况处理

| 情况 | 可能原因 | 应对方案 |
|------|---------|---------|
| exec_success_rate 提升 < 10pp | SFT 数据质量差、过拟合 | 检查 QA logs 中的失败案例 |
| HumanEval 显著退化 (> 5pp) | 灾难性遗忘 | 降低 lr、减少 epochs、或混合 general data |
| syntax_error_rate 不降反升 | 训练数据中有语法错误 | 过滤 BEE 数据中的语法错误样本 |
| 所有指标无变化 | 模型未学到有效信息 | 检查 loss 是否下降、检查 data pipeline |

### 3.4 报告生成脚本

`generate_report.py`:

```bash
PHASE1_DIR="coding_model_project/phase_1_ SFT"
python "$PHASE1_DIR/generate_report.py" \
    --phase0_metrics coding_model_project/outputs/phase0_*/metrics.json \
    --phase1_metrics "$PHASE1_DIR/outputs/eval_final/metrics.json" \
    --output_dir "$PHASE1_DIR/outputs/report/"
```

输出:
- `report/comparison_table.md` — Markdown 格式对比表
- `report/comparison_data.json` — JSON 格式原始数据
- `report/summary.txt` — 文字总结

---

## 4. GRPO Phase 3 检查点交接

### 4.1 主路径: hf_model 已在检查点中

由于训练配置中 `save_contents` 包含 `"hf_model"`，最佳检查点已经包含完整 HuggingFace 格式模型。

```
最佳 checkpoint 路径:
  phase_1_ SFT/checkpoints/rlvr_coding_model/phase1_sft_qwen7b_coder/
    global_step_{BEST}/
      huggingface/
        config.json
        generation_config.json
        model.safetensors (~14GB)
        tokenizer.json
        tokenizer_config.json
        special_tokens_map.json
        vocab.json
```

**GRPO 配置直接引用**:

```yaml
# Phase 3 GRPO 配置:
model:
  partial_pretrain: /path/to/phase_1_ SFT/checkpoints/.../global_step_{BEST}/huggingface
```

### 4.2 验证 HF 模型可加载

```python
# 验证脚本
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "phase_1_ SFT/checkpoints/.../global_step_{BEST}/huggingface"

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="bfloat16",
    trust_remote_code=True,
)
print(f"Model loaded: {model.config.architectures}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 2. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# 3. 简单生成测试
inputs = tokenizer("def hello():", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 4.3 验证 vLLM 可加载

```bash
# vLLM 加载测试
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/huggingface \
    --dtype bfloat16 \
    --max-model-len 6144 \
    --port 8000

# 验证
curl http://localhost:8000/v1/models
```

### 4.4 备选路径: FSDP → HF 转换

如果 `hf_model` 未被保存（或保存失败），可以从 FSDP 分片重建：

```python
# convert_fsdp_to_hf.py
"""将 FSDP 分片检查点转换为 HuggingFace 格式"""

# 需要用与训练相同数量的 GPU 运行:
# torchrun --nproc_per_node=8 convert_fsdp_to_hf.py --ckpt_dir <path> --output_dir <path>

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from verl.utils.fsdp_utils import get_fsdp_full_state_dict

# 1. 初始化 FSDP 模型（需要与训练相同的配置）
# 2. 加载分片检查点
# 3. 调用 get_fsdp_full_state_dict() 聚合
# 4. 在 rank 0 上 save_pretrained()
```

**注意**: 这个备选方案需要与训练相同数量的 GPU 和相同的 FSDP 配置。主路径（hf_model in save_contents）是更简单可靠的选择。

---

## 5. Phase 1 交付物清单

### 5.1 脚本文件

| 文件 | 用途 | 状态 |
|------|------|------|
| `phase_1_ SFT/prepare_sft_data.py` | BEE JSONL → Parquet | 待实现 |
| `phase_1_ SFT/run_sft.sh` | SFT 训练启动脚本 | 待实现 |
| `phase_1_ SFT/phase1_eval.py` | Checkpoint 评测流水线 | 待实现 |
| `phase_1_ SFT/run_eval_checkpoints.sh` | 批量评测自动化 | 待实现 |
| `phase_1_ SFT/generate_report.py` | Phase 0 vs 1 对比报告 | 待实现 |
| `phase_1_ SFT/convert_fsdp_to_hf.py` | FSDP → HF 转换（备选） | 待实现 |

### 5.2 verl 修改

| 文件 | 修改 | 状态 |
|------|------|------|
| `verl/trainer/fsdp_sft_trainer.py:527-531` | 添加 `"train/grad_norm"` 到返回字典 | 待实施 |

### 5.3 数据文件

| 文件 | 内容 | 状态 |
|------|------|------|
| `phase_1_ SFT/data/sft_train.parquet` | 训练数据（messages 格式） | 待生成 |
| `phase_1_ SFT/data/sft_val.parquet` | 验证数据（2% 随机划分） | 待生成 |

### 5.4 输出文件

| 文件 | 内容 | 何时生成 |
|------|------|---------|
| `phase_1_ SFT/checkpoints/.../global_step_*/huggingface/` | HF 格式模型 | 每 500 步 |
| `phase_1_ SFT/outputs/eval_step_*/metrics.json` | 各 checkpoint 评测指标 | 每次评测 |
| `phase_1_ SFT/outputs/eval_step_*/qa_logs/*_qa.jsonl` | QA 日志 | 每次评测 |
| `phase_1_ SFT/outputs/best_checkpoint.json` | 最佳 checkpoint 记录 | 评测完成时 |
| `phase_1_ SFT/outputs/eval_final/` | 最终评测结果 | Phase 结束时 |
| `phase_1_ SFT/outputs/report/` | 对比报告 | Phase 结束时 |

### 5.5 文档文件

| 文件 | 内容 |
|------|------|
| `phase_1_ SFT/SFT_impl_doc/01_data_preprocessing.md` | 本文档 |
| `phase_1_ SFT/SFT_impl_doc/02_training_config_and_flow.md` | 本文档 |
| `phase_1_ SFT/SFT_impl_doc/03_training_monitoring.md` | 本文档 |
| `phase_1_ SFT/SFT_impl_doc/04_checkpoint_eval_pipeline.md` | 本文档 |
| `phase_1_ SFT/SFT_impl_doc/05_end_of_phase_and_grpo_handoff.md` | 本文档 |

---

## 6. 端到端验证计划

### 6.1 数据预处理验证

- [ ] `sft_train.parquet` 可正确加载
- [ ] Messages 格式正确（system/user/assistant）
- [ ] valid_big 500 题已从训练集中排除
- [ ] 随机抽样 10 条记录，检查 prompt/response 分离正确

### 6.2 训练验证

- [ ] Smoke test（10 步）成功完成
- [ ] 检查点包含 `huggingface/` 目录
- [ ] `huggingface/` 中的模型可被 `AutoModelForCausalLM.from_pretrained()` 加载
- [ ] WandB 中可见 train/loss, train/lr, train/grad_norm, val/loss
- [ ] 完整训练（2 epochs）成功完成

### 6.3 评测验证

- [ ] Tier 1 评测（valid + MBPP）在第一个 checkpoint 上成功运行
- [ ] 指标计算正确并记录到 WandB
- [ ] QA 日志正确保存
- [ ] Tier 2 评测（valid_big）在 step 2000 成功运行
- [ ] best_checkpoint.json 自动生成

### 6.4 最终评测验证

- [ ] Tier 3 评测（test + HumanEval）在最佳 checkpoint 上运行
- [ ] 对比报告生成，所有指标有值
- [ ] HumanEval pass@1 无显著退化
- [ ] exec_success_rate 有预期提升

### 6.5 GRPO 交接验证

- [ ] 最佳 checkpoint 的 HF 模型路径可用于 GRPO 配置
- [ ] vLLM 可加载该模型进行推理
- [ ] 模型可生成合理的 Python 代码
