# Part 7: 评测管线与 GRPO 交接

> 对应实现文档: `04_checkpoint_eval_pipeline.md` + `05_end_of_phase_and_grpo_handoff.md`
> 核心源码: `src/eval_config.py`, `src/utils/metrics.py`, `src/utils/qa_logger.py`

---

## 1. 为什么需要独立的评测管线？

在 Part 6 中我们看到，verl 的 `fit()` 循环只记录两类指标：
- `train/loss`: 训练交叉熵损失
- `val/loss`: 验证集交叉熵损失

但 **val_loss 低 ≠ 模型会写好代码**。举个极端例子：

```
模型 A: val_loss=0.8, 但每次生成代码都 import 不存在的库 → runtime_error
模型 B: val_loss=1.0, 但生成的代码能正确运行 → success
```

Phase 1 SFT 的核心目标是**降低 syntax/runtime/timeout 错误**。这些指标只能通过：

```
checkpoint → 加载模型 → 生成代码 → 在沙箱中执行 → 判断结果
```

这就是为什么我们需要一个独立于 verl 训练循环的 **Checkpoint Evaluation Pipeline**。

```
verl 训练循环 (fit())                    独立评测管线 (phase1_eval.py)
───────────────────                     ──────────────────────────────
每步: train/loss, train/lr              每个 checkpoint:
每 500 步: val/loss                       1. 加载 huggingface/ 模型
每 500 步: 保存 checkpoint →───────────→  2. 启动 vLLM 服务
                                          3. 生成代码 (greedy decoding)
只看"学得好不好"                           4. SandboxFusion 执行
                                          5. 计算 exec_success_rate, accepted@1
                                          6. 分析 error_type 分布
                                        看"用得好不好"
```

---

## 2. 三层评测架构

### 2.1 为什么分三层？

评测的开销与数据集大小成正比（需要生成代码 + 沙箱执行），我们需要在**评测精度**和**计算成本**之间取平衡：

| Tier | 精度 | 成本 | 用途 |
|------|------|------|------|
| Tier 1 | 中等（样本少） | 低（频繁跑） | 快速迭代反馈 |
| Tier 2 | 较高（样本多） | 中等（偶尔跑） | 确认趋势 |
| Tier 3 | 最高（全量测试集） | 高（只跑一次） | 最终报告 |

### 2.2 层级定义

| Tier | 数据集 | 题目数 | 评测频率 | WandB Key 前缀 |
|------|--------|--------|---------|----------------|
| **Tier 1** | codecontests_valid | 117 | 每 500 步 | `eval/codecontests_valid/` |
| **Tier 1** | MBPP_reg | 200 | 每 500 步 | `eval/mbpp_reg/` |
| **Tier 2** | codecontests_valid_big | 500 | 每 2000 步 | `eval/codecontests_valid_big/` |
| **Tier 3** | codecontests_test | 165 | Phase 结束 | `eval/codecontests_test/` |
| **Tier 3** | HumanEval | 164 | Phase 结束 | `eval/humaneval/` |

### 2.3 执行时间线

假设总训练步数约 3000，`save_freq=500`：

```
训练步数    保存 Checkpoint    评测内容
─────────  ──────────────    ──────────────────────────────
Step 500    ✅               Tier 1: valid(117) + MBPP(200)
Step 1000   ✅               Tier 1: valid(117) + MBPP(200)
Step 1500   ✅               Tier 1: valid(117) + MBPP(200)
Step 2000   ✅               Tier 1 + Tier 2: +valid_big(500)
Step 2500   ✅               Tier 1: valid(117) + MBPP(200)
Step 3000   ✅ (Final)       Tier 1 + Tier 2 + Tier 3: +test(165) +HumanEval(164)
```

### 2.4 数据集角色划分

```
                          训练数据
                             │
                     ┌───────┴───────┐
                     │               │
              BEE 训练数据      codecontests_valid_big (500)
              (SFT 学习用)         (从训练集拆出)
                                      │
                                  Tier 2 评测
                                  （较可靠信号）

                          验证/测试数据
                     ┌────────┼────────┐
                     │        │        │
             codecontests   codecontests  HumanEval
              _valid(117)   _test(165)     (164)
                  │             │            │
              Tier 1        Tier 3       Tier 3
             (快速反馈)   (最终报告)   (回归检测)

                        MBPP_reg (200)
                             │
                          Tier 1
                        (回归检测)
```

**为什么 codecontests_valid 是主选模指标？**
- 它 **不在训练集中** → 无信息泄漏
- 它有 **足够数量** (117) → 统计上有意义
- 它是 **竞赛编程题** → 与最终目标一致
- 它 **题目数适中** → 每次评测不会太慢

---

## 3. 评测流程：端到端解析

### 3.1 单个 Checkpoint 评测流程

```
phase1_eval.py --checkpoint_dir <path> --step 500

  ├─ [Step 1] 定位模型
  │     checkpoint_dir/global_step_500/huggingface/
  │     → 检查 config.json 和 model.safetensors 存在
  │
  ├─ [Step 2] 启动 vLLM 推理服务
  │     python -m vllm.entrypoints.openai.api_server \
  │       --model <huggingface_path> \
  │       --dtype bfloat16 \
  │       --max-model-len 6144 \          # 4096 prompt + 2048 generation
  │       --gpu-memory-utilization 0.85 \
  │       --tensor-parallel-size 1 \
  │       --port 8000
  │     → 等待健康检查通过
  │
  ├─ [Step 3] 加载评测问题
  │     根据 tier 从 manifest 文件加载:
  │     - data/manifests/codecontests_valid_manifest.jsonl
  │     - MBPP: 内置 id_range=(11, 210)
  │
  ├─ [Step 4] 生成代码 (EVAL@1 协议)
  │     使用 aiohttp 调用 vLLM OpenAI-compatible API:
  │     {
  │       "model": <model_path>,
  │       "messages": [{"role": "system", ...}, {"role": "user", ...}],
  │       "temperature": 0.0,        ← greedy decoding (确定性)
  │       "top_p": 1.0,
  │       "max_tokens": 2048
  │     }
  │
  ├─ [Step 5] 沙箱执行 (SandboxFusion)
  │     对每道题的生成代码:
  │     - CodeContests: 逐 testcase 传入 stdin，比较 stdout
  │     - HumanEval: test_code + check(entry_point)
  │     - MBPP: test_setup_code + assert 列表
  │     配置: run_timeout=30s, memory_limit_mb=1024
  │
  ├─ [Step 6] 收集指标
  │     MetricsCollector 汇总:
  │     - exec_success_rate, accepted@1, pass_ratio
  │     - syntax_error_rate, runtime_error_rate, timeout_rate
  │     - avg_gen_tokens, throughput
  │
  ├─ [Step 7] 记录到 WandB
  │     wandb.log({"eval/codecontests_valid/exec_success_rate": 0.65, ...}, step=500)
  │
  ├─ [Step 8] 保存 QA 日志
  │     QALogger 分层抽样保存代表性案例 (success/WA/syntax/runtime/timeout)
  │
  └─ [Step 9] 更新最佳 checkpoint
        if exec_success_rate > best: update best_checkpoint.json
```

### 3.2 EVAL@1 协议

我们项目使用 **EVAL@1 协议**（也称 pass@1 with greedy decoding）：

```
EVAL@1 协议:
  - 每题只生成 1 次代码
  - 使用 greedy decoding (temperature=0.0)
  - 结果完全确定性（可复现）
  - 评测的是"模型最好的一次猜测"

vs EVAL@k (如 pass@k):
  - 每题生成 k 次代码（如 k=200）
  - 使用 temperature>0 的采样
  - 评测的是"模型多次尝试中至少一次正确的概率"
  - 更昂贵，适合最终报告
```

这些参数统一定义在 `src/eval_config.py`:

```python
# eval_config.py

EVAL_CONSTANTS = {
    "temperature": 0.0,           # greedy decoding
    "top_p": 1.0,
    "max_new_tokens": 2048,       # 最大生成长度
    "run_timeout": 30,            # 沙箱执行超时(秒)
    "memory_limit_mb": 1024,      # 沙箱内存限制
}
```

### 3.3 为什么复用 Phase 0 组件？

Phase 0（Baseline）和 Phase 1（SFT）使用**完全相同的评测逻辑**，确保对比公平：

| 组件 | 文件 | Phase 0 | Phase 1 |
|------|------|---------|---------|
| 指标计算 | `src/utils/metrics.py` | ✅ 使用 | ✅ 复用 |
| QA 日志 | `src/utils/qa_logger.py` | ✅ 使用 | ✅ 复用 |
| 评测配置 | `src/eval_config.py` | ✅ 使用 | ✅ 复用 |
| Prompt 模板 | `src/phase0_eval.py` | ✅ 使用 | ✅ 复用 |
| 沙箱评测 | `src/phase0_eval.py` | ✅ 使用 | ✅ 复用 |

**唯一区别**: Phase 1 的模型路径来自 checkpoint 的 `huggingface/` 目录，而非原始预训练模型。

---

## 4. 指标体系详解

### 4.1 指标分类

评测产出两类文件：`metrics.json`（WandB 记录）和 `summary.json`（选模决策）。

**质量指标（最重要）：**

| 指标 | 定义 | 重要性 |
|------|------|--------|
| `exec_success_rate` | 代码能成功执行的比例（含 wrong_answer） | **主选模指标** |
| `accepted_at_1` | 通过所有测试用例的比例 | 最终正确率 |
| `pass_ratio_mean` | 平均通过测试用例比例 | 部分正确率 |
| `pass_ratio_p50` | 通过率中位数 | 中间水平 |
| `pass_ratio_p90` | 通过率 90 百分位 | 最好的题的表现 |

**错误分布指标：**

| 指标 | 定义 | Phase 1 期望 |
|------|------|-------------|
| `syntax_error_rate` | 语法错误占比 | ↓ 显著下降 |
| `runtime_error_rate` | 运行时错误占比 | ↓ 下降 |
| `timeout_rate` | 超时占比 | ↓ 下降 |
| `wrong_answer_rate` | 答案错误占比 | ≈ 不变或小幅上升 |
| `api_error_rate` | API/系统错误 | ≈ 趋近 0 |

**成本效率指标：**

| 指标 | 定义 | 用途 |
|------|------|------|
| `avg_gen_tokens` | 平均生成 token 数 | 监控生成长度 |
| `avg_judge_time` | 平均判题时间 | 监控执行效率 |
| `throughput` | 每秒评测的问题数 | 评测速度 |
| `truncation_rate` | 截断率（超过 max_tokens） | 生成长度合理性 |

### 4.2 exec_success_rate vs accepted_at_1

这两个指标的关系是 Phase 1 的核心：

```
所有问题 (100%)
  ├─ 代码无法执行: syntax_error + runtime_error + timeout
  │    → Phase 1 的主要攻击目标
  │
  └─ 代码能执行: exec_success_rate
       ├─ 答案正确: accepted_at_1  ← 最终目标
       └─ 答案错误: wrong_answer   ← Phase 3 GRPO 的目标
```

```
Phase 0 (Base):
  syntax_error: 15% ─┐
  runtime_error: 20% ─┤── 这 45% 根本没法运行
  timeout:       10% ─┘
  wrong_answer:  47%
  accepted:       8% ─── exec_success_rate = 55%

Phase 1 目标 (SFT):
  syntax_error:  5% ─┐
  runtime_error: 10% ─┤── 降到 20%
  timeout:        5% ─┘
  wrong_answer:  55% ─── 更多代码能跑了，但可能答错
  accepted:      25% ─── exec_success_rate = 80% ← ↑25pp!
```

### 4.3 MetricsCollector 的使用

`MetricsCollector`（`src/utils/metrics.py`）负责聚合所有评测结果：

```python
from utils.metrics import MetricsCollector, EvalResult

collector = MetricsCollector()

# 逐题添加结果
for problem in problems:
    collector.add_result(
        dataset="codecontests_valid",
        result=EvalResult(
            problem_id=problem.id,
            accepted=True,                    # 是否通过所有用例
            pass_ratio=0.8,                   # 通过率 [0, 1]
            error_type="wrong_answer",        # 错误类型
            judge_time=2.5,                   # 判题耗时
            gen_tokens=350,                   # 生成 token 数
            gen_time=1.2,                     # 生成耗时
        ),
    )

# 设置总耗时（用于计算 throughput）
collector.set_wall_clock_time("codecontests_valid", 120.0)

# 获取汇总
summary = collector.get_summary()
exec_rate = summary["datasets"]["codecontests_valid"]["exec_success_rate"]
```

`EvalResult` 的 `error_type` 是从沙箱执行结果推断的：

```
SandboxFusion 返回          →  error_type 映射
─────────────────           ──────────────────
SyntaxError / IndentError   →  "syntax_error"
NameError / TypeError / ... →  "runtime_error"
TimeLimit / 超时             →  "timeout"
执行成功但输出不匹配          →  "wrong_answer"
所有用例都通过               →  "success"
API 调用失败                →  "api_error"
```

---

## 5. QA 日志：人工审查的基础

### 5.1 为什么需要 QA 日志？

数字指标告诉你"有多好"，但不告诉你"好在哪/差在哪"。QA 日志保存代表性的 prompt-response 对，让你可以：

- 看看模型**答对的题**是什么样的
- 分析 **syntax_error** 是哪类语法问题（缩进？括号？import？）
- 检查 **runtime_error** 的具体原因
- 判断 **wrong_answer** 是逻辑错误还是 I/O 格式错误

### 5.2 分层抽样策略

不是保存所有结果（太大），而是按错误类型分层抽样：

| 评测时机 | 数据集 | 采样总数 | 分层方式 |
|---------|--------|---------|---------|
| 每 500 步 | codecontests_valid | 30 | 每种 error_type 5 个 + 5 random |
| 每 500 步 | MBPP_reg | 20 | 按 error_type 分层 |
| 每 2000 步 | codecontests_valid_big | 50 | 按 error_type 分层 |
| Phase 结束 | codecontests_test | 50 | 每种 error_type 10 个 |
| Phase 结束 | HumanEval | 30 | 每种 error_type 6 个 |

### 5.3 QA Log 格式

每条记录包含完整的 prompt-response-judgment 信息：

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

### 5.4 输出目录结构

```
coding_model_project/phase_1_ SFT/outputs/
  ├── eval_step_500/
  │   ├── metrics.json              # 所有数值指标
  │   ├── summary.json              # 聚合摘要 + 错误分布
  │   └── qa_logs/
  │       ├── codecontests_valid_qa.jsonl    # 分层抽样的 QA 日志
  │       ├── mbpp_reg_qa.jsonl
  │       └── qa_summary.json               # 抽样统计
  │
  ├── eval_step_1000/
  │   └── (同上)
  │
  ├── eval_final/                   # 最终评测（全量）
  │   ├── metrics.json
  │   ├── summary.json
  │   └── qa_logs/
  │       ├── codecontests_valid_qa.jsonl
  │       ├── codecontests_valid_big_qa.jsonl
  │       ├── codecontests_test_qa.jsonl
  │       ├── humaneval_qa.jsonl
  │       ├── mbpp_reg_qa.jsonl
  │       └── qa_summary.json
  │
  └── best_checkpoint.json          # 最佳 checkpoint 记录
```

---

## 6. 最佳 Checkpoint 选择

### 6.1 选择标准

**主选模指标**: `exec_success_rate` on `codecontests_valid`

```python
# 选择逻辑（概念性伪代码）

best_checkpoint = {"step": None, "exec_success_rate": 0.0}

for step, eval_result in all_eval_results.items():
    exec_rate = eval_result["summary"]["datasets"]["codecontests_valid"]["exec_success_rate"]
    if exec_rate > best_checkpoint["exec_success_rate"]:
        best_checkpoint = {
            "step": step,
            "exec_success_rate": exec_rate,
            "accepted_at_1": eval_result["summary"]["datasets"]["codecontests_valid"]["accepted_at_1"],
            "val_loss": eval_result.get("val/loss", None),
        }
```

**为什么用 exec_success_rate 而不是 accepted_at_1？**

| 指标 | 对比 |
|------|------|
| `exec_success_rate` | 衡量"代码能跑"的比例 → **Phase 1 的直接目标** |
| `accepted_at_1` | 衡量"答案完全正确"的比例 → Phase 3 GRPO 的目标 |

Phase 1 SFT 的目标是让模型"学会写能运行的代码"，所以用 exec_success_rate 选模。"答案正确"是 Phase 3 用 RL 进一步优化的目标。

### 6.2 best_checkpoint.json

```json
{
    "best_step": 1000,
    "best_exec_success_rate": 0.65,
    "best_accepted_at_1": 0.08,
    "model_path": "phase_1_ SFT/checkpoints/.../global_step_1000/huggingface",
    "selection_metric": "summary.datasets.codecontests_valid.exec_success_rate",
    "all_checkpoints": {
        "500":  {"exec_success_rate": 0.55, "accepted_at_1": 0.05},
        "1000": {"exec_success_rate": 0.65, "accepted_at_1": 0.08},
        "1500": {"exec_success_rate": 0.62, "accepted_at_1": 0.07}
    }
}
```

### 6.3 val_loss vs exec_success_rate 的关系

```
val_loss 曲线:     1.5 → 1.3 → 1.1 → 0.9 → 0.8    ← 持续下降
exec_success_rate: 0.5 → 0.6 → 0.65 → 0.62 → 0.58  ← 先升后降！

最低 val_loss 的 checkpoint (step 2500) ≠ 最高 exec_rate 的 checkpoint (step 1500)
```

这就是为什么我们不能只看 val_loss —— **val_loss 继续下降可能意味着过拟合到训练数据的特定模式上**，反而损害了泛化到新题目的能力。真正的"好不好"只有通过实际生成代码并执行才能判断。

---

## 7. Phase 0 vs Phase 1 对比报告

### 7.1 对比表格模板

Phase 1 结束后，生成与 Phase 0 Baseline 的对比报告：

**质量指标对比:**

| 指标 | Phase 0 (Base) | Phase 1 (SFT) | 变化 | 预期 |
|------|---------------|---------------|------|------|
| **exec_success_rate** (valid) | X% | Y% | +Z pp | ↑ +20-40pp |
| accepted_at_1 (valid) | X% | Y% | +Z pp | ↑ +1-5pp |
| pass_ratio_mean (valid) | X | Y | +Z | ↑ |
| HumanEval pass@1 | X% | Y% | ΔZ pp | ≈ (不应退化) |
| MBPP_reg pass@1 | X% | Y% | ΔZ pp | ≈ (不应退化) |

**错误分布对比:**

| 错误类型 | Phase 0 | Phase 1 | 变化 | 预期 |
|---------|---------|---------|------|------|
| syntax_error | X% | Y% | -Z% | ↓ -30-60% relative |
| runtime_error | X% | Y% | -Z% | ↓ -10-30% |
| timeout | X% | Y% | -Z% | ↓ -10-30% |
| wrong_answer | X% | Y% | +Z% | ≈ 或小幅 ↑ |

### 7.2 预期结果解读

```
Phase 0 (原始模型):
  ┌─────────────────────────────────────────┐
  │ syntax  ████████      15%               │
  │ runtime ██████████████ 20%              │
  │ timeout ██████        10%               │
  │ WA      ████████████████████████ 47%    │
  │ success ██████        8%                │
  └─────────────────────────────────────────┘
  exec_success_rate = 55%

Phase 1 (SFT 后):
  ┌─────────────────────────────────────────┐
  │ syntax  ███           5%                │ ← 大幅下降
  │ runtime ███████       10%               │ ← 下降
  │ timeout ███           5%                │ ← 下降
  │ WA      ██████████████████████████ 55%  │ ← 更多代码能跑 → WA 占比上升
  │ success ████████████████████ 25%        │ ← 上升
  └─────────────────────────────────────────┘
  exec_success_rate = 80%  ← +25pp 提升!
```

**关键洞察**: syntax/runtime/timeout 的下降会"转化"为 wrong_answer 和 success 的上升。WA 占比上升不是坏事 —— 说明更多代码能跑了。真正的正确率提升留给 Phase 3 GRPO。

### 7.3 异常情况处理

| 异常 | 可能原因 | 应对 |
|------|---------|------|
| exec_success_rate 提升 < 10pp | 训练数据质量差 / 过拟合 | 检查 QA logs 中的失败案例 |
| HumanEval 退化 > 5pp | 灾难性遗忘 | 降低 lr、减少 epochs、混合通用数据 |
| syntax_error 不降反升 | 训练数据本身有语法错误 | 过滤 BEE 数据 |
| 所有指标无变化 | 模型未学到有效信息 | 检查 loss 下降情况、检查数据管线 |

---

## 8. GRPO Phase 3 交接

### 8.1 交接的核心：HuggingFace 格式模型

回顾 Part 6，每个 checkpoint 都包含 `huggingface/` 目录。GRPO 需要的就是这个：

```
Phase 1 SFT 输出:
  global_step_{BEST}/huggingface/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors (~14GB)     ← 完整模型权重
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.json

                    │
                    ▼

Phase 3 GRPO 配置:
  model:
    partial_pretrain: /path/to/global_step_{BEST}/huggingface
```

**GRPO 从这里加载模型作为初始策略（policy model）和参考模型（reference model）**。

### 8.2 为什么可以直接交接？

```
FSDP 分片 checkpoint        HuggingFace checkpoint
  ├─ model_rank_0.pt         ├─ model.safetensors (完整)
  ├─ model_rank_1.pt         ├─ config.json
  ├─ ...                     ├─ tokenizer.json
  └─ optim_rank_0.pt         └─ ...

  只能同 GPU 数恢复训练        任何框架都能加载:
  需要 FSDP 上下文              - HuggingFace transformers
                                - vLLM (评测推理)
                                - verl GRPO (下一阶段)
                                - SGLang
```

`save_contents` 中的 `"hf_model"` 选项正是为这个目的设计的 —— 在每次保存 checkpoint 时，额外导出一份完整的 HuggingFace 格式模型。

### 8.3 交接前验证

在交给 GRPO 之前，需要验证 HF 模型的完整性：

**验证 1: HuggingFace 加载**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "phase_1_ SFT/checkpoints/.../global_step_{BEST}/huggingface"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="bfloat16")
print(f"Model: {model.config.architectures}")         # ['Qwen2ForCausalLM']
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")  # ~7B

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Vocab: {tokenizer.vocab_size}")

# 简单生成测试
inputs = tokenizer("def hello():", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**验证 2: vLLM 加载**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/huggingface \
    --dtype bfloat16 \
    --max-model-len 6144 \
    --port 8000

# 验证服务可用
curl http://localhost:8000/v1/models
```

**验证 3: 生成质量抽查**

用几道简单的编程题手动检查生成的代码是否合理（不是 SFT 之前的混乱输出）。

### 8.4 备选路径: FSDP 分片 → HF 转换

如果 checkpoint 中没有 `huggingface/` 目录（比如 `save_contents` 没有包含 `"hf_model"`），可以手动从 FSDP 分片重建：

```bash
# 需要与训练相同数量的 GPU
torchrun --nproc_per_node=8 convert_fsdp_to_hf.py \
    --ckpt_dir phase_1_ SFT/checkpoints/.../global_step_1000 \
    --output_dir phase_1_ SFT/checkpoints/.../global_step_1000/huggingface
```

内部逻辑：

```python
# 1. 初始化 FSDP 模型（需要与训练完全相同的配置）
# 2. 每个 rank 加载自己的分片
# 3. 调用 get_fsdp_full_state_dict() 聚合到 rank 0
# 4. rank 0 执行 save_pretrained()
```

这个方案更复杂，**推荐在训练配置中直接启用 `hf_model`**。

---

## 9. Phase 1 完整工作流总结

把所有 7 部分串起来，Phase 1 SFT 的完整流程：

```
                    Phase 1 SFT 完整工作流
╔════════════════════════════════════════════════════════╗
║                                                        ║
║  [1] 数据准备 (Part 2)                                  ║
║     BEE JSONL → parse → OpenAI messages → Parquet      ║
║                                                        ║
║  [2] 训练配置 (Part 5)                                  ║
║     sft_trainer.yaml + run_sft.sh                      ║
║                                                        ║
║  [3] Smoke Test (Part 5)                               ║
║     10 步验证 → 检查 loss/checkpoint/HF 导出            ║
║                                                        ║
║  [4] 正式训练 (Part 3-4)                               ║
║     torchrun → FSDPSFTTrainer → fit()                  ║
║     │                                                  ║
║     ├─ 每步: training_step → train/loss, lr, grad_norm ║
║     ├─ 每 500 步: validation → val/loss                ║
║     ├─ 每 500 步: save_checkpoint → huggingface/       ║
║     └─ WandB 实时监控 (Part 6)                         ║
║                                                        ║
║  [5] Checkpoint 评测 (Part 7)                          ║
║     │                                                  ║
║     ├─ 每 500 步: Tier 1 (valid + MBPP)                ║
║     ├─ 每 2000 步: Tier 2 (+valid_big)                 ║
║     ├─ Phase 结束: Tier 3 (+test + HumanEval)          ║
║     └─ 更新 best_checkpoint.json                       ║
║                                                        ║
║  [6] 最终报告                                          ║
║     Phase 0 vs Phase 1 对比表                          ║
║     exec_success_rate 预期 ↑ 20-40pp                   ║
║                                                        ║
║  [7] GRPO 交接                                         ║
║     best checkpoint huggingface/ → Phase 3 配置         ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
```

---

## 10. 知识点总结

### 核心概念

| 概念 | 定义 | 在项目中的体现 |
|------|------|---------------|
| **EVAL@1 协议** | greedy decoding，每题一次，确定性结果 | `temperature=0.0`, `top_p=1.0` |
| **exec_success_rate** | 代码能成功执行的比例 | Phase 1 主选模指标 |
| **accepted_at_1** | 通过所有测试用例的比例 | 最终正确率 |
| **三层评测** | Tier 1 (快速) / Tier 2 (确认) / Tier 3 (最终) | 精度-成本平衡 |
| **QA 日志** | 分层抽样的 prompt-response-judgment 记录 | 人工审查和错误分析 |
| **best_checkpoint** | exec_success_rate 最高的 checkpoint | 交给 GRPO 的模型 |
| **HF 模型交接** | SFT 输出 HuggingFace 格式 → GRPO 加载 | `partial_pretrain` 配置 |
| **val_loss vs 真实指标** | val_loss 低不等于代码质量好 | 独立评测管线的必要性 |

### Phase 间的衔接

```
Phase 0 (Baseline)
  输出: 基准指标 (exec_success_rate, accepted@1, error 分布)
  目的: 建立对比基线
     │
     ▼
Phase 1 (SFT)  ← 你在这里
  输入: 预训练模型 + BEE 代码数据
  输出: SFT 微调模型 (huggingface/ 格式)
  目标: ↓ syntax/runtime/timeout, ↑ exec_success_rate
     │
     ▼
Phase 3 (GRPO)
  输入: Phase 1 最佳 checkpoint 的 huggingface/ 模型
  目标: ↑ accepted@1, 进一步提高正确率
  方法: 强化学习（生成代码 → 执行 → 奖励信号 → 策略更新）
```

---

## 11. 思考题

1. **为什么评测使用 greedy decoding (temperature=0.0) 而不是采样？这对评测结果有什么影响？**

2. **exec_success_rate 从 55% 提升到 80%，但 accepted_at_1 只从 8% 到 10%，这说明了什么？对 Phase 3 GRPO 有什么启示？**

3. **如果 Tier 1 评测（117 题）显示 exec_success_rate = 70%，但 Tier 2 评测（500 题）显示 exec_success_rate = 55%，你会相信哪个？为什么？**

4. **为什么 val_loss 持续下降而 exec_success_rate 可能先升后降？这种现象叫什么？**

5. **Phase 1 SFT 的 HuggingFace 模型为什么可以直接被 GRPO 加载？GRPO 需要这个模型做什么？**

---

## 12. 全系列回顾

恭喜你完成了 verl SFT 全部 7 部分的学习！回顾一下完整知识图谱：

```
Part 1: SFT 基础        → 为什么做 SFT、Loss Masking、Teacher Forcing
Part 2: 数据流水线      → BEE 数据 → OpenAI messages → Parquet
Part 3: Trainer 架构    → torchrun → run_sft() → FSDPSFTTrainer
Part 4: 训练循环        → training_step 10 步、loss 计算、梯度累积
Part 5: 配置与超参      → sft_trainer.yaml、run_sft.sh、Smoke Test
Part 6: 检查点与监控    → FSDP 分片保存、HF 导出、Resume、WandB
Part 7: 评测与交接      → 三层评测、选模标准、GRPO 交接
```

接下来就可以动手实施了：**数据准备 → Smoke Test → 正式训练 → 评测 → 报告 → 交接**。
