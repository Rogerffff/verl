# RLVR Coding Model Project - 项目介绍与文件结构

---

## 一、项目概述

### 项目名称
**RLVR Coding Post-Training (Offline DPO + Online GRPO with Verifiable Rewards)**

### 项目目标
构建一个端到端的 LLM 后训练闭环，使用可验证奖励（代码判题）完成：
**SFT → (离线 DPO) → 在线 GRPO → 多轮修复**

产出工业界认可的"后训练流程 + 可信评测 + 成本/稳定性面板"，作为简历项目展示 RL/LLM 后训练能力。

### 技术栈
- **训练框架**: verl (分布式 RL 训练框架)
- **推理引擎**: vLLM (高性能 LLM 推理)
- **代码评测**: SandboxFusion (安全沙盒执行环境)
- **基础模型**: Qwen2.5-Coder-7B-Instruct
- **实验追踪**: Weights & Biases (WandB)

### 五阶段训练流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            训练流程总览                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│   Phase 0          Phase 1         Phase 2           Phase 3      Phase 4  │
│   Baseline    →      SFT      →   DPO (可选)    →    GRPO     →  多轮修复   │
│     │                 │               │               │           (可选)    │
│     ↓                 ↓               ↓               ↓             ↓       │
│   建立基线       提升格式/执行    偏好对齐冷启动    在线RL优化    Agentic扩展 │
└─────────────────────────────────────────────────────────────────────────────┘
```

| 阶段 | 目标 | 核心指标 | 状态 |
|------|------|----------|------|
| Phase 0 | 建立基线 | accepted@1, pass_ratio, error_breakdown | ✅ 已完成 |
| Phase 1 | 降低低级错误 | exec_success_rate ↑, syntax_error_rate ↓ | 待实现 |
| Phase 2 | 偏好对齐冷启动 | pass_ratio ↑, zero_reward_rate ↓ | 可选 |
| Phase 3 | 在线 RL 优化 | CodeContests_test accepted@1 +3~10pp | 核心阶段 |
| Phase 4 | 多轮修复 | recovery_rate 20~40% | 加分项 |

---

## 二、数据集角色定义

| 数据集 | 角色 | 题目数 | 使用规则 |
|--------|------|--------|----------|
| **CodeContests_train** | Train | ~13k | 训练/构造偏好对 |
| **CodeContests_valid** | Dev/Val | 117 | 高频回归、早停、选超参 |
| **CodeContests_test** | Test | - | 阶段结束评测，禁止训练/调参 |
| **HumanEval** | Test only | 164 | 行业对标，禁止训练 |
| **MBPP_reg** | Dev/Val | 100-200 | 快速回归监控 |

---

## 三、完整文件结构

```
verl/coding_model_project/
│
├── experiment_design/                    # 📋 核心实验设计文档
│   ├── final_experiment_design.md        # ★ 完整五阶段实验设计（最权威参考）
│   ├── project_plan.md                   # 简历版本项目计划
│   ├── eval_protocol.md                  # 评测协议定义（EVAL@1/k/@budget）
│   ├── data_governance.md                # 数据治理原则（去重/泄漏检查）
│   ├── reward_design.md                  # 奖励函数设计（Dense vs Sparse）
│   ├── guardrails.md                     # 防 reward hacking 约束
│   ├── grpo_minimal_hparams.md           # GRPO 超参最小集
│   ├── metric_templates.md               # 指标定义模板
│   └── resource_plan.md                  # GPU 资源规划
│
├── phase_0_ Baseline/                    # 📊 Phase 0 基线评测
│   ├── PARAMETERS.md                     # Phase 0 参数说明
│   ├── phase0_implementation_plan.md     # 详细实施计划（1900+ 行）
│   ├── verl_standalone_rollout_guide.md  # verl 架构深度讲解
│   ├── data_governance_guide.md          # 数据治理完整指南
│   ├── metrics_collection_spec.md        # 指标收集规范
│   └── implement_explain_doc/            # 代码讲解文档
│       ├── 01_data_governance.md
│       ├── 02_eval_script.md
│       ├── 03_output_files_and_metrics.md
│       └── eval_script代码讲解.md
│
├── src/                                  # 💻 核心实现代码
│   ├── phase0_eval.py                    # ★ Phase 0 评测主脚本（2000+ 行）
│   │                                     # 功能：vLLM 服务器启动、代码生成、
│   │                                     # SandboxFusion 评测、指标收集
│   │
│   ├── eval_config.py                    # 评测常量与配置（280+ 行）
│   │                                     # 管理：EVAL_CONSTANTS、DATASET_CONFIGS、
│   │                                     # CONCURRENCY_CONFIGS
│   │
│   ├── data_governance.py                # 数据治理脚本（数据获取+去重+验证）
│   │
│   ├── utils/
│   │   ├── metrics.py                    # 指标收集器（MetricsCollector）
│   │   │                                 # 包含：EvalResult、DatasetMetrics
│   │   └── qa_logger.py                  # 问答日志（分层抽样保存）
│   │
│   ├── config/
│   │   └── phase0_config.yaml            # YAML 配置文件
│   │
│   └── temp/                             # 临时脚本目录
│       ├── test_sandbox_eval.py
│       ├── verify_dedup.py
│       └── add_mbpp_entry_point.py
│
├── data/                                 # 📁 数据管理
│   ├── manifests/                        # 去重后数据的索引文件
│   │   ├── humaneval_manifest.jsonl
│   │   ├── mbpp_reg_manifest.jsonl
│   │   ├── codecontests_train_manifest.jsonl
│   │   ├── codecontests_valid_manifest.jsonl
│   │   ├── codecontests_test_manifest.jsonl
│   │   └── codecontests_*_duplicates_intrasplit.jsonl
│   │
│   └── raw/                              # 原始数据文件
│       ├── humaneval_raw.jsonl
│       ├── mbpp_reg_raw.jsonl
│       ├── codecontests_train_raw.jsonl
│       ├── codecontests_valid_raw.jsonl
│       ├── codecontests_test_raw.jsonl
│       └── dataset_samples.jsonl
│
├── outputs/                              # 📈 评测结果输出
│   └── phase0_YYYYMMDD_HHMMSS/           # 时间戳输出目录
│       ├── metrics.json                  # 主要指标（按数据集聚合）
│       ├── summary.json                  # 完整汇总
│       ├── run_info.json                 # 可审计的运行信息
│       └── qa_logs/                      # 问答日志（分层抽样）
│           ├── humaneval_qa.jsonl
│           ├── mbpp_reg_qa.jsonl
│           ├── codecontests_valid_qa.jsonl
│           └── qa_summary.json
│
├── scripts/
│   └── run_phase0.sh                     # Phase 0 启动脚本
│
├── reports/
│   └── data_audit_report.md              # 数据治理审计报告
│
├── phase_1_sft/                          # Phase 1 SFT 目录（待实现）
│
├── verl基础讲解/                         # 📚 学习资料
│   ├── 01_从RayWorkerGroup到Replica.md
│   └── claude.md
│
├── agent.md                              # 本文件 - 项目介绍
└── claude.md                             # AI 助手上下文说明
```

---

## 四、核心文件详解

### 4.1 评测主脚本 `src/phase0_eval.py`

**功能架构**：

```python
# 第1层：配置与数据
class EvalConfig           # 评测配置数据类
SYSTEM_PROMPT             # LLM 系统提示
PROMPT_TEMPLATES          # 不同数据集的 prompt 模板

# 第2层：服务器管理
async start_rollout_servers()      # 启动 verl Standalone 服务器
async fetch_openai_models()        # 查询服务端的实际模型

# 第3层：代码生成
async generate_code()              # 单个代码生成（async）
async batch_generate()             # 批量并发生成（负载均衡）

# 第4层：代码评测
evaluate_with_submit_api()         # SandboxFusion submit() 评测
evaluate_with_run_code()           # SandboxFusion run_code() 评测

# 第5层：数据加载
load_prompts()                     # 从 manifest 或 SandboxFusion 加载

# 第6层：主评测流程
async evaluate_dataset()           # 评测单个数据集
async run_evaluation()             # 完整评测闭环
```

**关键设计**：
- **Async 并发生成**：使用 `asyncio.Semaphore` 限制并发数（默认64）
- **Round-Robin 负载均衡**：将请求均匀分发到多个 vLLM replica
- **多种评测方式支持**：`submit()` / `run_code()` / `compute_score()`
- **错误类型自动分类**：syntax/runtime/timeout/wrong_answer
- **可审计的运行信息**：记录配置、服务器地址、实际加载的模型 ID

### 4.2 指标收集器 `src/utils/metrics.py`

```python
class EvalResult:              # 单个问题的评测结果
  problem_id, accepted, pass_ratio, error_type, judge_time, gen_tokens

class DatasetMetrics:          # 数据集级聚合指标
  # 质量：accepted_at_1, pass_ratio_mean/p50/p90, exec_success_rate
  # 错误分布：各错误类型的比例
  # 成本：avg_judge_time, p50/p95_judge_time, avg_gen_tokens, throughput
  # 成本比率：cost_per_solved_tokens/judge_time

class MetricsCollector:        # 指标收集器
  def add_result()             # 逐个添加评测结果
  def get_dataset_metrics()    # 计算聚合指标（numpy 统计）
  def get_summary()            # 多数据集汇总
  def get_wandb_metrics()      # 返回 WandB 格式字典
```

### 4.3 评测配置 `src/eval_config.py`

```python
EVAL_CONSTANTS = {
    "temperature": 0.0,         # EVAL@1 协议（greedy decoding）
    "top_p": 1.0,
    "max_new_tokens": 2048,
    "run_timeout": 30,          # SandboxFusion 超时
    "memory_limit_mb": 1024,
}

DATASET_CONFIGS = {
    "humaneval": {...},
    "mbpp_reg": {...},
    "codecontests_valid": {...},
    ...
}
```

---

## 五、Phase 0 实际运行结果

| 数据集 | accepted@1 | pass_ratio_mean | exec_success_rate | avg_gen_tokens | throughput |
|--------|------------|-----------------|-------------------|----------------|------------|
| HumanEval | 87.2% | 0.872 | 100% | 122.5 | 5.85 问题/秒 |
| MBPP_reg | 58.0% | 0.58 | 100% | 54.1 | 7.9 问题/秒 |
| CodeContests_valid | 3.4% | 0.120 | 88.9% | 267.4 | 0.17 问题/秒 |

**错误分布（CodeContests_valid）**：
- Wrong Answer: ~70%
- Timeout: 10.3%
- Runtime Error: ~10%
- Syntax Error: <1%

---

## 六、简历写法建议

### Bullet Points 示例

1. **端到端闭环**
   - Built an end-to-end LLM post-training pipeline with verifiable rewards from judging feedback
   - Implemented SFT → (optional offline DPO) → online GRPO with strict train/dev/test isolation

2. **质量改进数据**
   - Improved CodeContests_test accepted@1 from X% → Y% (2 seeds, mean±std)
   - Reduced syntax/runtime errors by X% through SFT phase

3. **关键消融与稳定性**
   - Conducted critical ablations: dense (pass_ratio) vs sparse (accepted) reward signals
   - Demonstrated faster convergence and lower variance with dense rewards

4. **工程细节**
   - Implemented async code generation + load balancing, achieving 5.8x throughput improvement
   - Established auditable data isolation to prevent train/test leakage

---

## 七、快速开始

```bash
# 1. 启动 vLLM 服务
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-Coder-7B-Instruct

# 2. 启动 SandboxFusion 服务
docker run -p 8080:8080 volcengine/sandbox-fusion:server-20250609

# 3. 运行 Phase 0 评测
python src/phase0_eval.py \
    --mode simple \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --vllm_url http://localhost:8000 \
    --sandbox_url http://localhost:8080 \
    --datasets humaneval mbpp_reg codecontests_valid \
    --output_dir outputs/phase0
```

---

## 八、关键文档导航

| 文档 | 用途 | 路径 |
|------|------|------|
| **final_experiment_design.md** | 完整五阶段设计（最权威） | experiment_design/ |
| **project_plan.md** | 简历版本（HR 看这个） | experiment_design/ |
| **phase0_implementation_plan.md** | Phase 0 详细执行 | phase_0_ Baseline/ |
| **verl_standalone_rollout_guide.md** | verl 架构深度讲解 | phase_0_ Baseline/ |
| **eval_protocol.md** | EVAL@1/k/@budget 口径对齐 | experiment_design/ |
| **data_governance.md** | 数据去重+泄漏检查 | experiment_design/ |
