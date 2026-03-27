# RLVR Coding Model 项目进度文档

> 最后更新：2026-03-19

---

## 项目概览

本项目构建端到端 LLM 后训练流水线，使用可验证奖励（代码判题反馈）完成 **SFT → GRPO** 训练闭环。基础模型为 Qwen2.5-Coder-7B-Instruct，训练框架为 verl，代码评测使用 SandboxFusion 沙盒。

**当前决策**：Phase 0 评测结果显示 CodeContests 上有足够的奖励密度（pass_ratio_mean 14-26%），可直接进入 GRPO 阶段。Phase 1 SFT 因数据选取问题效果不佳（不如 base model），已搁置。

---

## 阶段状态

| 阶段 | 名称 | 状态 | 说明 |
|------|------|------|------|
| Phase 0 | Baseline 评测 | ✅ 已完成 | 完整评测基础设施、数据治理、基线指标 |
| Phase 1 | SFT | ⚠️ 已实现/已搁置 | 代码和流水线就绪，但训练效果不佳 |
| Phase 2 | DPO | ⏭️ 跳过 | 可选阶段，暂不执行 |
| Phase 3 | GRPO | 🔜 下一步 | 直接从 base model 开始在线 RL |
| Phase 4 | 多轮修复 | 📋 待定 | 可选 Agentic 扩展 |

---

## Phase 0: Baseline 评测（已完成）

### 关键指标

| 数据集 | accepted@1 | pass_ratio_mean | exec_success_rate | avg_gen_tokens | throughput |
|--------|-----------|-----------------|-------------------|----------------|------------|
| HumanEval | 87.2% | 87.2% | 100% | 124.2 | 5.63 prob/s |
| MBPP_reg | 58.5% | 58.5% | 100% | 53.8 | 8.35 prob/s |
| CodeContests_valid_big | 9.2% | 25.7% | - | 238.8 | 0.20 prob/s |
| CodeContests_valid | 1.7% | 14.3% | 88.9% | 264.0 | 0.13 prob/s |
| CodeContests_test | 3.0% | 13.8% | - | 273.6 | 0.16 prob/s |

### 错误分布（CodeContests_valid）

- Wrong Answer: ~70%
- Timeout: ~10%
- Runtime Error: ~10%
- Syntax Error: <1%

### 关键结论

- HumanEval/MBPP 基线强（87%/58%）
- CodeContests 挑战性大但 **pass_ratio_mean 14-26% 提供了充足的 dense reward 信号**
- 错误以 Wrong Answer 为主（70%），说明模型能执行但逻辑不对 → 适合 RL 优化

### 产出文件

- 评测结果：`outputs/phase0_20260206_154702/` （metrics.json, summary.json, run_info.json, qa_logs/）
- 数据审计：`reports/data_audit_report.md`

---

## Phase 1: SFT（已实现/已搁置）

### 已构建基础设施

- **训练脚本**：`phase_1_ SFT/run_sft.sh`（使用 verl FSDP SFT trainer）
- **数据处理**：`phase_1_ SFT/prepare_sft_data.py`、`dedup_one_per_problem.py` 等
- **评测流水线**：`phase_1_ SFT/phase1_eval.py`（2000+ 行，复用 Phase 0 基础设施）
- **文档**：`phase_1_ SFT/SFT_impl_doc/`（13 个讲解文档，涵盖数据处理到 GRPO 衔接）

### 训练数据

- `phase_1_ SFT/data/sft_train.parquet`（2.1 MB）
- `phase_1_ SFT/data/sft_val.parquet`（244 KB）
- 来源：BeeCodeContests 数据集，经过去重和质量过滤

### 结论

SFT 训练效果不如 base model，原因是 SFT 数据质量/分布与目标任务不匹配。鉴于 Phase 0 已证明有足够的奖励密度，决定跳过 SFT 直接进入 GRPO。

---

## Phase 3: GRPO（下一步）

### 已有基础

- **设计文档**：`experiment_design/grpo_minimal_hparams.md`（GRPO 超参最小集）
- **奖励设计**：`experiment_design/reward_design.md`（Dense vs Sparse 消融设计）
- **评测基础设施**：Phase 0/1 的评测脚本可直接复用
- **数据**：CodeContests_train（12,285 题去重后）用于 rollout
- **SandboxFusion**：已集成为 git submodule，用于在线奖励计算

### 核心消融

1. **Dense reward**（pass_ratio）vs **Sparse reward**（accepted）
2. 关键超参：group_size、KL 系数、学习率

### 关键设计参考

- 完整实验设计：`experiment_design/final_experiment_design.md`
- GRPO 超参检查清单：`experiment_design/grpo_minimal_hparams.md`
- 防 reward hacking：`experiment_design/guardrails.md`
- verl GRPO 示例：`examples/grpo_trainer/`（50+ 配置脚本）

---

## 代码与数据资产清单

### 核心代码

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/phase0_eval.py` | 2000+ | 评测主脚本（生成+评测+指标收集） |
| `src/eval_config.py` | 280+ | 评测常量与配置管理 |
| `src/data_governance.py` | 45K+ | 数据去重/治理/manifest 生成 |
| `src/utils/metrics.py` | 150+ | 指标收集器（EvalResult/DatasetMetrics/MetricsCollector） |
| `src/utils/qa_logger.py` | - | 问答日志分层抽样 |
| `phase_1_ SFT/phase1_eval.py` | 2000+ | Phase 1 评测脚本 |
| `scripts/run_phase0.sh` | 120 | Phase 0 启动脚本 |

### 数据文件

| 路径 | 大小 | 说明 |
|------|------|------|
| `data/raw/codecontests_train_raw.jsonl` | 785 MB | 训练集原始数据（13,328 题） |
| `data/raw/codecontests_valid_big_raw.jsonl` | 31 MB | 扩展验证集（500 题） |
| `data/raw/codecontests_test_raw.jsonl` | 5.8 MB | 测试集（165 题） |
| `data/raw/codecontests_valid_raw.jsonl` | 12 MB | 验证集（117 题） |
| `data/raw/humaneval_raw.jsonl` | 309 KB | HumanEval（164 题） |
| `data/raw/mbpp_reg_raw.jsonl` | 177 KB | MBPP 回归集（200 题） |
| `data/manifests/*.jsonl` | 6.3 MB | 去重后数据 manifest（9 个文件） |
| `phase_1_ SFT/data/*.parquet` | 2.3 MB | SFT 训练/验证数据 |

### 设计文档

| 文件 | 说明 |
|------|------|
| `experiment_design/final_experiment_design.md` | 完整五阶段实验设计（最权威） |
| `experiment_design/project_plan.md` | 简历版项目计划 |
| `experiment_design/eval_protocol.md` | 评测协议（EVAL@1/k/@budget） |
| `experiment_design/data_governance.md` | 数据治理原则 |
| `experiment_design/reward_design.md` | 奖励函数设计 |
| `experiment_design/guardrails.md` | 防 reward hacking |
| `experiment_design/grpo_minimal_hparams.md` | GRPO 超参最小集 |

---

## 仓库结构

### Git 分支与 Worktree

| 路径 | 分支 | 说明 |
|------|------|------|
| `/Users/xiaohui/Desktop/verl/verl` | `main` | 主分支（verl 框架） |
| `/Users/xiaohui/Desktop/verl/verl-sft` | `feature/sft-development` | SFT 开发分支 |
| `/Users/xiaohui/Desktop/verl/verl-grpo` | `feature/grpo-development` | GRPO 开发分支（活跃） |

### Git 子模块

| 名称 | 路径 | 远程仓库 | 说明 |
|------|------|----------|------|
| recipe | `recipe/` | verl-project/verl-recipe | verl 训练配方 |
| SandboxFusion | `SandboxFusion/` | Rogerffff/sandbox | 代码评测沙盒 |

### 技术栈

- **训练框架**: verl（分布式 RL）
- **推理引擎**: vLLM
- **代码评测**: SandboxFusion
- **基础模型**: Qwen2.5-Coder-7B-Instruct
- **实验追踪**: WandB
