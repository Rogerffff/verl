# RLVR Coding Model - 详细实验设计

---

## 一、项目概述与训练流程

### 1.1 项目目标

构建一个端到端的 LLM 后训练闭环，使用可验证奖励（代码判题）完成 SFT → (离线 DPO) → 在线 GRPO 训练，产出工业界认可的"后训练流程 + 可信评测 + 成本/稳定性面板"。

### 1.2 训练流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            训练流程总览                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Phase 0          Phase 1         Phase 2           Phase 3      Phase 4  │
│   Baseline    →      SFT      →   DPO (可选)    →    GRPO     →  多轮修复   │
│     │                 │               │               │           (可选)    │
│     ↓                 ↓               ↓               ↓             ↓       │
│   建立基线       提升格式/执行    偏好对齐冷启动    在线RL优化    Agentic扩展 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 数据集角色定义

| 数据集 | 角色 | 使用规则 |
|--------|------|----------|
| **CodeContests_train** | Train | 训练/构造偏好对 |
| **CodeContests_valid** | Dev/Val | 高频回归、早停、选超参 |
| **CodeContests_test** | Test | 阶段结束评测，禁止训练/调参 |
| **HumanEval** | Test only | 行业对标，禁止训练/构造偏好对/选超参 |
| **MBPP_reg** (100-200题) | Dev/Val | 回归监控，固定题号列表 |

---

## 二、Phase 详细设计

---

### Phase 0: Baseline（基线评测）

#### 目的

- **建立对照基准**：为所有后续实验提供可信的对照点
- **证明训练增益**：记录未训练模型的真实能力，后续任何提升都以此为参照
- **建立成本基线**：记录执行时间、吞吐量等指标，为后续优化提供参照

#### 做什么

1. 使用 Base 模型 (Qwen2.5-Coder-7B-Instruct) 在所有数据集上评测
2. 覆盖 4 个数据集：
   - Dev/Val: CodeContests_valid + MBPP_reg
   - Test: CodeContests_test + HumanEval
3. 收集三类指标：质量、成本、错误分布
4. 使用 SandboxFusion 执行代码判题，获取详细的测试结果和元数据

#### 必须产出

**质量指标**：
| 数据集 | 指标 |
|--------|------|
| CodeContests_valid/test | accepted@1, pass_ratio(mean/p50/p90), exec_success_rate, error breakdown |
| HumanEval | pass@1 (行业对标基线) |
| MBPP_reg | pass@1 + error breakdown (回归监控基线) |

**成本效率指标**：
- avg_total_gen_tokens / problem（见 [eval_protocol.md](./eval_protocol.md)）
- avg_total_judge_time / problem（sandbox duration 汇总）
- throughput (problems/sec, wall clock)
- cost_per_solved_tokens + cost_per_solved_judge_time（分别统计；不要把 tokens 和秒直接相加）

**错误分布指标**：
- Syntax Error Rate (编译/语法错误)
- Runtime Error Rate (运行时错误)
- Timeout Rate (超时)
- Wrong Answer Rate (答案错误)

#### 可加分点

- 记录 p95_total_judge_time（后续系统优化有对照）
- 记录 problems/sec 吞吐量基线（系统性能参照）

#### 日志与记录要求

**WandB/训练指标**：
| 指标类别 | 指标名 | 说明 |
|----------|--------|------|
| 评测结果 | `eval/accepted_at_1` | 按数据集分别记录 |
| 评测结果 | `eval/pass_ratio_mean` | 按数据集分别记录 |
| 评测结果 | `eval/exec_success_rate` | 代码可执行率 |
| 错误分布 | `eval/syntax_error_rate` | 语法错误率 |
| 错误分布 | `eval/runtime_error_rate` | 运行时错误率 |
| 错误分布 | `eval/timeout_rate` | 超时率 |
| 成本 | `eval/avg_total_gen_tokens` | 平均总生成 tokens（含所有 candidates/rounds） |
| 成本 | `eval/avg_total_judge_time` | 平均总判题时间（含所有 candidates/rounds） |
| 成本 | `eval/cost_per_solved_tokens` | 每 solved 的 tokens 成本 |
| 成本 | `eval/cost_per_solved_judge_time` | 每 solved 的判题时间成本 |
| 成本 | `eval/throughput` | problems/sec（wall clock） |

**问答日志记录**：
| 数据集 | 记录数量 | 记录内容 |
|--------|----------|----------|
| CodeContests_valid | **50 条** | prompt, response, ground_truth, pass_ratio, error_type, execution_output |
| CodeContests_test | **30 条** | 同上 |
| HumanEval | **20 条** | 同上 |
| MBPP_reg | **20 条** | 同上 |

> 建议按 error_type 分层抽样：成功/WA/Syntax/Runtime/Timeout 各抽若干条

**Checkpoint 保存**：
- ❌ **不需要保存**：Phase 0 是评测阶段，使用原始 Base 模型

---

### Phase 1: SFT（监督微调）

#### 目的

- **降低低级错误**：减少 Syntax/Runtime/Timeout 错误，让模型输出"能跑的代码"
- **为 RL 打基础**：为 GRPO 阶段提供更好的起点，避免 RL 预算浪费在格式错误上
- **稳定输出格式**：让模型学会遵循 stdin/stdout 的 IO 格式

#### 做什么

1. **训练数据**：使用 CodeContests_train 进行监督微调
2. **可选课程学习**：从 easy/短题子集开始 warm-up，再扩展到更大子集
3. **高频回归**：每 N steps 在 CodeContests_valid + MBPP_reg 上评测
   - 监控 exec_success_rate 提升
   - 监控错误分布变化
4. **阶段结束测试**：在 CodeContests_test + HumanEval 上完整评测

#### 必须产出

**核心指标（对比 Phase 0）**：
| 指标 | 预期变化 | 说明 |
|------|----------|------|
| exec_success_rate | ↑ 显著 | **最重要指标**，证明代码能正常运行 |
| Syntax Error Rate | ↓ 显著 | 编译/语法错误应大幅减少 |
| Runtime Error Rate | ↓ | 运行时错误减少 |
| Timeout Rate | ↓ | 超时减少 |
| accepted@1, pass_ratio | ↑ 小幅或持平 | 重点是"能跑"，不是"全对" |

**对标指标**：
- HumanEval pass@1：阶段末跑一次，证明泛化能力未退化

#### 可加分点

- **模板合规率分析**：输出是否只含代码、stdin/stdout 格式是否正确
- **Failure Analysis**：抽样 50 个失败 case 分类归因
  - I/O 格式错误
  - 超时
  - 逻辑错误
  - 其他

#### 日志与记录要求

**WandB/训练指标（每 step 记录）**：
| 指标类别 | 指标名 | 记录频率 |
|----------|--------|----------|
| 训练损失 | `train/loss` | 每 step |
| 训练损失 | `train/grad_norm` | 每 step |
| 学习率 | `train/lr` | 每 step |
| 验证损失 | `val/loss` | 每 N steps |

**WandB/评测指标（每 100-500 steps）**：
| 指标类别 | 指标名 | 说明 |
|----------|--------|------|
| 质量 | `eval/exec_success_rate` | **最重要**，监控提升 |
| 质量 | `eval/accepted_at_1` | 按数据集分别记录 |
| 质量 | `eval/pass_ratio_mean` | 按数据集分别记录 |
| 质量 | `eval/pass_ratio_p50` | 按数据集分别记录 |
| 质量 | `eval/pass_ratio_p90` | 按数据集分别记录 |
| 错误分布 | `eval/syntax_error_rate` | 期望显著下降 |
| 错误分布 | `eval/runtime_error_rate` | 期望下降 |
| 错误分布 | `eval/timeout_rate` | 期望下降 |
| 错误分布 | `eval/wrong_answer_rate` | 监控逻辑错误占比 |
| 成本 | `eval/avg_total_gen_tokens` | 平均总生成 tokens（含所有 candidates/rounds） |
| 成本 | `eval/avg_total_judge_time` | 平均总判题时间（含所有 candidates/rounds） |
| 成本 | `eval/cost_per_solved_tokens` | 每 solved 的 tokens 成本 |
| 成本 | `eval/cost_per_solved_judge_time` | 每 solved 的判题时间成本 |
| 成本 | `eval/throughput` | problems/sec（wall clock） |

**问答日志记录**：
| 时机 | 数据集 | 记录数量 | 记录内容 |
|------|--------|----------|----------|
| 每 500 steps | CodeContests_valid | **30 条** | prompt, response, ground_truth, pass_ratio, error_type |
| 每 500 steps | MBPP_reg | **20 条** | 同上 |
| 阶段结束 | CodeContests_test | **50 条** | 同上 + execution_output |
| 阶段结束 | HumanEval | **30 条** | 同上 |

> 建议对比记录：同一题目在不同 checkpoint 的输出变化

**Checkpoint 保存**：
| 保存时机 | 保存内容 | 用途 |
|----------|----------|------|
| 每 500 steps | 完整模型 (HF 格式) | 断点恢复 |
| exec_success_rate 最高点 | 完整模型 | **用于 GRPO 初始化** |
| 阶段结束 | 完整模型 | 最终 SFT checkpoint |

> ⚠️ 至少保留 **3 个 checkpoint**：最新、最佳、最终

---

### Phase 2: Offline DPO（暂时跳过）

#### 目的

- **偏好对齐冷启动**：用离线偏好数据把策略推向"更可能通过更多测试点"的分布
- **稳定 GRPO 起点**：让 GRPO 更稳、更省 rollouts
- **降低在线训练成本**：减少 GRPO 阶段需要的在线采样量

> **注意**：verl 不内置 DPO 训练器，需要使用外部工具（如 HuggingFace TRL）或跳过此阶段。

#### 做什么

1. **采样解答**：使用 SFT checkpoint 在 CodeContests_train 每题采样 K=2-4 个解答
2. **评测解答**：用 SandboxFusion 判题得到 pass_ratio 和 error_type
3. **构造偏好对**：
   - chosen: pass_ratio 最高（并列选更短/更快）
   - rejected: pass_ratio 最低（优先 Syntax/Timeout）
4. **过滤噪声**：
   - Δpass_ratio ≥ 0.3 才保留
   - "全 0 vs 全 0" 的对丢弃
5. **DPO 训练**：使用 HuggingFace TRL 等外部工具训练

**替代方案**（如果跳过 DPO）：
- 方案 A：直接 SFT → GRPO（推荐，流程简化）
- 方案 B：使用 verl 内置的 Preference Feedback PPO (`use_pf_ppo=True`)

#### 必须产出

**质量指标（对比 Phase 1）**：
| 指标 | 预期变化 | 说明 |
|------|----------|------|
| CodeContests_valid accepted@1 | ↑ | 作为可验证主指标的补充 |
| CodeContests_valid pass_ratio | ↑ (mean/p50/p90) | DPO 效果体现 |
| zero-reward rate | ↓ | pass_ratio=0 的比例应下降 |
| MBPP_reg | 稳定 | 回归不退化 |

**成本指标**：
- DPO 数据构建判题成本（总判题次数）
- 每题平均采样数

#### 可加分点

- **偏好数据质量报告**：Δpass_ratio 分布图
- **error_type 分布分析**：chosen vs rejected 的错误类型分布对比
- **pair 构造策略消融**：是否加入 judge_time 作为 tie-breaker

#### 日志与记录要求

**WandB/训练指标（每 step 记录）**：
| 指标类别 | 指标名 | 说明 |
|----------|--------|------|
| 训练损失 | `train/dpo_loss` | DPO 损失 |
| 训练损失 | `train/chosen_reward` | chosen 样本的隐式奖励 |
| 训练损失 | `train/rejected_reward` | rejected 样本的隐式奖励 |
| 训练损失 | `train/reward_margin` | chosen - rejected margin |
| 学习率 | `train/lr` | 学习率 |
| 梯度 | `train/grad_norm` | 梯度范数 |

**WandB/评测指标（每 100-500 steps）**：
| 指标类别 | 指标名 | 说明 |
|----------|--------|------|
| 质量 | `eval/pass_ratio_mean` | 期望 ↑ |
| 质量 | `eval/pass_ratio_p50` | 期望 ↑ |
| 质量 | `eval/pass_ratio_p90` | 期望 ↑ |
| 质量 | `eval/zero_reward_rate` | pass_ratio=0 的比例，期望 ↓ |
| 质量 | `eval/accepted_at_1` | 期望小幅 ↑ |
| 质量 | `eval/exec_success_rate` | 监控可执行率不退化 |
| 错误分布 | `eval/syntax_error_rate` | 监控低级错误 |
| 错误分布 | `eval/runtime_error_rate` | 监控低级错误 |
| 错误分布 | `eval/timeout_rate` | 监控低级错误 |
| 错误分布 | `eval/wrong_answer_rate` | 监控逻辑错误占比 |
| 成本 | `eval/avg_total_gen_tokens` | 平均总生成 tokens（含所有 candidates/rounds） |
| 成本 | `eval/avg_total_judge_time` | 平均总判题时间（含所有 candidates/rounds） |
| 成本 | `eval/throughput` | problems/sec（wall clock） |

**偏好对数据日志**：
| 记录内容 | 数量 | 说明 |
|----------|------|------|
| 偏好对样本 | **100 对** | prompt, chosen, rejected, chosen_pass_ratio, rejected_pass_ratio, Δpass_ratio |
| chosen 错误分布 | 全量统计 | error_type 分布 |
| rejected 错误分布 | 全量统计 | error_type 分布 |

**问答日志记录**：
| 时机 | 数据集 | 记录数量 |
|------|--------|----------|
| 每 500 steps | CodeContests_valid | **30 条** |
| 阶段结束 | CodeContests_test | **50 条** |

**Checkpoint 保存**：
| 保存时机 | 保存内容 | 用途 |
|----------|----------|------|
| 每 500 steps | 完整模型 | 断点恢复 |
| pass_ratio 最高点 | 完整模型 | **用于 GRPO 初始化** |
| 阶段结束 | 完整模型 | 最终 DPO checkpoint |

---

### Phase 3: Online GRPO（主训练阶段）

#### 目的

- **直接优化可验证奖励**：让模型学会"通过更多测试点"
- **核心阶段**：这是整个项目的核心，产出主要实验结果
- **可控成本提升 accepted@1**：在合理的计算预算下最大化性能

#### 做什么

1. **加载 checkpoint**：从 SFT（或 DPO）checkpoint 开始在线 RL 训练
2. **训练数据**：CodeContests_train
   - 建议课程学习：easy 子集 → 全量/更难子集
3. **高频验证**：
   - CodeContests_valid（主验证）
   - MBPP_reg（防漂移）
4. **阶段结束测试**：CodeContests_test + HumanEval
5. **必须完成消融**：Dense vs Sparse reward 对比

> GRPO 关键超参的“可复现最小集”见 [grpo_minimal_hparams.md](./grpo_minimal_hparams.md)。

**Dense vs Sparse Reward 定义**：
| 类型 | 定义 | 值域 | 说明 |
|------|------|------|------|
| **Dense (主推)** | reward = pass_ratio | [0, 1] 连续 | 更丰富的梯度信号 |
| **Sparse (对照)** | reward = 1[accepted] | {0, 1} 离散 | 只有全对才得分 |

#### 必须产出

**主结果**：
| 数据集 | 指标 | 说明 |
|--------|------|------|
| CodeContests_test | accepted@1 | **主指标** |
| CodeContests_test | pass_ratio(mean/p50/p90) | dense 信号质量 |
| CodeContests_test | pass@k (k=5/10, 可选) | 加分项 |
| HumanEval | pass@1 | 对标 & 泛化检测 |

**评测协议要求（必须成本对齐）**：
- 必须同时报告 `EVAL@1`（主结果口径）+ `EVAL@k`（推理预算曲线，可选）+ `EVAL@budget`（强对照，固定 tokens 预算）的结果与成本面板（详见 [eval_protocol.md](./eval_protocol.md)）。

**稳定性指标**：
- 2 seeds mean±std（关键实验必须）
- 训练曲线（保存到 repo）：
  - reward mean / reward std
  - KL（均值/最大值）
  - timeout rate 随训练变化

**成本效率指标**：
- avg_total_gen_tokens / problem（见 [eval_protocol.md](./eval_protocol.md)）
- avg_total_judge_time / problem
- throughput (problems/sec, wall clock)
- cost_per_solved_tokens + cost_per_solved_judge_time

**消融结论**：
| 对比项 | 需要回答的问题 |
|--------|----------------|
| Dense vs Sparse | 哪个收敛更快？方差更低？最终性能更好？ |

#### 可加分点

- **并发判题 + 限流**：吞吐↑、timeout↓
- **缓存/短路策略**：
  - 非代码/空输出/超长输出短路
  - 重复 code hash 缓存
  - 报告 cache hit rate
- **自动化回归系统**：每 N steps 跑固定 valid 子集与 MBPP_reg，生成报表；timeout rate 飙升 early stop
- **Failure-driven data flywheel**：把 Timeout/Syntax 失败回流做 DPO hard negatives 或小比例 format repair SFT
- **Curriculum vs No-Curriculum 消融**：只改采样策略的对比

#### 日志与记录要求

**WandB/训练指标（每 step 记录）**：
| 指标类别 | 指标名 | 说明 |
|----------|--------|------|
| 奖励 | `critic/score/mean` | 平均奖励（pass_ratio） |
| 奖励 | `critic/score/std` | 奖励标准差 |
| 奖励 | `critic/score/max` | 最大奖励 |
| 奖励 | `critic/score/min` | 最小奖励 |
| 优势 | `critic/advantages/mean` | 平均优势值 |
| Actor | `actor/loss` | Actor 损失 |
| Actor | `actor/grad_norm` | Actor 梯度范数 |
| Actor | `actor/clip_ratio` | PPO 裁剪比例 |
| KL | `actor/kl_loss` | KL 散度损失 |
| KL | `actor/kl_mean` | 平均 KL 散度 |
| KL | `actor/kl_max` | 最大 KL 散度 |
| 响应 | `response_length/mean` | 平均响应长度 |
| 响应 | `response_length/max` | 最大响应长度 |

**WandB/评测指标（每 50-100 steps）**：
| 指标类别 | 指标名 | 说明 |
|----------|--------|------|
| 质量 | `eval/accepted_at_1` | **主指标**，按数据集分别记录 |
| 质量 | `eval/pass_ratio_mean` | 平均通过率 |
| 质量 | `eval/pass_ratio_p50` | 通过率中位数 |
| 质量 | `eval/pass_ratio_p90` | 通过率 90 分位 |
| 质量 | `eval/zero_reward_rate` | 零奖励样本比例 |
| 错误分布 | `eval/syntax_error_rate` | 语法错误率 |
| 错误分布 | `eval/runtime_error_rate` | 运行时错误率 |
| 错误分布 | `eval/timeout_rate` | **重点监控**，飙升需 early stop |
| 错误分布 | `eval/wrong_answer_rate` | 答案错误率 |
| 成本 | `eval/avg_total_gen_tokens` | 平均总生成 tokens（含所有 candidates/rounds） |
| 成本 | `eval/avg_total_judge_time` | 平均总判题时间（含所有 candidates/rounds） |
| 成本 | `eval/throughput` | 评测吞吐量 |
| 成本 | `eval/cost_per_solved_tokens` | 每 solved 的 tokens 成本 |
| 成本 | `eval/cost_per_solved_judge_time` | 每 solved 的判题时间成本 |

**WandB/自定义指标（通过 reward_extra_info 收集）**：
| 指标名 | 说明 |
|--------|------|
| `reward/pass_ratio_dist` | pass_ratio 分布直方图 |
| `reward/error_type_dist` | error_type 分布饼图 |

**问答日志记录**：
| 时机 | 数据集 | 记录数量 | 记录内容 |
|------|--------|----------|----------|
| 每 100 steps | CodeContests_valid | **50 条** | prompt, response, pass_ratio, error_type, execution_output |
| 每 100 steps | MBPP_reg | **20 条** | 同上 |
| 每 500 steps | 全量训练样本 | **100 条** | prompt, response, reward, advantage, group_mean |
| 阶段结束 | CodeContests_test | **100 条** | 完整记录 + 对比 SFT 输出 |
| 阶段结束 | HumanEval | **50 条** | 完整记录 |

> 建议分层抽样：
> - 高奖励样本 (pass_ratio > 0.8): 30%
> - 中等奖励样本 (0.3 < pass_ratio ≤ 0.8): 40%
> - 低奖励样本 (pass_ratio ≤ 0.3): 30%

**Checkpoint 保存**：
| 保存时机 | 保存内容 | 用途 |
|----------|----------|------|
| 每 100 steps | 完整模型 + optimizer state | 断点恢复 |
| accepted@1 最高点 | 完整模型 | **最佳模型** |
| pass_ratio 最高点 | 完整模型 | 备选最佳 |
| 阶段结束 | 完整模型 | 最终 GRPO checkpoint |
| 2 seeds 各自 | 完整模型 | 复现性验证 |

> ⚠️ **重要**：GRPO 阶段至少保留 **5 个 checkpoint**
> - 用于分析训练动态
> - 用于 2 seeds 对比
> - 用于 Phase 4 初始化

---

### Phase 4: 多轮修复（可选加分 - Agentic 扩展）

> **定位**：建议在 Phase 3 (GRPO-dense) 跑通后作为加分扩展。

#### 目的

- **展示 Agentic 能力**：展示你具备 tool/exec feedback 驱动的 agentic loop 能力
- **量化收益 vs 成本**：用明确指标证明多轮带来增益，同时能控制成本与不确定性
- **对齐工业范式**：对齐 coding agent / post-training 中常见的"反复跑测试-修复"范式

#### 做什么

**流程 (Mode A: 自动判题)**：

| 轮次 | 输入 | 输出 | 判题 |
|------|------|------|------|
| Round 1 | prompt | code_1 | → accepted_1, pass_ratio_1, error_type_1, feedback_1 |
| Round 2 | prompt + feedback_1 | code_2 | → accepted_2, pass_ratio_2, error_type_2 |
| Round 3 (可选) | prompt + feedback_2 | code_3 | 仅当 Round 2 仍有提升空间时 |

**反馈摘要设计**（强约束、短、结构化，200-400 tokens）：
- `error_type`: SYNTAX/RUNTIME/TIMEOUT/WA
- `pass_ratio`: passed/total
- `failed_cases`: 最多 K 个失败用例
- `traceback_tail`: 最后 N 行错误信息
- `judge_time` (可选)

**Reward 设计（二选一）**：
- **方案 A（推荐）**：最终轮结果 + 轮数成本
  ```
  R = pass_ratio_last + 0.2 * I[accepted_last] - λ * (rounds_used - 1)
  ```
- **方案 B**：每轮即时奖励 + 轮数成本
  ```
  R = Σ_t pass_ratio_t + 0.2 * I[accepted_last] - λ * (rounds_used - 1)
  ```

> λ 建议很小 (0.02-0.05)，只表达额外轮次的真实成本

#### 必须产出

**核心指标（在 CodeContests_valid/test 上）**：
| 指标 | 定义 | 说明 |
|------|------|------|
| accepted@multi | 多轮后是否通过 | **主指标** |
| recovery_rate | Round1 未通过 → 最终通过 的比例 | **最能体现修复价值** |
| Δpass_ratio | pass_ratio_last - pass_ratio_1 | 报 mean/p50/p90 |
| avg_rounds_used | 平均轮数 | 成本指标 |
| total_gen_tokens/prob | 每题总生成 tokens（分解到每轮 + 总和） | 成本指标 |
| total_judge_time/prob | 每题总判题时间（分解到每轮 + 总和） | 成本指标 |
| cost_per_solved_multi_tokens | 多轮每 solved 的 tokens 成本 | 与单轮对比 |
| cost_per_solved_multi_judge_time | 多轮每 solved 的判题时间成本 | 与单轮对比 |
| error transition | error_type 从 Round1→Round2 的转移矩阵 | 分析修复模式 |

**回归监控**：
- MBPP_reg: pass@1 + error breakdown（防退化）
- HumanEval: pass@1（阶段结束，展示泛化不退化）

**必须做的对照**：
| 对比 | 说明 |
|------|------|
| 单轮 (Phase 3) vs 多轮 (Phase 4) | 同 checkpoint、同解码参数、尽量同预算 |

报告：accepted@1 vs accepted@multi、recovery_rate、cost_per_solved_multi_*

#### 可加分点

- **"按需修复"策略**：只对 Round 1 失败样本启用 Round 2，报告节省的成本
- **无效输出短路**：空/非代码/超长输出短路减少无意义判题
- **Case Study**：在 README 展示 3-5 个典型题目从 错误→修复→通过 的全过程
- **2 轮 vs 3 轮消融**：证明边际收益是否值得额外成本

#### 日志与记录要求

**WandB/训练指标（每 step 记录）**：
| 指标类别 | 指标名 | 说明 |
|----------|--------|------|
| 多轮奖励 | `reward/final_reward` | 最终奖励 |
| 多轮奖励 | `reward/round1_pass_ratio` | Round 1 pass_ratio |
| 多轮奖励 | `reward/round2_pass_ratio` | Round 2 pass_ratio |
| 多轮奖励 | `reward/delta_pass_ratio` | pass_ratio 提升 |
| 轮数 | `rounds/avg_rounds_used` | 平均使用轮数 |
| 轮数 | `rounds/round2_triggered_rate` | Round 2 触发率 |
| 轮数 | `rounds/round3_triggered_rate` | Round 3 触发率 |
| 修复 | `recovery/rate` | 修复成功率 |
| 成本 | `cost/output_tokens_round_1_mean` | Round 1 平均输出 tokens |
| 成本 | `cost/output_tokens_round_2_mean` | Round 2 平均输出 tokens |
| 成本 | `cost/total_gen_tokens_mean` | 多轮总生成 tokens 均值 |
| 成本 | `cost/judge_time_round_1_mean` | Round 1 平均判题时间 |
| 成本 | `cost/judge_time_round_2_mean` | Round 2 平均判题时间 |
| 成本 | `cost/total_judge_time_mean` | 多轮总判题时间均值 |

**WandB/评测指标（每 50-100 steps）**：
| 指标类别 | 指标名 | 说明 |
|----------|--------|------|
| 核心 | `eval/accepted_at_multi` | **主指标** |
| 核心 | `eval/recovery_rate` | Round1 失败→最终成功 |
| 核心 | `eval/delta_pass_ratio_mean` | pass_ratio 提升均值 |
| 核心 | `eval/delta_pass_ratio_p50` | pass_ratio 提升中位数 |
| 对比 | `eval/accepted_at_1_vs_multi` | 单轮 vs 多轮对比 |
| 成本 | `eval/cost_per_solved_multi_tokens` | 多轮每 solved 的 tokens 成本 |
| 成本 | `eval/cost_per_solved_multi_judge_time` | 多轮每 solved 的判题时间成本 |

**WandB/错误转移分析**：
| 指标名 | 说明 |
|--------|------|
| `error_transition/syntax_to_*` | Syntax Error 转移去向 |
| `error_transition/runtime_to_*` | Runtime Error 转移去向 |
| `error_transition/timeout_to_*` | Timeout 转移去向 |
| `error_transition/wa_to_*` | Wrong Answer 转移去向 |

**问答日志记录（多轮格式）**：
| 时机 | 数据集 | 记录数量 | 记录内容 |
|------|--------|----------|----------|
| 每 100 steps | CodeContests_valid | **50 条** | 完整多轮对话：prompt, code_1, feedback_1, code_2, [feedback_2, code_3] |
| 阶段结束 | CodeContests_test | **100 条** | 同上 + 每轮的 pass_ratio, error_type, execution_output |

**详细问答日志格式**：
```json
{
  "problem_id": "xxx",
  "prompt": "...",
  "rounds": [
    {
      "round": 1,
      "code": "...",
      "pass_ratio": 0.3,
      "error_type": "wrong_answer",
      "failed_cases": [...],
      "execution_output": "..."
    },
    {
      "round": 2,
      "feedback_input": "...",
      "code": "...",
      "pass_ratio": 0.8,
      "error_type": "wrong_answer",
      "failed_cases": [...]
    }
  ],
  "final_accepted": false,
  "recovery": false,
  "total_tokens": 1234,
  "total_judge_time": 5.6
}
```

> **Case Study 要求**：挑选 **5 条典型修复成功** + **3 条典型修复失败** 的完整日志

**Checkpoint 保存**：
| 保存时机 | 保存内容 | 用途 |
|----------|----------|------|
| 每 100 steps | 完整模型 | 断点恢复 |
| accepted@multi 最高点 | 完整模型 | 最佳多轮模型 |
| recovery_rate 最高点 | 完整模型 | 最佳修复能力模型 |
| 阶段结束 | 完整模型 | 最终 Phase 4 checkpoint |

---

## 三、实验矩阵与优先级

### 必须实验（5 组）

| 编号 | 配置 | 主要对比点 | 优先级 |
|------|------|-----------|--------|
| 1 | Base (无训练) | 基线 | ★★★ |
| 2 | SFT | exec_success_rate ↑ | ★★★ |
| 3 | SFT + best-of-n (inference only) | **强对照 baseline**：控制“只靠更多采样/更多 tokens” | ★★★ |
| 4 | SFT + GRPO (dense) | accepted@1 ↑，**主结果** | ★★★ |
| 5 | SFT + GRPO (sparse) | Dense vs Sparse 消融 | ★★★ |

### 强烈建议实验（2 组）

| 编号 | 配置 | 主要对比点 | 优先级 |
|------|------|-----------|--------|
| 6 | SFT → DPO → GRPO | DPO warm-start 效果 | ★★ |
| 7 | Curriculum vs No-Curriculum | 采样策略消融 | ★★ |

### 可选加分实验（1 组）

| 编号 | 配置 | 主要对比点 | 优先级 |
|------|------|-----------|--------|
| 8 | 单轮 vs 多轮修复 | 多轮收益 vs 成本 | ★ |

---

## 四、最终交付物清单

### 必须产出

| 产出物 | 内容 | 格式 |
|--------|------|------|
| **主表** | Base / SFT / SFT-best-of-n / GRPO 在 CodeContests_test + HumanEval + MBPP_reg 的指标（至少含 `EVAL@1` + `EVAL@budget`） | CSV/表格 |
| **学习曲线** | accepted@1, pass_ratio, reward, KL (2 seeds + 均值) | 图表 |
| **消融表** | Dense vs Sparse; DPO vs None | 表格 |
| **错误分布** | Syntax/Runtime/Timeout/WA 随阶段变化 | 堆叠图 |
| **成本面板** | throughput, avg_total_gen_tokens, p50/p95 total_judge_time, cost_per_solved_*, timeout rate | 表格 |

### 简历可写目标（目标区间，跑完后用真实数值替换）

> 说明：以下是“目标区间/验收标准”，不是性能承诺；以 `EVAL@1` 为主口径，并在报告中同步给出 `EVAL@budget` 的成本对齐对照（见 [eval_protocol.md](./eval_protocol.md)）。

| 阶段 | 建议写进简历的主句 | 目标区间（示例） |
|------|-------------------|------------------|
| Phase 1（SFT） | “显著提升可执行率，减少低级错误” | exec_success_rate **+20～40pp**；syntax_error_rate **-30～60%（相对）**；HumanEval pass@1 **不退化**（±1～2pp） |
| Phase 3（GRPO-dense） | “在可验证奖励下提升 accepted@1，并给出成本/稳定性” | CodeContests_test accepted@1 **+3～10pp**（vs SFT@1）；pass_ratio_mean **+0.05～0.15**；2 seeds std **可控**（例如 <1～2pp） |
| Phase 3（dense vs sparse） | “完成关键消融，解释收敛速度/方差差异” | dense 相比 sparse：更快到达同等 accepted@1，或 reward_std 更低（任选其一量化） |
| Phase 4（多轮修复） | “用执行反馈驱动修复，量化收益 vs 成本” | accepted@multi 相比 accepted@1 **+5～15pp**；recovery_rate **20～40%**；avg_rounds_used **≤1.3～1.6**；成本在预算口径下可解释 |

### 可选产出

| 产出物 | 内容 |
|--------|------|
| Failure Analysis | 典型失败 case + 修复策略 |
| Phase 4 对照页 | 单轮 vs 多轮完整对比 |
| Case Study | 3-5 个典型题目的全过程日志 |
| 偏好数据报告 | DPO 偏好对质量分析 |

---

## 五、设计原则与约束

### 数据隔离原则

| 原则 | 说明 |
|------|------|
| HumanEval: Test-only | 禁止用于训练、构造偏好对、选择超参、早停 |
| CodeContests_test: Test-only | 禁止训练与调参 |
| CodeContests_valid: Dev/Val only | 仅用于 checkpoint 选择/早停/超参对比 |

### 数据去重与泄漏防护（必做）

CodeContests 类题库存在重复题/近似题风险，必须做到“可审计的数据隔离”：

- split 内去重 + 跨 split overlap 检查（train/valid/test 交集必须为 0）
- HumanEval/MBPP 与训练数据的泄漏检查（交集必须为 0）
- 固定 manifest（题目 ID 列表 + prompt hash）进入 repo，保证可复现

详见 [data_governance.md](./data_governance.md)。

### 解码协议统一

| 场景 | 协议 | 参数 |
|------|------|------|
| 训练 rollout | 采样 | temperature, top_p 固定并记录 |
| 评测 | greedy 或固定低温 | 固定并记录 |

### 评测协议与成本对齐（必须）

为避免“只是采样更多 / 用更多 tokens”造成的虚假增益，本项目统一采用三种评测口径（见 [eval_protocol.md](./eval_protocol.md)）：

- `EVAL@1`：best-of-1（主结果口径，用于 Base/SFT/GRPO 公平对比）
- `EVAL@k`：best-of-k（推理预算曲线，必须同步汇报 tokens 与判题时间）
- `EVAL@budget`：固定每题 tokens 预算（强对照；best-of-n / 多轮修复必须纳入该预算口径，否则对比无效）

指标定义与口径（accepted/pass、pass_ratio、exec_success、error breakdown、成本面板等）详见 [metric_templates.md](./metric_templates.md)。建议不同 protocol 的评测用独立 eval run 或不同前缀记录，避免指标混淆。

### Guardrails（防 reward hacking / 质量控制）

RLVR coding 容易出现超长输出、无效输出、重复、以及环境利用等问题，建议把 Guardrails 作为默认配置并写入报告：

- 输出短路（空/非代码/超长）、重复 code hash 缓存、timeout 防爆
- 沙盒约束（禁网/限文件/限系统调用/固定 time&mem limit）
- 多轮反馈脱敏（只回传结构化摘要，避免泄漏隐藏测试）

详见 [guardrails.md](./guardrails.md)。

### 资源与预算（GPU）

本项目默认以 7B 模型为目标规模，Phase 0/4 主要是推理 + 判题，Phase 1/3 是训练主成本；建议按“经济适用”优先选择 `8×4090`，必要时用 `4×H100` 缩短墙钟时间，或用高显存卡降低工程复杂度。详见 [resource_plan.md](./resource_plan.md)。

### 可复现性要求

- [ ] 固定回归集题号（MBPP_reg + CodeContests_dev subset）
- [ ] 记录 seed、checkpoint、评测参数、题目 ID 列表版本
- [ ] 关键实验 2 seeds，报告 mean ± std

---

## 六、日志与 Checkpoint 总览

### 6.1 WandB 指标记录总览

| Phase | 训练指标 | 评测指标 | 记录频率 |
|-------|----------|----------|----------|
| Phase 0 | 无 | 质量/成本/错误分布 | 一次性 |
| Phase 1 | loss, grad_norm, lr | exec_success_rate, error rates | 每 step / 每 100-500 steps |
| Phase 2 | dpo_loss, reward_margin | pass_ratio, zero_reward_rate | 每 step / 每 100-500 steps |
| Phase 3 | score, advantage, actor_loss, kl | accepted@1, pass_ratio, error rates | 每 step / 每 50-100 steps |
| Phase 4 | multi-round reward, recovery | accepted@multi, recovery_rate | 每 step / 每 50-100 steps |

### 6.2 问答日志记录总览

| Phase | 数据集 | 训练中记录 | 阶段结束记录 | 总量估计 |
|-------|--------|------------|--------------|----------|
| Phase 0 | 4 个数据集 | - | 120 条 | ~120 条 |
| Phase 1 | valid + reg | 50 条/500 steps | 80 条 | ~200 条 |
| Phase 2 | valid + 偏好对 | 30 条/500 steps + 100 对 | 50 条 | ~200 条 |
| Phase 3 | valid + reg + train | 170 条/100 steps | 150 条 | ~500 条 |
| Phase 4 | valid + test | 50 条/100 steps | 100 条 | ~300 条 |

> **总计**：整个实验流程约 **1000-1500 条** 详细问答日志

### 6.3 Checkpoint 保存策略总览

| Phase | 需要保存 | 保存数量 | 关键 Checkpoint |
|-------|----------|----------|-----------------|
| Phase 0 | ❌ | 0 | 使用原始 Base 模型 |
| Phase 1 | ✅ | ≥3 | exec_success_rate 最高点 → GRPO 初始化 |
| Phase 2 | ✅ | ≥3 | pass_ratio 最高点 → GRPO 初始化 |
| Phase 3 | ✅ | ≥5 | accepted@1 最高点 → 主结果/Phase 4 |
| Phase 4 | ✅ | ≥4 | accepted@multi 最高点 → 最终结果 |

### 6.4 日志存储建议

```
logs/
├── wandb/                    # WandB 自动管理
├── qa_logs/
│   ├── phase0/
│   │   ├── codecontests_valid_50.jsonl
│   │   ├── codecontests_test_30.jsonl
│   │   ├── humaneval_20.jsonl
│   │   └── mbpp_reg_20.jsonl
│   ├── phase1/
│   │   ├── step_500_valid_30.jsonl
│   │   ├── step_1000_valid_30.jsonl
│   │   └── final_test_80.jsonl
│   ├── phase3/
│   │   ├── step_100_valid_50.jsonl
│   │   ├── step_100_train_sample_100.jsonl
│   │   └── final_test_150.jsonl
│   └── phase4/
│       ├── multiround_valid_50.jsonl
│       └── case_study_8.jsonl
└── checkpoints/
    ├── sft/
    │   ├── step_500/
    │   ├── best_exec_success/
    │   └── final/
    ├── grpo/
    │   ├── step_100/
    │   ├── best_accepted/
    │   ├── best_pass_ratio/
    │   └── final/
    └── phase4/
        ├── best_accepted_multi/
        └── final/
```

---

## 附录 A：技术实现映射

### verl 核心组件

| 功能 | 文件路径 | 说明 |
|------|----------|------|
| SFT Trainer | `verl/trainer/sft_trainer.py` | 监督微调训练器 |
| GRPO 优势估计 | `verl/trainer/ppo/core_algos.py` | `compute_grpo_outcome_advantage()` |
| PPO 训练器 | `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer` |
| 奖励管理器 | `verl/workers/reward_manager/naive.py` | `NaiveRewardManager` |
| 奖励路由 | `verl/utils/reward_score/__init__.py` | `default_compute_score()` |
| SandboxFusion 集成 | `verl/utils/reward_score/sandbox_fusion/__init__.py` | `compute_score()` |
| 正确性检查 | `verl/utils/reward_score/sandbox_fusion/utils.py` | `check_correctness()` |

### SandboxFusion 返回值

| 字段 | 类型 | 说明 |
|------|------|------|
| `results[i]` | `True/False/-1/-2/-3/-4` | 单个测试用例结果 |
| `metadata[i]["status"]` | str | `"success"/"wrong_answer"/"runtime_error"/"timeout"/"compile_error"` |
| `metadata[i]["duration"]` | float | 执行时间（秒） |

### 算法配置关键项

```yaml
algorithm:
  adv_estimator: grpo              # GRPO 优势估计器
  norm_adv_by_std_in_grpo: True    # 标准 GRPO 归一化

actor_rollout_ref:
  actor:
    use_kl_loss: True
    kl_loss_coef: 0.001
    clip_ratio: 0.2
  rollout:
    n: 5                           # 每 prompt 采样数 (GRPO 关键)
    temperature: 0.8
```

---

## 附录 B：指标定义详表

详见 [metric_templates.md](./metric_templates.md)
