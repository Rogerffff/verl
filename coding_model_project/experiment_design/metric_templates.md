# RLVR Coding Model - 指标表格模板

本文档提供每个阶段（Phase 0/1/2/3/4）的指标表格模板，包含质量、稳定性、成本效率、系统可靠性四类指标。评测协议与成本对齐见 [eval_protocol.md](./eval_protocol.md)。

---

## 0) 评测协议与通用指标定义（所有 Phase 共用）

### 0.1 评测协议（Protocol）

评测必须在以下协议中至少覆盖 `EVAL@1`，并对涉及“更多采样/多轮”的方法额外覆盖 `EVAL@budget`（强对照）：

- `EVAL@1`：best-of-1（n=1）
- `EVAL@k`：best-of-k（n=k）
- `EVAL@budget`：固定每题预算（`gen_tokens_budget_per_problem` 必选；`judge_time_budget_per_problem` 可选）

同一组指标在不同协议下都需要输出，差别仅在 **candidate 选择策略** 与 **总成本统计方式**。

### 0.2 关键对象（用于定义口径）

- **candidate**：一次生成得到的 1 份候选代码（以及其对应的判题结果）。
- **selected candidate**：在某个协议下最终用于统计质量/错误分布的候选代码：
  - `EVAL@1`：唯一 candidate
  - `EVAL@k`：k 个 candidate 中按 `pass_ratio` 选最优（如存在 `pass_ratio==1` 必选其一）
  - `EVAL@budget`：预算内尝试的 candidates/rounds 中按 `pass_ratio` 选最优（或按你的实现固定用“最终轮”，但必须写清）

### 0.3 Outcome / error_type（按“题目/问题”聚合）

对每个 **candidate 在一个 problem 上**定义一个 `final_status`（按优先级归并）：

1. `compile_error` / `syntax_error`
2. `runtime_error`
3. `timeout`
4. `wrong_answer`（可执行但至少 1 个用例失败）
5. `success`（全部用例通过；等价于 accepted/pass）
6. `api_error` / `sandbox_error` / `other_error`（如果 SandboxFusion 返回了更细的状态，按实际落盘）

**error breakdown**：`final_status` 的分布（Count / Total），通常至少输出：

- `accepted_rate`（`final_status == success`）
- `wrong_answer_rate`
- `syntax_error_rate`
- `runtime_error_rate`
- `timeout_rate`
-（可选）`api_error_rate`、`sandbox_error_rate`

**exec_success_rate**：`final_status ∈ {success, wrong_answer}` 的比例（即“能跑完/能执行”，不要求正确）。

> 说明：error breakdown 与 exec_success_rate 默认以 **selected candidate** 为口径；系统可靠性指标（api/sandbox）也可以额外提供按 candidate 统计的版本（见各 Phase 的系统可靠性表）。

### 0.4 质量指标（Quality）

**CodeContests（多测试用例）**

- `accepted@1 / accepted@k / accepted@budget`：在对应协议下 `final_status == success` 的问题比例
- `pass_ratio_mean/p50/p90`：对 selected candidate 的 `pass_ratio` 做统计（`pass_ratio = passed/total`，值域 [0,1]）

**HumanEval / MBPP_reg（binary 为主）**

- `pass@1 / pass@k / pass@budget`：在对应协议下通过的比例（如你用 SandboxFusion 统一判题，可直接复用 `final_status == success` 口径）
-（可选）如能拿到分用例信息，也可报 `pass_ratio_*`，但主口径仍是 pass@k

**命名建议（报告 vs 日志）**

- 报告中使用：`accepted@{1,k,budget}` / `pass@{1,k,budget}`、`pass_ratio_mean/p50/p90`
- 日志/WandB 中推荐使用：`accepted_at_1`、`accepted_at_k`、`accepted_at_budget`、`pass_at_1`…（用 `at` 替代 `@`）

### 0.5 成本指标（Cost）

按照 [eval_protocol.md](./eval_protocol.md) 的口径，必须记录：

- `output_tokens_selected`：selected candidate 的输出 tokens
- `total_gen_tokens`：该题在该协议下所有 candidates/rounds 的输出 tokens 之和
- `judge_time_selected`：selected candidate 的判题时间（建议为该 candidate 的所有测试用例 `duration` 求和）
- `total_judge_time`：该题在该协议下所有 candidates/rounds 的判题时间之和
- `throughput`：`total_problems / wall_clock_time`（端到端）

成本对齐常用派生指标：

- `cost_per_solved_tokens = Σ(total_gen_tokens) / solved_count`
- `cost_per_solved_judge_time = Σ(total_judge_time) / solved_count`

> 注意：不要把“tokens”和“秒”直接相加成一个标量；如需统一成美元成本，必须引入显式换算系数并写入报告。

---

## Phase 0: Baseline 指标表

### 质量指标

| 指标名 | 定义/公式 | 采样方式 | 数据源 | 统计方法 |
|--------|----------|----------|--------|----------|
| accepted@1 | `EVAL@1` 下 `final_status == success` 的比例 | 按问题 | SandboxFusion/判题汇总 | Count / Total |
| pass_ratio_mean | `mean(pass_ratio_selected)` | 按问题 | `compute_score()` 的 `pass_ratio` | Mean |
| pass_ratio_p50 | `p50(pass_ratio_selected)` | 按问题 | `pass_ratio` 分布 | 50th percentile |
| pass_ratio_p90 | `p90(pass_ratio_selected)` | 按问题 | `pass_ratio` 分布 | 90th percentile |
| exec_success_rate | `final_status ∈ {success, wrong_answer}` 的比例 | 按问题 | `final_status` | Count / Total |

### 错误分布指标

| 指标名 | 定义 | 采样方式 | 数据源 | 统计方法 |
|--------|------|----------|--------|----------|
| syntax_error_rate | `final_status == compile_error` 的比例 | 按问题 | `final_status` | Count / Total |
| runtime_error_rate | `final_status == runtime_error` 的比例 | 按问题 | `final_status` | Count / Total |
| timeout_rate | `final_status == timeout` 的比例 | 按问题 | `final_status` | Count / Total |
| wrong_answer_rate | `final_status == wrong_answer` 的比例 | 按问题 | `final_status` | Count / Total |

### 成本效率指标

| 指标名 | 定义 | 采样方式 | 数据源 | 统计方法 |
|--------|------|----------|--------|----------|
| avg_total_gen_tokens | `mean(total_gen_tokens)` | 按问题 | tokens 统计 | Mean |
| avg_total_judge_time | `mean(total_judge_time)` | 按问题 | sandbox `duration` 汇总 | Mean |
| p95_total_judge_time | `p95(total_judge_time)` | 按问题 | sandbox `duration` 汇总 | 95th percentile |
| throughput | `total_problems / wall_clock_time` | 批量 | 端到端 wall clock | 计算值 |
| cost_per_solved_tokens | `Σ(total_gen_tokens) / solved_count` | 按 solved | tokens 统计 | 计算值 |
| cost_per_solved_judge_time | `Σ(total_judge_time) / solved_count` | 按 solved | sandbox `duration` 汇总 | 计算值 |

### 系统可靠性指标

| 指标名 | 定义 | 采样方式 | 数据源 | 统计方法 |
|--------|------|----------|--------|----------|
| api_error_rate | `final_status == api_error` 的比例（或按 candidate 统计） | 按问题/candidate | `final_status` | Count / Total |
| sandbox_error_rate | `final_status == sandbox_error` 的比例（或按 candidate 统计） | 按问题/candidate | `final_status` | Count / Total |

---

## Phase 1: SFT 指标表

### 训练指标

| 指标名 | 定义 | 采样方式 | verl 数据源 | 频率 |
|--------|------|----------|-------------|------|
| train/loss | SFT 交叉熵损失 | 每 step | `sft_trainer.py` 训练循环 | 每 step |
| train/grad_norm | 梯度范数 | 每 step | `sft_trainer.py` | 每 step |
| train/lr | 当前学习率 | 每 step | optimizer | 每 step |
| val/loss | 验证集损失 | 验证时 | `sft_trainer.py` 验证循环 | 每 N steps |

### 质量指标（评测时）

| 指标名 | 预期变化 | 采样方式 | 数据源 | 对比 |
|--------|----------|----------|--------|------|
| exec_success_rate | ↑ 显著 | 按问题 | SandboxFusion | vs Phase 0 |
| syntax_error_rate | ↓ 显著 | 按问题 | `final_status` | vs Phase 0 |
| runtime_error_rate | ↓ | 按问题 | `final_status` | vs Phase 0 |
| timeout_rate | ↓ | 按问题 | `final_status` | vs Phase 0 |
| accepted@1 | ↑ 小幅或持平 | 按问题 | `final_status == success` | vs Phase 0 |
| pass_ratio_mean | ↑ 小幅 | 按问题 | `pass_ratio` | vs Phase 0 |
| pass_ratio_p50/p90 | ↑ 或持平 | 按问题 | `pass_ratio` 分布 | vs Phase 0 |
| wrong_answer_rate | ↓ 或持平 | 按问题 | `final_status` | vs Phase 0 |

### 稳定性指标

| 指标名 | 定义 | 数据源 | 要求 |
|--------|------|--------|------|
| loss_curve | loss 随 step 变化 | train/loss | 平滑下降 |
| val_loss_curve | 验证 loss 变化 | val/loss | 不过拟合 |

### 成本效率指标

| 指标名 | 定义 | 采样方式 | 统计方法 |
|--------|------|----------|----------|
| training_time | 总训练时间 | 全局 | 记录值 |
| samples_per_second | 每秒训练样本数 | 批量 | 计算值 |

---

## Phase 2: DPO 指标表 (如实施)

### 数据构建指标

| 指标名 | 定义 | 采样方式 | 数据源 |
|--------|------|----------|--------|
| total_pairs | 构建的偏好对总数 | 全局 | 数据构建脚本 |
| delta_pass_ratio_mean | chosen 与 rejected 的 pass_ratio 差异均值 | 按对 | 偏好对数据 |
| delta_pass_ratio_dist | 差异分布 (P25/P50/P75) | 按对 | 偏好对数据 |
| chosen_error_dist | chosen 样本的错误类型分布 | 按样本 | 偏好对数据 |
| rejected_error_dist | rejected 样本的错误类型分布 | 按样本 | 偏好对数据 |

### 训练指标

| 指标名 | 定义 | 数据源 | 频率 |
|--------|------|--------|------|
| dpo_loss | DPO 损失 | TRL/自定义训练 | 每 step |
| chosen_log_prob | chosen 样本的 log prob | 训练循环 | 每 step |
| rejected_log_prob | rejected 样本的 log prob | 训练循环 | 每 step |

### 质量指标（评测时）

| 指标名 | 预期变化 | 对比 |
|--------|----------|------|
| accepted@1 | ↑ 或持平 | vs SFT |
| pass_ratio_mean | ↑ | vs SFT |
| pass_ratio_p50/p90 | ↑ | vs SFT |
| exec_success_rate | ↑ 或持平 | vs SFT |
| error breakdown | 语法/运行时/超时/WA 分布 | vs SFT |
| zero_reward_rate | ↓ | vs SFT |

### 成本效率指标

| 指标名 | 定义 | 说明 |
|--------|------|------|
| pair_construction_cost | 构建偏好对的总判题次数 | 每题平均 K 次采样 |
| dpo_training_time | DPO 训练时间 | 全局 |

---

## Phase 3: GRPO 指标表

### 训练指标（每 step）

| 指标名 | 定义 | verl 字段 | 记录频率 |
|--------|------|-----------|----------|
| critic/score/mean | 平均奖励 (sequence-level) | `batch["token_level_scores"].sum(-1).mean()` | 每 step |
| critic/score/max | 最大奖励 | `.max()` | 每 step |
| critic/score/min | 最小奖励 | `.min()` | 每 step |
| critic/rewards/mean | 含 KL 惩罚的平均奖励 | `batch["token_level_rewards"]` | 每 step |
| critic/advantages/mean | 平均优势估计 | `batch["advantages"].mean()` | 每 step |
| actor/loss | PPO actor 损失 | `actor_output.meta_info["loss"]` | 每 step |
| actor/grad_norm | actor 梯度范数 | `actor_output.meta_info["grad_norm"]` | 每 step |
| actor/clip_ratio | PPO 裁剪比例 | `compute_policy_loss_vanilla()` 返回 | 每 step |
| actor/kl_loss | KL 散度损失 | `kl_loss_coef * kl` | 每 step |
| actor/entropy | 策略熵 | 如启用 | 每 step |
| response_length/mean | 平均响应长度 | `response_mask.sum(-1).mean()` | 每 step |
| response_length/clip_ratio | 达到最大长度的比例 | 统计 | 每 step |

### 自定义指标（通过 reward_extra_info）

| 指标名 | 定义 | 实现方式 | 说明 |
|--------|------|----------|------|
| pass_ratio_mean | 平均通过率 | `compute_score()` 返回 dict | 需扩展返回值 |
| accepted_rate_train | 训练样本中 `pass_ratio==1` 的比例 | `score == 1.0` | 训练监控（非主结果） |
| timeout_rate_train | 训练中超时比例 | 统计 `final_status/metadata["status"]` | 训练监控（重点） |
| zero_reward_rate | 零奖励样本比例 | `(score == 0).mean()` | 需计算 |

### 稳定性指标

| 指标名 | 定义 | 数据源 | 要求 |
|--------|------|--------|------|
| reward_mean_curve | 奖励均值随训练变化 | critic/score/mean | 2 seeds, mean ± std |
| reward_std_curve | 奖励方差变化 | critic/score/std | 2 seeds |
| kl_curve | KL 散度变化 | actor/kl_loss | 2 seeds, 监控爆炸 |
| timeout_rate_curve | 超时率变化 | 自定义 | 监控异常上升 |
| gradient_norm_curve | 梯度范数变化 | actor/grad_norm | 监控爆炸 |

### 质量指标（验证/测试，按评测协议输出）

评测协议与成本口径统一遵循 [eval_protocol.md](./eval_protocol.md)。每次评测 run 需要输出的指标集合见本文档 **0.3～0.5**（accepted/pass、pass_ratio、exec_success、error breakdown、成本面板等）。

| 协议 | 数据集 | 频率 | 说明 |
|------|--------|------|------|
| `EVAL@1` | CodeContests_valid | 每 N steps | checkpoint 选择/早停 |
| `EVAL@1` | MBPP_reg | 每 N steps | 回归监控 |
| `EVAL@budget` | CodeContests_valid | 阶段末 + 最佳 ckpt | 成本对齐强对照（必做） |
| `EVAL@1` | CodeContests_test | 阶段末 | test-only 主口径 |
| `EVAL@budget` | CodeContests_test | 阶段末 | test-only 成本对齐对照（必做） |
| `EVAL@1` | HumanEval | 阶段末 | test-only 对标 |
| `EVAL@k`（可选） | CodeContests_test | 阶段末 | 推理预算曲线（k∈{5,10}） |

### 消融实验指标

| 对比项 | 主指标 | 次要指标 | 统计方法 |
|--------|--------|----------|----------|
| Dense vs Sparse | Δaccepted@1 | 收敛 step 数, reward_std | 2 seeds, t-test |
| DPO vs No-DPO | Δaccepted@1 | 初始 reward, 收敛速度 | 2 seeds |
| Curriculum vs None | Δaccepted@1 | 训练稳定性 | 2 seeds |

### 成本效率指标

| 指标名 | 定义 | 采样方式 | 说明 |
|--------|------|----------|------|
| avg_total_gen_tokens | `mean(total_gen_tokens)` | 按 prompt | rollout 阶段（包含所有 samples） |
| avg_total_judge_time | `mean(total_judge_time)` | 按 prompt | reward 阶段（包含所有 samples） |
| throughput_train | 训练吞吐量 (problems/sec) | 批量 | 端到端 |
| cost_per_solved_tokens_train | `Σ(total_gen_tokens)/solved_count` | 按 solved | tokens 口径 |
| cost_per_solved_judge_time_train | `Σ(total_judge_time)/solved_count` | 按 solved | judge_time 口径 |
| total_training_time | 总训练时间 | 全局 | 记录 |
| total_gpu_hours | GPU 时间 | 全局 | 记录 |

### 系统可靠性指标

| 指标名 | 定义 | 数据源 | 阈值 |
|--------|------|--------|------|
| sandbox_timeout_rate | SandboxFusion 超时比例 | API 调用 | < 5% |
| sandbox_error_rate | SandboxFusion 错误比例 | API 调用 | < 1% |
| checkpoint_save_success | checkpoint 保存成功率 | 训练循环 | 100% |

---

## Phase 4: 多轮修复指标表

### 核心指标

| 指标名 | 定义 | 对比项 | 统计方法 |
|--------|------|--------|----------|
| accepted@multi | 多轮后全部通过比例 | vs accepted@1 | Count / Total |
| recovery_rate | (Round1 失败 → 最终成功) / Round1 失败数 | 多轮独有 | Count / Failed_R1 |
| delta_pass_ratio_mean | Σ(pass_ratio_last - pass_ratio_1) / N | 多轮独有 | Mean |
| delta_pass_ratio_p50 | delta 的中位数 | 多轮独有 | 50th percentile |
| delta_pass_ratio_p90 | delta 的 90 分位 | 多轮独有 | 90th percentile |
| avg_rounds_used | 平均使用轮数 | 成本 | Mean |

### 轮次成本分解

| 指标名 | 定义 | 按轮分解 |
|--------|------|----------|
| output_tokens_round_1 | 第 1 轮输出 tokens | Round 1 |
| output_tokens_round_2 | 第 2 轮输出 tokens | Round 2 |
| total_gen_tokens | 总生成 tokens（多轮所有输出求和） | Sum |
| judge_time_round_1 | 第 1 轮判题时间 | Round 1 |
| judge_time_round_2 | 第 2 轮判题时间 | Round 2 |
| total_judge_time | 总判题时间（多轮所有判题求和） | Sum |
| cost_per_solved_multi_tokens | `Σ(total_gen_tokens)/solved_count` | vs 单轮 |
| cost_per_solved_multi_judge_time | `Σ(total_judge_time)/solved_count` | vs 单轮 |

### 错误转移分析

| 分析项 | 定义 | 数据结构 |
|--------|------|----------|
| error_transition_matrix | Round1 → Round2 错误类型转移 | 5x5 矩阵 (Success/Syntax/Runtime/Timeout/WA) |
| syntax_to_success_rate | Syntax → Success 的比例 | 转移矩阵提取 |
| timeout_to_success_rate | Timeout → Success 的比例 | 转移矩阵提取 |
| wa_to_success_rate | WrongAnswer → Success 的比例 | 转移矩阵提取 |

### 对照实验指标

| 对比项 | 主指标 | 成本指标 | 结论关注点 |
|--------|--------|----------|-----------|
| 单轮 vs 2轮 | Δaccepted, recovery_rate | cost_per_solved_* | 收益是否值得成本 |
| 2轮 vs 3轮 | Δaccepted (边际) | 边际 cost | 边际收益递减点 |

---

## 汇总：结果报告表格模板

### 主结果表（`EVAL@1`，必须）

| 方法 | CodeContests_test accepted@1 | CodeContests_test pass_ratio (mean/p50/p90) | HumanEval pass@1 | MBPP_reg pass@1 |
|------|----------------------------|---------------------------------------------|------------------|-----------------|
| Base | X.X% | X.X/X.X/X.X | X.X% | X.X% |
| SFT | X.X% | X.X/X.X/X.X | X.X% | X.X% |
| GRPO-dense | X.X% ± X.X% | X.X/X.X/X.X | X.X% | X.X% |
| GRPO-sparse | X.X% ± X.X% | X.X/X.X/X.X | X.X% | X.X% |

> 说明：`EVAL@1` 只比较 best-of-1；`best-of-n` 与 `多轮修复` 的主结果请在 `EVAL@budget`（强对照）或 `EVAL@k`（预算曲线）下汇报。

### 成本对齐对照表（`EVAL@budget`，必须）

| 方法 | 预算（示例） | CodeContests_test accepted@budget | CodeContests_test pass_ratio (mean/p50/p90) | HumanEval pass@budget | MBPP_reg pass@budget |
|------|-------------|-----------------------------------|---------------------------------------------|-----------------------|----------------------|
| SFT + best-of-n（inference only） | gen_tokens=____ | X.X% | X.X/X.X/X.X | X.X% | X.X% |
| GRPO-dense | gen_tokens=____ | X.X% | X.X/X.X/X.X | X.X% | X.X% |
| Phase 4 多轮修复 | gen_tokens=____ | X.X% | X.X/X.X/X.X | X.X% | X.X% |

### 错误分布表（`EVAL@1`，必须）

| 方法 | syntax_error_rate | runtime_error_rate | timeout_rate | wrong_answer_rate | exec_success_rate |
|------|--------|---------|---------|-------------|--------------|
| Base | X.X% | X.X% | X.X% | X.X% | X.X% |
| SFT | X.X% | X.X% | X.X% | X.X% | X.X% |
| GRPO-dense | X.X% | X.X% | X.X% | X.X% | X.X% |

### 错误分布表（`EVAL@budget`，必须）

| 方法 | syntax_error_rate | runtime_error_rate | timeout_rate | wrong_answer_rate | exec_success_rate |
|------|-------------------|--------------------|--------------|-------------------|------------------|
| SFT + best-of-n（inference only） | X.X% | X.X% | X.X% | X.X% | X.X% |
| GRPO-dense | X.X% | X.X% | X.X% | X.X% | X.X% |
| Phase 4 多轮修复 | X.X% | X.X% | X.X% | X.X% | X.X% |

### 成本效率表（`EVAL@1`，必须）

| 方法 | avg_total_gen_tokens | avg_total_judge_time | throughput | cost_per_solved_tokens | cost_per_solved_judge_time |
|------|----------------------|----------------------|------------|-------------------------|----------------------------|
| Base | X | X.Xs | X prob/s | X | X |
| SFT | X | X.Xs | X prob/s | X | X |
| GRPO-dense | X | X.Xs | X prob/s | X | X |

### 成本效率表（`EVAL@budget`，必须）

| 方法 | avg_total_gen_tokens | avg_total_judge_time | throughput | cost_per_solved_tokens | cost_per_solved_judge_time |
|------|----------------------|----------------------|------------|-------------------------|----------------------------|
| SFT + best-of-n（inference only） | X | X.Xs | X prob/s | X | X |
| GRPO-dense | X | X.Xs | X prob/s | X | X |
| Phase 4 多轮修复 | X | X.Xs | X prob/s | X | X |

### 稳定性表（2 seeds，`EVAL@1`）

| 方法 | accepted@1 (seed1) | accepted@1 (seed2) | mean ± std |
|------|-------------------|-------------------|------------|
| GRPO-dense | X.X% | X.X% | X.X% ± X.X% |
| GRPO-sparse | X.X% | X.X% | X.X% ± X.X% |

### 推理预算曲线（`EVAL@k`，可选）

| 方法 | k | CodeContests_test accepted@k | pass_ratio (mean/p50/p90) | avg_total_gen_tokens | avg_total_judge_time |
|------|---|------------------------------|---------------------------|----------------------|----------------------|
| SFT | 5/10 | X.X% | X.X/X.X/X.X | X | X.Xs |
| GRPO-dense | 5/10 | X.X% | X.X/X.X/X.X | X | X.Xs |

---

## 数据源总结

### verl 日志字段

| 字段路径 | 内容 | 获取方式 |
|----------|------|----------|
| `batch["token_level_scores"]` | token 级别奖励 | DataProto |
| `batch["token_level_rewards"]` | 含 KL 的奖励 | DataProto |
| `batch["advantages"]` | 优势估计 | DataProto |
| `batch.non_tensor_batch["reward_extra_info"]` | 自定义奖励信息 | NaiveRewardManager |
| `actor_output.meta_info["loss"]` | actor 损失 | 训练输出 |
| `actor_output.meta_info["grad_norm"]` | 梯度范数 | 训练输出 |

### SandboxFusion 字段

| 字段路径 | 内容 | 来源 |
|----------|------|------|
| `results[i]` | 测试结果 True/False/-1/-2/-3/-4 | `check_correctness()` |
| `metadata["status"]` | 状态字符串 | `check_correctness()` |
| `metadata["duration"]` | 执行时间(秒) | `check_correctness()` |
| `metadata["stdout"]` | 标准输出 | `check_correctness()` |
| `metadata["stderr"]` | 标准错误 | `check_correctness()` |
| `metadata["exit_code"]` | 退出码 | `check_correctness()` |
| `score` | pass_ratio | `compute_score()` |
