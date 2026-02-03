# RLVR Coding Model - 评测协议与成本对齐

目标：让 Base / SFT / GRPO / 多轮修复 的对比在**同一预算口径**下成立，避免“只是采样更多/用更多 tokens”导致的虚假增益。

---

## 1) 术语与指标口径（统一定义）

- **best-of-1**：每题只生成 1 个候选并直接判题（无 rerank）。等价于 `@1` 指标。
- **best-of-n**：每题生成 n 个候选，逐个判题，取“最好”的那个用于统计（通常按 `pass_ratio` 排序，tie-break 用更短 tokens / 更快 exec_time）。
- **CodeContests accepted@k**：k 次采样中“存在至少 1 个候选全测例通过”的题目比例。
- **HumanEval/MBPP pass@k**：同上（以 benchmark 的通过定义为准）。
- **pass_ratio**：通过用例数 / 总用例数，值域 [0,1]。
- **总生成成本**：`total_gen_tokens = Σ candidates/rounds 的输出 tokens`（输入 tokens 可选单列）。
- **总判题成本**：`total_judge_time = Σ sandbox 执行时间`（建议同时记录 wall-clock 与 sandbox reported duration）。

> 约定：本项目的“主结果”默认使用 `@1`（best-of-1）口径；`@k`/best-of-n 作为推理预算曲线单独展示，并必须同步汇报成本。

---

## 2) 三种评测协议（推荐都保留）

### Protocol A：`EVAL@1`（主协议，最公平）

- **n=1**（best-of-1）
- 解码：`greedy` 或固定低温采样（必须写死并记录）
- 输出：accepted@1 / pass@1 / pass_ratio(mean/p50/p90) + 成本面板

用途：回答“模型参数更新（SFT/GRPO）是否真的变强”，不引入推理预算差异。

### Protocol B：`EVAL@k`（推理预算曲线）

- **n=k**（best-of-k），建议 k ∈ {5, 10}（或与你的 rollout_n 对齐）
- 解码参数与 `EVAL@1` 一致（除 n 之外）
- 统计：accepted@k / pass@k + pass_ratio@k + 成本面板（tokens 与 judge_time 会近似线性增长）

用途：回答“同一模型在更多推理预算下能提高多少”，同时给出成本-效果曲线。

### Protocol C：`EVAL@budget`（固定预算口径，强对照）

核心规则：对任意方法（best-of-n / 多轮修复），都必须满足同一条预算约束：

- 固定每题预算：`gen_tokens_budget_per_problem`（必选） + `judge_time_budget_per_problem`（可选）
- 运行策略（示例）：
  - best-of-n：按顺序生成候选并判题，达到预算上限就停止，取预算内最优候选
  - 多轮修复：Round1→Round2…，达到预算上限就停止，取最终轮（或预算内最优轮）

用途：回答“如果成本对齐，GRPO 的收益是否仍成立”，这是面试最抗质疑的一页。

---

## 3) 必须记录的“成本对齐字段”（用于审计）

对每个 run（甚至每个样本）至少记录：

- 解码参数：`temperature/top_p/max_new_tokens/n`
- token 统计：`output_tokens`（必要），`input_tokens`（可选）
- 判题统计：`exec_time`（sandbox reported），`wall_time`（端到端）
- 预算口径：`EVAL@1` / `EVAL@k` / `EVAL@budget` + 预算数值

---

## 4) 使用规则（避免数据泄漏与调参争议）

- `CodeContests_valid` + `MBPP_reg`：允许高频回归、选 checkpoint、早停、选超参。
- `CodeContests_test` + `HumanEval`：**test-only**，只允许阶段结束跑一次“报告型评测”，禁止基于 test 结果做任何选择。
- best-of-n 与多轮修复：可以在 test-only 上跑“报告”，但必须遵守上面的 test-only 约束，并把成本一并报告。
