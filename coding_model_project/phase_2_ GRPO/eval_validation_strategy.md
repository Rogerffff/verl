# Phase 3 Eval Throughput and Validation Strategy

> 本文是当前 Phase 3 的**评测吞吐诊断与验证策略说明**。
>
> 它的目标不是重新汇总 reward 设计，而是回答下面几个更工程化的问题：
>
> 1. 为什么 `CodeContests_valid` 的 full-test eval 明显比 `HumanEval` / `MBPP` 慢很多
> 2. 当前“慢”的主因到底是 timeout 长尾，还是 sandbox 并发不足，还是别的因素
> 3. 训练期间应该如何区分高频验证、中频验证和正式 baseline
> 4. 哪些加速手段可以先做，哪些不该过早写死
>
> 配套 baseline 结果见：
> [eval_analysis.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/outputs/phase0_fullval_20260331/eval_analysis.md)
>
> 配套 reward 主线见：
> [formal_reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/formal_reward_design.md)

---

## 1. 一句话结论

当前 `CodeContests_valid` eval 明显慢，**总 wall time 的主因不是“完全没并发”，也不是 timeout 长尾独占了大头，而是 full-test 工作量本身非常大，同时 shared verifier 的全局 sandbox 并发预算还比较保守**。

更准确地说：

- **总耗时主因**：`117` 题一共要跑 `6207` 个 testcase，而当前 shared verifier 的全局 limiter 默认只有 `8`
- **长尾主因**：`30s timeout` 主要解释了最慢那部分题的 `p95 / tail latency`
- **因此**：
  - timeout 长尾解释“为什么有些题会跑到几百秒”
  - 低并发相对于总 testcase 规模解释“为什么整体 full eval 这么慢”

这两个结论并不矛盾。

---

## 2. 当前已经确认的事实

以下结论基于 `2026-03-31` 的 `phase0_fullval_20260331` 结果。

### 2.1 数据集层面

- `HumanEval`: `164` 题，`avg_judge_time = 0.19s`
- `MBPP_reg`: `200` 题，`avg_judge_time = 0.21s`
- `CodeContests_valid`: `117` 题，`avg_judge_time = 143.36s`

### 2.2 CodeContests 的工作量规模

- `117` 题总共对应 `6207` 个 testcase
- 平均每题 `53.05` 个 testcase
- `pass_ratio_mean = 0.1422`
- `accepted@1 = 2.56%`

### 2.3 timeout 的真实影响

- testcase-level `timeout` 一共 `306` 个
- 有 timeout testcase 的题只有 `10` 道
- 这些题里最坏的 timeout-case 数分别是：
  - `48`
  - `46`
  - `43`
  - `41`
  - `40`
  - `35`
- 这些“带 timeout case 的题”总 judge time 约占 `14.3%`
- 其余 `85.7%` 的 judge time 发生在**没有 timeout case 的题**上

### 2.4 当前单实例 sandbox 不是完全没并发

从之前远端日志看，单实例 sandbox 在高峰时已经能打出较高请求并发，因此当前不能把问题简单归因成：

- “sandbox 完全没并发”
- “一定得先上多实例才能继续”

更准确的说法是：

- 当前单实例 sandbox 是在工作的
- 但对于 `6207` 个 testcase 的 full-test workload，**全局 limiter 只有 `8` 仍然偏保守**

---

## 3. 为什么之前会感觉“是 timeout 导致 eval 很慢”

这个感觉不是错觉，但它解释的是**长尾**，不是**总耗时**。

### 3.1 timeout 确实会制造极慢个例

有些题的 `judge_time_s` 会被拉到：

- `200s+`
- `300s+`
- 甚至 `400s+`

这些题通常都带有大量 timeout testcase，因此非常显眼。

### 3.2 但 timeout 不是总时间的大头

如果把全部 `117` 题的总 judge time 分开看：

- timeout-bearing problems 只占约 `14.3%`
- 没 timeout 的题反而占了约 `85.7%`

这说明：

- timeout 是**尾部放大器**
- 不是“整体慢的唯一主因”

### 3.3 整体慢更像是“大量普通 testcase 被低并发慢慢吃完”

因为：

- 每题平均 `53` 个 testcase
- 总计 `6207` 个 testcase
- 当前全局 limiter 只有 `8`

所以即使没有大量 timeout，光是完整跑完这么多 testcase，本身就会慢。

---

## 4. 当前最准确的根因判断

建议以后统一采用下面这套表述：

### 4.1 正式表述

`CodeContests_valid` full-test eval 慢，主要有两层原因：

1. **主因是 full-test 总工作量过大**
   - testcase 数量非常多
   - 当前 shared verifier 的全局 sandbox 并发预算偏低

2. **次因是 `30s timeout` 制造了明显长尾**
   - 它显著抬高了最慢样本的 `p95 / tail latency`
   - 但不是总 wall time 的主要组成部分

### 4.2 不建议再使用的表述

不建议再把问题描述成：

- “eval 主要是因为 timeout 才慢”
- “现在基本是串行跑 testcase”

更准确的说法应该是：

- 不是串行
- 是**全局并发存在，但相对 workload 仍然偏低**

---

## 5. 为什么 RL probe 里没有那么强烈地感觉 sandbox 慢

这不是矛盾，而是工作负载不同。

主要原因有四个：

- RL probe 的样本量远小于 full eval
- RL reward 使用的是更短的 `run_timeout_s=15`
- RL probe 常常只跑 `1 step`，不是 `117` 题完整 full validation
- RL step 的总耗时里，`update_actor` 常常还是第一瓶颈，reward judge 只是第二瓶颈

所以：

- 在 RL probe 中，sandbox 慢的问题被 actor update 的 GPU 开销部分遮住了
- 在纯 eval 中，judge 本身就是主工作负载，所以 slowdown 更显眼

---

## 6. 后续验证策略：不要把所有 eval 混成一类

当前最重要的策略是：

- **把训练期在线验证**
- **和离线正式 baseline**
- **明确分层**

而不是每次都跑同一套 full validation。

### 6.1 正式 baseline eval

用途：

- 作为可复现、可汇报、可对比的正式结果
- 用于 reward 版本对比、A0/A1/A2 对比、checkpoint 阶段性汇报

建议口径：

- `CodeContests_valid`
- full external tests
- `run_timeout = 30s`
- 完整保存 `per_case_results`

这套口径应该保持稳定，不为了提速随意改。

### 6.2 训练期高频验证

用途：

- 快速判断训练是否在朝正确方向走
- 及时发现 reward 崩坏、输出退化、明显 overfit 或明显性能回退

高频验证不应该等同于正式 baseline。

建议原则：

- 使用**固定、可复现、覆盖不同难度层级**的 `CodeContests` 子集
- 规模不要在文档里写死成固定题数
- 由**目标 wall time budget** 和**覆盖度要求**来决定
- 允许使用比正式 baseline 更短的 timeout

更直白地说：

- 训练期 fast val 应该是一个**中等规模、固定组成、可重复**的 subset
- 但不在文档中提前写死成“20-30 题”

### 6.3 中频验证

用途：

- 比高频验证更可靠
- 但又不必像正式 baseline 那样昂贵

建议：

- 仍然使用 full external tests
- 题量大于高频验证
- 频率低于高频验证
- 可以作为 checkpoint 前的“准正式评估”

### 6.4 低频正式评估

用途：

- 阶段性里程碑
- reward 对比结论
- 简历 / 面试汇报材料

建议：

- `CodeContests_valid` 保持 30s 口径
- `valid_big` 只在低频 checkpoint 或阶段性实验后跑
- 不要把 `valid_big` 作为训练期高频监控指标

---

## 7. 后续加速 eval 的优先级

### 7.1 第一优先级：分层验证，而不是每次都跑 full validation

这是收益最大、工程风险最小的改法。

核心思路：

- 高频验证看趋势
- 中频验证看稳态
- 低频 full validation 看正式结论

### 7.2 第二优先级：调大 verifier 并发预算

建议后续压测：

- `verifier_limiter_budget = 8`
- `verifier_limiter_budget = 12`
- `verifier_limiter_budget = 16`

关注：

- total wall time
- `sandbox_error_rate`
- `judge_time_p95`
- CPU 占用和系统稳定性

如果单实例在更高 limiter 下仍然稳定，那么这是最直接的提速手段。

### 7.3 第三优先级：为训练期 fast val 使用更短 timeout

建议原则：

- 正式 baseline 继续 `30s`
- 高频验证可考虑 `15s` 或 `20s`

这样做的目的不是改变正式指标，而是：

- 降低尾部耗时
- 增强训练过程中的反馈频率

但必须明确：

- fast val 指标不能直接与 `30s` baseline 混算

### 7.4 第四优先级：多 sandbox 实例

如果出现下面情况，再考虑多实例：

- 单实例 `limiter_budget` 提到更高后仍然很慢
- `sandbox_error_rate` 仍可接受
- CPU 资源仍足够
- full-test judge 已经明确成为训练或验证的主要 bottleneck

多实例是合理的下一阶段优化，但不应该在没有单实例压测证据前就默认成为首选。

---

## 8. 训练期推荐的验证原则

为了方便后续 agent 直接沿用，这里把当前推荐口径写成约束性更强的版本。

### 8.1 当前推荐

- 不把训练期在线验证等同于正式 baseline
- 不把 `valid_big` 放进高频评测链路
- 不把高频验证题量提前写死为某个固定值

### 8.2 当前推荐的配置方向

- 高频验证：
  - 固定子集
  - fixed seed
  - full external tests
  - timeout 可短于正式 baseline
  - 目标是 wall time 可接受、趋势可解释

- 中频验证：
  - 题量更大
  - timeout 可与高频验证一致或略高
  - 用于阶段性 sanity check

- 低频正式评估：
  - `CodeContests_valid`
  - full external tests
  - `run_timeout = 30s`
  - 保持可比较性

---

## 9. 不建议现在就写死的东西

以下事项当前都不建议提前写死进正式策略：

- 高频验证固定成“20-30 题”
- 一上来就默认必须多起多个 sandbox
- 因为 eval 慢就把正式 baseline timeout 从 `30s` 改掉
- 因为 RL probe 用 `15s` 成功就直接把离线 baseline 也全部切到 `15s`

这些都太早了。

更稳的做法是：

- 保留正式 baseline 的可信口径
- 再单独优化训练期监控口径

---

## 10. 建议后续 agent 优先回答的问题

后续如果要继续优化 eval / validation，建议优先回答下面几个问题：

1. 单实例 sandbox 在 `limiter_budget=12/16` 下是否仍然稳定
2. 训练期 fast val 的 wall time budget 应该定在多少分钟以内
3. 高频验证子集应该如何覆盖：
   - 容易题
   - 中等题
   - 长尾慢题
   - mixed-case 题
4. `15s / 20s / 30s` 三种 timeout 对 fast val 排名一致性影响有多大
5. 在不牺牲解释性的前提下，是否值得引入第二个 sandbox 实例

---

## 11. 当前冻结结论

截至当前版本，建议冻结为：

- 正式 baseline：
  - `CodeContests_valid`
  - full external tests
  - `run_timeout = 30s`

- 训练期验证策略：
  - 分层进行
  - 高频验证使用固定 subset，但**题量不在文档中写死**
  - 允许使用比正式 baseline 更短的 timeout
  - 优先通过提高 verifier 并发预算来加速

- 对 eval 慢的正式解释：
  - **总吞吐慢的主因是 full-test 工作量大 + 当前全局并发预算偏保守**
  - **timeout 长尾主要解释尾部，而不是总 wall time 的大头**

