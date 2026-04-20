# Phase 3: Formal Reward Design

> 本文是当前 coding RL 项目在 Phase 3 的**正式 reward 设计定稿**。
>
> 它的目标不是罗列所有候选公式，而是回答五个问题：
>
> 1. 当前项目为什么选这条 reward 主线
> 2. reward v1 的正式定义是什么
> 3. 每个分支判断为什么这样设计
> 4. 当前 reward 如何与 A0 / A1 / A2 算法路线配合
> 5. 面试官最可能追问什么、应该怎么回答
>
> 配套算法路线见：
> [algorithm_decision_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/algorithm_decision_guide.md)
>
> 配套 baseline 分析见：
> [eval_analysis.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/outputs/phase0_fullval_20260331/eval_analysis.md)

---

## 1. 一句话结论

当前项目 Phase 3 的正式 reward v1 固定为：

- **`anchored_dense + guardrails`**

正式定义如下：

```text
pass_ratio_all = passed_tests / total_tests   # 必须基于完整 testcase

if infra_error or sandbox_error or no_test_cases:
    reward = INVALID_FOR_RL
elif truncated_by_max_tokens:
    reward = INVALID_FOR_RL
elif error_type in {empty_output, extraction_failure, syntax_error}:
    reward = -1.0
else:
    reward = 0.8 * pass_ratio_all + 0.2 * accepted
```

其中：

- `accepted in {0,1}`
- `accepted=1` 当且仅当**全量 testcase 全部通过**
- `pass_ratio_all` 当且仅当来自 shared verifier 的 full external tests
- `INVALID_FOR_RL` 表示该样本不参与 RL 更新，等价语义是 `advantage = 0`，同时单独记日志

当前 reward v1 的默认命名固定为：

- **`anchored_dense_v1`**

当前 reward 的强对照固定为：

- **`dense_anchor_v1`**

其定义为：

```text
if infra_error or sandbox_error or no_test_cases:
    reward = INVALID_FOR_RL
elif truncated_by_max_tokens:
    reward = INVALID_FOR_RL
elif error_type in {empty_output, extraction_failure, syntax_error}:
    reward = -1.0
else:
    reward = clip(-0.2 + 1.0 * pass_ratio_all + 0.2 * accepted, -1.0, 1.0)
```

---

## 2. 为什么是这条主线

### 2.1 当前 baseline 已经说明必须用 dense reward

截至 **2026-03-31** 的 `codecontests_valid` baseline：

- `accepted@1 = 2.56%`
- `pass_ratio_mean = 0.1422`
- `58/117` 题满足 `pass_ratio_all > 0`
- `11/117` 题满足 `pass_ratio_all >= 0.5`

这说明当前模型不是“完全不会”，而是已经存在大量：

- 能通过一部分 testcase
- 但离完全 AC 还有距离

因此：

- **纯 `accepted` reward 太稀疏**
- **verifier-based dense reward 是必须的**

### 2.2 为什么不选 problem-level penalty 主线

当前 `codecontests_valid` 的 mixed-case 很多：

- `Broad mixed = 70/117`
- `Strict mixed-failure = 25/117`

也就是说同一题内部经常同时出现：

- `success + wrong_answer`
- `success + runtime_error`
- `success + timeout`
- 甚至 `success + wrong_answer + runtime_error`

这会直接带来一个工程结论：

- **不能仅凭题目级 `error_type` 就对整题 reward 做 hard override**

否则就会把同题内部已经存在的 partial correctness 一起抹掉。

这也是为什么当前 reward v1 明确不把：

- `problem-level runtime_error penalty`
- `problem-level timeout penalty`

作为主线设计的一部分。

### 2.3 为什么主线不直接选 `dense_anchor_v1`

`dense_anchor_v1` 是一个很有价值的对照，但当前不作为正式主线，原因不是它不好，而是：

1. 当前正式算法主线是：
   - `A0 = Stable GRPO baseline`
   - `A1 = GRPO with DAPO-style stabilizations`
2. 在这两条路线上，文档已固定：
   - `algorithm.norm_adv_by_std_in_grpo = true`
3. 这意味着在标准 GRPO 的组内均值中心化和标准差归一化下，`dense_anchor_v1` 里那条 `-0.2` 负截距的重要性会被明显削弱

换句话说：

- `anchored_dense`
- `dense_anchor_v1`

在当前 A0 / A1 口径下，训练行为不会像静态 reward 分布看上去那样差很多。

因此 reward v1 主线更应该优先满足：

- 语义简单
- 可解释
- 可复现
- 面试容易讲清楚

而 `anchored_dense` 比 `dense_anchor_v1` 更符合这个目标。

### 2.4 为什么 accepted 仍然要保留 anchor

如果只用：

- `reward = pass_ratio_all`

那 reward 虽然有 dense signal，但“全部通过”并没有被单独抬出来。

当前项目最终主汇报指标仍是：

- `accepted@1`

因此 reward 主线必须显式保留一个：

- **outcome anchor**

这就是 `0.8 * pass_ratio_all + 0.2 * accepted` 里 `+ 0.2 * accepted` 的作用：

- 保留 dense partial correctness
- 同时明确对齐最终 AC

---

## 3. 正式 reward 定义

### 3.1 输入字段

reward v1 只依赖以下 verifier 字段：

- `accepted`
- `passed_tests`
- `total_tests`
- `pass_ratio_all`
- `error_type`
- `invalid_for_rl`
- `invalid_reason`
- `finish_reason` 或等价截断标记

训练期还会额外记录但**不进入主 reward**的字段：

- `judge_time_s`
- `gen_tokens`
- `raw_completion_length`
- `response_length`

### 3.2 正式公式

```text
pass_ratio_all = passed_tests / total_tests

if invalid_for_rl:
    reward = INVALID_FOR_RL
elif truncated_by_max_tokens:
    reward = INVALID_FOR_RL
elif error_type in {empty_output, extraction_failure, syntax_error}:
    reward = -1.0
else:
    reward = 0.8 * pass_ratio_all + 0.2 * accepted
```

### 3.3 `INVALID_FOR_RL` 的固定语义

以下情况必须视为 `INVALID_FOR_RL`：

- `infra_error`
- `sandbox_error`
- `api_error`
- `no_test_cases`
- `truncated_by_max_tokens`

这些样本的特点是：

- 不能稳定代表模型能力
- 会污染 GRPO 的组内 reward 统计
- 不应该当成“正常失败样本”去学习

因此它们必须：

- `advantage = 0`
- 单独进入日志和 dashboard
- 不参与主 reward 比较

### 3.4 为什么 `truncated_by_max_tokens` 也要 `INVALID_FOR_RL`

这是一条非常重要的设计选择。

如果样本因为 `max_tokens` 被截断，那么：

- 输出可能是不完整代码
- 它很容易表现成 `syntax_error`
- 但这类错误不一定来自“模型不会写”，而可能来自“生成被硬切断”

如果把这种样本直接当成：

- `syntax_error -> -1.0`

去学习，实际上是在把：

- **generation budget 问题**

误当成：

- **程序能力问题**

这会污染 reward 语义。

因此当前 reward v1 明确要求：

- `truncated_by_max_tokens` 单独作为 invalid 处理
- 不和普通 `syntax_error` 混在一起

实现提醒：

- 这条语义成立的前提，是 rollout 侧必须把 `finish_reason` 或等价的 `truncated` 标记传进 reward 路径
- 如果当前训练 batch 里还没有这个字段，那么这不是设计无效，而是实现待补齐
- 在正式对外宣称“reward v1 已完全落地”前，必须确认这条链路真实存在

### 3.5 为什么 `empty_output / extraction_failure / syntax_error` 给 `-1.0`

这三类样本属于同一个层级的问题：

- 模型没有形成有效程序
- 或者程序结构已经坏到还没进入“功能部分正确”的讨论空间

它们和：

- `wrong_answer`
- `partial pass`
- `timeout after some passed tests`

不是同一个层级。

因此 reward v1 把它们统一当成：

- **hard negative**

这样做的好处是：

- 简单
- 可解释
- 符合“先形成有效程序，再追求功能正确”的工程逻辑

### 3.6 为什么 `runtime_error / timeout` 不做 problem-level penalty

当前 shared verifier 已经能拿到：

- `per_case_results`
- `passed_tests`
- `pass_ratio_all`

而你的 baseline 也已经证明：

- 很多 `runtime_error` / `timeout` 题内部其实已经通过了不少 testcase

例如：

- 题目级是 `runtime_error`
- 但 `pass_ratio_all` 可能仍有 `0.36`、`0.38`、`0.74`

在这种情况下，如果整题直接因为：

- `runtime_error`
- `timeout`

就整体减分，反而会抹掉已经存在的 verifier dense signal。

因此当前 v1 固定原则是：

- `runtime_error / timeout` 先不做 problem-level hard penalty
- 它们的影响自然体现在 `pass_ratio_all` 上
- `error_type` 只用于日志和后续分析

### 3.7 为什么主线不引入长度惩罚 / judge-time 惩罚

当前 v1 明确不引入：

- token length penalty
- judge_time penalty
- generation length penalty

原因有三点：

1. 这会把 reward 从“功能正确性”拖向“成本/效率”的混合目标
2. coding 任务有些题确实需要更长程序，长度惩罚容易误伤
3. 当前项目第一版更强调：
   - 正确性
   - 可复现
   - 可答辩

如果未来确实要做效率约束，应该作为：

- v2 独立消融

而不是混进 reward v1 主线。

---

## 4. 为什么主线不是 sparse / dense_anchor_v1 / penalized_anchor

### 4.1 为什么不是 sparse_accepted

如果只用：

```text
reward = accepted
```

那当前 `accepted@1 = 2.56%` 的 baseline 下：

- 绝大多数样本 reward 都是 0
- 对 GRPO 组内相对优势来说，早期训练会非常稀疏

所以 sparse 只能作为：

- 必要对照

不能作为主线。

### 4.2 为什么不是 dense_anchor_v1

`dense_anchor_v1` 的定义是：

```text
clip(-0.2 + pass_ratio_all + 0.2 * accepted, -1, 1)
```

它的问题不是错误，而是：

- 在当前 A0 / A1 的标准 GRPO 口径下，`-0.2` 的重要性会被组内归一化削弱
- 因此它更适合做：
  - 强对照
  - 而不是 reward v1 主线

它依然非常有价值，因为：

- 它比 `anchored_dense` 更保守
- 比 problem-level penalty 版本更少误伤 mixed-case

所以当前定位固定为：

- **主线 reward 的强对照 ablation**

### 4.3 为什么不是 penalized_anchor

当前静态分析已经证明：

- `penalized_anchor` 的正奖励题数只有 `18/117`
- 负奖励题数达到 `98/117`

更重要的是：

- 它对 `runtime_error / timeout` 的 problem-level penalty 会误伤 mixed-case

因此这条路不适合作为 v1 主线。

---

## 5. 与算法路线 A0 / A1 / A2 的关系

### 5.1 当前固定关系

reward v1 冻结后，算法路线实验矩阵应按下面顺序推进：

#### 第一阶段：固定 reward，比算法

- `A0 + anchored_dense_v1`
- `A1 + anchored_dense_v1`
- `A2 + anchored_dense_v1`

目的：

- 在同一 reward 下比较算法路线

#### 第二阶段：固定算法，比 reward

优先在 `A1` 下比较：

- `anchored_dense_v1`
- `dense_anchor_v1`
- `sparse_accepted`

目的：

- 在主线算法固定后，再验证 reward 取舍

### 5.2 为什么先固定 reward 再比算法

如果算法和 reward 同时变化，那么后面无法回答：

- 到底是算法路线更好
- 还是 reward 口径变化改变了训练信号

因此当前工程要求固定为：

- **先冻结 reward v1**
- **再做 A0 / A1 / A2 比较**

---

## 6. 日志与指标要求

reward v1 必须稳定记录以下字段：

- `reward_raw`
- `accepted`
- `pass_ratio_all`
- `passed_tests`
- `total_tests`
- `error_type`
- `invalid_for_rl`
- `invalid_reason`
- `judge_time_s`
- `finish_reason`
- `truncated_by_max_tokens`
- `extraction_status`

训练监控至少要有：

- `verifier/pass_ratio_all_mean`
- `verifier/accepted_rate`
- `verifier/invalid_for_rl_rate`
- `verifier/truncated_by_max_tokens_rate`
- `verifier/reward_raw_valid_count`
- `verifier/reward_raw_valid_rate`
- `verifier/reward_raw_mean`
- `verifier/runtime_error_rate`
- `verifier/timeout_rate`
- `verifier/syntax_error_rate`
- `verifier/empty_output_rate`
- `verifier/extraction_failure_rate`
- `verifier/judge_time_s_mean`
- `verifier/judge_time_s_p95`

额外冻结两条实现口径：

- `reward_raw` 对 `INVALID_FOR_RL` 样本必须记为 `NaN`，训练与 validation 聚合都只统计 finite 值
- 训练侧额外记录 `grpo/all_invalid_group_count` 与 `grpo/all_invalid_group_rate`，按每个 prompt-group 的 `uid` 统计

---

## 7. VeRPO 路线的后续增强

当前 reward v1 明确**不做**：

- weighted testcase difficulty
- weighted pass ratio
- extra outcome correction beyond `accepted`

但这不意味着 VeRPO 思路被否定。

当前对 VeRPO 的正式定位是：

- **reward v2 的候选增强方向**

### 7.1 何时考虑升级到 VeRPO-lite

只有在以下情况出现时，才考虑进入 VeRPO-lite：

1. `anchored_dense_v1` 已稳定优于 sparse
2. 但 `accepted@1` 提升仍明显慢于 `pass_ratio_mean`
3. 分析发现模型主要在“通过很多简单 testcase，但 AC 提升有限”

这时可考虑升级为：

```text
reward = alpha * weighted_pass_ratio + beta * accepted
```

其中：

- `weighted_pass_ratio` 来自 testcase difficulty weighting
- `accepted` 继续保留 outcome anchor

### 7.2 为什么 VeRPO 不进 v1

原因不是它不强，而是：

1. 需要额外维护 testcase 难度统计
2. 会增加 reward 解释成本
3. 会把“reward 本身的升级”和“当前主线是否跑通”绑在一起

所以当前项目把它明确定位为：

- **后续可选增强**
- **不是 reward v1 的前置条件**

---

## 8. 面试官可能的问答

### Q1：为什么不用纯 sparse reward？

因为当前 CodeContests baseline 已经显示：

- `accepted@1` 很低
- 但接近一半题目已经有非零 full-test pass ratio

这说明 partial correctness signal 是真实存在的。如果只学 `accepted`，训练早期绝大多数组都拿不到有效区分信号，所以我选了 verifier-based dense reward。

### Q2：为什么 reward 要加 accepted anchor？

因为最终主汇报指标是：

- `accepted@1`

如果只用 `pass_ratio_all`，模型会被鼓励“多过一些测试”，但“全部通过”并没有被明确抬出来。`+0.2 * accepted` 的作用，就是在保留 dense signal 的同时，把最终 AC 对齐进主目标。

### Q3：为什么不是 `dense_anchor_v1` 做主线？

它是一个很好的强对照，但不是当前主线。原因是当前 A0 / A1 走的是标准 GRPO 口径，组内中心化和标准差归一化会削弱负截距的作用，所以我不想把 reward 主线绑定在一个在当前算法设置下不一定有决定性效果的偏移项上。主线更适合用语义更简单、也更好答辩的 `anchored_dense`。

### Q4：为什么 `truncated_by_max_tokens` 要记成 `INVALID_FOR_RL`？

因为被截断的样本很容易表现成“不完整代码”，进而表面上像 `syntax_error`。如果直接把它当普通语法错误去学，相当于把 generation budget 问题误当成程序能力问题。为了保持 reward 语义干净，我把截断样本单独 mask 掉，只记日志不参与更新。

### Q5：为什么 `syntax_error` 给 `-1.0`，而 `runtime_error/timeout` 却不额外惩罚？

`syntax_error`、`empty_output`、`extraction_failure` 属于“还没形成有效程序”的错误层级，所以适合作为 hard negative。  
而 `runtime_error/timeout` 在当前 shared verifier 下往往仍伴随 partial pass，很多 mixed-case 题已经通过了不少 testcase。如果仅按题目级错误类型整题减分，会抹掉这些 verifier dense signal。

### Q6：为什么不用 learned reward model？

因为当前任务已经有强、可验证、可复现的 execution feedback。对第一版求职项目来说，我优先选择 fully verifiable reward，而不是引入额外 reward model / critic 的系统复杂度和解释负担。

### Q7：为什么不直接上 VeRPO 的 weighted reward？

VeRPO 是很好的后续增强方向，但它需要在线维护 testcase difficulty 或额外统计逻辑。当前项目 v1 更重要的是先把：

- full-test shared verifier
- verifier-based dense reward
- A0 / A1 / A2 算法比较

这三件事做扎实。等主线稳定后，再升级到 weighted dense 才更容易归因，也更适合面试叙事。

### Q8：如果面试官问“这是不是拍脑袋定的 0.8 / 0.2”？

我的回答会是：

不是拍脑袋，而是工程化取舍。当前 baseline 说明 partial correctness 很多，所以主 reward 需要主要由 `pass_ratio_all` 提供连续信号；同时最终项目汇报还是以 `accepted@1` 为目标，所以保留一个明确的 outcome anchor。`0.8 / 0.2` 是在“dense signal 充分”与“最终 AC 对齐”之间的简单、可解释折中。

---

## 9. 最终执行口径

当前 reward v1 的正式执行口径固定为：

- 主线：`anchored_dense_v1`
- 强对照：`dense_anchor_v1`
- 必要控制组：`sparse_accepted`

当前 reward v1 明确排除：

- problem-level runtime penalty
- problem-level timeout penalty
- weighted dense v1
- RM-based reward v1
- length / judge-time penalty

只有在 reward v1 冻结、训练链路稳定、A0 / A1 / A2 跑通之后，才进入：

- VeRPO-lite
- 更复杂 reward shaping
- weighted testcase difficulty

的下一阶段讨论。
