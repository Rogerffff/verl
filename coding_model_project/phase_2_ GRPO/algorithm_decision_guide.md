# Phase 3: GRPO 正式算法决策与实施指南

> 本文是当前 coding RL 项目在 GRPO 阶段的**正式算法决策正文**。
>
> 它不负责重复介绍全部数据治理、评测协议和奖励推导，而是专门回答四个问题：
>
> 1. 当前项目为什么选这条算法路线
> 2. 应该如何做算法比较
> 3. 实现前哪些配置要先写死
> 4. 面试时应该如何讲清楚这些取舍
>
> 配套总览文档见：
> [README.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/README.md)
>
> 配套学习材料见：
> [phase3_algorithm_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/experiment_design/phase3_algorithm_guide.md)
>
> 配套奖励设计见：
> [reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/experiment_design/reward_design.md)
>
> Phase 3 正式 reward 定稿见：
> [formal_reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/formal_reward_design.md)

---

## 1. 文档目的与使用方式

### 1.1 这份文档给谁看

这份文档面向三类读者：

- **当前项目的实现者**
- **未来回看项目的你自己**
- **需要了解算法取舍的面试官/评审者**

它的定位不是“学习笔记”，而是 **GRPO 阶段的正式算法决策与实施指南**。

### 1.2 它和 README 的关系

当前目录下的 [README.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/README.md) 负责：

- 项目背景
- Phase 0 基线
- 数据文件说明
- 评测脚本设计
- 奖励函数设计摘要
- 超参与实验矩阵总览

而本文只负责：

- 锁定当前 **算法路线**
- 规定 **A0 / A1 / A2** 的比较方式
- 固定当前默认配置口径
- 写清采用标准、风险与面试表述

### 1.3 一句话结论

当前项目在 Phase 3 的正式主线算法确定为：

- **`GRPO with DAPO-style stabilizations`**

同时保留：

- **`Dr.GRPO` 作为关键替代路线消融**

这不是“追最新算法名词”，而是基于 `verl` 当前支持度、coding verifier 场景适配性、训练成本与面试可解释性做出的工程决策。

---

## 2. 决策边界

### 2.1 这份文档锁定什么

本文只锁定：

- 算法主线
- 路线级实验设计
- 默认配置口径
- 采用标准

### 2.2 这份文档不锁定什么

本文**不重新展开**以下内容：

- 完整 reward 设计细节
- 所有数据治理与去重细节
- 所有评测脚本实现细节
- 所有训练资源与调度问题

这些内容请分别参考：

- 奖励设计： [reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/experiment_design/reward_design.md)
- 算法学习： [phase3_algorithm_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/experiment_design/phase3_algorithm_guide.md)
- 阶段总览： [README.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/README.md)

### 2.3 当前不作为主线的路线

这些路线都已经被评估过，但当前不作为第一版 headline algorithm：

- `GSPO`
- `pass@k / PKPO / grpo_passk`
- `VeRPO` 作为 optimizer headline
- `full DAPO reproduction`
- `原生 GRPO` 作为正式主线

原因不是它们没有价值，而是：

- 当前阶段优先级是**稳定、可复现、可答辩**
- 不是做完整算法论文式拆解

---

## 3. 当前推荐算法路线

### 3.1 正式表述

当前项目 Phase 3 的正式算法路线表述固定为：

**`GRPO with DAPO-style stabilizations`**

中文可写为：

**以 GRPO 为主干，吸收 DAPO 中最成熟、最适合 coding verifier 场景的一组稳定化改进。**

当前主线只纳入以下 **已验证或可直接落盘** 的 DAPO-style 改动：

- `clip-higher`
- `no-KL`
- 与主线共享的稳定聚合设置 `token-mean`
- 与 reward 解耦的执行层优化（例如当前项目的 batched verifier 路径）

当前**不纳入正式 A1 定义**的 DAPO 组件：

- `filter_groups / dynamic sampling`
- reward-side overlong shaping

原因是：

- `filter_groups` 在当前 workspace 中尚未看到主训练循环的明确消费证据
- overlong shaping 会直接改变 reward，不适合混入当前算法路线比较
- `reward_manager=dapo` 虽然在仓库中有实现，但其核心是 overlong shaping，不适合作为当前 verifier reward v1 的默认入口

### 3.2 明确不采用的 headline 表述

当前不采用以下表述：

- `原生 GRPO`
- `full DAPO reproduction`
- `GSPO mainline`
- `pass@k RL mainline`

### 3.3 为什么不是原生 GRPO

原生 GRPO 当然可以跑，但它已经不是最适合当前项目的最终叙事。

如果继续把“原生 GRPO”作为主线，会有两个问题：

1. 到 2026 年 3 月，已经有充分公开经验指出：
   - 长响应训练不稳
   - 无效 group 浪费采样
   - KL 过强可能限制探索
2. 面试时很容易被追问：
   - 为什么没有吸收 2025 年后的成熟稳定化经验

因此：

- **原生 GRPO 更适合做 baseline**
- **不适合作为最终主线 headline**

### 3.4 为什么不是 full DAPO

我们认可 DAPO 的公开价值，但不把当前项目写成 full DAPO reproduction，原因有三点：

1. DAPO 更像 **bundle recipe**，不是一个边界非常单一的 estimator
2. 如果一开始全盘打包，会削弱后续归因能力
3. 当前项目还要单独严肃讨论 reward 设计，不能把 reward-side 与 optimizer-side 全部绑死

因此当前更准确的写法是：

- **DAPO-style**
- 而不是 **full DAPO**

### 3.5 为什么 Dr.GRPO 不是唯一主线

Dr.GRPO 解决的是一个很重要的问题：

- 原生 GRPO 的标准差归一化和 loss 聚合是否引入偏置

它很值得做，而且非常适合作为答辩中的“关键算法思考”。

但它不适合作为当前项目唯一主线，原因是：

- 它更像偏置修正路线
- 不像 DAPO-style 那样天然带着完整的稳定训练叙事
- 当前项目第一优先级是先把 verifier-based coding RL 稳定跑通

所以：

- **Dr.GRPO 是高价值替代路线**
- **但当前阶段更适合做关键消融，而不是唯一主线**

### 3.6 为什么当前阶段不选 GSPO / pass@k

#### GSPO

GSPO 更偏目标函数级创新，理论味更强，但在当前项目里有两个问题：

- 实现与解释成本更高
- 第一版简历项目不需要优先承担“更换 objective”带来的额外风险

#### pass@k / PKPO / `grpo_passk`

pass@k 训练目标更适合：

- sample-and-select
- rerank
- inference-time budget 优化

而当前项目主结果强调：

- `accepted@1`
- `pass_ratio_mean/p50/p90`

如果第一版就把训练目标换成 pass@k，会明显增加“训练目标与汇报目标不完全一致”的答辩负担。

---

## 4. 路线级实验设计

### 4.1 总原则

本阶段采用：

- **路线级消融**

而不是：

- **DAPO / Dr.GRPO 各组件的全因子拆解**

### 4.2 为什么不做组件级全拆

原因写死如下：

1. DAPO 是 bundle recipe，不是单点开关
2. Dr.GRPO 也是联合改动，不只是一个 bool
3. 两条路线改动层级不同，组件之间存在交互
4. 当前 coding RL 训练成本高，judge cost 高，不适合做组合爆炸实验

因此本阶段回答的是：

- **哪条算法路线更适合作为当前项目主线**

而不是：

- **每个细小组件的精确边际贡献**

### 4.3 三组主实验

#### `A0`: Stable GRPO baseline

作用：

- 提供稳定对照组
- 证明不是“任何 RL 配置都一样”

#### `A1`: DAPO-style GRPO mainline

作用：

- 当前正式主线
- 回答“现代稳定化 recipe 是否优于 stable baseline”

#### `A2`: Dr.GRPO alternative

作用：

- 回答“偏置修正路线是否比当前主线更值得升级”

### 4.4 比较逻辑

固定比较逻辑如下：

- `A0 -> A1`
  - 回答：**DAPO-style stabilizations 是否值得作为主线**

- `A1 -> A2`
  - 回答：**Dr.GRPO 这条偏置修正路线是否比当前主线更值得升级**

### 4.5 A2 的写法原则

`A2` 必须尽量继承 `A1` 的公共设置，只替换 Dr.GRPO 核心改动。

这样做的原因是：

- 让 `A1 vs A2` 更接近“路线核心差异”的比较
- 避免把太多无关配置同时变化，导致归因变脏

必须在文档中显式提醒：

- **A2 不是另起炉灶的完全新 recipe**
- **而是基于当前主线配置，替换为 Dr.GRPO 核心改动的替代路线**

### 4.6 `filter_groups` 的当前定位

当前文档将 `filter_groups / dynamic sampling` 定位为：

- **已评估、但暂不纳入正式 A1 默认配置的候选增强项**

原因是当前 workspace 中：

- 有配置定义
- 有脚本示例
- 有文档说明
- 也能在实验性脚本中看到 DAPO 相关配置样例

但尚未在 `verl/` 主 Python 训练代码中确认其已被当前训练入口明确消费。

因此在本阶段：

- 正式 A1 默认配置先不依赖 `filter_groups`
- 若后续在训练器中补齐实现，或 smoke run 证明其确实生效，可单列为：
  - `A1+fg` 或下一轮主线增强

必须再补一条实现判断：

- 当前 `filter_groups` 更接近“配置已定义、样例存在、主链未验证”的状态
- 因此它不能作为当前项目对外宣称“已稳定采用 DAPO dynamic sampling”的证据

如果未来真的要在 coding reward 下启用 group filtering，当前建议也不是直接照搬 binary acc 逻辑，而是改成 **coding-specific filtering**：

- 优先过滤：
  - `all invalid_for_rl` 的 group
- 可考虑过滤：
  - `accepted` 全相同且 `pass_ratio_all` 组内方差接近 0 的 group
- 不应过滤：
  - 虽然都没 AC，但 `pass_ratio_all` 仍有明显差异的 group
  - mixed-case 明显、已有 partial pass 差异的 group

换句话说：

- 当前 reward 不是全 0 / 全 1 binary acc
- 所以后续若启用 `filter_groups.metric=seq_reward`，必须先确认它的过滤语义与当前 `anchored_dense` reward 一致，而不是默认沿用“全错组 / 全对组”二元逻辑

---

## 5. 当前推荐配置基线

这一节给出当前实现前默认采用的 v1 配置口径。

### 5.1 A0：Stable GRPO baseline

| 配置键 | 值 |
|--------|----|
| `algorithm.adv_estimator` | `grpo` |
| `algorithm.norm_adv_by_std_in_grpo` | `true` |
| `actor_rollout_ref.actor.loss_agg_mode` | `token-mean` |
| `actor_rollout_ref.actor.clip_ratio_low` | `0.2` |
| `actor_rollout_ref.actor.clip_ratio_high` | `0.2` |
| `actor_rollout_ref.actor.use_kl_loss` | `true` |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` |
| `actor_rollout_ref.rollout.n` | `8` |
| `reward_model.reward_manager` | `batch` |
| `algorithm.filter_groups.enable` | `false` |

解释：

- 这不是论文原始 GRPO 的机械复刻
- A0 与 A1 共享 `token-mean`，目的是隔离 clip-higher / no-KL 等差异
- A0 是更适合 `verl` 当前工程环境的 stable baseline，不代表论文原始 GRPO

### 5.2 A1：DAPO-style GRPO mainline

| 配置键 | 值 |
|--------|----|
| `algorithm.adv_estimator` | `grpo` |
| `algorithm.norm_adv_by_std_in_grpo` | `true` |
| `actor_rollout_ref.actor.loss_agg_mode` | `token-mean` |
| `actor_rollout_ref.actor.clip_ratio_low` | `0.2` |
| `actor_rollout_ref.actor.clip_ratio_high` | `0.28` |
| `actor_rollout_ref.actor.use_kl_loss` | `false` |
| `actor_rollout_ref.rollout.n` | `8` |
| `reward_model.reward_manager` | `batch` |
| `reward_model.use_reward_loop` | `false` |
| `reward_model.launch_reward_fn_async` | `false` |
| `algorithm.filter_groups.enable` | `false` |

解释：

- 主干仍是 GRPO
- 吸收 DAPO-style 中当前最适合正式主线的一组稳定化改动
- batched reward manager 是执行层优化，不改变当前算法 headline
- 当前 A1 **显式排除** reward-side DAPO shaping，不使用 `reward_model.reward_manager=dapo`
- 实施时不得额外传入任何 DAPO overlong buffer 配置，例如 `+reward_model.reward_kwargs.overlong_buffer_cfg.enable=true`
- 当前 A1 **显式不依赖** `filter_groups`，待训练器侧实现或验证通过后再单列增强版本
- 当前不把 overlong reward shaping 写进 headline algorithm 定义，也不把它混入 A1 结果

### 5.3 A2：Dr.GRPO alternative

| 配置键 | 值 |
|--------|----|
| `algorithm.adv_estimator` | `grpo` |
| `algorithm.norm_adv_by_std_in_grpo` | `false` |
| `actor_rollout_ref.actor.loss_agg_mode` | `seq-mean-token-sum-norm` |
| `actor_rollout_ref.actor.loss_scale_factor` | `2048` |
| `actor_rollout_ref.actor.clip_ratio_low` | `0.2` |
| `actor_rollout_ref.actor.clip_ratio_high` | `0.28` |
| `actor_rollout_ref.actor.use_kl_loss` | `false` |
| `actor_rollout_ref.rollout.n` | `8` |
| `reward_model.reward_manager` | `batch` |
| `algorithm.filter_groups.enable` | `false` |

实现提醒：

- A2 应尽量继承 A1 的公共设置
- 只替换 Dr.GRPO 核心改动，避免比较失真

### 5.4 `filter_groups` 的实现提醒

`filter_groups` 是当前主线中最需要谨慎声明的一项。

必须保留以下实现提醒：

- 当前 workspace 中，配置、脚本和文档都明确支持 `filter_groups`
- 但在正式对外宣称“已作为主线稳定生效”之前，必须先确认当前训练入口确实消费了该配置
- 在未确认前，A1 / A2 正式配置都不依赖它

换句话说：

- 配置存在 ≠ 训练入口已经按预期生效

### 5.5 rescue 配置

如果 `A1` 在 pilot run 中出现明显漂移，可以启用 rescue 配置：

| 配置键 | rescue 值 |
|--------|-----------|
| `actor_rollout_ref.actor.use_kl_loss` | `true` |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` |

但要注意：

- rescue run 是稳定性兜底措施
- 不改变当前主线 headline 仍为 `DAPO-style GRPO`

### 5.6 rescue 触发条件

当前最小触发条件固定为：

- 连续两次 Tier-1 评测中，`timeout_rate` 或 `runtime_error_rate` 相比 A0 同 seed baseline 翻倍
- 或 `response_length_mean` 相比 A0 同 seed baseline 增长超过 `50%`

满足任一条件时，可启动 small-KL rescue run。

---

## 6. 与 reward 设计的边界

### 6.1 为什么必须单独写这一节

算法主线和 reward 设计如果混在一起，后面会有两个问题：

1. 很难归因
2. 很难答辩

所以当前必须明确区分：

- **算法主线 = 如何更新 policy**
- **reward 设计 = verifier 输出如何映射成训练分数**

### 6.2 当前文档对 reward 的固定口径

本文现在固定到这一层：

- 主奖励类型为 **verifier-based anchored dense reward**
- `pass_ratio_all` 必须基于**完整 test cases** 计算，而不是子集近似
- 当前 reward v1 主线固定为：
  - `anchored_dense + guardrails`
- 当前 reward 强对照固定为：
  - `dense_anchor_v1 + same guardrails`

当前主线 reward 的正式语义固定为：

```text
if infra_error or sandbox_error or no_test_cases:
    reward = INVALID_FOR_RL
elif truncated_by_max_tokens:
    reward = INVALID_FOR_RL
elif error_type in {empty_output, extraction_failure, syntax_error}:
    reward = -1.0
else:
    reward = 0.8 * pass_ratio_all + 0.2 * accepted
```

这里的固定原则是：

- 保留 verifier dense signal
- 给 `accepted` 一个明确 outcome anchor
- 不把 `problem-level runtime/timeout penalty` 写进主线
- 不把 weighted reward / RM reward 混进 reward v1

完整细节统一参考：

- [formal_reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/formal_reward_design.md)

### 6.3 为什么 `filter_groups.metric=seq_reward` 会和 reward 设计耦合

这是当前实现中必须明确写给实现者的一句提醒。

如果：

- `filter_groups.metric = seq_reward`

那么 group filtering 判断一个 group 是否“全一样、无信息量”，用的就是：

- 当前 reward 定义出来的序列级分数

因此：

- reward 从纯 `pass_ratio` 改成 piecewise dense reward
- filtering 行为就会跟着变化

所以 `filter_groups` 不是完全独立于 reward 的纯算法开关。

### 6.4 当前工程要求

因此当前阶段的要求是：

- **先冻结 reward v1**
- **先用同一 reward 跑 A0 / A1 / A2**
- **再做 reward ablation**

否则你无法分清：

- 是算法路线本身更好
- 还是 reward 口径变化改变了有效样本分布

---

## 7. 采用标准与升级规则

### 7.1 主线采用标准

固定判断逻辑如下：

1. `A1` 至少完成 **2 个 seeds**
2. 且在 `CodeContests_valid_big` 上稳定优于 `A0`
3. 且 `MBPP_reg` 上无明显退化
4. 则 `A1` 保持为正式主线

### 7.2 升级规则

若满足以下条件：

1. `A2` 在至少 **2 个 seeds** 下持续优于 `A1`
2. 且成本与稳定性不更差

则：

- `A2` 升级为强候选主线

### 7.3 保留为消融的条件

如果：

- `A2` 只是与 `A1` 接近
- 或略好但不稳定

则：

- `A2` 保留为高质量算法消融
- 不替换当前主线

### 7.4 漂移时的处理

如果 `A1` 出现明显漂移：

- 启用 small-KL rescue run

但要保持：

- headline 仍写作 `DAPO-style GRPO`

### 7.5 主汇报指标

主汇报指标固定为：

- `accepted@1`
- `pass_ratio_mean`
- `pass_ratio_p50`
- `pass_ratio_p90`
- `exec_success_rate`
- `runtime_error_rate`
- `timeout_rate`
- `cost_per_solved_tokens`
- `cost_per_solved_judge_time`

这些指标同时服务：

- 算法路线决策
- 实施监控
- 面试汇报

---

## 8. 实施清单

在真正写训练脚本和跑实验前，必须先完成下面这份 checklist。

### 8.1 数据与评测

- [ ] 确认训练集文件：`data/manifests/codecontests_train_wo_valid_big_manifest.jsonl`
- [ ] 确认高频验证集文件：`data/manifests/codecontests_valid_big_manifest.jsonl`
- [ ] 确认回归验证集：`data/manifests/mbpp_reg_manifest.jsonl`
- [ ] 确认评测协议与 README 保持一致

### 8.2 reward

- [ ] 冻结 reward v1
- [ ] 确认 reward v1 的 `pass_ratio` 基于完整 test cases 计算
- [ ] 确认 reward v1 主线固定为 `anchored_dense + guardrails`
- [ ] 确认 reward 强对照固定为 `dense_anchor_v1 + same guardrails`
- [ ] 确认 `truncated_by_max_tokens -> INVALID_FOR_RL`
- [ ] 确认 `empty_output / extraction_failure / syntax_error -> -1.0`
- [ ] 确认不使用 `problem-level runtime/timeout penalty`
- [ ] 明确 reward v1 的日志字段
- [ ] 确认 reward 定义已与 `filter_groups.metric=seq_reward` 的耦合关系被记录

### 8.3 算法配置

- [ ] 将 A0 / A1 / A2 配置单独落盘
- [ ] 固定 `actor_rollout_ref.rollout.n=8`
- [ ] 固定 seeds
- [ ] 固定 WandB config
- [ ] 固定 Tier 评测频率

### 8.4 `filter_groups`

- [ ] 若计划启用 `filter_groups`，先做 smoke run
- [ ] 若计划启用 `filter_groups`，检查当前训练入口是否真的消费该配置
- [ ] 若计划启用 `filter_groups`，检查日志中是否能看出 group 过滤行为

### 8.5 稳定性兜底

- [ ] 预留 small-KL rescue 配置
- [ ] 预定义漂移判据

---

## 9. 面试答辩摘要

### Q1：为什么不是原生 GRPO？

原生 GRPO 可以跑，但已经不是最完整的工程实践。对 coding RL，后续公开经验已经指出长响应不稳、无效 group 和 KL 过强等问题。我的做法不是否定原生 GRPO，而是把它保留为 baseline，再在主干不变的前提下吸收更成熟的稳定化改动。

### Q2：为什么不是 full DAPO？

DAPO 更像一套 recipe，而不是边界很清晰的单点 estimator。对求职项目来说，如果直接写成 full DAPO，会把太多设计绑死，也不利于后续归因。更稳妥的表述是保留 GRPO 主干，只吸收最成熟、最适合 coding verifier 场景的 DAPO-style 改动。

### Q3：为什么 Dr.GRPO 是消融而不是唯一主线？

Dr.GRPO 很有价值，因为它指出了原生 GRPO 可能存在的归一化与长度相关偏置。但当前项目第一优先级是先把 verifier-based coding RL 稳定跑通，所以更适合把它作为关键替代路线消融，而不是第一版唯一主线。

### Q4：为什么不用 GSPO / pass@k？

GSPO 更偏目标函数级创新，pass@k 更偏 sample-and-select 目标。它们都不是没价值，只是第一版会显著增加实现与答辩复杂度。当前项目主结果强调 `accepted@1`，因此更适合先把 GRPO family 的稳定主线做扎实。

### Q5：为什么 no-KL 是工程选择，不是信条？

coding 任务常常要求模型学习新的程序结构和边界处理模式，过强的 KL 容易限制探索，所以 no-KL 值得优先尝试。但如果 pilot run 里出现明显漂移，我们仍然预留 small-KL rescue run。这是工程上的风险控制，而不是意识形态选择。

### Q6：你如何证明这是路线级决策，而不是追热点？

我没有做组件级全因子拆解，而是做路线级比较：Stable GRPO、DAPO-style GRPO、Dr.GRPO alternative。这个设计更符合 coding RL 的真实训练成本，也足够支撑实现与答辩。最终主线的选择依赖的是验证集表现、稳定性和成本，而不是算法名字是否更新。

### Q7：为什么 reward 主线是 anchored dense，而不是更“保守”的 dense_anchor_v1？

因为当前正式主线 A0 / A1 仍然是标准 GRPO 口径，组内中心化和标准差归一化会明显削弱负截距的作用。在这种前提下，reward 主线更适合选择一个语义更简单、面试更好解释、同时又保留 outcome anchor 的版本。`dense_anchor_v1` 仍然保留为强对照，但不直接取代主线。

### Q8：为什么 reward 主线不加入 runtime / timeout penalty？

因为当前 verifier 已经能拿到 full-test `pass_ratio_all`，而 baseline 分析显示 mixed-case 很多。很多题虽然题目级错误类型是 `runtime_error` 或 `timeout`，但实际上已经通过了不少 testcase。此时如果再做整题 penalty，会误伤 partial correctness signal，不利于 coding RL 主线训练。

---

## 10. 最终执行口径

当前 Phase 3 的正式执行口径固定为：

- `A0` 作为 stable baseline
- `A1` 作为正式主线
- `A2` 作为关键替代路线
- `anchored_dense + guardrails` 作为 reward v1 主线
- `dense_anchor_v1 + same guardrails` 作为 reward 强对照

当前正式 headline 为：

- **`GRPO with DAPO-style stabilizations`**

在 reward v1 冻结、`filter_groups` 生效验证通过、A0/A1/A2 配置落盘之后，方可进入正式训练实现阶段。
