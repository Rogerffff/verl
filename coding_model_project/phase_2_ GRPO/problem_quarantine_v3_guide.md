# Problem Quarantine v3 说明（2026-04-06）

本文档记录 `problem_quarantine_v3` 这一轮数据清洗/治理的目标、策略、执行结果，以及它相对 `v2` 的改进点。

它对应的是以下脚本和产物：

- `v2` 主清单：[../data/problem_quarantine_v2.json](../data/problem_quarantine_v2.json)
- 共享 helper：[../src/problem_quarantine.py](../src/problem_quarantine.py)
- pre-step640 review 资产构建脚本：[../src/prepare_step640_review_assets.py](../src/prepare_step640_review_assets.py)
- review 结果标准化脚本：[../src/materialize_step640_review_ledgers.py](../src/materialize_step640_review_ledgers.py)
- `v3` 主清单构建脚本：[../src/build_problem_quarantine_v3.py](../src/build_problem_quarantine_v3.py)
- `v3` 主清单：[../data/problem_quarantine_v3.json](../data/problem_quarantine_v3.json)

这份文档的定位不是宣称“全库已经彻底洗干净”，而是明确：

1. `v3` 是怎么从 `v2` 继续往前推进的
2. 这轮新增治理到底改进了什么
3. 当前哪些结果已经足够接入训练/评测
4. 后续如果继续做 `v4+`，该沿什么方向增量推进

---

## 1. 为什么要从 v2 继续做到 v3

`problem_quarantine_v2` 已经完成了：

- CodeContests 5 个 raw split 的自动轻筛
- clear-hard 脏题的自动隔离
- 一轮 Claude 复核
- 三态主清单落地：
  - `hard_blacklist`
  - `caution`
  - `unresolved`

但在 `step600 -> step620 -> step640` 的 curriculum RL 迭代里，我们又遇到了两个更细的问题：

1. **评测可信度还需要再提高**
   - 某些 `eval_priority_review` 题仍然带有结构性误报
   - 需要对 `delta69 / valid_big / eval-critical` 相关题做更保守的人审

2. **训练清洗和 quarantine 治理不能混成一个动作体系**
   - `quarantine` 的问题是“这题测试合同有没有问题”
   - `curriculum` 的问题是“这题是否适合留在 A/C 桶，或是否值得做 anchor stabilizer”
   - 这两者不能再共用一个模糊的 action vocabulary

因此 `v3` 的目标不是重做一遍全库轻筛，而是把治理从“静态初筛”推进到：

- **更可信的 eval-critical 复核**
- **更清晰的三态主清单**
- **和 curriculum 人审动作解耦的治理账本**

---

## 2. v3 的核心策略

### 2.1 保持三态主清单，不新增第四个顶层 bucket

`v3` 沿用了 `v2` 的三态结构：

- `hard_blacklist`
- `caution`
- `unresolved`

这里特意没有把 `clean_reviewed` 加成第四个顶层 bucket。

原因是：

- `hard_blacklist / caution / unresolved` 是**风险状态**
- `clean_reviewed` 更像**审计 provenance**

所以在 `v3` 中：

- 风险状态仍由主清单承载
- “这题为何被复核、是谁复核、复核范围是什么、最终判 clean 还是 needs_more_evidence” 放进独立 ledger

### 2.2 把 quarantine 决策和 curriculum 动作彻底拆开

`v3` 明确把两类结果拆成不同账本：

1. **quarantine review ledger**
   - 只记录：
     - `hard_blacklist`
     - `caution`
     - `clean`
     - `needs_more_evidence`

2. **curriculum bucket actions**
   - 只记录当前 consumer 真能吃的动作：
     - `keep_A`
     - `keep_C`
     - `move_to_B`
     - `move_to_U`
     - `drop`

3. **curriculum anchor actions**
   - 单独处理 focused-manifest v3 的 stabilizer 增补：
     - `add_to_A`
     - `keep_out`
     - `drop`

这比 `v2` 更稳，因为不会再出现“审了一堆动作，但没有 builder 真正消费”的问题。

### 2.3 引入版本冻结和 pre-step640 queue

`v3` 没有等 `step640` 全部资产落齐后才开始，而是先基于“最新完整资产”冻结了一版：

- `queue_tag = step640_v1`
- `freeze_basis = latest_complete_assets_step620_v2`
- `is_pre_step640 = true`

对应 freeze manifest：

- [review_freeze_manifest_step640_v1.json](review_assets/step640_v1/review_freeze_manifest_step640_v1.json)

这一步的意义是：

- 避免训练继续推进时，人审 queue 和实际输入资产失配
- 让这轮外部人审有稳定输入，不需要随着 `step640` 中途改版

### 2.4 先做 eval-critical，再做 train-critical

这轮 `v3` 不是“再对全库打一遍”，而是优先吃高价值切片：

1. `eval-critical`
   - 先做去重 queue
   - 当前 queue 只保留真正需要重审的 28 题

2. `train-critical`
   - 先做 `A_retention` 高优先级样本
   - 再做 `C_hard_partial`
   - 再做 anchor stabilizer additions

这让治理顺序从“全库一锅端”变成了“先保 eval 可信，再保训练高价值切片干净”。

---

## 3. 这版相对 v2 的改进

这是 `v3` 最重要的部分。

### 3.1 从单次初筛升级为“可冻结、可复用”的 review 资产

`v2` 已经有 review 资产，但更多还是面向一次性的初筛和 Claude 复核。

`v3` 新增了：

- [../src/prepare_step640_review_assets.py](../src/prepare_step640_review_assets.py)

它会自动生成：

- freeze manifest
- eval review queue
- A/C bucket review queue
- anchor action review queue
- 各自的 decision template

也就是说，`v3` 已经不是“手工挑一些文件发给外部 agent”，而是**程序化地产出一轮审核资产**。

### 3.2 从“混合动作字段”升级为“两个账本”

旧计划里的动作字段曾经混杂了：

- quarantine 风险判定
- curriculum 移桶
- anchor 增补

`v3` 则把它拆成：

- quarantine review ledger
- bucket actions
- anchor actions

这让每一类人工审核结果都能明确落到下游 consumer 上，不再停留在 JSON 里。

### 3.3 从“重复审 eval 子集”升级为按 problem_id 去重

旧思路里如果分别发 `valid_big500` 和 `delta69` 去审，很容易重复审同一题。

`v3` 先把 eval 相关资产按 `problem_id` 去重，再在 row 上加：

- `slice_memberships`
- `latest_eval_roles`
- `current_quarantine_state`
- `current_quarantine_subdecision`

这样一题只审一次，但仍保留它在多个 eval 切片中的角色信息。

### 3.4 从“只看 unresolved 大类”升级为区分 subdecision

`v2` 里 `unresolved` 其实混有两类：

- `clean`
- `needs_more_evidence`

`v3` 明确把：

- `current_quarantine_state`
- `current_quarantine_subdecision`

写进 queue 和 ledger，后续筛查时就能更精细地区分：

- `caution`
- `unresolved(clean)`
- `unresolved(needs_more_evidence)`

这比简单地把所有 `unresolved` 一起重审更合理。

### 3.5 引入冲突消解协议

`v3` 的 ledger 构建里显式约定：

- 同一 `queue_tag` 下冲突决策直接报错
- 跨 `queue_tag` 取 `reviewed_at` 更新的结果
- `manual_review` 优先于自动推断

这让后续继续做 `v4+` 时，review 结果可以累加，而不是手工覆盖。

---

## 4. 这轮 v3 的实际执行结果

### 4.1 pre-step640 review 资产

对应目录：

- [review_assets/step640_v1](review_assets/step640_v1)

关键 summary：

- [review_asset_summary_step640_v1.json](review_assets/step640_v1/review_asset_summary_step640_v1.json)

本轮生成的立即审核 queue 规模是：

- eval review queue：`28`
- A bucket review queue：`26`
- C bucket review queue：`1`
- anchor review queue：`21`

### 4.2 外部 agent 审核结果

外部 agent 返回的决策已经被标准化成：

- [problem_quarantine_review_ledger_step640_v1.jsonl](review_assets/step640_v1/problem_quarantine_review_ledger_step640_v1.jsonl)
- [curriculum_bucket_actions_step640_v1.jsonl](review_assets/step640_v1/curriculum_bucket_actions_step640_v1.jsonl)
- [curriculum_anchor_actions_step640_v1.jsonl](review_assets/step640_v1/curriculum_anchor_actions_step640_v1.jsonl)

标准化 summary：

- [materialized_review_ledgers_step640_v1.json](review_assets/step640_v1/materialized_review_ledgers_step640_v1.json)

其中：

- quarantine review ledger：`28`
  - `clean = 22`
  - `hard_blacklist = 3`
  - `caution = 3`
- bucket actions：`27`
  - `move_to_U = 22`
  - `move_to_B = 4`
  - `drop = 1`
- anchor actions：`21`
  - `add_to_A = 3`
  - `keep_out = 17`
  - `drop = 1`

### 4.3 v3 主清单结果

当前主清单：

- [../data/problem_quarantine_v3.json](../data/problem_quarantine_v3.json)

构建摘要：

- [problem_quarantine_v3_build_summary.json](review_assets/step640_v1/problem_quarantine_v3_build_summary.json)

当前计数：

- `hard_blacklist = 780`
- `caution = 22`
- `unresolved = 83`

相对 `v2` 的变化：

- `hard_blacklist: 777 -> 780`
- `caution: 32 -> 22`
- `unresolved: 76 -> 83`

transition 摘要：

- `caution -> clean = 10`
- `caution -> hard_blacklist = 1`
- `caution -> caution = 2`
- `unresolved -> clean = 12`
- `unresolved -> hard_blacklist = 2`
- `unresolved -> caution = 1`

这说明 `v3` 的核心效果不是“盲目扩大 hard blacklist”，而是：

- 把一部分旧的 `caution` 纠正回 `clean`
- 同时抓出少量新的高置信 hard 问题
- 让主清单状态更接近真实风险分布

---

## 5. 当前接入建议

### 5.1 训练/资产构建

如果下游只是做硬过滤，当前应优先接入：

- [../data/problem_quarantine_v3.json](../data/problem_quarantine_v3.json)

并继续沿用：

- `hard_blacklist`：硬过滤
- `caution`：避免作为高价值 eval seed / curriculum seed
- `unresolved`：保留，但继续跟踪

### 5.2 curriculum / focused manifest

当前的：

- [curriculum_bucket_actions_step640_v1.jsonl](review_assets/step640_v1/curriculum_bucket_actions_step640_v1.jsonl)
- [curriculum_anchor_actions_step640_v1.jsonl](review_assets/step640_v1/curriculum_anchor_actions_step640_v1.jsonl)

已经足够作为：

- focused-manifest v3 的人工输入
- A/C 桶局部清洗的 adapter 输入

但要注意：

- `bucket_actions` 目前是**局部高优先级 queue** 的结果
- 它不是全量 A/C 清洗结果
- 所以不能直接拿来当“全量 A/C cleaned manifest”

### 5.3 eval / clean metrics

`v3` 已经为 clean metrics 铺好了基础：

- eval queue 去重
- freeze manifest
- quarantine review ledger

但当前还没有把 clean metrics 全面接进所有 eval 脚本。

所以当前更准确的表述是：

- `v3` 已经建立了 clean metrics 的治理协议基础
- 但 clean metrics 还不是所有评测脚本的默认输出

---

## 6. 这版还没有解决什么

`problem_quarantine_v3` 仍然不是“最终真相”。

当前还没有完全做完的包括：

1. `step640` 落盘后的 delta refresh
2. 更系统的 eval clean allowlist 固化
3. 更广的 train-critical 复核
4. 更精细的自动规则 v3

所以当前更合理的认知是：

- `v3` 是 `v2` 之后的一个高价值增量治理版本
- 它已经足够用于当前训练/评测的污染控制
- 但后面仍然可以继续往 `v4+` 推

---

## 7. 当前最重要的 takeaway

如果只记住一句话，那就是：

**`problem_quarantine_v3` 的价值不只是“又多抓了几道脏题”，而是把 quarantine 从一次性的初筛，推进成了一个可以冻结版本、复用 queue、拆分账本、并与 curriculum 人审动作协同工作的治理流程。**

这也是 `v3` 相对 `v2` 最实质的改进。
