# 当前资产与数据治理总览（2026-04-07）

这份文档的目的只有一个：

**在当前训练和评测链条变复杂之后，给人一个“现在有哪些资产、它们是干什么的、当前最该关注哪些文件、数据清洗如何影响训练和评测”的总地图。**

如果你现在觉得目录、清洗流程、manifest、quarantine、curriculum review、eval 输出混在一起，这份文档就是用来把这些东西拆开的。

---

## 1. 一句话总图

当前 Phase 2 的数据链可以理解成 5 层：

1. **原始题目数据**
   - CodeContests raw + manifests
2. **共享数据治理**
   - `problem_quarantine_v3.json`
   - 处理“这题测试合同有没有问题”
3. **训练数据构建**
   - RL parquet builder
   - curriculum manifest builder / focused manifest
   - 处理“这题该不该进训练、该进哪个桶”
4. **运行期状态**
   - curriculum state snapshots
   - rollout dumps
   - checkpoint
5. **评测输出**
   - `delta69`
   - `valid_big500`
   - `valid117`
   - `canary`

最重要的区分是：

- **quarantine** 解决的是“脏题/脏测试”
- **curriculum** 解决的是“训练桶和训练重点”

它们有关系，但不是同一件事。

---

## 2. 当前最重要的资产是什么

如果你现在只想知道“我在训练和评测里最应该盯什么”，先看这一节。

### 2.1 数据治理主清单

当前最新共享 quarantine 主清单：

- [problem_quarantine_v3.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v3.json)

这是现在所有训练/评测数据治理的**风险状态单一真相来源**。

当前计数：

- `hard_blacklist = 780`
- `caution = 22`
- `unresolved = 83`

它的作用：

- `hard_blacklist`：应从训练构建和新资产构建里硬过滤
- `caution`：不建议进入高价值 seed / 高价值 eval 构建
- `unresolved`：保留治理状态，但默认不硬过滤

### 2.2 当前 quarantine 说明文档

- [problem_quarantine_v3_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/problem_quarantine_v3_guide.md)

当你忘了：

- `v3` 相比 `v2` 改进了什么
- 这轮 review queue 是怎么切的
- `v3` 为什么是三态

就看这个文档。

### 2.3 当前 curriculum 运行链核心代码

这 4 个文件是现在 curriculum RL 真正跑起来的核心：

- [step580_curriculum_builder.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_builder.py)
- [step580_curriculum_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_dataset.py)
- [step580_curriculum_sampler.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_sampler.py)
- [run_grpo_a1_resume580_to660_curriculum.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_grpo_a1_resume580_to660_curriculum.sh)

它们分别负责：

- builder：离线课程 manifest 和 eval seed
- dataset：runtime 注入 `index / bucket metadata`
- sampler：在线分桶和 curriculum 采样
- launcher：真正起训练

### 2.4 当前 curriculum 局部清洗和 focused manifest 资产

当前最相关的本地目录有两个：

- [curriculum_assets/step600_v3_review_local](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_review_local)
- [review_assets/step640_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1)

这两个目录分别代表：

1. `step600_v3_review_local`
   - focused manifest v3 的本地工作目录
   - 包含：
     - `curriculum_train_manifest_step600_v3.jsonl`
     - `c_hard_partial_review_candidates_step600_v3.jsonl`
     - `anchor_common_stabilizer_candidates_step600_v3.jsonl`
     - `step600_delta69_per_problem.jsonl`
     - `step620_v2_delta69_per_problem.jsonl`

2. `step640_v1`
   - pre-step640 的冻结 review 资产
   - 包含：
     - eval review queue
     - bucket actions
     - anchor actions
     - materialized ledgers
     - `problem_quarantine_v3` build summary

### 2.5 当前最该盯的 review 结果

当前已经完成并可直接消费的结果是：

- [problem_quarantine_review_ledger_step640_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1/problem_quarantine_review_ledger_step640_v1.jsonl)
- [curriculum_bucket_actions_step640_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1/curriculum_bucket_actions_step640_v1.jsonl)
- [curriculum_anchor_actions_step640_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1/curriculum_anchor_actions_step640_v1.jsonl)

这三份分别回答：

- eval 可疑题，应该进 `hard/caution/clean/needs_more_evidence` 哪个治理状态
- A/C 桶里哪些题应该移桶或删除
- 哪些新候选值得作为 `A_retention` 的 stabilizer addition

---

## 3. 资产目录按用途怎么分

### 3.1 原始数据层

路径：

- [data/raw](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/raw)
- [data/manifests](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/manifests)

最关键的文件：

- [codecontests_train_raw.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/raw/codecontests_train_raw.jsonl)
- [codecontests_train_wo_valid_big_raw.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/raw/codecontests_train_wo_valid_big_raw.jsonl)
- [codecontests_valid_raw.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/raw/codecontests_valid_raw.jsonl)
- [codecontests_valid_big_raw.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/raw/codecontests_valid_big_raw.jsonl)
- [codecontests_test_raw.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/raw/codecontests_test_raw.jsonl)

以及对应 manifest：

- [codecontests_train_manifest.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/manifests/codecontests_train_manifest.jsonl)
- [codecontests_train_wo_valid_big_manifest.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/manifests/codecontests_train_wo_valid_big_manifest.jsonl)
- [codecontests_valid_manifest.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/manifests/codecontests_valid_manifest.jsonl)
- [codecontests_valid_big_manifest.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/manifests/codecontests_valid_big_manifest.jsonl)
- [codecontests_test_manifest.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/manifests/codecontests_test_manifest.jsonl)

这些是所有后续 builder 的原料。

### 3.2 共享数据治理层

路径：

- [data/problem_quarantine_v1.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v1.json)
- [data/problem_quarantine_v2.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v2.json)
- [data/problem_quarantine_v3.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v3.json)
- [src/problem_quarantine.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/problem_quarantine.py)

这里：

- `v1` 是最初的人工已知名单
- `v2` 是初筛 + Claude review 后的三态主清单
- `v3` 是在 `pre-step640` review 后更新的最新主清单

### 3.3 quarantine 审计与 review 资产层

路径：

- [data/quarantine_audit](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit)
- [review_assets/step640_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1)

这两层的区别：

1. `data/quarantine_audit`
   - 偏向“全库初筛”和 `v2` 的形成过程
   - 例如：
     - [problem_contract_screen_summary_v1.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/problem_contract_screen_summary_v1.json)
     - [problem_quarantine_candidates_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/problem_quarantine_candidates_v2.jsonl)

2. `review_assets/step640_v1`
   - 偏向“当前这一轮继续治理”的冻结 review 队列和结果

### 3.4 curriculum 资产层

路径：

- [curriculum_assets](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets)

当前最相关的三个子目录：

1. [step600_v2_review_local](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v2_review_local)
   - `manifest v2` 时的人工 review 资产

2. [step600_v3_review_local](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_review_local)
   - focused manifest v3 的核心本地目录
   - 当前最重要

3. [step600_v3_qv2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_qv2)
   - 经过 quarantine v2 过滤后的 manifest 版本

### 3.5 SFT 资产层

路径：

- [sft_repair_data](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data)

当前更偏“历史背景 / 参考”的目录：

- `v1`
- `v2a`
- `v2b_step400`
- `final_v1`

如果你当前主线在做 curriculum RL，这部分不是最优先，但仍然重要，因为：

- 这里是 step400 那轮 repair + anchor + quarantine 经验的来源
- `step400_sft_v1_postmortem.md` 仍然能帮你理解后面为什么要保 retention、为什么不能只看小验证集

---

## 4. 当前数据清洗流程有哪些

当前有两条不同但相关的清洗流程：

1. **quarantine 治理流程**
2. **curriculum 数据清洗流程**

一定要分开理解。

---

## 5. quarantine 数据清洗流程

### 5.1 它解决什么问题

问题定义：

**“这道题的测试合同是否有问题？”**

例如：

- hidden tests 格式坏了
- query 数量和 header 对不上
- 区间越界
- marker contamination
- 截断或说明文本混入 input

### 5.2 它的当前主流程

当前主流程是：

1. 自动轻筛
   - [light_screen_problem_contracts.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/light_screen_problem_contracts.py)
2. review 资产整理
   - [prepare_problem_quarantine_review_assets.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/prepare_problem_quarantine_review_assets.py)
3. 构建 `problem_quarantine_v2`
   - [build_problem_quarantine_v2.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_problem_quarantine_v2.py)
4. 生成 pre-step640 review 资产
   - [prepare_step640_review_assets.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/prepare_step640_review_assets.py)
5. 标准化 review ledger
   - [materialize_step640_review_ledgers.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/materialize_step640_review_ledgers.py)
6. 构建 `problem_quarantine_v3`
   - [build_problem_quarantine_v3.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_problem_quarantine_v3.py)

### 5.3 当前它的输出是什么

最终输出就是：

- [problem_quarantine_v3.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v3.json)

以及辅助结果：

- [problem_quarantine_review_ledger_step640_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1/problem_quarantine_review_ledger_step640_v1.jsonl)
- [problem_quarantine_v3_build_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1/problem_quarantine_v3_build_summary.json)

### 5.4 它如何影响训练和评测

**对训练**

- `hard_blacklist` 应该从训练构建里过滤掉
- `caution` 不要拿去做高价值 seed
- `unresolved` 先保留，但继续治理

**对评测**

- `v3` 已经足够作为 clean overlay 的基础
- 但历史 benchmark 不能静默改口径
- 所以后续更合理的是：
  - `raw metrics`
  - `clean metrics (v3 overlay)`

而不是只报 clean 后的新数

---

## 6. curriculum 数据清洗流程

### 6.1 它解决什么问题

问题定义：

**“这道题是否适合作为当前 curriculum 训练样本？如果适合，它应该留在哪个桶？”**

它不是在问题脏不脏，而是在问：

- 该不该留在 `A_retention`
- 该不该降到 `B_near_miss`
- 该不该移到 `U_unseen`
- `C_hard_partial` 里哪些其实是噪声
- 哪些新题值得作为 `anchor stabilizer` 加进 `A`

### 6.2 它的当前主流程

当前 relevant 流程是：

1. 课程 manifest 构建
   - [step580_curriculum_builder.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_builder.py)
2. runtime dataset 注入
   - [step580_curriculum_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_dataset.py)
3. online sampler 分桶
   - [step580_curriculum_sampler.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_sampler.py)
4. 针对 `step600 -> v2 -> v3` 的人工清洗 / focused manifest
   - [export_curriculum_review_candidates.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/export_curriculum_review_candidates.py)
   - [apply_curriculum_review_decisions.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/apply_curriculum_review_decisions.py)
   - [prepare_focused_manifest_v3_assets.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/prepare_focused_manifest_v3_assets.py)
   - [apply_focused_manifest_v3.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/apply_focused_manifest_v3.py)

### 6.3 当前它的关键本地资产是什么

当前最该关注：

- [curriculum_train_manifest_step600_v3.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_review_local/curriculum_train_manifest_step600_v3.jsonl)
- [c_hard_partial_review_candidates_step600_v3.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_review_local/c_hard_partial_review_candidates_step600_v3.jsonl)
- [anchor_common_stabilizer_candidates_step600_v3.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_review_local/anchor_common_stabilizer_candidates_step600_v3.jsonl)
- [curriculum_bucket_actions_step640_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1/curriculum_bucket_actions_step640_v1.jsonl)
- [curriculum_anchor_actions_step640_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1/curriculum_anchor_actions_step640_v1.jsonl)

### 6.4 它如何影响训练

**对训练内容**

- 决定 A/B/C/U 哪些题被优先采样
- 决定 `A_retention` 是否过脏、`C_hard_partial` 是否过杂
- 决定是否额外加入少量 `anchor stabilizer`

**对训练目标**

- `A_retention`：守住已会题
- `B_near_miss`：提升接近做对的题
- `C_hard_partial`：保留少量 hard exposure
- `anchor stabilizer`：少量新增进 A 的稳定器样本

### 6.5 它如何影响评测

它**不会直接改动评测题本身**，但会通过训练数据选择影响：

- retention
- solve-set churn
- delta69
- valid_big500

所以：

- quarantine 影响评测“题是否可信”
- curriculum 影响评测“模型是否学得稳”

---

## 7. 当前训练和评测里，你最应该盯哪些资产

如果现在只想抓最关键的，我建议按这个优先级看：

### 第一优先级

1. [problem_quarantine_v3.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v3.json)
   - 当前最新的数据治理主清单

2. [problem_quarantine_v3_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/problem_quarantine_v3_guide.md)
   - 当前治理逻辑说明

3. [run_grpo_a1_resume580_to660_curriculum.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_grpo_a1_resume580_to660_curriculum.sh)
   - 当前 curriculum RL 启动脚本

### 第二优先级

4. [step580_curriculum_builder.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_builder.py)
5. [step580_curriculum_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_dataset.py)
6. [step580_curriculum_sampler.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_sampler.py)

### 第三优先级

7. [curriculum_train_manifest_step600_v3.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_review_local/curriculum_train_manifest_step600_v3.jsonl)
8. [curriculum_bucket_actions_step640_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1/curriculum_bucket_actions_step640_v1.jsonl)
9. [curriculum_anchor_actions_step640_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/review_assets/step640_v1/curriculum_anchor_actions_step640_v1.jsonl)

### 评测侧最该盯的

10. `delta69` 的 per-problem 结果
11. `valid_big500` 的 summary + per-problem 结果
12. 后续是否同时输出 `raw + clean(v3 overlay)`

这里要特别注意：

**评测前，不要静默切 benchmark 口径。**

也就是说：

- 历史结果继续看 `raw`
- 如果接入 `v3`，请同时算 `clean overlay`

---

## 8. 当前最容易混淆的几个概念

### 8.1 quarantine vs curriculum

- `quarantine`：题目测试合同是否有问题
- `curriculum`：题目是否适合当前训练桶

### 8.2 caution vs unresolved

- `caution`：确认有轻度真实问题
- `unresolved`：还没到可硬判的程度
  - 其中又分：
    - `clean`
    - `needs_more_evidence`

### 8.3 bucket actions vs anchor actions

- `bucket actions`
  - 处理已有 A/C 桶里的题
  - 动作：
    - `keep_A / keep_C / move_to_B / move_to_U / drop`

- `anchor actions`
  - 处理是否新增 `anchor stabilizer` 进 A
  - 动作：
    - `add_to_A / keep_out / drop`

---

## 9. 当前最建议的工作方式

如果你接下来还要继续做训练和评测，我建议：

1. 先把当前训练和评测都默认视为两层：
   - 模型训练层
   - 数据治理层

2. 训练相关变更时，先看：
   - `curriculum manifest`
   - `bucket/anchor actions`
   - sampler / dataset / launcher

3. 评测相关变更时，先看：
   - `problem_quarantine_v3.json`
   - 是否要做 `clean overlay`
   - 历史结果能否继续 raw 对比

4. 如果一个问题是：
   - “这题脏不脏” → 去看 quarantine
   - “这题该不该留在 A/C” → 去看 curriculum

---

## 10. 最后一句话

如果你现在只记住一句：

**当前最核心的主线文件是 `problem_quarantine_v3.json`（治理）、`step580_curriculum_*` 三件套（训练逻辑）、以及 `step600_v3_review_local + review_assets/step640_v1`（当前训练数据清洗与人工决策入口）。**

其他目录大多是这些主线的历史版本、辅助脚本或背景材料。
