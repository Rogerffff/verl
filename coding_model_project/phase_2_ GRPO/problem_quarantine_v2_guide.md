# Problem Quarantine v2 说明（2026-04-06）

本文档记录 2026-04-06 这一轮 CodeContests 数据“脏题/脏测试”初筛的目标、策略、产物和当前接入建议。

它对应的是：

- 自动轻筛脚本：[../src/light_screen_problem_contracts.py](../src/light_screen_problem_contracts.py)
- review 资产整理脚本：[../src/prepare_problem_quarantine_review_assets.py](../src/prepare_problem_quarantine_review_assets.py)
- 三态 quarantine 构建脚本：[../src/build_problem_quarantine_v2.py](../src/build_problem_quarantine_v2.py)
- 共享 helper：[../src/problem_quarantine.py](../src/problem_quarantine.py)
- 主清单：[../data/problem_quarantine_v2.json](../data/problem_quarantine_v2.json)

本文档的定位不是“证明全库所有脏题都已被发现”，而是把这一轮已经完成的治理过程和当前可用结论讲清楚，方便后续 RL / SFT / curriculum / eval 共用。

---

## 1. 为什么这轮要做 quarantine

在做 `step400` SFT 和后续 curriculum RL 过程中，我们已经确认过一批题目的隐藏测试本身存在明显问题，例如：

- `AIZU/p02353`
- `Codeforces/302/A`
- `HackerEarth/angles-2`
- `p03016 AtCoder Beginner Contest 129 - Takahashi's Basics in Education and Learning`（较轻，属于约束漂移）

这些问题不会只影响训练，还会同时影响：

1. reward / verifier 信号质量
2. curriculum seed 检索质量
3. checkpoint 间的评测可比性
4. SFT repair / anchor 的样本选择

因此需要把“已知严重污染”统一做成共享 quarantine 入口，而不是每条链路自己维护临时 blacklist。

---

## 2. 这轮筛查的范围

本轮自动轻筛覆盖了当前主线最相关的 5 个 CodeContests raw split：

- [../data/raw/codecontests_train_raw.jsonl](../data/raw/codecontests_train_raw.jsonl)
- [../data/raw/codecontests_train_wo_valid_big_raw.jsonl](../data/raw/codecontests_train_wo_valid_big_raw.jsonl)
- [../data/raw/codecontests_valid_raw.jsonl](../data/raw/codecontests_valid_raw.jsonl)
- [../data/raw/codecontests_valid_big_raw.jsonl](../data/raw/codecontests_valid_big_raw.jsonl)
- [../data/raw/codecontests_test_raw.jsonl](../data/raw/codecontests_test_raw.jsonl)

这里没有做“全量语义判题”，而是只做高置信的结构性轻筛。

---

## 3. 自动轻筛策略

### 3.1 目标

自动轻筛的目标是抓出“高信号的合同异常”，不是去自动证明一个题一定错误。

也就是说，这轮策略优先追求：

- 抓到明显坏题
- 给人审准备一个规模可控的 review queue
- 不因为过度激进而把大量正常题误伤进 hard blacklist

### 3.2 主要规则

当前轻筛规则包括：

1. `missing_tests`
  - `test_cases.tests` 为空或缺失
2. `malformed_tests`
  - hidden tests 没有 `input/output`
  - 或字段类型不对
3. `marker_contamination`
  - hidden tests 中出现明显的标记文本，例如：
    - `SAMPLE`
    - `EXPLANATION`
    - `OUTPUT`
    - `INPUT`
    - `RAMPLE`
    - `ELBMPT`
4. `uppercase_blob_tail`
  - 额外的大写 token 尾巴
  - 这条规则现在只作为弱信号使用，避免误杀合法的大写字符串题
5. `query_count_mismatch`
  - 题面强提示为 `n q` / `q lines of queries`
  - 但 hidden tests 实际 query 行数和题面合同明显冲突
6. `opcode_arity_inconsistency`
  - 数字 opcode 查询题中，同一 opcode 在 hidden tests 中出现不合理的 token 长度冲突
7. `range_violation`
  - 对明确的 `l_i, r_i` 区间查询题，hidden tests 出现不满足题面约束的区间
8. `second_line_length_mismatch`
  - 题面强提示“第二行/下一行有 N 或 K 个整数”
  - 但 hidden tests 第二行 token 数明显不符

### 3.3 这轮做过的误报控制

为了降低噪声，这轮已经对规则做过几轮收紧，重点包括：

- `uppercase_blob_tail` 不再单独作为 hard 证据
- query 规则会尝试跳过前导数组 / 边列表 / tree parent 列表
- `opcode_arity_inconsistency` 只在更像“typed query problem”的题面上触发
- 聚合后 `merged_signals.count` 改成单 split 最大计数，避免 `train` 和 `train_wo_valid_big` 重复放大

现在 `merged_signals` 的字段含义是：

- `count`: 单个 split 的最大计数
- `raw_count`: 跨 split 聚合的总计数
- `split_count`: 命中 split 数
- `split_names`: 命中的 split 名

例如：

- `Codeforces/1452/F` 现在是 `count=1, raw_count=2`
- `AIZU/p00628` 现在是 `count=50, raw_count=100`

---

## 4. 自动轻筛的第一阶段结果

自动轻筛第一阶段产物：

- split 级明细：[../data/quarantine_audit/problem_contract_screen_rows_v1.jsonl](../data/quarantine_audit/problem_contract_screen_rows_v1.jsonl)
- 聚合候选：[../data/quarantine_audit/problem_quarantine_candidates_v2.jsonl](../data/quarantine_audit/problem_quarantine_candidates_v2.jsonl)
- 手工审查摘要：[../data/quarantine_audit/problem_contract_screen_summary_v1.json](../data/quarantine_audit/problem_contract_screen_summary_v1.json)
- markdown 报告：[../data/quarantine_audit/problem_contract_audit_report_v1.md](../data/quarantine_audit/problem_contract_audit_report_v1.md)

核心数字：

- `split_candidate_row_count = 2996`
- `candidate_count = 1504`
- `hard_blacklist_new_count = 864`
- `manual_review_count = 636`
- `existing_hard_count = 3`
- `existing_caution_count = 1`

按 split 看：

- `codecontests_train`: `1492`
- `codecontests_train_wo_valid_big`: `1476`
- `codecontests_valid`: `6`
- `codecontests_valid_big`: `16`
- `codecontests_test`: `6`

这一步的结论是：

- 明显污染题已经能抓出来
- 但结构型 query 题误报仍然存在，不能直接把 `864` 全部当最终 hard blacklist

---

## 5. Review 资产是怎么切的

为了避免把全部 `manual_review` 都丢给 Claude，这一轮把 review 资产拆成了几层：

### 5.1 clear-hard-only

文件：

- [../data/quarantine_audit/review_assets_v1/problem_quarantine_auto_hard_v2.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_auto_hard_v2.jsonl)

它只包含：

- `hard_blacklist`
- 且命中 `marker/malformed/missing`
- 且**没有** structural overlap

当前数量：

- `auto_hard_count = 745`

注意：

- 这不是“所有明显脏题总数”
- 它只表示 `clear-hard-only`

summary 里已经明确了这个口径：

- [../data/quarantine_audit/review_assets_v1/problem_quarantine_review_summary_v1.json](../data/quarantine_audit/review_assets_v1/problem_quarantine_review_summary_v1.json)

### 5.2 structural review

文件：

- [../data/quarantine_audit/review_assets_v1/problem_quarantine_structural_review_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_structural_review_v1.jsonl)

这些题是：

- 自动规则认为可能 hard
- 但信号主要来自 query/range/opcode/second-line 这种结构规则
- 误报风险更高，必须人审

当前数量：

- `119`

### 5.3 train-only structural manual review

这是后续补上的一批，之前 review queue 一开始漏掉了。

文件：

- [../data/quarantine_audit/review_assets_v1/problem_quarantine_structural_manual_train_only_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_structural_manual_train_only_v1.jsonl)

当前数量：

- `9`

代表题：

- `Codeforces/1452/F`
- `Codeforces/301/D`
- `Codeforces/622/C`
- `Codeforces/940/F`
- `Codeforces/959/F`
- `CodeChef/botm`

### 5.4 eval-priority review

文件：

- [../data/quarantine_audit/review_assets_v1/problem_quarantine_eval_priority_review_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_eval_priority_review_v1.jsonl)

这些题来自：

- `valid`
- `valid_big`
- `test`

所以审核口径必须更保守。

当前数量：

- `28`

### 5.5 Claude review queue

最终给 Claude 的 review queue 是三者并集：

- structural review
- train-only structural manual review
- eval-priority review

文件：

- 第一版 queue: [../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_review_queue_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_review_queue_v1.jsonl)
- 修正后的 queue: [../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_review_queue_v2.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_review_queue_v2.jsonl)

修正后最终数量：

- `136`

---

## 6. Claude 人审结果

Claude 先审了主 queue，再补审了 `9` 条 supplement。

相关文件：

- 主 decision：
  - [../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_decisions_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_decisions_v1.jsonl)
- supplement decision：
  - [../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_decisions_supplement_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_decisions_supplement_v1.jsonl)

补充 review 的代表结论：

- `HackerEarth/climbing-ladder-1` -> `hard_blacklist`
- `HackerEarth/comrades-ii-6` -> `hard_blacklist`
- `HackerEarth/comrades-i-3` -> `caution`
- `CodeChef/botm` -> `needs_more_evidence`
- `Codeforces/301/D`、`622/C` -> `clean`
- `Codeforces/1452/F`、`940/F`、`959/F` -> `clean`

这一步的价值在于：

- 把结构型误报从 quarantine 主名单里剔出去
- 同时保住 eval 高价值切片不被自动规则误伤

---

## 7. 最终得到的三态 quarantine v2

最终主文件：

- [../data/problem_quarantine_v2.json](../data/problem_quarantine_v2.json)

构建摘要：

- [../data/quarantine_audit/review_assets_v1/problem_quarantine_v2_build_summary.json](../data/quarantine_audit/review_assets_v1/problem_quarantine_v2_build_summary.json)

当前最终数字：

- `hard_blacklist = 777`
- `caution = 32`
- `unresolved = 76`

共享 helper 也已经升级成三态接口：

- [../src/problem_quarantine.py](../src/problem_quarantine.py)

现在通过 helper 读主文件，可以拿到：

- `hard_blacklist_ids`
- `caution_ids`
- `unresolved_ids`
- `all_ids`

当前计数是：

- `hard = 777`
- `caution = 32`
- `unresolved = 76`
- `all_ids = 885`

---

## 8. 三态该怎么理解

### 8.1 hard_blacklist

定义：

- 已知明显坏测试 / 高置信污染题
- 应立即从训练和自动资产构建中剔除

本轮建议：

- RL train parquet builder：直接过滤
- curriculum manifest builder：直接过滤
- SFT asset builder：直接过滤
- eval asset builder：默认不纳入“clean metrics”

### 8.2 caution

定义：

- 存在真实但较轻的合同问题、约束漂移、或轻度结构异常
- 还不够直接拉进 hard blacklist

本轮建议：

- 不要直接全量硬删
- 但不要优先拿来做 curriculum seed / high-value eval seed
- 需要在后续 targeted audit 中继续跟进

### 8.3 unresolved

定义：

- 当前审查结论是 `clean` 或 `needs_more_evidence`
- 它们被保留在主清单里，是为了保留治理状态，而不是为了当前训练时一起硬过滤

注意：

- `unresolved` 不等于“脏题”
- 它表示“这题进入过治理流程，但当前结论不是 hard/caution”

---

## 9. 这份 v2 现在能不能直接用

可以，但要区分用途。

### 9.1 作为训练过滤名单

可以直接用。

当前下游大多数训练入口如果只读取：

- `hard_blacklist_ids`

那么行为是合理的，不会误把 `clean / needs_more_evidence` 一起删掉。

### 9.2 作为完整治理状态总表

现在也可以直接用。

因为主 JSON 和共享 helper 都已经支持三态，后续如果有人要统计：

- 已确认 hard
- 已确认 caution
- 尚未定案/保留状态

都能从同一份文件里拿到。

### 9.3 它是不是“最终真相”

不是。

更准确地说，它是：

- 当前这轮自动轻筛 + Claude 复核之后的**最新最佳已知** quarantine 主清单

它足够用于：

- 这一轮 `600 -> 630` 前的数据污染控制
- curriculum / parquet / eval builder 的统一接入

但后续仍然可能发现新的脏题，尤其是：

- 结构型 query 题
- 低样本数题
- 有轻度 contract drift 的题

---

## 10. 本轮最推荐的接入方式

如果当前要继续 `step600 -> 630` 的 curriculum / RL 主线，建议：

1. 直接接入 [../data/problem_quarantine_v2.json](../data/problem_quarantine_v2.json)
2. 训练过滤至少读取 `hard_blacklist_ids`
3. curriculum / seed 选择时，`caution` 不要优先进入高价值 seed
4. `unresolved` 先保留，不做硬过滤

一句话版：

- `hard_blacklist`: 现在就该拦
- `caution`: 先降权，不必全删
- `unresolved`: 保留治理状态，不作为这轮硬过滤依据

---

## 11. 相关产物索引

### 自动轻筛

- [../data/quarantine_audit/problem_contract_screen_rows_v1.jsonl](../data/quarantine_audit/problem_contract_screen_rows_v1.jsonl)
- [../data/quarantine_audit/problem_quarantine_candidates_v2.jsonl](../data/quarantine_audit/problem_quarantine_candidates_v2.jsonl)
- [../data/quarantine_audit/problem_contract_screen_summary_v1.json](../data/quarantine_audit/problem_contract_screen_summary_v1.json)
- [../data/quarantine_audit/problem_contract_audit_report_v1.md](../data/quarantine_audit/problem_contract_audit_report_v1.md)

### review 资产

- [../data/quarantine_audit/review_assets_v1/problem_quarantine_review_summary_v1.json](../data/quarantine_audit/review_assets_v1/problem_quarantine_review_summary_v1.json)
- [../data/quarantine_audit/review_assets_v1/problem_quarantine_auto_hard_v2.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_auto_hard_v2.jsonl)
- [../data/quarantine_audit/review_assets_v1/problem_quarantine_structural_review_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_structural_review_v1.jsonl)
- [../data/quarantine_audit/review_assets_v1/problem_quarantine_structural_manual_train_only_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_structural_manual_train_only_v1.jsonl)
- [../data/quarantine_audit/review_assets_v1/problem_quarantine_eval_priority_review_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_eval_priority_review_v1.jsonl)
- [../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_review_queue_v2.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_review_queue_v2.jsonl)
- [../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_decisions_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_decisions_v1.jsonl)
- [../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_decisions_supplement_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_claude_decisions_supplement_v1.jsonl)

### 最终主清单

- [../data/problem_quarantine_v1.json](../data/problem_quarantine_v1.json)
- [../data/problem_quarantine_v2.json](../data/problem_quarantine_v2.json)
- [../data/quarantine_audit/review_assets_v1/problem_quarantine_v2_build_summary.json](../data/quarantine_audit/review_assets_v1/problem_quarantine_v2_build_summary.json)
- [../data/quarantine_audit/review_assets_v1/problem_quarantine_unresolved_v1.jsonl](../data/quarantine_audit/review_assets_v1/problem_quarantine_unresolved_v1.jsonl)

