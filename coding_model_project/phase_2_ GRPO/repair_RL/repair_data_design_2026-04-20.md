# Repair RL Data Design (2026-04-20)

这份文档只负责一件事：

- 固定当前 `repair RL` 的数据范围、主数据源、缺失字段与过滤口径

它不负责：

- reward 公式本身
- `build_repair_rl_parquet.py` / `repair_grpo_batch_reward.py` 的实现细节
- 完整训练排期

相关文档：

- [repair_reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_RL/repair_reward_design.md)
- [repair_rl_engineering_spec_2026-04-20.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_RL/repair_rl_engineering_spec_2026-04-20.md)

---

## 1. 这次要确认的核心问题

当前要先回答四个问题：

1. 当前 `step1300` repair 线到底有多少条 first-pass repair 样本可用？
2. 如果采用另一个 agent 提出的数据范围：
   - `70%–80%`: `pass_ratio_first >= 0.6`
   - `20%–30%`: `0.2 <= pass_ratio_first < 0.6`
   真实可用量是多少？
3. 这些样本里是否已经有 `repair_delta_v0 / v1` 需要的关键字段？
4. 哪一层资产应该作为 repair RL 的主表，哪几层只能作为补充表？

---

## 2. 正式结论

### 2.1 repair RL 的主数据源不应该用 `final_keep` 或 `shortdiag_pure`

当前正式结论：

- **repair RL 主表应使用**
  - [student_references_step1300_repair_cond_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/student_references_step1300_repair_cond_v2.jsonl)
- **补充表应使用**
  - [full_teacher_requests_step1300_repair_cond_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/full_teacher_requests_step1300_repair_cond_v2.jsonl)
  - [codecontests_train_wo_valid_big_raw.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/raw/codecontests_train_wo_valid_big_raw.jsonl)

不推荐把下面两个直接当 repair RL 主表：

- [step1300_teacher_keep_set_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.jsonl)
- [step1300_shortdiag_pure_sft_dataset_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.jsonl)

原因很简单：

- `final_keep` / `shortdiag_pure` 是为 teacher-SFT 压缩过的产物，
- 它们更适合“teacher 成功修复样本”，
- 但 repair RL 需要的是“first-pass buggy sample + verifier feedback + 可重跑 test contract”。

### 2.2 当前最合适的 repair RL 口径

当前正式建议的数据口径是：

```text
base table:
    student_references_step1300_repair_cond_v2.jsonl

join #1:
    full_teacher_requests_step1300_repair_cond_v2.jsonl
    by (problem_id, prompt_sha256)

join #2:
    codecontests_train_wo_valid_big_raw.jsonl
    by (problem_id, prompt_sha256)
```

其中：

- `student_references` 提供 first-pass 的真实 verifier 状态与 first-pass code
- `full_teacher_requests` 提供现成的 `repair_feedback` 与 `request_id`
- raw dataset 提供 `test_cases`

### 2.3 `pass_ratio_first` 的正式含义

本轮文档里提到的 `pass_ratio_first`，正式映射为：

- `student_references.pass_ratio`
- 以及等价的 `student_references.pass_ratio_all`

当前检查结果：

- `pass_ratio` 与 `pass_ratio_all` 在 `1050 / 1050` 行上一致
- mismatch 数量为 `0`

因此后续实现里应统一标准字段名为：

- `first_pass_pass_ratio_all`

而不是继续混用：

- `pass_ratio`
- `pass_ratio_all`
- `first_pass_pass_ratio`

---

## 3. 当前资产盘点

### 3.1 candidate pool

文件：

- [candidate_pool_step1300_repair_cond_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/candidate_pool_step1300_repair_cond_v2.jsonl)
- [candidate_pool_step1300_repair_cond_v2.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/candidate_pool_step1300_repair_cond_v2.summary.json)

当前规模：

- 输入 curriculum 行数：`1064`
- candidate pool 保留：`1050`
- bucket：
  - `B_near_miss = 371`
  - `C_hard_partial = 679`

它的作用：

- 适合定义“哪些题进入 repair 资产池”
- **不适合直接做 repair RL 主表**

原因：

- 它没有 first-pass code
- 没有 accepted / finish_reason / invalid_for_rl
- 没有 repair prompt 所需的 verifier feedback
- 没有 test contract

### 3.2 student references

文件：

- [student_references_step1300_repair_cond_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/student_references_step1300_repair_cond_v2.jsonl)
- [student_references_step1300_repair_cond_v2.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/student_references_step1300_repair_cond_v2.summary.json)

当前规模：

- 总行数：`1050`
- `accepted_count = 20`
- 失败样本：`1030`

它已经具备的关键字段：

- `problem_id`
- `dataset`
- `prompt`
- `prompt_sha256`
- `full_extracted_code`
- `student_completion`
- `accepted`
- `pass_ratio`
- `pass_ratio_all`
- `error_type`
- `finish_reason`
- `details.invalid_for_rl`
- `details.invalid_reason`
- `details.extraction_status`
- `details.per_case_results`

这份表是当前最适合做 repair RL base table 的资产。

### 3.3 full teacher requests

文件：

- [full_teacher_requests_step1300_repair_cond_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/full_teacher_requests_step1300_repair_cond_v2.jsonl)
- [full_teacher_requests_step1300_repair_cond_v2.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/full_teacher_requests_step1300_repair_cond_v2.summary.json)

当前规模：

- request 数：`1030`

它已经具备的关键字段：

- `problem_id`
- `prompt_sha256`
- `request_id`
- `prompt`
- `full_extracted_code`
- `pass_ratio`
- `pass_ratio_all`
- `error_type`
- `repair_feedback`
- `teacher_prompt_mode`

它缺少的关键 first-pass verifier 字段：

- `accepted`
- `finish_reason`
- `invalid_for_rl`
- `invalid_reason`
- `extraction_status`

因此它**不能单独做主表**，但很适合作为补充表。

### 3.4 final keep

文件：

- [step1300_teacher_keep_set_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.jsonl)
- [step1300_teacher_keep_set_v2.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.summary.json)

当前规模：

- `553` 行

它的定位：

- 是 teacher 成功修复后留下的 keep set
- 适合 SFT
- 不适合直接当 repair RL 主表

当前检查到的关键缺口：

- `student_extracted_code` 只有 `32 / 553` 行非空
- `generation_finish_reason` 只有 `32 / 553` 行非空
- 没有 `test_cases`
- 数据语义已经偏向 teacher 成功样本

### 3.5 shortdiag pure SFT dataset

文件：

- [step1300_shortdiag_pure_sft_dataset_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.jsonl)
- [step1300_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.summary.json)

当前规模：

- `553` 行

它的定位：

- 是已经面向 SFT 的 messages 产物
- 不适合做 repair RL 主表

原因：

- 它保留了 `messages`
- 但已经丢失了 repair RL 需要的大量 first-pass verifier 原始字段

---

## 4. 当前可用数据量

### 4.1 以 `student_references` 为 base table 的真实可用量

先按 first-pass 真正需要的基础过滤：

- `accepted == false`
- 不看 teacher 是否修复成功

得到：

- 总失败样本：`1030`

再按另一个 agent 提出的范围：

- high：`pass_ratio_first >= 0.6`
- mid：`0.2 <= pass_ratio_first < 0.6`

得到：

- high：`250`
- mid：`440`
- high + mid 合计：`690`

剩余未进入本轮 repair RL 范围的失败样本：

- `< 0.2`：`340`

### 4.2 这 690 条样本的质量检查

当前 `690` 条 selected failures 满足：

- `finish_reason == "stop"`：`690 / 690`
- `details.invalid_for_rl == false`：`690 / 690`
- 与 `full_teacher_requests` 的 `(problem_id, prompt_sha256)` join 缺失数：`0`
- 与 raw dataset 的 `(problem_id, prompt_sha256)` join 缺失数：`0`

这说明：

- 这批样本当前数据质量是够高的
- 它们已经是一个非常合格的 repair RL 起始池

### 4.3 high / mid 不等于 B / C bucket

这一点必须明确写死：

- **不能**直接把 `B_near_miss` 当成 `>= 0.6`
- **不能**直接把 `C_hard_partial` 当成 `0.2 ~ 0.6`

真实分布是：

- high：
  - `B_near_miss = 182`
  - `C_hard_partial = 68`
- mid：
  - `B_near_miss = 152`
  - `C_hard_partial = 288`

所以这轮数据过滤必须按：

- `pass_ratio_first`

而不是按：

- `curriculum_bucket`

### 4.4 selected failures 的 error_type 组成

high (`250`)：

- `wrong_answer = 204`
- `timeout = 32`
- `runtime_error = 14`

mid (`440`)：

- `wrong_answer = 373`
- `timeout = 45`
- `runtime_error = 22`

这说明：

- 当前 proposed range 不是纯 `wrong_answer`
- 但 runtime / timeout 量级不大，仍然是可控补充项

---

## 5. audit-suspect 样本问题

当前 `step1300` repair 资产里存在一类不能忽略的风险：

- high-pass testcase issue
- judge instability

在 `final_keep` 中，来自下列 `teacher_keep_source` 的样本属于 audit-derived 可疑组：

- `remaining_high_pass_audit_0p85_to_0p9`
- `remaining_high_pass_audit_0p7_to_0p85`
- `near_miss_testcase_audit_high_confidence`
- `near_miss_testcase_audit_regen_round1`

这类行在 `final_keep` 里共有：

- `56` 行

把这些问题映射回 `student_references` 的当前 selected failures（high + mid）之后，可识别出的可疑样本数为：

- `37`
  - high：`19`
  - mid：`18`

这批样本对 SFT 可以是“teacher 修复示例”，
但对 RL 不一样：

- repair RL 的 reward 直接依赖 verifier
- 如果 testcase / judge 自己就有问题，reward 会被污染

因此本轮正式数据设计里应默认：

- **排除这 37 条 audit-suspect selected failures**

---

## 6. 正式数据范围设计

### 6.1 正式 base filter

repair RL 正式 base filter 写死为：

```text
source:
    student_references_step1300_repair_cond_v2.jsonl

keep only if:
    accepted == false
    finish_reason == "stop"
    details.invalid_for_rl == false
    pass_ratio_first >= 0.2
    source_split == "codecontests_train_wo_valid_big"

exclude:
    audit-suspect rows matched by (problem_id, prompt_sha256)
```

过滤后得到的 clean pool：

- clean high：`231`
- clean mid：`422`
- clean total：`653`

### 6.2 正式比例设计

当前接受另一个 agent 的方向：

- `70%–80%`: `pass_ratio_first >= 0.6`
- `20%–30%`: `0.2 <= pass_ratio_first < 0.6`

但正式写法应改成：

```text
high slice:
    first_pass_pass_ratio_all >= 0.6

mid slice:
    0.2 <= first_pass_pass_ratio_all < 0.6
```

### 6.3 这个比例下的真实可用量

在 clean pool 上，若坚持 high/mid 比例：

- `80 / 20` 最多可做：`288`
  - `231 high + 57 mid`
- `75 / 25` 最多可做：`308`
  - `231 high + 77 mid`
- `70 / 30` 最多可做：`330`
  - `231 high + 99 mid`

因此：

- 如果你坚持 `70%–80%` high 占比，
- 当前 clean pool 的实际 repair RL probe 规模大约就是 `288 ~ 330`。

### 6.4 当前正式推荐

当前正式推荐默认值是：

```text
repair RL v0 probe data:
    75 / 25 clean mix
    = 231 high + 77 mid
    = 308 rows
```

理由：

- 落在你接受的 `70%–80% / 20%–30%` 范围正中间
- 吃满当前 clean high slice
- 仍然保留一小部分 mid slice，避免 reward 完全只学 near-solved cases
- 数据规模足够做 first probe，但不会把 low-signal 样本一次性放太多进去

如果后续希望稍微放大 probe，而不改字段设计，可以再放到：

- `70 / 30`
- `330` 行

---

## 7. 字段审计：当前缺什么，哪里补

### 7.1 当前已经有的关键字段

以 `student_references` 为 base table，当前已经有：

| repair RL 需要的字段 | 当前来源 |
|---|---|
| `problem_id` | `student_references.problem_id` |
| `dataset` | `student_references.dataset` |
| `prompt` | `student_references.prompt` |
| `first_pass.code` | `student_references.full_extracted_code` |
| `first_pass.accepted` | `student_references.accepted` |
| `first_pass.pass_ratio_all` | `student_references.pass_ratio` / `pass_ratio_all` |
| `first_pass.error_type` | `student_references.error_type` |
| `first_pass.finish_reason` | `student_references.finish_reason` |
| `first_pass.invalid_for_rl` | `student_references.details.invalid_for_rl` |
| `first_pass.invalid_reason` | `student_references.details.invalid_reason` |
| `first_pass.extraction_status` | `student_references.details.extraction_status` |
| `per_case_results` | `student_references.details.per_case_results` |

### 7.2 当前还需要补的字段

当前最重要的缺口有两个：

| 缺口字段 | 补法 |
|---|---|
| `repair_feedback` | 直接从 `full_teacher_requests` join，或从 `student_references.details.per_case_results` 现算 |
| `test_cases` | 从 raw dataset join 补齐 |

### 7.3 join 设计已经可行

当前已经验证：

- `student_references` -> `full_teacher_requests`
  - join key：`(problem_id, prompt_sha256)`
  - selected failures 缺失：`0`
- `student_references` -> raw dataset
  - join key：`(problem_id, prompt_sha256)`
  - selected failures 缺失：`0`

因此 repair parquet builder 可以稳定采用：

```text
base:
    student_references

left join:
    full_teacher_requests
    on (problem_id, prompt_sha256)

left join:
    raw codecontests dataset
    on (problem_id, prompt_sha256)
```

### 7.4 对 `repair_delta_v1` 的影响

`repair_delta_edit_v1` 需要 `first_pass.code`。

这点当前不是 blocker，因为：

- `student_references.full_extracted_code`
  - 在 `1050 / 1050` 行上非空

所以：

- **数据层面对 `v1` 已经基本够用**
- `v1` 的额外工作主要是 builder / reward adapter / trainer 侧 plumbing

---

## 8. 当前正式不采用的资产

本轮先明确不采用：

- `final_keep` 直接做 repair RL 主表
- `shortdiag_pure` 直接做 repair RL 主表
- 按 `curriculum_bucket` 直接代替 `pass_ratio_first` 分桶
- 把 `< 0.2` 的 low / zero slice 一起并入首轮 probe
- 把 audit-suspect 样本混进首轮 RL 数据

---

## 9. Final Decision

当前正式数据设计结论是：

1. repair RL 主表使用：
   - [student_references_step1300_repair_cond_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/student_references_step1300_repair_cond_v2.jsonl)
2. 用 `(problem_id, prompt_sha256)` 补 join：
   - [full_teacher_requests_step1300_repair_cond_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/full_teacher_requests_step1300_repair_cond_v2.jsonl)
   - [codecontests_train_wo_valid_big_raw.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/raw/codecontests_train_wo_valid_big_raw.jsonl)
3. 首轮 repair RL 只保留：
   - `accepted == false`
   - `finish_reason == "stop"`
   - `invalid_for_rl == false`
   - `first_pass_pass_ratio_all >= 0.2`
4. 默认排除：
   - `37` 条 audit-suspect high/mid 样本
5. 首轮 probe 的正式推荐数据范围：
   - `75 / 25 clean mix`
   - `231 high + 77 mid`
   - `308` 行

这份结论确认之后，就可以进入下一步：

- 按这个数据口径写最终的 repair RL 实现计划。
