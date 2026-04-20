# Step900 Teacher Generation And QC Plan

## 0. Scope

这份文档只覆盖当前 `step900 repair-conditioned SFT` 的下一段：

- `teacher generation`
- `candidate normalization`
- `verifier QC`
- `stability rejudge`
- `verified keep set` 的产物定义

它讨论的是 `step900 repair-conditioned SFT` 的 teacher 侧流程。

2026-04-13 的状态更新是：

- `Stage B/C` 的 judge 协议已经切到：
  - `patched sandbox`
  - `client RR`
  - `direct backend URL`
  - `8081..8088`
- `SelectiveC` 已经从 probe 规模扩到：
  - `44 requests`
  - `88 generation units`
- 但这轮 `SelectiveC` 的真实语义不是“纯 reviewed-C”
  - 而是：
    - `legacy_reviewed_c = 9`
    - `heuristic_supplemental_c = 35`
- teacher request / shard 侧也已经补齐：
  - `source_truncated`
  - `source_char_cap`
  - `source_provenance`
  - `selection_mode`
  - `c_trust_class`
  - `c_source_tag`
  - `c_decision_reason`
  - `c_reviewer`

因此，这份文档里更早的这些表述都应视为**历史口径**，不再作为当前执行基线：

- `step900_candidate_v1`
- `241 requests / 348 generation units`
- `SelectiveC too small / tiny C probe`
- “teacher generation 还不能开始”的 pre-gate 表述

当前真正的执行基线是：

- audited local teacher requests
- audited local teacher shards
- `281 requests / 429 generation units / 9 shards`

除非某一节显式给出更具体的本地绝对路径，
否则这份文档后面仍残留的旧 `*_step900_repair_cond_v1*` 文件名，
都应视为 **stage-name placeholder**，
实际执行时统一映射到当前：

- `step900_candidate_v2_backend_rr`
- audited local shard / request namespace

本轮也 **不包含**：

- 最终 parquet 组装
- 训练脚本
- prompt ablation 的第二条 `code-only` 分支

当前目标是先把：

- 一套已经 canonicalize 且补齐 provenance 的 teacher requests

稳定地转成：

- 一批高精度、可追溯、可复判的 repair-conditioned teacher AC 样本

---

## 1. Current Inputs

### 1.0 Current canonical execution assets

当前真正进入 teacher generation / QC 的输入资产是：

- audited teacher requests  
  `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/teacher_generation_requests_step900_candidate_v2_backend_rr.jsonl`
- audited teacher request summary  
  `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/teacher_generation_requests_step900_candidate_v2_backend_rr.summary.json`
- enriched student references  
  `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/student_references_step900_candidate_v2_backend_rr_enriched.jsonl`
- shard manifest  
  `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/manifest.json`

其中 upstream first-pass / judge 口径已经固定为：

- `patched sandbox`
- `client RR`
- `direct backend URL`
- `8081..8088`

### 1.1 Current canonical queue statistics

当前 canonical queue 的真实规模是：

- `request_count = 281`
- `generation_units = 429`
- `Core = 133`
- `Expansion = 104`
- `SelectiveC = 44`

其中：

- `teacher_best_of_n = 1` for `Core`
- `teacher_best_of_n = 2` for `Expansion / SelectiveC`

### 1.2 Current `SelectiveC` semantics

这轮 `SelectiveC` 已经不再是 tiny probe，
但它也不是“纯 reviewed-C”。

当前 `SelectiveC = 44 requests` 的组成是：

- `legacy_reviewed_c = 9`
- `heuristic_supplemental_c = 35`

这里要按历史背景理解这两个来源：

- `legacy_reviewed_c`
  - 对应的是 `step600` 冷启动阶段的 bootstrap C
  - 当时因为 `C` 桶几乎为空，才需要对从训练集按评测信号启发式搜出的题做人工 review
- `heuristic_supplemental_c`
  - 主要对应 `step600 -> step900` 训练过程中真实进入 `C_hard_partial` 的题
  - 再从这些更贴近课程训练分布的 C 题里，按当前信号筛出 retained C

因此本轮更准确的描述应该是：

- `Core + Expansion` 主体
- 加一个已经可用、但属于 mixed C provenance 的 `SelectiveC`

后续 QC / keep accounting / 汇报中，
都应该保留这个区分，
但不要把它解释成“reviewed 一定更高质量”的排序。

更稳妥的表述是：

- `legacy_reviewed_c` = bootstrap reviewed-C provenance
- `heuristic_supplemental_c` = current-training retained-C provenance
- 二者要分来源记账
- 不自动做质量高低假设

### 1.3 Current request-side provenance contract

当前 request / shard 侧已经显式携带：

- `source_truncated`
- `source_char_cap`
- `source_provenance`
- `review_status`
- `selection_mode`
- `c_trust_class`
- `c_source_tag`
- `c_decision_reason`
- `c_reviewer`

因此 teacher 侧的 reject ledger / QC / verified keep set，
都不应再依赖脆弱的 upstream 回表来证明：

- source-integrity hard gate
- `SelectiveC` 的 provenance 分组

这里字段名 `c_trust_class` 只是兼容当前资产的历史命名；
解释时应把它视为：

- `C provenance label`
- 不是质量分数，也不是置信分数
- heuristic vs legacy reviewed C 的来源差异

---

## 2. Goal

本轮 teacher generation / QC 的目标不是“尽快拿到一堆看起来不错的答案”。

真正目标是：

1. 用外部 teacher 为每条 request 产出 repair candidate
2. 把这些 candidate 归一化成统一 schema
3. 在 `patched sandbox + client RR` 下做严格 verifier QC
4. 对所有 provisional AC 再做 stability rejudge
5. 只保留：
   - extraction 正常
   - finish reason 合法
   - verifier 通过
   - 复判稳定
   的 teacher AC 样本

一句话说：

**这一阶段要追求的是“高精度 verified repair corpus”，不是单纯的 teacher 覆盖率。**

---

## 3. High-Level Flow

canonical 版本建议固定成下面 8 步：

1. 冻结当前 audited local queue 的 C provenance 解释
2. 复用当前 canonicalized Stage B/C 资产
3. `teacher_requests` 扩成 `generation_units`
4. `generation_units` 切成 Claude Code shard
5. Claude Code 逐 shard 生成 raw responses
6. raw responses 归一化成 `teacher_candidates`
7. primary QC verifier pass
8. 对 provisional accepts 做 stability rejudge，并生成 `verified_keep_set`

---

## 4. Stage D0: Expand Requests Into Generation Units

### 4.1 Why expand first

不要直接把 `teacher_best_of_n = 2` 留给 Claude Code 在一个 request 里自由返回两个答案。

应该先把 request 展开成独立 generation unit。

原因：

- 每个 generation unit 只要求一个 completion，接口最简单
- 后续 normalization / QC / retry 都是按单 candidate 做
- 可以避免一个 request 里混出两个 completion、后处理变脏
- 更适合分 shard、重试、补跑

### 4.2 Expanded unit schema

建议从当前 audited local requests：

- `teacher_generation_requests_step900_candidate_v2_backend_rr.jsonl`

派生出 generation units。

每条 generation unit 至少包含：

- `generation_unit_id`
- `request_id`
- `attempt_index`
- `attempt_count`
- `problem_id`
- `dataset`
- `source_split`
- `curriculum_bucket`
- `repair_stratum`
- `teacher_prompt_mode`
- `prompt`
- `prompt_sha256`
- `student_completion`
- `full_extracted_code`
- `source_truncated`
- `source_char_cap`
- `source_provenance`
- `error_type`
- `pass_ratio`
- `pass_ratio_all`
- `passed_tests`
- `total_tests`
- `judge_time_s`
- `review_status`
- `repair_feedback`

### 4.3 ID convention

建议：

- `generation_unit_id = "{request_id}__attempt{idx}"`

例如：

- `step900_rc_aizu_p00950__attempt1`
- `step900_rc_cf_1234_a__attempt2`

这样后续：

- Claude 输出
- candidate normalization
- QC
- rejudge

都能稳定按 `generation_unit_id` 对齐。

---

## 5. Stage D1: Shard For Claude Code

### 5.1 Why shard

`429` 个 generation units 不适合一次性塞给 Claude Code。

推荐切 shard，原因：

- 降低单次上下文压力
- 降低长任务中途失败的返工成本
- 更方便按 stratum 看质量
- 某个 shard 出问题时可以局部重跑

### 5.2 Recommended shard plan

在 canonical queue 重新冻结之前，
不要把 shard 数量和每 shard 行数当成硬规则。

当前 audited local queue 已经切成 `9` 个 shard。

推荐分法：

- `core_01`: `45`
- `core_02`: `45`
- `core_03`: `44`
- `exp_01`: `40`
- `exp_02`: `40`
- `exp_03`: `40`
- `exp_04`: `41`
- `selc_01`: `12`

其中：

- `Core` 每 request 只 1 条 generation unit
- `Expansion / SelectiveC` 已经在 D0 里按 `best_of_n=2` 展开

### 5.3 Shard outputs

建议每个 shard 单独落成：

- `teacher_generation_shards/shard_core_01.jsonl`
- `teacher_generation_shards/shard_core_02.jsonl`
- ...

并额外生成：

- `teacher_generation_shards/manifest.json`

记录：

- shard 名称
- row count
- stratum
- generation unit id 范围

---

## 6. Stage D2: Claude Code Generation Contract

### 6.1 Teacher role

当前默认 teacher 是：

- Claude Code

它的任务不是做 judge，也不是做后处理。

它只负责：

- 读取一个 shard 的 generation units
- 逐条按 `short_diagnosis_code` 模式生成 repair output
- 把结果写回规定 schema 的 JSONL

### 6.2 Teacher-visible input

每条 generation unit 给 teacher 的有效输入应固定成：

- 原始题面 `prompt`
- student 错代码 `full_extracted_code`
- grounded `repair_feedback`
- 固定的 `teacher_prompt_mode`

不要额外把：

- queue 构造逻辑
- curriculum 设计解释
- selection strategy 的实验元信息

暴露给 teacher。

teacher 需要 grounded signal，不需要 pipeline 元信息。

### 6.3 Raw response schema

Claude Code 返回的 raw responses 建议统一成：

- `generation_unit_id`
- `request_id`
- `attempt_index`
- `teacher_prompt_mode`
- `teacher_model`
- `teacher_completion`
- `generation_finish_reason`
- `notes`

其中：

- `teacher_completion` 是完整原始文本
- `generation_finish_reason` 建议固定写 `stop`
- `notes` 可选，默认空串

### 6.4 One completion per unit

每个 generation unit 只能返回：

- **一个** completion

不要在一条记录里塞：

- 多个 code block
- 多个 candidate
- 带分析再带第二个版本的“补充答案”

如果要多个 candidate，应该在 D0 就展开成多个 generation unit。

---

## 7. Stage D3: Normalize Teacher Candidates

### 7.1 Goal

Claude Code 返回的 raw JSONL 要先规范化，再进入 QC。

建议生成：

- `teacher_candidates_step900_repair_cond_v1.jsonl`

### 7.2 Candidate schema

每条 candidate 至少包含：

- `generation_unit_id`
- `request_id`
- `attempt_index`
- `problem_id`
- `dataset`
- `source_split`
- `curriculum_bucket`
- `repair_stratum`
- `teacher_prompt_mode`
- `prompt_sha256`
- `student_completion`
- `full_extracted_code_student`
- `source_truncated`
- `source_char_cap`
- `source_provenance`
- `teacher_completion_raw`
- `generation_finish_reason`
- `teacher_model`

这里先不要做 verifier 结论，只做 normalization。

### 7.3 Normalization rules

必须做到：

- `generation_unit_id` 唯一
- 每个 shard 的 generation unit 全覆盖
- 不允许重复返回同一 `generation_unit_id`
- 缺失条目必须显式报错，不允许静默跳过

---

## 8. Stage E1: Metadata And Extraction Preflight

在跑 sandbox verifier 之前，先做一层便宜但严格的 preflight。

### 8.1 Hard reject conditions

以下情况直接 reject：

- `generation_finish_reason != "stop"`
- 没有 `<code>...</code>`
- `normalize_candidate(...).extraction_status != "ok"`
- 提取出的 code 为空
- 单条输出里出现多个互相冲突的 `<code>` block

### 8.2 Preflight outputs

建议额外产出：

- `teacher_preflight_results_step900_repair_cond_v1.jsonl`
- `teacher_preflight_summary_step900_repair_cond_v1.json`

用于记录：

- finish reason 分布
- extraction status 分布
- 格式拒绝数
- 每个 shard 的 completeness

---

## 9. Stage E2: Primary Verifier QC

### 9.1 QC environment

primary QC 必须固定使用：

- `patched sandbox`
- `client RR`
- 与 first-pass 一致的 sandbox URL 集合

当前推荐：

- `http://localhost:8081,http://localhost:8082,http://localhost:8083,http://localhost:8084,http://localhost:8085,http://localhost:8086,http://localhost:8087,http://localhost:8088`

也就是：

- 直接打 backend URL
- 由 client 侧做 RR
- 不走 nginx 多-upstream LB

如果 direct-backend RR 在当前并发下仍然存在明显噪声，
应该优先：

- 降低 QC 并发
- 提高 stability rejudge 严格度

而不是回退到 `8090..8093` 这种 RR-over-LB 路径。

### 9.2 QC inputs

每条 teacher candidate 用：

- teacher 提取代码
- `problem_id`
- 训练集 test cases

跑一遍 verifier。

### 9.3 QC outputs

建议生成：

- `teacher_qc_primary_step900_repair_cond_v1.jsonl`
- `teacher_qc_primary_step900_repair_cond_v1.summary.json`

每条至少包含：

- `generation_unit_id`
- `request_id`
- `attempt_index`
- `problem_id`
- `repair_stratum`
- `teacher_prompt_mode`
- `teacher_completion_raw`
- `teacher_code`
- `source_truncated`
- `source_char_cap`
- `source_provenance`
- `accepted`
- `pass_ratio`
- `error_type`
- `judge_time_s`
- `invalid_for_rl`
- `qc_status`
- `reject_reason`
- `primary_qc_pass`

### 9.4 Keep rule after primary QC

primary QC 后只把下面这类样本送去 stability rejudge：

- `accepted == true`
- `pass_ratio == 1.0`
- `error_type == success`
- `invalid_for_rl == false`

其他样本先保留在 reject ledger，不进入下一步。

---

## 10. Stage E3: Stability Rejudge

### 10.1 Why rejudge is mandatory

当前项目已经确认：

- patched sandbox + client RR 仍然不是 100% judge-deterministic

所以：

- 单次 `accepted=true`

还不足以直接进入最终高精度 SFT 集。

### 10.2 Rejudge policy

对所有 provisional accepts：

1. 再跑一次相同 verifier，得到 `rejudge2`
2. 如果 `run1=AC` 且 `run2=AC`：
   - 直接 `stable_accept`
3. 如果 `run1/run2` 不一致：
   - 再跑一次 `rejudge3`
   - 采用 `2-of-3` 规则

### 10.3 Reject rules

以下情况 reject：

- `AC / non-AC / non-AC`
- `AC / AC / sandbox_error` 这种带 infra 污染、无法稳定确认的边界情况
- 任一复判出现明显 extraction / format 问题

### 10.4 Stability outputs

建议生成：

- `teacher_qc_stability_step900_repair_cond_v1.jsonl`
- `teacher_qc_stability_step900_repair_cond_v1.summary.json`

字段至少包含：

- `generation_unit_id`
- `primary_qc_status`
- `run1_status`
- `run2_status`
- `run3_status`
- `stable_keep`
- `stability_label`
- `final_decision`

其中 `stability_label` 建议取值：

- `stable_2_of_2`
- `stable_2_of_3`
- `unstable_reject`
- `infra_reject`

---

## 11. Final Keep Set Definition

最终进入下一阶段 SFT 组装的 keep set 必须同时满足：

1. preflight extraction 正常
2. finish reason 合法
3. primary QC `accepted=true`
4. stability rejudge 通过
5. 不属于人工 blacklist / manual reject

建议最终生成：

- `teacher_verified_keep_step900_repair_cond_v1.jsonl`

每条是一个已经稳定验证通过的 teacher repair 样本。

建议在最终 keep set 中也保留：

- `source_truncated`
- `source_char_cap`
- `source_provenance`
- `primary_qc_status`
- `stability_label`
- `run1_status`
- `run2_status`
- `run3_status`

这样后续不用再跨 4 份中间文件回溯：

- source 是否完整
- primary QC 是否通过
- stability 是 `2/2` 还是 `2/3`

同时保留：

- `teacher_reject_ledger_step900_repair_cond_v1.jsonl`

这样后面我们还能分析：

- 哪类请求最容易失败
- 哪类是 extraction 问题
- 哪类是 judge instability

---

## 12. Quota Control

不要让最终 kept set 自然塌缩成全 `Core`。

建议在 stability QC 结束后，再做一次配额检查。

目标区间：

- `Core: 50% ~ 65%`
- `Expansion: 25% ~ 35%`
- `SelectiveC: 10% ~ 15%`

如果最终 kept set 偏差过大，处理顺序应是：

1. 先补跑 `Expansion`
2. 再补跑 `SelectiveC`
3. 不优先继续加 `Core`

也就是说：

- request 阶段的 `134/101/6`
- 不是最终训练集的强制配比

最终训练集比例应该按 **verified keep set** 再校正。

---

## 13. Suggested File Layout

不要把新的 canonical rerun 继续写回当前：

- `provisional_repair_conditioned/step900_candidate_v1/`

这套路径应保留给：

- 当前 RR-over-LB 的 provisional 资产

建议新的 canonical teacher-generation 线固定在一个新 namespace，例如：

- `phase_2_ GRPO/sft_repair_data/provisional_repair_conditioned/step900_candidate_v2_backend_rr/teacher_generation/`

或在 freeze 之后直接切到：

- `phase_2_ GRPO/sft_repair_data/step900_repair_conditioned_canonical_v1/teacher_generation/`

然后把这一阶段的输出固定在该 canonical namespace 下：

- `.../teacher_generation/`

下面，分成：

- `generation_units_step900_repair_cond_v1.jsonl`
- `teacher_generation_shards/`
- `teacher_raw_responses/`
- `teacher_candidates_step900_repair_cond_v1.jsonl`
- `teacher_preflight_results_step900_repair_cond_v1.jsonl`
- `teacher_qc_primary_step900_repair_cond_v1.jsonl`
- `teacher_qc_stability_step900_repair_cond_v1.jsonl`
- `teacher_verified_keep_step900_repair_cond_v1.jsonl`
- `teacher_reject_ledger_step900_repair_cond_v1.jsonl`
- `teacher_generation_manifest_step900_repair_cond_v1.json`

这样路径会比把所有文件平铺在根目录里更清楚。

---

## 14. Execution Order

建议严格按下面顺序执行：

1. freeze 当前 `SelectiveC` 的 provenance 解释
2. 复用当前 audited local canonical assets
3. 把 teacher 侧产物写入新 namespace
4. materialize `generation_units`
5. cut shard manifests
6. 你 review shard format
7. 我给你 Claude Code teacher prompt
8. 你把 shard 发给 Claude Code 生成 raw responses
9. raw responses 回收后做 normalization
10. 跑 primary QC
11. 跑 stability rejudge
12. 生成 verified keep set

不要跳步直接让 teacher 先生成。

原因很简单：

- 一旦 raw generation 开始，大部分 schema 和命名就应该冻结
- 否则会出现 “老师已经生成了一批，但后处理字段名又改了” 的返工

---

## 15. What Needs Review Before Generation

在真正发给 Claude Code 之前，建议你重点 review 这 8 件事：

1. 是否接受当前 teacher 输入绑定 audited local `step900_candidate_v2_backend_rr`
2. 是否接受把 `281 requests` 展开成 `429 generation units`
3. 是否接受当前 `9` 个 Claude Code shard
4. 是否接受当前只跑 `short_diagnosis_code`
5. 是否接受 `SelectiveC` 明确按 `legacy_reviewed_c + heuristic_supplemental_c` 记账
6. 是否接受 `AC -> rejudge2 -> 必要时 rejudge3` 的稳定性 QC
7. 是否接受最终 keep set 只基于稳定 verified AC
8. 是否接受这一轮先不做 `code-only` teacher 数据

如果这 6 条没问题，下一步我就可以直接给：

- Claude Code 的 generation prompt
- raw response schema
- shard manifest 使用说明

---

## 16. Recommended Immediate Next Step

当前最自然的下一步就是直接基于当前 audited local queue 开始 teacher generation。

实现层已经具备：

- audited `teacher_generation_requests`
- audited local shard manifests
- per-shard Claude Code prompts

所以你 review 完这份文档后，
后续发给 Claude Code 的 prompt 应直接绑定到：

- 一个具体 shard 输入文件
- 一个明确 raw output 文件

而不是停留在“请帮我生成一批 repair 数据”的模糊描述上。
