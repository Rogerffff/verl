# Step900 Repair-Conditioned SFT Data Plan

## 0. Decision

当前最适合落地的 Step 3 方案是：

- **checkpoint anchor**：`step900`，但只作为 **provisional anchor**
- **task type**：`repair-conditioned SFT`
- **target format**：`short_diagnosis+code`
- **data source**：训练集 `train_wo_valid_big` 为主，必要时补 `train`
- **queue strategy**：
  - 不是只收 `pass_ratio >= 0.6`
  - 也不是直接把全量 `C` 灌进去
  - 而是采用 **`B_high + B_mid + selected_C` 三层混合**

一句话总结：

**先把 `step900` 作为 provisional repair anchor，用它的 curriculum bucket 当骨架，再叠加 canonical first-pass repairability 证据，构造一版高精度但不只会修最后一公里的 repair-conditioned SFT 数据。**

但这里有一个前置 gate：

- 在大规模 teacher 数据生成前，必须先完成 **同协议、同 sandbox 口径下的 `step900 vs step1300` reuse-first-pass final check**
- 只有当这个 final check 继续支持：
  - `step900` 是更强的 repair base
  - `step1300` 只是 partial-credit / near-miss probe
  才正式冻结 `step900`

在这一步完成前，文档中的 `step900` 都应理解为：

- **candidate default**
- 不是 irrevocable final anchor

---

## 1. Goal

这版 SFT 的目标不是直接提升 single-shot `accepted@1`。

它要优化的是：

- 给定
  - 题面
  - student 错代码
  - grounded verifier feedback
- 输出
  - 极短诊断
  - 极短修复计划
  - 修正后的完整代码

也就是：

- **提升“看到反馈后修代码”的能力**
- 而不是直接提升“单轮一次做对”的能力

对应项目表述应固定成：

- RL 主线：
  - 提升 single-shot `accepted@1`
- repair-conditioned SFT 支线：
  - 提升 verifier-guided self-repair

后者如果有效，再决定要不要做：

- `repair-distilled single-turn SFT`

去反蒸馏回单轮模型。

---

## 2. Why `step900`, Not `step1300`

当前已有受控评测说明：

- `step1300` 是更强的 partial-credit / `pass_ratio_mean` checkpoint
- 但 **不是** 更强的 one-turn repair base

当前正式结论是：

- `step900`
  - 仍是更好的 one-turn repair 主 base
- `step1300`
  - 更适合作为 near-miss / partial-credit probe checkpoint
  - 不适合优先拿来做第一版 repair-conditioned SFT 数据锚点

原因很简单：

- `step900` 在 `valid_big500` 的 `WA/RE + pass_ratio >= 0.6 + reuse-first-pass` 切片上：
  - `code_only`: `54 -> 64 / 500`
  - `short_diagnosis_code`: `54 -> 66 / 500`
  - `conditional_repair_success = 13.70% / 16.44%`
- `step1300` 在相同受控协议下：
  - `code_only`: `61 -> 65 / 500`
  - `short_diagnosis_code`: `61 -> 65 / 500`
  - `conditional_repair_success = 6.06% / 6.06%`

因此当前倾向是：

- `step900` 更 repair-ready
- `step1300` 更 high-partial

但这个结论在进入大规模数据生成前，还需要通过一次 **最新 `patched sandbox + client RR` 口径下的 same-protocol final check** 再确认。

### 2.1 Anchor freeze gate

冻结 student anchor 前，必须满足：

1. `step900` 和 `step1300` 都已经在 **相同协议** 下完成：
   - `reuse-first-pass`
   - `patched sandbox + client RR`
   - 相同 `repair_prompt_mode`
   - 相同 trigger slice
2. `step900` 在该协议下仍然是更强的 repair base
3. 结论不依赖旧的 pre-patch source run

如果这 3 条里有任意一条不成立，则：

- 暂缓冻结 `step900`
- 不进入 Stage D 之后的大规模 teacher 生成

也就是说：

- **Stage A/B/C 可以先搭**
- **Stage D/E/F 的大规模产出，必须等 anchor freeze gate 通过**

### 2.2 Provisional outputs must not use final names

在 freeze gate 通过之前，任何 probe / pre-gate 产物都必须放在：

- provisional namespace
- 或 scratch namespace

例如：

- `phase_2_ GRPO/sft_repair_data/provisional_anchor_check/`
- `phase_2_ GRPO/sft_repair_data/provisional_repair_conditioned/`

不允许在 gate 通过之前，就把产物写成看起来像最终正式数据的路径，例如：

- `v3_step900_repair_conditioned/...`

这些 `step900_repair_conditioned` final 命名，只保留给：

- gate 通过之后
- 正式冻结 anchor 后
- 可被后续 Stage D/E/F 复用的 canonical artifacts

---

## 3. Data Source Boundary

### 3.1 Allowed source splits

优先使用：

- `codecontests_train_wo_valid_big`

只有在样本量明显不足时，才补：

- `codecontests_train`

但这里有一个额外限制：

- `step900` 的 curriculum skeleton 只天然覆盖已经进入该 curriculum state 的训练宇宙
- 如果补充的 `codecontests_train` 行没有对应的 `B_near_miss / C_hard_partial` 标签，
  就不能直接进入 `Core / Expansion / Selective C`

因此 v1 的默认规则是：

- **只用 `codecontests_train_wo_valid_big`**
- `codecontests_train` 补样本先视为 **disabled by default**

只有在先实现下面这个独立步骤后，才允许把 `train` 补进来：

- `non_curriculum_bucket_derivation_v1`

该步骤必须为非-curriculum 题显式生成：

- `derived_repair_bucket`
- `derivation_reason`
- `derivation_source`

在这一步落地前，不允许把没有 `step900` curriculum 标签的 `train` 样本直接混进 v1 主 candidate queue。

### 3.2 Forbidden source splits

不能进入 SFT 主数据的题：

- `valid_big500`
- `delta69`
- `valid117`
- `canary_v1`
- `test`

### 3.3 Hygiene exclusions

无论来自哪个 train split，都必须排除：

- `problem_quarantine_v3.json` 中：
  - `hard_blacklist`
  - `caution`
  - `unresolved`
- 与任何 eval slice 重叠的题
- 已知 prompt / tests 不稳定题
- source response 已截断的样本

这里“已知不稳定题”不能只写成原则，必须物化成显式文件。

### 3.4 Instability blacklist must be materialized

第一版必须单独产出一份 instability blacklist，例如：

- `phase_2_ GRPO/sft_repair_data/v3_step900_repair_conditioned/instability_blacklist_v1.json`

来源应明确来自：

- fixed-response rejudge 的 same-response mismatch 题
- judge-only 重判反复波动题
- prompt/test 已知不稳定题

建议字段：

- `problem_id`
- `reason`
- `source_artifact`
- `evidence_count`
- `added_at`

并要求：

- `Stage A` 显式读入这份 blacklist
- 不允许靠人工口头排除
- 后续 agent 必须复用同一份 blacklist，而不是各自判断

原则：

- **训练题只能来自训练宇宙**
- **任何 eval 题、quarantine 题都不能混进来**

---

## 4. Candidate Queue: How To Cut It

这里最重要的结论是：

- **不能只靠 `step900` 的 `B/C` 标签切**
- 也不能只靠 `pass_ratio` 切

正确切法是：

`step900 curriculum bucket`
`∩`
`step900 canonical first-pass behavior`
`∩`
`teacher verified success potential`

### 4.1 First layer: curriculum skeleton

先用 `step900` 的 curriculum 状态把训练题打上结构标签：

- `B_near_miss`
- `C_hard_partial`

不要直接把全量 `C` 收进来。

`C` 只作为小比例扩张池。

### 4.2 Second layer: canonical first-pass evidence

对候选题，必须跑一遍 **canonical step900 first pass**，得到：

- `response`
- `accepted`
- `pass_ratio`
- `pass_ratio_all`
- `error_type`
- grounded verifier feedback

这个 first-pass sweep 是必需的，因为：

- `B/C` 是 curriculum bucket
- 不是当前 checkpoint 的真实 repair 状态

也就是说：

- `B` 不等于“一定适合 repair”
- `C` 也不等于“一定不适合 repair”

必须用 canonical first-pass 再切一次。

### 4.3 Third layer: repair-oriented strata

第一版 candidate queue 切成 3 层：

#### A. Core queue

主干高质量样本。

条件：

- `curriculum_bucket == B_near_miss`
- `error_type in {wrong_answer, runtime_error}`
- `pass_ratio >= 0.6`
- first pass 未 AC

作用：

- 保证高成功率
- 先把“最后一公里修复”学稳

#### B. Expansion queue

向更低 through-rate 扩 repair horizon。

条件：

- `curriculum_bucket == B_near_miss`
- `error_type in {wrong_answer, runtime_error}`
- `0.2 <= pass_ratio < 0.6`
- first pass 未 AC

作用：

- 教模型修不只是 near-AC 的题
- 但仍然保持在“已经进入正确算法邻域”的范围内

#### C. Selective C queue

只小比例引入 hard partial。

条件：

- `curriculum_bucket == C_hard_partial`
- 仅限当前明确保留的 retained C provenance
  - 包含两类：
    - `step600` 冷启动时人工 review 通过的 bootstrap `keep_C`
    - `step600 -> step900` 训练过程中真实进入 `C_hard_partial` 后，再按当前信号保留的 retained C
- `error_type in {wrong_answer, runtime_error}`
- `pass_ratio >= 0.2` 或 `RE + pass_ratio > 0`
- first pass 未 AC

作用：

- 轻量扩张到 harder repair cases
- 避免 SFT 只会修最容易的 near-miss
- 保留 bootstrap C 与 current-training C 两种来源
- 但不要把前者自动当成更高质量；两者首先是 provenance 区分

### 4.4 Explicitly excluded from v1

第一版不收：

- `0-pass`
- `syntax_error`
- `timeout`
- 全量 `C`
- `D_dead_hard`

原因：

- `0-pass` 太容易把任务变成“重写整题”，不再是 repair-conditioned
- `syntax_error` 可以单独做低成本清洗，但不是当前主收益来源
- `timeout` 在现有 one-turn evidence 上没有稳定收益
- 全量 `C/D` 太容易让数据分布变脏

### 4.5 Queue proportions vs final dataset proportions

这里要明确区分两件事：

- **request-time mix**
- **final verified dataset mix**

`60/30/10` 不能只定义在 teacher request 阶段。

因为：

- `Core`
- `Expansion`
- `Selective C`

三层的 teacher verified success rate 一定不同。

如果只控制 request 比例，而 Stage E 之后只保留 verified AC，
最终 parquet 很可能会自动塌缩成：

- 远高于 `60%` 的 `Core`
- 远低于 `30/10%` 的 `Expansion / Selective C`

所以真正应该冻结的是：

- **final verified dataset quota**

而不是只冻结：

- request-time quota

---

## 5. Recommended Mixture

第一版 request queue 建议先这样配比：

- `60%`：Core queue
  - `B_near_miss`
  - `pass_ratio >= 0.6`
- `30%`：Expansion queue
  - `B_near_miss`
  - `0.2 <= pass_ratio < 0.6`
- `10%`：Selective C queue
  - reviewed / retained `C`

这个配比背后的逻辑是：

- `>= 0.6` 保证质量
- `0.2~0.6` 负责扩张 repair horizon
- `selected C` 负责给模型一点 harder repair exposure

如果第一版 teacher verified success 太少，可按下面顺序扩张：

1. 先扩大 `B 0.2~0.6`
2. 再扩大 `selected C`
3. 最后才考虑 `syntax_error`

不要反过来。

### 5.1 Final verified dataset quota

主训练集的目标配比应定义在 **Stage E 之后**。

建议目标：

- `Core`: `50% ~ 65%`
- `Expansion`: `25% ~ 35%`
- `Selective C`: `10% ~ 15%`

如果 Stage E 后的 verified AC 自然分布偏离过大，应当：

1. 对 `Expansion` 先做 request 侧 oversampling
2. 对 `Selective C` 再做 request 侧 oversampling
3. 如果仍不足，再单独提高这两层的 `teacher best-of-n`

而不是直接接受最终数据自动塌成纯 `Core`。

换句话说：

- `60/30/10` 是 **初始 request 配比**
- `final parquet` 需要单独做 **post-QC quota control**

---

## 6. End-to-End Data Flow

完整流程建议固定成 6 个阶段。

### Stage A. Build candidate pool

输入：

- `step900` curriculum state
- train raw / manifest
- bootstrap reviewed-C assets + current retained-C assets
- quarantine v3
- eval overlap blacklist
- instability blacklist

输出：

- `candidate_pool_v1_step900_repair_cond.jsonl`

每条至少包含：

- `dataset`
- `problem_id`
- `curriculum_bucket`
- `source_split`
- `review_status`
- `prompt`
- `prompt_sha256`

### Stage B. Build canonical student references

对 candidate pool 跑一遍 canonical `step900` first pass。

要求：

- 使用 patched sandbox + client RR
- 单独输出目录
- 不覆盖历史结果
- 一旦生成完成，即视为 **immutable canonical artifact**
- Stage C/D/E/F 只能复用，不允许重跑覆盖同一路径

这里最关键的硬约束是：

- teacher 看到的 student 错代码，必须是 **可证明完整、不截断** 的 source

### Stage B.1 Full-source capture mechanism must be explicit

这里不能只写“希望有完整代码”，必须把采集机制写死。

Stage B 必须走 **dedicated student-reference capture path**，并满足下面二选一：

1. 单独的 no-cap / effectively-no-cap student-reference run
2. 单独落盘 raw completion artifact，再从 raw completion artifact 提取 `full_extracted_code`

不允许的做法：

- 直接从普通 eval `per_problem["response"]` 回推 `full_extracted_code`
- 直接复用可能受 `max_response_chars` 影响的 capped `response`

换句话说：

- `full_extracted_code` 的来源必须是 **full-response capture**
- 不是普通 eval 的 summary / capped response

建议把 provenance 写成枚举值，例如：

- `raw_completion_full_capture`
- `student_reference_uncapped_run`

而明确禁止：

- `per_problem_response_capped`

因此 Stage B 不能只落：

- `response`
- `details`

还必须额外落：

- `full_extracted_code`
- `source_truncated`
- `source_char_cap`
- `source_provenance`

并要求：

- 只有 `source_truncated == false`
- 且 `full_extracted_code` 非空

的样本，才允许进入 Stage C。

如果无法证明 source 完整，则：

- 直接 drop
- 不进入 teacher generation

这里的“无法证明”包括：

- 只能拿到 capped `response`
- 无法确认 `max_response_chars` 是否命中
- 无法从 raw completion artifact 恢复完整代码

输出：

- `student_references_step900_repair_cond_v1.jsonl`

每条至少包含：

- `problem_id`
- `response`
- `full_extracted_code`
- `source_truncated`
- `source_char_cap`
- `source_provenance`
- `accepted`
- `pass_ratio`
- `pass_ratio_all`
- `error_type`
- `details`
- `per_case_results` 摘要

### Stage C. Derive teacher request queue

把 Stage B 的结果再切成真正的 teacher request queue。

保留条件：

- first pass 未 AC
- `error_type in {wrong_answer, runtime_error}`
- 满足 `Core / Expansion / Selective C` 其中之一
- `source_truncated == false`
- `full_extracted_code` 非空

输出：

- `teacher_generation_requests_step900_repair_cond_v1.jsonl`

### Stage D. Teacher generation

teacher 输入：

- 原始题面
- student 错代码
- grounded verifier feedback

teacher 输出：

- `BUG_SUMMARY`
- `FIX_PLAN`
- `<code>...</code>`

默认 teacher 模板：

- `short_diagnosis+code`

可选对照：

- `code-only`

但 `code-only` 只建议保留少量控制组，不要当第一版主数据来源。

此外，建议预留一个很小的 teacher variance buffer：

- `Core`: 默认 `best_of_n = 1`
- `Expansion`: 可选 `best_of_n = 2`
- `Selective C`: 可选 `best_of_n = 2`

目的不是追求大规模 search，而是避免：

- `Expansion / Selective C`

因为单次 teacher 方差被过度淘汰。

输出：

- `teacher_responses_step900_repair_cond_v1.jsonl`

### Stage E. Verifier-based QC

对 teacher 输出再次 verifier 回判。

这里不能只做单次 `accepted=true` 判定。

因为项目已经确认：

- 即使在 `patched sandbox + client RR` 下
- judge-only accepted drift 仍然不是 0

因此 v1 主数据的 kept 样本，必须走 **双次稳定性确认**。

### Stage E.1 Stability confirmation policy

对于所有候选 kept 样本，至少执行：

1. `QC pass 1`
   - patched sandbox + client RR
   - 记录 `accepted / pass_ratio / error_type`
2. `QC pass 2`
   - 对同一份 teacher repaired code 再独立重判一次
   - 协议与 `QC pass 1` 保持一致

v1 主数据的硬条件建议是：

- `accepted = true` on pass 1
- `accepted = true` on pass 2

如果两次不一致，则：

- 不进入主训练集
- 可以进入 `weak / supplemental audit set`

对下列更高风险样本，应优先严格处理：

- timeout-adjacent accepts
- 只在 very small margin 上转正的 borderline accepts
- 命中过 instability blacklist 邻域的题

如果后续想进一步放宽，可以升级为：

- majority-of-3

但 v1 至少要有双次确认，不能只靠单次 QC。

输出：

- `teacher_qc_results_step900_repair_cond_v1.jsonl`

### Stage F. Build final parquet

从通过 QC 的 teacher 样本生成最终：

- `repair_conditioned_train_step900_v1.parquet`
- `repair_conditioned_val_step900_v1.parquet`
- `repair_conditioned_test_step900_v1.parquet`

这里必须明确：

- final parquet 的 strata 配比以 **Stage E 结果** 为准
- 需要执行一次 post-QC quota materialization
- 不允许默认沿用 Stage D request 比例

---

## 7. Teacher Output Policy

### 7.1 Should `short_diagnosis` come from the teacher?

**Yes.**

而且应当和 repaired code 来自同一条 teacher 响应。

不要做：

- student 自己先写 diagnosis
- 再让 teacher 只写 code

也不要做：

- diagnosis 由另一个模型补写

更稳的做法是：

- teacher 一次输出完整目标：
  - `BUG_SUMMARY`
  - `FIX_PLAN`
  - repaired code

### 7.2 Why teacher-generated diagnosis is better

因为你要训练的是：

- 看 feedback 后的修复能力

所以最自然的 supervision 就是：

- teacher 对 feedback 的短解释
- teacher 的修复方向
- teacher 的最终代码

这样任务是闭环的。

---

## 8. Acceptance Rules For Final SFT Samples

最终进入主训练集的样本，必须同时满足：

1. teacher 输出可解析
   - 能抽到 `BUG_SUMMARY`
   - 能抽到 `FIX_PLAN`
   - 能抽到 `<code>...</code>`

2. diagnosis 形式合规
   - `BUG_SUMMARY` 最多 1 句
   - `FIX_PLAN` 最多 1 句

3. repaired code 通过 verifier
   - v1 主数据硬条件：
     - `accepted = true` on QC pass 1
     - `accepted = true` on QC pass 2

4. 无 source truncation
   - student 错代码不能是被截断的 source

5. 无 quarantine / eval overlap
   - 再次复核一次 problem id

6. diagnosis 不明显与 feedback 冲突
   - 可人工 spot-check
   - 也可加轻规则过滤

第一版主数据**不建议**保留“只是 `pass_ratio` 提升但没 AC”的 teacher 样本。

原因：

- 它会把监督目标变得模糊
- 容易让 SFT 学到不稳定的“部分修对”

如果以后要扩大，可把它们作为：

- auxiliary weak set

但不要放进 v1 主数据。

---

## 9. Final Training Schema

建议同时保留两层表示：

### 9.1 Audit JSONL schema

用于可追溯和 QC。

字段建议：

- `dataset`
- `problem_id`
- `source_split`
- `curriculum_bucket`
- `repair_queue`
- `first_pass_pass_ratio`
- `first_pass_error_type`
- `first_pass_code`
- `verifier_feedback`
- `teacher_model`
- `teacher_bug_summary`
- `teacher_fix_plan`
- `teacher_repaired_code`
- `teacher_verified_accepted`
- `teacher_verified_pass_ratio`
- `prompt_sha256`
- `student_checkpoint`
- `teacher_prompt_mode`

### 9.2 Final SFT parquet schema

用于训练。

字段建议：

- `messages`
  - user:
    - 题面
    - 错代码
    - verifier feedback
  - assistant:
    - `BUG_SUMMARY`
    - `FIX_PLAN`
    - `<code>...</code>`
- `problem_id`
- `curriculum_bucket`
- `repair_queue`
- `first_pass_pass_ratio`
- `first_pass_error_type`
- `teacher_verified_accepted`

---

## 10. Split Strategy

split 必须按 `problem_id` 做，不能按 response 行随机切。

默认建议：

- `80%` train
- `10%` val
- `10%` test

并做分层抽样，保持以下维度大致平衡：

- `Core / Expansion / Selective C`
- `wrong_answer / runtime_error`
- `B / C`

val/test 的作用不是对外汇报 solve，而是：

- 看训练是否真的学会 repair-conditioned format
- 看不同 strata 上是否有明显 overfit

### 10.1 Minimum-per-stratum floor before finalizing split

即使采用 `80/10/10`，对于：

- `Selective C`

这样的稀疏层，val/test 仍然可能太薄。

因此 split 不能只按比例切完就结束，还必须满足最小 strata floor。

建议最小值：

- `Core`
  - val 至少 `25`
  - test 至少 `25`
- `Expansion`
  - val 至少 `15`
  - test 至少 `15`
- `Selective C`
  - val 至少 `10`
  - test 至少 `10`

如果切分后任一层达不到 floor：

1. 先增加该层 upstream request / kept 样本
2. 再重新 split
3. 如果仍达不到，则明确把该层标记为：
   - `supplemental only`
   - 不作为 v1 strata-level hard conclusion 的主要依据

也就是说：

- **split 比例只是起点**
- **最小 strata floor 才是最终约束**

---

## 11. Suggested Data Volume

第一版不建议一口气做太大。

### v1 pilot target

candidate pool 目标：

- `Core queue`: `1200`
- `Expansion queue`: `800`
- `Selective C`: `300`

总 teacher requests：

- `2300`

这是一个兼顾：

- 数据覆盖
- teacher 成本
- QC 可控性

的合理起点。

### v1 success criterion

如果最终 verified success 主数据能达到：

- `600 ~ 1200` 条

我认为已经足够支撑第一版 repair-conditioned SFT 试验。

如果 verified success 太少，再按顺序扩张：

1. 先加 `Expansion queue`
2. 再加 `Selective C`
3. 最后才考虑更难 strata

### 11.1 Request counts vs expected final kept counts

第一版应同时写两套数字：

- request target
- expected kept target

建议初始目标：

- request：
  - `Core`: `1200`
  - `Expansion`: `800`
  - `Selective C`: `300`
- expected kept：
  - `Core`: `350 ~ 600`
  - `Expansion`: `150 ~ 300`
  - `Selective C`: `50 ~ 120`

如果 `Expansion / Selective C` 的 kept 数过低，
优先增加这两层 request，而不是继续加 `Core`。

---

## 12. Concrete File Layout

建议新建：

- `phase_2_ GRPO/sft_repair_data/v3_step900_repair_conditioned/`

但要注意：

- 这个目录只给 **post-gate final artifacts**
- pre-gate probe 产物必须写到 provisional 目录

里面固定这些产物：

- `spec_v3_step900_repair_conditioned.json`
- `instability_blacklist_v1.json`
- `candidate_pool_v3_step900_repair_conditioned.jsonl`
- `student_references_step900_v3.jsonl`
- `teacher_generation_requests_step900_v3.jsonl`
- `teacher_responses_step900_v3.jsonl`
- `teacher_qc_results_step900_v3.jsonl`
- `repair_conditioned_split_manifest_v3.json`
- `repair_conditioned_final_quota_report_v3.json`
- `repair_conditioned_train_v3.parquet`
- `repair_conditioned_val_v3.parquet`
- `repair_conditioned_test_v3.parquet`
- `qc_report_v3.md`

---

## 13. Implementation Order

推荐执行顺序：

1. 先通过 `step900 vs step1300` same-protocol final check
2. 冻结 provisional anchor 为正式 student checkpoint
3. 物化 `instability_blacklist_v1.json`
4. 构造 `candidate_pool`
5. 跑 canonical student references，并冻结为 immutable artifact
6. 按 `Core / Expansion / Selective C` 切 teacher request queue
7. 跑 teacher generation
8. verifier 回判 + QC
9. 做 post-QC quota materialization
10. 落 final parquet
11. 再启动一轮小型 repair-conditioned SFT

不要反过来：

- 先拼 teacher 数据
- 再回头补 canonical student reference

那样很容易把 source 口径搞混。

---

## 14. What Not To Do In v1

第一版明确不做：

- 用 `step1300` 当主数据锚点
- 只收 `pass_ratio >= 0.6`
- 全量 `C`
- `timeout` 主导数据集
- 未验证成功的 teacher 样本进入主训练集
- 直接把这版 SFT 当成提升 single-shot `accepted@1` 的主要手段
- 在 Stage B 之后重跑覆盖 canonical student references

---

## 15. Final Recommendation

如果目标是：

- **最快做出一个完整、可讲清楚、技术闭环干净的简历项目**

那第一版最值得执行的是：

1. 先通过 anchor freeze gate，再冻结 `step900`
2. 用冻结后的 student checkpoint 切一版 repair-conditioned candidate queue
3. 数据范围采用：
   - `60% B_high`
   - `30% B_mid`
   - `10% selected_C`
4. teacher 主格式使用：
   - `short_diagnosis+code`
5. 只保留 verified AC 样本进入主数据
6. 但最终 parquet 要做 post-QC quota control，不能默认自然塌成 `Core`
7. 先做一轮小型 repair-conditioned SFT
8. 评测主打：
   - `accepted after 1 repair`
   - `conditional repair success`

也就是说：

- `>= 0.6` 应继续作为 **repair eval 的甜点区**
- 但 SFT 数据收集应当**适度扩大到 `0.2~0.6` 和 selected `C`**
- 这样模型学到的才不只是“最后一点 polishing”

这版方案的核心不是“让模型变成 solve winner”，而是：

**把当前已经验证过的 verifier-guided repair 能力，变成一个可训练、可复现、可写进简历的能力闭环。**
