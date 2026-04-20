# Step1300 Repair-Conditioned SFT v2 Plan

## 0. Decision

当前下一轮扩大版 repair-SFT 的默认锚点应切到：

- `student anchor = step1300`
- `task type = repair-conditioned SFT`
- `primary format = short_diagnosis_code`
- `data source = codecontests_train_wo_valid_big`
- `judge protocol = patched sandbox + direct-backend client RR + 8081..8088`

这轮不再沿用 `step900 v1` 的“小而精 near-miss 修补器”思路，而是改成：

- 更强 first-pass 底座
- 更大规模 `B/C` repair horizon
- 更明确的 strict / supplemental 双层数据设计

一句话总结：

**v2 的目标不是继续证明 `step900` 会修最后一公里，而是围绕更强的 `step1300` 部署底座，构造一版更大规模、更贴近当前训练分布的 repair-conditioned SFT。**

---

## 1. Why Switch To `step1300`

### 1.1 当前已知事实

基于当前正式结果：

- [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)
- [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)

可以稳定下来的结论是：

- `step40`
  - 在 `valid_big500` 的 repair 能力上最好
  - 说明小规模 repair-SFT 这条线是有效的
- `step900`
  - 是更好的“小 repair base”起点
  - 但已经不是当前最强的 deployed model
- `step1300`
  - 是当前最强的 RL 主线模型
  - 在 `Protocol B / codecontests_test` 上给出最好的端到端修复结果
  - 同时在当前课程训练后期拥有更丰富的 `B/C` 题目分布

### 1.2 v2 的目标已经变化

`step900 v1` 的价值已经证明：

- repair-conditioned SFT 能带来真实提升
- `short_diagnosis_code` 是可行主线

但它的局限也很明确：

- 数据太小
- 过度偏向 `pass_ratio >= 0.6`
- `SelectiveC` 太小
- 没有把最强 deployed 能力拉到 `step1300` 之上

因此 v2 应该改成：

- 保留最强 first-pass 底座
- 在更大的当前训练分布上补 repair skill

这就是选择 `step1300` 的核心原因。

---

## 2. Goal

这轮 v2 的目标是：

1. 扩大 repair-conditioned SFT 数据规模
2. 从 `near-AC` 扩展到更广的 `B/C` repair horizon
3. 在不放松主集精度的前提下，利用更多 teacher 样本
4. 训练一个更贴近真实部署的 repair model

它优化的能力仍然是：

- 给定
  - 原始题面
  - student 错代码
  - grounded verifier feedback
- 输出
  - 极短诊断
  - 极短修复计划
  - 修正后的完整代码

也就是说：

- RL 主线：
  - 提升 single-shot `accepted@1`
- repair-SFT 支线：
  - 提升 verifier-guided repair

当前 v2 不追求：

- 再做 prompt family 大消融
- 把 `code_only` 重新并入主线
- 上多轮 repair / agent loop

主线仍然是：

- `short_diagnosis_code`

---

## 3. Success Criteria

### 3.1 数据成功标准

这轮先不把绝对配额写成 pre-freeze 承诺。

在 `Stage 0 freeze` 完成前，v2 只先冻结：

- 默认比例
- strict / supplemental 双层 keep 设计

默认目标写成两层：

- `strict main keep`
  - 高精主集
  - 目标：`800 ~ 1200` 条 request-unique rows
- `relaxed supplemental keep`
  - 放宽补充集
  - 目标：`300 ~ 900` 条 request-unique rows

也就是说：

- 主集仍然追求高精度
- 但总数据规模不再被主集硬上限卡死
- 若 supply 允许，这轮总训练语料可以做到：
  - `1200 ~ 2200` 条 request-unique rows

不过这些绝对数都必须在 `Stage 0 freeze summary` 和 `pilot QC summary` 出来后才正式解锁。

freeze 前只先固定默认比例：

- `CoreNearMiss`：`35% ~ 45%`
- `ExpansionB`：`25% ~ 35%`
- `CurrentCRecoverable`
  - `strict main keep` 默认 `10% ~ 20%`
  - 只有 pilot 证明稳定 yield 足够时，才上调到 `20% ~ 25%`
- `CurrentCHardProbe`
  - 默认不进入 strict main keep
  - 只进入 supplemental / audit-only

因此，这轮对外的预期也要写得更准确：

- v2 的核心目标是 `repair-skill shaping + 更强部署修复能力`
- 不是在 freeze 前就承诺一定会出现大幅 headline jump

### 3.2 模型成功标准

训练后至少应重新比较：

- `Protocol A`
  - `valid_big500`
  - `codecontests_test`
- `Protocol B`
  - `valid_big500`
  - `codecontests_test`

和以下底座做对照：

- `step900`
- `step1300`
- `step40`

这轮真正要回答的问题是：

**在更强 first-pass 底座上，扩大版 repair-SFT 是否能同时保住部署能力，并提升修复能力。**

---

## 4. Canonical Source Boundary

### 4.1 Allowed splits

默认只允许：

- `codecontests_train_wo_valid_big`

原因：

- 最容易保证不碰 eval 题
- 不碰 `valid_big500` / `codecontests_test`
- 不引入不透明的 train 全量补样本逻辑

### 4.2 Supplemental `train` 先禁用

这轮虽然是 v2，但仍然不建议把 `codecontests_train` 直接补进主线。

原因不是数据不够，而是：

- `step1300` 当前 curriculum skeleton 天然覆盖的是当前训练宇宙
- 对没有 bucket 标签的 `train` 题，仍然缺一个可复现的 `bucket-derivation path`

因此 v2 继续采用：

- `train disabled by default`

如果后续真的要补 `train`，必须单独实现：

- `non_curriculum_bucket_derivation_v2`

在那之前：

- 不允许把没有 `step1300` bucket 标签的题混进主 keep set

### 4.3 Forbidden splits

以下题目不得进入 v2 数据：

- `valid_big500`
- `delta69`
- `valid117`
- `canary_v1`
- `codecontests_test`
- 任何 regression eval 题

---

## 5. Hygiene And Immutable Filters

所有 candidate 都必须统一吃这几层过滤：

1. quarantine 过滤
2. eval overlap 过滤
3. instability blacklist 过滤
4. source integrity 过滤

### 5.1 Quarantine

必须显式读取：

- [problem_quarantine_v3.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v3.json)

一律排除：

- `hard_blacklist`
- `caution`
- `unresolved`

### 5.2 Eval overlap

必须排除与以下切片重叠的题：

- `valid_big500`
- `delta69`
- `valid117`
- `canary_v1`
- `codecontests_test`
- 当前 phase2 所有固定回归集

### 5.3 Instability blacklist

必须物化一份 v2 blacklist，例如：

- `phase_2_ GRPO/sft_repair_data/step1300_repair_conditioned_v2/instability_blacklist_v1.json`

来源至少包括：

- fixed-response rejudge mismatch
- judge-only rejudge mismatch
- prompt/test 已知不稳定题

建议字段：

- `problem_id`
- `reason`
- `source_artifact`
- `evidence_count`
- `added_at`

### 5.4 Source integrity

任何 student reference 进入 teacher 阶段之前，必须满足：

- `full_extracted_code` 存在
- `source_truncated = false`
- `source_provenance` 明确

并明确禁止：

- 从 capped `response` 字段反推 `full_extracted_code`

---

## 6. Step1300 Strata Design

这轮不再用 `B_high + B_mid + selected_C` 这组偏 step900-v1 的名字，而是改成更贴近当前目标的四层结构。

### 6.1 Stratum A: `CoreNearMiss`

定义：

- 来自 `step1300` 的 `B_near_miss`
- 以 `wrong_answer / runtime_error` 为主
- `pass_ratio >= 0.6`

目标：

- 保住 high-ROI last-mile 修复能力

### 6.2 Stratum B: `ExpansionB`

定义：

- 来自 `step1300` 的 `B_near_miss`
- `0.2 <= pass_ratio < 0.6`
- 允许 `wrong_answer / runtime_error / 少量 timeout`

目标：

- 从“最后一公里”扩到“中等 partial”

### 6.3 Stratum C: `CurrentCRecoverable`

定义：

- 来自 `step1300` 的 `C_hard_partial`
- 优先当前训练过程中真实进入 `C` 的题

优先筛法：

- `pass_ratio > 0`
- 非纯噪声 timeout
- 存在可执行 signal
- 最近窗口内不是极端不稳定题

目标：

- 让模型学会修“比 B 更难、但仍可救”的题

这是 v2 和 `step900 v1` 最大的区别之一。

### 6.4 Stratum D: `CurrentCHardProbe`

定义：

- 仍来自当前 `C_hard_partial`
- 但更难、更边界、更低通过率

目的：

- 只保留很小比例做 probe
- 默认不作为 strict main keep 的主干

建议：

- 默认只进入 `supplemental / audit-only`
- 只有 pilot 证明：
  - stable-AC yield 不是极低
  - suspect rate 也没有明显失控
  才允许小比例进入 relaxed supplemental keep

### 6.5 C provenance 解释

这轮直接采用用户最新决策：

- `reviewed-C` 不再作为 v2 设计中的单独保留对象
- v2 的 C 主线只围绕 `step1300` 当前训练过程中真实进入 `C_hard_partial` 的题展开

因此这轮的 C 语义固定成：

- `current-training C = 当前训练分布 provenance`

如果后续还需要回头审计更早期的 bootstrap C，那应作为单独旁路资产处理，而不是进入 v2 主设计。

---

## 7. Freeze-Gated Scale And Quotas

`step900 v1` 的主问题之一，是 request-time 设计和 final keep 分布差得太远。

v2 必须直接按 **最终 kept rows** 去设计配额，但这次配额解锁分 3 步：

1. `freeze inventory`
2. `pilot QC`
3. `full expansion`

### 7.1 Final keep targets

在 `freeze inventory` 完成前，只先保留区间目标：

- `strict main keep`
  - acceptable：`800+`
  - ideal：`1000 ~ 1200`
- `relaxed supplemental keep`
  - acceptable：`300+`
  - ideal：`400 ~ 800`
- combined stretch：
  - `1500 ~ 2200`

这意味着：

- `800 ~ 1200` 不再是这轮唯一总目标
- 它只对应 strict main keep
- 扩大数据规模主要通过 supplemental keep 解决

### 7.2 Final kept quotas

这一步不再在 freeze 前写死绝对数。

改成：

- 先固定默认比例
- freeze 后再解锁绝对数

freeze 后必须先产出：

- `candidate_pool_step1300_repair_cond_v2.summary.json`
- `student_references_step1300_repair_cond_v2.summary.json`
- `pilot_qc_step1300_repair_cond_v2.summary.json`

只有在这 3 份 summary 都存在后，才允许把默认比例转换成绝对 kept 数。

默认比例：

- `CoreNearMiss`：`35% ~ 45%`
- `ExpansionB`：`25% ~ 35%`
- `CurrentCRecoverable`
  - pre-pilot 默认按 strict main keep 的 `10% ~ 20%`
  - 通过 pilot gate 后，才上调到 `20% ~ 25%`
- `CurrentCHardProbe`
  - 默认 `0` strict main keep
  - supplemental keep 再单独解锁

### 7.3 C 扩张 gate

`CurrentCRecoverable` 只有在 pilot 满足下面条件时，才允许扩大 share：

1. `accept_2_of_2` kept rate 没有显著塌缩
2. suspect testcase / contract issue rate 没有明显高于 `ExpansionB`
3. source integrity 依然稳定

如果 pilot 不满足这些条件，则：

- `CurrentCRecoverable` 只保持 `10% ~ 20%`
- `CurrentCHardProbe` 继续只做 supplemental / audit-only

### 7.4 Request-time oversampling

因为 harder strata 的 teacher success rate 更低，所以 request-time 必须过采样。

建议 full queue 的 request/generation-unit 配额：

- `CoreNearMiss`
  - request 占比 `25% ~ 35%`
  - `best_of_n = 1`
- `ExpansionB`
  - request 占比 `30% ~ 35%`
  - `best_of_n = 2`
- `CurrentCRecoverable`
  - request 占比 `25% ~ 35%`
  - `best_of_n = 2`
- `CurrentCHardProbe`
  - request 占比 `5% ~ 10%`
  - `best_of_n = 2`

这样最终 generation units 大致落在：

- `1800 ~ 3200`

但这里的 request 配额同样要区分：

- `pilot queue`
- `full queue`

也就是说：

- 不先做 pilot，就不直接把整批 C 全量放大

### 7.5 双层 keep 设计

为了兼顾“数据量更大”和当前 judge/QC 仍有噪声这两个现实，v2 明确改成：

- `strict main keep`
  - 训练主集
  - 高精度
- `relaxed supplemental keep`
  - 放宽补充集
  - 只要满足更弱但可追溯的条件，就允许进入

这比“把所有 QC 都放宽到一个池子里”更稳，因为：

- 主集精度不会被一起拉低
- 同时又能把更多 teacher 样本利用起来

---

## 8. Stage 0: Freeze Current Step1300 Assets

在真正开始构数据前，先冻结一轮 `step1300` 当前资产。

### 8.1 必须冻结的输入

至少包括：

- `step1300` curriculum state
- `step1300` 当前 focused manifest / seed manifests
- `problem_quarantine_v3.json`
- instability blacklist
- eval overlap blacklist
- canonical raw eval references

建议产物：

- `phase_2_ GRPO/sft_repair_data/step1300_repair_conditioned_v2/freeze_manifest_step1300_repair_cond_v2.json`

### 8.2 当前缺口

本地目前没有看到现成的：

- `curriculum_state_step_1300.json`

所以 Stage 0 的第一件事是：

- 从远端把 `step1300` curriculum state 与相关 manifests materialize 到本地 canonical 路径

在 freeze manifest 落盘前，不进入 Stage B。

### 8.3 Freeze 后必须先做 pilot

freeze 完成后，不直接全量 teacher generation。

先做一轮 `pilot queue`：

- 覆盖 `CoreNearMiss / ExpansionB / CurrentCRecoverable / CurrentCHardProbe`
- 规模建议：
  - `200 ~ 400` generation units

pilot 的作用是先回答：

- `step1300` 当前 B/C 库存到底有多少
- C 的真实稳定 kept rate 到底如何
- 哪些 strata 适合放进 strict main keep

只有 pilot summary 落盘后，才解锁 full queue。

---

## 9. Stage A: Build Canonical Step1300 Candidate Pool

复用并参数化当前脚本：

- [build_repair_conditioned_candidate_pool.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_repair_conditioned_candidate_pool.py)

但要把它从“step900 provisional”真正改成：

- `--anchor_step`
- `--curriculum_state`
- `--bucket_policy`
- `--instability_blacklist`

### 9.1 Stage A 输出

建议输出：

- `candidate_pool_step1300_repair_cond_v2.jsonl`
- `candidate_pool_step1300_repair_cond_v2.summary.json`

### 9.2 每条 candidate 至少要带

- `problem_id`
- `dataset`
- `anchor_step`
- `bucket_name`
- `seed_bucket`
- `dominant_error_type`
- `ema_group_pass_ratio`
- `ema_timeout_rate`
- `selection_mode`
- `source_manifest`
- `curriculum_membership`
- `c_source_tag`
- `c_provenance`

### 9.3 选题原则

- `B_near_miss`
  - 继续保留大头
- `C_hard_partial`
  - 只要符合 recoverable / probe 条件就显式进入
- reviewed-C 保留为 seed，不再是唯一合法 C 来源

---

## 10. Stage B: Materialize Student References

复用脚本：

- [build_repair_conditioned_student_references.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_repair_conditioned_student_references.py)

但这轮必须明确两条硬约束。

### 10.1 必须走 full-response capture

禁止从普通 eval `per_problem["response"]` 直接构 student reference。

必须使用：

- no-cap full response
- 或单独 raw completion artifact

并显式产出：

- `full_extracted_code`
- `source_truncated`
- `source_char_cap`
- `source_provenance`

### 10.2 Judge 协议写死

student reference eval 必须统一使用：

- patched sandbox
- direct-backend client RR
- `8081..8088`

不再接受：

- RR-over-LB
- 旧 pre-patch sandbox

### 10.3 Stage B 输出

- `student_references_step1300_repair_cond_v2.jsonl`
- `student_references_step1300_repair_cond_v2.summary.json`

---

## 11. Stage C: Build Teacher Requests

复用并扩展：

- [build_repair_conditioned_teacher_requests.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_repair_conditioned_teacher_requests.py)

这一步要从 `step900` 风格的 strata 逻辑，改成 `step1300 v2` 的四层结构。

### 11.1 request schema 必须保留 provenance

teacher request / shard 侧不能只保留最小 prompt 字段。

必须继续携带：

- `source_truncated`
- `source_char_cap`
- `source_provenance`
- `selection_mode`
- `bucket_name`
- `stratum`
- `c_source_tag`
- `c_provenance`
- `c_decision_reason`
- `anchor_step`

这样后面 teacher 输出、reject ledger、QC ledger 才能独立审计。

### 11.2 shard 设计

建议 shard 大小：

- `50 ~ 80 generation units`

因为 v2 总规模更大，shard 太小会给 Claude 分发带来额外摩擦。

---

## 12. Stage D: Teacher Generation

这轮 teacher generation 仍然采用：

- `short_diagnosis_code`

不把 `code_only` 重新并入主线。

### 12.1 Why

当前我们已经知道：

- `short_diagnosis_code` 是可用的
- v2 的主要瓶颈在规模和分布
- 不是 prompt 样式还没试够

### 12.2 best-of-n policy

建议：

- `CoreNearMiss`：`n=1`
- `ExpansionB`：`n=2`
- `CurrentCRecoverable`：`n=2`
- `CurrentCHardProbe`：`n=2`

不要全层统一 `n=1`，否则 harder strata 会被过度淘汰。

---

## 13. Stage E: QC And Stability Rejudge

这一步是 v2 的生命线，但这次不再只有一个“全-or-无”的 keep 门槛。

复用：

- [qc_repair_conditioned_teacher_candidates.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/qc_repair_conditioned_teacher_candidates.py)
- [assemble_repair_conditioned_teacher_candidates.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/assemble_repair_conditioned_teacher_candidates.py)

### 13.1 Primary QC

统一使用：

- patched sandbox
- direct-backend client RR
- same timeout contract

### 13.2 Strict main keep

`v1` 已经证明 judge-only 漂移不是 0。

所以 v2 的 `strict main keep` 必须采用：

- `accept_2_of_2`

也就是：

1. primary QC accepted
2. rejudge accepted

才允许进 strict main keep。

### 13.3 Relaxed supplemental keep

为了扩大数据规模，这轮新增：

- `relaxed supplemental keep`

允许进入 supplemental 的最小条件建议是：

1. `primary QC` 通过
2. `source_truncated = false`
3. 没有：
   - `suspect_testcase_or_judge_issue`
   - `suspect_prompt_or_contract_issue`
4. 不是明显高风险 timeout 噪声
5. 相对 student 有明确改进，或至少 teacher 结果为 AC

也就是说：

- 不再要求所有训练样本都必须是 `accept_2_of_2`
- 但也不会把“只过一次且明显可疑”的样本直接混入主集

### 13.4 Borderline accepts

对于：

- timeout-adjacent
- suspect testcase
- contract 不稳

这些行要：

- 进入 reject / audit ledger
- 不进入 strict main keep

若只是不满足 `accept_2_of_2`，但没有明显 suspect flag，则：

- 可以进入 `relaxed supplemental keep` 候选池
- 但不能自动进入 strict main keep

### 13.5 Recommended usage

默认训练配方建议分两条：

- `v2-main`
  - 只用 `strict main keep`
- `v2-main+supp`
  - `strict main keep + capped supplemental keep`

这样后面如果要比较“放宽 QC 到底值不值”，也能直接做受控对照。

### 13.6 Stage E 输出

- `teacher_candidates_step1300_repair_cond_v2.jsonl`
- `qc_primary_step1300_repair_cond_v2.jsonl`
- `qc_rejudge_step1300_repair_cond_v2.jsonl`
- `verified_keep_strict_step1300_repair_cond_v2.jsonl`
- `verified_keep_supplemental_step1300_repair_cond_v2.jsonl`
- `reject_ledger_step1300_repair_cond_v2.jsonl`

---

## 14. Stage F: Final Dataset Materialization

### 14.1 Final keep policy

主训练集默认只吃：

- request-unique
- `short_diagnosis_code`
- `accept_2_of_2`
- `source_truncated = false`

但 v2 这次新增一个补充版训练集：

- `request-unique`
- `short_diagnosis_code`
- 来源于 `verified_keep_supplemental`
- 且单独打标 provenance

也就是说，Stage F 最终至少会产出两套 dataset：

- strict-only
- strict+supplemental

### 14.2 Request-unique 规则

和 `step900 v1` 一样，最终训练集不应保留多条同 request 的 accepted attempt。

对同一 request 的多个 AC candidate：

- 只保留 1 条 canonical row

建议优先级：

1. rejudge 稳定
2. `short_diagnosis` 更简洁
3. 代码长度更合理
4. 输出结构更干净

对于 supplemental keep，也采用相同的 request-unique 规则，但：

- 不允许和 strict keep 争抢同一条 canonical row
- 如果同一 request 已经存在 strict row，则 supplemental row 直接丢弃

### 14.3 Final deliverables

建议最终输出：

- `step1300_shortdiag_repair_sft_dataset_v2_strict.jsonl`
- `step1300_shortdiag_repair_sft_dataset_v2_strict.parquet`
- `step1300_shortdiag_repair_sft_dataset_v2_strict.summary.json`
- `step1300_shortdiag_repair_sft_dataset_v2_main_plus_supp.jsonl`
- `step1300_shortdiag_repair_sft_dataset_v2_main_plus_supp.parquet`
- `step1300_shortdiag_repair_sft_dataset_v2_main_plus_supp.summary.json`

---

## 15. Training Handoff

v2 训练默认应直接训在：

- `step1300` HF merged model

不再采用：

- `step900` 起训，再看是否能迁移

### 15.1 Training recipe

这份文档不展开训练超参，但训练侧原则应继承当前经验：

- 保留 `short_diagnosis_code`
- 先做 `v2-main`
- 再做 `v2-main+supp` 作为规模增益对照
- 先做 canary，再做正式 run

### 15.2 Eval recipe after training

训练后至少要补：

- `Protocol A / valid_big500`
- `Protocol A / codecontests_test`
- `Protocol B / valid_big500`
- `Protocol B / codecontests_test`

对照：

- `step1300`
- 当前最佳 repair-SFT checkpoint
- `step900`
- `step40`

---

## 16. Required Script Changes

为了落地 v2，至少要做这些改造。

### 16.1 Reusable with parameterization

优先复用：

- [build_repair_conditioned_candidate_pool.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_repair_conditioned_candidate_pool.py)
- [build_repair_conditioned_student_references.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_repair_conditioned_student_references.py)
- [build_repair_conditioned_teacher_requests.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_repair_conditioned_teacher_requests.py)
- [assemble_repair_conditioned_teacher_candidates.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/assemble_repair_conditioned_teacher_candidates.py)
- [qc_repair_conditioned_teacher_candidates.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/qc_repair_conditioned_teacher_candidates.py)
- [prepare_repair_sft_parquet.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/prepare_repair_sft_parquet.py)

### 16.2 Must be extended

需要显式扩展的点：

1. `build_repair_conditioned_candidate_pool.py`
   - 支持 `step1300`
   - 支持四层 strata 的前置标签
2. `build_repair_conditioned_teacher_requests.py`
   - 支持 `CoreNearMiss / ExpansionB / CurrentCRecoverable / CurrentCHardProbe`
   - 支持 strict/supplemental 目标和 request-time oversampling
3. student reference materialization
   - 明确 full-response capture contract
4. final keep builder
   - 直接支持：
     - strict keep
     - supplemental keep
     - request-unique canonical selection

### 16.3 Optional helper scripts

如果不想把旧脚本改得太重，可以新增：

- `build_step1300_repair_conditioned_candidate_pool_v2.py`
- `build_step1300_teacher_requests_v2.py`
- `build_step1300_request_unique_sft_dataset_v2.py`

---

## 17. Execution Order

建议按下面顺序执行：

1. Stage 0
   - 冻结 `step1300` 当前 curriculum state / manifests / blacklist
2. pilot queue
   - 先估算真实 B/C supply 和 QC yield
3. Stage A
   - 构建 canonical candidate pool
4. Stage B
   - 产出 full-code student references
5. Stage C
   - 按 strict/supplemental 目标反推 teacher requests
6. Stage D
   - teacher generation
7. Stage E
   - primary QC + rejudge + strict/supp keep
8. Stage F
   - request-unique dataset + parquet
9. training
10. Protocol A/B eval

---

## 18. Go / No-Go Rules

只有同时满足下面条件，才进入 full teacher generation：

1. `step1300` curriculum state 已 freeze
2. instability blacklist 已 materialize
3. student references 已证明：
   - `source_truncated_count = 0`
4. pilot summary 已落盘
5. `CurrentCRecoverable` 的 pilot kept rate 不至于明显失控

如果这些条件不满足，就先停在 Stage 0 / pilot，不要直接全量 teacher。

---

## 19. Final Recommendation

这轮 v2 的执行基线可以直接定成：

- `anchor = step1300`
- `format = short_diagnosis_code`
- `source split = train_wo_valid_big`
- `judge = patched sandbox + direct-backend RR`
- `main expansion direction = 当前 B/C，尤其是 current-training C_hard_partial`
- `dataset design = strict main keep + relaxed supplemental keep`

与 `step900 v1` 相比，v2 的核心变化不是“再试一种 prompt”，而是：

- 更强 base
- 更大规模
- 更多 current-training C
- 先 freeze，再 pilot，再全量
- 不再把所有数据都绑死在 `accept_2_of_2` 一条线上

如果后续要我继续推进，最自然的下一步就是：

1. 先把 `step1300` freeze inputs 清单和 pilot schema 冻住
2. 再把现有 `step900` 脚本参数化成 `step1300 v2`
