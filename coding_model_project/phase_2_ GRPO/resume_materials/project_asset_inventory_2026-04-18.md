# 项目资产总盘点（2026-04-18）

## 0. 用途

这份 inventory 的目标不是挑“最适合简历的少数亮点”，而是先把当前仓库里与本项目相关的**文档、代码、评测结果、数据治理资产、SFT/teacher/QC 资产**系统盘清楚，方便后续：

1. 写简历与项目页
2. 让其他 agent 接手继续做实验
3. 回溯某条旧实验线到底做过什么
4. 判断哪些资产属于当前权威，哪些属于历史线索

后续从资产盘点跳到“可写进简历的 claim 素材”时，直接看：

- [resume_claim_evidence_map_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md)

这里优先覆盖：

- `phase_2_ GRPO/output` 下的结果资产
- curriculum RL 训练主线
- repair-SFT 全链路
  - 早期几百步 teacher / anchor / step400 repair-SFT
  - `step900` repair-conditioned 与 `shortdiag pure`
  - `step1300` repair-conditioned 与 `shortdiag pure`
- quarantine / teacher QC / reject audit / testcase audit / curriculum review
- 支撑这些实验的关键代码入口

---

## 1. 阅读顺序与权威层级

### 1.1 当前权威文档锚点

这些文档是后续引用时应优先相信的“当前口径”。

- [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/experiment_handoff.md)
  - 当前总接手文档
  - 负责说明：
    - 当前 strongest held-out deployed base
    - repair-SFT v1 当前 best ckpt
    - `v2` 是否值得推进、该怎么推进

- [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_eval_current_contract_2026-04-18.md)
  - 当前 repair 评测总合同
  - 负责说明：
    - Protocol A / Protocol B 的意义
    - `reuse_step900` / `reuse_step1300` 的角色
    - 哪些结果已经完整，哪些还只是次级 probe

- [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_protocolA_results_2026-04-18.md)
  - 固定输入 repair 结果总表

- [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_protocolB_results_2026-04-18.md)
  - 自举 first-pass repair 结果总表

### 1.2 仍重要但更偏“主线设计/历史解释”的文档

- [README.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/README.md)
- [algorithm_decision_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/algorithm_decision_guide.md)
- [formal_reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/formal_reward_design.md)
- [curriculum_rl_pilot_v8_explainer.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/curriculum_rl_pilot_v8_explainer.md)
- [current_asset_and_data_flow_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/current_asset_and_data_flow_guide.md)
- [shared_verifier_infra_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/shared_verifier_infra_guide.md)
- [full_code_path_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/full_code_path_guide.md)

### 1.3 历史但仍值得保留的计划/复盘

- [step400_repair_sft_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step400_repair_sft_plan.md)
- [step400_repair_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/v2b_step400/step400_repair_plan.md)
- [step400_sft_v1_postmortem.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/v2b_step400/step400_sft_v1_postmortem.md)
- [step900_repair_conditioned_sft_data_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/step900_repair_conditioned_sft_data_plan.md)
- [step1300_repair_conditioned_sft_v2_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/step1300_repair_conditioned_sft_v2_plan.md)

---

## 2. `phase_2_ GRPO` 文档资产全景

### 2.1 Curriculum RL 主线文档

- [README.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/README.md)
  - GRPO 阶段总览

- [algorithm_decision_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/algorithm_decision_guide.md)
  - 为什么选当前 RL 主线与对照口径

- [formal_reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/formal_reward_design.md)
  - 正式 reward 设计

- [curriculum_rl_pilot_v8_explainer.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/curriculum_rl_pilot_v8_explainer.md)
  - curriculum RL 的设计背景、演化与 bucket 逻辑

- [curriculum_manifest_v2_review_instructions.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/curriculum_manifest_v2_review_instructions.md)
  - step600 review 阶段的人工审核说明

- [checkpoint_selection_and_valid_big_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/checkpoint_selection_and_valid_big_plan.md)
  - checkpoint 选择与 valid_big 策略

- [eval_validation_strategy.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/eval_validation_strategy.md)
  - 评测吞吐与分层验证策略

- [a1_formal_run_metrics_summary.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/a1_formal_run_metrics_summary.md)
  - A1 正式 RL 运行指标汇总

### 2.2 Repair 评测与协议文档

- [repair_phase4_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_phase4_design.md)
  - Phase 4 的总设计草案

- [phase4_step1_repair_eval_implementation_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/phase4_step1_repair_eval_implementation_plan.md)
  - one-turn repair 第一步的实现计划

- [repair_eval_protocol_matrix.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_eval_protocol_matrix.md)
  - Protocol A / B 的矩阵定义

- [repair_eval_protocolA_step900_vs_sft30_60.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_eval_protocolA_step900_vs_sft30_60.md)
  - 旧 repair-SFT 线的 Protocol A 对照

- [repair_eval_protocolB_step30_60.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_eval_protocolB_step30_60.md)
  - 旧 repair-SFT 线的 Protocol B 对照

- [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_eval_current_contract_2026-04-18.md)
- [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_protocolA_results_2026-04-18.md)
- [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_protocolB_results_2026-04-18.md)
  - 当前 repair 口径与结果的正式引用页

### 2.3 Prompt / testcase 规则 / repair analysis 文档

- [repair_prompt_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_prompt_design.md)
  - `code_only` / `short_diagnosis_code` 的 prompt 设计

- [testcase_selection_rule_audit_v1.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/testcase_selection_rule_audit_v1.md)
- [testcase_selection_rule_recommendation_v1.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/testcase_selection_rule_recommendation_v1.json)
- [testcase_selection_instance_audit_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/testcase_selection_instance_audit_v1.jsonl)
  - testcase 选取规则的审计与实例级结果

- [step100_valid_big_failure_mode_analysis.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/step100_valid_big_failure_mode_analysis.md)
- [step200_failure_mode_analysis.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/step200_failure_mode_analysis.md)
- [step200_centered_repair_priority_list.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/step200_centered_repair_priority_list.md)
- [valid_big_transition_analysis_and_sft_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/valid_big_transition_analysis_and_sft_plan.md)
- [train_repair_method_and_next_steps.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/train_repair_method_and_next_steps.md)
- [train_repair_retrieval.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/train_repair_retrieval.md)
  - 属于 repair 思考、失败模式与数据构建的分析页

### 2.4 Sandbox / verifier / infra 文档

- [shared_verifier_infra_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/shared_verifier_infra_guide.md)
- [full_code_path_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/full_code_path_guide.md)
- [sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sandbox_judge_repair.md)
- [sandbox_repair_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sandbox_repair_plan.md)
- [sandbox_concurrency_standard.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sandbox_concurrency_standard.md)
- [new_machine_teacher_qc_runbook_2026-04-15.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/new_machine_teacher_qc_runbook_2026-04-15.md)

---

## 3. `phase_2_ GRPO/output` 结果资产总览

## 3.1 目录结构惯例

当前 `output/` 下至少有三类结果树：

- [output/outputs](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs)
  - 当前主要结果树

- [output/未命名](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/未命名)
  - 与 `outputs/` 存在部分镜像/历史复制

- [output/rl_codecontests_validbig500_repair](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/rl_codecontests_validbig500_repair) 等直接挂在 `output/` 下的旧 repair 导出
  - 代表某些阶段的中间或旧式导出结构

常见关键文件语义：

- `summary.json`
  - 主评测汇总

- `repair_summary.json`
  - repair 专用汇总，包含 attempt/success/cost

- `run_info.json`
  - 运行上下文与配置唯一真值来源

- `per_problem/*.jsonl`
  - 逐题结果

- `qa_logs/qa_summary.json`
  - 抽样日志与 QA 聚合

- repair 目录中的 `first_pass_summary.json`、`first_pass_metrics.json`
  - 首轮生成与修复轮的对照资产

## 3.2 `phase0_*` 基础验证 / smoke / same-protocol 对照

位于 [output/outputs](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs) 下：

- [phase0_fullval_20260331](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_fullval_20260331)
- [phase0_fullval_baseline_valid_big_lb24_multi2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_fullval_baseline_valid_big_lb24_multi2)
- [phase0_fullval_step100_same_protocol](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_fullval_step100_same_protocol)
- [phase0_fullval_step100_valid_big_lb24_multi2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_fullval_step100_valid_big_lb24_multi2)
- [phase0_fullval_step120_lb24_multi2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_fullval_step120_lb24_multi2)
- [phase0_fullval_step160_lb24_multi2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_fullval_step160_lb24_multi2)
- [phase0_fullval_step180_lb24_multi2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_fullval_step180_lb24_multi2)
- [phase0_fullval_step200_lb24_multi2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_fullval_step200_lb24_multi2)
- [phase0_fullval_step200_valid_big_lb24_multi2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_fullval_step200_valid_big_lb24_multi2)
- [phase0_schema_smoke_20260331](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_schema_smoke_20260331)
- [phase0_smoke_remote](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_smoke_remote)
- [phase0_smoke_remote_v2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_smoke_remote_v2)
- [phase0_smoke_reboot](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase0_smoke_reboot)

用途：

- 早期全量基线
- same-protocol 对照
- schema / smoke / reboot 后的环境自检

## 3.3 RL eval 主线结果树

### `valid117`

- [rl_codecontests_valid117](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_valid117)

这条线主要包含：

- 早期 RL step 的验证集评估
- `phase2_repair_sft_step400_r3a2_clean_v1_step10/20` 这类旧 SFT 线在 `valid117` 上的跟踪

### `valid_big500`

- [rl_codecontests_validbig500](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_validbig500)

这是当前最重要的 RL checkpoint 大盘之一，包含：

- 早期阶段：
  - `step320`
  - `step360`
  - `step400`
  - `step460`
  - `step520`
  - `step580`

- curriculum 主线：
  - `step600`
  - `step620`
  - `step640`
  - `step700`
  - `step800`
  - `step900`
  - `step1000`
  - `step1100`
  - `step1200`
  - `step1300`

- patched / rerun / fixed-response 稳定性相关：
  - `raw_patched_rr818x_vastai3_retry2`
  - `raw_patched_rr828x_vastai3`
  - `*_rerun1`

- 旧 SFT 线跟踪：
  - `phase2_repair_sft_step400_r3a2_clean_v1_step10_codecontests_validbig500`
  - `phase2_repair_sft_step400_r3a2_clean_v1_step20_codecontests_validbig500`

### `delta69`

- [rl_codecontests_delta69](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_delta69)

这条线主要用于：

- curriculum 主线 checkpoint 的中小规模高灵敏对照
- raw / clean overlay / quarantine 影响分析
- `abfocus` 等新训练段对比

### canary / watchlist / retention

- [rl_codecontests_canary_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_canary_v1)

包含：

- `mid_val_codecontests_32`
- `retention_canary_codecontests`
- `structural_hard_watchlist`
- `timeout_canary_codecontests`

### testset regression

- [rl_testset_regression](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_testset_regression)

用途：

- 持出测试集上的回归检查
- 常见同时覆盖：
  - `codecontests_test`
  - `humaneval`
  - `mbpp`

## 3.4 Repair eval 结果树

### `validbig500` repair 主线

- [rl_codecontests_validbig500_repair](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_validbig500_repair)
- [output/rl_codecontests_validbig500_repair](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/rl_codecontests_validbig500_repair)

主要包含：

- `raw`
- `reuse_firstpass`
- `reuse_firstpass_full`
- `reuse_firstpass_aligncheck`

用途：

- one-turn repair 最初评测线
- first-pass 重用与对齐检查

### `delta69` repair

- [rl_codecontests_delta69_repair](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_delta69_repair)

用途：

- 小规模高信号 repair 提升验证
- 典型 headline 结果来源之一

### `codecontests_test` repair

- [rl_codecontests_test_repair](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_test_repair)
- [rl_codecontests_test_repair_protocolA](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_test_repair_protocolA)
- [rl_codecontests_test_repair_protocolA_step1300sftv1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_test_repair_protocolA_step1300sftv1)
- [rl_codecontests_test_repair_step1300sftv1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_test_repair_step1300sftv1)

用途：

- held-out test 上的 Protocol A / Protocol B repair 对照
- `step1300_rl` 与 `step1300_sft_v1_step60` 的关键比较资产

### `validbig500` Protocol A / B 拆分评测

- [rl_codecontests_validbig500_repair_protocolA_full](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolA_full)
- [rl_codecontests_validbig500_repair_protocolB](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB)
- [rl_codecontests_validbig500_repair_protocolA_step1300sftv1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolA_step1300sftv1)
- [rl_codecontests_validbig500_repair_protocolB_step1300sftv1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB_step1300sftv1)
- [rl_codecontests_validbig500_repair_protocolA_step1300src](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolA_step1300src)

用途：

- 当前 repair-SFT v1 与 baseline 的核心对照结果树

### 其他 repair 结果树

- [rl_codecontests_validbig500_repair_ablation](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_validbig500_repair_ablation)
  - prompt / mode /设置的小消融

- [rl_codecontests_validbig500_protocolB](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_validbig500_protocolB)
  - 非 repair-SFT 当前线但同属 Protocol B 相关结果

- [rl_codecontests_validbig500_fixed_rejudge](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/rl_codecontests_validbig500_fixed_rejudge)
  - fixed-response rejudge 稳定性结果

## 3.5 Student reference / teacher / SFT eval 结果树

### student reference eval

- [repair_conditioned_student_reference_eval](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/repair_conditioned_student_reference_eval)
  - 包含：
    - `step900_repair_conditioned_student_ref_v1`
    - `step900_candidate_v2_backend_rr_student_ref`
    - `step1300_repair_cond_v2_student_ref`

- [repair_v2a_student_reference_eval](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/repair_v2a_student_reference_eval)
- [repair_v2b_step400_student_reference_eval](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/repair_v2b_step400_student_reference_eval)

用途：

- 冻结 student response
- 物化 teacher 生成前的 canonical reference
- 对齐 step400 / step900 / step1300 repair-conditioned 数据线

### SFT eval suite

- [phase2_sft_eval_suite_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/output/outputs/phase2_sft_eval_suite_v1)

当前能确认的 step 包括：

- `phase2_sft_pre_sft_step200_fullsuite_v1_retry1`
- `phase2_sft_step8_fullsuite_v1`
- `phase2_sft_step12_fullsuite_v1`
- `phase2_sft_step16_fullsuite_v1`
- `phase2_sft_step20_fullsuite_v1_retry1`

每个 step 下再按任务拆成：

- `humaneval_mini`
- `mbpp_reg_mini`
- `repair_val`
- `retention_canary_codecontests`
- `structural_hard_watchlist`
- `timeout_canary_codecontests`

这是一条非常重要的“旧 SFT 全套评估”资产线。

---

## 4. `sft_repair_data` 资产总览

## 4.1 早期 SFT / teacher / anchor 主线

- [sft_repair_data/v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/v1)
  - 早期 teacher candidates、teacher QC、train/val parquet、token report

- [sft_repair_data/final_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/final_v1)
  - 旧 phase2 混合 SFT 最终数据

- [sft_repair_data/phase2_sft_runbook.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/phase2_sft_runbook.md)
- [sft_repair_data/phase2_sft_runbook.zh.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/phase2_sft_runbook.zh.md)
- [sft_repair_data/repair_sft_training_defaults.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/repair_sft_training_defaults.md)
  - 旧 SFT 默认训练口径

### anchor 资产

- [sft_repair_data/anchor_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/anchor_v1)
- [sft_repair_data/anchor_v2a](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/anchor_v2a)
- [anchor_data_strategy.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/anchor_data_strategy.md)
- [anchor_v1_build_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/anchor_v1_build_plan.md)
- [claude_anchor_generation_prompt.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/claude_anchor_generation_prompt.md)

## 4.2 step400 / v2b_step400 资产

- [sft_repair_data/v2b_step400](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/v2b_step400)

这条线包含：

- seed batches
- teacher generation requests
- Claude teacher outputs
- anchor batches / anchor regen
- student reference subset
- spec / draft spec
- QC 结果
- postmortem

是当前仓库里“几百步时 teacher SFT / repair SFT”最完整的一条历史资产树。

## 4.3 step900 repair-conditioned 与 shortdiag pure 资产

### step900 candidate / student reference / teacher generation

- [sft_repair_data/provisional_repair_conditioned/step900_candidate_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/provisional_repair_conditioned/step900_candidate_v1)
  - `step900` repair-conditioned 的早期候选冻结

- [teacher_generation_shards/step900_candidate_v2_backend_rr](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr)
  - canonical teacher 请求与 prompt shard

- [teacher_generation_outputs/step900_candidate_v2_backend_rr](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr)
  - raw teacher outputs、preflight、primary QC、rejudge、final keep

- [reject_audit_with_cases/step900_candidate_v2_backend_rr](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/reject_audit_with_cases/step900_candidate_v2_backend_rr)
- [reject_audit_outputs_actionable_with_cases/step900_candidate_v2_backend_rr](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/reject_audit_outputs_actionable_with_cases/step900_candidate_v2_backend_rr)
  - reject audit 与可救回决策链

### step900 最终 keep 与 shortdiag pure

- [sft_repair_data/step900_request_unique_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step900_request_unique_v1)
- [sft_repair_data/step900_shortdiag_pure_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step900_shortdiag_pure_v1)
- [step900_shortdiag_pure_sft_training_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/step900_shortdiag_pure_sft_training_guide.md)

## 4.4 step1300 repair-conditioned 与 shortdiag pure 资产

### step1300 repair-conditioned 候选与 freeze

- [sft_repair_data/step1300_repair_conditioned_v2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair_conditioned_v2)
  - 冻结输入、candidate pool、curriculum state 的新主线

### step1300 主工作树

- [sft_repair_data/step1300_repair](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair)

这条树当前包含：

- candidate pool 与 summary
- full teacher requests
- full teacher shards
- full teacher shards `n=1`
- full teacher raw outputs
- QC 结果
- redo QC 结果
- regen queues / regen round2
- testcase audit / near miss 审计
- final keep v1 / v2

### step1300 pure shortdiag SFT 数据

- [sft_repair_data/step1300_shortdiag_pure_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_shortdiag_pure_v1)
- [step1300_shortdiag_pure_sft_training_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/step1300_shortdiag_pure_sft_training_guide.md)

---

## 5. 数据治理资产总览

## 5.1 Quarantine

### 核心代码

- [problem_quarantine.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/problem_quarantine.py)
- [build_problem_quarantine_v2.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_problem_quarantine_v2.py)
- [build_problem_quarantine_v3.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_problem_quarantine_v3.py)
- [prepare_problem_quarantine_review_assets.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/prepare_problem_quarantine_review_assets.py)
- [prepare_test_full_review_assets.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/prepare_test_full_review_assets.py)

### 主定义与审计结果

- [problem_quarantine_v1.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v1.json)
- [problem_quarantine_v2.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v2.json)
- [problem_quarantine_v3.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v3.json)
- [problem_contract_audit_report_v1.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/problem_contract_audit_report_v1.md)
- [problem_contract_screen_rows_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/problem_contract_screen_rows_v1.jsonl)
- [problem_contract_screen_summary_v1.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/problem_contract_screen_summary_v1.json)
- [problem_manual_review_candidates_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/problem_manual_review_candidates_v1.jsonl)
- [problem_quarantine_candidates_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/problem_quarantine_candidates_v2.jsonl)

### review 资产树

- [data/quarantine_audit/review_assets_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/review_assets_v1)
  - canonical quarantine review 树

- [data/quarantine_audit/test_review_assets_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/test_review_assets_v1)
  - test split 全量复审树

## 5.2 Curriculum review 与 ledger

### 代码入口

- [export_curriculum_review_candidates.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/export_curriculum_review_candidates.py)
- [apply_curriculum_review_decisions.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/apply_curriculum_review_decisions.py)
- [materialize_step640_review_ledgers.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/materialize_step640_review_ledgers.py)
- [filter_curriculum_manifest_by_quarantine.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/filter_curriculum_manifest_by_quarantine.py)
- [step580_curriculum_builder.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_builder.py)
- [step580_curriculum_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_dataset.py)
- [step580_curriculum_sampler.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_sampler.py)

### 资产树

- [curriculum_assets/step600_v2_review_local](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/curriculum_assets/step600_v2_review_local)
- [curriculum_assets/step600_v3_review_local](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/curriculum_assets/step600_v3_review_local)
- [curriculum_assets/step600_v3_qv2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/curriculum_assets/step600_v3_qv2)
- [curriculum_assets/step600_v3_qv3](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/curriculum_assets/step600_v3_qv3)
- [review_assets/step640_v1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/review_assets/step640_v1)
- [remote_logs/curriculum_assets](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/remote_logs/curriculum_assets)

## 5.3 Teacher QC

### 方案与脚本

- [step900_teacher_generation_qc_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/step900_teacher_generation_qc_plan.md)
- [new_machine_teacher_qc_runbook_2026-04-15.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/new_machine_teacher_qc_runbook_2026-04-15.md)
- [step900_teacher_prompt_and_schema_v2_backend_rr.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/step900_teacher_prompt_and_schema_v2_backend_rr.md)
- [step900_teacher_shards_v2_backend_rr.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/step900_teacher_shards_v2_backend_rr.md)
- [sft_repair_data/scripts/build_teacher_generation_requests.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/build_teacher_generation_requests.py)
- [sft_repair_data/scripts/build_teacher_generation_shards.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/build_teacher_generation_shards.py)
- [sft_repair_data/scripts/qc_teacher_candidates.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/qc_teacher_candidates.py)
- [sft_repair_data/scripts/materialize_step900_teacher_keep_and_reject_audit.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/materialize_step900_teacher_keep_and_reject_audit.py)
- [sft_repair_data/scripts/merge_step900_actionable_repair_keep_set.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/merge_step900_actionable_repair_keep_set.py)

### 资产树

- [teacher_generation_shards/step900_candidate_v2_backend_rr](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr)
- [teacher_generation_outputs/step900_candidate_v2_backend_rr](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr)
- [sft_repair_data/v1/teacher_qc_results_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/v1/teacher_qc_results_v1.jsonl)
- [sft_repair_data/v1/repair_sft_qc_report_v1.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/v1/repair_sft_qc_report_v1.md)

## 5.4 Reject audit

- [reject_audit/step900_candidate_v2_backend_rr](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/reject_audit/step900_candidate_v2_backend_rr)
- [reject_audit_outputs/step900_candidate_v2_backend_rr](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/reject_audit_outputs/step900_candidate_v2_backend_rr)
- [reject_audit_with_cases/step900_candidate_v2_backend_rr](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/reject_audit_with_cases/step900_candidate_v2_backend_rr)
- [reject_audit_outputs_actionable_with_cases/step900_candidate_v2_backend_rr](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/reject_audit_outputs_actionable_with_cases/step900_candidate_v2_backend_rr)

## 5.5 Testcase audit

- [repair_analysis/testcase_selection_rule_audit_v1.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/testcase_selection_rule_audit_v1.md)
- [repair_analysis/testcase_selection_rule_recommendation_v1.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/testcase_selection_rule_recommendation_v1.json)
- [repair_analysis/testcase_selection_instance_audit_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis/testcase_selection_instance_audit_v1.jsonl)
- [phase_1_ SFT/match_sft_with_testcases.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_1_ SFT/match_sft_with_testcases.py)
- [sft_repair_data/scripts/audit_step900_shortdiag_token_lengths.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/audit_step900_shortdiag_token_lengths.py)
- [sft_repair_data/step1300_repair/near_miss_testcase_audit](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/near_miss_testcase_audit)
- [sft_repair_data/step1300_repair/near_miss_testcase_audit_regen_round1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/near_miss_testcase_audit_regen_round1)
- [sft_repair_data/step1300_repair/near_miss_testcase_audit_regen_round2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/near_miss_testcase_audit_regen_round2)
- [sft_repair_data/step1300_repair/remaining_high_pass_audit_candidates](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/remaining_high_pass_audit_candidates)
- [sft_repair_data/step1300_repair/regen_queues](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/regen_queues)
- [sft_repair_data/step1300_repair/regen_round2](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/regen_round2)

---

## 6. 关键代码资产

## 6.1 评测与 repair 主入口

- [phase0_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py)
- [phase4_repair_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py)
- [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)
- [eval_config.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/eval_config.py)
- [prompting.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/prompting.py)
- [verifier/__init__.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/__init__.py)
- [verifier/shared.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py)

## 6.2 数据治理与构建

- [data_governance.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/data_governance.py)
- [build_grpo_parquet.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_grpo_parquet.py)
- [build_valid_big_split.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_valid_big_split.py)
- [build_eval_slice_from_problem_ids.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_eval_slice_from_problem_ids.py)
- [grpo_batch_reward.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/grpo_batch_reward.py)
- [compute_eval_clean_overlay.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/compute_eval_clean_overlay.py)
- [light_screen_problem_contracts.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/light_screen_problem_contracts.py)
- [validate_sft_data.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/validate_sft_data.py)

## 6.3 SandboxFusion 关键代码

这些文件位于独立仓 [SandboxFusion](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion)。

- [sandbox/runners/base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py)
- [sandbox/utils/execution.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/execution.py)
- [sandbox/server/server.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/server/server.py)
- [sandbox/server/sandbox_api.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/server/sandbox_api.py)
- [sandbox/server/online_judge_api.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/server/online_judge_api.py)
- [sandbox/utils/sandbox_client.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/sandbox_client.py)
- [sandbox/runners/isolation.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/isolation.py)
- [sandbox/utils/extraction.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/extraction.py)

## 6.4 Ops 脚本资产

- [ops/README.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/ops/README.md)
- `sandbox_backend_*`
- `setup_*sandbox*`
- `run_phase0_*`
- `run_phase2_sft_*`
- `run_phase4_repair_eval_*`
- `run_checkpoint_eval_queue.sh`
- `summarize_checkpoint_eval.py`

这些主要位于：

- [phase_2_ GRPO/ops](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/ops)

---

## 7. 当前应如何使用这份 inventory

### 7.1 如果目标是继续做实验

优先看：

- [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/experiment_handoff.md)
- [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_eval_current_contract_2026-04-18.md)
- [shared_verifier_infra_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/shared_verifier_infra_guide.md)

### 7.2 如果目标是回溯某条历史 SFT / teacher 线

优先按这条顺序找：

1. `sft_repair_data/v1` 与 `final_v1`
2. `sft_repair_data/v2b_step400`
3. `step900_*` 文档 + `teacher_generation_*`
4. `sft_repair_data/step1300_repair`

### 7.3 如果目标是写简历或项目页

这份 inventory 先解决“有哪些资产、它们在什么位置、它们属于哪一阶段”的问题。

真正面向简历的精炼写法，应再结合：

- [resume_project_writing_plan_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/resume_materials/resume_project_writing_plan_2026-04-18.md)

---

## 8. 当前最重要的提醒

1. 不是所有资产都属于当前权威口径。
2. 当前 repair 结论必须优先按 `Protocol A / Protocol B` 区分。
3. `output/outputs` 是主结果树，但 `output/未命名` 和 `output/` 下部分旧直挂目录也保留了历史结果，不能简单忽略。
4. `step400`、`step900`、`step1300` 三条 repair-SFT 线都真实存在，不能混成一条时间线讲。
5. quarantine / teacher QC / reject audit / testcase audit 已经形成完整治理链，这部分是当前项目的重要资产，而不是附属材料。
