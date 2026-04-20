# 简历项目编写计划（2026-04-18）

## 0. 目标

这份计划的目标不是继续补实验，而是先基于当前仓库里**已经落地、可核验、可追溯**的资产，产出一版适合写进简历和面试材料的项目包装方案。

配套资产索引见：

- [project_asset_inventory_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/resume_materials/project_asset_inventory_2026-04-18.md)
- [resume_claim_evidence_map_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md)

这里的核心原则只有两条：

1. 优先展示已经被代码、文档、结果三方共同支撑的优势
2. 对当前还不够漂亮的部分，不回避，但要把它们转化成“实验严谨性、系统排障能力、研究判断力”的证据

---

## 1. 推荐的项目主叙事

当前最适合对外讲的，不是“我做了一堆零散 RL / repair / sandbox 实验”，而是下面这条完整主线：

> 围绕算法代码生成任务，搭建并迭代了一条 `curriculum-aware RL + shared verifier + verifier-guided repair` 的完整实验链路；不仅做了训练和评测，还把数据治理、沙箱稳定性、固定输入/自举 repair 协议、以及后续 repair-SFT 数据构建串成了一套可复现流程。

这条主叙事的优点：

- 同时覆盖 `ML / RL / LLM post-training`
- 同时覆盖 `evaluation infra / sandbox / debugging`
- 同时覆盖 `data governance / curriculum / audit`
- 不依赖某一个单独分数就能成立

---

## 2. 简历里最该强调的 4 个优势

### 2.1 不是只调模型，而是把整条实验链做完整

建议突出：

- 你不只是跑 RL，而是把：
  - 数据切分
  - 课程采样
  - shared verifier
  - sandbox 评测
  - checkpoint 汇总
  - one-turn repair
  - repair-SFT 数据准备
  串成了统一链路

最能支撑这一点的资产：

- [experiment_handoff.md](../experiment_handoff.md)
- [shared_verifier_infra_guide.md](../shared_verifier_infra_guide.md)
- [full_code_path_guide.md](../full_code_path_guide.md)
- [phase0_eval.py](../../src/phase0_eval.py)
- [phase4_repair_eval.py](../../src/phase4_repair_eval.py)
- [verifier/shared.py](../../src/verifier/shared.py)

### 2.2 你做了严谨的协议控制，而不是只看单次分数

建议突出：

- 明确区分 `Protocol A` 和 `Protocol B`
- 通过 `reuse-first-pass` 消除 first-pass 漂移
- 做过 fixed-response rejudge，识别 judge instability
- 对旧/新 lineage 的同名 checkpoint 做了口径隔离

最能支撑这一点的资产：

- [repair_eval_current_contract_2026-04-18.md](../repair_eval_current_contract_2026-04-18.md)
- [repair_protocolA_results_2026-04-18.md](../repair_protocolA_results_2026-04-18.md)
- [repair_protocolB_results_2026-04-18.md](../repair_protocolB_results_2026-04-18.md)

### 2.3 你不仅会训练，还会做数据治理和审计

建议突出：

- 版本化 quarantine
- curriculum bucket 与 quarantine 联动
- review ledger / teacher QC / reject audit
- 把人工审查和自动流水线分开

最能支撑这一点的资产：

- [problem_quarantine_v3_guide.md](../problem_quarantine_v3_guide.md)
- [problem_quarantine.py](../../src/problem_quarantine.py)
- [build_problem_quarantine_v2.py](../../src/build_problem_quarantine_v2.py)
- [build_problem_quarantine_v3.py](../../src/build_problem_quarantine_v3.py)
- [filter_curriculum_manifest_by_quarantine.py](../../src/filter_curriculum_manifest_by_quarantine.py)
- [step1300_repair](../sft_repair_data/step1300_repair)

### 2.4 你做过“评测系统本身”的定位和修复

建议突出：

- 识别并修复 SandboxFusion 在并发下的输出截断 / nondeterminism 主问题
- 验证 patched sandbox + direct-backend RR
- 让 reward、eval、repair 三条线共享同一套可信 judge 契约

最能支撑这一点的资产：

- [sandbox_judge_repair.md](../sandbox_judge_repair.md)
- [sandbox_repair_plan.md](../sandbox_repair_plan.md)
- [base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py)
- [execution.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/execution.py)

---

## 3. 最适合写进简历 headline 的硬结果

这里优先列“适合上简历 bullet 的结果”，不是“所有已经做过的结果”。

### 3.1 RL 主线结果

优先引用：

- `CodeContests valid_big500` 上的 checkpoint 对比
- `step1000` 作为 exact-solve winner

当前最安全的表述：

- 在 `valid_big500` 上完成多 checkpoint 对比，最佳 checkpoint 达到 `69/500` solved，`accepted@1 = 13.8%`

支撑资产：

- [validbig500_step1300_vs_baselines.md](../eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)

### 3.2 Repair 主线结果

优先引用：

- `delta69` 上 one-turn repair 的明显提升

当前最安全的表述：

- 在 `CodeContests delta69` repair 评测上，将 `accepted@1` 从 `65.2%` 提升到 `72.46%`

支撑资产：

- [repair_summary.json](../output/outputs/rl_codecontests_delta69_repair/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_delta69_repair_raw/repair_summary.json)

### 3.3 Repair-SFT v1 的正确定位

这部分不建议写成 headline，但适合写成 supporting line。

当前最安全的表述：

- 基于 `step1300` 构建并验证了 `short_diagnosis_code` repair-SFT v1，开发集 `valid_big500` 上出现明确 repair gain，但 held-out `codecontests_test` 仍未超过 `step1300_rl` baseline，因此据此制定了更平衡的 v2 数据配方

支撑资产：

- [repair_protocolA_results_2026-04-18.md](../repair_protocolA_results_2026-04-18.md)
- [repair_protocolB_results_2026-04-18.md](../repair_protocolB_results_2026-04-18.md)

---

## 4. 不建议作为简历 headline 的内容

这些内容不是不能讲，而是不适合放在最前面。

### 4.1 不要把 repair-SFT v1 讲成“已经成功替代主模型”

原因：

- `step1300_sft_v1_step60` 在 `valid_big500` 上是正收益
- 但在 `codecontests_test` 上没有超过 `step1300_rl`

正确讲法：

- v1 是一次成功的 repair-specialization probe
- 它证明了方向可行，但也揭示了 v2 必须改 recipe

### 4.2 不要把所有旧 lineage 的 `step40/50/60` 混成一个故事

原因：

- 旧 `step900` repair-SFT 线和当前 `step1300`-base v1 线都存在 `step40/60`
- 如果简历里不区分，会显得实验口径混乱

正确讲法：

- 对外一律只保留：
  - `legacy repair-SFT line`
  - `step1300 repair-SFT v1`

### 4.3 不要把尚未闭环的 v2 写成已完成

正确讲法：

- 当前已经完成的是：
  - v1 数据资产
  - teacher / QC / audit 流程
  - v2 设计与候选队列
- 但 v2 本体仍是下一阶段工作

---

## 5. 推荐的简历项目包装方式

建议至少准备 3 个层次的材料，而不是只写 2 条 bullet。

### 5.1 简历短版

用途：

- 2 到 3 条 bullet
- 强调结果 + 你负责的核心系统

适合保留的关键词：

- curriculum-aware RL
- shared verifier
- multi-sandbox evaluation
- repair-guided post-training
- data governance / quarantine

### 5.2 项目说明中版

用途：

- 求职平台项目描述
- 个人主页 / Notion / PDF 项目页

结构建议：

1. 背景与任务
2. 训练与评测系统
3. 数据治理与审计
4. repair 实验与结论
5. 当前 v2 方向

### 5.3 面试展开长版

用途：

- 面试时根据岗位偏好切换展开重点

建议准备 3 种展开口径：

1. `ML / Applied Research`
2. `LLM Systems / Infra`
3. `Data quality / evaluation reliability`

---

## 6. 推荐的对外措辞方向

### 6.1 如果岗位更偏 Research / Applied Scientist

重点突出：

- curriculum RL
- repair protocol design
- fixed-input vs self-first-pass evaluation
- data slicing and checkpoint analysis

少讲：

- 具体 nginx / shell 运维细节

### 6.2 如果岗位更偏 ML Systems / Infra

重点突出：

- shared verifier
- sandbox orchestration
- direct-backend RR
- deterministic evaluation / judge debugging
- training-eval-repair pipeline integration

少讲：

- 某个单独 checkpoint 的微小分数涨跌

### 6.3 如果岗位更偏通用 LLM Engineer

重点突出：

- 端到端项目 ownership
- 从数据、训练、评测、修复到分析的完整闭环
- 真实 debug / protocol control / regression handling

---

## 7. 推荐引用资产地图

### 7.1 核心结论文档

- [experiment_handoff.md](../experiment_handoff.md)
- [repair_eval_current_contract_2026-04-18.md](../repair_eval_current_contract_2026-04-18.md)
- [repair_protocolA_results_2026-04-18.md](../repair_protocolA_results_2026-04-18.md)
- [repair_protocolB_results_2026-04-18.md](../repair_protocolB_results_2026-04-18.md)
- [validbig500_step1300_vs_baselines.md](../eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)

### 7.2 最值得引用的实现文件

- [phase0_eval.py](../../src/phase0_eval.py)
- [phase4_repair_eval.py](../../src/phase4_repair_eval.py)
- [repair_feedback.py](../../src/repair_feedback.py)
- [verifier/shared.py](../../src/verifier/shared.py)
- [step580_curriculum_sampler.py](../../src/step580_curriculum_sampler.py)
- [problem_quarantine.py](../../src/problem_quarantine.py)
- [base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py)
- [execution.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/execution.py)

### 7.3 最值得引用的结果资产

- [summary.json](../../outputs/phase0_fullval_20260331/summary.json)
- [repair_summary.json](../output/outputs/rl_codecontests_delta69_repair/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_delta69_repair_raw/repair_summary.json)
- [step900_shortdiag_pure_sft_dataset_v1.summary.json](../sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.summary.json)
- [step1300_shortdiag_pure_sft_training_guide.md](../step1300_shortdiag_pure_sft_training_guide.md)

---

## 8. 下一步材料产出顺序

推荐顺序：

1. 先写一版 `claim -> evidence` 对照表
2. 再写一版中文项目主描述
3. 再压成中文简历 bullet
4. 再翻成英文简历 bullet
5. 最后准备一版面试展开稿

如果时间有限，至少先完成前 3 项。

---

## 9. 当前对外叙事的安全边界

一句话版本：

- 可以把这个项目写成“我搭建并迭代了一条代码生成 RL + repair 的完整实验与评测系统，并通过数据治理、协议控制和 sandbox 修复提升了实验可信度与 repair 开发集效果”
- 但不要写成“我已经通过 repair-SFT 在 held-out test 上稳定超过主 RL baseline”

这是当前最能放大优势、同时不越界的写法。
