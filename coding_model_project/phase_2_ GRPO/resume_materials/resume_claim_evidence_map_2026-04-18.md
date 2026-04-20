# 项目 Claim-Evidence Map（2026-04-18）

## 0. 用途

这份文档不是正式简历，也不是项目介绍成稿，而是一份尽量全覆盖的**项目素材与证据地图**。它的目标是：

1. 把这个项目里所有值得在简历、项目页、面试问答里提到的工作先系统列出来
2. 把每条 claim 背后的文档、代码、数据和评测结果挂清楚
3. 明确哪些结果文件才是后续应该长期引用的 canonical 结果锚点
4. 让后续写简历时可以从这里挑素材，而不是重新在仓库里找证据

配套资产总盘点见：

- [project_asset_inventory_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/project_asset_inventory_2026-04-18.md)
- [resume_project_writing_plan_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_project_writing_plan_2026-04-18.md)

---

## 1. 使用规则

### 1.1 Claim 强度标签

- `A 级`
  - 可以稳定写进简历或项目页
  - 有比较扎实的文档、代码、结果三方支撑
  - 升级门槛：同时具备 (1) 设计/决策文档；(2) 实际代码 / 数据 artifact；(3) 至少一份未退役的 canonical `summary.json` 或等价结果；(4) 结论经过 held-out 或独立复现覆盖

- `B 级`
  - 很适合面试展开，或者作为项目页的深挖点
  - 证据充分，但更偏过程价值、研究判断或系统能力
  - 升降级门槛：通常是 (1) held-out 证据还缺；或 (2) 主要是 recipe/工程判断，不直接挂硬数字；或 (3) 数据 / 结果规模较小，不适合做 headline

- `C 级`
  - 只能保守表述
  - 更适合说“探索过 / 建好流程 / 验证出边界”，不适合写成硬结果 headline
  - 适用情形：(1) 只有开发集证据；(2) 样本量小、噪声大；(3) 尚未独立复现过；(4) 仍处在原型或单次实验阶段

### 1.2 结果文件引用原则

`phase_2_ GRPO/output` 下结果比较杂，后续引用时建议遵守下面的规则：

- **规则 1：canonical 都在 `output/outputs`**
  - 所有最终要被简历、问答、handoff 引用的 `summary.json` / `repair_summary.json` 都在 [output/outputs](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs) 这棵树下
  - 示例：`output/outputs/rl_testset_regression/step1300_..._raw_rr808x_vastai5/summary.json` 就是 step1300 held-out raw 的权威结果

- **规则 2：`output/未命名` 只作镜像**
  - [output/未命名](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/%E6%9C%AA%E5%91%BD%E5%90%8D) 里的目录视作历史备份，不作为首选引用源
  - 遇到同名目录同时存在于 `outputs/` 和 `未命名/` 时，以前者为准

- **规则 3：新跑法优先于旧跑法**
  - 同一实验有「旧 raw 跑法」和「patched sandbox + direct-backend RR 跑法」两份结果时，后者优先
  - 具体识别：目录名带 `_rr808x_` / `_rr828x_` / `_patched_` 的是新跑法

- **对 repair 相关结论的细则**
  - 开发集（`valid_big500` / `delta69`）看 `Protocol A + reuse_step900` 和 `Protocol B` 的组合
  - held-out test 部署口径只看 `Protocol B`（端到端自修复），因为 Protocol A 是纯修复能力度量

- **对 general raw 能力结论的细则**
  - 优先看 `rl_testset_regression/` 下的目录，它是专门为 raw generalization 留的 canonical 树
  - 绝对不要用某条 repair run 的 first-pass 字段去替代 raw regression 结果，因为 repair run 的 first-pass 采样参数和 raw 跑可能不一致

### 1.3 当前权威口径文档

当前如果要统一外部表述，优先信这些文档：

- [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
- [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)
- [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)
- [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)

注意：这 4 份文件是「**冲突仲裁文档**」——当下面各 claim 引用的支撑文档与这 4 份出现口径冲突时，以这 4 份为准。各 claim 下面挂的 `Doc` / `Key Result Files` 是「**详细支撑**」，两类角色不应混用。

### 1.4 术语小字典（跨文件共享）

这份字典在 [resume_bullet_candidates_cn_2026-04-20.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_bullet_candidates_cn_2026-04-20.md) 里也出现一份相同内容，用来保证两个文件对同一术语的措辞完全一致。

- **Shared Verifier**
  - 评测脚本（`phase0_eval.py`）与 RL reward（`grpo_batch_reward.py`）共用的判题模块 `src/verifier/shared.py`
  - 对每题跑 **全部 testcase**（而不是 verl 内置 sandbox_fusion 的前 10 条），结果落到统一的 `accepted` / `pass_ratio_all` 字段上
  - 保证 eval 与 RL reward 在同一口径下判题，避免 reward hacking 与评测漂移

- **Protocol A（fixed-input repair）**
  - 所有被测 checkpoint 都在同一套冻结的错代码输入上做 one-turn repair
  - 度量的是「**纯修复能力**」——第一遍生成的差异被消掉了

- **Protocol B（end-to-end self-repair）**
  - 每个 checkpoint 先自己生成 first-pass，再对**自己**的错代码做 one-turn repair
  - 度量的是「**端到端部署口径**」——更接近真实部署，但 repair gain 会受 first-pass 分布影响

- **Strongest deployed base**
  - 在 held-out test 上「raw accepted@1 + Protocol B repair gain」综合最高的 checkpoint
  - 当前是 `step1300_rl`（raw 7.88% → after repair 10.91%）

- **两类 checkpoint winner**
  - `step1000` 是 **solve-count winner**（valid_big500 dev accepted@1 最高：13.8%）
  - `step1300` 是 **partial-credit winner**（pass_ratio_mean 最高：35.76%）
  - 两者不是一个模型；checkpoint 选择需要两个指标一起看

- **step 命名约定**
  - `step900 / step1000 / step1300` = **RL 主线** checkpoint（A1 curriculum GRPO 训练的结果）
  - `step40` = 旧 repair-SFT 线（`step900`-base）的一个 SFT step
  - `step1300_sft_v1_step60` = 当前主线的 repair-SFT v1 第 60 个 step，base 是 `step1300_rl`
  - 凡出现 `stepXXX_sft_v1_stepYY` 都是 repair-SFT 线；单独 `stepXXX` 一般是 RL 线

---

## 2. 值得长期引用的 `output/` 结果文件

这一节专门回答“`output/` 里哪些结果文件以后真的值得在简历编写和项目问答中反复引用”。

### 2.1 基础 baseline 与早期 full-val

这些文件适合回答：

- base model 起点是什么
- 早期 full-val 基线是什么
- 后面所有提升是相对谁来的

推荐长期引用：

- [eval_analysis.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase0_fullval_20260331/eval_analysis.md)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase0_fullval_20260331/summary.json)
- [run_info.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase0_fullval_20260331/run_info.json)

这些文件支撑的关键数字（**这一组是 base `Qwen2.5-Coder-7B-Instruct` 的 full-val 起点，不是 held-out test**，held-out 数字见 §2.2）：

- `humaneval accepted@1 = 0.87195`
- `mbpp_reg accepted@1 = 0.585`
- `codecontests_valid accepted@1 = 0.02564`（这是 117 题的 tier-1 开发集，不是 165 题的 held-out test）
- `codecontests_valid pass_ratio_mean = 0.14221`

### 2.2 主线 raw generalization regression

这些文件适合回答：

- base model 和主线 RL / SFT 模型在 held-out test 与回归集上的原始能力差异
- `step1300_rl` 为什么被视作当前 strongest held-out deployed base

推荐长期引用：

- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/qwen25coder7b_instruct_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/step1300_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/step40_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)

这些文件支撑的关键数字（**这一组是 held-out test 集 `CodeContests test` 165 题 + `HumanEval` 164 题 + `MBPP_reg` 200 题的 raw generation 结果，不含任何 repair**）：

- base model（`Qwen2.5-Coder-7B-Instruct`，未做 post-training）:
  - `codecontests_test accepted@1 = 0.01818`（绝对数 `3/165`）
  - `codecontests_test pass_ratio_mean = 0.12888`
  - `humaneval accepted@1 = 0.87195`
  - `mbpp_reg accepted@1 = 0.58`
- `step1300_rl`（当前主线 RL，当前 strongest deployed base）:
  - `codecontests_test accepted@1 = 0.07879`（绝对数 `13/165`，相比 base 多 10 道 exact solve，倍率约 4.3x）
  - `codecontests_test pass_ratio_mean = 0.25010`（相比 base 的 0.1289 几乎翻倍，说明提升不只是"多碰对几题"）
  - `humaneval accepted@1 = 0.89024`（+1.83 pp）
  - `mbpp_reg accepted@1 = 0.625`（+4.5 pp）
  - 说明：这条线既提升了 codecontests，也没有损坏通用 coding 回归集
- `step40 repair-SFT v1`（**注意**：这是 step900-base 旧线 SFT 第 40 step，不是 step1300_sft_v1_step60；两条 SFT 线完全不同）:
  - `codecontests_test accepted@1 = 0.06667`（绝对数 `11/165`）
  - `humaneval accepted@1 = 0.89634`
  - `mbpp_reg accepted@1 = 0.60`
  - 说明：**step40 是历史探索线的 raw regression 参照**，不是当前推荐部署的 checkpoint

### 2.3 RL 主线 checkpoint 选择

这些文件适合回答：

- `valid_big500` 上谁是 solve-count winner
- `step900 / step1000 / step1300` 各自代表什么
- 为什么 late-stage RL 不能只看一个指标

推荐长期引用：

- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step620_v3q3_to900_seed0_step900_codecontests_validbig500_raw_patched_rr828x_vastai3/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_codecontests_validbig500_raw_patched_rr828x_vastai3/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3_step1300_codecontests_validbig500_raw_patched_rr828x_vastai3/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_fixed_rejudge/step1300_rejudge1/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_fixed_rejudge/step1300_rejudge2/summary.json)

这些文件支撑的关键数字（**这一组是 dev 集 `valid_big500` 500 题、raw generation 的结果；不是 held-out**；用来做 checkpoint 对比与 late-stage RL 诊断）：

- `step900`: `accepted@1 = 0.128`（64/500），`pass_ratio_mean = 0.34722`
- `step1000`: `accepted@1 = 0.138`（69/500，**solve-count winner**），`pass_ratio_mean = 0.34440`
- `step1300`: `accepted@1 = 0.122`（61/500），`pass_ratio_mean = 0.35755`（**partial-credit winner**）
- `step1300` fixed-response rejudge（同一份 response 跑两次 judge，用来隔离 judge instability）:
  - rerun1 `accepted@1 = 0.122`, `pass_ratio_mean = 0.35747`
  - rerun2 `accepted@1 = 0.116`, `pass_ratio_mean = 0.35735`
  - 说明：rerun 之间的漂移量级 ≤ 0.6 pp accepted@1，属可接受噪声范围

这组文件非常适合在问答里解释：

- `step1000` 是 solve-count winner
- `step1300` 更像 partial-credit / broader-quality 强点
- patched 后 judge 漂移仍非零，但量级已可量化

### 2.4 one-turn repair 的代表性结果文件

这些文件适合回答：

- repair 到底有没有真实收益
- 开发集与 held-out test 的 repair 行为有什么不同
- 当前哪条 repair-SFT 线是真的有效，哪条只是开发集有效

#### 2.4.1 Focused delta slice

- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_delta69_repair/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_delta69_repair_raw/repair_summary.json)

这组文件回答的是「**high-partial near-miss 失败题上的纯 repair 能力**」问题。19 题是在 `valid_big500` 里挑出的 near-miss / high-partial 子集，不是随机子集，也不是全量。

关键数字：

- `repair_attempt_count = 19`（在 delta69 高 partial bucket 中做 one-turn repair）
- `repair_success_count = 5`
- `conditional_repair_success = 0.26316`（**这是 near-miss slice 的条件成功率，不是任意 repair 的全量成功率；不要直接当 headline**）
- first-pass `accepted@1 = 0.6522`，after-repair `accepted@1 = 0.7246`（repair_gain +7.25 pp）

#### 2.4.2 ValidBig500 Protocol A

- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolA_full/step900_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolA_full/step900_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5/repair_summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolA_full/step40_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolA_full/step40_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5/repair_summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolA_full/step1300_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolA_full/step1300_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5/repair_summary.json)

这组文件回答的是「**Protocol A = 纯修复能力**」：所有 checkpoint 在同一套冻结的错代码上做 one-turn repair，消掉 first-pass 生成差异这一 confound。

这组文件支撑：

- fixed-input 下 `step40` 是 dev 侧最好结果之一（注意 `step40` 是 **step900-base 旧 repair-SFT 线**的 SFT step，不是当前主线 step1300-base 线的一员）
- `step1300_rl` 没有在 fixed-input dev slice 上明显超过 `step900_rl`，说明 late-stage RL 并未显著提升"纯修复能力"，真正的收益集中在 partial-credit 与 first-pass 质量

#### 2.4.3 ValidBig500 Protocol B

- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB/step900_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB/step900_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/repair_summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB/step40_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB/step40_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/repair_summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB/step1300_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB/step1300_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/repair_summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB_step1300sftv1/step1300sftv1_step60_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB_step1300sftv1/step1300sftv1_step60_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/repair_summary.json)

这组文件回答的是「**Protocol B = 端到端部署口径**」：每个 checkpoint 先自己生成 first-pass，再对自己的错代码做 one-turn repair。更接近真实部署场景，但会受 first-pass 分布影响。

这组文件支撑：

- `step40`（**旧 repair-SFT 线，step900-base**，历史参照）在 dev 侧 end-to-end repair 有明显强点
- `step1300_sft_v1_step60`（**当前主线 repair-SFT v1，step1300-base**）在 current-lineage dev 集上有真实 repair gain：`repair_success_count = 9`, `conditional_repair_success ≈ 0.04523`（相比 step1300_rl 的 5 提升到 9）
- 重要：`step40` 与 `step1300_sft_v1_step60` **不构成 A/B 选择，而是时间顺序上的两条独立演化线**；前者是早期在 step900-base 上的探索线，后者是当前主线在 step1300-base 上的 v1 实验

#### 2.4.4 CodeContests Test Protocol A / B

- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair_protocolA/step1300_codecontests_test_repair_protocolA_shortdiag_p02_reuse_step900_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair_protocolA/step1300_codecontests_test_repair_protocolA_shortdiag_p02_reuse_step900_rr808x_vastai5/repair_summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair/step1300_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair/step1300_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/repair_summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair_step1300sftv1/step1300sftv1_step60_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair_step1300sftv1/step1300sftv1_step60_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/repair_summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair_protocolA_step1300sftv1/step1300sftv1_step60_codecontests_test_repair_protocolA_shortdiag_p02_reuse_step900_rr808x_vastai5/summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair_protocolA_step1300sftv1/step1300sftv1_step60_codecontests_test_repair_protocolA_shortdiag_p02_reuse_step900_rr808x_vastai5/repair_summary.json)

这组文件回答的是「**held-out test 的 repair 能力与部署口径**」——既看 Protocol A 的纯修复能力，也看 Protocol B 的端到端部署口径，用来判断 dev gain 能否外推到 held-out。

这组文件支撑：

- held-out `Protocol A` 很窄，当前几乎没有 solve gain：说明目前 repair pipeline 的"纯修复能力"在 held-out test 上还没有足够的提升
- held-out `Protocol B` 说明 `step1300_rl` 仍是 **strongest deployed repair base**（raw `7.88%` → after-repair `10.91%`，`repair_attempt_count=150, repair_success_count=5, conditional_repair_success=3.33%`）
- `step1300_sft_v1_step60` 在 held-out test 上没有超过 `step1300_rl`（raw 相同，after-repair `9.70%`，`repair_success_count=3`）——这是当前 repair-SFT v1 的关键边界：**dev gain 是真的，但尚未外推到 held-out**

### 2.5 repair prompt ablation

这组结果适合回答：

- 你有没有真正比较 `code_only` 和 `short_diagnosis_code`
- prompt 设计是靠拍脑袋还是做过小消融

推荐长期引用：

- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_ablation/grpo_a1_step1300_validbig500_repair_reuse_ablate_code_only_p06_wa_re/repair_summary.json)
- [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_ablation/grpo_a1_step1300_validbig500_repair_reuse_ablate_short_diag_p06_wa_re/repair_summary.json)

当前这两份文件显示：

- 两者在该小 ablation 下都得到 `repair_success_count = 4`
- `conditional_repair_success = 0.06061`

这更适合支持“做过 prompt 格式对照”，不适合支持“short diagnosis 已显著优于 code-only”。

### 2.6 早期 teacher / repair-SFT 探索线

这些文件适合回答：

- 项目不是从 `step900` 才开始做 SFT / teacher
- 早期 `step200/step400` 线已经做过一轮完整探索、复盘和换线

推荐长期引用：

- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase2_sft_eval_suite_v1/phase2_sft_pre_sft_step200_fullsuite_v1_retry1/repair_val/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase2_sft_eval_suite_v1/phase2_sft_step20_fullsuite_v1_retry1/repair_val/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase2_sft_eval_suite_v1/phase2_sft_pre_sft_step200_fullsuite_v1_retry1/humaneval_mini/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase2_sft_eval_suite_v1/phase2_sft_step20_fullsuite_v1_retry1/humaneval_mini/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/phase2_repair_sft_step400_r3a2_clean_v1_step10_codecontests_validbig500/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/phase2_repair_sft_step400_r3a2_clean_v1_step20_codecontests_validbig500/summary.json)
- [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/repair_v2b_step400_student_reference_eval/global_step_400_stepref_v2b_step400/summary.json)

这些文件支撑：

- `step200 -> early SFT` 在 mini/fullsuite 上做过系统探索
- `step400` repair-SFT v1 在 `valid117` / canary 上有局部现象，但在 `valid_big500` 上没有超过 pre-SFT `step400`
- `step400` student-reference strict eval 已经真实打通

---

## 3. Claim 目录

下面的 claim 不是最终简历语言，而是“这个项目里我真实做过哪些事”的展开素材。

---

## 4. 项目总叙事与系统范围

### Claim 4.1

- `Claim`
  - 我搭建并持续迭代了一条面向算法代码生成的完整后训练链路，覆盖 `curriculum-aware RL -> shared verifier evaluation -> one-turn repair -> repair-conditioned SFT`。
- `强度`
  - `A`
- `一句话 claim`
  - 端到端打通了 curriculum RL、统一判题、one-turn repair、repair-SFT 的后训练链路，各环节互相对接而非孤立 demo
- `详细展开`
  - **发生了什么**：项目要把通用 coder base 推到能做竞赛编程题的 RL checkpoint，单纯跑 RL 已不足——late-stage RL 会出现"partial-credit 继续涨但 solve-count 不涨"的饱和现象，需要补 one-turn repair 把 near-miss 拉成 solve，且 repair 本身也需要 SFT 进一步固化。
  - **我做了什么**：把四个原本独立的阶段串成一条数据闭环——RL 训练用 bucket-aware curriculum 决定采样分布；评测与 reward 走同一套 shared verifier（`src/verifier/shared.py`）保证口径一致；repair pipeline 复用 RL 训练中的 first-pass 做 diagnosis + second-pass；teacher 生成 + QC 后的高精度样本再回流做 repair-conditioned SFT。
  - **体现的能力**：不是"跑过几次 RL"的单点实验，而是负责过一条覆盖训练 / 评测 / 数据治理 / 运维的完整后训练系统；对相邻阶段的接口、数据流和评测契约都有设计级别的把控。
- `Doc`
  - [README.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/README.md)
  - [curriculum_rl_pilot_v8_explainer.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_rl_pilot_v8_explainer.md)
  - [repair_phase4_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_phase4_design.md)
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
- `Key Result Files`
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_codecontests_validbig500_raw_patched_rr828x_vastai3/summary.json)
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_delta69_repair/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_delta69_repair_raw/repair_summary.json)
  - [step1300_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.summary.json)
- `Code`
  - [phase0_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py)
  - [phase4_repair_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py)
  - [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)
  - [verifier/shared.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py)

### Claim 4.2

- `Claim`
  - 这个项目不是单纯“调模型分数”，而是同时覆盖实验设计、评测协议、数据治理、teacher/QC、sandbox 稳定性和远端运行编排。
- `强度`
  - `A`
- `一句话 claim`
  - 同时承担 ML post-training + 评测系统 + 数据治理 + 远端实验运维，技能跨度贯穿研究到工程
- `详细展开`
  - **发生了什么**：一个人跑 RL 项目最容易的失败模式是把一切都包给第三方框架或教程脚本，结果评测口径不对、数据脏、判题不稳都会让训练结论失去意义。
  - **我做了什么**：在 verl + SandboxFusion 底座上，自己补齐了 shared verifier 契约、curriculum 数据闭环、Protocol A/B repair 评测、teacher-QC 与 reject audit 流水线、sandbox 并发修复、远端租机 runbook 与 handoff 文档，让每一层都可以被独立审查。
  - **体现的能力**：项目内 ownership 跨越 ML / systems / evaluation / data quality / reproducibility 五条轴；这一条是"你在项目里到底负责什么"类问题的总括答案。
- `Doc`
  - [current_asset_and_data_flow_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/current_asset_and_data_flow_guide.md)
  - [shared_verifier_infra_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/shared_verifier_infra_guide.md)
  - [sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_judge_repair.md)
  - [new_machine_teacher_qc_runbook_2026-04-15.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/new_machine_teacher_qc_runbook_2026-04-15.md)
- `Code`
  - [build_problem_quarantine_v3.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_problem_quarantine_v3.py)
  - [build_repair_conditioned_teacher_requests.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_repair_conditioned_teacher_requests.py)
  - [run_phase4_repair_eval_checkpoint.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_phase4_repair_eval_checkpoint.sh)
  - [run_phase2_repair_sft.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_phase2_repair_sft.sh)

---

## 5. RL 主线、Checkpoint 分析与原始能力提升

### Claim 5.1

- `Claim`
  - 我做过多 checkpoint 的 curriculum RL 主线对比，并把 checkpoint 选择建立在系统化评测上，而不是某个单次跑出来的分数。
- `强度`
  - `A`
- `一句话 claim`
  - 在 500 题开发集上对齐比较 step620/700/800/900/1000/1200/1300 多个 checkpoint，按部署/repair/SFT anchor 角色分别选人
- `详细展开`
  - **发生了什么**：跑完 1300 步 GRPO（A1 variant，asymmetric clip + token-mean + no KL）之后，如果只看"最后一个 checkpoint"，会错判 late-stage RL 的动态——后期 solve-count 实际上是先升后降（step1000 峰值 13.8%，step1300 回到 12.2%），但 pass_ratio_mean 仍在爬（step1300 反而是 partial-credit 最高）。
  - **我做了什么**：把 step620 / step700 / step800 / step900 / step1000 / step1200 / step1300 这 7 个关键点全部在 `valid_big500`（500 题 dev）上跑 raw eval，明确划分三个 role：step1000 = solve-count winner、step1300 = partial-credit winner + strongest held-out deployed base、step900 = fixed-input Protocol A 的 reuse anchor。
  - **体现的能力**：不用单次 run 拍板，而是把 checkpoint selection 建成可追溯的多指标决策；下游的 repair base、repair-SFT base 选择都能直接引用这一轮对比的结果。
- `Doc`
  - [a1_formal_run_metrics_summary.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/a1_formal_run_metrics_summary.md)
  - [checkpoint_selection_and_valid_big_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/checkpoint_selection_and_valid_big_plan.md)
  - [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)
- `Key Result Files`
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step620_v3q3_to900_seed0_step900_codecontests_validbig500_raw_patched_rr828x_vastai3/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_codecontests_validbig500_raw_patched_rr828x_vastai3/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3_step1300_codecontests_validbig500_raw_patched_rr828x_vastai3/summary.json)
- `Evidence detail`
  - `step900`: `accepted@1 = 0.128`, `pass_ratio_mean = 0.34722`
  - `step1000`: `accepted@1 = 0.138`, `pass_ratio_mean = 0.34440`
  - `step1300`: `accepted@1 = 0.122`, `pass_ratio_mean = 0.35755`
  - 这直接支持：
    - `step1000` 是 solve-count winner
    - `step1300` 是 stronger partial-credit / broader-quality checkpoint

### Claim 5.2

- `Claim`
  - 我把 solve-count 和 pass-ratio 拆开分析 late-stage RL，因此能够解释为什么不同 checkpoint 看起来“都很强”，但强的维度不一样。
- `强度`
  - `A`
- `一句话 claim`
  - 用 exact-solve 与 partial-credit 双轴解释 late-stage RL，识别出不同 checkpoint 的强项维度不同
- `详细展开`
  - **发生了什么**：后期 RL 训练常见一个"看起来在涨"但单指标会误导的现象——valid_big500 上 step1000 到 step1300，`accepted@1` 从 13.8% 跌到 12.2%，但 `pass_ratio_mean` 反而从 34.4% 爬到 35.8%。单看 accepted@1 会得出"training 掉了"，单看 pass_ratio 会得出"training 还在涨"，两者都不完整。
  - **我做了什么**：把这两类信号**解耦成两个独立角色**：exact-solve 指标用来选 solve-count winner（部署偏重 cold output 正确性时用）；partial-credit 指标用来选 partial-credit winner（更接近 repair base 的选法，因为 near-miss 更多意味着 repair 更有杠杆）。同时在 handoff / checkpoint review 文档里显式记录"同一次 RL run 两个 winner 不是一个 checkpoint"。
  - **体现的能力**：对 RL 训练动态有过真实观察，能够在缺少 held-out 信号时只用开发集双指标完成合理 checkpoint 分工。
- `Doc`
  - [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
- `Key Result Files`
  - 同 Claim 5.1 的三份 `valid_big500` summary
- `边界`
  - 这条 claim 适合解释“研究判断”，不适合写成某个单独数字 headline

### Claim 5.3

- `Claim`
  - `step1300_rl` 相比 base model 在 held-out `codecontests_test / HumanEval / MBPP` 上都有 raw 能力提升，是当前 strongest held-out deployed base。
- `强度`
  - `A`
- `一句话 claim`
  - held-out CodeContests test raw accepted@1 从 1.8% 提升到 7.9%（4.3x），同时 HumanEval / MBPP 没有退步
- `详细展开`
  - **发生了什么**：RL 训练如果只在 dev 集上看结果，很容易被 dev overfit。需要一份完全 held-out、训练过程中完全不可见的 test 集来验证 raw generation 的真实泛化。
  - **我做了什么**：用 `rl_testset_regression/` 下的 canonical 跑法（patched sandbox + direct-backend RR）在 3 个 held-out 上同时跑 base 和 step1300_rl 的 raw generation——无任何 repair、无任何 teacher 数据参与。CodeContests test 165 题 accepted@1 从 `3/165 = 1.82%` 提升到 `13/165 = 7.88%`（绝对 +6.06 pp，倍率 4.33x）；pass_ratio_mean 从 `0.129` 提升到 `0.250`（几乎翻倍，说明提升不只是"多碰对几题"而是整体分布上移）；HumanEval `+1.83 pp`（87.20% → 89.02%）、MBPP `+4.5 pp`（58.00% → 62.50%）。
  - **体现的能力**：raw generalization 提升不是 dev-only 的 signal，是 3 个 held-out 数据集的一致提升；且附带回归集 guardrail（模型没有被训成只会 codecontests 的 narrow specialist）。这是整个项目**最经得起追问**的一条数字 headline。
- `Doc`
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
- `Key Result Files`
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/qwen25coder7b_instruct_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/step1300_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)
- `Evidence detail`
  - base model:
    - `codecontests_test accepted@1 = 0.01818`
    - `humaneval accepted@1 = 0.87195`
    - `mbpp_reg accepted@1 = 0.58`
  - `step1300_rl`:
    - `codecontests_test accepted@1 = 0.07879`
    - `humaneval accepted@1 = 0.89024`
    - `mbpp_reg accepted@1 = 0.625`

### Claim 5.4

- `Claim`
  - 我不是只在训练结束后看一个 summary，而是持续跟踪 entropy、actor step 时延、reward 长尾和 benchmark 漂移，用这些信号诊断训练 regime。
- `强度`
  - `B`
- `一句话 claim`
  - 训练中持续看 entropy / step latency / reward tail / benchmark drift 四类信号，把 RL 当白盒而不是黑盒
- `详细展开`
  - **发生了什么**：GRPO 训练跑 1000+ step 时，最容易出的问题是 entropy 坍塌（policy 过早确定 → 探索不足）或 reward long-tail 拉崩单 batch 时延，但很多项目只在最后看一个 summary，这类问题会被忽略。
  - **我做了什么**：训练过程中用 wandb 持续观察 4 组信号——(a) policy entropy 轨迹；(b) `update_actor` 单步时延；(c) per-sample reward 分布的长尾；(d) 阶段性在 `valid_big500` 上的 benchmark 漂移。把观察写进 `a1_formal_run_metrics_summary.md`，并据此判断"当前 regime 还健康，不需要触发 rescue config"。
  - **体现的能力**：对 RL 训练动力学有在线诊断意识，能从 wandb 信号主动发现潜在问题，而不是等事后解释。
- `Doc`
  - [a1_formal_run_metrics_summary.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/a1_formal_run_metrics_summary.md)
- `Evidence detail`
  - policy entropy 从约 `0.168` 收敛到约 `0.113`，**没有触发 collapse 预警阈值**（算法决策指南里设的阈值约 `0.05`）
  - `update_actor` 单步时延稳定在 `43~44s`，说明 RL 主循环没有被 rollout / reward 阻塞
  - reward long-tail 被识别为主瓶颈之一——少数长题会把 batch 拖慢，后续通过 reward 层 timeout 约束缓解

### Claim 5.5

- `Claim`
  - 早期就建立了 full-val / same-protocol / smoke/schema baseline，给后续协议漂移和结果解释提供了对照基座。
- `强度`
  - `B`
- `详细展开`
  - 这个项目里后来出现过模型生成漂移、judge 漂移、repair first-pass 漂移等问题。早期这些基础 baseline 的存在，使得后面排查“到底是模型变了、协议变了，还是 judge 变了”成为可能。
- `Doc`
  - [eval_analysis.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase0_fullval_20260331/eval_analysis.md)
- `Key Result Files`
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase0_fullval_20260331/summary.json)
  - [run_info.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase0_fullval_20260331/run_info.json)
- `Evidence detail`
  - `humaneval = 0.87195`
  - `mbpp_reg = 0.585`
  - `codecontests_valid = 0.02564`

---

## 6. Curriculum 设计与数据分桶

### Claim 6.1

- `Claim`
  - 我设计并推进了 bucket-aware curriculum，而不是把所有训练题都视为同一难度分布。
- `强度`
  - `A`
- `详细展开`
  - 项目里后续几乎所有关键动作都和 bucket 相关：RL 采样、checkpoint 理解、repair 候选切片、teacher request 分层、SFT 数据配比。也就是说，bucket 不只是一个“分析标签”，而是真正驱动了实验主线。
- `Doc`
  - [curriculum_rl_pilot_v8_explainer.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_rl_pilot_v8_explainer.md)
  - [curriculum_manifest_v2_review_instructions.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_manifest_v2_review_instructions.md)
- `Data`
  - [runtime_curriculum_coverage.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/runtime_curriculum_coverage.json)
  - [curriculum_apply_report_step600_v3.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_review_local/curriculum_apply_report_step600_v3.json)
  - [focused_manifest_v3_review_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_review_local/focused_manifest_v3_review_summary.json)
- `Code`
  - [build_curriculum_manifest.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_curriculum_manifest.py)
  - [apply_curriculum_updates.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/apply_curriculum_updates.py)

### Claim 6.2

- `Claim`
  - 我做过 focused curriculum review，把 `C_hard_partial` 和 anchor 题目的去留变成可审计的 manifest 决策。
- `强度`
  - `A`
- `详细展开`
  - 这件事的重要性在于：curriculum 不是完全自动化黑箱，而是对关键桶做过显式 review。对后续面试来说，这可以很好地说明我对“数据分布如何影响训练”和“高噪声 hard bucket 如何进入系统”是有控制意识的。
- `Doc`
  - [curriculum_manifest_v2_review_instructions.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_manifest_v2_review_instructions.md)
- `Data`
  - [curriculum_apply_report_step600_v3.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_review_local/curriculum_apply_report_step600_v3.json)
  - [focused_manifest_v3_review_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_assets/step600_v3_review_local/focused_manifest_v3_review_summary.json)
- `Evidence detail`
  - `keep_C = 40`
  - `move_to_U = 21`
  - `move_to_B = 3`
  - `add_to_A = 14`

---

## 7. 评测协议、漂移控制与实验严谨性

### Claim 7.1

- `Claim`
  - 我明确区分了 fixed-input repair 能力和 end-to-end self-repair 能力，并把它们制度化为 `Protocol A / Protocol B`。
- `强度`
  - `A`
- `一句话 claim`
  - 把 repair 评测拆成 Protocol A（纯修复能力）+ Protocol B（端到端部署口径）两套正交协议，让不同 checkpoint 可以被公平比较
- `详细展开`
  - **发生了什么**：repair 评测常见的混淆是——不同 checkpoint 生成的 first-pass 本来就不同（有的错 100 道有的错 150 道），然后又在各自的错代码上做 repair，最终的 after-repair accepted@1 究竟反映的是"谁的修复能力强"还是"谁本来 first-pass 就少错"完全分不清。
  - **我做了什么**：设计两套正交协议——**Protocol A（fixed-input）**：所有 checkpoint 都在同一套冻结的错代码（由 step900_rl 生成并冻结）上做 repair，消掉 first-pass 差异，得到"纯修复能力"；**Protocol B（end-to-end self-repair）**：每个 checkpoint 自己生成 first-pass 再修自己，得到"部署口径"。两套协议都落成 `phase4_repair_eval.py` 的具体 flag，并写进 `repair_eval_current_contract_2026-04-18.md` 作为 canonical 契约。
  - **体现的能力**：评测协议设计不是"加一个 flag"，而是把"你到底想度量什么"拆成正交信号；这套拆法在面试里是最容易被追问、也最能展示 evaluation rigor 的一条。
- `Doc`
  - [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)
  - [repair_eval_protocol_matrix.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_protocol_matrix.md)
  - [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)
  - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)
- `Code`
  - [phase4_repair_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py)

### Claim 7.2

- `Claim`
  - 我实现了 `reuse-first-pass`，用来冻结 first-pass 响应，避免 repair 评测被生成漂移污染。
- `强度`
  - `A`
- `一句话 claim`
  - 把 first-pass 冻结成 artifact 供 repair 评测复用，消掉采样抖动这一 confound，让 repair 对比可重复
- `详细展开`
  - **发生了什么**：在贪心采样下同一模型同一题理论上 deterministic，但实际跑下来 sandbox 调度 + judge 随机性会让 first-pass 的 accepted 在不同 run 之间有 1~2 pp 抖动。repair 评测如果每次现跑 first-pass，就相当于同时让 first-pass 和 second-pass 都在变，结论会被噪声淹没。
  - **我做了什么**：把每个 `(checkpoint, problem)` 的 first-pass 响应存成 artifact（JSONL），后续 repair eval 跑的时候只消费这份冻结 artifact 做 diagnosis + second-pass。校验方式：repair eval 启动时对 `(model_id, problem_id)` 做一一对应检查，**缺任何一题 hard-fail**，不允许补采样。truncated 源（即 first-pass 被 max_tokens 截断）单独打标，避免当 repairable 误算入分母。
  - **体现的能力**：实验可解释性 + 数据契约意识——把"怎么保证 repair 实验之间可比"从口头约定变成文件级合约。
- `Doc`
  - [phase4_step1_repair_eval_implementation_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/phase4_step1_repair_eval_implementation_plan.md)
  - [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)
- `Code`
  - [phase4_repair_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py)

### Claim 7.3

- `Claim`
  - 我做过 fixed-response rejudge，专门用来分离 residual judge instability 和生成漂移。
- `强度`
  - `A`
- `一句话 claim`
  - 用同一份冻结 response 做两次 judge，把"评测为什么不稳定"拆成生成漂移 vs judge 漂移两个独立信号
- `详细展开`
  - **发生了什么**：项目早期出现过同一个 checkpoint 跑两次 eval 结果不一致的情况。这里有两种可能的原因：(a) 采样随机性导致生成不一样；(b) sandbox 判题本身不稳定。不拆开两者就无法解释评测噪声到底有多大，也无法说服面试官"数字可信"。
  - **我做了什么**：用 `valid_big500` 的 step1300 一份冻结 response 跑两次 judge（rerun1 / rerun2），观察差异。结果：`accepted@1` 0.122 vs 0.116（差 0.6 pp），`pass_ratio_mean` 0.35747 vs 0.35735（几乎相同）。这量化了"judge 漂移量级 ≤ 0.6 pp accepted@1"这个边界。
  - **体现的能力**：评测系统意识——不是声称 judge deterministic，而是**把 residual nondeterminism 定位、量化并写进协议解释**，让后续所有 repair 对比都能用这个 bound 去判断"差异是否可信"。
- `Doc`
  - [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
- `Key Result Files`
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_fixed_rejudge/step1300_rejudge1/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_fixed_rejudge/step1300_rejudge2/summary.json)
- `边界`
  - 这条不应写成“彻底消灭了 nondeterminism”
  - 更准确的说法是“定位并量化了主要来源”

### Claim 7.4

- `Claim`
  - 我把 bucket-level repair yield 也纳入结果解释，识别出 repair 更像 near-miss fallback，而不是对所有失败题同等有效。
- `强度`
  - `B`
- `详细展开`
  - 这条 claim 更偏结果理解。它的重要性在于后续的 trigger 选择、prompt 预算分配和 SFT 数据切片，都是建立在“高 through-rate near-miss 更值得 repair”这个发现上的。
- `Doc`
  - [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)
  - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)

---

## 8. One-turn repair 设计、prompt 和结果

### Claim 8.1

- `Claim`
  - 我实现了完整的 verifier-guided one-turn repair pipeline，支持错误类型感知的 failure summary、feedback 构造与 second-pass 评测。
- `强度`
  - `A`
- `详细展开`
  - 这条工作覆盖的不是一个 prompt，而是一条端到端流水线：从 first-pass response、逐题 verifier 结果、错误类型、failure case 选取，到 second-pass prompt，再到新的 verifier 结果与成本统计。这条链是后续 repair eval 和 repair-SFT 数据构建的共同底座。
- `Doc`
  - [repair_phase4_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_phase4_design.md)
  - [phase4_step1_repair_eval_implementation_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/phase4_step1_repair_eval_implementation_plan.md)
- `Code`
  - [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)
  - [phase4_repair_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py)

### Claim 8.2

- `Claim`
  - 我审计并替换了 naive 的“first failing testcase”规则，改成更接近最小反例 / 最清晰异常的 deterministic case selection。
- `强度`
  - `A`
- `一句话 claim`
  - 把 "repair prompt 里塞哪条失败样例" 从 naive 取第一条改成按错误类型分流的 deterministic 规则，审计后替换
- `详细展开`
  - **发生了什么**：repair prompt 里给模型看的"失败样例"质量极大决定修复效果。原 pipeline 取 `per_case_results` 的第一条 fail 作为 diagnosis 输入，但审计后发现两个问题：(a) 前几条 case 里可能是 `syntax_error` 或 `timeout` 这种信息噪声大的类型，淹没了 WA 的反例信号；(b) `per_case_results` 字段本身有截断现象，直接取第一条可能拿到被截断的大 case。
  - **我做了什么**：先做 case selection instance-level audit（见 `testcase_selection_rule_audit_v1.md`），确认问题确实存在；再落成 deterministic 选取规则——WA 取**最小反例**（输入长度最短的 fail）、RE 取**最清晰异常栈**、TLE 取**最短输入**；同时在 prompt 构造阶段主动过滤掉被截断的 case。规则直接写进 `repair_feedback.py`，所有 repair eval / teacher generation 都共用这一套规则。
  - **体现的能力**：数据接口审计能力——不是发现"数字不对"就调参数，而是反推到 prompt 输入源头做规则替换，同时留下可追溯的 audit artifact。
- `Doc`
  - [testcase_selection_rule_audit_v1.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_analysis/testcase_selection_rule_audit_v1.md)
  - [testcase_selection_rule_recommendation_v1.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_analysis/testcase_selection_rule_recommendation_v1.json)
  - [phase4_step1_repair_eval_implementation_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/phase4_step1_repair_eval_implementation_plan.md)
- `Data`
  - [testcase_selection_instance_audit_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_analysis/testcase_selection_instance_audit_v1.jsonl)
- `Code`
  - [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)

### Claim 8.3

- `Claim`
  - 我设计并集成了 `code_only` 与 `short_diagnosis_code` 两类 repair prompt，并把它们纳入统一的切换式评测链。
- `强度`
  - `A`
- `一句话 claim`
  - 设计 code-only 与 short-diagnosis 两种 repair prompt，并将 prompt family 与 teacher generation / SFT 训练统一成同一套可控结构
- `详细展开`
  - **发生了什么**：repair prompt 有两种常见选择——(1) `code_only` 让模型只输出新代码；(2) `short_diagnosis_code` 让模型先写一段简短诊断再给新代码。前者 output token 短但缺可审核的中间步骤，后者多一段 diagnosis 便于 teacher QC 和 SFT training。
  - **我做了什么**：在 `repair_feedback.py` 里把两种 prompt 做成可切换 mode，评测链、teacher 生成、SFT 训练共用同一套 prompt builder。最终选择 `short_diagnosis_code` 作为主线，不是因为性能更高（见下方 caveat），而是因为：(a) 有可审核的中间 diagnosis，teacher QC 和 SFT training 都能利用；(b) 结构化输出更适合做后续 reward shaping 或规则约束。
  - **体现的能力**：prompt 设计不是"拍脑袋改一句话"，而是从 evaluation → teacher generation → SFT training 整个 pipeline 看 prompt 结构化的价值。
- `Doc`
  - [repair_prompt_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_prompt_design.md)
- `Key Result Files`
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_ablation/grpo_a1_step1300_validbig500_repair_reuse_ablate_code_only_p06_wa_re/repair_summary.json)
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_ablation/grpo_a1_step1300_validbig500_repair_reuse_ablate_short_diag_p06_wa_re/repair_summary.json)
- `Code`
  - [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)
- `边界`
  - 当前 ablation 两组都只有 `conditional_repair_success ≈ 0.06061`（`repair_success_count = 4`，样本量约 66 题），**样本量太小，统计上不可区分**
  - 因此只能支持"**做过 prompt 格式对照**"这种过程 claim，不适合直接表述成"short diagnosis 显著优于 code-only"的结果 claim
  - 面试被问到时应主动说明样本量不足，避免被追到"你怎么证明是 diagnosis 带来的提升"时答崩

### Claim 8.4

- `Claim`
  - 我在 `delta69`、`valid_big500`、`codecontests_test` 三层集合上都验证过 one-turn repair，能够清楚地区分“focused slice 成功”与“held-out generalized repair 还不够强”。
- `强度`
  - `A`
- `一句话 claim`
  - 在三个难度递增的集合（focused near-miss → full dev → held-out test）上验证 repair，划清楚"哪里成了 / 哪里还没成"的边界
- `详细展开`
  - **发生了什么**：repair 项目最常见的 pitfall 是在一个 focused slice 上拿到漂亮数字（如"26.3% 修复率"）就停下来作为结论，但这个数字根本无法外推。
  - **我做了什么**：**三层递进式验证**——(1) `delta69` focused slice 上验证 "repair 在 near-miss 高 partial 样本上有可见收益"（conditional 26.3%）；(2) `valid_big500` full dev 上验证"同样的 repair pipeline 在全量开发集 Protocol B 上也有增益"（`step1300_sft_v1_step60` 成功 9 题 vs step1300_rl 成功 5 题）；(3) `codecontests_test` held-out 上验证是否外推——结果发现 `step1300_sft_v1_step60` 反而退步（held-out 成功 3 题 vs step1300_rl 5 题），由此得出"v1 recipe 在 held-out 上尚未超越 RL baseline，需要 v2 recipe"的结论。
  - **体现的能力**：不用 slice gain 包装整体结论；能主动把实验拉到更难的层看 transfer，并在失败时诚实地在 handoff 文档里记录"下一步 v2 为什么必须换 recipe"。
- `Doc`
  - [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)
  - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
- `Key Result Files`
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_delta69_repair/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_delta69_repair_raw/repair_summary.json)
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB_step1300sftv1/step1300sftv1_step60_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/repair_summary.json)
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair/step1300_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/repair_summary.json)
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair_step1300sftv1/step1300sftv1_step60_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/repair_summary.json)
- `Evidence detail`
  - `delta69`: `19 -> 5` successful repairs, `conditional_repair_success ≈ 26.3%`
  - `valid_big500 / step1300_sft_v1_step60`: `repair_success_count = 9`
  - `codecontests_test / step1300_rl`: `repair_success_count = 5`
  - `codecontests_test / step1300_sft_v1_step60`: `repair_success_count = 3`

---

## 9. Repair-conditioned SFT 数据构建、teacher generation 与 QC

### Claim 9.1

- `Claim`
  - 我构建的 repair-conditioned SFT 数据不是“teacher 单次生成直接入训练”，而是经过 request-unique、QC、regen、审计和最终 keep set 过滤的高精度数据资产。
- `强度`
  - `A`
- `一句话 claim`
  - repair-SFT 数据经过 request-unique → QC → regen → audit → keep-set 5 步筛，不把 LLM teacher 当无条件真值
- `详细展开`
  - **发生了什么**：最常见的 repair-SFT 失败模式是"teacher 模型跑一遍 repair，成功的就当 ground truth 入 SFT"——这种 pipeline 有两个致命问题：(a) teacher 自己也会有 WA 但因为某条测试通过就被收入训练；(b) 同一个 request_id 可能有多个 candidate answer 同时进训练，造成 SFT 分布偏移。
  - **我做了什么**：设计 5 阶段数据治理 pipeline——**(1) request-unique**：同一 `(problem, first_pass_artifact)` 在最终 parquet 只保留 1 行；**(2) primary QC**：teacher 输出的代码必须跑 shared verifier 全测试通过 (`accept_2_of_2`)；**(3) regen round**：primary QC 没过的 request 允许定向 regen 最多 2 轮；**(4) near-miss testcase audit**：对高 partial 但没 accept 的样本做 testcase / judge 层人工审核，决定是 salvage 还是 reject；**(5) final keep set**：所有阶段的保留 IDs 合并成一份带来源标记的 ledger。
  - **体现的能力**：把 LLM-as-teacher 的 trust 边界量化成具体数据门槛；整条 pipeline 在 `step900` 沉淀出 267 条 stable keep、在 `step1300` 沉淀出 553 条 expanded keep，每一条都可追溯来源。
- `Doc`
  - [step900_repair_conditioned_sft_data_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_repair_conditioned_sft_data_plan.md)
  - [step1300_repair_conditioned_sft_v2_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_repair_conditioned_sft_v2_plan.md)
- `Data`
  - [teacher_qc_v2_full.stable_keep_set.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/final_keep/teacher_qc_v2_full.stable_keep_set.summary.json)
  - [step1300_teacher_keep_set_v2.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.summary.json)

### Claim 9.2

- `Claim`
  - `step900` 这条 repair-conditioned 线最终沉淀出一份高精度 keep set 和一份更纯的 short-diagnosis-only 训练集。
- `强度`
  - `A`
- `详细展开`
  - 这条线很适合作为“repair-SFT v1 数据管线已经真实跑通”的证据。它同时说明我能做 request-level 去重、稳定性确认和 prompt-mode 控制，不只是把若干 JSONL 凑起来。
- `Data`
  - [teacher_qc_v2_full.stable_keep_set.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/final_keep/teacher_qc_v2_full.stable_keep_set.summary.json)
  - [step900_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.summary.json)
- `Evidence detail`
  - stable keep set: `267` rows
  - pure shortdiag dataset: `182` rows, `182` distinct requests
  - 全部 `short_diagnosis_code`
  - 全部 `accept_2_of_2`

### Claim 9.3

- `Claim`
  - `step1300` 这条 repair-conditioned 线在 primary QC 之外，还叠加了 regen round、near-miss testcase audit 和高通过率边界样本复核。
- `强度`
  - `A`
- `详细展开`
  - 这一条最能说明“数据集规模扩大不是靠放松门槛瞎灌”。相反，`step1300` 的 keep set 扩大是通过多轮可追溯的补救流程完成的：先 primary QC，再重生成，再对高通过率 near-miss 做 testcase/judge 审核，最后再合并进主 keep set。
- `Data`
  - [step1300_teacher_keep_set_v2.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.summary.json)
  - [step1300_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.summary.json)
- `Evidence detail`
  - 最终 `553` rows
  - 来源包括：
    - `primary_qc = 389`
    - `regen_round1_qc = 96`
    - `regen_round2_qc = 12`
    - `near_miss_testcase_audit_high_confidence = 13`
    - `near_miss_testcase_audit_regen_round1 = 11`
    - `remaining_high_pass_audit_0p85_to_0p9 = 11`
    - `remaining_high_pass_audit_0p7_to_0p85 = 21`

### Claim 9.4

- `Claim`
  - 我把 repair 数据按 `Core / Expansion / C` 等 strata 贯穿到了 teacher request、QC 和最终 dataset summary 中，而不是后验再去贴标签。
- `强度`
  - `A`
- `详细展开`
  - 这说明数据分层不是事后分析，而是真正影响了 teacher generation 配比、QC 解释和训练数据构成。后续如果要解释为什么某一版 SFT 更偏 hard repair，这些 strata 统计就是直接证据。
- `Data`
  - [step900_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.summary.json)
  - [step1300_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.summary.json)
- `Evidence detail`
  - `step900`: `Core = 87`, `Expansion = 70`, `SelectiveC = 25`
  - `step1300`: `CoreNearMiss = 119`, `ExpansionB = 118`, `CurrentCRecoverable = 316`
- `边界`
  - `step1300` 这版不能说是“平衡数据集”
  - 它更准确的描述是 `C_hard_partial` 偏重的 harder-repair-specialized 数据集

### Claim 9.5

- `Claim`
  - teacher 不通过样本并没有被简单丢弃，而是进入 reject audit / testcase audit / salvage 流程。
- `强度`
  - `A`
- `详细展开`
  - 这条 claim 很适合回答“你如何保证 teacher 数据质量”。项目里的做法不是“teacher 没过就是模型差”，而是把失败拆成 teacher bug、testcase issue、judge instability、needs manual review 等不同来源，然后有条件地救回边界样本。
- `Data`
  - [near_miss_wrong_answer_ge_0p9_testcase_audit_labels.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/near_miss_testcase_audit/agent_outputs/near_miss_wrong_answer_ge_0p9_testcase_audit_labels.jsonl)
  - [near_miss_wrong_answer_ge_0p9_regen_round1_testcase_audit_labels.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/near_miss_testcase_audit_regen_round1/agent_outputs/near_miss_wrong_answer_ge_0p9_regen_round1_testcase_audit_labels.jsonl)
  - [wrong_answer_0p7_to_0p85_remaining_unaudited_testcase_audit_labels.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/remaining_high_pass_audit_candidates/agent_outputs/wrong_answer_0p7_to_0p85_remaining_unaudited_testcase_audit_labels.jsonl)

---

## 10. Repair-SFT 训练本身与结论

### Claim 10.1

- `Claim`
  - 我在 `step900` 和 `step1300` 两条不同 base 上都跑过 repair-SFT，不是只做过一条偶然成功或失败的线。
- `强度`
  - `A`
- `详细展开`
  - 这条 claim 的价值在于说明我不是把一条 SFT 线的结果绝对化。项目里既有较早的 `step400` / `step900` repair-SFT 探索线，也有后来的 `step1300`-base current-lineage v1，这让“哪些结论是 recipe 问题，哪些是 base checkpoint 问题”可以被拆开。
- `Doc`
  - [step900_shortdiag_pure_sft_training_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_shortdiag_pure_sft_training_guide.md)
  - [step1300_shortdiag_pure_sft_training_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_shortdiag_pure_sft_training_guide.md)
  - [step400_sft_v1_postmortem.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/v2b_step400/step400_sft_v1_postmortem.md)
- `Key Result Files`
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/phase2_repair_sft_step400_r3a2_clean_v1_step10_codecontests_validbig500/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/phase2_repair_sft_step400_r3a2_clean_v1_step20_codecontests_validbig500/summary.json)
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB_step1300sftv1/step1300sftv1_step60_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/repair_summary.json)

### Claim 10.2

- `Claim`
  - `step1300`-base repair-SFT v1 在开发集 repair 上带来了真实增益，但在 held-out `codecontests_test` 上没有超过 `step1300_rl`，因此我明确得出“v2 必须换 recipe”的结论。
- `强度`
  - `A`
- `一句话 claim`
  - repair-SFT v1 dev gain 真，held-out 没跟上，因此不做 naive scale-up 而是转向 v2 recipe 重设计
- `详细展开`
  - **发生了什么**：跑完 repair-SFT v1（step1300-base + 553 条 shortdiag pure 数据），dev 集 `valid_big500` 的 Protocol B 确实出现 repair_success_count 从 5 提升到 9（conditional 约 4.5%）的真实增益；但在 held-out `codecontests_test` 上，Protocol B 的 repair_success_count 反而从 5 降到 3（after-repair `accepted@1` 从 `10.91%` 降到 `9.70%`）。
  - **我做了什么**：没有把这次 dev gain 包装成"repair-SFT 成功了"的 headline；也没有直接做 naive scale-up（"既然 553 条有效，就继续加到 1000 条"）。而是写下明确的 v2 要求——下一版 SFT 数据的 strata 分布需要 rebalance（当前 v1 在 `CurrentCRecoverable` 偏重 316 条 / 总 553 条，学成了 harder-repair-specialized shaper）；v2 plan 在 `step1300_repair_conditioned_sft_v2_plan.md` 已落文档但尚未执行。
  - **体现的能力**：对 dev → held-out 的 gap 有客观判断，不让自己被 dev-only gain 推着走；能把失败转成"下一版为什么必须这样改"的具体 plan，而不是简单搁置。
- `Doc`
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
  - [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)
  - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)
  - [step1300_repair_conditioned_sft_v2_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_repair_conditioned_sft_v2_plan.md)
- `Key Result Files`
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB_step1300sftv1/step1300sftv1_step60_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/repair_summary.json)
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair/step1300_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/repair_summary.json)
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair_step1300sftv1/step1300sftv1_step60_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/repair_summary.json)
- `Evidence detail`
  - `valid_big500 / step1300_sft_v1_step60`: `repair_success_count = 9`
  - `codecontests_test / step1300_rl`: `repair_success_count = 5`
  - `codecontests_test / step1300_sft_v1_step60`: `repair_success_count = 3`

### Claim 10.3

- `Claim`
  - 我在启动 repair-SFT 训练前做过 tokenizer-length audit、1-sample canary、checkpoint 保留策略与后续 repair eval 接力设计。
- `强度`
  - `B`
- `详细展开`
  - 这条更偏工程化训练实践。它体现的是：训练不是“parquet 一备好就 launch”，而是把长度风险、数据 schema、模型路径、磁盘空间和后续评测链一起考虑了。
- `Doc`
  - [step900_shortdiag_pure_sft_training_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_shortdiag_pure_sft_training_guide.md)
  - [step1300_shortdiag_pure_sft_training_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_shortdiag_pure_sft_training_guide.md)
- `Code`
  - [run_phase2_repair_sft.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_phase2_repair_sft.sh)

### Claim 10.4

- `Claim`
  - 我已经从 `step1300` pure shortdiag v1 得到一个比较清晰的 recipe 结论：小规模 pure repair-only 训练能提升 dev-side repair，但在 `CurrentCRecoverable` 偏重时容易学成 harder-repair-specialized shaper。
- `强度`
  - `B`
- `详细展开`
  - 这条 claim 很适合在项目问答里展示研究判断。它说明我没有只盯着“有没有一点提升”，而是会进一步解释“为什么提升只出现在 dev、为什么 held-out 没跟上、下一版 recipe 应该怎么改”。
- `Doc`
  - [step1300_shortdiag_pure_sft_training_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_shortdiag_pure_sft_training_guide.md)
  - [step1300_repair_conditioned_sft_v2_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_repair_conditioned_sft_v2_plan.md)

---

## 11. Sandbox、judge 稳定性与评测系统治理

### Claim 11.1

- `Claim`
  - 我定位并修复了 SandboxFusion 在并发场景下的主 nondeterminism 问题。
- `强度`
  - `A`
- `一句话 claim`
  - 把 sandbox 评测不稳定这条线索一路追到子进程清理与 stdout/stderr drain 竞态，写补丁收敛
- `详细展开`
  - **发生了什么**：评测早期出现过"同一份 response 跑两次 judge 会有不同结果"的诡异现象。最容易的误判是"是不是模型不 deterministic"或"是不是 vLLM 采样的 seed 问题"，但我在 fixed-response rejudge 里已经把 response 冻结了，生成侧不可能出问题——剩下的只能是 judge 侧。
  - **我做了什么**：沿着 SandboxFusion 的进程链一路追查，定位到两个 race：(a) 子进程在 `wait()` 前 stdout/stderr pipe 没完全 drain 就被 kill，部分输出丢失；(b) 并发场景下进程清理顺序不稳定，偶发返回旧 session 残留。在 `SandboxFusion/sandbox/runners/base.py` 和 `execution.py` 里写补丁（完整 drain + 强制同步 cleanup），把正式评测 judge 路径从 LB 改成 direct-backend RR 以绕开第三方调度层。
  - **体现的能力**：系统调试深度——不满足于"大概可能是某某问题"的表层描述，能在外部大型 codebase（SandboxFusion）里定位并提交具体修补；给整个项目的评测可信度奠基。
- `Doc`
  - [sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_judge_repair.md)
  - [sandbox_repair_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_repair_plan.md)
- `Code`
  - [base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py)
  - [execution.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/execution.py)
- `边界`
  - 不宜写成“彻底修好 judge”
  - 更准确的说法是“修掉主要 instability source，并把剩余漂移量级缩小到可测范围”

### Claim 11.2

- `Claim`
  - 我把正式评测标准收敛到 `patched sandbox + direct-backend client RR`，避免继续依赖更脆弱的 LB 路径。
- `强度`
  - `A`
- `详细展开`
  - 这个决定看起来像运维细节，但实际上直接影响实验口径。后面所有修复、teacher QC、repair eval、held-out test 对比，都是建立在这套收敛后的 judge 契约上。
- `Doc`
  - [sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_judge_repair.md)
  - [sandbox_concurrency_standard.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_concurrency_standard.md)
  - [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)

### Claim 11.3

- `Claim`
  - shared verifier、repair eval 和 teacher QC 不是三套各自独立的 judge 体系，而是被我逐步拉回到了同一套基础契约上。
- `强度`
  - `B`
- `详细展开`
  - 这条 claim 的重点是实验可解释性。只有当这些链路共享 judge 假设时，`raw -> repair -> teacher-QC -> SFT` 之间的比较才有意义。
- `Doc`
  - [shared_verifier_infra_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/shared_verifier_infra_guide.md)
  - [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)
- `Code`
  - [verifier/shared.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py)
  - [verifier/run_verifier_server.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/run_verifier_server.py)

---

## 12. 数据治理、quarantine 与 testcase 审计

### Claim 12.1

- `Claim`
  - 我建立了版本化 quarantine 体系，用来隔离污染题、判题不稳题和需要谨慎对待的题。
- `强度`
  - `A`
- `一句话 claim`
  - 建立 v1→v2→v3 三版 quarantine ledger，把"哪些题不可信"从口头知识变成可 diff、可复用的 JSONL
- `详细展开`
  - **发生了什么**：竞赛编程题数据集常见的问题是——测试用例覆盖不全、题面歧义、同题多答案、训练/测试污染（某道题在 train 里出现过同一问题的变种）。如果把这些题都不加区分地放进 RL 采样或 SFT 数据，会让评测信号失真。
  - **我做了什么**：建立 3 版 quarantine ledger（`problem_quarantine_v1/v2/v3.jsonl`），每道题标 `hard_blacklist` / `caution` / `unresolved` 三类并带原因，累计 `hard_blacklist = 777`、`caution = 32`、`unresolved = 76`。这份 ledger 不只是审阅文档，它被直接挂到 curriculum manifest 构建（`filter_curriculum_manifest_by_quarantine.py`）、teacher candidate 构建（`build_repair_conditioned_teacher_requests.py`）和 SFT 数据构建三条上游流水线。
  - **体现的能力**：数据治理能力——认识到"训练集里有什么就全吃"是一种风险，并把风险降维成具体的、可审计的 artifact；对数据分布污染有主动设计级的防御。
- `Doc`
  - [problem_quarantine_v3_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/problem_quarantine_v3_guide.md)
- `Data`
  - [problem_quarantine_v2_build_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/review_assets_v1/problem_quarantine_v2_build_summary.json)
  - [problem_quarantine_v3.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/problem_quarantine_v3.jsonl)
- `Code`
  - [problem_quarantine.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/problem_quarantine.py)
  - [build_problem_quarantine_v2.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_problem_quarantine_v2.py)
  - [build_problem_quarantine_v3.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_problem_quarantine_v3.py)
- `Evidence detail`
  - `hard_blacklist_count = 777`
  - `caution_count = 32`
  - `unresolved_count = 76`

### Claim 12.2

- `Claim`
  - quarantine 不是纸面规则，而是和 curriculum manifest、teacher candidate 构建、SFT 数据构建存在实际联动。
- `强度`
  - `A`
- `详细展开`
  - 这条可以很好地回答“你怎么防止脏题流入训练”。项目里不是手工记住几道坏题，而是把 quarantine 变成上游筛选的一部分。
- `Code`
  - [filter_curriculum_manifest_by_quarantine.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/filter_curriculum_manifest_by_quarantine.py)
  - [build_curriculum_manifest.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_curriculum_manifest.py)
  - [build_repair_conditioned_teacher_requests.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_repair_conditioned_teacher_requests.py)

### Claim 12.3

- `Claim`
  - 我做过 testcase-level 审计，用来区分 teacher bug、testcase issue、judge instability 和真正不值得保留的样本。
- `强度`
  - `A`
- `详细展开`
  - 这条 claim 在面试里很有价值，因为它能说明我对“数据不通过不一定全是模型错”有很强的意识。尤其是在 competitive programming 这类任务里，题面歧义、多个合法答案、坏测试、历史污染都是真问题。
- `Data`
  - [near_miss_testcase_audit](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/near_miss_testcase_audit)
  - [near_miss_testcase_audit_regen_round1](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/near_miss_testcase_audit_regen_round1)
  - [remaining_high_pass_audit_candidates](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/remaining_high_pass_audit_candidates)

---

## 13. 远端实验运维、迁移与协作

### Claim 13.1

- `Claim`
  - 我能在多台租用 GPU 机器之间迁移完整实验链，包括 repo、sandbox 池、teacher QC、repair eval 和训练恢复。
- `强度`
  - `B`
- `详细展开`
  - 这条 claim 不一定放在简历第一屏，但非常适合系统/平台/ML infra 岗位问答。它说明我不是只能在本地或单机跑一个实验，而是能在不稳定、会重启、会丢环境的远端租机环境里保持项目推进。
- `Doc`
  - [new_machine_teacher_qc_runbook_2026-04-15.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/new_machine_teacher_qc_runbook_2026-04-15.md)
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)

### Claim 13.2

- `Claim`
  - 我为长跑训练和后续续跑设计过 watcher / handoff guard / launch-check 机制，减少 GPU 时间因人工延迟而浪费。
- `强度`
  - `B`
- `详细展开`
  - 这条更偏实验运维成熟度。项目里不是所有事情都能手动盯着做，因此续跑 watcher、launch marker、source-process exit guard 这类机制可以很好地展示“我会把实验做成可持续运行的系统”。
- `Code`
  - [wait_launch_grpo_a1_resume1200_to1400_overnight_vastai3.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/wait_launch_grpo_a1_resume1200_to1400_overnight_vastai3.sh)
  - [run_grpo_a1_resume1200_to1400_overnight_vastai3.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_grpo_a1_resume1200_to1400_overnight_vastai3.sh)

### Claim 13.3

- `Claim`
  - 我有完整的 handoff / runbook / inventory 文档化习惯，能把项目做成别人可接手的状态。
- `强度`
  - `A`
- `详细展开`
  - 这一点很适合在面试里作为协作能力证据。大型实验项目最大的风险之一是只有本人知道现状。这里已经沉淀出 handoff、runbook、asset inventory、claim map 等多层文档，不依赖口头记忆。
- `Doc`
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
  - [project_asset_inventory_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/project_asset_inventory_2026-04-18.md)
  - [resume_project_writing_plan_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_project_writing_plan_2026-04-18.md)

---

## 14. 早期探索线与历史资产

### Claim 14.1

- `Claim`
  - 在 `step900 / step1300` 之前，项目已经做过一轮较完整的 early teacher / repair-SFT 探索线，涵盖 `step200` fullsuite 和 `step400` repair-SFT v1。
- `强度`
  - `B`
- `详细展开`
  - 这条 claim 的价值在于说明项目不是“一条直线走到今天”。前期已经探索过 teacher mix、repair val、mini regression、step400 anchor 和 student reference strict eval。即使后面这些线不是最终主线，它们也为后续 `step900` / `step1300` 的数据构建与 recipe 选择提供了经验。
- `Doc`
  - [step400_repair_sft_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step400_repair_sft_plan.md)
  - [step400_repair_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/v2b_step400/step400_repair_plan.md)
  - [step400_sft_v1_postmortem.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/v2b_step400/step400_sft_v1_postmortem.md)
  - [phase2_sft_runbook.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/phase2_sft_runbook.md)
- `Key Result Files`
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase2_sft_eval_suite_v1/phase2_sft_pre_sft_step200_fullsuite_v1_retry1/repair_val/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase2_sft_eval_suite_v1/phase2_sft_step20_fullsuite_v1_retry1/repair_val/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/phase2_repair_sft_step400_r3a2_clean_v1_step10_codecontests_validbig500/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/phase2_repair_sft_step400_r3a2_clean_v1_step20_codecontests_validbig500/summary.json)

### Claim 14.2

- `Claim`
  - 我不仅做过 early SFT，还把“这条线为什么没有成为最终主线”复盘清楚了。
- `强度`
  - `B`
- `详细展开`
  - 这条 claim 很适合展示研究判断：有些项目会把失败线直接忘掉，但这里保留了完整 postmortem，说明我能把“没有赢 baseline”的实验也转化成方法经验，而不是简单忽略。
- `Doc`
  - [step400_sft_v1_postmortem.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/v2b_step400/step400_sft_v1_postmortem.md)
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)

---

## 15. 当前最适合后续简历与项目问答使用的素材池

如果后面要快速挑素材，优先从下面这些 claim 出发：

- `Claim 4.1`
  - 完整实验链路范围
- `Claim 5.1`
  - 多 checkpoint RL 主线与系统化对比
- `Claim 5.3`
  - `step1300_rl` 相比 base model 的 held-out raw generalization 提升
- `Claim 7.1`
  - `Protocol A / Protocol B` 双协议
- `Claim 8.2`
  - testcase 选择规则审计与 deterministic 修复
- `Claim 8.4`
  - one-turn repair 在 delta/dev/test 三层集合上的结果与边界
- `Claim 9.1`
  - 高精度 repair-conditioned SFT 数据构建链
- `Claim 10.2`
  - `step1300`-base repair-SFT v1 的真实结论：dev gain 存在，但 held-out 未替代 baseline
- `Claim 11.1`
  - SandboxFusion nondeterminism 修复
- `Claim 12.1`
  - quarantine 数据治理体系

这些点组合起来，已经足够支撑三种不同方向的项目叙事：

- 偏 `Applied Research / Post-training`
- 偏 `ML Systems / Evaluation Infra`
- 偏 `Data Quality / Experiment Reliability`

后续如果继续补：

- `step1300 repair-SFT v2`
- 更大规模 teacher/QC 资产
- 更新的 held-out repair 结果

---

## 16. 面试/简历引用速查表

这一节是给 Roger 写简历、做项目答辩时用的"一页纸"。每一行对应一个**常被问到的点**，点进去能直接定位权威口径、关键数字、证据文件与容易翻车的地方。其他几节是素材深度，这一节是入口索引。

### 16.1 RL raw 能力提升（held-out generalization）

- **权威口径**：[experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md) §0.1
- **关键数字**：held-out `CodeContests test` 165 题 raw `accepted@1`：`1.82%` → `7.88%`（4.33x，+10 绝对 solve）；`pass_ratio_mean`：`0.129` → `0.250`；`HumanEval +1.83 pp`、`MBPP +4.5 pp`
- **证据文件**：[base summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/qwen25coder7b_instruct_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)、[step1300 summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/step1300_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)
- **容易答崩**：把 dev 集 `13.8%` 说成 held-out `7.9%`；把"CodeContests test"和"CodeContests valid"混为一谈

### 16.2 Protocol A（fixed-input repair，纯修复能力）

- **权威口径**：[repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)、[repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)
- **关键数字**：held-out 上 Protocol A 几乎没有 solve gain（"纯修复能力"在 held-out 还没有足够提升）；dev 上 step40（旧线）表现较好但不是当前主线
- **证据文件**：[step1300 Protocol A on test summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair_protocolA/step1300_codecontests_test_repair_protocolA_shortdiag_p02_reuse_step900_rr808x_vastai5/summary.json)
- **容易答崩**：把 Protocol A 结果说成"端到端部署效果"（那是 Protocol B）；忘记说 A 是 fixed-input，消掉了 first-pass 差异

### 16.3 Protocol B（end-to-end self-repair，部署口径）

- **权威口径**：[repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)
- **关键数字**：held-out test 上 step1300_rl raw `7.88%` → after-repair `10.91%`（+5 solve，`repair_attempt_count=150, repair_success_count=5, conditional=3.33%`）
- **证据文件**：[step1300 test repair summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair/step1300_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/summary.json)、同目录 `repair_summary.json`
- **容易答崩**：把 `10.9%` 说成"纯修复能力大幅提升"；把 conditional 3.33% 说成全量 repair 成功率

### 16.4 Checkpoint 双 winner（late-stage RL 诊断）

- **权威口径**：[validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)、[checkpoint_selection_and_valid_big_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/checkpoint_selection_and_valid_big_plan.md)
- **关键数字**：`valid_big500` 上 step900 `accepted@1=12.8%` / step1000 `=13.8%`（solve winner）/ step1300 `=12.2%` 但 `pass_ratio_mean=35.76%`（partial winner）
- **证据文件**：3 份 `valid_big500` summary（见 §2.3）
- **容易答崩**：只说一个"最佳 checkpoint"，不区分 exact-solve vs partial-credit winner vs held-out deployed base 三种角色

### 16.5 Delta69 near-miss repair（focused slice，面试高危区）

- **权威口径**：[repair_phase4_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_phase4_design.md)
- **关键数字**：`valid_big500` 的 delta69 高 partial bucket 19 题 one-turn repair 成功 5 题，`conditional_repair_success = 26.32%`；first-pass `65.2%` → after-repair `72.5%`
- **证据文件**：[delta69 repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_delta69_repair/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_delta69_repair_raw/repair_summary.json)
- **容易答崩**：把 `26.3%` 说成全量 repair 成功率；忘记说它是 high-partial near-miss 的**子集条件成功率**

### 16.6 Repair-SFT 结论（当前主线：有效但边界清晰）

- **权威口径**：[experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)、[step1300_repair_conditioned_sft_v2_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_repair_conditioned_sft_v2_plan.md)
- **关键数字**：`valid_big500` 上 step1300_sft_v1_step60 repair_success_count 从 5 提升到 9（conditional ≈ 4.5%）；held-out test 上反而从 5 降到 3 → 因此标记 "v2 需换 recipe"
- **证据文件**：[valid_big500 protocolB sft_v1 summary](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_repair_protocolB_step1300sftv1/step1300sftv1_step60_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5/repair_summary.json)、[test protocolB sft_v1 summary](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair_step1300sftv1/step1300sftv1_step60_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/repair_summary.json)
- **容易答崩**：把 dev gain 包装成"repair-SFT 全面超过 RL baseline"；混淆 `step40`（旧 step900-base 线）与 `step1300_sft_v1_step60`（当前主线）

### 16.7 Teacher / QC 数据治理（高精度 SFT 数据）

- **权威口径**：[step900_repair_conditioned_sft_data_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_repair_conditioned_sft_data_plan.md)、[step1300_repair_conditioned_sft_v2_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_repair_conditioned_sft_v2_plan.md)
- **关键数字**：step900 stable keep `267` rows → pure shortdiag `182` rows（全部 `accept_2_of_2`）；step1300 expanded keep `553` rows，来源细分：primary_qc 389 + regen_round1 96 + regen_round2 12 + near_miss_audit 24 + remaining_high_pass_audit 32
- **证据文件**：[step900 keep set summary](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/final_keep/teacher_qc_v2_full.stable_keep_set.summary.json)、[step1300 keep set summary](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.summary.json)
- **容易答崩**：被问"为什么相信 teacher 数据"时答不出具体 QC 步骤；把 step900 和 step1300 的 keep 数字混用

### 16.8 Sandbox 稳定性（系统治理）

- **权威口径**：[sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_judge_repair.md)、[sandbox_concurrency_standard.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_concurrency_standard.md)
- **关键数字**：fixed-response rejudge 量化 judge 残余漂移 ≤ `0.6 pp accepted@1`；`valid_big500` rerun1 vs rerun2 accepted@1 差 `0.122 vs 0.116`
- **证据文件**：两个 [rejudge summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_fixed_rejudge)、`SandboxFusion/sandbox/runners/base.py`、`execution.py` 修补
- **容易答崩**：说成"彻底解决 judge 不稳定"（应改为"定位主要来源并量化剩余上限"）

### 16.9 使用方式

- 写简历时：每条 bullet 都能从这张表找到对应行，不用再翻长文档
- 自述项目时：按 16.1 → 16.3 → 16.4 → 16.6 的顺序讲 4 个高点，足够 5 分钟答辩
- 面试追问时：把"容易答崩"那列提前背熟，被追问时主动承认边界，反而显得靠谱

这份 map 可以继续增量更新，而不需要重写整个项目叙事。
