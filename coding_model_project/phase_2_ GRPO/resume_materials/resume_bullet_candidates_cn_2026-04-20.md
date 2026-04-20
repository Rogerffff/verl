# 中文简历 Bullet 候选集（2026-04-20）

## 0. 用途

这份文档不是最终简历成稿，而是一个尽量丰富的中文 bullet 素材库，目标是：

1. 尽量多地覆盖当前项目里值得写到简历上的点
2. 先把不同方向的表述都摊开，不急着收敛到最后 2 到 4 条
3. 尽量突出项目优势、方法完整性和你的 ownership
4. 避免在 bullet 层直接暴露不必要的内部实验细节，例如内部 checkpoint 命名、临时分支名、远端机器名等

配套证据地图见：

- [resume_claim_evidence_map_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md)

---

## 1. 阅读方式

### 1.1 字段说明

- `ID`
  - 方便后续筛选、讨论和面试问答引用。命名规则：`H` = 总括、`R` = 结果、`J` = 研究判断、`D` = 数据治理、`S` = 系统/Infra、`O` = 运维/平台、`M` = 模型 × 数据方法论
- `方向`
  - 这条 bullet 更偏结果、系统、数据还是研究判断
- `包装度`
  - `稳健`：更接近保守真实表述，适合 HR 初筛 / 岗位 JD 不明确时兜底
  - `平衡`：比较适合直接上简历，既有数字又不过度承诺
  - `进攻`：更有冲击力，但要求后续问答更稳；只在岗位 JD 明确吃这口风格时才用
- `Bullet`
  - 可直接进一步改写的候选句子
- `证据锚点`
  - 对应的 claim map、关键结果或代码资产
- `展开解释（不上简历，仅供自述用）`（**本次新增**）
  - 针对核心 bullet 提供的"一段 3-5 句的自述脚本"，用于面试自我介绍或答辩开场；不会印到简历上，但 Roger 读完就能直接用

### 1.2 使用建议

简历最终一般只能放 2-4 条项目 bullet。按照岗位 JD 的偏向，推荐组合：

- **Applied Research / LLM Post-training**
  - 最小组合（2 条）：`H01 + R01`
  - 完整组合（3 条）：`H01 + R01 + J02`（加一条研究判断）
  - 进阶组合（4 条）：`H01 + R01 + R04 + J07`

- **ML Systems / Infra / Evaluation**
  - 最小组合：`H01 + S04`
  - 完整组合：`H01 + S04 + S02`（加 reuse-first-pass）
  - 进阶组合：`H01 + S04 + S02 + D01`（加 quarantine 数据治理）

- **平台 + 模型复合型**
  - 最小组合：`H01 + R04 + D03`
  - 完整组合：`H01 + R01 + R04 + D03`
  - 这种岗位更喜欢看到 Roger 既有模型结果、又有 repair 的端到端落地、又有数据治理

这组推荐与下面 §9「最值得优先筛选」保持一致。两处推荐的 bullet 必须同步——如果对 bullet 的判断有变动，两处都要改。

### 1.3 术语小字典（跨文件共享）

这份字典在 [resume_claim_evidence_map_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md) §1.4 里有一份完全相同的内容，保证两个文件对同一术语措辞一致。

- **Shared Verifier**：评测（`phase0_eval.py`）与 RL reward（`grpo_batch_reward.py`）共用的判题模块 `src/verifier/shared.py`，对每题跑全部 testcase，落到统一的 `accepted` / `pass_ratio_all` 字段
- **Protocol A（fixed-input repair）**：对同一套冻结的错代码输入做 one-turn repair，度量**纯修复能力**
- **Protocol B（end-to-end self-repair）**：每个模型先自己生成 first-pass，再对自己的错代码做 one-turn repair，度量**端到端部署口径**
- **Strongest deployed base**：在 held-out test 上 raw 能力 + repair gain 综合最高的 checkpoint，当前是 `step1300_rl`
- **两类 checkpoint winner**：`step1000` = solve-count winner（dev accepted@1 最高 13.8%）；`step1300` = partial-credit winner（pass_ratio_mean 最高 35.76%）
- **step 命名约定**：单独 `stepXXX` 一般是 **RL 主线**；`stepXXX_sft_v1_stepYY` 是 repair-SFT 线；`step40` 是**历史** step900-base 旧 SFT 线，`step1300_sft_v1_step60` 是**当前**主线 v1

---

## 2. 项目总括候选

### H01

- `方向`
  - 总括 / 端到端
- `包装度`
  - `平衡`
- `Bullet`
  - 围绕算法代码生成任务搭建了一条 `课程式 RL → 共享判题器评测 → verifier-guided one-turn repair → repair-conditioned SFT` 的完整后训练链路；链路同时覆盖训练、评测协议、数据治理与远端实验运维，各环节共用同一套判题契约而不是彼此独立。
- `证据锚点`
  - [resume_claim_evidence_map_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md)
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
- `展开解释（不上简历，仅供自述用）`
  - 这个项目在 Qwen2.5-Coder-7B-Instruct 上做 post-training，目标是让模型在竞赛编程题（CodeContests）上能真正稳定解题。
  - 单做 RL 会遇到 late-stage 饱和——pass_ratio 还在涨，但 exact solve 不涨。所以我往后加了 one-turn repair（让模型看测试反馈改自己的代码），再把 repair 成功的高精度样本做 repair-conditioned SFT。
  - 四个阶段不是各做各的，都走同一套 `src/verifier/shared.py` shared verifier 判题，保证 RL reward、evaluation、teacher QC 用同一口径，避免 reward hacking 和评测漂移。
  - 我对每个阶段的接口、数据契约、评测协议都自己设计过，不是简单包了 verl。
- `不要用于`
  - HR 初筛且岗位 JD 没有明确强调 RL/后训练时——这条 bullet 在 30 秒内不好讲清楚"四阶段链路"是什么。可以退回 H03 或 R01。

### H02

- `方向`
  - 总括 / 研究系统一体化
- `包装度`
  - `进攻`
- `Bullet`
  - 主导搭建算法代码生成的后训练实验平台，把 curriculum RL、repair 评测、teacher-QC、quarantine 和 sandbox 稳定性治理整合成一条可复现研究流水线。
- `证据锚点`
  - [resume_claim_evidence_map_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md)
  - [project_asset_inventory_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/project_asset_inventory_2026-04-18.md)
- `不要用于`
  - HR 初筛 / 非技术面试：用 "主导搭建...平台" 这种表述时，对方可能会追问团队规模、协作模式等问题；作为独立项目要答清"主导"的范围（个人项目 vs 团队项目）

### H03

- `方向`
  - 总括 / ownership
- `包装度`
  - `平衡`
- `Bullet`
  - 独立推进一个面向 CodeContests 的 LLM post-training 项目，不仅负责模型训练，还负责评测协议设计、sandbox 修复、数据治理、teacher 生成与 SFT 数据落地。
- `不要用于`
  - 岗位更偏纯 Research Scientist / 理论方向时——这条 bullet 过于强调"ownership 覆盖面"，对方可能期待看到方法创新。可以退回 H01 用"搭建了一条完整后训练链路"这种更聚焦的总括。
- `证据锚点`
  - [resume_claim_evidence_map_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md)

### H04

- `方向`
  - 总括 / 面向系统岗
- `包装度`
  - `平衡`
- `Bullet`
  - 在一个高噪声、强依赖判题器的代码生成项目中，同时负责模型、评测系统、数据管线和远端 GPU 运行编排，把实验从“能跑”推进到“可解释、可复现、可交接”。
- `证据锚点`
  - [resume_claim_evidence_map_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md)

---

## 3. 结果导向 Bullet 候选

### R01

- `方向`
  - RL 结果
- `包装度`
  - `平衡`
- `Bullet`
  - 将 held-out `CodeContests test` 的原始 `accepted@1` 从 `1.8%` 提升到 `7.9%`，同时保持 `HumanEval 89.0%`、`MBPP 62.5%` 的回归集表现。
- `证据锚点`
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/qwen25coder7b_instruct_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/step1300_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)
- `展开解释（不上简历，仅供自述用）`
  - base 模型是 `Qwen2.5-Coder-7B-Instruct`，`7.9%` 对应训练到 1300 step 的 RL checkpoint（`step1300_rl`）。
  - **Held-out**：`CodeContests test` 165 题是完全在训练过程中不可见的集合；绝对数从 base 的 `3/165` 提升到 `13/165`（多 10 道 exact solve，倍率 ≈ 4.3x）。
  - 同时 `pass_ratio_mean` 从 `0.129` 提升到 `0.250`（几乎翻倍），说明不只是"多碰对几题"而是整体部分分分布在上移——对 repair 构成可利用的 near-miss 池。
  - `HumanEval` 从 `87.2%` → `89.0%`（+1.83 pp），`MBPP` 从 `58.0%` → `62.5%`（+4.5 pp），保住了通用代码生成的 guardrail，没把模型训成 codecontests narrow specialist。
  - 算法用的是 DAPO 增强的 GRPO（A1 variant：asymmetric clip 0.2/0.28、token-mean、no KL），奖励是 anchored_dense_v1，硬件 4× RTX 5090。
- `不要用于`
  - 简历里只允许写 1 条项目 bullet 时——这条数字很亮但需要 30 秒展开上下文；这种场合可以换 R02（4.3x 倍率包装更短）或 H03（总括 ownership 优先）。

### R02

- `方向`
  - RL 结果
- `包装度`
  - `进攻`
- `Bullet`
  - 在 competitive-programming 代码生成任务上，把一个通用 7B coder base 推进成可部署的 RL checkpoint，使 held-out `CodeContests test` 原始通过率提升约 `4.3x`。
- `证据锚点`
  - 同 R01

### R03

- `方向`
  - checkpoint 选择
- `包装度`
  - `平衡`
- `Bullet`
  - 在 `500` 题 CodeContests 开发基准上完成多 checkpoint 系统对比，最佳 exact-solve checkpoint 达到 `69/500 solved`、`13.8% accepted@1`。
- `证据锚点`
  - [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_codecontests_validbig500_raw_patched_rr828x_vastai3/summary.json)

### R04

- `方向`
  - repair 结果
- `包装度`
  - `平衡`
- `Bullet`
  - 构建 verifier-guided 单轮修复链路，在 held-out `CodeContests test` 的 end-to-end 自修复协议下，将 `accepted@1` 从 `7.9%` 提升到 `10.9%`。
- `证据锚点`
  - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_test_repair/step1300_codecontests_test_repair_reuse_selfraw_rr808x_vastai5/repair_summary.json)
- `展开解释（不上简历，仅供自述用）`
  - `10.9%` 来自 **Protocol B**（end-to-end self-repair）：step1300_rl 先自己生成 first-pass，再对自己的错代码做 one-turn repair，度量**端到端部署口径**，而不是纯修复能力。
  - 具体数字：`repair_attempt_count = 150`（对 first-pass 失败的 150 题触发 repair），其中 `repair_success_count = 5`（新增 5 道 solve），`conditional_repair_success = 3.33%`。
  - 注意这个 3.33% **不同于** R05 的 26.3%——R05 是在 delta69 high-partial near-miss 子集上的**条件**成功率，只在高价值样本上触发；R04 是 held-out 全量失败题。两者的分母不同，不要混淆。
  - 让这条 bullet 更稳的讲法：这是"部署口径下的真实修复 gain"，比纯 repair skill 评测更接近实际部署价值，但也更受 trigger 触发策略（什么时候触发 repair）的影响。
- `不要用于`
  - 简历只想放"一个代码能力提升"数字时——这条必须和 R01 一起说，否则读者会困惑"10.9% 是修复能力还是原始能力"。

### R05

- `方向`
  - repair 结果
- `包装度`
  - `平衡`
- `Bullet`
  - 在高价值 near-miss 子集上实现 `26.3%` 的条件修复成功率，验证了 verifier-guided one-turn repair 在局部逻辑错误场景下的有效性。
- `证据锚点`
  - [repair_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_delta69_repair/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_delta69_repair_raw/repair_summary.json)
- `展开解释（不上简历，仅供自述用）`
  - **这个 26.3% 是子集条件成功率，不是全量 repair 成功率**——一定要主动说清楚，否则容易被追问。
  - 具体范围：`valid_big500` 开发集里 step900→step1000 的 delta69 "high-partial near-miss" 子集，即 pass_ratio 很高但 accepted=False 的题（离通过只差一点点）。共 19 题，one-turn repair 后成功 5 题，所以 `conditional = 5/19 = 26.3%`。
  - first-pass `accepted@1 = 65.22%`，after-repair = `72.46%`（repair gain +7.25 pp）。
  - 这条数字说明 repair 不是对所有失败题均匀有效，而是**更像 near-miss fallback**——局部逻辑错、off-by-one、边界条件这类错误修得好；完全不会的题修不起来。这个判断直接指导了后续 repair trigger 策略和 repair-SFT 数据切片。
- `不要用于`
  - 对方只有 15 秒时——容易被截断成"26.3% 修复率"然后以为是全局数字。可以先讲 R04 让全局数字扎根，再用 R05 作为"在高价值子集上更高"的补充。

### R06

- `方向`
  - repair-SFT 结果
- `包装度`
  - `平衡`
- `Bullet`
  - 基于 `553` 条高精度 repair-conditioned 数据训练 short-diagnosis SFT，在开发集 `valid_big500` Protocol B 上将 repair 成功数从 `5` 提升到 `9`（conditional ≈ 4.5%），并据 held-out 未跟上的证据推导出下一版更平衡的 v2 数据配方。
- `证据锚点`
  - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)
  - [step1300_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.summary.json)
  - [step1300_repair_conditioned_sft_v2_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_repair_conditioned_sft_v2_plan.md)
- `展开解释（不上简历，仅供自述用）`
  - Base 是 `step1300_rl`，数据是 553 条 short-diagnosis + code 格式的 repair-conditioned SFT（经过 primary QC + 2 轮 regen + near-miss testcase audit + remaining high-pass audit 5 阶段治理）。
  - Dev 侧结果：`valid_big500` Protocol B `repair_success_count` 从 step1300_rl 的 5 提升到 step1300_sft_v1_step60 的 9；`conditional_repair_success` 约 4.5%。
  - **Held-out 侧结果**：`codecontests_test` 反而从 5 降到 3，after-repair accepted@1 从 10.91% 跌到 9.70%。所以当前的合适表述是"**repair-SFT v1 dev gain 真，held-out 尚未外推**"。
  - 这个 gap 我的判断不是"repair-SFT 失败"，而是 v1 数据的 strata 分布偏重 CurrentCRecoverable（316/553），学成了 harder-repair-specialized shaper；下一版 v2 plan 要 rebalance strata，这个已经写进 v2 plan 文档（尚未执行）。
  - 这种"dev gain 真但 held-out 不跟上"的结论比硬吹 dev 成功更能显示研究判断的成熟度。
- `不要用于`
  - 面试只有时间说 1-2 条结果数字时——这条边界很多，容易被追问到岔路。优先用 R01/R04。

### R07

- `方向`
  - 结果 + 研究判断
- `包装度`
  - `进攻`
- `Bullet`
  - 不仅把 raw 生成能力做上去，还进一步把“生成后自修复”做成可量化的第二增益来源，形成 `raw solve + repair gain` 的双层评估框架。
- `证据锚点`
  - [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)
  - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)
- `展开解释（不上简历，仅供自述用）`
  - "双层评估框架"的具体含义：
    - **第一层 raw**：first-pass `accepted@1` 与 `pass_ratio_mean`，度量模型直接生成解的能力（R01 的 7.9%）。
    - **第二层 repair**：在 raw 之上再叠一层 one-turn repair gain，按 **Protocol A（纯修复能力）**与 **Protocol B（部署口径）**分别量化。
  - 为什么要拆：如果只看 after-repair accepted@1，无法区分"模型本来 first-pass 就强"vs"模型修复能力强"，这对下游 repair-SFT 投入方向是两种完全不同的判断。
  - 实际数据：step1300_rl 第一层 `7.88%`，第二层 Protocol B 叠加后 `10.91%`（gain +3.03 pp），Protocol A 几乎持平（"纯修复能力"目前还没突破）。
  - 这套框架的价值：在下游选 repair base 和决定 SFT 资源投入时，第一层强但第二层弱的 checkpoint vs 第二层强但第一层弱的 checkpoint，选择逻辑完全不同。
- `不要用于`
  - 被问到"双层框架有没有显著改善 held-out"时——诚实答案是"第一层有显著改善，第二层 gain 可量化但幅度较小"。如果对方追 headline，这条包装度偏 `进攻`，可能被过度解读。

### R08

- `方向`
  - 结果 + checkpoint analysis
- `包装度`
  - `平衡`
- `Bullet`
  - 用 exact-solve 与 partial-credit 双指标做 checkpoint 选择，识别出“最佳 solve checkpoint”和“最佳 partial checkpoint”并不总是同一个模型。
- `证据锚点`
  - [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)

### R09

- `方向`
  - generalization
- `包装度`
  - `平衡`
- `Bullet`
  - 在提升 CodeContests 原始通过率的同时保住了通用代码回归集表现，而不是把模型训成只会特定数据分布的窄专长模型。
- `证据锚点`
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/qwen25coder7b_instruct_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_testset_regression/step1300_codecontests_test_humaneval_mbpp_raw_rr808x_vastai5/summary.json)

### R10

- `方向`
  - baseline establishing
- `包装度`
  - `稳健`
- `Bullet`
  - 在项目初期完成 full-val baseline，建立 `HumanEval / MBPP / CodeContests` 三类任务的统一起点，为后续 RL、repair 和 SFT 提供可比较基座。
- `证据锚点`
  - [eval_analysis.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/phase0_fullval_20260331/eval_analysis.md)

---

## 4. 系统 / Infra / 评测治理 Bullet 候选

### S01

- `方向`
  - 评测协议
- `包装度`
  - `平衡`
- `Bullet`
  - 设计并落地了 fixed-input 与 end-to-end self-repair 两套 repair 评测协议，把“模型修复能力”和“真实部署行为”明确拆开。
- `证据锚点`
  - [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)
  - [repair_eval_protocol_matrix.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_protocol_matrix.md)

### S02

- `方向`
  - 评测协议
- `包装度`
  - `进攻`
- `Bullet`
  - 实现 `reuse-first-pass` 机制，消除了 repair 对比中 first-pass 生成漂移的主要干扰项，让不同模型能够在同一批冻结错误程序上做公平比较。
- `证据锚点`
  - [phase4_repair_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py)
  - [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)
- `展开解释（不上简历，仅供自述用）`
  - 背景：repair 评测同时包含两个可变项——first-pass 生成和 second-pass repair。如果每次现跑 first-pass，生成侧的 1-2 pp 抖动会让 repair 差异无法区分是"真的 repair 变好"还是"first-pass 更多样"。
  - 具体做法：把每个 `(checkpoint, problem)` 的 first-pass response 存成 JSONL artifact，后续 repair eval 直接消费这份冻结 artifact 做 diagnosis 和 second-pass，不允许重新采样。
  - 校验机制：repair eval 启动时对 `(model_id, problem_id)` 做一一对应检查，缺任意一题直接 **hard-fail**；被 `max_tokens` 截断的 first-pass 单独打 `truncated_source` 标签，不进 repair_attempt_count 分母（否则会把 trigger 设计的锅甩给 repair 能力）。
  - 这一机制是 Protocol A 能成立的基础——它把"公平比较 repair 能力"从口头约定变成文件级合约。
- `不要用于`
  - 面试对方更关心高层次结果时——这条偏 infra 细节，放在 Applied Research 岗简历上可能被认为过于技术细节。可以给 ML Systems/Infra 岗用。

### S03

- `方向`
  - judge reliability
- `包装度`
  - `平衡`
- `Bullet`
  - 通过 fixed-response rejudge 把 judge 漂移和生成漂移拆开量化，避免把评测不稳定误判成模型能力波动。
- `证据锚点`
  - [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_fixed_rejudge/step1300_rejudge1/summary.json)
  - [summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/outputs/rl_codecontests_validbig500_fixed_rejudge/step1300_rejudge2/summary.json)

### S04

- `方向`
  - sandbox debugging
- `包装度`
  - `平衡`
- `Bullet`
  - 定位并修复 SandboxFusion 在高并发下的输出截断 / 非确定性主问题，把正式评测收敛到 patched sandbox + direct-backend round-robin。
- `证据锚点`
  - [sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_judge_repair.md)
  - [base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py)
  - [execution.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/execution.py)

### S05

- `方向`
  - shared infra
- `包装度`
  - `平衡`
- `Bullet`
  - 统一 raw eval、repair eval 和 teacher-QC 的 judge 契约，减少不同实验链各自使用不同评测假设造成的口径漂移。
- `证据锚点`
  - [shared_verifier_infra_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/shared_verifier_infra_guide.md)
  - [verifier/shared.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py)

### S06

- `方向`
  - failure summarization
- `包装度`
  - `平衡`
- `Bullet`
  - 替换了 naive 的“第一条失败样例”规则，改用按错误类型分流的 deterministic failure summarization，使 repair prompt 更接近最小反例和清晰异常信号。
- `证据锚点`
  - [testcase_selection_rule_audit_v1.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_analysis/testcase_selection_rule_audit_v1.md)
  - [testcase_selection_rule_recommendation_v1.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_analysis/testcase_selection_rule_recommendation_v1.json)
  - [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)
- `展开解释（不上简历，仅供自述用）`
  - 原规则：repair prompt 里给模型看的 diagnosis 就是 `per_case_results` 里第一条 fail。
  - 审计发现两个问题：(a) 前几条 case 类型可能是 syntax_error 或 timeout，信息噪声大，淹没 WA 的反例信号；(b) `per_case_results` 字段里有明显截断现象，第一条 fail 可能是损坏的大 case。
  - 新规则按错误类型分流：
    - `WA (wrong_answer)` → 取**最小反例**（输入长度最短的 fail，便于模型对照）
    - `RE (runtime_error)` → 取**最清晰的异常栈**（带完整 traceback 的）
    - `TLE (timeout)` → 取**最短输入**（说明即使最短输入都超时，就是算法复杂度问题）
  - 同时在 prompt 构造阶段主动过滤掉被截断的 case，避免把脏数据喂给模型。
  - 规则写进 `repair_feedback.py`，audit 证据见 `testcase_selection_rule_audit_v1.md`（逐样本审计记录）和 `testcase_selection_instance_audit_v1.jsonl`（原始 labels）。
- `不要用于`
  - 把它讲成"解决了所有 prompt 质量问题"——这只是规则层的改进，对"模型 repair 能力边界"本身没有直接影响。

### S07

- `方向`
  - runtime ops
- `包装度`
  - `平衡`
- `Bullet`
  - 在多台临时租用 GPU 机器之间迁移完整实验栈，覆盖代码仓、sandbox 池、teacher QC、repair eval 和训练恢复，保证长周期实验可持续推进。
- `证据锚点`
  - [new_machine_teacher_qc_runbook_2026-04-15.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/new_machine_teacher_qc_runbook_2026-04-15.md)
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)

### S08

- `方向`
  - runtime ops
- `包装度`
  - `进攻`
- `Bullet`
  - 为长跑训练和续跑 handoff 设计 watcher、launch guard 和自动恢复逻辑，降低人工等待和 GPU 时间浪费。
- `证据锚点`
  - [wait_launch_grpo_a1_resume1200_to1400_overnight_vastai3.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/wait_launch_grpo_a1_resume1200_to1400_overnight_vastai3.sh)

### S09

- `方向`
  - documentation / collaboration
- `包装度`
  - `平衡`
- `Bullet`
  - 为复杂实验链持续维护 handoff、runbook、asset inventory 和 protocol 文档，让项目状态可以被他人直接接手，而不是依赖个人记忆。
- `证据锚点`
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
  - [project_asset_inventory_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/project_asset_inventory_2026-04-18.md)
  - [resume_claim_evidence_map_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md)

### S10

- `方向`
  - pipeline engineering
- `包装度`
  - `平衡`
- `Bullet`
  - 将 teacher generation 做成带 manifest、shard、schema、reject ledger 和 regen queue 的可审计流水线，而不是一次性离线生成脚本。
- `证据锚点`
  - [step900_teacher_generation_qc_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_teacher_generation_qc_plan.md)
  - [step900_teacher_prompt_and_schema_v2_backend_rr.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_teacher_prompt_and_schema_v2_backend_rr.md)

---

## 5. 数据治理 / 数据工程 / QC Bullet 候选

### D01

- `方向`
  - 数据治理
- `包装度`
  - `平衡`
- `Bullet`
  - 建立了版本化 quarantine 体系，系统隔离污染题、判题不稳题和高风险问题，避免脏数据渗入训练与评测结论。
- `证据锚点`
  - [problem_quarantine_v3_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/problem_quarantine_v3_guide.md)
  - [problem_quarantine_v2_build_summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/quarantine_audit/review_assets_v1/problem_quarantine_v2_build_summary.json)

### D02

- `方向`
  - 数据治理
- `包装度`
  - `进攻`
- `Bullet`
  - 版本化管理 `777+` 道 hard blacklist 问题，并将 quarantine 规则回接到 curriculum 和 teacher-data 构建链，提升训练数据和评测结论的可信度。
- `证据锚点`
  - 同 D01
  - [filter_curriculum_manifest_by_quarantine.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/filter_curriculum_manifest_by_quarantine.py)

### D03

- `方向`
  - repair-conditioned SFT data
- `包装度`
  - `平衡`
- `Bullet`
  - 构建了 request-unique、高精度的 repair-conditioned SFT 数据集，使用 teacher-QC、rejudge、regen 和 testcase audit 多轮过滤后再入训练。
- `证据锚点`
  - [teacher_qc_v2_full.stable_keep_set.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/final_keep/teacher_qc_v2_full.stable_keep_set.summary.json)
  - [step1300_teacher_keep_set_v2.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.summary.json)

### D04

- `方向`
  - repair-conditioned SFT data
- `包装度`
  - `平衡`
- `Bullet`
  - 沉淀出两版可直接训练的 repair-SFT 数据：一版 `182` 条高精度 pure short-diagnosis 数据，一版 `553` 条带多轮审计与补救的扩大版数据。
- `证据锚点`
  - [step900_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.summary.json)
  - [step1300_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.summary.json)

### D05

- `方向`
  - teacher QC
- `包装度`
  - `平衡`
- `Bullet`
  - 把 teacher 失败样本拆成 teacher bug、testcase issue、judge instability 和边界可救回样本，而不是简单把所有 reject 一刀切丢弃。
- `证据锚点`
  - [step1300_teacher_keep_set_v2.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.summary.json)
  - [near_miss_testcase_audit](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/near_miss_testcase_audit)

### D06

- `方向`
  - dataset slicing
- `包装度`
  - `平衡`
- `Bullet`
  - 在 repair 数据构建中保留 `Core / Expansion / C` 等 strata 标签，并将其贯穿到 teacher 请求配比、QC 分析和最终训练集构成。
- `证据锚点`
  - [step900_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.summary.json)
  - [step1300_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.summary.json)

### D07

- `方向`
  - curation rigor
- `包装度`
  - `进攻`
- `Bullet`
  - 不是把 teacher 当成默认真值，而是把 teacher generation 变成一条可审计的数据生产线：请求切片、分 shard 生成、primary QC、regen、testcase 审计、最终 keep set 汇总全部留痕。
- `证据锚点`
  - [step900_teacher_generation_qc_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_teacher_generation_qc_plan.md)
  - [step1300_repair](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair)

### D08

- `方向`
  - curriculum + data linkage
- `包装度`
  - `平衡`
- `Bullet`
  - 将 curriculum bucket、quarantine、teacher 候选和 repair-SFT 数据构建打通，使“训练分布、评测分布、teacher 数据分布”三者之间存在显式映射关系。
- `证据锚点`
  - [curriculum_rl_pilot_v8_explainer.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_rl_pilot_v8_explainer.md)
  - [step900_repair_conditioned_sft_data_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_repair_conditioned_sft_data_plan.md)
  - [step1300_repair_conditioned_sft_v2_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_repair_conditioned_sft_v2_plan.md)

---

## 6. 研究判断 / 实验设计 / 方法论 Bullet 候选

### J01

- `方向`
  - 研究判断
- `包装度`
  - `平衡`
- `Bullet`
  - 不把 repair 当成单一分数游戏，而是拆成 fixed-input repair skill 与 end-to-end self-repair behavior 两种问题分别评估。
- `证据锚点`
  - [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)

### J02

- `方向`
  - 研究判断
- `包装度`
  - `平衡`
- `Bullet`
  - 通过协议化评测得出“开发集 repair gain 不等于 held-out deploy gain”的结论，并据此阻止不成熟 recipe 被误当成成功路线继续放大。
- `证据锚点`
  - [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)
  - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)

### J03

- `方向`
  - 研究判断
- `包装度`
  - `进攻`
- `Bullet`
  - 把 noisy 的评测现象转化成可执行的实验决策：识别哪些是模型能力变化，哪些是 judge 漂移，哪些是 prompt/trigger 设计导致的表面波动。
- `证据锚点`
  - [sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_judge_repair.md)
  - [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)

### J04

- `方向`
  - 研究判断
- `包装度`
  - `平衡`
- `Bullet`
  - 用 exact-solve 和 partial-credit 双指标解释 late-stage RL，避免把“整体质量更好”和“真正做对更多题”混为一谈。
- `证据锚点`
  - [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)

### J05

- `方向`
  - 研究判断
- `包装度`
  - `平衡`
- `Bullet`
  - 识别出 one-turn repair 更像 near-miss fallback，而不是对所有失败题均匀有效，并据此把修复触发与后续 SFT 数据收集聚焦到高价值 failure strata。
- `证据锚点`
  - [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)
  - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)

### J06

- `方向`
  - 研究判断
- `包装度`
  - `平衡`
- `Bullet`
  - 不是只保留“成功实验”，而是对早期 `step200/step400` 的 teacher / repair-SFT 探索线做过完整复盘，并把失败经验反哺到后续主线。
- `证据锚点`
  - [step400_sft_v1_postmortem.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/v2b_step400/step400_sft_v1_postmortem.md)
  - [phase2_sft_runbook.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/phase2_sft_runbook.md)

### J07

- `方向`
  - 研究判断
- `包装度`
  - `进攻`
- `Bullet`
  - 将 repair-SFT v1 定位为“有效的 dev-side specialization probe”而不是盲目宣称成功，并基于 held-out 结果提出更平衡的 v2 数据配方。
- `证据锚点`
  - [step1300_repair_conditioned_sft_v2_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_repair_conditioned_sft_v2_plan.md)
  - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)

---

## 7. 数据与模型结合型 Bullet 候选

### M01

- `方向`
  - 模型 + 数据
- `包装度`
  - `平衡`
- `Bullet`
  - 以 curriculum bucket 为中轴，把 RL 训练、repair 触发、teacher 候选切片和 SFT 数据配比串成一套统一的数据闭环。
- `证据锚点`
  - [curriculum_rl_pilot_v8_explainer.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/curriculum_rl_pilot_v8_explainer.md)
  - [step900_repair_conditioned_sft_data_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_repair_conditioned_sft_data_plan.md)

### M02

- `方向`
  - 模型 + 数据
- `包装度`
  - `平衡`
- `Bullet`
  - 将 repair prompt、teacher 输出格式和最终 SFT 数据格式统一到 `short-diagnosis + code` 协议，降低 teacher-QC 和训练衔接成本。
- `证据锚点`
  - [repair_prompt_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_prompt_design.md)
  - [step900_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.summary.json)
  - [step1300_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.summary.json)

### M03

- `方向`
  - 模型 + 数据
- `包装度`
  - `进攻`
- `Bullet`
  - 把“看反馈修错代码”从一次性 prompt 实验升级成可训练的数据范式，完成从 verifier-guided repair 到 repair-conditioned SFT 的闭环验证。
- `证据锚点`
  - [repair_phase4_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_phase4_design.md)
  - [step900_shortdiag_pure_sft_training_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_shortdiag_pure_sft_training_guide.md)
  - [step1300_shortdiag_pure_sft_training_guide.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step1300_shortdiag_pure_sft_training_guide.md)

### M04

- `方向`
  - 模型 + 数据
- `包装度`
  - `平衡`
- `Bullet`
  - 在同一个项目中同时推进 raw RL 增强和 post-execution self-repair 两条能力线，并用协议化评测分析两者是互补还是冲突。
- `证据锚点`
  - [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)

---

## 8. 更偏系统岗 / 平台岗的候选

### O01

- `方向`
  - 系统
- `包装度`
  - `平衡`
- `Bullet`
  - 在高并发、多 backend、存在判题不稳定风险的环境里，把模型评测链路收敛成可复现、可追责、可复核的实验系统。
- `证据锚点`
  - [sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_judge_repair.md)
  - [repair_eval_current_contract_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_current_contract_2026-04-18.md)

### O02

- `方向`
  - 系统
- `包装度`
  - `进攻`
- `Bullet`
  - 把一个高噪声 LLM 实验环境中的“看起来像模型问题”的现象拆解为协议问题、judge 问题、数据问题和真实模型问题，显著提高实验结论的可信度。
- `证据锚点`
  - [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)
  - [sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_judge_repair.md)

### O03

- `方向`
  - 平台化
- `包装度`
  - `平衡`
- `Bullet`
  - 为复杂实验项目持续维护 protocol 文档、runbook、handoff 和资产清单，降低多人协作和跨机器迁移的接手成本。
- `证据锚点`
  - [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
  - [project_asset_inventory_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/project_asset_inventory_2026-04-18.md)

### O04

- `方向`
  - 平台化
- `包装度`
  - `平衡`
- `Bullet`
  - 把 teacher 生成、QC、regen、审计和 parquet 落地做成一条可持续扩展的数据生产链，而不是一次性的离线数据脚本。
- `证据锚点`
  - [step900_teacher_generation_qc_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_teacher_generation_qc_plan.md)
  - [step1300_repair](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair)

---

## 9. 当前最值得优先筛选的一组

如果下一步要快速收敛成一版较强的中文简历项目 bullet，最推荐先从下面 8 条里筛：

- `H01`、`R01`、`R03`、`R04`、`S02`、`S04`、`D03`、`J02`

这 8 条与 §1.2 的岗位推荐组合对齐（如果下一版 bullet 有增删，两处都要同步改，防止漂移）。

**按场景的快速选法**：

| 场景 | 最小 2 条 | 平衡 3 条 | 进阶 4 条 |
|------|----------|----------|----------|
| Applied Research / LLM Post-training | `H01 + R01` | `H01 + R01 + J02` | `H01 + R01 + R04 + J07` |
| ML Systems / Infra | `H01 + S04` | `H01 + S04 + S02` | `H01 + S04 + S02 + D01` |
| 平台 + 模型复合 | `H01 + R04 + D03`（3 条起步） | 同左 | `H01 + R01 + R04 + D03` |

这组的共同优点：

- 既有硬结果（R01/R04）又有协议与系统意识（S02/S04）
- 能体现数据治理（D03）和研究判断（J02）
- 不依赖暴露内部 checkpoint 命名或远端机器名

---

## 10. 当前不建议直接上简历的写法（附改写建议）

每条"不建议写法"后面配一条**保守版改写**，可以直接替换使用：

- ❌ "repair-SFT 已经全面超过主 RL 模型"
  - ✅ 改为："repair-conditioned SFT 在开发集 `valid_big500` 上稳定带来 repair gain（success_count 5 → 9），held-out test 上仍以 RL 主线为 strongest deployed base"
  - 为什么：dev gain 是真的，但 held-out 上 sft_v1 反而退步；不说清楚边界会被追问到崩

- ❌ "完全解决了 judge nondeterminism"
  - ✅ 改为："定位并修复 SandboxFusion 并发场景下的主 nondeterminism 来源，将 residual judge 漂移量化到 ≤ 0.6 pp accepted@1 的可测范围"
  - 为什么：完全 deterministic 是做不到的；但"量化剩余边界"是实打实的系统治理贡献

- ❌ "发明了新的 reasoning 格式"
  - ✅ 改为："设计 short-diagnosis + code 两段式 repair prompt，让 diagnosis 成为可审核、可训练的中间产物，贯穿 evaluation、teacher generation、SFT 训练三条链路"
  - 为什么：不要用"发明"这种强词；同一套 prompt 结构贯穿多条链路是更真实的贡献

- ❌ 任何带内部 step 编号、远端机器名、临时评测路径名的 bullet
  - ✅ 改为：内部命名只在"展开解释"或面试问答里用，简历 bullet 层统一改成 "当前 RL 主线 checkpoint" / "当前 repair-SFT v1" 这种角色描述
  - 为什么：内部 ID 会让简历读者困惑，还可能暴露不必要的实验细节

这些内容不是不能讲，而是更适合放到后续问答里解释，而不是直接出现在简历第一屏。

---

## 11. 面试前一分钟速查

这一节是 Roger 在面试前扫一眼就能稳住的 "5 条必须能 30 秒内复述的数字"。不追求完整，只追求对 headline 不犹豫。

### 11.1 R01（raw RL generalization）

- held-out `CodeContests test` 165 题 raw `accepted@1`：**`1.82%` → `7.88%`**（`3/165 → 13/165`，倍率 4.33x）
- `pass_ratio_mean`：`0.129` → `0.250`（几乎翻倍）
- `HumanEval`：`87.2%` → `89.0%`（+1.83 pp）；`MBPP`：`58.0%` → `62.5%`（+4.5 pp）
- 设置：base `Qwen2.5-Coder-7B-Instruct` → step1300_rl；A1 GRPO（DAPO-style）；4× RTX 5090

### 11.2 R04（Protocol B end-to-end repair）

- held-out test 上 step1300_rl：raw `7.88%` → after-repair `10.91%`（+5 solve）
- `repair_attempt_count = 150`、`repair_success_count = 5`、`conditional_repair_success = 3.33%`
- **记住**：这是端到端部署口径（Protocol B），不是纯修复能力（Protocol A）

### 11.3 R05（delta69 near-miss）

- `valid_big500` 的 delta69 高 partial bucket 19 题 one-turn repair 成功 5 题
- `conditional_repair_success = 26.32%`；first-pass `65.2%` → after-repair `72.5%`
- **重要**：这是子集条件成功率，**不等于全量 repair 率**；被问到时必须主动说"这是 near-miss 高价值 slice 上的条件成功率"

### 11.4 Checkpoint 双 winner

- `valid_big500` dev 集：step1000 `accepted@1 = 13.8%`（**solve-count winner**）/ step1300 `= 12.2%` 但 `pass_ratio_mean = 35.76%`（**partial-credit winner**）
- held-out 部署上 `step1300_rl` 是 **strongest deployed base**（综合 raw + repair）
- 三个角色不能混：solve winner ≠ partial winner ≠ deployed base，必要时都讲清楚

### 11.5 训练规模与算法

- 4× RTX 5090，A1 GRPO（DAPO 增强：asymmetric clip 0.2/0.28、token-mean、no KL）
- curriculum-aware 采样，1300 step 主线；reward = anchored_dense_v1
- 底座 verl + SandboxFusion；shared verifier 用来保证 eval 与 RL reward 口径一致

### 11.6 面试时的总原则

- 数字优先给 held-out，dev 数字只在被追问时补充
- Protocol B repair 和 Protocol A repair 先明确是哪个，再给数字
- 26.3% / 3.33% / 10.91% 三个数字容易混；按 "**全局 3.33% / near-miss 子集 26.3% / after-repair 绝对 10.91%**" 的顺序复述
- 不要把"dev 赢"说成"held-out 赢"；repair-SFT 的结论是"dev gain 真，held-out v2 需换 recipe"
