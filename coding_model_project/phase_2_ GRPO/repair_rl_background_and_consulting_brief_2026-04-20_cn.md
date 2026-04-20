# Repair RL 背景与咨询简报 (2026-04-20)

## 0. 目的

本文档是一个自包含的背景简报，面向需要快速了解以下内容的其他 agent 或合作者：

1. 这个 coding RL 项目是什么；
2. 已经构建并验证了什么；
3. 当前与 repair 相关的瓶颈是什么；
4. 为什么现在开始考虑 repair RL；
5. 哪些外部工作与此最相关；
6. 当前最有价值的建议类型是什么。

本文档并不打算取代项目中既有的权威文档。它是一份面向咨询的汇总文档，将当前项目状态集中到一个地方，使其他 agent 无需通读整个代码仓库即可给出更高质量的建议。

---

## 1. 一段话的执行摘要

本项目是一个构建在 `verl` 之上的 coding RL 后训练项目，使用基于 verifier 的 GRPO 来提升模型在竞赛级编程任务上的代码生成能力。正式的 Phase 3 主线是 `带 DAPO 风格稳定化的 GRPO`，使用 `anchored_dense + guardrails` 奖励和一套共享 verifier 基础设施，从而让评测和 RL 训练消费同一份基于执行的 ground-truth 信号。主要的 raw RL 故事已经足够稳固：held-out 的 raw coding 能力相对于 base model 有显著提升，同时项目还累积了大量系统、评测、数据治理、repair 和 QC 相关的基础设施。但是，当前的 repair-SFT 路线还没有转化为 held-out 意义上的 headline 胜利：dev 侧的 repair 收益是真实的，但目前最好的 repair-SFT checkpoint 在 held-out 部署风格的 repair 评测上仍然打不过 `step1300_rl`。当前的问题是：相比再做一轮 repair-SFT，做一轮聚焦的 repair RL 是否是更好的下一步？尤其是考虑到 repair-SFT 数据似乎很难进一步扩大规模，而 repair RL 可能更便宜、对数据准备的依赖也更低。

---

## 2. 仓库与分支上下文

### 2.1 主要仓库

- 项目主仓库：
  - `/Users/roger/Desktop/coding_RL_project/verl`
- 用于执行 / 判定的 sandbox 仓库：
  - `/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion`

任何 agent 都必须注意的重点：

- `verl` 与 `SandboxFusion` 是两个独立的 git 仓库；
- 如果问题涉及 sandbox API 行为、执行语义、返回字段或运行时问题，应查 `SandboxFusion`，而不只是查 `verl`。

### 2.2 当前的分支层级定位

- 当前正在推进的路线是这个 coding RL 项目的 GRPO 开发路线；
- 当前正式的算法选择是：
  - `带 DAPO 风格稳定化的 GRPO`；
- `filter_groups` 明确不在当前主线定义范围内。

本地规范文档：

- `README.md`
- `algorithm_decision_guide.md`
- `formal_reward_design.md`
- `shared_verifier_infra_guide.md`
- `experiment_handoff.md`

---

## 3. 当前项目叙事

最清晰、诚实的对外叙事是：

1. 从一个较强的 coding base model 出发；
2. 观察到目标错误模式主要是 wrong-answer / 部分正确，而不是语法类错误；
3. 在 `verl` 中构建基于 verifier 的 RL 基础设施，并使用共享的基于执行的 truth source；
4. 跑 curriculum RL 并取得真实的 held-out raw 收益；
5. 构建 one-turn repair 评测、repair 反馈 prompt、repair-conditioned SFT、teacher/QC、quarantine 与 rejudge 机制，以理解部分正确的 gain 能否被转化为最终 AC；
6. 发现 repair 在 dev 切片和部分部署风格 case 上确实有帮助，但当前的 repair-SFT 作为部署模型仍然打不过最强的 held-out RL checkpoint。

换句话说：

- 项目已经有一个稳固的 raw RL 故事，
- 有一个稳固的 ML 系统 / 评测严谨性故事，
- 以及一个稳固的数据质量 / 调试 / 基础设施故事，
- 但 repair 分支还不是一个干净的 held-out headline 成功。

---

## 4. 推荐优先阅读的本地规范文档

如果其他 agent 只有时间读几份文件，下列最有价值。

### 4.1 当前权威文档

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/experiment_handoff.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_eval_current_contract_2026-04-18.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_protocolA_results_2026-04-18.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_protocolB_results_2026-04-18.md`

### 4.2 主线设计文档

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/README.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/algorithm_decision_guide.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/formal_reward_design.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/shared_verifier_infra_guide.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_phase4_design.md`

### 4.3 简历 / 资产汇总文档

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/resume_materials/project_asset_inventory_2026-04-18.md`

---

## 5. 当前技术主线

### 5.1 训练目标与算法

正式的 Phase 3 主线标题是：

- `带 DAPO 风格稳定化的 GRPO`

当前主线包含：

- `clip-higher`
- `no-KL`
- `token-mean`
- verifier 侧 / 执行侧的 batching 与稳定性改进

明确不在当前主线内：

- `filter_groups / 动态采样`
- 奖励侧的 overlong shaping
- 完整的 DAPO 复现

来源：

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/algorithm_decision_guide.md`

### 5.2 奖励

正式奖励是：

- `anchored_dense + guardrails`
- 默认名称：`anchored_dense_v1`

公式：

- 如果出现 infra/sandbox/API/无测试/截断等问题：
  - `INVALID_FOR_RL`
- 如果 `error_type` 属于 extraction / empty / non-code / syntax 类：
  - `-1.0`
- 否则：
  - `0.8 * pass_ratio_all + 0.2 * accepted`

关键语义选择：

- 奖励基于来自共享 verifier 的完整外部测试；
- 保留 `accepted` 作为显式 anchor；
- invalid 样本在 RL 更新中被置零。

来源：

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/formal_reward_design.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/grpo_batch_reward.py`

### 5.3 共享 verifier 基础设施

本项目在共享 verifier 设计上已经比较扎实：

- 评测和 RL 使用同一层 verifier；
- 核心执行 truth 契约存放在 `verifier/shared.py`；
- `phase0_eval.py` 与 RL 奖励都消费这一层；
- verifier 的指标通过 trainer 路径记录；
- invalid-for-RL 语义被显式集成到 trainer 侧的 advantage 处理中。

主要代码 / 文档：

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/grpo_batch_reward.py`
- `/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/ppo/ray_trainer.py`
- `/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/ppo/metric_utils.py`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/shared_verifier_infra_guide.md`

---

## 6. 当前实验状态

### 6.1 当前最强的 raw 部署模型

当前最强的 held-out 部署 base 仍然是：

- `step1300_rl`

这才是目前最主要的结果锚点，而不是 repair-SFT 路线。

相关本地参考：

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/experiment_handoff.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md`

### 6.2 最强的 raw held-out 故事

当前可用于简历的最强结果故事是 raw held-out RL 收益：

- base `codecontests_test accepted@1 = 0.01818`
- `step1300_rl codecontests_test accepted@1 = 0.07879`
- `HumanEval accepted@1` 同样有提升
- `MBPP_reg accepted@1` 同样有提升

这支持"主 RL 线已经带来了有意义的外部收益"的声明。

### 6.3 Checkpoint 读数的微妙之处

Checkpoint 的解读不是单指标的：

- `step1000` 是更偏 exact-solve 友好的点；
- `step1300` 在 partial-credit / pass-ratio 质量上更强。

这一点很重要，因为当前 repair 问题的一部分是：partial 更强的 checkpoint 能否被转换为更多最终 AC。

### 6.4 Repair 当前状态

Repair 是真实的，但仍然带有限定条件。

当前状态：

- 固定输入的 repair 能力在 dev 上是真实的；
- 端到端 self-repair 在部署风格评测中是真实的；
- 当前 lineage 的 repair-SFT v1 提升了 dev 侧 repair；
- 但当前 repair-SFT 作为 held-out 部署模型仍然打不过 `step1300_rl`。

具体地：

- 当前 lineage 最好的 repair-SFT checkpoint 是：
  - `step1300_sft_v1_step60`
- 最强 held-out 部署 base 是：
  - `step1300_rl`

Held-out 关键读数：

- `step1300_rl` 在 `codecontests_test / Protocol B` 上：
  - `0.1091`
- `step1300_sft_v1_step60` 在同样的 held-out Protocol B 上：
  - `0.0970`

权威参考：

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/experiment_handoff.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_protocolB_results_2026-04-18.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_eval_current_contract_2026-04-18.md`

### 6.5 One-turn repair 的当前定位

One-turn repair 仍然有用，但在内部当前应解读为：

- 一个真实的能力；
- 一个良好的诊断 / 兜底工具；
- 但尚不是项目级 headline 改进的主引擎。

当前内部定位已经被收窄为：

- 定向兜底；
- repair-prior 的诊断工具；
- 而不是新的主训练引擎。

来源：

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/experiment_handoff.md`

---

## 7. 与 Repair RL 相关的现有资产

本项目已经有很多使 repair RL 可行的资产。

### 7.1 评测与 repair 流水线资产

- One-turn repair 评测实现：
  - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py`
- Repair prompt / 反馈构造器：
  - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py`
- 当前 repair prompt 设计文档：
  - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_prompt_design.md`

### 7.2 奖励与 verifier 资产

- 共享 verifier；
- 批式 reward adapter；
- trainer 对 verifier 指标和 invalid 屏蔽的集成。

这意味着 repair RL 不必在基础设施上从零开始。

### 7.3 既有的 repair 评测资产

本地输出中已经有完整的 Protocol A / Protocol B 结果树和汇总，包括：

- `valid_big500`
- `codecontests_test`
- delta 切片
- per-problem 日志
- first-pass 与 after-repair 的汇总

### 7.4 既有的 repair-SFT 资产

项目已经包含：

- repair-conditioned SFT 规划文档；
- repair SFT 数据准备资产；
- teacher 生成与 QC 资产；
- anchor / 通用代码 mix 的规划。

但当前的担忧是：这条 repair-SFT 数据线从当前 `step1300` lineage 的 trace 中已经越来越难进一步扩大规模。

### 7.5 简历 / 咨询资产

项目在"用于 agent 咨询的元资产"上相对罕见地完善：

- claim-evidence map；
- 项目资产清单；
- handoff 文档；
- protocol 文档。

这些让其他 agent 在拿到合适的背景文档后能快速对项目进行推理。

---

## 8. 当前瓶颈

实际瓶颈是：

1. 当前 repair-SFT 还不是一个 held-out headline 胜利；
2. 基于 `step1300` / 当前 ABCD bucket trace 的 repair-SFT 数据看起来很难再扩大多少；
3. 再做一轮小规模 repair-SFT 可能会受限于同样的数据支撑不足问题；
4. Repair RL 可能更便宜，因为它有希望复用已有的 buggy-code + verifier-feedback 资产，省掉再一次大规模 teacher/SFT 数据建设。

这改变了决策框架。

问题不再只是：

- "Repair RL 在理论上是否优雅？"

现在同时也是：

- "在当前本地约束下，repair RL 是不是测试'二次训练能否带来 held-out repair 收益'这一假设的最便宜可信路径？"

---

## 9. 为什么现在要考虑 Repair RL

### 9.1 高层动机

动机是测试：当前 near-miss / partial-correctness 的 gain 能否通过定向的二次训练被转化为最终 AC，同时不必再构建一份大规模的 repair-SFT 数据集。

### 9.2 为什么不继续当前的 repair-SFT 路径？

因为当前担忧是：

- 可用的 repair-SFT 数据池从现有 trace 中可能已接近饱和；
- 想进一步扩大可能困难或昂贵；
- 当前 repair-SFT 已经显示出 dev 收益，但仍未超过最强 held-out RL checkpoint。

### 9.3 为什么 repair RL 在当前情境下有吸引力

在这个特定情境下，repair RL 有吸引力是因为它可能：

- 复用已有 buggy-code 与 verifier-feedback 资产；
- 避免再走一次大规模 teacher 生成环节；
- 比窄口径 repair-SFT 更好地保留 raw coding 通用能力；
- 在相同 verifier 契约下直接优化二次 repair 成功率；
- 对"是否值得继续追求二次训练"给出更明确的是/否答案。

### 9.4 Repair RL 要测试的具体假设

具体假设是：

- 第一次 RL 已经产生了有意义的 partial 正确性；
- 这些失败并非一律无解；
- One-turn verifier-guided repair 可以直接通过 RL 训练；
- 二次 RL 策略可以把 held-out 部署风格的 repair 提升到足以超过当前 `step1300_rl` repair baseline。

换句话说：

- 在当前 repair-SFT 还没跨过线的情况下，二次训练能否带来真实的 held-out 收益？

---

## 10. 为什么 Repair RL 并非天然容易

本节展开主要的注意点。

### 10.1 奖励分布比单轮 coding RL 更偏斜

在普通的单轮 coding RL 中，奖励只问：

- 最终代码有多好？

在 repair RL 中，任务变为：

- 给定 buggy 的第一次解与 verifier 反馈，模型的修复效果有多好？

这会产生一个更偏斜的奖励分布：

1. 很多样本仍然接近无解，继续停留在 0 附近；
2. 一小部分高 partial 的 near-miss 样本会突然跳升；
3. 一部分样本因为模型过度改写而反而变差。

所以一个 repair-RL group 可能长这样：

- 很多低或近 0 的值；
- 少量非常大的 win；
- 少量由于退化编辑造成的中等偏高但具有误导性的值。

这会让 group-relative estimation 比普通单轮 RL 更噪声。

### 10.2 触发策略非常敏感

Repair 并不是对所有失败都同样有用。

如果 repair 候选选取过宽：

- 过多无解样本进入训练；
- 奖励方差塌缩；
- judge 成本上升；
- 训练效率变低。

如果过窄：

- 训练只覆盖 trivial 的 near-miss 修复；
- 表面 repair 成功率看起来很高；
- 但覆盖面和泛化能力仍然弱。

这意味着 repair RL 强依赖于：

- 允许哪些 bucket；
- 使用多大的最低 pass-ratio 阈值；
- 包含哪些 error type；
- 反馈中展示多少个失败 case。

### 10.3 Judge 成本更高

完全在线的 repair 流水线可能需要：

1. 第一次生成；
2. 第一次 judge；
3. 反馈构造；
4. 第二次生成；
5. 第二次 judge。

这比单遍 RL 样本要重得多。

可以通过使用缓存的第一次 trace、只训练第二次来部分缓解，但 repair RL 仍然会比普通单轮 RL 制造更高的执行成本压力。

### 10.4 朴素 `anchored_dense` 可能不够

当前 reward 只看最终 solution 的质量。

对于 repair，这忽略了三个重要事实：

1. 第二次是否相对第一次有改进；
2. 是否退化并破坏了本已正确的行为；
3. 编辑是否过大或破坏性过强。

失败模式示例：

- 第一次：`0.9`
- 第二次：`0.7`

在朴素的最终分数 reward 下，这仍然看起来像个不错的正样本。
但从 repair 的角度来看这是一次糟糕的 repair，因为模型让答案变差了。

这就是为什么二次 repair 可能需要比当前朴素 final-score `anchored_dense` 更具编辑感知的 reward。

### 10.5 过度编辑是一个真实风险

Repair 不只是"让最终测试通过"。
它同时也是：

- 不要不必要地覆盖正确代码；
- 不要把一次局部 bug 修复变成一次不稳定的完整重写；
- 尽量保留已经正确的结构。

一个只看最终 pass-ratio 的 reward 会无意间鼓励：

- 重写密集的行为；
- 不稳定的 repair 风格；
- 更高的退化风险。

---

## 11. 外部工作回顾

下面是与本决策最相关的外部工作的紧凑 review。

### 11.1 CodeRL (2022)

- 标题：
  - `CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning`
- 链接：
  - https://arxiv.org/abs/2207.01780
- 相关性：
  - 表明使用执行 / 功能正确性信号的 RL 能够改进代码生成；
  - 为通用代码生成引入了基于 critic 的稠密反馈设计。
- 可迁移经验：
  - verifier / 测试反馈是一种合法的 coding RL 信号；
  - 但这仍然主要是从零生成代码的 RL 论文，而不是针对 repair 的配方。

### 11.2 RLTF (2023)

- 标题：
  - `RLTF: Reinforcement Learning from Unit Test Feedback`
- 链接：
  - https://arxiv.org/abs/2307.04349
- 相关性：
  - 论证基于单元测试反馈的在线 RL 是有用的；
  - 强调多粒度反馈而不仅仅是粗粒度最终结果。
- 可迁移经验：
  - 在线 verifier-guided RL 是合理方向；
  - 但这同样是通用 coding RL 证据，而不是直接的二次 repair 设计。

### 11.3 ChatRepair / Conversational APR (2023)

- 标题：
  - `Conversational Automated Program Repair`
- 链接：
  - https://arxiv.org/abs/2301.13246
- 相关性：
  - 是使用 validation 反馈构成 repair loop 的重要 repair 参考；
  - 在对话式过程中交替进行 patch 生成与反馈。
- 可迁移经验：
  - 基于 validation 反馈的 repair 流水线已较成熟；
  - 公共 repair 文献在反馈循环式的 repair 流水线上比在 repair-RL 作为主训练线上更成熟。

### 11.4 Self-Debugging (2023)

- 标题：
  - `Teaching Large Language Models to Self-Debug`
- 链接：
  - https://arxiv.org/abs/2304.05128
- 相关性：
  - 表明迭代式调试 / 修复能显著提升代码性能。
- 可迁移经验：
  - 二次改进是真实的；
  - 但该论文更聚焦在 prompting / self-debugging 行为上，而不是 repair 专用 RL。

### 11.5 VRpilot / Validation-feedback repair (2024)

- 标题：
  - `A Case Study of LLM for Automated Vulnerability Repair: Assessing Impact of Reasoning and Patch Validation Feedback`
- 链接：
  - https://arxiv.org/abs/2405.15690
- 相关性：
  - 表明 reasoning + validation 反馈在安全场景中对 repair 有显著帮助。
- 可迁移经验：
  - 外部工具反馈对 repair 非常有用；
  - 但这同样更直接支持 repair 流水线，而非 repair RL 配方。

### 11.6 PSGPO (2024/2025)

- 标题：
  - `Process Supervision-Guided Policy Optimization for Code Generation`
- 链接：
  - https://arxiv.org/abs/2410.17621
- 相关性：
  - 指出只用最终单元测试 reward 可能过于稀疏；
  - 为 coding RL 提出 line-level / process reward 建模。
- 可迁移经验：
  - 如果 repair RL 遭遇稀疏或分布糟糕的 reward，更丰富的过程信号可能会有帮助；
  - 但这仍是通用 coding RL 而不是直接的 repair RL。

### 11.7 SWE-RL (2025)

- 标题：
  - `SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution`
- 链接：
  - https://arxiv.org/abs/2502.18449
- 相关性：
  - 是一个重要信号，表明在软件工程数据上做 RL 可能泛化到窄 benchmark 之外。
- 可迁移经验：
  - 带软件工程风味的 RL 能提升通用能力；
  - 但 SWE-RL 比单轮 repair 广得多，且使用不同的数据前提。

### 11.8 ReCode (2025)

- 标题：
  - `ReCode: Updating Code API Knowledge with Reinforcement Learning`
- 链接：
  - https://arxiv.org/abs/2506.20495
- 相关性：
  - 编辑式 code RL 而非 bug repair；
  - 在保留通用代码能力方面相对 SFT 表现更好。
- 可迁移经验：
  - 编辑式 RL 可能比窄口径 SFT 对通用 coding 能力的破坏更小；
  - 这是 repair RL 在当前情境下有吸引力的最强外部理由之一。

### 11.9 GAPO (2025/2026)

- 标题：
  - `GAPO: Robust Advantage Estimation for Real-World Code LLMs`
- 链接：
  - https://arxiv.org/abs/2510.21830
- 相关性：
  - 直接研究真实世界代码编辑中的 group-relative RL。
- 可迁移经验：
  - 真实代码编辑的 reward 分布是偏斜且嘈杂的；
  - Repair RL 应预期 group reward 不稳定；
  - 可能需要更鲁棒的 advantage estimation 或更干净的样本过滤。

### 11.10 PRepair (2026)

- 标题：
  - `QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization`
- 链接：
  - https://arxiv.org/abs/2604.05963
- 相关性：
  - 目前能看到的最接近 repair 专用 GRPO 式 RL 的直接先例。
- 重要性：
  - 显式地把过度编辑当作核心失败模式；
  - 使用 edit-aware GRPO reward 而不是仅 final pass/fail。
- 可迁移经验：
  - 对 repair RL 最直接的外部支持不是"只用 final test reward"；
  - 而是"用 repair-aware reward，尤其是 edit-aware reward"。

---

## 12. 对本项目最相关的外部要点

文献回顾表明：

1. 基于 verifier / 单元测试反馈的 coding RL 有充分支持；
2. 带 validation 反馈的迭代式 repair 有充分支持；
3. 公共的 repair 流水线工作比公共的"repair-RL 作为主训练线"工作更成熟；
4. 在某些场景下，编辑式 RL 比窄口径 SFT 更好地保留通用能力；
5. 真实代码编辑 RL 的 reward 分布偏斜且嘈杂；
6. 对 repair-RL 最强的直接信号指向 edit-aware reward 设计，而不是朴素 final-score reward。

所以文献并没有说：

- "Repair RL 显然是下一步的标准选择。"

相反，它说的是：

- Repair RL 是可行的，支持度也在上升；
- 但需要谨慎对待；
- 尤其是在 reward 设计和样本过滤上。

---

## 13. 当前建议

### 13.1 在当前约束下的更新建议

鉴于新说明的约束——即当前 repair-SFT 数据很难显著扩大规模——建议是：

- **值得尝试 repair RL**；
- 但作为一次**小规模、可控的二次 RL 探针**；
- 而不是作为新的大型项目主线。

这比笼统的"继续做 repair-SFT"的建议更积极，因为当前瓶颈明显与数据规模和准备成本相关。

### 13.2 为什么建议改变了

如果 repair-SFT 数据易于扩大，更高 ROI 的选择可能仍是：

- 规模更大的、配方升级的 repair-SFT v2。

但在当前情况下：

- Repair-SFT 数据似乎接近饱和；
- 额外的 teacher / QC 工作可能无法将其大幅扩大；
- Repair RL 可能用更低的成本测试关键假设。

这使得 repair RL 成为一个合理的下一轮实验。

### 13.3 不应该做什么

不应立即启动：

- 覆盖所有可 repair 失败的大规模 repair RL 活动；
- 只照搬朴素 `anchored_dense` 的 reward；
- 包含大量无解 `bucket_0` / 死硬 case 的宽触发器；
- 一次性改动太多变量的设置。

---

## 14. 一次最小 Repair RL 探针的建议形态

这还不是最终的实验方案，但它是当前建议的形态。

### 14.1 作用域

将任务定义为：

- `题目陈述 + buggy 第一次代码 + verifier 反馈 -> 修复后的代码`

应将其视为二次任务，而不是普通的单轮 coding。

### 14.2 Base Model

优先：

- `step1300_rl`

可能的对比 checkpoint：

- `step1300_sft_v1_step60`

### 14.3 候选样本策略

从窄开始。

初期可能的候选切片：

- 较高 partial 的 `A/B` 风格样本；
- Near-miss 失败；
- 可能是 `wrong_answer` 和 `runtime_error`；
- 一开始避免把 `timeout` 作为主要焦点；
- 避免被 `bucket_0` 这种无解失败淹没。

### 14.4 数据来源策略

优先复用：

- 已有的第一次 trace；
- 已缓存的 buggy 代码；
- 已有的 verifier 反馈逻辑；
- 当前的 repair prompt 构造器。

这保持较低成本，并避免再做一次大规模 teacher 数据建设。

### 14.5 Reward 方向

不要只用 final-score 的 `anchored_dense`。

一个更合理的起步方向是：

- 最终正确性项；
- 加上"相对第一次的改进"项；
- 可选的 edit-preservation / edit-size 正则。

粗略地说：

- 奖励好的 repair；
- 惩罚退化；
- 不必要时不鼓励破坏性的完整重写。

### 14.6 成功判据

主要成功判据应为：

- 在部署风格 repair 评测上打败当前 `step1300_rl` 的 held-out repair baseline。

具体来说，关键目标是：

- `codecontests_test / Protocol B`

同时要检查：

- 通用 raw 能力不应塌缩。

### 14.7 何时提前停止

如果一个小规模 repair RL 探针出现：

- 奖励不稳定；
- Held-out 没有增益；
- 很大的 judge 成本换来的改进很小；
- 或明显的通用能力退化，

那么应提前停止，不要推进为新的长周期活动。

---

## 15. 可向其他 Agent 提出的问题

如果把本文分享给其他 agent，最有价值的问题可能是：

1. 在当前本地约束下，一次小规模 repair RL 探针的 ROI 是否高于再做一轮 repair-SFT？
2. 在当前 verifier 契约之上，最合理的 repair 专用 reward 设计是什么？
3. 什么样的第一次 bucket / 触发策略最可能给出稳定的 repair RL 信号？
4. Repair RL 应该作为纯粹的二次 RL（使用缓存的第一次 trace）来训练，还是应该保持完全在线？
5. 应如何衡量成功，才能让结果具备简历相关性，而不仅是学术上的有趣？
6. 如果 repair RL 失败，最佳兜底是什么：
   - repair-SFT v2，
   - repair-distilled 单轮 SFT，
   - 还是不再继续做 repair 训练？

---

## 16. 最终工作立场

本简报背后的当前工作立场是：

- 即使没有新的 repair-RL 结果，项目本身也已经足够用于简历；
- 但当前 repair-SFT 数据规模看起来受限，因此一次小规模 repair RL 探针是合理的；
- 而且 repair RL 可能是剩下的最便宜方式，用来测试"二次训练是否能成为真实的 held-out 改进，而不只是 dev 侧或窄切片效应"。

主要告诫是：不应把 repair RL 视为"只是换了个 repair prompt 的普通 coding RL"。
它很可能需要：

- repair-aware 的样本选择；
- repair-aware 的 reward 塑形；
- 对 judge 成本的细致控制。

这是其他 agent 在给出建议时最需要牢记的一点。
