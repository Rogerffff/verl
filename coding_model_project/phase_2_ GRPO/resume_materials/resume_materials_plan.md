# 简历材料编写计划

## 0. 目标

当前目标不是把所有实验都补完，而是先基于**已经可核验的 RL / eval / repair / sandbox 工作**，尽快整理出一版可以直接支撑投递的项目材料。

这份计划要解决 4 件事：

1. 先固定哪些内容已经可以写进简历，哪些只能写成“进行中 / 已设计”
2. 先固定后续要产出的文档结构，避免反复改口径
3. 先固定所有可引用的文档、日志结论和代码锚点
4. 对尚未完成的 SFT 数据与结果，提供**仅供草稿使用**的合理估计区间，并明确禁止把估计值伪装成已完成结果

---

## 1. 当前工作假设

这份计划先按下面 4 个假设推进：

1. 简历主目标岗位是：
   - `LLM / Applied Research Engineer`
   - 或 `ML Systems / RL Engineer`
2. 你需要的是：
   - 一版可直接投递的项目描述
   - 一版更长的项目说明材料
   - 一版面试时可展开讲的证据索引
3. 当前 GPU 不可用，因此：
   - 不等待新一轮远端 SFT 或新增 eval 完成
   - 先用现有本地文档与同步下来的日志结论写材料
4. 未完成的 SFT 工作允许在**内部草稿**中使用 placeholder estimate，但：
   - 不进入最终对外简历 bullet
   - 不写成“已完成指标”

需要额外注意的一点是：

- 本地 `phase_2_ GRPO/output/` 目录当前只保留了少量历史文件
- 当前真正完整、可信的证据主要来自：
  - `experiment_handoff.md`
  - `eval_analysis/`
  - `repair_*` 文档
  - `remote_logs/`
  - 以及相关源码实现

如果后续你的目标岗位更偏：

- `研究 / 算法`
- `系统 / 基础设施`
- `生成式 AI 应用`

只需要调整最后的叙事侧重，不需要重写整个证据骨架。

---

## 2. 真实性边界

这部分必须写死，后面所有简历文案都遵守。

### 2.1 可以直接写成“已完成 / 已实现 / 已验证”的内容

1. 基于 `verl + vLLM + SandboxFusion` 实现并跑通了 curriculum-aware GRPO 训练链路
2. 为 `CodeContests` 任务实现了 standalone eval、shared verifier、multi-sandbox 评测链
3. 实现了 quarantine v3 与 curriculum bucket 的联动过滤
4. 实现了 one-turn verifier-guided repair eval，并支持 `reuse-first-pass`
5. 做过 `code_only` 与 `short_diagnosis+code` 的 prompt ablation
6. 修复过 sandbox 并发下的判题输出竞态 / nondeterminism 主问题，并完成本地 smoke 验证
7. 已经完成多轮 checkpoint 对比分析，并整理出 exact-solve / partial-credit / repairability 的分层结论

### 2.2 只能写成“设计完成 / 正在推进 / 计划中”的内容

1. `step900` repair-conditioned SFT 主数据构建
2. `short_diagnosis+code` 正式 teacher 数据生成
3. 基于新 sandbox 协议跑出来的最终 SFT 训练结果
4. 任何当前尚未落地的新机远端指标

### 2.3 明确禁止的写法

1. 不能把估计的 SFT 样本量写成已完成样本量
2. 不能把预估的 SFT 提升写成已达成指标
3. 不能把老协议下、受 sandbox 污染或未 protocol-match 的结果写成最终结论
4. 不能把设计文档里的 future work 写成 shipped feature

---

## 3. 当前最值得写进简历的项目主线

### 3.1 推荐主叙事

最推荐的简历主线不是“我做了很多零碎实验”，而是下面这个完整故事：

> 我围绕算法代码生成任务，搭建并迭代了一条 curriculum-aware RL + verifier-guided repair 的完整实验链。
> 一方面做 GRPO 训练、分桶 curriculum、数据清洗和 checkpoint 分析；
> 另一方面做 shared verifier、多 sandbox 评测、repair prompt ablation 和 sandbox 稳定性修复；
> 最终把“模型能力变化”、“评测协议可信度”和“后续 SFT 数据构建路线”这三件事打通。

这条主线有 3 个优点：

1. 同时覆盖 `ML algorithm + eval infra + debugging`
2. 不依赖尚未完成的 SFT 才能成立
3. 和你现有本地证据高度一致

### 3.2 简历里建议拆成的 3 个模块

#### 模块 A：Curriculum RL 主线

强调：

- GRPO 训练
- A/B/C/D bucket 课程设计
- quarantine 过滤
- checkpoint 分析
- solve / pass-ratio 分化结论

#### 模块 B：Verifier + Sandbox 基础设施

强调：

- shared verifier
- multi-sandbox 评测
- round-robin / client RR
- deterministic / nondeterministic 排查
- SandboxFusion 补丁

#### 模块 C：Repair 实验主线

强调：

- one-turn repair eval
- reuse-first-pass 固定 first-pass 响应
- code-only vs short-diagnosis+code
- repair-ready checkpoint 与 exact-solve winner 的分离

---

## 4. 当前可直接引用的核心结果

下面这些是当前最适合进入简历与项目说明的“硬结论”。

### 4.1 RL / eval 结果

来自：

- [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
- [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)

当前最稳的说法：

1. `step1000` 是当前 `valid_big500` 的 exact-solve winner
   - `69 / 500`
   - `accepted@1 = 13.8%`
2. `step1300` 不是 exact-solve winner，但在 fixed-response rejudge 下是当前最强的 partial-credit checkpoint
   - `pass_ratio_mean ≈ 0.3574`
3. `step900` 是当前更好的 one-turn repair base，而不是 `step1300`

### 4.2 Repair 结果

来自：

- [repair_phase4_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_phase4_design.md)
- [phase4_step1_repair_eval_implementation_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/phase4_step1_repair_eval_implementation_plan.md)

当前最稳的说法：

1. `step900 + valid_big500 + reuse-first-pass + WA/RE + pass_ratio>=0.6` 上：
   - `code_only` 的 `conditional_repair_success = 12.33%`
   - `short_diagnosis_code` 的 `conditional_repair_success = 16.44%`
2. `short_diagnosis+code` 在 `step900` repair-ready slice 上有明确正信号
3. 同样的优势没有迁移到 `step1300`
4. 这说明：
   - partial-credit 更强的 checkpoint，不一定更 repairable
   - repair prompt 的收益和 checkpoint 状态强相关

### 4.3 Sandbox / infra 结果

来自：

- [sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_judge_repair.md)
- [sandbox_repair_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_repair_plan.md)
- `SandboxFusion` 实现文件

当前最稳的说法：

1. 找到了 sandbox 并发下相同 response 判题结果漂移的重要来源
2. 根因是：
   - 进程退出后过早 kill process tree
   - drain stdout/stderr 方式存在竞态
3. 已在 `SandboxFusion` 最小修复中引入：
   - request-scoped process group
   - guarded drain
   - 显式 `DrainTimeout`
   - selective second drain

---

## 5. 当前建议的简历结构

这次不建议只写一个 bullet。建议同时准备 4 份材料。

### 5.1 文档 A：简历主文案

建议新建：

- `resume_materials/resume_project_master.md`

用途：

- 写一版完整的项目说明
- 后面再从这里压缩出 1-3 条简历 bullet

结构建议：

1. 项目一句话概述
2. 目标问题
3. 你具体做了什么
4. 最强结果与可信边界
5. 当前未完成但正在推进的部分

### 5.2 文档 B：中英双语 bullet

建议新建：

- `resume_materials/resume_bullets_cn_en.md`

用途：

- 生成：
  - 中文简历 bullet
  - 英文简历 bullet
  - 不同岗位版本

建议至少准备 3 组版本：

1. `ML / Applied Research`
2. `LLM Infra / Systems`
3. `General SWE + AI project`

### 5.3 文档 C：claim-evidence map

建议新建：

- `resume_materials/resume_claims_evidence_map.md`

用途：

- 每一条简历表述都追溯到：
  - 哪个文档
  - 哪个日志结论
  - 哪个代码文件

这样后续无论改中文、改英文、改岗位风格，都不容易越界。

### 5.4 文档 D：面试展开稿

建议新建：

- `resume_materials/resume_interview_story_bank.md`

用途：

- 准备面试时可以展开讲的：
  - 背景
  - 决策
  - 指标
  - debugging
  - trade-off

---

## 6. 证据索引：文档与代码锚点

后续简历材料至少应引用下面这些本地文档或代码。

### 6.1 项目结论类文档

1. [experiment_handoff.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/experiment_handoff.md)
2. [validbig500_step1300_vs_baselines.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/eval_analysis/2026-04-09_validbig500_checkpoint_review/validbig500_step1300_vs_baselines.md)
3. [repair_phase4_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_phase4_design.md)
4. [phase4_step1_repair_eval_implementation_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/phase4_step1_repair_eval_implementation_plan.md)
5. [repair_prompt_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_prompt_design.md)
6. [step900_repair_conditioned_sft_data_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_repair_conditioned_sft_data_plan.md)
7. [sandbox_judge_repair.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_judge_repair.md)
8. [sandbox_repair_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sandbox_repair_plan.md)

### 6.2 关键实现文件

#### RL / curriculum / quarantine

1. [step580_curriculum_builder.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_builder.py)
2. [step580_curriculum_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_dataset.py)
3. [step580_curriculum_sampler.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/step580_curriculum_sampler.py)
4. [problem_quarantine.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/problem_quarantine.py)
5. [build_grpo_parquet.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_grpo_parquet.py)

#### Eval / verifier / repair

1. [phase0_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py)
2. [verifier/shared.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py)
3. [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)
4. [phase4_repair_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py)

#### Sandbox / runtime

1. [SandboxFusion/sandbox/utils/execution.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/execution.py)
2. [SandboxFusion/sandbox/runners/base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py)

#### Trainer / failure diagnosis

1. [verl/trainer/ppo/ray_trainer.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/ppo/ray_trainer.py)

---

## 7. 简历中最值得强调的技术点

### 7.1 可以主打的工程能力

1. 复杂 LLM 训练 / 评测链路搭建
2. 多组件系统集成
   - `verl`
   - `vLLM`
   - `SandboxFusion`
   - custom verifier
3. 实验协议设计与可信性控制
4. nondeterminism / race condition 排查与最小修复
5. 数据治理与 curriculum 资产管理

### 7.2 可以主打的算法 / 研究能力

1. 把 single-shot solve 与 partial-credit 拆成两条评价轴
2. 用 fixed-response rejudge 拆分：
   - generation drift
   - judge drift
3. 用 reuse-first-pass 控制 repair 评测协议
4. 设计 repair prompt ablation，而不是只堆 prompt trick
5. 设计 repair-conditioned SFT 数据方案，并引入稳定性二次 QC

### 7.3 不建议主打的点

1. 未完成的新 SFT 最终指标
2. 太老、未 protocol-match 的 step200 / step400 历史分支
3. 没有经过 patched sandbox + client RR 口径复核的旧表格

---

## 8. Placeholder estimate 策略

这部分只给**内部草稿**用。

### 8.1 允许使用 placeholder 的场景

1. 写项目长说明时，为了描述“下一步会做什么”
2. 写内部版本的 English project brief
3. 写面试时的 roadmap

### 8.2 不允许使用 placeholder 的场景

1. 最终投递版一页简历 bullet
2. 对外公开项目总结
3. 面试中把 placeholder 当成既成事实

### 8.3 当前建议的 placeholder 区间

仅供草稿使用，后续必须替换或删除：

1. `repair-conditioned SFT` 预计可保留的 verified 样本量：
   - `~600 - 1200`
2. `short_diagnosis+code` 相对 `code_only` 在训练后预期的离线 repair 提升：
   - `conditional repair success +3 ~ +6 pct`
3. 对 focused repair slice 的预期 solve / accepted 改善：
   - `+1 ~ +2 absolute points`

更安全的写法不是写成结果，而是写成：

> designed a repair-conditioned SFT pipeline targeting ~600-1200 verified training pairs

而不是：

> trained on 1200 verified pairs and improved X%

---

## 9. 具体执行顺序

### Step 1. 先完成这份计划文档

目标：

- 固定结构
- 固定真实性边界
- 固定证据来源

### Step 2. 产出一版项目总说明

输出：

- `resume_project_master.md`

要求：

1. 先写长版
2. 明确区分：
   - 已完成
   - 已验证
   - 设计中

### Step 3. 压缩成简历 bullet

输出：

- `resume_bullets_cn_en.md`

要求：

1. 先写 6-8 条长 bullet
2. 再压缩成 2-3 条最终简历 bullet
3. 每条 bullet 都要能追溯到证据

### Step 4. 做 claim-evidence map

输出：

- `resume_claims_evidence_map.md`

要求：

每条 claim 至少对应：

1. 一个文档结论
2. 一个代码实现
3. 如有指标，再给一个指标来源

### Step 5. 做 interview story bank

输出：

- `resume_interview_story_bank.md`

建议准备的题：

1. 为什么 RL 的 exact-solve 和 partial-credit 会分离
2. 为什么要做 reuse-first-pass
3. 为什么要修 sandbox，而不是简单多跑几次
4. 为什么 `step900` 更 repairable，但 `step1300` partial-credit 更强
5. 为什么下阶段是 repair-conditioned SFT，而不是继续盲目长跑 RL

---

## 10. 推荐的最终简历表述方向

### 10.1 最稳版本

强调你完成了：

1. 训练链
2. 评测链
3. repair eval
4. sandbox 修复
5. checkpoint / protocol 分析

这版**完全不依赖未来 SFT 指标**，最适合立刻投递。

### 10.2 强一点的版本

在最稳版本基础上，加一句：

> Designed a repair-conditioned SFT data pipeline as the next stage for converting RL-induced near-miss solutions into verified passes.

这句话可以写，因为它描述的是**设计完成**，不是结果完成。

### 10.3 当前不建议的版本

不建议把简历主卖点写成：

- “我已经完成 repair SFT 并显著提升”

因为当前你手里的最强证据其实是：

- RL + eval + repair + sandbox protocol 的整链能力

而不是已经闭环完成的新一轮 SFT 提升。

---

## 11. 本计划之后的直接下一步

这份计划写完后，下一步就按下面顺序继续：

1. 写 `resume_project_master.md`
2. 写 `resume_bullets_cn_en.md`
3. 写 `resume_claims_evidence_map.md`
4. 写 `resume_interview_story_bank.md`

如果后续新机配置好、SFT 跑完，只需要：

1. 更新 `claim-evidence map`
2. 替换 placeholder
3. 重新压缩一次 final bullet

整体结构不需要重做。
