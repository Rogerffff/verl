
---

# RLVR Coding Model（Resume Project）

**Goal**：用可验证 reward（单测/判题）完成离线偏好优化（可选 DPO）+ 在线 GRPO，做出工业界认可的“端到端后训练闭环 + 可信评测 + 成本/稳定性面板”。

---

## 0) 项目背景（写给招聘方看的）

我做这个项目是为了求职 **LLM RL / Post-training / Alignment / RLHF/RLVR** 方向算法工程师岗位。岗位通常希望你不仅“会一个算法名”，而是能把下面几件事做成闭环：

* **端到端后训练闭环**：SFT / 偏好数据构建 / RL 训练 / 评测与迭代
* **鲁棒评测与可信实验**：严格数据划分、消融、多 seed、可复现
* **工程化与效率**：吞吐、延迟、成本（tokens/判题时间）、监控与回归
* **安全与约束**：代码执行沙盒隔离、超时/资源限制、防 reward hacking

关键是把这些点**量化成指标面板**，并在每个阶段输出“面试官能追问也站得住”的证据链。

---

## 1) 你最终要交付的“指标面板”（主表 + 曲线）

指标分 4 类：质量、稳定性、成本效率、系统可靠性。简历不必全写，但 repo/报告里必须有。

### 1.1 质量（效果）指标（必须）

**主评测（贯穿）**：CodeContests_valid / CodeContests_test

* **accepted@1**：一次生成是否全通过（主指标之一）
* **pass_ratio**：通过测试数 / 总测试数（dense 信号；报告 mean、P50、P90）
* **pass@k（k=5/10 可选）**：同题采样 k 次成功率（很加分）
* **exec-success rate**：能否正常运行（不语法错/不崩溃/不超时）
* **error breakdown**：Syntax / Runtime / Timeout / Wrong Answer 比例（非常能体现工程能力）

**标准对标 Test（必须加入）**：HumanEval（只做 test，不训练）

* **pass@1**（可选 pass@5/10）

**轻量回归 Dev/Val（必须加入）**：MBPP 回归子集（例如 100–200 题）

* pass@1 / solved-rate（回归监控）
* error breakdown（Syntax/Runtime/Timeout）
* 输出合规率（是否只输出代码/函数题格式是否稳定）

> 这样你既有：**竞赛主线（CodeContests）** + **行业对标（HumanEval）** + **快速回归（MBPP）**，工业可信度明显提升。

---

### 1.2 稳定性（RL 训练是否可控）指标（强烈建议）

* **2 seeds：mean ± std**（最少门槛，建议对 CodeContests_test accepted@1 报）
* 训练曲线（至少保存到 repo）：

  * reward mean / reward std
  * KL（均值/最大值）
  * timeout rate 随训练变化（很关键）

---

### 1.3 成本效率（工业界非常看重）

* avg output tokens / problem
* avg execution_time / problem（判题耗时）
* throughput（problems/sec 或 tests/sec）
* **cost-per-solved**：每解出 1 题平均花多少 tokens + 判题时间（最贴工业视角）

---

### 1.4 系统可靠性/可运维（加分项）

* 并发判题下：queue length / timeout rate / retry rate
* cache hit rate（如果做了缓存）
* 自动回归：regression fail count；异常比例过高 early stop

---

## 2) 数据与划分（主线 CodeContests + 外部对标 HumanEval + 回归 MBPP）所以数据集在

### 2.1 三套数据集的**明确角色**（写死在设计里）

**CodeContests（主线）**

* Train：CodeContests_train（训练/构造偏好对）
* Dev/Val：CodeContests_valid（训练中高频回归、早停、选超参）
* Test：CodeContests_test（只在阶段结束评测，不调参）

**HumanEval（标准对标，只做 Test）**

* Test only：HumanEval 全量 164（只在阶段结束评测）

**MBPP（轻量回归子集，只做 Dev/Val 监控）**

* MBPP_reg：固定 100–200 题（训练中高频跑，监控退化/格式）
* 不作为主结论（除非你完全不在 MBPP 上训练且严格独立拆分）

> 额外两条防质疑：

* 固定 decoding 协议（训练用/评测用分开写清楚）
* 记录：seed、checkpoint、评测参数、题目 ID 列表版本（可复现）

---

## 3) 实验步骤（每一步产出哪些指标/图表；哪里加工程细节）

我按 Phase 0/1/2/3 写成“执行手册”。每阶段都包含：目的、做什么、必须产出、可加分点。

---

### Phase 0：Baseline（基线评测）

**目的**：Base → SFT → (DPO) → GRPO 全链路曲线从这里起。

**做什么**：base 7B（未训练）评测三套：

* Dev/Val：CodeContests_valid + MBPP_reg
* Test：CodeContests_test + HumanEval

**必须产出指标**

* CodeContests_valid/test：accepted@1、pass_ratio(mean/P50/P90)、exec-success、error breakdown
* MBPP_reg：pass@1 + error breakdown（作为回归基线）
* HumanEval：pass@1（对标基线）
* 成本：avg tokens、avg execution_time、throughput、cost-per-solved（至少对 CodeContests）

**可加分点（工程）**

* 记录 problems/sec 与 p95 execution_time（后续系统优化有对照）

---

### Phase 1：SFT（让 IO 格式与可执行性先稳定）

**目的**：降低 Syntax/Runtime/Timeout，提升 exec-success，让 RL 不浪费预算在低级错误。

**做什么**

* 训练：CodeContests_train（可先 easy/短题子集 warm-up，再扩到更大子集）
* 回归（高频）：CodeContests_valid + MBPP_reg
* 阶段结束测试（低频）：CodeContests_test + HumanEval

**这一阶段必须产出的指标（岗位很爱看）**

* exec-success rate ↑（最重要）
* Syntax/Runtime/Timeout 显著下降（error breakdown）
* accepted@1、pass_ratio 小幅提升或持平都合理（重点是“能跑”）
* HumanEval pass@1：阶段末跑一次，证明“没退化/可对标”

**可加分实验/工程细节（选 1–2 个）**

* 模板合规率：只含代码、stdin/stdout 格式正确
* Failure analysis：抽样 50 个失败 case 分类归因（I/O 格式、超时、逻辑错）

---

### Phase 2（可选但强烈建议）：Offline DPO（偏好对冷启动）

**目的**：用离线偏好数据把策略推向“更可能通过更多测试点”的分布，让 GRPO 更稳、更省 rollouts。

**做什么**

* 用 SFT checkpoint，在 CodeContests_train 每题采样 K=2~4 个解
* 判题得到 pass_ratio，构造偏好对：

  * chosen：pass_ratio 最高（并列选更短/更快）
  * rejected：pass_ratio 最低（优先 Syntax/Timeout）
* 过滤噪声：Δpass_ratio ≥ 0.3；“全 0 vs 全 0”丢弃
* 用偏好对训练 DPO

**必须产出的指标（让 DPO 不是摆设）**

* CodeContests_valid：pass_ratio(mean/P50/P90) ↑；zero-reward rate ↓（pass_ratio=0 的比例）
* 成本：DPO 数据构建判题成本（总判题次数、每题平均判题次数）
* MBPP_reg：回归不退化（至少稳定）

**可加分点**

* 偏好数据质量报告：Δpass_ratio 分布、chosen/rejected 的 error_type 分布
* pair 构造策略小消融（是否加入 execution_time tie-breaker）

---

### Phase 3：Online GRPO（主训练）

**目的**：直接优化可验证奖励，让模型学会“通过更多测试点”，并在可控成本下提升 accepted@1。

**做什么**

* 训练：CodeContests_train（建议课程学习：easy 子集 → 全量/更难子集）
* 验证：CodeContests_valid（高频）+ MBPP_reg（防漂移）
* 测试：CodeContests_test + HumanEval（阶段结束）

**必须固定做的 2 个 variant（简历最值钱的消融）**

* GRPO-dense：reward = pass_ratio（主推）
* GRPO-sparse：reward = 1[accepted]（对照）

**这一阶段必须产出的核心结果**

* CodeContests_test：accepted@1（主结果）、pass_ratio(mean/P50/P90)、可选 pass@k
* HumanEval：pass@1（对标 & 泛化检测）
* 稳定性：2 seeds mean±std；reward/KL/timeout 曲线
* 成本效率：avg tokens、avg execution_time、problems/sec、cost-per-solved
* 消融结论：dense vs sparse（更快收敛/更低方差/更少 rollouts 到阈值 —— 选一个你能量化的）

**可加分工程细节（最贴岗位，选 2–3 个）**

* 并发判题 + 限流（吞吐↑、timeout↓）
* 缓存/短路：非代码/空输出/超长输出短路；重复 code hash 缓存（报 cache hit rate）
* 自动化回归：每 N steps 跑固定 valid 子集与 MBPP_reg，生成报表；timeout rate 飙升 early stop
* Failure-driven data flywheel：把 Timeout/Syntax 失败回流（做 DPO hard negatives 或小比例 format repair SFT）

---

### Phase 4（可选加分）：多轮修复（2–3 轮，轻量 Agentic 扩展）

> 说明：Phase 4 建议作为 **Phase 3（GRPO-dense）跑通后**的加分扩展，核心价值是展示你具备 **tool/exec feedback 驱动的 agentic loop** 能力，并且能量化 **收益 vs 成本**。

**目的**

* 引入“执行反馈 → 修复迭代”的轻量多轮 loop，让模型从 **partial correctness → full correctness** 的修复过程更稳定；
* 对齐工业 coding agent / post-training 中常见的“反复跑测试-修复”范式；
* 用明确指标证明：多轮带来增益，但你也能控制成本与不确定性。

**做什么（流程：Mode A 自动判题）**

* **Round 1**：生成 `code_1` → 判题得到 `accepted_1 / pass_ratio_1 / error_type_1 / exec_time_1`，并产出短反馈摘要 `feedback_1`
* **Round 2**：把 `feedback_1` 拼接到 prompt（只保留摘要，不堆日志）→ 生成 `code_2` → 再判题得到 `accepted_2 / pass_ratio_2 / error_type_2 / exec_time_2`
* **Round 3（可选）**：同上（仅当 Round 2 仍未通过且 `pass_ratio_2` 有提升空间时启用）

**反馈摘要（强约束、短、结构化）**

建议固定字段并控制长度（例如 200–400 tokens）：

* `error_type`（SYNTAX/RUNTIME/TIMEOUT/WA）
* `pass_ratio = passed/total`
* `failed_cases`（最多 K 个）
* `traceback_tail`（最后 N 行）
* `exec_time`（可选）

**Reward/Return（两种推荐，二选一即可）**

* **方案 A（最稳）**：用最终轮结果 + 轮数成本
  `R = pass_ratio_last + 0.2 * I[accepted_last] - λ * (rounds_used - 1)`
* **方案 B（过程友好）**：每轮即时奖励 + 轮数成本
  `R = Σ_t pass_ratio_t + 0.2 * I[accepted_last] - λ * (rounds_used - 1)`

> `λ` 建议很小（0.02–0.05），只表达“额外轮次的真实成本”，避免模型拒绝修复。

**Phase 4 必须记录的指标（和 Phase 3 区分开）**

在 **CodeContests_valid/test** 上至少记录：

* **accepted@multi（最终轮）**：多轮后是否通过（主指标）
* **recovery rate**：Round 1 未通过但 Round 2/3 通过的比例（最能体现修复价值）
* **Δpass_ratio**：`pass_ratio_last - pass_ratio_1`（mean / P50 / P90）
* **avg_rounds_used**：平均轮数（成本）
* **tokens/prob（分解到每轮 + 总和）**
* **execution_time/prob（分解到每轮 + 总和）**
* **cost-per-solved（multi-round）**：成功解题的平均 tokens + 判题时间（与单轮对比）
* **error transition**：error_type 从 Round1→Round2 的转移（例如 SYNTAX→WA、TIMEOUT→WA）

并在回归/对标上保持：

* **MBPP_reg**：pass@1 与 error breakdown（防格式/基础能力退化）
* **HumanEval（test only）**：pass@1（阶段结束跑一次，展示泛化不退化）

**Phase 4 必须做的对照（否则很难说服面试官）**

1. **单轮（Phase 3 GRPO-dense） vs 多轮（Phase 4）**：同 checkpoint、同解码参数、尽量同预算对比
   报：accepted@1/accepted@multi、recovery rate、cost-per-solved
2. **2 轮 vs 3 轮（可选）**：证明边际收益是否值得额外成本

**可加分点（工程/实验细节，选 1–2 个）**

* 只对 Round 1 的失败样本启用 Round 2（“按需修复”）并报告节省的成本
* 对明显无效输出短路（空/非代码/超长）减少无意义判题
* 在 README 加一节 “Case Study”：展示 3–5 个典型题目从错误→修复→通过的全过程（结构化日志）

---

## 4) 最小可行实验矩阵（能写简历 + 不超预算）

**必须（5 组）**

1. Base（valid/test + HumanEval test + MBPP_reg）
2. SFT（同上）
3. **强对照 baseline**：SFT checkpoint + best-of-n（inference only，成本对齐口径下对比）
4. SFT + GRPO(dense)（同上）
5. 消融：SFT + GRPO(sparse)（同上）

**强烈建议加分（2 组）**
6) SFT→DPO→GRPO(dense) vs SFT→GRPO(dense)
7) curriculum vs no-curriculum（只改采样策略）

**Phase 4 加分项（推荐至少做 1 组）**
8) 单轮 GRPO(dense) vs 多轮修复(2 rounds)（对比收益 vs 成本）

---

## 5) 你简历上怎么写（岗位语言 + 量化指标 + 工程闭环）

**项目标题（建议）**
**RLVR Coding Post-Training (Offline DPO + Online GRPO, Verifiable Judge Reward)**

**3–4 条 bullet（直接替换数字）**

* Built an end-to-end LLM post-training pipeline with verifiable rewards from judging feedback; implemented **SFT → (offline DPO) → online GRPO** with strict train/dev/test isolation.
* Improved **CodeContests_test accepted@1** from **X% → Y%** (**2 seeds, mean±std**) and increased **pass_ratio P50/P90** by **Δ**, demonstrating stronger partial-to-full correctness under dense verifier feedback.
* Ran key ablations: **dense (pass_ratio) vs sparse (accepted)** reward and **with/without DPO warm-start**, showing faster convergence / lower variance / fewer rollouts to reach target performance.
* (Optional) Added a lightweight **multi-round repair loop (2–3 iterations)** using structured execution feedback, improving **recovery rate** by **Δ** while quantifying **tokens/judge-time trade-offs**; monitored regressions on **MBPP_reg** and external benchmarks on **HumanEval**.

---

## 6) 最终“结果清单”（repo/报告必须有的那几页）

* 主表：Base/SFT/DPO/GRPO 在 **CodeContests_test + HumanEval + MBPP_reg** 的指标与成本面板
* 学习曲线：accepted@1 / pass_ratio / reward / KL（2 seeds + 均值）
* 消融表：dense vs sparse；DPO vs none；（可选）curriculum vs none
* 错误类型分布：Syntax/Runtime/Timeout/WA 随阶段变化
* 成本面板：problems/sec、p50/p95 execution_time、timeout rate
* Failure analysis：典型失败 case + 修复策略（很加分）
* **Phase 4 专属对照页**：单轮 vs 多轮（accepted、recovery rate、Δpass_ratio、avg_rounds、cost-per-solved）

---

补充：设计原则：
* HumanEval：test-only 禁止用于训练、构造偏好对、选择超参、早停
* CodeContests_test：test-only 禁止训练与调参
* CodeContests_valid：只用于 checkpoint 选择/早停/超参对比
* 统一解码协议 训练 rollout（temperature/top_p）与评测 decoding（greedy/低温）必须分别固定并记录
* 固定回归集题号（MBPP_reg + CodeContests_dev subset）训练过程中回归评估必须使用“固定题号列表”，否则趋势不可比
