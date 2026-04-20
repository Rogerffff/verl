# 中文简历 Bullet 面试问答素材库（2026-04-20）

## 0. 用途

这份文档配套：

- [resume_bullet_candidates_cn_2026-04-20.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/resume_materials/resume_bullet_candidates_cn_2026-04-20.md)

目标不是压低 bullet 强度，而是为每条高价值 bullet 预先准备：

1. 面试官最可能追问的问题
2. 这些问题实际在 probe 什么
3. 候选人应该怎么答，既显得扎实，又不把项目讲弱
4. 哪些点容易答崩，需要提前避坑

---

## 1. 使用方式

- 如果要准备 `Applied Research / LLM Post-training`
  - 优先看第 2 节
- 如果要准备 `ML Systems / Infra / Evaluation`
  - 优先看第 3 节
- 如果要准备综合型面试或 hiring manager 面
  - 优先看第 4 节

---

## 2. 研究向问答

### H01

- `高频追问`
  - 你说是完整后训练体系，链路到底长什么样，哪些是你亲自搭的，哪些是复用现有框架？
  - 为什么不是只继续做 RL，而要引入 repair 和 repair-SFT？
  - 你怎么证明这不是很多零散实验的拼接？
- `面试官在 probe`
  - 看你是否真的做了端到端整合，以及研究路线是否是被数据推动而不是拍脑袋扩项目。
- `强回答要点`
  - 先讲清 `curriculum RL -> shared verifier -> raw regression -> Protocol A/B repair eval -> teacher generation/QC -> repair-conditioned SFT` 这条链。
  - 明确 `verl + SandboxFusion` 是复用底座，你自己补的是 shared verifier 契约、curriculum 数据闭环、repair 协议、teacher/QC 与 SFT 数据链。
  - 用 late-stage RL 的现象解释路线转向：后期 checkpoint 更像把题推向 partial correctness，而不是稳定转成 AC，所以才引入 one-turn repair。
  - 不把话说满：当前证据支持 raw RL、repair、repair-SFT 都跑通了，但 held-out 上最强 deployed base 仍是 RL checkpoint，而不是 SFT 版。
- `容易翻车的点`
  - 把“完整体系”说成“所有环节都在 held-out 上已经 fully 成功”。
  - 把现有框架里的内容也包装成自己从零搭建。

### H02

- `高频追问`
  - 你说是可复现研究流水线，可复现具体体现在哪些 frozen artifacts 和 protocol 上？
  - teacher-QC 为什么不是 teacher 一次生成过了就收？
  - sandbox 稳定性治理做了什么，为什么还敢信结果？
- `面试官在 probe`
  - 看 reproducibility、数据治理和系统治理是否真的落到了 artifact 层。
- `强回答要点`
  - 直接点名 canonical 资产：full-val baseline、raw regression summary、Protocol A/B contract、quarantine v3 ledger、handoff、inventory、claim-evidence map。
  - 说明 teacher 数据不是一次性脚本：step900 形成 `182` 条 `accept_2_of_2` 纯 shortdiag 数据；step1300 则是 `553` 条 keep，来自 primary QC、regen、testcase/judge salvage。
  - sandbox 这条线不是“已经零噪声”，而是“主要竞态已定位并修补，剩余噪声已量化并写进协议解释里”。
- `容易翻车的点`
  - 把“可复现”讲成 deterministic。
  - 忽略 residual judge instability 仍然存在这一事实。

### H03

- `高频追问`
  - 如果这是你独立推进的项目，你做过的 owner-level 决策是什么？
  - 模型、判题、数据、运维都要管时，你是怎么排优先级的？
  - 哪些是你自建，哪些是站在现有框架上做的？
- `面试官在 probe`
  - 看 ownership 是否真实，是否会因为想显得厉害而过度抢功。
- `强回答要点`
  - 最重要的判断不是“继续训”，而是先统一 ground truth 和评测协议，再决定是否扩大训练或 SFT。
  - 举真实例子：`step1300_sft_v1_step60` 在 dev 上有增益，但 held-out 没超过 `step1300_rl`，所以没有直接把 v1 配方放大。
  - 清楚区分底座与自建：训练框架和沙箱来自现有 repo，自己做的是 protocol、verifier、curriculum、QC、repair、远端实验运维。
- `容易翻车的点`
  - 听起来像“我从零实现了整个训练框架和沙箱”。

### H04

- `高频追问`
  - 你说同时负责模型、评测系统、数据管线和远端编排，这条链具体是什么？
  - 最难的非模型问题是什么？
  - 你怎么证明它从“能跑”变成“可解释、可复现、可交接”？
- `面试官在 probe`
  - 看系统整合能力、debug 深度和 handoff 意识。
- `强回答要点`
  - 直接把闭环讲成 `curriculum-aware RL -> shared verifier eval/reward -> one-turn repair -> teacher generation/QC -> repair-SFT parquet`。
  - 非模型难题举 sandbox/judge 漂移最合适，因为它逼你把问题拆成 protocol、judge、数据、模型四层。
  - “可解释、可复现、可交接”来自 Protocol A/B、fixed-response rejudge、run_info/per_problem/summary、handoff/inventory/runbook 的联合落地。
- `容易翻车的点`
  - 范围说得很大，但落不到一份 schema、一个接口或一份结果文件。

### R01

- `高频追问`
  - `1.8% -> 7.9%` 具体是什么 set、什么协议、什么模型对比？
  - 除了 accepted@1 还有什么指标一起变了？
  - 你怎么验证模型没被训成 narrow specialization？
- `面试官在 probe`
  - 看 headline 是否经得住拆、是否只会背一个漂亮数字。
- `强回答要点`
  - 这是 held-out `CodeContests test` 的 raw 生成结果，题量 `165`；base `Qwen2.5-Coder-7B-Instruct` raw `accepted@1 = 1.818%`，`step1300_rl` raw `accepted@1 = 7.879%`。
  - 同时 `pass_ratio_mean 0.1289 -> 0.2501`、`exec_success_rate 0.691 -> 0.824`，说明不是只多对了几题。
  - general-code guardrail 方面，`HumanEval 87.2% -> 89.0%`、`MBPP 58.0% -> 62.5%`，所以更稳的讲法是“当前 recipe 没明显把模型训坏”。
- `容易翻车的点`
  - 把 dev 集 `13.8%` 和 held-out `7.9%` 混为一谈。

### R02

- `高频追问`
  - 为什么敢说是“可部署 RL checkpoint”？
  - `4.3x` 背后的绝对数是多少？
  - 为什么不是把 exact-solve winner 当 deployed base？
- `面试官在 probe`
  - 看你是否会用倍率包装小结果，以及 checkpoint 选择逻辑是否清晰。
- `强回答要点`
  - 这里的 deployable 只能指“在当前 held-out + repair pipeline 下最值得部署的 base”，不是产品 SLA。
  - 绝对数大约是 `3/165 -> 13/165`，多了约 `10` 道 exact solves。
  - exact-solve winner 和 deployed base 可以不是一个模型：dev 上 solve winner 是另一点位，但 held-out raw + repair pipeline 下更强的是 `step1300_rl`。
- `容易翻车的点`
  - 把“deployable”说成“已经可以做真实在线竞赛产品”。

### R03

- `高频追问`
  - 500 题 checkpoint review 到底回答了什么问题？
  - 为什么 best exact-solve checkpoint 不等于你后面一直使用的 checkpoint？
  - rerun / judge drift 怎么处理？
- `面试官在 probe`
  - 看你是否区分 dev checkpoint selection 与最终 deployment judgment。
- `强回答要点`
  - 500 题 review 是 dev-side checkpoint selection，不是最终部署结论。
  - best exact-solve checkpoint 达到 `69/500 solved`、`13.8% accepted@1`，但后续还要结合 partial-credit、held-out、repair utility 来选 deployed base。
  - 对 rerun / judge drift 的处理不是“挑一次好看的”，而是依赖 fixed-response rejudge 和 protocol-matched canonical results。
- `容易翻车的点`
  - 直接说“最佳 checkpoint 就是 X”，不区分 exact-solve 与 partial-credit / held-out。

### R04

- `高频追问`
  - end-to-end self-repair 协议具体是什么？
  - 这个 gain 到底多大？
  - 这个 gain 有多少来自更宽的 trigger？
- `面试官在 probe`
  - 看你是否懂 `Protocol B` 的含义，以及是否会主动讲 caveat。
- `强回答要点`
  - `Protocol B` 是每个模型先自己生成 first pass，再对自己的错代码做 one-turn repair，所以回答的是 practical deployment utility。
  - held-out `CodeContests test` 上，`step1300_rl` 从 `7.88%` 提升到 `10.91%`，约 `+5 solved`，`repair_attempt_count = 150`，`conditional_repair_success = 3.33%`。
  - 要主动补一句：这不是纯 repair skill，比 `Protocol A` 更接近部署口径，也更受 trigger 设计影响。
- `容易翻车的点`
  - 把 `10.9%` 直接表述成“纯修复能力大幅提升”。

### R05

- `高频追问`
  - 为什么要挑 near-miss 子集，而不是看全量失败池？
  - `26.3%` 怎么算出来的？
  - repairable failures 长什么样？
- `面试官在 probe`
  - 看你是否理解 repair success 的结构性，而不是只背一个漂亮比例。
- `强回答要点`
  - 这是高价值 slice，不是全局 repair 成功率。
  - `19` 次 repair attempt 中成功 `5` 次，所以 `conditional_repair_success = 26.3%`；first-pass `65.2%`，after-repair `72.46%`。
  - 这一批成功几乎都来自高 partial bucket，更像局部逻辑修补，不是从完全不会直接修成 AC。
- `容易翻车的点`
  - 把 `26.3%` 当作全数据集 repair 成功率宣传。

### R06

- `高频追问`
  - repair-conditioned SFT v1 的实际 dev gain 是多少？
  - 为什么还不能算成功？
  - 它给下一轮 recipe 带来了什么判断？
- `面试官在 probe`
  - 看你是否愿意承认“dev gain 有，但 held-out 不够”的中间态结果。
- `强回答要点`
  - `step1300_sft_v1_step60` 在 `valid_big500` 上 `Protocol B 0.130 -> 0.144`，`conditional_repair_success 3.02% -> 4.52%`；`Protocol A 0.146 -> 0.150`。
  - 但 held-out `codecontests_test` 上 `Protocol B 9.70% < 10.91%`，所以它更像 dev-side specialization probe，而不是 deployed winner。
  - v2 的设计因此转向更平衡的数据配方，而不是简单把 v1 继续放大。
- `容易翻车的点`
  - 只讲 dev 增益，不讲 held-out 没赢过 RL base。

### R07

- `高频追问`
  - 为什么要强调 `raw solve + repair gain` 双层框架？
  - 这个框架真的改变过项目决策吗？
  - 怎么避免双重计算？
- `面试官在 probe`
  - 看 measurement 框架是否真的被用来做决策。
- `强回答要点`
  - raw solve 反映 first-pass 能力，repair gain 反映 second-pass / pipeline utility，两者不拆会把“更会生成”和“更会修”混为一谈。
  - 这个框架直接影响了对 `step1300_sft_v1_step60` 的判断：它在 dev repair 上看起来更强，但没有超过 `step1300_rl` 的 held-out deployed result，所以没有被误升级为主模型。
  - `Protocol A` 固定输入，`Protocol B` 看端到端，raw regression 独立，不互相顶替。
- `容易翻车的点`
  - 把 final repaired acc 直接当作 raw 生成能力提升。

### R08

- `高频追问`
  - 为什么 exact-solve 和 partial-credit 都要看？
  - 你这里最典型的分歧例子是什么？
  - 如果目标不同，你会怎么选 checkpoint？
- `面试官在 probe`
  - 看你是否理解 late-stage RL 的常见动态。
- `强回答要点`
  - late-stage RL 经常把更多题推向 near-correct，但不一定稳定转成 AC，只看 solved 或只看 partial-credit 都会误导。
  - 当前项目里就出现了 exact-solve winner 和 partial-credit stronger checkpoint 不同的情况。
  - 如果目标是 benchmark solve count 选 solve winner；如果目标是 held-out deployed base + self-repair utility，则看综合表现更强的 raw base。
- `容易翻车的点`
  - 说成“partial-credit 更高就一定更好”。

### R09

- `高频追问`
  - 你凭什么说不是 narrow specialization？
  - 除了 benchmark 分数，还看了什么？
  - 这个 claim 应该怎么保守表述？
- `面试官在 probe`
  - 看你是否会把“没明显回归”错误包装成“已证明广泛泛化”。
- `强回答要点`
  - 回归集没掉，`HumanEval / MBPP` 还有轻微上涨，这是最好的守法证据。
  - 同时 raw `CodeContests test` 的 `exec_success_rate`、`pass_ratio_mean` 也同步提升。
  - 最稳妥的说法是“当前 recipe 没把模型训成只会这个 slice 的窄专长”，而不是“已经证明广泛泛化”。
- `容易翻车的点`
  - 把回归不掉说成通用代码能力全面增强。

### R10

- `高频追问`
  - full-val baseline 的价值是什么？
  - baseline 给了什么研究判断？
  - baseline 阶段有没有做 reward 设计分析？
- `面试官在 probe`
  - 看你是否重视 baseline，而不是一开始就盲目开训。
- `强回答要点`
  - baseline 给出统一起点：`HumanEval 87.2% / MBPP 58.5% / CodeContests_valid 2.56%`。
  - 更重要的是它暴露出 `CodeContests_valid` 只有很低 AC，但有大量题 `pass_ratio > 0`，这直接支撑了 dense reward 和 repair 的可行性。
  - baseline 还做过 reward 设计对比，帮助避免过早把 dense signal 压得过稀。
- `容易翻车的点`
  - 把 baseline 说成只是“跑个 sanity check”。

### J01

- `高频追问`
  - 为什么 repair 要拆成 fixed-input skill 和 end-to-end self-repair 两类问题？
  - 这两种协议下 model ordering 会变化吗？
  - 如果只看其中一个，会错什么决策？
- `面试官在 probe`
  - 看你是否理解“评测问题的定义”本身。
- `强回答要点`
  - `Protocol A` 回答“同一份坏代码谁更会修”，`Protocol B` 回答“上线后整条 pipeline 谁更强”，问题本身不同。
  - 当前项目里确实出现了 ordering 不完全一致的情况。
  - 只看 dev fixed-input 会高估 SFT v1 的部署价值，只看 end-to-end 又看不清 second-pass skill 是否真的提升。
- `容易翻车的点`
  - 说不清 `reuse_step900` / canonical raw 和 `selfraw` 的差别。

### J02

- `高频追问`
  - “dev repair gain 不等于 held-out deploy gain”最硬的证据是什么？
  - 这个结论最后阻止了什么？
  - 后续 recipe 因此怎么改？
- `面试官在 probe`
  - 看你是否会因为 dev 上涨就急着宣称方案成功。
- `强回答要点`
  - 最硬证据是 `valid_big500` 上 `step60` 比 `step1300_rl` 强，但 `codecontests_test` 上反而 `0.0970 < 0.1091`。
  - 它阻止了把 v1 继续当作成功配方无脑放大。
  - v2 因此不是“再多训一点”，而是换更强 base、改数据分布、引入更稳的 keep / supplemental 设计。
- `容易翻车的点`
  - 只会背口号，不会背 dev/test 对照数字。

### J03

- `高频追问`
  - 分别给一个模型变化、judge 漂移、prompt/trigger 设计造成波动的例子。
  - 你怎么把 judge 漂移和模型变化剥离开？
  - 这些分析最后带来了哪些 protocol 改动？
- `面试官在 probe`
  - 看你是否能把复杂噪声拆解成不同来源。
- `强回答要点`
  - 模型变化看 raw regression 和 checkpoint review；judge 漂移看 fixed-response rejudge 与 sandbox targeted repro；prompt/trigger 影响看 Protocol A/B 和 trigger 口径差异。
  - 真正的 isolation 手段包括 reuse-first-pass、fixed-response rejudge、single backend / LB 对照、quarantine / testcase audit。
  - 这些结果最终变成了协议：canonical first-pass、Protocol A/B 分工、对小 solved 差异更保守。
- `容易翻车的点`
  - 把所有异常都归因给 judge，反而削弱自己的主结果。

### J04

- `高频追问`
  - 你怎么解释 late-stage RL？
  - 为什么说它更像往 partial frontier 推，而不是 solve 持续增长？
  - 这个判断改变了什么实验选择？
- `面试官在 probe`
  - 看你是否真的理解训练动态，而不是只看最后一张表。
- `强回答要点`
  - 后期训练不是 solve-count 单调增长，而是更多题从 `zero` 往 `mid/high` 桶移动。
  - fixed-response rejudge 进一步支持 `step1300` 更像 partial-credit stronger，而不是 solve winner。
  - 这直接把后续重点从“继续训 raw RL”切到“repair / repair-SFT 能不能把这部分 frontier 转成 AC”。
- `容易翻车的点`
  - 用单一指标把 late-stage RL 讲成“退化”或“继续稳定提升”。

### J05

- `高频追问`
  - 为什么说 one-turn repair 更像 near-miss fallback？
  - 但 held-out test 不是还有 bucket_0 的成功吗？
  - 这个判断怎样影响后续数据收集？
- `面试官在 probe`
  - 看你是否会被反例打穿。
- `强回答要点`
  - dev 上成功明显集中在高 partial bucket，所以 near-miss fallback 是合理主判断。
  - 但 held-out test 也出现过 bucket_0 成功，所以最准确的说法是“非均匀有效”，而不是“只对 near-miss 有效”。
  - v2 的切片设计因此不是均匀扫失败池，而是围绕 `CoreNearMiss / ExpansionB / CurrentCRecoverable / HardProbe` 分层。
- `容易翻车的点`
  - 说成“repair 只对 high-partial 有效”。

### J06

- `高频追问`
  - 早期几百步的 repair-SFT 线哪里失败了？
  - 你怎么知道不是 simply 训练步数不够？
  - 失败经验怎么反哺主线？
- `面试官在 probe`
  - 看你是否能诚实复盘失败，而不是事后合理化。
- `强回答要点`
  - 早期 SFT 线在 canary / 小集上有亮点，但 full valid 上没超过 pre-SFT，因此不能算成功。
  - 问题不只是 optimization 没收敛，更关键是 solve-set churn 和 anti-regression 没保住。
  - 之后更强调 full-suite / full-valid 才能拍板，cheap-screen 只做 shortlist。
- `容易翻车的点`
  - 把失败包装成“基本成功，只差多训几步”。

### J07

- `高频追问`
  - 为什么把 repair-SFT v1 定义成 dev-side specialization probe？
  - v2 真正不同在哪里？
  - 你会怎么防止 v2 重复 v1 的失败？
- `面试官在 probe`
  - 看你是否真的从一轮不完全成功的结果里学到了 recipe 判断。
- `强回答要点`
  - v1 在 dev 上有 gain，但 held-out 没超过 RL base，所以更像 specialization probe。
  - v2 不只是加数据，而是更强 base、更平衡 strata、更清楚的 strict / supplemental keep 和 pilot gating。
  - success criterion 也要同时看 `valid_big500` 和 `codecontests_test` 的 Protocol A/B，而不是只看 dev。
- `容易翻车的点`
  - 把 v2 讲成“把 v1 放大三倍就行”。

### M01

- `高频追问`
  - curriculum bucket 为什么能当整个数据闭环的中轴？
  - 为什么不直接按 pass_ratio 切数据？
  - 这个设计产生过什么可见价值？
- `面试官在 probe`
  - 看你是否真正把训练状态和数据生产连接起来。
- `强回答要点`
  - bucket 既驱动 RL 采样，又定义 near-miss / hard-partial 结构，后续 repair trigger、teacher queue、SFT 配比都围绕它组织。
  - pass_ratio 只是静态数值，bucket 还带 retention / failure frontier 的在线语义。
  - 这个设计帮助解释过训练中 `B` 桶被吃进更难桶的问题，也指导了后续 v2 再平衡。
- `容易翻车的点`
  - 只会讲 bucket 名字，不会讲它如何流到 repair / SFT。

### M02

- `高频追问`
  - 为什么统一成 `short-diagnosis + code`？
  - 这个格式是否已经被大规模证明明显强于 code-only？
  - 它具体省了什么工程成本？
- `面试官在 probe`
  - 看你是否会把格式选择过度包装成算法贡献。
- `强回答要点`
  - 这个格式兼顾 grounded reasoning 和稳定 extraction，便于 eval、teacher 输出、SFT 数据全链统一。
  - 证据是有限支持，不是压倒性证明，所以面试里不要把它讲成核心科学结论。
  - 工程收益是真实的：teacher-QC、解析、数据装配、训练输入都能用同一套 schema。
- `容易翻车的点`
  - 把 shortdiag 讲成已被严格证明显著优于 code-only。

### M03

- `高频追问`
  - 什么叫把 prompt 实验升级成 trainable data paradigm？
  - 有哪些 artifact 证明这个闭环真搭起来了？
  - 这条闭环还有哪些环节其实没完成？
- `面试官在 probe`
  - 看你能否把“prompting”讲成可持续数据系统，而不是一次性技巧。
- `强回答要点`
  - 核心不是 prompt trick，而是把 `题面 + 错代码 + verifier feedback -> 短诊断 + 修复代码` 固化成可生成、可 QC、可训练、可再评测的数据任务。
  - 对应 artifact 有 repair eval、teacher keep set、shortdiag SFT parquet、training guide 和回测结果。
  - 也要保留边界：held-out deploy gain 还没被 SFT 跑赢，multi-turn 和 repair-distilled single-turn 仍是后续设计。
- `容易翻车的点`
  - 把设计文档里的未来阶段说成已经完成。

### M04

- `高频追问`
  - raw RL 和 post-execution self-repair 是互补还是冲突？
  - 你这里最典型的互补证据是什么？
  - 最典型的冲突证据是什么？
- `面试官在 probe`
  - 看你是否能同时承认 synergy 和 trade-off。
- `强回答要点`
  - 两者都可能发生：更强 raw first pass 能给 repair 留出更高价值的 frontier，但 repair-SFT 太窄也可能伤害 held-out deployed mix。
  - 互补证据是 `step1300_rl` raw held-out `7.88%`，加 self-repair 到 `10.91%`。
  - 冲突证据是 `step1300_sft_v1_step60` 在 dev repair 上更强，但 held-out `Protocol B 9.70% < 10.91%`。
- `容易翻车的点`
  - 给出“永远互补”或“天然冲突”这种一刀切结论。

---

## 3. 系统向问答

### S01

- `高频追问`
  - 为什么必须拆 Protocol A 和 Protocol B？
  - 如果不拆，会具体错判什么？
  - 这两个 protocol 最后分别支持了哪些项目决策？
- `面试官在 probe`
  - 看实验设计意识，以及你是否真的用协议来约束结论。
- `强回答要点`
  - `Protocol A` 固定 canonical first-pass artifact，只比 second-pass repair；`Protocol B` 让模型先自己生成再自己修，回答部署式 end-to-end utility。
  - 如果不拆，raw first-pass 更强的模型会被误读成 repair skill 更强。
  - 当前 headline 的分工是：纯 repair 比较看 `Protocol A`，部署式自修复看 `Protocol B`。
- `容易翻车的点`
  - 把 `Protocol A` 说成“更科学，所以全都该看它”。

### S02

- `高频追问`
  - reuse-first-pass 在代码里怎么实现？
  - 你怎么保证复用的 first-pass 真的和当前样本一一对应？
  - 复用旧 artifact 时有哪些边界问题？
- `面试官在 probe`
  - 看实现深度、contract enforcement 和数据脏点意识。
- `强回答要点`
  - `phase4_repair_eval.py` 支持直接加载固定 `per_problem`，跳过 first-pass 生成与判题。
  - 校验包括 `dataset`、`problem_id`、`prompt_sha256` 等，避免 prompt 漂移或错行复用。
  - 对原始落盘 char cap 造成的截断 source 会单独标记并统计，避免把坏 first-pass 再喂回 repair。
- `容易翻车的点`
  - 把 reuse 讲成“只是读一个 JSONL”。

### S03

- `高频追问`
  - fixed-response rejudge 的实验设计是什么？
  - 你量化出的结论到底是什么，不是什么？
  - 这些结果怎样影响你解释 checkpoint 能力变化？
- `面试官在 probe`
  - 看控制变量能力和结果边界感。
- `强回答要点`
  - 同一批固定 response 重判两次，把 judge 漂移和生成漂移拆开。
  - `step1300` rejudge1 / rejudge2 的 `accepted@1` 和 `pass_ratio_mean` 很接近，说明 residual judge instability 还在，但量级小于 raw generation drift。
  - 这使得你对 `step1300` 的正确解释是“partial-credit 更强，但 exact solve 仍不赢 solve winner”。
- `容易翻车的点`
  - 说成“judge 已经 deterministic”。

### S04

- `高频追问`
  - 你是怎么把问题定位到 sandbox / LB 路径而不是模型本身的？
  - 真正修的是什么竞态？
  - 为什么最后收敛到 patched sandbox + direct-backend client RR？
- `面试官在 probe`
  - 看系统 debug 的方法论和代码级修复能力。
- `强回答要点`
  - 用 single backend、single-upstream LB、multi-upstream LB 的对照，把“像模型不稳定”的现象定位成输出截断 / 判题不稳定。
  - 根因在 SandboxFusion 的 kill-before-drain 和极短输出读取 timeout，修补点在 `sandbox/runners/base.py` 与 `sandbox/utils/execution.py`。
  - 后续正式评测口径改成 patched sandbox + direct backend URL，由客户端 RR，但仍保留“高并发下有 residual mismatch”的保守解释。
- `容易翻车的点`
  - 说成“彻底修好 judge，已经完全稳定”。

### S05

- `高频追问`
  - shared verifier 统一了哪些契约？
  - eval、RL reward、teacher QC 为什么必须共享同一套 judge 语义？
  - 这套共享层的核心数据结构是什么？
- `面试官在 probe`
  - 看平台化抽象和口径统一意识。
- `强回答要点`
  - 统一了代码提取、candidate normalization、CodeContests 判题、invalid-for-RL 判定。
  - 否则 raw、repair、teacher-QC、SFT 会建立在不同真值假设上，后续无法解释增益来源。
  - 核心结构是 `VerificationSummary`，包含 `accepted/pass_ratio_all/error_type/invalid_for_rl/per_case_results` 等字段。
- `容易翻车的点`
  - 只会说“统一了 judge”，却说不出具体字段或 consumer。

### S06

- `高频追问`
  - 为什么“第一条失败样例”会误导 repair？
  - 你改成了什么 deterministic rule？
  - 有什么证据说明这不是拍脑袋改 prompt？
- `面试官在 probe`
  - 看你对失败摘要质量和存储截断问题是否真正有证据。
- `强回答要点`
  - 审计发现 `per_case_results` 里 `stdin / stderr / expected / actual` 会受落盘 cap 截断，直接取第一条失败常常把坏 case 喂给模型。
  - WA 用 `simplest_counterexample`，RE 用 `clearest_exception_then_shortest_stdin`，TLE 用 `shortest_timeout_stdin + complexity hint`。
  - 审计样本规模、disagreement 比例和 selected-stdin 长度下降都能拿出来说。
- `容易翻车的点`
  - 只说“prompt 更清晰了”，不提 storage truncation 和具体审计结果。

### S07

- `高频追问`
  - 多机迁移时你认定哪些目录是 source-of-truth，哪些是 runtime junk？
  - 新机器 bring-up 后怎么做完整性检查？
  - 迁移的不只是代码，还包括哪些实验状态？
- `面试官在 probe`
  - 看运维成熟度和实验资产意识。
- `强回答要点`
  - 真正迁移的是 repo、checkpoint、curriculum state、teacher shards/outputs、sandbox bring-up 方法和 canonical contract。
  - `/root/sandboxfusion-venv`、`/tmp/ray` 这类运行态不迁。
  - 完整性检查要落到 manifest、summary、期望计数，而不是只看目录有没有。
- `容易翻车的点`
  - 把“迁移”讲成单纯 scp 目录。

### S08

- `高频追问`
  - watcher / launch guard 具体防什么故障？
  - 怎样避免重复 launch 或抢占 source run？
  - 你怎么确认 launch 真成功了？
- `面试官在 probe`
  - 看幂等、防重和可靠启动确认。
- `强回答要点`
  - watcher 不是只盯 checkpoint 文件，还要等 source 训练退出和 GPU 资源空闲。
  - 防重依赖 `flock`、launch marker、目标进程探测和目标 ckpt 目录探测。
  - 不是 `nohup` 一下就算成功，而是要在限定时间内确认 trainer 进程真的起来了再写 marker。
- `容易翻车的点`
  - 把它包装成“大型调度平台”。

### S09

- `高频追问`
  - 你写的 handoff / inventory 文档为什么不是项目日记？
  - 你如何区分 canonical source-of-truth 和历史资料？
  - 这些文档具体让别人少踩了什么坑？
- `面试官在 probe`
  - 看知识治理是否服务真实协作。
- `强回答要点`
  - handoff 的结构是“最新结论在前，旧背景下沉”，inventory 则专门做 authority hierarchy 与结果树导航。
  - 重要的是告诉后来人“先信哪份 contract、哪张表、哪类结果”，而不是堆日志。
  - 这些文档直接服务新机器 bring-up、简历写作、面试问答和继续实验，不是装饰品。
- `容易翻车的点`
  - 全讲“我很爱写文档”，但说不出文档如何改变后续人的行为。

### S10

- `高频追问`
  - 为什么敢把 teacher generation 叫“可审计流水线”？
  - 为什么要先 expand request 再切 shard，而不是让模型一次吐多个答案？
  - 这条线怎么处理 partial failure、regen 和稳定性确认？
- `面试官在 probe`
  - 看你是否有数据平台视角，而不是一次性脚本心态。
- `强回答要点`
  - request 会先扩成 generation unit，保留稳定主键和 provenance，之后才能做 shard、重试和 ledger 记账。
  - `Core` 与 `Expansion/C` 可以用不同 `best_of_n`，但最终都走统一 manifest、preflight、QC、regen、audit merge。
  - 讲具体计数最有说服力：比如 `281 requests -> 429 generation units -> 9 shards -> 267 stable keep`。
- `容易翻车的点`
  - 只会说 manifest / shard / ledger 这些词，却拿不出真实数字。

### D01

- `高频追问`
  - quarantine v3 的状态机是什么？
  - v3 相比早期版本升级了什么？
  - 这套体系怎么避免手工名单越来越乱？
- `面试官在 probe`
  - 看数据治理抽象是否成体系。
- `强回答要点`
  - 保持 `hard_blacklist / caution / unresolved` 三态，不把 clean-review 混成风险状态。
  - v3 把 quarantine review ledger、curriculum bucket actions、anchor actions 明确拆开，并引入 freeze manifest 与 queue_tag。
  - 冲突通过 ledger 规则解决，而不是靠口头约定覆盖。
- `容易翻车的点`
  - 把 quarantine 讲成“我拉了个 blacklist”。

### D02

- `高频追问`
  - 这批 blacklist / caution / unresolved 到底怎么消费到下游？
  - 你默认过滤哪些状态，为什么？
  - teacher-data 与 curriculum 如何接上 quarantine？
- `面试官在 probe`
  - 看数据治理是否真的落到 consumer，而不是只停在 ledger。
- `强回答要点`
  - consumer 脚本会默认移除 `hard_blacklist` 与 `caution`，也可选移除 `unresolved`。
  - 这样做是为了保证主训练、主评测与 teacher queue 都遵守同一 hygiene 契约。
  - step900 / step1300 的 teacher 构数计划都明确禁止 quarantine、eval overlap 与 truncated-source 样本进入主线。
- `容易翻车的点`
  - 只会背 blacklist 数量，不会说 consumer 如何使用。

### D03

- `高频追问`
  - request-unique / high-precision 具体是什么意思？
  - keep set 到最终 train set 中间做了哪些过滤？
  - 你怎么证明这不是 teacher 吐一遍就拿去训？
- `面试官在 probe`
  - 看数据主键意识、QC 深度和 precision-first 的严谨程度。
- `强回答要点`
  - request-unique 指 `row_count == distinct_request_count`，最终 train parquet 不保留同一 request 的多答案竞争。
  - keep set 需要经历 primary QC、stability rejudge、regen、testcase audit、final keep，最后再导出 pure shortdiag parquet。
  - step900 与 step1300 两条线都能拿出从原始 generation 到最终纯训练集的完整缩减路径。
- `容易翻车的点`
  - 混淆 keep set、request-unique set、最终 parquet。

### D04

- `高频追问`
  - 两版可训练 repair-SFT 数据分别是什么？
  - 为什么同时保留小而纯和大而广两版？
  - 这两版的分布差异是什么？
- `面试官在 probe`
  - 看你是否真正掌握数据资产。
- `强回答要点`
  - 小版是 step900 的 `182` 行 pure shortdiag，大版是 step1300 的 `553` 行 pure shortdiag。
  - 小版更像高精度 seed，大版保留了更多多轮补救带来的覆盖。
  - 最大差异在 strata 分布：step1300 明显更偏 `CurrentCRecoverable`，所以更像 hard-repair specialized corpus。
- `容易翻车的点`
  - 把这两版误说成 train / val split。

### D05

- `高频追问`
  - teacher reject taxonomy 怎么定义？
  - 哪些 reject 可以 salvage，哪些必须丢？
  - 给一个真实 audit 分布。
- `面试官在 probe`
  - 看 QC 体系是否细，是否真做过审计。
- `强回答要点`
  - reject 至少拆成 `teacher_logic_bug / testcase_issue / judge_instability / ambiguous_contract`。
  - 只有 testcase / judge 类才进入 salvage 流程，teacher bug 通常直接 reject。
  - near-miss 高通过率 audit 的真实标签分布和最终 salvage 数量是最好用的例子。
- `容易翻车的点`
  - 只会说“我们做了人工审查”，却说不出 taxonomy 和计数。

### D06

- `高频追问`
  - Core / Expansion / C 这些 strata 是后验分析标签还是前置决策变量？
  - 你怎么从 student reference 派生这些 strata？
  - 这些标签在哪里被保留下来了？
- `面试官在 probe`
  - 看分层是不是实打实进入了数据生产。
- `强回答要点`
  - 这些 strata 不是后验分析标签，而是 request builder 的前置决策变量。
  - 派生规则结合 `curriculum_bucket + error_type + pass_ratio`，例如 `B_near_miss` 且高 partial 的样本进入 `Core`。
  - 标签保存在 teacher requests、manifest、keep set summary 和最终 SFT summary 里。
- `容易翻车的点`
  - 把 strata 说成“为了后面做图方便”。

### D07

- `高频追问`
  - 按 stage 把 teacher 数据生产线从头到尾走一遍。
  - 这条线里哪些地方保证可审计？
  - 如果某个 shard 缺失或某轮 regen 失败，你如何恢复？
- `面试官在 probe`
  - 看 pipeline 是否 truly operational。
- `强回答要点`
  - 典型顺序是 queue -> generation units -> shards -> raw generation -> preflight -> assembled candidates -> primary QC -> stability rejudge -> regen -> testcase audit -> final keep。
  - manifest、generation_unit_id、preflight summary、regen round ledger 是关键可审计点。
  - 因为主键稳定，所以恢复依赖补 shard / 补 unit / 补 regen，而不是整线重跑。
- `容易翻车的点`
  - 只会说“做成 pipeline 了”，但说不清主键和恢复机制。

### D08

- `高频追问`
  - 你怎么把 curriculum bucket、quarantine、teacher candidate、repair-SFT 接成显式映射？
  - 为什么 source split 默认只用 train_wo_valid_big？
  - 这条映射对后续 recipe 判断有什么价值？
- `面试官在 probe`
  - 看数据 lineage 和 hygiene 边界是否清楚。
- `强回答要点`
  - 切法本质是 `curriculum skeleton ∩ canonical first-pass behavior ∩ teacher verified success potential`，并排除 quarantine、eval overlap、truncated source。
  - 默认只用 `codecontests_train_wo_valid_big`，因为非 curriculum `train` 题在没有派生 bucket 前不能安全混入主 strata。
  - 这让你能解释“为什么某轮 SFT 更像 hard-repair specialized shaper”，而不是泛泛说“数据更大所以效果变了”。
- `容易翻车的点`
  - 把“打通”讲成一句空话。

### O01

- `高频追问`
  - 你说把评测链路做成可复现 / 可追责，具体靠哪些 artifact？
  - 发生结果争议时你怎么追到某个 run 的判题口径？
  - 这套系统为什么在高噪声环境里仍然值得信？
- `面试官在 probe`
  - 看 observability 和 reproducibility engineering。
- `强回答要点`
  - 每次 run 都有 `run_info.json`、`summary.json`、`per_problem/*.jsonl`、repair 还额外记录 reused source。
  - 协议、`prompt_sha256`、`prompt_sha_mismatch_count`、`source_path`、`sandbox_url` 都能帮你回放当前 run 的口径。
  - 真正的信任基础不是“零噪声”，而是 contract 明确、噪声可量化、结果可回放、争议可归因。
- `容易翻车的点`
  - 把“可复现”讲成 deterministic。

### O02

- `高频追问`
  - 给一个“看起来像模型问题，最后拆成 protocol / judge / data / 模型四类”的真实案例。
  - 你分别用了什么实验去隔离这四类因素？
  - 拆完之后，项目结论有哪些被改写？
- `面试官在 probe`
  - 看因果分解和研究诚信。
- `强回答要点`
  - 最好讲 raw / repair 漂移与 checkpoint 波动这个案例，因为它同时涉及 judge bug、bad testcase、protocol confound 和真实模型变化。
  - 隔离手段可以依次讲 reuse-first-pass、fixed-response rejudge、sandbox targeted repro、quarantine / testcase audit。
  - 改写后的结论包括：`step1300` 是 stronger partial-credit base 但不是 exact-solve winner；`step1300_sft_v1_step60` dev 有 gain 但 held-out 不够。
- `容易翻车的点`
  - 只会说“我很重视实验可信度”，却没有真正的 isolation experiment。

### O03

- `高频追问`
  - 你维护了哪些层次的协作文档？
  - 文档怎样减少跨机器 / 跨人 handoff 成本？
  - 如果新人今天接手，你建议怎么读？
- `面试官在 probe`
  - 看 onboarding 能力和知识治理意识。
- `强回答要点`
  - 文档至少分 handoff、runbook、inventory、claim-evidence map 几层，职责不同。
  - 它们的作用不是“记日记”，而是告诉接手者“哪份是当前权威、该先看什么、遇到争议去哪里查证”。
  - 推荐阅读顺序本身也能体现你对资料层级的理解。
- `容易翻车的点`
  - 说得像 PM，而不是技术 owner。

### O04

- `高频追问`
  - 为什么敢把 teacher/QC/regen/audit/parquet 叫“数据生产链”而不是离线脚本？
  - 如果下一轮换 base checkpoint 或增加新 stratum，这条线怎么扩？
  - 最终落地给训练系统的产物是什么？
- `面试官在 probe`
  - 看平台化思维和 extensibility。
- `强回答要点`
  - 关键不在于有多少脚本，而在于有稳定 schema、manifest、QC、regen、audit merge 和 final export。
  - 换 base 时复用的是 request / generation-unit / QC schema，只换 upstream student reference 与 slicing 规则。
  - 真正交给训练系统的是 request-unique `short_diagnosis_code` parquet，例如 step1300 纯训练集。
- `容易翻车的点`
  - 把“可扩展”讲成抽象形容词，拿不出 schema、主键和 source-counts。

---

## 4. 综合型 / 挑剔型问答

### 4.1 当前最强的 12 条素材

- `R01`
  - 最强原因：有 held-out 数字，有回归集 guardrail，最像正式项目成果。
  - 面试时要主动补：这是 raw 生成结果，不是 repair 或 SFT 的结果。
- `S04`
  - 最强原因：很能体现 debug、infra、研究可靠性治理三者结合。
  - 面试时要主动补：不是“彻底零噪声”，而是“主要竞态修掉 + 残余噪声量化”。
- `S02`
  - 最强原因：reuse-first-pass 是非常像研究工程 owner 才会做的事情。
  - 面试时要主动补：固定的是 first-pass artifact，不是整个评测世界都 frozen。
- `S03`
  - 最强原因：fixed-response rejudge 很适合展示实验方法论。
  - 面试时要主动补：rejudge 证明的是 residual judge drift 仍有，但不是主导项。
- `D03`
  - 最强原因：把 SFT 数据讲成 request-unique、高精度、可审计，比“我做了数据清洗”强很多。
  - 面试时要主动补：高精度不等于完全无噪声，仍有 testcase / judge salvage 机制。
- `J02`
  - 最强原因：很能体现你不会因为 dev gain 就自我说服。
  - 面试时要主动补：最好直接给出 dev/test 的对照数字。
- `S01`
  - 最强原因：Protocol A/B 很适合展现实验设计能力。
  - 面试时要主动补：两者回答的问题不同，不是简单谁更科学。
- `R04`
  - 最强原因：held-out end-to-end self-repair 是很好讲的第二层 headline。
  - 面试时要主动补：这是部署式 utility，不是纯 repair skill。
- `R03`
  - 最强原因：说明你认真做过 checkpoint review，不是拍脑袋选模型。
  - 面试时要主动补：best exact-solve checkpoint 和最终 deployed base 不一定一致。
- `J04`
  - 最强原因：exact-solve 与 partial-credit 双指标视角很像成熟后训练研究者会说的话。
  - 面试时要主动补：不要把 partial-credit 更高直接等同于“模型更好”。
- `D01`
  - 最强原因：quarantine v3 能很好展示数据治理成熟度。
  - 面试时要主动补：它不是随手 blacklist，而是有三态与 ledger 规则。
- `S06`
  - 最强原因：failure summarization 容易让人看到“你真的碰过脏数据、长上下文和修复 prompt”的现场问题。
  - 面试时要主动补：核心证据是 storage truncation audit，而不是 prompt 灵感。

### 4.2 当前最容易被挑刺的 8 类说法

- `H03`
  - 风险：容易被听成“从零实现整个训练框架与沙箱”。
  - 修正方式：强调自己是 end-to-end owner，不是底层框架原创者。
- `H02`
  - 风险：过于像“平台 owner”，容易让人追问你是不是做了大规模通用平台。
  - 修正方式：说清是基于 verl + SandboxFusion 的研究平台整合与标准化。
- `M03`
  - 风险：把 design 文档里的未来阶段也说成已经落地。
  - 修正方式：只讲已跑通的 teacher/QC/SFT/re-eval 闭环，未来阶段单列为 next step。
- `R06`
  - 风险：只说“repair-SFT 有稳定 dev gain”，会被 held-out 一问就穿。
  - 修正方式：明确它是 dev-side specialization signal，不是最终 deployed winner。
- `R05`
  - 风险：`26.3%` 容易被误听成全局 repair 成功率。
  - 修正方式：反复强调它是 near-miss 高价值 slice。
- `R02`
  - 风险：倍率大但 base 低，容易被质疑包装。
  - 修正方式：同时给绝对数 `3/165 -> 13/165`。
- `J03`
  - 风险：如果只说“结果有噪声”，会像是在给结果找借口。
  - 修正方式：必须带 isolation experiment 和被改写过的具体结论。
- `S08`
  - 风险：watcher/launch guard 容易被说成只是 shell glue。
  - 修正方式：把它讲成轻量但真实解决 GPU 窗口浪费与重复 launch 的 operational glue，不要包装成调度系统。

---

## 5. 最值得优先背熟的问答

### 第一优先级

- `R01`
  - 你要能在 30 秒内说清：held-out `CodeContests test`、`165` 题、raw、`1.8% -> 7.9%`、`pass_ratio_mean 0.1289 -> 0.2501`、`HumanEval 89.0`、`MBPP 62.5`。
- `R04`
  - 你要能在 30 秒内说清：这是 `Protocol B`、end-to-end self-repair、`7.88% -> 10.91%`、约 `+5 solved`、trigger caveat。
- `S02`
  - 你要能在 30 秒内说清：reuse-first-pass 冻结了什么、消掉了什么 confound、如何校验一一对应、如何处理 truncated source。
- `S04`
  - 你要能在 45 秒内说清：为什么最开始像模型问题、后来如何定位到 sandbox / LB 竞态、最终协议为什么变成 patched sandbox + direct backend RR。
- `D03`
  - 你要能在 30 秒内说清：request-unique、高精度、从 teacher generation 到 final parquet 之间经历了哪些 QC / salvage 环节。

### 第二优先级

- `J02`
  - 你要能在 30 秒内说清：为什么 dev gain 不能直接外推到 held-out deploy gain，以及这件事如何阻止你继续放大 v1 recipe。
- `R03`
  - 你要能在 30 秒内说清：best exact-solve checkpoint 与 strongest deployed base 不一定一致，原因是 exact-solve、partial-credit 和 held-out utility 回答不同问题。
- `S06`
  - 你要能在 30 秒内说清：first failing case 为什么会误导 repair、你怎么重写 deterministic summarization、审计证据是什么。

### 面试时的总原则

- 不要把内部 checkpoint 命名搬进 bullet，但在被问到时要能说清“dev solve winner、partial-credit stronger base、held-out deployed base”这三者的区别。
- 不要把“可复现”说成 deterministic。
- 不要把任何 dev-only 的正结果说成已经在 held-out 上证明成立。
- 该大胆的时候大胆：你确实做了从 RL、repair、teacher/QC、SFT 到 sandbox/debug/handoff 的整条链。
