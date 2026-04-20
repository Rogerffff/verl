# Step580 Curriculum RL Pilot v8 说明文档

## 1. 这份文档是做什么的

这份文档不是实现规范，也不是 launch 脚本。

它的目的，是用尽量基础、完整的方式解释：

- 为什么当前要做 `Curriculum RL`
- 这次 pilot 想回答什么问题
- 它和之前的普通 RL / SFT 有什么不同
- 训练时到底会发生什么
- 你作为项目 owner 最需要关心哪些点

如果你不想从多轮 review 和设计稿里反复拼上下文，这份文档可以当成当前 `step580 curriculum pilot` 的主说明。

### 1.1 当前实现状态（2026-04-07）

和最早的设计稿相比，这条 pilot 现在已经不只是方案，项目内代码已经落地了 4 个核心组件：

- 离线资产构建器：
  - `coding_model_project/src/step580_curriculum_builder.py`
- 自定义 dataset：
  - `coding_model_project/src/step580_curriculum_dataset.py`
- 动态 curriculum sampler：
  - `coding_model_project/src/step580_curriculum_sampler.py`
- 启动脚本：
  - `coding_model_project/phase_2_ GRPO/ops/run_grpo_a1_resume580_to660_curriculum.sh`

当前已经完成的验证：

- `python3 -m py_compile` 通过
- launcher 的 `bash -n` 通过
- builder 的 `--help` 正常
- builder 的一个小型逻辑 smoke 已经验证过：
  - 同一 key 同时命中 `A/B` 时，最终会提升到更高优先级的 `A`

当前这份文档下面的大部分正文，仍然保留了最初 `step580 curriculum pilot` 的设计解释。它们依然能帮助理解“为什么要这样做”，但如果你要接当前主线，必须再看下面新增的 `1.2` 和文末 `28.*`。

### 1.2 这份文档现在承担两个角色

截至 `2026-04-07`，这份 explainer 不再只是“准备上 smoke 的设计说明”，而是同时记录：

1. 最初的 curriculum RL 设计动机与实现语义
2. 这条主线后来真实跑出来的版本演化、配额变化、分桶结果和阶段性结论

也就是说：

- 前面的正文主要回答：
  - 这套 curriculum RL 是怎么设计出来的
  - 各组件在语义上应该做什么
- 文末新增的 `28.*` 主要回答：
  - 实际上后来跑了哪些版本
  - 每一段用了什么分桶策略
  - 为什么从探索型切到 exploitation 型
  - 当前最可信的实验结论是什么

---

## 2. 当前背景：为什么现在考虑 Curriculum RL

### 2.1 当前主线状态

到目前为止，项目已经经历了两条主线：

1. `RL` 主线
  从 GRPO + DAPO-enhanced A1 继续推进，近期 `step580` 的 `valid_big500` 指标已经超过 `step400`。
2. `SFT repair` 主线
  基于 `step400` 做过一版 repair + anchor 的 step400 SFT。  
   这版 SFT 在 `117 / canary` 上有局部收益，但在 `valid_big500` 上没有超过纯 RL 的 `step400`。

因此当前判断是：

- **RL 主线继续推进是对的**
- 但单纯继续“按原样随机采样训练集”可能仍然会有 `solve-set churn`
- 也就是：模型可能新会一些题，但同时把原来会的题掉掉一部分

这正是这次 `Curriculum RL Pilot` 想解决的问题。

### 2.2 这次 pilot 想回答的核心问题

它想回答的不是：

- “Curriculum RL 能不能让分数一定大涨？”

它真正想回答的是：

- **如果只改训练样本的采样方式，不改 reward、不改 filter_groups、不混入 SFT 变量，能不能降低 retention/churn 问题？**

换句话说：

- 现在模型并不是完全不会做题
- 它更像是会一批、掉一批、再会一批
- 这次要试的是：**让训练过程更有节奏地看到不同类型的样本，能否让模型更稳，而不是一直“学一块、忘一块”**

---

## 3. 什么叫 Curriculum RL

### 3.1 最朴素的理解

普通 RL 训练里，训练样本通常是：

- 训练集里全部题目
- 大致随机地被抽到

Curriculum RL 的意思是：

- 不同类型的题，不再被完全一样地对待
- 训练过程会根据题目的“当前状态”动态决定更偏向抽哪些题

这和学校里的课程安排有点像：

- 有些题模型已经基本会了，应该偶尔回顾，防止忘
- 有些题模型已经接近做对，应该多给一点机会，把它推过线
- 有些题特别难，只能适度暴露，不能让它们过度主导训练
- 还有很多模型还没怎么见过的题，仍然要保持一定探索

所以这次 curriculum 不是简单的 “easy-to-hard 固定顺序”，而是：

- **根据训练过程里的在线表现，动态调整采样**

### 3.2 这次不是做什么

这次 pilot 明确**不做**下面这些事：

- 不改 reward 公式
- 不打开 `filter_groups`
- 不切 SFT 主线
- 不把 `valid_big` 题直接拿回训练
- 不在一条实验里同时改很多变量

这样做的原因很简单：

- 如果一条实验同时改 4 个东西，最后分数变了也不知道是谁起作用

所以这次实验很“克制”：

- **只改采样策略**

---

## 4. 为什么这次基座选 `step580`

当前判断是：

- `step580` 是现在更强的 RL-vnext base
- 但它依然存在 churn / retention 风险

因此策略是：

- **SFT 主锚点仍然保持 `step400`**
- **RL-vnext 的 curriculum pilot 则从 `step580` 接着做**

这两个角色不同：

- `step400` 更像 SFT 的稳定锚点
- `step580` 更像 RL 继续探索的更强起点

所以这里不是“用 curriculum 替代 SFT”，而是：

- 在 RL 线上，试一个新的采样方法

---

## 5. 这次 pilot 的一句话版本

一句话说清楚，就是：

> 从 `global_step_580` 恢复训练，继续执行 label-step `581..660`，但不再按普通随机方式抽训练题，而是给每个训练样本分桶，并在训练过程中根据模型在线表现动态重分桶、动态调采样比例。

---

## 6. 这次实验为什么这么复杂

因为当前 `verl` 的标准训练流程本身并不知道你的“课程桶”概念。

它只知道：

- 有一个 dataset
- 有一个 sampler
- 从 sampler 拿 batch
- rollout
- reward
- update

而这次 curriculum pilot 要额外实现这些能力：

1. 离线先把训练集样本分成不同 seed bucket
2. 训练运行时给每条样本一个稳定的在线身份
3. 每个 batch 结束后根据结果更新样本状态
4. 下一个 batch 再按新的状态决定抽谁
5. 中途保存时，还要把 curriculum 自己的状态也存下来

这就是为什么它需要：

- `offline manifest`
- `custom dataset`
- `dynamic sampler`
- `launcher`

而不是只写一个小配置文件就完事。

---

## 7. 这次 pilot 的四个核心组件

### 7.1 Offline curriculum manifest builder

这是离线构建脚本。

它负责做的事是：

- 读取 `step400` 和 `step580` 的 `valid_big500` 结果
- 找出：
  - 哪些题是 `step400` 会但 `step580` 不会的
  - 哪些题是 `step580` 新会的
  - 哪些题是 `step580` 接近做对但还没完全过线的
  - 哪些题表现出 `runtime_error / timeout / hard partial`
- 用这些“诊断种子”去 train 集里检索相似题
- 只在 `train_wo_valid_big` 内构建课程桶

它最终会产出：

- `curriculum_train_manifest_step580_v1.jsonl`
- `curriculum_eval_seed_asset_step580_v1.json`
- `delta69_eval.parquet`
- `curriculum_hygiene_report_step580_v1.md`

当前实现里，它还额外做了两件很重要的事：

- collapse 时真正按 `A_retention > B_near_miss > C_hard_partial` 选最终 `initial_bucket`
- summary 会区分：
  - `bucket_selected_pre_collapse`
  - `bucket_actual`

这样你后面复盘时可以区分：

- “这个样本最早是在哪个桶里被选中的”
- “最终它在 manifest 中被归到了哪个更高优先级桶”

### 7.2 Custom dataset

这是运行时 dataset 包装层。

为什么需要它？

因为当前 RL parquet 里**没有稳定的 dataset index** 可直接拿来做 curriculum 状态主键。

而 curriculum sampler 需要知道：

- 当前这条训练样本究竟是谁
- 它最初属于哪个 bucket
- 它的离线 family 是什么

所以 custom dataset 会在运行时给每条样本补上：

- `index = item`
- `extra_info.index = item`
- `extra_info.dataset`
- `extra_info.problem_id`
- `extra_info.initial_bucket`
- `extra_info.primary_seed_family`

它还会在 runtime filtering 完成后生成：

- `runtime_curriculum_coverage.json`

这个文件很重要，因为它告诉你：

- 离线 manifest 设计得再好，真正开训时还剩多少能用

当前实现里，custom dataset 还会额外做这些 fail-fast：

- 如果 runtime filtering 后 `A/B/C` 任一桶低于阈值，直接退出
- 如果过滤后的 dataset 中 `(dataset, problem_id)` 仍然重复，直接退出

这一步的目的很明确：

- **宁可不开训，也不要在 runtime bucket 残缺或 key 冲突的情况下带病启动**

### 7.3 Dynamic sampler

这是这次实验最核心的部件。

它不是普通“随机 sampler”，而是一个会随着训练结果动态改行为的 sampler。

它负责：

- 按当前 bucket 配额抽样
- 记录某题已经被看过多少次
- 根据 online reward/verifier 结果给题重分桶
- 定期写 curriculum snapshot

当前实现里，sampler 的关键运行语义也已经写死了：

- `invalid_for_rl=true` 的 whole-group 样本不会参与在线状态更新
- snapshot 只会在 `update(batch)` 末尾触发
- `JSON snapshot` 是 resume 时唯一的 sampler 真相来源
- dataloader 自动恢复出来的 sampler state 只做 echo 校验，不负责真正恢复课程状态

这是整个 experiment 里真正体现 “Curriculum RL” 的地方。

### 7.4 Launcher

Launcher 负责把所有东西接起来。

它需要显式传入：

- resume checkpoint
- curriculum manifest path
- custom dataset class
- custom sampler class
- curriculum state dir
- smoke 或正式 pilot 的 `TOTAL_TRAINING_STEPS`

它的作用不是“写算法”，而是防止实验配置飘掉。

当前 launcher 已经补上了几条最容易踩坑的保护：

- resume 到 `step > 580` 时，必须显式提供 `RESUME_STATE_PATH`
- 会在启动前校验：
  - `snapshot.global_step == resume step`
  - `snapshot.local_update_step == resume step - 580`
- 基础 smoke 和 resume smoke 都支持通过覆盖 `TOTAL_TRAINING_STEPS` 精确停在：
  - `584`
  - `604`

---

## 8. 什么是离线 seed bucket

这次离线 manifest 先把 train 集样本分成三类 seed bucket：

### 8.1 `A_retention`

这类样本来自：

- `anchor_common`
- `only400`

它代表的是：

- 模型之前比较稳定的一类能力
- 或者 `step400` 会、`step580` 掉了的那类能力

你可以把它理解成：

- **保底题**
- **保住不能再掉的能力**

### 8.2 `B_near_miss`

这类样本来自：

- `only580`
- `step580` 的高 partial / 中 partial `wrong_answer`

它代表的是：

- 模型已经“差一点点就过线”的题型

你可以把它理解成：

- **最值得优先转化的题**

因为这类题通常 ROI 最高。

### 8.3 `C_hard_partial`

这类样本来自：

- partial 的 `runtime_error`
- partial 的 `timeout`
- 更硬的 hard partial

它代表的是：

- 模型还明显没搞定，但不是完全没希望

你可以把它理解成：

- **高难暴露题**

需要看，但不能让它们把训练完全带偏。

---

## 9. 为什么 `valid_big` 不能直接回灌训练

这是这次计划里非常重要的一条数据纪律。

`valid_big` 的角色是：

- 做诊断
- 做 winner 决策
- 做外部评测

它**不能**直接放回训练集。

原因：

1. 会造成数据泄漏
2. 会让最终 `valid_big500` 失去判断意义
3. 会让“curriculum 是否真的泛化有效”变得不清楚

所以这次做法是：

- 用 `valid_big` 结果来定义 seed family
- 再从 `train_wo_valid_big` 里找相似 train 题

这是“用验证集指导训练方向”，但**不直接训练验证题本身**。

---

## 10. 为什么 join key 要用 `(dataset, problem_id)`

之前 reviewer 最担心的一点是：

- 如果只用 `problem_id`，不同数据源里可能会有冲突
- 运行时映射就可能错

因此现在强制用：

- `(dataset, problem_id)`

作为离线主键。

这意味着一条课程样本的身份，不再只是：

- `Codeforces/1407/B`

而是更完整的：

- `(codecontests, Codeforces/1407/B)` 这一类键

这让离线 manifest、runtime dataset 和后续日志都更稳。

---

## 11. Runtime coverage report 是干什么的

很多人第一眼会觉得：

- 离线 manifest 都已经构好了，为什么还要再出一份 runtime report？

原因是：

- dataset 在真正进入训练前，还会做 runtime filtering
- 有些题可能因为 prompt 太长或别的原因，在 runtime 被剔除

所以离线构出来的 `A=192/B=384/C=192`，不代表训练时真的还能保留这个数量。

因此 custom dataset 必须额外输出：

- `runtime_curriculum_coverage.json`

它至少要告诉你：

- 过滤后的 dataset 有多大
- A/B/C/U 四个 bucket 还剩多少
- 哪些 bucket 掉得最严重
- 缺失是因为 runtime filter，还是因为根本找不到
- 过滤后是否出现 `(dataset, problem_id)` 仍然重复

### 11.1 为什么要 fail-fast

如果 runtime 后发现：

- `A_retention` 剩太少
- `B_near_miss` 剩太少
- `C_hard_partial` 剩太少

那继续训练其实已经不是设计中的 curriculum 了。

所以这里宁可直接停掉，也不要“带残缺 bucket 强行开跑”。

当前硬阈值是：

- `A < 154` 退出
- `B < 308` 退出
- `C < 154` 退出
- `duplicate_key_count_post_filter > 0` 退出

在当前实现里，这些检查已经真实写进 dataset 初始化流程，而不只是文档约定。

---

## 12. 运行期 bucket 是什么

离线 bucket 是训练开始前的初始分类。  
运行期 bucket 才是训练过程中真正会变的状态。

这次运行时有 5 个 bucket：

### 12.1 `U_unseen`

还没怎么被在线观察过的样本。

它的作用是：

- 保持探索
- 避免整个训练只围着已知 seed 池转

### 12.2 `A_retention`

模型已经比较会、或者明显接近稳定会的样本。

这类样本的作用是：

- 保住已有能力
- 防止训练只顾着新收益，结果老题大面积回退

### 12.3 `B_near_miss`

模型接近做对，但还没过线的样本。

这类通常是最优先转化对象。

### 12.4 `C_hard_partial`

模型有部分能力，但距离完全做对还比较远，尤其容易伴随：

- runtime error
- timeout
- 更难的 partial

### 12.5 `D_dead_hard`

目前看起来很难、短期内收益很小的样本。

不是永远不看，而是：

- 少量暴露
- 防止它们吞掉过多训练预算

---

## 13. 为什么这次是 dynamic sampler，而不是固定 easy-to-hard

因为当前模型不是一个“从 0 开始学会做题”的状态。

它已经是一个很强、但行为会波动的 RL checkpoint。

这种情况下，固定 easy-to-hard 顺序通常不够好：

- 有些题以前会，现在掉了
- 有些题快会了，应该多推一把
- 有些题当前根本不该吃太多预算

所以更适合的是：

- **根据在线表现动态调整采样**

这就是 dynamic sampler 的核心价值。

---

## 14. 为什么强调 lazy sampling

如果 sampler 一开始就把整轮 epoch 全部排好：

- 后面 `update(batch)` 再怎么更新状态
- 也影响不了已经排好的后续 batch

那就不是“在线课程学习”，而只是“开头排了一个有点偏好的随机顺序”。

所以这次要求：

- sampler 必须是 lazy sampling
- 也就是下一批抽谁，要等上一批结果出来再决定

---

## 15. `invalid_for_rl` 为什么要单独处理

有些 rollout 会被 verifier 认定为：

- `invalid_for_rl = true`

比如抽取失败、非代码、空输出等。

这类样本如果直接参与在线 bucket 更新，会污染状态：

- 你会误以为某题很难
- 但其实只是这次 rollout 根本不该被算进 RL 更新统计

所以这次规则是：

- group 统计只在 `valid_mask = ~invalid_for_rl` 上算
- 如果某个 `uid` group 全 invalid：
  - 不更新该题 online state
  - 不加 `visits`
  - 不做 EMA / bucket 更新
  - 但仍然记录“这次采样过”

这样做的好处是：

- 训练记录不丢
- 但 curriculum 状态不会被脏 rollout 误导

---

## 16. `local_update_step` 到底是什么

这是整个计划里最容易混淆、但最重要的概念之一。

它不是：

- 当前 trainer 的 `global_steps`
- 也不是 checkpoint label

它的定义是：

- **已经完成了多少次 sampler update**

当前约定：

- resume 到 `global_step_580` 后
- 进入主循环前 trainer 会把 `global_steps` 从 `580` 加到 `581`
- 第一批训练完成后才执行 `sampler.update(batch)`
- 这时 `local_update_step` 从 `0` 变成 `1`

所以：

- 训练 batch 开始前看的，是“当前 `local_update_step`”
- `update(batch)` 结束后，`local_update_step += 1`
- 下一批才用新的 step

### 16.1 它为什么重要

因为 phase 切换是靠它定义的：

- `phase0`: `0..19`
- `phase1`: `20..59`
- `phase2`: `60..79`

这正好对应：

- `581..600`
- `601..640`
- `641..660`

如果这个口径错一拍，整套 curriculum 配额都会错一拍。

---

## 17. 三个 phase 分别在干什么

### 17.1 Phase 0：预热期（581..600）

配额：

- `U/A/B/C/D = 8/3/3/1/1`

目标：

- 先建立在线统计
- 先让更多未充分观察过的样本被看一遍
- 不让 seed bucket 一上来就被反复过采样

### 17.2 Phase 1：主训练期（601..640）

配额：

- `U/A/B/C/D = 3/4/6/2/1`

目标：

- 把重心放在 `B_near_miss`
- 但仍保留 `A_retention`
- 让模型既能转化快会的题，又尽量别忘太多

### 17.3 Phase 2：收敛期（641..660）

配额：

- `U/A/B/C/D = 2/5/6/2/1`

目标：

- 进一步偏重 retention 和 near-miss
- 降低最后阶段的遗忘
- 保留一点 hard exposure，但不让它主导

---

## 18. 为什么 snapshot 这么重要

普通训练里，只保存 checkpoint 往往就够了。

但 curriculum 训练不是。

因为模型状态之外，还有一套“课程状态”：

- 每题现在属于哪个 bucket
- 每题被看了几次
- EMA pass ratio 是多少
- 最近看过哪些题
- promotion/demotion 计数是多少

这些东西如果不单独保存：

- 训练中断后恢复出来的 curriculum 就会变样
- 那就不再是同一条实验

所以这次要求：

- `curriculum_state_step_600.json`
- `curriculum_state_step_620.json`
- `curriculum_state_step_640.json`
- `curriculum_state_step_660.json`

这些 snapshot 必须明确存在。

---

## 19. 为什么 snapshot 只能在 `update(batch)` 里写

这是由 trainer 当前真实调用顺序决定的。

当前 trainer 是：

1. 先完成这一步训练
2. 先做验证和保存 checkpoint
3. 再调用 `sampler.update(batch)`

这意味着：

- `checkpoint@600` 表示模型已经完成 `label 600`
- 但 curriculum 状态只有在 `sampler.update(batch)` 跑完后，才真正完成 `label 600` 的状态更新

所以：

- `curriculum_state_step_600.json` 必须表示“post-label-600 curriculum state”
- 这个 snapshot 只能在 `update(batch)` 末尾写

这不是随便定的，而是和 trainer 的真实执行顺序绑定的。

---

## 20. 为什么 JSON snapshot 是 sampler 唯一真相来源

因为 trainer 自己会恢复 dataloader state，但它并不知道你 curriculum 的“完整统计状态”是什么意思。

所以这里必须把责任划清：

- trainer / dataloader 的 state：
  - 负责最小迭代恢复
- curriculum JSON snapshot：
  - 负责真正的 sampler 运行状态恢复

当前约定是：

- 当 `resume_state_path` 存在时
- sampler 必须在第一次 `__iter__()` 前且仅一次加载 snapshot
- 这次加载要覆盖掉 dataloader 自动恢复后 sampler 的运行状态

当前实现还专门加了一条单行日志，便于远端查 resume：

- `CURRICULUM_SNAPSHOT_LOADED step=... local_update_step=...`

换句话说：

- **真正信的是 JSON**
- dataloader 里的 sampler state 只拿来做 echo 校验

---

## 21. 为什么还要做两种 smoke

### 21.1 基础 smoke：`581..584`

这个 smoke 的目的不是看分数。

它主要验证：

- custom dataset 正常注入元数据
- sampler 能正常取数
- `uid/index/(dataset, problem_id)` 聚合正确
- `runtime_curriculum_coverage.json` 正常生成
- runtime bucket 没有因为过滤问题直接崩掉

这一步本质上是在问：

- **这套 plumbing 能不能正常跑起来**

### 21.2 Mid-run resume smoke：`600 -> 604`

这个 smoke 是这次计划里非常关键的一步。

它验证的是：

- 从 `checkpoint@600 + snapshot@600` 恢复时
- curriculum 状态是否真的能正确接上

它要求：

- resumed run 第一批必须是 `label 601`
- `local_update_step` 从 `20` 接着走到 `24`
- 不允许重复 `600`
- 不允许跳过 `601`

这一步本质上是在问：

- **这套 resume 语义是真的对，还是只是纸面上对**

---

## 22. 为什么 resume smoke 要和正式 pilot 隔离目录

因为你不希望测试 resume 的副产物污染正式 run。

如果混用目录，可能出现：

- rollout dump 混在一起
- validation dump 混在一起
- checkpoint 名字冲突
- curriculum state 被覆盖

所以现在要求：

- resume smoke 必须用独立目录
- 独立 `EXPERIMENT_NAME`
- 独立 checkpoint dir
- 独立 rollout/validation/state dir

这是运维保护，不是算法本身，但非常必要。

当前实现里，launcher 也已经支持把 resume smoke 跑到独立目录；实际执行时要显式覆盖：

- `EXPERIMENT_NAME`
- checkpoint dir
- rollout dump dir
- validation dump dir
- curriculum state dir

---

## 23. 外部评测为什么仍然要看 `valid_big500`

这次 pilot 不是不看 cheap-screen，而是：

- cheap-screen 只能当开发可见性
- `valid_big500` 才是正式 winner 判定集

因为你们之前已经见过：

- `117 + canary` 看起来不错
- 但最后不一定能赢 `valid_big500`

所以这次标准更严格：

- `600/620/640/660` 都要过外部 gate

---

## 24. 这次成功标准是什么意思

当前设定是：

- `valid_big500 solved >= 58`
- `pass_ratio_mean >= 0.3316`
- `step400 solved retention >= 43/53`
- `lost_vs_step400 <= 10`

这四条合在一起，表达的是：

1. **分数要涨**
2. **整体 partial 不能比当前基线更差**
3. **保留住 step400 的核心已会题**
4. **不能靠大规模掉题换取少量新题**

这套标准的核心思想是：

- 不接受“只是换了一批会做的题”
- 要的是“更强，且更稳”

### 24.1 当前远端 reward / sandbox 运行前提已经恢复

在最新一次实例重启之后，共享 reward 池已经重新恢复完成，当前远端可直接用于这条 curriculum pilot 的 verifier 链路是：

- `8090 -> 8081..8088`

也就是：

- 1 个 Nginx LB
- 8 个 sandbox backend

当前已经完成的远端验证包括：

- `8081..8088` 全部 `health: up`
- 对 `http://localhost:8090` 的 `32 requests / 16 workers` probe 成功
- `unexpected_stdout_count = 0`
- `failure_count = 0`

所以这条 curriculum pilot 当前默认应继续使用：

- `SANDBOX_URL=http://localhost:8090`
- `LIMITER_BUDGET=128`

这意味着：

- 训练前置的 reward infra 已经不是 blocker
- 下一步可以直接进入 trainer 级 smoke，而不是再做一轮基础运维准备

---

## 25. 你作为 owner 最需要关心的事情

如果你不打算盯具体实现细节，那最重要的是盯下面这几件事：

### 25.1 先看 runtime coverage

如果 `runtime_curriculum_coverage.json` 一开始就不达标：

- 这条 pilot 直接不该跑

### 25.2 再看两个 smoke

如果：

- 基础 smoke 过不了
- 或 `600 -> 604` resume smoke 过不了

那说明实现语义没站稳，不能直接上正式 pilot。

### 25.3 正式 pilot 时不要只看 fast-val

fast-val 可以看趋势，但它不是最终 winner 指标。

真正要看的还是：

- `delta69`
- `valid_big500`

### 25.4 最后重点看 retention/churn

这次 pilot 的核心不是“某一次瞬时分数多高”，而是：

- `step400` 那些关键 solved case 保住了多少
- `step580` 新增的能力有没有继续积累
- `lost_vs_step400` 有没有明显下降

---

## 26. 你可以把这次 pilot 理解成什么

你可以把它理解成：

> 在不改 reward、不改 verifier、不改主算法的前提下，给 `step580` 装上一层“会根据在线表现动态调课表”的训练控制器，看看它能不能减少当前最头疼的 churn。

这不是终局方案。

它更像一个非常重要的中间问题验证：

- 如果只靠 curriculum sampling 就能显著改善 retention/churn，那么后续很多改动都可以更有把握地围绕它展开。
- 如果连这一步都不行，那就说明瓶颈更可能在 reward 设计、filter_groups、或更深层的数据分布问题上。

---

## 27. 一句话总结

这次 `Step580 Curriculum RL Pilot v8` 的本质是：

> 从更强的 `step580` 出发，不改 reward，只改“训练时题目被抽到的方式”，并用严格的 snapshot / resume / external eval 约束，验证 curriculum sampling 能不能真正减少 solve-set churn、提升 retention。

### 27.1 现在最直接的执行顺序

如果按当前已经落地的实现和环境状态继续往前推进，最自然的顺序就是：

1. 在远端跑 `581..584` 基础 smoke
- `TOTAL_TRAINING_STEPS=584`

2. 跑到 `checkpoint@600 + curriculum_state_step_600.json`

3. 在独立目录里跑 `600 -> 604` 的 mid-run resume smoke
- `TOTAL_TRAINING_STEPS=604`

4. 两个 smoke 都通过后，再起正式 pilot
- label-step `581..660`

也就是说，这份文档当前对应的是一个已经进入“可执行 smoke 阶段”的实验，而不是还停留在纯设计讨论阶段。

## 28. 2026-04-07 之后的实际推进记录

上面 `1..27` 解释的是最初 `step580 curriculum pilot` 的设计和语义。下面这部分补的是：**后来这条 RL 主线实际是怎么推进的。**

这部分不是新的设计稿，而是当前已经发生过的真实 rollout 历史。

### 28.1 `step600_v2`：先做人工清洗，把 `A/C` 收紧

最初的目标不是直接大改 trainer，而是先把离线课程桶做得更干净。

这一步做了两类人工 review：

- `A_retention` review
- `C_hard_partial` review

最终得到的 `manifest v2` 方向是：

- `A` 更小、更偏 anti-regression
- `B` 继续做 near-miss 主桶
- `C` 明显收缩

当时的代表性目标桶大小是：

- `A = 128`
- `B = 408`
- `C = 64`

配额上采用过一版偏 `A/B` 的方案：

- `phase1 = 3/5/7/1/0`

这版的教训是：

- `delta69` 上能看到局部修复
- 但 `valid_big500` 没有守住 solve-count

也就是说：

- **方向不是完全错**
- 但只靠这一轮清洗，还不足以把局部修复变成正式 benchmark 的稳定胜利

### 28.2 `step600_v3`：focused re-clean，再补一小批 stabilizer

在 `v2` 之后，又做了一轮更 focused 的人工收缩和补强：

- `C_hard_partial` 继续从 `64` 缩到 `40`
- 额外做了 `anchor_common stabilizer shortlist`
- 从候选里挑出 `14` 条真正值得补进 `A` 的 stabilizer

这一轮形成的 `focused manifest v3` 最终大小大致是：

- `A = 142`
- `B = 411`
- `C = 40`

这一步的真实意图是：

- 不再继续放大 `C`
- 让 `A` 除了 anti-regression family 之外，再补一点稳的 `anchor_common`

### 28.3 `qv2/qv3`：把数据治理真正接进训练入口

后面我们确认到：已知脏题虽然没有直接污染当前 `delta69 / valid_big500` 的结论，但它们仍然在 RL train parquet 入口里。

所以这条主线后来又叠加了 quarantine 治理。

当前真正采用的是 `problem_quarantine_v3`：

- `hard_blacklist = 780`
- `caution = 22`
- `unresolved = 83`

接入语义是：

- `train parquet`：只硬过滤 `hard_blacklist`
- `curriculum manifest / 高价值 eval seed`：过滤 `hard_blacklist + caution`
- `unresolved`：保留

对应产物：

- `grpo_parquet_qv3/train.parquet`
- `step600_v3_qv3 manifest`

当前 `qv3` 过滤后的主 manifest 大小是：

- `A = 139`
- `B = 409`
- `C = 39`

这一步很重要，因为：

- 训练分布更干净
- 但又没有因为过度硬删而把训练集规模砍得太狠

### 28.4 `step620 -> 900`：探索型长跑，目标是建立 `U` 图谱

在 `step600/620` 这一轮判断之后，主线没有继续沿 `step640` 推，而是选了：

- 用 `step620` 做 base
- 开一条更长的 curriculum RL
- 目标从“再试 20 步”改成“把 `U` 的在线难度图谱真正建起来”

这条长跑的关键代码变化有三件：

1. `U_revisit` 配额
- `U` 不再只是 first-touch
- 明确保留一部分给 `visits == 1` 的样本第二次访问

2. `prefer_low_visits`
- 同 bucket 内优先抽更低 `visits` 的样本
- 减少一直在相似 `A/B/C` 里打转

3. `U` 观测字段
- `u_first_touch_sampled_count`
- `u_revisit_sampled_count`
- `u_to_A/B/C/D`
- `u_visit1_backlog`

这条长跑实际用了两段式策略：

- `621..700`
  - `U/A/B/C/D = 8/3/4/1/0`
  - `U_revisit_quota = 3`
- `701..900`
  - `U/A/B/C/D = 6/4/5/1/0`
  - `U_revisit_quota = 3`

这里要特别注意一个实现语义：

- 当前 sampler 实现只真正使用前两个 `phase_boundaries`
- 所以这条 run 实际上是一个**两段式**，不是三段式

这条长跑最重要的结果不是单个 checkpoint 的分数，而是我们终于看清了 `U` 图谱的真实形状。

到 `step900`：

- `U` 从 `9616` 降到 `8776`
- 说明更多题确实被触达并完成了定级

但更关键的是流向：

- `U -> A = 207`
- `U -> B = 164`
- `U -> C = 277`
- `U -> D = 192`

也就是：

- `U -> A/B = 371`
- `U -> C/D = 469`

这告诉我们：

- 探索型 curriculum 的目的达到了：更多题被看到了
- 但新发现的题里，偏 hard/dead-hard 的比例比预期更高

这也解释了为什么：

- `step900` 的 `valid_big500 pass_ratio_mean` 能明显抬升
- 但 solve-count 仍然没有超过 `step600`

### 28.5 `step700/800/900` 的真实评测形状

这条长跑里最值得记住的不是单个点，而是整体趋势：

`delta69`

- `step600 = 50/69`, `pass_ratio_mean = 0.8892`
- `step700 = 47/69`, `0.8234`
- `step800 = 43/69`, `0.7699`
- `step900 = 48/69`, `0.8549`

`valid_big500 clean`

- `step600 = 57/498`, `pass_ratio_mean = 0.3319`
- `step700 = 51/498`, `0.3314`
- `step800 = 39/498`, `0.3094`
- `step900 = 53/498`, `0.3467`

所以这轮最准确的读法是：

- `step600`：solve-count winner
- `step900`：broad partial / pass-ratio winner
- `step800`：明显 bad point

也就是说：

- 这轮 RL 的收益不是假的
- 但收益主要先表现成 **partial correctness**
- 还没有稳定穿过 solve 的门槛

这就是为什么后面不再继续沿当前探索型配方往前推，而是切到了 exploitation 分支。

### 28.6 `step900 -> 1000`：从探索型切到 `A/B` 强化 exploitation

到了 `step900`，我们已经得到两个明确结论：

1. 继续高 `U` 只会继续把更多样本推向 `C/D`
2. 当前更需要的是：
  - 清掉 `u_visit1_backlog`
  - 把已经摸出来的可学题尽量拉进 `A/B`

所以后续从 `step900` 开了一条新的短分支，策略改成：

- 少量继续看新题
- 更大力度做 revisit 和利用
- 明显增加 `A/B`
- `D` 继续禁用

当前采用的两段式配额是：

- `901..940`
  - `U/A/B/C/D = 4/5/6/1/0`
  - `U_revisit_quota = 3`
- `941..1000`
  - `U/A/B/C/D = 2/6/7/1/0`
  - `U_revisit_quota = 2`

也就是说，这一段的核心目标已经不是“继续摸图谱”，而是：

- **把 backlog 定级完**
- **把更多质量送回 `A/B`**

### 28.7 为什么 `step900 -> 1000` 先用 `200` 并发，后来又改成 `150`

这条 exploitation 分支最早试过一版：

- `limiter_budget = 200`

这版失败了。

关键点是：

- 不是 GPU OOM
- 不是数据坏了
- 也不是 sandbox 本身挂掉

真正的根因是：

- Ray 的 **CPU/RAM memory pressure** 保护先杀了 worker
- 表面上的报错发生在 `_compute_old_log_prob()`
- 但本质上是 reward 阶段残留的对象 + old-log-prob 前向叠加，触发了瞬时内存峰值

这条经验后来直接影响了新分支的启动参数。

当前真正继续跑的版本是：

- `limiter_budget = 150`
- `save_freq = 35`
- `snapshot_steps = [910,945,980,1000]`

也就是：

- reward 并发比 `200` 更保守
- 保存更密一点，避免这一段再挂时完全没有可恢复点

### 28.8 读这条主线时，要特别注意的三个“语义细节”

1. `U` 观测字段当前更适合看趋势
- 它们不等于“严格的 U-slot 命中计数”
- 因为 fallback 抽到的 `U` 也会被计进去

2. `reset_state_on_dataset_mismatch=True` 不是硬 reset
- 当前实现是：
  - 先按 `index_keys` 做 remap
  - 保留能保留的旧状态
  - 不够时才 reset

3. `qv3` 的正确性依赖显式路径
- 例如 `build_grpo_parquet.py` 的默认 quarantine 路径仍不是 `v3`
- 所以当前 run 的正确性来自“显式传 `v3`”，不是默认值自己对了

### 28.9 截至现在，这条 curriculum RL 主线最可信的结论

截至 `2026-04-07`，当前最可信的结论是：

- Curriculum RL 已经不是纸面设计，而是已经真实改变了训练分布
- `U` 图谱也确实在被建立
- 但当前探索型配方更容易先把收益落在 `partial correctness`
- 真正要把收益转成 solve-count，还需要更偏 `A/B` 的 exploitation 分支继续验证

所以当前这条主线不应该再被理解成：

- “再多跑一点探索就自然会更好”

更准确的说法是：

- “探索型阶段已经完成了它该做的事情”
- “下一步必须看 exploitation 阶段能不能把 partial gain 转成 solved gain”
