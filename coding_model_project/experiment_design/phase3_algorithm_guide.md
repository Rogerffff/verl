# Phase 3 算法学习指南：从原生 GRPO 到 DAPO / Dr.GRPO

面向对象：已经知道原生 GRPO 是什么，但还不熟悉 DAPO、Dr.GRPO、GSPO 等后续改进；希望把这些算法真正用于当前 coding RL 项目，并且在面试里讲清楚为什么这样选。

当前项目语境：

- 框架：`verl`
- 基座模型：`Qwen2.5-Coder-7B-Instruct`
- 奖励来源：`SandboxFusion` 判题反馈
- 任务：单轮 Python code generation
- 主要指标：`accepted@1`、`pass_ratio_mean/p50/p90`

一句话总判断：

- **本项目当前推荐的主线算法是 `DAPO-style GRPO`。**
- **`Dr.GRPO` 不是被否定，而是更适合作为关键消融。**
- **我们不是追最新名词，而是在 GRPO 主干上选择一组更适合 coding verifier 场景的稳定化改进。**

---

## 1. 为什么要看这篇

如果你只懂原生 GRPO，第一次接触 DAPO 和 Dr.GRPO 时最容易出现两个误解：

1. 误以为 DAPO 是一个“完全替代 GRPO 的新算法”
2. 误以为 Dr.GRPO 和 DAPO 是同一类改进，只是名字不同

这两个理解都不准确。

这篇文档的目标不是把所有论文都背一遍，而是帮你建立下面这条清晰的认知链：

1. 原生 GRPO 到底在做什么
2. 它在 coding RL 里会暴露哪些真实问题
3. DAPO 和 Dr.GRPO 分别想修什么问题
4. 它们为什么不是一回事
5. 回到当前项目，为什么主线选择 `DAPO-style GRPO`，而不是“纯原生 GRPO”或“纯 Dr.GRPO”

---

## 2. 先把原生 GRPO 讲透

### 2.1 原生 GRPO 的核心思想

原生 GRPO 可以理解成一种 **不训练 critic 的 PPO 类方法**。它不显式训练 value model，而是对同一个 prompt 采样多个回答，用 **组内相对奖励** 来决定哪个回答更值得强化。

设一个 prompt 为 `x`，我们从当前策略采样 `n` 个回答：

```text
G(x) = {y1, y2, ..., yn}
```

每个回答经过 verifier 打分，得到标量奖励：

```text
r1, r2, ..., rn
```

原生 GRPO 的直觉是：

- 比组内平均表现更好的回答，优势为正
- 比组内平均表现更差的回答，优势为负
- 所以不需要单独训练一个 critic 去估计 value

原生 GRPO 常见的组内优势写法是：

```text
ai = (ri - mean(r_group)) / std(r_group)
```

然后把这个标量优势广播到该回答的所有 response token 上，再进入 PPO-style clipped objective。

### 2.2 为什么说它是 critic-free

PPO 的经典做法通常要训练一个 value model，去估计每一步 action 的 value 或 return。

GRPO 不这样做。它直接用“同一题的多个候选答案之间的相对比较”来构造优势：

- 不训练 critic
- 省显存
- 系统更简单
- 对可验证任务尤其方便，因为你本来就可以拿到 outcome reward

这也是为什么 GRPO 特别适合：

- 数学推理
- 代码生成
- 可执行、可验证任务

### 2.3 在 `verl` 里，原生 GRPO 对应什么

你当前项目里最重要的本地实现入口是：

- `../../verl/trainer/ppo/core_algos.py`
- `../../examples/grpo_trainer/README.md`

在 `verl` 中：

- `algorithm.adv_estimator=grpo` 表示使用 GRPO 风格的优势估计
- `algorithm.norm_adv_by_std_in_grpo=True` 表示保留原生 GRPO 的组内标准差归一化
- `actor_rollout_ref.actor.loss_agg_mode` 决定 policy loss 如何在 token / sequence 维度聚合

`verl` 本地 README 还明确提醒了一件很重要的事：

- 原始 GRPO 论文更偏 `seq-mean-token-mean`
- 但 `verl` 的 GRPO 示例默认更偏 `token-mean`
- 原因是长 CoT / 长 response 场景下，sample-level 聚合可能更不稳定

这句话很关键，因为它说明：

- **你现在就算“跑 GRPO”，也不一定是在跑最教科书的原始 recipe**
- **很多所谓 DAPO 解决的问题，在 `verl` 的默认工程实践里已经部分被绕开了**

### 2.4 一个 4 个候选回答的最小例子

假设同一个 CodeContests 题目，我们采样了 4 个候选回答，判题结果如下：

| 候选 | verifier 结果 | `pass_ratio` | response 长度 |
|------|---------------|--------------|---------------|
| `y1` | 全部通过 | `1.0` | 220 tokens |
| `y2` | 通过一部分测试 | `0.6` | 420 tokens |
| `y3` | 只通过少量测试 | `0.2` | 760 tokens |
| `y4` | 全错 | `0.0` | 900 tokens |

组内平均奖励：

```text
mean = (1.0 + 0.6 + 0.2 + 0.0) / 4 = 0.45
```

如果按原生 GRPO 的直觉：

- `y1` 的优势最大，应该被强化
- `y2` 也略高于平均，应被轻度强化
- `y3` 和 `y4` 低于平均，应被抑制

如果启用标准差归一化，优势大致会变成“更拉开”的版本：

```text
y1: 强正优势
y2: 小正优势
y3: 小负优势
y4: 强负优势
```

如果关闭标准差归一化，只减均值，不除标准差，那么它们会变成“更保守”的版本：

```text
y1: +0.55
y2: +0.15
y3: -0.25
y4: -0.45
```

这已经能帮助你直观看到：

- **原生 GRPO 和 Dr.GRPO 的第一个差异在 advantage 的尺度上**
- **DAPO-style GRPO 并不一定改这里，它更多改的是 loss、采样和稳定化 recipe**

### 2.5 这 4 个候选例子里，DAPO-style 和 Dr.GRPO 各自会怎么看

对这个“混合质量”的 group：

- **原生 GRPO**
  - 保留组内 mean/std 归一化
  - 用默认 GRPO 主干更新

- **DAPO-style GRPO**
  - 组内相对比较这件事没有变
  - 但会在更新方式上做稳定化：
    - 更偏 `token-mean`
    - 可以用 `clip-higher`
    - 可以保留 mixed-quality group，过滤全对或全错组
    - 常见做法是去掉 KL

- **Dr.GRPO**
  - 更关注“原生 GRPO 的归一化和聚合是否引入偏置”
  - 它不是主要在采样上做文章，而是更直接地改：
    - `norm_adv_by_std_in_grpo=False`
    - `loss_agg_mode=seq-mean-token-sum-norm`
    - 可配 `loss_scale_factor`

如果再看一个“全错组”：

| 候选 | verifier 结果 | `pass_ratio` |
|------|---------------|--------------|
| `z1` | 全错 | `0.0` |
| `z2` | 全错 | `0.0` |
| `z3` | 全错 | `0.0` |
| `z4` | 全错 | `0.0` |

那么这个 group 在组内几乎没有学习信号。

- 原生 GRPO：理论上仍然可以进训练，但有效优势接近 0
- DAPO-style 的 group filtering：倾向把这种 group 过滤掉，减少无信息采样
- Dr.GRPO：不会自动解决“全错组没有信息”这个问题，因为它主要修的是偏置，而不是采样效率

这就是为什么：

- **DAPO 更像“稳定训练 recipe”**
- **Dr.GRPO 更像“偏置修正路线”**

---

## 3. 原生 GRPO 在 coding RL 里会遇到什么问题

这一节只讲与当前项目强相关的问题，不展开通用 RL 理论。

### 3.1 问题一：长代码、长响应下的长度偏置与训练不稳

代码任务与短答案任务最大的不同，是 response 长度跨度非常大。

在 CodeContests 场景里，模型经常要输出：

- 完整函数
- 输入解析
- 边界处理
- 主逻辑
- 输出格式

这会带来两个风险：

1. 较长 response 的 token-level 梯度更容易被“平均掉”
2. 某些聚合方式会让错误但很长的答案产生不合适的更新强度

这也是为什么后续论文会围绕下面两件事做文章：

- DAPO：更强调 token-level loss、clip-higher、过长响应处理
- Dr.GRPO：更强调 GRPO 的归一化和聚合本身可能带来偏置

### 3.2 问题二：全对组 / 全错组经常没有什么学习价值

GRPO 的核心信号是“组内相对差异”。

如果一个 group：

- 全都答对
- 或者全都答错

那这个 group 提供的相对学习信号就很弱。

这在 coding RL 里很常见，因为有些题：

- 太简单，模型全过
- 太难，模型全错

当前项目的 baseline 已经说明：

- CodeContests 上 `pass_ratio_mean` 大约在 `14%-26%`
- `Wrong Answer` 是主要错误类型

这说明它既不是“几乎全过”，也不是“完全学不到”，但 mixed-quality group 与 degenerate group 会同时存在。

这正是 DAPO-style 里 group filtering 有吸引力的原因。

### 3.3 问题三：KL 太强会限制 coding policy 探索

在对话型 RLHF 场景里，KL 常被用来避免策略漂移太大。

但 coding 任务有个不同点：

- 你往往希望模型学会“新的程序结构”
- 希望它更敢尝试边界处理
- 希望它摆脱初始模型的一些错误模式

如果 KL 过强，模型可能：

- 更不敢偏离 base model
- 更难探索新的代码分布
- 学得更保守

所以 2025 年后的很多公开 coder RL 系统都倾向于弱化甚至去掉 KL。

但这里也要记住：

- **去 KL 不是宗教信条**
- 如果出现格式漂移、输出发散、timeout 暴涨，开一个小 KL 的 rescue run 仍然是合理工程做法

### 3.4 问题四：coding verifier 很贵，group size 不能盲目加

GRPO 的一个朴素想法是：group size 越大，组内比较越充分。

但 coding RL 里，group size 不是免费午餐。

因为每增加一个候选回答，你都要付出：

- 生成成本
- 执行成本
- 判题成本
- 更长的 wall-clock 时间

而当前项目的奖励又来自 `SandboxFusion`，这意味着每个 rollout 都需要真正跑代码。

所以 group size 的选择是一个典型工程权衡：

- 太小：组内相对信号不够稳定
- 太大：judge cost 爆炸，训练吞吐掉得很厉害

这也是为什么当前项目把 `group_size=4` 当成更合理的起步点，而不是一上来就上 `8` 或更大。

### 3.5 这一节的结论

`论文结论`：

- 原生 GRPO 在长响应任务、组内退化组、以及稳定性方面都有后续改进空间。

`本项目工程判断`：

- 对当前 coding RL 项目，最值得优先处理的不是“把目标函数换到最新”，而是：
  - 稳定更新
  - 避免无效 group
  - 不让 KL 阻碍探索
  - 控住 judge cost

---

## 4. DAPO 到底是什么

### 4.1 先给一句最重要的定义

**DAPO 更像是一套构建在 GRPO 之上的稳定化 recipe，而不是与 GRPO 完全脱离的新范式。**

如果用一句更通俗的话说：

- 原生 GRPO 回答的是“组内相对奖励怎么用”
- DAPO 回答的是“既然还在用 GRPO 主干，训练应该怎样更稳地跑起来”

### 4.2 DAPO 的四个核心部件

| 组件 | 它改了什么 | 它想解决什么 | coding RL 是否一定要用 |
|------|------------|--------------|-------------------------|
| `clip-higher` | 上下裁剪区间不对称，常见 `0.2 / 0.28` | 允许更优 token 概率上升得更充分 | 不一定，但作为主线增强很值得试 |
| `dynamic sampling / group filtering` | 过滤掉全对或全错组，必要时继续采样 | 减少低信息量 group 浪费 | 对 verifier 任务很有吸引力 |
| `token-level loss` | 更偏 token 维度聚合，而不是 sample 维度 | 缓解长响应下的不稳定 | 对长代码任务通常值得保留 |
| `overlong reward shaping` | 对过长但未硬截断的输出施加线性惩罚 | 防止模型学成无节制拉长 | 不一定要进 headline algorithm |

### 4.3 逐项讲清楚

#### 4.3.1 Clip-Higher

经典 PPO / GRPO 常用一个对称 clip ratio，例如 `0.2`。

DAPO 的想法是：

- 对坏方向仍然严格限制
- 对好方向稍微放松一些

典型形式是：

```text
clip_ratio_low = 0.2
clip_ratio_high = 0.28
```

直觉上，它是在说：

- 不希望更新太激进，避免崩
- 但对于真正更好的 token，希望“向上走”的空间稍微更大

对 coding RL 来说，这件事的吸引力在于：

- 好的代码答案往往不是“略微更好”，而是结构上明显更对
- 适当鼓励正向更新，有机会提高学习效率

#### 4.3.2 Dynamic Sampling / Group Filtering

这部分是 DAPO 非常工程化、也非常适合 verifier 场景的一点。

它的核心直觉是：

- 如果一个 group 全都 1 分，没什么相对学习信号
- 如果一个 group 全都 0 分，也没什么相对学习信号
- 既然这样，不如过滤掉这些 group，把训练预算留给 mixed-quality group

对 coding RL 来说，这很自然，因为：

- verifier reward 是 outcome-oriented
- group 的信息量是否充足，很容易从 reward 结果里直接看到

但也要注意一个实现侧边界：

- 如果你写 `filter_groups.metric=seq_reward`
- 那你其实已经和 reward 定义产生了耦合

换句话说：

- 如果后面 reward 从纯 `pass_ratio` 改成 piecewise dense reward
- group filtering 的行为也会跟着变

所以它不是“完全独立于 reward 的纯算法开关”。

#### 4.3.3 Token-Level Loss

这个点最容易被误解。

很多人会说：

- DAPO 的一个贡献是 token-level loss

这话不算错，但对你当前项目来说必须补充一句：

- **`verl` 的 GRPO 示例本来就默认偏向 `token-mean`**

也就是说，`token-mean` 不是只有“开启 full DAPO”才能拥有的能力。

因此更准确的理解应该是：

- DAPO 强调 token-level aggregation 的价值
- 但在 `verl` 工程实践里，这个思路已经被部分吸收到了更常规的 GRPO 配置中

#### 4.3.4 Overlong Reward Shaping

这部分很有用，但不一定应该直接写进你的 headline algorithm 定义。

它做的事情是：

- 当 response 长度接近或进入一个“过长缓冲区”时
- 对 reward 增加线性惩罚

它要解决的是：

- 模型为了逃避错误或拖延结束，不断输出更长内容
- 长输出导致吞吐恶化、成本膨胀、甚至触发截断

对于当前项目，我更推荐的态度是：

- **理解它**
- **在需要时把它作为 reward-side 工具**
- **但不要在第一版 headline algorithm 里把它和主算法绑死**

这样你后续 reward 设计还能保持独立解释空间。

### 4.4 DAPO 在 `verl` 里的对应关系

本地可直接对照的材料：

- `../../docs/algo/dapo.md`
- `../../tests/special_e2e/run_dapo.sh`
- `../../verl/workers/reward_manager/dapo.py`

你需要知道的几个点：

1. `verl` 已经把 DAPO 的核心 recipe 拆成若干配置项，而不是一个神秘黑箱
2. `reward_manager=dapo` 更多是 reward manager / recipe 入口，不代表“整个算法已经自动替换成另一个完全不同的优化器”
3. 文档和脚本已经把 `clip-higher`、`filter_groups`、`token-mean`、`overlong_buffer` 的典型配置写出来了

### 4.5 对 DAPO 的一句准确总结

`论文结论`：

- DAPO 证明了一组围绕 GRPO 的稳定化设计，在长推理 RL 中有效。

`本项目工程判断`：

- 当前项目不需要“full DAPO 全盘照搬”。
- 更合理的说法是：**以 GRPO 为主干，吸收 DAPO 中成熟、可解释、适合 coding verifier 场景的稳定化部件。**

---

## 5. Dr.GRPO 到底改了什么

### 5.1 Dr.GRPO 不是 recipe，而是偏置修正路线

如果说 DAPO 更像“训练 recipe”，那 Dr.GRPO 更像在问一个更尖锐的问题：

> 原生 GRPO 的优势归一化与 loss 聚合方式，会不会本身就引入优化偏置？

它的核心关注点不是：

- 怎么过滤 group
- 怎么延长采样
- 怎么做 overlong shaping

而是：

- 原生 GRPO 的 `(r - mean) / std` 是否会放大某些不该被放大的差异
- 长 response 在现有聚合方式下是否被不公平地处理

### 5.2 `norm_adv_by_std_in_grpo=False` 到底是什么意思

原生 GRPO 常见写法：

```text
ai = (ri - mean(r_group)) / std(r_group)
```

Dr.GRPO 认为这里的 `std` 归一化在某些情况下会产生问题：

- 如果组内所有回答都很接近
- `std` 非常小
- 那么一些本来差异不大的回答，也可能被放大成很显著的优势差

于是 Dr.GRPO 的一个关键动作是：

```text
ai = ri - mean(r_group)
```

也就是：

- 仍然做组内中心化
- 但不再除以组内标准差

在 `verl` 对应的就是：

```yaml
algorithm.norm_adv_by_std_in_grpo = false
```

### 5.3 `seq-mean-token-sum-norm` 在做什么

Dr.GRPO 的另一个重点在 loss 聚合方式。

它不满足于只改 advantage 的归一化，还希望改 response 长度参与 loss 的方式。

`seq-mean-token-sum-norm` 的直觉可以这样理解：

- 先把每个 sequence 的 token loss 汇总
- 再用一个全局常数或统一尺度去归一
- 避免不同长度 response 在 loss 中产生不合理偏差

在 `verl` 里，这对应：

```yaml
actor_rollout_ref.actor.loss_agg_mode = seq-mean-token-sum-norm
```

### 5.4 `loss_scale_factor` 在做什么

如果只说“做归一化”，还不够稳定，因为不同 batch 的 response 长度分布会变。

所以 Dr.GRPO 还常配：

```yaml
actor_rollout_ref.actor.loss_scale_factor = max_response_length
```

它的作用是：

- 用一个固定常数当归一化尺度
- 避免 batch-to-batch 的尺度变化太大

你可以把它理解成：

- 不再让 loss 的有效尺度过度依赖“这一批恰好生成了多长的答案”

### 5.5 Dr.GRPO 在 `verl` 里的对应关系

本地可直接对照：

- `../../examples/grpo_trainer/README.md`
- `../../verl/trainer/ppo/core_algos.py`

你需要记住的三个开关：

```yaml
algorithm.norm_adv_by_std_in_grpo = false
actor_rollout_ref.actor.loss_agg_mode = seq-mean-token-sum-norm
actor_rollout_ref.actor.loss_scale_factor = fixed max_response_length
```

README 里还建议：

- Dr.GRPO 一般不再配 KL loss

### 5.6 对 Dr.GRPO 的一句准确总结

`论文结论`：

- Dr.GRPO 主要是在指出 GRPO 的归一化和聚合方式可能引入偏置，并提出更干净的修正路线。

`本项目工程判断`：

- Dr.GRPO 很适合作为一个“理论明确、面试可讲、改动边界清晰”的关键消融。
- 但它不是最像“第一版稳定主线”的选择，因为它更像偏置修正，而不是完整训练 recipe。

---

## 6. DAPO vs Dr.GRPO：新手最容易混淆的地方

### 6.1 一张表看懂

| 对比项 | DAPO | Dr.GRPO |
|--------|------|---------|
| 修改层级 | 训练 recipe 级 | 偏置修正级 |
| 核心问题 | 训练稳定性、无效 group、更新方式 | advantage 归一化和长度相关偏置 |
| 是否是 bundle recipe | 是，多个部件打包 | 否，边界更聚焦 |
| 是否更偏理论修正 | 中等 | 强 |
| 对 `verl` 的现成支持度 | 高 | 高 |
| 对简历项目的解释成本 | 较低 | 中等 |
| 是否天然解决无效 group | 倾向能 | 不能 |
| 是否天然涉及 overlong 处理 | 是，常一起讨论 | 否 |

### 6.2 为什么它们不是互斥关系

这点非常重要。

很多人会把问题问成：

> “到底选 DAPO 还是选 Dr.GRPO？”

更准确的问法应该是：

> “我是更想要一套稳定训练 recipe，还是更想先验证 GRPO 的偏置修正？”

因为它们关注的层面并不完全相同：

- DAPO 更关心“训练怎么更稳、更高效”
- Dr.GRPO 更关心“原生 GRPO 的数学处理是否有偏”

所以它们不是理论上不能共存，而是你在一个简历项目里要决定：

- 主线更强调工程稳定性
- 还是主线更强调理论修正

### 6.3 为什么当前项目主线不选“纯 Dr.GRPO”

因为当前项目的第一优先级是：

- 稳定
- 可复现
- 可解释
- 能围绕 verifier-based coding RL 讲出完整闭环

在这个优先级下，Dr.GRPO 的问题在于：

- 它很值得做
- 但更像一条“我怀疑原生 GRPO 有偏，我来专门验证”的研究路线
- 不像 DAPO-style 那样天然带着一整套“工程上更稳”的叙事

所以它更适合作为关键消融：

- 很有价值
- 很值得跑
- 但不一定是第一版 headline route

### 6.4 为什么当前项目也不写成“full DAPO”

因为“full DAPO”有两个问题：

1. 容易把 reward shaping、group filtering、clip、loss aggregation 全绑死在一个标签里
2. 后续你很难解释“到底哪部分带来了收益”

所以对当前项目，更好的表述是：

- **`DAPO-style GRPO`**

这句话同时保留了三层含义：

1. 主干还是 GRPO
2. 我吸收了 DAPO 的成熟稳定化经验
3. 我没有宣称自己在做 full DAPO reproduction

---

## 7. 还要知道，但当前不主选的路线

这一节的目标不是深入研究，而是避免你在面试中被问到时完全没概念。

### 7.1 GSPO

GSPO 可以理解成：如果 reward 是 sequence-level 的，那 policy ratio 和 clipping 也应该更一致地在 sequence 层面建模。

它更偏目标函数级创新。

优点：

- 理论味更强
- 对 sequence-level reward 的表述更自然

为什么当前不作为主线：

- 对简历项目来说解释成本更高
- 与当前 `accepted@1 + verifier reward` 主线相比，未必比 DAPO-style 更容易讲清收益
- 第一版系统更应先证明“闭环做通且稳定”，而不是把主要风险押在更换目标函数上

### 7.2 pass@k / PKPO / `grpo_passk`

这类方法的核心是：

- 不只看单个 candidate
- 更关心一组候选里“最好那个”能不能成功

它适合的问题是：

- 你真正关心的是 inference-time rerank / sample-and-select

为什么当前不作为主线：

- 当前项目主结果强调的是 `accepted@1`
- 如果训练目标改成偏 `pass@k`
- 那就会出现“训练目标”和最终汇报目标不完全一致”的解释负担

一句话记住：

- **当前项目不是不承认 pass@k 有价值，而是第一版主线不想把叙事搞复杂。**

### 7.3 VeRPO

VeRPO 对当前项目很重要，但它更重要的部分在 **reward 设计**，不在 headline optimizer。

它主要强调：

- 利用可验证任务的 dense reward
- 更细粒度地利用 unit-test 信号
- 把 reward 设计得更适合代码生成

为什么当前不作为主算法：

- 它更像 reward-side 方案
- 你后面会单独做 reward 设计决策
- 所以不应该在“算法主线”这一节把它和优化器混写

### 7.4 这一节的结论

`论文结论`：

- GSPO、pass@k RL、VeRPO 都在各自方向上解决了真实问题。

`本项目工程判断`：

- 第一版简历项目应优先选择：
  - 主干清楚
  - 实现成本可控
  - 面试表达成本低
  - 能稳定跑出结果

因此它们当前都不是第一版主线。

---

## 8. 算法路线图：当前有哪些可选方向

| 路线 | 你可以把它理解成什么 | 主要解决什么 | 当前项目推荐级别 |
|------|----------------------|--------------|------------------|
| 原生 GRPO | 最经典的 critic-free 组相对优化 | 先把 verifier RL 跑起来 | 只做 baseline |
| DAPO-style GRPO | GRPO + 稳定化 recipe | 训练稳定性、无效 group、探索与效率 | 主线推荐 |
| Dr.GRPO | GRPO 偏置修正路线 | std 归一化和长度相关偏置 | 关键消融 |
| GSPO | sequence-level objective 改写 | 更一致的序列级 policy update | 当前不主选 |
| pass@k RL | best-of-k 训练目标 | 优化 sample-and-select 结果 | 当前不主选 |

---

## 9. 回到你的项目：最终算法决策怎么落地

这一节只讨论算法主线，不讨论 reward 细节实现。

### 9.1 当前推荐的三组实验

#### A0：Stable GRPO Baseline

用途：

- 给后续主线一个清晰对照
- 证明不是“任何 RL 配置都一样”

建议配置：

| 字段 | 建议值 |
|------|--------|
| `adv_estimator` | `grpo` |
| `norm_adv_by_std_in_grpo` | `true` |
| `loss_agg_mode` | `token-mean` |
| `clip_ratio_low` | `0.2` |
| `clip_ratio_high` | `0.2` |
| `use_kl_loss` | `true` |
| `kl_loss_coef` | `0.001` |
| `group_size` | `4` |
| `filter_groups` | `off` |

解释：

- 这不是“最原始论文复刻版 GRPO”
- 而是更适合当前 `verl` 工程环境的稳定 baseline

#### A1：DAPO-style GRPO Mainline

用途：

- 作为当前主线方案
- 代表最终对外采用的算法路线

建议配置：

| 字段 | 建议值 |
|------|--------|
| `adv_estimator` | `grpo` |
| `norm_adv_by_std_in_grpo` | `true` |
| `loss_agg_mode` | `token-mean` |
| `clip_ratio_low` | `0.2` |
| `clip_ratio_high` | `0.28` |
| `use_kl_loss` | `false` |
| `group_size` | `4` |
| `filter_groups.enable` | `true` |
| `filter_groups.metric` | `seq_reward` |
| `filter_groups.max_num_gen_batches` | `10` |

解释：

- 主干仍然是 GRPO
- 但吸收了 DAPO 风格的稳定化改动
- 这里不把 overlong shaping 写进 headline algorithm 定义

#### A2：Dr.GRPO Ablation

用途：

- 回答“是否应该优先修正 GRPO 偏置，而不是采用 DAPO-style recipe”

建议配置：

| 字段 | 建议值 |
|------|--------|
| `adv_estimator` | `grpo` |
| `norm_adv_by_std_in_grpo` | `false` |
| `loss_agg_mode` | `seq-mean-token-sum-norm` |
| `loss_scale_factor` | `fixed max_response_length` |
| `use_kl_loss` | `false` |
| `group_size` | `4` |

解释：

- A2 更像另一条“偏置修正路线”
- 它和 A1 的对比，不是为了证明谁绝对正确
- 而是为了让你的算法决策变得可验证、可答辩
- 如果实验预算允许，A2 最好尽量继承 A1 的公共设置，只替换 Dr.GRPO 相关开关，这样归因会更干净

### 9.2 为什么主线是 A1，而不是 A0 或 A2

选择 A1 的理由可以压缩成四句话：

1. 它仍然属于 GRPO 家族，critic-free，系统复杂度低
2. 它比“纯原生 GRPO”更现代，体现了你做过算法筛选
3. 它比“纯 Dr.GRPO”更像稳定第一版主线
4. 它比 GSPO / pass@k objective 更容易围绕 `accepted@1` 和 verifier-based coding RL 讲出完整故事

### 9.3 算法主线与 reward 设计的边界

这一点必须记住。

当前项目里：

- 算法主线 = 如何更新 policy
- reward 设计 = verifier 输出被映射成什么分数

这两个不能完全混在一起，否则后面很难归因。

因此当前推荐写法是：

- headline algorithm 写成 `DAPO-style GRPO`
- reward 只写成 `verifier-based dense reward`
- 不在 headline algorithm 里塞进具体的 piecewise penalty 细节

### 9.4 一个必须写进实现说明的注意事项

`filter_groups` 这一项，当前必须保留一个实现侧提醒：

- **在本地 workspace 中，配置、脚本和文档对 `filter_groups` 的支持是明确存在的**
- **但在正式对外宣称它已作为主线稳定生效前，仍应先通过实际 smoke run 和日志验证它是否在当前训练入口里真的被消费**

换句话说：

- 这不是“不要用”
- 而是“不要只因为配置写上了就默认它一定生效”

---

## 10. 面试答辩速记版

下面这些问答，不是论文标准答案，而是更适合简历项目口径的回答模板。

### Q1：为什么不直接用原生 GRPO？

原生 GRPO当然能跑，但它已经不是 2025 年以后最完整的工程实践。对于 coding RL，后续公开经验已经指出了长响应不稳、无效 group、KL 过保守等问题。我的选择不是否定原生 GRPO，而是把它作为 baseline，再在主干不变的前提下吸收更成熟的稳定化改动。

### Q2：为什么不直接写成 full DAPO？

因为 DAPO 更像一套 recipe，而不是一个边界极清晰的单点 estimator。对简历项目来说，如果一上来就全盘打包，很难解释到底哪一项改动带来了收益。我更希望保留“GRPO 主干 + 选取最成熟的 DAPO-style 稳定化部件”这个表述，让算法决策和 reward 决策还能分开讲。

### Q3：为什么 Dr.GRPO 不是唯一主线？

Dr.GRPO 很有价值，它解决的是原生 GRPO 的归一化和聚合偏置问题。但对当前项目来说，第一优先级是先把 verifier-based coding RL 稳定跑通，而不是一开始就把主要风险押在偏置修正路线本身。所以我把它放在关键消融位置，让这个问题可以被验证，而不是直接变成唯一主线。

### Q4：为什么不用 GSPO？

GSPO 更偏目标函数级创新，研究味更强。它不是没有价值，而是对第一版简历项目来说，实现与解释成本都更高。当前项目更适合先把 `accepted@1 + verifier reward + GRPO family` 这一闭环做扎实，再考虑更换 objective。

### Q5：为什么不用 pass@k 训练目标？

因为当前项目主结果强调的是 `accepted@1`，而 pass@k 更适合“采样多个候选后选最好一个”的场景。第一版如果把训练目标改成 pass@k，会增加“训练目标与汇报目标不完全一致”的解释成本。我更希望先保证训练目标和对外结果尽可能一致。

### Q6：为什么 coding RL 里可以考虑 no-KL？

因为代码任务常常要求模型学习新的程序结构和边界处理模式，过强的 KL 容易把策略拉回 base model 附近，限制探索。不过 no-KL 不是信条，如果 pilot run 里出现明显漂移，我仍然会保留小 KL 的 rescue 配置。这是一个工程权衡，而不是意识形态选择。

### Q7：你如何保证这不是“追热点”而是工程决策？

我的判断标准不是论文名字，而是三件事：能不能在 `verl` 里稳定落地，能不能围绕 verifier-based coding RL 形成完整闭环，能不能在面试里把收益和代价讲清楚。最终选择 `DAPO-style GRPO`，正是因为它同时满足实现成熟度、叙事清晰度和实验可控性。

### Q8：为什么说 DAPO 和 Dr.GRPO 不是一回事？

因为它们关注的问题不同。DAPO 更像训练 recipe，解决的是训练稳定性、采样效率和更新方式；Dr.GRPO 更像偏置修正路线，解决的是 advantage 归一化和长度相关偏置。它们不是互斥关系，但在简历项目里需要决定哪条更适合作为主线。

---

## 11. 你现在真正需要记住的 10 句话

1. 原生 GRPO 的核心是组内相对奖励，不需要 critic。
2. `verl` 里的常规 GRPO 工程实践，本来就不完全等于原始论文 recipe。
3. DAPO 不是“完全替代 GRPO 的新算法”，而是一套稳定化 recipe。
4. Dr.GRPO 不是 recipe，而是更偏数学偏置修正路线。
5. DAPO 更关心训练怎么更稳，Dr.GRPO 更关心原生 GRPO 的归一化是否有偏。
6. coding RL 里，长响应、无效 group、KL 过强、judge cost 都是真问题。
7. `token-mean` 很重要，但不是 DAPO 独占的能力。
8. `filter_groups` 很有吸引力，但必须先验证在当前训练入口中真的生效。
9. 对这个项目，`DAPO-style GRPO` 是更好的主线叙事，`Dr.GRPO` 是更好的关键消融。
10. 面试时不要说“我用了最新算法”，而要说“我在 GRPO 主干上选择了最适合 coding verifier 场景的一组稳定化改进”。

---

## 12. 参考资料

### 论文

- DeepSeekMath / 原始 GRPO：<https://arxiv.org/abs/2402.03300>
- DAPO：<https://arxiv.org/abs/2503.14476>
- Dr.GRPO / Understanding R1-Zero-Like Training：<https://arxiv.org/abs/2503.20783>
- GSPO：<https://arxiv.org/abs/2507.18071>
- VeRPO：<https://arxiv.org/abs/2601.03525>
- Pass@k 相关训练目标参考：<https://arxiv.org/abs/2503.19595>
- Why Pass@k Optimization Can Degrade Pass@1：<https://arxiv.org/abs/2602.21189>

### 本地实现与文档

- `../../examples/grpo_trainer/README.md`
- `../../verl/trainer/ppo/core_algos.py`
- `../../docs/algo/dapo.md`
- `../../tests/special_e2e/run_dapo.sh`
- `../../verl/workers/reward_manager/dapo.py`
- `../../SandboxFusion/sandbox/utils/testing.py`
- `../PROGRESS.md`

---

## 13. 最后的提醒

这篇文档的目的不是替你提前宣布实验结论，而是帮你把“为什么选这条路线”讲清楚。

所以请一直记住下面这个边界：

- **DAPO-style GRPO 是当前最合理的主线推荐**
- **不是已经被实验最终证明的唯一真理**

真正让这个项目变强的，不只是算法名字，而是三件事一起成立：

1. 奖励口径可信
2. 训练配置可复现
3. 评测与消融能支撑你的项目叙事

只要这三件事做好了，这个项目在简历和面试里的说服力就会很强。
