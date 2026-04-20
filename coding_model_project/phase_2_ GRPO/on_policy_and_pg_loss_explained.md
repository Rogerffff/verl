# on_policy 分支与 pg_loss 指标详解

本文档解释 10-step probe 中观察到的 pg_loss ≈ 0 现象：这到底是 bug 还是正常行为？模型到底有没有在更新？

---

## 0. 前置知识：从 REINFORCE 到 PPO 到 GRPO 的演进

要理解 pg_loss，需要先理解策略梯度算法的演进脉络。这三个算法是层层改进的关系。

### 0.1 REINFORCE（最原始的策略梯度）

**核心思想**：直接对策略求梯度，让获得高奖励的动作概率上升。

**目标函数**：最大化期望回报
```
J(θ) = E[R × log π_θ(a|s)]
```

**梯度**：
```
∇_θ J = E[R × ∇_θ log π_θ(a|s)]
```

**直觉理解**：
- `log π_θ(a|s)` 是策略对动作 a 的对数概率
- `∇_θ log π_θ(a|s)` 是"让这个动作概率上升"的方向
- `R` 是奖励的标量权重
- 如果 R > 0：推动策略增加这个动作的概率
- 如果 R < 0：推动策略降低这个动作的概率

**缺点**：
1. **高方差**：R 可能很大或很小，导致梯度估计噪声大
2. **没有信任域约束**：一步可能更新太大，导致策略崩溃

### 0.2 加入 Baseline → Advantage

为了降低方差，把 R 替换成 Advantage（优势）：

```
A(s,a) = R(s,a) - baseline
```

baseline 可以是：
- **V(s)**：状态价值函数（PPO 用 critic 网络估计）
- **组内均值**：同一 prompt 的多个回复的平均 reward（GRPO 的做法）

用 Advantage 后梯度变成：
```
∇_θ J = E[A × ∇_θ log π_θ(a|s)]
```

A 有正有负，均值接近 0，比直接用 R 方差小得多。

### 0.3 PPO（Proximal Policy Optimization）

**解决的问题**：REINFORCE + Advantage 没有限制更新幅度。如果某一步的梯度特别大，策略可能一步跳到很远的地方，然后崩溃（这在 RL 中很常见）。

**PPO 的核心改进**：引入 importance sampling ratio 和 clip。

```
ratio = π_θ(a|s) / π_old(a|s)     ← 新旧策略的概率比
L = -min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
```

**为什么需要 ratio？**

在标准 REINFORCE 中，数据必须是当前策略 π_θ 生成的（on-policy）。但实际训练中，我们想在同一批数据上做多次更新（多个 mini_batch 或多个 epoch）。第二次更新时，策略已经变了，数据不再是"当前策略生成的"。

Importance sampling 通过 `ratio = π_θ / π_old` 来修正这个偏差：
- 如果 π_θ(a) > π_old(a)（新策略更倾向这个动作），ratio > 1 → 放大梯度
- 如果 π_θ(a) < π_old(a)（新策略不那么倾向），ratio < 1 → 缩小梯度

**clip 的作用**：限制 ratio 的范围，防止单步更新太大。

```
当 A > 0（好动作）：ratio 被 clip 在 [1-ε, 1+ε]
  → 策略最多把概率提高到 (1+ε) 倍，不能更多

当 A < 0（差动作）：ratio 被 clip 在 [1-ε, 1+ε]
  → 策略最多把概率降低到 (1-ε) 倍，不能更少
```

### 0.4 GRPO（Group Relative Policy Optimization）

**GRPO 相对于 PPO 的改变**：

| | PPO | GRPO |
|---|---|---|
| Advantage 来源 | Critic 网络估计 V(s) | 组内奖励归一化（不需要 critic） |
| 数据 | 每个 prompt 生成 1 个回复 | 每个 prompt 生成 n 个回复（如 n=8） |
| Advantage 计算 | `A = R - V(s)`（GAE） | `A_i = (R_i - μ_group) / σ_group` |
| Policy loss | PPO clipped loss | **完全相同的 PPO clipped loss** |

**关键点：GRPO 和 PPO 使用完全相同的 policy loss 函数。**

区别只在 advantage 怎么算。一旦 advantage 算好了，后面的 ratio、clip、pg_loss 公式完全一样。所以 pg_loss 这个指标对 GRPO 和 PPO 的含义完全相同。

在 verl 代码中，PPO 和 GRPO 走的是同一个 `compute_policy_loss_vanilla()` 函数（core_algos.py:1170），区别只在之前的 `compute_advantage()` 阶段选择了不同的 advantage estimator。

### 0.5 三者的关系图

```
REINFORCE:   L = -A × log π_θ(a|s)
                 │
                 │ 加入 importance sampling ratio
                 ▼
PPO:         L = -min(ratio × A, clip(ratio) × A)
                 │                    │
                 │ ratio = π_θ/π_old  │ clip 限制更新幅度
                 │                    │
                 │ 改变 advantage 计算方式
                 ▼
GRPO:        L = -min(ratio × A, clip(ratio) × A)   ← loss 公式完全相同
             但 A = (R_i - μ_group) / σ_group         ← advantage 不同
             不需要 critic                              ← 架构更简单
```

### 0.6 为什么说"on_policy 下 PPO loss 退化成 REINFORCE"

当 `ratio = 1.0`（on_policy 模式）时：
```
L_PPO = -min(1.0 × A, clip(1.0, 1-ε, 1+ε) × A)
      = -min(A, A)
      = -A
```

梯度：
```
∇L = -A × ∇log π_θ(a|s)
```

这和 REINFORCE 的梯度公式**完全一样**。PPO 的 importance sampling 和 clip 都不起作用了，因为 ratio 恒等于 1.0。

但这**不是坏事**——在严格 on-policy 场景下（数据就是当前策略生成的），REINFORCE 梯度本来就是无偏的，不需要 importance sampling 修正。PPO 的 ratio 和 clip 只在 off-policy 场景下才有意义。

---

## 1. PPO/GRPO 的 ratio 和 loss：公式与代码

### 1.1 Importance Sampling Ratio

```
ratio = π_θ(a|s) / π_old(a|s)
```

在代码中（core_algos.py:1220-1223）：
```python
negative_approx_kl = log_prob - old_log_prob      # log(π_θ/π_old)
ratio = torch.exp(negative_approx_kl)              # π_θ/π_old
```

ratio 的含义：
- ratio = 1.0 → 新策略和旧策略对这个 token 的概率一样
- ratio > 1.0 → 新策略更倾向于输出这个 token
- ratio < 1.0 → 新策略不那么倾向于输出这个 token

### 1.2 PPO Clipped Loss（GRPO 也用这个）

```
L_clip = -E[min(ratio × A, clip(ratio, 1-ε_low, 1+ε_high) × A)]
```

在代码中（core_algos.py:1226-1253）：
```python
pg_losses1 = -advantages * ratio                                           # 无 clip
pg_losses2 = -advantages * clamp(ratio, 1 - clip_low, 1 + clip_high)      # 有 clip
pg_loss = max(pg_losses1, pg_losses2)                                       # 取较大的（更保守的）
```

clip 的作用：当 ratio 偏离 1.0 太多时，loss 不再随 ratio 变化。这限制了单步更新幅度。

**DAPO 风格（A1 配置）**：`clip_low=0.2, clip_high=0.28`，非对称 clip 允许好动作有更大的概率提升空间。

### 1.3 监控指标的含义

| 指标 | 公式 | 含义 | PPO/GRPO 是否相同 |
|---|---|---|---|
| **pg_loss** | `mean(L_clip)` | 策略正在被推动改变的程度 | 完全相同 |
| **clipfrac** | `mean(pg_losses2 > pg_losses1)` | 被 clip 截断的 token 比例 | 完全相同 |
| **ppo_kl** | `mean(old_log_prob - log_prob)` | 新旧策略的近似 KL 散度 | 完全相同 |
| **grad_norm** | `‖∇L‖₂` | 梯度的 L2 范数 | 完全相同 |

这些指标在 GRPO 和 PPO 中含义完全一致，因为它们都是从同一个 `compute_policy_loss_vanilla()` 函数输出的。

**注意**：这里的 `ppo_kl` 是 `π_old` 和 `π_θ` 之间的 KL，不是和 ref model 的 KL。这是两个 agent 混淆的点。`π_old` 是同一步内的"更新前快照"，`π_ref` 是训练开始时冻结的初始模型。

---

## 2. on_policy 分支是什么

### 2.1 代码位置

`dp_actor.py:541-594`：

```python
# Line 541: 把数据按 mini_batch_size 拆分
mini_batches = data.split(self.config.ppo_mini_batch_size)

# Line 543: 判断是否为 on_policy 模式
on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

# Line 581-584: 前向传播得到当前策略的 log_prob
outputs = self._forward_micro_batch(model_inputs, ...)
log_prob = outputs["log_probs"]

# Line 591-594: on_policy 模式下替换 old_log_prob
if on_policy:
    old_log_prob = log_prob.detach()    # ← 关键：用当前的 log_prob 替换
else:
    old_log_prob = model_inputs["old_log_probs"]   # 用预计算的
```

### 2.2 `log_prob.detach()` 做了什么

`log_prob` 是当前策略 π_θ 前向传播的输出，它依赖于模型参数 θ。

`.detach()` 把这个张量从计算图中断开——它的值不变，但不再追踪对 θ 的梯度。

所以 `old_log_prob = log_prob.detach()` 之后：
```
ratio = exp(log_prob - log_prob.detach())
```

从**数值**上看：`log_prob - log_prob.detach() = 0`，`ratio = 1.0`。
从**梯度**上看：`log_prob` 有梯度，`log_prob.detach()` 没有梯度。

### 2.3 为什么 verl 要这么做

这是一个 on-policy 优化。当只有 1 个 mini_batch 且只做 1 个 epoch 时：

- 你训练用的数据就是刚刚生成的
- 前向传播时的 `log_prob` 就是当前策略对这些 token 的概率
- 如果用之前预计算的 `old_log_probs`（rollout 阶段或 compute_log_prob 阶段算的），由于浮点精度、padding 方式、micro_batch 拆分方式等差异，可能和当前前向传播的 `log_prob` 有微小数值偏差
- 这些偏差会导致 ratio 不严格等于 1.0，产生虚假的 `ppo_kl > 0`，误导监控

用 `log_prob.detach()` 可以消除这些数值噪声，让 on-policy 场景下的 loss 更干净。

---

## 3. 为什么当前配置触发了 on_policy

### 3.1 ppo_mini_batch_size 的归一化

`fsdp_workers.py:242-243`：
```python
self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
self.config.actor.ppo_mini_batch_size //= self.device_mesh.size()
```

你的 config：
```
ppo_mini_batch_size = 8 (原始值)
rollout.n = 8
device_mesh.size() = 4 (4 张 GPU)
```

归一化计算：
```
step 1: 8 × 8 = 64
step 2: 64 ÷ 4 = 16  ← 这是每张 GPU 上的 ppo_mini_batch_size
```

### 3.2 每张 GPU 上的数据量

dispatch 把全局 batch 均分到 4 张 GPU：
```
全局 batch = train_batch_size × rollout.n = 8 × 8 = 64
每张 GPU = 64 ÷ 4 = 16
```

### 3.3 mini_batch 数量

```python
mini_batches = data.split(ppo_mini_batch_size)
# = 16 条数据.split(16)
# = [16 条]  ← 只有 1 个 mini_batch
```

### 3.4 触发条件

```python
on_policy = len(mini_batches) == 1 and ppo_epochs == 1
#         = (1 == 1)              and (1 == 1)
#         = True
```

### 3.5 核心公式

判断 on_policy 是否触发的通用公式：

```
per_gpu_data = (train_batch_size × rollout.n) ÷ num_gpus
per_gpu_mini_batch = (ppo_mini_batch_size × rollout.n) ÷ num_gpus
num_mini_batches = per_gpu_data ÷ per_gpu_mini_batch

简化后：
num_mini_batches = train_batch_size ÷ ppo_mini_batch_size

如果 train_batch_size == ppo_mini_batch_size → num_mini_batches = 1 → on_policy = True
```

**关键发现**：rollout.n 和 num_gpus 在分子分母中相互抵消了。判断 on_policy 只取决于 `train_batch_size` 和 `ppo_mini_batch_size` 的比值。

你的配置：`train_batch_size = 8, ppo_mini_batch_size = 8` → 比值 = 1 → on_policy = True。

---

## 4. on_policy 模式下梯度是否还在流动

**这是最关键的问题。答案是：梯度仍然在流动，模型仍然在更新。**

### 4.1 数学推导

on_policy 模式下：
```
old_log_prob = log_prob.detach()
ratio = exp(log_prob - log_prob.detach()) = exp(0) = 1.0   ← 数值上
```

代入 PPO loss（忽略 clip，因为 ratio=1.0 不会触发 clip）：
```
L = -advantage × ratio = -advantage × exp(log_prob - log_prob.detach())
```

对 θ 求梯度（链式法则）：
```
∂L/∂θ = -advantage × exp(log_prob - log_prob.detach()) × ∂(log_prob)/∂θ
                                                          └──────────────┘
                                                          log_prob.detach() 对 θ 没有梯度
       = -advantage × 1.0 × ∂(log_prob)/∂θ
       = -advantage × ∂(log_prob)/∂θ
```

**这就是标准的 REINFORCE 策略梯度！**

```
∇_θ L = -A × ∇_θ log π_θ(a|s)
```

### 4.2 直觉理解

on_policy 模式下，PPO loss 退化成了没有 importance sampling 修正的 REINFORCE：

| 模式 | Loss | 梯度 |
|---|---|---|
| off_policy (正常 PPO) | `-A × (π_θ/π_old)` | `-A × (π_θ/π_old) × ∇log π_θ` |
| on_policy (当前) | `-A × 1.0` | `-A × ∇log π_θ` |

两者的**梯度方向完全相同**（都是 `-A × ∇log π_θ`），差异只在 importance sampling 的权重。在 on-policy 场景下 `π_θ ≈ π_old`，权重本来就接近 1.0，所以两种模式的实际效果几乎一样。

### 4.3 为什么 pg_loss 数值接近 0

```
pg_loss = mean(-advantage × ratio) = mean(-advantage × 1.0) = -mean(advantage)
```

GRPO 的 advantage 经过组内归一化后均值接近 0（因为减去了组内均值）。所以 `pg_loss ≈ 0` 是数学必然，不是"没有梯度"。

类比：函数 f(x) = x² 在 x=0 处的值也是 0，但梯度 f'(0) = 0。不过这个类比不完全准确——这里 pg_loss ≈ 0 但梯度不为 0，因为 advantage 对每个 token 的值不同（有正有负），只是均值接近 0。

### 4.4 grad_norm 的证据

probe 的 grad_norm 在 0.6 - 1.3 之间，**如果梯度为零，grad_norm 应该严格为 0**。

```
Step  grad_norm
1     1.349
2     1.079
3     0.658
4     1.304
5     0.611
6     0.794
7     0.689
8     0.896
9     0.752
10    0.789
```

这证实了模型参数确实在被梯度推动。`optimizer.step()` 也在正常执行（远端日志没有 `grad_norm is not finite` 跳过更新的警告）。

---

## 5. 监控指标的诊断价值

### 5.1 on_policy 模式下失去诊断价值的指标

| 指标 | on_policy 下的行为 | 是否有诊断价值 |
|---|---|---|
| **pg_loss** | 恒等于 `-mean(advantage)` ≈ 0 | 没有 — 无法区分"学得好"和"没在学" |
| **clipfrac** | 恒等于 0（ratio=1.0 永远不触发 clip） | 没有 |
| **ppo_kl** | 恒等于 0（log_prob - log_prob.detach() = 0） | 没有 |

### 5.2 仍然有效的指标

| 指标 | 含义 | 是否正常 |
|---|---|---|
| **grad_norm** | 梯度的 L2 范数 | 0.6-1.3，正常 |
| **entropy** | 策略的探索度 | 0.15-0.21，正常 |
| **train score_mean** | 训练 batch 的平均 reward | 从 0.03 上升到 0.26，有信号 |
| **val metrics** | 验证集指标 | 10 步太少 + val 集小，暂时平坦是合理的 |

### 5.3 为什么 validation 平坦不能说明"没学到东西"

- CodeContests valid 只有 117 题，val_batch_size=32 可能只采了一部分
- `accepted@1` 以 1/9 ≈ 11.1% 的粒度跳（因为 val 中可能只有 9 道 CC 题），10 步内不可能看到连续变化
- `pass_ratio_all/mean` 从 0.151 到 0.162 有微弱波动，在采样噪声范围内
- **10 步是 infra probe，不是训练实验**。判断 policy 是否在改善至少需要 50-100 步

---

## 6. 如何恢复指标诊断能力

如果你想让 pg_loss / clipfrac / ppo_kl 重新变得有意义，有两种方法：

### 方法 A：让每张 GPU 有多个 mini_batch

```bash
# 关键：ppo_mini_batch_size < train_batch_size
TRAIN_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=4   # 8/4 = 2 个 mini_batch per GPU → on_policy=False
```

效果：
```
per_gpu_data = (8 × 8) / 4 = 16
per_gpu_mini_batch = (4 × 8) / 4 = 8
num_mini_batches = 16 / 8 = 2 → on_policy = False
```

此时第二个 mini_batch 训练时，策略已经被第一个 mini_batch 更新过了，`old_log_probs` 和当前的 `log_prob` 会有真实差异，ratio ≠ 1.0，pg_loss / clipfrac / ppo_kl 就会显示有意义的值。

### 方法 B：增加 ppo_epochs

```bash
PPO_EPOCHS=2   # 或 4
```

效果：即使只有 1 个 mini_batch，第二个 epoch 时策略已经被第一个 epoch 更新过了，同理。

### 要不要改？

**不一定需要改。** on_policy 模式下梯度仍然有效（本质是 REINFORCE），模型仍然在更新。改这个只是为了让监控面板更好看。

PPO 原始论文建议 ppo_epochs=3-10，多个 epoch 可以更充分利用每个 batch 的数据。但这也增加了过拟合当前 batch 的风险，需要 clip 来控制。

如果你的训练在 50-100 步后 validation 指标开始上升，那说明即使在 on_policy 模式下训练也是有效的，不需要改配置。只有在验证指标长期完全不动时才需要考虑调整。

---

## 7. 总结

| 问题 | 答案 |
|---|---|
| pg_loss ≈ 0 是 bug 吗？ | 不是。是 `on_policy=True` 分支的预期行为 |
| 模型在更新吗？ | 在更新。grad_norm > 0 证实梯度在流动 |
| 为什么 ratio 恒等于 1.0？ | `old_log_prob = log_prob.detach()`，数值上相等但梯度只从 `log_prob` 流过 |
| on_policy 下 loss 等价于什么？ | REINFORCE 策略梯度：`∇L = -A × ∇log π_θ` |
| 为什么 verl 要做这个优化？ | 消除 on-policy 场景下的浮点精度噪声 |
| 什么配置触发 on_policy？ | `train_batch_size == ppo_mini_batch_size` 且 `ppo_epochs == 1` |
| 需要修复吗？ | 不紧急。如果想恢复监控指标，降 ppo_mini_batch_size 或增 ppo_epochs |
| 10 步 validation 平坦正常吗？ | 正常。10 步太少，val 集太小，不能作为判断依据 |

---

# 第二部分：verl 数据流完整介绍

本部分系统性地介绍 verl 中数据从 parquet 文件到 GPU 张量的完整生命周期。理解这条数据流是理解 `train_batch_size`、`ppo_mini_batch_size`、`rollout.n`、on_policy 触发条件等所有训练旋钮的前提。

---

## 8. 数据相关的核心配置

verl 的数据配置散落在 4 个层级，理解它们的关系是理解整个训练流程的关键。

### 8.1 配置层级

| 配置 | 脚本中的位置 | 含义 | 影响阶段 |
|------|-------------|------|----------|
| `data.train_files` | 脚本顶层 | parquet 文件路径 | Dataset 加载 |
| `data.train_batch_size` | 脚本顶层 | dataloader 每步取多少 **prompt** | Dataloader |
| `data.max_prompt_length` | 脚本顶层 | prompt 最大 token 数 | Dataset 过滤 |
| `data.max_response_length` | 脚本顶层 | response 最大 token 数 | Rollout 截断 |
| `data.filter_overlong_prompts` | 脚本顶层 | 是否过滤超长 prompt | Dataset 加载 |
| `data.truncation` | 脚本顶层 | 超长时的处理策略 | tokenizer |
| `data.shuffle` | 脚本顶层 | dataloader 是否 shuffle | Sampler |
| `data.seed` | 脚本顶层 | shuffle 的随机种子 | Sampler |
| `actor_rollout_ref.rollout.n` | 顶层 | 每个 prompt 生成多少个 response | Rollout 后 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 顶层（脚本值）| 每次 optimizer step 处理的 sample 数 | Actor update |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 顶层 | 每次 forward+backward 的 sample 数 | Actor update |

### 8.2 几个最容易混淆的概念

**prompt vs sample 的区别**：

- **prompt**：dataset 中的一行原始数据，对应一道题
- **sample**：rollout 后的一个完整 trajectory（prompt + response），一个 prompt 经过 `rollout.n=8` 后会变成 8 个 sample

**全局 vs 每 GPU 的区别**：

- `train_batch_size = 8` 是**全局**的（dataloader 一次取 8 个 prompt）
- `ppo_mini_batch_size = 8` 在脚本里写的是**全局**的，但会被 verl 做归一化（见 §11.2）
- `ppo_micro_batch_size_per_gpu = 1` 是**每 GPU** 的

**步数关系**：

```
1 个 training step
  = dataloader 取 1 次
  = train_batch_size 个 prompt
  = train_batch_size × rollout.n 个 sample
  = (train_batch_size × rollout.n) / num_gpus 个 sample per GPU
```

---

## 9. 数据流第一阶段：Parquet → Dataset

### 9.1 Dataset 类：`RLHFDataset`

**位置**：`verl/utils/dataset/rl_dataset.py`，line 70-421

继承自 `torch.utils.data.Dataset`，是一个标准的 PyTorch Dataset。

### 9.2 启动时数据加载流程

`RLHFDataset.__init__()` 在 trainer 启动时执行一次，做以下事情（rl_dataset.py:143-179）：

```python
# Step 1: 把远端 parquet 拷到本地缓存
self._download()
# 拷贝到 ~/.cache/verl/rlhf/
# 支持 HTTP/HTTPS 和本地路径

# Step 2: 用 HuggingFace datasets 加载并 concat 多个文件
self._read_files_and_tokenize()
# self.dataframe = datasets.load_dataset("parquet", data_files=...)["train"]
# 多个文件会被 concat 成一个

# Step 3: 过滤超长 prompt（如果开启）
self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
# 用多进程 tokenize 每个 sample，丢弃 token 数 > max_prompt_length 的
```

### 9.3 数据所在位置

**关键：整个 dataset 加载完成后，全部存在 CPU 内存中（通过 HuggingFace `datasets.Dataset` 对象管理）**。

verl 不是 streaming 的——dataset 在启动时一次性全部加载到内存。对于 11785 题的 CodeContests 训练集来说，每行包含 prompt + 完整 testcases，可能占几 GB 的 CPU 内存。

### 9.4 `__getitem__` 返回什么

每次 dataloader 取一个 sample，调用 `__getitem__(idx)`（rl_dataset.py:336-357）：

```python
def __getitem__(self, item):
    row_dict: dict = self.dataframe[item]              # 从 HF Dataset 取一行
    row_dict["raw_prompt"] = self._build_messages(row_dict)  # 构造 chat messages
    row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)  # 占位 tensor
    
    # 提取 extra_info 中的字段
    row_dict["index"] = row_dict.get("extra_info", {}).get("index", 0)
    row_dict["tools_kwargs"] = ...
    row_dict["interaction_kwargs"] = ...
    return row_dict
```

返回的字典字段（对 CodeContests）：
- `prompt`：原始 message list（未 tokenize）
- `data_source`：数据集名称（如 `codecontests_train_wo_valid_big`）
- `reward_model`：嵌套 dict，包含 `ground_truth.test_cases`
- `extra_info`：嵌套 dict，包含 `problem_id` 等
- `raw_prompt`：构造好的 chat messages
- `dummy_tensor`：torch.Tensor (1,)，仅用于让 DataProto 的 batch 不为空

### 9.5 关键观察：tokenize 是 lazy 的

**Dataset `__getitem__` 不做 tokenize**。`raw_prompt` 是 message list，真正的 tokenize 发生在后续的 `SingleTurnAgentLoop.run()` 阶段（rollout 时）。

例外：`filter_overlong_prompts=True` 时，启动阶段会 tokenize 一次用来过滤，但过滤后丢弃 token id，只保留原始数据。

---

## 10. 数据流第二阶段：Dataset → DataLoader → DataProto

### 10.1 DataLoader 创建

**位置**：`verl/trainer/ppo/ray_trainer.py:408-415`

```python
self.train_dataloader = StatefulDataLoader(
    dataset=self.train_dataset,
    batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
    num_workers=self.config.data["dataloader_num_workers"],
    drop_last=True,
    collate_fn=collate_fn,
    sampler=train_sampler,
)
```

注意：这里用的是 `torchdata.stateful_dataloader.StatefulDataLoader`，**不是** `torch.utils.data.DataLoader`。区别在于它支持保存/恢复迭代状态（用于 checkpoint resume）。

### 10.2 Sampler 选择

**位置**：`verl/trainer/main_ppo.py:395-439`

verl 支持三种 sampler：

```python
def create_rl_sampler(data_config, dataset):
    # 情况 1: 自定义 curriculum sampler（高级用法）
    if data_config.sampler is not None and data_config.sampler.get("class_path"):
        sampler = curriculum_class(data_source=dataset, data_config=data_config)
    
    # 情况 2: shuffle=True 时用 RandomSampler
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    
    # 情况 3: shuffle=False 时用 SequentialSampler
    else:
        sampler = SequentialSampler(data_source=dataset)
```

注意：用的是 `torchdata.stateful_dataloader.sampler.RandomSampler`，**不是** `torch.utils.data.RandomSampler`。区别同样是支持 checkpoint resume。

`data.seed` 只对 `RandomSampler` 有意义。SequentialSampler 是确定性的不需要 seed。

### 10.3 collate_fn 做什么

**位置**：`verl/utils/dataset/rl_dataset.py:39-67`

```python
def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    
    # 区分 tensor 和非 tensor 字段
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)
    
    # tensor 沿 dim=0 stack
    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)
    
    # 非 tensor 转为 numpy object array
    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))
    
    return {**tensors, **non_tensors}
```

输出：
- Tensor 字段：`torch.Tensor` of shape `(batch_size, ...)`，**仍在 CPU**
- 非 Tensor 字段：`np.ndarray(dtype=object)` of shape `(batch_size,)`，**在 CPU**

### 10.4 DataProto 创建

**位置**：`ray_trainer.py:1440-1449`

```python
for batch_dict in self.train_dataloader:  # batch_dict 是 collate_fn 的输出
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
    
    # 为每个 sample 分配 uid（GRPO 分组用）
    batch.non_tensor_batch["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
    )
```

`DataProto.from_single_dict` 自动把 dict 拆成两部分：
- `batch`：所有 `torch.Tensor` 字段，封装为 `TensorDict`
- `non_tensor_batch`：所有 `np.ndarray` 字段

**此时的 DataProto 完全在 CPU 上**。

### 10.5 DataProto 是什么

**位置**：`verl/protocol.py:328-339`

```python
@dataclass
class DataProto:
    batch: TensorDict = None              # tensor 数据，可以 .to(device)
    non_tensor_batch: dict = field(...)   # numpy object 数据，永远在 CPU
    meta_info: dict = field(...)          # Python dict 元信息
```

**关键约束**：`batch` 中所有 tensor 的第 0 维必须相等，`non_tensor_batch` 中所有 array 的长度也必须等于 batch_size。

**关键特性**：
- `batch.to(device)` 把所有 tensor 搬到 GPU
- `non_tensor_batch` 永远不上 GPU（numpy 没法上 GPU）
- `meta_info` 是 Python 标量字典，跟着对象传递

---

## 11. 数据流第三阶段：DataProto → 多 GPU Worker

### 11.1 rollout.n 重复

**位置**：`ray_trainer.py:1461-1463`

```python
gen_batch_output = gen_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n,  # n=8
    interleave=True
)
```

`interleave=True` 的语义（protocol.py:982-1024）：

```
原 batch (train_batch_size=8 个 prompt):
  [p1, p2, p3, p4, p5, p6, p7, p8]

repeat(8, interleave=True) 后 (64 个 sample):
  [p1, p1, p1, p1, p1, p1, p1, p1,
   p2, p2, p2, p2, p2, p2, p2, p2,
   ...
   p8, p8, p8, p8, p8, p8, p8, p8]
```

每个 prompt 被复制 8 次，连续放在一起。后续 vLLM 生成时会对每个 sample 独立采样，得到 8 个不同的 response。

**此时 DataProto 仍在 CPU**。

### 11.2 ppo_mini_batch_size 的归一化

**这是非常容易混淆的一步。**

在 worker 内部初始化时，verl 会对 `ppo_mini_batch_size` 做归一化（fsdp_workers.py:242-243）：

```python
if self._is_actor:
    self.config.actor.ppo_mini_batch_size *= self.config.rollout.n           # × 8
    self.config.actor.ppo_mini_batch_size //= self.device_mesh.size()        # ÷ 4
```

**翻译**：
- 你脚本里写的 `ppo_mini_batch_size=8` 是**全局 prompt 级**的
- × `rollout.n=8` → 变成**全局 sample 级**：8 × 8 = 64
- ÷ `n_gpus=4` → 变成**每 GPU sample 级**：64 / 4 = 16

所以 actor 实际看到的 `ppo_mini_batch_size = 16`，**而不是脚本里写的 8**。

### 11.3 Worker dispatch：64 sample 怎么分到 4 GPU

`actor_rollout_wg.generate_sequences(gen_batch_output)` 不是简单的函数调用，而是经过 verl 的 dispatch 机制：

**位置**：`verl/single_controller/base/decorator.py`

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
def generate_sequences(self, prompts: DataProto):
    ...
```

调用流程：
1. **Dispatch 阶段**：`dispatch_lazy_compute_data_proto` 把 64 sample 的 DataProto 切成 4 个 chunk（每个 16 sample），用 `DataProto.chunk(4)`
2. **远程执行**：4 个 chunk 通过 Ray ObjectRef 发送给 4 个 worker，每个 worker 在自己的 GPU 上独立处理
3. **Collect 阶段**：4 个 worker 完成后，`collect_lazy_compute_data_proto` 把 4 个结果用 `DataProto.concat()` 拼回成 64 sample 的完整 batch

`DataProto.chunk()`（protocol.py:875-914）：

```python
def chunk(self, chunks: int) -> list["DataProto"]:
    # 沿 dim=0 切分 batch
    batch_lst = self.batch.chunk(chunks=chunks, dim=0)
    
    # 同步切分 non_tensor_batch
    chunk_indices = np.cumsum([batch.batch_size[0] for batch in batch_lst])[:-1]
    for key, val in self.non_tensor_batch.items():
        non_tensor_lst = np.array_split(val, chunk_indices.tolist())
    
    # 返回 chunks 个 DataProto，meta_info 共享
    return [DataProto(batch=..., non_tensor_batch=..., meta_info=self.meta_info) for ...]
```

### 11.4 数据上 GPU 的精确时刻

**位置**：`verl/workers/fsdp_workers.py:960-963`

```python
def generate_sequences(self, prompts: DataProto):
    assert self._is_rollout
    prompts = prompts.to(get_device_id())  # ← 这一行才把数据搬到 GPU
    ...
```

在这一刻之前，数据一直在 CPU（甚至跨进程通过 Ray serialization 传输都是 CPU 形式）。只有当某个 worker 真正要用这份数据计算时，才把它搬到自己负责的 GPU 上。

`DataProto.to(device)`（protocol.py:597-609）：

```python
def to(self, device) -> "DataProto":
    if self.batch is not None:
        self.batch = self.batch.to(device)  # 只搬 TensorDict
    return self
```

**关键**：只有 `batch` 被搬到 GPU，`non_tensor_batch` 始终在 CPU（numpy 不能在 GPU 上）。

### 11.5 rollout 完成后回到 CPU

**位置**：`fsdp_workers.py:1002`

```python
output = output.to("cpu")
get_torch_device().empty_cache()
return output
```

rollout 完成后，结果立即搬回 CPU 准备序列化返回给 trainer。这样做是因为：
1. Ray 的对象序列化只支持 CPU tensor
2. vLLM 需要释放 GPU 显存为下一阶段（compute_log_prob 或 actor update）让路

---

## 12. 数据流第四阶段：从 rollout 结果到 actor update

### 12.1 rollout 输出回到 trainer

`gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)` 返回时：
- 4 个 worker 的结果已经被 `collect_lazy_compute_data_proto` 拼回
- 完整 64 sample 的 DataProto 在 trainer 进程的 CPU 上

### 12.2 与原 batch 合并

**位置**：`ray_trainer.py:1551-1558`

```python
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)
if "response_mask" not in batch.batch.keys():
    batch.batch["response_mask"] = compute_response_mask(batch)
```

`union` 把原 prompt 信息（reward_model.ground_truth, extra_info 等）和 rollout 生成的 response 信息合并。两个 DataProto 的 batch_size 必须相等（都是 64）。

### 12.3 reward 计算

reward 阶段不在 GPU 上做（除非用 reward model）。`grpo_batch_reward.compute_score()` 是纯 CPU + sandbox 调用。

**关键**：reward 阶段批量调用 sandbox 对 64 个 sample 做并发判题，结果填回到 `non_tensor_batch` 中。

### 12.4 advantage 计算

**位置**：`ray_trainer.py:243-248`

```python
advantages, returns = compute_grpo_outcome_advantage(
    token_level_rewards=data.batch["token_level_rewards"],
    response_mask=grpo_calculation_mask,
    index=data.non_tensor_batch["uid"],
    invalid_mask=data.non_tensor_batch.get("invalid_for_rl"),
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
)
data.batch["advantages"] = advantages
```

advantage 计算在 trainer 进程的 CPU 上做（PyTorch 操作，但数据在 CPU）。算完后塞回 `data.batch["advantages"]`，仍在 CPU。

### 12.5 actor update：再次分发到 4 GPU

`actor_rollout_wg.update_actor(batch)` 经过和 rollout 类似的 dispatch 流程：
1. 64 sample 切成 4 chunk × 16
2. 每个 worker 收到一个 chunk
3. Worker 内部 `data.to(get_device_id())` 把 chunk 搬到自己的 GPU
4. 在 GPU 上做 ppo_mini_batch_size 切分（每 GPU 16 → 1 个 mini-batch），再做 ppo_micro_batch_size_per_gpu 切分（16 / 1 = 16 个 micro-batch）
5. 16 次 forward+backward，1 次 optimizer step
6. 返回 metrics 字典（无大数据）

---

## 13. 数据流总结表

| 阶段 | 组件 | 数据位置 | 数据格式 | shape |
|------|------|---------|----------|-------|
| 1 | parquet 文件 | 远端/本地磁盘 | binary | - |
| 2 | `RLHFDataset.__init__` | CPU 内存 | `datasets.Dataset` | 11785 行 |
| 3 | `Dataset.__getitem__` | CPU 内存 | `dict[str, Tensor/ndarray]` | 1 行 |
| 4 | `collate_fn` 输出 | CPU 内存 | `dict`（tensor stacked） | (8, ...) |
| 5 | `DataProto.from_single_dict` | CPU 内存 | DataProto | batch_size=8 |
| 6 | `repeat(rollout.n=8)` | CPU 内存 | DataProto | batch_size=64 |
| 7 | `chunk(4)` for dispatch | CPU 内存 | List[DataProto] | 4 × (batch_size=16) |
| 8 | Ray 序列化跨进程传输 | CPU（Ray object store）| 序列化字节 | - |
| 9 | Worker 收到，未上 GPU | CPU 内存 | DataProto | batch_size=16 |
| 10 | `prompts.to(get_device_id())` | **GPU** | DataProto (batch on GPU) | batch_size=16 |
| 11 | vLLM 生成 | **GPU** | DataProto + responses | batch_size=16 |
| 12 | `output.to("cpu")` | CPU 内存 | DataProto | batch_size=16 |
| 13 | Ray 序列化返回 | CPU | 序列化字节 | - |
| 14 | trainer 收到 4 个 chunk 拼回 | CPU 内存 | DataProto | batch_size=64 |
| 15 | reward 计算（sandbox 调用） | CPU + sandbox | DataProto + scores | batch_size=64 |
| 16 | advantage 计算（CPU PyTorch） | CPU 内存 | DataProto + advantages | batch_size=64 |
| 17 | dispatch 给 actor 4 worker | CPU → 4 chunk | List[DataProto] | 4 × 16 |
| 18 | actor worker 收到 | CPU 内存 | DataProto | 16 |
| 19 | actor 内部上 GPU | **GPU** | DataProto | 16 |
| 20 | 切 mini-batch (1 个) → micro-batch (16 个) | **GPU** | sub DataProto | 1 → 1 |
| 21 | 16 次 fwd+bwd + 1 次 optim step | **GPU** | gradients | - |
| 22 | 返回 metrics | CPU | dict | - |

---

## 14. 核心数据流关键事实

### 14.1 数据在 CPU 还是 GPU？默认在 CPU

verl 的数据流默认在 CPU 上传输和处理。**只有真正要做 GPU 计算的时刻**，才显式 `.to(get_device_id())` 把它搬到 GPU。计算完成后立即 `.to("cpu")` 搬回。

这个设计的好处：
- Ray 跨进程传输不需要管 GPU
- vLLM 和 FSDP actor 共享 GPU 显存（混合引擎）
- 不需要 GPU 之间直接通信传 batch 数据（只需 FSDP 内部通信传 weight/grad）

### 14.2 数据从来不会"留在"某个 GPU

每次 worker 调用结束后，数据会被搬回 CPU 序列化返回。下次同一个 worker 再被调用时，数据会重新从 CPU 上传到 GPU。这意味着：
- 同一个 sample 会**多次**经过 PCIe 上 GPU（rollout 一次、compute_log_prob 一次、update_actor 一次）
- 但每次只是元数据级的 tensor（prompt + response），不是模型参数级的传输
- 所以 PCIe 4.0 vs 5.0 对 batch 数据传输影响很小

### 14.3 dataloader 是 stateful 的

`StatefulDataLoader` 的 `state_dict()` 和 `load_state_dict()` 用于 checkpoint resume：
- 训练每个 step 后，dataloader 的迭代位置可以保存到 `data.pt`
- resume 时从精确的位置继续
- 如果设置 `trainer.skip_dataloader_state_load=True`，则忽略 dataloader 状态从头开始

### 14.4 dataset 永远在 CPU 内存中

`RLHFDataset` 不是 streaming 的——所有 11785 行训练数据在 trainer 启动时就加载到 CPU 内存。这意味着：
- 启动时一次性的内存开销可能较大（尤其是 codecontests 这种带完整 testcases 的数据）
- 但运行时 dataloader 取数极快（直接内存索引）
- 适合 ~10^4 量级的训练集；如果是 ~10^7 级别需要换 streaming dataset

### 14.5 Ray ObjectRef 实现"零拷贝"传递

Ray 的 plasma object store 使得 DataProto 在 worker 之间可以高效传递。但 verl 的设计是：
- ObjectRef 在 trainer 进程内是引用
- 跨进程传递时通过 Ray 的零拷贝机制（共享内存）
- 只有当 worker 真正要"访问"数据时才 deserialize

### 14.6 prompt 的 tokenization 在 rollout 时才发生

回顾：dataset `__getitem__` 不做 tokenize。tokenize 真正发生在：
1. （可选）启动时为了过滤超长 prompt 做一次（结果丢弃）
2. `SingleTurnAgentLoop.run()` 中调用 `apply_chat_template` 时（结果传给 vLLM）

这个设计让 dataset 可以保留原始的 message 格式，方便支持多轮对话和工具调用。

---

## 15. 关键代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| Dataset 类 | `verl/utils/dataset/rl_dataset.py` | 70-421 |
| Dataset `__getitem__` | `verl/utils/dataset/rl_dataset.py` | 336-357 |
| filter_overlong_prompts | `verl/utils/dataset/rl_dataset.py` | 181-261 |
| collate_fn | `verl/utils/dataset/rl_dataset.py` | 39-67 |
| dataloader 创建 | `verl/trainer/ppo/ray_trainer.py` | 408-415 |
| sampler 选择 | `verl/trainer/main_ppo.py` | 395-439 |
| DataProto 类定义 | `verl/protocol.py` | 328-339 |
| `DataProto.from_single_dict` | `verl/protocol.py` | 491-504 |
| `DataProto.to(device)` | `verl/protocol.py` | 597-609 |
| `DataProto.repeat` | `verl/protocol.py` | 982-1024 |
| `DataProto.chunk` | `verl/protocol.py` | 875-914 |
| `DataProto.union` | `verl/protocol.py` | 792-809 |
| `DataProto.concat` | `verl/protocol.py` | 928-972 |
| dispatch 机制 | `verl/single_controller/base/decorator.py` | 273-322 |
| `make_nd_compute_dataproto_dispatch_fn` | `verl/single_controller/base/decorator.py` | 325-329 |
| trainer 主循环数据准备 | `verl/trainer/ppo/ray_trainer.py` | 1440-1469 |
| ppo_mini_batch_size 归一化 | `verl/workers/fsdp_workers.py` | 240-247 |
| `generate_sequences` 上 GPU | `verl/workers/fsdp_workers.py` | 960-1006 |
| actor update 数据切分 | `verl/workers/actor/dp_actor.py` | 539-558 |
| dataloader 保存 | `verl/trainer/ppo/ray_trainer.py` | 1003-1007 |
| dataloader 加载 | `verl/trainer/ppo/ray_trainer.py` | 1073-1086 |

