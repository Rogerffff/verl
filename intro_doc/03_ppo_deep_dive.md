# PPO 算法深度解析

## 1. PPO 算法原理回顾

Proximal Policy Optimization (PPO) 是 OpenAI 于 2017 年提出的策略梯度算法，通过限制策略更新幅度来提高训练稳定性。

### 1.1 核心思想

PPO 的核心目标函数：

```
L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
```

其中：
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` 是重要性采样比率
- `A_t` 是优势估计
- `ε` 是裁剪范围（通常为 0.2）

### 1.2 PPO 在 LLM 中的应用

在 LLM 的 RLHF 训练中，PPO 被用于：
1. **策略模型 (Actor)**：生成响应的语言模型
2. **价值模型 (Critic)**：估计状态价值，用于计算 GAE
3. **参考模型 (Reference)**：用于 KL 惩罚，防止策略偏离太远
4. **奖励模型 (Reward)**：评估响应质量

---

## 2. GAE 优势估计实现

**文件位置**：`verl/trainer/ppo/core_algos.py`，第 214-262 行

### 2.1 GAE 原理

Generalized Advantage Estimation (GAE) 是一种在偏差和方差之间权衡的优势估计方法：

```
A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
```

其中 TD 误差 `δ_t = r_t + γ * V(s_{t+1}) - V(s_t)`

### 2.2 代码实现

```python
@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    values: torch.Tensor,               # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    gamma: torch.Tensor,                # 折扣因子
    lam: torch.Tensor,                  # GAE lambda
):
    """计算 GAE 优势和回报"""
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        # 从后向前遍历（因为需要未来的价值）
        for t in reversed(range(gen_len)):
            # TD 误差: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]

            # GAE: A_t = δ_t + γλ * A_{t+1}
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # 使用 mask 处理 padding tokens
            nextvalues = values[:, t] * response_mask[:, t] + \
                         (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + \
                         (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)

        # 反转得到正确顺序
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # 计算回报: R_t = A_t + V_t
        returns = advantages + values

        # 白化优势（均值为0，标准差为1）
        advantages = verl_F.masked_whiten(advantages, response_mask)

    return advantages, returns
```

### 2.3 关键点解析

1. **反向遍历**：从序列末尾开始，因为需要 `V(s_{t+1})`
2. **Mask 处理**：对于 padding tokens，保持之前的值
3. **优势白化**：使优势值标准化，提高训练稳定性

---

## 3. PPO Clipped Loss 实现

**文件位置**：`verl/trainer/ppo/core_algos.py`，第 1159-1250 行

### 3.1 损失函数结构

```python
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,     # 旧策略的 log prob
    log_prob: torch.Tensor,         # 当前策略的 log prob
    advantages: torch.Tensor,       # 优势估计
    response_mask: torch.Tensor,    # 响应 mask
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
```

### 3.2 核心计算逻辑

```python
# 1. 计算重要性采样比率
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)  # 数值稳定性
ratio = torch.exp(negative_approx_kl)

# 2. 计算 KL 散度（用于监控）
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

# 3. 计算未裁剪的策略损失
pg_losses1 = -advantages * ratio

# 4. 计算裁剪后的策略损失
pg_losses2 = -advantages * torch.clamp(
    ratio, 1 - cliprange_low, 1 + cliprange_high
)

# 5. 取两者的最大值（悲观估计）
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
```

### 3.3 双重裁剪 (Dual-Clip PPO)

verl 支持 Dual-Clip PPO（论文：https://arxiv.org/pdf/1912.09729）：

```python
# 当优势为负时，使用额外的下界裁剪
pg_losses3 = -advantages * clip_ratio_c  # clip_ratio_c 默认为 3.0
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

# 根据优势符号选择最终损失
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

### 3.4 DAPO 风格的非对称裁剪

verl 支持 DAPO（https://arxiv.org/abs/2503.14476）风格的非对称裁剪：

```python
clip_ratio_low = config.clip_ratio_low   # 下界裁剪范围
clip_ratio_high = config.clip_ratio_high # 上界裁剪范围

# 非对称裁剪
pg_losses2 = -advantages * torch.clamp(
    ratio, 1 - cliprange_low, 1 + cliprange_high
)
```

---

## 4. 损失聚合模式

**文件位置**：`verl/trainer/ppo/core_algos.py`

verl 支持多种损失聚合模式：

### 4.1 聚合模式说明

| 模式 | 描述 | 公式 |
|-----|------|------|
| `token-mean` | 所有 token 平均 | `Σ loss / Σ mask` |
| `seq-mean-token-sum` | 每个序列 token 求和后取平均 | `mean(Σ_t loss_t)` |
| `seq-mean-token-mean` | 每个序列 token 平均后取平均 | `mean(mean_t(loss_t))` |
| `seq-mean-token-sum-norm` | DrGRPO 归一化 | 按序列长度归一化 |

### 4.2 实现代码

```python
def agg_loss(
    loss_mat: torch.Tensor,      # (bs, seq_len)
    loss_mask: torch.Tensor,     # (bs, seq_len)
    loss_agg_mode: str = "token-mean",
    **kwargs,
) -> torch.Tensor:
    """根据模式聚合损失"""

    if loss_agg_mode == "token-mean":
        # 全局 token 平均
        return verl_F.masked_mean(loss_mat, loss_mask)

    elif loss_agg_mode == "seq-mean-token-sum":
        # 每个序列求和，然后取序列平均
        seq_losses = (loss_mat * loss_mask).sum(dim=-1)
        return seq_losses.mean()

    elif loss_agg_mode == "seq-mean-token-mean":
        # 每个序列平均，然后取序列平均
        seq_lengths = loss_mask.sum(dim=-1).clamp(min=1)
        seq_losses = (loss_mat * loss_mask).sum(dim=-1) / seq_lengths
        return seq_losses.mean()
```

---

## 5. 价值函数损失

**文件位置**：`verl/trainer/ppo/core_algos.py`

### 5.1 价值损失函数

```python
def compute_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float = 0.2,
    loss_type: str = "l2",  # "l2" 或 "huber"
) -> tuple[torch.Tensor, dict]:
    """计算价值函数损失"""

    if loss_type == "l2":
        # L2 损失
        vf_loss = (values - returns) ** 2
    elif loss_type == "huber":
        # Huber 损失（更鲁棒）
        vf_loss = F.huber_loss(values, returns, reduction="none")

    # 可选：价值裁剪
    if cliprange_value is not None:
        values_clipped = torch.clamp(
            values,
            old_values - cliprange_value,
            old_values + cliprange_value
        )
        vf_loss_clipped = (values_clipped - returns) ** 2
        vf_loss = torch.maximum(vf_loss, vf_loss_clipped)

    return verl_F.masked_mean(vf_loss, response_mask)
```

---

## 6. KL 散度控制

**文件位置**：`verl/trainer/ppo/core_algos.py`，第 152-211 行

### 6.1 KL 惩罚类型

verl 支持多种 KL 惩罚计算方式：

```python
# 在 algorithm.py 中定义
kl_penalty_types = ["kl", "abs", "mse", "low_var_kl", "full"]
```

### 6.2 固定 KL 控制器

```python
class FixedKLController:
    """固定 KL 系数"""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        # 不更新，保持固定
        pass
```

### 6.3 自适应 KL 控制器

```python
class AdaptiveKLController:
    """自适应 KL 控制器 (https://arxiv.org/pdf/1909.08593.pdf)"""

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """根据当前 KL 散度调整系数"""
        target = self.target
        # 计算比例误差
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        # 更新系数
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
```

### 6.4 使用示例

```yaml
# 配置文件
algorithm:
  kl_ctrl:
    type: adaptive  # 或 "fixed"
    kl_coef: 0.02
    target_kl: 0.01
    horizon: 10000
```

---

## 7. 熵正则化

### 7.1 熵损失计算

```python
def compute_entropy_loss(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """计算熵损失（负熵，最小化时增加探索）"""
    # 熵 = -Σ p(x) * log p(x)
    # 这里使用 log_prob 近似: entropy ≈ -log_prob
    entropy = -log_probs
    return verl_F.masked_mean(entropy, response_mask)
```

### 7.2 总损失组合

```python
# PPO 总损失
total_loss = (
    policy_loss                           # 策略损失
    + vf_coef * value_loss                # 价值损失
    - entropy_coef * entropy_loss         # 熵奖励（负号因为要最大化熵）
    + kl_coef * kl_penalty                # KL 惩罚
)
```

---

## 8. 完整 PPO 训练流程

### 8.1 数据流

```
1. Rollout Phase:
   prompts → Actor (vLLM/SGLang) → responses

2. Compute Phase:
   (prompts, responses) → Reference Model → ref_log_probs
   (prompts, responses) → Actor Model → log_probs
   (prompts, responses) → Critic Model → values
   (prompts, responses) → Reward Model → rewards

3. Advantage Computation:
   (rewards, values) → GAE → advantages, returns

4. Update Phase:
   For each mini-batch:
       (log_probs, old_log_probs, advantages) → PPO Loss → Actor Update
       (values, returns) → Value Loss → Critic Update
```

### 8.2 配置示例

```yaml
algorithm:
  gamma: 1.0           # 折扣因子
  lam: 0.95            # GAE lambda
  adv_estimator: gae   # 优势估计器

actor:
  clip_ratio: 0.2      # PPO 裁剪范围
  clip_ratio_low: null # 非对称裁剪下界
  clip_ratio_high: null # 非对称裁剪上界
  entropy_coef: 0.01   # 熵系数

critic:
  value_loss_coef: 0.5     # 价值损失系数
  cliprange_value: 0.2     # 价值裁剪范围
```

---

## 9. 关键代码路径

| 功能 | 文件 | 函数/类 |
|-----|------|--------|
| GAE 优势估计 | core_algos.py:214 | `compute_gae_advantage_return()` |
| PPO 策略损失 | core_algos.py:1159 | `compute_policy_loss_vanilla()` |
| 价值损失 | core_algos.py:1799 | `compute_value_loss()` |
| KL 控制器 | core_algos.py:152 | `AdaptiveKLController` |
| 损失聚合 | core_algos.py | `agg_loss()` |

---

## 10. 下一步

- 了解 GRPO 等无需 Critic 的算法：[04_grpo_deep_dive.md](04_grpo_deep_dive.md)
- 了解 PPO Trainer 的完整实现：[05_ppo_trainer.md](05_ppo_trainer.md)
