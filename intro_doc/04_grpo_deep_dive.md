# GRPO 算法深度解析

## 1. GRPO 算法原理

GRPO (Group Relative Policy Optimization) 是一种无需 Critic 模型的策略优化算法，通过组内相对比较来估计优势。

### 1.1 与 PPO 的关键差异

| 特性 | PPO | GRPO |
|-----|-----|------|
| Critic 模型 | 需要 | **不需要** |
| 优势估计 | GAE (基于 Critic) | 组内相对奖励 |
| 内存占用 | 较高（需存储 Critic） | 较低 |
| 实现复杂度 | 较高 | 较低 |
| 适用场景 | 密集奖励 | 稀疏/结果奖励 |

### 1.2 GRPO 核心思想

对于同一个 prompt 生成的多个响应，GRPO 使用组内归一化来计算优势：

```
A_i = (r_i - μ_g) / σ_g
```

其中：
- `r_i`：第 i 个响应的奖励
- `μ_g`：组内奖励均值
- `σ_g`：组内奖励标准差

---

## 2. GRPO 优势估计实现

**文件位置**：`verl/trainer/ppo/core_algos.py`，第 266-330 行

### 2.1 标准 GRPO 实现

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length)
    response_mask: torch.Tensor,         # (bs, response_length)
    index: np.ndarray,                   # 组索引
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算 GRPO 优势（仅基于结果奖励）

    Args:
        token_level_rewards: token 级别奖励（通常只有最后一个 token 有奖励）
        response_mask: 响应 mask
        index: 每个样本的组 ID（同一 prompt 的样本共享相同 ID）
        norm_adv_by_std_in_grpo: 是否除以标准差
            - True: 标准 GRPO
            - False: Dr.GRPO (https://arxiv.org/abs/2503.20783)
    """
    # 1. 计算每个样本的总奖励（结果奖励）
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # 2. 按组收集奖励
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        # 3. 计算每组的均值和标准差
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # 单样本组：无法归一化
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)

        # 4. 计算归一化优势
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                # 标准 GRPO: (r - μ) / σ
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                # Dr.GRPO: r - μ (不除以标准差)
                scores[i] = scores[i] - id2mean[index[i]]

        # 5. 广播到 token 级别
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
```

### 2.2 向量化 GRPO 实现

更高效的向量化版本：

```python
@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """向量化 GRPO 实现"""
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)

        # 转换索引为 torch tensor
        g = as_torch_index(index, device=scores.device)

        # 使用 group_mean_std 向量化计算
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)

        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]

        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages
```

---

## 3. GRPO 变体算法

### 3.1 GRPO Pass@K

**文件位置**：`verl/trainer/ppo/core_algos.py`，第 360-419 行

Pass@K 变体只奖励组内最佳响应：

```python
@register_adv_est(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pass@K 优势估计

    仅最佳响应获得非零优势：advantage = r_max - r_second_max
    基于论文：https://arxiv.org/abs/2503.19595
    """
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k 需要每组至少 2 个样本，组 {idx} 只有 {rewards.numel()} 个"
                )

            # 找到最大和次大奖励
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]

            # 最佳响应的索引
            i_max = id2indices[idx][topk_idx[0].item()]

            # 只有最佳响应获得优势
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)

            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages
```

### 3.2 Dr.GRPO (不除以标准差)

Dr.GRPO 是 GRPO 的简化版本，不除以组内标准差：

```python
# 在 GRPO 中设置 norm_adv_by_std_in_grpo = False
# 优势计算: A = r - μ (不是 (r - μ) / σ)
```

**优点**：
- 更稳定的梯度
- 避免标准差接近 0 时的数值问题

---

## 4. 相关算法：RLOO

**文件位置**：`verl/trainer/ppo/core_algos.py`，第 476-525 行

RLOO (Leave-One-Out) 使用不同的基线估计：

```python
@register_adv_est(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RLOO 优势估计 (https://arxiv.org/abs/2402.14740)

    使用 leave-one-out 均值作为基线
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            else:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))

        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                # RLOO 公式：修正偏差
                # A_i = r_i * n/(n-1) - μ * n/(n-1)
                scores[i] = (
                    scores[i] * response_num / (response_num - 1) -
                    id2mean[index[i]] * response_num / (response_num - 1)
                )

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
```

---

## 5. 相关算法：REINFORCE++

**文件位置**：`verl/trainer/ppo/core_algos.py`，第 582-618 行

```python
@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    REINFORCE++ 优势估计 (https://arxiv.org/abs/2501.03262)

    使用累积回报并进行白化
    """
    assert config is not None
    gamma = config.gamma

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        # 反向计算累积回报
        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # 在 EOS 后重置
            running_return = running_return * response_mask[:, t]

        # 白化回报作为优势
        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns
```

---

## 6. 相关算法：ReMax

**文件位置**：`verl/trainer/ppo/core_algos.py`，第 621-654 行

```python
@register_adv_est(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,     # 基线奖励 (bs,)
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ReMax 优势估计 (https://arxiv.org/abs/2310.10505)

    使用贪心解码的奖励作为基线
    """
    with torch.no_grad():
        # 计算累积回报（从当前到结束）
        returns = (token_level_rewards * response_mask)\
                  .flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # 优势 = 回报 - 基线
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns
```

---

## 7. DAPO 算法

DAPO (Decoupled Alignment Policy Optimization) 是 GRPO 的增强版本，由 verl 团队提出。

### 7.1 DAPO 关键特性

1. **动态采样**：根据奖励多样性过滤组
2. **非对称裁剪**：不同的上下裁剪范围
3. **超长惩罚**：对超长响应施加惩罚

### 7.2 DAPO 奖励管理器

**文件位置**：`verl/workers/reward_manager/dapo.py`

```python
class DAPORewardManager(NaiveRewardManager):
    """DAPO 奖励管理器"""

    def _compute_score(self, response_str, **kwargs) -> float:
        """计算奖励，包含超长惩罚"""
        score = self.compute_score_fn(response_str, **kwargs)

        # 超长惩罚
        if hasattr(self, 'overlong_buffer_len') and self.overlong_buffer_len > 0:
            exceed_len = len(response_tokens) - max_response_length
            if exceed_len > 0:
                # 线性惩罚
                penalty = min(-exceed_len / self.overlong_buffer_len * penalty_factor, 0)
                score += penalty

        return score
```

### 7.3 动态组过滤

```python
class FilterGroupsConfig:
    """DAPO 动态组过滤配置"""
    enable: bool = False
    metric_to_filter: str = "score"  # 用于过滤的指标
    max_num_gen_batches: int = 4     # 最大生成批次数
```

---

## 8. Token 级别归一化 (DrGRPO)

DrGRPO 使用 token 级别的损失聚合：

```python
# 使用 seq-mean-token-sum-norm 模式
loss_agg_mode = "seq-mean-token-sum-norm"

# 该模式将每个序列的损失除以序列长度，
# 使短序列和长序列获得相同的梯度权重
```

**优点**：
- 避免长序列主导梯度
- 更公平的样本权重

---

## 9. GRPO 配置示例

```yaml
algorithm:
  adv_estimator: grpo           # 使用 GRPO 优势估计
  norm_adv_by_std_in_grpo: true # 标准 GRPO（除以标准差）
  gamma: 1.0                    # 折扣因子

actor:
  clip_ratio: 0.2
  loss_agg_mode: token-mean     # 或 seq-mean-token-sum-norm for DrGRPO
  policy_loss_fn: vanilla       # PPO 风格的裁剪损失

# GRPO 不需要 Critic 配置
# critic:
#   ...
```

---

## 10. 算法对比总结

| 算法 | 基线 | 归一化 | 特点 |
|-----|------|--------|------|
| GRPO | 组均值 | 除以标准差 | 标准实现 |
| GRPO_VECTORIZED | 组均值 | 除以标准差 | 高效向量化 |
| Dr.GRPO | 组均值 | 不除以标准差 | 更稳定 |
| GRPO_PASSK | 次优奖励 | 可选 | 只奖励最佳 |
| RLOO | Leave-one-out | 无 | 无偏估计 |
| REINFORCE++ | 白化 | 白化 | 累积回报 |
| ReMax | 贪心解码 | 无 | 使用贪心基线 |

---

## 11. 代码路径总结

| 算法 | 函数 | 行号 |
|-----|------|------|
| GRPO | `compute_grpo_outcome_advantage` | 266 |
| GRPO_VECTORIZED | `compute_grpo_vectorized_outcome_advantage` | 333 |
| GRPO_PASSK | `compute_grpo_passk_outcome_advantage` | 360 |
| RLOO | `compute_rloo_outcome_advantage` | 476 |
| REINFORCE++ | `compute_reinforce_plus_plus_outcome_advantage` | 582 |
| ReMax | `compute_remax_outcome_advantage` | 621 |

---

## 12. 下一步

- 了解 PPO Trainer 完整实现：[05_ppo_trainer.md](05_ppo_trainer.md)
- 了解 Worker 实现细节：[06_workers.md](06_workers.md)
