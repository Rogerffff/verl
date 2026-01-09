# PPO Trainer 深度解析

## 1. 概述

PPO Trainer 是 verl 中的核心训练编排器，负责协调分布式 PPO/GRPO 训练的完整流程。

**文件位置**：`verl/trainer/ppo/ray_trainer.py`

---

## 2. RayPPOTrainer 类结构

### 2.1 初始化

```python
class RayPPOTrainer:
    """使用 Ray 的分布式 PPO 训练器"""

    def __init__(
        self,
        config,                           # 训练配置
        tokenizer,                        # 分词器
        role_worker_mapping: dict[Role, WorkerType],  # 角色到 Worker 的映射
        resource_pool_manager: ResourcePoolManager,   # 资源池管理器
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,                   # 多模态处理器
        reward_fn=None,                   # 训练奖励函数
        val_reward_fn=None,               # 验证奖励函数
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        # ...
```

### 2.2 核心属性

```python
# Worker 组
self.actor_rollout_wg: RayWorkerGroup  # Actor + Rollout Worker
self.critic_wg: RayWorkerGroup         # Critic Worker
self.ref_policy_wg: RayWorkerGroup     # Reference Policy Worker
self.reward_model_wg: RayWorkerGroup   # Reward Model Worker

# 数据加载
self.train_dataloader: StatefulDataLoader
self.val_dataloader: StatefulDataLoader

# KL 控制器
self.kl_ctrl: AdaptiveKLController | FixedKLController
```

---

## 3. ResourcePoolManager

资源池管理器负责 GPU 资源的分配和管理。

```python
@dataclass
class ResourcePoolManager:
    """定义资源池规格"""

    resource_pool_spec: dict[str, list[int]]  # 资源池规格
    mapping: dict[Role, str]                   # 角色到资源池的映射
    resource_pool_dict: dict[str, RayResourcePool]  # 资源池字典

    def create_resource_pool(self):
        """创建 Ray 资源池"""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count=3: actor_critic_ref, rollout, reward_model
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=3,
                name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """获取指定角色的资源池"""
        return self.resource_pool_dict[self.mapping[role]]
```

---

## 4. 训练循环

### 4.1 主训练循环

```python
def fit(self):
    """执行完整训练流程"""

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            # 1. 准备数据
            batch = DataProto.from_single_dict(batch_dict)

            # 2. 执行训练步骤
            metrics = self.training_step(batch)

            # 3. 记录指标
            self.log_metrics(metrics)

            # 4. 保存检查点
            if self.should_save_checkpoint():
                self.save_checkpoint()

            # 5. 执行验证
            if self.should_validate():
                self.validation_step()
```

### 4.2 训练步骤详解

```python
def training_step(self, batch: DataProto) -> dict:
    """单个训练步骤"""

    # ========== 阶段 1: Rollout 生成 ==========
    with marked_timer("rollout"):
        # 同步权重到推理引擎
        self.actor_rollout_wg.sync_weights()

        # 生成响应序列
        batch = self.actor_rollout_wg.generate_sequences(batch)

    # ========== 阶段 2: 计算奖励 ==========
    with marked_timer("reward"):
        # 使用奖励函数或奖励模型
        batch = compute_reward(batch, self.reward_fn, self.reward_model_wg)

    # ========== 阶段 3: 计算 Reference Log Probs ==========
    with marked_timer("ref_log_prob"):
        if self.need_ref_policy:
            batch = self.ref_policy_wg.compute_log_prob(batch)

    # ========== 阶段 4: 计算 Values (PPO only) ==========
    with marked_timer("values"):
        if self.need_critic:
            batch = self.critic_wg.compute_values(batch)

    # ========== 阶段 5: 计算优势 ==========
    with marked_timer("advantage"):
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
        )

    # ========== 阶段 6: 更新策略 ==========
    with marked_timer("update_policy"):
        actor_metrics = self.actor_rollout_wg.update_policy(batch)

    # ========== 阶段 7: 更新价值函数 (PPO only) ==========
    with marked_timer("update_critic"):
        if self.need_critic:
            critic_metrics = self.critic_wg.update_critic(batch)

    return {**actor_metrics, **critic_metrics}
```

---

## 5. KL 惩罚应用

```python
def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty="kl"):
    """将 KL 惩罚应用到 token 级别奖励"""

    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # 计算 ref_policy 和 current_policy 之间的 KL 散度
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"],
        data.batch["ref_log_prob"],
        kl_penalty=kl_penalty
    )
    kld = kld * response_mask
    beta = kl_ctrl.value

    # 奖励 = 原始奖励 - β * KL
    token_level_rewards = token_level_scores - beta * kld

    # 计算当前 KL 并更新控制器
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)

    data.batch["token_level_rewards"] = token_level_rewards

    return data, {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}
```

---

## 6. 优势计算

```python
def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """计算优势估计"""

    response_mask = data.batch["response_mask"]
    token_level_rewards = data.batch["token_level_rewards"]
    index = data.non_tensor_batch.get("uid", None)

    # 获取优势估计函数
    adv_fn = core_algos.get_adv_estimator_fn(adv_estimator)

    # 根据算法类型调用不同的优势估计器
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = adv_fn(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
    elif adv_estimator in [AdvantageEstimator.GRPO, AdvantageEstimator.RLOO]:
        advantages, returns = adv_fn(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
    # ... 其他算法

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

    return data
```

---

## 7. 梯度累积和 Mini-Batch

### 7.1 Mini-Batch 迭代器

```python
def make_minibatch_iterator(data: DataProto, mini_batch_size: int):
    """创建 mini-batch 迭代器"""
    total_size = len(data)
    indices = torch.randperm(total_size)

    for start in range(0, total_size, mini_batch_size):
        end = min(start + mini_batch_size, total_size)
        batch_indices = indices[start:end]
        yield data.select_idxs(batch_indices)
```

### 7.2 梯度累积

```python
def update_policy_with_grad_accum(self, data: DataProto):
    """带梯度累积的策略更新"""

    optimizer.zero_grad()

    for micro_batch in make_minibatch_iterator(data, micro_batch_size):
        # 前向传播
        loss = compute_policy_loss(micro_batch)

        # 缩放损失（梯度累积）
        scaled_loss = loss / num_accumulation_steps

        # 反向传播
        scaled_loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # 更新参数
    optimizer.step()
    scheduler.step()
```

---

## 8. 检查点管理

### 8.1 保存检查点

```python
def save_checkpoint(self):
    """保存训练检查点"""

    checkpoint = {
        "global_step": self.global_step,
        "epoch": self.epoch,
        "optimizer_state": self.optimizer.state_dict(),
        "scheduler_state": self.scheduler.state_dict(),
        "kl_ctrl_value": self.kl_ctrl.value,
    }

    # 保存 Actor 权重
    self.actor_rollout_wg.save_checkpoint(checkpoint_path)

    # 保存 Critic 权重
    if self.need_critic:
        self.critic_wg.save_checkpoint(checkpoint_path)
```

### 8.2 恢复检查点

```python
def load_checkpoint(self, checkpoint_path: str):
    """加载训练检查点"""

    checkpoint = torch.load(checkpoint_path)

    self.global_step = checkpoint["global_step"]
    self.epoch = checkpoint["epoch"]
    self.optimizer.load_state_dict(checkpoint["optimizer_state"])
    self.scheduler.load_state_dict(checkpoint["scheduler_state"])
    self.kl_ctrl.value = checkpoint["kl_ctrl_value"]

    # 加载模型权重
    self.actor_rollout_wg.load_checkpoint(checkpoint_path)
    if self.need_critic:
        self.critic_wg.load_checkpoint(checkpoint_path)
```

---

## 9. 指标收集

### 9.1 数据指标

```python
def compute_data_metrics(batch: DataProto) -> dict:
    """计算数据相关指标"""
    return {
        "data/response_length": batch.batch["responses"].shape[1],
        "data/prompt_length": batch.batch["prompts"].shape[1],
        "data/reward_mean": batch.batch["rewards"].mean().item(),
        "data/reward_std": batch.batch["rewards"].std().item(),
        "data/advantage_mean": batch.batch["advantages"].mean().item(),
    }
```

### 9.2 吞吐量指标

```python
def compute_throughout_metrics(timing_info: dict, batch_size: int) -> dict:
    """计算吞吐量指标"""
    return {
        "throughput/samples_per_second": batch_size / timing_info["total_time"],
        "throughput/tokens_per_second": total_tokens / timing_info["total_time"],
        "timing/rollout_time": timing_info["rollout_time"],
        "timing/update_time": timing_info["update_time"],
    }
```

---

## 10. 验证流程

```python
def validation_step(self):
    """执行验证"""

    val_metrics = defaultdict(list)

    for batch_dict in self.val_dataloader:
        batch = DataProto.from_single_dict(batch_dict)

        # 生成响应
        batch = self.actor_rollout_wg.generate_sequences(batch)

        # 计算奖励
        batch = compute_reward(batch, self.val_reward_fn)

        # 收集指标
        val_metrics["val/reward"].append(batch.batch["rewards"].mean().item())
        val_metrics["val/response_length"].append(
            batch.batch["responses"].shape[1]
        )

    # 聚合指标
    final_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

    return final_metrics
```

---

## 11. 工作流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RayPPOTrainer                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  for batch in train_dataloader:                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 1. Rollout: generate_sequences(batch)                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 2. Reward: compute_reward(batch)                              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 3. Ref LogProb: ref_policy.compute_log_prob(batch)            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 4. Values: critic.compute_values(batch)  [PPO only]           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 5. Advantage: compute_advantage(batch)                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 6. Update: actor.update_policy(batch)                         │  │
│  │           critic.update_critic(batch)  [PPO only]             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 7. Log & Checkpoint                                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. 配置示例

```yaml
trainer:
  total_epochs: 1
  total_training_steps: 1000
  save_freq: 100
  val_freq: 50
  logging_freq: 10

  # 梯度累积
  gradient_accumulation_steps: 4
  micro_batch_size: 8

  # 检查点
  default_local_dir: "./checkpoints"
  resume_mode: "auto"  # auto, must, disable
```

---

## 13. 关键代码路径

| 功能 | 文件 | 类/函数 |
|-----|------|--------|
| 主训练器 | ray_trainer.py | `RayPPOTrainer` |
| 资源管理 | ray_trainer.py | `ResourcePoolManager` |
| KL 惩罚 | ray_trainer.py | `apply_kl_penalty()` |
| 优势计算 | ray_trainer.py | `compute_advantage()` |
| 指标收集 | metric_utils.py | `compute_*_metrics()` |

---

## 14. 下一步

- 了解 Worker 实现细节：[06_workers.md](06_workers.md)
- 了解分布式训练：[07_distributed_training.md](07_distributed_training.md)
