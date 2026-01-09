# Worker 实现详解

## 1. Worker 体系结构

verl 使用 Worker 模式来分离不同的计算任务，每个 Worker 负责特定的功能。

```
┌─────────────────────────────────────────────────────────────────┐
│                         Worker 层次结构                          │
└─────────────────────────────────────────────────────────────────┘

                        BasePPOActor
                             │
              ┌──────────────┼──────────────┐
              │                             │
    DataParallelPPOActor            MegatronPPOActor
    (FSDP 后端)                      (Megatron 后端)


                        BasePPOCritic
                             │
              ┌──────────────┼──────────────┐
              │                             │
    DataParallelPPOCritic           MegatronPPOCritic
    (FSDP 后端)                      (Megatron 后端)


                         BaseRollout
                             │
              ┌──────────────┼──────────────┐
              │                             │
        vLLMAsyncRollout              ServerAdapter
        (vLLM 后端)                   (SGLang 后端)
```

---

## 2. Actor Worker

### 2.1 基类 (BasePPOActor)

**文件位置**：`verl/workers/actor/base.py`

```python
class BasePPOActor(ABC):
    """PPO Actor 基类"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """计算给定 token 的对数概率

        Args:
            data: 包含 input_ids, attention_mask, position_ids

        Returns:
            DataProto: 包含 log_probs
        """
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> dict:
        """更新策略模型

        Returns:
            Dict: 包含 loss, grad_norm 等统计信息
        """
        pass
```

### 2.2 DataParallelPPOActor (FSDP 实现)

**文件位置**：`verl/workers/actor/dp_actor.py`

```python
class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor"""

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None
    ):
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        # 配置选项
        self.use_remove_padding = config.get("use_remove_padding", False)
        self.use_fused_kernels = config.get("use_fused_kernels", False)
        self.ulysses_sequence_parallel_size = config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
```

### 2.3 前向传播 (Micro-Batch)

```python
def _forward_micro_batch(
    self,
    micro_batch: dict[str, torch.Tensor],
    temperature: float,
    calculate_entropy: bool = False
) -> dict[str, torch.Tensor]:
    """单个 micro-batch 的前向传播

    Returns:
        log_probs: (bs, response_len)
        entropys: (bs, response_len) [可选]
    """
    response_length = micro_batch["responses"].size(-1)

    # 1. 准备输入
    input_ids = micro_batch["input_ids"]
    attention_mask = micro_batch["attention_mask"]
    position_ids = micro_batch["position_ids"]

    # 2. 模型前向传播
    with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = output.logits

    # 3. 提取响应部分的 logits
    response_logits = logits[:, -response_length-1:-1, :]

    # 4. 计算对数概率
    response_ids = micro_batch["responses"]
    log_probs = logprobs_from_logits(
        logits=response_logits / temperature,
        labels=response_ids
    )

    # 5. 可选：计算熵
    if calculate_entropy:
        entropys = self.compute_entropy_from_logits(response_logits)
        return {"log_probs": log_probs, "entropys": entropys}

    return {"log_probs": log_probs}
```

### 2.4 compute_log_prob 实现

```python
def compute_log_prob(self, data: DataProto) -> DataProto:
    """计算对数概率"""

    self.actor_module.eval()

    all_log_probs = []
    all_entropys = []

    # 遍历 micro-batches
    for micro_batch in data.chunk(micro_batch_size):
        with torch.no_grad():
            output = self._forward_micro_batch(
                micro_batch.batch,
                temperature=self.config.temperature,
                calculate_entropy=True
            )

        all_log_probs.append(output["log_probs"])
        all_entropys.append(output["entropys"])

    # 合并结果
    data.batch["old_log_probs"] = torch.cat(all_log_probs, dim=0)
    data.batch["entropys"] = torch.cat(all_entropys, dim=0)

    return data
```

### 2.5 update_policy 实现

```python
def update_policy(self, data: DataProto) -> dict:
    """更新策略"""

    self.actor_module.train()
    metrics = defaultdict(list)

    # PPO epochs
    for epoch in range(self.config.ppo_epochs):
        for micro_batch in data.shuffle().chunk(self.config.ppo_mini_batch_size):
            # 1. 前向传播
            output = self._forward_micro_batch(
                micro_batch.batch,
                temperature=self.config.temperature,
                calculate_entropy=True
            )

            log_probs = output["log_probs"]
            old_log_probs = micro_batch.batch["old_log_probs"]
            advantages = micro_batch.batch["advantages"]
            response_mask = micro_batch.batch["response_mask"]

            # 2. 计算策略损失
            loss_fn = get_policy_loss_fn(self.config.policy_loss_fn)
            pg_loss, pg_metrics = loss_fn(
                old_log_prob=old_log_probs,
                log_prob=log_probs,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode=self.config.loss_agg_mode,
                config=self.config,
            )

            # 3. 计算熵损失
            entropy_loss = verl_F.masked_mean(output["entropys"], response_mask)

            # 4. 总损失
            total_loss = pg_loss - self.config.entropy_coef * entropy_loss

            # 5. 反向传播
            self.actor_optimizer.zero_grad()
            total_loss.backward()

            # 6. 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(),
                self.config.max_grad_norm
            )

            # 7. 优化器步骤
            self.actor_optimizer.step()

            # 收集指标
            append_to_dict(metrics, pg_metrics)
            metrics["actor/entropy_loss"].append(entropy_loss.item())
            metrics["actor/grad_norm"].append(grad_norm.item())

    return {k: np.mean(v) for k, v in metrics.items()}
```

---

## 3. Critic Worker

### 3.1 基类 (BasePPOCritic)

**文件位置**：`verl/workers/critic/base.py`

```python
class BasePPOCritic(ABC):
    """PPO Critic 基类"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_values(self, data: DataProto) -> torch.Tensor:
        """计算状态价值"""
        pass

    @abstractmethod
    def update_critic(self, data: DataProto):
        """更新价值函数"""
        pass
```

### 3.2 DataParallelPPOCritic (FSDP 实现)

**文件位置**：`verl/workers/critic/dp_critic.py`

```python
class DataParallelPPOCritic(BasePPOCritic):
    """FSDP DataParallel PPO Critic"""

    def __init__(
        self,
        config,
        critic_module: nn.Module,
        critic_optimizer: torch.optim.Optimizer
    ):
        super().__init__(config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer

    def compute_values(self, data: DataProto) -> DataProto:
        """计算状态价值"""

        self.critic_module.eval()
        all_values = []

        for micro_batch in data.chunk(micro_batch_size):
            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch.batch)
            all_values.append(values)

        data.batch["values"] = torch.cat(all_values, dim=0)
        return data

    def update_critic(self, data: DataProto) -> dict:
        """更新价值函数"""

        self.critic_module.train()
        metrics = defaultdict(list)

        for epoch in range(self.config.ppo_epochs):
            for micro_batch in data.shuffle().chunk(mini_batch_size):
                # 1. 前向传播
                values = self._forward_micro_batch(micro_batch.batch)

                # 2. 计算价值损失
                returns = micro_batch.batch["returns"]
                old_values = micro_batch.batch["values"]
                response_mask = micro_batch.batch["response_mask"]

                vf_loss = compute_value_loss(
                    values=values,
                    old_values=old_values,
                    returns=returns,
                    response_mask=response_mask,
                    cliprange_value=self.config.cliprange_value,
                )

                # 3. 反向传播和优化
                self.critic_optimizer.zero_grad()
                vf_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic_module.parameters(),
                    self.config.max_grad_norm
                )
                self.critic_optimizer.step()

                metrics["critic/vf_loss"].append(vf_loss.item())
                metrics["critic/grad_norm"].append(grad_norm.item())

        return {k: np.mean(v) for k, v in metrics.items()}
```

---

## 4. Rollout Worker

### 4.1 基类 (BaseRollout)

**文件位置**：`verl/workers/rollout/base.py`

```python
class BaseRollout(ABC):
    """Rollout 基类"""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        self.config = omega_conf_to_dataclass(config)
        self.model_config = omega_conf_to_dataclass(model_config)
        self.device_mesh = device_mesh

    @abstractmethod
    async def resume(self, tags: list[str]):
        """恢复 rollout 权重或 KV cache"""
        pass

    @abstractmethod
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """更新 rollout 模型的权重"""
        pass

    @abstractmethod
    async def release(self):
        """释放 GPU 内存"""
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """批量生成序列"""
        raise NotImplementedError
```

### 4.2 vLLM Rollout 实现

**文件位置**：`verl/workers/rollout/vllm_rollout/vllm_rollout.py`

```python
class vLLMAsyncRollout(BaseRollout):
    """vLLM 异步 Rollout"""

    def __init__(self, config, model_config, device_mesh):
        super().__init__(config, model_config, device_mesh)

        # 初始化 vLLM 引擎
        self.engine = AsyncLLMEngine(...)

    async def update_weights(self, weights, **kwargs):
        """更新 vLLM 模型权重"""
        # 从训练引擎获取权重并广播到 vLLM
        for name, param in weights:
            # 将 FSDP 分片权重转换为 vLLM 格式
            await self.engine.update_weight(name, param)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """生成序列"""
        # 准备采样参数
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
        )

        # 批量生成
        outputs = self.engine.generate(prompts, sampling_params)

        # 处理输出
        responses = [output.outputs[0].text for output in outputs]
        return DataProto.from_dict({"responses": responses})
```

### 4.3 SGLang Rollout 实现

**文件位置**：`verl/workers/rollout/sglang_rollout/sglang_rollout.py`

```python
class ServerAdapter(BaseRollout):
    """SGLang Server 适配器"""

    def __init__(self, config, model_config, device_mesh):
        super().__init__(config, model_config, device_mesh)

        # 连接到 SGLang 服务器
        self.client = sglang.Client(...)

    async def update_weights(self, weights, **kwargs):
        """通过 HTTP 更新 SGLang 服务器权重"""
        for name, param in weights:
            await self.client.update_weight(name, param)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """通过 SGLang 服务器生成序列"""
        responses = self.client.generate(
            prompts=prompts,
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )
        return DataProto.from_dict({"responses": responses})
```

---

## 5. Reward Manager

### 5.1 NaiveRewardManager

**文件位置**：`verl/workers/reward_manager/naive.py`

```python
class NaiveRewardManager:
    """基础奖励管理器"""

    def __init__(self, tokenizer, compute_score_fn):
        self.tokenizer = tokenizer
        self.compute_score_fn = compute_score_fn

    def compute_reward(self, data: DataProto) -> DataProto:
        """计算奖励"""
        rewards = []

        for i in range(len(data)):
            response_str = self.tokenizer.decode(data.batch["responses"][i])
            ground_truth = data.non_tensor_batch["ground_truth"][i]

            # 调用评分函数
            score = self.compute_score_fn(response_str, ground_truth=ground_truth)
            rewards.append(score)

        # 转换为 token 级别奖励（只在最后一个 token 给奖励）
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float)
        response_lengths = data.batch["response_mask"].sum(dim=-1)
        for i, (reward, length) in enumerate(zip(rewards, response_lengths)):
            token_level_rewards[i, length - 1] = reward

        data.batch["token_level_scores"] = token_level_rewards
        return data
```

### 5.2 DAPORewardManager

**文件位置**：`verl/workers/reward_manager/dapo.py`

```python
class DAPORewardManager(NaiveRewardManager):
    """DAPO 奖励管理器（带超长惩罚）"""

    def __init__(self, tokenizer, compute_score_fn, overlong_buffer_len=512, penalty_factor=1.0):
        super().__init__(tokenizer, compute_score_fn)
        self.overlong_buffer_len = overlong_buffer_len
        self.penalty_factor = penalty_factor

    def _compute_score(self, response_str, **kwargs) -> float:
        score = self.compute_score_fn(response_str, **kwargs)

        # 超长惩罚
        response_length = len(response_str)
        max_length = kwargs.get("max_response_length", float("inf"))

        if response_length > max_length:
            exceed_len = response_length - max_length
            penalty = min(-exceed_len / self.overlong_buffer_len * self.penalty_factor, 0)
            score += penalty

        return score
```

---

## 6. FSDP Workers 组合类

**文件位置**：`verl/workers/fsdp_workers.py`

组合多个 Worker 到单个进程：

```python
class ActorRolloutRefWorker:
    """组合 Actor + Rollout + Reference 的 Worker"""

    def __init__(self, config):
        # 初始化模型
        self.model = create_model(config)

        # 初始化 Actor
        self.actor = DataParallelPPOActor(
            config=config.actor,
            actor_module=self.model,
            actor_optimizer=create_optimizer(self.model, config)
        )

        # 初始化 Reference (无优化器)
        self.ref = DataParallelPPOActor(
            config=config.ref,
            actor_module=self.model,
            actor_optimizer=None
        )

        # 初始化 Rollout
        self.rollout = get_rollout_class(config.rollout.name)(
            config=config.rollout,
            model_config=config.model,
        )
```

---

## 7. 权重同步机制

### 7.1 FSDP 到 vLLM 权重同步

```python
def sync_weights_to_rollout(actor, rollout):
    """从 FSDP Actor 同步权重到 vLLM Rollout"""

    # 1. 收集 FSDP 分片权重
    def get_weights_generator():
        for name, param in actor.actor_module.named_parameters():
            # 从所有 ranks 收集完整参数
            full_param = param.full_tensor()
            yield name, full_param

    # 2. 更新到 Rollout
    await rollout.update_weights(get_weights_generator())
```

### 7.2 DTensor 权重加载器

```python
class DTensorWeightLoader:
    """DTensor 权重加载器（用于 FSDP2）"""

    def load_weights(self, weights):
        for name, param in weights:
            # 处理 DTensor 格式
            if isinstance(param, DTensor):
                local_param = param.to_local()
            else:
                local_param = param

            # 更新模型权重
            self.model.update_weight(name, local_param)
```

---

## 8. 配置类

### 8.1 ActorConfig

```python
@dataclass
class ActorConfig:
    # 模型配置
    temperature: float = 1.0

    # PPO 配置
    clip_ratio: float = 0.2
    clip_ratio_low: Optional[float] = None
    clip_ratio_high: Optional[float] = None
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # 训练配置
    ppo_epochs: int = 1
    ppo_mini_batch_size: int = 8

    # 损失配置
    loss_agg_mode: str = "token-mean"
    policy_loss_fn: str = "vanilla"

    # 优化
    use_remove_padding: bool = False
    use_fused_kernels: bool = False
    ulysses_sequence_parallel_size: int = 1
```

### 8.2 CriticConfig

```python
@dataclass
class CriticConfig:
    # 价值损失配置
    cliprange_value: float = 0.2
    value_loss_coef: float = 0.5
    max_grad_norm: float = 1.0

    # 训练配置
    ppo_epochs: int = 1
    ppo_mini_batch_size: int = 8
```

---

## 9. 关键代码路径

| 组件 | 文件 | 类 |
|-----|------|---|
| Actor 基类 | `workers/actor/base.py` | `BasePPOActor` |
| FSDP Actor | `workers/actor/dp_actor.py` | `DataParallelPPOActor` |
| Megatron Actor | `workers/actor/megatron_actor.py` | `MegatronPPOActor` |
| Critic 基类 | `workers/critic/base.py` | `BasePPOCritic` |
| FSDP Critic | `workers/critic/dp_critic.py` | `DataParallelPPOCritic` |
| Rollout 基类 | `workers/rollout/base.py` | `BaseRollout` |
| vLLM Rollout | `workers/rollout/vllm_rollout/` | `vLLMAsyncRollout` |
| SGLang Rollout | `workers/rollout/sglang_rollout/` | `ServerAdapter` |
| Reward Manager | `workers/reward_manager/` | `NaiveRewardManager` |

---

## 10. 下一步

- 了解分布式训练架构：[07_distributed_training.md](07_distributed_training.md)
- 了解配置系统：[08_configuration.md](08_configuration.md)
