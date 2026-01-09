# 配置系统

## 1. 概述

verl 使用 YAML 配置文件和 Python dataclass 来管理训练参数。配置系统采用层次化设计，支持继承和覆盖。

---

## 2. 配置类层次结构

```
BaseConfig
    │
    ├── AlgoConfig          # 算法配置
    ├── KLControlConfig     # KL 控制配置
    ├── FilterGroupsConfig  # 组过滤配置 (DAPO)
    ├── RolloutCorrectionConfig  # Rollout 校正配置
    │
    ├── ActorConfig         # Actor 配置
    ├── CriticConfig        # Critic 配置
    ├── RolloutConfig       # Rollout 配置
    │
    ├── FSDPEngineConfig    # FSDP 引擎配置
    ├── MegatronEngineConfig # Megatron 引擎配置
    │
    └── TrainerConfig       # Trainer 配置
```

---

## 3. 算法配置 (AlgoConfig)

**文件位置**：`verl/trainer/config/algorithm.py`

### 3.1 核心参数

```python
@dataclass
class AlgoConfig(BaseConfig):
    """算法配置"""

    # 折扣因子
    gamma: float = 1.0

    # GAE lambda
    lam: float = 1.0

    # 优势估计器类型
    adv_estimator: str = "grpo"
    # 可选: "gae", "grpo", "grpo_vectorized", "rloo", "reinforce_plus_plus", "remax", "gpg"

    # 是否在奖励中使用 KL 惩罚
    use_kl_in_reward: bool = False

    # KL 惩罚类型
    kl_penalty: str = "kl"
    # 可选: "kl", "abs", "mse", "low_var_kl", "full"

    # GRPO 特定：是否除以标准差
    norm_adv_by_std_in_grpo: bool = True

    # KL 控制器配置
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)

    # 组过滤配置 (DAPO)
    filter_groups: FilterGroupsConfig = field(default_factory=FilterGroupsConfig)

    # Rollout 校正配置
    rollout_correction: Optional[RolloutCorrectionConfig] = None
```

### 3.2 KL 控制配置

```python
@dataclass
class KLControlConfig(BaseConfig):
    """KL 控制配置"""

    # 控制器类型: "fixed" 或 "adaptive"
    type: str = "fixed"

    # KL 系数
    kl_coef: float = 0.001

    # 自适应控制器参数
    horizon: int = 10000
    target_kl: float = 0.1
```

### 3.3 组过滤配置 (DAPO)

```python
@dataclass
class FilterGroupsConfig(BaseConfig):
    """组过滤配置"""

    # 是否启用
    enable: bool = False

    # 过滤指标
    metric: Optional[str] = None
    # 可选: "acc", "score", "seq_reward", "seq_final_reward"

    # 最大生成批次数
    max_num_gen_batches: int = 0
```

### 3.4 Rollout 校正配置

```python
@dataclass
class RolloutCorrectionConfig(BaseConfig):
    """Rollout 校正配置（处理离策略问题）"""

    # 重要性采样聚合级别
    rollout_is: Optional[str] = "sequence"
    # 可选: None, "token", "sequence"

    # IS 权重阈值
    rollout_is_threshold: float = 2.0

    # 拒绝采样级别
    rollout_rs: Optional[str] = None
    # 可选: None, "token", "sequence", "geometric"

    # 拒绝采样阈值
    rollout_rs_threshold: Optional[float] = None

    # Bypass 模式
    bypass_mode: bool = False

    # 损失类型（Bypass 模式下）
    loss_type: str = "ppo_clip"
    # 可选: "reinforce", "ppo_clip"

    # 预设工厂方法
    @classmethod
    def decoupled_token_is(cls):
        """Token-TIS 预设"""
        return cls(rollout_is="token", rollout_is_threshold=2.0)

    @classmethod
    def decoupled_seq_is(cls):
        """Seq-TIS 预设"""
        return cls(rollout_is="sequence", rollout_is_threshold=5.0)

    @classmethod
    def bypass_ppo_clip(cls):
        """Bypass PPO-Clip 预设"""
        return cls(bypass_mode=True, loss_type="ppo_clip")
```

---

## 4. Actor 配置

**文件位置**：`verl/workers/config/actor.py`

```python
@dataclass
class ActorConfig(BaseConfig):
    """Actor 配置"""

    # 采样温度
    temperature: float = 1.0

    # PPO 裁剪范围
    clip_ratio: float = 0.2
    clip_ratio_low: Optional[float] = None   # 非对称裁剪下界
    clip_ratio_high: Optional[float] = None  # 非对称裁剪上界

    # 双重裁剪 (Dual-Clip PPO)
    clip_ratio_c: float = 3.0

    # 熵系数
    entropy_coef: float = 0.01

    # 梯度裁剪
    max_grad_norm: float = 1.0

    # 训练配置
    ppo_epochs: int = 1
    ppo_mini_batch_size: int = 8
    gradient_accumulation_steps: int = 1

    # 损失聚合模式
    loss_agg_mode: str = "token-mean"
    # 可选: "token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"

    # 策略损失函数
    policy_loss_fn: str = "vanilla"
    # 可选: "vanilla", "gspo", "sapo", "gpg", "clip_cov", "kl_cov", "cispo"

    # 优化选项
    use_remove_padding: bool = False
    use_fused_kernels: bool = False
    use_torch_compile: bool = True

    # 序列并行
    ulysses_sequence_parallel_size: int = 1

    # FSDP 配置
    strategy: str = "fsdp2"
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
```

---

## 5. Critic 配置

```python
@dataclass
class CriticConfig(BaseConfig):
    """Critic 配置"""

    # 价值损失系数
    value_loss_coef: float = 0.5

    # 价值裁剪范围
    cliprange_value: float = 0.2

    # 梯度裁剪
    max_grad_norm: float = 1.0

    # 训练配置
    ppo_epochs: int = 1
    ppo_mini_batch_size: int = 8

    # 价值损失类型
    value_loss_type: str = "l2"
    # 可选: "l2", "huber"

    # 策略
    strategy: str = "fsdp2"
```

---

## 6. Rollout 配置

```python
@dataclass
class RolloutConfig(BaseConfig):
    """Rollout 配置"""

    # 推理引擎
    name: str = "vllm"
    # 可选: "vllm", "sglang"

    # 采样参数
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1

    # 生成长度
    max_new_tokens: int = 512
    min_new_tokens: int = 1

    # 张量并行
    tensor_parallel_size: int = 1

    # 每个 prompt 的响应数
    n: int = 1
```

---

## 7. FSDP 配置

```python
@dataclass
class FSDPConfig(BaseConfig):
    """FSDP 配置"""

    # 数据类型
    dtype: str = "bfloat16"

    # CPU 卸载
    offload_policy: bool = False

    # 梯度检查点
    gradient_checkpointing: bool = True

    # 分片策略
    sharding_strategy: str = "FULL_SHARD"
    # 可选: "FULL_SHARD" (ZeRO-3), "HYBRID_SHARD" (ZeRO-2)
```

---

## 8. Trainer 配置

```python
@dataclass
class TrainerConfig(BaseConfig):
    """Trainer 配置"""

    # 训练轮数
    total_epochs: int = 1
    total_training_steps: Optional[int] = None

    # 日志和保存频率
    logging_freq: int = 10
    save_freq: int = 100
    val_freq: int = 50

    # 检查点
    default_local_dir: str = "./checkpoints"
    resume_mode: str = "auto"
    # 可选: "auto", "must", "disable"

    # 验证
    val_before_train: bool = False
    val_generations_to_log_to_wandb: int = 0
```

---

## 9. 完整配置示例

### 9.1 GRPO 训练配置

```yaml
# config/grpo_qwen2_7b.yaml

data:
  train_files: data/train.parquet
  val_files: data/val.parquet
  prompt_key: prompt
  response_key: response

algorithm:
  gamma: 1.0
  lam: 1.0
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true
  use_kl_in_reward: false

  kl_ctrl:
    type: fixed
    kl_coef: 0.001

actor_rollout_ref:
  actor:
    strategy: fsdp2
    clip_ratio: 0.2
    entropy_coef: 0.01
    max_grad_norm: 1.0
    ppo_epochs: 1
    ppo_mini_batch_size: 8
    loss_agg_mode: token-mean
    policy_loss_fn: vanilla

    fsdp_config:
      dtype: bfloat16
      offload_policy: false
      gradient_checkpointing: true

  rollout:
    name: vllm
    temperature: 1.0
    top_p: 1.0
    max_new_tokens: 512
    tensor_parallel_size: 2
    n: 4  # 每个 prompt 生成 4 个响应

  ref:
    strategy: fsdp2
    fsdp_config:
      dtype: bfloat16

trainer:
  total_epochs: 1
  logging_freq: 10
  save_freq: 100
  val_freq: 50
  default_local_dir: ./checkpoints
```

### 9.2 PPO 训练配置

```yaml
# config/ppo_llama3_8b.yaml

algorithm:
  gamma: 1.0
  lam: 0.95
  adv_estimator: gae  # PPO 使用 GAE
  use_kl_in_reward: true

  kl_ctrl:
    type: adaptive
    kl_coef: 0.02
    target_kl: 0.01
    horizon: 10000

actor_rollout_ref:
  actor:
    strategy: fsdp2
    clip_ratio: 0.2
    entropy_coef: 0.01
    ppo_epochs: 4  # PPO 通常多轮更新
    ppo_mini_batch_size: 4

critic:
  strategy: fsdp2
  value_loss_coef: 0.5
  cliprange_value: 0.2
  ppo_epochs: 4
  ppo_mini_batch_size: 4

reward_model:
  strategy: fsdp2
  # 或使用函数奖励
  # enable: false

trainer:
  total_epochs: 1
  logging_freq: 10
  save_freq: 100
```

### 9.3 DAPO 训练配置

```yaml
# config/dapo_qwen2_32b.yaml

algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: false  # Dr.GRPO style

  filter_groups:
    enable: true
    metric: score
    max_num_gen_batches: 4

actor_rollout_ref:
  actor:
    clip_ratio_low: 0.1   # 非对称裁剪
    clip_ratio_high: 0.2
    loss_agg_mode: seq-mean-token-sum-norm  # DrGRPO 归一化

  rollout:
    n: 8  # 更多响应用于组内比较
```

---

## 10. 配置加载和使用

### 10.1 从 YAML 加载

```python
from omegaconf import OmegaConf

# 加载配置
config = OmegaConf.load("config/grpo_qwen2_7b.yaml")

# 合并命令行参数
cli_config = OmegaConf.from_cli()
config = OmegaConf.merge(config, cli_config)
```

### 10.2 转换为 Dataclass

```python
from verl.utils.config import omega_conf_to_dataclass
from verl.trainer.config import AlgoConfig

# 转换为类型安全的 dataclass
algo_config = omega_conf_to_dataclass(config.algorithm, AlgoConfig)
```

### 10.3 命令行覆盖

```bash
python train.py \
    --config config/grpo_qwen2_7b.yaml \
    algorithm.adv_estimator=grpo_vectorized \
    actor_rollout_ref.actor.clip_ratio=0.1 \
    trainer.total_epochs=2
```

---

## 11. 配置验证

verl 提供配置验证功能：

```python
from verl.utils.config import validate_config

# 验证配置
errors = validate_config(config)
if errors:
    for error in errors:
        print(f"配置错误: {error}")
```

---

## 12. 关键配置文件路径

| 配置类型 | 文件路径 |
|---------|---------|
| 算法配置 | `verl/trainer/config/algorithm.py` |
| Actor 配置 | `verl/workers/config/actor.py` |
| Critic 配置 | `verl/workers/config/critic.py` |
| Rollout 配置 | `verl/workers/config/rollout.py` |
| FSDP 配置 | `verl/workers/config/engine.py` |
| 示例配置 | `examples/*/config/*.yaml` |

---

## 13. 下一步

- 了解示例代码：[09_examples.md](09_examples.md)
