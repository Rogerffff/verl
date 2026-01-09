# verl 系统架构详解

## 1. 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              PPO Ray Trainer                                  │
│                        (verl/trainer/ppo/ray_trainer.py)                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         训练循环 (Training Loop)                         │ │
│  │  1. 生成序列 (Rollout)                                                   │ │
│  │  2. 计算奖励 (Reward)                                                    │ │
│  │  3. 计算优势 (Advantage)                                                 │ │
│  │  4. 更新策略 (Policy Update)                                             │ │
│  │  5. 更新价值函数 (Value Update) - 仅 PPO                                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Rollout Worker    │    │    Actor Worker     │    │   Critic Worker     │
│  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │
│  │    vLLM       │  │    │  │    FSDP       │  │    │  │    FSDP       │  │
│  │      or       │  │    │  │      or       │  │    │  │      or       │  │
│  │   SGLang      │  │    │  │  Megatron-LM  │  │    │  │  Megatron-LM  │  │
│  └───────────────┘  │    │  └───────────────┘  │    │  └───────────────┘  │
│                     │    │                     │    │                     │
│  - 序列生成          │    │  - compute_log_prob │    │  - compute_values   │
│  - 权重同步          │    │  - update_policy    │    │  - update_critic    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │
                                      ▼
                          ┌─────────────────────┐
                          │   Reward Manager    │
                          │  ┌───────────────┐  │
                          │  │  奖励函数      │  │
                          │  │  或           │  │
                          │  │  奖励模型      │  │
                          │  └───────────────┘  │
                          └─────────────────────┘
```

---

## 2. Worker 体系结构

verl 采用 Worker 模式来分离不同的计算任务。每个 Worker 负责特定的功能。

### 2.1 Worker 基类

**文件位置**：`verl/single_controller/base/worker.py`

```python
class Worker:
    """Worker 基类，管理分布式环境信息"""

    def __init__(self):
        # 分布式 Rank 信息
        self.rank_info: DistRankInfo  # 包含 TP, DP, PP, CP 等维度
        self.global_info: DistGlobalInfo  # 全局信息

        # 环境配置
        # WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
```

### 2.2 Actor Worker

**文件位置**：`verl/workers/actor/base.py`

Actor Worker 负责策略模型的训练和推理。

```python
class BasePPOActor(ABC):
    """PPO Actor 基类"""

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

        Args:
            data: 包含训练数据的 DataProto

        Returns:
            Dict: 包含 loss, grad_norm 等统计信息
        """
        pass
```

**实现类**：
- `DataParallelPPOActor`（`dp_actor.py`）：使用 FSDP 的 Actor
- `MegatronPPOActor`（`megatron_actor.py`）：使用 Megatron-LM 的 Actor

### 2.3 Critic Worker

**文件位置**：`verl/workers/critic/base.py`

Critic Worker 负责价值函数的估计（仅 PPO 需要，GRPO 不需要）。

```python
class BasePPOCritic(ABC):
    """PPO Critic 基类"""

    @abstractmethod
    def compute_values(self, data: DataProto) -> torch.Tensor:
        """计算状态价值

        Args:
            data: 包含状态信息的 DataProto

        Returns:
            价值估计 tensor
        """
        pass

    @abstractmethod
    def update_critic(self, data: DataProto):
        """更新价值函数"""
        pass
```

**实现类**：
- `DataParallelPPOCritic`（`dp_critic.py`）
- `MegatronPPOCritic`（`megatron_critic.py`）

### 2.4 Rollout Worker

**文件位置**：`verl/workers/rollout/base.py`

Rollout Worker 负责使用推理引擎生成序列。

```python
class BaseRollout(ABC):
    """Rollout 基类"""

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
        """释放 GPU 内存中的权重和 KV cache"""
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """批量生成序列"""
        pass
```

**注册的实现类**：
```python
_ROLLOUT_REGISTRY = {
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.vLLMAsyncRollout",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
}
```

### 2.5 Reward Manager

**文件位置**：`verl/workers/reward_manager/`

Reward Manager 负责计算奖励信号。

- `NaiveRewardManager`：基础奖励计算
- `DAPORewardManager`：DAPO 特定奖励计算（含超长惩罚）
- `PRIMERewardManager`：PRIME 算法奖励计算

---

## 3. Engine 后端

verl 支持多种训练引擎后端。

### 3.1 FSDP (Fully Sharded Data Parallel)

**文件位置**：`verl/workers/engine/fsdp/`

FSDP 是 PyTorch 原生的分布式训练方案，实现 ZeRO-3 级别的内存优化。

**特性**：
- 完全分片模型参数、梯度和优化器状态
- 支持混合精度训练
- 支持激活检查点
- 支持 CPU 卸载

**配置示例**：
```yaml
actor:
  strategy: fsdp2  # 使用 FSDP2
  fsdp_config:
    offload_policy: True  # CPU 卸载
```

### 3.2 Megatron-LM

**文件位置**：`verl/workers/engine/megatron/`

Megatron-LM 是 NVIDIA 的大规模模型训练框架。

**特性**：
- 张量并行 (Tensor Parallelism)
- 流水线并行 (Pipeline Parallelism)
- 专家并行 (Expert Parallelism) - 用于 MoE 模型
- 支持 DeepSeek-671B、Qwen3-235B 等超大模型

---

## 4. 推理引擎集成

### 4.1 vLLM 集成

**文件位置**：`verl/workers/rollout/vllm_rollout/`

vLLM 是高性能 LLM 推理引擎，使用 PagedAttention 技术。

**集成方式**：
```
Training Engine (FSDP/Megatron)
         │
         │ 权重同步
         ▼
    vLLM Engine
         │
         │ 生成序列
         ▼
    Generated Sequences
```

**权重同步模式**：
- DTensor 权重加载器（推荐用于 FSDP）
- HuggingFace 权重加载器
- Megatron 权重加载器

### 4.2 SGLang 集成

**文件位置**：`verl/workers/rollout/sglang_rollout/`

SGLang 是另一个高性能推理框架，支持结构化生成。

**运行模式**：
- Hybrid 模式：与训练引擎共享 GPU，进程内权重同步
- Standalone/Colocated 模式：独立 GPU，Server 模式运行

---

## 5. 数据协议 (DataProto)

**文件位置**：`verl/protocol.py`

`DataProto` 是 verl 中统一的数据交换协议，用于 Worker 之间的数据传递。

### 5.1 DataProto 结构

```python
@dataclass
class DataProto:
    """
    标准数据交换协议

    Attributes:
        batch: TensorDict，存储 tensor 数据（相同 batch size）
        non_tensor_batch: dict，存储非 tensor 数据（如字符串）
        meta_info: dict，存储元信息
    """
    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)
```

### 5.2 常用字段

在 RL 训练中，DataProto 通常包含以下字段：

| 字段名 | 类型 | 描述 |
|-------|------|------|
| `input_ids` | Tensor | 输入 token IDs |
| `attention_mask` | Tensor | 注意力掩码 |
| `position_ids` | Tensor | 位置 IDs |
| `responses` | Tensor | 生成的响应 |
| `log_probs` | Tensor | 对数概率 |
| `old_log_probs` | Tensor | 旧策略的对数概率 |
| `values` | Tensor | 价值估计 |
| `rewards` | Tensor | 奖励信号 |
| `advantages` | Tensor | 优势估计 |

### 5.3 核心操作

```python
# 创建 DataProto
data = DataProto.from_single_dict({"input_ids": ids, "attention_mask": mask})

# 索引操作
item = data[0]           # 单个样本，返回 DataProtoItem
subset = data[10:20]     # 切片，返回 DataProto
selected = data[[1,3,5]] # 选择索引，返回 DataProto

# 拼接
combined = DataProto.concat([data1, data2])

# 切分
chunks = data.chunk(num_chunks)

# 填充到可整除
padded, pad_size = pad_dataproto_to_divisor(data, divisor)
```

---

## 6. 训练循环时序图

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│  Trainer   │   │  Rollout   │   │   Actor    │   │   Critic   │   │   Reward   │
└─────┬──────┘   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
      │                │                │                │                │
      │ 1. 同步权重     │                │                │                │
      ├───────────────►│                │                │                │
      │                │                │                │                │
      │ 2. 生成序列     │                │                │                │
      │◄───────────────┤                │                │                │
      │                │                │                │                │
      │ 3. 计算奖励     │                │                │                │
      ├────────────────┼────────────────┼────────────────┼───────────────►│
      │◄───────────────┼────────────────┼────────────────┼────────────────┤
      │                │                │                │                │
      │ 4. 计算 ref log_prob            │                │                │
      ├────────────────┼───────────────►│                │                │
      │◄───────────────┼────────────────┤                │                │
      │                │                │                │                │
      │ 5. 计算 values（PPO only）       │                │                │
      ├────────────────┼────────────────┼───────────────►│                │
      │◄───────────────┼────────────────┼────────────────┤                │
      │                │                │                │                │
      │ 6. 计算优势     │                │                │                │
      ├────────────────┼────────────────┼────────────────┤                │
      │                │                │                │                │
      │ 7. 更新策略     │                │                │                │
      ├────────────────┼───────────────►│                │                │
      │◄───────────────┼────────────────┤                │                │
      │                │                │                │                │
      │ 8. 更新价值函数（PPO only）       │                │                │
      ├────────────────┼────────────────┼───────────────►│                │
      │◄───────────────┼────────────────┼────────────────┤                │
      │                │                │                │                │
```

### 详细步骤说明

1. **同步权重**：将训练好的 Actor 权重同步到 Rollout 引擎
2. **生成序列**：使用 vLLM/SGLang 生成响应序列
3. **计算奖励**：使用奖励函数或奖励模型计算奖励
4. **计算 Reference Log Prob**：计算参考策略的对数概率（用于 KL 惩罚）
5. **计算 Values**：使用 Critic 计算状态价值（仅 PPO）
6. **计算优势**：根据算法类型计算优势估计
7. **更新策略**：使用计算的优势更新 Actor 策略
8. **更新价值函数**：更新 Critic 网络（仅 PPO）

---

## 7. WorkerGroup 和分发机制

### 7.1 WorkerGroup

**文件位置**：`verl/single_controller/base/worker_group.py`

WorkerGroup 管理一组相同类型的 Worker。

```python
class WorkerGroup:
    """管理一组 Worker"""

    def __init__(self, resource_pool: ResourcePool):
        self.resource_pool = resource_pool
        self._workers = []  # Worker 列表
```

### 7.2 Dispatch/Collect 模式

verl 使用装饰器模式来绑定 Worker 方法的分发和收集逻辑。

```python
from verl.single_controller.base.decorator import register

@register(dispatch_mode=DispatchMode.DP_COMPUTE_PROTO)
def compute_log_prob(self, data: DataProto) -> DataProto:
    """
    DP_COMPUTE_PROTO 模式：
    - dispatch: 将数据按 batch 维度切分到各 Worker
    - execute: 各 Worker 并行计算
    - collect: 将结果按 batch 维度合并
    """
    pass
```

**分发模式 (DispatchMode)**：

| 模式 | 描述 |
|-----|------|
| `RANK_ZERO` | 仅在 rank 0 执行 |
| `ONE_TO_ALL` | 复制数据到所有 Worker |
| `ALL_TO_ALL` | 发送到所有 Worker（all gather） |
| `DP_COMPUTE` | 数据并行计算，按 batch 切分 |
| `DP_COMPUTE_PROTO` | 数据并行计算，使用 DataProto |

---

## 8. 3D-HybridEngine

verl 的核心创新之一是 3D-HybridEngine，用于高效地在训练和推理之间切换模型。

### 8.1 问题背景

在 RLHF 训练中，模型需要在两种模式间切换：
- **训练模式**：使用 FSDP/Megatron 进行梯度计算
- **推理模式**：使用 vLLM/SGLang 进行序列生成

传统方法需要完整复制模型，浪费内存。

### 8.2 解决方案

3D-HybridEngine 通过高效的模型重分片（resharding）来消除内存冗余：

```
训练状态 (FSDP ZeRO-3 分片)        推理状态 (Tensor Parallel 分片)
┌────────────────────────┐        ┌────────────────────────┐
│  GPU 0: Shard 0        │        │  GPU 0: Layer 0-N/2    │
│  GPU 1: Shard 1        │   =>   │  GPU 1: Layer N/2-N    │
│  GPU 2: Shard 2        │        │  GPU 0,1: TP 分片      │
│  GPU 3: Shard 3        │        │                        │
└────────────────────────┘        └────────────────────────┘
```

### 8.3 权重同步流程

```python
# 从训练引擎提取权重
weights_generator = actor.get_weights_generator()

# 更新到推理引擎
await rollout.update_weights(weights_generator)
```

---

## 9. 总结

verl 的架构设计遵循以下原则：

1. **模块化**：清晰分离 Actor、Critic、Rollout、Reward 职责
2. **可扩展**：支持多种训练后端（FSDP、Megatron）和推理引擎（vLLM、SGLang）
3. **高效**：3D-HybridEngine 消除训练/推理切换的内存冗余
4. **统一协议**：DataProto 提供标准化的数据交换格式
5. **分布式**：Ray 提供灵活的分布式调度能力

下一章将深入分析 PPO 算法的具体实现。
