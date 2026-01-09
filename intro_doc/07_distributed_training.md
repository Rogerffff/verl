# 分布式训练架构

## 1. 概述

verl 使用 Ray 作为分布式训练的基础设施，支持 FSDP 和 Megatron-LM 两种训练后端。

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Driver Node                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      RayPPOTrainer                           │    │
│  │  - 训练循环协调                                               │    │
│  │  - 数据加载                                                   │    │
│  │  - 检查点管理                                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Worker Node 1  │  │  Worker Node 2  │  │  Worker Node N  │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │  GPU 0-7  │  │  │  │  GPU 0-7  │  │  │  │  GPU 0-7  │  │
│  │           │  │  │  │           │  │  │  │           │  │
│  │ - Actor   │  │  │  │ - Actor   │  │  │  │ - Actor   │  │
│  │ - Critic  │  │  │  │ - Critic  │  │  │  │ - Critic  │  │
│  │ - Rollout │  │  │  │ - Rollout │  │  │  │ - Rollout │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 2. Ray 集成

### 2.1 RayResourcePool

**文件位置**：`verl/single_controller/ray/base.py`

RayResourcePool 管理 GPU 资源分配：

```python
class RayResourcePool:
    """Ray 资源池"""

    def __init__(
        self,
        process_on_nodes: list[int],  # 每个节点的进程数
        use_gpu: bool = True,
        max_colocate_count: int = 3,
        name_prefix: str = "",
    ):
        self.process_on_nodes = process_on_nodes
        self.use_gpu = use_gpu
        self.max_colocate_count = max_colocate_count

    def get_placement_groups(self) -> list[PlacementGroup]:
        """创建 Ray Placement Groups"""
        pgs = []
        for node_idx, n_procs in enumerate(self.process_on_nodes):
            bundles = [{"GPU": 1} for _ in range(n_procs)]
            pg = placement_group(
                bundles,
                strategy="STRICT_PACK",  # 所有 GPU 在同一节点
                name=f"{self.name_prefix}_node{node_idx}"
            )
            pgs.append(pg)
        return pgs

    def split_resource_pool(self, ratios: list[float]) -> list["RayResourcePool"]:
        """按比例拆分资源池"""
        pools = []
        for ratio in ratios:
            new_process_on_nodes = [int(n * ratio) for n in self.process_on_nodes]
            pools.append(RayResourcePool(
                process_on_nodes=new_process_on_nodes,
                use_gpu=self.use_gpu,
            ))
        return pools
```

### 2.2 RayWorkerGroup

RayWorkerGroup 管理一组 Ray Actor：

```python
class RayWorkerGroup:
    """Ray Worker 组"""

    def __init__(
        self,
        resource_pool: RayResourcePool,
        ray_cls_with_init: RayClassWithInitArgs,
    ):
        self.resource_pool = resource_pool
        self.ray_cls = ray_cls_with_init

        # 创建 Ray Actors
        self._workers = []
        for pg in resource_pool.get_placement_groups():
            worker = ray_cls_with_init.cls.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                )
            ).remote(*ray_cls_with_init.args, **ray_cls_with_init.kwargs)
            self._workers.append(worker)

    def execute(self, method_name: str, *args, **kwargs):
        """在所有 Workers 上执行方法"""
        refs = [getattr(w, method_name).remote(*args, **kwargs) for w in self._workers]
        return ray.get(refs)
```

### 2.3 Dispatch/Collect 模式

使用装饰器绑定方法的分发和收集逻辑：

```python
from verl.single_controller.base.decorator import register, DispatchMode

class MyWorkerGroup(RayWorkerGroup):

    @register(dispatch_mode=DispatchMode.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """
        DP_COMPUTE_PROTO 模式:
        - dispatch: 将 DataProto 按 batch 维度切分到各 Worker
        - execute: 各 Worker 并行计算
        - collect: 将结果按 batch 维度合并
        """
        pass
```

**分发模式 (DispatchMode)**：

| 模式 | 描述 | 使用场景 |
|-----|------|---------|
| `RANK_ZERO` | 仅在 rank 0 执行 | 日志、保存 |
| `ONE_TO_ALL` | 复制数据到所有 Worker | 广播 |
| `ALL_TO_ALL` | 发送到所有 Worker | All-Gather |
| `DP_COMPUTE` | 数据并行计算 | 前向/反向 |
| `DP_COMPUTE_PROTO` | 数据并行 + DataProto | 大多数情况 |

---

## 3. FSDP 后端

### 3.1 FSDP 概述

FSDP (Fully Sharded Data Parallel) 实现 ZeRO-3 级别的内存优化：

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FSDP 分片策略                               │
│                                                                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │
│  │   GPU 0   │  │   GPU 1   │  │   GPU 2   │  │   GPU 3   │        │
│  │           │  │           │  │           │  │           │        │
│  │ Shard 0   │  │ Shard 1   │  │ Shard 2   │  │ Shard 3   │        │
│  │ (1/4 参数) │  │ (1/4 参数) │  │ (1/4 参数) │  │ (1/4 参数) │        │
│  │ (1/4 梯度) │  │ (1/4 梯度) │  │ (1/4 梯度) │  │ (1/4 梯度) │        │
│  │ (1/4 优化器)│  │ (1/4 优化器)│  │ (1/4 优化器)│  │ (1/4 优化器)│       │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │
│                                                                      │
│  前向/反向时: All-Gather 收集完整参数 → 计算 → Reduce-Scatter 梯度   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 FSDP Engine

**文件位置**：`verl/workers/engine/fsdp/`

```python
class FSDPEngine:
    """FSDP 训练引擎"""

    def __init__(self, config: FSDPEngineConfig):
        self.config = config

    def wrap_model(self, model: nn.Module) -> FSDP:
        """用 FSDP 包装模型"""
        fsdp_config = {
            "sharding_strategy": ShardingStrategy.FULL_SHARD,  # ZeRO-3
            "mixed_precision": MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
            "cpu_offload": CPUOffload(offload_params=self.config.offload_policy),
            "auto_wrap_policy": size_based_auto_wrap_policy,
        }
        return FSDP(model, **fsdp_config)
```

### 3.3 FSDP2 支持

verl 支持 PyTorch 最新的 FSDP2：

```yaml
# 配置文件中启用 FSDP2
actor_rollout_ref.ref.strategy: fsdp2
actor_rollout_ref.actor.strategy: fsdp2
critic.strategy: fsdp2
reward_model.strategy: fsdp2

# CPU 卸载（FSDP2 支持与梯度累积兼容）
actor_rollout_ref.actor.fsdp_config.offload_policy: True
```

---

## 4. Megatron-LM 后端

### 4.1 Megatron 概述

Megatron-LM 支持多维度并行：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Megatron 3D 并行                                │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     数据并行 (DP)                            │    │
│  │  ┌───────────────────┐  ┌───────────────────┐               │    │
│  │  │   DP Replica 0    │  │   DP Replica 1    │               │    │
│  │  │                   │  │                   │               │    │
│  │  │  ┌─────────────┐  │  │  ┌─────────────┐  │               │    │
│  │  │  │张量并行(TP)  │  │  │  │张量并行(TP)  │  │               │    │
│  │  │  │ GPU0 | GPU1 │  │  │  │ GPU4 | GPU5 │  │               │    │
│  │  │  └─────────────┘  │  │  └─────────────┘  │               │    │
│  │  │  ┌─────────────┐  │  │  ┌─────────────┐  │               │    │
│  │  │  │流水线(PP)    │  │  │  │流水线(PP)    │  │               │    │
│  │  │  │ Stage 0→1   │  │  │  │ Stage 0→1   │  │               │    │
│  │  │  └─────────────┘  │  │  └─────────────┘  │               │    │
│  │  └───────────────────┘  └───────────────────┘               │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 并行维度

| 并行类型 | 描述 | 适用场景 |
|---------|------|---------|
| **张量并行 (TP)** | 切分单个层的权重 | 单节点内 GPU 通信 |
| **流水线并行 (PP)** | 切分模型层到不同设备 | 跨节点，减少内存 |
| **数据并行 (DP)** | 复制模型，切分数据 | 扩展吞吐量 |
| **专家并行 (EP)** | MoE 模型的专家分布 | 大规模 MoE |

### 4.3 Megatron Engine

**文件位置**：`verl/workers/engine/megatron/`

```python
class MegatronEngine:
    """Megatron 训练引擎"""

    def __init__(self, config):
        self.config = config

        # 初始化并行组
        self.tp_size = config.tensor_parallel_size
        self.pp_size = config.pipeline_parallel_size
        self.ep_size = config.expert_parallel_size

    def init_parallel_groups(self):
        """初始化 Megatron 并行组"""
        from megatron.core import parallel_state

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            expert_model_parallel_size=self.ep_size,
        )
```

---

## 5. 3D-HybridEngine

### 5.1 问题背景

RLHF 训练需要在训练模式和推理模式间切换：
- **训练模式**：FSDP/Megatron 分片
- **推理模式**：vLLM/SGLang 的张量并行

传统方法需要复制模型，浪费内存。

### 5.2 3D-HybridEngine 解决方案

```
┌─────────────────────────────────────────────────────────────────────┐
│                        3D-HybridEngine                               │
│                                                                      │
│  训练阶段 (FSDP)              推理阶段 (vLLM)                         │
│  ┌───────────────┐            ┌───────────────┐                     │
│  │  GPU 0: Shard │            │  GPU 0: TP 0  │                     │
│  │  GPU 1: Shard │    ══►     │  GPU 1: TP 1  │                     │
│  │  GPU 2: Shard │  权重重分片  │  GPU 2: TP 0  │                     │
│  │  GPU 3: Shard │            │  GPU 3: TP 1  │                     │
│  └───────────────┘            └───────────────┘                     │
│                                                                      │
│  特点:                                                               │
│  - 零内存冗余                                                         │
│  - 高效权重广播                                                       │
│  - 异步切换                                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 权重同步流程

```python
async def sync_weights(actor, rollout):
    """从 Actor 同步权重到 Rollout"""

    # 1. 释放 Rollout 的 GPU 内存
    await rollout.release()

    # 2. 从 FSDP 收集权重
    def weights_generator():
        for name, param in actor.module.named_parameters():
            # FSDP: All-Gather 收集完整参数
            full_param = param.full_tensor()
            yield name, full_param

    # 3. 更新到 Rollout (vLLM/SGLang)
    await rollout.update_weights(weights_generator())

    # 4. 恢复 Rollout
    await rollout.resume(["weights"])
```

---

## 6. 设备映射和资源调度

### 6.1 ResourcePoolManager

```python
@dataclass
class ResourcePoolManager:
    """资源池管理器"""

    resource_pool_spec: dict[str, list[int]]  # 资源池规格
    mapping: dict[Role, str]                   # 角色映射

    def create_resource_pool(self):
        """创建资源池"""
        for name, process_on_nodes in self.resource_pool_spec.items():
            pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=3,
            )
            self.resource_pool_dict[name] = pool
```

### 6.2 设备映射示例

```python
# 4 节点 x 8 GPU 集群配置
resource_pool_spec = {
    "actor_rollout": [8, 8, 8, 8],  # 32 GPUs for Actor + Rollout
    "critic": [8, 8],               # 16 GPUs for Critic
    "reward": [8],                  # 8 GPUs for Reward Model
}

mapping = {
    Role.ActorRollout: "actor_rollout",
    Role.Critic: "critic",
    Role.RewardModel: "reward",
}

manager = ResourcePoolManager(
    resource_pool_spec=resource_pool_spec,
    mapping=mapping,
)
```

---

## 7. 序列并行 (Ulysses)

### 7.1 Ulysses 概述

Ulysses 是 DeepSpeed 的序列并行方案，用于处理超长序列：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Ulysses 序列并行                                 │
│                                                                      │
│  输入序列: [T1, T2, T3, T4, T5, T6, T7, T8]                          │
│                                                                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │
│  │   GPU 0   │  │   GPU 1   │  │   GPU 2   │  │   GPU 3   │        │
│  │ [T1, T2]  │  │ [T3, T4]  │  │ [T5, T6]  │  │ [T7, T8]  │        │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │
│                                                                      │
│  注意力计算: All-to-All 通信交换 K/V                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 配置

```yaml
actor:
  ulysses_sequence_parallel_size: 4  # 序列并行度
```

---

## 8. 配置示例

### 8.1 FSDP 配置

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp2
    fsdp_config:
      dtype: bfloat16
      offload_policy: false
      gradient_checkpointing: true

  rollout:
    name: vllm
    tensor_parallel_size: 2
```

### 8.2 Megatron 配置

```yaml
actor_rollout_ref:
  actor:
    strategy: megatron
    megatron_config:
      tensor_parallel_size: 8
      pipeline_parallel_size: 2
      expert_parallel_size: 4  # for MoE
```

---

## 9. 关键代码路径

| 组件 | 文件 | 类/函数 |
|-----|------|--------|
| Ray 资源池 | `single_controller/ray/base.py` | `RayResourcePool` |
| Ray Worker 组 | `single_controller/ray/base.py` | `RayWorkerGroup` |
| Worker 基类 | `single_controller/base/worker.py` | `Worker` |
| FSDP Engine | `workers/engine/fsdp/` | `FSDPEngine` |
| Megatron Engine | `workers/engine/megatron/` | `MegatronEngine` |
| 分片管理 | `workers/sharding_manager/` | `BaseShardingManager` |

---

## 10. 性能调优

### 10.1 关键参数

| 参数 | 描述 | 建议值 |
|-----|------|-------|
| `micro_batch_size` | 单次前向的批大小 | 根据 GPU 内存调整 |
| `gradient_accumulation_steps` | 梯度累积步数 | 增大可减少通信 |
| `tensor_parallel_size` | 张量并行度 | 通常 2-8 |
| `ulysses_sp_size` | 序列并行度 | 长序列时使用 |

### 10.2 性能监控

```python
# 使用内置的性能分析器
from verl.utils.profiler import GPUMemoryLogger

with GPUMemoryLogger():
    trainer.fit()
```

---

## 11. 下一步

- 了解配置系统：[08_configuration.md](08_configuration.md)
- 了解示例代码：[09_examples.md](09_examples.md)
