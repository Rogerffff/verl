# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
verl Rollout Replica 模块

=== 核心概念：什么是 Replica？===

Replica（副本）是 verl 中管理推理服务器的核心抽象。

想象你有一台 8 GPU 的服务器，想用 vLLM 做推理：
- 如果模型需要 2 GPU（Tensor Parallel=2），你可以启动 4 个独立的推理服务器
- 每个服务器就是一个 "Replica"
- 4 个 Replica 可以并行处理请求，提高吞吐量

公式：num_replicas = total_gpus / tensor_parallel_size
例如：8 GPU / TP=2 = 4 个 Replica

=== Replica 的三种部署模式 ===

1. STANDALONE（独立模式）← Phase 0 评测使用
   - Replica 独占 GPU 资源
   - 模型从磁盘加载（load_format="auto"）
   - 通过 HTTP API 对外提供服务
   - 适用：纯推理、评测、off-policy 训练

2. HYBRID（混合模式）← GRPO/PPO 训练使用
   - Replica 与训练引擎在同一进程
   - 需要权重同步（训练后同步到推理引擎）
   - 适用：on-policy 训练

3. COLOCATED（共置模式）
   - Replica 与训练引擎共享 GPU，但在不同进程
   - 不需要权重同步
   - 适用：GRM (LLM as a Judge)

=== 使用示例 ===

```python
from verl.workers.rollout.replica import get_rollout_replica_class

# 1. 获取 Replica 类（vLLM 或 SGLang）
replica_class = get_rollout_replica_class("vllm")

# 2. 创建 Replica 实例
replica = replica_class(
    replica_rank=0,           # 第几个 Replica（从 0 开始）
    config=rollout_config,    # 推理配置
    model_config=model_config, # 模型配置
    gpus_per_node=8,          # 每节点 GPU 数
)

# 3. 初始化（三选一）
await replica.init_standalone()   # 独立模式（Phase 0 用这个）
await replica.init_hybrid(worker_group)  # 混合模式
await replica.init_colocated(resource_pool)  # 共置模式

# 4. 使用
server_address = replica.server_address  # 获取 HTTP 地址
# 然后通过 OpenAI API 调用：POST http://{server_address}/v1/chat/completions
```
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional

from omegaconf import DictConfig
from pydantic import BaseModel
from ray.actor import ActorHandle

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayResourcePool, ResourcePoolManager
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig

logger = logging.getLogger(__file__)


class TokenOutput(BaseModel):
    """
    Token 输出结构，用于 Token-in-Token-out 生成模式

    这是 Replica 的 generate() 方法返回的数据结构，包含生成的 token 及其元信息。
    """
    token_ids: list[int]
    """生成的 token ID 列表（response 部分，不含 prompt）"""

    log_probs: Optional[list[float]] = None
    """每个 token 的 log 概率，用于计算 PPO/GRPO 中的 ratio"""

    routed_experts: Optional[Any] = None
    """MoE 模型中路由到的专家信息（普通模型为 None）"""

    stop_reason: Optional[str] = None
    """停止原因：'completed'（正常结束）, 'aborted'（被中断）, None（未知）"""


class RolloutMode(Enum):
    """
    Rollout 部署模式枚举

    verl 支持三种部署模式，根据是否需要训练、是否需要权重同步来选择：

    ┌─────────────┬────────────┬────────────┬─────────────────┐
    │ 模式        │ 需要训练   │ 权重同步   │ 适用场景        │
    ├─────────────┼────────────┼────────────┼─────────────────┤
    │ STANDALONE  │ ❌         │ ❌         │ Phase 0 评测    │
    │ HYBRID      │ ✅         │ ✅         │ GRPO/PPO 训练   │
    │ COLOCATED   │ ✅         │ ❌         │ GRM (Judge)     │
    └─────────────┴────────────┴────────────┴─────────────────┘
    """

    # 混合模式：推理引擎和训练引擎（FSDP/Megatron）在同一进程中
    # 特点：共享 GPU，需要权重同步（每次训练后把新权重同步到推理引擎）
    # 用途：on-policy 训练（GRPO、PPO）
    # load_format 应设为 "dummy"（空壳模型，由训练引擎同步权重）
    HYBRID = "hybrid"

    # 共置模式：推理引擎和训练引擎在同一 Ray Placement Group，但不同进程
    # 特点：共享 GPU，不需要权重同步（推理用独立的固定权重）
    # 用途：GRM (LLM as a Judge)，用另一个模型做评分
    COLOCATED = "colocated"

    # 独立模式：推理引擎独占 GPU 资源，与训练完全分离
    # 特点：独立资源，不需要权重同步，模型从磁盘一次性加载
    # 用途：Phase 0 评测、off-policy 训练、批量推理
    # load_format 应设为 "auto"（从磁盘加载真实权重）
    # ⚠️ 重要：Phase 0 必须使用此模式！
    STANDALONE = "standalone"


class RolloutReplica(ABC):
    """
    Rollout Replica 抽象基类 - verl 推理服务器的核心抽象

    === 什么是 Replica？===

    Replica 是一个独立的推理服务器实例。在多 GPU 环境下，你可以启动多个 Replica
    来并行处理推理请求，每个 Replica 管理一组 GPU。

    例如，8 GPU 服务器，TP=2 时：
    - num_replicas = 8 / 2 = 4
    - Replica 0: GPU 0-1
    - Replica 1: GPU 2-3
    - Replica 2: GPU 4-5
    - Replica 3: GPU 6-7

    === 与命令行启动的对应关系 ===

    这个类等价于在命令行启动多个推理服务器：

    SGLang 命令行:
    ```bash
    python -m sglang.launch_server --node-rank 0 --nnode 2 ...
    python -m sglang.launch_server --node-rank 1 --nnode 2 ...
    ```

    vLLM 命令行:
    ```bash
    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 0 ...
    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 8 ...
    ```

    === 使用方式 ===

    ```python
    # 1. 获取具体实现类
    replica_class = get_rollout_replica_class("vllm")  # 或 "sglang"

    # 2. 创建实例
    replica = replica_class(replica_rank=0, config=..., model_config=...)

    # 3. 初始化（三选一）
    await replica.init_standalone()   # ← Phase 0 用这个
    await replica.init_hybrid(...)
    await replica.init_colocated(...)

    # 4. 获取服务地址并调用
    address = replica.server_address  # 如 "192.168.1.1:8000"
    # POST http://{address}/v1/chat/completions
    ```

    Args:
        replica_rank: 当前 Replica 的序号（从 0 开始）
        config: RolloutConfig 推理配置（温度、top_p、max_tokens 等）
        model_config: HFModelConfig 模型配置（模型路径、dtype 等）
        gpus_per_node: 每个节点的 GPU 数量（默认 8）
        is_reward_model: 是否是奖励模型（用于 GRM 场景）
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ) -> None:
        # === 基本属性 ===
        self.replica_rank = replica_rank  # 当前 Replica 的序号
        self.config = omega_conf_to_dataclass(config)  # 推理配置
        self.model_config: HFModelConfig = model_config  # 模型配置

        # === 计算并行度 ===
        # world_size = TP * DP * PP（这个 Replica 需要多少个 GPU）
        # 对于单 Replica：通常 DP=1, PP=1，所以 world_size = TP
        self.world_size = (
            self.config.tensor_model_parallel_size      # 张量并行（模型切分到多个 GPU）
            * self.config.data_parallel_size            # 数据并行（多个模型副本）
            * self.config.pipeline_model_parallel_size  # 流水线并行（模型按层切分）
        )

        # === 计算节点分布 ===
        self.gpus_per_node = min(gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_node == 0, (
            f"world_size {self.world_size} must be divisible by gpus_per_node {self.gpus_per_node}"
        )
        self.nnodes = self.world_size // self.gpus_per_node  # 需要多少个节点
        self.is_reward_model = is_reward_model

        # === 运行时状态（初始化后填充）===
        self.rollout_mode: RolloutMode = None  # 部署模式
        self.workers: list[ActorHandle] = []   # Ray Worker Actor 列表
        self.resource_pool: RayResourcePool = None  # GPU 资源池

        # === HTTP 服务器相关 ===
        self.servers: list[ActorHandle] = []   # 每个节点的 HTTP 服务器 Actor
        self._server_address: str = None       # 主服务器地址（用于 API 调用）
        self._server_handle: ActorHandle = None  # 主服务器 Actor 句柄

    async def init_hybrid(self, worker_group: RayWorkerGroup):
        """
        初始化混合模式（HYBRID）- 用于 GRPO/PPO 训练

        混合模式下，推理引擎和训练引擎在同一进程中：
        - 训练时：使用 FSDP/Megatron 进行梯度计算
        - 推理时：使用 vLLM/SGLang 生成样本
        - 每次训练后需要同步权重到推理引擎

        配置要求：load_format="dummy"（创建空壳模型，由训练引擎同步权重）

        Args:
            worker_group: 已初始化训练引擎的 RayWorkerGroup
                          （这些 Worker 已经加载了模型权重）
        """
        self.rollout_mode = RolloutMode.HYBRID
        # 从 worker_group 中切片出属于这个 Replica 的 Workers
        # 例如：Replica 0 用 workers[0:2]，Replica 1 用 workers[2:4]
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]
        await self.launch_servers()

    # TODO(sgm): this should be the default solution, but need to make the RolloutMode more clear.
    async def init_colocated(self, resource_pool: RayResourcePool):
        """
        初始化共置模式（COLOCATED）- 用于 GRM (LLM as a Judge)

        共置模式下，推理引擎和训练引擎共享 GPU，但在不同进程：
        - 推理进程：运行评分模型（如 Judge LLM）
        - 训练进程：运行被训练的模型
        - 不需要权重同步（评分模型权重固定）

        典型用途：使用另一个 LLM 对生成结果进行评分

        Args:
            resource_pool: 已创建的 Ray Placement Group（GPU 资源池）
                          训练进程已在其中启动
        """
        self.rollout_mode = RolloutMode.COLOCATED
        self.resource_pool = resource_pool

        # 在同一资源池中创建新的 Worker（与训练 Worker 共存）
        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,  # 不进行紧凑排布，保持与训练 Worker 的对应关系
            name_prefix=f"rollout_colocate_{self.replica_rank}"
            if not self.is_reward_model
            else f"rollout_reward_colocate_{self.replica_rank}",
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    async def init_standalone(self):
        """
        初始化独立模式（STANDALONE）- ⭐ Phase 0 评测使用此方法 ⭐

        独立模式下，推理引擎独占 GPU 资源，与训练完全分离：
        - 创建独立的 GPU 资源池
        - 模型从磁盘一次性加载
        - 通过 HTTP API 对外提供服务
        - 运行期间模型权重保持不变

        === Phase 0 使用示例 ===
        ```python
        replica = vLLMReplica(replica_rank=0, config=..., model_config=...)
        await replica.init_standalone()
        address = replica.server_address  # "192.168.1.1:8000"
        # 然后通过 OpenAI API 调用
        ```

        ⚠️ 重要配置：load_format="auto"（从磁盘加载真实权重）
        如果使用 "dummy"，模型将不会加载任何权重，输出全是随机的！
        """
        # === Step 1: 创建独立的 GPU 资源池 ===
        self.rollout_mode = RolloutMode.STANDALONE
        resource_pool_name = (
            f"rollout_pool_{self.replica_rank}"
            if not self.is_reward_model
            else f"rollout_pool_reward_{self.replica_rank}"
        )
        # 资源池规格：每个节点分配 gpus_per_node 个 GPU
        # 例如：{"rollout_pool_0": [8, 8]} 表示 2 个节点，每节点 8 GPU
        resource_pool_spec = {
            resource_pool_name: [self.gpus_per_node] * self.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=None)
        resource_pool_manager.create_resource_pool()  # 创建 Ray Placement Group
        self.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]

        # === Step 2: 创建 Worker 组 ===
        # Worker 是 Ray Actor，每个 Worker 管理一个 GPU 上的模型分片
        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),  # 获取具体的 Worker 类
            bin_pack=False,
            name_prefix=f"rollout_standalone_{self.replica_rank}"
            if not self.is_reward_model
            else f"rollout_reward_standalone_{self.replica_rank}",
        )
        self.workers = worker_group.workers

        # === Step 3: 启动 HTTP 服务器 ===
        await self.launch_servers()

    @abstractmethod
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """
        获取 Ray Worker Actor 类及其初始化参数

        子类必须实现此方法，返回具体的 Worker 类（如 vLLMWorker）。
        Worker 是实际运行推理引擎的 Ray Actor。

        Returns:
            RayClassWithInitArgs: 包含 Worker 类和初始化参数的封装对象
        """
        raise NotImplementedError

    @abstractmethod
    async def launch_servers(self):
        """
        启动 HTTP 服务器

        子类必须实现此方法，在每个节点上启动 HTTP 服务器。
        服务器提供 OpenAI 兼容的 API（/v1/chat/completions）。

        调用后会设置：
        - self.servers: 所有服务器的 Actor 句柄列表
        - self._server_address: 主服务器的地址（如 "192.168.1.1:8000"）
        - self._server_handle: 主服务器的 Actor 句柄
        """
        raise NotImplementedError

    @property
    def server_address(self) -> str:
        """
        获取 HTTP 服务器地址

        用于通过 OpenAI API 调用推理服务：
        ```python
        import aiohttp
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"http://{replica.server_address}/v1/chat/completions",
                json={"model": "xxx", "messages": [...], "temperature": 0}
            )
        ```

        Returns:
            str: 服务器地址，格式为 "ip:port"
        """
        return self._server_address

    @property
    def server_handle(self) -> ActorHandle:
        """
        获取服务器 Ray Actor 句柄

        用于 Token-in-Token-out 生成模式（直接调用 Actor 方法，而非 HTTP）：
        ```python
        result = await replica.server_handle.generate.remote(
            prompt_token_ids=[...],
            sampling_params=...
        )
        ```

        Returns:
            ActorHandle: Ray Actor 句柄
        """
        return self._server_handle

    async def wake_up(self):
        """
        唤醒所有服务器（从休眠状态恢复）

        在休眠后重新激活服务器，准备处理请求。
        用于节省 GPU 显存的场景（训练时让推理服务器休眠）。
        """
        await asyncio.gather(*[server.wake_up.remote() for server in self.servers])

    async def sleep(self):
        """
        让所有服务器进入休眠状态

        休眠状态下会释放部分 GPU 显存，用于：
        - 训练阶段不需要推理时
        - 需要临时释放显存给其他任务
        """
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def clear_kv_cache(self):
        """
        清空所有服务器的 KV Cache

        KV Cache 是推理引擎用于加速生成的缓存，清空后：
        - 释放显存
        - 下次请求需要重新计算（可能更慢但结果一致）

        典型用途：批量评测之间清理状态
        """
        await asyncio.gather(*[server.clear_kv_cache.remote() for server in self.servers])


class RolloutReplicaRegistry:
    """
    Rollout Replica 注册表 - 工厂模式实现

    管理不同推理引擎（vLLM、SGLang）的 Replica 实现类。
    使用延迟加载，只在需要时才导入对应的模块。

    支持的引擎：
    - "vllm": vLLM 推理引擎（推荐，更成熟）
    - "sglang": SGLang 推理引擎（可选）
    """

    # 注册表：name -> loader function
    _registry: dict[str, Callable[[], type[RolloutReplica]]] = {}

    @classmethod
    def register(cls, name: str, loader: Callable[[], type[RolloutReplica]]) -> None:
        """
        注册新的 Replica 类型

        Args:
            name: 引擎名称（如 "vllm", "sglang"）
            loader: 延迟加载函数，返回对应的 Replica 类
        """
        cls._registry[name] = loader

    @classmethod
    def get(cls, name: str) -> type[RolloutReplica]:
        """
        获取 Replica 类

        Args:
            name: 引擎名称

        Returns:
            对应的 Replica 类（如 vLLMReplica）

        Raises:
            ValueError: 如果引擎名称未注册
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown rollout mode: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]()  # 调用 loader 函数


# === 内置引擎的加载函数 ===

def _load_vllm():
    """
    延迟加载 vLLM Replica

    vLLM 是高性能 LLM 推理引擎，特点：
    - PagedAttention（高效 KV Cache 管理）
    - 连续批处理（高吞吐量）
    - OpenAI 兼容 API
    """
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
    return vLLMReplica


def _load_sglang():
    """
    延迟加载 SGLang Replica

    SGLang 是另一个 LLM 推理引擎，特点：
    - RadixAttention（前缀缓存优化）
    - 结构化生成支持
    - 与 vLLM 兼容的 API

    注：此加载函数包含一些兼容性处理代码（mock vllm 模块）
    """
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"

    # === vLLM 兼容性处理 ===
    # SGLang 可能依赖 vllm 的某些模块，如果没安装 vllm 则 mock 掉
    try:
        import vllm  # noqa: F401
    except ImportError:
        import sys
        import types
        from unittest.mock import Mock

        # 创建 mock 的 vllm 模块结构
        mock_vllm = types.ModuleType("vllm")

        mock_custom_ops = types.ModuleType("vllm._custom_ops")
        mock_custom_ops.scaled_fp8_quant = Mock()
        mock_vllm._custom_ops = mock_custom_ops

        mock_model_executor = types.ModuleType("vllm.model_executor")
        mock_layers = types.ModuleType("vllm.model_executor.layers")
        mock_activation = types.ModuleType("vllm.model_executor.layers.activation")

        class GeluAndMul:  # noqa: N801
            pass

        class SiluAndMul:  # noqa: N801
            pass

        mock_activation.GeluAndMul = GeluAndMul
        mock_activation.SiluAndMul = SiluAndMul
        mock_layers.activation = mock_activation
        mock_model_executor.layers = mock_layers
        mock_vllm.model_executor = mock_model_executor

        # 注册 mock 模块
        sys.modules["vllm"] = mock_vllm
        sys.modules["vllm._custom_ops"] = mock_custom_ops
        sys.modules["vllm.model_executor"] = mock_model_executor
        sys.modules["vllm.model_executor.layers"] = mock_layers
        sys.modules["vllm.model_executor.layers.activation"] = mock_activation

    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica

    del os.environ["SGLANG_USE_CPU_ENGINE"]
    return SGLangReplica


# === 注册内置引擎 ===
RolloutReplicaRegistry.register("vllm", _load_vllm)
RolloutReplicaRegistry.register("sglang", _load_sglang)


# === 便捷函数（推荐使用）===

def get_rollout_replica_class(rollout: str) -> type[RolloutReplica]:
    """
    获取 Rollout Replica 类 - ⭐ 推荐入口函数 ⭐

    这是获取 Replica 类的主要方式，支持 vLLM 和 SGLang 两种引擎。

    === 使用示例 ===
    ```python
    from verl.workers.rollout.replica import get_rollout_replica_class

    # 获取 vLLM Replica 类
    vllm_replica_class = get_rollout_replica_class("vllm")

    # 创建实例并初始化（Phase 0 示例）
    replica = vllm_replica_class(
        replica_rank=0,
        config=rollout_config,
        model_config=model_config,
        gpus_per_node=8,
    )
    await replica.init_standalone()

    # 获取服务地址
    print(f"Server ready at: {replica.server_address}")
    ```

    Args:
        rollout: 引擎名称，支持 "vllm" 或 "sglang"

    Returns:
        type[RolloutReplica]: Replica 类（vLLMReplica 或 SGLangReplica）

    Raises:
        ValueError: 如果 rollout 参数不是支持的引擎名称
    """
    return RolloutReplicaRegistry.get(rollout)
