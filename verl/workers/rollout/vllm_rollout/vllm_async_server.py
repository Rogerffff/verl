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
verl vLLM HTTP Server 模块

=== 模块概述 ===

本模块实现了 verl 框架中基于 vLLM 的分布式推理服务器。核心功能是：
1. 启动 vLLM 推理引擎
2. 提供 OpenAI 兼容的 HTTP API
3. 通过 ZeroMQ 与 GPU Workers 通信

=== 架构图 ===

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         vLLMReplica 架构                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   外部请求 (OpenAI API)                                                  │
│   POST http://ip:port/v1/chat/completions                               │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │           vLLMHttpServer (Ray Actor)                              │  │
│   │           - 运行在 CPU 上，不占用 GPU                             │  │
│   │           - 持有 vLLM AsyncLLM 引擎                              │  │
│   │           - 提供 /v1/chat/completions 等端点                     │  │
│   │                                                                   │  │
│   │   ┌────────────────────────────────────────────────────────┐     │  │
│   │   │     ExternalZeroMQDistributedExecutor                   │     │  │
│   │   │     - vLLM 的分布式执行后端                             │     │  │
│   │   │     - 通过 ZMQ 连接到各 Worker                          │     │  │
│   │   └────────────────────────────────────────────────────────┘     │  │
│   └──────────────────────────┬───────────────────────────────────────┘  │
│                              │                                           │
│                              │ ZMQ REQ-REP 模式                          │
│                              │ (请求-响应，同步通信)                     │
│                              │                                           │
│          ┌───────────────────┼───────────────────┐                      │
│          │                   │                   │                      │
│          ▼                   ▼                   ▼                      │
│   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐          │
│   │   Worker 0      │ │   Worker 1      │ │   Worker N      │          │
│   │   (Ray Actor)   │ │   (Ray Actor)   │ │   (Ray Actor)   │          │
│   │   GPU 0         │ │   GPU 1         │ │   GPU N         │          │
│   │   ZMQ :5555     │ │   ZMQ :5556     │ │   ZMQ :555N     │          │
│   │                 │ │                 │ │                 │          │
│   │   模型分片 0    │ │   模型分片 1    │ │   模型分片 N    │          │
│   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘          │
│            │                   │                   │                    │
│            └───────────────────┼───────────────────┘                    │
│                                │                                         │
│                         NCCL 集合通信                                    │
│                    （跨 GPU 同步张量数据）                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

=== 核心类说明 ===

1. **vLLMReplica** (继承自 RolloutReplica)
   - 管理一个完整的推理服务器实例
   - 负责创建 Workers 和启动 HTTP Server
   - 对外提供 server_address 属性

2. **vLLMHttpServer** (Ray Actor)
   - 运行在单个节点上的 HTTP 服务器
   - 接收 OpenAI API 请求
   - 持有 vLLM 的 AsyncLLM 引擎

3. **ExternalZeroMQDistributedExecutor**
   - vLLM 的分布式执行器
   - 通过 ZMQ 与 Workers 通信
   - 将推理请求分发到各 GPU

=== 使用示例 ===

```python
from verl.workers.rollout.replica import get_rollout_replica_class

# 1. 获取 vLLMReplica 类
replica_class = get_rollout_replica_class("vllm")

# 2. 创建实例
replica = replica_class(
    replica_rank=0,
    config=rollout_config,      # 推理配置
    model_config=model_config,  # 模型配置
    gpus_per_node=8,
)

# 3. 初始化（选择 standalone 模式用于 Phase 0 评测）
await replica.init_standalone()

# 4. 获取服务地址
server_address = replica.server_address  # 如 "192.168.1.1:8000"

# 5. 通过 OpenAI API 调用
import aiohttp
async with aiohttp.ClientSession() as session:
    resp = await session.post(
        f"http://{server_address}/v1/chat/completions",
        json={
            "model": "your-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
            "max_tokens": 100,
        }
    )
    result = await resp.json()
    print(result["choices"][0]["message"]["content"])
```

=== ZeroMQ 通信原理 ===

为什么用 ZMQ 而不是 Ray 远程调用？
- **性能**: ZMQ 是高性能消息队列，延迟极低（微秒级）
- **集成**: vLLM 内部使用 ZMQ 进行分布式通信
- **灵活**: 支持多种通信模式，这里用 REQ-REP（请求-响应）

通信流程:
1. Worker 启动时创建 ZMQ socket 监听端口
2. HTTP Server 启动时获取所有 Workers 的 ZMQ 地址
3. Server 收到推理请求 → 通过 ZMQ 发送给 Workers
4. Workers 执行推理 → 返回结果给 Server
5. Server 组装响应 → 返回 HTTP 响应
"""
import argparse
import asyncio
import inspect
import json
import logging
import os
from concurrent.futures import Future
from pprint import pprint
from typing import Any, Callable, Optional

import cloudpickle as pickle
import numpy as np
import ray
import vllm.entrypoints.cli.serve
import zmq
from packaging import version
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_app,
    init_app_state,
)
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager
from vllm.v1.executor.abstract import Executor

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import (
    get_free_port,
    get_max_position_embeddings,
    is_valid_ipv6_address,
    run_unvicorn,
)
from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

# === vLLM 版本兼容处理 ===
# 不同版本的 vLLM API 可能有差异，这里做兼容
_VLLM_VERSION = version.parse(vllm.__version__)

if _VLLM_VERSION > version.parse("0.11.0"):
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.utils.network_utils import get_tcp_uri

    if _VLLM_VERSION == version.parse("0.12.0"):
        from vllm.entrypoints.harmony_utils import get_encoding

        get_encoding()
else:
    from vllm.utils import FlexibleArgumentParser, get_tcp_uri
if _VLLM_VERSION >= version.parse("0.12.0"):
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# ============================================================================
# ExternalZeroMQDistributedExecutor - ZMQ 分布式执行器
# ============================================================================

class ExternalZeroMQDistributedExecutor(Executor):
    """
    外部 ZeroMQ 分布式执行器 - verl 的核心通信组件

    === 什么是 Executor？===

    Executor 是 vLLM 中负责分布式执行的抽象类。默认 vLLM 使用 Ray 来管理 Workers，
    但 verl 有自己的 Worker 管理方式（通过 RayWorkerGroup），所以需要自定义 Executor。

    这个类的作用是：让 vLLM 引擎通过 ZeroMQ 与 verl 管理的 Workers 通信。

    === 为什么用 ZeroMQ？===

    1. **高性能**: ZMQ 是极低延迟的消息队列（微秒级）
    2. **与 vLLM 集成**: vLLM 内部就用 ZMQ 做分布式通信
    3. **解耦**: HTTP Server 和 Workers 可以在不同进程

    === 通信模式 ===

    使用 REQ-REP（请求-响应）模式：
    ```
    HTTP Server (REQ)  →  "execute_model"  →  Worker (REP)
                       ←   result          ←
    ```

    === 架构示意 ===

    ```
    ┌───────────────────────────────────────────────────────────────┐
    │              vLLMHttpServer                                    │
    │                                                                │
    │   ┌────────────────────────────────────────────────────────┐  │
    │   │  ExternalZeroMQDistributedExecutor                      │  │
    │   │                                                         │  │
    │   │  self.sockets = [                                       │  │
    │   │      zmq.REQ → tcp://192.168.1.1:5555  (连接 Worker 0) │  │
    │   │      zmq.REQ → tcp://192.168.1.1:5556  (连接 Worker 1) │  │
    │   │  ]                                                      │  │
    │   │                                                         │  │
    │   │  collective_rpc("execute_model", args=(...))           │  │
    │   │  → 广播到所有 Workers，收集结果                        │  │
    │   └────────────────────────────────────────────────────────┘  │
    │                                                                │
    └───────────────────────────────────────────────────────────────┘
                                    │
                     ZMQ (tcp://...)│
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │   Worker 0    │       │   Worker 1    │       │   Worker N    │
    │   ZMQ REP     │       │   ZMQ REP     │       │   ZMQ REP     │
    │   :5555       │       │   :5556       │       │   :555N       │
    │   GPU 0       │       │   GPU 1       │       │   GPU N       │
    └───────────────┘       └───────────────┘       └───────────────┘
    ```
    """

    # 标记不使用 Ray 管理（我们自己管理）
    uses_ray: bool = False

    def _init_executor(self) -> None:
        """
        初始化执行器 - 建立与所有 Workers 的 ZMQ 连接

        执行流程：
        1. 从环境变量获取所有 Workers 的 ZMQ 地址
        2. 为每个 Worker 创建一个 ZMQ REQ socket
        3. 通过 collective_rpc 初始化所有 Workers
        """
        # === 获取并行配置 ===
        # dp_rank_local: 当前节点的数据并行 rank
        # tp_size: 张量并行大小（模型分片数）
        dp_rank_local = self.vllm_config.parallel_config.data_parallel_rank_local
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size

        # === 获取 Workers 的 ZMQ 地址 ===
        # 地址格式: "tcp://192.168.1.1:5555,tcp://192.168.1.1:5556,..."
        # 由 HTTP Server 在启动时设置到环境变量
        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        # 根据数据并行 rank 选择对应的 Workers
        # 例如: TP=2, DP=2 时，rank0 用 workers[0:2], rank1 用 workers[2:4]
        addresses = addresses[dp_rank_local * tp_size : (dp_rank_local + 1) * tp_size]

        # === 创建 ZMQ 上下文和 sockets ===
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            # 创建 REQ（请求）类型的 socket
            socket = self.context.socket(zmq.REQ)
            # 处理 IPv6 地址
            if address.startswith("tcp://["):
                socket.setsockopt(zmq.IPV6, 1)
            # 连接到 Worker
            socket.connect(address)
            self.sockets.append(socket)

        # === 初始化所有 Workers ===
        # 通过 RPC 调用 Worker 的初始化方法
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        # 1. 初始化 Worker
        self.collective_rpc("init_worker", args=([kwargs],))
        # 2. 初始化设备（CUDA）
        self.collective_rpc("init_device")
        # 3. 加载模型
        self.collective_rpc("load_model")

    # === vLLM 0.12+ 的新接口 ===
    if _VLLM_VERSION >= version.parse("0.12.0"):

        def execute_model(
            self, scheduler_output: "SchedulerOutput", non_block: bool = False
        ) -> "ModelRunnerOutput | None | Future[ModelRunnerOutput | None]":
            """
            执行模型推理 - 核心方法

            这是 vLLM 引擎调用的主要方法，用于执行一个批次的推理。

            流程：
            1. 将 scheduler_output（包含要处理的请求）发送给所有 Workers
            2. Workers 在各自 GPU 上执行推理
            3. 收集并返回结果

            Args:
                scheduler_output: 调度器输出，包含本批次要处理的请求信息
                non_block: 是否非阻塞模式（返回 Future）

            Returns:
                ModelRunnerOutput: 推理结果（token 概率等）
            """
            output = self.collective_rpc("execute_model", args=(scheduler_output,))
            result = output[0]  # 取第一个 Worker 的结果（其他 Worker 结果相同）
            if non_block:
                # 非阻塞模式：包装成 Future
                f = Future()
                f.set_result(result)
                return f
            return result

        def sample_tokens(
            self, grammar_output: "GrammarOutput | None", non_block: bool = False
        ) -> "ModelRunnerOutput | None | Future[ModelRunnerOutput | None]":
            """
            采样 tokens - 从概率分布中采样生成 token

            Args:
                grammar_output: 语法约束输出（用于 constrained decoding）
                non_block: 是否非阻塞模式

            Returns:
                ModelRunnerOutput: 采样结果
            """
            output = self.collective_rpc("sample_tokens", args=(grammar_output,))
            result = output[0]
            if non_block:
                f = Future()
                f.set_result(result)
                return f
            return result

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        **kwargs_extra: Any,
    ) -> list[Any]:
        """
        集体 RPC 调用 - 向所有 Workers 广播调用请求

        这是 ZMQ 通信的核心方法。它会：
        1. 将方法名和参数序列化
        2. 发送给所有 Workers
        3. 等待所有 Workers 返回结果
        4. 收集并返回结果列表

        === 通信流程图 ===

        ```
        collective_rpc("execute_model", args=(data,))
                │
                ▼
        ┌───────────────────────────────────────────────┐
        │  1. 序列化: pickle.dumps((method, args, {})) │
        │  2. 广播: socket.send(message) 到所有 Worker │
        │  3. 等待: socket.recv() 收集所有结果        │
        └───────────────────────────────────────────────┘
                │
        ┌───────┴───────┬───────────────┐
        │               │               │
        ▼               ▼               ▼
    Worker 0        Worker 1        Worker N
    execute_model() execute_model() execute_model()
        │               │               │
        └───────────────┴───────────────┘
                        │
                        ▼
                [result0, result1, ..., resultN]
        ```

        Args:
            method: 要调用的方法名（字符串）或 callable
            timeout: 超时时间（秒），None 表示无超时
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            list[Any]: 每个 Worker 返回的结果列表

        Raises:
            Exception: 如果任何 Worker 返回异常
        """
        # 处理方法名：字符串直接用，callable 需要序列化
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        # 序列化整个消息
        message = pickle.dumps((sent_method, args, kwargs or {}))

        # 非阻塞发送给所有 Workers
        # zmq.DONTWAIT: 不等待，立即返回
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        # 收集所有 Workers 的响应
        outputs = []
        for socket in self.sockets:
            # recv() 会阻塞直到收到响应
            outputs.append(pickle.loads(socket.recv()))

        # 检查是否有异常
        for output in outputs:
            if isinstance(output, Exception):
                raise output

        return outputs

    def check_health(self):
        """健康检查 - 目前为空实现"""
        return


# ============================================================================
# vLLMHttpServerBase - HTTP 服务器基类
# ============================================================================

class vLLMHttpServerBase:
    """
    vLLM HTTP 服务器基类

    === 功能说明 ===

    这个类实现了单节点的 vLLM HTTP 服务器，功能等价于命令行：
    ```bash
    vllm serve /path/to/model --tensor-parallel-size=8 --port=8000
    ```

    但与命令行方式不同的是，这里的 Workers 是由 verl 的 RayWorkerGroup 管理的，
    而不是 vLLM 自己启动的。

    === 核心组件 ===

    ```
    vLLMHttpServerBase
    ├── config: RolloutConfig        # 推理配置（温度、top_p 等）
    ├── model_config: HFModelConfig  # 模型配置（路径、dtype 等）
    ├── workers: list[ActorHandle]   # Worker Actor 句柄列表
    ├── engine: AsyncLLM             # vLLM 异步推理引擎
    └── _server_port: int            # HTTP 服务端口
    ```

    === 关键方法 ===

    - launch_server(): 启动 HTTP 服务器
    - generate(): Token-in-Token-out 生成（直接调用，不走 HTTP）
    - wake_up()/sleep(): 控制服务器休眠/唤醒
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        """
        初始化 HTTP 服务器

        Args:
            config: 推理配置，包含：
                - temperature: 采样温度
                - top_p: nucleus sampling 参数
                - max_model_len: 最大上下文长度
                - tensor_model_parallel_size: TP 大小
                - load_format: 权重加载方式（"auto" 或 "dummy"）
                  ⚠️ Phase 0 必须用 "auto"！

            model_config: 模型配置，包含：
                - local_path: 模型路径
                - trust_remote_code: 是否信任远程代码
                - hf_config: HuggingFace 配置

            rollout_mode: 部署模式
                - STANDALONE: Phase 0 评测用
                - HYBRID: GRPO/PPO 训练用
                - COLOCATED: GRM 用

            workers: Worker Actor 句柄列表
                - 每个 Worker 管理一个 GPU
                - 通过 ZMQ 与本服务器通信

            replica_rank: Replica 序号（0, 1, 2, ...）
                - 用于日志和命名

            node_rank: 节点序号（多节点场景）
                - 0 = Master 节点（提供 HTTP API）
                - 其他 = Headless 节点（参与推理但不提供 API）

            gpus_per_node: 每节点 GPU 数量

            nnodes: 节点总数
        """
        super().__init__()

        # === 配置处理 ===
        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        # 从 HF config 获取最大位置编码长度
        self.config.max_model_len = get_max_position_embeddings(self.model_config.hf_config)
        self.rollout_mode = rollout_mode
        self.workers = workers

        # === 节点信息 ===
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        # === 关键：load_format 自动修正 ===
        # 非 HYBRID 模式下，如果 load_format 是 "dummy"，自动改为 "auto"
        # 因为 "dummy" 不会加载真实权重，只有 HYBRID 模式（训练时同步权重）才能用
        # ⚠️ 这是防止 Phase 0 配置错误的安全措施！
        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # === HTTP 服务器配置 ===
        # 获取本机 IP 地址
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None  # 启动后填充

        # === 数据并行配置（多节点场景）===
        # node_rank=0 是 Master，负责提供 HTTP API 和协调其他节点
        if self.node_rank == 0:
            self._master_address = self._server_address
            # 获取两个空闲端口：master_port 和 dp_master_port
            self._master_port, self._master_sock = get_free_port(self._server_address)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address)
            logger.info(
                f"vLLMHttpServer, replica_rank: {self.replica_rank}, master address: {self._master_address}, "
                f"master port: {self._master_port}, data parallel master port: {self._dp_master_port}"
            )
        else:
            # 非 Master 节点，等待从 Master 获取地址
            self._master_address = None
            self._master_port = None

    def get_master_address(self):
        """
        获取 Master 地址和端口

        用于数据并行场景，非 Master 节点需要知道 Master 的地址。

        Returns:
            tuple[str, int]: (master_address, master_port)
        """
        return self._master_address, self._master_port

    def get_server_address(self):
        """
        获取 HTTP 服务器地址和端口

        Returns:
            tuple[str, int]: (server_address, server_port)

        Raises:
            AssertionError: 如果服务器未启动
        """
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        """
        启动 HTTP 服务器 - 核心方法

        这个方法会：
        1. 配置 vLLM 引擎参数
        2. 设置 ZMQ 通信
        3. 启动 Uvicorn HTTP 服务器

        === 执行流程 ===

        ```
        launch_server()
            │
            ├── 1. 配置 vLLM CLI 参数
            │       - dtype, load_format, tensor_parallel_size
            │       - temperature, top_p, max_tokens
            │
            ├── 2. 设置分布式执行器
            │       - 获取 Workers 的 ZMQ 地址
            │       - 设置 VERL_VLLM_ZMQ_ADDRESSES 环境变量
            │
            └── 3. 启动服务器
                    - Master 节点: run_server() → Uvicorn HTTP
                    - 其他节点: run_headless() → 仅参与推理
        ```

        Args:
            master_address: Master 节点地址（非 Master 节点需要提供）
            master_port: Master 节点端口
        """
        # 非 Master 节点需要知道 Master 地址
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        # ====================================================================
        # Step 1: 配置 vLLM CLI 参数
        # ====================================================================

        # 获取额外的引擎参数
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}

        # 多图像支持
        if self.config.get("limit_images", None):
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}

        # CUDA Graph 优化
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        # 默认生成配置（可被每个请求覆盖）
        override_generation_config = dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )
        logger.info(f"override_generation_config: {override_generation_config}")

        # 休眠模式配置
        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments
            set_expandable_segments(True)

        # === 量化配置 ===
        quantization = self.config.quantization
        if quantization is not None:
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao"]
            if quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {quantization}")

            if quantization == "fp8":
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
                apply_vllm_fp8_patches()

        hf_overrides = {}
        if quantization is not None and self.config.quantization_config_file is not None:
            hf_overrides["quantization_config_file"] = self.config.quantization_config_file
        if quantization == "fp8":
            hf_overrides["quantization_config"] = fp8_block_quant_kwargs

        # === 组装参数字典 ===
        args = {
            "dtype": self.config.dtype,               # 数据类型（float16, bfloat16）
            "load_format": self.config.load_format,   # ⚠️ 关键！"auto" 或 "dummy"
            "skip_tokenizer_init": False,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,  # 最大上下文长度
            "max_num_seqs": self.config.max_num_seqs,    # 最大并发序列数
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "disable_custom_all_reduce": True,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,  # GPU 显存利用率
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,  # TP 大小
            "seed": self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            **engine_kwargs,
        }

        # Prometheus 监控配置
        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        # === 专家并行配置（MoE 模型）===
        if self.config.expert_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to dp_size_local "
            )
            f"({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"

            args.update(
                {
                    "enable_expert_parallel": self.config.expert_parallel_size > 1,
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": data_parallel_size_local,
                    "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._master_port,
                }
            )

        # === LoRA 配置 ===
        if self.model_config.lora_rank > 0:
            args.update(
                {
                    "enable_lora": True,
                    "max_loras": 1,
                    "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank),
                }
            )

        # MoE 路由重放
        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        # === 构建命令行参数 ===
        # 格式: ["serve", "/path/to/model", "--dtype", "bfloat16", ...]
        server_args = ["serve", self.model_config.local_path]
        for k, v in args.items():
            if isinstance(v, bool):
                if v:
                    server_args.append(f"--{k}")
            elif v is not None:
                server_args.append(f"--{k}")
                server_args.append(json.dumps(v) if isinstance(v, dict) else str(v))

        # 打印参数（仅 Replica 0 打印，避免日志刷屏）
        if self.replica_rank == 0:
            pprint(server_args)

        # 解析命令行参数
        CMD_MODULES = [vllm.entrypoints.cli.serve]
        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # ====================================================================
        # Step 2: 设置分布式执行器
        # ====================================================================

        # 如果有 Workers，使用 ZMQ 执行器；否则为 None（单进程模式）
        distributed_executor_backend = ExternalZeroMQDistributedExecutor if len(self.workers) > 0 else None
        server_args.distributed_executor_backend = distributed_executor_backend

        # 获取所有 Workers 的 ZMQ 地址
        # 例如: ["tcp://192.168.1.1:5555", "tcp://192.168.1.1:5556"]
        zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in self.workers])
        logger.info(
            f"replica_rank={self.replica_rank}, node_rank={self.node_rank}, nnodes={self.nnodes}, "
            f"get worker zmq addresses: {zmq_addresses}"
        )
        # 设置环境变量，供 ExternalZeroMQDistributedExecutor 使用
        os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        # ====================================================================
        # Step 3: 启动服务器
        # ====================================================================

        if self.node_rank == 0:
            # Master 节点：启动完整的 HTTP 服务器
            await self.run_server(server_args)
        else:
            # 其他节点：无头模式，仅参与推理
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        """
        运行 HTTP 服务器（Master 节点）

        这个方法会：
        1. 创建 vLLM 异步引擎
        2. 构建 FastAPI 应用
        3. 启动 Uvicorn HTTP 服务器

        Args:
            args: 解析后的命令行参数
        """
        # === 创建 vLLM 配置 ===
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        # === 创建 vLLM 异步引擎 ===
        # AsyncLLM 是 vLLM 的异步推理引擎，支持高并发
        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # 清理多模态缓存
        await engine_client.reset_mm_cache()

        # === 构建 FastAPI 应用 ===
        # build_app() 创建包含 /v1/chat/completions 等端点的 FastAPI 应用
        app = build_app(args)
        if _VLLM_VERSION > version.parse("0.11.0"):
            await init_app_state(engine_client, app.state, args)
        else:
            await init_app_state(engine_client, vllm_config, app.state, args)

        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        # 保存引擎引用
        self.engine = engine_client

        # === 启动 Uvicorn HTTP 服务器 ===
        # run_unvicorn() 返回 (port, task)
        self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """
        运行无头模式（非 Master 节点）

        无头模式下，节点参与分布式推理但不提供 HTTP API。
        这用于大模型跨多节点部署的场景。

        Args:
            args: 解析后的命令行参数
        """
        # 创建引擎配置（headless=True）
        engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context, headless=True)

        parallel_config = vllm_config.parallel_config
        local_engine_count = parallel_config.data_parallel_size_local

        host = parallel_config.data_parallel_master_ip
        port = engine_args.data_parallel_rpc_port
        handshake_address = get_tcp_uri(host, port)

        # 创建引擎管理器
        self.engine_manager = CoreEngineProcManager(
            target_fn=EngineCoreProc.run_engine_core,
            local_engine_count=local_engine_count,
            start_index=vllm_config.parallel_config.data_parallel_rank,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=False,
            handshake_address=handshake_address,
            executor_class=Executor.get_class(vllm_config),
            log_stats=not engine_args.disable_log_stats,
        )

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """
        Token-in-Token-out 生成 - 直接调用引擎（不走 HTTP）

        这个方法用于直接调用 vLLM 引擎生成文本，返回 token 级别的结果。
        与 HTTP API 不同，这里输入输出都是 token IDs。

        === 使用场景 ===

        - GRPO/PPO 训练中需要获取 log_probs
        - 需要精细控制生成过程
        - 批量生成时减少 HTTP 开销

        === 示例 ===

        ```python
        # 调用 generate（通过 Ray）
        result = await server_handle.generate.remote(
            prompt_ids=[1, 2, 3, 4, 5],  # tokenized prompt
            sampling_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100,
                "logprobs": True,  # 返回 log 概率
            },
            request_id="req_001",
        )

        # result 是 TokenOutput
        print(result.token_ids)   # [6, 7, 8, ...]  生成的 tokens
        print(result.log_probs)   # [-1.2, -0.8, ...]  每个 token 的 log 概率
        print(result.stop_reason) # "completed" 或 "aborted"
        ```

        Args:
            prompt_ids: 输入 prompt 的 token IDs（已 tokenize）
            sampling_params: 采样参数字典
                - temperature: 采样温度
                - top_p: nucleus sampling
                - max_tokens / max_new_tokens: 最大生成 token 数
                - logprobs: 是否返回 log 概率
            request_id: 请求 ID（用于追踪和取消）
            image_data: 图像数据（多模态模型）
            video_data: 视频数据（多模态模型）

        Returns:
            TokenOutput: 包含生成的 token IDs 和元信息
        """
        # === 计算可用 token 空间 ===
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        # === 确定 max_tokens ===
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            # 兼容 sglang 风格的参数名
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            # 默认值
            max_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)

        # 限制在有效范围内
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        assert max_tokens <= max_possible_tokens, (
            f"max_tokens {max_tokens} exceeds available context space {max_possible_tokens}"
        )

        # === 处理 logprobs 参数 ===
        # vLLM 的 logprobs 参数是 int（返回 top-k 的 logprobs）或 None
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))

        # 创建 vLLM SamplingParams
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)

        # Qwen2.5-VL 特殊处理：去重图像 tokens
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)

        # === 构建多模态输入 ===
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        # === LoRA 处理 ===
        lora_request = None
        if self.model_config.lora_rank > 0:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        # === 调用 vLLM 引擎生成 ===
        generator = self.engine.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
        )

        # 获取最终结果（遍历所有中间输出）
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        # === 提取结果 ===
        token_ids = final_res.outputs[0].token_ids

        # log_probs
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]

        # MoE 路由信息
        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            routed_experts = final_res.outputs[0].routed_experts

        # 停止原因
        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason

        return TokenOutput(
            token_ids=token_ids, log_probs=log_probs, routed_experts=routed_experts, stop_reason=stop_reason
        )

    async def wake_up(self):
        """
        唤醒服务器 - 从休眠状态恢复

        根据不同的 rollout_mode 有不同的行为：
        - HYBRID: 调用所有 Workers 的 wake_up（切换到推理模式）
        - COLOCATED: 调用引擎的 wake_up
        - STANDALONE: 无操作（Standalone 模式不休眠）
        """
        if self.rollout_mode == RolloutMode.HYBRID:
            # HYBRID 模式：Workers 需要在训练模式和推理模式之间切换
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # COLOCATED 模式：直接调用引擎
            if self.node_rank == 0:
                await self.engine.wake_up(tags=["kv_cache", "weights"])
        elif self.rollout_mode == RolloutMode.STANDALONE:
            # STANDALONE 模式：不需要休眠/唤醒
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        """
        让服务器进入休眠状态

        休眠可以释放 GPU 显存，用于训练阶段。

        不同模式的行为：
        - HYBRID: 重置缓存，调用 Workers 休眠
        - COLOCATED: 重置缓存，引擎休眠
        - STANDALONE: 无操作
        """
        if self.rollout_mode == RolloutMode.HYBRID:
            if self.node_rank == 0:
                await self.engine.reset_prefix_cache()
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            if self.node_rank == 0:
                await self.engine.reset_prefix_cache()
                await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def clear_kv_cache(self):
        """
        清空 KV Cache

        KV Cache 是 Transformer 推理时缓存的 Key/Value 张量，
        清空可以释放显存，但会降低后续推理速度。
        """
        if self.node_rank == 0:
            await self.engine.reset_prefix_cache()

    async def wait_for_requests_to_drain(self):
        """
        等待所有请求处理完成

        用于安全地关闭服务器或进入休眠状态。
        """
        await self.engine.wait_for_requests_to_drain()

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """
        中止所有正在进行的生成请求

        用于紧急停止或超时处理。

        Args:
            reset_prefix_cache: 是否重置前缀缓存

        Returns:
            dict: {
                "aborted_count": 中止的请求数,
                "request_ids": 中止的请求 ID 列表
            }
        """
        try:
            # 原子快照，避免与引擎线程竞争
            request_states_snapshot = list(self.engine.output_processor.request_states.items())
            request_ids = [req_id for req_id, _ in request_states_snapshot]

            if not request_ids:
                return {"aborted_count": 0, "request_ids": []}

            from vllm.v1.engine import FinishReason

            # 为每个请求创建中止输出
            for _, req_state in request_states_snapshot:
                request_output = req_state.make_request_output(
                    [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
                )
                req_state.queue.put(request_output)

            # 在处理器和引擎核心中中止请求
            self.engine.output_processor.abort_requests(request_ids)
            await self.engine.engine_core.abort_requests_async(request_ids)

            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info("Prefix cache reset after abort")

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """
        中止特定请求

        Args:
            request_id: 要中止的请求 ID
            reset_prefix_cache: 是否重置前缀缓存

        Returns:
            dict: {"aborted": True/False, "request_id": ...}
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}


# ============================================================================
# vLLMHttpServer - Ray Actor 包装器
# ============================================================================

@ray.remote(num_cpus=1)
class vLLMHttpServer(vLLMHttpServerBase):
    """
    vLLM HTTP 服务器 Ray Actor

    这是 vLLMHttpServerBase 的 Ray Actor 包装。通过 @ray.remote 装饰后，
    这个类的实例会运行在独立的 Ray Actor 进程中。

    === 为什么需要 Ray Actor？===

    1. **进程隔离**: HTTP 服务器运行在独立进程，不影响其他组件
    2. **资源控制**: 可以指定 CPU 数量（num_cpus=1）
    3. **远程调用**: 可以通过 Ray 远程调用方法

    === 使用方式 ===

    ```python
    # 创建 Actor（注意 .remote()）
    server = vLLMHttpServer.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=..., soft=False),
        name="vllm_server_0_0",
    ).remote(
        config=config,
        model_config=model_config,
        rollout_mode=RolloutMode.STANDALONE,
        workers=workers,
        replica_rank=0,
        node_rank=0,
        gpus_per_node=8,
        nnodes=1,
    )

    # 调用方法（注意 .remote()）
    await server.launch_server.remote()
    address, port = await server.get_server_address.remote()
    ```

    注：num_cpus=1 表示这个 Actor 只占用 1 个 CPU，不占用 GPU。
    GPU 由 Workers 占用。
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        super().__init__(config, model_config, rollout_mode, workers, replica_rank, node_rank, gpus_per_node, nnodes)


# Worker Actor 类（带 @ray.remote 装饰）
# vLLMAsyncRollout 是实际在 GPU 上运行的 Worker
_rollout_worker_actor_cls = ray.remote(vLLMAsyncRollout)


# ============================================================================
# vLLMReplica - Replica 实现
# ============================================================================

class vLLMReplica(RolloutReplica):
    """
    vLLM Replica 实现 - 管理完整的推理服务器

    === 什么是 Replica？===

    Replica 是 verl 中管理推理服务器的核心抽象。一个 Replica 包含：
    - 一组 Workers（每个 Worker 管理一个 GPU）
    - 一个或多个 HTTP Servers（每个节点一个）

    例如，TP=2 的单节点场景：
    ```
    vLLMReplica
    ├── workers: [Worker0(GPU0), Worker1(GPU1)]
    └── servers: [vLLMHttpServer]
                      │
                      └── HTTP :8000
    ```

    === Phase 0 使用方式 ===

    ```python
    from verl.workers.rollout.replica import get_rollout_replica_class

    # 1. 获取类
    replica_class = get_rollout_replica_class("vllm")

    # 2. 创建实例
    replica = replica_class(
        replica_rank=0,
        config=rollout_config,
        model_config=model_config,
        gpus_per_node=8,
    )

    # 3. 初始化（Standalone 模式）
    await replica.init_standalone()

    # 4. 使用
    address = replica.server_address  # "192.168.1.1:8000"
    ```
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        """
        初始化 vLLM Replica

        Args:
            replica_rank: Replica 序号（0, 1, 2, ...）
            config: 推理配置
            model_config: 模型配置
            gpus_per_node: 每节点 GPU 数量
            is_reward_model: 是否是奖励模型（用于 GRM）
        """
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        # 指定 HTTP Server 类
        self.server_class = vLLMHttpServer

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """
        获取 Worker Actor 类和初始化参数

        这个方法被 init_standalone() 和 init_colocated() 调用，
        用于创建 Workers。

        Returns:
            RayClassWithInitArgs: 包含 Worker 类和初始化参数
        """
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,  # vLLMAsyncRollout（已装饰 @ray.remote）
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """
        启动 HTTP 服务器 - vLLMReplica 的核心方法

        这个方法在 init_standalone()/init_hybrid()/init_colocated() 中被调用，
        负责在每个节点上启动 vLLMHttpServer。

        === 执行流程 ===

        ```
        launch_servers()
            │
            ├── 1. 验证 Workers 数量
            │
            ├── 2. 获取每个 Worker 的节点 ID
            │       （确定 Worker 在哪台物理机）
            │
            ├── 3. 为每个节点创建 vLLMHttpServer Actor
            │       - 使用 NodeAffinitySchedulingStrategy
            │       - 保证 Server 和 Workers 在同一节点
            │
            ├── 4. 启动所有服务器
            │       - 获取 Master 地址
            │       - 并发启动所有服务器
            │
            └── 5. 保存服务器地址
                    - self._server_address = "ip:port"
        ```

        === 单节点 vs 多节点 ===

        单节点（常见）:
        - 只有 1 个 HTTP Server
        - 所有 Workers 在同一台机器

        多节点（大模型）:
        - 每个节点 1 个 Server
        - Node 0 是 Master（提供 HTTP API）
        - 其他节点是 Headless（参与推理）
        """
        # === Step 1: 验证 Workers 数量 ===
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # === Step 2: 获取每个 Worker 的节点 ID ===
        # 这是为了确保 HTTP Server 和它的 Workers 在同一个节点上
        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.workers
            ]
        )

        # === 处理数据并行配置 ===
        # 非数据并行场景：只有一个 Server
        nnodes, gpus_per_node = self.nnodes, self.gpus_per_node
        if self.config.data_parallel_size == 1:
            nnodes = 1
            gpus_per_node = self.world_size

        # === Step 3: 为每个节点创建 vLLMHttpServer Actor ===
        for node_rank in range(nnodes):
            # 获取该节点上的 Workers
            workers = self.workers[node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node]
            node_id = worker_node_ids[node_rank * gpus_per_node]

            # 命名（方便调试）
            name = (
                f"vllm_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"vllm_server_reward_{self.replica_rank}_{node_rank}"
            )

            # 创建 Server Actor
            # NodeAffinitySchedulingStrategy: 强制调度到指定节点
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,  # 硬约束：必须在该节点
                ),
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            self.servers.append(server)

        # === Step 4: 启动所有服务器 ===
        # 获取 Master 地址
        master_address, master_port = await self.servers[0].get_master_address.remote()
        # 并发启动所有服务器
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # === Step 5: 保存服务器地址 ===
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        # 处理 IPv6 地址格式
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """
        让所有服务器进入休眠状态

        先等待请求处理完成，再让服务器休眠。
        """
        # 先等待请求处理完成
        await self.servers[0].wait_for_requests_to_drain.remote()
        # 再让所有服务器休眠
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self) -> dict[str, Any]:
        """
        中止所有服务器上的所有请求

        Returns:
            dict: 汇总的中止结果
        """
        results = await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """
        中止特定请求

        由于不知道请求在哪个服务器上，会尝试所有服务器。

        Args:
            request_id: 请求 ID

        Returns:
            dict: 中止结果
        """
        # TODO(petersh6): 应该只在拥有该请求的服务器上中止
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}


# ============================================================================
# 辅助函数
# ============================================================================

def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """
    Qwen2.5-VL 图像 token 去重

    Qwen2.5-VL 模型的特殊处理：vLLM 会根据 image_data 自动复制 <|image_pad|> token，
    所以输入时需要将连续的图像 token 去重。

    === 示例 ===

    原始 token 序列:
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    ```

    去重后:
    ```
    <|vision_start|><|image_pad|><|vision_end|>
    ```

    Args:
        prompt_ids: token ID 列表
        processor: HuggingFace processor（用于获取特殊 token ID）

    Returns:
        list[int]: 去重后的 token ID 列表
    """
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)

        # 创建 mask，True 表示保留
        mask = np.ones(len(prompt_ids), dtype=bool)

        # 找出图像/视频 token
        is_value = (prompt_ids == processor.image_token_id) | (prompt_ids == processor.video_token_id)

        # 去除连续重复：如果当前和前一个都是图像 token，则去除当前
        mask[1:] &= ~(is_value[1:] & is_value[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids
