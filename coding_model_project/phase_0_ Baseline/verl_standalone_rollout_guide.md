# verl Standalone Rollout 模式详解

> 本文档为 Phase 0 Baseline 评测提供 verl 框架的深度技术讲解，帮助新手理解 Standalone Rollout 模式的工作原理和代码执行流程。
>
> **📚 相关文档**：
> - [phase0_implementation_plan.md](./phase0_implementation_plan.md) — Phase 0 整体实施计划、评测脚本和指标收集
> - [data_governance_guide.md](./data_governance_guide.md) — 数据治理详细指南
> - [metrics_collection_spec.md](./metrics_collection_spec.md) — 指标收集规范

---

## 一、概述

### 1.1 什么是 Standalone Rollout 模式

**Rollout** 在 verl 框架中指的是使用 LLM 推理引擎（vLLM 或 SGLang）生成文本序列的过程。verl 提供三种 Rollout 部署模式，定义在 `verl/verl/workers/rollout/replica.py`:

```python
class RolloutMode(Enum):
    # Rollout 与训练引擎融合在同一进程
    HYBRID = "hybrid"

    # Rollout 与训练引擎共享 GPU，独立进程
    COLOCATED = "colocated"

    # 独立 GPU 资源，disaggregated 架构
    STANDALONE = "standalone"
```

**Standalone 模式**的核心特征：
- **独立 GPU 资源**：为推理分配专用 GPU，不与训练共享
- **无权重同步**：模型权重从磁盘一次性加载，运行期间保持静态
- **HTTP API**：提供 OpenAI 兼容的 REST API 接口

### 1.2 为什么 Phase 0 使用 Standalone 模式

Phase 0 是纯评测阶段，**不涉及任何训练**，其目标是：
- 在未训练的 Base 模型上建立性能基线
- 验证评测流水线（SandboxFusion 判题）正常工作
- 收集质量、成本、错误分布等指标

**Standalone 模式完美契合这些需求**：

| 特性 | Standalone | Hybrid | 说明 |
|------|------------|--------|------|
| 需要训练引擎 | ❌ 不需要 | ✅ 需要 | Phase 0 无训练 |
| 权重同步 | ❌ 不需要 | ✅ 需要 | 模型静态不变 |
| GPU 独占推理 | ✅ 最大化利用 | ⚠️ 需与训练共享 | 评测吞吐量更高 |
| 实现复杂度 | ⭐ 低 | ⭐⭐⭐ 高 | 更容易调试 |

### 1.3 三种模式对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         verl Rollout 模式对比                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   HYBRID (混合模式)                                                          │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │  同一进程                                                     │          │
│   │  ┌─────────────┐    权重同步     ┌─────────────┐            │          │
│   │  │ 训练引擎     │ ◄────────────► │ Rollout 引擎 │            │          │
│   │  │ (FSDP)      │                │ (vLLM)      │            │          │
│   │  └─────────────┘                └─────────────┘            │          │
│   └─────────────────────────────────────────────────────────────┘          │
│   适用：On-policy 训练 (GRPO/PPO)                                           │
│                                                                             │
│   COLOCATED (共置模式)                                                       │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │  同一 GPU，不同进程                                           │          │
│   │  ┌─────────────┐    无权重同步    ┌─────────────┐            │          │
│   │  │ 训练引擎     │ ─ ─ ─ ─ ─ ─ ─  │ Rollout 引擎 │            │          │
│   │  │ (进程 A)    │                │ (进程 B)     │            │          │
│   │  └─────────────┘                └─────────────┘            │          │
│   └─────────────────────────────────────────────────────────────┘          │
│   适用：GRM (LLM as a Judge)                                               │
│                                                                             │
│   STANDALONE (独立模式) ← Phase 0 使用                                       │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │  独立 GPU 资源                                                │          │
│   │                         ┌─────────────┐                     │          │
│   │      无训练引擎          │ Rollout 引擎 │                     │          │
│   │                         │ (vLLM/SGLang)│                     │          │
│   │                         │ HTTP Server  │                     │          │
│   │                         └─────────────┘                     │          │
│   └─────────────────────────────────────────────────────────────┘          │
│   适用：Phase 0 评测、Off-policy 训练、批量推理                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、verl Rollout 架构详解

### 2.1 核心类层次结构

```
RolloutReplica (抽象基类)
├── 定义三种初始化方法：
│   ├── init_hybrid(worker_group)     # 混合模式
│   ├── init_colocated(resource_pool) # 共置模式
│   └── init_standalone()             # 独立模式 ← Phase 0
│
├── vLLMReplica (vLLM 实现)
│   └── verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
│
└── SGLangReplica (SGLang 实现)
    └── verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py
```

### 2.2 RolloutReplica 基类

**文件位置**：`verl/verl/workers/rollout/replica.py`

```python
class RolloutReplica(ABC):
    """
    Rollout replica 是一个独立的服务器实例，可部署在单节点或多节点上。

    等效于命令行启动：
    - SGLang: python -m sglang.launch_server --node-rank 0 --nnode 2 ...
    - vLLM:   vllm serve --data-parallel-size 16 ...

    参数：
        replica_rank: int, 当前 replica 的编号
        config: RolloutConfig, 推理配置
        model_config: HFModelConfig, 模型配置
        gpus_per_node: int, 每节点 GPU 数量
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
    ) -> None:
        self.replica_rank = replica_rank
        self.config = config
        self.model_config = model_config

        # 计算并行度
        self.world_size = (
            config.tensor_model_parallel_size
            * config.data_parallel_size
            * config.pipeline_model_parallel_size
        )
        self.gpus_per_node = min(gpus_per_node, self.world_size)
        self.nnodes = self.world_size // self.gpus_per_node

        # 运行时状态
        self.rollout_mode: RolloutMode = None
        self.workers: list[ActorHandle] = []
        self.resource_pool: RayResourcePool = None
        self._server_address: str = None      # HTTP 服务器地址
        self._server_handle: ActorHandle = None
```

### 2.3 init_standalone() 详解

这是 Phase 0 评测使用的核心方法：

```python
async def init_standalone(self):
    """
    初始化 Standalone Rollout 服务器

    流程：
    1. 创建独立的 GPU 资源池
    2. 创建 Worker 组
    3. 启动 HTTP 服务器
    """
    # Step 1: 设置模式
    self.rollout_mode = RolloutMode.STANDALONE

    # Step 2: 创建资源池
    # 每个 replica 拥有独立的 GPU 资源
    resource_pool_name = f"rollout_pool_{self.replica_rank}"
    resource_pool_spec = {
        resource_pool_name: [self.gpus_per_node] * self.nnodes,
    }

    # 使用 ResourcePoolManager 管理 Ray placement group
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=None
    )
    resource_pool_manager.create_resource_pool()
    self.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]

    # Step 3: 创建 Worker 组
    # Worker 是 Ray Actor，运行 vLLM/SGLang 推理引擎
    worker_group = RayWorkerGroup(
        resource_pool=self.resource_pool,
        ray_cls_with_init=self.get_ray_class_with_init_args(),  # 抽象方法
        bin_pack=False,
        name_prefix=f"rollout_standalone_{self.replica_rank}",
    )
    self.workers = worker_group.workers

    # Step 4: 启动 HTTP 服务器
    # 提供 OpenAI-compatible API
    await self.launch_servers()  # 抽象方法
```

**关键概念解释**：

| 概念 | 说明 |
|------|------|
| `ResourcePool` | Ray placement group 的封装，管理 GPU 资源分配 |
| `RayWorkerGroup` | 管理一组 Ray Actor，提供数据分发和收集 |
| `replica_rank` | 当前 replica 的编号（0, 1, 2, ...） |
| `world_size` | 总 GPU 数量 = TP × DP × PP |

### 2.4 vLLMReplica 实现

**文件位置**：`verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`

```python
class vLLMReplica(RolloutReplica):
    """vLLM 后端的 Rollout Replica 实现"""

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """返回 vLLM Worker 类及其初始化参数"""
        worker_dict_cls = RayClassWithInitArgs(
            cls=ray.remote(vLLMAsyncRollout),  # Ray Actor 类
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,  # Standalone 模式不需要 device mesh
        )
        return worker_dict_cls

    async def launch_servers(self):
        """启动 vLLM HTTP 服务器"""
        # 在每个节点上启动一个 HTTP 服务器
        self._http_server = vLLMHttpServer(
            workers=self.workers,
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
        )

        # 启动服务器并获取地址
        self._server_address = await self._http_server.start()
        self._server_handle = self._http_server
```

### 2.5 HTTP Server 架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         vLLM HTTP Server 架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Client (评测脚本)                                                          │
│       │                                                                     │
│       │ POST /v1/chat/completions                                          │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │  vLLMHttpServer (Ray Actor)                                  │          │
│   │                                                              │          │
│   │  ┌─────────────────────────────────────────────────────┐    │          │
│   │  │  FastAPI Application                                  │    │          │
│   │  │                                                       │    │          │
│   │  │  /v1/chat/completions  ──────────────────────────┐   │    │          │
│   │  │  /v1/completions       ──────────────────────────┤   │    │          │
│   │  │  /health               ──────────────────────────┤   │    │          │
│   │  └──────────────────────────────────────────────────┤───┘    │          │
│   │                                                      │        │          │
│   │                                                      ▼        │          │
│   │  ┌─────────────────────────────────────────────────────┐    │          │
│   │  │  AsyncLLM Engine (vLLM v1)                          │    │          │
│   │  │                                                       │    │          │
│   │  │  ┌───────────────┐  ┌───────────────┐               │    │          │
│   │  │  │  GPU 0 (TP=0) │  │  GPU 1 (TP=1) │  ...          │    │          │
│   │  │  │  Attention    │  │  Attention    │               │    │          │
│   │  │  │  FFN          │  │  FFN          │               │    │          │
│   │  │  └───────────────┘  └───────────────┘               │    │          │
│   │  └─────────────────────────────────────────────────────┘    │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、代码执行流程（完整追踪）

### 3.1 端到端流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase 0 Baseline 评测流程                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 环境初始化                                                               │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │  ray.init(runtime_env={                                      │        │
│     │      "env_vars": {                                           │        │
│     │          "TOKENIZERS_PARALLELISM": "true",                   │        │
│     │          "NCCL_DEBUG": "WARN",                               │        │
│     │          "VLLM_USE_V1": "1"  # 使用 vLLM v1 引擎             │        │
│     │      }                                                       │        │
│     │  })                                                          │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  2. 创建 Rollout Replica                                                    │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │  rollout_class = get_rollout_replica_class("vllm")           │        │
│     │                                                              │        │
│     │  num_replicas = total_gpus / tensor_parallel_size            │        │
│     │  例如: 8 GPU, TP=2 → 4 replicas                              │        │
│     │                                                              │        │
│     │  replicas = [                                                │        │
│     │      rollout_class(replica_rank=i, config=..., model=...)    │        │
│     │      for i in range(num_replicas)                            │        │
│     │  ]                                                           │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  3. 初始化 Standalone 模式                                                   │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │  await asyncio.gather(*[                                     │        │
│     │      replica.init_standalone()                               │        │
│     │      for replica in replicas                                 │        │
│     │  ])                                                          │        │
│     │                                                              │        │
│     │  内部执行：                                                   │        │
│     │  ├── 创建 ResourcePool (Ray placement group)                 │        │
│     │  ├── 创建 RayWorkerGroup (Ray Actors)                        │        │
│     │  ├── 加载模型权重 (load_format="auto")                       │        │
│     │  └── 启动 HTTP 服务器                                        │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  4. 获取服务器地址                                                           │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │  server_addresses = [replica._server_address for replica...] │        │
│     │  例如: ["10.0.0.1:8000", "10.0.0.1:8001", ...]              │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  5. 加载评测数据                                                             │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │  data = pd.read_parquet("codecontests_valid.parquet")        │        │
│     │  prompts = data['prompt'].tolist()                           │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  6. 批量生成（异步 HTTP 请求）                                               │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │  async def generate(prompts):                                │        │
│     │      # 将数据分发到多个 replica（负载均衡）                   │        │
│     │      chunks = np.array_split(prompts, num_replicas)          │        │
│     │                                                              │        │
│     │      # 并行请求所有 replica                                  │        │
│     │      results = await asyncio.gather(*[                       │        │
│     │          generate_per_replica(server_addresses[i], chunks[i])│        │
│     │          for i in range(num_replicas)                        │        │
│     │      ])                                                      │        │
│     │      return results                                          │        │
│     │                                                              │        │
│     │  async def generate_per_replica(server_addr, prompts):       │        │
│     │      async with aiohttp.ClientSession() as session:          │        │
│     │          response = await session.post(                      │        │
│     │              f"http://{server_addr}/v1/chat/completions",    │        │
│     │              json={                                          │        │
│     │                  "model": model_path,                        │        │
│     │                  "messages": [{"role":"user","content":...}],│        │
│     │                  "temperature": 0.0,  # EVAL@1 greedy        │        │
│     │                  "max_tokens": 2048                          │        │
│     │              }                                               │        │
│     │          )                                                   │        │
│     │      return response.choices[0].message.content              │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  7. SandboxFusion 评测                                                      │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │  from sandbox_fusion import submit, SubmitRequest, TestConfig│        │
│     │                                                              │        │
│     │  for completion, record in zip(completions, data):           │        │
│     │      result = submit(SubmitRequest(                          │        │
│     │          dataset=record['sandbox_dataset'],                  │        │
│     │          id=record['sandbox_id'],                            │        │
│     │          completion=completion,                              │        │
│     │          config=TestConfig(language='python', run_timeout=10)│        │
│     │      ))                                                      │        │
│     │                                                              │        │
│     │      metrics.append({                                        │        │
│     │          "accepted": result.accepted,                        │        │
│     │          "pass_ratio": 1.0 if result.accepted else 0.0,      │        │
│     │      })                                                      │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  8. 指标聚合与日志                                                           │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │  wandb.log({                                                 │        │
│     │      "eval/codecontests_valid/accepted_at_1": 0.08,          │        │
│     │      "eval/codecontests_valid/pass_ratio_mean": 0.23,        │        │
│     │      ...                                                     │        │
│     │  })                                                          │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Ray 集群初始化详解

```python
import ray

# 初始化 Ray 集群
ray.init(
    runtime_env={
        "env_vars": {
            # 启用 tokenizer 并行化
            "TOKENIZERS_PARALLELISM": "true",

            # NCCL 调试级别（WARN 减少输出）
            "NCCL_DEBUG": "WARN",

            # 使用 vLLM v1 引擎（推荐）
            "VLLM_USE_V1": "1",
        }
    }
)
```

**为什么需要这些环境变量？**

| 变量 | 作用 |
|------|------|
| `TOKENIZERS_PARALLELISM` | 允许 HuggingFace tokenizer 并行处理，提高效率 |
| `NCCL_DEBUG` | NVIDIA 集合通信库日志级别，WARN 减少噪音 |
| `VLLM_USE_V1` | 启用 vLLM 新版引擎，性能更好 |

### 3.3 Replica 数量计算

```python
# 配置参数
n_gpus_per_node = 8
nnodes = 1
tensor_model_parallel_size = 2  # TP 并行度

# 计算总 GPU 数量
total_gpus = n_gpus_per_node * nnodes  # = 8

# 计算 replica 数量
# 每个 replica 需要 TP 个 GPU
num_replicas = total_gpus // tensor_model_parallel_size  # = 4

# 结果：4 个 replica，每个使用 2 个 GPU (TP=2)
# replica 0: GPU 0-1
# replica 1: GPU 2-3
# replica 2: GPU 4-5
# replica 3: GPU 6-7
```

### 3.4 HTTP API 请求格式

verl 的 Rollout 服务器提供 **OpenAI 兼容 API**：

```python
# 请求
POST /v1/chat/completions
{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [
        {"role": "user", "content": "Write a Python function..."}
    ],
    "temperature": 0.0,      # EVAL@1 协议使用 greedy decoding
    "top_p": 1.0,
    "max_tokens": 2048
}

# 响应
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1706745600,
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "```python\ndef solution():\n    ..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "total_tokens": 300
    }
}
```

---

## 四、SandboxFusion 集成

### 4.1 两种评测方式对比

| 方式 | API | 简单性 | 与 GRPO 一致性 | 推荐场景 |
|------|-----|--------|---------------|---------|
| `submit()` | SandboxFusion SDK | ⭐⭐⭐ 简单 | ❌ 不同 | Phase 0 快速评测 |
| `compute_score()` | verl 内部 | ⭐⭐ 中等 | ✅ 一致 | 需要与训练对齐 |

### 4.2 使用 submit() API（推荐用于 Phase 0）

```python
from sandbox_fusion import submit, SubmitRequest, TestConfig

def evaluate_with_submit(completion: str, record: dict) -> dict:
    """
    使用 SandboxFusion submit() API 评测

    优点：
    - 无需管理测试用例
    - 代码简洁
    - 直接返回 accepted 结果
    """
    result = submit(SubmitRequest(
        dataset=record['sandbox_dataset'],  # e.g., "humaneval"
        id=record['sandbox_id'],             # e.g., "0"
        completion=completion,
        config=TestConfig(
            language='python',
            run_timeout=10
        )
    ))

    return {
        "accepted": result.accepted,
        "tests": result.tests if hasattr(result, 'tests') else None,
    }
```

### 4.3 使用 compute_score()（与 GRPO 一致）

```python
from verl.utils.reward_score.sandbox_fusion import compute_score

def evaluate_with_compute_score(
    completion: str,
    test_cases: dict,
    sandbox_url: str = "http://localhost:8080/run_code"
) -> tuple[float, list]:
    """
    使用 verl compute_score() 评测

    优点：
    - 与 GRPO 训练阶段完全一致
    - 返回详细的 metadata

    参数：
        test_cases: {"inputs": [...], "outputs": [...]}
    """
    score, metadata = compute_score(
        sandbox_fusion_url=sandbox_url,
        concurrent_semaphore=None,
        memory_limit_mb=1024,
        completion=completion,
        test_cases=test_cases,
        continuous=False,
        timeout=10,
    )

    return score, metadata
```

### 4.4 返回值与状态码

| 结果值 | 含义 | 统计类别 |
|--------|------|---------|
| `True` | 测试通过 | success |
| `False` | 输出错误 (Wrong Answer) | wrong_answer |
| `-1` | API/Sandbox 错误 | api_error |
| `-2` | 运行时错误 (Runtime Error) | runtime_error |
| `-3` | 超时 (Timeout) | timeout |
| `-4` | 编译错误 (Compile Error) | syntax_error |

```python
def determine_final_status(results: list) -> str:
    """根据测试结果确定最终状态"""
    for r in results:
        if r == -4:
            return "syntax_error"
        elif r == -2:
            return "runtime_error"
        elif r == -3:
            return "timeout"

    if all(r is True for r in results):
        return "success"
    elif any(r is False for r in results):
        return "wrong_answer"
    else:
        return "api_error"
```

---

## 五、官方参考实现分析

### 5.1 main_generation_server.py 完整解析

**文件位置**：`verl/verl/trainer/main_generation_server.py`

这是 verl 官方提供的 Standalone 模式参考实现，Phase 0 评测可以直接基于此修改。

```python
"""
Generate responses given a dataset of prompts
"""

import os
import aiohttp
import hydra
import numpy as np
import ray

# 环境变量设置
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import asyncio
from pprint import pprint
import pandas as pd
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletion

from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.replica import get_rollout_replica_class


async def start_server(config):
    """
    创建并初始化 Standalone Rollout 服务器

    Returns:
        server_handles: Ray Actor 句柄列表
        server_addresses: HTTP 服务器地址列表
    """
    # 计算 replica 数量
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size

    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model

    # 获取对应的 Replica 类（vLLM 或 SGLang）
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)

    # 创建所有 replica 实例
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]

    # 并行初始化所有 replica（Standalone 模式）
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    # 收集服务器信息
    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]

    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    return server_handles, server_addresses


async def submit_request(server_address, **chat_complete_request):
    """
    向单个服务器提交请求

    使用 aiohttp 而非 openai 库，避免大量请求时的死锁问题
    """
    try:
        extra_headers = chat_complete_request.pop("extra_headers", {})
        timeout = aiohttp.ClientTimeout(total=None)  # 无超时限制
        session = aiohttp.ClientSession(timeout=timeout)

        async with session.post(
            url=f"http://{server_address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},
            json=chat_complete_request,
        ) as resp:
            data = await resp.json()
            return ChatCompletion(**data)
    finally:
        await session.close()


async def generate_per_replica(server_address, model_path: str, n_samples: int,
                               sampling_params: dict, chat_lst: list):
    """
    在单个 replica 上生成

    Args:
        n_samples: 每个 prompt 生成的样本数
    """
    # 构建请求列表
    chat_complete_request = [
        {
            "model": model_path,
            "messages": messages,
            **sampling_params,
        }
        for messages in chat_lst
        for _ in range(n_samples)  # 每个 prompt 重复 n_samples 次
    ]

    # 并行提交所有请求
    tasks = [submit_request(server_address, **req) for req in chat_complete_request]
    results = await asyncio.gather(*tasks)
    return results


async def generate(server_addresses: list, model_path: str, n_samples: int,
                   sampling_params: dict, chat_numpy: np.ndarray):
    """
    在多个 replica 上并行生成（负载均衡）
    """
    num_replicas = len(server_addresses)

    # 将数据均匀分配到各 replica
    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]

    assert len(server_addresses) == len(chat_sub_array)

    # 并行调用所有 replica
    results = await asyncio.gather(*[
        generate_per_replica(
            server_addresses[i], model_path, n_samples,
            sampling_params, chat_sub_array[i]
        )
        for i in range(num_replicas)
    ])
    return results


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """主入口"""
    # 初始化 Ray
    ray.init(runtime_env={
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_USE_V1": "1"
        }
    })

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    n_samples = config.actor_rollout_ref.rollout.n

    # 验证采样参数
    if config.actor_rollout_ref.rollout.temperature == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."
    assert n_samples >= 1

    # 采样参数
    sampling_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        "max_tokens": config.actor_rollout_ref.rollout.response_length,
    }

    # 加载数据
    train_files = config.data.train_files
    if not isinstance(train_files, list):
        train_files = [train_files]

    datasets = [pd.read_parquet(f) for f in train_files]
    dataset = pd.concat(datasets, axis=0, ignore_index=True)

    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]
    chat_numpy = np.array(chat_lst)

    # 启动服务器
    server_handles, server_addresses = asyncio.run(start_server(config))

    # 生成
    gen_results = asyncio.run(
        generate(server_addresses, config.actor_rollout_ref.model.path,
                 n_samples, sampling_params, chat_numpy)
    )

    # 处理结果
    import itertools
    results = list(itertools.chain.from_iterable(gen_results))
    results = np.array([r.choices[0].message.content for r in results])
    results = np.reshape(results, (-1, n_samples))

    assert results.shape == (len(chat_lst), n_samples)

    dataset["responses"] = results.tolist()

    # 保存结果
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {config.data.output_path}")
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
```

### 5.2 关键函数总结

| 函数 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `start_server()` | 初始化 Standalone 服务器 | config | (handles, addresses) |
| `submit_request()` | 单个 HTTP 请求 | server_addr, request | ChatCompletion |
| `generate_per_replica()` | 单 replica 批量生成 | addr, prompts | results |
| `generate()` | 多 replica 负载均衡 | addrs, prompts | all_results |

---

## 六、配置参数详解

### 6.1 RolloutConfig 关键参数

**文件位置**：`verl/verl/workers/config/rollout.py`

```python
@dataclass
class RolloutConfig:
    # 推理引擎选择
    name: str = "vllm"  # "vllm" 或 "sglang"
    mode: str = "async"  # 仅支持 "async"

    # 并行度配置
    tensor_model_parallel_size: int = 2   # TP 并行度
    data_parallel_size: int = 1           # DP 并行度
    pipeline_model_parallel_size: int = 1 # PP 并行度

    # 采样参数
    temperature: float = 1.0   # 温度（EVAL@1 用 0.0）
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1                 # 每 prompt 生成数量

    # 长度限制
    prompt_length: int = 4096
    response_length: int = 2048

    # 内存与性能
    gpu_memory_utilization: float = 0.8
    enforce_eager: bool = True           # 禁用 CUDA Graph
    enable_prefix_caching: bool = True   # 启用前缀缓存
    enable_chunked_prefill: bool = True  # 启用分块预填充

    # ⚠️ 关键参数
    load_format: str = "auto"  # Standalone 必须用 "auto"

    # 批处理
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192

    # 日志
    disable_log_stats: bool = True
```

### 6.2 load_format 参数（重点）

**这是 Standalone 模式最关键的参数**：

| load_format | 说明 | 适用模式 |
|-------------|------|---------|
| `"auto"` | 从磁盘/HDFS 自动加载模型权重 | **STANDALONE** |
| `"dummy"` | 创建空壳模型（权重由训练引擎同步） | HYBRID |
| `"safetensors"` | 强制使用 safetensors 格式 | 特殊情况 |

**Phase 0 必须使用 `load_format: "auto"`**，否则：
- 模型权重不会被加载
- 推理输出为随机值
- 评测结果无意义

### 6.3 Hydra 配置文件示例

```yaml
# config/phase0_eval.yaml

defaults:
  - _self_

trainer:
  n_gpus_per_node: 8
  nnodes: 1
  device: "cuda"

actor_rollout_ref:
  model:
    path: "Qwen/Qwen2.5-Coder-7B-Instruct"
    trust_remote_code: true
    lora_rank: 0

  rollout:
    name: "vllm"
    mode: "async"

    # 并行度
    tensor_model_parallel_size: 2
    data_parallel_size: 1
    pipeline_model_parallel_size: 1

    # EVAL@1 协议
    temperature: 0.0
    top_p: 1.0
    n: 1

    # 长度
    prompt_length: 4096
    response_length: 2048

    # 内存
    dtype: "bfloat16"
    gpu_memory_utilization: 0.8

    # ⚠️ 关键：必须为 "auto"
    load_format: "auto"

    # 性能
    enforce_eager: true
    enable_prefix_caching: true
    enable_chunked_prefill: true
    max_num_seqs: 256
    max_num_batched_tokens: 8192
    disable_log_stats: true

data:
  train_files:
    - "data/codecontests_valid.parquet"
  prompt_key: "prompt"
  output_path: "outputs/phase0/results.parquet"

ray_kwargs:
  ray_init:
    num_cpus: null
    runtime_env:
      env_vars:
        TOKENIZERS_PARALLELISM: "true"
        NCCL_DEBUG: "WARN"
        VLLM_USE_V1: "1"
```

---

## 七、常见问题与排查

### 7.1 模型未加载（load_format 错误）

**症状**：
- 模型输出为随机 token
- 所有评测结果为 0
- 日志显示 "Loading model with dummy weights"

**原因**：使用了 `load_format: "dummy"`

**解决**：确保配置中 `load_format: "auto"`

```yaml
rollout:
  load_format: "auto"  # 不是 "dummy"!
```

### 7.2 NCCL 错误

**症状**：
```
NCCL error: unhandled system error
```

**解决**：
1. 检查 GPU 驱动版本
2. 设置环境变量：
```python
os.environ["NCCL_DEBUG"] = "INFO"  # 获取更多信息
os.environ["NCCL_P2P_DISABLE"] = "1"  # 禁用 P2P（如果有问题）
```

### 7.3 SandboxFusion 超时

**症状**：
```
aiohttp.ClientError: Connection timeout
```

**解决**：
1. 确认 SandboxFusion 服务运行中：
```bash
curl http://localhost:8080/health
```
2. 增加超时时间：
```python
config=TestConfig(run_timeout=30)  # 增加到 30 秒
```
3. 减少并发请求数

### 7.4 奖励全为 0

**症状**：
- `compute_score()` 总是返回 0
- 所有测试用例标记为失败

**排查步骤**：
1. 检查代码提取逻辑（是否正确处理 ```python``` 块）
2. 检查测试用例格式：
```python
# 正确格式
test_cases = {
    "inputs": ["1 2\n", "3 4\n"],
    "outputs": ["3\n", "7\n"]
}
```
3. 手动测试单个用例：
```python
from sandbox_fusion import run_code, RunCodeRequest

result = run_code(RunCodeRequest(
    code="print(sum(map(int, input().split())))",
    stdin="1 2\n",
    language="python",
    run_timeout=10
))
print(result.stdout)  # 应该是 "3\n"
```

---

## 附录

### A. 关键文件索引

| 用途 | 文件路径 |
|------|----------|
| RolloutMode 枚举 | `verl/verl/workers/rollout/replica.py:44-57` |
| RolloutReplica 基类 | `verl/verl/workers/rollout/replica.py:60-210` |
| init_standalone() | `verl/verl/workers/rollout/replica.py:149-176` |
| get_rollout_replica_class() | `verl/verl/workers/rollout/replica.py:286-287` |
| vLLMReplica | `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` |
| SGLangReplica | `verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py` |
| main_generation_server.py | `verl/verl/trainer/main_generation_server.py` |
| compute_score() | `verl/verl/utils/reward_score/sandbox_fusion/__init__.py` |
| check_correctness() | `verl/verl/utils/reward_score/sandbox_fusion/utils.py` |
| RolloutConfig | `verl/verl/workers/config/rollout.py` |
| HFModelConfig | `verl/verl/workers/config/model.py` |

### B. 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| Rollout | - | 使用 LLM 生成文本序列的过程 |
| Replica | - | 一个独立的推理服务器实例 |
| TP | Tensor Parallelism | 张量并行，将模型层切分到多 GPU |
| DP | Data Parallelism | 数据并行，相同模型处理不同数据 |
| PP | Pipeline Parallelism | 流水线并行，不同层在不同 GPU |
| Ray Actor | - | Ray 框架中的远程对象，可调用方法 |
| Placement Group | - | Ray 中的资源分配单位 |
| ResourcePool | - | verl 对 Placement Group 的封装 |
| WorkerGroup | - | verl 中管理一组 Worker 的类 |

### C. 参考链接

- [verl 官方文档](https://github.com/volcengine/verl)
- [vLLM 文档](https://docs.vllm.ai/)
- [SGLang 文档](https://github.com/sgl-project/sglang)
- [Ray 文档](https://docs.ray.io/)
- [SandboxFusion](https://github.com/bytedance/SandboxFusion)

---

*文档版本：v1.0*
*创建日期：2026-01-31*
*适用版本：verl 0.7+, vLLM 0.8+*
