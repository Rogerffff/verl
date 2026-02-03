# 从 RayWorkerGroup 到 Replica：深入理解 verl 的分布式架构

> 本文档从你已经学过的 `base00_verl的ray的入门使用.ipynb` 出发，逐步讲解 verl 中 `init_standalone()` 的工作原理。

---

## 第一部分：回顾 base00 中的核心概念

### 1.1 你已经学过的内容

在 `base00_verl的ray的入门使用.ipynb` 中，你学到了以下关键概念：

```python
# 1. Ray Actor：用 @ray.remote 装饰的类，运行在独立进程中
@ray.remote
class GPUAccumulator(Worker):
    def __init__(self):
        super().__init__()
        self.value = torch.zeros(1, device="cuda") + self.rank

    def add(self, x):
        self.value += x
        return self.value.cpu()

# 2. RayResourcePool：GPU 资源池，告诉 Ray "我需要多少 GPU"
resource_pool = RayResourcePool([2], use_gpu=True)  # 需要 2 个 GPU

# 3. RayClassWithInitArgs：封装 Actor 类和它的初始化参数
class_with_args = RayClassWithInitArgs(cls=GPUAccumulator)

# 4. RayWorkerGroup：管理一组 Worker，可以批量调用方法
work_group = RayWorkerGroup(
    resource_pool=resource_pool,
    ray_cls_with_init=class_with_args
)

# 5. 使用 WorkerGroup 调用方法
value = work_group.execute_all_sync("add", x=[1, 1])
# 返回：[tensor([1.]), tensor([2.])]
```

### 1.2 关键理解点

让我用图来说明这些概念的关系：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        base00 中学到的结构                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   RayResourcePool([2], use_gpu=True)                                    │
│   ─────────────────────────────────                                     │
│   "我需要 2 个 GPU"                                                      │
│                                                                          │
│                    ↓ 传入                                                │
│                                                                          │
│   RayWorkerGroup(resource_pool, ray_cls_with_init)                      │
│   ─────────────────────────────────────────────────                     │
│   "在这 2 个 GPU 上，各启动一个 GPUAccumulator Actor"                    │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                     RayWorkerGroup                            │      │
│   │                                                               │      │
│   │   ┌─────────────────────┐    ┌─────────────────────┐         │      │
│   │   │   Worker 0          │    │   Worker 1          │         │      │
│   │   │   (GPUAccumulator)  │    │   (GPUAccumulator)  │         │      │
│   │   │   GPU 0             │    │   GPU 1             │         │      │
│   │   │   rank = 0          │    │   rank = 1          │         │      │
│   │   └─────────────────────┘    └─────────────────────┘         │      │
│   │                                                               │      │
│   │   work_group.execute_all_sync("add", x=[1, 1])               │      │
│   │   → Worker 0.add(1), Worker 1.add(1) → [tensor(1), tensor(2)]│      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 问题检查点

在继续之前，请确认你理解以下概念：

1. **Ray Actor** 是什么？（运行在独立进程中的类实例）
2. **RayResourcePool** 做什么？（声明需要多少 GPU 资源）
3. **RayWorkerGroup** 做什么？（在资源池的 GPU 上启动多个 Worker Actor）
4. `work_group.execute_all_sync("add", x=[1, 1])` 发生了什么？（对每个 Worker 调用 add 方法）

**如果你理解了以上内容，请告诉我，我们继续下一部分：`init_standalone()` 中的资源池创建。**

---

## 第二部分：从 RayResourcePool 到 ResourcePoolManager

### 2.1 base00 中的简单方式 vs init_standalone 中的方式

让我们对比一下两种创建资源池的方式：

```python
# ========== base00 中的简单方式 ==========
resource_pool = RayResourcePool([2], use_gpu=True)
# 直接创建，一步到位

# ========== init_standalone 中的方式 ==========
resource_pool_spec = {
    "rollout_pool_0": [2],  # 资源池名称 → GPU 分配列表
}
resource_pool_manager = ResourcePoolManager(
    resource_pool_spec=resource_pool_spec,
    mapping=None
)
resource_pool_manager.create_resource_pool()  # 需要显式调用创建
resource_pool = resource_pool_manager.resource_pool_dict["rollout_pool_0"]
```

**为什么 `init_standalone` 要用更复杂的方式？**

### 2.2 为什么 init_standalone 使用 ResourcePoolManager？

在 `init_standalone()` 中，每个 Replica 实例只创建**一个**属于自己的资源池：

```
┌─────────────────────────────────────────────────────────────────────────┐
│              init_standalone() 中的资源池创建                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【核心理解】                                                            │
│                                                                          │
│  每个 Replica 实例只创建自己的资源池：                                   │
│  ───────────────────────────────────                                     │
│                                                                          │
│  # 在某个 Replica 实例中（假设 replica_rank = 0）                        │
│  resource_pool_name = f"rollout_pool_{self.replica_rank}"               │
│  # → "rollout_pool_0"                                                    │
│                                                                          │
│  resource_pool_spec = {                                                  │
│      "rollout_pool_0": [2],   ← 只有这一个条目！                         │
│  }                                                                       │
│                                                                          │
│  【为什么用 ResourcePoolManager 而不是直接用 RayResourcePool？】         │
│                                                                          │
│  1. 命名：资源池有名字 "rollout_pool_0"，方便调试和追踪                  │
│  2. 统一接口：verl 的其他模块都使用 ResourcePoolManager                  │
│  3. 扩展性：虽然这里只创建一个池，但接口支持更复杂的场景                 │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │         Replica 0 的 init_standalone()                           │   │
│  │                                                                   │   │
│  │   resource_pool_spec = {"rollout_pool_0": [2]}                   │   │
│  │                    ↓                                              │   │
│  │   ResourcePoolManager(resource_pool_spec)                        │   │
│  │                    ↓                                              │   │
│  │   .create_resource_pool()                                        │   │
│  │                    ↓                                              │   │
│  │   self.resource_pool = ...["rollout_pool_0"]                     │   │
│  │                                                                   │   │
│  │   结果：获得一个包含 2 个 GPU 的资源池                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**重要澄清**：多个 Replica 的场景发生在更上层的代码中（如 `main_generation_server.py`），而不是在单个 `init_standalone()` 调用中：

```
┌─────────────────────────────────────────────────────────────────────────┐
│           多个 Replica 是如何产生的？（上层视角）                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  # 在 main_generation_server.py 中（简化示意）                          │
│  replicas = [                                                            │
│      Replica(replica_rank=0, ...),  # 调用 init_standalone() → pool_0  │
│      Replica(replica_rank=1, ...),  # 调用 init_standalone() → pool_1  │
│  ]                                                                       │
│                                                                          │
│  # 每个 Replica 实例独立调用 init_standalone()                          │
│  # 每个只创建自己的资源池，互不知道其他 Replica 的存在                   │
│                                                                          │
│  ┌─────────────────┐          ┌─────────────────┐                       │
│  │   Replica 0     │          │   Replica 1     │                       │
│  │   replica_rank=0│          │   replica_rank=1│                       │
│  │                 │          │                 │                       │
│  │ init_standalone │          │ init_standalone │                       │
│  │       ↓         │          │       ↓         │                       │
│  │ rollout_pool_0  │          │ rollout_pool_1  │                       │
│  └─────────────────┘          └─────────────────┘                       │
│         独立创建                      独立创建                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 init_standalone 中的具体代码解析

现在让我们逐行看 `init_standalone()` 中的资源池创建代码：

```python
async def init_standalone(self):
    # === Step 1: 创建独立的 GPU 资源池 ===
    self.rollout_mode = RolloutMode.STANDALONE

    # 给资源池起个名字，包含 replica_rank 以区分不同 Replica
    resource_pool_name = f"rollout_pool_{self.replica_rank}"
    # 例如：Replica 0 → "rollout_pool_0"
    #       Replica 1 → "rollout_pool_1"

    # 定义资源规格：每个节点分配多少 GPU
    resource_pool_spec = {
        resource_pool_name: [self.gpus_per_node] * self.nnodes,
    }
    # 例如：gpus_per_node=2, nnodes=1
    #       → {"rollout_pool_0": [2]}
    #       意思是：1 个节点，每节点 2 个 GPU

    # 例如：gpus_per_node=8, nnodes=2（跨两台机器）
    #       → {"rollout_pool_0": [8, 8]}
    #       意思是：2 个节点，每节点 8 个 GPU

    # 创建 ResourcePoolManager
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=None  # 不指定具体 GPU 映射，让 Ray 自动分配
    )

    # 实际创建 Ray Placement Group（资源池）
    resource_pool_manager.create_resource_pool()

    # 从 manager 中取出我们需要的资源池
    self.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]
```

### 2.4 类比理解

```
把 GPU 资源比作停车场：

【base00 的方式】
直接说：我要 2 个车位
→ RayResourcePool([2])

【init_standalone 的方式】
1. 先规划：这个停车场叫 "rollout_pool_0"，有 2 个车位
   → resource_pool_spec = {"rollout_pool_0": [2]}

2. 创建停车场管理系统
   → ResourcePoolManager(...)

3. 实际建造停车场
   → .create_resource_pool()

4. 获取停车场的入口
   → .resource_pool_dict["rollout_pool_0"]

为什么要这么麻烦？
- 因为可能要管理多个停车场（多个 Replica）
- 每个停车场需要有名字（方便追踪和调试）
- 需要统一管理和分配
```

---

## 问题检查点

在继续之前，请确认你理解：

1. 在 `init_standalone()` 中，每个 Replica 创建几个资源池？
   （只创建**一个**属于自己的资源池）

2. `resource_pool_spec = {"rollout_pool_0": [2]}` 是什么意思？
   （创建一个叫 "rollout_pool_0" 的资源池，包含 1 个节点，每节点 2 个 GPU）

3. 为什么用 ResourcePoolManager 而不是直接用 RayResourcePool？
   （命名、统一接口、扩展性）

4. 多个 Replica 的资源池是在哪里创建的？
   （每个 Replica 独立调用 init_standalone()，各自创建自己的池）

**理解了请告诉我，我们继续下一部分：WorkerGroup 的创建和 HTTP Server 启动。**

---

## 第三部分：WorkerGroup 的创建

### 3.1 回顾 base00 中的 WorkerGroup

在 base00 中，你学到了如何创建 WorkerGroup：

```python
# base00 中的方式
work_group = RayWorkerGroup(
    resource_pool=resource_pool,
    ray_cls_with_init=RayClassWithInitArgs(cls=GPUAccumulator)
)
```

`init_standalone()` 中的方式几乎一样：

```python
# init_standalone 中的方式
worker_group = RayWorkerGroup(
    resource_pool=self.resource_pool,
    ray_cls_with_init=self.get_ray_class_with_init_args(),  # 获取 Worker 类
    bin_pack=False,
    name_prefix=f"rollout_standalone_{self.replica_rank}",
)
self.workers = worker_group.workers
```

### 3.2 两者的区别

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    base00 vs init_standalone 的 WorkerGroup             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【base00】                                                              │
│  ─────────                                                               │
│  RayClassWithInitArgs(cls=GPUAccumulator)                               │
│  - 直接传入你定义的 Worker 类                                            │
│  - 简单明了                                                              │
│                                                                          │
│  【init_standalone】                                                     │
│  ─────────────────                                                       │
│  self.get_ray_class_with_init_args()                                    │
│  - 调用抽象方法获取 Worker 类                                            │
│  - 由子类（vLLMReplica 或 SGLangReplica）实现                           │
│  - 返回的是 vLLM 或 SGLang 的专用 Worker 类                             │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  为什么用抽象方法？                                               │   │
│  │                                                                   │   │
│  │  RolloutReplica（基类）                                          │   │
│  │       │                                                           │   │
│  │       ├── vLLMReplica（子类）                                    │   │
│  │       │      └── get_ray_class_with_init_args()                  │   │
│  │       │             → 返回 vLLMWorker 类                         │   │
│  │       │                                                           │   │
│  │       └── SGLangReplica（子类）                                  │   │
│  │              └── get_ray_class_with_init_args()                  │   │
│  │                     → 返回 SGLangWorker 类                       │   │
│  │                                                                   │   │
│  │  基类定义流程，子类提供具体实现（模板方法模式）                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Worker 与 GPU 的对应关系

这里再次强调一个关键点：**1 Worker = 1 GPU = 1 Ray Actor**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Worker-GPU 一一对应关系                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  假设：TP=2（模型需要 2 个 GPU）                                        │
│                                                                          │
│  resource_pool = RayResourcePool([2])  ← 申请 2 个 GPU                  │
│                                                                          │
│  RayWorkerGroup 创建后：                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     WorkerGroup                                  │    │
│  │                                                                   │    │
│  │   workers[0]                    workers[1]                       │    │
│  │   ┌─────────────────┐          ┌─────────────────┐               │    │
│  │   │   Worker 0      │          │   Worker 1      │               │    │
│  │   │   (Ray Actor)   │          │   (Ray Actor)   │               │    │
│  │   │   GPU 0         │          │   GPU 1         │               │    │
│  │   │   rank = 0      │          │   rank = 1      │               │    │
│  │   │                 │          │                 │               │    │
│  │   │  持有模型的     │          │  持有模型的     │               │    │
│  │   │  第 0 个分片    │◄────────►│  第 1 个分片    │               │    │
│  │   │                 │  NCCL    │                 │               │    │
│  │   └─────────────────┘  通信    └─────────────────┘               │    │
│  │                                                                   │    │
│  │   两个 Worker 协作完成一次推理（TP=2 时模型切分到 2 个 GPU）      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 init_standalone 中的 WorkerGroup 创建代码

```python
async def init_standalone(self):
    # ... Step 1: 创建资源池（上一部分已讲）...

    # === Step 2: 创建 WorkerGroup ===
    worker_group = RayWorkerGroup(
        resource_pool=self.resource_pool,
        ray_cls_with_init=self.get_ray_class_with_init_args(),
        bin_pack=False,
        name_prefix=f"rollout_standalone_{self.replica_rank}",
    )

    # 保存 workers 列表，供后续使用
    self.workers = worker_group.workers
    # workers = [Worker0_ActorHandle, Worker1_ActorHandle, ...]
```

**参数说明**：

| 参数 | 含义 |
|------|------|
| `resource_pool` | 上一步创建的 GPU 资源池 |
| `ray_cls_with_init` | Worker 类和初始化参数（由子类提供） |
| `bin_pack=False` | 不紧凑排布，保持 Worker 与 GPU 的顺序对应 |
| `name_prefix` | Worker Actor 的名称前缀（方便调试） |

---

## 问题检查点

在继续之前，请确认你理解：

1. `get_ray_class_with_init_args()` 是什么？
   （抽象方法，由子类 vLLMReplica/SGLangReplica 实现，返回具体的 Worker 类）

2. 为什么使用抽象方法而不是直接传入 Worker 类？
   （模板方法模式：基类定义流程，子类提供具体实现）

3. `self.workers` 是什么？
   （Worker 的 Ray Actor 句柄列表，用于后续操作）

**理解了请告诉我，我们继续下一部分：HTTP Server 的启动。**

---

## 补充章节：Ray Placement Group 资源分配机制

### 为什么需要这个补充？

在第二部分我们讲到，每个 Replica 独立调用 `init_standalone()` 创建自己的资源池。你可能会问：**如何保证多个 Replica 的资源池不会使用同一个 GPU？**

答案是：**Ray Placement Group 机制保证了资源隔离**。

### Ray Placement Group 是什么？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Ray Placement Group 简介                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Placement Group 是 Ray 的资源分配单元：                                 │
│                                                                          │
│  - 向 Ray 集群申请一组资源（如 2 个 GPU）                               │
│  - 申请成功后，这些资源被 **独占**                                       │
│  - 其他 Placement Group 无法使用已被占用的资源                          │
│                                                                          │
│  类比：在公司预订会议室                                                  │
│  - 你预订了 A 会议室 10:00-12:00                                        │
│  - 这个时间段 A 会议室只属于你                                          │
│  - 其他人无法重复预订                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 多个 Replica 的资源分配过程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    资源分配时间线（8 GPU，TP=2）                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Ray 集群初始状态：                                                      │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                      │
│  │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │  全部空闲            │
│  │  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │                      │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                      │
│                                                                          │
│  T1: Replica 0 调用 init_standalone()                                   │
│      → create_resource_pool() 向 Ray 申请 2 GPU                         │
│      → Ray 分配 GPU 0, 1（标记为已占用）                                │
│                                                                          │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                      │
│  │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │                      │
│  │  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │                      │
│  └──▲──┴──▲──┴─────┴─────┴─────┴─────┴─────┴─────┘                      │
│     └──┬──┘                                                              │
│   rollout_pool_0（独占）                                                 │
│                                                                          │
│  T2: Replica 1 调用 init_standalone()                                   │
│      → create_resource_pool() 向 Ray 申请 2 GPU                         │
│      → Ray 从剩余资源中分配 GPU 2, 3                                    │
│                                                                          │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                      │
│  │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │                      │
│  │  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │                      │
│  └──▲──┴──▲──┴──▲──┴──▲──┴─────┴─────┴─────┴─────┘                      │
│     └──┬──┘     └──┬──┘                                                  │
│   pool_0       pool_1                                                    │
│                                                                          │
│  T3, T4: Replica 2, 3 类似...                                           │
│                                                                          │
│  最终状态：                                                              │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                      │
│  │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │ GPU │                      │
│  │  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │                      │
│  └──▲──┴──▲──┴──▲──┴──▲──┴──▲──┴──▲──┴──▲──┴──▲──┘                      │
│     └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘                          │
│   pool_0       pool_1       pool_2       pool_3                          │
│  (Replica 0) (Replica 1) (Replica 2) (Replica 3)                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 谁负责计算 Replica 数量？

关键点：**`init_standalone()` 本身不关心其他 Replica**，它只管申请自己需要的资源。资源总量的控制由上层代码负责。

在 `main_generation_server.py` 的 `start_server()` 函数中（第 40-56 行）：

```python
# 源文件：verl/verl/trainer/main_generation_server.py:40-56

async def start_server(config):
    # 计算 Replica 数量
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size
    # 例如：8 GPU / TP=2 = 4 个 Replica

    # 获取 Replica 类
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)

    # 创建所有 Replica 实例
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)  # 0, 1, 2, 3
    ]

    # 并发初始化所有 Replica
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])
```

### 如果资源不足会怎样？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    资源不足的场景                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  假设：8 GPU，但配置错误，尝试启动 5 个 Replica（每个需要 2 GPU）        │
│  总需求：5 × 2 = 10 GPU > 8 GPU（实际可用）                             │
│                                                                          │
│  Replica 0-3：成功创建，占用 GPU 0-7                                    │
│  Replica 4：create_resource_pool() 阻塞等待...                          │
│             直到有资源释放，或者超时失败                                 │
│                                                                          │
│  【正确做法】                                                            │
│  上层代码正确计算 num_replicas：                                         │
│  num_replicas = total_gpus // tp_size = 8 // 2 = 4                      │
│  只创建 4 个 Replica，正好用完所有 GPU                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 关键总结

| 问题 | 答案 |
|------|------|
| 多个 Replica 会用到同一个 GPU 吗？ | **不会**，Ray Placement Group 是独占式分配 |
| 谁保证资源隔离？ | Ray 集群的 Placement Group 机制 |
| 谁负责计算启动多少个 Replica？ | 上层代码（如 `main_generation_server.py`） |
| `init_standalone()` 知道其他 Replica 吗？ | **不知道**，它只管申请自己需要的资源 |

---

## 第四部分：HTTP Server 的启动

### 4.1 init_standalone 的最后一步

在前面的部分中，我们完成了：
- **Step 1**: 创建资源池（Placement Group）
- **Step 2**: 创建 WorkerGroup（在 GPU 上启动 Worker Actor）

现在是最后一步：

```python
async def init_standalone(self):
    # Step 1: 创建资源池（已完成）
    # Step 2: 创建 WorkerGroup（已完成）

    # === Step 3: 启动 HTTP Server ===
    await self.launch_servers()
```

`launch_servers()` 是一个抽象方法，由子类 `vLLMReplica` 或 `SGLangReplica` 实现。下面我们以 `vLLMReplica` 为例详细讲解。

### 4.2 为什么需要 HTTP Server？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    为什么需要 HTTP Server？                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【问题】Workers 已经在 GPU 上加载了模型，如何让外部调用？               │
│                                                                          │
│  【方案】启动一个 HTTP Server，提供 OpenAI 兼容的 API                   │
│                                                                          │
│  外部调用方式：                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  curl http://localhost:8000/v1/chat/completions \                │   │
│  │    -H "Content-Type: application/json" \                         │   │
│  │    -d '{                                                          │   │
│  │      "model": "your-model",                                       │   │
│  │      "messages": [{"role": "user", "content": "Hello!"}]         │   │
│  │    }'                                                             │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  【优势】                                                                │
│  - 标准化：兼容 OpenAI API 格式，可直接使用各种 SDK                     │
│  - 解耦：调用方不需要了解 Ray、vLLM 等内部实现                         │
│  - 灵活：支持各种客户端（Python、curl、前端应用等）                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 HTTP Server 与 Workers 的关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                HTTP Server 与 Workers 的架构关系                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【关键理解】                                                            │
│  - Workers：持有模型权重，在 GPU 上执行实际推理计算                     │
│  - HTTP Server：接收 API 请求，协调 Workers 完成推理                   │
│  - 通信方式：通过 ZeroMQ (ZMQ) 进行高效的进程间通信                     │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        Replica 内部架构                           │   │
│  │                                                                   │   │
│  │   外部请求                                                        │   │
│  │      │                                                            │   │
│  │      ▼                                                            │   │
│  │  ┌───────────────────────────────────────────────────────────┐   │   │
│  │  │              HTTP Server (vLLMHttpServer)                  │   │   │
│  │  │              - 监听端口（如 8000）                          │   │   │
│  │  │              - 接收 /v1/chat/completions 请求              │   │   │
│  │  │              - 持有 vLLM 的 AsyncLLM 引擎                  │   │   │
│  │  └─────────────────────────┬─────────────────────────────────┘   │   │
│  │                            │                                      │   │
│  │                            │ ZMQ 通信                             │   │
│  │                            │                                      │   │
│  │            ┌───────────────┼───────────────┐                     │   │
│  │            │               │               │                     │   │
│  │            ▼               ▼               ▼                     │   │
│  │  ┌─────────────────┐ ┌─────────────────┐                        │   │
│  │  │   Worker 0      │ │   Worker 1      │  ...                   │   │
│  │  │   (Ray Actor)   │ │   (Ray Actor)   │                        │   │
│  │  │   GPU 0         │ │   GPU 1         │                        │   │
│  │  │   模型分片 0    │ │   模型分片 1    │                        │   │
│  │  └─────────────────┘ └─────────────────┘                        │   │
│  │                            │                                      │   │
│  │                            │ NCCL 集合通信                        │   │
│  │                            │ （跨 GPU 同步）                      │   │
│  │                            ▼                                      │   │
│  │                    推理结果返回给 Server                          │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 launch_servers() 的实现流程

让我们看 `vLLMReplica.launch_servers()` 的代码（简化版）：

```python
# 源文件：verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:713-776

async def launch_servers(self):
    """在每个节点上启动 HTTP Server"""

    # === Step 3.1: 验证 Worker 数量 ===
    assert len(self.workers) == self.world_size
    # 确保 Worker 数量与 world_size 一致（1 Worker = 1 GPU）

    # === Step 3.2: 获取每个 Worker 所在的节点 ID ===
    worker_node_ids = await asyncio.gather(*[
        worker.__ray_call__.remote(
            lambda self: ray.get_runtime_context().get_node_id()
        )
        for worker in self.workers
    ])
    # 这是为了确保 HTTP Server 和它的 Workers 在同一个节点上

    # === Step 3.3: 在每个节点创建 HTTP Server Actor ===
    for node_rank in range(nnodes):
        # 获取该节点上的 Workers
        workers = self.workers[node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node]
        node_id = worker_node_ids[node_rank * gpus_per_node]

        # 创建 vLLMHttpServer Actor，使用节点亲和性调度
        server = self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,  # 强制在同一节点
                soft=False,
            ),
            name=f"vllm_server_{self.replica_rank}_{node_rank}",
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=workers,  # 传入该节点的 Workers
            replica_rank=self.replica_rank,
            node_rank=node_rank,
            gpus_per_node=gpus_per_node,
            nnodes=nnodes,
        )
        self.servers.append(server)

    # === Step 3.4: 启动所有节点的 HTTP Server ===
    master_address, master_port = await self.servers[0].get_master_address.remote()
    await asyncio.gather(*[
        server.launch_server.remote(
            master_address=master_address,
            master_port=master_port
        )
        for server in self.servers
    ])

    # === Step 3.5: 获取 Server 地址 ===
    server_address, server_port = await self.servers[0].get_server_address.remote()
    self._server_handle = self.servers[0]
    self._server_address = f"{server_address}:{server_port}"
    # 例如："192.168.1.100:8000"
```

### 4.5 流程图解

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    launch_servers() 执行流程                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  前置状态：Workers 已创建（Step 2 完成）                                │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 3.1: 验证 Workers 数量                                     │    │
│  │            len(self.workers) == self.world_size                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                               │                                          │
│                               ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 3.2: 获取每个 Worker 的节点 ID                            │    │
│  │            （确定 Worker 在哪台物理机上）                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                               │                                          │
│                               ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 3.3: 为每个节点创建 vLLMHttpServer Actor                  │    │
│  │            - 使用 NodeAffinitySchedulingStrategy                │    │
│  │            - 保证 Server 和 Workers 在同一节点                  │    │
│  │            - 传入该节点的 Workers 列表                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                               │                                          │
│                               ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 3.4: 调用 server.launch_server()                          │    │
│  │            - 配置 vLLM 引擎参数                                  │    │
│  │            - 建立与 Workers 的 ZMQ 连接                         │    │
│  │            - 启动 Uvicorn HTTP 服务                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                               │                                          │
│                               ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 3.5: 保存 Server 地址                                      │    │
│  │            self._server_address = "192.168.1.100:8000"          │    │
│  │            （供外部调用使用）                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  完成！Replica 可以接收 HTTP 请求了                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.6 ZeroMQ (ZMQ) 通信机制

你可能好奇：HTTP Server 如何与 Workers 通信？答案是 **ZeroMQ**。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ZeroMQ 通信机制                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【为什么用 ZMQ 而不是 Ray 的远程调用？】                               │
│  - 性能：ZMQ 是高性能的消息队列，延迟极低                              │
│  - 集成：vLLM 内部使用 ZMQ 进行分布式通信                              │
│  - 灵活：支持多种通信模式（REQ-REP、PUB-SUB 等）                       │
│                                                                          │
│  【通信流程】                                                            │
│                                                                          │
│  1. Worker 启动时，每个 Worker 创建 ZMQ socket 并监听                   │
│     → worker.get_zeromq_address() 返回 "tcp://192.168.1.100:5555"      │
│                                                                          │
│  2. HTTP Server 启动时，获取所有 Workers 的 ZMQ 地址                    │
│     → zmq_addresses = ["tcp://...:5555", "tcp://...:5556", ...]         │
│                                                                          │
│  3. HTTP Server 创建 ExternalZeroMQDistributedExecutor                  │
│     → 连接到所有 Workers 的 ZMQ socket                                 │
│                                                                          │
│  4. 收到推理请求时：                                                     │
│     Server → ZMQ → Workers（执行推理）→ ZMQ → Server → HTTP 响应       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │            HTTP Server (vLLMHttpServer)                           │   │
│  │                                                                   │   │
│  │   ┌─────────────────────────────────────────────────────────┐    │   │
│  │   │     ExternalZeroMQDistributedExecutor                    │    │   │
│  │   │                                                          │    │   │
│  │   │   self.sockets = [                                       │    │   │
│  │   │       zmq.socket → tcp://...:5555 (Worker 0)            │    │   │
│  │   │       zmq.socket → tcp://...:5556 (Worker 1)            │    │   │
│  │   │   ]                                                      │    │   │
│  │   │                                                          │    │   │
│  │   │   collective_rpc("execute_model", args=(...))           │    │   │
│  │   │   → 向所有 Workers 发送命令，收集结果                   │    │   │
│  │   └─────────────────────────────────────────────────────────┘    │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.7 launch_server() 内部细节

HTTP Server 启动的核心逻辑在 `vLLMHttpServerBase.launch_server()` 中：

```python
# 源文件：verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:242-408

async def launch_server(self, master_address: str = None, master_port: int = None):
    # === 1. 配置 vLLM CLI 参数 ===
    args = {
        "dtype": self.config.dtype,
        "load_format": self.config.load_format,  # 重要！Standalone 必须是 "auto"
        "tensor_parallel_size": self.config.tensor_model_parallel_size,
        "max_model_len": self.config.max_model_len,
        # ... 更多参数
    }

    # === 2. 设置分布式执行后端 ===
    # 使用 ZMQ 与 Workers 通信
    distributed_executor_backend = ExternalZeroMQDistributedExecutor

    # 获取所有 Workers 的 ZMQ 地址
    zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in self.workers])
    os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

    # === 3. 启动 vLLM 引擎和 HTTP 服务 ===
    await self.run_server(server_args)

async def run_server(self, args):
    # 创建 vLLM 异步引擎
    engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, ...)

    # 构建 FastAPI 应用（OpenAI 兼容 API）
    app = build_app(args)
    await init_app_state(engine_client, app.state, args)

    # 保存引擎引用
    self.engine = engine_client

    # 启动 Uvicorn HTTP 服务器
    self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)
```

### 4.8 完整架构图

现在让我们看 `init_standalone()` 完成后的完整架构：

```
┌─────────────────────────────────────────────────────────────────────────┐
│             init_standalone() 完成后的 Replica 架构                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  假设：TP=2，单节点                                                     │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Replica 0                                     │   │
│  │                                                                   │   │
│  │   外部 API 调用                                                   │   │
│  │   POST http://192.168.1.100:8000/v1/chat/completions             │   │
│  │          │                                                        │   │
│  │          ▼                                                        │   │
│  │   ┌──────────────────────────────────────────────────────────┐   │   │
│  │   │           vLLMHttpServer (Ray Actor)                      │   │   │
│  │   │           - HTTP 端口：8000                               │   │   │
│  │   │           - OpenAI 兼容 API                               │   │   │
│  │   │           - vLLM AsyncLLM 引擎                           │   │   │
│  │   │                                                           │   │   │
│  │   │   ┌──────────────────────────────────────────────────┐   │   │   │
│  │   │   │     ExternalZeroMQDistributedExecutor            │   │   │   │
│  │   │   │     ZMQ sockets → Workers                        │   │   │   │
│  │   │   └──────────────────────────────────────────────────┘   │   │   │
│  │   └──────────────────────────┬───────────────────────────────┘   │   │
│  │                              │                                    │   │
│  │                              │ ZMQ REQ-REP                        │   │
│  │                              │                                    │   │
│  │          ┌───────────────────┼───────────────────┐               │   │
│  │          │                   │                   │               │   │
│  │          ▼                   ▼                                   │   │
│  │   ┌─────────────────┐ ┌─────────────────┐                       │   │
│  │   │   Worker 0      │ │   Worker 1      │                       │   │
│  │   │   (Ray Actor)   │ │   (Ray Actor)   │                       │   │
│  │   │   GPU 0         │ │   GPU 1         │                       │   │
│  │   │   ZMQ :5555     │ │   ZMQ :5556     │                       │   │
│  │   │                 │ │                 │                       │   │
│  │   │   模型分片 0    │◄┼►│   模型分片 1    │  NCCL                │   │
│  │   │   (权重加载自   │ │   (权重加载自   │  集合通信            │   │
│  │   │    磁盘)        │ │    磁盘)        │                       │   │
│  │   └─────────────────┘ └─────────────────┘                       │   │
│  │                                                                   │   │
│  │   资源池：rollout_pool_0 (Placement Group, 独占 GPU 0-1)         │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.9 单节点 vs 多节点

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    单节点 vs 多节点的 HTTP Server                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【单节点】（常见场景）                                                  │
│  - 只有 1 个 HTTP Server                                                │
│  - 所有 Workers 在同一台机器                                            │
│                                                                          │
│  【多节点】（大模型跨机场景）                                            │
│  - 每个节点有 1 个 HTTP Server                                          │
│  - 节点 0 是 Master，提供对外 API                                       │
│  - 其他节点是 Headless，参与分布式推理                                  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                 多节点架构示例 (2 节点，TP=16)                    │    │
│  │                                                                   │    │
│  │   Node 0 (Master)                     Node 1 (Headless)          │    │
│  │   ┌───────────────────┐               ┌───────────────────┐      │    │
│  │   │  HTTP Server 0    │               │  HTTP Server 1    │      │    │
│  │   │  (对外提供 API)   │◄──────────────►│  (无 HTTP 端口)   │      │    │
│  │   │  端口 8000        │   RPC 通信    │  Headless 模式    │      │    │
│  │   └────────┬──────────┘               └────────┬──────────┘      │    │
│  │            │                                   │                  │    │
│  │   Workers 0-7                         Workers 8-15               │    │
│  │   GPU 0-7                             GPU 0-7                    │    │
│  │                                                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 问题检查点

在继续之前，请确认你理解：

1. HTTP Server 的作用是什么？
   （提供 OpenAI 兼容的 API，接收外部请求，协调 Workers 执行推理）

2. HTTP Server 和 Workers 如何通信？
   （通过 ZeroMQ，而不是 Ray 的远程调用）

3. `launch_servers()` 做了什么？
   （创建 vLLMHttpServer Actor，启动 vLLM 引擎，启动 Uvicorn HTTP 服务）

4. `self._server_address` 的作用？
   （保存 HTTP Server 的地址，供外部调用，如 "192.168.1.100:8000"）

5. 为什么要使用 NodeAffinitySchedulingStrategy？
   （确保 HTTP Server 和它的 Workers 在同一个物理节点上，减少网络延迟）

**理解了以上内容，你就完整掌握了 `init_standalone()` 的整个流程！**

---

