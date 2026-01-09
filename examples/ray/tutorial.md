# VeRL Ray API 教程

本教程将帮助你理解 verl 中如何使用 Ray 进行分布式计算。

**前置知识**：
- Python 基础
- PyTorch 基础

**学习目标**：
- 理解 Ray 的基本概念（远程函数、Actor）
- 掌握 RayResourcePool 和 RayWorkerGroup 的使用
- 了解数据分发（Dispatch）和收集（Collection）机制
- 学习 Megatron 并行的集成方式

---

## 第一章：Ray 基础

Ray 是一个用于构建分布式应用的开源框架。它的核心概念包括：

1. **远程函数 (Remote Functions)**：可以在集群中任意节点上执行的函数
2. **Actor**：有状态的远程对象，可以保持内部状态并响应方法调用
3. **Object Store**：分布式内存存储，用于在节点间共享数据

在 verl 中，我们主要使用 Ray 的 Actor 模式来管理 Worker（如 Actor Worker、Critic Worker 等）。

```python
import os
```

```python
import warnings

import ray
import torch

warnings.filterwarnings("ignore")
```

```python
# 启动本地 Ray 集群
# Head 节点和 Worker 节点都在本机上
ray.init()
```

### 1.1 Ray Actor 示例

下面实现一个简单的累加器类。

**关键点**：
- `@ray.remote` 装饰器将普通类转换为 Ray Actor
- Actor 是一个独立的进程，可以保持状态
- 调用 Actor 方法使用 `.remote()` 后缀

```python
@ray.remote
class Accumulator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x

    def get_value(self):
        return self.value
```

```python
# 实例化一个累加器
# Accumulator 可以看作是一个进程，充当 RPC 服务
accumulator = Accumulator.remote()
```

```python
# 查看当前值
# 注意：.remote() 调用会立即返回，不会等待远程执行完成
# 返回的是一个 ObjectRef（对象引用）
value_ref = accumulator.get_value.remote()

# 使用 ray.get() 获取实际值（这会阻塞直到远程执行完成）
value = ray.get(value_ref)
print(value)
```

```python
# 执行累加操作，然后查看结果
# 同样，这里的 add 也会立即返回
accumulator.add.remote(10)

# 获取新值
new_value = ray.get(accumulator.get_value.remote())
print(new_value)
```

---

## 第二章：资源池（Resource Pool）和 RayWorkerGroup

在上一个例子中，我们使用了简单的单进程 Worker。

在实际的 RL 训练中，我们需要：
1. **多 GPU 并行**：每个 Worker 绑定一个 GPU
2. **Worker 分组**：将多个 Worker 组织成一个 WorkerGroup 进行协同工作
3. **资源管理**：灵活分配和复用 GPU 资源

verl 提供了 `RayResourcePool` 和 `RayWorkerGroup` 来满足这些需求。

### 核心概念

- **RayResourcePool**：GPU 资源池，定义可用的 GPU 资源
- **RayWorkerGroup**：Worker 组，将多个 Worker 映射到资源池上
- **Worker**：verl 中的基础 Worker 类，继承后实现具体功能

```python
from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, merge_resource_pool
```

```python
resource_pool = RayResourcePool([4], use_gpu=True)
```

```python
@ray.remote
class GPUAccumulator(Worker):
    def __init__(self) -> None:
        super().__init__()
        # The initial value of each rank is the same as the rank
        # 每个 rank 的初始值与其 rank 相同
        self.value = torch.zeros(size=(1,), device="cuda") + self.rank

    def add(self, x):
        self.value += x
        print(f"rank {self.rank}, value: {self.value}")
        return self.value.cpu()
```

```python
# Each worker's initial value is its rank, and then each rank's value is incremented by 1, so the values obtained on each rank are [1, 2, 3, 4]
# 每个 worker 的初始值是它的 rank，然后每个 rank 的值加 1，所以每个 rank 上得到的值是 [1, 2, 3, 4]
class_with_args = RayClassWithInitArgs(cls=GPUAccumulator)
worker_group = RayWorkerGroup(resource_pool, class_with_args)
print(worker_group.execute_all_sync("add", x=[1, 1, 1, 1]))
```

参数传递原理：输入参数是一个长度为 world_size 的列表，列表中的每个元素分别分发给 RayWorkerGroup 中的每个 worker。
返回参数也是一个列表，对应每个 worker 的返回值。

### GPU 资源共享

映射到同一资源池的 RayWorkerGroups 共享 GPU。在这个例子中，我们实现了三个资源池：第一个占用 4 个 GPU，第二个也占用 4 个 GPU，最后一个占用所有 8 个 GPU。其中，第一个资源池复用了上面提到的资源池。

```python
# Create a new resource pool and then merge the newly created resource pool with the previous one.
# 创建一个新的资源池，然后将新创建的资源池与前一个合并。
resource_pool_1 = RayResourcePool([4], use_gpu=True, name_prefix="a")
resource_pool_merge = merge_resource_pool(resource_pool, resource_pool_1)
```

```python
# Establish a RayWorkerGroup on the newly created resource pool.
# 在新创建的资源池上建立一个 RayWorkerGroup。
worker_group_1 = RayWorkerGroup(resource_pool_1, class_with_args)
worker_group_merge = RayWorkerGroup(resource_pool_merge, class_with_args)
```

```python
# Run 'add' on the second set of 4 GPUs; the result should be [2, 3, 4, 5].
# 在第二组 4 个 GPU 上运行 'add'；结果应该是 [2, 3, 4, 5]。
output_1 = worker_group_1.execute_all_sync("add", x=[2, 2, 2, 2])
print(output_1)
```

```python
# Run 'add' on the merged set of 8 GPUs; the result should be [3, 4, 5, 6, 7, 8, 9, 10].
# 在合并后的 8 个 GPU 上运行 'add'；结果应该是 [3, 4, 5, 6, 7, 8, 9, 10]。
output_merge = worker_group_merge.execute_all_sync("add", x=[3, 3, 3, 3, 3, 3, 3, 3])
print(output_merge)
```

```python
print(worker_group.world_size, worker_group_1.world_size, worker_group_merge.world_size)
```

---

## 第三章：数据分发、执行和收集

在上面的例子中，我们使用 RayWorkerGroup 中的 `execute_all_sync` 函数将数据从 driver 分发到每个 worker。这对于编码来说非常不方便。
在本章中，我们使用函数装饰器的形式，允许 RayWorkerGroup 直接调用在 Worker 中编写的函数，并大大简化参数传递。

```python
from verl.single_controller.base.decorator import Dispatch, Execute, register
```

```python
@ray.remote
class GPUAccumulatorDecorator(Worker):
    def __init__(self) -> None:
        super().__init__()
        # The initial value of each rank is the same as the rank
        # 每个 rank 的初始值与其 rank 相同
        self.value = torch.zeros(size=(1,), device="cuda") + self.rank

    # map from a single input to all the worker
    # 将单个输入映射到所有 worker
    @register(Dispatch.ONE_TO_ALL)
    def add(self, x):
        print(x)
        self.value = self.value + x
        print(f"rank {self.rank}, value: {self.value}")
        return self.value.cpu()
```

```python
class_with_args = RayClassWithInitArgs(cls=GPUAccumulatorDecorator)
gpu_accumulator_decorator = RayWorkerGroup(resource_pool_merge, class_with_args)
```

```python
# As we can see, 10 is automatically dispatched to each Worker in this RayWorkerGroup.
# 可以看到，10 被自动分发到这个 RayWorkerGroup 中的每个 Worker。
print(gpu_accumulator_decorator.add(x=10))
```

### 自定义分发和收集

用户可以自定义 `dispatch` 和 `collection` 函数。你只需要自己编写 `dispatch_fn` 和 `collect_fn` 函数即可。我们还支持仅在 rank_zero 上执行 RPC，具体示例如下。

```python
from verl.single_controller.base.decorator import Dispatch, collect_all_to_all, register
```

```python
def two_to_all_dispatch_fn(worker_group, *args, **kwargs):
    """
    Assume the input is a list of 2. Duplicate the input interleaved and pass to each worker.
    假设输入是一个长度为 2 的列表。将输入交替复制并传递给每个 worker。
    """
    for arg in args:
        assert len(arg) == 2
        for i in range(worker_group.world_size - 2):
            arg.append(arg[i % 2])
    for k, v in kwargs.items():
        assert len(v) == 2
        for i in range(worker_group.world_size - 2):
            v.append(v[i % 2])
    return args, kwargs


@ray.remote
class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    # TODO: 传递 *args 和 **kwargs 容易出错且不太可靠
    def __init__(self, x) -> None:
        super().__init__()
        self._x = x

    def foo(self, y):
        return self._x + y

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def foo_rank_zero(self, x, y):
        return self._x + y + x

    @register(dispatch_mode={"dispatch_fn": two_to_all_dispatch_fn, "collect_fn": collect_all_to_all})
    def foo_custom(self, x, y):
        return self._x + y + x
```

```python
class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)
worker_group = RayWorkerGroup(resource_pool, class_with_args)
```

```python
output_ref = worker_group.foo_custom(x=[1, 2], y=[5, 6])
assert output_ref == [8, 10, 8, 10]

output_ref = worker_group.foo_rank_zero(x=1, y=2)
assert output_ref == 5
```

```python
print(gpu_accumulator_decorator.world_size)
```

```python
# Shutdown ray cluster
# 关闭 Ray 集群
ray.shutdown()
```

---

## 第四章：NVMegatronRayWorkerGroup

由于 Ray 的问题，目前我们在 RayResourcePool 中只能支持 max_colocate_count=1。
这意味着每个 GPU 只能有一个进程。
当应用这个 pull request 后，我们可以支持 max_colocate > 1：https://github.com/ray-project/ray/pull/44385

因此，我们需要重启 Ray 并初始化一个新的 resource_pool 来演示 **NVMegatronRayWorkerGroup**

```python
# Build a local ray cluster. The head node and worker node are on this machine
# 构建一个本地 Ray 集群。Head 节点和 Worker 节点都在本机上
ray.init()
```

最后，我们实现一个 `NVMegatronRayWorkerGroup`，在其中创建一个 Megatron，然后运行一个张量并行（tp）切分的 Llama MLP 层。这里，我们使用一个复杂的分发模式 `Megatron_COMPUTE`。这个分发模式假设用户传递的数据是按 DP 维度分区的。数据被分发到同一 dp 组内的所有 tp/pp rank，最终只从 tp=0 和最后一个 pp 收集输出数据。这样，对于只在 driver 上编写代码的用户来说，RPC 后面的 Megatron 变得透明了。

```python
import sys

current_pythonpath = os.environ.get("PYTHONPATH", "")

new_path = "/opt/tiger/Megatron-LM"

new_pythonpath = f"{new_path}:{current_pythonpath}" if current_pythonpath else new_path

os.environ["PYTHONPATH"] = new_pythonpath

print(new_path)
sys.path.append(new_path)

import megatron

print(megatron.__file__)
```

```python
from megatron.core import parallel_state as mpu
from omegaconf import OmegaConf

from verl.single_controller.base.decorator import Dispatch, Execute, register
from verl.single_controller.base.megatron.worker import MegatronWorker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
```

```python
resource_pool = RayResourcePool([4], use_gpu=True, max_colocate_count=1)
```

```python
@ray.remote
class MLPLayerWorker(MegatronWorker):
    def __init__(self):
        super().__init__()
        rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            nccl_communicator_config_path=None,
        )
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(10)

    @register(Dispatch.ONE_TO_ALL)
    def init_model(self, config):
        from omegaconf import OmegaConf

        from verl.models.llama.megatron.layers import ParallelLlamaMLP
        from verl.utils.megatron_utils import init_model_parallel_config

        megatron_config = OmegaConf.create(
            {
                "sequence_parallel": False,
                "param_dtype": "fp32",
                "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
                "pipeline_model_parallel_rank": mpu.get_pipeline_model_parallel_rank(),
                "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
                "virtual_pipeline_model_parallel_rank": mpu.get_virtual_pipeline_model_parallel_rank(),
                "virtual_pipeline_model_parallel_size": mpu.get_virtual_pipeline_model_parallel_world_size(),
            }
        )

        megatron_config = init_model_parallel_config(megatron_config)
        self.parallel_layer = ParallelLlamaMLP(config=config, megatron_config=megatron_config)

    @register(Dispatch.ONE_TO_ALL)
    def get_weights(self):
        output = {}
        for key, val in self.parallel_layer.named_parameters():
            output[key] = val
        return output

    @register(Dispatch.MEGATRON_COMPUTE)
    def run_layer(self, x):
        x = x.to("cuda")
        y = self.parallel_layer(x)
        return y
```

```python
layer_cls = RayClassWithInitArgs(cls=MLPLayerWorker)
layer_worker_group = NVMegatronRayWorkerGroup(
    resource_pool=resource_pool,
    ray_cls_with_init=layer_cls,
)
```

```python
print(layer_worker_group.world_size, layer_worker_group.tp_size, layer_worker_group.pp_size, layer_worker_group.dp_size)
```

```python
ffn_hidden_size = 11008
batch_size = 16
seq_len = 2048
hidden_size = 4096

config = OmegaConf.create(
    {
        "hidden_size": hidden_size,
        "intermediate_size": ffn_hidden_size,
        "hidden_act": "silu",
        "pretraining_tp": 1,
        "tp": layer_worker_group.tp_size,
    }
)
```

```python
x = torch.rand(size=(seq_len, batch_size, hidden_size), dtype=torch.float32)
```

```python
layer_worker_group.init_model(config)
```

```python
output = layer_worker_group.run_layer(
    [x]
)  # This must be a list of size 1, ensuring that the input equals the data parallel (dp).
# 这必须是一个大小为 1 的列表，确保输入等于数据并行度（dp）。
print(output[0].shape)
```

```python
# Shutdown ray cluster
# 关闭 Ray 集群
ray.shutdown()
```

