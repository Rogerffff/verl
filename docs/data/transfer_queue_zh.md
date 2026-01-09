# TransferQueue 数据系统

最后更新：2026年1月7日

本文档介绍 [TransferQueue](https://gitcode.com/Ascend/TransferQueue)，这是一个用于高效后训练的异步流式数据管理系统。

🔥 **TransferQueue 现已在 [GitCode](https://gitcode.com/Ascend/TransferQueue) 正式开源。我们很快将提供 [Github 镜像仓库](https://github.com/Ascend/TransferQueue) 以便社区贡献。<span style="color: #FF0000;">欢迎在任一平台提交贡献或提出新想法！**</span>


> 与此同时，早期开发历史仍可在此访问：https://github.com/TransferQueue/TransferQueue。

<h2 id="overview"> 概述</h2>

TransferQueue 是一个高性能的数据存储和传输模块，具有全景数据可见性和流式调度能力，专为后训练工作流中的高效数据流而优化。

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/tq_arch.png?raw=true" width="70%">
</p>

TransferQueue 提供**细粒度的样本级**数据管理和**负载均衡**（开发中）能力，作为数据网关解耦计算任务之间的显式数据依赖。这实现了分而治之的方法，显著简化了算法控制器的设计。

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/main_func.png?raw=true" width="70%">
</p>

<h2 id="updates"> 更新日志</h2>

 - **2025年12月30日**：**TransferQueue x verl** 集成已在 DAPO 算法中进行大规模测试 **（64个节点，1024张卡）**。它显著优化了主机内存利用率并加速了数据传输。敬请期待更多细节！
 - **2025年12月20日**：🔥 正式[教程](https://github.com/TransferQueue/TransferQueue/tree/main/tutorial)发布！欢迎查看。
 - **2025年11月10日**：我们从 TransferQueueController 中解耦了数据检索逻辑 [PR#101](https://github.com/TransferQueue/TransferQueue/pull/101)。现在您可以实现自己的 `Sampler` 来控制数据消费方式。
 - **2025年11月5日**：我们提供了一个 `KVStorageManager`，简化了与基于 KV 的存储后端的集成 [PR#96](https://github.com/TransferQueue/TransferQueue/pull/96)。第一个可用的基于 KV 的后端是 [Yuanrong](https://gitee.com/openeuler/yuanrong-datasystem)。
 - **2025年11月4日**：数据分区功能在 [PR#98](https://github.com/TransferQueue/TransferQueue/pull/98) 中可用。现在您可以定义逻辑数据分区来管理训练/验证/测试数据集。
 - **2025年10月25日**：我们在 [PR#66](https://github.com/TransferQueue/TransferQueue/pull/66) 中使存储后端可插拔。现在您可以尝试将自己的存储后端与 TransferQueue 集成！
 - **2025年10月21日**：正式集成到 verl 已准备就绪 [verl/pulls/3649](https://github.com/volcengine/verl/pull/3649)。后续 PR 将通过完全解耦数据和控制流来优化单控制器架构。
 - **2025年7月22日**：我们在<a href="https://zhuanlan.zhihu.com/p/1930244241625449814">知乎 1</a>、<a href="https://zhuanlan.zhihu.com/p/1933259599953232589">2</a>上发布了一系列中文博客。
 - **2025年7月21日**：我们在 verl 社区启动了一个 RFC [verl/RFC#2662](https://github.com/volcengine/verl/discussions/2662)。
 - **2025年7月2日**：我们发布了论文 [AsyncFlow](https://arxiv.org/abs/2507.01663)。

<h2 id="components"> 组件</h2>

### 控制平面：全景数据管理

在控制平面中，`TransferQueueController` 跟踪每个训练样本的**生产状态**和**消费状态**作为元数据。当所有必需的数据字段都准备就绪（即已写入 `TransferQueueStorageManager`）时，我们知道此数据样本可以被下游任务消费。

对于消费状态，我们记录每个计算任务（例如 `generate_sequences`、`compute_log_prob` 等）的消费记录。因此，即使不同的计算任务需要相同的数据字段，它们也可以独立消费数据而不会相互干扰。

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/control_plane.png?raw=true" width="70%">
</p>

为了使数据检索过程更加可定制，我们提供了一个 `Sampler` 类，允许用户定义自己的数据检索和消费逻辑。详见[自定义](#customize)部分。

> 未来，我们计划在控制平面支持**负载均衡**和**动态批处理**能力。此外，我们将支持分布式框架的数据管理，其中每个 rank 自己管理数据检索，而不是由单个控制器协调。

### 数据平面：分布式数据存储

在数据平面中，我们提供了一个可插拔的设计，使 TransferQueue 能够根据用户需求与不同的存储后端集成。

具体来说，我们提供了一个 `TransferQueueStorageManager` 抽象类，定义了核心 API 如下：

- `async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None`
- `async def get_data(self, metadata: BatchMeta) -> TensorDict`
- `async def clear_data(self, metadata: BatchMeta) -> None`

该类封装了 TransferQueue 系统内的核心交互逻辑。您只需编写一个简单的子类即可集成自己的存储后端。详见[自定义](#customize)部分。

目前，我们支持以下存储后端：

- SimpleStorageUnit：一个基本的 CPU 内存存储，数据格式约束最少，易于使用。
- [Yuanrong](https://gitcode.com/openeuler/yuanrong-datasystem)（beta，[#PR107](https://github.com/TransferQueue/TransferQueue/pull/107)，[#PR96](https://github.com/TransferQueue/TransferQueue/pull/96)）：一个 Ascend 原生数据系统，提供包括 HBM/DRAM/SSD 的分层存储接口。
- [Mooncake Store](https://github.com/kvcache-ai/Mooncake)（alpha，[#PR162](https://github.com/TransferQueue/TransferQueue/pull/162)）：一个高性能的基于 KV 的分层存储，支持 GPU 和 DRAM 之间的 RDMA 传输。
- [Ray Direct Transport](https://docs.ray.io/en/master/ray-core/direct-transport.html)（alpha，[#PR167](https://github.com/TransferQueue/TransferQueue/pull/167)）：Ray 的新功能，允许 Ray 直接在 Ray actor 之间存储和传递对象。

其中，`SimpleStorageUnit` 作为我们的默认存储后端，由 `AsyncSimpleStorageManager` 类协调。每个存储单元可以部署在单独的节点上，实现分布式数据管理。

`SimpleStorageUnit` 采用如下二维数据结构：

- 每一行对应一个训练样本，在相应的全局批次中分配一个唯一索引。
- 每一列代表计算任务的输入/输出数据字段。

这种数据结构设计源于后训练过程的计算特性，其中每个训练样本以流水线方式在任务管道中生成。它提供了精确的寻址能力，允许以流式方式进行细粒度的并发数据读/写操作。

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/data_plane.png?raw=true" width="70%">
</p>

### 用户接口：异步和同步客户端

TransferQueue 系统的交互工作流程如下：

1. 一个进程向 `TransferQueueController` 发送读取请求。
2. `TransferQueueController` 扫描每个样本（行）的生产和消费元数据，并根据负载均衡策略动态组装一个微批次元数据。这种机制实现了样本级数据调度。
3. 该进程使用控制器提供的元数据从分布式存储单元检索实际数据。

为了简化 TransferQueue 的使用，我们将这个过程封装到 `AsyncTransferQueueClient` 和 `TransferQueueClient` 中。这些客户端为数据传输提供异步和同步接口，使用户能够轻松地将 TransferQueue 集成到他们的框架中。

> 未来，我们将为分布式框架提供一个 `StreamingDataLoader` 接口，如 [issue#85](https://github.com/TransferQueue/TransferQueue/issues/85) 和 [verl/RFC#2662](https://github.com/volcengine/verl/discussions/2662) 中所讨论的。利用这种抽象，每个 rank 可以像 PyTorch 中的 `DataLoader` 一样自动获取自己的数据。TransferQueue 系统将处理由不同并行策略引起的底层数据调度和传输逻辑，显著简化分布式框架的设计。

<h2 id="show-cases">🔥 案例展示</h2>

### 通用用法

主要交互点是 `AsyncTransferQueueClient` 和 `TransferQueueClient`，作为与 TransferQueue 系统的通信接口。

核心接口：

- `(async_)get_meta(data_fields: list[str], batch_size:int, partition_id: str, mode: str, task_name:str, sampling_config: Optional[dict[str, Any]]) -> BatchMeta`
- `(async_)get_data(metadata: BatchMeta) -> TensorDict`
- `(async_)put(data: TensorDict, metadata: Optional[BatchMeta], partition_id: Optional[str])`
- `(async_)clear_partition(partition_id: str)` 和 `(async_)clear_samples(metadata: BatchMeta)`

<span style="color: #FF0000;">**详细示例请参考我们的[教程](https://github.com/TransferQueue/TransferQueue/tree/main/tutorial)。**</span>


### verl 示例

现在将 TransferQueue 集成到 verl 的主要动机是**缓解单控制器 `RayPPOTrainer` 的数据传输瓶颈**。目前，所有 `DataProto` 对象都必须通过 `RayPPOTrainer` 路由，导致整个后训练系统的单点瓶颈。

![verl_dataflow_DataProto](https://github.com/TransferQueue/community_doc/blob/main/docs/verl_workflow.jpeg?raw=true)


利用 TransferQueue，我们通过以下方式将经验数据传输与元数据分发分离：

- 用 `BatchMeta`（元数据）和 `TensorDict`（实际数据）结构替换 `DataProto`
- 通过 BatchMeta 保留 verl 原有的分发/收集逻辑（保持单控制器可调试性）
- 通过 TransferQueue 的分布式存储单元加速数据传输

![verl_dataflow_TransferQueue](https://github.com/TransferQueue/community_doc/blob/main/docs/verl_workflow_with_tq.jpeg?raw=true)


您可以参考[配方](https://github.com/TransferQueue/TransferQueue/tree/dev/recipe/simple_use_case)，其中我们在异步和同步场景中模拟了 verl 的使用。verl 的正式集成现在也可以在 [verl/pulls/3649](https://github.com/volcengine/verl/pull/3649) 获得（后续 PR 将进一步优化集成）。


### 使用 Python 包
```bash
pip install TransferQueue
```

### 从源代码构建 wheel 包

按照以下步骤构建和安装：
1. 从 GitHub 仓库克隆源代码
   ```bash
   git clone https://github.com/TransferQueue/TransferQueue/
   cd TransferQueue
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 构建和安装
   ```bash
   python -m build --wheel
   pip install dist/*.whl
   ```

<h2 id="performance">📊 性能</h2>

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/performance_0.1.1.dev2.png?raw=true" width="100%">
</p>

> 注意：上述 TransferQueue 的基准测试基于我们简单的 `SimpleStorageUnit` 后端。通过引入高性能存储后端并优化序列化/反序列化，我们期望实现更好的性能。热烈欢迎社区贡献！

有关详细的性能基准测试，请参考[此博客](https://www.yuque.com/haomingzi-lfse7/hlx5g0/tml8ke0zkgn6roey?singleDoc#)。

我们还提供了一个[压力测试报告](https://www.yuque.com/haomingzi-lfse7/hlx5g0/ydbwgo5k2umaag78?singleDoc#)，演示了**768个并发客户端在4个节点上向 TransferQueue 写入1.4 TB 数据**。系统保持稳定，没有任何崩溃或数据丢失，实现了80%的带宽利用率。

<h2 id="customize"> 🛠️ 自定义 TransferQueue</h2>

### 定义您自己的数据检索逻辑
我们提供了一个 `BaseSampler` 抽象类，定义了以下接口：

```python3
@abstractmethod
def sample(
    self,
    ready_indexes: list[int],
    batch_size: int,
    *args: Any,
    **kwargs: Any,
) -> tuple[list[int], list[int]]:
    """从就绪索引中采样一批索引。

    参数：
        ready_indexes: 全局索引列表，对应样本的所有必需字段都已生成，
        且样本在相应任务中未被标记为已消费。
        batch_size: 要选择的样本数量
        *args: 特定采样器实现的额外位置参数
        **kwargs: 特定采样器实现的额外关键字参数

    返回：
        长度为 batch_size 的采样全局索引列表
        长度为 batch_size 的全局索引列表，应标记为已消费
        （将来永远不会被检索）

    异常：
        ValueError: 如果 batch_size 无效或 ready_indexes 不足
    """
    raise NotImplementedError("子类必须实现 sample")
```

在这个设计中，我们通过两个返回值分离数据检索和数据消费，这使我们能够轻松控制样本替换。我们已经实现了两个参考设计：`SequentialSampler` 和 `GRPOGroupNSampler`。

`Sampler` 类或实例应在初始化期间传递给 `TransferQueueController`。在每次 `get_meta` 调用期间，您可以向 `Sampler` 提供动态采样参数。

```python3
from transfer_queue import TransferQueueController, TransferQueueClient, GRPOGroupNSampler, process_zmq_server_info

# 选项 1：将采样器类传递给 TransferQueueController
controller = TransferQueueController.remote(GRPOGroupNSampler)

# 选项 2：将采样器实例传递给 TransferQueueController（如果需要自定义配置）
your_own_sampler = YourOwnSampler(config)
controller = TransferQueueController.remote(your_own_sampler)

# 使用采样器
batch_meta = client.get_meta(
    data_fields=["input_ids", "attention_mask"],
    batch_size=8,
    partition_id="train_0",
    task_name="generate_sequences",
    sampling_config={"n_samples_per_prompt": 4}  # 在这里放置所需的采样参数
)
```

<span style="color: #FF0000;">**更多细节请参考[tutorial/04_custom_sampler.py](https://github.com/TransferQueue/TransferQueue/blob/main/tutorial/04_custom_sampler.py)。**</span>


### 如何集成新的存储后端

数据平面的组织结构如下：
```text
  transfer_queue/
  ├── storage/
  │   ├── __init__.py
  │   │── simple_backend.py             # TQ 的默认分布式存储后端（SimpleStorageUnit）
  │   ├── managers/                     # Managers 是封装与 TQ 系统交互逻辑的上层接口
  │   │   ├── __init__.py
  │   │   ├──base.py                    # TransferQueueStorageManager, KVStorageManager
  │   │   ├──simple_backend_manager.py  # AsyncSimpleStorageManager
  │   │   ├──yuanrong_manager.py        # YuanrongStorageManager
  │   │   ├──mooncake_manager.py        # MooncakeStorageManager
  │   │   └──factory.py                 # TransferQueueStorageManagerFactory
  │   └── clients/                      # Clients 是直接操作目标存储后端的下层接口
  │   │   ├── __init__.py
  │   │   ├── base.py                   # TransferQueueStorageKVClient
  │   │   ├── yuanrong_client.py        # YuanrongStorageClient
  │   │   ├── mooncake_client.py        # MooncakeStorageClient
  │   │   ├── ray_storage_client.py     # RayStorageClient
  │   │   └── factory.py                # TransferQueueStorageClientFactory
```

要将 TransferQueue 与自定义存储后端集成，首先实现一个继承自 `TransferQueueStorageManager` 的子类。该子类充当 TransferQueue 系统和目标存储后端之间的适配器。对于基于 KV 的存储后端，您可以简单地继承 `KVStorageManager`，它可以作为所有基于 KV 的后端的通用管理器。

分布式存储后端通常带有自己的原生客户端作为存储系统的接口。在这种情况下，可以为这个客户端编写一个底层适配器，遵循 `storage/clients` 目录中提供的示例。

为 `StorageManager` 和 `StorageClient` 都提供了工厂类，以便于轻松集成。在工厂类中添加必要参数的描述有助于增强整体用户体验。

<h2 id="contribution"> ✏️ 贡献指南</h2>

<span style="color: #FF0000;">**热烈欢迎贡献！**</span>

欢迎新想法、功能建议和用户体验反馈——随时提交 issue 或 PR。我们会尽快回复。

我们推荐使用 pre-commit 以获得更好的代码格式。

```bash
# 安装 pre-commit
pip install pre-commit

# 在您的仓库文件夹中运行以下命令，然后在提交代码之前修复检查
pre-commit install && pre-commit run --all-files --show-diff-on-failure --color=always
```


<h2 id="citation"> 引用</h2>
如果您发现这个仓库有用，请引用我们的论文：

```bibtex
@article{han2025asyncflow,
  title={AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training},
  author={Han, Zhenyu and You, Ansheng and Wang, Haibo and Luo, Kui and Yang, Guang and Shi, Wenqi and Chen, Menglong and Zhang, Sicheng and Lan, Zeshun and Deng, Chunshi and others},
  journal={arXiv preprint arXiv:2507.01663},
  year={2025}
}
```

