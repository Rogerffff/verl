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
verl Standalone Rollout 模式的官方参考实现

=== 文件用途 ===
这是 verl 提供的批量生成脚本示例，展示如何使用 Standalone Rollout 模式进行大规模推理。
Phase 0 评测脚本可以直接参考本文件的实现方式。

=== 核心流程 ===
1. 初始化 Ray 集群
2. 创建多个 Replica（推理服务器）
3. 每个 Replica 调用 init_standalone() 初始化
4. 通过 HTTP API 向 Replica 发送请求
5. 收集生成结果并保存

=== Python 异步编程基础 ===

【什么是异步编程？】
同步编程：一个任务完成后才能开始下一个（排队买奶茶，一个一个来）
异步编程：多个任务可以"同时"进行（点餐后等待，期间可以做其他事）

【关键概念】
1. async def：定义一个异步函数（协程函数）
   - 普通函数：def foo(): return 1
   - 异步函数：async def foo(): return 1

2. await：等待一个异步操作完成
   - 只能在 async def 函数内使用
   - await some_async_func() 表示"等这个操作完成再继续"

3. asyncio.gather()：并发执行多个异步任务
   - results = await asyncio.gather(task1, task2, task3)
   - 三个任务"同时"执行，全部完成后返回结果列表

4. asyncio.run()：运行异步函数（从同步代码进入异步世界）
   - asyncio.run(async_function()) 启动事件循环并运行

【为什么用异步？】
发送 HTTP 请求时，大部分时间在等待网络响应。
- 同步方式：发请求 → 等待 → 收响应 → 发下一个请求 → 等待...
- 异步方式：发请求1 → 发请求2 → 发请求3 → 等待 → 收响应1,2,3

异步可以在等待时发送更多请求，大幅提高吞吐量。

=== 使用方式 ===
python -m verl.trainer.main_generation_server \
    actor_rollout_ref.model.path=/path/to/model \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.n_gpus_per_node=8 \
    data.train_files=/path/to/prompts.parquet \
    data.output_path=/path/to/output.parquet
"""

import os

import aiohttp  # 异步 HTTP 客户端库（比 requests 更适合异步场景）
import hydra  # 配置管理库，支持 YAML 配置和命令行覆盖
import numpy as np
import ray  # 分布式计算框架

# 设置环境变量（在导入其他模块前设置）
os.environ["NCCL_DEBUG"] = "WARN"  # NCCL 调试级别，WARN 减少日志输出
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 允许 tokenizer 多线程
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import asyncio  # Python 异步编程核心库
from pprint import pprint  # 格式化打印（Pretty Print）

import pandas as pd  # 数据处理库
from omegaconf import OmegaConf  # Hydra 的配置对象
from openai.types.chat import ChatCompletion  # OpenAI API 响应类型

from verl.utils.hdfs_io import makedirs  # 支持 HDFS 的目录创建
from verl.workers.rollout.replica import get_rollout_replica_class  # 获取 Replica 类


async def start_server(config):
    """
    启动多个 Replica（推理服务器）

    === 函数说明 ===
    这是 Standalone Rollout 模式的核心启动函数。
    它会根据 GPU 数量和 TP 大小计算需要多少个 Replica，然后创建并初始化它们。

    === 参数 ===
    config: Hydra 配置对象，包含：
        - config.actor_rollout_ref.rollout.tensor_model_parallel_size: TP 大小
        - config.trainer.n_gpus_per_node: 每节点 GPU 数
        - config.trainer.nnodes: 节点数
        - config.actor_rollout_ref.rollout.name: 推理引擎名称 ("vllm" 或 "sglang")
        - config.actor_rollout_ref.model: 模型配置

    === 返回值 ===
    (server_handles, server_addresses): 元组
        - server_handles: 所有 Replica 的 Ray Actor 句柄列表
        - server_addresses: 所有 Replica 的 HTTP 地址列表

    === 示例 ===
    假设：8 GPU，TP=2
    → num_replicas = 8 / 2 = 4
    → 创建 4 个 Replica，每个使用 2 个 GPU
    → 返回 4 个服务器地址，如 ["192.168.1.1:8000", "192.168.1.1:8001", ...]
    """
    # === Step 1: 计算 Replica 数量 ===
    # TP (Tensor Parallel) 大小：模型切分到多少个 GPU
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size

    # 计算可以启动多少个 Replica
    # 公式：num_replicas = 总 GPU 数 / 每个 Replica 需要的 GPU 数
    # 例如：8 GPU / TP=2 = 4 个 Replica
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size

    # 获取配置
    rollout_config = config.actor_rollout_ref.rollout  # 推理配置（温度、max_tokens 等）
    model_config = config.actor_rollout_ref.model  # 模型配置（路径、dtype 等）

    # === Step 2: 获取 Replica 类 ===
    # get_rollout_replica_class("vllm") → 返回 vLLMReplica 类
    # get_rollout_replica_class("sglang") → 返回 SGLangReplica 类
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)

    # === Step 3: 创建所有 Replica 实例 ===
    # 使用列表推导式创建多个 Replica
    # replica_rank 从 0 开始，用于区分不同的 Replica
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,  # 第几个 Replica（0, 1, 2, 3）
            config=rollout_config,  # 推理配置
            model_config=model_config,  # 模型配置
            gpus_per_node=config.trainer.n_gpus_per_node,  # 每节点 GPU 数（用于计算资源分配）
        )
        for replica_rank in range(num_replicas)  # 循环创建 num_replicas 个
    ]
    # 此时只是创建了 Python 对象，还没有分配 GPU 资源

    # === Step 4: 并发初始化所有 Replica ===
    # asyncio.gather() 并发执行多个异步任务
    # 每个 server.init_standalone() 会：
    #   1. 创建 Ray Placement Group（申请 GPU 资源）
    #   2. 创建 WorkerGroup（启动 Ray Actor）
    #   3. 启动 HTTP 服务器
    #
    # 【异步编程说明】
    # await asyncio.gather(task1, task2, task3) 的含义：
    # - 同时启动 task1, task2, task3
    # - 等待所有任务完成
    # - 如果是同步方式，需要 task1 完成 → task2 完成 → task3 完成（串行）
    # - 异步方式可以并发执行，节省时间
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])
    # * 是解包操作符，把列表 [a, b, c] 解包成 a, b, c
    # 等价于：await asyncio.gather(server0.init_standalone(), server1.init_standalone(), ...)

    # === Step 5: 获取服务器信息 ===
    # 每个 Replica 初始化后会设置 _server_handle 和 _server_address
    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]

    # 验证数量正确
    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    return server_handles, server_addresses


async def submit_request(server_address, **chat_complete_request):
    """
    向单个 Replica 发送一个 HTTP 请求

    === 函数说明 ===
    这是最底层的请求函数，向指定的服务器地址发送 OpenAI 格式的 chat completion 请求。

    === 参数 ===
    server_address: str, 服务器地址，如 "192.168.1.1:8000"
    **chat_complete_request: 请求参数（使用 ** 接收关键字参数）
        - model: 模型路径
        - messages: 对话消息列表
        - temperature: 采样温度
        - max_tokens: 最大生成长度
        - 等等...

    === 返回值 ===
    ChatCompletion: OpenAI 格式的响应对象

    === 示例 ===
    response = await submit_request(
        "192.168.1.1:8000",
        model="/path/to/model",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100
    )
    print(response.choices[0].message.content)  # 模型生成的文本

    === 异步 HTTP 请求说明 ===
    使用 aiohttp 而不是 requests，因为：
    - requests 是同步库，发请求时会阻塞整个程序
    - aiohttp 是异步库，发请求后可以做其他事，响应回来时再处理

    async with session.post(...) as resp:
        这是异步上下文管理器，自动处理连接的打开和关闭
    """
    try:
        # 提取额外的请求头（如果有的话）
        extra_headers = chat_complete_request.pop("extra_headers", {})

        # 创建 HTTP 会话
        # timeout=None 表示不设置超时（等待直到响应）
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)

        # 发送 POST 请求到 OpenAI 兼容的 API 端点
        # /v1/chat/completions 是 OpenAI API 的标准路径
        async with session.post(
            url=f"http://{server_address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},  # 认证头（vLLM 需要）
            json=chat_complete_request,  # 请求体（JSON 格式）
        ) as resp:
            # 解析 JSON 响应
            data = await resp.json()
            # 转换为 ChatCompletion 对象（OpenAI SDK 的类型）
            return ChatCompletion(**data)
    finally:
        # 确保关闭会话，释放连接资源
        await session.close()


async def generate_per_replica(server_address, model_path: str, n_samples: int, sampling_params: dict, chat_lst: list):
    """
    使用单个 Replica 生成多个请求的响应

    === 函数说明 ===
    这是中间层函数，负责：
    1. 把一批对话消息转换成请求
    2. 对每个消息生成 n_samples 个样本
    3. 并发发送所有请求并收集结果

    === 参数 ===
    server_address: str, 目标 Replica 的地址
    model_path: str, 模型路径（用于请求中的 model 字段）
    n_samples: int, 每个 prompt 生成几个样本
    sampling_params: dict, 采样参数 {"temperature": 0.7, "max_tokens": 100, ...}
    chat_lst: list, 对话消息列表，每个元素是一个 messages 列表
        例如：[
            [{"role": "user", "content": "问题1"}],
            [{"role": "user", "content": "问题2"}],
        ]

    === 返回值 ===
    list[ChatCompletion]: 响应列表

    === 示例 ===
    假设 chat_lst = [消息1, 消息2], n_samples = 3
    → 生成 6 个请求：消息1×3 + 消息2×3
    → 返回 6 个响应

    === 并发请求的实现 ===
    使用 asyncio.gather() 同时发送所有请求，而不是一个一个发送。
    这样可以充分利用网络带宽，大幅提高吞吐量。

    【对比】
    同步方式（假设每个请求 1 秒）：
    请求1 → 等待1秒 → 请求2 → 等待1秒 → ... → 总共 6 秒

    异步方式：
    同时发送请求1,2,3,4,5,6 → 等待 → 几乎同时收到所有响应 → 约 1 秒
    """
    # 这里注释掉了使用 OpenAI SDK 的方式（也可以用，但 aiohttp 更灵活）
    # client = AsyncOpenAI(
    #     api_key="123-abc",
    #     base_url=f"http://{server_address}/v1",
    # )

    # === Step 1: 构建所有请求 ===
    # 使用嵌套列表推导式，为每个消息生成 n_samples 个请求
    chat_complete_request = [
        {
            "model": model_path,
            "messages": messages,
            **sampling_params,  # 展开采样参数
        }
        for messages in chat_lst  # 遍历每个对话消息
        for _ in range(n_samples)  # 每个消息重复 n_samples 次
    ]
    # 例如：chat_lst=[msg1, msg2], n_samples=3
    # → [msg1请求, msg1请求, msg1请求, msg2请求, msg2请求, msg2请求]

    # === Step 2: 创建所有异步任务 ===
    # 注意：这里只是创建任务对象，还没有执行
    tasks = [submit_request(server_address, **req) for req in chat_complete_request]

    # === Step 3: 并发执行所有任务 ===
    # asyncio.gather(*tasks) 同时执行所有任务，返回结果列表
    # 结果顺序与任务顺序一致
    results = await asyncio.gather(*tasks)
    return results


async def generate(
    server_addresses: list, model_path: str, n_samples: int, sampling_params: dict, chat_numpy: np.ndarray
):
    """
    使用多个 Replica 并行生成响应（负载均衡）

    === 函数说明 ===
    这是最顶层的生成函数，负责：
    1. 把数据均匀分配给多个 Replica
    2. 并发调用所有 Replica 进行生成
    3. 收集所有结果

    === 参数 ===
    server_addresses: list[str], 所有 Replica 的地址列表
    model_path: str, 模型路径
    n_samples: int, 每个 prompt 生成几个样本
    sampling_params: dict, 采样参数
    chat_numpy: np.ndarray, 所有对话消息的数组

    === 返回值 ===
    list[list[ChatCompletion]]: 嵌套列表，外层是 Replica，内层是该 Replica 的结果

    === 示例 ===
    假设：4 个 Replica，100 条数据
    → 数据分割：[25条, 25条, 25条, 25条]
    → 每个 Replica 处理 25 条
    → 返回 [[Replica0的结果], [Replica1的结果], [Replica2的结果], [Replica3的结果]]

    === 负载均衡说明 ===
    np.array_split() 尽可能均匀分割数据。
    如果 100 条数据分给 4 个 Replica：每个 25 条
    如果 100 条数据分给 3 个 Replica：34 + 33 + 33 条
    """
    num_replicas = len(server_addresses)

    # === Step 1: 数据分割 ===
    # np.array_split() 把数组尽可能均匀地分成 num_replicas 份
    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    # 例如：100 条数据，4 个 Replica → [[25条], [25条], [25条], [25条]]

    # 转换为 Python 列表（numpy 数组 → list）
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]

    # 验证分割正确
    assert len(server_addresses) == len(chat_sub_array)

    # === Step 2: 并发调用所有 Replica ===
    # 每个 Replica 处理自己的那份数据
    results = await asyncio.gather(
        *[
            generate_per_replica(server_addresses[i], model_path, n_samples, sampling_params, chat_sub_array[i])
            for i in range(num_replicas)
        ]
    )
    # results 是嵌套列表：[[Replica0结果], [Replica1结果], ...]

    return results


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """
    主函数：完整的生成流程

    === Hydra 装饰器说明 ===
    @hydra.main() 让这个函数成为 Hydra 应用的入口：
    - config_path="config": 配置文件目录
    - config_name="ppo_trainer": 默认配置文件名（ppo_trainer.yaml）
    - 命令行参数可以覆盖配置：python script.py key=value

    === 完整流程 ===
    1. 初始化 Ray 集群
    2. 读取数据集
    3. 启动 Replica 服务器
    4. 发送请求并收集结果
    5. 保存结果到文件

    === 配置示例 ===
    actor_rollout_ref:
      model:
        path: /path/to/model
      rollout:
        name: vllm
        tensor_model_parallel_size: 2
        temperature: 0.7
        top_p: 0.95
        response_length: 1024
        n: 1  # 每个 prompt 生成几个样本

    trainer:
      n_gpus_per_node: 8
      nnodes: 1

    data:
      train_files: /path/to/prompts.parquet
      prompt_key: messages
      output_path: /path/to/output.parquet
    """
    # === Step 1: 初始化 Ray 集群 ===
    # runtime_env 设置 Ray 工作进程的环境变量
    ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}})

    # 打印配置（调试用）
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True 解析变量引用
    OmegaConf.resolve(config)  # 解析配置中的变量

    # === Step 2: 解析采样参数 ===
    n_samples = config.actor_rollout_ref.rollout.n  # 每个 prompt 生成几个样本

    # 验证参数合理性
    if config.actor_rollout_ref.rollout.temperature == 0.0:
        # temperature=0 是贪婪解码，结果确定性，多次采样无意义
        assert n_samples == 1, "When temperature=0, n_samples must be 1."
    assert n_samples >= 1, "n_samples should always >= 1"

    # 构建采样参数字典
    sampling_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,  # 采样温度（0=贪婪，越高越随机）
        "top_p": config.actor_rollout_ref.rollout.top_p,  # nucleus sampling 阈值
        # "top_k": config.actor_rollout_ref.rollout.top_k,  # top-k sampling
        "max_tokens": config.actor_rollout_ref.rollout.response_length,  # 最大生成长度
    }

    # === Step 3: 读取数据集 ===
    from omegaconf import ListConfig

    train_files = config.data.train_files
    # 统一转换为列表格式（支持单文件和多文件）
    if not isinstance(train_files, list | ListConfig):
        train_files = [train_files]

    # 读取所有 parquet 文件
    # 注意：数据集应该已经是 chat template 格式（messages 列表）
    datasets = []
    for train_file in train_files:
        dataset = pd.read_parquet(train_file)
        datasets.append(dataset)

    # 合并数据集
    dataset = pd.concat(datasets, axis=0, ignore_index=True)

    # 提取对话消息列
    # prompt_key 指定哪一列包含 messages，例如 "messages" 或 "prompt"
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]  # 确保是 Python 列表
    chat_numpy = np.array(chat_lst)  # 转换为 numpy 数组方便分割

    # === Step 4: 启动服务器 ===
    # asyncio.run() 是从同步代码调用异步函数的入口
    # 它会创建事件循环，运行异步函数，然后关闭事件循环
    #
    # 【为什么需要 asyncio.run()？】
    # main() 是普通函数（不是 async def），不能直接用 await
    # asyncio.run() 作为"桥梁"，让同步代码可以调用异步代码
    server_handles, server_addresses = asyncio.run(start_server(config))

    # === Step 5: 批量生成 ===
    gen_results = asyncio.run(
        generate(server_addresses, config.actor_rollout_ref.model.path, n_samples, sampling_params, chat_numpy)
    )
    # gen_results 是嵌套列表：[[Replica0结果], [Replica1结果], ...]

    # === Step 6: 整理结果 ===
    import itertools

    # 把嵌套列表展平成一维列表
    # itertools.chain.from_iterable() 把 [[a,b], [c,d]] 变成 [a, b, c, d]
    results = list(itertools.chain.from_iterable(gen_results))

    # 提取生成的文本内容
    # ChatCompletion.choices[0].message.content 是模型生成的文本
    results = np.array([result.choices[0].message.content for result in results])

    # 重塑数组形状
    # 原来：[样本1-1, 样本1-2, 样本1-3, 样本2-1, 样本2-2, 样本2-3, ...]
    # 重塑：[[样本1-1, 样本1-2, 样本1-3], [样本2-1, 样本2-2, 样本2-3], ...]
    # 形状：(num_prompts, n_samples)
    results = np.reshape(results, (-1, n_samples))

    # 验证形状正确
    assert results.shape == (len(chat_lst), n_samples)

    # 转换为 Python 列表
    results = results.tolist()

    # === Step 7: 保存结果 ===
    # 把生成结果添加到原数据集
    dataset["responses"] = results

    # 确保输出目录存在
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)

    # 保存为 parquet 文件
    print(f"Saving results to {config.data.output_path}")
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
