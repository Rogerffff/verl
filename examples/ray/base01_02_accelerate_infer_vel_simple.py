"""
verl 框架简单应用示例：使用 Ray + Accelerate 进行分布式推理

本示例演示了如何：
1. 使用 verl 的 RayResourcePool 管理 GPU 资源
2. 使用 RayWorkerGroup 创建分布式 Worker 组
3. 使用 @register 装饰器简化分布式调用
4. 结合 HuggingFace Accelerate 进行模型推理

核心概念：
- Ray: 分布式计算框架，用于在多个 GPU/节点上并行执行任务
- verl: 基于 Ray 的强化学习训练框架，提供了便捷的分布式抽象
- Accelerate: HuggingFace 的分布式训练工具，简化多 GPU 训练/推理
"""

# ============================================================================
# 第一部分：导入必要的库
# ============================================================================

import logging  # Python 标准日志库，用于记录程序运行信息
import os  # 操作系统接口，用于访问环境变量等
import time  # 时间相关功能，用于计时等
import warnings  # 警告控制模块

import ray  # Ray 分布式计算框架的核心库
import torch  # PyTorch 深度学习框架

# 忽略所有警告信息，使输出更清晰（生产环境建议移除）
warnings.filterwarnings("ignore")

from typing import List, Tuple  # Python 类型注解，提高代码可读性

# ============================================================================
# Accelerate 相关导入
# ============================================================================
# PartialState: Accelerate 提供的分布式状态管理类
# 它封装了分布式训练所需的设备信息、进程 rank 等
# 与完整的 Accelerator 不同，PartialState 是轻量级的，不会自动包装模型
from accelerate import PartialState

# ============================================================================
# Transformers 相关导入
# ============================================================================
# AutoModelForCausalLM: 自动加载因果语言模型（如 GPT、Qwen 等）
# AutoTokenizer: 自动加载对应模型的分词器
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# verl 框架相关导入
# ============================================================================
# Worker: verl 的基础 Worker 类，所有自定义 Worker 都需要继承它
# Worker 提供了 self.rank（当前进程编号）和 self.world_size（总进程数）等属性
from verl.single_controller.base import Worker

# Dispatch: 数据分发模式枚举类，定义了如何将数据从 driver 分发到各个 worker
#   - ONE_TO_ALL: 将同一份数据复制到所有 worker
#   - ALL_TO_ALL: 按顺序将数据列表中的每个元素分发到对应 worker
#   - MEGATRON_COMPUTE: 专为 Megatron 并行设计的分发模式
# register: 装饰器，用于注册 Worker 方法的分发和收集行为
from verl.single_controller.base.decorator import Dispatch, register  # noqa: E402

# RayClassWithInitArgs: 封装 Ray Actor 类及其初始化参数的工具类
# RayResourcePool: GPU 资源池，定义可用的计算资源
# RayWorkerGroup: Worker 组，将多个 Worker 映射到资源池上，统一管理和调用
from verl.single_controller.ray.base import (  # noqa: E402
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)

# verl 提供的设备工具函数
# get_device_name: 获取当前设备类型（如 "cuda"、"npu" 等）
# get_nccl_backend: 获取 NCCL 通信后端名称（NVIDIA GPU 间通信库）
from verl.utils.device import (  # noqa: E402
    get_device_name,
    get_nccl_backend,
)

# ============================================================================
# 第二部分：初始化 Ray 集群
# ============================================================================
# ray.init() 启动一个本地 Ray 集群
# 如果已经有 Ray 集群在运行，会自动连接到现有集群
# 可以传入参数如 ray.init(num_gpus=4) 来指定资源
# 在集群环境中，可以用 ray.init(address="auto") 连接到已有集群
ray.init()

# 获取当前设备名称，通常是 "cuda"（NVIDIA GPU）或 "npu"（华为昇腾）
device_name = get_device_name()


# ============================================================================
# 第三部分：定义分布式 Worker 类
# ============================================================================

# @ray.remote 装饰器将普通 Python 类转换为 Ray Actor
# Ray Actor 是一个有状态的远程对象，运行在独立的进程中
# 每个 Actor 实例都有自己的内存空间，可以保持状态（如加载的模型）
# Actor 之间通过 RPC（远程过程调用）进行通信
@ray.remote
class TestAccelerateWorker(Worker):
    """
    继承自 verl 的 Worker 基类的测试 Worker

    Worker 基类提供的重要属性：
    - self.rank: 当前 worker 在 worker group 中的编号（从 0 开始）
    - self.world_size: worker group 中的 worker 总数

    这些属性由 RayWorkerGroup 在创建 worker 时自动设置
    """

    def __init__(self):
        # 调用父类 Worker 的初始化方法
        # 这一步很重要，它会初始化 rank、world_size 等属性的占位符
        # 实际值会在 RayWorkerGroup 创建时通过 _setup_worker 方法注入
        super().__init__()

        # ====================================================================
        # 从环境变量获取分布式训练的配置
        # ====================================================================
        # RANK: 当前进程在所有进程中的全局编号（由 verl 框架设置）
        # 默认值 0 表示单进程模式
        rank = int(os.environ.get("RANK", 0))

        # WORLD_SIZE: 参与分布式训练的总进程数（由 verl 框架设置）
        # 默认值 1 表示单进程模式
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # ====================================================================
        # 初始化 Accelerate 的 PartialState
        # ====================================================================
        # PartialState 是 Accelerate 提供的轻量级分布式状态管理器
        # 它不会自动包装模型或优化器，只提供设备和进程信息
        #
        # 参数说明：
        # - backend: 通信后端配置
        #   格式为 "cpu_backend:gloo,gpu_backend:nccl"
        #   - gloo: Facebook 开发的 CPU 通信库
        #   - nccl: NVIDIA 的 GPU 通信库（高效的 GPU 间通信）
        #   例如: "cpu:gloo,cuda:nccl"
        #
        # - rank: 当前进程的编号
        #
        # - world_size: 总进程数
        #
        # - init_method: 进程间同步的初始化方法
        #   常见值：
        #   - "env://": 从环境变量获取（MASTER_ADDR, MASTER_PORT）
        #   - "tcp://hostname:port": 使用 TCP 连接
        #   - "file:///path/to/file": 使用共享文件系统
        #   verl 框架会自动设置 DIST_INIT_METHOD 环境变量
        self.distributed_state = PartialState(
            backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
            rank=rank,
            world_size=world_size,
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )

    # ========================================================================
    # 使用 @register 装饰器注册 Worker 方法
    # ========================================================================
    # @register 装饰器用于定义方法在分布式环境下的行为
    #
    # dispatch_mode 参数定义数据如何从 driver（主进程）分发到各个 worker：
    # - Dispatch.ONE_TO_ALL: 将相同的数据广播到所有 worker
    #   例如: driver 调用 show_info()，所有 worker 都执行 show_info()
    #
    # - Dispatch.ALL_TO_ALL: 数据按顺序分发
    #   例如: driver 调用 func([a, b, c, d])，worker 0 收到 a，worker 1 收到 b...
    #
    # - Dispatch.MEGATRON_COMPUTE: 专为 Megatron 并行设计
    #   数据按 DP（数据并行）维度分发，在 TP/PP（张量/流水线并行）组内广播

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def show_info(self):
        """
        显示当前 Worker 的信息

        由于使用 ONE_TO_ALL 模式，driver 调用一次，所有 worker 都会执行
        返回值会被收集成一个列表返回给 driver
        """
        info = {
            # Accelerate 管理的设备（如 cuda:0, cuda:1）
            "acc_device": self.distributed_state.device,
            # verl Worker 的 rank（由 RayWorkerGroup 设置）
            "rank": self.rank,
            # verl Worker 的 world_size
            "world_size": self.world_size,
            # Accelerate PartialState 的字符串表示
            "acc_world_size": str(self.distributed_state),
        }
        return info

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_model(self, model_name: str):
        """
        加载预训练语言模型

        参数:
            model_name: HuggingFace 模型名称或本地路径

        由于使用 ONE_TO_ALL 模式：
        - driver 只需要调用一次 load_model(model_name)
        - 所有 worker 都会收到相同的 model_name 并加载模型
        - 每个 worker 的模型会加载到各自的 GPU 上
        """
        # AutoModelForCausalLM.from_pretrained: 加载因果语言模型
        # 参数说明：
        # - model_name: 模型标识符（HuggingFace Hub 上的名称或本地路径）
        # - device_map: 指定模型放置的设备
        #   这里使用 PartialState 管理的设备，确保每个 worker 使用不同 GPU
        # - torch_dtype: 模型参数的数据类型
        #   torch.float16 (FP16) 可以减少显存占用，加速推理
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.distributed_state.device,
            torch_dtype=torch.float16,
        )

        # 加载对应的分词器
        # 分词器负责将文本转换为模型可以理解的 token ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 设置 padding token
        # 许多模型（如 GPT 系列）没有专门的 padding token
        # 在生成任务中，需要设置一个 padding token 用于批处理
        # 通常将 eos_token（结束符）作为 padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 返回模型所在设备和模型对象的内存地址（用于调试）
        return self.model.device, id(self.model)

    def _infer(self, prompts: list[str]):
        """
        内部推理方法（不带 @register 装饰器）

        这是一个普通的实例方法，只在当前 worker 内部调用
        不通过 RayWorkerGroup 的分发机制

        参数:
            prompts: 要处理的提示词列表
        """

        def formmat_prompt_func(prompt: str):
            """
            将用户输入格式化为模型期望的对话格式

            大多数聊天模型（如 Qwen、ChatGPT）都有特定的对话模板
            这个函数将简单的用户输入转换为包含系统提示和用户消息的格式
            """
            # 构建对话消息列表
            # 标准的 Chat 格式包含 role（角色）和 content（内容）
            messages = [
                {
                    "role": "system",  # 系统提示，定义 AI 的行为
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},  # 用户的输入
            ]

            # apply_chat_template: 将消息列表转换为模型期望的文本格式
            # 不同模型有不同的模板格式（如 ChatML、Llama 格式等）
            # 参数说明：
            # - tokenize=False: 不进行分词，只返回格式化后的文本
            # - add_generation_prompt=True: 添加生成提示符
            #   这会在末尾添加助手回复的开始标记，提示模型开始生成
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return text

        res = []
        # 遍历所有提示词进行推理
        for batch_simple in prompts:
            # 格式化提示词
            batch = formmat_prompt_func(batch_simple)

            # 分词并转换为模型输入格式
            # tokenizer() 返回包含 input_ids 和 attention_mask 的字典
            # return_tensors="pt": 返回 PyTorch tensor
            # .to(device): 将数据移动到 GPU
            model_inputs = self.tokenizer([batch], return_tensors="pt").to(
                self.distributed_state.device
            )

            # 使用模型生成文本
            # model.generate() 是 HuggingFace 的自动回归生成方法
            # **model_inputs: 解包输入字典（input_ids, attention_mask 等）
            # max_new_tokens: 最多生成的新 token 数量
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)

            # 提取生成的部分（去除输入部分）
            # generated_ids 包含 [输入 + 生成]，我们只需要生成的部分
            # output_ids[len(input_ids):] 从输入长度处开始切片
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            # 将 token ID 解码为文本
            # batch_decode: 批量解码
            # skip_special_tokens=True: 跳过特殊 token（如 <eos>、<pad>）
            # [0][:10]: 取第一个结果的前 10 个字符（这里只是示例，实际应用不需要截断）
            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0][:10]

            # 将结果添加到返回列表
            res.append({"query": batch_simple, "response": response})
        return res

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def infer(self, prompt: str | list[str]):
        """
        分布式推理入口方法

        这个方法展示了一种简单的数据并行策略：
        1. 所有 worker 都收到完整的 prompt 列表（ONE_TO_ALL）
        2. 每个 worker 根据自己的 rank 选择处理哪些 prompt
        3. 各自处理后返回结果

        参数:
            prompt: 单个提示词字符串，或提示词列表
        """
        # 统一转换为列表格式
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # ====================================================================
        # 在 worker 内部进行数据分割
        # ====================================================================
        # 这是一种简单的数据并行策略：
        # 每个 worker 都有完整数据，但只处理属于自己的那部分

        # 创建 world_size 个空列表，用于分配数据
        splits = [[] for _ in range(self.world_size)]

        # 使用轮询（round-robin）方式分配数据
        # prompt 0 -> worker 0
        # prompt 1 -> worker 1
        # prompt 2 -> worker 0 (如果只有 2 个 worker)
        # ...
        for i, prompt in enumerate(prompts):
            splits[i % self.world_size].append(prompt)

        # 获取当前 worker 应该处理的数据
        # 每个 worker 的 self.rank 不同，所以会获取不同的数据
        split_world_prompts = splits[self.rank]

        # 调用内部推理方法处理数据
        return self._infer(split_world_prompts)


# ============================================================================
# 第四部分：创建资源池和 Worker 组
# ============================================================================

# RayResourcePool: GPU 资源池，定义可用的计算资源
#
# 参数说明：
# - [2]: 进程/GPU 布局列表
#   [2] 表示在一个节点上使用 2 个 GPU
#   [4] 表示使用 4 个 GPU
#   [2, 2] 表示在 2 个节点上各使用 2 个 GPU（共 4 个）
#   [4, 4] 表示在 2 个节点上各使用 4 个 GPU（共 8 个）
#
# - use_gpu=True: 是否使用 GPU
#   如果为 True，每个 worker 会绑定一个 GPU
#   如果为 False，worker 只使用 CPU
resource_pool = RayResourcePool([2], use_gpu=True)

# RayClassWithInitArgs: 封装 Ray Actor 类及其初始化参数
#
# 这个类的作用是延迟实例化：
# - 不立即创建 Actor 实例
# - 只保存类和初始化参数
# - 由 RayWorkerGroup 在需要时创建实际的 Actor
#
# 参数说明：
# - cls: Ray Actor 类（必须用 @ray.remote 装饰）
# - 其他参数会传递给 Actor 的 __init__ 方法
#   例如: RayClassWithInitArgs(cls=MyWorker, arg1=1, arg2="test")
class_with_args = RayClassWithInitArgs(cls=TestAccelerateWorker)

# RayWorkerGroup: 创建并管理一组 Worker
#
# 工作原理：
# 1. 根据 resource_pool 的定义创建多个 Ray Actor
# 2. 每个 Actor 绑定到一个 GPU
# 3. 为每个 Actor 设置环境变量（RANK, WORLD_SIZE, DIST_INIT_METHOD 等）
# 4. 调用每个 Actor 的 _setup_worker 方法初始化 rank 和 world_size
#
# 创建后，worker_group 可以像调用普通对象方法一样调用 Worker 的方法
# 但实际上这些调用会被分发到所有 Worker 上执行
worker_group = RayWorkerGroup(resource_pool, class_with_args)

# ============================================================================
# 第五部分：使用 Worker 组
# ============================================================================

# 调用 worker_group.show_info()
# 由于 show_info 使用了 @register(dispatch_mode=Dispatch.ONE_TO_ALL)
# 所以这个调用会：
# 1. 将调用广播到所有 worker
# 2. 每个 worker 执行 show_info() 并返回结果
# 3. 收集所有 worker 的返回值，组成列表返回
# 返回值: [worker_0_info, worker_1_info, ...]
show_info = worker_group.show_info()

# 打印每个 worker 的信息（调试用）
# for i in show_info:
#     print(i)

# 模型路径：可以是 HuggingFace Hub 上的模型名，或本地路径
# 例如：
# - "Qwen/Qwen2.5-0.5B-Instruct" (从 HuggingFace Hub 下载)
# - "/path/to/local/model" (本地路径)
model_name = "/home/yuanz/documents/weights/Qwen/Qwen2.5-0.5B-Instruct"

# 加载模型到所有 worker
# 由于 load_model 使用 ONE_TO_ALL 模式：
# - 所有 worker 都收到相同的 model_name
# - 每个 worker 在自己的 GPU 上加载模型
# - 返回每个 worker 的 (device, model_id) 列表
model_device = worker_group.load_model(model_name=model_name)
print(model_device)  # 例如: [(cuda:0, 12345), (cuda:1, 67890)]

# 准备测试查询
query_list = [
    "你是谁",
    "1+1=几",
    "十个字介绍一下杭州",
]

# ============================================================================
# 第六部分：执行分布式推理
# ============================================================================

# 方式 1：使用 infer 方法进行分布式推理
# 工作流程：
# 1. driver 调用 worker_group.infer(prompt=query_list)
# 2. ONE_TO_ALL 模式：所有 worker 都收到完整的 query_list
# 3. 每个 worker 内部根据 rank 选择处理哪些 query
#    - worker 0 处理: ["你是谁", "十个字介绍一下杭州"]
#    - worker 1 处理: ["1+1=几"]
# 4. 各 worker 执行推理并返回结果
# 5. 结果被收集成列表返回给 driver

# response_list = worker_group.infer(prompt=query_list)

# print(response_list)
# 输出示例:
# [
#   [{"query": "你是谁", "response": "..."}, {"query": "十个字介绍一下杭州", "response": "..."}],  # worker 0 的结果
#   [{"query": "1+1=几", "response": "..."}]  # worker 1 的结果
# ]

# ============================================================================
# 第七部分：清理资源
# ============================================================================

# 关闭 Ray 集群
# 这会：
# 1. 终止所有 Ray Actor（Worker 进程）
# 2. 释放所有 GPU 资源
# 3. 清理 Ray 的 Object Store
# 在脚本结束时调用，确保资源被正确释放
ray.shutdown()
