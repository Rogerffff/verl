# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
多轮 SFT 数据集，支持在多轮对话数据上进行有监督微调训练

本模块的核心功能：
1. 读取 Parquet 格式的对话数据
2. 将多轮对话转换为模型可训练的格式（input_ids, attention_mask, loss_mask）
3. 支持多模态输入（图片、视频）
4. 支持 function calling（tools）
5. 支持 thinking mode（思考模式，如 Qwen3 的 enable_thinking）

数据格式示例：
Parquet 文件中的每一行应包含 "messages" 字段，格式如下：
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "3+3 equals 6."}
]

训练时只对 assistant 的回复计算 loss，其他部分（system、user）不计算 loss。
"""

import logging
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import hf_tokenizer
from verl.utils.chat_template import extract_system_prompt_and_generation
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.vision_utils import process_image, process_video
from verl.utils.fs import copy_local_path_from_hdfs

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def convert_nested_value_to_list_recursive(data_item):
    """
    递归地将嵌套数据结构中的 numpy 数组转换为 Python 列表

    为什么需要这个函数？
    因为 Parquet 文件读取出的数据可能包含 numpy.ndarray，而后续处理需要标准 Python 列表。

    Args:
        data_item: 任意数据类型（dict、list、np.ndarray 或基础类型）

    Returns:
        转换后的数据，所有 numpy.ndarray 都被转为 list

    示例：
        输入: {'content': np.array(['Hello', 'World'])}
        输出: {'content': ['Hello', 'World']}

        输入: [np.array([1, 2, 3]), {'nested': np.array([4, 5])}]
        输出: [[1, 2, 3], {'nested': [4, 5]}]
    """
    if isinstance(data_item, dict):
        # 递归处理字典的每个值
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        # 递归处理列表的每个元素
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # 将 numpy 数组转换为列表，然后递归处理
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # 基础类型（int、str、float、bool 等）直接返回
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class MultiTurnSFTDataset(Dataset):
    """
    多轮对话 SFT 数据集类

    核心设计理念：
    - 将多轮对话中的每个 assistant 回复都作为训练目标
    - 只有 assistant 的回复计算 loss，user/system 的内容不计算 loss
    - 支持多模态（图片/视频）和工具调用

    工作流程：
    1. 读取 Parquet 文件中的对话数据
    2. 对每条消息单独 tokenize
    3. 拼接所有消息的 token，生成 loss_mask 标记哪些位置需要计算 loss
    4. 处理填充/截断，返回模型可用的格式

    Args:
        parquet_files (str or list): Parquet 文件路径，支持单个文件或多个文件
        tokenizer (PreTrainedTokenizer): 用于文本 tokenization 的 tokenizer
        config (DictConfig): 配置选项，包括：
            - pad_mode: 填充模式 ("right" 或 "no_padding")
            - truncation: 截断策略 ("error", "left", "right")
            - max_length: 最大序列长度
            - messages_key: 消息字段名（默认 "messages"）
            - image_key: 图片字段名（默认 "images"）
            - video_key: 视频字段名（默认 "videos"）
            - tools_key: 工具字段名（默认 "tools"）
            - enable_thinking_key: 思考模式字段名（默认 "enable_thinking"）
            - shuffle: 是否打乱数据
            - seed: 随机种子
            - ignore_input_ids_mismatch: 是否忽略 input_ids 不匹配错误
        processor (ProcessorMixin, optional): 多模态处理器，用于处理图片/视频
        max_samples (int, optional): 最大样本数，-1 表示使用全部数据

    使用示例：
        ```python
        from transformers import AutoTokenizer
        from omegaconf import OmegaConf

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        config = OmegaConf.create({
            "max_length": 2048,
            "pad_mode": "right",
            "truncation": "right",
        })

        dataset = MultiTurnSFTDataset(
            parquet_files="train_data.parquet",
            tokenizer=tokenizer,
            config=config,
        )

        # 获取一条数据
        sample = dataset[0]
        # sample 包含: input_ids, attention_mask, position_ids, loss_mask
        ```
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # =======================
        # 1. 解析并设置配置参数
        # =======================
        # Set defaults and extract parameters from config if provided
        config = config or {}

        # 填充模式：
        # - "right": 在序列右侧填充至 max_length
        # - "no_padding": 不填充，保持原始长度（适用于动态 batch）
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )

        # 截断策略：当序列超过 max_length 时的处理方式
        # - "error": 抛出异常
        # - "left": 从左侧截断（保留最近的对话）
        # - "right": 从右侧截断（保留开头的对话）
        self.truncation = config.get("truncation", "error")
        assert self.truncation in ["error", "left", "right"]

        # for right padding
        self.max_length = config.get("max_length", 1024)

        # =======================
        # 2. 数据字段名称配置
        # =======================
        # Get messages_key from the new multiturn config structure
        # 消息字段名，Parquet 文件中存储对话的列名
        self.messages_key = config.get("messages_key", "messages")
        # 图片字段名，存储图片路径或数据的列名
        self.image_key = config.get("image_key", "images")
        # 视频字段名，存储视频路径或数据的列名
        self.video_key = config.get("video_key", "videos")
        # 图像 patch 大小，用于视觉模型
        self.image_patch_size = config.get(
            "image_patch_size", processor.image_processor.patch_size if processor else None
        )
        # 工具字段名，存储 function calling 工具定义的列名
        self.tools_key = config.get("tools_key", "tools")
        # 思考模式字段名（用于 Qwen3 等支持思考模式的模型）
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")

        # apply_chat_template 的额外参数
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # =======================
        # 3. 数据采样配置
        # =======================
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples

        # 是否忽略 input_ids 不匹配的问题
        # 某些模型（如 Qwen Thinking）在最后一轮会添加特殊标签，导致逐轮拼接的结果与整体 tokenize 不一致
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)

        # =======================
        # 4. 初始化数据
        # =======================
        # 确保 parquet_files 是列表格式
        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files

        # 支持传入 tokenizer 路径字符串，自动加载
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

        # 下载文件（如果是 HDFS 路径）并读取处理
        self._download()
        self._read_files_and_process()

    def _download(self):
        """
        从 HDFS 下载 Parquet 文件到本地

        如果文件路径是 HDFS 路径（如 hdfs://...），会将其复制到本地临时目录。
        如果是本地路径，则保持不变。
        """
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        """
        读取并处理 Parquet 文件

        主要步骤：
        1. 读取所有 Parquet 文件并合并
        2. 如果设置了 max_samples，进行采样
        3. 提取消息、工具、思考模式等字段
        4. 提取 system_prompt 和 generation_prompt 用于后续处理
        """
        def series_to_item(ls):
            """将单元素 Series/array 解包为标量"""
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        # =======================
        # 1. 读取并合并所有 Parquet 文件
        # =======================
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        # =======================
        # 2. 采样处理
        # =======================
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                # 随机采样
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                # 顺序采样前 max_samples 条
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        # =======================
        # 3. 提取各字段数据
        # =======================
        # 提取并转换消息列表（将 numpy 数组转为 list）
        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()

        # 提取工具定义（如果存在）
        # tools 示例：[{"name": "get_weather", "description": "获取天气", "parameters": {...}}]
        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None

        # 提取思考模式开关（如果存在）
        # enable_thinking 是 Qwen3 系列的特性，控制是否启用思考模式
        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

        # =======================
        # 4. 提取 chat template 的特殊标记
        # =======================
        # system_prompt 示例: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        # generation_prompt 示例: <|im_start|>assistant\n
        # 这些用于后续处理单条消息时去除重复的系统提示和计算 loss_mask
        # system prompt: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        # generation prompt: <|im_start|>assistant\n
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.messages)

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        处理单条消息，返回其 tokenized 表示

        这是数据处理的核心方法，将一条消息转换为模型输入格式。

        Args:
            index: 消息在对话中的索引（用于判断是否是第一条消息）
            message: 单条消息字典，格式如 {"role": "user", "content": "Hello"}
            tools: 工具定义列表（只在第一条消息时使用）
            enable_thinking: 是否启用思考模式

        Returns:
            tuple: (input_ids, loss_mask, attention_mask, multi_modal_inputs)
            - input_ids: token ID 列表
            - loss_mask: 标记哪些位置计算 loss（1=计算，0=不计算）
            - attention_mask: 注意力掩码
            - multi_modal_inputs: 多模态输入（如 pixel_values）

        处理逻辑：
        1. 使用 apply_chat_template 将消息转换为 token
        2. 如果不是第一条消息，去除重复的 system_prompt
        3. 根据消息角色设置 loss_mask：
           - assistant: loss_mask=1（计算 loss），但 generation_prompt 部分为 0
           - user/system: loss_mask=0（不计算 loss）

        示例：
            假设处理一条 user 消息：
            message = {"role": "user", "content": "What is 2+2?"}

            tokenize 后可能得到：
            input_ids = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198]
            对应文本: <|im_start|>user\nWhat is 2+2?<|im_end|>\n

            由于是 user 消息，loss_mask 全为 0：
            loss_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            假设处理一条 assistant 消息：
            message = {"role": "assistant", "content": "4"}

            tokenize 后可能得到：
            input_ids = [151644, 77091, 198, 19, 151645, 198]
            对应文本: <|im_start|>assistant\n4<|im_end|>\n

            generation_prompt = [151644, 77091, 198] (即 "<|im_start|>assistant\n")
            loss_mask 中 generation_prompt 部分为 0，其余为 1：
            loss_mask = [0, 0, 0, 1, 1, 1]
            只有 "4<|im_end|>\n" 部分计算 loss
        """
        # 选择使用 processor（多模态）或 tokenizer（纯文本）
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}

        # 如果指定了 enable_thinking，添加到参数中
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        # 使用 chat template 进行 tokenization
        # 注意：这里传入的是 [message]（单条消息列表），而不是整个对话
        inputs = processor.apply_chat_template(
            [message],
            tools=tools,
            add_generation_prompt=False,  # 不添加生成提示，因为我们分别处理每条消息
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]  # 移除 batch 维度
        attention_mask = inputs.pop("attention_mask")[0]

        # =======================
        # 去除重复的 system_prompt
        # =======================
        # 当处理第 2 条及之后的消息时，apply_chat_template 会自动添加 system_prompt
        # 但我们是逐条处理后拼接的，所以需要去除重复的 system_prompt
        # remove system prompt if exists
        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        # =======================
        # 设置 loss_mask
        # =======================
        # loss_mask 决定哪些 token 参与 loss 计算
        # - assistant 消息：计算 loss（除了 generation_prompt 部分）
        # - user/system 消息：不计算 loss
        if message["role"] == "assistant":
            loss_mask = torch.ones_like(attention_mask)
            # mask out generation prompt if assistant message
            # generation_prompt 是 "<|im_start|>assistant\n" 这部分，不应计算 loss
            loss_mask[: len(self.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask, inputs

    def _build_messages(self, example: dict):
        """
        构建多模态消息，将 <image> 和 <video> 占位符替换为实际的图片/视频数据

        这个方法处理消息中的多模态占位符，将其转换为 processor.apply_chat_template 所需的格式。

        Args:
            example: DataFrame 中的一行数据（字典格式）

        Returns:
            messages: 处理后的消息列表

        输入示例：
            example = {
                "messages": [
                    {"role": "user", "content": "这张图片里有什么？<image>"},
                    {"role": "assistant", "content": "这是一只可爱的猫咪。"}
                ],
                "images": ["/path/to/cat.jpg"]
            }

        输出示例：
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这张图片里有什么？"},
                        {"type": "image", "image": <processed_image>}
                    ]
                },
                {"role": "assistant", "content": "这是一只可爱的猫咪。"}
            ]

        处理逻辑：
        1. 遍历每条消息
        2. 使用正则表达式将内容按 <image> 和 <video> 分割
        3. 将分割后的片段转换为相应的格式：
           - 普通文本 -> {"type": "text", "text": "..."}
           - <image> -> {"type": "image", "image": processed_image}
           - <video> -> {"type": "video", "video": processed_video}
        """
        # Replace <image> and <video> placeholder in messages with corresponding image and video
        # which is required by processor.apply_chat_template.
        # - <image>: {"type": "image", "image": image}
        # - <video>: {"type": "video", "video": video}

        messages: list = example[self.messages_key]
        images = example[self.image_key] if self.image_key in example else []
        videos = example[self.video_key] if self.video_key in example else []

        # 用偏移量跟踪当前处理到第几张图片/视频
        image_offset, video_offset = 0, 0

        for message in messages:
            # 如果没有图片和视频数据，跳过处理
            if self.image_key not in example and self.video_key not in example:
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content = message["content"]
            # 如果内容不是字符串（可能已经是多模态格式），跳过
            if not isinstance(content, str):
                continue

            content_list = []
            # 使用正则表达式按 <image> 和 <video> 分割内容
            # 示例: "请描述<image>这张图片<image>和这张" 
            # -> ["请描述", "<image>", "这张图片", "<image>", "和这张"]
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]  # 过滤空字符串

            for segment in segments:
                if segment == "<image>":
                    # 处理图片：加载并预处理
                    image = process_image(images[image_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    # 处理视频：加载并预处理
                    video = process_video(videos[video_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "video", "video": video})
                    video_offset += 1
                else:
                    # 普通文本
                    content_list.append({"type": "text", "text": segment})

            message["content"] = content_list

        # 验证所有图片和视频都被使用
        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages

    def __getitem__(self, item):
        """
        获取第 item 条数据

        这是 Dataset 的核心方法，返回模型训练所需的完整数据。

        Args:
            item: 样本索引

        Returns:
            dict: 包含以下字段：
            - input_ids: (seq_len,) token ID 序列
            - attention_mask: (seq_len,) 注意力掩码
            - position_ids: (seq_len,) 或 (4, seq_len) 位置编码
            - loss_mask: (seq_len,) 损失掩码，标记参与 loss 计算的位置
            - multi_modal_inputs: (可选) 多模态输入，如 pixel_values

        处理流程：
        1. 构建多模态消息
        2. 逐条消息 tokenize 并拼接
        3. 处理位置编码（特别是 Qwen2-VL 的 3D rope）
        4. 处理填充或截断

        示例输出：
            对于一条简单的两轮对话：
            user: "2+2等于几？"
            assistant: "4"

            返回：
            {
                "input_ids": tensor([151644, 8948, 198, ...]),  # 完整对话的 token
                "attention_mask": tensor([1, 1, 1, ...]),       # 有效 token 为 1
                "position_ids": tensor([0, 1, 2, ...]),         # 位置编码
                "loss_mask": tensor([0, 0, 0, ..., 1, 1, 0]),   # 只有 assistant 回复为 1
            }
        """
        # 获取原始数据
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        # =======================
        # 1. 逐条消息 tokenize
        # =======================
        # 1. tokenize each message
        input_ids, loss_mask, attention_mask, multi_modal_inputs = [], [], [], {}
        for i, message in enumerate(messages):
            _input_ids, _loss_mask, _attention_mask, _inputs = self._process_single_message(
                index=i,
                message=message,
                tools=tools if i == 0 else None,  # 工具定义只在第一条消息时传入
                enable_thinking=enable_thinking,
            )
            input_ids.append(_input_ids)
            loss_mask.append(_loss_mask)
            attention_mask.append(_attention_mask)
            # 收集多模态输入
            for k, v in _inputs.items():
                multi_modal_inputs.setdefault(k, []).append(v)

        # 拼接所有消息的 token
        input_ids = torch.cat(input_ids, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        assert input_ids.shape == loss_mask.shape == attention_mask.shape, (
            f"Shape mismatch: {input_ids.shape}, {loss_mask.shape}, {attention_mask.shape}"
        )

        # 验证拼接结果的正确性
        self.sanity_check(input_ids, messages, tools, enable_thinking)

        # =======================
        # 处理多模态输入
        # =======================
        # Since the tokenizer may return user-customized results, we need to filter out inconsistent tensor shapes
        # 过滤掉形状不一致的张量（某些自定义 tokenizer 可能返回不规则结果）
        keys_to_remove = []
        for k, v in multi_modal_inputs.items():
            if len(v) > 0 and v[0] is not None and isinstance(v[0], torch.Tensor):
                # Check if all tensors in the list have the same shape
                first_shape = v[0].shape[1:]
                if not all(tensor.shape[1:] == first_shape for tensor in v):
                    keys_to_remove.append(k)

        for k in keys_to_remove:
            del multi_modal_inputs[k]

        # 拼接多模态输入
        for k, v in multi_modal_inputs.items():
            multi_modal_inputs[k] = torch.concat(v, dim=0)

        # =======================
        # 2. 处理位置编码
        # =======================
        # 2. handle position_ids for Qwen-VL series models
        # Qwen2-VL 使用特殊的 3D RoPE，需要额外处理
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # 获取视觉内容的网格信息
            image_grid_thw = multi_modal_inputs.get("image_grid_thw", None)
            video_grid_thw = multi_modal_inputs.get("video_grid_thw", None)
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts", None)

            # 计算视觉内容的 3D 位置编码
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )  # (3, seq_len) - 时间、高度、宽度三个维度

            # 文本使用标准的 1D 位置编码
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)  # (1, seq_len)
            # 拼接文本和视觉的位置编码
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            # 普通模型使用标准 1D 位置编码
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (seq_len,)

        # =======================
        # 3. 处理填充和截断
        # =======================
        # 3. handle padding
        sequence_length = input_ids.shape[0]

        # Handle sequence length
        if self.pad_mode == DatasetPadMode.RIGHT:
            # ---- 右填充模式 ----
            if sequence_length < self.max_length:
                # 序列长度不足，需要填充
                # Pad sequences
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                # 创建填充 tensor
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                # 拼接原始数据和填充
                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(position_ids, (0, self.max_length - sequence_length), value=0)

            elif sequence_length > self.max_length:
                # 序列过长，需要截断
                if self.truncation == "left":
                    # 左截断：保留最近的对话（适合生成场景）
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[..., -self.max_length :]
                elif self.truncation == "right":
                    # 右截断：保留开头的对话
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[..., : self.max_length]
                elif self.truncation == "error":
                    # 抛出错误
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            # 构建返回结果
            res = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res

        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            # ---- 无填充模式 ----
            # 适用于动态 batch（不同样本长度不同，由 DataLoader 的 collate_fn 处理填充）
            # truncate input_ids if it is longer than max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[..., : self.max_length]

            # return nested tensor with out padding
            res = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def sanity_check(self, input_ids: torch.Tensor, messages: list[dict], tools: list[dict], enable_thinking: bool):
        """
        完整性检查：验证逐条 tokenize 后拼接的结果是否与整体 tokenize 一致

        为什么需要这个检查？
        我们采用的策略是逐条消息 tokenize 后拼接，而不是对整个对话一次性 tokenize。
        这两种方式的结果在某些模型上可能不一致，特别是：
        - Qwen Thinking 系列：会在最后一轮添加 <think></think> 标签
        - 某些模型的特殊 token 处理

        Args:
            input_ids: 逐条拼接得到的 input_ids
            messages: 原始消息列表
            tools: 工具定义
            enable_thinking: 是否启用思考模式

        Raises:
            AssertionError: 如果结果不一致且 ignore_input_ids_mismatch=False

        示例：
            Qwen3 with enable_thinking=True 的情况：

            逐条处理的结果：
            1. system: "<|im_start|>system\n...<|im_end|>\n"
            2. user: "<|im_start|>user\n...<|im_end|>\n"
            3. assistant: "<|im_start|>assistant\n...<|im_end|>\n"

            一次性处理的结果（会自动添加思考标签）：
            1. system: "<|im_start|>system\n...<|im_end|>\n"
            2. user: "<|im_start|>user\n...<|im_end|>\n"
            3. assistant: "<|im_start|>assistant\n<think>...</think>...<|im_end|>\n"

            此时两者的 input_ids 会不一致。设置 ignore_input_ids_mismatch=True 可以忽略此问题。
        """
        # Check concatenated input_ids of apply_chat_template to each turn equals
        # apply_chat_template to whole messages.
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        # 对整个对话一次性 tokenize
        inputs = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        error_message = (
            "MultiTurnSFTDataset apply_chat_template to each turn separately and concat `input_ids` "
            "as a whole sequence, which may not equal to apply_chat_template to whole messages at once.\n"
            "For example, Qwen Thinking series models add <think></think> tags to last turn, please check "
            "your tokenizer chat template settings.\n"
            "Set `ignore_input_ids_mismatch=True` to ignore input_ids mismatch and use the concatenated "
            "input_ids as the final input_ids. "
        )

        # 比较两种方式的结果
        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):
            if self.ignore_input_ids_mismatch:
                # 只警告一次
                logger.warning_once(error_message)
            else:
                raise AssertionError(error_message)
