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
一个轻量级的基于 FSDP (Fully Sharded Data Parallel) 的 SFT (Supervised Fine-Tuning) 训练器。

【核心概念解释】
- SFT (Supervised Fine-Tuning): 监督式微调，即在预训练大语言模型基础上，使用标注数据进行有监督的微调训练，
  使模型学会遵循指令或适应特定任务。例如：用问答对数据微调 LLaMA 模型使其学会对话。
- FSDP (Fully Sharded Data Parallel): PyTorch 提供的全分片数据并行策略，
  将模型参数、梯度和优化器状态分片到多个 GPU 上，以节省显存、支持更大模型训练。
  例如：一个 7B 参数的模型在单卡放不下时，FSDP 会将参数切分到多张 GPU 上。

【本文件的工作流程】
1. run_sft() 入口函数：初始化分布式环境 → 构建 tokenizer 和数据集 → 创建训练器
2. FSDPSFTTrainer.__init__(): 构建 dataloader → 构建模型和优化器 → 加载 checkpoint
3. FSDPSFTTrainer.fit(): 执行训练循环（含验证和保存 checkpoint）

TODO(zhangchi.usc1992)
- Add calculation of mfu (模型浮点运算利用率)
- Add validation (已实现)
"""

import os

# ==================== 环境变量设置 ====================
# NCCL 是 NVIDIA 的多 GPU 通信库，设置 WARN 级别减少调试输出
os.environ["NCCL_DEBUG"] = "WARN"
# 允许 tokenizer 使用多线程并行处理，加速数据预处理
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
import time
from contextlib import nullcontext

# ==================== 第三方库导入 ====================
# Hydra: Facebook 开发的配置管理框架，支持从 YAML 文件加载和组合配置
import hydra
import torch
import torch.distributed
# OmegaConf: 与 Hydra 配合使用的配置库，支持嵌套字典式配置访问
from omegaconf import DictConfig, OmegaConf
# PEFT (Parameter-Efficient Fine-Tuning): HuggingFace 提供的高效微调库
# LoRA 是一种常用的高效微调方法，只训练少量低秩矩阵而非全部参数
from peft import LoraConfig, TaskType, get_peft_model
# TensorDict: PyTorch 提供的类字典张量容器，方便批量操作多个张量
from tensordict import TensorDict
from torch import nn
# DeviceMesh: PyTorch 的设备网格抽象，用于定义多 GPU 的逻辑拓扑
# 例如: 8 张 GPU 可以被组织为 (4, 2) 的网格，4 路数据并行 × 2 路序列并行
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
# FSDP 相关组件：CPU 卸载、混合精度、分片策略
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DistributedSampler
# StatefulDataLoader: 支持保存和恢复状态的 DataLoader，用于断点续训
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

# ==================== verl 内部工具导入 ====================
import verl.utils.hdfs_io as hdfs_io  # HDFS 文件系统操作（用于云端存储 checkpoint）
# 注意力机制相关工具函数（用于 remove padding 和序列并行）
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
# Checkpoint 管理：查找最新的 checkpoint、获取跟踪文件名
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset import SFTDataset  # 单轮 SFT 数据集
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset  # 多轮对话 SFT 数据集
from verl.utils.device import (
    auto_set_device,
    get_device_id,
    get_device_name,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local  # 将远程模型路径复制到本地
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,  # FSDP2 (PyTorch >= 2.4 的新版 fully_shard API)
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,  # 获取 FSDP 的自动包装策略（决定哪些层被 FSDP 包装）
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger import log_with_rank
from verl.utils.profiler import log_gpu_memory_usage  # GPU 显存使用监控
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
# 学习率调度器：余弦退火 和 WSD (Warmup-Stable-Decay)
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking  # 训练指标跟踪（如 WandB / TensorBoard）
# Ulysses 序列并行：DeepSpeed 提出的序列并行方案，将长序列切分到多个 GPU 上
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.config.optimizer import build_optimizer
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
# 日志级别可通过环境变量 VERL_SFT_LOGGING_LEVEL 控制，默认 WARN
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    """
    从 checkpoint 路径中提取训练步数。

    例如:
        extract_step("/checkpoints/global_step_1000") → 1000
        extract_step("/checkpoints/random_name") → None
    """
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    """
    基于 FSDP 的 SFT 训练器。

    【主要功能】
    - 使用 FSDP 实现多 GPU 分布式训练，支持 FSDP1 和 FSDP2 两种策略
    - 支持 LoRA 等参数高效微调方法
    - 支持 Ulysses 序列并行（处理超长序列）
    - 支持 remove padding（去除 padding token 以提高计算效率）
    - 支持梯度累积（micro batch）、混合精度训练、梯度 checkpoint 等
    - 支持断点续训（checkpoint 保存与恢复）

    【使用示例】
    trainer = FSDPSFTTrainer(
        config=config,            # Hydra 配置对象
        device_mesh=device_mesh,  # FSDP 设备网格
        ulysses_device_mesh=ulysses_device_mesh,  # 序列并行设备网格
        tokenizer=tokenizer,      # HuggingFace tokenizer
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.fit()  # 开始训练
    """

    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        """
        初始化 SFT 训练器。

        Args:
            config: Hydra 配置对象，包含所有训练超参数
            device_mesh: FSDP 设备网格，定义了 GPU 之间的分片拓扑
                例如: 4 张 GPU → DeviceMesh("cuda", [0,1,2,3], mesh_dim_names=("fsdp",))
            ulysses_device_mesh: Ulysses 序列并行设备网格
                例如: 4 GPU, SP=2 → DeviceMesh("cuda", [[0,1],[2,3]], mesh_dim_names=("dp","sp"))
                其中 dp=数据并行维度, sp=序列并行维度
            tokenizer: HuggingFace tokenizer，用于文本编解码
            train_dataset: 训练数据集
            val_dataset: 验证数据集
        """
        self.config = config
        self.device_mesh = device_mesh  # FSDP 分片用的设备网格
        self.ulysses_device_mesh = ulysses_device_mesh  # 序列并行用的设备网格
        # ShardingManager 负责管理序列并行时的数据分片和聚合
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # 根据数据并行度 (dp_size) 调整每个 GPU 的 batch size
        # 例如: 全局 batch_size=32, 4卡 → 每卡 batch_size=8
        self._normalize_config_bsz()

        # 设置序列并行大小，默认为 1（不使用序列并行）
        # 序列并行将一条长序列切分到多个 GPU 上计算，适合处理超长文本
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        # remove_padding: 去除 batch 中的 padding token，只计算有效 token
        # 这可以显著提高计算效率，尤其是 batch 内序列长度差异大时
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)

        # 判断是否使用 LoRA 微调（有 adapter 路径或 lora_rank > 0）
        self.lora = self.config.model.get("lora_adapter_path") is not None or self.config.model.lora_rank > 0

        # Initialize resume-related variables
        self.resume_global_step = 0

        # build model
        self._build_model_optimizer()

        # Initialize checkpoint manager
        self._init_checkpoint_manager()

        self.load_checkpoint()

        if self.device_mesh.get_rank() == 0:
            print(self.config)

        self.device_name = self.config.trainer.device

    def _normalize_config_bsz(self):
        """
        将全局 batch size 转换为每个数据并行 rank 的 batch size。

        【计算逻辑】
        假设:
          - 全局 train_batch_size = 32（配置文件中设定的总 batch 大小）
          - dp_size = 4（4 个数据并行的 rank/GPU 组）
          - micro_batch_size_per_gpu = 4（每次前向传播处理的样本数）
        则:
          - 每卡 batch_size = 32 / 4 = 8（每轮每卡处理 8 个样本）
          - 梯度累积步数 = 8 / 4 = 2（每卡需要 2 次 micro batch 前向后累积梯度后再更新）
        """
        # 获取数据并行大小：如果使用了 Ulysses 序列并行，dp_size 是第 0 维（dp 维度）的大小
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        # 全局 batch size 必须能被 dp_size 整除
        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        # 将全局 batch size 转为每个 dp rank 的 batch size
        self.config.data.train_batch_size //= dp_size

        # 确保每卡的 batch size 能被 micro_batch_size 整除（用于梯度累积）
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        """
        构建训练和验证的 DataLoader。

        【关键设计】
        - 使用 DistributedSampler 确保每个 GPU 获得不同的数据子集
        - 使用 StatefulDataLoader 支持断点续训（可保存/恢复读取位置）
        - 序列并行时，同一 SP 组内的 GPU 获得相同数据（因为它们合作处理同一条序列）

        【例如】
        4 GPU, SP=2 的数据分配:
          GPU 0,1 是 SP 组 1 → 它们获得相同的样本，各处理序列的一半
          GPU 2,3 是 SP 组 2 → 它们获得另一批相同的样本
        """
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # 根据是否使用序列并行，确定数据分发的 rank 和 world_size
        if self.config.ulysses_sequence_parallel_size > 1:
            # 序列并行模式：使用 dp 维度的 rank（同一 SP 组内的 GPU 获得相同数据）
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            # 普通模式：每个 GPU 获得不同的数据
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        # 获取设备名称（"cuda" 或 "npu"），用于 pin_memory 配置
        device_name = get_device_name()

        # 构建训练 DataLoader
        # DistributedSampler 会自动将数据集分片到各 GPU
        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        # StatefulDataLoader 支持 state_dict()/load_state_dict()，实现断点续训
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,  # 已经是每卡的 batch size
            sampler=self.train_sampler,
            num_workers=8,  # 8 个工作进程并行加载数据
            pin_memory=True,  # 使用锁页内存加速 CPU→GPU 传输
            drop_last=True,  # 丢弃最后一个不完整的 batch
            pin_memory_device=device_name,
        )

        # 构建验证 DataLoader（不 shuffle，使用 micro_batch_size）
        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,  # 验证时用 micro batch size
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

    def _build_model_optimizer(self):
        """
        构建模型、FSDP 包装、优化器和学习率调度器。

        【整体流程】
        1. 加载预训练模型（如 LLaMA, Qwen 等）
        2. 可选：应用 monkey patch（用于 remove padding / 序列并行）
        3. 可选：应用 Liger kernel（加速等效核函数）
        4. 可选：应用 LoRA 高效微调
        5. 用 FSDP 包装模型（分片到多 GPU）
        6. 构建优化器和学习率调度器
        """
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights

        # 将模型从远程路径（如 HDFS / S3）复制到本地
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        # 支持加载外部自定义模型代码（例如自定义的 modeling_xxx.py）
        if self.config.model.get("external_lib", None) is not None:
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # 获取模型的数据类型（默认 fp32，通常设为 bf16 以节省显存）
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # 加载模型配置（不加载权重）
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        # 确保模型的最大位置编码能容纳训练数据的最大序列长度
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        # 序列并行必须开启 remove_padding（因为需要平坦化序列后再切分）
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # 使用权重初始化上下文管理器（支持 meta tensor 模式以节省内存）
        # meta tensor: 先创建模型结构但不分配实际权重内存，等 FSDP 分片后再加载
        # 注意：当模型使用 tie_word_embeddings 时不能用 meta tensor
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            # ========== 步骤 1: 加载预训练模型 ==========
            # 使用 HuggingFace AutoModelForCausalLM 自动识别模型架构并加载权重
            # attn_implementation="flash_attention_2": 使用 Flash Attention 2 加速注意力计算
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # ========== 步骤 2: 应用 monkey patch（可选）==========
            # 当使用 remove_padding 或序列并行时，需要修改模型的注意力层
            # 以支持变长序列输入（去除 padding 后序列长度不一）
            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # ========== 步骤 3: 应用 Liger Kernel（可选）==========
            # Liger Kernel 提供了融合的 Triton 核函数，可加速 CrossEntropy/RMSNorm/SwiGLU 等操作
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            # ========== 步骤 4: 应用 LoRA（可选）==========
            # LoRA (Low-Rank Adaptation) 只训练少量的低秩矩阵，大幅减少可训练参数
            # 例如: 7B 模型 + LoRA (rank=16) 只需训练 ~0.1% 的参数
            if self.lora:
                # LoRA 需要开启输入梯度（因为原始模型参数被冻结）
                self.model.enable_input_require_grads()

                lora_adapter_path = self.config.model.get("lora_adapter_path")
                if lora_adapter_path is not None:
                    # 情况 A: 加载已有的 LoRA adapter 继续训练
                    from peft import PeftModel

                    print(f"Loading pre-trained LoRA adapter for sft from: {lora_adapter_path}")

                    local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.use_shm)

                    self.model = PeftModel.from_pretrained(self.model, local_adapter_path, is_trainable=True)
                    peft_config = self.model.peft_config["default"]
                    if isinstance(peft_config.task_type, str):
                        peft_config.task_type = TaskType.CAUSAL_LM
                else:
                    # 情况 B: 创建新的 LoRA adapter
                    # target_modules 指定应用 LoRA 的层，如 ["q_proj", "v_proj"]
                    lora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "r": self.config.model.lora_rank,        # LoRA 秩，越大参数越多但表达能力越强
                        "lora_alpha": self.config.model.lora_alpha,  # LoRA 缩放因子
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "bias": "none",
                    }
                    self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                self.model = self.model.to(torch_dtype)

        # ========== 步骤 5: 梯度 Checkpointing（可选）==========
        # 用时间换空间：前向传播时不保存中间激活值，反向传播时重新计算
        # 可以显著减少显存占用，但会增加计算时间
        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        # ========== 步骤 6: FSDP 包装 ==========
        # 混合精度配置：
        # - param_dtype=bf16: 参数以 bf16 存储（节省显存）
        # - reduce_dtype=fp32: 梯度归约用 fp32（保证数值稳定性）
        # - buffer_dtype=fp32: 缓冲区用 fp32
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        # 获取 FSDP 包装策略，决定模型的哪些层应该被单独包装为 FSDP 单元
        # 通常每个 Transformer 层（TransformerDecoderLayer）是一个 FSDP 单元
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.lora,
        )

        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        # CPU 卸载配置：将不参与当前计算的参数卸载到 CPU 以节省 GPU 显存
        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        # 支持两种 FSDP 策略：
        # - "fsdp": PyTorch 原生 FSDP1 API
        # - "fsdp2": PyTorch >= 2.4 的新版 fully_shard API，更灵活更高效
        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            # FSDP1: 使用 FullyShardedDataParallel 包装整个模型
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,  # 从 meta tensor 初始化实际参数
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片（最节省显存）
                mixed_precision=mixed_precision,
                sync_module_states=True,  # 同步所有 rank 的模型状态
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            # FSDP2: 使用 fully_shard API（更现代的实现）
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,  # 前向传播后重新分片（节省显存）
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        # ========== 步骤 7: 构建优化器 ==========
        self.optimizer = build_optimizer(self.fsdp_model.parameters(), self.config.optim)

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        # 计算训练步数
        self.steps_per_epoch = len(self.train_dataloader)  # 每个 epoch 的步数
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs  # 总训练步数

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        # ========== 步骤 8: 构建学习率调度器 ==========
        # warmup 步数 = 总步数 × warmup 比例
        num_warmup_steps = int(self.total_steps * self.config.optim.lr_warmup_steps_ratio)

        # 支持两种学习率调度策略：
        # - cosine: 余弦退火（最常用），学习率从 warmup 后按余弦函数逐渐下降
        # - wsd: Warmup-Stable-Decay，先 warmup、然后保持稳定、最后衰减
        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
        """
        计算损失并执行反向传播。支持普通模式和 Ulysses 序列并行模式。

        【SFT 损失计算原理】
        SFT 使用“下一个 token 预测”的方式计算损失：
          输入:  [What, is, the, capital, of, France, ?]
          标签:  [is,   the, capital, of, France, ?,  Paris]   # 左移一位
          模型预测每个位置的下一个 token，用 CrossEntropyLoss 计算损失。

        【loss_mask 的作用】
        在 SFT 中，我们通常只计算"assistant 回复"部分的损失，而不计算"user 提问"部分。
        loss_mask 是一个 0/1 向量，1 表示需要计算损失的位置。
        例如: "User: 你好 Assistant: 你好！" → loss_mask = [0, 0, 1, 1, 1]

        Args:
            batch: 包含 input_ids, attention_mask, position_ids, loss_mask 的数据批次
            do_backward: 是否执行反向传播（验证时为 False）
            n_micro_batches: micro batch 的数量，用于归一化损失（梯度累积）

        Returns:
            loss: 当前 micro batch 的归一化损失
        """
        # 判断是否使用序列并行（需要同时开启 remove_padding 和 SP）
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # 将输入移动到 GPU 并准备 loss_mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        # loss_mask[:, 1:]: 去掉第一个 token 的 mask（因为标签是左移一位的）
        loss_mask = batch.pop("loss_mask")[:, 1:].reshape(-1).to(self.device_name)
        # reduction="none": 不自动求均值，保留每个 token 的损失（用于后续 loss_mask 加权）
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # 序列并行时使用 sharding_manager 上下文，否则用空上下文
        context = self.sharding_manager if use_sp else nullcontext()
        # torch.autocast: 自动混合精度，前向传播时自动将计算转为 bf16
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # ========== 普通模式（无序列并行） ==========
                # 标签 = 输入左移一位（下一个 token 预测）
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits  # (batch_size, seq_len, vocab_size)

                # 将 logits 和 labels 对齐：logits 去掉最后一个位置
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # 展平为 (batch_size * seq_len, vocab_size) 和 (batch_size * seq_len,)
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                # 计算每个 token 的交叉熵损失
                loss = loss_fct(shift_logits, shift_labels)
                # 用 loss_mask 只保留需要计算的位置（如 assistant 回复）
                loss = loss * loss_mask.to(loss.device)
            else:
                # ========== 序列并行模式 (Ulysses SP) ==========
                # 【核心假设】
                # 每个 SP 组合作处理同一条序列，即：
                # 1. 同一 SP 组内的所有 GPU 收到相同的 batch
                # 2. 不同 SP 组收到不同的 batch
                # 这是由 DistributedSampler 保证的

                batch_size, seqlen = input_ids.shape

                # 步骤 1: 去除 padding（unpad）
                # 将 (batch_size, seq_len) 的 padded 序列压缩为 (total_nnz,) 的紧凑序列
                # total_nnz = batch 中所有非 padding token 的总数
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # 也对 position_ids 做同样的 unpad 操作
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # 步骤 2: 为序列并行做 padding 和切分
                # 将序列平均分给 SP 组内的各个 GPU
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )

                # 步骤 3: 准备损失计算的标签（左移一位）
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                # 步骤 4: 前向传播（每个 GPU 只处理序列的一部分）
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Flash Attention varlen 不需要 attention_mask
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # 步骤 5: 在各 GPU 上局部计算损失，然后聚合
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # 从所有 SP rank 收集损失并去除 padding
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # 步骤 6: 将损失恢复为原始形状并应用 loss_mask
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # 去掉最后一个 token 的损失
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            # ========== 损失归一化 ==========
            # 计算有效 token 数（loss_mask 为 1 的位置）
            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                # 在所有 dp rank 间聚合有效 token 数，确保每个 rank 的损失权重一致
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            # 损失 = 所有有效 token 损失之和 / 有效 token 总数 × dp_size
            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            # 除以 micro batch 数量（梯度累积时的归一化）
            loss = loss / n_micro_batches

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        """
        执行一个完整的训练步骤（包含梯度累积）。

        【流程】
        1. 清除梯度 → 2. 将 batch 切分为 micro batches → 3. 前向+后向传播
        → 4. 梯度裁剪 → 5. 优化器更新 → 6. 学习率调度

        Args:
            batch: 一个完整的训练 batch

        Returns:
            dict: 包含训练损失、学习率、耗时等指标
        """
        start_time = time.time()

        self.fsdp_model.train()  # 设置为训练模式

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()  # 清除上一步的梯度

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        # ========== 梯度累积 (Gradient Accumulation) ==========
        # 将一个大 batch 切分为多个 micro batch，逐个计算损失并累积梯度
        # 这样可以在显存有限时模拟更大的有效 batch size
        # 例如: batch_size=8, micro_batch=4 → 2 次前向后向+累积梯度，再更新一次参数
        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch, n_micro_batches=n_micro_batches)
            step_loss += loss.item()

        # ========== 梯度裁剪 (Gradient Clipping) ==========
        # 防止梯度爆炸，将梯度范数限制在 clip_grad 以内
        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # 如果梯度范数不是有限值（NaN/Inf），跳过本次参数更新
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()  # 更新模型参数

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()  # 更新学习率

        # 获取当前学习率
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)

        # 计算本步耗时
        end_time = time.time()
        spend_time_per_step = end_time - start_time

        # 在所有 dp rank 间求平均损失（确保日志记录的是全局平均）
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
        return {
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/time(s)": spend_time_per_step,
        }

    def validation_step(self, batch: TensorDict):
        """
        执行一个验证步骤（不计算梯度，不更新参数）。
        """
        self.fsdp_model.eval()  # 设置为评估模式
        with torch.no_grad():  # 不计算梯度，节省显存
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            # 在所有 rank 间求平均损失
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    def save_checkpoint(self, step):
        """
        保存训练 checkpoint，包含模型、优化器和 dataloader 状态。

        【保存内容】
        - 模型权重 + 优化器状态（由 FSDPCheckpointManager 管理）
        - DataLoader 状态（用于断点续训时恢复数据读取位置）
        - Checkpoint 跟踪文件（记录最新的 checkpoint 步数）

        【保存目录结构示例】
        default_local_dir/
        ├── global_step_100/
        │   ├── model/              # 模型权重
        │   ├── optimizer/          # 优化器状态
        │   └── data.pt             # DataLoader 状态
        ├── global_step_200/
        └── latest_checkpointed_iteration.txt  # 跟踪文件

        Args:
            step: 当前的全局训练步数
        """
        from verl.utils.fs import local_mkdir_safe

        # 构建 checkpoint 保存路径
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # 最多保留的 checkpoint 数量（超出时自动删除最旧的）
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

        # 使用 checkpoint 管理器保存模型和优化器状态
        self.checkpoint_manager.save_checkpoint(
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # 只在 rank 0 保存 dataloader 状态和跟踪文件（避免多个进程重复写入）
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")

            # 保存 StatefulDataLoader 的状态（包括当前读取位置等）
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

            # 更新 checkpoint 跟踪文件（原子写入，防止写到一半时崩溃破坏文件）
            tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)  # 原子重命名
            print(f"Updated checkpoint tracker: {tracker_file}")

        # 如果配置了 HDFS 路径，将 checkpoint 复制到 HDFS（云端备份）
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        # 等待所有 rank 完成保存
        torch.distributed.barrier()

    def _init_checkpoint_manager(self):
        """
        初始化 checkpoint 管理器，配置要保存和加载的内容。

        save_contents / load_contents 可以包含:
        - "model": 模型权重
        - "optimizer": 优化器状态（包含动量等）
        - "extra": 额外信息（如学习率调度器状态）
        """
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        checkpoint_config_dict = DictConfig({
            "load_contents": load_contents,
            "save_contents": save_contents,
        })

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        """
        加载 checkpoint 以恢复训练（断点续训）。

        【流程】
        1. 确定要加载的 checkpoint 路径
        2. 提取训练步数
        3. 加载模型 + 优化器状态
        4. 加载 DataLoader 状态

        Returns:
            int: 恢复的训练步数，如果没有 checkpoint 则返回 0
        """
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # 从路径中提取训练步数（例如 global_step_1000 → 1000）
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # 加载模型和优化器状态
        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        log_with_rank(
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # 加载 DataLoader 状态（恢复数据读取位置）
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """从 checkpoint 中加载 DataLoader 状态，用于恢复数据读取位置。"""
        dataloader_path = os.path.join(checkpoint_path, "data.pt")

        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )
        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """
        根据 resume_mode 配置确定要恢复的 checkpoint 路径。

        【三种模式】
        - "disable": 不恢复，从头开始训练
        - "auto": 自动查找最新的 checkpoint（或使用指定路径）
        - "resume_path": 使用指定的 checkpoint 路径
        """
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # 尝试在默认目录中查找最新的 checkpoint
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """在默认本地目录中查找最新的 checkpoint。"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def fit(self):
        """
        执行完整的训练循环。

        【训练循环结构】
        for epoch in range(total_epochs):
            for step, batch in enumerate(train_dataloader):
                1. 执行 training_step (forward + backward + update)
                2. 记录训练指标
                3. 定期执行验证 (test_freq)
                4. 定期保存 checkpoint (save_freq)
                5. 达到总步数时提前结束
        """
        rank = self.device_mesh.get_rank()

        # 初始化训练指标跟踪（仅 rank 0，如 WandB / TensorBoard）
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # 从断点处继续
        last_valid_metric = None

        # 计算总训练步数（主要用于提前结束训练）
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        # 支持通过配置直接指定总步数（覆盖计算值）
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # StatefulDataLoader 会自动从上次保存的位置继续读取数据
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        # 计算从哪个 epoch 开始（用于 sampler.set_epoch）
        start_epoch = global_step // self.steps_per_epoch

        train_time = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            # 设置 epoch 编号，确保每个 epoch 的数据顺序不同（shuffle）
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,  # 只在 rank 0 显示进度条
                )
            ):
                global_step += 1
                # 将数据转为 TensorDict 并移动到 GPU
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data)
                train_time += metric["train/time(s)"]
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                # 判断是否需要验证或保存
                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # ========== 验证阶段 ==========
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                            self.device_name
                        )
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()  # 同步所有 rank

                # ========== 保存 Checkpoint ==========
                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                # ========== 提前结束 ==========
                if is_last_step:
                    if rank == 0:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return

# ==================== 模块级别函数 ====================


def run_sft(config):
    """
    SFT 训练的主入口函数。

    【整体流程】
    1. 初始化分布式环境（NCCL/HCCL 后端）
    2. 创建设备网格（DeviceMesh）
    3. 加载 tokenizer 和数据集
    4. 创建 FSDPSFTTrainer 并开始训练
    5. 销毁分布式进程组

    【使用示例】
    通常通过命令行调用:
        torchrun --nproc_per_node=4 fsdp_sft_trainer.py
    这会启动 4 个进程，每个进程对应一张 GPU。

    Args:
        config: Hydra 配置对象
    """
    device_name = get_device_name()  # "cuda" 或 "npu"
    # 初始化分布式进程组（设置 NCCL 后端、分配 rank 等）
    local_rank, rank, world_size = initialize_global_process_group()

    # 创建 FSDP 设备网格（所有 GPU 在一个维度上）
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    # 创建 Ulysses 序列并行设备网格
    # 例如: 8 GPU, SP=2 → dp_size=4, mesh_shape=(4, 2)
    # 意味着 4 组数据并行，每组 2 个 GPU 做序列并行
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    # 加载 tokenizer
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    # 创建训练和验证数据集
    train_dataset = create_sft_dataset(
        config.data.train_files, config.data, tokenizer, max_samples=config.data.get("train_max_samples", -1)
    )
    val_dataset = create_sft_dataset(
        config.data.val_files, config.data, tokenizer, max_samples=config.data.get("val_max_samples", -1)
    )

    # 创建训练器并开始训练
    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()

    # 训练完成后销毁分布式进程组
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    """
    程序入口，由 Hydra 装饰器自动加载 YAML 配置文件。

    配置文件位置: config/sft_trainer.yaml
    使用方式: torchrun --nproc_per_node=4 fsdp_sft_trainer.py [Hydra 覆写参数]
    例如: torchrun --nproc_per_node=4 fsdp_sft_trainer.py model.partial_pretrain=/path/to/model
    """
    # 自动检测硬件并设置设备类型（如华为昇腾 NPU）
    auto_set_device(config)

    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, max_samples=-1):
    """
    创建 SFT 数据集，支持三种数据集类型。

    【数据集选择优先级】
    1. 自定义数据集类 (custom_cls): 用户自定义的数据集实现
    2. 多轮对话数据集 (MultiTurnSFTDataset): 处理多轮对话格式的数据
    3. 单轮数据集 (SFTDataset): 默认的单轮问答数据集

    Args:
        data_paths: 数据文件路径（支持 parquet 格式）
        data_config: 数据配置
        tokenizer: HuggingFace tokenizer
        max_samples: 最大样本数（-1 表示不限制）

    Returns:
        Dataset: 构建好的 SFT 数据集
    """
    # 优先检查是否指定了自定义数据集类
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object

        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
    # 其次检查是否启用多轮对话数据集
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # 默认使用单轮数据集
    else:
        dataset_cls = SFTDataset

    # 创建并返回数据集实例
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config, max_samples=max_samples)
    return dataset


if __name__ == "__main__":
    main()
