# Part 3: verl SFT Trainer 架构全景

> 本文是 verl SFT 讲解系列的第 3 部分，从 `torchrun` 命令启动开始，完整走一遍 verl SFT 的初始化和训练全流程。
> 对应实现文档：`02_training_config_and_flow.md`

---

## 1. 从一条命令说起

启动 verl SFT 训练，只需要一条命令：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=sft_train.parquet \
    data.val_files=sft_val.parquet \
    data.multiturn.enable=true \
    model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \
    trainer.total_epochs=2 \
    ...
```

这条命令涉及几个关键概念，我们逐个拆解。

### 1.1 `torchrun` — 分布式启动器

`torchrun` 是 PyTorch 提供的分布式训练启动工具。它做的事情是：

```
torchrun --nproc_per_node=8 ...
                    ↓
    启动 8 个进程，每个进程绑定一张 GPU
    每个进程拥有独立的 rank（0-7）
    进程之间通过 NCCL 通信

    GPU 0 ← 进程 rank=0
    GPU 1 ← 进程 rank=1
    GPU 2 ← 进程 rank=2
    ...
    GPU 7 ← 进程 rank=7
```

**关键参数**：
- `--nnodes=1`：只用 1 台机器
- `--nproc_per_node=8`：这台机器上启动 8 个进程（对应 8 张 GPU）
- `--standalone`：单机模式，不需要多机协调

### 1.2 `-m verl.trainer.fsdp_sft_trainer` — 入口文件

`-m` 告诉 Python "把这个模块当脚本运行"。实际执行的是：

```
verl/trainer/fsdp_sft_trainer.py 的 main() 函数
```

### 1.3 Hydra 配置覆盖

`data.train_files=...` 这些是 **Hydra** 的配置覆盖语法。verl 使用 Hydra 管理配置：

```
默认配置: verl/trainer/config/sft_trainer.yaml
    ↓
命令行覆盖: data.train_files=xxx model.partial_pretrain=xxx ...
    ↓
最终配置: 合并后的完整 config 字典
```

Hydra 的好处是：不用每次修改配置文件，直接在命令行上覆盖想改的参数。

---

## 2. 程序启动全流程

以下是从 `main()` 开始，到训练循环前的**完整初始化流程**：

```
main()                               ← fsdp_sft_trainer.py:843
  │
  ├─ auto_set_device(config)          ← 自动检测设备(CUDA/NPU)
  │
  └─ run_sft(config)                  ← fsdp_sft_trainer.py:806
       │
       ├─ ① 分布式初始化
       │   initialize_global_process_group()
       │   → 返回 local_rank, rank, world_size
       │
       ├─ ② 创建设备网格 (Device Mesh)
       │   device_mesh = init_device_mesh(shape=(8,))         # FSDP 用
       │   ulysses_device_mesh = init_device_mesh(shape=(4,2)) # 序列并行用(可选)
       │
       ├─ ③ 加载分词器
       │   tokenizer = hf_tokenizer("Qwen/Qwen2.5-Coder-7B-Instruct")
       │
       ├─ ④ 创建数据集
       │   train_dataset = create_sft_dataset(...)  # → MultiTurnSFTDataset
       │   val_dataset = create_sft_dataset(...)
       │
       ├─ ⑤ 创建 Trainer 实例
       │   trainer = FSDPSFTTrainer(config, device_mesh, ..., train_dataset, val_dataset)
       │   │
       │   │  FSDPSFTTrainer.__init__() 内部:
       │   ├─ (a) _normalize_config_bsz()    # 按 GPU 数归一化 batch size
       │   ├─ (b) _build_dataloader()        # 构建 DataLoader + Sampler
       │   ├─ (c) _build_model_optimizer()   # 加载模型 + FSDP 包装 + 优化器 + 调度器
       │   ├─ (d) _init_checkpoint_manager() # 初始化检查点管理器
       │   └─ (e) load_checkpoint()          # 自动恢复（如果有之前的检查点）
       │
       ├─ ⑥ 开始训练
       │   trainer.fit()
       │
       └─ ⑦ 清理
           destroy_global_process_group()
```

下面我们逐步详细讲解每个阶段。

---

## 3. 阶段 ①②: 分布式初始化与设备网格

### 3.1 什么是设备网格 (Device Mesh)？

Device Mesh 是 PyTorch 对多 GPU 拓扑的抽象。把它想象成一个"GPU 矩阵"：

```
单纯 FSDP 模式 (默认):
  device_mesh = shape (8,)
  维度名: ("fsdp",)

  ┌──────────────────────────────┐
  │ GPU0  GPU1  GPU2  GPU3       │
  │ GPU4  GPU5  GPU6  GPU7       │ ← 8个GPU组成一个 FSDP 组
  └──────────────────────────────┘
  所有 GPU 一起做 FSDP 参数分片
```

```
FSDP + 序列并行 (SP) 模式 (可选):
  ulysses_device_mesh = shape (4, 2)
  维度名: ("dp", "sp")

  ┌──────────────────────────────┐
  │  DP组0:  [GPU0, GPU1]  ← SP组 │  数据并行组0，2个GPU做序列并行
  │  DP组1:  [GPU2, GPU3]  ← SP组 │
  │  DP组2:  [GPU4, GPU5]  ← SP组 │
  │  DP组3:  [GPU6, GPU7]  ← SP组 │  数据并行组3
  └──────────────────────────────┘
  dp=4: 4个数据并行组，各看不同的数据
  sp=2: 每组内2个GPU，分摊同一条长序列
```

对应源码（`fsdp_sft_trainer.py:808-816`）：

```python
# FSDP 设备网格
device_mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(world_size,),       # (8,)
    mesh_dim_names=("fsdp",)
)

# 序列并行设备网格
dp_size = world_size // config.ulysses_sequence_parallel_size  # 8 // 2 = 4
ulysses_device_mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),  # (4, 2)
    mesh_dim_names=("dp", "sp")
)
```

### 3.2 对于我们的项目

我们的配置是 `ulysses_sequence_parallel_size=2`（2 路序列并行），所以：
- 8 张 GPU 被分成 4 个数据并行组
- 每组 2 张 GPU 共同处理一条序列
- 序列并行适合处理长序列（我们的 max_length=4096）

---

## 4. 阶段 ⑤(a): Batch Size 归一化

这是一个容易困惑的点。我们配置的是 **全局 batch size**，但每张 GPU 只处理一部分。

```python
# _normalize_config_bsz() 源码 (fsdp_sft_trainer.py:144-155)
def _normalize_config_bsz(self):
    dp_size = 4  # 数据并行组数
    # 全局 batch size 除以 数据并行数 = 每个 DP 组的 batch size
    self.config.data.train_batch_size //= dp_size
    # 128 // 4 = 32 (每个 DP 组处理 32 个样本)
```

完整的 batch size 分解：

```
全局 batch size = 128
       ÷
数据并行组数 (dp_size) = 4       ← 4 个 DP 组各看不同数据
       =
每 DP 组 batch size = 32
       ÷
micro_batch_size_per_gpu = 2     ← 每次喂给 GPU 的小批量
       =
梯度累积步数 = 16               ← 累积 16 次梯度再更新一次

实际效果: 每步 128 个样本的梯度参与一次参数更新
         但每次 forward 只处理 2 个样本（省显存）
```

**为什么要这样？** 因为 7B 模型 + 4096 序列长度已经占用大量显存。一次性放 32 个样本到 GPU 会 OOM（内存不够）。梯度累积让我们"分批算梯度，攒够了再更新"。

---

## 5. 阶段 ⑤(b): DataLoader 构建

DataLoader 负责把 Dataset 的样本打包成 batch，并分发给不同 GPU。

```python
# _build_dataloader() 源码 (fsdp_sft_trainer.py:157-205)

# DistributedSampler: 确保每个 GPU 看到不同的数据
self.train_sampler = DistributedSampler(
    self.train_dataset,
    shuffle=True,          # 训练数据随机打乱
    num_replicas=world_size,  # GPU 总数
    rank=rank,             # 当前 GPU 的编号
    drop_last=True         # 丢弃最后不完整的 batch
)

# StatefulDataLoader: 支持保存/恢复状态的 DataLoader
self.train_dataloader = StatefulDataLoader(
    dataset=self.train_dataset,
    batch_size=config.data.train_batch_size,  # 归一化后的 batch size
    sampler=self.train_sampler,
    num_workers=8,         # 8 个后台进程预加载数据
    pin_memory=True,       # 锁页内存，加速 CPU→GPU 传输
    drop_last=True,
)
```

### DistributedSampler 如何工作？

假设 10000 条数据，4 个 DP 组：

```
全部数据: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]  (共 10000 条)
                        ↓ 随机打乱
         [7, 2, 9, 0, 5, 3, 1, 8, 4, 6, ...]
                        ↓ 按 DP 组分配
  DP 组 0 (rank 0): [7, 5, 4, ...]   ← 每4个取1个
  DP 组 1 (rank 1): [2, 3, 6, ...]
  DP 组 2 (rank 2): [9, 1, ...]
  DP 组 3 (rank 3): [0, 8, ...]
```

每个 DP 组只看到 2500 条数据，互不重叠。

### StatefulDataLoader 的特殊之处

普通 DataLoader 没有"记忆"——如果训练中断，重启后从头开始。但 StatefulDataLoader 可以保存当前状态（读到哪条数据了），恢复训练时从断点继续。

---

## 6. 阶段 ⑤(c): 模型构建与 FSDP 包装

这是最核心的部分。分为 4 个子步骤：

### 6.1 加载预训练模型

```python
# _build_model_optimizer() 源码 (fsdp_sft_trainer.py:207-246)

# 1. 加载模型配置
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# 2. 加载模型权重
with init_context():  # 控制内存分配方式
    self.model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype=torch.bfloat16,              # 用 bf16 节省显存
        attn_implementation="flash_attention_2",   # Flash Attention 加速
    )

# 3. 开启梯度检查点（用计算换内存）
self.model.gradient_checkpointing_enable()
```

**Flash Attention 2**：一种高效的注意力计算算法，显著减少显存使用和计算时间。
**Gradient Checkpointing**：正常训练会保存所有中间激活值用于反向传播。梯度检查点只保存部分，需要时重新计算。牺牲约 30% 速度换取约 50% 显存节省。

### 6.2 FSDP 包装 — 核心概念

**FSDP (Fully Sharded Data Parallelism)** 是 PyTorch 的分布式训练策略，它做的事情是：

```
传统数据并行 (DDP):
  每张 GPU 都有模型的完整副本
  GPU 0: [完整模型 7B] + [完整优化器状态 14B] = 21B 参数
  GPU 1: [完整模型 7B] + [完整优化器状态 14B] = 21B 参数
  ...
  问题: 7B 模型 + 优化器 ≈ 42GB，单卡 24GB 放不下！

FSDP (完全分片):
  把模型参数切成碎片，分摊到各 GPU
  GPU 0: [模型碎片 0.875B] + [优化器碎片 1.75B]  ≈ 5.25GB
  GPU 1: [模型碎片 0.875B] + [优化器碎片 1.75B]  ≈ 5.25GB
  ...
  需要计算时，临时 all-gather 收集完整参数
  计算完后，再分片释放
```

FSDP 的工作流程：

```
┌────────────────────────────────────────────────────┐
│                 FSDP 一次 forward-backward           │
│                                                      │
│  1. Forward 开始                                     │
│     Layer 1: all-gather 收集完整参数                  │
│              → 计算 forward                          │
│              → 释放非本地参数 (reshard)                │
│     Layer 2: all-gather → 计算 → reshard             │
│     ...                                              │
│     Layer N: all-gather → 计算 → reshard             │
│                                                      │
│  2. Backward 开始                                    │
│     Layer N: all-gather → 计算梯度 → reshard          │
│     ...                                              │
│     Layer 1: all-gather → 计算梯度 → reshard          │
│                                                      │
│  3. 梯度 reduce-scatter                              │
│     每个 GPU 只保留自己负责的梯度分片                    │
│                                                      │
│  4. Optimizer step                                   │
│     每个 GPU 只更新自己的参数分片                       │
└────────────────────────────────────────────────────┘
```

### 6.3 FSDP2 包装源码

我们用的是 FSDP2（`model.strategy=fsdp2`），对应源码（`fsdp_sft_trainer.py:325-340`）：

```python
# Mixed Precision 策略
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,   # 参数用 bf16 (省显存)
    reduce_dtype=torch.float32,    # 梯度归约用 fp32 (保精度)
    cast_forward_inputs=True,
)

fsdp_kwargs = {
    "mesh": self.device_mesh,
    "mp_policy": mp_policy,
    "reshard_after_forward": True,  # forward 后释放收集的参数
}

# 对模型应用 FSDP2 分片
apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
```

**Mixed Precision (混合精度)**：
- 参数和计算用 `bfloat16`（16位浮点数，省一半显存）
- 梯度归约用 `float32`（32位浮点数，避免精度损失累积）
- 这是"既省显存又保精度"的黄金组合

### 6.4 优化器和学习率调度器

```python
# 构建优化器 (fsdp_sft_trainer.py:346)
self.optimizer = build_optimizer(self.fsdp_model.parameters(), config.optim)
# → 创建 AdamW 优化器，lr=2e-5, weight_decay=0.01

# 计算 warmup 步数
num_warmup_steps = int(total_steps * 0.05)  # 总步数的 5%

# 创建 Cosine 学习率调度器 (fsdp_sft_trainer.py:361-364)
self.lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=self.optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=self.total_steps
)
```

**Cosine Schedule with Warmup** 长什么样：

```
学习率
  ↑
  │         ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
  │        ╱                    ╲
  │       ╱                      ╲
  │      ╱                        ╲
  │     ╱                          ╲
  │    ╱                            ╲
  │   ╱                              ╲
  │──╱                                ╲──
  └──────────────────────────────────────→ 步数
    warmup    peak(2e-5)     cosine 衰减

  前 5% 步数: 线性从 0 升到 2e-5 (warmup 热身)
  之后: 余弦曲线从 2e-5 缓慢降到 ~0
```

**为什么要 warmup？** 训练开始时模型参数是随机方向的，梯度噪声大。如果一开始就用大学习率，可能震荡不收敛。先用小学习率"热身"，等梯度方向稳定后再加大。

---

## 7. 阶段 ⑤(d)(e): 检查点管理与自动恢复

### 7.1 检查点管理器初始化

```python
# _init_checkpoint_manager() 源码 (fsdp_sft_trainer.py:587-612)

save_contents = ["model", "optimizer", "extra", "hf_model"]
#                  ↑          ↑          ↑          ↑
#              FSDP分片     优化器分片   调度器+RNG    完整HF模型

self.checkpoint_manager = FSDPCheckpointManager(
    model=self.fsdp_model,
    optimizer=self.optimizer,
    lr_scheduler=self.lr_scheduler,
    processing_class=self.tokenizer,
    checkpoint_config=checkpoint_config,
)
```

`save_contents` 中 `"hf_model"` 很重要——它会在检查点中额外保存一份**完整的 HuggingFace 格式模型**。这样每个检查点都可以直接用 vLLM 加载进行评测，不需要手动合并 FSDP 分片。

### 7.2 自动恢复逻辑

```python
# load_checkpoint() → _determine_resume_path() (fsdp_sft_trainer.py:673-696)

resume_mode 有三种:
  "auto"     → 自动寻找最新检查点，有就恢复，没有就从头开始
  "disable"  → 忽略任何已有检查点，强制从头开始
  "resume_path" → 从指定路径恢复
```

恢复时加载什么：
```
检查点目录 global_step_500/
├── model_world_size_8_rank_0.pt    → 恢复模型参数 (rank 0 的分片)
├── optim_world_size_8_rank_0.pt    → 恢复优化器状态
├── extra_state_world_size_8_rank_0.pt → 恢复学习率调度器 + 随机数状态
├── data.pt                          → 恢复 DataLoader 位置
└── huggingface/                     → (不用于恢复，用于评测)
```

---

## 8. 阶段 ⑥: 训练主循环 `fit()` 概览

`fit()` 是训练的主循环（`fsdp_sft_trainer.py:713-803`），结构如下：

```python
def fit(self):
    # 初始化 WandB 追踪
    tracking = Tracking(project_name=..., experiment_name=...)

    global_step = self.resume_global_step  # 从恢复点开始

    for epoch in range(total_epochs):           # ← 外层循环: 遍历 epoch
        self.train_sampler.set_epoch(epoch)      # 设置随机种子保证每 epoch 不同顺序

        for data in self.train_dataloader:       # ← 内层循环: 遍历 batch
            global_step += 1

            # ① 训练一步
            metric = self.training_step(data)
            tracking.log(metric, step=global_step)

            # ② 验证 (每 500 步)
            if global_step % test_freq == 0:
                val_loss = 对 val_dataloader 计算平均 loss
                tracking.log({"val/loss": val_loss}, step=global_step)

            # ③ 保存检查点 (每 500 步)
            if global_step % save_freq == 0:
                self.save_checkpoint(step=global_step)

            # ④ 提前退出
            if global_step >= total_training_steps:
                return
```

具体的 `training_step()` 内部细节将在 **Part 4** 中逐行解析。

---

## 9. 完整架构图

把所有组件放在一起看：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     verl SFT 训练架构                                │
│                                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐        │
│  │ Hydra    │  │  torchrun    │  │  sft_trainer.yaml      │        │
│  │ Config   │→ │  8 进程启动   │→ │  默认配置 + 命令行覆盖   │        │
│  └──────────┘  └──────────────┘  └────────────────────────┘        │
│                                           ↓                         │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                    run_sft()                                │     │
│  │                                                            │     │
│  │  ┌────────────────┐   ┌────────────────────────────────┐  │     │
│  │  │ Device Mesh    │   │ Tokenizer + Dataset             │  │     │
│  │  │ (8,) FSDP      │   │ MultiTurnSFTDataset             │  │     │
│  │  │ (4,2) DP+SP    │   │  → Parquet → tokenize → tensor  │  │     │
│  │  └────────────────┘   └────────────────────────────────┘  │     │
│  │           ↓                          ↓                     │     │
│  │  ┌────────────────────────────────────────────────────┐   │     │
│  │  │              FSDPSFTTrainer                         │   │     │
│  │  │                                                    │   │     │
│  │  │  ┌───────────────┐  ┌────────────────────────┐    │   │     │
│  │  │  │ FSDP Model    │  │ StatefulDataLoader       │    │   │     │
│  │  │  │ Qwen2.5-7B    │  │ + DistributedSampler    │    │   │     │
│  │  │  │ bf16 + FA2    │  └────────────────────────┘    │   │     │
│  │  │  └───────────────┘               ↓                │   │     │
│  │  │         ↕                    fit() 训练循环         │   │     │
│  │  │  ┌───────────────┐    ┌─────────────────────┐    │   │     │
│  │  │  │ AdamW 优化器   │    │ training_step()     │    │   │     │
│  │  │  │ Cosine LR     │ ←  │ → forward → loss    │    │   │     │
│  │  │  │ Grad Clip     │    │ → backward → update  │    │   │     │
│  │  │  └───────────────┘    └─────────────────────┘    │   │     │
│  │  │         ↕                        ↓                │   │     │
│  │  │  ┌───────────────┐    ┌─────────────────────┐    │   │     │
│  │  │  │ CheckpointMgr │    │ WandB Tracking       │    │   │     │
│  │  │  │ FSDP shards   │    │ train/loss, val/loss │    │   │     │
│  │  │  │ + HF export   │    │ lr, grad_norm, time  │    │   │     │
│  │  │  └───────────────┘    └─────────────────────┘    │   │     │
│  │  └────────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. 知识点总结

| 概念 | 一句话解释 |
|------|-----------|
| **torchrun** | PyTorch 分布式启动器，每 GPU 启动一个进程 |
| **Hydra** | 配置管理框架，支持 YAML 默认配置 + 命令行覆盖 |
| **Device Mesh** | GPU 拓扑的抽象，定义哪些 GPU 组成 FSDP 组 / SP 组 |
| **FSDP** | 完全分片数据并行，把模型参数/梯度/优化器状态分片到各 GPU |
| **FSDP2** | FSDP 的新版 API（PyTorch ≥ 2.4），使用 `fully_shard` 接口 |
| **Mixed Precision** | 参数用 bf16 省显存，梯度归约用 fp32 保精度 |
| **Flash Attention 2** | 高效注意力计算，减少显存和计算时间 |
| **Gradient Checkpointing** | 不保存全部中间激活，需要时重算，用时间换空间 |
| **DistributedSampler** | 确保每个 GPU 看到不同的数据子集 |
| **StatefulDataLoader** | 支持保存/恢复状态的 DataLoader，训练中断后能从断点继续 |
| **Cosine Schedule** | 学习率先 warmup 再余弦衰减，兼顾稳定性和训练效果 |
| **Gradient Accumulation** | 多个 micro-batch 的梯度累积后再更新，等效于大 batch 但省显存 |

---

## 11. 与实现文档的对应

| 本讲内容 | 对应实现文档 |
|----------|-------------|
| 启动命令和入口 | `02_training_config_and_flow.md` 第 2-3 节 |
| FSDP 配置 | `02_training_config_and_flow.md` 第 4 节 |
| 超参数（概览） | `02_training_config_and_flow.md` 第 5 节（Part 5 详讲） |
| 训练循环概览 | `02_training_config_and_flow.md` 第 6 节（Part 4 详讲） |

**下一部分** 将进入训练循环内部，逐行解析 `training_step()` 和 `_compute_loss_and_backward()` 的代码。

---

> **思考题**：
> 1. 如果我们有 4 张 GPU（不是 8 张），全局 batch size 仍然是 128，`micro_batch_size_per_gpu=2`，那梯度累积步数是多少？
> 2. FSDP 把模型参数分片到各 GPU，那在做 forward 计算时每个 GPU 怎么拿到完整参数？
> 3. 为什么 `save_contents` 要包含 `"hf_model"`？如果不包含，评测时会遇到什么问题？
