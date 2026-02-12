# Part 6: 检查点管理与训练监控

> 对应实现文档: `03_training_monitoring.md`
> 核心源码: `verl/utils/checkpoint/fsdp_checkpoint_manager.py`, `verl/utils/tracking.py`

---

## 1. 为什么检查点和监控如此重要？

SFT 训练通常需要数小时甚至数天，在这期间可能遇到：
- **硬件故障**: GPU OOM、NCCL 超时、节点宕机
- **训练异常**: loss 爆炸、梯度消失、过拟合
- **资源限制**: 云实例预付时间到期、抢占式实例被回收

如果没有检查点机制，一次中断就意味着从头再来；如果没有训练监控，你可能训练了几千步才发现模型一直没在学习。

检查点（Checkpoint）和监控（Monitoring）就像训练的"存档"和"仪表盘"：

```
训练循环                         监控系统
  │                               │
  ├─ step 1 → loss=2.5 ────────→ WandB: 记录 train/loss=2.5
  ├─ step 2 → loss=2.3 ────────→ WandB: 记录 train/loss=2.3
  ├─ ...                          │
  ├─ step 500 → val_loss=1.5 ──→ WandB: 记录 val/loss=1.5
  │  └─ 保存 checkpoint ────────→ 磁盘: global_step_500/
  ├─ ...                          │
  ├─ step 800 → 💥 GPU 故障!      │
  │                               │
  ├─ 重启训练 ────────────────────→ 加载 global_step_500/
  ├─ step 501 → 继续训练          │  ← 只丢失 300 步的工作
  └─ ...
```

---

## 2. verl 检查点架构：两层设计

verl 的检查点系统采用两层设计：

```
FSDPSFTTrainer.save_checkpoint()    ← 上层：协调保存流程
  │
  ├─ FSDPCheckpointManager.save_checkpoint()  ← 下层：保存模型+优化器
  │     ├─ 每个 rank 保存自己的分片文件
  │     ├─ rank 0 保存 HF config + tokenizer
  │     └─ (可选) rank 0 导出完整 HF 模型
  │
  ├─ 保存 DataLoader 状态 (data.pt)        ← 上层独有
  └─ 更新 tracker 文件                      ← 上层独有
```

### 2.1 上层: `FSDPSFTTrainer.save_checkpoint()`

这是 Trainer 中直接调用的保存方法：

```python
# fsdp_sft_trainer.py: save_checkpoint()

def save_checkpoint(self, step):
    # 1. 构建保存路径
    local_global_step_folder = os.path.join(
        self.config.trainer.default_local_dir,  # 如 "checkpoints/rlvr_coding_model/phase1_sft"
        f"global_step_{step}"                    # 如 "global_step_500"
    )

    # 2. 获取最大保留数量（如 5）
    max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

    # 3. 调用 CheckpointManager 保存模型和优化器
    self.checkpoint_manager.save_checkpoint(
        local_path=local_global_step_folder,
        global_step=step,
        max_ckpt_to_keep=max_ckpt_to_keep
    )

    # 4. 只在 rank 0 保存额外信息
    if self.device_mesh.get_rank() == 0:
        # 保存 DataLoader 状态（记住当前读到了哪个 batch）
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, os.path.join(local_global_step_folder, "data.pt"))

        # 原子更新 tracker 文件
        tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
        temp_tracker_file = tracker_file + ".tmp"
        with open(temp_tracker_file, "w") as f:
            f.write(str(step))
        os.rename(temp_tracker_file, tracker_file)  # 原子重命名，防止写一半崩溃

    # 5. 如果配置了 HDFS，还会把 checkpoint 复制到远端
    torch.distributed.barrier()  # 等待所有 rank 完成
```

**关键细节：原子写入 tracker 文件**

```python
# 为什么不直接 write？
# 如果 write 到一半机器崩了，tracker 文件会损坏，导致无法恢复

# 正确做法：先写临时文件，再原子重命名
with open(tracker_file + ".tmp", "w") as f:
    f.write(str(step))
os.rename(tracker_file + ".tmp", tracker_file)  # os.rename 在同一文件系统上是原子操作
```

### 2.2 下层: `FSDPCheckpointManager`

继承自 `BaseCheckpointManager`，专门处理 FSDP 模型的保存和加载。

**类的继承关系：**

```
BaseCheckpointManager (checkpoint_manager.py)
  ├─ 管理 save/load 的内容控制 (should_save_model, should_load_optimizer 等)
  ├─ 检查点轮转 (remove_previous_save_local_path)
  ├─ RNG 状态管理 (get_rng_state, load_rng_state)
  └─ find_latest_ckpt_path() ← 模块级函数

FSDPCheckpointManager (fsdp_checkpoint_manager.py)
  ├─ 实现 save_checkpoint()：FSDP 分片保存
  └─ 实现 load_checkpoint()：FSDP 分片加载
```

---

## 3. 检查点保存：`save_checkpoint()` 深度解析

### 3.1 保存的内容和控制

`_init_checkpoint_manager()` 初始化时决定保存什么：

```python
# fsdp_sft_trainer.py: _init_checkpoint_manager()

save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
load_contents = checkpoint_config.get("load_contents", save_contents)

# 在我们项目中，还需要加上 "hf_model":
# save_contents = ["model", "optimizer", "extra", "hf_model"]
```

| 内容 | 属性检查 | 说明 | 必要性 |
|------|----------|------|--------|
| `model` | `should_save_model` | FSDP 分片模型权重 | **必须** - 恢复训练 |
| `optimizer` | `should_save_optimizer` | FSDP 分片优化器状态（含 Adam 的 m 和 v） | **推荐** - 无则动量归零 |
| `extra` | `should_save_extra` | LR scheduler + RNG 状态 | **推荐** - 无则 lr 从头开始 |
| `hf_model` | `should_save_hf_model` | 完整 HuggingFace 模型 | **项目必须** - 评测用 |

### 3.2 FSDP 分片保存原理

FSDP 将模型参数分片到各个 GPU，所以保存时每个 rank 只保存自己持有的那一份：

```
8 GPU 训练，模型 14GB
  ├─ GPU 0 持有 ~1.75GB 的模型分片 → model_world_size_8_rank_0.pt
  ├─ GPU 1 持有 ~1.75GB 的模型分片 → model_world_size_8_rank_1.pt
  ├─ ...
  └─ GPU 7 持有 ~1.75GB 的模型分片 → model_world_size_8_rank_7.pt
```

```python
# fsdp_checkpoint_manager.py: save_checkpoint() 核心逻辑

# 1. 设置保存模式为 SHARDED（分片保存，而不是聚合后保存）
state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
    # 2. 保存模型分片
    model_path = f"model_world_size_{self.world_size}_rank_{self.rank}.pt"
    model_state_dict = self.model.state_dict()     # 只获取本 rank 的分片
    torch.save(model_state_dict, model_path)

    # 3. 保存优化器分片（含 Adam 的 m 和 v 动量）
    optim_path = f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"
    optimizer_state_dict = self.optimizer.state_dict()
    torch.save(optimizer_state_dict, optim_path)

    # 4. 保存 LR scheduler 和 RNG 状态
    extra_path = f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
    extra_state_dict = {
        "lr_scheduler": self.lr_scheduler.state_dict(),
        "rng": self.get_rng_state(),  # CPU/GPU/numpy/random 的随机种子
    }
    torch.save(extra_state_dict, extra_path)
```

**为什么用 SHARDED 模式？**

| 保存模式 | 原理 | 优点 | 缺点 |
|----------|------|------|------|
| FULL_STATE_DICT | 先 all-gather 合并完整权重，然后 rank 0 保存 | 保存一个文件 | 需要 1 张 GPU 能放下整个模型 |
| SHARDED_STATE_DICT | 每个 rank 直接保存自己的分片 | 不需要额外内存 | 恢复需要相同 GPU 数 |

对于 7B 模型 + 8 GPU，SHARDED 模式更实际。

### 3.3 HuggingFace 模型导出

这是我们项目的关键功能 —— 评测需要完整的 HF 格式模型：

```python
# fsdp_checkpoint_manager.py: save_checkpoint() 中的 HF 模型导出

if self.should_save_hf_model:
    # 1. 从所有 rank 收集完整模型权重到 rank 0 的 CPU
    state_dict = get_fsdp_full_state_dict(
        self.model,
        offload_to_cpu=True,     # 放到 CPU，避免 GPU OOM
        rank0_only=True          # 只在 rank 0 聚合
    )

    if self.rank == 0:
        # 2. 创建一个空壳模型（不分配内存）
        with init_empty_weights():
            save_model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.bfloat16)
        save_model.to_empty(device="cpu")

        # 3. 用完整权重保存为 HF 格式
        save_model.save_pretrained(hf_local_path, state_dict=state_dict)
        # 输出: model.safetensors (~14GB), config.json 等
```

**为什么需要 HF 格式？**

```
FSDP 分片格式                    HuggingFace 格式
model_rank_0.pt + ... + _7.pt   →  model.safetensors (单文件)
  ↓                                 ↓
只能 FSDP 恢复训练               vLLM / HF pipeline 直接加载推理
（需要相同 GPU 数量）              （任意 GPU 数量皆可）
                                    ↓
                                 我们的评测脚本使用 vLLM 加载
```

### 3.4 检查点轮转（自动清理旧检查点）

磁盘空间有限，不可能保留所有检查点。`max_ckpt_to_keep` 控制保留数量：

```python
# fsdp_checkpoint_manager.py: save_checkpoint() 中的轮转逻辑

if max_ckpt_to_keep and len(self.previous_saved_paths) >= max_ckpt_to_keep:
    # 计算需要删除的旧检查点数量
    keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
    # 删除最旧的检查点
    self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
    self.previous_saved_paths = self.previous_saved_paths[keep_start:]
```

```
max_ckpt_to_keep=3 时的行为:

保存 step 500:  [step_500]                     ← 保留
保存 step 1000: [step_500, step_1000]           ← 保留
保存 step 1500: [step_500, step_1000, step_1500] ← 满了
保存 step 2000: 删除 step_500 → [step_1000, step_1500, step_2000]
保存 step 2500: 删除 step_1000 → [step_1500, step_2000, step_2500]
```

**磁盘空间估算（我们项目）：**

```
每个检查点:
  模型分片 (8 rank):  8 × 1.75GB ≈ 14GB
  优化器分片 (8 rank): 8 × 3.5GB  ≈ 28GB   ← Adam 有 m + v 两份
  Extra 状态:          ~100MB
  HF 模型:             ~14GB
  Data.pt:             ~1KB
  ────────────────────────────────
  合计:                ~56GB / checkpoint

max_ckpt_to_keep=5 → 5 × 56GB ≈ 280GB 磁盘空间
```

---

## 4. 检查点目录结构

完整的目录树如下：

```
{default_local_dir}/
│
├── latest_checkpointed_iteration.txt     # 内容: "1500"（纯整数）
│
├── global_step_500/
│   ├── model_world_size_8_rank_0.pt      # FSDP 分片模型 (rank 0)
│   ├── model_world_size_8_rank_1.pt      # FSDP 分片模型 (rank 1)
│   ├── ...                               # rank 2-6
│   ├── model_world_size_8_rank_7.pt      # FSDP 分片模型 (rank 7)
│   │
│   ├── optim_world_size_8_rank_0.pt      # FSDP 分片优化器 (rank 0)
│   ├── ...                               # rank 1-7
│   ├── optim_world_size_8_rank_7.pt
│   │
│   ├── extra_state_world_size_8_rank_0.pt # LR scheduler + RNG (rank 0)
│   ├── ...                                # rank 1-7
│   ├── extra_state_world_size_8_rank_7.pt
│   │
│   ├── fsdp_config.json                  # {"FSDP_version": 1, "world_size": 8}
│   ├── data.pt                           # DataLoader 状态
│   │
│   └── huggingface/                      # 完整 HF 格式模型
│       ├── config.json                   # 模型架构配置
│       ├── generation_config.json        # 生成参数配置
│       ├── model.safetensors             # 完整模型权重 (~14GB)
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       └── vocab.json
│
├── global_step_1000/
│   └── (同上结构)
│
└── global_step_1500/
    └── (同上结构)
```

**每个文件的用途总结：**

| 文件 | 谁写的 | 谁读的 | 用途 |
|------|--------|--------|------|
| `latest_checkpointed_iteration.txt` | Trainer (rank 0) | `find_latest_ckpt_path()` | 恢复训练时找到最新 checkpoint |
| `model_world_size_8_rank_X.pt` | CheckpointManager (每个 rank) | CheckpointManager | 恢复 FSDP 模型权重 |
| `optim_world_size_8_rank_X.pt` | CheckpointManager (每个 rank) | CheckpointManager | 恢复优化器状态（Adam 动量） |
| `extra_state_*.pt` | CheckpointManager (每个 rank) | CheckpointManager | 恢复 LR schedule + RNG |
| `fsdp_config.json` | CheckpointManager (rank 0) | 调试/检查 | 记录 FSDP 版本和 world_size |
| `data.pt` | Trainer (rank 0) | Trainer | 恢复 DataLoader 读取位置 |
| `huggingface/` | CheckpointManager (rank 0) | 评测脚本 (vLLM) | 评测推理 + GRPO 交接 |

---

## 5. 训练恢复（Resume）机制

### 5.1 三种恢复模式

```python
# fsdp_sft_trainer.py: _determine_resume_path()

resume_mode = getattr(self.config.trainer, "resume_mode", "auto")  # 默认 "auto"
```

| 模式 | 行为 | 使用场景 |
|------|------|----------|
| `"auto"` | 自动查找最新 checkpoint | **默认模式** — 训练中断后重新运行 |
| `"disable"` | 不恢复，从头训练 | 确定要从零开始（如更换数据集） |
| `"resume_path"` | 指定具体 checkpoint 路径 | 从特定步骤恢复 |

### 5.2 自动恢复流程

当使用 `resume_mode=auto`（默认），完整的恢复链条如下：

```
bash run_sft.sh 8
  │
  └─ torchrun → FSDPSFTTrainer.__init__() → load_checkpoint()
       │
       ├─ _determine_resume_path()
       │    └─ _find_latest_checkpoint()
       │         └─ find_latest_ckpt_path(default_local_dir)
       │              │
       │              ├─ 读取 latest_checkpointed_iteration.txt → "1000"
       │              └─ 返回 "checkpoints/.../global_step_1000"
       │
       ├─ extract_step("global_step_1000") → 1000
       │
       ├─ checkpoint_manager.load_checkpoint("global_step_1000")
       │    │
       │    ├─ 每个 rank 加载自己的 model_rank_X.pt
       │    │    └─ model.load_state_dict(model_state_dict)
       │    │
       │    ├─ 每个 rank 加载自己的 optim_rank_X.pt
       │    │    └─ optimizer.load_state_dict(optimizer_state_dict)
       │    │
       │    ├─ 每个 rank 加载自己的 extra_state_rank_X.pt
       │    │    ├─ lr_scheduler.load_state_dict(...)
       │    │    └─ load_rng_state(rng)  ← 恢复随机种子
       │    │
       │    └─ torch.distributed.barrier()  ← 等所有 rank 加载完
       │
       ├─ _load_dataloader_state("global_step_1000")
       │    └─ train_dataloader.load_state_dict(data.pt)
       │
       └─ self.resume_global_step = 1000
            │
            └─ fit() 中: global_step 从 1000 开始
```

### 5.3 加载源码逐行解析

```python
# fsdp_checkpoint_manager.py: load_checkpoint()

def load_checkpoint(self, local_path):
    # 1. 设置 SHARDED 状态字典上下文（与保存对应）
    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

    with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
        if self.should_load_model:
            # 2. 每个 rank 加载自己的分片
            remote_model_path = os.path.join(
                local_path,
                f"model_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            # copy_to_local: 如果是远程路径(HDFS)则先下载，本地路径直接返回
            local_model_path = copy_to_local(remote_model_path)
            model_state_dict = torch.load(local_model_path, weights_only=False)
            self.model.load_state_dict(model_state_dict)

        if self.should_load_optimizer:
            # 3. 加载优化器分片
            remote_optim_path = os.path.join(
                local_path,
                f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_optim_path = copy_to_local(remote_optim_path)
            optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
            self.optimizer.load_state_dict(optimizer_state_dict)

    if self.should_load_extra:
        # 4. 加载 LR scheduler 和 RNG（不在 FSDP 上下文中）
        remote_extra_path = os.path.join(
            local_path,
            f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
        )
        extra_state_dict = torch.load(copy_to_local(remote_extra_path), weights_only=False)

        # 恢复随机种子（保证恢复后数据的随机顺序一致）
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

        # 恢复 LR scheduler（保证学习率从中断处继续）
        if extra_state_dict["lr_scheduler"] is not None and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])

    # 5. 等待所有 rank 完成加载
    torch.distributed.barrier()
```

### 5.4 RNG 状态恢复

RNG（Random Number Generator）状态恢复保证了训练的**可复现性**：

```python
# checkpoint_manager.py: BaseCheckpointManager

@staticmethod
def get_rng_state():
    rng_state = {
        "cpu": torch.get_rng_state(),          # PyTorch CPU 随机种子
        "numpy": np.random.get_state(),         # NumPy 随机种子
        "random": random.getstate(),            # Python 内置 random
    }
    if get_device_name() != "cpu":
        rng_state["cuda"] = torch.cuda.get_rng_state()  # GPU 随机种子
    return rng_state

@staticmethod
def load_rng_state(rng_state):
    torch.set_rng_state(rng_state["cpu"])
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["random"])
    if get_device_name() != "cpu":
        torch.cuda.set_rng_state(rng_state["cuda"])
```

**为什么要恢复 RNG？**

| 不恢复 RNG | 恢复 RNG |
|------------|----------|
| Dropout 的随机 mask 不同 | Dropout mask 完全一致 |
| 数据 shuffle 顺序不同 | 数据顺序完全一致 |
| 结果不可复现 | 结果完全可复现 |

### 5.5 DataLoader 状态恢复

verl 使用 `StatefulDataLoader`（来自 `torchdata`），它可以保存和恢复自己的读取位置：

```python
# 保存时 (Trainer.save_checkpoint)
dataloader_state_dict = self.train_dataloader.state_dict()
torch.save(dataloader_state_dict, "data.pt")

# 加载时 (Trainer._load_dataloader_state)
dataloader_state_dict = torch.load("data.pt", map_location="cpu")
self.train_dataloader.load_state_dict(dataloader_state_dict)
```

这意味着恢复后，DataLoader 会**跳过已经训练过的数据**，从中断的 batch 继续读取。

### 5.6 恢复约束

| 约束 | 原因 | 后果 |
|------|------|------|
| GPU 数量必须相同 | FSDP 分片是 per-rank 的 | 8 GPU 保存的不能用 4 GPU 加载 |
| 同一文件系统 | checkpoint 路径必须可访问 | 换机器需复制 checkpoint |
| 配置兼容 | 模型结构/FSDP 策略必须一致 | 不能改模型再恢复 |

> **如果需要换 GPU 数量怎么办？** 使用 `huggingface/` 下的完整模型重新开始训练（会丢失优化器动量和 LR 状态）。

---

## 6. 训练监控：Tracking 系统

### 6.1 Tracking 类架构

verl 的 `Tracking` 类（`verl/utils/tracking.py`）支持多种日志后端：

```python
class Tracking:
    supported_backend = [
        "wandb",         # Weights & Biases — 最常用，云端可视化
        "mlflow",        # MLflow — 开源替代
        "swanlab",       # SwanLab
        "tensorboard",   # TensorBoard — 本地可视化
        "console",       # 控制台打印
        "clearml",       # ClearML
        "trackio",       # TrackIO
        "file",          # JSONL 文件日志
    ]
```

**初始化：**

```python
# fsdp_sft_trainer.py: fit() 中 (仅 rank 0 执行)

if rank == 0:
    tracking = Tracking(
        project_name=self.config.trainer.project_name,      # "rlvr_coding_model"
        experiment_name=self.config.trainer.experiment_name, # "phase1_sft_qwen7b_coder"
        default_backend=self.config.trainer.logger,          # ["console", "wandb"]
        config=OmegaConf.to_container(self.config, resolve=True),  # 完整配置
    )
```

**为什么只在 rank 0 初始化？**

- 所有 rank 计算的 loss 已经通过 `all_reduce` 平均过了
- 只需要一个进程上报指标，避免重复记录
- WandB 等服务也不应该被多个进程同时写入

### 6.2 WandB 初始化细节

当后端包含 `"wandb"` 时：

```python
# tracking.py: __init__() 中的 wandb 初始化

import wandb

entity = os.environ.get("WANDB_ENTITY", None)  # WandB 团队名（可选）
wandb.init(
    project=project_name,       # WandB 项目名
    name=experiment_name,       # 本次运行名称
    entity=entity,              # 团队名
    config=config,              # 完整配置作为超参数记录
    settings=settings,          # 代理设置（可选）
)
```

**配置会被自动记录：** WandB 会把整个 Hydra 配置保存为 run 的 config，方便后续对比不同实验的超参数。

### 6.3 指标记录流程

```python
# fsdp_sft_trainer.py: fit() 中的指标记录

for step_in_epoch, data in enumerate(self.train_dataloader):
    global_step += 1

    # 1. 训练一步，返回指标
    metric = self.training_step(data)
    # metric = {"train/loss": 1.23, "train/lr(1e-3)": 0.02, "train/time(s)": 3.5}

    # 2. 记录到所有后端
    if rank == 0:
        tracking.log(data=metric, step=global_step)

    # 3. 验证阶段
    if is_valid_step:
        val_loss = torch.mean(torch.stack(val_losses))
        if rank == 0:
            tracking.log(data={"val/loss": val_loss.item()}, step=global_step)
```

`tracking.log()` 会同时写入所有已配置的后端：

```python
# tracking.py: log()

def log(self, data, step, backend=None):
    for default_backend, logger_instance in self.logger.items():
        if backend is None or default_backend in backend:
            logger_instance.log(data=data, step=step)
    # 如果配置了 ["console", "wandb"]
    # → console 打印: "step=500, train/loss=1.23, ..."
    # → wandb 上传: 数据点 (step=500, train/loss=1.23)
```

---

## 7. 训练指标详解

### 7.1 训练指标（每步记录）

| 指标名 | 来源 | 说明 |
|--------|------|------|
| `train/loss` | `training_step()` 返回 | SFT 交叉熵 loss，跨所有 rank 平均 |
| `train/lr(1e-3)` | `training_step()` 返回 | 当前学习率 × 1000（方便在图表上看） |
| `train/time(s)` | `training_step()` 返回 | 本步训练耗时（秒） |
| `train/grad_norm` | **需添加** | 梯度范数（检测训练稳定性） |

> **注意：** `training_step()` 默认不返回 `grad_norm`。我们在实现文档中计划添加这个指标，只需在返回字典中加一项：
> ```python
> return {
>     "train/loss": step_loss.detach().item(),
>     "train/lr(1e-3)": lr * 1e3,
>     "train/time(s)": spend_time_per_step,
>     "train/grad_norm": grad_norm.item(),  # ← 添加这一行
> }
> ```

### 7.2 验证指标（每 test_freq 步记录）

| 指标名 | 来源 | 说明 |
|--------|------|------|
| `val/loss` | `fit()` 中计算 | BEE 验证集的平均 loss |

验证流程：

```python
# fit() 中的验证逻辑

if is_valid_step:
    val_losses = []
    for val_data in self.val_dataloader:     # 遍历整个验证集
        val_loss = self.validation_step(val_data)
        # validation_step 内部:
        #   self.fsdp_model.eval()   ← 关闭 dropout
        #   with torch.no_grad():   ← 不计算梯度
        #     loss = _compute_loss_and_backward(batch, do_backward=False)
        val_losses.append(val_loss)

    val_loss = torch.mean(torch.stack(val_losses))  # 对所有 batch 求平均
    tracking.log({"val/loss": val_loss.item()}, step=global_step)
```

### 7.3 fit() 中的保存和验证触发

```python
# fit() 循环核心逻辑

for epoch in range(total_epochs):
    for step_in_epoch, data in enumerate(train_dataloader):
        global_step += 1

        metric = self.training_step(data)            # 训练
        tracking.log(data=metric, step=global_step)  # 记录

        # 判断三个条件
        is_last_step  = global_step >= total_training_steps
        is_valid_step = global_step % test_freq == 0   # 每 500 步
        is_save_step  = global_step % save_freq == 0   # 每 500 步

        # 验证（最后一步也会验证）
        if is_last_step or (test_freq > 0 and is_valid_step):
            run_validation()

        # 保存（最后一步也会保存）
        if is_last_step or (save_freq > 0 and is_save_step):
            self.save_checkpoint(step=global_step)

        # 提前结束
        if is_last_step:
            print(f"Total train time: {train_time:.2f}s")
            print(f"Final val metrics: {last_valid_metric}")
            return
```

---

## 8. WandB Dashboard 设计

### 8.1 推荐面板布局

**Panel 1: Loss Curves（最重要）**

```
train/loss (每步)  ─────── 应稳定下降
val/loss (每 500 步) ────── 用于检测过拟合
```

- 正常模式：两条线都在下降，val/loss 略高于 train/loss
- 过拟合信号：train/loss 继续下降，val/loss 开始上升

**Panel 2: Learning Rate & Gradient Norm**

```
train/lr(1e-3) ───── 确认 warmup → cosine 曲线正确
train/grad_norm ──── 监控训练稳定性
```

**Panel 3: Training Speed**

```
train/time(s) ────── 每步训练时间
```

### 8.2 正常范围参考

| 指标 | 训练初期 | 训练中期 | 训练后期 | 异常信号 |
|------|---------|---------|---------|---------|
| `train/loss` | 1.5 - 3.0 | 0.8 - 1.5 | 0.5 - 1.0 | > 5.0 或 NaN |
| `val/loss` | 1.5 - 2.5 | 1.0 - 1.8 | 0.8 - 1.5 | 持续上升 |
| `train/grad_norm` | 1.0 - 10.0 | 0.5 - 5.0 | 0.1 - 2.0 | > 100 持续 |
| `train/lr(1e-3)` | 0→0.02 (warmup) | 0.02→0.01 | 0.01→~0 | 不符合 cosine 曲线 |
| `train/time(s)` | 2 - 5 | 2 - 5 | 2 - 5 | > 30 (I/O 瓶颈) |

### 8.3 过拟合检测示例

```
Step  500: train/loss=1.2, val/loss=1.5  ← 正常，两者都在下降
Step 1000: train/loss=0.8, val/loss=1.3  ← 正常，gap 稳定
Step 1500: train/loss=0.5, val/loss=1.4  ← ⚠️ val/loss 反弹
Step 2000: train/loss=0.3, val/loss=1.6  ← 🚨 过拟合确认
```

**应对策略：** 选择 val/loss 最低的那个 checkpoint（step 1000）。但注意：val/loss 只是辅助参考，最终应该以代码执行评测（exec_success_rate）为准。

---

## 9. 训练健康检查清单

### 9.1 训练开始后 100 步内

- [ ] `train/loss` 在下降（从 ~2.5 逐步降到 ~2.0）
- [ ] `train/lr(1e-3)` 在 warmup 阶段线性上升
- [ ] `train/grad_norm` 在合理范围 (0.1 - 10.0)
- [ ] `train/time(s)` 稳定，无持续增长
- [ ] GPU 利用率 > 80% (`nvidia-smi` 检查)
- [ ] 无 NCCL timeout 或 OOM 错误
- [ ] WandB Dashboard 能正常看到数据

### 9.2 每个检查点时

- [ ] `val/loss` 没有持续上升（相比上一次检查）
- [ ] `huggingface/` 目录已生成且包含完整文件
- [ ] 检查点大小合理（~56GB per checkpoint）
- [ ] `latest_checkpointed_iteration.txt` 已更新为最新 step

### 9.3 训练结束时

- [ ] 总 step 数符合预期（~3000 步）
- [ ] 最终 `train/loss` 在合理范围（0.5 - 1.5）
- [ ] 所有预期的检查点已保存
- [ ] WandB run 状态为 finished（非 crashed）
- [ ] 最终 checkpoint 的 `huggingface/` 可以被 vLLM 加载

---

## 10. 知识点总结

### 核心概念

| 概念 | 定义 | 在 verl 中的体现 |
|------|------|------------------|
| **Checkpoint** | 训练状态的磁盘快照 | `FSDPCheckpointManager` 管理 |
| **FSDP Sharded Save** | 每个 rank 只保存自己的参数分片 | `StateDictType.SHARDED_STATE_DICT` |
| **Checkpoint Rotation** | 自动删除旧检查点，保留最新 N 个 | `max_ckpt_to_keep` 参数 |
| **Tracker File** | 记录最新 checkpoint 步数的文件 | `latest_checkpointed_iteration.txt` |
| **Atomic Write** | 先写临时文件再重命名，防止写一半崩溃 | `os.rename(temp, final)` |
| **RNG State** | CPU/GPU/numpy/random 随机种子 | `get_rng_state()` / `load_rng_state()` |
| **StatefulDataLoader** | 可保存/恢复读取位置的 DataLoader | `data.pt` 保存/加载 |
| **HF Model Export** | FSDP 分片 → 完整 HuggingFace 模型 | `should_save_hf_model` + `save_pretrained` |
| **Tracking** | 多后端统一日志接口 | `Tracking` 类支持 wandb/console/mlflow 等 |

### 保存 vs 加载 对照

```
保存 (save_checkpoint)                加载 (load_checkpoint)
─────────────────────                ──────────────────────
model.state_dict()  → .pt           .pt → model.load_state_dict()
optimizer.state_dict() → .pt        .pt → optimizer.load_state_dict()
lr_scheduler.state_dict() → .pt     .pt → lr_scheduler.load_state_dict()
get_rng_state() → .pt               .pt → load_rng_state()
train_dataloader.state_dict() → .pt .pt → train_dataloader.load_state_dict()
write(step) → tracker.txt           read(tracker.txt) → step
```

---

## 11. 思考题

1. **为什么优化器状态通常比模型权重大 2 倍？** （提示：Adam 优化器保存了什么额外信息？）

2. **如果训练到 step 1200 时机器崩溃，`save_freq=500`，会丢失多少步的训练？从哪一步恢复？**

3. **为什么 `huggingface/` 目录对我们的项目至关重要，而纯 FSDP 分片不能直接用于评测？**

4. **如果想从 8 GPU 训练切换到 4 GPU 继续训练，应该怎么做？**

---

> **下一部分预告：** Part 7 将讲解评测管线与 GRPO 交接 —— 如何利用 `huggingface/` 下的模型进行三层评测，以及如何选出最优 checkpoint 交给 GRPO 阶段。
