# Phase 1 SFT - Step 2: Training Configuration & verl Execution Flow

> verl FSDPSFTTrainer 完整执行流程、超参数配置、训练启动脚本

---

## 1. Overview

使用 verl 的 `FSDPSFTTrainer` 对 `Qwen2.5-Coder-7B-Instruct` 进行 SFT 训练。

**入口文件**: `verl/trainer/fsdp_sft_trainer.py` (873 行)
**默认配置**: `verl/trainer/config/sft_trainer.yaml`
**启动方式**: `torchrun -m verl.trainer.fsdp_sft_trainer [Hydra config overrides]`

---

## 2. verl SFT 完整执行流程

### 2.1 端到端调用链（含源码行号）

```
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    [Hydra config overrides]
    │
    ├─ main(config)                           # Hydra 入口 (line 843-848)
    │   ├─ auto_set_device(config)            # 自动检测 CUDA/NPU (line 846)
    │   └─ run_sft(config)                    # 主函数 (line 806-840)
    │
    └─ run_sft(config):
        ├─ [1] initialize_global_process_group()    # torch.distributed 初始化 (line 808)
        │       → 返回 (local_rank, rank, world_size)
        │
        ├─ [2] init_device_mesh()                   # 创建 FSDP device mesh (line 810-816)
        │       → device_mesh: 1D mesh shape=(world_size,)
        │       → ulysses_device_mesh: 2D mesh shape=(dp_size, sp_size)
        │
        ├─ [3] hf_tokenizer(local_model_path)       # 加载 tokenizer (line 820-821)
        │       → 从 model.partial_pretrain 路径加载
        │       → trust_remote_code 由配置控制
        │
        ├─ [4] create_sft_dataset()                  # 创建数据集 (line 822-827, 851-868)
        │   ├─ 检查 data.multiturn.enable → True
        │   ├─ dataset_cls = MultiTurnSFTDataset
        │   └─ MultiTurnSFTDataset(parquet_files, tokenizer, config)
        │       ├─ pd.read_parquet()                 # 读取 parquet 文件
        │       ├─ self.messages = dataframe[messages_key]  # 提取 messages 列
        │       ├─ extract_system_prompt_and_generation()   # 提取系统 prompt 模式
        │       └─ __getitem__(item):                # 按需 tokenize
        │           ├─ _build_messages(row_dict)      # 获取 messages 列表
        │           ├─ For each message:
        │           │   └─ _process_single_message()
        │           │       ├─ tokenizer.apply_chat_template()
        │           │       ├─ 去除非首条 message 的 system prompt
        │           │       └─ 设置 loss_mask:
        │           │           ├─ assistant → 1 (除 generation prompt)
        │           │           └─ system/user → 0
        │           ├─ torch.cat(input_ids), torch.cat(loss_mask), ...
        │           ├─ sanity_check()                # 验证 tokenize 一致性
        │           ├─ position_ids = torch.arange(seq_len)
        │           └─ Padding/Truncation 到 max_length
        │               → 返回 {input_ids, attention_mask, position_ids, loss_mask}
        │
        ├─ [5] FSDPSFTTrainer.__init__()             # 初始化 trainer (line 96-142)
        │   ├─ _normalize_config_bsz()               # [5a] 按 DP 并行度归一化 batch size
        │   │   └─ train_batch_size //= dp_size      # 全局 → 每卡 (line 144-156)
        │   │   └─ 验证: train_batch_size % micro_batch_size_per_gpu == 0
        │   │
        │   ├─ _build_dataloader()                   # [5b] 构建 DataLoader (line 157-205)
        │   │   ├─ DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        │   │   └─ StatefulDataLoader(                # 支持状态保存/恢复
        │   │       dataset=train_dataset,
        │   │       batch_size=per_dp_batch_size,     # 已归一化的每卡 batch
        │   │       num_workers=8, pin_memory=True)
        │   │
        │   ├─ _build_model_optimizer()              # [5c] 构建模型和优化器 (line 207-370)
        │   │   ├─ copy_to_local(model.partial_pretrain)   # 下载/复制模型到本地
        │   │   ├─ AutoConfig.from_pretrained()             # 加载模型配置
        │   │   ├─ AutoModelForCausalLM.from_pretrained()   # 加载预训练模型 (line 240-246)
        │   │   │   → torch_dtype=bf16, attn_implementation="flash_attention_2"
        │   │   ├─ model.gradient_checkpointing_enable()    # 启用梯度检查点 (line 287-288)
        │   │   ├─ [FSDP2 路径] (line 325-342):
        │   │   │   ├─ MixedPrecisionPolicy(param=bf16, reduce=fp32)
        │   │   │   ├─ apply_fsdp2(model, fsdp_kwargs, config)  # 应用 FSDP2 分片
        │   │   │   └─ fsdp2_load_full_state_dict()              # 加载初始权重
        │   │   ├─ build_optimizer(params, config.optim)    # AdamW 优化器 (line 346)
        │   │   ├─ 计算 steps_per_epoch, total_steps (line 350-357)
        │   │   ├─ num_warmup_steps = total_steps * warmup_ratio (line 359)
        │   │   └─ get_cosine_schedule_with_warmup()        # LR 调度器 (line 361-364)
        │   │
        │   ├─ _init_checkpoint_manager()            # [5d] 初始化检查点管理器
        │   │   → FSDPCheckpointManager(model, optimizer, lr_scheduler, ...)
        │   │
        │   └─ load_checkpoint()                     # [5e] 尝试恢复训练
        │       → resume_mode="auto": 查找 latest_checkpointed_iteration.txt（内容为整数 step）
        │
        └─ [6] trainer.fit()                         # 主训练循环 (line 713-803)
            ├─ Tracking(project, experiment, logger)  # WandB/Console 追踪 (line 718-723)
            ├─ total_training_steps 计算 (line 729-734)
            │
            └─ For each epoch in range(start_epoch, total_epochs):
                ├─ train_sampler.set_epoch(epoch)     # 确保每 epoch 不同 shuffle
                │
                └─ For each (step_in_epoch, data) in train_dataloader:
                    ├─ global_step += 1
                    ├─ data = TensorDict(data).to(device)
                    │
                    ├─ metric = training_step(data)   # [核心] 训练一步 (line 473-531)
                    │   ├─ fsdp_model.train()
                    │   ├─ optimizer.zero_grad()
                    │   ├─ micro_batches = batch.split(micro_batch_size_per_gpu) # 梯度累积
                    │   ├─ For each micro_batch:
                    │   │   └─ loss = _compute_loss_and_backward(micro_batch)
                    │   │       ├─ model(input_ids, attention_mask, position_ids)
                    │   │       ├─ CrossEntropyLoss(shift_logits, shift_labels)
                    │   │       ├─ loss *= loss_mask  (只在 assistant token 上计算)
                    │   │       ├─ loss /= n_micro_batches  (归一化)
                    │   │       └─ loss.backward()
                    │   ├─ grad_norm = fsdp2_clip_grad_norm_(params, max_norm)  # 梯度裁剪
                    │   ├─ if grad_norm is finite: optimizer.step()
                    │   ├─ lr_scheduler.step()
                    │   ├─ all_reduce(step_loss, ReduceOp.AVG)  # 跨 rank 平均 loss
                    │   └─ return {"train/loss", "train/lr(1e-3)", "train/time(s)"}
                    │
                    ├─ tracking.log(metric, global_step)     # 记录训练指标
                    │
                    ├─ [If global_step % test_freq == 0]:    # 验证步 (line 780-794)
                    │   ├─ For each val_batch in val_dataloader:
                    │   │   └─ val_loss = validation_step(val_batch)
                    │   │       └─ _compute_loss_and_backward(batch, do_backward=False)
                    │   ├─ val_loss = mean(val_losses)
                    │   └─ tracking.log({"val/loss": val_loss}, global_step)
                    │
                    ├─ [If global_step % save_freq == 0]:    # 保存检查点 (line 796-797)
                    │   └─ save_checkpoint(step=global_step)
                    │       └─ FSDPCheckpointManager.save_checkpoint()
                    │           ├─ 每个 rank 保存自己的 model/optim/extra shard
                    │           ├─ rank 0 保存 HF config + tokenizer 到 huggingface/
                    │           └─ [If "hf_model" in save_contents]:
                    │               ├─ get_fsdp_full_state_dict() → 聚合完整模型
                    │               └─ save_model.save_pretrained(huggingface/)
                    │
                    └─ [If is_last_step]: 打印总训练时间, return
```

### 2.2 关键代码路径总结

| 阶段 | 文件 | 行号 | 函数 |
|------|------|------|------|
| 入口 | `fsdp_sft_trainer.py` | 843-848 | `main()` → `run_sft()` |
| 分布式初始化 | `fsdp_sft_trainer.py` | 808-816 | `initialize_global_process_group()`, `init_device_mesh()` |
| 数据集创建 | `fsdp_sft_trainer.py` | 851-868 | `create_sft_dataset()` → `MultiTurnSFTDataset` |
| Trainer 初始化 | `fsdp_sft_trainer.py` | 96-142 | `FSDPSFTTrainer.__init__()` |
| Batch size 归一化 | `fsdp_sft_trainer.py` | 144-156 | `_normalize_config_bsz()` |
| DataLoader 构建 | `fsdp_sft_trainer.py` | 157-205 | `_build_dataloader()` |
| 模型/优化器构建 | `fsdp_sft_trainer.py` | 207-370 | `_build_model_optimizer()` |
| 训练循环 | `fsdp_sft_trainer.py` | 713-803 | `fit()` |
| 单步训练 | `fsdp_sft_trainer.py` | 473-531 | `training_step()` |
| Loss 计算 | `fsdp_sft_trainer.py` | 372-471 | `_compute_loss_and_backward()` |
| 验证步 | `fsdp_sft_trainer.py` | 533-543 | `validation_step()` |
| 检查点保存 | `fsdp_checkpoint_manager.py` | 180-370 | `FSDPCheckpointManager.save_checkpoint()` |

---

## 3. 超参数配置

### 3.1 Hydra 配置系统

verl SFT 使用 Hydra 配置管理。默认配置在 `verl/trainer/config/sft_trainer.yaml`，通过命令行 override 覆盖。

配置层次：
```
sft_trainer.yaml (默认值)
    └── 命令行 overrides (优先级更高)
```

### 3.2 完整超参数表

#### 数据配置 (`data.*`)

| 参数 | 值 | 默认值 | 说明 |
|------|-----|--------|------|
| `data.train_files` | `${DATA_DIR}/sft_train.parquet` | `~/data/gsm8k/train.parquet` | 训练数据路径 |
| `data.val_files` | `${DATA_DIR}/sft_val.parquet` | `~/data/gsm8k/test.parquet` | 验证数据路径 |
| `data.multiturn.enable` | `true` | `false` | 启用 MultiTurnSFTDataset |
| `data.multiturn.messages_key` | `messages` | `messages` | Parquet 中 messages 列名 |
| `data.max_length` | `4096` | `1024` | 最大序列长度（token） |
| `data.truncation` | `right` | `error` | 截断策略（right/left/error） |
| `data.train_batch_size` | `128` | `256` | **全局** batch size（所有 GPU 合计） |
| `data.micro_batch_size_per_gpu` | `2` | `4` | 每 GPU micro-batch 大小 |
| `data.balance_dp_token` | `True` | `False` | 平衡 DP 各 rank 的 token 数量 |

**max_length=4096 的理由**: 根据 BEE 数据分析，99.6% 的样本在 4096 token 内。允许长解题代码不被截断。

**Batch size 计算链**:
```
全局 batch = 128
  ÷ 8 GPUs (DP size) = 16 per GPU
  ÷ 2 micro_batch = 8 梯度累积步数
```

#### 模型配置 (`model.*`)

| 参数 | 值 | 默认值 | 说明 |
|------|-----|--------|------|
| `model.partial_pretrain` | `Qwen/Qwen2.5-Coder-7B-Instruct` | `~/models/gemma-1.1-7b-it` | 预训练模型 |
| `model.trust_remote_code` | `true` | `false` | 信任远程代码（Qwen 需要） |
| `model.enable_gradient_checkpointing` | `true` | `true` | 梯度检查点，节省显存 |
| `model.strategy` | `fsdp2` | `fsdp2` | FSDP 版本（使用 v2） |
| `model.fsdp_config.model_dtype` | `bf16` | `fp32` | 模型精度 |
| `model.lora_rank` | `0` | `0` | LoRA 秩（0=全参数微调） |

**gradient_checkpointing**: 对 7B 模型 + max_length=4096 至关重要，否则 OOM。代价是约 30% 训练速度下降。

#### 优化器配置 (`optim.*`)

| 参数 | 值 | 默认值 | 说明 |
|------|-----|--------|------|
| `optim.lr` | `2e-5` | `1e-5` | 学习率 |
| `optim.betas` | `[0.9, 0.95]` | `[0.9, 0.95]` | AdamW beta 参数 |
| `optim.weight_decay` | `0.01` | `0.01` | 权重衰减 |
| `optim.lr_warmup_steps_ratio` | `0.05` | `0.1` | Warmup 比例 |
| `optim.clip_grad` | `1.0` | `1.0` | 梯度裁剪阈值 |
| `optim.lr_scheduler` | `cosine` | `cosine` | LR 调度策略 |

**lr=2e-5**: 对已经 instruction-tuned 的 7B 模型，使用保守学习率防止灾难性遗忘。

#### Trainer 配置 (`trainer.*`)

| 参数 | 值 | 默认值 | 说明 |
|------|-----|--------|------|
| `trainer.total_epochs` | `2` | `4` | 训练轮数 |
| `trainer.save_freq` | `500` | `-1` | 保存检查点频率（步） |
| `trainer.test_freq` | `500` | `-1` | 验证频率（步） |
| `trainer.max_ckpt_to_keep` | `5` | `null` | 保留最近 N 个检查点 |
| `trainer.logger` | `["console", "wandb"]` | `["console", "wandb"]` | 日志后端 |
| `trainer.seed` | `42` | `1` | 随机种子 |
| `trainer.resume_mode` | `auto` | `auto` | 自动恢复训练 |
| `trainer.project_name` | `rlvr_coding_model` | `gsm8k-sft` | WandB 项目名 |
| `trainer.experiment_name` | `phase1_sft_qwen7b_coder` | `test` | WandB 实验名 |

#### 检查点配置 (`trainer.checkpoint.*`)

| 参数 | 值 | 默认值 | 说明 |
|------|-----|--------|------|
| `trainer.checkpoint.save_contents` | `["model","optimizer","extra","hf_model"]` | `["model","optimizer","extra"]` | 保存内容 |

**添加 `"hf_model"`**: 确保每个检查点自动导出完整 HuggingFace 模型到 `huggingface/` 子目录，可直接被 vLLM 加载用于评测。

### 3.3 训练步数计算

```
输入样本数: N_final（最终过滤后由用户提供）
val_ratio: 0.02
→ 训练集约 floor(0.98 * N_final)，验证集约 ceil(0.02 * N_final)

全局 batch size: 128
Steps per epoch: ceil(train_samples / 128)
Total epochs: 2
Total steps: steps_per_epoch × 2

checkpoints: 每 500 步保存一次，直到最终 step
Warmup steps: total_steps × 0.05
```

---

## 4. verl 框架修改: 添加 `grad_norm` 日志

### 4.1 问题

`training_step()` 方法（line 473-531）在 line 492-494 计算了 `grad_norm`：

```python
if self.config.model.strategy == "fsdp":
    grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
elif self.config.model.strategy == "fsdp2":
    grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
```

但返回值（line 527-531）**没有**包含 `grad_norm`：

```python
return {
    "train/loss": step_loss.detach().item(),
    "train/lr(1e-3)": lr * 1e3,
    "train/time(s)": spend_time_per_step,
}
```

### 4.2 修改方案

**文件**: `verl/trainer/fsdp_sft_trainer.py`
**位置**: line 527-531
**改动**: 添加一行

```python
# 修改后:
return {
    "train/loss": step_loss.detach().item(),
    "train/lr(1e-3)": lr * 1e3,
    "train/time(s)": spend_time_per_step,
    "train/grad_norm": grad_norm.detach().item(),  # ← 新增
}
```

### 4.3 为什么需要 grad_norm

1. **训练健康监控**: grad_norm 突增/突降是训练不稳定的早期信号
2. **超参数调优**: 如果 grad_norm 持续很大，可能需要降低 lr 或增加 clip_grad
3. **loss spike 诊断**: 当 loss 突然升高时，grad_norm 能帮助定位原因（数据异常 vs 学习率过大）
4. **项目展示**: 完整的训练曲线（loss + lr + grad_norm）是 resume 项目的标准展示内容

---

## 5. 检查点策略

### 5.1 save_contents 配置

```yaml
trainer.checkpoint.save_contents: ["model", "optimizer", "extra", "hf_model"]
```

| 内容 | 文件格式 | 说明 |
|------|---------|------|
| `model` | `model_world_size_8_rank_X.pt` | FSDP 分片模型权重 |
| `optimizer` | `optim_world_size_8_rank_X.pt` | FSDP 分片优化器状态 |
| `extra` | `extra_state_world_size_8_rank_X.pt` | LR scheduler 状态 + RNG 状态 |
| `hf_model` | `huggingface/model.safetensors` | **完整** HuggingFace 格式模型 |

### 5.2 hf_model 保存流程（fsdp_checkpoint_manager.py:309-368）

```python
# 1. 聚合分片模型到完整状态字典（仅 rank 0 保留）
state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

# 2. rank 0 创建空模型并加载状态字典
if self.rank == 0:
    with init_empty_weights():
        save_model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.bfloat16)
    save_model.save_pretrained(hf_local_path, state_dict=state_dict)
```

### 5.3 磁盘开销估算

| 组件 | 每个检查点大小 | 说明 |
|------|-------------|------|
| FSDP shards (model + optim + extra) | ~28 GB | 8 个 rank 文件 |
| hf_model (huggingface/) | ~14 GB | 完整 bf16 模型 |
| **合计** | ~42 GB | 每个检查点 |

- `max_ckpt_to_keep=5` → 最多保留 5 个 → ~210 GB 磁盘空间
- hf_model 保存耗时约 30-60 秒（需要 all_gather 然后 rank 0 写盘）

---

## 6. 训练启动脚本: `run_sft.sh`

```bash
#!/bin/bash
set -x

# ============================================================
# Phase 1 SFT 训练脚本
# 使用 verl FSDPSFTTrainer 进行 Qwen2.5-Coder-7B-Instruct SFT
# ============================================================

NPROC_PER_NODE=${1:-8}
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_DIR="${PROJECT_DIR}/data"
SAVE_DIR="${PROJECT_DIR}/checkpoints"

echo "========================================"
echo "Phase 1 SFT Training"
echo "GPUs: ${NPROC_PER_NODE}"
echo "Data: ${DATA_DIR}"
echo "Save: ${SAVE_DIR}"
echo "========================================"

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${DATA_DIR}/sft_train.parquet \
    data.val_files=${DATA_DIR}/sft_val.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=4096 \
    data.truncation=right \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=2 \
    data.balance_dp_token=True \
    model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.strategy=fsdp2 \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=2e-5 \
    optim.betas='[0.9,0.95]' \
    optim.weight_decay=0.01 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.project_name=rlvr_coding_model \
    trainer.experiment_name=phase1_sft_qwen7b_coder \
    trainer.total_epochs=2 \
    trainer.save_freq=500 \
    trainer.test_freq=500 \
    trainer.max_ckpt_to_keep=5 \
    trainer.logger='["console","wandb"]' \
    trainer.seed=42 \
    trainer.resume_mode=auto \
    trainer.checkpoint.save_contents='["model","optimizer","extra","hf_model"]'
```

### 6.1 启动命令

```bash
cd "coding_model_project/phase_1_ SFT"
bash run_sft.sh 8       # 8 GPU
bash run_sft.sh 4       # 4 GPU（需调整 batch size 使其整除）
```

### 6.2 Smoke Test（验证 pipeline）

```bash
# 只跑 10 步，验证完整流程无报错
PHASE1_DIR="$(pwd)"
DATA_DIR="${PHASE1_DIR}/data"
SAVE_DIR="${PHASE1_DIR}/checkpoints"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${DATA_DIR}/sft_train.parquet \
    data.val_files=${DATA_DIR}/sft_val.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=4096 \
    data.truncation=right \
    data.train_batch_size=4 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.strategy=fsdp2 \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=2e-5 \
    trainer.default_local_dir=${SAVE_DIR}/smoke_test \
    trainer.project_name=rlvr_coding_model \
    trainer.experiment_name=smoke_test \
    trainer.total_training_steps=10 \
    trainer.total_epochs=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.logger='["console"]' \
    trainer.seed=42
```

---

## 7. 潜在问题与应对

| 问题 | 症状 | 应对 |
|------|------|------|
| OOM | CUDA out of memory | 1. 已启用 gradient_checkpointing; 2. 减小 micro_batch=1; 3. 减小 max_length=2048 |
| Sanity check 失败 | `MultiTurnSFTDataset` 报 AssertionError | 设置 `data.multiturn.ignore_input_ids_mismatch=true` |
| 模型下载超时 | HuggingFace 连接失败 | 提前下载到本地，使用本地路径 |
| batch size 不整除 | assert 失败 | 确保 train_batch_size % (n_gpus × micro_batch_size_per_gpu) == 0 |
| hf_model 保存超时 | 训练卡在 save_checkpoint | 正常现象，7B 模型 gather+save 需要 30-60 秒 |
| resume 后 step 不连续 | global_step 计数异常 | `StatefulDataLoader` + `resume_mode=auto` 自动处理 |
