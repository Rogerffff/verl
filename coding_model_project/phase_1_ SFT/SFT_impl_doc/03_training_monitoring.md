# Phase 1 SFT - Step 3: Training Monitoring

> WandB 指标体系、检查点目录结构、训练恢复、WandB Dashboard 设计

---

## 1. Overview

verl FSDPSFTTrainer 通过 `Tracking` 类 (`verl/utils/tracking.py`) 自动记录训练指标到 WandB 和 Console。本文档定义 Phase 1 SFT 阶段需要监控的全部指标、检查点目录结构和训练恢复策略。

---

## 2. 训练指标体系

### 2.1 verl 自动记录的指标（每步）

这些指标由 `FSDPSFTTrainer.training_step()` 返回，在 `fit()` 循环中通过 `tracking.log(data=metric, step=global_step)` 记录。

| 指标名 | 来源 | 类型 | 说明 |
|--------|------|------|------|
| `train/loss` | `fsdp_sft_trainer.py:528` | float | SFT 交叉熵 loss（跨 rank 平均） |
| `train/lr(1e-3)` | `fsdp_sft_trainer.py:529` | float | 当前学习率 × 1000 |
| `train/time(s)` | `fsdp_sft_trainer.py:530` | float | 本步训练耗时（秒） |
| `train/grad_norm` | **需修改** (见 02 文档) | float | 梯度范数（修改后新增） |

#### Loss 计算细节

```
_compute_loss_and_backward() (line 372-471):
  1. shift_logits = logits[..., :-1, :]
  2. shift_labels = input_ids[..., 1:]
  3. loss = CrossEntropyLoss(shift_logits, shift_labels, reduction='none')
  4. loss *= loss_mask[..., 1:]     ← 只在 assistant token 上计算
  5. loss = sum(loss) / valid_tokens  ← 按有效 token 数归一化
  6. loss /= n_micro_batches         ← 按梯度累积步数归一化
  7. loss.backward()

training_step() (line 473-531):
  - step_loss 累加所有 micro_batch 的 loss
  - all_reduce(step_loss, ReduceOp.AVG)  ← 跨 DP rank 平均
```

### 2.2 验证指标（每 test_freq 步）

| 指标名 | 来源 | 类型 | 说明 |
|--------|------|------|------|
| `val/loss` | `fsdp_sft_trainer.py:791` | float | BEE 2% 验证集的平均 loss |

#### val_loss 计算细节

```
fit() (line 780-794):
  if is_valid_step:
    val_losses = []
    for val_batch in val_dataloader:
        val_loss = validation_step(val_batch)
            → _compute_loss_and_backward(batch, do_backward=False)
        val_losses.append(val_loss)
    val_loss = mean(val_losses)
    tracking.log({"val/loss": val_loss}, step=global_step)
```

### 2.3 指标记录频率总结

| 指标类别 | 频率 | 条件 |
|---------|------|------|
| `train/loss`, `train/lr`, `train/time(s)`, `train/grad_norm` | 每步 | 始终 |
| `val/loss` | 每 500 步 | `trainer.test_freq=500` |
| Checkpoint 保存 | 每 500 步 | `trainer.save_freq=500` |

---

## 3. WandB 配置

### 3.1 Tracking 初始化

```python
# fit() (line 717-723):
if rank == 0:
    tracking = Tracking(
        project_name=self.config.trainer.project_name,    # "rlvr_coding_model"
        experiment_name=self.config.trainer.experiment_name, # "phase1_sft_qwen7b_coder"
        default_backend=self.config.trainer.logger,         # ["console", "wandb"]
        config=OmegaConf.to_container(self.config, resolve=True),  # 完整配置
    )
```

verl 的 `Tracking` 类 (`verl/utils/tracking.py`) 支持多种后端：
- `console`: 打印到 stdout
- `wandb`: WandB 在线追踪
- `tensorboard`: TensorBoard 日志
- `mlflow`: MLflow 追踪

### 3.2 WandB 项目结构

```
WandB Project: rlvr_coding_model
  ├─ Run: phase1_sft_qwen7b_coder     # SFT 训练 run
  ├─ Run: phase1_eval_step_500         # Eval pipeline runs (见 04 文档)
  ├─ Run: phase1_eval_step_1000
  └─ ...
```

所有训练和评测 run 共享同一个 WandB project，便于在 Dashboard 中对比。

---

## 4. 检查点目录结构

### 4.1 完整目录树

```
phase_1_ SFT/checkpoints/rlvr_coding_model/phase1_sft_qwen7b_coder/
│
├── latest_checkpointed_iteration.txt     # 最新检查点标记文件
│                                         # 内容示例: "1500"（纯整数 step）
│
├── global_step_500/
│   ├── model_world_size_8_rank_0.pt      # FSDP 分片模型权重 (rank 0)
│   ├── model_world_size_8_rank_1.pt      # FSDP 分片模型权重 (rank 1)
│   ├── ...                               # rank 2-7
│   ├── model_world_size_8_rank_7.pt
│   ├── optim_world_size_8_rank_0.pt      # FSDP 分片优化器状态 (rank 0)
│   ├── ...                               # rank 1-7
│   ├── optim_world_size_8_rank_7.pt
│   ├── extra_state_world_size_8_rank_0.pt # LR scheduler + RNG 状态 (rank 0)
│   ├── ...                               # rank 1-7
│   ├── extra_state_world_size_8_rank_7.pt
│   ├── fsdp_config.json                  # FSDP 版本和 world_size 信息
│   ├── data.pt                           # StatefulDataLoader 状态（恢复用）
│   └── huggingface/                      # ★ 完整 HF 模型（由 hf_model 触发）
│       ├── config.json                   # 模型配置
│       ├── generation_config.json        # 生成配置
│       ├── model.safetensors             # 完整模型权重 (~14GB)
│       ├── tokenizer.json               # tokenizer
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       └── vocab.json
│
├── global_step_1000/
│   └── (同上结构)
│
├── global_step_1500/
│   └── (同上结构)
│
└── global_step_XXXX/                     # 最终检查点（训练结束时）
    └── (同上结构)
```

### 4.2 关键文件说明

| 文件 | 说明 | 用途 |
|------|------|------|
| `latest_checkpointed_iteration.txt` | 记录最新检查点 step（整数） | `resume_mode=auto` 查找入口 |
| `model_world_size_8_rank_X.pt` | FSDP 分片权重 | 恢复训练（需要相同 GPU 数） |
| `optim_world_size_8_rank_X.pt` | FSDP 分片优化器状态 | 恢复训练（需要相同 GPU 数） |
| `extra_state_world_size_8_rank_X.pt` | LR scheduler + RNG | 恢复训练，保证可复现性 |
| `data.pt` | DataLoader 状态 | 恢复训练，从中断的 batch 继续 |
| `fsdp_config.json` | FSDP 运行时配置 | 恢复训练的兼容性检查 |
| `huggingface/` | 完整 HF 格式模型 | **评测用** — vLLM 直接加载 |

### 4.3 检查点轮转

配置 `trainer.max_ckpt_to_keep=5`，FSDPCheckpointManager 在保存新检查点时自动删除最旧的：

```python
# fsdp_checkpoint_manager.py:205-214
if max_ckpt_to_keep and len(self.previous_saved_paths) >= max_ckpt_to_keep:
    keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
    self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
```

实际磁盘占用: 5 × ~42GB = ~210GB

---

## 5. 训练恢复 (Resume)

### 5.1 自动恢复流程

配置 `trainer.resume_mode=auto`，verl 的恢复逻辑如下：

```python
# FSDPSFTTrainer.load_checkpoint() 调用链:
1. _find_latest_checkpoint()
   → find_latest_ckpt_path(checkpoint_dir)
   → 读取 latest_checkpointed_iteration.txt（例如 `1000`）
   → 构造并返回路径 (e.g., "global_step_1000")

2. FSDPCheckpointManager.load_checkpoint(latest_path)
   → 每个 rank 加载自己的 model_rank_X.pt, optim_rank_X.pt, extra_rank_X.pt
   → 恢复 lr_scheduler 状态
   → 恢复 RNG 状态（保证可复现性）

3. StatefulDataLoader 自动恢复
   → 从 data.pt 恢复 dataloader 状态
   → 跳到正确的 batch 位置

4. global_step 恢复
   → self.resume_global_step = extracted_step
   → fit() 中 global_step 从 resume_global_step 开始计数
```

### 5.2 恢复约束

| 约束 | 说明 |
|------|------|
| **GPU 数量必须相同** | FSDP 分片是 per-rank 的，不同 world_size 不兼容 |
| **同一台机器或相同文件路径** | 检查点路径必须可访问 |
| **配置兼容** | 模型结构、FSDP 策略等必须一致 |

### 5.3 恢复使用方式

```bash
# 训练中断后，直接重新运行相同脚本即可
bash run_sft.sh 8
# → 自动检测 latest_checkpointed_iteration.txt
# → 打印: "Found checkpoint: .../global_step_1000"
# → 从 step 1000 继续训练
```

---

## 6. WandB Dashboard 设计建议

### 6.1 训练监控面板

**Panel 1: Loss Curves**
- `train/loss` (每步): 训练 loss 趋势，应稳定下降
- `val/loss` (每 500 步): 验证 loss，用于检测过拟合
- 对比: 如果 val/loss 开始上升而 train/loss 继续下降 → 过拟合信号

**Panel 2: Learning Rate & Gradient Norm**
- `train/lr(1e-3)`: 确认 cosine schedule 工作正常（warmup → peak → decay）
- `train/grad_norm`: 监控训练稳定性
  - 正常范围: 0.1 - 10.0
  - 警告: > 100 持续出现 → 可能需要降低 lr
  - 警告: 趋近 0 → 可能梯度消失

**Panel 3: Training Speed**
- `train/time(s)`: 每步训练时间
  - 正常: ~2-5 秒/步（7B + 8xA100 + max_length=4096）
  - 周期性 spike: 正常（检查点保存时）

### 6.2 关键看板指标

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| `train/loss` (初始) | 1.5 - 3.0 | > 5.0 → 数据/模型问题 |
| `train/loss` (final) | 0.5 - 1.5 | < 0.1 → 过拟合 |
| `val/loss` (final) | 0.8 - 2.0 | 持续上升 → 停止训练 |
| `train/grad_norm` | 0.1 - 10.0 | > 100 持续 → 不稳定 |
| `train/time(s)` | 2 - 5 | > 30 → 可能有 I/O 瓶颈 |

### 6.3 过拟合检测

```
Step 500:  train/loss=1.2, val/loss=1.5
Step 1000: train/loss=0.8, val/loss=1.3  ← val/loss 也在下降，正常
Step 1500: train/loss=0.5, val/loss=1.4  ← val/loss 开始上升，注意
Step 2000: train/loss=0.3, val/loss=1.6  ← 过拟合，应选择 step 1000 的 checkpoint
```

注意: val/loss 只是参考指标。真正的模型能力评测通过 Eval Pipeline (04 文档) 的 `exec_success_rate` 决定。

---

## 7. 训练健康检查清单

### 7.1 训练开始后 100 步内检查

- [ ] `train/loss` 在下降
- [ ] `train/lr(1e-3)` 在 warmup 阶段上升
- [ ] `train/grad_norm` 在合理范围 (0.1 - 10.0)
- [ ] `train/time(s)` 稳定，无持续增长
- [ ] GPU 利用率 > 80% (通过 `nvidia-smi` 检查)
- [ ] 无 NCCL timeout 或 OOM 错误

### 7.2 每个检查点时检查

- [ ] `val/loss` 没有持续上升
- [ ] `huggingface/` 目录生成且包含完整文件
- [ ] 检查点大小合理 (~42GB per checkpoint)
- [ ] `latest_checkpointed_iteration.txt` 已更新

### 7.3 训练结束时检查

- [ ] 总 step 数符合预期
- [ ] 最终 `train/loss` 在合理范围
- [ ] 所有预期的检查点已保存
- [ ] WandB run 状态为 finished
