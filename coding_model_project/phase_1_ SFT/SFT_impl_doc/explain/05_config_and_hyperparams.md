# Part 5: 配置项与超参数详解

> 本文是 verl SFT 讲解系列的第 5 部分，逐个讲解 `sft_trainer.yaml` 中的每个配置项，以及我们项目中具体参数值背后的选择理由。
> 对应实现文档：`02_training_config_and_flow.md` 第 3 节

---

## 1. verl 的配置系统：Hydra

verl 用 **Hydra** 管理配置。理解 Hydra 的关键在于：

```
                ┌──────────────────────────┐
                │ sft_trainer.yaml (默认值)  │
                │ lr: 1e-5                  │
                │ total_epochs: 4           │
                │ max_length: 1024          │
                └───────────┬──────────────┘
                            │
                 命令行 override (优先级更高)
                            │
                            ↓
                ┌──────────────────────────┐
                │ 最终合并的 config          │
                │ lr: 2e-5        ← 被覆盖   │
                │ total_epochs: 2 ← 被覆盖   │
                │ max_length: 4096← 被覆盖   │
                └──────────────────────────┘
```

**规则**：YAML 文件中写默认值，命令行上写你想改的值，两者合并得到最终配置。没有在命令行覆盖的参数就用 YAML 的默认值。

---

## 2. 配置项全景速查表

先看全局，后面逐个详解。下表中 **"我们的值"** 是本项目的设定，**"默认值"** 是 `sft_trainer.yaml` 中的默认：

| 分类 | 配置项 | 我们的值 | 默认值 | 重要度 |
|------|--------|---------|--------|-------|
| **数据** | `data.max_length` | **4096** | 1024 | ★★★ |
| | `data.train_batch_size` | **128** | 256 | ★★★ |
| | `data.micro_batch_size_per_gpu` | **2** | 4 | ★★★ |
| | `data.multiturn.enable` | **true** | false | ★★★ |
| | `data.truncation` | **right** | error | ★★ |
| | `data.balance_dp_token` | **True** | False | ★ |
| **模型** | `model.partial_pretrain` | **Qwen2.5-Coder-7B-Instruct** | gemma-7b | ★★★ |
| | `model.strategy` | **fsdp2** | fsdp2 | ★★ |
| | `model.enable_gradient_checkpointing` | **true** | true | ★★★ |
| | `model.fsdp_config.model_dtype` | **bf16** | fp32 | ★★ |
| **优化器** | `optim.lr` | **2e-5** | 1e-5 | ★★★ |
| | `optim.weight_decay` | **0.01** | 0.01 | ★ |
| | `optim.clip_grad` | **1.0** | 1.0 | ★★ |
| | `optim.lr_warmup_steps_ratio` | **0.05** | 0.1 | ★ |
| | `optim.lr_scheduler` | **cosine** | cosine | ★★ |
| **训练** | `trainer.total_epochs` | **2** | 4 | ★★★ |
| | `trainer.save_freq` | **500** | -1 | ★★★ |
| | `trainer.test_freq` | **500** | -1 | ★★★ |
| | `trainer.max_ckpt_to_keep` | **5** | null | ★★ |
| | `checkpoint.save_contents` | **+hf_model** | 无 hf_model | ★★★ |

---

## 3. 数据配置详解 (`data.*`)

### 3.1 `data.max_length = 4096`

**含义**：所有样本都会被 padding/truncation 到这个长度。

**为什么是 4096？**

```
BEE 数据 token 分布 (使用 Qwen tokenizer):
  p50 (中位数):  ~800 tokens
  p90:           ~2000 tokens
  p99:           ~3500 tokens
  p99.6:         ~4096 tokens    ← 覆盖 99.6% 数据
  max:           ~8000+ tokens

选 4096 的理由:
  ✅ 覆盖 99.6% 数据，极少样本被截断
  ✅ 是 2 的幂次，GPU 计算效率最高
  ✅ 8 张 A100/4090 能放得下 (配合梯度检查点)
  ❌ 如果选 8192: 显存可能不够，或 micro_batch 只能设 1
  ❌ 如果选 2048: 约 5% 样本被截断，丢失完整解法
```

### 3.2 `data.train_batch_size = 128`

**含义**：全局 batch size，所有 GPU 合计一步处理 128 个样本。

**这是全局的，不是每个 GPU 的！** verl 内部会自动除以 DP 数：

```
你设置: train_batch_size = 128 (全局)
verl 内部: 128 / 4 (dp_size) = 32 (每个 DP 组)
实际效果: 每步 128 个样本的梯度参与一次参数更新
```

**为什么是 128？**
- 太小（如 32）：梯度噪声大，训练不稳定
- 太大（如 512）：训练步数减少，可能欠拟合
- 128 是 7B 模型 SFT 的常用值，平衡稳定性和训练效率

### 3.3 `data.micro_batch_size_per_gpu = 2`

**含义**：每次 forward/backward 实际喂给 GPU 的样本数。

**为什么只能是 2？**

```
显存预算估算 (单张 A100 80GB):
  模型参数 (bf16):     ~14 GB (7B × 2 bytes)
  FSDP 分片后 (8卡):   ~1.75 GB
  优化器状态 (分片):    ~3.5 GB
  梯度 (分片):          ~1.75 GB
  激活值 (每个样本):    ~2-4 GB (4096 序列长度, 有梯度检查点)
  ────────────────────────
  micro_batch=2 时:    ~12-15 GB 激活值
  总计:                ~20-22 GB   ← 80GB 够用

  micro_batch=4 时:    ~24-30 GB 激活值
  总计:                ~32-38 GB   ← 可能紧张

  4090 (24GB) 的话:    micro_batch=1 或 2，视序列长度
```

### 3.4 `data.multiturn.enable = true`

激活 `MultiTurnSFTDataset`（Part 2 已详细讲解）。配合 `messages_key=messages` 使用。

### 3.5 `data.truncation = right`

超过 `max_length` 的样本从右侧截断（截掉代码尾部）：

```
"right":  保留题目 + 代码开头，截掉代码尾部
          → 模型至少能看到完整题目，代码可能不完整
          → 对 99.6% 不超长的样本无影响

"left":   截掉题目开头，保留代码尾部
          → 模型看不到完整题目，基本无法学习

"error":  直接报错
          → 安全但需要数据预过滤
```

### 3.6 `data.balance_dp_token = True`

不同 DP 组的样本长度可能不同，导致有效 token 数不均衡。开启后，loss 归一化时会用**全局有效 token 数**而非单个 GPU 的：

```
关闭 (False):
  GPU 0: 300 个有效token, loss = sum / 300
  GPU 1: 200 个有效token, loss = sum / 200   ← 短样本的 loss 被放大了
  → 短样本对梯度的贡献被高估

开启 (True):
  全局有效 token = 300 + 200 = 500
  GPU 0: loss = sum / 500 * 2(dp_size)
  GPU 1: loss = sum / 500 * 2
  → 每个 token 的权重一致，更公平
```

---

## 4. 模型配置详解 (`model.*`)

### 4.1 `model.partial_pretrain = Qwen/Qwen2.5-Coder-7B-Instruct`

从 HuggingFace 加载的预训练模型。**Instruct** 版本意味着已经做过基础的指令微调，我们在此基础上用代码数据继续微调。

### 4.2 `model.strategy = fsdp2`

使用 PyTorch ≥ 2.4 的 FSDP2 API。相比 FSDP1 更灵活，支持更细粒度的分片控制。

### 4.3 `model.enable_gradient_checkpointing = true`

**必须开启**。对于 7B 模型 + 4096 序列长度，不开梯度检查点 **一定会 OOM**。

原理回顾：
```
不开:  forward 时保存所有层的激活值 → 反向传播直接用 → 快但费显存
       显存占用: O(层数 × 序列长度 × 隐藏维度)

开了:  forward 时只保存部分层的激活值 → 反向传播时重新计算丢弃的 → 慢但省显存
       显存占用: O(√层数 × 序列长度 × 隐藏维度)
       速度下降: ~30%
```

### 4.4 `model.fsdp_config.model_dtype = bf16`

参数存储用 `bfloat16`（和 Mixed Precision 配合），内存减半：

```
fp32: 每参数 4 bytes → 7B × 4 = 28 GB
bf16: 每参数 2 bytes → 7B × 2 = 14 GB  ← 省一半
```

---

## 5. 优化器配置详解 (`optim.*`)

### 5.1 `optim.lr = 2e-5`

学习率是 SFT 最敏感的超参数。

```
学习率选择思路:

  预训练 (from scratch):  lr = 1e-4 ~ 3e-4    (大幅调整)
  SFT (已有基础):         lr = 1e-5 ~ 5e-5    (轻微调整)
  LoRA SFT:              lr = 1e-4 ~ 5e-4    (只改少量参数，可以大一些)

  我们选 2e-5:
    ✅ 足够大，2 个 epoch 内能学到代码格式
    ✅ 足够小，不会破坏 Qwen 的通用能力 (防 catastrophic forgetting)
    ★ 如果训练 loss 不下降 → 试 5e-5
    ★ 如果 HumanEval 退化严重 → 试 1e-5
```

### 5.2 `optim.betas = [0.9, 0.95]`

AdamW 优化器的一阶/二阶动量衰减系数：

```
beta1 = 0.9:   一阶动量（梯度的指数移动平均）
                → 平滑梯度方向，减少震荡
                → 0.9 表示"记住 90% 的历史方向"

beta2 = 0.95:  二阶动量（梯度平方的指数移动平均）
                → 自适应学习率，对不同参数用不同步长
                → 0.95 比默认的 0.999 衰减更快
                → LLM 训练常用 0.95（梯度变化大，需要更快适应）
```

### 5.3 `optim.weight_decay = 0.01`

权重衰减（L2 正则化），防止过拟合：

```
每步更新: θ = θ - lr × (梯度 + weight_decay × θ)
                                    ↑
                         让参数不要变得太大

0.01 是 LLM 微调的标准值:
  太大 (0.1):  过度正则化，可能学不够
  太小 (0):    没有正则化，过拟合风险
```

### 5.4 `optim.lr_warmup_steps_ratio = 0.05`

总训练步数的前 5% 做 warmup：

```
假设 total_steps = 2000
warmup_steps = 2000 × 0.05 = 100 步

步 1:    lr = 0          (从 0 开始)
步 50:   lr = 1e-5       (线性升到一半)
步 100:  lr = 2e-5       (达到 peak)
步 101+: lr 开始 cosine 衰减
```

### 5.5 `optim.clip_grad = 1.0`

梯度范数裁剪阈值（Part 4 已详细讲解）。1.0 是标准值。

### 5.6 `optim.lr_scheduler = cosine`

Cosine 衰减学习率调度，verl 还支持 `wsd`（Warmup-Stable-Decay）：

```
cosine (我们选的):
  warmup → peak → 余弦平滑衰减到 ~0
  优点: 平滑，end-of-training 自然收敛

wsd:
  warmup → 保持 peak → 快速衰减
  优点: 稳定阶段更长，适合长训练
```

---

## 6. Trainer 配置详解 (`trainer.*`)

### 6.1 `trainer.total_epochs = 2`

**为什么只 2 个 epoch？**

```
SFT 过拟合风险:
  epoch 1: 模型学会格式和基本代码模式 → exec_success_rate ↑
  epoch 2: 进一步强化 → 通常达到最佳点
  epoch 3+: 开始"背诵"训练数据 → val_loss ↑, HumanEval ↓

经验规则:
  - 数据量大 (>50K):  1-2 epochs
  - 数据量小 (<10K):  2-4 epochs
  - 我们的 BEE 数据约 12K (去重后): 2 epochs 合适
```

### 6.2 `trainer.save_freq = 500` & `trainer.test_freq = 500`

每 500 步保存一次检查点，同时计算 val_loss：

```
假设 total_steps ≈ 2000 (12K 数据, batch 128, 2 epochs)

步 500:   保存 ckpt + 验证  → global_step_500/
步 1000:  保存 ckpt + 验证  → global_step_1000/
步 1500:  保存 ckpt + 验证  → global_step_1500/
步 2000:  保存 ckpt + 验证  → global_step_2000/ (最终)

→ 一共 4 个检查点，每个都有 val_loss
→ 每个检查点的 huggingface/ 子目录可以独立拿去评测
```

### 6.3 `trainer.max_ckpt_to_keep = 5`

只保留最近 5 个检查点，旧的自动删除：

```
单个检查点大小:
  FSDP 分片:     ~28 GB (8 rank × 3.5 GB)
  HF 模型:       ~14 GB
  总计:          ~42 GB

5 个检查点:     ~210 GB

如果不限制:     可能填满磁盘！
```

### 6.4 `trainer.checkpoint.save_contents += "hf_model"`

这是本项目**非常关键**的配置。默认只保存 FSDP 分片，加上 `"hf_model"` 后：

```
默认 (无 hf_model):
  global_step_500/
  ├── model_world_size_8_rank_0.pt    ← FSDP 分片，不能直接用
  ├── model_world_size_8_rank_1.pt
  └── ...
  ❌ 无法直接用 vLLM 加载评测
  ❌ 需要手动写脚本合并分片

加了 hf_model:
  global_step_500/
  ├── model_world_size_8_rank_0.pt
  ├── ...
  └── huggingface/                    ← 完整 HF 模型
      ├── config.json
      ├── model.safetensors           ← ~14 GB
      ├── tokenizer.json
      └── ...
  ✅ 直接用 vLLM 加载: vllm serve huggingface/
  ✅ 直接用 transformers 加载: AutoModelForCausalLM.from_pretrained("huggingface/")
```

---

## 7. 启动脚本完整解读

我们项目的启动脚本 `run_sft.sh` 长这样：

```bash
#!/bin/bash

# ===== 路径配置 =====
PHASE1_DIR="coding_model_project/phase_1_ SFT"
DATA_DIR="${PHASE1_DIR}/data"
CKPT_DIR="${PHASE1_DIR}/checkpoints/rlvr_coding_model/phase1_sft_qwen7b_coder"

# ===== 启动训练 =====
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    \
    data.train_files=${DATA_DIR}/sft_train.parquet \          # 训练数据
    data.val_files=${DATA_DIR}/sft_val.parquet \              # 验证数据
    data.multiturn.enable=true \                              # 使用多轮格式
    data.multiturn.messages_key=messages \                    # messages 列名
    data.max_length=4096 \                                    # 最大序列长度
    data.truncation=right \                                   # 右截断
    data.train_batch_size=128 \                               # 全局 batch
    data.micro_batch_size_per_gpu=2 \                         # micro batch
    data.balance_dp_token=True \                              # 平衡 DP token
    \
    model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \  # 预训练模型
    model.trust_remote_code=true \                            # Qwen 需要
    model.enable_gradient_checkpointing=true \                # 省显存
    model.strategy=fsdp2 \                                    # FSDP 版本
    model.fsdp_config.model_dtype=bf16 \                      # 模型精度
    \
    optim.lr=2e-5 \                                           # 学习率
    optim.lr_warmup_steps_ratio=0.05 \                        # warmup 5%
    optim.clip_grad=1.0 \                                     # 梯度裁剪
    optim.lr_scheduler=cosine \                               # cosine 调度
    \
    ulysses_sequence_parallel_size=2 \                        # 2 路序列并行
    use_remove_padding=true \                                 # 配合 SP 使用
    \
    trainer.total_epochs=2 \                                  # 训练 2 轮
    trainer.save_freq=500 \                                   # 每 500 步保存
    trainer.test_freq=500 \                                   # 每 500 步验证
    trainer.max_ckpt_to_keep=5 \                              # 保留 5 个 ckpt
    trainer.default_local_dir=${CKPT_DIR} \                   # 检查点目录
    trainer.project_name=rlvr_coding_model \                  # WandB 项目
    trainer.experiment_name=phase1_sft_qwen7b_coder \         # WandB 实验
    trainer.logger='["console","wandb"]' \                    # 日志后端
    trainer.seed=42 \                                         # 随机种子
    trainer.resume_mode=auto \                                # 自动恢复
    trainer.checkpoint.save_contents='["model","optimizer","extra","hf_model"]'
```

---

## 8. Smoke Test — 训练前的快速验证

正式训练前，先做一个 10 步的冒烟测试，确认一切正常：

```bash
# Smoke test: 只跑 10 步，验证流程能跑通
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${DATA_DIR}/sft_train.parquet \
    data.val_files=${DATA_DIR}/sft_val.parquet \
    data.multiturn.enable=true \
    data.max_length=4096 \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.strategy=fsdp2 \
    optim.lr=2e-5 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.total_training_steps=10 \       # ← 只跑 10 步!
    trainer.total_epochs=999 \               # ← 设大，让 total_training_steps 生效
    trainer.save_freq=5 \                    # ← 第 5 步保存一次
    trainer.test_freq=5 \                    # ← 第 5 步验证一次
    trainer.default_local_dir=/tmp/sft_smoke_test \
    trainer.logger='["console"]' \           # ← 不发 WandB
    trainer.resume_mode=disable              # ← 不恢复，从头开始
```

**Smoke test 要验证的点**：
1. 不 OOM（显存够用）
2. loss 在下降（数值正常，不是 nan/inf）
3. 检查点能保存（`/tmp/sft_smoke_test/global_step_5/` 存在）
4. `huggingface/` 子目录存在
5. 第 5 步有 val_loss 输出

---

## 9. 知识点总结

| 概念 | 一句话解释 |
|------|-----------|
| **Hydra** | YAML 默认值 + 命令行覆盖 = 最终配置，方便实验管理 |
| **全局 batch size** | 所有 GPU 合计的 batch，verl 内部自动除以 DP 数 |
| **micro_batch_size** | 每次实际 forward 的样本数，受显存限制 |
| **梯度累积步数** | = (per_dp_batch_size) / micro_batch_size |
| **bf16** | 16位浮点数，比fp32省一半显存，精度足够 |
| **lr=2e-5** | SFT 的保守学习率，平衡学习效果和防遗忘 |
| **weight_decay=0.01** | L2 正则化，防止参数过大导致过拟合 |
| **warmup 5%** | 训练初期用小学习率稳定梯度，然后升到 peak |
| **total_epochs=2** | 防过拟合的 epoch 数，SFT 不宜训练过多轮 |
| **save_contents: hf_model** | 每个检查点导出完整 HF 模型，直接可用于 vLLM 评测 |
| **Smoke test** | 正式训练前跑 10 步验证流程正确，避免浪费 GPU 时间 |

---

> **思考题**：
> 1. 如果你的 GPU 只有 24GB 显存（4090），`micro_batch_size_per_gpu` 可能需要设成多少？如果设成 1 但 per_dp_batch_size=32，梯度累积步数是多少？
> 2. 为什么 `trainer.total_training_steps` 优先于 `trainer.total_epochs`？什么场景下你会直接指定步数而不是 epoch 数？
> 3. 如果训练数据从 12K 增加到 50K，你会调整哪些超参数？为什么？
