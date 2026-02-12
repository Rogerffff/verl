# Part 4: 训练循环逐行解析

> 本文是 verl SFT 讲解系列的第 4 部分，深入 `training_step()` 和 `_compute_loss_and_backward()` 的每一行代码，理解模型到底是怎么"学"的。
> 对应实现文档：`02_training_config_and_flow.md` 第 6 节

---

## 1. 训练一步的全景

先用一张图回顾一步训练（`training_step()`）的完整流程：

```
                        一个 batch (32个样本)
                               │
                    ┌──────────┴──────────┐
                    │  拆成 16 个 micro-batch │  (每个 2 个样本)
                    │  (梯度累积)            │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ↓                     ↓                     ↓
   micro-batch 1         micro-batch 2    ...   micro-batch 16
   ┌───────────┐         ┌───────────┐         ┌───────────┐
   │ forward   │         │ forward   │         │ forward   │
   │ loss计算   │         │ loss计算   │         │ loss计算   │
   │ backward  │         │ backward  │         │ backward  │
   │ 梯度累加   │         │ 梯度累加   │         │ 梯度累加   │
   └───────────┘         └───────────┘         └───────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │  梯度裁剪 (clip_grad)  │
                    │  optimizer.step()     │
                    │  lr_scheduler.step()  │
                    │  all_reduce 平均 loss  │
                    └──────────────────────┘
                               ↓
                    返回 {train/loss, train/lr, train/time}
```

---

## 2. `training_step()` 逐行解析

源码位置：`fsdp_sft_trainer.py:473-531`（原始代码行号可能有偏移，以中文注释版为准）

```python
def training_step(self, batch: TensorDict):
    start_time = time.time()                          # ① 计时开始

    self.fsdp_model.train()                            # ② 设为训练模式
    # train模式 vs eval模式 的区别:
    #   train: Dropout 启用, BatchNorm 用当前batch统计
    #   eval:  Dropout 关闭, BatchNorm 用全局统计

    self.optimizer.zero_grad()                         # ③ 清零梯度
    # 为什么要清零? 因为 PyTorch 默认梯度是"累加"的
    # 如果不清零，上一步的梯度会混进来

    # ④ 拆分 micro-batches
    micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
    n_micro_batches = len(micro_batches)               # = 32 / 2 = 16
    step_loss = 0

    # ⑤ 对每个 micro-batch 做 forward + backward
    for micro_batch in micro_batches:
        loss = self._compute_loss_and_backward(
            batch=micro_batch,
            n_micro_batches=n_micro_batches            # 传入总数用于归一化
        )
        step_loss += loss.item()                       # 累加 loss 值(用于日志)

    # ⑥ 梯度裁剪
    if self.config.model.strategy == "fsdp2":
        grad_norm = fsdp2_clip_grad_norm_(
            self.fsdp_model.parameters(),
            max_norm=self.config.optim.clip_grad        # max_norm = 1.0
        )
    # 返回裁剪前的梯度范数，用于监控训练健康

    # ⑦ 安全检查: 如果梯度异常(inf/nan)，跳过更新
    if not torch.isfinite(grad_norm):
        print(f"WARN: grad_norm is not finite: {grad_norm}")
        self.optimizer.zero_grad()                     # 丢弃异常梯度
    else:
        self.optimizer.step()                          # ⑧ 更新参数

    self.lr_scheduler.step()                           # ⑨ 更新学习率

    # ⑩ 跨所有 GPU 平均 loss
    step_loss = torch.tensor(step_loss).to(self.device_name)
    torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)

    end_time = time.time()
    return {
        "train/loss": step_loss.detach().item(),       # 平均 loss
        "train/lr(1e-3)": lr * 1e3,                    # 当前学习率
        "train/time(s)": end_time - start_time,        # 本步耗时
        # 注意: grad_norm 没有返回! 这是需要修复的 bug
    }
```

### 关键步骤详解

#### ③ `optimizer.zero_grad()` — 为什么要清零梯度？

PyTorch 的梯度是**累加**的（additive）。这在梯度累积时很有用，但每次参数更新前必须清零：

```
如果不清零:
  step 1 梯度: [0.1, -0.2, 0.3]
  step 2 梯度: [0.05, -0.1, 0.2]    ← backward 会加到 step 1 上!
  实际梯度:    [0.15, -0.3, 0.5]    ← 不是 step 2 的真实梯度!

正确做法:
  zero_grad()
  step 2 梯度: [0.05, -0.1, 0.2]    ← 这才是 step 2 的真实梯度
```

#### ⑥ 梯度裁剪 (Gradient Clipping) — 防止梯度爆炸

```
                  梯度范数 = √(g₁² + g₂² + ... + gₙ²)

                  如果梯度范数 > max_norm (1.0):
                    所有梯度按比例缩小:
                    g_new = g * (max_norm / 梯度范数)

效果示意:
  正常梯度:    [0.3, -0.2, 0.1]   范数=0.37  < 1.0  → 不变
  爆炸梯度:    [5.0, -8.0, 3.0]   范数=9.9   > 1.0  → 缩放到范数=1.0
  缩放后:      [0.5, -0.8, 0.3]   范数=1.0           → 安全了
```

**为什么会梯度爆炸？** 长序列、不稳定的 loss、学习率太大等都可能导致。裁剪是一道安全网。

#### ⑩ `all_reduce` — 跨 GPU 平均 loss

每个 GPU 只看到一部分数据，算出来的 loss 不同。`all_reduce(AVG)` 把所有 GPU 的 loss 取平均，这样日志中记录的是"全局 loss"：

```
GPU 0 loss: 2.3  ─┐
GPU 1 loss: 2.1   │  all_reduce(AVG)     所有 GPU
GPU 2 loss: 2.5   ├─────────────────→    都得到 2.3
GPU 3 loss: 2.3  ─┘
```

---

## 3. `_compute_loss_and_backward()` 逐行解析

这是计算 loss 的核心函数。源码位置：`fsdp_sft_trainer.py:538-`

### 3.1 输入准备

```python
def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
    # 将数据移到 GPU
    input_ids = batch["input_ids"].to("cuda")         # (batch, seq_len)
    attention_mask = batch["attention_mask"].to("cuda") # (batch, seq_len)
    position_ids = batch["position_ids"].to("cuda")    # (batch, seq_len)

    # ★ 关键: loss_mask 要从第2个位置开始取
    loss_mask = batch.pop("loss_mask")[:, 1:].reshape(-1).to("cuda")
    #                                  ↑ ↑
    #                          去掉第1个  展平成一维
```

**为什么 `loss_mask[:, 1:]` 要从第 2 个位置开始？**

这与 Next Token Prediction 的"错位"有关：

```
位置:        0     1     2     3     4     5     6     7
input_ids: [sys  sys  user  user  asst  asst  asst  pad]
loss_mask: [0    0    0     0     1     1     1     0  ]  ← 原始 loss_mask

标签(labels) = input_ids 左移1位:
labels:    [sys  user  user  asst  asst  asst  pad   ×]
           位置 0预测1  1预测2  2预测3  3预测4  4预测5  5预测6  6预测7

logits:    位置 0    1     2     3     4     5     6
           预测1   预测2  预测3  预测4  预测5  预测6  预测7

对齐后:
shift_logits: [位置0, 位置1, 位置2, 位置3, 位置4, 位置5, 位置6]  (seq_len - 1)
shift_labels: [位置1, 位置2, 位置3, 位置4, 位置5, 位置6, 位置7]  (seq_len - 1)
loss_mask[:, 1:]: [0,    0,    0,     1,     1,     1,     0]    (seq_len - 1)
                                      ↑     ↑     ↑
                         只有这3个位置计算loss，对应预测assistant的token
```

`loss_mask[:, 1:]` 正好和 `shift_logits`、`shift_labels` 长度一致（都是 `seq_len - 1`）。

### 3.2 前向传播与 Loss 计算（普通模式）

```python
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    # reduction="none" 表示不自动求平均，返回每个 token 的 loss

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # autocast: 自动把计算转成 bf16，节省显存和加速
        # 但 loss 计算等关键步骤仍用 fp32

        # ① 构造标签: 输入左移 1 位
        labels = input_ids[:, 1:].contiguous()
        # input_ids: [t0, t1, t2, t3, t4, t5, t6, t7]
        # labels:    [t1, t2, t3, t4, t5, t6, t7]

        # ② 前向传播: 模型输出每个位置对所有 token 的概率分布
        output = self.fsdp_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False    # 训练时不需要 KV cache
        )
        logits = output.logits  # 形状: (batch, seq_len, vocab_size)
        #                               (2,     4096,    151936)  Qwen词表大小

        # ③ 对齐 logits 和 labels
        shift_logits = logits[..., :-1, :].contiguous()   # 去掉最后一个位置
        shift_labels = labels.contiguous()
        # shift_logits: (batch, seq_len-1, vocab_size)
        # shift_labels: (batch, seq_len-1)

        # ④ 展平
        shift_logits = shift_logits.view(-1, vocab_size)  # (batch*(seq_len-1), vocab_size)
        shift_labels = shift_labels.view(-1)               # (batch*(seq_len-1),)

        # ⑤ 计算每个 token 的 cross-entropy loss
        loss = loss_fct(shift_logits, shift_labels)        # (batch*(seq_len-1),)

        # ⑥ 用 loss_mask 过滤: 只保留 assistant 回复的 loss
        loss = loss * loss_mask
        # loss_mask 中 0 的位置 loss 被乘成 0 (不参与)
        # loss_mask 中 1 的位置 loss 被保留
```

### 3.3 Loss 归一化

```python
        # ⑦ 统计有效 token 数
        valid_token_this_rank = torch.sum(loss_mask)
        # 例如: loss_mask 中有 300 个 1 → 300 个有效 token

        # ⑧ 可选: 跨 GPU 平衡 token 数
        if self.config.data.balance_dp_token:
            torch.distributed.all_reduce(valid_token_this_rank)
            dp_size = torch.distributed.get_world_size()
        else:
            dp_size = 1

        # ⑨ 计算平均 loss
        loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size
        # sum(loss) = 所有有效 token 的 loss 之和
        # / valid_token = 平均到每个有效 token
        # * dp_size = 如果开了 balance_dp_token，补偿 all_reduce 的影响
        # + 1e-8 = 防止除以零

        # ⑩ 除以 micro-batch 数
        loss = loss / n_micro_batches
        # 因为梯度会累加，如果有 16 个 micro-batch，
        # 每个的 loss 要 /16 才等效于一个大 batch 的平均 loss

        # ⑪ 反向传播
        if do_backward:
            loss.backward()
            # 计算所有参数的梯度
            # 梯度会累加到 param.grad 中

        return loss
```

### 3.4 为什么 `loss / n_micro_batches`？

这是梯度累积的关键。假设大 batch 有 32 个样本，分成 16 个 micro-batch（每个 2 个样本）：

```
理想情况 (一次算 32 个):
  Loss = 平均(32个样本的loss)
  梯度 = ∂Loss/∂θ

梯度累积 (分 16 次算):
  第1次:  loss_1 = 平均(2个样本的loss) / 16
          backward → grad += ∂loss_1/∂θ

  第2次:  loss_2 = 平均(2个样本的loss) / 16
          backward → grad += ∂loss_2/∂θ
  ...
  第16次: loss_16 = 平均(2个样本的loss) / 16
          backward → grad += ∂loss_16/∂θ

  最终 grad = Σ(∂loss_i/∂θ) = ∂(Σloss_i)/∂θ
            = ∂(平均(32个样本的loss))/∂θ  ← 和一次性算32个等价!
```

除以 `n_micro_batches` 保证了**梯度累积等价于大 batch 训练**。

---

## 4. 用一个具体例子走一遍

假设我们有一条训练样本：

```
messages = [
  {"role": "system", "content": "You are an expert Python programmer."},
  {"role": "user", "content": "Solve: read n, print n*2"},
  {"role": "assistant", "content": "n=int(input())\nprint(n*2)"}
]
```

### Step 1: MultiTurnSFTDataset 输出

经过 tokenize + padding 后（假设 max_length=20 简化）：

```
位置:        0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19
input_ids:  [sys  sys  sys  usr  usr  usr  usr  gen  gen  n   =   int  (   inp  )   \n  pri  )   pad  pad]
attn_mask:  [1    1    1    1    1    1    1    1    1    1   1    1   1    1    1    1    1   1    0    0]
loss_mask:  [0    0    0    0    0    0    0    0    0    1   1    1   1    1    1    1    1   1    0    0]
             └──system──┘  └───user────┘  └gen─┘  └──────assistant 回复───────────────┘  └pad┘
                                          prompt
```

### Step 2: loss_mask 错位

```python
loss_mask[:, 1:] →
位置:        1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18
loss_mask:  [0    0    0    0    0    0    0    0    1    1    1    1    1    1    1    1    1    0]
```

### Step 3: 前向传播

```
模型输入:  位置 0-19 的 input_ids
模型输出:  位置 0-19 每个位置预测下一个 token 的概率 (logits)

shift_logits = logits[:, :-1, :]   → 位置 0-18 的预测
shift_labels = input_ids[:, 1:]    → 位置 1-19 的真实 token
```

### Step 4: 计算 loss

```
位置 0: 预测位置1应该是"sys"  → loss=0.5, loss_mask=0 → 0     (system，不算)
位置 1: 预测位置2应该是"sys"  → loss=0.3, loss_mask=0 → 0     (system，不算)
...
位置 8: 预测位置9应该是"n"   → loss=2.1, loss_mask=1 → 2.1  ★ 算loss!
位置 9: 预测位置10应该是"="  → loss=1.8, loss_mask=1 → 1.8  ★ 算loss!
位置 10: 预测位置11应该是"int" → loss=0.9, loss_mask=1 → 0.9 ★ 算loss!
...
位置 17: 预测位置18应该是"pad"→ loss=0.4, loss_mask=0 → 0    (padding，不算)

最终 loss = sum(有效loss) / sum(loss_mask) = (2.1+1.8+0.9+...) / 9
```

---

## 5. validation_step() — 验证步骤

验证和训练的区别很简单（`fsdp_sft_trainer.py:533-542`）：

```python
def validation_step(self, batch: TensorDict):
    self.fsdp_model.eval()               # ① 设为评估模式(关闭Dropout)
    with torch.no_grad():                 # ② 不计算梯度(省显存+加速)
        loss = self._compute_loss_and_backward(
            batch,
            do_backward=False              # ③ 不做反向传播!
        )
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
    return loss
```

**和训练的 3 个区别**：
1. `eval()` 模式：关闭 Dropout（如果有的话）
2. `torch.no_grad()`：不追踪计算图，不计算梯度，省约 50% 显存
3. `do_backward=False`：只算 loss 值，不更新任何参数

---

## 6. fit() 主循环中的验证和保存逻辑

```python
# fit() 中的关键判断逻辑 (fsdp_sft_trainer.py:775-803)

for data in train_dataloader:
    global_step += 1
    metric = self.training_step(data)

    is_last_step  = global_step >= total_training_steps
    is_valid_step = global_step % test_freq == 0      # 每500步
    is_save_step  = global_step % save_freq == 0      # 每500步

    # 验证: 最后一步 OR 每500步
    if is_last_step or (test_freq > 0 and is_valid_step):
        val_losses = []
        for val_data in val_dataloader:
            val_loss = self.validation_step(val_data)
            val_losses.append(val_loss)
        mean_val_loss = torch.mean(torch.stack(val_losses))
        tracking.log({"val/loss": mean_val_loss}, step=global_step)

    # 保存: 最后一步 OR 每500步
    if is_last_step or (save_freq > 0 and is_save_step):
        self.save_checkpoint(step=global_step)

    # 提前退出
    if is_last_step:
        return
```

一个典型训练过程的时间线：

```
步数:    1    2  ...  499   500   501  ...  999  1000  ... 2000  ... 最后
         │    │       │      │     │        │     │         │        │
训练:    ████████████████    ████████████████     ██████    █████████
验证:                   ────┤              ────┤          ────┤   ────┤
保存:                   ────┤              ────┤          ────┤   ────┤
WandB:   每步记录loss ──→   记录val/loss ──→        ──→         ──→
```

---

## 7. 需要修复的 Bug: grad_norm 未返回

当前 `training_step()` 的返回值没有包含 `grad_norm`（`fsdp_sft_trainer.py:527-531`）：

```python
# 当前代码
return {
    "train/loss": step_loss.detach().item(),
    "train/lr(1e-3)": lr * 1e3,
    "train/time(s)": spend_time_per_step,
    # grad_norm 没有返回!
}
```

**需要修改为**：

```python
return {
    "train/loss": step_loss.detach().item(),
    "train/lr(1e-3)": lr * 1e3,
    "train/time(s)": spend_time_per_step,
    "train/grad_norm": grad_norm.detach().item(),  # ← 加这一行
}
```

**为什么 grad_norm 重要？**

| grad_norm 范围 | 含义 |
|---------------|------|
| 0.1 - 10.0 | 正常，训练健康 |
| > 100 | 梯度爆炸，可能需要降低 lr |
| 持续 = max_norm (1.0) | 经常被裁剪，可能 lr 太大 |
| 突然飙升 | loss spike，可能有异常数据 |
| ≈ 0 | 梯度消失，模型不在学习 |

---

## 8. 知识点总结

| 概念 | 一句话解释 |
|------|-----------|
| **Next Token Prediction** | 模型看到前 n 个 token，预测第 n+1 个 token |
| **Shift (错位)** | logits 去掉最后一个，labels 去掉第一个，让它们对齐"预测→真实" |
| **CrossEntropyLoss(reduction="none")** | 不自动求平均，返回每个 token 的 loss，方便用 loss_mask 过滤 |
| **loss_mask[:, 1:]** | 和 shift 后的序列长度对齐，只在 assistant 回复位置计算 loss |
| **Gradient Accumulation** | 多个 micro-batch 梯度累加，等效于大 batch 但省显存 |
| **loss / n_micro_batches** | 保证梯度累积和大 batch 数学等价 |
| **Gradient Clipping** | 梯度范数超过阈值时按比例缩小，防止梯度爆炸 |
| **all_reduce(AVG)** | 所有 GPU 的值取平均，用于同步 loss |
| **torch.autocast(bf16)** | 自动把前向计算转 bf16，省显存加速，关键步骤用 fp32 |
| **torch.no_grad()** | 不追踪计算图，验证时使用，省约 50% 显存 |
| **balance_dp_token** | 跨 GPU 统一有效 token 数，避免不同 GPU loss 权重不均 |

---

> **思考题**：
> 1. 如果 `n_micro_batches=1`（不做梯度累积），loss 的计算和有梯度累积时有什么不同？
> 2. 为什么 `loss_fct = nn.CrossEntropyLoss(reduction="none")` 要用 `"none"` 而不是 `"mean"`？如果用 `"mean"` 会怎样？
> 3. `torch.isfinite(grad_norm)` 检测到异常后只是 `zero_grad()` 跳过，为什么不直接终止训练？这样做有什么好处和风险？
