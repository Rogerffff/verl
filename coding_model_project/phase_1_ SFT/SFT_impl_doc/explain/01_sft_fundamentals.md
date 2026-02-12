# Part 1: SFT 基础知识 — SFT 是什么 & 为什么要做

> 本文是 verl SFT 讲解系列的第 1 部分，帮助你建立 SFT 的核心概念，理解它在整个项目中的定位。

---

## 1. LLM 训练的三个阶段

要理解 SFT，首先要知道大语言模型(LLM)的训练通常分为三个阶段：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Pre-training │ ──▶ │     SFT      │ ──▶ │      RL      │
│   (预训练)     │     │  (监督微调)   │     │  (强化学习)   │
└──────────────┘     └──────────────┘     └──────────────┘
   海量文本             人类示范数据            奖励信号
   学会"语言"           学会"遵循指令"          学会"做得更好"
```

| 阶段 | 输入 | 目标 | 类比 |
|------|------|------|------|
| **Pre-training** | 互联网文本（TB 级） | 学会语言模式、世界知识 | 读完图书馆所有书 |
| **SFT** | (指令, 正确回答) 对 | 学会按指令格式回答 | 老师一对一教你做题 |
| **RL (GRPO)** | 奖励信号（对/错） | 自我探索，提升质量 | 自己刷题，看对错反馈进步 |

### 我们项目的位置

```
Phase 0 (Baseline)  →  Phase 1 (SFT)  →  Phase 3 (GRPO)
     ↑                      ↑                   ↑
  评测原始模型          你现在在这里！        在线强化学习
```

我们使用的 Base 模型是 **Qwen2.5-Coder-7B-Instruct**，它已经完成了预训练和基础的指令微调。我们要在它的基础上，用**编程竞赛的正确解答**再做一次 SFT，让它更擅长写代码。

---

## 2. SFT 到底在做什么？

### 2.1 核心思想：Teacher Forcing（教师强制）

SFT 的训练方式叫 **Teacher Forcing**，简单说就是：

> **给模型看正确答案的每一步，让它学会预测下一个 token。**

```
输入序列:  [系统提示] [题目描述] [正确代码...]
            ↓          ↓         ↓
模型要学:   ×          ×        预测每个代码token
           不算loss    不算loss    计算loss
```

举个具体例子：

```python
# 假设正确代码是: n = int(input())
# 模型看到 "n" 后，要预测下一个 token 是 " ="
# 模型看到 "n =" 后，要预测下一个 token 是 " int"
# 模型看到 "n = int" 后，要预测下一个 token 是 "("
# ...以此类推
```

这就是 **Next Token Prediction（下一个 token 预测）**，是所有 GPT 类模型的核心训练方式。

### 2.2 Cross-Entropy Loss（交叉熵损失）

模型对每个位置会输出一个概率分布（词表中每个 token 的概率），我们用 **Cross-Entropy Loss** 来衡量这个预测和正确答案之间的差距：

```
                    模型预测的概率分布
                    ┌───────────────┐
token "int":        │  0.3  ←←←←←← │ ← 正确答案，希望这个概率尽量大
token "str":        │  0.2          │
token "float":      │  0.1          │
token "print":      │  0.05         │
...其他 tokens:     │  ...          │
                    └───────────────┘

Cross-Entropy Loss = -log(正确token的预测概率)
                   = -log(0.3)
                   ≈ 1.2

如果模型预测正确token的概率是0.9:
Loss = -log(0.9) ≈ 0.1  ← loss 越小越好
```

**训练目标**：最小化所有位置上 Cross-Entropy Loss 的平均值。

### 2.3 Loss Masking（损失遮罩）—— 只在回答上计算 Loss

这是 SFT 最关键的概念之一。我们**不希望模型学会"背诵题目"**，只希望它学会"写出正确代码"。所以：

```
位置:     [  系统提示  ] [  用户题目  ] [  助手回答（代码）  ]
loss_mask: [0, 0, 0, ...] [0, 0, 0, ...] [1, 1, 1, ..., 1, 0]
           ← 不计算loss →  ← 不计算loss →  ← 计算loss →  ↑
                                                        最后一个token
                                                        没有下一个要预测
                                                        的，所以也是 0
```

**为什么最后一个 token 的 loss_mask 也是 0？**
因为 Next Token Prediction 需要"下一个 token"作为标签，最后一个 token 后面没有 token 了，所以无法计算 loss。

在 verl 的代码中（`multiturn_sft_dataset.py` 第 211-216 行），这个逻辑非常清晰：

```python
# 来自 MultiTurnSFTDataset._process_single_message()
if message["role"] == "assistant":
    loss_mask = torch.ones_like(attention_mask)   # 全1：计算 loss
    loss_mask[:len(self.generation_prompt)] = 0    # 生成提示符也要mask掉
else:
    loss_mask = torch.zeros_like(attention_mask)   # 全0：不计算 loss
```

- `system` 消息 → `loss_mask = 全0`（不学习系统提示）
- `user` 消息 → `loss_mask = 全0`（不学习题目）
- `assistant` 消息 → `loss_mask = 全1`（学习代码），但 generation_prompt 部分也 mask 掉

> **什么是 generation_prompt？**
>
> 在 Qwen 的 chat template 中，assistant 回复前面有一个特殊标记，类似 `<|im_start|>assistant\n`。这是模板格式，不是模型要"学"的内容，所以也被 mask 掉。

---

## 3. 在我们项目中，SFT 要解决什么问题？

回顾 Phase 0 基线评测的结果，原始模型存在大量**低级错误**：

| 错误类型 | 问题 | SFT 能否解决 |
|----------|------|-------------|
| **Syntax Error** | 代码语法错误，无法编译 | **能显著改善** |
| **Runtime Error** | 运行时崩溃（IndexError, ValueError...） | **能改善** |
| **Timeout** | 代码超时（死循环、效率差） | **部分改善** |
| **Wrong Answer** | 逻辑错误，结果不对 | 改善有限（需要 RL） |

### SFT 的核心目标

```
Phase 0 的问题:           Phase 1 SFT 的目标:
├── 输出格式不稳定          ├── 稳定 stdin/stdout 格式
├── 经常语法错误            ├── 语法错误率 ↓ 30-60%
├── 运行时异常多            ├── 运行时错误 ↓ 10-30%
├── 代码不可执行            ├── exec_success_rate ↑ 20-40pp ★最重要
└── accepted 率低          └── accepted@1 小幅 ↑ (1-5pp)
```

简单说：**SFT 的目标是让模型输出"能跑的代码"，而不是"全对的代码"。**

"全对"是 Phase 3 GRPO 的任务，RL 通过奖励信号让模型学会选择更好的解法。

---

## 4. SFT 的训练数据是什么样的？

我们使用 **BEE CodeContests** 数据集，包含编程竞赛题目和经过验证的 Python 正确解答：

```json
{
  "name": "1283_D. Christmas Trees",
  "source": 2,
  "difficulty": 10,
  "language": "PYTHON3",
  "text": "### Prompt\n[题目描述...]\n\n### Response\n```python3\n[正确代码]\n```"
}
```

经过数据处理后，会变成 verl 需要的格式（OpenAI messages format）：

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert Python programmer..."},
    {"role": "user", "content": "在Python3中解决以下问题: [题目描述]"},
    {"role": "assistant", "content": "<code>\nimport sys\nn = int(input())\n...\n</code>"}
  ]
}
```

这就是一条训练样本。模型会从这样的**（题目, 正确解答）** 配对中学习。

---

## 5. 需要注意的风险：Catastrophic Forgetting（灾难性遗忘）

SFT 有一个重要风险：**模型可能在学会新任务的同时，忘记之前学过的能力。**

```
比喻: 一个学生拼命练数学，结果语文全忘了

在我们的项目中:
  - 模型学会了写 CodeContests 风格的代码  ✅
  - 但 HumanEval 的 pass@1 可能下降      ⚠️ 灾难性遗忘！
```

**缓解措施**：
1. **使用较小的学习率** (lr=2e-5)：微调而不是重新学习
2. **只训练 2 个 epoch**：避免过拟合训练数据
3. **监控 HumanEval pass@1**：如果退化超过 5pp，说明遗忘太严重
4. **监控 MBPP_reg pass@1**：作为回归测试的"金丝雀"

---

## 6. SFT vs RL（GRPO）的区别

| 维度 | SFT | RL (GRPO) |
|------|-----|-----------|
| **学习信号** | 正确答案（监督信号） | 奖励分数（pass_ratio） |
| **训练方式** | 模仿人类示范 | 自我探索 + 奖励最大化 |
| **目标** | 学会"能跑的代码" | 学会"通过更多测试的代码" |
| **数据需求** | 需要正确解答 | 只需要题目 + 判题器 |
| **风险** | 过拟合训练数据 | reward hacking、KL 发散 |
| **verl 中的实现** | `fsdp_sft_trainer.py` | `ray_trainer.py` (PPO/GRPO) |

**为什么不直接做 RL，跳过 SFT？**

因为 RL 的探索需要一个好的起点。如果原始模型 70% 的输出都是语法错误，RL 的采样效率极低（大部分采样 reward = 0），训练信号太稀疏。SFT 先让模型"能跑代码"，RL 再在这个基础上提升"代码质量"。

---

## 7. 知识点总结

| 概念 | 一句话解释 |
|------|-----------|
| **SFT (Supervised Fine-Tuning)** | 用 (指令, 正确回答) 对微调模型，让它学会按指令回答 |
| **Teacher Forcing** | 训练时给模型看正确答案的每一步，学习预测下一个 token |
| **Cross-Entropy Loss** | 衡量模型预测概率分布和真实标签之间的差距 |
| **Loss Masking** | 只在 assistant 回复 token 上计算 loss，忽略 prompt |
| **Next Token Prediction** | 模型在每个位置预测下一个 token，这是 GPT 的核心训练方式 |
| **Catastrophic Forgetting** | 模型学新任务时忘记旧能力，需要用小 lr 和监控来缓解 |
| **exec_success_rate** | 代码可执行率，SFT 阶段最重要的指标 |

---

## 8. 与实现文档的对应关系

本部分内容对应你的 SFT 实现文档的**背景知识**部分：
- 为什么要做 SFT → 对应 `final_experiment_design.md` Phase 1 的"目的"
- SFT 的预期效果 → 对应实验设计中的"必须产出"表格
- 数据格式概览 → 对应 `01_data_preprocessing.md` 的输入输出设计

**下一部分** 将详细讲解数据从 BEE 原始格式到 verl 训练格式的完整流水线。

---

> **思考题**（帮助检验理解）：
> 1. 为什么 loss_mask 中系统提示和用户输入的位置是 0？如果我们也在这些位置计算 loss 会怎样？
> 2. 为什么 SFT 使用较小的学习率（2e-5 而不是 1e-3）？
> 3. 如果训练数据中有错误的代码（不能通过测试的代码），会对 SFT 产生什么影响？
