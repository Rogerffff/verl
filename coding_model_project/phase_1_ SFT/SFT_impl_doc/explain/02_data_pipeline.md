# Part 2: 数据流水线 — 从 BEE 原始数据到 verl 训练格式

> 本文是 verl SFT 讲解系列的第 2 部分，详解数据是如何一步步从原始格式变成模型能吃进去的张量的。
> 对应实现文档：`01_data_preprocessing.md`

---

## 1. 数据流水线全景图

先看全局，再看细节。从原始数据到模型输入，经过了以下几个阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据流水线全景                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  阶段 1: 原始数据                                                    │
│  ┌──────────────────────┐                                           │
│  │ BEE JSONL            │  每行一个JSON对象                          │
│  │ {name, source,       │  text 里混着题目和代码                      │
│  │  difficulty, text}   │                                           │
│  └──────────┬───────────┘                                           │
│             ↓                                                       │
│  阶段 2: 解析 & 格式转换              [你需要写的脚本]                  │
│  ┌──────────────────────┐                                           │
│  │ prepare_sft_data.py  │  parse_bee_text() 解析                     │
│  │  • 分离题目/代码      │  构造 messages 列表                        │
│  │  • 加 system prompt  │  98%/2% 划分                              │
│  │  • 保存 Parquet      │                                           │
│  └──────────┬───────────┘                                           │
│             ↓                                                       │
│  阶段 3: Parquet 文件                                                │
│  ┌──────────────────────┐                                           │
│  │ sft_train.parquet    │  只有一列: messages                        │
│  │ sft_val.parquet      │  每行是 [system, user, assistant]          │
│  └──────────┬───────────┘                                           │
│             ↓                                                       │
│  阶段 4: verl Dataset 加载            [verl 框架自动处理]              │
│  ┌──────────────────────┐                                           │
│  │ MultiTurnSFTDataset  │  tokenize + loss_mask + padding           │
│  │  __getitem__() →     │  返回 {input_ids, attention_mask,         │
│  │  张量字典             │         position_ids, loss_mask}          │
│  └──────────┬───────────┘                                           │
│             ↓                                                       │
│  阶段 5: DataLoader 批处理            [PyTorch 自动处理]              │
│  ┌──────────────────────┐                                           │
│  │ StatefulDataLoader   │  batching + DistributedSampler             │
│  │  → batch of tensors  │  多 GPU 数据分发                            │
│  └──────────────────────┘                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

你需要负责的是**阶段 1→3**（写 `prepare_sft_data.py`），**阶段 4→5** 由 verl 框架自动处理。

---

## 2. 阶段 1: BEE 原始数据长什么样

打开 `bee_hq_python_deduped_filtered/train.jsonl`，每行是一个 JSON 对象：

```json
{
  "name": "1283_D. Christmas Trees",
  "source": 2,
  "difficulty": 10,
  "language": "PYTHON3",
  "text": "### Prompt\n\nIn Python3, your task is to solve...\n\n### Response\n\n```python3\nimport sys\nfrom collections import deque\n...\n```"
}
```

### 2.1 `text` 字段的内部结构

`text` 字段是一个大字符串，用特殊分隔符把"题目"和"代码"拼在一起：

```
┌──────────────────────────────────────────────────────┐
│ ### Prompt                                           │ ← 标记开始
│                                                      │
│ In Python3, your task is to solve the following...   │
│ [完整题目描述]                                        │
│ [Input/Output 格式说明]                               │
│ [Examples 样例]                                       │
│                                                      │
│ ### Response                      ← 分隔符，这行很关键 │
│                                                      │
│ ```python3                        ← 代码块开始         │
│ import sys                                           │
│ n, m = map(int, input().split())                     │
│ ...                                                  │
│ print(result)                                        │
│ ```                               ← 代码块结束         │
└──────────────────────────────────────────────────────┘
```

### 2.2 解析逻辑

解析分两步：

```python
def parse_bee_text(text: str) -> tuple[str, str]:
    # Step 1: 用 "### Response" 把题目和答案分开
    parts = text.split("### Response", 1)
    prompt = parts[0].replace("### Prompt\n", "").strip()  # 去掉标记

    # Step 2: 从 Response 部分提取代码块
    response_raw = parts[1].strip()
    code_match = re.search(r'```(?:python3?)\s*\n(.*?)```', response_raw, re.DOTALL)
    if code_match:
        response = code_match.group(1).strip()  # 只取代码内容
    else:
        response = response_raw.strip()          # 降级：取整个内容

    return prompt, response
```

**为什么用正则提取代码块？** 因为 BEE 数据中代码被 ` ```python3 ... ``` ` 包裹着，我们只需要里面的纯代码，不需要 markdown 格式标记。

---

## 3. 阶段 2: 转换为 Messages 格式

### 3.1 为什么选 MultiTurnSFTDataset？

verl 提供了两种 SFT 数据集类，它们的**核心区别**在于数据格式和功能：

```
┌─────────────────────────┐     ┌───────────────────────────┐
│     SFTDataset           │     │  MultiTurnSFTDataset       │
│  (sft_dataset.py)        │     │ (multiturn_sft_dataset.py) │
├─────────────────────────┤     ├───────────────────────────┤
│ Parquet 格式:            │     │ Parquet 格式:              │
│  - prompt 列             │     │  - messages 列             │
│  - response 列           │     │    [system, user, asst]    │
├─────────────────────────┤     ├───────────────────────────┤
│ ❌ 不支持 system prompt  │     │ ✅ 支持 system prompt      │
│ ❌ 手动拼接 chat template│     │ ✅ 自动 apply_chat_template│
│ ❌ loss mask 手动计算    │     │ ✅ 按角色自动 mask          │
└─────────────────────────┘     └───────────────────────────┘
         不选这个  ←                    → 选这个 ✅
```

**选择 MultiTurnSFTDataset 的 3 个关键理由**：

1. **System Prompt 支持**：我们的 Phase 0 评测使用了 system prompt 定义输出格式（`<code>` 标签等），训练时必须保持一致
2. **Chat Template 自动处理**：Qwen 模型有特殊 token（`<|im_start|>`、`<|im_end|>`），MultiTurnSFTDataset 会自动调用 `apply_chat_template()` 处理
3. **Loss Mask 精确控制**：自动按角色分配 loss_mask，assistant 回复计算 loss，其他不计算

### 3.2 verl 如何选择数据集类

在 `fsdp_sft_trainer.py` 的 `create_sft_dataset()` 函数中（约第 851-868 行）：

```python
# verl 的数据集选择逻辑（简化版）
if data_config.custom_cls.path is not None:
    dataset_cls = load_extern_object(...)        # 自定义类优先
elif data_config.multiturn.enable == True:        # ← 我们设置这个为 true
    dataset_cls = MultiTurnSFTDataset             # ← 走这个分支
else:
    dataset_cls = SFTDataset                      # 默认
```

**配置开关**：在启动训练时加上 `data.multiturn.enable=true`。

### 3.3 构造 Messages 列表

解析完题目和代码后，构造成 OpenAI 标准 messages 格式：

```python
SYSTEM_PROMPT = """You are an expert Python programmer.

Output rules:
1. Output Python code only.
2. Include necessary imports only if needed.
3. Wrap the entire code in <code> and </code>.
4. Do not write anything outside the <code> tags.
5. Follow dataset-specific constraints given by the user prompt."""

def bee_record_to_messages(record):
    prompt, response = parse_bee_text(record["text"])

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},   # 角色定义
        {"role": "user",      "content": prompt},           # 题目
        {"role": "assistant", "content": response},         # 正确代码
    ]
    return messages
```

**为什么 system prompt 要和 Phase 0 一致？**

因为模型在推理时（评测/实际使用）会收到同样的 system prompt。如果训练和推理的 system prompt 不一致，模型可能会"困惑"——训练时学的是一种格式，推理时被要求用另一种格式。

### 3.4 保存为 Parquet

```python
import pandas as pd

# 所有 messages 收集到列表中
all_messages = [bee_record_to_messages(rec) for rec in records]

# 随机划分 98% train / 2% val
# ...划分逻辑...

# 保存 Parquet — 只需要一列 "messages"
pd.DataFrame({"messages": train_messages}).to_parquet("sft_train.parquet")
pd.DataFrame({"messages": val_messages}).to_parquet("sft_val.parquet")
```

---

## 4. 阶段 3-4: verl 如何消费 Parquet 数据

这是最重要的部分——理解 verl 框架内部如何把你的 Parquet 文件变成模型能训练的张量。

### 4.1 MultiTurnSFTDataset 初始化流程

当训练启动时，verl 会创建 `MultiTurnSFTDataset` 实例（`multiturn_sft_dataset.py:68-327`）：

```python
# verl 内部初始化流程（简化）
class MultiTurnSFTDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config):
        # 1. 读取 Parquet 文件
        self.dataframe = pd.read_parquet(parquet_files)

        # 2. 提取 messages 列
        self.messages = self.dataframe["messages"].tolist()

        # 3. 提取 chat template 的特殊标记（关键！）
        self.system_prompt, self.generation_prompt = \
            extract_system_prompt_and_generation(tokenizer)
        # 对于 Qwen 模型:
        #   system_prompt   = tokenize("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        #   generation_prompt = tokenize("<|im_start|>assistant\n")
```

**`system_prompt` 和 `generation_prompt` 是什么？**

Qwen 模型的 chat template 会在每条消息前后加特殊标记：

```
<|im_start|>system
You are an expert Python programmer...
<|im_end|>
<|im_start|>user
In Python3, solve...
<|im_end|>
<|im_start|>assistant      ← 这就是 generation_prompt
import sys...
<|im_end|>
```

verl 需要知道这些标记的 token ID，才能正确处理 loss_mask。

### 4.2 `__getitem__()` — 核心处理流程

每次训练取一条数据时，会调用 `__getitem__(item)`（`multiturn_sft_dataset.py:531-`）：

```
输入: messages = [
    {"role": "system", "content": "You are an expert..."},
    {"role": "user", "content": "In Python3, solve..."},
    {"role": "assistant", "content": "import sys\n..."}
]

                    ┌──────────────────────────────┐
                    │  Step 1: 逐条消息 tokenize     │
                    └──────────────┬───────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────┐
    │ message[0] (system):                                    │
    │   input_ids = [151644, 8948, 198, ...]                  │
    │   loss_mask = [0, 0, 0, ...]         ← 全0，不计算loss   │
    │                                                         │
    │ message[1] (user):                                      │
    │   input_ids = [151644, 872, 198, ...]                   │
    │   loss_mask = [0, 0, 0, ...]         ← 全0，不计算loss   │
    │                                                         │
    │ message[2] (assistant):                                  │
    │   input_ids = [151644, 77091, 198, 475, ...]            │
    │   loss_mask = [0, 0, 0, 1, 1, ...]   ← generation_prompt│
    │               ↑────────↑  ↑──────↑      之后才是1       │
    │            gen_prompt部分  实际回复内容                    │
    └─────────────────────────────────────────────────────────┘
                                  ↓
                    ┌──────────────────────────────┐
                    │  Step 2: 拼接所有消息          │
                    └──────────────┬───────────────┘
                                  ↓
    input_ids    = [sys_tokens... | user_tokens... | asst_tokens...]
    loss_mask    = [0, 0, 0, ...  | 0, 0, 0, ...   | 0, 0, 0, 1, 1, 1, ...]
    attn_mask    = [1, 1, 1, ...  | 1, 1, 1, ...   | 1, 1, 1, 1, 1, 1, ...]
                                  ↓
                    ┌──────────────────────────────┐
                    │  Step 3: Padding / Truncation │
                    └──────────────┬───────────────┘
                                  ↓
    如果长度 < max_length (4096):   右边补 pad_token_id, attn_mask 补 0
    如果长度 > max_length (4096):   根据 truncation 策略截断 ("right")
                                  ↓
                    ┌──────────────────────────────┐
                    │  Step 4: 返回张量字典          │
                    └──────────────────────────────┘
    return {
        "input_ids":      tensor of shape (max_length,),
        "attention_mask":  tensor of shape (max_length,),
        "position_ids":   tensor of shape (max_length,),
        "loss_mask":      tensor of shape (max_length,),
    }
```

### 4.3 逐条消息 tokenize 的细节

这是最精妙的部分。来看 `_process_single_message()` 的源码（`multiturn_sft_dataset.py:333-436`）：

```python
def _process_single_message(self, index, message, tools=None, enable_thinking=None):
    # 1. 用 chat template tokenize 这一条消息
    inputs = processor.apply_chat_template(
        [message],                    # 注意：传入的是 [单条消息]
        add_generation_prompt=False,   # 不加额外的 assistant 提示
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]

    # 2. 去重：非第一条消息要去掉重复的 system_prompt
    #    因为 apply_chat_template 会自动给每次调用加 system prompt
    if index != 0 and message["role"] != "system":
        input_ids = input_ids[len(self.system_prompt):]
        attention_mask = attention_mask[len(self.system_prompt):]

    # 3. Loss Mask 分配
    if message["role"] == "assistant":
        loss_mask = torch.ones_like(attention_mask)    # 全1
        loss_mask[:len(self.generation_prompt)] = 0     # generation prompt 不算
    else:
        loss_mask = torch.zeros_like(attention_mask)    # 全0

    return input_ids, loss_mask, attention_mask, inputs
```

**为什么要去除 system_prompt？**

当你对**单条消息**调用 `apply_chat_template` 时，Qwen 的 tokenizer 会自动在前面加上默认的 system prompt。但我们是逐条消息处理后再拼接的，如果不去除，拼接后就会有多个 system prompt：

```
# 不去除的话，拼接结果会是：
<|im_start|>system\nYou are...<|im_end|>    ← 来自 message[0]
<|im_start|>system\nYou are...<|im_end|>    ← 重复！来自 message[1] 的自动添加
<|im_start|>user\n题目...<|im_end|>
<|im_start|>system\nYou are...<|im_end|>    ← 重复！来自 message[2] 的自动添加
<|im_start|>assistant\n代码...<|im_end|>

# 去除后，正确的拼接结果：
<|im_start|>system\nYou are...<|im_end|>    ← 只保留第一条的
<|im_start|>user\n题目...<|im_end|>
<|im_start|>assistant\n代码...<|im_end|>
```

### 4.4 Sanity Check — 验证正确性

verl 在拼接完所有消息后，会做一次 sanity check（`multiturn_sft_dataset.py:293`）：

```python
self.sanity_check(input_ids, messages, tools, enable_thinking)
```

它做的事情是：把所有 messages 一次性传给 `apply_chat_template`，得到的 token 序列应该和逐条处理后拼接的结果**完全一致**。如果不一致，说明去重或拼接逻辑有 bug。

---

## 5. SFTDataset vs MultiTurnSFTDataset 对比详解

为了加深理解，我们对比两个类如何处理同一条数据：

### 5.1 SFTDataset 的做法（`sft_dataset.py:136-204`）

```python
def __getitem__(self, item):
    prompt = "In Python3, solve..."
    response = "import sys\n..."

    # 1. 手动构造 chat template（只有 user 角色，没有 system）
    prompt_chat = [{"role": "user", "content": prompt}]
    prompt_chat_str = tokenizer.apply_chat_template(
        prompt_chat, add_generation_prompt=True, tokenize=False
    )
    response_chat_str = response + tokenizer.eos_token

    # 2. 分别 tokenize
    prompt_ids = tokenizer(prompt_chat_str)["input_ids"][0]
    response_ids = tokenizer(response_chat_str)["input_ids"][0]

    # 3. 拼接
    input_ids = torch.cat([prompt_ids, response_ids])

    # 4. 手动构造 loss_mask
    loss_mask = attention_mask.clone()
    loss_mask[:prompt_length - 1] = 0           # prompt 部分不算
    loss_mask[prompt_length + response_length - 1] = 0  # 最后一个 token 不算
```

**问题**：没有 system prompt，loss_mask 逻辑是手动硬编码的。

### 5.2 MultiTurnSFTDataset 的做法

```python
def __getitem__(self, item):
    messages = [
        {"role": "system", "content": "You are..."},     # ← 有 system prompt
        {"role": "user", "content": "In Python3..."},
        {"role": "assistant", "content": "import sys..."},
    ]

    # 逐条处理，按角色自动分配 loss_mask
    for i, message in enumerate(messages):
        ids, mask, attn, _ = self._process_single_message(i, message)
        # system → mask 全0
        # user → mask 全0
        # assistant → mask 全1（除了 generation_prompt）

    # 拼接 + padding
```

**优势**：system prompt 支持 + 自动 loss_mask + sanity check 保证正确性。

---

## 6. Padding 和 Truncation

### 6.1 为什么需要 Padding？

GPU 需要固定形状的张量才能高效计算。但不同样本的长度不同：
- 简单题目 + 短代码 → 500 tokens
- 复杂题目 + 长代码 → 3000 tokens

所以需要把所有样本补齐到同一长度 `max_length = 4096`：

```
样本 A (500 tokens):  [实际内容................] [pad pad pad ... pad]
样本 B (3000 tokens): [实际内容.........................................] [pad ...]
样本 C (4200 tokens): [实际内容................截断到4096] ← 被截断了！
                      ├────── max_length = 4096 ──────────┤
```

### 6.2 Truncation 策略

配置 `data.truncation` 控制超长样本的处理方式：

| 策略 | 行为 | 风险 |
|------|------|------|
| `"right"` | 从右侧截断，保留前 max_length 个 token | 代码尾部被截掉，可能不完整 |
| `"left"` | 从左侧截断，保留后 max_length 个 token | 题目被截掉 |
| `"error"` | 直接报错 | 安全但需要数据预过滤 |

我们选择 `"right"`，因为：
- BEE 数据中 99.6% 的样本在 4096 tokens 以内
- 极少数超长样本被截断影响不大
- 比 `"error"` 更鲁棒

### 6.3 attention_mask 的作用

padding 的 token 不应该参与注意力计算（它们是"假的"），所以：

```
input_ids:      [实际token ... 实际token, pad_id, pad_id, pad_id, ...]
attention_mask: [1, 1, 1, ..., 1,         0,      0,      0,      ...]
                 ↑ 参与注意力              ↑ 不参与注意力
```

模型的 self-attention 计算会用 attention_mask 来遮住 padding 位置。

---

## 7. 关于 val 划分的理解

### 7.1 2% val 的目的

```
最终过滤后的 SFT 数据
├── 98% → sft_train.parquet  → 用于训练
└──  2% → sft_val.parquet    → 用于计算 val_loss
```

val_loss 的作用是**检测过拟合**：

```
正常训练:                          过拟合了:
train_loss ↓↓↓                    train_loss ↓↓↓
val_loss   ↓↓                     val_loss   ↓ → ↑↑↑ ← 危险信号！
                                  模型在"背诵"训练数据而不是"学习"
```

### 7.2 val_loss ≠ 真正的评测

重要区分：
- **val_loss**：只是计算验证集上的 cross-entropy loss，和训练用同样的方式
- **真正的评测**：让模型**生成代码** → **执行代码** → **判题**，完全不同的流程

val_loss 下降只说明模型越来越会"预测正确答案的下一个 token"，但不等于模型能自己**从头写出**正确代码。所以我们还需要独立的三层评测管线（Part 7 详讲）。

---

## 8. 知识点总结

| 概念 | 一句话解释 |
|------|-----------|
| **Parquet 格式** | 列式存储的数据文件，比 JSON/CSV 更快，支持嵌套数据（如 messages 列表） |
| **OpenAI Messages 格式** | `[{"role": "system/user/assistant", "content": "..."}]`，已成为行业标准 |
| **Chat Template** | 模型特定的格式化规则，把 messages 转成模型能理解的 token 序列（含特殊标记） |
| **generation_prompt** | `<\|im_start\|>assistant\n`，告诉模型"现在该你回答了"的标记 |
| **Padding** | 把不同长度的样本补齐到相同长度，GPU 需要固定形状张量 |
| **Truncation** | 超过 max_length 的样本被截断，`"right"` 表示截掉右边（代码末尾） |
| **attention_mask** | 标记哪些位置是真实 token（1），哪些是 padding（0），防止模型关注无意义位置 |
| **loss_mask** | 标记哪些位置计算 loss（1=assistant回复），哪些不计算（0=system/user/padding） |
| **Sanity Check** | 逐条 tokenize 的拼接结果 必须= 整体 tokenize 的结果，验证数据处理的正确性 |

---

## 9. 与实现文档的对应

| 本讲内容 | 对应实现文档 |
|----------|-------------|
| BEE 数据解析 | `01_data_preprocessing.md` 第 3-4 节 |
| MultiTurnSFTDataset 选择 | `01_data_preprocessing.md` 第 2 节 |
| Parquet Schema | `01_data_preprocessing.md` 第 5 节 |
| prepare_sft_data.py 规划 | `01_data_preprocessing.md` 第 8 节 |
| 验证清单 | `01_data_preprocessing.md` 第 9 节 |

**下一部分** 将讲解 verl SFT Trainer 的整体架构——从 `torchrun` 命令启动，到 FSDP 分布式训练的初始化流程。

---

> **思考题**：
> 1. 如果我们不去除重复的 system_prompt（跳过 `_process_single_message` 中第 2 步），拼接后的 token 序列会有什么问题？
> 2. 为什么 generation_prompt（`<|im_start|>assistant\n`）的 loss_mask 是 0？模型不需要学会"在 assistant 标记后开始回答"吗？
> 3. 如果 BEE 数据中有一条记录的代码为空（`response = ""`），经过 MultiTurnSFTDataset 处理后 loss_mask 会是什么样？这样的数据应该保留还是过滤？
