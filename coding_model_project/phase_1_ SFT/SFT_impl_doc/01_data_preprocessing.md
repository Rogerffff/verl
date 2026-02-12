# Phase 1 SFT - Step 1: Data Preprocessing

> BEE CodeContests JSONL → verl Multi-turn Parquet

---

## 1. Overview

将 BEE-spoke-data/code_contests_instruct (hq-python-deduped) 的 JSONL 训练数据转换为 verl `MultiTurnSFTDataset` 所需的 Parquet 格式。

**输入**: `phase_1_ SFT/bee_hq_python_deduped_filtered/<FINAL_SFT_DATASET>.jsonl`（最终过滤后由用户稍后提供）
**输出**: `phase_1_ SFT/data/sft_train.parquet`, `phase_1_ SFT/data/sft_val.parquet`
**脚本**: `phase_1_ SFT/prepare_sft_data.py`

---

## 2. 为什么选择 Multi-turn 格式

verl 提供两种 SFT 数据集类：

| 类 | 文件 | 格式 | 特点 |
|---|------|------|------|
| `SFTDataset` | `verl/utils/dataset/sft_dataset.py` | 单轮 (prompt/response 列) | 不支持 system prompt |
| `MultiTurnSFTDataset` | `verl/utils/dataset/multiturn_sft_dataset.py` | 多轮 (messages 列) | 支持 system prompt，标准 OpenAI 格式 |

**选择 MultiTurnSFTDataset 的理由**：

1. **System Prompt 支持**: Phase 0 评测使用了 system prompt（见下文），SFT 训练应保持一致，且 Phase 3 GRPO 也需要 system prompt
2. **标准格式**: OpenAI messages 格式 (`[{"role": ..., "content": ...}]`) 是行业标准
3. **Chat Template 自动应用**: `MultiTurnSFTDataset` 会调用 tokenizer 的 `apply_chat_template()`，自动处理 Qwen 模型的特殊 token（`<|im_start|>`, `<|im_end|>` 等）
4. **Loss Mask 自动处理**: 只在 assistant 回复上计算 loss，system 和 user 消息被自动 mask 掉（`multiturn_sft_dataset.py:211-216`）

### verl 数据集选择逻辑

在 `fsdp_sft_trainer.py:851-868` 的 `create_sft_dataset()` 中：

```python
def create_sft_dataset(data_paths, data_config, tokenizer, max_samples=-1):
    if data_config.custom_cls.get("path", None):
        dataset_cls = load_extern_object(...)         # 自定义类优先
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset             # ← 我们使用这个
    else:
        dataset_cls = SFTDataset                      # 默认单轮
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config, ...)
    return dataset
```

**Hydra 配置开关**：`data.multiturn.enable=true`

---

## 3. BEE 原始数据格式

每条 JSONL 记录的关键字段：

```json
{
    "name": "1283_D. Christmas Trees",
    "source": 2,
    "difficulty": 10,
    "language": "PYTHON3",
    "text": "### Prompt\n\n[问题描述，包含 Input/Output/Examples]\n\n### Response\n\n```python3\n[解题代码]\n```"
}
```

### `text` 字段结构

```
### Prompt

In Python3, your task is to solve the following problem:

[完整题目描述，包含：]
- 题目背景和约束条件
- Input 格式说明
- Output 格式说明
- Examples（输入/输出样例）

### Response

```python3
[Python3 解题代码]
```
```

### 关键观察

1. **分隔符**: `### Response` 将题目和答案分开
2. **代码块**: 答案在 ` ```python3 ... ``` ` 代码围栏中
3. **source 字段**: `2` = Codeforces, `6` = AtCoder, 等等
4. **语言**: 全部为 `PYTHON3`（因为 hq-python-deduped 配置已过滤）

---

## 4. 解析逻辑

### 4.1 核心解析函数

```python
import re

def parse_bee_text(text: str) -> tuple[str, str]:
    """
    从 BEE text 字段解析出 prompt（题目描述）和 response（解题代码）。

    Returns:
        (prompt, response) 元组
    """
    # Step 1: 按 "### Response" 分割
    parts = text.split("### Response", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot split text by '### Response': {text[:100]}...")

    # Step 2: 提取 prompt（去除 "### Prompt\n" 前缀）
    prompt = parts[0].replace("### Prompt\n", "").strip()

    # Step 3: 从 Response 部分提取代码
    response_raw = parts[1].strip()
    # 匹配 ```python3 或 ```python 代码块
    code_match = re.search(r'```(?:python3?)\s*\n(.*?)```', response_raw, re.DOTALL)
    if code_match:
        response = code_match.group(1).strip()
    else:
        # 降级：如果没有代码围栏，取整个 Response 内容
        response = response_raw.strip()

    return prompt, response
```

### 4.2 System Prompt

与 Phase 0 评测保持一致，使用 `coding_model_project/src/phase0_eval.py:198-205` 定义的 SYSTEM_PROMPT：

```python
SYSTEM_PROMPT = """You are an expert Python programmer.

Output rules:
1. Output Python code only.
2. Include necessary imports only if needed.
3. Wrap the entire code in <code> and </code>.
4. Do not write anything outside the <code> tags.
5. Follow dataset-specific constraints given by the user prompt (function-only vs full program)."""
```

> **注意**: 这个 system prompt 在训练和评测之间保持一致非常重要。它定义了模型的角色和输出格式约束。

### 4.3 转换为 Messages 格式

```python
def bee_record_to_messages(record: dict) -> list[dict]:
    """
    将一条 BEE 记录转换为 OpenAI messages 格式。

    Args:
        record: BEE JSONL 的一条记录（包含 text, name, source 等字段）

    Returns:
        messages 列表，包含 system/user/assistant 三条消息
    """
    prompt, response = parse_bee_text(record["text"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    return messages
```

---

## 5. 目标 Parquet Schema

### 5.1 Schema 定义

Parquet 文件只需要一列 `messages`，其中每行是一个 messages 列表：

```python
{
    "messages": [
        {"role": "system", "content": "You are an expert Python programmer..."},
        {"role": "user", "content": "In Python3, your task is to solve..."},
        {"role": "assistant", "content": "import sys\nfrom collections import deque\n..."}
    ]
}
```

### 5.2 verl MultiTurnSFTDataset 如何消费此格式

`MultiTurnSFTDataset.__init__()` (multiturn_sft_dataset.py:68-111):

1. 读取 parquet: `pd.read_parquet(parquet_file)` (line 128)
2. 提取 messages 列: `self.dataframe[self.messages_key]` (line 146)
   - `messages_key` 默认为 `"messages"`，通过 `data.multiturn.messages_key` 配置

`MultiTurnSFTDataset.__getitem__()` (multiturn_sft_dataset.py:266-386):

1. 获取该行的 messages: `row_dict = self.dataframe.iloc[item].to_dict()` → `self._build_messages(row_dict)` (line 267-268)
2. 对每条 message 分别 tokenize: `self._process_single_message(index=i, message=message, ...)` (line 275-280)
3. **Loss mask 逻辑** (line 211-216):
   - `role == "assistant"`: `loss_mask = 1`（计算 loss），但 generation prompt token 被 mask 掉
   - `role != "assistant"` (system/user): `loss_mask = 0`（不计算 loss）
4. 拼接所有 message 的 input_ids, loss_mask, attention_mask (line 287-292)
5. Sanity check: 验证分别 tokenize 后拼接的结果等于整体 tokenize 的结果 (line 293, 388-419)
6. 处理 padding/truncation (line 330-386)

### 5.3 参考: verl 官方 Multi-turn 数据预处理示例

参见 `examples/data_preprocess/gsm8k_multiturn_sft.py`，pattern 如下：

```python
def process_fn(example, idx):
    data = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_raw},
        ],
    }
    return data

train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
```

我们的格式多了一个 system message，但 `MultiTurnSFTDataset` 完全支持。

---

## 6. 数据过滤要求

### 6.1 当前状态

- 候选数据与统计见 `phase_1_ SFT/bee_hq_python_deduped_filtered/DATA_README.md`
- 最终用于 SFT 的 JSONL 仍在外部流程中生成，本计划不预先固化为某个文件（例如 `train.jsonl`）

### 6.2 计划边界（本文件）

- 验证/替换流程由用户单独实施，不写入本计划
- 本计划只约定：`prepare_sft_data.py` 消费“最终过滤完成”的 JSONL
- 最终文件路径由用户后续给出后再填充命令

### 6.3 `prepare_sft_data.py` 输入约束

- 不硬编码输入文件名（不能假设一定是 `train.jsonl`）
- 不硬编码样本数量
- 支持从最终 JSONL 中读取 `prompt/solution`（若缺失再回退解析 `text` 字段）
- 输出处理统计（总数、解析失败、空解答、train/val 数）

---

## 7. Validation Split 策略

### 7.1 verl 内置 val_loss（最终 SFT JSONL 的 2% 随机划分）

```
训练数据 (最终过滤完成的 SFT JSONL)
    │
    ├── 98% → sft_train.parquet   (用于 SFT 训练)
    └──  2% → sft_val.parquet     (用于 verl 内置 validation loss 计算)
```

- 目的: 在训练过程中监控 val_loss 趋势，检测过拟合
- 频率: 每 500 步计算一次（由 `trainer.test_freq=500` 控制）
- 注意: 这只是 loss 指标，不涉及代码生成和执行评测

### 7.2 真正的评测（三层 Eval Pipeline）

真正的模型能力评测通过独立的 Checkpoint Eval Pipeline 完成（详见 `04_checkpoint_eval_pipeline.md`）：

| 层级 | 数据集 | 大小 | 频率 |
|------|--------|------|------|
| Tier 1 | codecontests_valid | 117 题 | 每 500 步 |
| Tier 2 | codecontests_valid_big | 500 题 | 每 2000 步 |
| Tier 3 | codecontests_test + HumanEval | 165 + 164 题 | 仅在 Phase 结束时 |

### 7.3 随机划分实现

```python
import random
random.seed(42)  # 固定种子，确保可复现

# 假设 all_records 是过滤后的所有 BEE 记录列表
indices = list(range(len(all_records)))
random.shuffle(indices)

val_size = int(len(all_records) * 0.02)
val_indices = set(indices[:val_size])
train_indices = set(indices[val_size:])
```

---

## 8. 脚本实现规划: `prepare_sft_data.py`

### 8.1 命令行接口

```bash
PHASE1_DIR="coding_model_project/phase_1_ SFT"
python "$PHASE1_DIR/prepare_sft_data.py" \
    --input_jsonl "$PHASE1_DIR/bee_hq_python_deduped_filtered/<FINAL_SFT_DATASET>.jsonl" \
    --output_dir "$PHASE1_DIR/data" \
    --val_ratio 0.02 \
    --seed 42
```

### 8.2 处理流程

```
1. 读取最终过滤完成的 JSONL（路径由用户后续提供）
2. 逐行处理:
   a. 优先读取 `prompt` + `solution`
   b. 若缺失则回退到 `text` 字段并调用 `parse_bee_text()`
   c. 构造 messages 列表
   d. 过滤空解答/解析失败样本
3. 随机划分 train/val (98%/2%)
4. 保存为 Parquet:
   - pd.DataFrame({"messages": train_messages}).to_parquet("sft_train.parquet")
   - pd.DataFrame({"messages": val_messages}).to_parquet("sft_val.parquet")
5. 输出统计报告
```

### 8.3 统计报告

```json
{
    "input_file": "path/to/final_sft_dataset.jsonl",
    "input_total": "count",
    "parse_errors": "count",
    "empty_solution_dropped": "count",
    "final_total": "count",
    "train_samples": "count",
    "val_samples": "count",
    "avg_prompt_length_chars": "N",
    "avg_response_length_chars": "N",
    "max_prompt_length_chars": "N",
    "max_response_length_chars": "N"
}
```

---

## 9. 验证检查清单

### 9.1 Parquet 文件验证

```python
import pandas as pd

df = pd.read_parquet("coding_model_project/phase_1_ SFT/data/sft_train.parquet")
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")  # 应该只有 ['messages']
print(f"First messages: {df['messages'].iloc[0]}")  # 检查格式

# 检查所有 messages 都是 3 条（system, user, assistant）
msg_lengths = df['messages'].apply(len)
print(f"Message count distribution: {msg_lengths.value_counts()}")
assert (msg_lengths == 3).all(), "All records should have exactly 3 messages"

# 检查角色顺序
for i in range(min(10, len(df))):
    msgs = df['messages'].iloc[i]
    assert msgs[0]['role'] == 'system'
    assert msgs[1]['role'] == 'user'
    assert msgs[2]['role'] == 'assistant'
    assert len(msgs[2]['content']) > 0, "Assistant response should not be empty"
```

### 9.2 Tokenizer 兼容性验证

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# 测试 chat template 能正确处理 messages
sample_messages = df['messages'].iloc[0]
tokens = tokenizer.apply_chat_template(
    sample_messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
    return_tensors="pt"
)
print(f"Token count: {tokens['input_ids'].shape}")
print(f"Decoded: {tokenizer.decode(tokens['input_ids'][0][:50])}")
```

### 9.3 verl MultiTurnSFTDataset 加载验证

```python
from omegaconf import OmegaConf
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

config = OmegaConf.create({
    "max_length": 4096,
    "truncation": "right",
    "pad_mode": "right",
    "messages_key": "messages",
})

dataset = MultiTurnSFTDataset(
    parquet_files="coding_model_project/phase_1_ SFT/data/sft_train.parquet",
    tokenizer=tokenizer,
    config=config,
)
print(f"Dataset length: {len(dataset)}")

# 取一个样本验证
sample = dataset[0]
print(f"Keys: {sample.keys()}")        # input_ids, attention_mask, position_ids, loss_mask
print(f"input_ids shape: {sample['input_ids'].shape}")
print(f"loss_mask sum: {sample['loss_mask'].sum()}")  # 应该 > 0（有 assistant 回复）
```

---

## 10. 潜在问题与应对

| 问题 | 应对方案 |
|------|---------|
| BEE text 格式不一致（缺少 `### Response`） | 记录并跳过，报告 parse_errors 数量 |
| 代码块不是 `python3`/`python`（如 `py`） | 扩展正则表达式匹配模式 |
| Sanity check 失败（分别 tokenize ≠ 整体 tokenize） | 设置 `data.multiturn.ignore_input_ids_mismatch=true`，或检查 tokenizer 版本 |
| Parquet 中 messages 列存储为嵌套结构 | `MultiTurnSFTDataset` 内部使用 `convert_nested_value_to_list_recursive()` 处理 numpy 数组到 list 的转换（line 43-53） |
| 样本过长（>4096 tokens） | 由 `data.truncation=right` 处理，配合 `data.max_length=4096` |
