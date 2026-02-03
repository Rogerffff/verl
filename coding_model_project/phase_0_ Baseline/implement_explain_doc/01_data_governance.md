# Step 1: 数据治理实施文档 (Data Governance)

## 概述

本文档详细说明 Phase 0 数据治理脚本 (`src/data_governance.py`) 的实现细节和使用方法。

**什么是数据治理？**

在机器学习项目中，数据治理是确保训练数据质量的关键步骤。对于代码生成模型，我们需要：
1. **获取数据**：从 Hugging Face 或 SandboxFusion 获取评测数据集
2. **数据清洗**：去除重复样本，确保数据唯一性
3. **泄漏检查**：防止测试集数据泄漏到训练集
4. **生成元数据**：创建 Manifest 文件，记录每个样本的唯一标识

---

## 1. 功能说明

### 1.1 核心功能

| 功能 | 说明 | 为什么重要 |
|------|------|-----------|
| 数据获取 | 从 Hugging Face 获取完整数据集 | 确保数据完整性和一致性 |
| **测试用例获取** | 同时获取每个问题的测试用例 | 用于评测模型生成的代码 |
| Prompt 规范化 | 统一换行符、去除空白、压缩空行 | 使相同内容的 prompt 产生相同的哈希值 |
| SHA256 哈希 | 对规范化后的 prompt 计算唯一标识 | 用于去重和泄漏检查 |
| Split 内去重 | 同一 split 内基于 hash 去重 | 避免重复样本影响评测 |
| 跨 Split 检查 | 检查 train/valid/test 是否有重叠 | 确保评测公平性 |
| 泄漏检查 | 检查 HumanEval/MBPP 是否泄漏到训练集 | 防止过拟合外部基准 |
| Manifest 生成 | 生成每个数据集的 manifest 文件 | 记录数据版本和元信息 |
| 审计报告 | 生成 Markdown 格式的审计报告 | 方便人工审核 |

### 1.2 支持的数据源

| 数据源 | 命令参数 | 优点 | 缺点 |
|--------|---------|------|------|
| **Hugging Face** | `--source huggingface` | 完整数据、包含测试用例 | 需要网络 |
| SandboxFusion | `--source sandbox` | 与评测环境一致 | 依赖服务运行 |

**推荐使用 Hugging Face 数据源**，因为它包含完整的测试用例。

---

## 2. 文件结构

运行数据治理脚本后，会生成以下文件结构：

```
coding_model_project/
├── src/
│   ├── __init__.py
│   ├── data_governance.py     # 数据治理主脚本
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py         # 指标收集模块
│       └── qa_logger.py       # 问答日志模块
│
├── data/
│   ├── manifests/             # Manifest 文件（元数据）
│   │   ├── humaneval_manifest.jsonl
│   │   ├── mbpp_reg_manifest.jsonl
│   │   ├── codecontests_train_manifest.jsonl
│   │   ├── codecontests_valid_manifest.jsonl
│   │   ├── codecontests_test_manifest.jsonl
│   │   └── *_duplicates_intrasplit.jsonl  # 重复项记录（用于审计）
│   │
│   └── raw/                   # 原始数据（包含测试用例）
│       ├── humaneval_raw.jsonl        # 164 条，含 test_code
│       ├── mbpp_reg_raw.jsonl         # 200 条，含 test_list
│       ├── codecontests_train_raw.jsonl   # ~12,285 条，含 stdin/stdout
│       ├── codecontests_valid_raw.jsonl   # 117 条
│       └── codecontests_test_raw.jsonl    # 165 条
│
└── reports/
    └── data_audit_report.md   # 审计报告
```

---

## 3. 数据集详解

### 3.1 数据集配置

| 数据集键名 | 来源 | Split | 角色 | 数量 | 说明 |
|-----------|------|-------|------|------|------|
| `humaneval` | OpenAI HumanEval | test | test_only | 164 | 经典代码生成基准 |
| `mbpp_reg` | Google MBPP | test | test_only | 200 | ID 11-210，Python 编程题 |
| `codecontests_train` | CodeContests | train | train | ~12,285 | 用于 RL 训练 |
| `codecontests_valid` | CodeContests | valid | validation | 117 | 超参数调优 |
| `codecontests_test` | CodeContests | test | test | 165 | 最终评测 |

### 3.2 最新运行结果（2026-02-02）

```
============================================================
数据治理完成！
============================================================

最终样本数统计:
  humaneval (test_only): 164 条
  mbpp_reg (test_only): 200 条
  codecontests_train (train): 12,285 条  ← 去重后（原 13,328 条，移除 1,043 重复）
  codecontests_valid (validation): 117 条
  codecontests_test (test): 165 条

跨 Split 检查: 全部通过 ✓
外部泄漏检查: 无泄漏 ✓
```

### 3.3 测试用例格式

**新版本的数据治理脚本会同时保存测试用例**，格式如下：

#### HumanEval 测试用例格式

```json
{
  "problem_id": "HumanEval/0",
  "prompt": "from typing import List\n\ndef has_close_elements(...",
  "test_cases": {
    "type": "humaneval",
    "test_code": "\nMETADATA = {...}\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9...]) == True\n    ...",
    "entry_point": "has_close_elements"
  },
  "canonical_solution": "    for idx, elem in enumerate(numbers):\n        ..."
}
```

**说明**：
- `test_code`：包含 `check()` 函数的 Python 代码
- `entry_point`：需要测试的函数名
- 评测时组装：`模型代码 + test_code + check(entry_point)`

#### MBPP 测试用例格式

```json
{
  "problem_id": "11",
  "prompt": "Write a function to remove all characters...",
  "test_cases": {
    "type": "mbpp",
    "test_list": [
      "assert check_k_elements([(4, 4), (4, 4, 4)], 4) == True",
      "assert check_k_elements([(7, 7, 7), (7, 7)], 7) == True",
      "assert check_k_elements([(9, 9), (9, 9, 9, 9)], 7) == False"
    ],
    "test_setup_code": "",
    "challenge_test_list": []
  },
  "canonical_solution": "def check_k_elements(test_tup, K):\n    ..."
}
```

**说明**：
- `test_list`：assert 语句列表
- 评测时组装：`test_setup_code + 模型代码 + test_list.join('\n')`

#### CodeContests 测试用例格式

```json
{
  "problem_id": "codeforces/1234/A",
  "prompt": "Problem description...\n\nInput\n...\nOutput\n...\nExample\n...",
  "test_cases": {
    "type": "codecontests",
    "tests": [
      {"input": "2 3\n1\n5\n6\n", "output": "9\n6\n1\n"},
      {"input": "1 1\n100\n", "output": "100\n"}
    ]
  },
  "solutions": ["# Python solution 1...", "# Python solution 2..."]
}
```

**说明**：
- `tests`：stdin/stdout 测试用例对
- 每个测试用例包含 `input`（标准输入）和 `output`（期望输出）
- 评测时：运行模型代码，传入 stdin，比较 stdout

---

## 4. 使用方法

### 4.1 环境准备

```bash
# 1. 进入项目目录
cd /Users/xiaohui/Desktop/verl/verl/coding_model_project

# 2. 确保 Python 环境中有 datasets 库
# 方式一：使用 miniconda base 环境（已安装 datasets）
/Users/xiaohui/miniconda3/bin/python --version

# 方式二：安装到当前环境
pip install datasets
```

### 4.2 运行数据治理（推荐命令）

```bash
# 从 Hugging Face 获取完整数据（推荐）
/Users/xiaohui/miniconda3/bin/python src/data_governance.py \
    --source huggingface \
    --output_dir data/

# 输出示例：
# [Step 1] 获取数据集...
#   Fetching HumanEval from Hugging Face...
#     Retrieved 164 problems (with test cases)
#   Fetching MBPP from Hugging Face...
#     Retrieved 200 problems (ID 11-210, with test cases)
#   ...
```

### 4.3 常用命令

```bash
# 只获取 HumanEval 和 MBPP（快速测试）
python src/data_governance.py \
    --source huggingface \
    --datasets humaneval mbpp_reg \
    --output_dir data/

# 从已有的 raw 文件加载（跳过网络获取）
python src/data_governance.py \
    --skip_fetch \
    --output_dir data/

# 使用 SandboxFusion 数据源（需要先启动服务）
python src/data_governance.py \
    --source sandbox \
    --endpoint http://localhost:8080 \
    --output_dir data/
```

### 4.4 命令行参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | huggingface | 数据源：`huggingface` 或 `sandbox` |
| `--endpoint` | http://localhost:8080 | SandboxFusion 服务地址（仅 sandbox 模式） |
| `--output_dir` | data/ | 输出目录 |
| `--datasets` | 全部 | 指定数据集，如 `humaneval mbpp_reg` |
| `--skip_fetch` | False | 跳过数据获取，从已有 raw 文件加载 |

---

## 5. 核心实现详解

### 5.1 Prompt 规范化

**为什么需要规范化？**

不同来源的 prompt 可能有细微差异（如换行符、空格），但内容相同。规范化确保相同内容产生相同的哈希值。

```python
def canonicalize_prompt(prompt: str) -> str:
    """
    规范化规则：
    1. \r\n -> \n（统一换行符）
    2. 去除每行行尾空白
    3. 压缩连续空行（保留最多一个）
    4. 去除首尾空白
    """
    # 1. 统一换行符
    text = prompt.replace('\r\n', '\n').replace('\r', '\n')

    # 2. 去除每行行尾空白
    lines = [line.rstrip() for line in text.split('\n')]

    # 3. 压缩连续空行
    compressed_lines = []
    prev_empty = False
    for line in lines:
        is_empty = len(line.strip()) == 0
        if is_empty:
            if not prev_empty:
                compressed_lines.append('')
            prev_empty = True
        else:
            compressed_lines.append(line)
            prev_empty = False

    # 4. 合并并去除首尾空白
    return '\n'.join(compressed_lines).strip()
```

### 5.2 SHA256 哈希计算

```python
import hashlib

def compute_sha256(text: str) -> str:
    """计算文本的 SHA256 哈希值"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
```

**作用**：
- 64 个十六进制字符的唯一标识
- 用于去重和泄漏检查
- 相同内容 → 相同哈希 → 被识别为重复

### 5.3 去重优先级

当不同 split 之间存在重叠时，根据优先级决定保留哪个：

```
优先级从高到低：
1. humaneval       (外部基准，必须保留)
2. mbpp_reg        (外部基准，必须保留)
3. codecontests_test    (测试集，保留)
4. codecontests_valid   (验证集，次优先)
5. codecontests_train   (训练集，最低优先)
```

**例子**：如果某个 prompt 同时出现在 `codecontests_train` 和 `humaneval` 中，从 train 中移除。

### 5.4 数据保存（含测试用例）

新版本的 `save_raw_data()` 函数会同时保存测试用例：

```python
def save_raw_data(dataset_key, problems, output_dir):
    """保存原始数据到 JSONL 文件（支持测试用例）"""

    for item in problems:
        if isinstance(item, dict):
            # 新格式：包含测试用例
            record = {
                "problem_id": item["problem_id"],
                "prompt": item["prompt"],
                "canonical_prompt": canonicalize_prompt(item["prompt"]),
                "prompt_sha256": compute_sha256(...),
                "test_cases": item.get("test_cases", {}),  # 测试用例
            }

            # 可选：参考解答
            if "canonical_solution" in item:
                record["canonical_solution"] = item["canonical_solution"]
```

---

## 6. 输出文件格式

### 6.1 Raw 数据文件 (JSONL) - 包含测试用例

每行一个 JSON 对象：

```json
{
  "problem_id": "HumanEval/0",
  "prompt": "from typing import List\n\ndef has_close_elements...",
  "canonical_prompt": "from typing import List\n\ndef has_close_elements...",
  "prompt_sha256": "797fe463e321b2524e948df205089eab05de3a3ff72d5f42328cfaca1d3952b5",
  "test_cases": {
    "type": "humaneval",
    "test_code": "\nMETADATA = {...}\n\ndef check(candidate):\n    assert ...",
    "entry_point": "has_close_elements"
  },
  "canonical_solution": "    for idx, elem in enumerate(numbers):\n        ..."
}
```

### 6.2 Manifest 文件 (JSONL)

记录元数据（不含测试用例，用于快速索引）：

```json
{
  "dataset": "humaneval_python",
  "split": "test",
  "problem_id": "HumanEval/0",
  "prompt_sha256": "797fe463e321b2524e948df205089eab...",
  "prompt_length": 500,
  "canonical_length": 480,
  "version": "20260202_213748"
}
```

### 6.3 审计报告 (Markdown)

查看 `reports/data_audit_report.md`：

```markdown
# 数据治理审计报告

生成时间: 2026-02-02 21:38:21

## 1. 样本数统计

| 数据集 | 去重前 | 去重后 | Split内重复 | 跨Split移除 |
|--------|--------|--------|-------------|-------------|
| humaneval | 164 | 164 | 0 | 0 |
| mbpp_reg | 200 | 200 | 0 | 0 |
| codecontests_train | 13328 | 12285 | 1043 | 0 |
| codecontests_valid | 117 | 117 | 0 | 0 |
| codecontests_test | 165 | 165 | 0 | 0 |

## 2. 跨 Split 精确重叠检查

**所有 Split 交集均为空 ✓**

## 3. 外部基准泄漏检查

**训练集与外部基准无泄漏 ✓**
```

---

## 7. 验收标准

运行完成后，检查以下条件：

| 检查项 | 预期结果 | 如何验证 |
|--------|----------|----------|
| Raw 文件生成 | 5 个 `*_raw.jsonl` 文件 | `ls data/raw/` |
| 包含测试用例 | 每条记录有 `test_cases` 字段 | `head -1 data/raw/humaneval_raw.jsonl \| python -m json.tool` |
| Manifest 生成 | 5 个 `*_manifest.jsonl` 文件 | `ls data/manifests/` |
| 跨 Split 重叠 | 全部为 0 | 查看审计报告 |
| 外部泄漏 | 无泄漏 | 查看审计报告 |
| 审计报告 | 已生成 | `cat reports/data_audit_report.md` |

**快速验证命令**：

```bash
# 验证测试用例是否保存
head -1 data/raw/humaneval_raw.jsonl | python -c "
import sys, json
d = json.loads(sys.stdin.read())
print('Keys:', list(d.keys()))
print('Test type:', d.get('test_cases', {}).get('type'))
"

# 预期输出：
# Keys: ['problem_id', 'prompt', 'canonical_prompt', 'prompt_sha256', 'test_cases', 'canonical_solution']
# Test type: humaneval
```

---

## 8. 故障排查

### 问题 1：ModuleNotFoundError: No module named 'datasets'

```bash
# 解决方案：安装 datasets 库
pip install datasets

# 或使用已安装 datasets 的环境
/Users/xiaohui/miniconda3/bin/python src/data_governance.py ...
```

### 问题 2：网络超时

```bash
# 解决方案 1：设置代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 解决方案 2：使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题 3：sandbox_fusion not installed

这是警告，不影响使用 Hugging Face 数据源：

```
Warning: sandbox_fusion not installed. Run: pip install sandbox-fusion
```

如需使用 SandboxFusion 数据源：
```bash
pip install sandbox-fusion
```

---

## 9. 后续步骤

完成数据治理后：

1. **检查审计报告**：确认所有检查通过
2. **验证测试用例**：确认 raw 文件包含 `test_cases` 字段
3. **继续 Step 2**：运行评测脚本 `phase0_eval.py`

```bash
# 下一步：使用外部测试用例进行评测
python src/phase0_eval.py \
    --mode simple \
    --vllm_url http://gpu-server:8000 \
    --manifest_dir data/manifests \
    --use_external_tests \
    --datasets humaneval mbpp_reg
```
