# Phase 0: 数据治理与 Manifest 生成指南

---

## 一、数据治理目标

数据治理是 Phase 0 的关键步骤，其目标是：

1. **可审计的数据隔离**：确保 train/valid/test 严格分离
2. **防止泄漏**：避免评测数据污染训练集
3. **可复现性**：通过 manifest 文件确保实验可重复
4. **透明度**：生成审计报告供面试时证明数据完整性

---

## 二、数据集概览

### 2.1 数据集角色定义

| 数据集 | 角色 | 用途 | 来源 |
|--------|------|------|------|
| **CodeContests_train** | Train | 训练/构造偏好对 | deepmind/code_contests |
| **CodeContests_valid** | Dev/Val | 高频回归、早停、选超参 | deepmind/code_contests |
| **CodeContests_test** | Test | 阶段结束评测，禁止训练/调参 | deepmind/code_contests |
| **HumanEval** | Test only | 行业对标基准 | openai_humaneval |
| **MBPP_reg** | Dev/Val | 回归监控（100-200 题子集） | google-research-datasets/mbpp |

### 2.2 数据集规模

| 数据集 | Split | 预估样本数 | 说明 |
|--------|-------|-----------|------|
| CodeContests | train | ~13,000 | 包含多个来源的竞赛题 |
| CodeContests | valid | ~100-200 | 验证集 |
| CodeContests | test | ~100-200 | 测试集 |
| HumanEval | test | 164 | 全量评测 |
| MBPP | test | 500 | 取子集 100-200 题 |

---

## 三、数据获取步骤

### 3.1 从 SandboxFusion SDK 获取（推荐）

使用 SandboxFusion SDK 直接获取数据集是最简单的方式，因为：
- 数据格式已经适配 SandboxFusion 判题
- 可以直接使用 `submit()` API 进行评测
- 无需处理原始测试用例格式

```python
# scripts/download_datasets_from_sandbox.py

from sandbox_fusion import (
    get_prompts,
    GetPromptsRequest,
    TestConfig,
    list_datasets,
)
from pathlib import Path
import json

def download_from_sandbox(output_dir: str = "data/sandbox"):
    """从 SandboxFusion SDK 获取所有需要的数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 获取 CodeContests
    print("Fetching CodeContests from SandboxFusion...")
    for split in ['valid', 'test']:  # Phase 0 不需要 train
        prompts = get_prompts(GetPromptsRequest(
            dataset='code_contests',  # SandboxFusion 中的数据集名
            config=TestConfig(
                language='python',
                locale='en',
                extra={'split': split}  # 指定 split
            ),
            offset=0,
            limit=100000
        ))

        records = []
        for item in prompts.prompts:
            records.append({
                "dataset": "codecontests",
                "split": split,
                "problem_id": item.id,
                "prompt": item.prompt,
                "sandbox_dataset": "code_contests",  # 用于 submit()
                "sandbox_id": item.id,               # 用于 submit()
            })

        save_path = output_path / f"codecontests_{split}.jsonl"
        with open(save_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"  {split}: {len(records)} samples -> {save_path}")

    # 2. 获取 HumanEval
    print("\nFetching HumanEval from SandboxFusion...")
    prompts = get_prompts(GetPromptsRequest(
        dataset='humaneval',
        config=TestConfig(language='python')
    ))

    records = []
    for item in prompts.prompts:
        records.append({
            "dataset": "humaneval",
            "split": "test",
            "problem_id": f"HumanEval/{item.id}",
            "prompt": item.prompt,
            "sandbox_dataset": "humaneval",
            "sandbox_id": item.id,
        })

    save_path = output_path / "humaneval.jsonl"
    with open(save_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"  test: {len(records)} samples -> {save_path}")

    # 3. 获取 MBPP
    print("\nFetching MBPP from SandboxFusion...")
    prompts = get_prompts(GetPromptsRequest(
        dataset='mbpp',
        config=TestConfig(is_fewshot=False)
    ))

    # 选取 ID 11-210 作为回归子集
    records = []
    for item in prompts.prompts:
        try:
            task_id = int(item.id)
            if 11 <= task_id <= 210:  # MBPP_reg 子集
                records.append({
                    "dataset": "mbpp",
                    "split": "test",
                    "problem_id": f"MBPP/{item.id}",
                    "prompt": item.prompt,
                    "sandbox_dataset": "mbpp",
                    "sandbox_id": item.id,
                })
        except ValueError:
            continue

    save_path = output_path / "mbpp_reg.jsonl"
    with open(save_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"  mbpp_reg: {len(records)} samples -> {save_path}")

    print(f"\nAll datasets saved to {output_path}")
    return output_path

if __name__ == "__main__":
    download_from_sandbox()
```

### 3.2 SandboxFusion SDK API 说明

| API | 用途 | 说明 |
|-----|------|------|
| `list_datasets()` | 列出可用数据集 | 返回支持的数据集列表 |
| `get_prompts()` | 获取题目 | 返回 prompt 列表，包含 id 和 prompt |
| `submit()` | 提交评测 | 直接评测代码，无需自己管理测试用例 |

```python
from sandbox_fusion import list_datasets, get_prompts, submit
from sandbox_fusion import GetPromptsRequest, SubmitRequest, TestConfig

# 查看可用数据集
datasets = list_datasets()
print(datasets)  # ['humaneval', 'mbpp', 'code_contests', ...]

# 获取题目
prompts = get_prompts(GetPromptsRequest(
    dataset='humaneval',
    config=TestConfig(language='python')
))

for p in prompts.prompts[:3]:
    print(f"ID: {p.id}")
    print(f"Prompt: {p.prompt[:100]}...")
    print("---")

# 提交评测（无需自己管理测试用例）
result = submit(SubmitRequest(
    dataset='humaneval',
    id=prompts.prompts[0].id,
    completion='def has_close_elements(numbers, threshold):\n    ...',
    config=TestConfig(language='python', run_timeout=10)
))
print(f"Accepted: {result.accepted}")
print(f"Tests: {result.tests}")
```

### 3.3 使用 submit() vs compute_score() 对比

| 方面 | `submit()` (SandboxFusion SDK) | `compute_score()` (verl) |
|------|-------------------------------|-------------------------|
| **测试用例管理** | 由 SandboxFusion 管理 | 需要自己提供 test_cases |
| **简单性** | ✅ 更简单 | 需要处理数据格式 |
| **与 GRPO 一致性** | ❌ GRPO 使用 compute_score | ✅ 完全一致 |
| **灵活性** | 受限于 SandboxFusion 数据 | 可自定义测试用例 |

**推荐**：Phase 0 评测可以使用 `submit()` 简化流程，但需要在文档中说明与 GRPO 训练阶段的差异。

### 3.4 从 HuggingFace 下载（备选）

如果需要原始数据或 SandboxFusion 中缺少某些数据集，可以从 HuggingFace 下载：

```python
from datasets import load_dataset

# CodeContests
codecontests = load_dataset("deepmind/code_contests")

# HumanEval
humaneval = load_dataset("openai_humaneval")

# MBPP
mbpp = load_dataset("mbpp")
```

---

## 四、数据标准化

### 4.1 统一数据格式

使用 SandboxFusion SDK 获取数据后，所有数据集使用统一格式：

```python
@dataclass
class ProblemRecord:
    """统一的问题记录格式（SandboxFusion SDK 版本）"""
    dataset: str           # "codecontests" / "humaneval" / "mbpp"
    split: str             # "train" / "valid" / "test"
    problem_id: str        # 唯一标识
    prompt: str            # 问题描述/提示
    canonical_prompt: str  # 规范化后的 prompt
    prompt_sha256: str     # prompt 的 SHA256 哈希
    # SandboxFusion 评测所需字段
    sandbox_dataset: str   # SandboxFusion 中的数据集名
    sandbox_id: str        # SandboxFusion 中的问题 ID
    metadata: Dict         # 其他元数据
```

> **注意**：由于使用 `submit()` API 评测，不需要在本地存储测试用例，SandboxFusion 会自动管理测试用例。

### 4.2 从 SandboxFusion 数据标准化

Section 3 中的下载脚本已经生成了基本格式，以下是添加规范化处理的完整转换函数：

```python
# scripts/standardize_sandbox_data.py

import json
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import List, Dict

def canonicalize_prompt(prompt: str) -> str:
    """规范化 prompt 文本（详见 Section 5.1）"""
    text = unicodedata.normalize('NFC', prompt)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.strip()
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def compute_hash(text: str) -> str:
    """计算文本的 SHA256 哈希"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def standardize_sandbox_data(input_dir: str, output_dir: str) -> Dict[str, List[Dict]]:
    """
    对 SandboxFusion SDK 下载的数据进行标准化处理

    Args:
        input_dir: Section 3 下载脚本的输出目录 (data/sandbox)
        output_dir: 标准化后的输出目录 (data/standardized)

    Returns:
        all_records: 按数据集名分组的记录字典
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_records = {}

    # 处理所有下载的 JSONL 文件
    for jsonl_file in input_path.glob("*.jsonl"):
        dataset_name = jsonl_file.stem  # e.g., "codecontests_valid", "humaneval"

        records = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                item = json.loads(line)

                # 规范化 prompt
                canonical = canonicalize_prompt(item['prompt'])

                # 创建标准化记录
                record = {
                    "dataset": item['dataset'],
                    "split": item['split'],
                    "problem_id": item['problem_id'],
                    "prompt": item['prompt'],
                    "canonical_prompt": canonical,
                    "prompt_sha256": compute_hash(canonical),
                    "sandbox_dataset": item['sandbox_dataset'],
                    "sandbox_id": item['sandbox_id'],
                    "metadata": {},
                }
                records.append(record)

        all_records[dataset_name] = records

        # 保存标准化后的数据
        save_path = output_path / f"{dataset_name}.jsonl"
        with open(save_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        print(f"Standardized {dataset_name}: {len(records)} records -> {save_path}")

    return all_records

if __name__ == "__main__":
    standardize_sandbox_data("data/sandbox", "data/standardized")
```

### 4.3 使用标准化数据进行评测

标准化后的数据可以直接用于 SandboxFusion 评测：

```python
from sandbox_fusion import submit, SubmitRequest, TestConfig

def evaluate_completion(record: Dict, completion: str) -> Dict:
    """
    使用 SandboxFusion submit() API 评测代码

    Args:
        record: 标准化后的问题记录
        completion: 模型生成的代码

    Returns:
        评测结果
    """
    result = submit(SubmitRequest(
        dataset=record['sandbox_dataset'],
        id=record['sandbox_id'],
        completion=completion,
        config=TestConfig(language='python', run_timeout=10)
    ))

    return {
        "problem_id": record['problem_id'],
        "accepted": result.accepted,
        "tests": result.tests if hasattr(result, 'tests') else None,
    }
```

### 4.4 与 HuggingFace 数据的对比

| 方面 | SandboxFusion SDK | HuggingFace |
|------|-------------------|-------------|
| **测试用例管理** | 由 SandboxFusion 管理 | 需要本地存储和解析 |
| **数据格式** | 统一的 Prompt 对象 | 每个数据集格式不同 |
| **评测方式** | `submit()` 一步完成 | 需要构造执行环境 |
| **去重逻辑** | 相同，基于 prompt hash | 相同 |

---

## 五、规范化与哈希计算

### 5.1 Prompt 规范化

```python
import re
import unicodedata

def canonicalize_prompt(prompt: str) -> str:
    """
    规范化 prompt 文本，用于计算哈希和去重

    规则：
    1. Unicode 规范化 (NFC)
    2. 统一换行符
    3. 去除首尾空白
    4. 多个连续空白压缩（保留换行）
    5. 去除常见噪声（版权声明等）
    """
    # Unicode 规范化
    text = unicodedata.normalize('NFC', prompt)

    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 去除首尾空白
    text = text.strip()

    # 多个空格压缩为单个（保留换行）
    text = re.sub(r'[^\S\n]+', ' ', text)

    # 多个连续换行压缩为两个（段落分隔）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 可选：去除版权/来源声明等噪声
    # text = re.sub(r'Copyright.*?\n', '', text, flags=re.IGNORECASE)

    return text
```

### 5.2 SHA256 计算

```python
import hashlib

def compute_hash(text: str) -> str:
    """计算文本的 SHA256 哈希"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
```

---

## 六、去重与泄漏检查

### 6.1 Split 内去重

```python
from typing import Tuple, Set

def deduplicate_split(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Split 内精确去重

    返回:
        (unique_records, duplicate_records)
    """
    seen_hashes: Set[str] = set()
    unique = []
    duplicates = []

    for record in records:
        h = record['prompt_sha256']
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(record)
        else:
            duplicates.append(record)

    return unique, duplicates
```

### 6.2 跨 Split 重叠检查

```python
def check_cross_split_overlap(
    split_a: List[Dict],
    split_b: List[Dict],
    split_a_name: str,
    split_b_name: str
) -> Dict:
    """
    检查两个 split 之间的重叠

    返回:
        {
            "overlap_count": int,
            "overlap_hashes": List[str],
            "overlap_problems": List[Dict]  # 重叠问题的详细信息
        }
    """
    hashes_a = {r['prompt_sha256']: r for r in split_a}
    hashes_b = {r['prompt_sha256']: r for r in split_b}

    overlap_hashes = set(hashes_a.keys()) & set(hashes_b.keys())

    overlap_problems = []
    for h in overlap_hashes:
        overlap_problems.append({
            f"{split_a_name}_problem_id": hashes_a[h]['problem_id'],
            f"{split_b_name}_problem_id": hashes_b[h]['problem_id'],
            "prompt_sha256": h,
        })

    return {
        "overlap_count": len(overlap_hashes),
        "overlap_hashes": list(overlap_hashes),
        "overlap_problems": overlap_problems,
    }
```

### 6.3 外部泄漏检查

```python
def check_external_leakage(
    train_records: List[Dict],
    external_records: List[Dict],
    external_name: str
) -> Dict:
    """
    检查训练数据与外部评测集的泄漏

    返回:
        {
            "leakage_count": int,
            "leakage_problems": List[Dict]
        }
    """
    train_hashes = {r['prompt_sha256']: r for r in train_records}
    external_hashes = {r['prompt_sha256']: r for r in external_records}

    overlap = set(train_hashes.keys()) & set(external_hashes.keys())

    leakage_problems = []
    for h in overlap:
        leakage_problems.append({
            "train_problem_id": train_hashes[h]['problem_id'],
            f"{external_name}_problem_id": external_hashes[h]['problem_id'],
            "prompt_sha256": h,
        })

    return {
        "leakage_count": len(overlap),
        "leakage_problems": leakage_problems,
    }
```

### 6.4 冲突处理策略

当发现重叠时：

| 冲突类型 | 处理策略 | 原因 |
|---------|---------|------|
| train ∩ valid | 从 train 删除 | 保持验证集完整性 |
| train ∩ test | 从 train 删除 | 保持测试集完整性 |
| valid ∩ test | 从 valid 删除 | 保持测试集完整性 |
| train ∩ HumanEval | 从 train 删除 | 防止对标基准泄漏 |
| train ∩ MBPP_reg | 从 train 删除 | 防止回归监控泄漏 |

```python
def resolve_conflicts(
    records: Dict[str, List[Dict]],
    conflicts: List[Tuple[str, str, List[str]]]
) -> Dict[str, List[Dict]]:
    """
    解决冲突，返回清理后的记录

    Args:
        records: {"train": [...], "valid": [...], "test": [...]}
        conflicts: [(source_split, target_split, overlap_hashes), ...]
    """
    for source, target, hashes in conflicts:
        if not hashes:
            continue

        # 从 source split 删除冲突样本
        hash_set = set(hashes)
        records[source] = [
            r for r in records[source]
            if r['prompt_sha256'] not in hash_set
        ]

        print(f"Removed {len(hashes)} samples from {source} (conflict with {target})")

    return records
```

---

## 七、Manifest 生成

### 7.1 Manifest 格式（SandboxFusion SDK 版本）

每个 manifest 文件是一个 JSONL 文件，每行一条记录。由于使用 `submit()` API 评测，manifest 包含 SandboxFusion 评测所需的字段：

```json
{"dataset": "codecontests", "split": "valid", "problem_id": "cc_valid_001", "prompt_sha256": "a1b2c3...", "sandbox_dataset": "code_contests", "sandbox_id": "001", "version": "2025-01-31"}
{"dataset": "humaneval", "split": "test", "problem_id": "HumanEval/0", "prompt_sha256": "d4e5f6...", "sandbox_dataset": "humaneval", "sandbox_id": "0", "version": "2025-01-31"}
...
```

### 7.2 生成脚本

```python
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

def generate_manifest(
    records: List[Dict],
    output_path: str,
    version: str = None
):
    """
    生成 manifest 文件（SandboxFusion SDK 版本）

    Args:
        records: 问题记录列表
        output_path: 输出文件路径
        version: 版本标识（默认使用当前日期）
    """
    if version is None:
        version = datetime.now().strftime("%Y-%m-%d")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            manifest_entry = {
                "dataset": record['dataset'],
                "split": record['split'],
                "problem_id": record['problem_id'],
                "prompt_sha256": record['prompt_sha256'],
                "prompt_length": len(record['canonical_prompt']),
                # SandboxFusion 评测所需字段
                "sandbox_dataset": record['sandbox_dataset'],
                "sandbox_id": record['sandbox_id'],
                "version": version,
            }
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')

    print(f"Manifest saved to {output_path} ({len(records)} records)")
```

### 7.3 输出文件结构（Phase 0）

```
data_manifests/
├── codecontests_valid.jsonl       # CodeContests 验证集 manifest
├── codecontests_test.jsonl        # CodeContests 测试集 manifest
├── humaneval.jsonl                # HumanEval manifest
├── mbpp_reg.jsonl                 # MBPP 回归子集 manifest
├── duplicates_intrasplit.jsonl    # Split 内重复记录
├── conflicts_resolved.jsonl       # 跨 split 冲突处理记录
└── audit_report.md                # 审计报告
```

---

## 八、审计报告生成

### 8.1 报告模板

```python
def generate_audit_report(
    stats: Dict,
    overlaps: Dict,
    output_path: str
):
    """生成审计报告"""
    report = f"""# 数据治理审计报告

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. 样本统计

| 数据集 | Split | 去重前 | 去重后 | 删除数 |
|--------|-------|--------|--------|--------|
"""

    for dataset, splits in stats.items():
        for split, counts in splits.items():
            report += f"| {dataset} | {split} | {counts['before']} | {counts['after']} | {counts['removed']} |\n"

    report += f"""
---

## 2. 跨 Split 精确重叠检查

| 检查对 | 重叠数 | 状态 | 处理 |
|--------|--------|------|------|
"""

    for check_name, result in overlaps.items():
        status = "✓" if result['count'] == 0 else "✗"
        action = "无需处理" if result['count'] == 0 else result.get('action', '已删除')
        report += f"| {check_name} | {result['count']} | {status} | {action} |\n"

    report += f"""
---

## 3. 外部泄漏检查

| 训练集 | 评测集 | 泄漏数 | 状态 |
|--------|--------|--------|------|
"""

    for check_name, result in overlaps.get('external', {}).items():
        status = "✓" if result['count'] == 0 else "✗ (已处理)"
        report += f"| CodeContests_train | {check_name} | {result['count']} | {status} |\n"

    report += f"""
---

## 4. MBPP_reg 题目列表

选择 MBPP ID {stats.get('mbpp_reg_range', '11-210')} 作为回归监控子集。

共 {stats.get('mbpp_reg_count', 200)} 题。

---

## 5. 版本信息

| 数据集 | 来源 | 版本 |
|--------|------|------|
| CodeContests | deepmind/code_contests | {stats.get('codecontests_version', 'latest')} |
| HumanEval | openai_humaneval | {stats.get('humaneval_version', 'latest')} |
| MBPP | google-research-datasets/mbpp | {stats.get('mbpp_version', 'latest')} |

---

## 6. 结论

"""

    all_ok = all(v['count'] == 0 for v in overlaps.values())
    if all_ok:
        report += "✓ **数据治理通过**：所有检查项均符合要求，无数据泄漏风险。"
    else:
        report += "⚠️ **注意**：发现数据冲突，已按策略处理。详见上述表格。"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Audit report saved to {output_path}")
```

---

## 九、完整执行流程

### 9.1 主脚本

```python
# scripts/data_governance.py

import json
from pathlib import Path
from sandbox_fusion import get_prompts, GetPromptsRequest, TestConfig

# 导入 Section 4 和 Section 5 的工具函数
from standardize_sandbox_data import canonicalize_prompt, compute_hash

def download_from_sandbox() -> dict:
    """
    从 SandboxFusion SDK 下载所有数据集（简化版，详见 Section 3.1）
    返回: {dataset_name: [records]}
    """
    all_records = {}

    # CodeContests (valid + test)
    for split in ['valid', 'test']:
        prompts = get_prompts(GetPromptsRequest(
            dataset='code_contests',
            config=TestConfig(language='python', locale='en', extra={'split': split}),
            offset=0, limit=100000
        ))
        records = []
        for item in prompts.prompts:
            canonical = canonicalize_prompt(item.prompt)
            records.append({
                "dataset": "codecontests",
                "split": split,
                "problem_id": item.id,
                "prompt": item.prompt,
                "canonical_prompt": canonical,
                "prompt_sha256": compute_hash(canonical),
                "sandbox_dataset": "code_contests",
                "sandbox_id": item.id,
            })
        all_records[f"codecontests_{split}"] = records

    # HumanEval
    prompts = get_prompts(GetPromptsRequest(
        dataset='humaneval',
        config=TestConfig(language='python')
    ))
    records = []
    for item in prompts.prompts:
        canonical = canonicalize_prompt(item.prompt)
        records.append({
            "dataset": "humaneval",
            "split": "test",
            "problem_id": f"HumanEval/{item.id}",
            "prompt": item.prompt,
            "canonical_prompt": canonical,
            "prompt_sha256": compute_hash(canonical),
            "sandbox_dataset": "humaneval",
            "sandbox_id": item.id,
        })
    all_records['humaneval'] = records

    # MBPP (回归子集 ID 11-210)
    prompts = get_prompts(GetPromptsRequest(
        dataset='mbpp',
        config=TestConfig(is_fewshot=False)
    ))
    records = []
    for item in prompts.prompts:
        try:
            task_id = int(item.id)
            if 11 <= task_id <= 210:
                canonical = canonicalize_prompt(item.prompt)
                records.append({
                    "dataset": "mbpp",
                    "split": "test",
                    "problem_id": f"MBPP/{item.id}",
                    "prompt": item.prompt,
                    "canonical_prompt": canonical,
                    "prompt_sha256": compute_hash(canonical),
                    "sandbox_dataset": "mbpp",
                    "sandbox_id": item.id,
                })
        except ValueError:
            continue
    all_records['mbpp_reg'] = records

    return all_records

def main():
    """数据治理主流程"""
    print("=" * 60)
    print("RLVR Coding Model - 数据治理 (SandboxFusion SDK)")
    print("=" * 60)

    # 1. 从 SandboxFusion SDK 下载数据
    print("\n[Step 1] 从 SandboxFusion SDK 加载数据集...")
    all_records = download_from_sandbox()

    for name, records in all_records.items():
        print(f"  {name}: {len(records)} records")

    # 2. Split 内去重
    print("\n[Step 2] Split 内去重...")
    stats = {}
    all_duplicates = []

    for name, records in all_records.items():
        before = len(records)
        unique, dups = deduplicate_split(records)
        all_records[name] = unique
        after = len(unique)

        stats[name] = {'before': before, 'after': after, 'removed': before - after}
        all_duplicates.extend(dups)

        if dups:
            print(f"  {name}: {before} -> {after} (removed {len(dups)})")

    # 3. 跨 split 检查
    # 注意：Phase 0 只下载 valid/test，不下载 train
    # 如果后续需要 train，可以扩展 download_from_sandbox()
    print("\n[Step 3] 跨 Split 重叠检查...")
    overlaps = {}

    checks = [
        ('codecontests_valid', 'codecontests_test'),
        ('codecontests_valid', 'humaneval'),
        ('codecontests_valid', 'mbpp_reg'),
        ('codecontests_test', 'humaneval'),
        ('codecontests_test', 'mbpp_reg'),
    ]

    for split_a, split_b in checks:
        if split_a not in all_records or split_b not in all_records:
            continue

        result = check_cross_split_overlap(
            all_records[split_a],
            all_records[split_b],
            split_a, split_b
        )
        key = f"{split_a} ∩ {split_b}"
        overlaps[key] = {'count': result['overlap_count']}
        print(f"  {key}: {result['overlap_count']} overlaps")

        # 处理冲突：从 valid 删除（保护 test 集完整性）
        if result['overlap_count'] > 0 and 'valid' in split_a:
            hash_set = set(result['overlap_hashes'])
            all_records[split_a] = [
                r for r in all_records[split_a]
                if r['prompt_sha256'] not in hash_set
            ]
            overlaps[key]['action'] = f"从 {split_a} 删除"

    # 4. 跨数据集重叠检查（用于报告）
    print("\n[Step 4] 跨数据集重叠检查...")
    cross_dataset_checks = [
        ('humaneval', 'mbpp_reg'),
    ]

    for split_a, split_b in cross_dataset_checks:
        result = check_cross_split_overlap(
            all_records[split_a],
            all_records[split_b],
            split_a, split_b
        )
        key = f"{split_a} ∩ {split_b}"
        overlaps[key] = {'count': result['overlap_count']}
        print(f"  {key}: {result['overlap_count']} overlaps")

    # 5. 生成 manifest
    print("\n[Step 5] 生成 Manifest...")
    manifest_dir = Path("data_manifests")
    manifest_dir.mkdir(exist_ok=True)

    for name, records in all_records.items():
        generate_manifest(records, manifest_dir / f"{name}.jsonl")

    # 保存重复记录
    if all_duplicates:
        with open(manifest_dir / "duplicates_intrasplit.jsonl", 'w') as f:
            for d in all_duplicates:
                f.write(json.dumps({
                    'problem_id': d['problem_id'],
                    'prompt_sha256': d['prompt_sha256'],
                }, ensure_ascii=False) + '\n')

    # 6. 生成审计报告
    print("\n[Step 6] 生成审计报告...")
    generate_audit_report(
        stats=stats,
        overlaps=overlaps,
        output_path=manifest_dir / "audit_report.md"
    )

    print("\n" + "=" * 60)
    print("数据治理完成 (SandboxFusion SDK)！")
    print(f"输出目录: {manifest_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## 十、验收清单

### 10.1 必须产出（Phase 0）

> **注意**：Phase 0 只需要评测数据，不需要训练数据。如后续阶段需要 train split，可扩展 `download_from_sandbox()` 函数。

- [ ] `data_manifests/codecontests_valid.jsonl`
- [ ] `data_manifests/codecontests_test.jsonl`
- [ ] `data_manifests/humaneval.jsonl`
- [ ] `data_manifests/mbpp_reg.jsonl`
- [ ] `data_manifests/audit_report.md`

### 10.2 验收标准

| 检查项 | 要求 | 验证方法 |
|--------|------|---------|
| valid ∩ test = ∅ | 0 重叠 | 审计报告 |
| valid ∩ HumanEval = ∅ | 0 重叠 | 审计报告 |
| valid ∩ MBPP_reg = ∅ | 0 重叠 | 审计报告 |
| HumanEval ∩ MBPP_reg = ∅ | 0 重叠 | 审计报告 |
| MBPP_reg 固定 | 200 题 | manifest 行数 |
| 每条记录包含 sandbox_dataset + sandbox_id | 可评测 | manifest 字段 |

### 10.3 面试时展示要点

1. **展示审计报告**：证明数据隔离的严谨性
2. **解释数据来源**：使用 SandboxFusion SDK 获取数据，与评测系统一致
3. **说明 manifest 作用**：确保可复现性
4. **展示评测流程**：使用 `submit()` API 直接评测，无需自己管理测试用例

### 10.4 后续阶段扩展

当进入 GRPO 训练阶段时，需要扩展数据治理：

1. **添加 codecontests_train**：在 `download_from_sandbox()` 中添加 train split 下载
2. **train ∩ test 检查**：确保训练数据不包含测试数据
3. **train ∩ HumanEval/MBPP 检查**：确保不泄漏外部评测数据

---

*文档版本：v2.0 (SandboxFusion SDK 版本)*
*创建日期：2024-01-31*
*更新日期：2025-01-31*
