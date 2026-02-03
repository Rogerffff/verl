# Step 2: 评测核心脚本实施文档 (Phase 0 Evaluation)

## 概述

本文档详细说明 Phase 0 评测脚本 (`src/phase0_eval.py`) 的实现细节和使用方法。

**什么是 Phase 0 评测？**

Phase 0 是基线评测阶段，目的是：
1. 在 RL 训练之前，测量模型的初始性能
2. 建立基准线，用于后续对比
3. 验证整个评测流程是否正确

---

## 1. 架构设计

### 1.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Phase 0 评测流程                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ 1. 加载数据   │ →  │ 2. 生成代码   │ →  │ 3. 评测代码   │                  │
│  │              │    │              │    │              │                  │
│  │ - manifest   │    │ - vLLM 推理  │    │ - 运行测试    │                  │
│  │ - raw data   │    │ - OpenAI API │    │ - 比较输出    │                  │
│  │ - test cases │    │              │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                    4. 输出结果                             │              │
│  │                                                          │              │
│  │  - metrics.json (指标汇总)                                │              │
│  │  - summary.json (详细统计)                                │              │
│  │  - qa_logs/ (问答日志)                                    │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 两种运行模式

| 模式 | 命令参数 | 说明 | 适用场景 |
|------|---------|------|---------|
| **verl 模式** | `--mode verl` | 使用 verl 框架启动 vLLM | GPU 服务器 |
| **simple 模式** | `--mode simple` | 连接已有的 vLLM 服务 | 本地测试 |

**初学者建议**：先使用 `simple` 模式熟悉流程，再尝试 `verl` 模式。

### 1.3 两种评测方式

| 方式 | 命令参数 | 说明 | 数据来源 |
|------|---------|------|---------|
| **外部测试用例** | `--use_external_tests` | 使用本地 raw 文件中的测试用例 | Hugging Face |
| **submit API** | `--use_submit_api` | 使用 SandboxFusion 内置数据 | SandboxFusion |

**推荐使用外部测试用例**，因为：
1. 数据完整性有保证
2. 不依赖 SandboxFusion 的内置数据
3. 测试用例格式统一

---

## 2. 文件结构

```
src/
├── phase0_eval.py           # 主评测脚本 (~970 行)
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # 指标收集模块
│   │   ├── MetricsCollector  # 收集评测指标
│   │   └── EvalResult        # 单个问题的评测结果
│   │
│   └── qa_logger.py         # 问答日志模块
│       └── QALogger          # 记录问题、模型输出、评测结果
│
└── config/
    └── phase0_config.yaml   # 配置文件（可选）

data/
├── manifests/               # Manifest 文件
│   └── *.jsonl
└── raw/                     # 原始数据（含测试用例）
    └── *.jsonl

outputs/phase0/              # 输出目录
├── metrics.json             # 汇总指标
├── summary.json             # 详细统计
└── qa_logs/
    ├── humaneval_qa.jsonl   # 问答日志
    └── qa_summary.json
```

---

## 3. 核心实现详解

### 3.1 配置类 (EvalConfig)

```python
@dataclass
class EvalConfig:
    """评测配置"""

    # 运行模式
    mode: str = "simple"          # "verl" 或 "simple"

    # 模型配置
    model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # 简化模式配置
    vllm_url: str = "http://localhost:8000"

    # 解码参数 (EVAL@1 协议：贪婪解码)
    temperature: float = 0.0      # 0.0 = 确定性输出
    top_p: float = 1.0
    max_new_tokens: int = 2048

    # SandboxFusion 配置
    sandbox_url: str = "http://localhost:8080"
    run_timeout: int = 10         # 代码执行超时（秒）

    # 评测方式
    use_submit_api: bool = True   # 使用 SandboxFusion submit API
    use_external_tests: bool = True  # 使用外部测试用例（推荐）

    # 数据配置
    datasets: List[str] = ["humaneval", "mbpp_reg"]
    manifest_dir: Optional[str] = None  # Manifest 目录

    # 输出
    output_dir: str = "outputs/phase0"
    qa_sample_size: int = 50      # 每个数据集保存的 QA 样本数
```

### 3.2 代码提取函数

模型生成的回复可能包含解释文字，需要提取纯代码：

```python
def _extract_code_from_completion(completion: str) -> str:
    """
    从模型输出中提取代码

    提取优先级：
    1. <code>...</code> 标签（推荐格式）
    2. ```python ... ``` markdown 代码块
    3. 原始内容（如果没有标签）
    """
    import re

    # 1. 优先尝试 <code>...</code> 标签
    code_tag_pattern = r'<code>(.*?)</code>'
    matches = re.findall(code_tag_pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()

    # 2. 尝试 markdown 代码块
    md_pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(md_pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()

    # 3. 返回原始内容
    return completion.strip()
```

**重要**：Prompt 设计应该引导模型将代码包裹在 `<code>` 标签中。

### 3.3 评测函数（使用外部测试用例）

#### HumanEval 评测

```python
def _evaluate_humaneval(completion, test_cases, problem_id, config, start_time):
    """评测 HumanEval 格式的测试用例"""

    test_code = test_cases.get("test_code", "")
    entry_point = test_cases.get("entry_point", "")

    # 1. 提取代码
    code = _extract_code_from_completion(completion)

    # 2. 组装完整测试代码
    #    格式: 模型代码 + 测试代码 + 调用 check 函数
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"

    # 3. 执行代码
    result = run_code(RunCodeRequest(
        code=full_code,
        language="python",
        run_timeout=config.run_timeout,
    ))

    # 4. 判断结果
    status = getattr(result, 'status', 'unknown')
    accepted = (status == "success" or status == "Finished")

    return EvalResult(
        problem_id=problem_id,
        accepted=accepted,
        pass_ratio=1.0 if accepted else 0.0,
        ...
    )
```

#### MBPP 评测

```python
def _evaluate_mbpp(completion, test_cases, problem_id, config, start_time):
    """评测 MBPP 格式的测试用例"""

    test_list = test_cases.get("test_list", [])
    test_setup_code = test_cases.get("test_setup_code", "")

    # 1. 提取代码
    code = _extract_code_from_completion(completion)

    # 2. 组装完整测试代码
    #    格式: setup_code + 模型代码 + assert 语句
    test_code = "\n".join(test_list)
    full_code = f"{test_setup_code}\n\n{code}\n\n{test_code}"

    # 3. 执行代码
    result = run_code(RunCodeRequest(
        code=full_code,
        language="python",
        run_timeout=config.run_timeout,
    ))

    # 4. 判断结果
    status = getattr(result, 'status', 'unknown')
    accepted = (status == "success" or status == "Finished")

    return EvalResult(...)
```

#### CodeContests 评测

```python
def _evaluate_codecontests(completion, test_cases, problem_id, config, start_time):
    """评测 CodeContests 格式的测试用例（stdin/stdout）"""

    tests = test_cases.get("tests", [])

    # 1. 提取代码
    code = _extract_code_from_completion(completion)

    passed = 0
    total = len(tests)

    # 2. 逐个执行测试用例
    for tc in tests:
        stdin_input = tc.get("input", "")
        expected_output = tc.get("output", "").strip()

        # 执行代码，传入 stdin
        result = run_code(RunCodeRequest(
            code=code,
            language="python",
            run_timeout=config.run_timeout,
            stdin=stdin_input,  # 标准输入
        ))

        # 比较输出
        actual_output = str(getattr(result, 'run_result', '')).strip()
        if actual_output == expected_output:
            passed += 1

    # 3. 计算通过率
    pass_ratio = passed / total if total > 0 else 0.0
    accepted = (passed == total)

    return EvalResult(
        problem_id=problem_id,
        accepted=accepted,
        pass_ratio=pass_ratio,
        ...
    )
```

### 3.4 数据加载（包含测试用例）

```python
def _load_from_manifest(dataset_key, manifest_dir):
    """从 manifest 加载数据（包括测试用例）"""

    manifest_path = Path(manifest_dir) / f"{dataset_key}_manifest.jsonl"
    raw_path = Path(manifest_dir).parent / "raw" / f"{dataset_key}_raw.jsonl"

    # 1. 加载 manifest 中的 problem_id 列表
    problem_ids = set()
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            problem_ids.add(entry["problem_id"])

    # 2. 从 raw 文件加载数据（包括测试用例）
    result = []
    with open(raw_path) as f:
        for line in f:
            record = json.loads(line)
            if record["problem_id"] in problem_ids:
                item = {
                    "problem_id": record["problem_id"],
                    "prompt": record["prompt"],
                    "sandbox_dataset": sandbox_dataset,
                }

                # 加载测试用例（关键！）
                if "test_cases" in record:
                    item["test_cases"] = record["test_cases"]

                result.append(item)

    return result
```

---

## 4. 使用方法

### 4.1 环境准备

```bash
# 1. 确保已完成数据治理
ls data/raw/*.jsonl
# 应该看到: humaneval_raw.jsonl, mbpp_reg_raw.jsonl, ...

# 2. 确保 SandboxFusion 服务运行（用于代码执行）
docker run -d --rm --privileged -p 8080:8080 \
    volcengine/sandbox-fusion:server-20250609

# 3. 验证服务
curl http://localhost:8080/health
# 应该返回: {"status": "ok"}
```

### 4.2 基本使用（推荐）

```bash
# 使用外部测试用例评测（推荐）
python src/phase0_eval.py \
    --mode simple \
    --vllm_url http://gpu-server:8000 \
    --sandbox_url http://localhost:8080 \
    --manifest_dir data/manifests \
    --use_external_tests \
    --datasets humaneval mbpp_reg \
    --output_dir outputs/phase0
```

### 4.3 命令行参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **运行模式** | | |
| `--mode` | simple | 运行模式：`verl` (GPU服务器) 或 `simple` (本地测试) |
| **模型配置** | | |
| `--model` | Qwen/Qwen2.5-Coder-7B-Instruct | 模型路径或名称 |
| `--vllm_url` | http://localhost:8000 | vLLM 服务地址（simple 模式） |
| **生成配置** | | |
| `--temperature` | 0.0 | 采样温度（0.0 = 贪婪解码） |
| `--max_tokens` | 2048 | 最大生成长度 |
| **评测配置** | | |
| `--sandbox_url` | http://localhost:8080 | SandboxFusion 服务地址 |
| `--run_timeout` | 10 | 代码执行超时（秒） |
| `--use_external_tests` | True | 使用外部测试用例 |
| `--use_submit_api` | False | 使用 SandboxFusion submit API |
| **数据配置** | | |
| `--datasets` | humaneval mbpp_reg | 要评测的数据集 |
| `--manifest_dir` | None | Manifest 目录路径 |
| **输出配置** | | |
| `--output_dir` | outputs/phase0 | 输出目录 |
| `--qa_sample_size` | 50 | 每个数据集保存的 QA 样本数 |
| **verl 模式专用** | | |
| `--rollout` | vllm | Rollout 引擎：vllm 或 sglang |
| `--tensor_parallel_size` | 2 | Tensor Parallel 大小 |
| `--n_gpus` | 8 | GPU 数量 |
| `--gpu_memory_utilization` | 0.85 | GPU 显存利用率 |

### 4.4 verl 模式（GPU 服务器）

```bash
# 在 GPU 服务器上运行（需要安装 verl）
python src/phase0_eval.py \
    --mode verl \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --rollout vllm \
    --tensor_parallel_size 2 \
    --n_gpus 8 \
    --sandbox_url http://localhost:8080 \
    --manifest_dir data/manifests \
    --use_external_tests \
    --datasets humaneval mbpp_reg codecontests_valid
```

**verl 模式说明**：
- 使用 Ray 集群协调多个 vLLM 实例
- 支持 Tensor Parallel 分布式推理
- 需要安装 verl 包：`pip install verl`

---

## 5. 输出文件

### 5.1 metrics.json（汇总指标）

```json
{
  "humaneval": {
    "total_problems": 164,
    "accepted_at_1": 0.35,      // 35% 的问题一次通过
    "pass_ratio_mean": 0.42,    // 平均测试通过率
    "total_gen_tokens": 150000,
    "total_gen_time": 245.6,
    "total_judge_time": 131.2,
    "avg_gen_time": 1.50,       // 平均生成时间（秒/题）
    "avg_judge_time": 0.80      // 平均评测时间（秒/题）
  },
  "mbpp_reg": {
    "total_problems": 200,
    "accepted_at_1": 0.45,
    ...
  }
}
```

### 5.2 summary.json（详细统计）

包含更详细的统计信息，如错误类型分布：

```json
{
  "humaneval": {
    "error_distribution": {
      "success": 57,
      "wrong_answer": 45,
      "runtime_error": 30,
      "syntax_error": 20,
      "timeout": 12
    },
    "pass_ratio_percentiles": {
      "p50": 0.5,
      "p90": 1.0
    }
  }
}
```

### 5.3 qa_logs/（问答日志）

每个数据集的问答日志，用于人工审核：

```json
{
  "problem_id": "HumanEval/0",
  "prompt": "from typing import List\n\ndef has_close_elements...",
  "response": "<code>\n    for idx, elem in enumerate(numbers):\n        ...\n</code>",
  "accepted": true,
  "pass_ratio": 1.0,
  "error_type": "success",
  "gen_time": 1.23,
  "judge_time": 0.45
}
```

---

## 6. 典型指标参考

### 6.1 基线指标（7B 模型）

| 数据集 | accepted@1 | 说明 |
|--------|-----------|------|
| HumanEval | 30-40% | 经典代码生成基准 |
| MBPP_reg | 40-50% | Python 编程题 |
| CodeContests | 5-10% | 竞赛题较难 |

### 6.2 错误类型说明

| 错误类型 | 说明 | 常见原因 |
|---------|------|---------|
| `success` | 通过所有测试 | - |
| `wrong_answer` | 输出不正确 | 逻辑错误 |
| `runtime_error` | 运行时错误 | 空指针、除零、越界 |
| `syntax_error` | 语法错误 | 代码格式问题 |
| `timeout` | 执行超时 | 算法复杂度太高 |

---

## 7. 评测流程图解

### 7.1 HumanEval 评测流程

```
1. 加载问题
   ┌─────────────────────────────────────┐
   │ prompt: "def has_close_elements..." │
   │ test_code: "def check(candidate):   │
   │             assert candidate(...)   │
   │             ..."                    │
   │ entry_point: "has_close_elements"   │
   └─────────────────────────────────────┘

2. 模型生成代码
   ┌─────────────────────────────────────┐
   │ <code>                              │
   │     for idx, elem in enumerate(...):│
   │         for idx2, elem2 in ...      │
   │             if abs(elem - elem2)... │
   │     return False                    │
   │ </code>                             │
   └─────────────────────────────────────┘

3. 提取代码（去除 <code> 标签）
   ┌─────────────────────────────────────┐
   │     for idx, elem in enumerate(...):│
   │         ...                         │
   └─────────────────────────────────────┘

4. 组装测试代码
   ┌─────────────────────────────────────┐
   │ # 模型代码                           │
   │     for idx, elem in enumerate(...):│
   │         ...                         │
   │                                     │
   │ # 测试代码                           │
   │ def check(candidate):               │
   │     assert candidate(...) == True   │
   │     ...                             │
   │                                     │
   │ # 执行测试                           │
   │ check(has_close_elements)           │
   └─────────────────────────────────────┘

5. 运行并判断结果
   ┌─────────────────────────────────────┐
   │ 状态: success                        │
   │ accepted: True                       │
   │ pass_ratio: 1.0                     │
   └─────────────────────────────────────┘
```

### 7.2 CodeContests 评测流程

```
1. 加载问题
   ┌─────────────────────────────────────┐
   │ prompt: "Given n and m, find..."    │
   │ tests: [                            │
   │   {input: "2 3\n1\n5\n6\n",         │
   │    output: "9\n6\n1\n"},            │
   │   {input: "1 1\n100\n",             │
   │    output: "100\n"}                 │
   │ ]                                   │
   └─────────────────────────────────────┘

2. 模型生成代码
   ┌─────────────────────────────────────┐
   │ <code>                              │
   │ n, m = map(int, input().split())    │
   │ for _ in range(n):                  │
   │     x = int(input())                │
   │     print(x + m)                    │
   │ </code>                             │
   └─────────────────────────────────────┘

3. 逐个执行测试用例
   ┌─────────────────────────────────────┐
   │ 测试 1:                              │
   │   stdin: "2 3\n1\n5\n6\n"           │
   │   expected: "9\n6\n1\n"             │
   │   actual: "9\n6\n1\n"               │
   │   结果: 通过 ✓                       │
   │                                     │
   │ 测试 2:                              │
   │   stdin: "1 1\n100\n"               │
   │   expected: "100\n"                 │
   │   actual: "100\n"                   │
   │   结果: 通过 ✓                       │
   └─────────────────────────────────────┘

4. 计算最终结果
   ┌─────────────────────────────────────┐
   │ passed: 2 / 2                       │
   │ pass_ratio: 1.0                     │
   │ accepted: True                      │
   └─────────────────────────────────────┘
```

---

## 8. 常见问题

### Q1: 为什么使用 `<code>` 标签？

模型生成的回复通常包含解释文字，如：

```
这个问题需要检查列表中是否有两个数字足够接近。我们可以使用双重循环来比较每对数字。

<code>
def has_close_elements(numbers, threshold):
    for i, x in enumerate(numbers):
        for j, y in enumerate(numbers):
            if i != j and abs(x - y) < threshold:
                return True
    return False
</code>

这个解法的时间复杂度是 O(n^2)。
```

使用 `<code>` 标签可以准确提取代码部分。

### Q2: 如何选择评测方式？

| 场景 | 推荐方式 |
|------|---------|
| 本地开发测试 | `--use_external_tests`（使用 Hugging Face 测试用例） |
| GPU 服务器完整评测 | `--use_external_tests`（确保数据一致性） |
| 与 SandboxFusion 对齐 | `--use_submit_api`（使用内置数据） |

### Q3: simple 模式和 verl 模式的区别？

| 方面 | simple 模式 | verl 模式 |
|------|------------|----------|
| vLLM 启动 | 需要手动启动 | 自动启动 |
| 分布式 | 不支持 | 支持 Ray 集群 |
| GPU 利用 | 单实例 | 多实例并行 |
| 依赖 | aiohttp | verl, ray |
| 适用场景 | 本地测试 | GPU 服务器 |

### Q4: 如何处理超时问题？

```bash
# 增加代码执行超时时间
python src/phase0_eval.py \
    --run_timeout 30 \
    ...
```

对于 CodeContests 这类复杂题目，可能需要更长的超时时间。

---

## 9. 故障排查

### 问题 1：SandboxFusion 连接失败

```bash
# 检查服务是否运行
curl http://localhost:8080/health

# 如果未运行，启动 Docker
docker run -d --rm --privileged -p 8080:8080 \
    volcengine/sandbox-fusion:server-20250609
```

### 问题 2：找不到测试用例

```
Warning: No test_cases in record, using submit API
```

原因：raw 数据文件中没有 `test_cases` 字段。

解决：重新运行数据治理脚本：
```bash
python src/data_governance.py --source huggingface --output_dir data/
```

### 问题 3：代码提取失败

如果模型输出没有 `<code>` 标签或 markdown 代码块，会使用原始输出。

建议：在 prompt 中明确要求模型使用特定格式输出代码。

### 问题 4：verl 模式启动失败

```
Error: verl mode requires verl package. Use --mode simple instead.
```

解决：安装 verl 或使用 simple 模式：
```bash
pip install verl
# 或
python src/phase0_eval.py --mode simple ...
```

---

## 10. 下一步

完成 Phase 0 评测后：

1. **分析结果**：查看 `metrics.json` 中的 `accepted@1` 指标
2. **审核日志**：检查 `qa_logs/` 中的失败案例
3. **建立基线**：记录初始性能，作为后续对比的基准
4. **进入 Phase 1**：开始 SFT（监督微调）或 RL（强化学习）训练

```bash
# 生成评测报告
python src/generate_report.py --input_dir outputs/phase0

# 查看结果摘要
cat outputs/phase0/metrics.json | python -m json.tool
```
