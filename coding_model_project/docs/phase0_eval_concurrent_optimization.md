# Phase0 评测脚本并发优化文档

## 修改日期
2026-02-06

## 修改概述

本次修改针对 `src/phase0_eval.py` 中的 `_evaluate_codecontests` 函数进行了**并发执行优化**，将 CodeContests 数据集的测试用例从串行执行改为并发执行，显著提升了评测速度。

## 问题背景

### 性能瓶颈分析

CodeContests 数据集的特点：
- **每题平均 50+ 个测试用例**（最多可达 230 个）
- **每个测试用例需要一次独立的 HTTP 调用**到 SandboxFusion 服务端
- **原始实现是串行执行**：`for tc in tests: run_code(...)`

**性能数据（修改前）：**
- `codecontests_valid`: 117 题，6,207 个测试用例，总耗时 **758.3 秒**
- 平均每题耗时：**6.5 秒**
- 平均每次 HTTP 调用耗时：**~0.122 秒**
- 吞吐率：**0.15 problems/sec**

**瓶颈原因：**
- 串行执行导致总耗时 = `N × avg_latency`（N 为测试用例数）
- 例如：50 个测试用例 × 0.12s = **6 秒/题**

## 修改内容

### 1. 新增导入

```python
from concurrent.futures import ThreadPoolExecutor  # 线程池，用于并发执行 sandbox 测试用例
```

### 2. EvalConfig 新增字段

```python
judge_concurrency: int = 20  # CodeContests 测试用例并发执行数（线程池大小）
```

### 3. 核心改动：`_evaluate_codecontests` 函数重构

#### 3.1 执行方式改变

**修改前（串行）：**
```python
for tc in tests:
    result = run_code(RunCodeRequest(...))  # 阻塞等待
    # 处理结果
    if error:
        break  # 遇到错误立即退出
```

**修改后（并发）：**
```python
# 1. 构建所有 RunCodeRequest
requests = [RunCodeRequest(...) for tc in tests]

# 2. 线程池并发执行
def _safe_run_code(req):
    try:
        return run_code(req)
    except Exception:
        return None  # 异常处理

with ThreadPoolExecutor(max_workers=config.judge_concurrency) as executor:
    run_results = list(executor.map(_safe_run_code, requests))

# 3. 汇总所有结果
for tc, result in zip(tests, run_results):
    # 处理结果
```

#### 3.2 错误类型判定逻辑改进

**修改前：**
- 依赖 `break` 的顺序隐式确定错误类型
- 第一个遇到的错误类型就是整体错误类型

**修改后：**
- 显式收集所有测试用例的错误类型标志
- 按优先级统一判定：`syntax_error > timeout > runtime_error > wrong_answer`

```python
has_syntax_error = False
has_timeout = False
has_runtime_error = False
has_wrong_answer = False

# 汇总时标记
for tc, result in zip(tests, run_results):
    if "SyntaxError" in actual_output:
        has_syntax_error = True
    # ...

# 按优先级确定整体错误类型
if accepted:
    error_type = "success"
elif has_syntax_error:
    error_type = "syntax_error"
elif has_timeout:
    error_type = "timeout"
# ...
```

#### 3.3 新增每个测试用例的错误类型记录

**新增功能：**
- 记录每个测试用例的详细错误信息（`test_case_errors`）
- 统计各错误类型的数量（`error_type_counts`）

```python
test_case_errors: List[Dict[str, Any]] = []

for tc_idx, (tc, result) in enumerate(zip(tests, run_results)):
    # ...
    test_case_errors.append({
        "test_index": tc_idx,
        "error_type": "wrong_answer",  # 或 "success", "syntax_error", "timeout", "runtime_error"
        "error_message": "...",
        "expected": "...",  # 仅 wrong_answer 有
        "got": "...",       # 仅 wrong_answer 有
    })

# 统计
error_type_counts = {}
for tc_err in test_case_errors:
    err_type = tc_err["error_type"]
    error_type_counts[err_type] = error_type_counts.get(err_type, 0) + 1
```

**返回的 details 中新增字段：**
```python
details={
    "test_case_errors": test_case_errors,      # 每个测试用例的详细错误
    "error_type_counts": error_type_counts,    # 各错误类型的统计
    # ... 其他字段
}
```

#### 3.4 异常处理增强

**新增 `_safe_run_code` 包装函数：**
- 捕获 `run_code()` 的异常，返回 `None` 而不是崩溃
- 单个测试用例失败不影响其他测试用例的执行

### 4. CLI 参数新增

```python
parser.add_argument("--judge_concurrency", type=int, default=20,
                    help="CodeContests 测试用例并发执行线程数（默认 20）")
```

## 预期性能提升

### 理论加速比

- **单题加速**：`50 个测试用例 / 20 并发 ≈ 2.5 轮` → 从 6s 降至 **~0.3s**（**~20x**）
- **整体加速**：`codecontests_valid` 从 758s 降至 **~40-60s**（**~15x**）

### 实际效果

（待运行测试后补充实际数据）

## 设计要点

### 1. 为什么使用 ThreadPoolExecutor 而不是 asyncio？

- SDK 的 `run_code()` 是**同步阻塞**的 HTTP POST 调用
- `ThreadPoolExecutor` 可以并发执行多个同步函数
- 服务端（FastAPI + asyncio）天然支持并发处理多个 HTTP 请求

### 2. 为什么不需要修改 SDK 或服务端？

- **SDK**：`run_code()` 是无状态的线程安全函数，多线程调用完全安全
- **服务端**：FastAPI + asyncio 天然支持并发，每个请求独立处理

### 3. 并发度选择

- 默认 `judge_concurrency=20`：平衡并发度和资源占用
- 可根据 sandbox 服务器性能调整：
  - 服务器资源充足：可提高到 30-50
  - 服务器资源紧张：可降低到 10-15

## 兼容性

### 向后兼容

- ✅ 所有现有功能保持不变
- ✅ 输出格式兼容（新增字段不影响现有解析）
- ✅ 默认参数保证行为一致

### 行为变化

1. **所有测试用例都会执行完**（不再提前 `break`）
   - 原始版本：遇到 `syntax_error` 或 `timeout` 会立即停止
   - 修改后：所有测试用例并发执行，无法提前退出
   - **影响**：略微增加执行时间，但并发带来的收益远大于此

2. **错误类型判定更精确**
   - 原始版本：依赖执行顺序
   - 修改后：按优先级统一判定，逻辑更清晰

## 使用示例

### 基本使用（使用默认并发度 20）

```bash
python src/phase0_eval.py \
    --mode simple \
    --vllm_url http://localhost:8000 \
    --datasets codecontests_valid
```

### 自定义并发度

```bash
python src/phase0_eval.py \
    --mode simple \
    --vllm_url http://localhost:8000 \
    --datasets codecontests_valid \
    --judge_concurrency 30  # 提高并发度
```

### 查看详细错误信息

修改后，`outputs/phase0_*/per_problem/codecontests_*.jsonl` 中每个问题的 `details` 包含：

```json
{
  "problem_id": "Codeforces/1548/C",
  "accepted": false,
  "error_type": "syntax_error",
  "details": {
    "passed": 0,
    "total": 50,
    "error_type_counts": {
      "syntax_error": 1,
      "wrong_answer": 49
    },
    "test_case_errors": [
      {
        "test_index": 0,
        "error_type": "syntax_error",
        "error_message": "SyntaxError: invalid syntax..."
      },
      {
        "test_index": 1,
        "error_type": "wrong_answer",
        "error_message": "Expected: 9, Got: 8",
        "expected": "9",
        "got": "8"
      }
      // ... 其他测试用例
    ]
  }
}
```

## 文件修改清单

- `src/phase0_eval.py`
  - 新增 `ThreadPoolExecutor` 导入
  - `EvalConfig` 新增 `judge_concurrency` 字段
  - `_evaluate_codecontests` 函数完全重构
  - CLI 新增 `--judge_concurrency` 参数

## 测试建议

1. **功能测试**：确保并发执行不影响评测结果的正确性
2. **性能测试**：对比修改前后的耗时
3. **边界测试**：测试异常情况（sandbox 服务不可用、网络超时等）

## 后续优化方向

1. **方案 2**：多题判题并行化（当前生成与判题分离，可进一步并行化）
2. **方案 3**：CodeContests early-exit 优化（在第一个 `wrong_answer` 时提前退出）
3. **方案 4**：合并测试到单次调用（类似 HumanEval/MBPP，将多个 stdin/stdout 合并成一个脚本）

## 相关文档

- 原始性能分析：见代码注释和终端日志
- SandboxFusion 架构：`/workspace/sandbox/intro_doc/`
