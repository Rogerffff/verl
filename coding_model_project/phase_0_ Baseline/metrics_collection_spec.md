# Phase 0: 指标收集规范

---

## 一、指标分类

Phase 0 需要收集的指标分为四大类：

| 类别 | 目的 | 关键指标 |
|------|------|---------|
| **质量指标** | 衡量模型代码生成能力 | accepted@1, pass_ratio, exec_success_rate |
| **错误分布指标** | 分析失败原因 | syntax_error_rate, runtime_error_rate, timeout_rate |
| **成本效率指标** | 衡量推理与判题开销 | avg_total_gen_tokens, avg_total_judge_time |
| **系统可靠性指标** | 监控评测系统稳定性 | api_error_rate, sandbox_error_rate |

---

## 二、质量指标详解

### 2.1 accepted@1 (主指标)

**定义**：在 EVAL@1 协议下，生成的代码通过所有测试用例的问题比例

**计算公式**：
```
accepted@1 = count(pass_ratio == 1.0) / total_problems
```

**数据来源**：
```python
# verl compute_score 返回的 score == 1.0 表示全部通过
score, metadata = compute_score(...)
accepted = (score == 1.0)
```

**WandB 记录**：
```python
wandb.log({
    "eval/{dataset}/accepted_at_1": accepted_at_1,
})
```

### 2.2 pass_ratio (密集信号)

**定义**：通过的测试用例数 / 总测试用例数

**计算公式**：
```
pass_ratio = passed_count / total_test_cases
```

**统计量**：
- `pass_ratio_mean`: 所有问题的 pass_ratio 均值
- `pass_ratio_p50`: 中位数（第 50 百分位）
- `pass_ratio_p90`: 第 90 百分位

**数据来源**：
```python
# compute_score 直接返回 pass_ratio
score, metadata = compute_score(...)
pass_ratio = score  # 值域 [0, 1]
```

**代码示例**：
```python
import numpy as np

def compute_pass_ratio_stats(pass_ratios: List[float]) -> Dict:
    """计算 pass_ratio 统计量"""
    arr = np.array(pass_ratios)
    return {
        "pass_ratio_mean": float(np.mean(arr)),
        "pass_ratio_p50": float(np.median(arr)),
        "pass_ratio_p90": float(np.percentile(arr, 90)),
    }
```

### 2.3 exec_success_rate

**定义**：代码能够正常执行（不语法错误、不崩溃、不超时）的问题比例

**计算公式**：
```
exec_success_rate = count(final_status in ['success', 'wrong_answer']) / total_problems
```

**说明**：
- `success`: 通过所有测试用例
- `wrong_answer`: 能运行但输出不对

这两种状态都说明代码是可执行的。

**数据来源**：
```python
def is_executable(metadata_list: List[Dict]) -> bool:
    """判断代码是否可执行"""
    # 至少有一个测试用例成功执行（通过或 WA）
    for m in metadata_list:
        status = m.get('status', 'unknown')
        if status in ['success', 'wrong_answer']:
            return True
    return False
```

---

## 三、错误分布指标详解

### 3.1 error_type 定义

| status | 对应结果码 | 含义 |
|--------|-----------|------|
| `success` | True | 测试通过 |
| `wrong_answer` | False | 能运行但输出错误 |
| `syntax_error` / `compile_error` | -4 | 编译/语法错误 |
| `runtime_error` | -2 | 运行时崩溃 |
| `timeout` | -3 | 超时 |
| `api_error` | -1 | SandboxFusion API 错误 |

### 3.2 final_status 确定逻辑

对于一个问题（可能有多个测试用例），按以下优先级确定 final_status：

```python
def determine_final_status(results: List, metadata: List[Dict]) -> str:
    """
    根据所有测试用例结果确定问题的 final_status

    优先级：syntax_error > runtime_error > timeout > wrong_answer > success
    """
    has_syntax = False
    has_runtime = False
    has_timeout = False
    has_wrong_answer = False
    all_passed = True

    for r, m in zip(results, metadata):
        status = m.get('status', 'unknown')

        if r == -4 or status in ['compile_error', 'compile_timeout']:
            has_syntax = True
        elif r == -2 or status == 'runtime_error':
            has_runtime = True
        elif r == -3 or status == 'timeout':
            has_timeout = True
        elif r is False or status == 'wrong_answer':
            has_wrong_answer = True
            all_passed = False
        elif r is True or status == 'success':
            pass
        else:  # API error or unknown
            has_runtime = True  # 归类为 runtime error

        if r is not True:
            all_passed = False

    # 按优先级返回
    if has_syntax:
        return 'syntax_error'
    elif has_runtime:
        return 'runtime_error'
    elif has_timeout:
        return 'timeout'
    elif has_wrong_answer:
        return 'wrong_answer'
    elif all_passed:
        return 'success'
    else:
        return 'unknown'
```

### 3.3 错误分布计算

```python
def compute_error_distribution(results_list: List[EvalResult]) -> Dict[str, float]:
    """
    计算错误分布

    返回:
        {
            "syntax_error_rate": 0.10,
            "runtime_error_rate": 0.08,
            "timeout_rate": 0.12,
            "wrong_answer_rate": 0.55,
            "success_rate": 0.15,
        }
    """
    n = len(results_list)
    if n == 0:
        return {}

    counts = {
        'success': 0,
        'wrong_answer': 0,
        'syntax_error': 0,
        'runtime_error': 0,
        'timeout': 0,
    }

    for r in results_list:
        status = r.final_status
        if status in counts:
            counts[status] += 1

    return {f"{k}_rate": v / n for k, v in counts.items()}
```

### 3.4 WandB 记录

```python
error_dist = compute_error_distribution(results)

wandb.log({
    "eval/{dataset}/syntax_error_rate": error_dist['syntax_error_rate'],
    "eval/{dataset}/runtime_error_rate": error_dist['runtime_error_rate'],
    "eval/{dataset}/timeout_rate": error_dist['timeout_rate'],
    "eval/{dataset}/wrong_answer_rate": error_dist['wrong_answer_rate'],
})
```

---

## 四、成本效率指标详解

### 4.1 avg_total_gen_tokens

**定义**：每个问题平均生成的 token 数

**计算公式**：
```
avg_total_gen_tokens = sum(output_tokens for all problems) / total_problems
```

**数据来源**：
```python
# 使用 tokenizer 计算
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)

def count_tokens(text: str) -> int:
    """计算文本的 token 数"""
    return len(tokenizer.encode(text, add_special_tokens=False))

# 对于每个 completion
output_tokens = count_tokens(completion)
```

### 4.2 avg_total_judge_time

**定义**：每个问题平均消耗的判题时间（秒）

**计算公式**：
```
avg_total_judge_time = sum(total_judge_time for all problems) / total_problems
```

**数据来源**：
```python
def compute_judge_time(metadata_list: List[Dict]) -> float:
    """
    计算一个问题的总判题时间

    从 metadata 的 duration 字段汇总
    """
    total_time = 0.0
    for m in metadata_list:
        # duration 是执行时间（秒）
        duration = m.get('duration', 0) or 0
        total_time += duration

        # 可选：加上编译时间
        compile_duration = m.get('compile_duration', 0) or 0
        total_time += compile_duration

    return total_time
```

### 4.3 p95_total_judge_time

**定义**：判题时间的第 95 百分位

**用途**：监控长尾，识别性能瓶颈

```python
import numpy as np

def compute_judge_time_percentiles(judge_times: List[float]) -> Dict:
    """计算判题时间百分位数"""
    arr = np.array(judge_times)
    return {
        "p50_total_judge_time": float(np.percentile(arr, 50)),
        "p95_total_judge_time": float(np.percentile(arr, 95)),
        "p99_total_judge_time": float(np.percentile(arr, 99)),
    }
```

### 4.4 throughput

**定义**：评测吞吐量，每秒处理的问题数

**计算公式**：
```
throughput = total_problems / wall_clock_time
```

**数据来源**：
```python
import time

start_time = time.time()

# 评测所有问题
for problem in problems:
    # ... 生成和评测
    pass

wall_clock_time = time.time() - start_time
throughput = len(problems) / wall_clock_time
```

### 4.5 cost_per_solved

**定义**：每解决一个问题的平均成本（分 tokens 和 judge_time 两个维度）

**计算公式**：
```
cost_per_solved_tokens = sum(output_tokens for all problems) / solved_count
cost_per_solved_judge_time = sum(judge_time for all problems) / solved_count
```

**重要说明**：
- 分子是**所有问题**的成本（包括未解决的）
- 分母是**已解决**的问题数
- 这样计算能反映"解决一道题的真实代价"

```python
def compute_cost_per_solved(
    results: List[EvalResult]
) -> Dict[str, float]:
    """计算 cost_per_solved"""
    total_tokens = sum(r.output_tokens for r in results)
    total_judge_time = sum(r.total_judge_time for r in results)
    solved_count = sum(r.accepted for r in results)

    if solved_count == 0:
        return {
            "cost_per_solved_tokens": float('inf'),
            "cost_per_solved_judge_time": float('inf'),
        }

    return {
        "cost_per_solved_tokens": total_tokens / solved_count,
        "cost_per_solved_judge_time": total_judge_time / solved_count,
    }
```

---

## 五、系统可靠性指标

### 5.1 api_error_rate

**定义**：SandboxFusion API 调用失败的比例

**数据来源**：
```python
def compute_api_error_rate(metadata_list: List[Dict]) -> float:
    """计算 API 错误率"""
    api_errors = sum(
        1 for m in metadata_list
        if m.get('status') == 'api_error' or m.get('api_request_error')
    )
    return api_errors / len(metadata_list) if metadata_list else 0
```

### 5.2 sandbox_error_rate

**定义**：Sandbox 内部错误的比例

```python
def compute_sandbox_error_rate(metadata_list: List[Dict]) -> float:
    """计算 Sandbox 错误率"""
    sandbox_errors = sum(
        1 for m in metadata_list
        if m.get('status') == 'sandbox_error'
    )
    return sandbox_errors / len(metadata_list) if metadata_list else 0
```

### 5.3 阈值与告警

| 指标 | 正常阈值 | 告警阈值 |
|------|---------|---------|
| api_error_rate | < 1% | > 5% |
| sandbox_error_rate | < 0.5% | > 2% |
| timeout_rate | < 15% | > 30% |

---

## 六、WandB 日志规范

### 6.1 命名约定

```
eval/{dataset}/{metric_name}
```

示例：
- `eval/codecontests_valid/accepted_at_1`
- `eval/humaneval/pass_ratio_mean`
- `eval/mbpp_reg/syntax_error_rate`

### 6.2 完整日志记录

```python
def log_eval_metrics(dataset_name: str, results: List[EvalResult]):
    """记录评测指标到 WandB"""

    # 1. 质量指标
    quality = compute_quality_metrics(results)
    for k, v in quality.items():
        wandb.log({f"eval/{dataset_name}/{k}": v})

    # 2. 错误分布
    error_dist = compute_error_distribution(results)
    for k, v in error_dist.items():
        wandb.log({f"eval/{dataset_name}/{k}": v})

    # 3. 成本效率
    cost = compute_cost_metrics(results)
    for k, v in cost.items():
        wandb.log({f"eval/{dataset_name}/{k}": v})

    # 4. 系统可靠性
    reliability = compute_reliability_metrics(results)
    for k, v in reliability.items():
        wandb.log({f"eval/{dataset_name}/{k}": v})
```

### 6.3 日志样例

```json
{
  "eval/codecontests_valid/accepted_at_1": 0.08,
  "eval/codecontests_valid/pass_ratio_mean": 0.23,
  "eval/codecontests_valid/pass_ratio_p50": 0.18,
  "eval/codecontests_valid/pass_ratio_p90": 0.52,
  "eval/codecontests_valid/exec_success_rate": 0.72,
  "eval/codecontests_valid/syntax_error_rate": 0.12,
  "eval/codecontests_valid/runtime_error_rate": 0.08,
  "eval/codecontests_valid/timeout_rate": 0.08,
  "eval/codecontests_valid/wrong_answer_rate": 0.64,
  "eval/codecontests_valid/avg_total_gen_tokens": 456.3,
  "eval/codecontests_valid/avg_total_judge_time": 2.45,
  "eval/codecontests_valid/p95_total_judge_time": 8.2,
  "eval/codecontests_valid/throughput": 4.5,
  "eval/codecontests_valid/cost_per_solved_tokens": 5703.8,
  "eval/codecontests_valid/cost_per_solved_judge_time": 30.6
}
```

---

## 七、问答日志规范

### 7.1 日志格式

```json
{
  "problem_id": "cc_valid_042",
  "dataset": "codecontests_valid",
  "timestamp": "2024-01-31T10:30:00Z",

  "prompt": "Given an array of integers...",
  "response": "```python\ndef solution():\n    ...\n```",

  "quality_metrics": {
    "pass_ratio": 0.80,
    "accepted": false,
    "final_status": "wrong_answer"
  },

  "cost_metrics": {
    "output_tokens": 156,
    "total_judge_time": 1.23
  },

  "error_breakdown": {
    "passed": 8,
    "wrong_answer": 2,
    "timeout": 0,
    "runtime_error": 0,
    "syntax_error": 0
  },

  "execution_details": {
    "first_failed_case": {
      "input": "10\n1 2 3 ...",
      "expected": "29",
      "actual": "28",
      "stderr": ""
    }
  },

  "metadata": {
    "model": "Qwen2.5-Coder-7B-Instruct",
    "temperature": 0.0,
    "max_new_tokens": 2048
  }
}
```

### 7.2 分层抽样策略

| 数据集 | 样本数 | success | WA | Syntax | Runtime | Timeout |
|--------|--------|---------|-----|--------|---------|---------|
| CodeContests_valid | 50 | 10 | 20 | 10 | 5 | 5 |
| CodeContests_test | 30 | 6 | 12 | 6 | 3 | 3 |
| HumanEval | 20 | 4 | 8 | 4 | 2 | 2 |
| MBPP_reg | 20 | 4 | 8 | 4 | 2 | 2 |

### 7.3 抽样代码

```python
import random
from collections import defaultdict

def stratified_sample(
    results: List[EvalResult],
    num_samples: int,
    seed: int = 42
) -> List[EvalResult]:
    """
    分层抽样

    按 final_status 分组，每组按比例抽取
    """
    random.seed(seed)

    # 按状态分组
    by_status = defaultdict(list)
    for r in results:
        by_status[r.final_status].append(r)

    # 计算每组应抽取数量
    num_statuses = len(by_status)
    base_per_status = num_samples // num_statuses
    extra = num_samples % num_statuses

    samples = []
    for i, (status, group) in enumerate(sorted(by_status.items())):
        n = base_per_status + (1 if i < extra else 0)
        n = min(n, len(group))  # 不能超过组内数量
        samples.extend(random.sample(group, n))

    return samples
```

---

## 八、汇总报告模板

### 8.1 Phase 0 Summary JSON

```json
{
  "meta": {
    "phase": 0,
    "model": "Qwen2.5-Coder-7B-Instruct",
    "timestamp": "2024-01-31T12:00:00Z",
    "protocol": "EVAL@1"
  },

  "codecontests_valid": {
    "quality": {
      "accepted_at_1": 0.08,
      "pass_ratio_mean": 0.23,
      "pass_ratio_p50": 0.18,
      "pass_ratio_p90": 0.52,
      "exec_success_rate": 0.72
    },
    "error_distribution": {
      "syntax_error_rate": 0.12,
      "runtime_error_rate": 0.08,
      "timeout_rate": 0.08,
      "wrong_answer_rate": 0.64
    },
    "cost": {
      "avg_total_gen_tokens": 456.3,
      "avg_total_judge_time": 2.45,
      "p95_total_judge_time": 8.2,
      "throughput": 4.5,
      "cost_per_solved_tokens": 5703.8,
      "cost_per_solved_judge_time": 30.6
    }
  },

  "codecontests_test": { ... },
  "humaneval": { ... },
  "mbpp_reg": { ... }
}
```

### 8.2 Markdown 报告

```markdown
# Phase 0 Baseline 评测报告

## 模型信息
- Model: Qwen2.5-Coder-7B-Instruct
- Protocol: EVAL@1 (greedy decoding)
- Date: 2024-01-31

---

## 质量指标

| Dataset | accepted@1 | pass_ratio_mean | pass_ratio_p50 | pass_ratio_p90 | exec_success |
|---------|------------|-----------------|----------------|----------------|--------------|
| CC_valid | 8.0% | 0.23 | 0.18 | 0.52 | 72.0% |
| CC_test | 7.5% | 0.21 | 0.16 | 0.48 | 70.0% |
| HumanEval | 35.0% | 0.42 | 0.35 | 0.85 | 88.0% |
| MBPP_reg | 45.0% | 0.58 | 0.55 | 0.92 | 92.0% |

---

## 错误分布

| Dataset | syntax | runtime | timeout | wrong_answer |
|---------|--------|---------|---------|--------------|
| CC_valid | 12.0% | 8.0% | 8.0% | 64.0% |
| CC_test | 13.0% | 9.0% | 8.0% | 63.0% |
| HumanEval | 5.0% | 3.0% | 4.0% | 53.0% |
| MBPP_reg | 3.0% | 2.0% | 3.0% | 47.0% |

---

## 成本效率

| Dataset | avg_tokens | avg_judge_time | throughput | cost/solved_tokens | cost/solved_time |
|---------|------------|----------------|------------|--------------------|--------------------|
| CC_valid | 456 | 2.45s | 4.5/s | 5704 | 30.6s |
| CC_test | 478 | 2.52s | 4.3/s | 6373 | 33.6s |
| HumanEval | 234 | 0.85s | 12.0/s | 669 | 2.4s |
| MBPP_reg | 189 | 0.62s | 15.0/s | 420 | 1.4s |

---

## 结论

Phase 0 基线评测完成。主要发现：
1. Base 模型在 CodeContests 上表现较差 (accepted@1 ~8%)
2. HumanEval/MBPP 上表现相对较好 (accepted@1 35-45%)
3. 错误主要集中在 wrong_answer (逻辑错误)
4. syntax_error 和 runtime_error 占比较低，说明模型能生成语法正确的代码
```

---

## 九、常见问题

### Q1: pass_ratio 和 accepted@1 的区别？

- `accepted@1`: 二值指标，只有全部通过才算 1，否则为 0
- `pass_ratio`: 连续指标，反映部分正确性

Phase 0 的意义是建立基线，后续 SFT/GRPO 训练时：
- `accepted@1` 是最终目标
- `pass_ratio` 提供更丰富的梯度信号（dense reward）

### Q2: 为什么 cost_per_solved 的分子包含未解决问题的成本？

这样计算反映了"为了解决一道题，平均需要尝试多少次/消耗多少资源"。

如果只计算 solved 问题的成本，会低估真实代价。

### Q3: timeout_rate 高怎么办？

可能原因：
1. 模型生成的算法效率低
2. 沙盒 timeout 设置过短

建议：
1. 检查生成的代码是否使用了低效算法（如 O(n²) 而不是 O(n log n)）
2. 考虑增加 timeout 阈值（但会增加成本）

---

*文档版本：v1.0*
*创建日期：2024-01-31*
