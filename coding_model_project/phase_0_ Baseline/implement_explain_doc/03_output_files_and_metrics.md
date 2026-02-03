# Phase 0 评测输出文件与指标说明

本文档说明 `phase0_eval.py` 评测脚本的输出文件结构和记录的指标。

---

## 一、输出文件结构

运行 `phase0_eval.py` 后，会在 `--output_dir`（默认 `outputs/phase0`）目录下生成以下文件：

```
{output_dir}/
├── metrics.json          # 每个数据集的汇总指标（简洁版）
├── summary.json          # 详细统计（包含错误分布等）
└── qa_logs/              # 问答日志目录
    ├── humaneval.jsonl   # HumanEval 数据集的问答日志
    ├── mbpp_reg.jsonl    # MBPP Regular 数据集的问答日志
    └── ...
```

---

## 二、metrics.json

**来源**：`evaluate_dataset()` 函数返回的 `dataset_metrics`

**用途**：快速查看每个数据集的核心指标

### 文件格式

```json
{
  "humaneval": {
    "total_problems": 164,

    // === 质量指标 ===
    "accepted_at_1": 0.35,      // 主指标：通过所有测试用例的问题比例
    "pass_ratio_mean": 0.42,    // pass_ratio 均值
    "pass_ratio_p50": 0.35,     // pass_ratio 中位数
    "pass_ratio_p90": 0.85,     // pass_ratio 第90百分位

    // === 成本指标 ===
    "total_gen_tokens": 38456,       // 总生成 token 数
    "avg_gen_tokens": 234.5,         // 平均每题生成 token 数
    "total_gen_time": 120.5,         // 总生成耗时（秒）
    "avg_gen_time": 0.73,            // 平均每题生成耗时
    "total_judge_time": 85.3,        // 总判题耗时（秒）
    "avg_judge_time": 0.52,          // 平均每题判题耗时
    "wall_clock_time": 205.8,        // 数据集评测总耗时（秒）
    "throughput": 0.80,              // 吞吐量（问题/秒）
    "cost_per_solved_tokens": 669.0, // 每解决一题的 token 成本
    "cost_per_solved_judge_time": 1.49  // 每解决一题的判题时间成本
  },
  "mbpp_reg": { ... },
  "codecontests_valid": { ... }
}
```

### 指标说明

| 指标 | 类型 | 说明 |
|------|------|------|
| `total_problems` | int | 数据集总题数 |
| `accepted_at_1` | float | **主指标**，EVAL@1 通过率 |
| `pass_ratio_mean` | float | 通过测试用例比例的均值 |
| `pass_ratio_p50` | float | 通过测试用例比例的中位数 |
| `pass_ratio_p90` | float | 通过测试用例比例的第90百分位 |
| `total_gen_tokens` | int | 所有问题生成的总 token 数 |
| `avg_gen_tokens` | float | 平均每题生成 token 数 |
| `total_gen_time` | float | 总生成耗时（秒） |
| `avg_gen_time` | float | 平均每题生成耗时（秒） |
| `total_judge_time` | float | 总判题耗时（秒） |
| `avg_judge_time` | float | 平均每题判题耗时（秒） |
| `wall_clock_time` | float | 数据集评测总耗时（秒） |
| `throughput` | float | 吞吐量（问题/秒） |
| `cost_per_solved_tokens` | float | 每解决一题消耗的总 token（分子含未解决问题） |
| `cost_per_solved_judge_time` | float | 每解决一题消耗的总判题时间（分子含未解决问题） |

> **注意**：当没有问题通过时，`cost_per_solved_*` 会是 `null`（JSON 中 inf 转为 null）

---

## 三、summary.json

**来源**：`MetricsCollector.get_summary()` 方法

**用途**：详细统计，包含错误分布、百分位数等完整指标

### 文件格式

```json
{
  "datasets": {
    "humaneval": {
      "dataset": "humaneval",
      "total_problems": 164,

      // === 质量指标 ===
      "accepted_at_1": 0.35,
      "pass_ratio_mean": 0.42,
      "pass_ratio_p50": 0.35,
      "pass_ratio_p90": 0.85,
      "exec_success_rate": 0.88,   // 可执行率（非语法错误/非超时/非API错误）

      // === 错误分布 ===
      "syntax_error_rate": 0.05,   // 语法错误率
      "runtime_error_rate": 0.03,  // 运行时错误率
      "timeout_rate": 0.04,        // 超时率
      "wrong_answer_rate": 0.53,   // 答案错误率
      "api_error_rate": 0.00,      // API 错误率
      "unknown_error_rate": 0.00,  // 未知错误率

      // === 成本指标 ===
      "avg_judge_time": 0.52,
      "total_judge_time": 85.3,
      "p50_judge_time": 0.45,      // 判题时间中位数
      "p95_judge_time": 1.23,      // 判题时间第95百分位
      "avg_gen_tokens": 234.5,
      "total_gen_tokens": 38456,
      "throughput": 0.80,
      "cost_per_solved_tokens": 669.0,
      "cost_per_solved_judge_time": 1.49
    },
    "mbpp_reg": { ... }
  },
  "overall": {
    "total_problems": 364,
    "accepted_at_1": 0.40,
    "pass_ratio_mean": 0.50,
    "pass_ratio_p50": 0.45,
    "pass_ratio_p90": 0.88,
    "total_judge_time": 170.6
  }
}
```

### 与 metrics.json 的区别

| 特性 | metrics.json | summary.json |
|------|--------------|--------------|
| 错误分布 | ❌ 无 | ✅ 有 |
| exec_success_rate | ❌ 无 | ✅ 有 |
| p50_judge_time | ❌ 无 | ✅ 有 |
| p95_judge_time | ❌ 无 | ✅ 有 |
| overall 汇总 | ❌ 无 | ✅ 有 |
| wall_clock_time | ✅ 有 | ❌ 无（在 throughput 中体现） |

---

## 四、qa_logs/*.jsonl

**来源**：`QALogger` 类

**用途**：保存问答详情，用于后续分析和调试

### 文件格式（每行一个 JSON 对象）

```json
{
  "problem_id": "HumanEval/42",
  "dataset": "humaneval",
  "timestamp": "2024-01-31T10:30:00Z",

  "prompt": "def add(a, b):\n    \"\"\"Add two numbers...",
  "response": "def add(a, b):\n    return a + b",

  "eval_result": {
    "accepted": true,
    "pass_ratio": 1.0,
    "error_type": "success",
    "judge_time": 0.45,
    "gen_tokens": 156,
    "gen_time": 0.82
  },

  "gen_metadata": {
    "prompt_tokens": 234,
    "completion_tokens": 156,
    "total_tokens": 390,
    "finish_reason": "stop"
  }
}
```

### 抽样策略

`QALogger` 会按 `error_type` 分层抽样，确保每种错误类型都有代表性样本：

| error_type | 含义 |
|------------|------|
| success | 通过所有测试 |
| wrong_answer | 能运行但输出错误 |
| syntax_error | 语法/编译错误 |
| runtime_error | 运行时崩溃 |
| timeout | 执行超时 |
| api_error | SandboxFusion API 错误 |

---

## 五、指标完整清单

### 质量指标（5个）

| 指标名 | metrics.json | summary.json | WandB | 说明 |
|--------|:------------:|:------------:|:-----:|------|
| accepted_at_1 | ✅ | ✅ | ✅ | **主指标** |
| pass_ratio_mean | ✅ | ✅ | ✅ | 密集信号均值 |
| pass_ratio_p50 | ✅ | ✅ | ✅ | 密集信号中位数 |
| pass_ratio_p90 | ✅ | ✅ | ✅ | 密集信号第90百分位 |
| exec_success_rate | ❌ | ✅ | ✅ | 可执行率 |

### 错误分布指标（6个）

| 指标名 | metrics.json | summary.json | WandB | 说明 |
|--------|:------------:|:------------:|:-----:|------|
| syntax_error_rate | ❌ | ✅ | ✅ | 语法错误率 |
| runtime_error_rate | ❌ | ✅ | ✅ | 运行时错误率 |
| timeout_rate | ❌ | ✅ | ✅ | 超时率 |
| wrong_answer_rate | ❌ | ✅ | ✅ | 答案错误率 |
| api_error_rate | ❌ | ✅ | ✅ | API 错误率 |
| unknown_error_rate | ❌ | ✅ | ❌ | 未知错误率 |

### 成本效率指标（10个）

| 指标名 | metrics.json | summary.json | WandB | 说明 |
|--------|:------------:|:------------:|:-----:|------|
| total_gen_tokens | ✅ | ✅ | ✅ | 总生成 token |
| avg_gen_tokens | ✅ | ✅ | ✅ | 平均生成 token |
| total_gen_time | ✅ | ❌ | ✅ | 总生成耗时 |
| avg_gen_time | ✅ | ❌ | ✅ | 平均生成耗时 |
| total_judge_time | ✅ | ✅ | ✅ | 总判题耗时 |
| avg_judge_time | ✅ | ✅ | ✅ | 平均判题耗时 |
| p50_judge_time | ❌ | ✅ | ✅ | 判题时间中位数 |
| p95_judge_time | ❌ | ✅ | ✅ | 判题时间第95百分位 |
| throughput | ✅ | ✅ | ✅ | 吞吐量 |
| wall_clock_time | ✅ | ❌ | ✅ | 数据集评测总耗时 |
| cost_per_solved_tokens | ✅ | ✅ | ✅ | 每解决一题的 token 成本 |
| cost_per_solved_judge_time | ✅ | ✅ | ✅ | 每解决一题的判题时间 |

---

## 六、WandB 日志格式

启用 `--use_wandb` 后，指标会以以下格式记录：

```
eval/{dataset}/{metric_name}
```

示例：
- `eval/humaneval/accepted_at_1`
- `eval/mbpp_reg/pass_ratio_mean`
- `eval/codecontests_valid/syntax_error_rate`

---

## 七、使用示例

### 读取 metrics.json

```python
import json

with open("outputs/phase0/metrics.json") as f:
    metrics = json.load(f)

for dataset, m in metrics.items():
    print(f"{dataset}: accepted@1 = {m['accepted_at_1']:.2%}")
```

### 读取 summary.json

```python
import json

with open("outputs/phase0/summary.json") as f:
    summary = json.load(f)

# 查看错误分布
for dataset, m in summary["datasets"].items():
    print(f"\n{dataset} 错误分布:")
    print(f"  syntax_error: {m['syntax_error_rate']:.2%}")
    print(f"  runtime_error: {m['runtime_error_rate']:.2%}")
    print(f"  timeout: {m['timeout_rate']:.2%}")
    print(f"  wrong_answer: {m['wrong_answer_rate']:.2%}")
```

### 读取问答日志

```python
import json

# 读取 HumanEval 的问答日志
with open("outputs/phase0/qa_logs/humaneval.jsonl") as f:
    for line in f:
        record = json.loads(line)
        if record["eval_result"]["error_type"] == "syntax_error":
            print(f"语法错误样例: {record['problem_id']}")
            print(f"  Response: {record['response'][:200]}...")
```

---

*文档版本：v1.0*
*创建日期：2024-02-03*
