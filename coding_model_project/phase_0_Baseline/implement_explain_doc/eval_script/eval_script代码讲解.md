# Phase 0 评测脚本讲解 (Simple 模式)

本文档按照 **代码执行顺序** 讲解 `phase0_eval.py` 的实现。

---

## 第一部分：程序入口与配置初始化

### 1.1 程序入口点

```python
# 第 1896-1897 行
if __name__ == "__main__":
    main()
```

Python 程序从这里开始执行。`__name__ == "__main__"` 确保只有直接运行脚本时才执行 `main()`，被 import 时不执行。

---

### 1.2 main() 函数：命令行解析与配置创建

```python
# 第 1768-1892 行
def main():
    """命令行入口函数"""

    # ========== 第一步：定义命令行参数 ==========
    parser = argparse.ArgumentParser(
        description="Phase 0 Baseline Evaluation (verl Standalone Rollout)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="..."  # 使用示例
    )

    # 模式选择
    parser.add_argument("--mode", type=str, default="simple",
                        choices=["verl", "simple"],
                        help="运行模式: verl (分布式) 或 simple (简化)")

    # 模型配置
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-Coder-7B-Instruct")

    # vLLM 服务器地址（simple 模式使用）
    parser.add_argument("--vllm_url", type=str,
                        default="http://localhost:8000")

    # SandboxFusion 配置
    parser.add_argument("--sandbox_url", type=str,
                        default="http://localhost:8080")

    # 数据集列表
    parser.add_argument("--datasets", nargs="+", type=str,
                        default=["humaneval", "mbpp_reg"])

    # ... 更多参数 ...

    args = parser.parse_args()
```

#### argparse 关键语法

| 参数 | 说明 | 示例 |
|------|------|------|
| `type=str` | 参数类型 | 字符串 |
| `default="simple"` | 默认值 | 不传参数时使用 |
| `choices=[...]` | 限制可选值 | 只能是列表中的值 |
| `nargs="+"` | 接收多个值 | `--datasets humaneval mbpp_reg` |
| `action="store_true"` | 布尔开关 | 存在则为 True |

---

### 1.3 创建 EvalConfig 配置对象

```python
    # 第 1862-1884 行
    # ========== 第二步：创建配置对象 ==========
    config = EvalConfig(
        mode=args.mode,                    # "simple"
        model_path=args.model,             # 模型路径
        vllm_url=args.vllm_url,            # vLLM 服务器地址
        sandbox_url=args.sandbox_url,      # SandboxFusion 地址
        run_timeout=args.run_timeout,      # 代码执行超时
        temperature=args.temperature,      # 采样温度（0.0 = greedy）
        max_new_tokens=args.max_tokens,    # 最大生成长度
        datasets=args.datasets,            # 数据集列表
        manifest_dir=args.manifest_dir,    # 数据目录
        output_dir=args.output_dir,        # 输出目录
        max_concurrent_requests=args.max_concurrent,  # 最大并发数
        batch_size=args.batch_size,        # 批处理大小
        # ... 更多配置 ...
    )
```

#### EvalConfig 数据类（第 130-189 行）

```python
@dataclass
class EvalConfig:
    """
    评测配置 - 使用 Python dataclass 自动生成 __init__ 等方法
    """
    # === 运行模式 ===
    mode: str = "verl"  # "simple" 时连接已有 vLLM 服务器

    # === 模型配置 ===
    model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # === 简化模式配置 ===
    vllm_url: str = "http://localhost:8000"  # vLLM 服务器地址

    # === 解码参数（EVAL@1 协议：贪婪解码） ===
    temperature: float = 0.0  # 0.0 = greedy，确保可复现
    top_p: float = 1.0
    max_new_tokens: int = 2048

    # === SandboxFusion 配置 ===
    sandbox_url: str = "http://localhost:8080"
    run_timeout: int = 30     # 代码执行超时（秒）
    memory_limit_mb: int = 1024

    # === 并发控制 ===
    max_concurrent_requests: int = 64  # 最大并发请求数
    batch_size: int = 50               # 批处理大小

    # === 数据配置 ===
    datasets: List[str] = field(default_factory=lambda: ["humaneval", "mbpp_reg"])
```

#### @dataclass 装饰器

`@dataclass` 是 Python 3.7+ 的特性，自动生成：
- `__init__()` 方法
- `__repr__()` 方法
- `__eq__()` 方法

```python
# 等价于手写：
class EvalConfig:
    def __init__(self, mode="verl", model_path="...", ...):
        self.mode = mode
        self.model_path = model_path
        # ...
```

#### field(default_factory=...) 的作用

```python
# 错误写法！可变默认值会被所有实例共享
datasets: List[str] = ["humaneval"]  # 危险！

# 正确写法：每次创建新实例时调用 lambda 生成新列表
datasets: List[str] = field(default_factory=lambda: ["humaneval", "mbpp_reg"])
```

---

### 1.4 启动异步事件循环

```python
    # 第 1892 行
    # ========== 第三步：运行评测（异步） ==========
    asyncio.run(run_evaluation(config))
```

**这是整个脚本的关键转折点！**

#### asyncio.run() 做了什么？

```
┌─────────────────────────────────────────────────────────────┐
│  同步世界 (main 函数)                                        │
│                                                              │
│  asyncio.run(run_evaluation(config))                        │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  异步世界 (事件循环 Event Loop)                      │    │
│  │                                                      │    │
│  │  run_evaluation() 协程开始执行                       │    │
│  │       │                                              │    │
│  │       ├── await evaluate_dataset() ──┐              │    │
│  │       │                               │              │    │
│  │       │   await batch_generate() ─────┤              │    │
│  │       │                               │ 并发执行     │    │
│  │       │   await generate_code() ──────┤              │    │
│  │       │   await generate_code() ──────┤              │    │
│  │       │   await generate_code() ──────┘              │    │
│  │       │                                              │    │
│  │       ▼                                              │    │
│  │  协程完成，返回结果                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│       │                                                      │
│       ▼                                                      │
│  asyncio.run() 返回，回到同步世界                            │
└─────────────────────────────────────────────────────────────┘
```

`asyncio.run()` 的作用：
1. **创建事件循环** (Event Loop)
2. **运行协程** 直到完成
3. **关闭事件循环**
4. **返回协程的结果**

---

### 1.5 本部分的配置参数总结（Simple 模式）

运行命令示例：
```bash
python src/phase0_eval.py \
    --mode simple \
    --vllm_url http://localhost:8000 \
    --sandbox_url http://localhost:8080 \
    --datasets humaneval mbpp_reg \
    --temperature 0.0 \
    --max_tokens 2048 \
    --max_concurrent 64 \
    --batch_size 50 \
    --output_dir outputs/phase0
```

| 参数 | 值 | 说明 |
|------|----|----|
| `mode` | `simple` | 连接已有 vLLM 服务器 |
| `vllm_url` | `http://localhost:8000` | vLLM OpenAI-compatible API 地址 |
| `sandbox_url` | `http://localhost:8080` | SandboxFusion 判题服务地址 |
| `datasets` | `["humaneval", "mbpp_reg"]` | 要评测的数据集 |
| `temperature` | `0.0` | 贪婪解码，确保结果可复现 |
| `max_concurrent` | `64` | 最多同时 64 个并发请求 |

---

## 小结

**执行流程到目前为止：**

```
1. if __name__ == "__main__": main()
       │
       ▼
2. main() 函数
       │
       ├── argparse 解析命令行参数
       │
       ├── 创建 EvalConfig 配置对象
       │
       └── asyncio.run(run_evaluation(config))
              │
              ▼
3. 进入异步世界... (下一部分讲解)
```

**关键概念：**
- `argparse`: 命令行参数解析
- `@dataclass`: 自动生成数据类方法
- `field(default_factory=...)`: 可变默认值的正确写法
- `asyncio.run()`: 同步世界到异步世界的入口

---

**请确认你理解了第一部分后，我将继续讲解第二部分：run_evaluation() 主流程。**

---

## 第二部分：run_evaluation() 主流程

### 2.1 协程函数的定义

```python
# 第 1616 行
async def run_evaluation(config: EvalConfig):
    """
    运行完整评测流程
    """
```

#### 什么是协程 (Coroutine)？

**协程** 是可以暂停和恢复执行的函数。`async def` 定义的函数叫做 **协程函数**，调用它不会立即执行，而是返回一个 **协程对象**。

```python
# 普通函数：调用立即执行
def normal_func():
    return 42
result = normal_func()  # 立即执行，result = 42

# 协程函数：调用返回协程对象，不会立即执行
async def async_func():
    return 42
coro = async_func()     # 不执行！返回 <coroutine object>
result = await coro     # 现在才执行，result = 42
```

#### 协程 vs 线程 vs 进程

```
┌─────────────────────────────────────────────────────────────────────┐
│                          对比表                                      │
├─────────────┬──────────────┬──────────────┬────────────────────────┤
│             │   进程        │   线程        │   协程                  │
├─────────────┼──────────────┼──────────────┼────────────────────────┤
│ 内存占用     │   最大        │   中等        │   最小（几KB）          │
│ 切换开销     │   最大        │   中等        │   最小（用户态）        │
│ 并行能力     │   真并行      │   受 GIL 限制 │   单线程并发           │
│ 适用场景     │   CPU 密集    │   混合场景    │   I/O 密集（网络）     │
│ 数量上限     │   几十个      │   几百个      │   几万个               │
└─────────────┴──────────────┴──────────────┴────────────────────────┘
```

**本脚本的场景**：评测需要大量网络 I/O（调用 vLLM 和 SandboxFusion），协程是最佳选择。

---

### 2.2 run_evaluation() 执行流程（Simple 模式）

```python
async def run_evaluation(config: EvalConfig):
    # ========== 步骤1：打印配置信息 ==========
    print(f"Mode: {config.mode}")        # "simple"
    print(f"Model: {config.model_path}") # "Qwen/Qwen2.5-Coder-7B-Instruct"
    print(f"Datasets: {config.datasets}")# ["humaneval", "mbpp_reg"]

    # ========== 步骤2：创建输出目录 ==========
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # parents=True: 自动创建父目录
    # exist_ok=True: 目录已存在时不报错
```

#### pathlib.Path 用法

```python
from pathlib import Path

# 创建 Path 对象（比字符串拼接更安全）
output_dir = Path("outputs/phase0")

# 创建目录
output_dir.mkdir(parents=True, exist_ok=True)

# 路径拼接（用 / 运算符，不用 os.path.join）
metrics_file = output_dir / "metrics.json"  # "outputs/phase0/metrics.json"

# 常用方法
output_dir.exists()      # 是否存在
output_dir.is_dir()      # 是否是目录
output_dir.iterdir()     # 遍历目录
```

---

### 2.3 获取服务器地址（Simple 模式的关键分支）

```python
    # 第 1639-1649 行
    # ========== 步骤3：获取服务器地址 ==========
    if config.mode == "verl":
        # verl 分布式模式：启动多个 vLLM replica（跳过）
        rollout_servers, server_addresses = await start_rollout_servers(config)
    else:
        # ★ Simple 模式：连接已有的 vLLM 服务器 ★
        print(f"\n[Simple Mode] Connecting to {config.vllm_url}")
        # 去掉 http:// 前缀，因为后续代码会重新添加
        server_addresses = [config.vllm_url.replace("http://", "")]
        # 例如：["localhost:8000"]
        rollout_servers = None
```

**Simple 模式**的核心：
- 不启动任何服务器
- 直接使用用户提供的 `--vllm_url` 地址
- `server_addresses` 列表只有一个元素

---

### 2.4 初始化组件

```python
    # 第 1651-1664 行
    # ========== 步骤4：初始化组件 ==========

    # 1. 指标收集器：收集 accepted@1、pass_ratio 等统计信息
    metrics_collector = MetricsCollector()

    # 2. 问答日志：保存生成的代码和评测结果（用于调试）
    qa_logger = QALogger(
        output_dir / "qa_logs",
        sample_size=config.qa_sample_size  # 默认 20
    )

    # 3. WandB（可选）：实验追踪平台
    if config.use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, name=run_name)
        wandb.config.update(asdict(config))  # 记录配置

    all_metrics = {}  # 存储所有数据集的评测结果
```

#### asdict() 函数

```python
from dataclasses import asdict

# 将 dataclass 转换为字典
config_dict = asdict(config)
# {
#     "mode": "simple",
#     "model_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
#     "vllm_url": "http://localhost:8000",
#     ...
# }
```

---

### 2.5 主循环：评测每个数据集

```python
    # 第 1668-1699 行
    # ========== 步骤5：评测每个数据集 ==========
    try:
        for dataset_key in config.datasets:  # ["humaneval", "mbpp_reg"]
            print(f"\n[Loading {dataset_key}]")

            # 5.1 加载题目
            prompts = load_prompts(dataset_key, config)
            # prompts 是一个列表，每个元素包含：
            # {
            #     "problem_id": "HumanEval/0",
            #     "prompt": "def func():\n    ...",
            #     "sandbox_dataset": "HumanEval",
            #     "test_cases": {...}  # 可选
            # }

            if not prompts:
                print(f"  No prompts found, skipping...")
                continue

            print(f"  Loaded {len(prompts)} problems")

            # 5.2 评测数据集（核心！）
            dataset_metrics = await evaluate_dataset(
                dataset_key,       # "humaneval"
                prompts,           # 题目列表
                server_addresses,  # ["localhost:8000"]
                config,            # 配置
                metrics_collector, # 指标收集器
                qa_logger,         # 日志记录器
            )
            # ↑ await: 等待协程执行完成

            all_metrics[dataset_key] = dataset_metrics

# ★ evaluate_dataset() 返回值示例 ★
# dataset_metrics 是一个字典，包含该数据集的统计信息：
dataset_metrics = {
    # 基础统计
    "total_problems": 164,              # 总题数

    # 质量指标（核心）
    "accepted_at_1": 0.7256,            # 通过率 = 119/164
    "pass_ratio_mean": 0.8234,          # 平均测试用例通过比例
    "pass_ratio_p50": 1.0,              # 中位数（很多题全部通过）
    "pass_ratio_p90": 1.0,              # 90% 分位

    # Token 统计
    "total_gen_tokens": 40278,          # 总生成 token 数
    "avg_gen_tokens": 245.6,            # 平均每题生成 token 数

    # 时间统计
    "total_gen_time": 387.5,            # 总生成时间（秒）
    "avg_gen_time": 2.36,               # 平均每题生成时间
    "total_judge_time": 85.3,           # 总判题时间
    "avg_judge_time": 0.52,             # 平均每题判题时间
    "wall_clock_time": 45.2,            # 实际耗时（包含并发）

    # 效率指标
    "throughput": 3.63,                 # 吞吐量 = 164/45.2 题/秒
    "cost_per_solved_tokens": 338.5,    # 每解决一题消耗的 token
    "cost_per_solved_judge_time": 0.72, # 每解决一题的判题时间

    # 异常统计
    "truncation_count": 2,              # 被截断的题数
    "truncation_rate": 0.012,           # 截断率 = 2/164
    "timeout_count": 1,                 # 超时题数
    "timeout_rate": 0.006,              # 超时率

    # 错误分布
    "error_distribution": {
        "success": 119,
        "wrong_answer": 32,
        "runtime_error": 8,
        "syntax_error": 3,
        "timeout": 1,
        "empty_output": 1
    }
}

    finally:
        # 清理资源（verl 模式才需要）
        if rollout_servers:
            print("\n[Shutting down Rollout Servers]")
```

#### await 关键字详解

```python
# await 只能在 async 函数内使用
async def run_evaluation(config):
    # await 做了两件事：
    # 1. 等待协程执行完成
    # 2. 获取协程的返回值
    dataset_metrics = await evaluate_dataset(...)
    #                 ↑
    #     暂停当前协程，让出控制权给事件循环
    #     事件循环可以去执行其他协程
    #     当 evaluate_dataset 完成后，恢复执行
```

**执行流程图**：

```
run_evaluation()                    事件循环
      │                                 │
      ├── await evaluate_dataset() ────►│ 暂停 run_evaluation
      │                                 │ 执行 evaluate_dataset
      │                                 │   ├── await batch_generate()
      │                                 │   │      └── 执行 HTTP 请求
      │                                 │   └── 返回结果
      │◄────────────────────────────────┤ 恢复 run_evaluation
      │                                 │
      ├── 处理结果                       │
      └── 继续下一个数据集               │
```

#### try...finally 保证清理

```python
try:
    # 可能抛出异常的代码
    for dataset_key in config.datasets:
        dataset_metrics = await evaluate_dataset(...)
finally:
    # 无论是否异常，都会执行清理
    if rollout_servers:
        print("[Shutting down]")
```

---

### 2.6 保存结果

```python
    # 第 1705-1735 行
    # ========== 步骤6：保存结果 ==========

    # 6.1 处理 JSON 不支持的值（inf → null）
    def handle_inf(obj):
        """递归处理 inf 值"""
        if isinstance(obj, dict):
            return {k: handle_inf(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [handle_inf(v) for v in obj]
        elif isinstance(obj, float) and obj == float('inf'):
            return None  # JSON 不支持 inf
        return obj

    # 6.2 保存指标
    with open(output_dir / "metrics.json", 'w', encoding='utf-8') as f:
        json.dump(handle_inf(all_metrics), f, indent=2, ensure_ascii=False)

    # 6.3 保存问答日志
    qa_logger.save()

    # 6.4 保存详细统计
    summary = metrics_collector.get_summary()
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(handle_inf(summary), f, indent=2)
```

#### ★ all_metrics 最终结构示例 ★

```python
# 评测完所有数据集后，all_metrics 的结构：
all_metrics = {
    "humaneval": {
        "total_problems": 164,
        "accepted_at_1": 0.7256,        # 72.56% 通过率
        "pass_ratio_mean": 0.8234,
        "pass_ratio_p50": 1.0,
        "pass_ratio_p90": 1.0,
        "total_gen_tokens": 40278,
        "avg_gen_tokens": 245.6,
        "avg_gen_time": 2.36,
        "avg_judge_time": 0.52,
        "throughput": 3.63,
        "cost_per_solved_tokens": 338.5,
        "truncation_rate": 0.012,
        "timeout_rate": 0.006,
        "error_distribution": {
            "success": 119,
            "wrong_answer": 32,
            "runtime_error": 8,
            "syntax_error": 3,
            "timeout": 1,
            "empty_output": 1
        }
    },
    "mbpp_reg": {
        "total_problems": 200,
        "accepted_at_1": 0.685,          # 68.5% 通过率
        "pass_ratio_mean": 0.7856,
        "pass_ratio_p50": 1.0,
        "pass_ratio_p90": 1.0,
        "total_gen_tokens": 52340,
        "avg_gen_tokens": 261.7,
        "avg_gen_time": 2.89,
        "avg_judge_time": 0.48,
        "throughput": 3.21,
        "cost_per_solved_tokens": 382.0,
        "truncation_rate": 0.015,
        "timeout_rate": 0.005,
        "error_distribution": {
            "success": 137,
            "wrong_answer": 45,
            "runtime_error": 12,
            "syntax_error": 4,
            "timeout": 1,
            "empty_output": 1
        }
    }
}

# 这个结构会被保存到 metrics.json 文件
```

#### handle_inf() 递归函数

```python
# 问题：JSON 不支持 Python 的 float('inf')
json.dumps({"value": float('inf')})  # 报错！

# 解决：递归替换为 None
data = {
    "dataset1": {
        "pass_rate": 0.85,
        "cost_per_solved": float('inf')  # 没有通过的题目时
    }
}
handle_inf(data)
# → {"dataset1": {"pass_rate": 0.85, "cost_per_solved": None}}
```

---

### 2.7 本部分小结

**执行流程到目前为止**：

```
asyncio.run(run_evaluation(config))
       │
       ▼
run_evaluation() 协程开始执行
       │
       ├── 1. 打印配置信息
       │
       ├── 2. 创建输出目录 (Path.mkdir)
       │
       ├── 3. 获取服务器地址 (Simple: 直接用 vllm_url)
       │
       ├── 4. 初始化组件 (MetricsCollector, QALogger)
       │
       ├── 5. 主循环
       │      ├── load_prompts() ──────► 加载题目（下一部分讲解）
       │      │
       │      └── await evaluate_dataset() ──► 评测（第四部分讲解）
       │
       └── 6. 保存结果 (metrics.json, qa_logs, summary.json)
```

**关键概念**：
- `async def`: 定义协程函数
- **协程对象**: 调用协程函数返回的对象，需要 `await` 执行
- `await`: 暂停当前协程，等待另一个协程完成
- `pathlib.Path`: 现代化的路径处理
- `try...finally`: 保证清理代码执行
- `asdict()`: dataclass 转字典

---

**请确认你理解了第二部分后，我将继续讲解第三部分：load_prompts() 数据加载。**

---

## 第三部分：load_prompts() 数据加载与 Prompt 模板

### 3.1 执行流程回顾

```
run_evaluation()
    │
    for dataset_key in config.datasets:  # ["humaneval", "mbpp_reg"]
        │
        ▼
        prompts = load_prompts(dataset_key, config)  ← 我们现在在这里
        │
        ▼
        await evaluate_dataset(prompts, ...)
```

---

### 3.2 load_prompts() 入口函数

```python
# 第 1262-1276 行
def load_prompts(dataset_key: str, config: EvalConfig) -> List[Dict[str, Any]]:
    """
    加载评测数据

    两种数据源：
    1. manifest_dir: 从本地 manifest + raw 文件加载（包含测试用例）
    2. SandboxFusion: 从在线服务加载（仅 prompt）
    """
    if config.manifest_dir:
        # 优先使用本地数据（包含测试用例）
        return _load_from_manifest(dataset_key, config.manifest_dir)
    else:
        # 从 SandboxFusion 在线服务加载
        return _load_from_sandbox(dataset_key, config.sandbox_url)
```

**注意**：这是普通函数（不是 `async def`），因为文件读取是 CPU 操作，不需要异步。

#### 两种数据源对比

| 数据源 | 包含测试用例 | 适用场景 |
|--------|-------------|----------|
| 本地 manifest | ✅ 是 | 生产评测，需要外部测试用例 |
| SandboxFusion | ❌ 否 | 快速测试，使用内置测试用例 |

---

### 3.3 数据集配置映射

```python
# 第 341-362 行
DATASET_SANDBOX_CONFIG = {
    "humaneval": {
        "sandbox_dataset": "humaneval_python",  # SandboxFusion 中的名称
        "language": "python",
    },
    "mbpp_reg": {
        "sandbox_dataset": "mbpp",
        "language": "python",
        "id_range": (11, 210),  # MBPP Regular 子集（200题）
    },
    "codecontests_train": {
        "sandbox_dataset": "code_contests",
        "language": "python",
    },
    # ...
}
```

**作用**：将脚本内部的 `dataset_key` 映射到 SandboxFusion 的数据集名称。

| 脚本 dataset_key | SandboxFusion 名称 | 说明 |
|------------------|-------------------|------|
| `humaneval` | `humaneval_python` | HumanEval 164 题 |
| `mbpp_reg` | `mbpp` | MBPP Regular 200 题 (ID 11-210) |

---

### 3.4 从本地加载：_load_from_manifest()

```python
# 第 1327-1395 行
def _load_from_manifest(dataset_key: str, manifest_dir: str) -> List[Dict[str, Any]]:
    """
    文件结构：
    manifest_dir/
        humaneval_manifest.jsonl   # 去重后的 problem_id 列表
    manifest_dir/../raw/
        humaneval_raw.jsonl        # 完整数据（含测试用例）
    """
    manifest_path = Path(manifest_dir) / f"{dataset_key}_manifest.jsonl"
    raw_path = Path(manifest_dir).parent / "raw" / f"{dataset_key}_raw.jsonl"
```

#### 文件结构示意

```
data/
├── manifest/
│   ├── humaneval_manifest.jsonl   # {"problem_id": "HumanEval/0"}
│   └── mbpp_reg_manifest.jsonl    # {"problem_id": "11"}
└── raw/
    ├── humaneval_raw.jsonl        # 完整数据 + 测试用例
    └── mbpp_reg_raw.jsonl
```

#### 加载流程

```python
    # 步骤1：从 manifest 读取要评测的题目 ID（去重后的）
    problem_ids = set()
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            entry = json.loads(line)
            problem_ids.add(entry["problem_id"])
    # problem_ids = {"HumanEval/0", "HumanEval/1", ...}

    # 步骤2：从 raw 文件加载完整数据，只保留 manifest 中的题目
    result = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record["problem_id"] in problem_ids:  # 过滤
                item = {
                    "problem_id": record["problem_id"],
                    "prompt": record["prompt"],
                    "sandbox_dataset": sandbox_dataset,
                }
                # 加载测试用例（如果存在）
                if "test_cases" in record:
                    item["test_cases"] = record["test_cases"]
                result.append(item)

    return result
```

#### JSONL 格式

JSONL (JSON Lines)：每行一个 JSON 对象，便于流式处理大文件。

```jsonl
{"problem_id": "HumanEval/0", "prompt": "def func():\n    ..."}
{"problem_id": "HumanEval/1", "prompt": "def another():\n    ..."}
```

**优点**：
- 可以逐行读取，不需要一次加载整个文件到内存
- 便于追加写入
- 便于用 `grep`/`wc -l` 等工具处理

---

### 3.5 从 SandboxFusion 加载：_load_from_sandbox()

```python
# 第 1279-1324 行
def _load_from_sandbox(dataset_key: str, sandbox_url: str) -> List[Dict[str, Any]]:
    """从 SandboxFusion 在线服务加载数据"""

    # 检查 SDK 是否可用
    if not SANDBOX_AVAILABLE:
        print(f"  Warning: SandboxFusion SDK not available")
        return []

    cfg = DATASET_SANDBOX_CONFIG.get(dataset_key, {})
    sandbox_dataset = cfg.get("sandbox_dataset", dataset_key)
    id_range = cfg.get("id_range")  # MBPP Regular 的 ID 范围

    # 设置 SandboxFusion 服务器地址
    set_sandbox_endpoint(sandbox_url)

    # 调用 SDK 获取题目列表
    prompts = get_prompts(GetPromptsRequest(
        dataset=sandbox_dataset,
        config={"language": cfg.get("language", "python")}
    ))

    result = []
    for p in prompts:
        pid = str(p.id)

        # ID 范围过滤（用于 MBPP Regular 子集）
        if id_range:
            id_num = int(pid)
            if id_num < id_range[0] or id_num > id_range[1]:
                continue  # 跳过不在范围内的题目

        result.append({
            "problem_id": pid,
            "prompt": p.prompt,
            "sandbox_dataset": sandbox_dataset,
            # 注意：没有 test_cases！
        })

    return result
```

**注意**：从 SandboxFusion 加载的数据 **不包含测试用例**，评测时必须使用 SandboxFusion 的内置测试。

---

### 3.6 返回数据的结构

#### ★ load_prompts() 返回值示例 ★

```python
prompts = load_prompts("humaneval", config)
# prompts 是一个列表，每个元素代表一道题目：

# 示例：HumanEval 数据集的前 3 题
prompts = [
    # 第 1 题：检查列表中是否有两个数字足够接近
    {
        "problem_id": "HumanEval/0",
        "prompt": '''from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
        "sandbox_dataset": "humaneval_python",
        "test_cases": {  # 只有从 manifest 加载时才有
            "entry_point": "has_close_elements",
            "test_code": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0], 0.3) == True\n..."
        }
    },

    # 第 2 题：分离括号组
    {
        "problem_id": "HumanEval/1",
        "prompt": '''from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those group into separate strings and return the list of those.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
''',
        "sandbox_dataset": "humaneval_python",
        "test_cases": {
            "entry_point": "separate_paren_groups",
            "test_code": "..."
        }
    },

    # 第 3 题：截断数字的小数部分
    {
        "problem_id": "HumanEval/2",
        "prompt": '''def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).
    >>> truncate_number(3.5)
    0.5
    """
''',
        "sandbox_dataset": "humaneval_python",
        "test_cases": {
            "entry_point": "truncate_number",
            "test_code": "..."
        }
    },

    # ... 共 164 题
]
```

**从 SandboxFusion 加载时（无 test_cases）**：
```python
prompts = load_prompts("humaneval", config)  # config.manifest_dir 为 None
# 返回结构相同，但没有 test_cases 字段：
[
    {
        "problem_id": "HumanEval/0",
        "prompt": "from typing import List\n\ndef has_close_elements...",
        "sandbox_dataset": "humaneval_python"
        # 注意：没有 test_cases！评测时必须用 SandboxFusion 内置测试
    },
    # ...
]
```

---

### 3.7 Prompt 模板（format_prompt）

加载完 prompts 后，在 `evaluate_dataset()` 中会用模板格式化：

```python
# 第 1459-1470 行（evaluate_dataset 内部）
if dataset_key == "mbpp_reg":
    prompt_texts = [
        format_prompt(
            p["prompt"],
            dataset_key,
            p.get("test_cases", {}).get("entry_point", ""),
            p.get("test_cases", {}).get("example_call", "")
        )
        for p in batch
    ]
else:
    prompt_texts = [format_prompt(p["prompt"], dataset_key) for p in batch]
```

#### format_prompt() 函数

```python
# 第 282-310 行
def format_prompt(
    prompt: str,
    dataset_key: str,
    entry_point: str = "",
    example_call: str = ""
) -> str:
    """
    根据数据集类型格式化 prompt

    Args:
        prompt: 原始题目描述
        dataset_key: 数据集名称
        entry_point: 函数名（MBPP 需要）
        example_call: 调用示例（MBPP 需要）
    """
    template = PROMPT_TEMPLATES.get(dataset_key, PROMPT_TEMPLATES["humaneval"])

    # 替换占位符
    formatted = template.format(
        prompt=prompt,
        entry_point=entry_point,
        example_call=example_call
    )
    return formatted
```

#### HumanEval 模板

```python
# 第 214-229 行
PROMPT_TEMPLATES = {
    "humaneval": """Complete the following Python function.

Rules:
- Keep the function name, parameters, and docstring unchanged.
- Output a complete, executable Python code snippet that defines the function.
- Use only Python standard library (no pip packages).
- Do NOT read from stdin and do NOT print anything.
- Do NOT include "if __name__ == '__main__':" or any top-level execution.
- Do NOT define a function named "check" (it is reserved for tests).

{prompt}

Output ONLY:
<code>
# python code
</code>""",
```

#### MBPP 模板

```python
    "mbpp_reg": """Implement a Python function for the following task.

Task:
{prompt}

Rules:
- The function name MUST be: {entry_point}
- Your function will be called like: {example_call}
- Use only Python standard library (no pip packages).
- Do NOT read from stdin and do NOT print anything.
- Do NOT include "if __name__ == '__main__':" or any top-level execution.

Output ONLY:
<code>
# python code
</code>""",
```

#### ★ format_prompt() 返回值示例 ★

**输入（HumanEval 原始 prompt）**：
```python
prompt = '''from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer...
    """
'''

formatted = format_prompt(prompt, "humaneval")
```

**输出（格式化后发给模型的完整 prompt）**：
```
Complete the following Python function.

Rules:
- Keep the function name, parameters, and docstring unchanged.
- Output a complete, executable Python code snippet that defines the function.
- Use only Python standard library (no pip packages).
- Do NOT read from stdin and do NOT print anything.
- Do NOT include "if __name__ == '__main__':" or any top-level execution.
- Do NOT define a function named "check" (it is reserved for tests).

from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer...
    """

Output ONLY:
<code>
# python code
</code>
```

**MBPP 数据集的格式化示例**：
```python
# MBPP 需要额外的 entry_point 和 example_call
prompt = "Write a function to find the similar elements from the given two tuple lists."
formatted = format_prompt(
    prompt,
    "mbpp_reg",
    entry_point="similar_elements",
    example_call="similar_elements((3, 4, 5, 6), (5, 7, 4, 10))"
)
```

**输出**：
```
Implement a Python function for the following task.

Task:
Write a function to find the similar elements from the given two tuple lists.

Rules:
- The function name MUST be: similar_elements
- Your function will be called like: similar_elements((3, 4, 5, 6), (5, 7, 4, 10))
- Use only Python standard library (no pip packages).
- Do NOT read from stdin and do NOT print anything.
- Do NOT include "if __name__ == '__main__':" or any top-level execution.

Output ONLY:
<code>
# python code
</code>
```

---

### 3.8 System Prompt

```python
# 第 198-205 行
SYSTEM_PROMPT = """You are an expert Python programmer.

Output rules:
1. Output Python code only.
2. Include necessary imports only if needed.
3. Wrap the entire code in <code> and </code>.
4. Do not write anything outside the <code> tags.
5. Follow dataset-specific constraints given by the user prompt (function-only vs full program)."""
```

**作用**：作为模型的"角色设定"，在每次 API 调用时传入。

---

### 3.9 本部分小结

**数据加载流程**：

```
load_prompts(dataset_key, config)
        │
        ├── config.manifest_dir 存在?
        │       │
        │       ├── 是 → _load_from_manifest()
        │       │         ├── 读取 manifest.jsonl (problem_id 列表)
        │       │         ├── 读取 raw.jsonl (完整数据)
        │       │         └── 返回 [{problem_id, prompt, test_cases}, ...]
        │       │
        │       └── 否 → _load_from_sandbox()
        │                 ├── 调用 SandboxFusion SDK
        │                 └── 返回 [{problem_id, prompt}, ...]  (无 test_cases)
        │
        ▼
prompts 列表传给 evaluate_dataset()
        │
        ▼
format_prompt() 格式化每个 prompt
        │
        ▼
发送给 vLLM 生成代码
```

**关键概念**：
- **JSONL 格式**：每行一个 JSON，便于流式处理
- **Manifest + Raw 分离**：manifest 控制评测哪些题目，raw 存储完整数据
- **Prompt 模板**：`{prompt}` 占位符，统一格式化指令
- **数据集配置映射**：`DATASET_SANDBOX_CONFIG` 处理命名差异

---

**请确认你理解了第三部分后，我将继续讲解第四部分：evaluate_dataset() 评测流程与 asyncio.gather 并发。**

---

## 第四部分：evaluate_dataset() 评测流程与异步并发

这是整个脚本中 **异步编程最核心** 的部分，会详细讲解 `asyncio.gather`、`Semaphore`、`async with` 等概念。

### 4.1 执行流程回顾

```
run_evaluation()
    │
    for dataset_key in config.datasets:
        │
        prompts = load_prompts(...)     ← 第三部分已讲
        │
        ▼
        await evaluate_dataset(...)     ← 我们现在在这里
              │
              ├── 分批处理 (batch)
              │     │
              │     ├── await batch_generate()   ← 并发生成代码
              │     │         │
              │     │         └── asyncio.gather(*tasks)
              │     │               ├── generate_code() × N
              │     │               ├── generate_code() × N
              │     │               └── ...
              │     │
              │     └── 逐个判题 evaluate_with_*()
              │
              └── 返回 dataset_metrics
```

---

### 4.2 evaluate_dataset() 整体结构

```python
# 第 1399-1427 行
async def evaluate_dataset(
    dataset_key: str,
    prompts: List[Dict[str, Any]],
    server_addresses: List[str],
    config: EvalConfig,
    metrics_collector: MetricsCollector,
    qa_logger: QALogger,
) -> Dict[str, Any]:
    """
    评测单个数据集

    流程：
    1. 分批处理（避免内存溢出）
    2. 批量生成代码（并发请求）   ← 异步
    3. 逐个判题                  ← 同步
    4. 收集指标和日志
    5. 返回统计信息
    """
```

**注意**：函数签名是 `async def`，所以这是一个协程函数。

---

### 4.3 分批处理 (Batching)

```python
    # 第 1449-1454 行
    # 分批处理：每批 batch_size 个题目
    for batch_start in range(0, len(prompts), config.batch_size):
        batch_end = min(batch_start + config.batch_size, len(prompts))
        batch = prompts[batch_start:batch_end]
        # batch_size = 50，则：
        # 第1批: prompts[0:50]
        # 第2批: prompts[50:100]
        # ...
```

**为什么分批？**
- 避免一次性发送太多请求
- 内存占用可控
- 方便显示进度

```python
        print(f"  Processing batch {batch_start//config.batch_size + 1}/...")
```

---

### 4.4 批量生成代码：batch_generate() ★核心★

```python
        # 第 1471-1478 行
        gen_results = await batch_generate(
            server_addresses,           # ["localhost:8000"]
            config.model_path,          # "Qwen/Qwen2.5-Coder-7B-Instruct"
            prompt_texts,               # 格式化后的 prompts (50个)
            sampling_params,            # {temperature: 0.0, max_tokens: 2048}
            config.max_concurrent_requests,  # 64
            system_prompt=SYSTEM_PROMPT,
        )
```

#### ★ batch_generate() 返回值示例 ★

`gen_results` 是一个列表，每个元素是 `(completion, metadata)` 元组：

```python
gen_results = [
    # 第 1 题：成功生成
    (
        "<code>\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n</code>",
        {
            "gen_time": 2.34,              # 生成耗时（秒）
            "prompt_tokens": 156,          # 输入 token 数
            "completion_tokens": 89,       # 输出 token 数
            "total_tokens": 245,           # 总 token 数
            "finish_reason": "stop"        # 结束原因：stop=正常结束
        }
    ),
    # 第 2 题：成功生成（较长）
    (
        "<code>\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    result = []\n    current = []\n    depth = 0\n    for char in paren_string:\n        if char == '(':\n            depth += 1\n            current.append(char)\n        elif char == ')':\n            depth -= 1\n            current.append(char)\n            if depth == 0:\n                result.append(''.join(current))\n                current = []\n    return result\n</code>",
        {
            "gen_time": 3.12,
            "prompt_tokens": 178,
            "completion_tokens": 124,
            "total_tokens": 302,
            "finish_reason": "stop"
        }
    ),
    # 第 3 题：生成被截断（达到 max_tokens 限制）
    (
        "<code>\ndef truncate_number(number: float) -> float:\n    return number - int(number)\n# This function extracts the decimal part of a floating point number...",
        {
            "gen_time": 5.67,
            "prompt_tokens": 145,
            "completion_tokens": 2048,     # 达到上限
            "total_tokens": 2193,
            "finish_reason": "length"      # 因长度限制截断
        }
    ),
    # 第 4 题：请求失败
    (
        "",                                # 空字符串表示失败
        {
            "error": "timeout",            # 错误类型
            "gen_time": 300.0              # 超时时间
        }
    ),
    # ... 共 50 个元素，与 prompt_texts 一一对应
]
```

**finish_reason 可能的值**：

| finish_reason | 含义 |
|---------------|------|
| `stop` | 正常结束（遇到 stop token 或生成完整） |
| `length` | 达到 `max_tokens` 限制被截断 |
| `content_filter` | 被内容过滤器拦截 |

现在让我们深入 `batch_generate()` 内部：

```python
# 第 554-597 行
async def batch_generate(
    server_addresses: List[str],
    model_path: str,
    prompts: List[str],
    sampling_params: dict,
    max_concurrent: int = 64,
    system_prompt: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    批量生成代码，负载均衡到多个 replica
    """

    # ★ 步骤1：创建信号量（限制并发数）★
    semaphore = asyncio.Semaphore(max_concurrent)  # 最多 64 个并发

    # ★ 步骤2：创建 HTTP 会话（连接池）★
    async with aiohttp.ClientSession() as session:

        # ★ 步骤3：创建协程任务列表 ★
        tasks = []
        for i, prompt in enumerate(prompts):
            # Round-Robin 负载均衡
            server_idx = i % len(server_addresses)
            server_address = server_addresses[server_idx]

            # 注意：这里只是创建协程对象，还没有执行！
            task = generate_code(
                session, server_address, model_path, prompt,
                sampling_params, semaphore, system_prompt
            )
            tasks.append(task)
        # tasks = [coroutine1, coroutine2, ..., coroutine50]

        # ★ 步骤4：并发执行所有任务 ★
        results = await asyncio.gather(*tasks)
        # gather 会：
        # 1. 把所有协程注册到事件循环
        # 2. 并发执行它们
        # 3. 等待全部完成
        # 4. 按原顺序返回结果

    return results
```

---

### 4.5 asyncio.gather() 详解

```python
results = await asyncio.gather(*tasks)
```

**`*tasks` 是什么？**

```python
tasks = [coro1, coro2, coro3]
asyncio.gather(*tasks)  # 等价于 asyncio.gather(coro1, coro2, coro3)
# * 是解包运算符，把列表展开为多个参数
```

**asyncio.gather() 的行为**：

```
         asyncio.gather(coro1, coro2, coro3)
                        │
                        ▼
    ┌──────────────────────────────────────────┐
    │           事件循环 (Event Loop)            │
    │                                           │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │
    │  │  coro1  │ │  coro2  │ │  coro3  │    │
    │  │         │ │         │ │         │    │
    │  │ HTTP ───┼─┼─ HTTP ──┼─┼─ HTTP ──│    │  ← 同时发出请求
    │  │ 等待... │ │ 等待... │ │ 等待... │    │
    │  │         │ │         │ │         │    │
    │  │ 响应 ◄──┼─┼─ 响应 ◄─┼─┼─ 响应 ◄─│    │  ← 陆续收到响应
    │  │ 完成!   │ │ 完成!   │ │ 完成!   │    │
    │  └─────────┘ └─────────┘ └─────────┘    │
    │                                           │
    │        全部完成，返回 [r1, r2, r3]         │
    └──────────────────────────────────────────┘
```

**关键特性**：
1. **并发执行**：所有协程"同时"开始
2. **保持顺序**：结果按 tasks 的顺序返回，不是按完成顺序
3. **等待全部**：所有协程完成后才返回

```python
# 示例
tasks = [
    fetch_data("url1"),  # 耗时 3 秒
    fetch_data("url2"),  # 耗时 1 秒
    fetch_data("url3"),  # 耗时 2 秒
]
results = await asyncio.gather(*tasks)
# 总耗时约 3 秒（最慢的那个），不是 6 秒
# results = [result1, result2, result3]  ← 按原顺序
```

---

### 4.6 asyncio.Semaphore 并发控制

**问题**：如果同时发起 1000 个请求会怎样？
- vLLM 服务器可能过载
- 网络连接可能耗尽
- 内存可能不足

**解决方案**：Semaphore（信号量）

```python
# 第 578-579 行
semaphore = asyncio.Semaphore(max_concurrent)  # max_concurrent = 64
```

**Semaphore 是什么？**

想象一个停车场只有 64 个车位：

```
停车场（Semaphore = 64）
┌────────────────────────────────────────┐
│ 🚗🚗🚗🚗🚗🚗...（64个车位）              │
│                                         │
│ 进入：semaphore.acquire() → 车位 -1     │
│ 离开：semaphore.release() → 车位 +1     │
└────────────────────────────────────────┘

当车位 = 0 时：
  新车必须等待，直到有车离开
```

**在 generate_code() 中使用**：

```python
# 第 506 行
async with semaphore:  # 获取一个"车位"
    # ... 发送 HTTP 请求 ...
    # ... 等待响应 ...
# 离开 async with 时自动释放"车位"
```

**执行过程**：

```
协程 1-64:  async with semaphore → 成功，进入执行
协程 65:    async with semaphore → 等待...（车位满了）
协程 1 完成: 释放 semaphore
协程 65:    获得 semaphore → 开始执行
...
```

---

### 4.7 async with 详解

`async with` 是 **异步上下文管理器**，用于管理需要异步初始化/清理的资源。

#### 普通 with vs async with

```python
# 普通 with（同步）
with open("file.txt") as f:
    data = f.read()
# 离开 with 时自动调用 f.close()

# async with（异步）
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.text()
# 离开时自动清理连接
```

#### 代码中的三层 async with

```python
# 第 1 层：HTTP 会话管理
async with aiohttp.ClientSession() as session:
    #       ↑ 创建连接池
    #       离开时自动关闭所有连接

    # 第 2 层：并发控制（在 generate_code 中）
    async with semaphore:
        #       ↑ 获取许可（如果没有许可则等待）
        #       离开时自动释放许可

        # 第 3 层：单个 HTTP 请求
        async with session.post(url, json=data) as resp:
            #       ↑ 发送请求，等待响应
            #       离开时自动关闭响应体
            result = await resp.json()
```

#### async with 的本质

```python
async with something as x:
    # 使用 x
```

等价于：

```python
x = await something.__aenter__()  # 异步进入
try:
    # 使用 x
finally:
    await something.__aexit__()   # 异步退出（清理）
```

---

### 4.8 generate_code() 单个请求的实现

```python
# 第 468-551 行
async def generate_code(
    session: aiohttp.ClientSession,
    server_address: str,
    model_path: str,
    prompt: str,
    sampling_params: dict,
    semaphore: asyncio.Semaphore,
    system_prompt: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    通过 OpenAI-compatible API 调用 vLLM 生成代码
    """

    # ★ 并发控制：最多 64 个协程能同时进入这里 ★
    async with semaphore:
        start_time = time.time()

        # 构建 messages（OpenAI 格式）
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # ★ 发送 HTTP POST 请求 ★
            async with session.post(
                url=f"http://{server_address}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model_path,
                    "messages": messages,
                    **sampling_params  # temperature, max_tokens
                },
                timeout=aiohttp.ClientTimeout(total=300),  # 5分钟超时
            ) as resp:
                # ★ 这里发生了什么？★
                # 1. 发送请求（几乎瞬间完成）
                # 2. 等待服务器处理（协程在这里暂停）
                # 3. 事件循环去执行其他协程
                # 4. 响应到达后，恢复这个协程

                if resp.status != 200:
                    error_text = await resp.text()
                    return "", {"error": f"API error {resp.status}"}

                # ★ 读取响应体（又一次 I/O 等待）★
                data = await resp.json()

                # ★ vLLM API 响应示例 ★
                # data 的结构（OpenAI 兼容格式）：
                # {
                #     "id": "cmpl-abc123",
                #     "object": "chat.completion",
                #     "created": 1699000000,
                #     "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                #     "choices": [
                #         {
                #             "index": 0,
                #             "message": {
                #                 "role": "assistant",
                #                 "content": "<code>\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n</code>"
                #             },
                #             "finish_reason": "stop"
                #         }
                #     ],
                #     "usage": {
                #         "prompt_tokens": 156,
                #         "completion_tokens": 89,
                #         "total_tokens": 245
                #     }
                # }

                # 提取结果
                completion = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                return completion, {
                    "gen_time": time.time() - start_time,
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "finish_reason": data["choices"][0].get("finish_reason"),
                }

        except asyncio.TimeoutError:
            return "", {"error": "timeout"}
        except Exception as e:
            return "", {"error": str(e)}
```

#### ★ generate_code() 返回值示例 ★

单个 `generate_code()` 调用返回一个元组 `(completion, metadata)`：

**成功案例**：
```python
(
    # completion: vLLM 返回的原始内容（包含 <code> 标签）
    "<code>\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer\n    to each other than given threshold.\n    \"\"\"\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n</code>",

    # metadata: 生成的元信息
    {
        "gen_time": 2.34,           # 从发送请求到收到响应的耗时
        "prompt_tokens": 156,       # 输入 prompt 的 token 数
        "completion_tokens": 89,    # 生成内容的 token 数
        "total_tokens": 245,        # 总 token 数
        "finish_reason": "stop"     # 正常结束
    }
)
```

**失败案例**：
```python
# 超时
("", {"error": "timeout", "gen_time": 300.0})

# API 错误
("", {"error": "API error 503: Service Unavailable", "gen_time": 0.5})

# 网络错误
("", {"error": "Cannot connect to host localhost:8000", "gen_time": 0.1})
```

---

### 4.9 完整的并发执行流程图

```
batch_generate(prompts=[p1, p2, ..., p50], max_concurrent=64)
        │
        ▼
创建 Semaphore(64)
        │
        ▼
async with aiohttp.ClientSession() as session:
        │
        ▼
创建 50 个协程对象（还没执行）
tasks = [generate_code(p1), generate_code(p2), ..., generate_code(p50)]
        │
        ▼
await asyncio.gather(*tasks)
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    事件循环开始调度                                │
│                                                                   │
│  时刻 T0:                                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ coro1: async with semaphore → 获取(剩余63)                   │ │
│  │ coro2: async with semaphore → 获取(剩余62)                   │ │
│  │ ...                                                          │ │
│  │ coro50: async with semaphore → 获取(剩余14)                  │ │
│  │                                                              │ │
│  │ 全部 50 个协程都获得了 semaphore（因为 50 < 64）              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  时刻 T1:                                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ coro1: session.post() → 发送请求 → 等待响应...              │ │
│  │ coro2: session.post() → 发送请求 → 等待响应...              │ │
│  │ ...                                                          │ │
│  │ coro50: session.post() → 发送请求 → 等待响应...             │ │
│  │                                                              │ │
│  │ 50 个 HTTP 请求几乎同时发出！                                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  时刻 T2~T10:                                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ coro5:  响应到达 → await resp.json() → 完成 ✓               │ │
│  │ coro12: 响应到达 → await resp.json() → 完成 ✓               │ │
│  │ coro1:  响应到达 → await resp.json() → 完成 ✓               │ │
│  │ ...                                                          │ │
│  │ (响应陆续到达，完成顺序不确定)                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  时刻 T_end:                                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 所有 50 个协程都完成                                         │ │
│  │ gather 返回 [result1, result2, ..., result50]               │ │
│  │ (按原顺序，不是完成顺序)                                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

### 4.10 aiohttp.ClientSession 的作用

```python
async with aiohttp.ClientSession() as session:
```

**ClientSession 的优势**：

1. **连接池复用**：不需要每个请求都建立新连接
2. **Cookie 管理**：自动处理 Cookie
3. **配置共享**：headers、timeout 等可以统一配置

```
不用 Session（慢）:
请求1: TCP握手 → 发送 → 响应 → 关闭
请求2: TCP握手 → 发送 → 响应 → 关闭  （又要握手）
请求3: TCP握手 → 发送 → 响应 → 关闭

用 Session（快）:
请求1: TCP握手 → 发送 → 响应 → (保持连接)
请求2: (复用连接) → 发送 → 响应 → (保持连接)
请求3: (复用连接) → 发送 → 响应 → 关闭
```

---

### 4.11 判题部分（同步）

生成代码后，逐个判题：

```python
        # 第 1480-1513 行
        # 2. 逐个判题
        for i, (generated_code, gen_meta) in enumerate(gen_results):
            problem_id = batch[i]["problem_id"]
            test_cases = batch[i].get("test_cases")

            # 根据配置选择评测方式
            if test_cases and config.use_external_tests:
                # 方式1：使用外部测试用例 + run_code API
                eval_result = evaluate_with_run_code(
                    generated_code, test_cases, problem_id, config
                )
            else:
                # 方式2：使用 submit API（SandboxFusion 内置测试）
                eval_result = evaluate_with_submit_api(
                    generated_code, sandbox_dataset, problem_id, config
                )
```

**为什么判题是同步的？**

评测函数 `evaluate_with_submit_api()` 和 `evaluate_with_run_code()` 都是普通函数（不是 `async def`），因为：
1. SandboxFusion SDK 的 `submit_safe()` 是同步 API
2. 判题通常比生成快得多，优化收益小
3. 保持代码简单

---

### 4.12 本部分小结

**异步编程核心概念**：

| 概念 | 作用 | 代码位置 |
|------|------|----------|
| `async def` | 定义协程函数 | 第 468, 554, 1399 行 |
| `await` | 等待协程完成 | 第 595, 1471 行 |
| `asyncio.gather()` | 并发执行多个协程 | 第 595 行 |
| `asyncio.Semaphore` | 限制并发数量 | 第 579 行 |
| `async with` | 异步上下文管理器 | 第 506, 517, 581 行 |
| `aiohttp.ClientSession` | HTTP 连接池 | 第 581 行 |

**执行流程**：

```
evaluate_dataset()
    │
    for batch in batches:
        │
        ├── batch_generate()
        │       │
        │       ├── Semaphore(64) ← 并发控制
        │       │
        │       ├── ClientSession() ← 连接池
        │       │
        │       ├── 创建 N 个协程
        │       │
        │       └── asyncio.gather() ← 并发执行！
        │               │
        │               └── 50 个 HTTP 请求同时发出
        │                   响应陆续返回
        │                   全部完成后返回结果
        │
        └── 逐个判题（同步）
```

**关键理解**：
- `asyncio.gather()` 是实现并发的核心
- `Semaphore` 防止并发过多导致资源耗尽
- `async with` 管理异步资源的生命周期
- 协程在 I/O 等待时让出控制权，实现"并发"

---

**请确认你理解了第四部分后，我将继续讲解第五部分：代码评测（SandboxFusion）与结果统计。**

---

## 第五部分：代码评测（SandboxFusion）与结果统计

### 5.1 评测流程概览

```
batch_generate() 返回生成的代码
        │
        ▼
for i, (generated_code, gen_meta) in enumerate(gen_results):
        │
        ├── 选择评测方式
        │       │
        │       ├── 方式1: evaluate_with_submit_api()
        │       │          └── 使用 SandboxFusion 内置测试
        │       │
        │       └── 方式2: evaluate_with_run_code()
        │                  └── 使用外部测试用例
        │
        ├── 获取 EvalResult
        │
        ├── 收集指标 (metrics_collector)
        │
        └── 记录日志 (qa_logger)
```

---

### 5.2 EvalResult 数据结构

```python
# utils/metrics.py 第 28-38 行
@dataclass
class EvalResult:
    """单个问题的评测结果"""
    problem_id: str          # 问题 ID
    accepted: bool           # 是否通过所有测试用例
    pass_ratio: float        # 通过的测试用例比例 [0, 1]
    error_type: str          # 错误类型
    judge_time: float        # 判题耗时（秒）
    gen_tokens: int = 0      # 生成的 token 数
    gen_time: float = 0.0    # 生成耗时（秒）
    details: Dict[str, Any] = field(default_factory=dict)  # 额外信息
```

**error_type 可能的值**：

| error_type | 说明 |
|------------|------|
| `success` | 通过所有测试用例 |
| `syntax_error` | 语法错误（代码无法解析） |
| `runtime_error` | 运行时错误（异常、段错误等） |
| `timeout` | 执行超时 |
| `wrong_answer` | 结果错误（代码能运行但输出不对） |
| `api_error` | API 调用错误 |
| `empty_output` | 模型输出为空 |

#### ★ EvalResult 返回值示例 ★

**成功通过所有测试**：
```python
EvalResult(
    problem_id="HumanEval/0",
    accepted=True,                    # 全部通过
    pass_ratio=1.0,                   # 10/10 测试用例通过
    error_type="success",
    judge_time=0.45,                  # 判题耗时 0.45 秒
    gen_tokens=89,                    # 生成了 89 个 token
    gen_time=2.34,                    # 生成耗时 2.34 秒
    details={
        "extracted_code": "def has_close_elements(numbers, threshold):\n    ...",
        "test_count": 10
    }
)
```

**部分通过**：
```python
EvalResult(
    problem_id="HumanEval/15",
    accepted=False,                   # 未全部通过
    pass_ratio=0.7,                   # 7/10 测试用例通过
    error_type="wrong_answer",        # 答案错误
    judge_time=0.38,
    gen_tokens=156,
    gen_time=3.21,
    details={
        "extracted_code": "def split_words(txt):\n    ...",
        "test_count": 10,
        "failed_tests": [3, 5, 8]     # 失败的测试用例编号
    }
)
```

**运行时错误**：
```python
EvalResult(
    problem_id="HumanEval/42",
    accepted=False,
    pass_ratio=0.0,
    error_type="runtime_error",
    judge_time=0.12,
    gen_tokens=78,
    gen_time=1.89,
    details={
        "extracted_code": "def incr_list(l):\n    return [x + 1 for x in l]",
        "test_count": 5,
        "error_message": "TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'"
    }
)
```

**空输出**：
```python
EvalResult(
    problem_id="HumanEval/99",
    accepted=False,
    pass_ratio=0.0,
    error_type="empty_output",
    judge_time=0.01,
    gen_tokens=0,
    gen_time=0.5,
    details={}
)
```

---

### 5.3 evaluate_with_submit_api() 详解

```python
# 第 604-706 行
def evaluate_with_submit_api(
    completion: str,        # 模型生成的代码（包含 <code> 标签）
    sandbox_dataset: str,   # "humaneval_python"
    sandbox_id: str,        # "HumanEval/0"
    config: EvalConfig,
) -> EvalResult:
    """
    使用 SandboxFusion submit() API 评测代码

    submit() API 特点：
    - 依赖 SandboxFusion 内置的测试用例数据
    - 自动处理代码提取、编译、执行
    - 返回详细的测试结果
    """
```

**执行流程**：

```python
    # 1. 检查 SDK 可用性
    if not SANDBOX_AVAILABLE:
        return EvalResult(..., error_type="sdk_unavailable")

    start_time = time.time()

    # 2. 空输出检查
    if not completion or not completion.strip():
        return EvalResult(..., error_type="empty_output")

    try:
        # 3. 设置服务地址
        set_sandbox_endpoint(config.sandbox_url)  # "http://localhost:8080"

        # 4. 调用 submit API
        result = submit_safe(SubmitRequest(
            dataset=sandbox_dataset,   # "humaneval_python"
            id=sandbox_id,             # "HumanEval/0"
            completion=completion,     # 模型生成的代码
            config=TestConfig(
                language='python',
                run_timeout=config.run_timeout,  # 30 秒
            )
        ))

        # 5. 解析结果
        accepted = result.accepted        # 是否全部通过
        tests = result.tests or []        # 每个测试用例的结果

        # 6. 计算 pass_ratio
        if tests:
            passed = sum(1 for t in tests if t.status == "success")
            pass_ratio = passed / len(tests)
        else:
            pass_ratio = 1.0 if accepted else 0.0

        # 7. 确定错误类型
        error_type = "success" if accepted else _determine_error_type(tests)

        return EvalResult(
            problem_id=sandbox_id,
            accepted=accepted,
            pass_ratio=pass_ratio,
            error_type=error_type,
            judge_time=time.time() - start_time,
            details={
                "extracted_code": result.extracted_code,  # 提取的代码
                "test_count": len(tests),
            },
        )

    except Exception as e:
        return EvalResult(..., error_type="api_error")
```

---

### 5.4 SandboxFusion 服务架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    SandboxFusion 服务器                          │
│                    (localhost:8080)                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    submit() API                           │  │
│  │                                                           │  │
│  │  输入:                                                    │  │
│  │    - dataset: "humaneval_python"                         │  │
│  │    - id: "HumanEval/0"                                   │  │
│  │    - completion: "<code>def func():...</code>"           │  │
│  │                                                           │  │
│  │  处理步骤:                                                │  │
│  │    1. 代码提取：从 <code>...</code> 中提取代码            │  │
│  │    2. 加载测试用例：从内置数据集获取测试用例              │  │
│  │    3. 沙箱执行：在隔离环境中运行代码                      │  │
│  │    4. 结果比对：比较输出与预期结果                        │  │
│  │                                                           │  │
│  │  输出:                                                    │  │
│  │    - accepted: True/False                                │  │
│  │    - tests: [{status: "success"}, {status: "failed"}]    │  │
│  │    - extracted_code: "def func():..."                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  内置数据集:                                                     │
│    - humaneval_python (164 题 + 测试用例)                       │
│    - mbpp (974 题 + 测试用例)                                   │
│    - code_contests (...)                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

### 5.5 两种评测方式对比

```python
# 第 1498-1506 行
if test_cases and config.use_external_tests:
    # 方式1：使用外部测试用例 + run_code API
    eval_result = evaluate_with_run_code(
        generated_code, test_cases, problem_id, config
    )
else:
    # 方式2：使用 submit API（SandboxFusion 内置测试）
    eval_result = evaluate_with_submit_api(
        generated_code, sandbox_dataset, problem_id, config
    )
```

| 方式 | 测试用例来源 | 适用场景 |
|------|-------------|----------|
| `submit_api` | SandboxFusion 内置 | 快速测试，标准数据集 |
| `run_code` | 外部 manifest/raw 文件 | 自定义测试用例，离线评测 |

---

### 5.6 MetricsCollector 指标收集

```python
# utils/metrics.py 第 78-112 行
class MetricsCollector:
    """指标收集器：收集评测结果并计算统计指标"""

    def __init__(self):
        # 按数据集存储结果
        self._results: Dict[str, List[EvalResult]] = defaultdict(list)

        # 错误类型计数
        self._error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # 数据集的 wall_clock_time
        self._wall_clock_time: Dict[str, float] = {}

    def add_result(self, dataset: str, result: EvalResult):
        """添加单个评测结果"""
        self._results[dataset].append(result)
        self._error_counts[dataset][result.error_type] += 1
```

**在 evaluate_dataset() 中使用**：

```python
# 第 1515-1530 行（简化）
for i, (generated_code, gen_meta) in enumerate(gen_results):
    # ... 评测 ...
    eval_result = evaluate_with_submit_api(...)

    # 收集指标
    metrics_collector.add_result(dataset_key, eval_result)

    # 记录日志（采样）
    qa_logger.log(dataset_key, {
        "problem_id": problem_id,
        "prompt": prompt,
        "generated_code": generated_code,
        "accepted": eval_result.accepted,
        "pass_ratio": eval_result.pass_ratio,
        "error_type": eval_result.error_type,
    })

    results.append({
        "problem_id": problem_id,
        "accepted": eval_result.accepted,
        "pass_ratio": eval_result.pass_ratio,
        ...
    })
```

---

### 5.7 指标计算（evaluate_dataset 结尾）

```python
# 第 1550-1590 行
# 计算统计指标
accepted_count = sum(1 for r in results if r["accepted"])
pass_ratios = np.array([r["pass_ratio"] for r in results])

# 计算 throughput（吞吐量）
wall_clock_time = time.time() - dataset_start_time
throughput = len(results) / wall_clock_time  # 问题数/秒

# 计算 cost_per_solved（每解决一题的成本）
if accepted_count > 0:
    cost_per_solved_tokens = total_gen_tokens / accepted_count
    cost_per_solved_judge_time = total_judge_time / accepted_count
else:
    cost_per_solved_tokens = float('inf')      # 没有通过的题目
    cost_per_solved_judge_time = float('inf')

# 返回数据集级别的统计信息
dataset_metrics = {
    "total_problems": len(results),

    # 质量指标
    "accepted_at_1": accepted_count / len(results),  # 主指标！
    "pass_ratio_mean": float(np.mean(pass_ratios)),
    "pass_ratio_p50": float(np.median(pass_ratios)),
    "pass_ratio_p90": float(np.percentile(pass_ratios, 90)),

    # 成本指标
    "total_gen_tokens": total_gen_tokens,
    "avg_gen_tokens": total_gen_tokens / len(results),
    "total_judge_time": total_judge_time,
    "avg_judge_time": total_judge_time / len(results),
    "throughput": throughput,
    "cost_per_solved_tokens": cost_per_solved_tokens,
    "cost_per_solved_judge_time": cost_per_solved_judge_time,

    # 异常指标
    "truncation_rate": truncation_count / len(results),
    "timeout_rate": timeout_count / len(results),
}
```

---

### 5.8 指标解释

#### 质量指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| `accepted_at_1` | **主指标**：通过率 | 通过题数 / 总题数 |
| `pass_ratio_mean` | 平均通过测试用例比例 | mean(每题的 pass_ratio) |
| `pass_ratio_p50` | 中位数通过比例 | median(pass_ratios) |
| `pass_ratio_p90` | 90% 分位通过比例 | percentile(pass_ratios, 90) |

**为什么需要 pass_ratio？**

```
题目 A: 10 个测试用例，通过 10 个 → accepted=True,  pass_ratio=1.0
题目 B: 10 个测试用例，通过 9 个  → accepted=False, pass_ratio=0.9
题目 C: 10 个测试用例，通过 0 个  → accepted=False, pass_ratio=0.0

accepted_at_1 = 1/3 = 33.3%
pass_ratio_mean = (1.0 + 0.9 + 0.0) / 3 = 63.3%

pass_ratio 比 accepted 更能反映代码质量的"接近程度"
```

#### 成本指标

| 指标 | 含义 | 用途 |
|------|------|------|
| `avg_gen_tokens` | 平均生成 token 数 | 估算 API 成本 |
| `avg_judge_time` | 平均判题时间 | 评估判题效率 |
| `throughput` | 吞吐量（问题/秒） | 评估整体效率 |
| `cost_per_solved_tokens` | 每解决一题的 token 成本 | 性价比评估 |

---

### 5.9 QALogger 问答日志

```python
# 用于调试：保存生成的代码和评测结果
qa_logger = QALogger(
    output_dir / "qa_logs",
    sample_size=config.qa_sample_size  # 默认 20
)
```

**输出文件结构**：

```
outputs/phase0/qa_logs/
├── humaneval_samples.jsonl    # 采样的问答记录
└── mbpp_reg_samples.jsonl
```

**每条记录包含**：

```json
{
    "problem_id": "HumanEval/0",
    "prompt": "from typing import List\ndef has_close_elements...",
    "generated_code": "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers))...",
    "accepted": true,
    "pass_ratio": 1.0,
    "error_type": "success",
    "gen_tokens": 156,
    "gen_time": 2.34,
    "judge_time": 0.45
}
```

**用途**：
- 调试模型输出
- 分析错误类型
- 人工检查代码质量

---

### 5.10 最终输出文件

```
outputs/phase0/
├── metrics.json       # 每个数据集的统计指标
├── summary.json       # 详细统计（错误分布等）
└── qa_logs/
    ├── humaneval_samples.jsonl
    └── mbpp_reg_samples.jsonl
```

**metrics.json 示例**：

```json
{
    "humaneval": {
        "total_problems": 164,
        "accepted_at_1": 0.7256,
        "pass_ratio_mean": 0.8234,
        "pass_ratio_p50": 1.0,
        "pass_ratio_p90": 1.0,
        "avg_gen_tokens": 245.6,
        "avg_judge_time": 0.52,
        "throughput": 12.34,
        "cost_per_solved_tokens": 338.5,
        "truncation_rate": 0.012,
        "timeout_rate": 0.006
    },
    "mbpp_reg": {
        "total_problems": 200,
        "accepted_at_1": 0.685,
        ...
    }
}
```

---

### 5.11 本部分小结

**评测流程**：

```
模型生成代码
    │
    ▼
evaluate_with_submit_api() 或 evaluate_with_run_code()
    │
    ├── SandboxFusion 执行代码
    │
    ├── 比对测试结果
    │
    └── 返回 EvalResult
            │
            ├── accepted: 是否全部通过
            ├── pass_ratio: 通过比例
            └── error_type: 错误类型
    │
    ▼
MetricsCollector 收集指标
    │
    ▼
计算统计指标
    │
    ├── accepted_at_1（主指标）
    ├── pass_ratio_mean/p50/p90
    ├── avg_gen_tokens, avg_judge_time
    └── cost_per_solved, throughput
    │
    ▼
保存结果
    │
    ├── metrics.json
    ├── summary.json
    └── qa_logs/*.jsonl
```

**关键概念**：
- **EvalResult**: 单个问题的评测结果数据结构
- **SandboxFusion**: 代码沙箱服务，提供安全的代码执行环境
- **accepted_at_1**: 主要评测指标（通过率）
- **pass_ratio**: 比 accepted 更细粒度的质量评估
- **cost_per_solved**: 性价比指标

---

## 总结：完整执行流程

```
┌──────────────────────────────────────────────────────────────────────┐
│  python src/phase0_eval.py --mode simple --vllm_url http://localhost:8000 │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  main()                                                               │
│    ├── argparse 解析命令行参数                                        │
│    ├── 创建 EvalConfig                                               │
│    └── asyncio.run(run_evaluation(config))                           │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  run_evaluation() [async]                                             │
│    ├── 获取服务器地址 (Simple: 直接用 vllm_url)                       │
│    ├── 初始化组件 (MetricsCollector, QALogger)                       │
│    │                                                                  │
│    ├── for dataset_key in ["humaneval", "mbpp_reg"]:                 │
│    │     │                                                            │
│    │     ├── prompts = load_prompts()                                │
│    │     │     └── 从 manifest/raw 或 SandboxFusion 加载             │
│    │     │                                                            │
│    │     └── await evaluate_dataset()                                │
│    │           │                                                      │
│    │           ├── for batch in batches:                             │
│    │           │     │                                                │
│    │           │     ├── format_prompt() 格式化                      │
│    │           │     │                                                │
│    │           │     ├── await batch_generate()                      │
│    │           │     │     │                                          │
│    │           │     │     ├── Semaphore(64)                         │
│    │           │     │     ├── ClientSession()                       │
│    │           │     │     └── asyncio.gather(*tasks)               │
│    │           │     │           └── 并发调用 vLLM API               │
│    │           │     │                                                │
│    │           │     └── for 每个生成结果:                           │
│    │           │           └── evaluate_with_submit_api()            │
│    │           │                 └── 调用 SandboxFusion 判题         │
│    │           │                                                      │
│    │           └── 返回 dataset_metrics                              │
│    │                                                                  │
│    └── 保存结果                                                       │
│          ├── metrics.json                                            │
│          ├── summary.json                                            │
│          └── qa_logs/*.jsonl                                         │
└──────────────────────────────────────────────────────────────────────┘
```

**核心技术点**：
1. **asyncio** - 异步编程框架
2. **asyncio.gather** - 并发执行多个协程
3. **Semaphore** - 并发数量控制
4. **aiohttp** - 异步 HTTP 客户端
5. **SandboxFusion** - 代码沙箱服务

---

**至此，Simple 模式评测代码的讲解全部完成！**

如果有任何问题，欢迎继续提问。
