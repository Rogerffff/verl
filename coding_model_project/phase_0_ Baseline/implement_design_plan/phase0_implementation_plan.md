# Phase 0: Baseline 详细实施计划

---

> **📚 相关文档**：本文档的技术细节（特别是 verl Standalone Rollout 模式的代码执行流程）可参考 [verl_standalone_rollout_guide.md](./verl_standalone_rollout_guide.md)，该文档提供了更深入的源码级解析。

---

## 一、Phase 0 概述

### 1.1 目标定位

Phase 0 是整个 RLVR Coding Model 项目的起点，其核心目标是：

1. **建立对照基准**：为后续所有阶段（SFT、DPO、GRPO）提供可信的性能参照点
2. **验证评测流水线**：确保 SandboxFusion 判题系统工作正常
3. **收集成本基线**：记录推理吞吐量、判题时间等指标
4. **验证数据治理**：确保数据划分正确、无泄漏

### 1.2 Phase 0 产出清单

| 产出类型 | 具体内容 | 重要性 |
|---------|---------|--------|
| **质量指标** | accepted@1, pass_ratio(mean/p50/p90), exec_success_rate, error breakdown | ★★★ |
| **成本指标** | avg_total_gen_tokens, avg_total_judge_time, throughput, cost_per_solved | ★★★ |
| **数据治理** | manifest 文件, 去重报告, 泄漏检查报告 | ★★★ |
| **问答日志** | 120 条详细日志（按数据集分层抽样） | ★★ |
| **WandB 面板** | 基线指标记录 | ★★ |

### 1.3 评测数据集

| 数据集 | 角色 | 样本数（预估） | 评测目的 |
|--------|------|---------------|---------|
| CodeContests_valid | Dev/Val | ~200-500 | 主验证集，高频回归 |
| CodeContests_test | Test | ~200-500 | 最终评测，禁止调参 |
| HumanEval | Test only | 164 | 行业对标基线 |
| MBPP_reg | Dev/Val | 100-200 | 回归监控基线 |

---

## 二、verl 框架核心概念（教学内容）

在开始实施之前，你需要理解 verl 框架的核心架构和数据流。

### 2.1 verl 整体架构

#### 训练模式（GRPO/PPO）架构
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              PPO Ray Trainer                                  │
│                        (verl/trainer/ppo/ray_trainer.py)                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         训练循环 (Training Loop)                         │ │
│  │  1. 生成序列 (Rollout)  ─────────────────────────────────────────────>   │ │
│  │  2. 计算奖励 (Reward)   <─────────────────────────────────────────────   │ │
│  │  3. 计算优势 (Advantage)                                                 │ │
│  │  4. 更新策略 (Policy Update)                                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Rollout Worker    │    │    Actor Worker     │    │   Reward Manager    │
│  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │
│  │    vLLM       │  │    │  │    FSDP       │  │    │  │   compute_    │  │
│  │      or       │  │    │  │      or       │  │    │  │   score()     │  │
│  │   SGLang      │  │    │  │  Megatron-LM  │  │    │  │               │  │
│  └───────────────┘  │    │  └───────────────┘  │    │  └───────────────┘  │
│                     │    │                     │    │         │           │
│  - 序列生成          │    │  - 策略更新          │    │         ▼           │
│  - 权重同步          │    │  - log_prob 计算     │    │  ┌───────────────┐  │
└─────────────────────┘    └─────────────────────┘    │  │ SandboxFusion │  │
                                                       │  │   API 调用     │  │
                                                       │  └───────────────┘  │
                                                       └─────────────────────┘
```

#### Phase 0 评测架构（Standalone 模式）

**重要**：Phase 0 是纯评测阶段，不涉及训练，因此使用 **Standalone Rollout** 模式。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Phase 0 评测流程                                     │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         Ray 集群协调                                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│            ┌───────────────────────┼───────────────────────┐                │
│            │                       │                       │                │
│            ▼                       ▼                       ▼                │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│   │  vLLM Replica 0 │     │  vLLM Replica 1 │     │  vLLM Replica N │      │
│   │  (GPU 0-1, TP=2)│     │  (GPU 2-3, TP=2)│     │  (GPU 2N-2N+1)  │      │
│   │                 │     │                 │     │                 │      │
│   │  HTTP Server    │     │  HTTP Server    │     │  HTTP Server    │      │
│   │  ip:port_0      │     │  ip:port_1      │     │  ip:port_N      │      │
│   └────────┬────────┘     └────────┬────────┘     └────────┬────────┘      │
│            │                       │                       │                │
│            └───────────────────────┴───────────────────────┘                │
│                                    │                                         │
│                                    ▼                                         │
│                     ┌─────────────────────────────┐                         │
│                     │   OpenAI-Compatible API     │                         │
│                     │   POST /v1/chat/completions │                         │
│                     └──────────────┬──────────────┘                         │
│                                    │                                         │
│                                    ▼                                         │
│                     ┌─────────────────────────────┐                         │
│                     │   Code Generation Results   │                         │
│                     │   (completions)             │                         │
│                     └──────────────┬──────────────┘                         │
│                                    │                                         │
│                                    ▼                                         │
│                     ┌─────────────────────────────┐                         │
│                     │   SandboxFusion 判题        │                         │
│                     │   方式A: submit() API       │                         │
│                     │   方式B: compute_score()    │                         │
│                     └──────────────┬──────────────┘                         │
│                                    │                                         │
│                                    ▼                                         │
│                     ┌─────────────────────────────┐                         │
│                     │   Metrics + QA Logs         │                         │
│                     │   - accepted@1              │                         │
│                     │   - pass_ratio              │                         │
│                     │   - error_breakdown         │                         │
│                     └─────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Phase 0 关注的组件

由于 Phase 0 是纯评测阶段（不训练），我们主要关注：

| 组件 | 文件位置 | Phase 0 用途 |
|------|----------|-------------|
| **Rollout Worker** | `verl/verl/workers/rollout/` | 使用 vLLM/SGLang 生成代码 |
| **Reward Manager** | `verl/verl/workers/reward_manager/` | 调用 SandboxFusion 计算 pass_ratio |
| **compute_score** | `verl/verl/utils/reward_score/sandbox_fusion/__init__.py` | 核心评分逻辑 |
| **check_correctness** | `verl/verl/utils/reward_score/sandbox_fusion/utils.py` | 逐测试用例判题 |

### 2.3 代码执行流程（Phase 0 视角）- 使用 verl 分布式架构

**核心要点**：Phase 0 评测使用 verl 的 **Standalone Rollout 模式**，通过 vLLM/SGLang 引擎提供高效的分布式推理。

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         1. Ray 集群初始化                                   │
│  ray.init(runtime_env={"env_vars": {...}})                                 │
│  设置环境: TOKENIZERS_PARALLELISM, NCCL_DEBUG, VLLM_USE_V1                 │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    2. 创建 Standalone Rollout Server                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  rollout_class = get_rollout_replica_class("vllm")  # 或 "sglang"   │  │
│  │                                                                      │  │
│  │  rollout_servers = [                                                 │  │
│  │      rollout_class(                                                  │  │
│  │          replica_rank=i,                                             │  │
│  │          config=RolloutConfig(...),      # 推理配置                  │  │
│  │          model_config=HFModelConfig(...) # 模型配置                  │  │
│  │      ) for i in range(num_replicas)                                  │  │
│  │  ]                                                                   │  │
│  │                                                                      │  │
│  │  await asyncio.gather(*[                                             │  │
│  │      server.init_standalone()  # 初始化独立推理服务器                │  │
│  │      for server in rollout_servers                                   │  │
│  │  ])                                                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  结果: 每个 replica 启动一个 HTTP 服务器，提供 OpenAI 兼容 API            │
│  server_addresses = ["ip:port", ...]                                       │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                       3. 加载评测数据集                                     │
│  data = pd.read_parquet("data/eval/codecontests_valid.parquet")            │
│  格式: {prompt, ground_truth (test_cases JSON), problem_id, ...}           │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│               4. 通过 OpenAI API 调用 vLLM/SGLang 生成代码                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  async with aiohttp.ClientSession() as session:                      │  │
│  │      response = await session.post(                                  │  │
│  │          f"http://{server_address}/v1/chat/completions",             │  │
│  │          json={                                                      │  │
│  │              "model": model_path,                                    │  │
│  │              "messages": [{"role": "user", "content": prompt}],      │  │
│  │              "temperature": 0.0,  # EVAL@1 协议: greedy               │  │
│  │              "max_tokens": 2048                                      │  │
│  │          }                                                           │  │
│  │      )                                                               │  │
│  │  completion = response.choices[0].message.content                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  特点: 异步并发请求，支持多 replica 负载均衡                              │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    5. 代码提取与预处理                                      │
│  从 completion 中提取代码块，处理 ```python ... ``` 格式                   │
│  应用 Guardrails: 空输出检测、超长截断、非代码过滤                         │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    6. SandboxFusion 判题                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  score, metadata = compute_score(                                    │  │
│  │      sandbox_fusion_url=sandbox_url,                                 │  │
│  │      completion=code,                                                │  │
│  │      test_cases=test_cases,                                          │  │
│  │      timeout=10,                                                     │  │
│  │      memory_limit_mb=1024                                            │  │
│  │  )                                                                   │  │
│  │                                                                      │  │
│  │  # score = pass_ratio (0.0 ~ 1.0)                                    │  │
│  │  # metadata = [每个测试用例的详细结果]                               │  │
│  │  #   - status: success/wrong_answer/runtime_error/timeout/compile    │  │
│  │  #   - duration: 执行时间                                            │  │
│  │  #   - stdout/stderr: 输出信息                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  返回码: True(通过) / False(WA) / -1(API错误) / -2(RE) / -3(TLE) / -4(CE)  │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    7. 指标聚合与日志记录                                    │
│  - 质量指标: accepted@1, pass_ratio (mean/p50/p90), exec_success_rate      │
│  - 成本指标: avg_gen_tokens, avg_judge_time, throughput, cost_per_solved   │
│  - 错误分布: syntax_error_rate, runtime_error_rate, timeout_rate, wa_rate  │
│  - WandB 记录 + JSONL 问答日志                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 verl Standalone 模式详解

**为什么使用 Standalone 模式？**

verl 提供三种 Rollout 模式：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **HYBRID** | Rollout 与训练引擎融合在同一进程 | On-policy 训练（GRPO/PPO） |
| **COLOCATED** | Rollout 与 Hybrid 引擎共享 GPU，独立进程 | GRM (LLM as a Judge) |
| **STANDALONE** | 独立 GPU 资源，disaggregated 架构 | **Phase 0 评测**、Off-policy 训练 |

对于 Phase 0 纯评测场景，**STANDALONE 模式**是最佳选择：
- 不需要训练引擎（无梯度计算）
- 可以独占 GPU 资源最大化推理吞吐
- 通过 HTTP API 提供灵活的接入方式

**关键代码路径**：

```
verl/verl/workers/rollout/replica.py
├── RolloutReplica (基类)
│   ├── init_standalone()  ← Phase 0 使用这个方法
│   ├── init_hybrid()      ← GRPO/PPO 训练使用
│   └── init_colocated()   ← GRM 使用
│
├── get_rollout_replica_class("vllm")
│   └── returns vLLMReplica
│
└── get_rollout_replica_class("sglang")
    └── returns SGLangReplica

verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
├── vLLMReplica
│   ├── launch_servers()   ← 启动 HTTP 服务器
│   └── generate()         ← Token-in-token-out 生成
│
└── vLLMHttpServer         ← 单节点 HTTP 服务器

# 官方参考实现（推荐参考）
verl/verl/trainer/main_generation_server.py
├── start_server()         ← 启动多 replica 服务器
├── submit_request()       ← OpenAI API 调用示例
└── generate()             ← 批量生成流程
```

> **💡 提示**：`main_generation_server.py` 是 verl 官方提供的 Standalone 模式参考实现，Phase 0 评测脚本可直接参考该文件的实现方式。详细解析见 [verl_standalone_rollout_guide.md](./verl_standalone_rollout_guide.md) 第五章。

### 2.5 SandboxFusion 返回值详解

```python
# check_correctness() 返回值
(results_list, metadata_list) = check_correctness(...)

# results_list 元素含义：
#   True:  测试通过
#   False: Wrong Answer（能运行但输出不对）
#   -1:    API 错误 / Sandbox 内部错误
#   -2:    Runtime Error（运行时崩溃）
#   -3:    Timeout（超时）
#   -4:    Compile Error（编译/语法错误）

# metadata_list 每个元素的结构：
{
    "case_index": 0,
    "input": "1 2\n",
    "expected_output": "3\n",
    "status": "success" | "wrong_answer" | "runtime_error" | "timeout" | "compile_error" | "api_error",
    "stdout": "3\n",
    "stderr": "",
    "exit_code": 0,
    "duration": 0.05,  # 执行时间（秒）
    "compile_duration": 0.01,
    "compile_stderr": None,
}
```

### 2.6 为什么不使用纯 HuggingFace？

| 方面 | 纯 HuggingFace | verl 分布式架构 |
|------|---------------|----------------|
| **模型加载** | `AutoModelForCausalLM.from_pretrained()` | vLLM/SGLang 引擎自动加载 |
| **推理** | `model.generate()` 同步串行 | HTTP API 异步并发 |
| **并行度** | 单 GPU 或 device_map="auto" | 真正的 Tensor Parallel |
| **吞吐量** | 低（无 PagedAttention） | 高（KV Cache 优化） |
| **与 GRPO 一致性** | 不一致 | **完全一致** |
| **可扩展性** | 受限 | 多节点多 replica |

**关键原因**：Phase 0 的基线需要与后续 Phase（SFT、GRPO）的评测保持一致。使用 verl 架构可以确保：

1. **公平对比**：同样的推理引擎和解码参数
2. **代码复用**：评测脚本可直接用于 GRPO 训练中的 rollout 阶段
3. **性能基线**：throughput 指标具有参考价值

---

## 三、SandboxFusion 使用指南（教学内容）

### 3.1 SandboxFusion 架构

SandboxFusion 是字节跳动开发的安全代码沙盒系统，支持：
- 30+ 编程语言
- 13+ 评测数据集（HumanEval, MBPP, CodeContests 等）
- 进程隔离与资源限制

### 3.2 核心 API 端点

| 端点 | 方法 | 用途 |
|------|------|------|
| `/run_code` | POST | 执行单段代码 |
| `/list_datasets` | GET | 列出可用数据集 |
| `/get_prompts` | POST | 获取数据集题目 |
| `/submit` | POST | 提交代码评测 |

### 3.3 两种使用模式

**模式 A：直接使用 `/run_code`（verl 默认方式）**

```python
# verl 使用这种方式：直接调用 /run_code 执行代码
# 测试用例通过 stdin/stdout 传入传出

payload = {
    "compile_timeout": 10,
    "run_timeout": 10,
    "code": "a, b = map(int, input().split())\nprint(a + b)",
    "stdin": "1 2\n",
    "memory_limit_MB": 1024,
    "language": "python",
}
response = requests.post(f"{sandbox_url}/run_code", json=payload)
# 响应包含 stdout, stderr, status, duration 等
```

**模式 B：使用 Dataset API（SandboxFusion SDK）**

```python
from sandbox_fusion import get_prompts, submit, GetPromptsRequest, SubmitRequest, TestConfig

# 1. 获取数据集题目
prompts = get_prompts(GetPromptsRequest(
    dataset='codecontests',
    config=TestConfig(language='python', locale='en')
))

# 2. 提交评测
result = submit(SubmitRequest(
    dataset='codecontests',
    id='problem_id',
    completion='<model generated code>',
    config=TestConfig(language='python', run_timeout=20)
))

# 3. 获取结果
print(result.accepted)  # True/False
print(result.tests)     # 每个测试用例的详细结果
```

### 3.4 数据集类型对比

| 类型 | 代表数据集 | 评测方式 | 数据结构 |
|------|-----------|---------|---------|
| **AutoEval** | HumanEval, MBPP | 函数测试 | `{id, content, test, canonical_solution}` |
| **CommonOJ** | CodeContests | stdin/stdout | `{id, content, test_cases: [{input, output}]}` |

### 3.5 Phase 0 推荐使用模式

对于 Phase 0，有两种评测方式可选：

#### 方式 A：使用 `submit()` API（推荐，更简单）

```python
from sandbox_fusion import submit, SubmitRequest, TestConfig

# 直接提交评测，无需管理测试用例
result = submit(SubmitRequest(
    dataset='humaneval',              # SandboxFusion 数据集名
    id=record['sandbox_id'],          # 问题 ID
    completion=generated_code,
    config=TestConfig(language='python', run_timeout=10)
))

print(f"Accepted: {result.accepted}")
```

**优点**：
- 无需自己管理测试用例
- 与数据获取使用相同的 SandboxFusion SDK
- 代码简洁

#### 方式 B：使用 `compute_score()` 函数

```python
from verl.utils.reward_score.sandbox_fusion import compute_score

score, metadata = compute_score(
    sandbox_fusion_url=sandbox_url,
    completion=code,
    test_cases=test_cases,  # 需要自己传入测试用例
    timeout=10,
)
```

**优点**：
- 返回的 metadata 包含详细的执行信息（duration, status 等）
- 与后续 Phase 的 GRPO 训练代码保持一致

**建议**：Phase 0 优先使用 `submit()` API 简化实现，但在代码中保留对 `compute_score()` 的支持，以便与 GRPO 阶段对比。

---

## 四、数据获取与治理

### 4.1 数据集获取步骤

#### 推荐方式：从 SandboxFusion SDK 获取（Phase 0）

使用 SandboxFusion SDK 获取数据是最简单的方式，因为数据格式已经适配评测系统：

```python
# scripts/download_datasets_from_sandbox.py
from sandbox_fusion import get_prompts, GetPromptsRequest, TestConfig
from pathlib import Path
import json

def download_from_sandbox(output_dir: str = "data/sandbox"):
    """从 SandboxFusion SDK 获取所有需要的数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CodeContests (valid + test)
    for split in ['valid', 'test']:
        prompts = get_prompts(GetPromptsRequest(
            dataset='code_contests',
            config=TestConfig(language='python', locale='en', extra={'split': split}),
            offset=0, limit=100000
        ))

        records = []
        for item in prompts.prompts:
            records.append({
                "dataset": "codecontests",
                "split": split,
                "problem_id": item.id,
                "prompt": item.prompt,
                "sandbox_dataset": "code_contests",
                "sandbox_id": item.id,
            })

        save_path = output_path / f"codecontests_{split}.jsonl"
        with open(save_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"  {split}: {len(records)} samples -> {save_path}")

    # HumanEval
    prompts = get_prompts(GetPromptsRequest(
        dataset='humaneval',
        config=TestConfig(language='python')
    ))
    records = [{"dataset": "humaneval", "split": "test",
                "problem_id": f"HumanEval/{item.id}", "prompt": item.prompt,
                "sandbox_dataset": "humaneval", "sandbox_id": item.id}
               for item in prompts.prompts]
    with open(output_path / "humaneval.jsonl", 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # MBPP (回归子集 ID 11-210)
    prompts = get_prompts(GetPromptsRequest(
        dataset='mbpp',
        config=TestConfig(is_fewshot=False)
    ))
    records = [{"dataset": "mbpp", "split": "test",
                "problem_id": f"MBPP/{item.id}", "prompt": item.prompt,
                "sandbox_dataset": "mbpp", "sandbox_id": item.id}
               for item in prompts.prompts
               if 11 <= int(item.id) <= 210]
    with open(output_path / "mbpp_reg.jsonl", 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    return output_path

if __name__ == "__main__":
    download_from_sandbox()
```

**优点**：
- 数据格式已适配 `submit()` API 评测
- 无需自己管理测试用例
- 代码简洁

#### 备选方式：从 HuggingFace 下载

如果需要测试用例（用于 `compute_score()`）或 SandboxFusion 数据库中缺少某些数据：

```python
from datasets import load_dataset

codecontests = load_dataset("deepmind/code_contests")
humaneval = load_dataset("openai_humaneval")
mbpp = load_dataset("mbpp")
```

### 4.2 数据预处理流程

```
┌─────────────────┐
│ 1. 原始数据下载  │  HuggingFace Datasets
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. 格式标准化   │  转换为统一的 {prompt, test_cases, metadata} 格式
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Canonicalize │  规范化 prompt 文本（去空白、统一换行符）
│                 │  计算 prompt_sha256
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Split 内去重  │  按 prompt_sha256 去重
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. 跨 Split 检查 │  确保 train ∩ valid = ∅
│                 │  确保 train ∩ test = ∅
│                 │  确保 valid ∩ test = ∅
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. 外部泄漏检查  │  CodeContests_train ∩ HumanEval = ∅
│                 │  CodeContests_train ∩ MBPP_reg = ∅
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 7. 生成 Manifest │  保存到 data_manifests/
└─────────────────┘
```

### 4.3 Manifest 文件格式

```python
# data_manifests/codecontests_valid.jsonl
# 每行一条记录

{
    "dataset": "codecontests",
    "split": "valid",
    "problem_id": "cc_valid_001",
    "prompt_sha256": "a1b2c3d4e5f6...",
    "prompt_length": 1234,
    "num_test_cases": 10,
    "version": "2024-01-31",
    "source_url": "https://huggingface.co/datasets/deepmind/code_contests"
}
```

### 4.4 数据治理脚本框架

```python
# 伪代码：scripts/data_governance.py

import hashlib
import json
from datasets import load_dataset
from typing import Dict, List, Set

def canonicalize_prompt(prompt: str) -> str:
    """规范化 prompt 文本"""
    # 1. 统一换行符
    prompt = prompt.replace('\r\n', '\n')
    # 2. 去除首尾空白
    prompt = prompt.strip()
    # 3. 多个空格压缩为单个（保留必要格式）
    # ... 根据需要添加更多规则
    return prompt

def compute_hash(text: str) -> str:
    """计算 SHA256 哈希"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def deduplicate_split(records: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """Split 内去重，返回 (去重后, 重复记录)"""
    seen_hashes: Set[str] = set()
    unique, duplicates = [], []

    for record in records:
        h = record['prompt_sha256']
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(record)
        else:
            duplicates.append(record)

    return unique, duplicates

def check_cross_split_overlap(split_a: List[Dict], split_b: List[Dict]) -> List[str]:
    """检查跨 split 重叠，返回重叠的 hash 列表"""
    hashes_a = {r['prompt_sha256'] for r in split_a}
    hashes_b = {r['prompt_sha256'] for r in split_b}
    return list(hashes_a & hashes_b)

def generate_manifest(records: List[Dict], output_path: str):
    """生成 manifest 文件"""
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

# 主流程
def main():
    # 1. 加载数据
    codecontests = load_dataset("deepmind/code_contests")

    # 2. 处理每个 split
    splits = {}
    for split_name in ['train', 'valid', 'test']:
        records = []
        for item in codecontests[split_name]:
            canonical = canonicalize_prompt(item['description'])
            records.append({
                'dataset': 'codecontests',
                'split': split_name,
                'problem_id': item['name'],
                'prompt_sha256': compute_hash(canonical),
                'num_test_cases': len(item.get('public_tests', {}).get('input', [])),
                'version': '2024-01-31',
            })

        # 3. Split 内去重
        unique, dups = deduplicate_split(records)
        splits[split_name] = unique

        # 保存去重记录
        if dups:
            generate_manifest(dups, f'data_manifests/duplicates_{split_name}.jsonl')

    # 4. 跨 split 检查
    overlaps_train_valid = check_cross_split_overlap(splits['train'], splits['valid'])
    overlaps_train_test = check_cross_split_overlap(splits['train'], splits['test'])

    assert len(overlaps_train_valid) == 0, f"train/valid overlap: {len(overlaps_train_valid)}"
    assert len(overlaps_train_test) == 0, f"train/test overlap: {len(overlaps_train_test)}"

    # 5. 生成 manifest
    for split_name, records in splits.items():
        generate_manifest(records, f'data_manifests/codecontests_{split_name}.jsonl')

    # 6. 输出审计报告
    print_audit_report(splits)
```

### 4.5 审计报告模板

```markdown
# 数据治理审计报告

## 1. 样本统计

| Split | 去重前 | 去重后 | 删除数 |
|-------|--------|--------|--------|
| train | 13328  | 13200  | 128    |
| valid | 117    | 117    | 0      |
| test  | 165    | 165    | 0      |

## 2. 跨 Split 精确重叠检查

| 检查对 | 重叠数 | 状态 |
|--------|--------|------|
| train ∩ valid | 0 | ✓ |
| train ∩ test | 0 | ✓ |
| valid ∩ test | 0 | ✓ |

## 3. 外部泄漏检查

| 检查对 | 重叠数 | 状态 |
|--------|--------|------|
| codecontests_train ∩ humaneval | 0 | ✓ |
| codecontests_train ∩ mbpp_reg | 0 | ✓ |

## 4. MBPP_reg 固定题号列表

选择 MBPP ID 11-210（共 200 题）作为回归监控子集。

## 5. 版本信息

- CodeContests: deepmind/code_contests @ 2024-01-31
- HumanEval: openai_humaneval @ 2024-01-31
- MBPP: google-research-datasets/mbpp @ 2024-01-31
```

---

## 五、Phase 0 评测实施步骤

### 5.1 环境准备

#### Step 1: 启动 SandboxFusion 服务

```bash
# 方式一：本地开发模式
cd SandboxFusion
make run  # 端口 8080

# 方式二：Docker 生产模式
docker run -d --rm --privileged -p 8080:8080 code_sandbox:server

# 验证服务
curl http://localhost:8080/health
```

#### Step 2: 安装依赖

```bash
# verl 依赖
pip install -e verl/
pip install vllm  # 或 sglang

# SandboxFusion SDK
pip install sandbox-fusion

# 其他
pip install wandb datasets transformers aiohttp
```

#### Step 3: verl 配置文件（可选，用于 Hydra 集成）

如果使用 Hydra 配置系统，可以创建配置文件：

```yaml
# config/phase0_eval.yaml

# Ray 配置
trainer:
  n_gpus_per_node: 8
  nnodes: 1

# 模型配置
actor_rollout_ref:
  model:
    path: "Qwen/Qwen2.5-Coder-7B-Instruct"
    trust_remote_code: true
    load_tokenizer: true
    lora_rank: 0

  # Rollout 配置（使用 vLLM Standalone 模式）
  rollout:
    name: "vllm"  # 或 "sglang"
    mode: "async"
    tensor_model_parallel_size: 2
    data_parallel_size: 1
    pipeline_model_parallel_size: 1

    # EVAL@1 协议：greedy decoding
    temperature: 0.0
    top_p: 1.0
    top_k: -1
    do_sample: false
    n: 1

    # 输出长度
    prompt_length: 4096
    response_length: 2048

    # 引擎配置
    dtype: "bfloat16"
    gpu_memory_utilization: 0.8
    load_format: "auto"  # Standalone 模式必须使用 "auto"
    enforce_eager: true
    enable_prefix_caching: true
    enable_chunked_prefill: true

    # 批处理
    max_num_seqs: 256
    max_num_batched_tokens: 8192

# 数据配置
data:
  eval_files:
    - "data/eval/codecontests_valid.parquet"
    - "data/eval/codecontests_test.parquet"
    - "data/eval/humaneval.parquet"
    - "data/eval/mbpp_reg.parquet"

# SandboxFusion 配置
sandbox:
  url: "http://localhost:8080/run_code"
  timeout: 10
  memory_limit_mb: 1024
```

> **⚠️ 关键配置：`load_format` 参数**
>
> | 模式 | `load_format` | 说明 |
> |------|---------------|------|
> | **STANDALONE** (Phase 0 评测) | `"auto"` | 从磁盘/HDFS 加载**真实**模型权重 |
> | **HYBRID** (GRPO/PPO 训练) | `"dummy"` | 创建空壳模型，由训练引擎同步权重 |
>
> **如果 Phase 0 使用 `load_format: "dummy"`，模型权重将不会被加载，所有生成结果都是随机的！** 这是最常见的配置错误之一。详见 [verl_standalone_rollout_guide.md](./verl_standalone_rollout_guide.md) 第七章。

### 5.2 数据准备

#### Step 1: 下载数据集

```python
# scripts/download_datasets.py
from datasets import load_dataset

# CodeContests
codecontests = load_dataset("deepmind/code_contests")
codecontests.save_to_disk("data/codecontests")

# HumanEval
humaneval = load_dataset("openai_humaneval")
humaneval.save_to_disk("data/humaneval")

# MBPP
mbpp = load_dataset("mbpp")
mbpp.save_to_disk("data/mbpp")
```

#### Step 2: 运行数据治理

```bash
python scripts/data_governance.py
# 输出:
#   - data_manifests/codecontests_train.jsonl
#   - data_manifests/codecontests_valid.jsonl
#   - data_manifests/codecontests_test.jsonl
#   - data_manifests/humaneval.jsonl
#   - data_manifests/mbpp_reg.jsonl
#   - data_manifests/audit_report.md
```

#### Step 3: 转换为 verl 数据格式

```python
# 伪代码：scripts/prepare_eval_data.py

def convert_codecontests_to_verl_format(dataset, split: str) -> List[Dict]:
    """转换 CodeContests 为 verl 评测格式"""
    records = []
    for item in dataset[split]:
        # CodeContests 使用 stdin/stdout 格式
        test_cases = {
            "inputs": item['public_tests']['input'] + item['private_tests']['input'],
            "outputs": item['public_tests']['output'] + item['private_tests']['output'],
        }

        records.append({
            "prompt": item['description'],
            "data_source": "codecontests",
            "ground_truth": json.dumps(test_cases),
            "problem_id": item['name'],
            "difficulty": item.get('difficulty', 'unknown'),
        })
    return records

def convert_humaneval_to_verl_format(dataset) -> List[Dict]:
    """转换 HumanEval 为 verl 评测格式"""
    records = []
    for item in dataset['test']:
        # HumanEval 使用函数调用 + assert 格式
        records.append({
            "prompt": item['prompt'],
            "data_source": "humaneval",
            "ground_truth": json.dumps({
                "test": item['test'],
                "entry_point": item['entry_point'],
            }),
            "problem_id": f"HumanEval/{item['task_id']}",
        })
    return records

# 保存为 parquet
import pandas as pd
pd.DataFrame(records).to_parquet("data/eval/codecontests_valid.parquet")
```

### 5.3 评测脚本框架（使用 verl 分布式推理）

**重要**: Phase 0 评测脚本使用 verl 的 **Standalone Rollout 模式**，通过 vLLM/SGLang 引擎进行分布式推理，而非纯 HuggingFace。

```python
# scripts/phase0_baseline_eval.py

"""
Phase 0 Baseline 评测脚本 - 使用 verl 分布式推理架构

核心组件：
- verl RolloutReplica: 管理 vLLM/SGLang 推理服务器
- Ray: 分布式协调
- OpenAI-compatible API: 统一的生成接口
- SandboxFusion compute_score: 代码评测

运行：
python scripts/phase0_baseline_eval.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --rollout vllm \
    --tensor_parallel_size 2 \
    --sandbox_url http://localhost:8080/run_code \
    --output_dir outputs/phase0
"""

import asyncio
import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import ray
import wandb
from tqdm.asyncio import tqdm_asyncio

# verl 核心组件
from verl.workers.rollout.replica import get_rollout_replica_class
from verl.utils.reward_score.sandbox_fusion import compute_score

# =============================================================================
# 配置类
# =============================================================================

@dataclass
class EvalConfig:
    """评测配置"""
    # 模型配置
    model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # verl Rollout 配置
    rollout_name: str = "vllm"  # "vllm" 或 "sglang"
    tensor_parallel_size: int = 2
    n_gpus_per_node: int = 8
    gpu_memory_utilization: float = 0.8

    # 解码参数 (EVAL@1 协议)
    temperature: float = 0.0  # greedy decoding
    top_p: float = 1.0
    max_new_tokens: int = 2048

    # SandboxFusion 配置
    sandbox_url: str = "http://localhost:8080/run_code"
    memory_limit_mb: int = 1024
    timeout: int = 10

    # 评测方式选择
    use_submit_api: bool = True  # True: 使用 submit() API (推荐)
                                  # False: 使用 compute_score() (与 GRPO 一致)

    # 并发控制
    max_concurrent_requests: int = 64

    # 输出
    output_dir: str = "outputs/phase0"
    wandb_project: str = "rlvr_coding_model"
    wandb_run_name: str = "phase0_baseline"

@dataclass
class EvalResult:
    """单个问题的评测结果"""
    problem_id: str
    prompt: str
    completion: str

    # 质量指标
    pass_ratio: float = 0.0
    accepted: bool = False
    final_status: str = "unknown"

    # 成本指标
    output_tokens: int = 0
    gen_time: float = 0.0
    judge_time: float = 0.0

    # 详细结果
    metadata: List = field(default_factory=list)

# =============================================================================
# verl Rollout Server 管理
# =============================================================================

async def start_rollout_servers(config: EvalConfig):
    """
    启动 verl Standalone Rollout 服务器

    关键步骤：
    1. 获取 RolloutReplica 类（vLLMReplica 或 SGLangReplica）
    2. 根据 GPU 配置计算 replica 数量
    3. 调用 init_standalone() 初始化独立服务器
    4. 返回服务器地址列表
    """
    from omegaconf import OmegaConf

    # 构建 RolloutConfig
    rollout_config = OmegaConf.create({
        "name": config.rollout_name,
        "mode": "async",
        "tensor_model_parallel_size": config.tensor_parallel_size,
        "data_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "response_length": config.max_new_tokens,
        "prompt_length": 4096,
        "dtype": "bfloat16",
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "load_format": "auto",  # Standalone 模式使用 "auto" 加载真实权重
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "disable_log_stats": True,
    })

    # 构建 HFModelConfig
    model_config = OmegaConf.create({
        "path": config.model_path,
        "trust_remote_code": True,
        "load_tokenizer": True,
        "lora_rank": 0,
    })

    # 计算 replica 数量
    num_replicas = config.n_gpus_per_node // config.tensor_parallel_size

    # 获取 Rollout 类并创建实例
    rollout_class = get_rollout_replica_class(config.rollout_name)

    rollout_servers = [
        rollout_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]

    # 初始化 Standalone 模式服务器
    print(f"Initializing {num_replicas} {config.rollout_name} rollout servers...")
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    # 获取服务器地址
    server_addresses = [server._server_address for server in rollout_servers]
    print(f"Rollout servers ready at: {server_addresses}")

    return rollout_servers, server_addresses

# =============================================================================
# 代码生成（通过 OpenAI-compatible API）
# =============================================================================

async def generate_code(
    session: aiohttp.ClientSession,
    server_address: str,
    model_path: str,
    prompt: str,
    sampling_params: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str, float]:
    """
    通过 OpenAI API 调用 vLLM/SGLang 生成代码

    返回: (completion, generation_time)
    """
    async with semaphore:
        start_time = time.time()

        try:
            async with session.post(
                url=f"http://{server_address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123"},
                json={
                    "model": model_path,
                    "messages": [{"role": "user", "content": prompt}],
                    **sampling_params
                },
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                data = await resp.json()
                completion = data["choices"][0]["message"]["content"]
                gen_time = time.time() - start_time
                return completion, gen_time

        except Exception as e:
            print(f"Generation error: {e}")
            return "", time.time() - start_time

async def batch_generate(
    server_addresses: List[str],
    model_path: str,
    prompts: List[str],
    sampling_params: dict,
    max_concurrent: int = 64,
) -> List[tuple[str, float]]:
    """
    批量生成代码，负载均衡到多个 replica
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            # Round-robin 负载均衡
            server_idx = i % len(server_addresses)
            server_address = server_addresses[server_idx]

            task = generate_code(
                session, server_address, model_path, prompt,
                sampling_params, semaphore
            )
            tasks.append(task)

        # 带进度条的并发执行
        results = await tqdm_asyncio.gather(*tasks, desc="Generating")

    return results

# =============================================================================
# 代码评测（使用 SandboxFusion）
# =============================================================================

# 方式 A：使用 submit() API（推荐，更简单）
def evaluate_code_with_submit(
    completion: str,
    sandbox_dataset: str,
    sandbox_id: str,
    config: EvalConfig,
) -> tuple[float, str, float]:
    """
    使用 SandboxFusion submit() API 评测代码

    参数:
        completion: 模型生成的代码
        sandbox_dataset: SandboxFusion 数据集名 (e.g., "humaneval")
        sandbox_id: 问题 ID

    返回: (pass_ratio, final_status, judge_time)
    """
    from sandbox_fusion import submit, SubmitRequest, TestConfig

    start_time = time.time()

    if not completion or not completion.strip():
        return 0.0, "empty_output", time.time() - start_time

    try:
        result = submit(SubmitRequest(
            dataset=sandbox_dataset,
            id=sandbox_id,
            completion=completion,
            config=TestConfig(language='python', run_timeout=config.timeout)
        ))

        judge_time = time.time() - start_time

        if result.accepted:
            return 1.0, "success", judge_time
        else:
            # 从 result.tests 推断错误类型（如果可用）
            return 0.0, "wrong_answer", judge_time

    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.0, "api_error", time.time() - start_time

# 方式 B：使用 compute_score()（与 GRPO 一致）
def evaluate_code_with_compute_score(
    completion: str,
    test_cases: Dict,
    config: EvalConfig,
) -> tuple[float, str, List, float]:
    """
    使用 verl compute_score() 评测代码

    返回: (pass_ratio, final_status, metadata, judge_time)
    """
    start_time = time.time()

    if not completion or not completion.strip():
        return 0.0, "empty_output", [], time.time() - start_time

    try:
        score, metadata_list = compute_score(
            sandbox_fusion_url=config.sandbox_url,
            memory_limit_mb=config.memory_limit_mb,
            completion=completion,
            test_cases=test_cases,
            continuous=False,
            timeout=config.timeout,
        )

        judge_time = time.time() - start_time

        if not metadata_list:
            final_status = "api_error"
        elif score == 1.0:
            final_status = "success"
        else:
            statuses = [m.get('status', '') for m in metadata_list]
            if any('compile' in s for s in statuses):
                final_status = "syntax_error"
            elif any('runtime' in s for s in statuses):
                final_status = "runtime_error"
            elif any('timeout' in s for s in statuses):
                final_status = "timeout"
            else:
                final_status = "wrong_answer"

        return score, final_status, metadata_list, judge_time

    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.0, "api_error", [], time.time() - start_time

# 统一接口：根据配置选择评测方式
def evaluate_code(
    completion: str,
    record: Dict,
    config: EvalConfig,
) -> tuple[float, str, List, float]:
    """
    评测代码 - 根据配置选择 submit() 或 compute_score()

    参数:
        record: 问题记录，包含 sandbox_dataset/sandbox_id 或 ground_truth
    """
    if config.use_submit_api and 'sandbox_dataset' in record:
        # 使用 submit() API
        pass_ratio, final_status, judge_time = evaluate_code_with_submit(
            completion,
            record['sandbox_dataset'],
            record['sandbox_id'],
            config
        )
        return pass_ratio, final_status, [], judge_time
    else:
        # 使用 compute_score()
        test_cases = json.loads(record.get('ground_truth', '{}'))
        return evaluate_code_with_compute_score(completion, test_cases, config)

# =============================================================================
# 指标聚合
# =============================================================================

def aggregate_metrics(results: List[EvalResult]) -> Dict:
    """聚合评测指标"""
    n = len(results)
    if n == 0:
        return {}

    pass_ratios = [r.pass_ratio for r in results]

    # 质量指标
    metrics = {
        "accepted_at_1": sum(r.accepted for r in results) / n,
        "pass_ratio_mean": np.mean(pass_ratios),
        "pass_ratio_p50": np.percentile(pass_ratios, 50),
        "pass_ratio_p90": np.percentile(pass_ratios, 90),
        "exec_success_rate": sum(
            r.final_status in ['success', 'wrong_answer'] for r in results
        ) / n,
    }

    # 错误分布
    for status in ['syntax_error', 'runtime_error', 'timeout', 'wrong_answer', 'empty_output']:
        count = sum(r.final_status == status for r in results)
        metrics[f"{status}_rate"] = count / n

    # 成本指标
    metrics["avg_gen_tokens"] = np.mean([r.output_tokens for r in results])
    metrics["avg_gen_time"] = np.mean([r.gen_time for r in results])
    metrics["avg_judge_time"] = np.mean([r.judge_time for r in results])

    solved = [r for r in results if r.accepted]
    if solved:
        metrics["cost_per_solved_tokens"] = sum(r.output_tokens for r in solved) / len(solved)
        metrics["cost_per_solved_time"] = sum(r.gen_time + r.judge_time for r in solved) / len(solved)

    return metrics

def sample_qa_logs(results: List[EvalResult], num_samples: int) -> List[Dict]:
    """分层抽样 QA 日志"""
    by_status = {}
    for r in results:
        if r.final_status not in by_status:
            by_status[r.final_status] = []
        by_status[r.final_status].append(r)

    samples = []
    samples_per_status = max(1, num_samples // max(len(by_status), 1))

    for status, group in by_status.items():
        for r in group[:samples_per_status]:
            samples.append({
                "problem_id": r.problem_id,
                "prompt": r.prompt[:500],
                "response": r.completion[:1000],
                "pass_ratio": r.pass_ratio,
                "final_status": r.final_status,
                "output_tokens": r.output_tokens,
                "gen_time": r.gen_time,
                "judge_time": r.judge_time,
            })

    return samples[:num_samples]

# =============================================================================
# 主函数
# =============================================================================

async def evaluate_dataset(
    dataset_name: str,
    data_path: str,
    server_addresses: List[str],
    config: EvalConfig,
    log_samples: int,
) -> tuple[Dict, List[Dict]]:
    """评测单个数据集"""
    print(f"\n{'='*60}")
    print(f"Evaluating {dataset_name}...")
    print(f"{'='*60}")

    # 加载数据
    df = pd.read_parquet(data_path)
    prompts = df['prompt'].tolist()

    # 批量生成
    sampling_params = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_new_tokens,
    }

    gen_results = await batch_generate(
        server_addresses, config.model_path, prompts,
        sampling_params, config.max_concurrent_requests
    )

    # 评测每个结果
    results = []
    for idx, ((completion, gen_time), row) in enumerate(zip(gen_results, df.itertuples())):
        # 构建记录字典（支持 submit() 和 compute_score() 两种方式）
        record = {
            'sandbox_dataset': getattr(row, 'sandbox_dataset', None),
            'sandbox_id': getattr(row, 'sandbox_id', None),
            'ground_truth': getattr(row, 'ground_truth', '{}'),
        }
        pass_ratio, final_status, metadata, judge_time = evaluate_code(
            completion, record, config
        )

        result = EvalResult(
            problem_id=getattr(row, 'problem_id', f'problem_{idx}'),
            prompt=row.prompt,
            completion=completion,
            pass_ratio=pass_ratio,
            accepted=(pass_ratio == 1.0),
            final_status=final_status,
            output_tokens=len(completion.split()),  # 简化计算
            gen_time=gen_time,
            judge_time=judge_time,
            metadata=metadata,
        )
        results.append(result)

    # 聚合指标
    metrics = aggregate_metrics(results)
    metrics["total_problems"] = len(results)

    # 抽样日志
    qa_logs = sample_qa_logs(results, log_samples)

    # 打印摘要
    print(f"  Total problems: {len(results)}")
    print(f"  accepted@1: {metrics['accepted_at_1']:.2%}")
    print(f"  pass_ratio_mean: {metrics['pass_ratio_mean']:.3f}")
    print(f"  exec_success_rate: {metrics['exec_success_rate']:.2%}")

    return metrics, qa_logs

async def main_async(config: EvalConfig):
    """异步主函数"""

    # 初始化 Ray
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )

    try:
        # 启动 Rollout 服务器
        rollout_servers, server_addresses = await start_rollout_servers(config)

        # 初始化 WandB
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

        # 评测数据集列表
        datasets_to_eval = [
            ("codecontests_valid", "data/eval/codecontests_valid.parquet", 50),
            ("codecontests_test", "data/eval/codecontests_test.parquet", 30),
            ("humaneval", "data/eval/humaneval.parquet", 20),
            ("mbpp_reg", "data/eval/mbpp_reg.parquet", 20),
        ]

        all_metrics = {}
        output_dir = Path(config.output_dir)

        for dataset_name, data_path, log_samples in datasets_to_eval:
            if not Path(data_path).exists():
                print(f"Warning: {data_path} not found, skipping...")
                continue

            metrics, qa_logs = await evaluate_dataset(
                dataset_name, data_path, server_addresses,
                config, log_samples
            )

            # 记录到 WandB
            for k, v in metrics.items():
                wandb.log({f"eval/{dataset_name}/{k}": v})

            all_metrics[dataset_name] = metrics

            # 保存 QA 日志
            log_path = output_dir / "qa_logs" / f"{dataset_name}.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                for log in qa_logs:
                    f.write(json.dumps(log, ensure_ascii=False) + '\n')

        # 保存汇总
        summary_path = output_dir / "phase0_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        wandb.finish()
        print(f"\n{'='*60}")
        print(f"Results saved to {config.output_dir}")
        print(f"{'='*60}")

    finally:
        ray.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Phase 0 Baseline Evaluation with verl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--rollout", default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--sandbox_url", default="http://localhost:8080/run_code")
    parser.add_argument("--output_dir", default="outputs/phase0")
    parser.add_argument("--use-submit-api", action="store_true", default=True,
                        help="使用 SandboxFusion submit() API 评测（推荐）")
    parser.add_argument("--use-compute-score", dest="use_submit_api", action="store_false",
                        help="使用 verl compute_score() 评测（与 GRPO 一致）")
    args = parser.parse_args()

    config = EvalConfig(
        model_path=args.model,
        rollout_name=args.rollout,
        tensor_parallel_size=args.tensor_parallel_size,
        n_gpus_per_node=args.n_gpus,
        sandbox_url=args.sandbox_url,
        output_dir=args.output_dir,
        use_submit_api=args.use_submit_api,
    )

    asyncio.run(main_async(config))

if __name__ == "__main__":
    main()
```

### 5.4 运行命令示例

```bash
# 单机 8 GPU，使用 vLLM，TP=2（4 个 replica）
python scripts/phase0_baseline_eval.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --rollout vllm \
    --tensor_parallel_size 2 \
    --n_gpus 8 \
    --sandbox_url http://localhost:8080/run_code \
    --output_dir outputs/phase0

# 单机 4 GPU，使用 SGLang，TP=2（2 个 replica）
python scripts/phase0_baseline_eval.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --rollout sglang \
    --tensor_parallel_size 2 \
    --n_gpus 4 \
    --sandbox_url http://localhost:8080/run_code \
    --output_dir outputs/phase0

# 快速测试（1 GPU）
python scripts/phase0_baseline_eval.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --rollout vllm \
    --tensor_parallel_size 1 \
    --n_gpus 1 \
    --output_dir outputs/phase0_test
```

---

## 六、指标收集详解

### 6.1 质量指标收集

| 指标 | 计算公式 | 数据来源 |
|------|----------|---------|
| accepted@1 | `sum(pass_ratio == 1.0) / total` | compute_score 返回值 |
| pass_ratio_mean | `mean(pass_ratios)` | compute_score 返回值 |
| pass_ratio_p50 | `median(pass_ratios)` | compute_score 返回值 |
| pass_ratio_p90 | `percentile(pass_ratios, 90)` | compute_score 返回值 |
| exec_success_rate | `sum(final_status in ['success', 'wrong_answer']) / total` | metadata 的 status 字段 |

### 6.2 错误分布指标收集

| 指标 | 对应 status | 判断逻辑 |
|------|-------------|---------|
| syntax_error_rate | compile_error | `result == -4` |
| runtime_error_rate | runtime_error | `result == -2` |
| timeout_rate | timeout | `result == -3` |
| wrong_answer_rate | wrong_answer | `result == False` |

```python
# 从 metadata 中提取错误类型
def extract_error_distribution(metadata_list):
    """提取错误分布"""
    counts = {
        'success': 0,
        'wrong_answer': 0,
        'syntax_error': 0,
        'runtime_error': 0,
        'timeout': 0,
        'api_error': 0,
    }

    for m in metadata_list:
        status = m.get('status', 'unknown')
        if status == 'success':
            counts['success'] += 1
        elif status == 'wrong_answer':
            counts['wrong_answer'] += 1
        elif status in ['compile_error', 'compile_timeout']:
            counts['syntax_error'] += 1
        elif status == 'runtime_error':
            counts['runtime_error'] += 1
        elif status == 'timeout':
            counts['timeout'] += 1
        else:
            counts['api_error'] += 1

    total = len(metadata_list)
    return {k: v/total for k, v in counts.items()}
```

### 6.3 成本指标收集

| 指标 | 计算公式 | 数据来源 |
|------|----------|---------|
| avg_total_gen_tokens | `mean(output_tokens per problem)` | tokenizer.encode(completion) |
| avg_total_judge_time | `mean(sum(duration per case) per problem)` | metadata 的 duration 字段 |
| p95_total_judge_time | `percentile(total_judge_times, 95)` | metadata 的 duration 字段 |
| throughput | `total_problems / wall_clock_time` | 端到端计时 |
| cost_per_solved_tokens | `sum(output_tokens) / solved_count` | 仅对 accepted 的问题 |
| cost_per_solved_judge_time | `sum(judge_time) / solved_count` | 仅对 accepted 的问题 |

```python
def compute_cost_metrics(results: List[EvalResult]) -> Dict:
    """计算成本指标"""
    total_tokens = sum(r.output_tokens for r in results)
    total_judge_time = sum(r.total_judge_time for r in results)
    solved_count = sum(r.accepted for r in results)

    metrics = {
        'avg_total_gen_tokens': total_tokens / len(results),
        'avg_total_judge_time': total_judge_time / len(results),
        'p95_total_judge_time': np.percentile(
            [r.total_judge_time for r in results], 95
        ),
    }

    if solved_count > 0:
        # 只计算 solved 的问题
        solved_tokens = sum(r.output_tokens for r in results if r.accepted)
        solved_time = sum(r.total_judge_time for r in results if r.accepted)

        metrics['cost_per_solved_tokens'] = solved_tokens / solved_count
        metrics['cost_per_solved_judge_time'] = solved_time / solved_count

    return metrics
```

### 6.4 WandB 日志格式

```python
# 记录到 WandB
wandb.log({
    # 按数据集分别记录
    "eval/codecontests_valid/accepted_at_1": 0.15,
    "eval/codecontests_valid/pass_ratio_mean": 0.25,
    "eval/codecontests_valid/pass_ratio_p50": 0.20,
    "eval/codecontests_valid/pass_ratio_p90": 0.45,
    "eval/codecontests_valid/exec_success_rate": 0.70,
    "eval/codecontests_valid/syntax_error_rate": 0.10,
    "eval/codecontests_valid/runtime_error_rate": 0.08,
    "eval/codecontests_valid/timeout_rate": 0.12,
    "eval/codecontests_valid/wrong_answer_rate": 0.55,
    "eval/codecontests_valid/avg_total_gen_tokens": 450,
    "eval/codecontests_valid/avg_total_judge_time": 2.5,
    "eval/codecontests_valid/cost_per_solved_tokens": 500,
    "eval/codecontests_valid/cost_per_solved_judge_time": 3.0,
    "eval/codecontests_valid/throughput": 5.0,

    # 同样格式记录其他数据集...
})
```

---

## 七、问答日志格式规范

### 7.1 日志文件结构

```
outputs/phase0/
├── qa_logs/
│   ├── codecontests_valid_50.jsonl
│   ├── codecontests_test_30.jsonl
│   ├── humaneval_20.jsonl
│   └── mbpp_reg_20.jsonl
├── metrics/
│   └── phase0_summary.json
└── audit/
    └── data_governance_report.md
```

### 7.2 单条日志格式

```json
{
    "problem_id": "cc_valid_042",
    "dataset": "codecontests_valid",
    "prompt": "Given an array of integers, find the maximum sum of a contiguous subarray...",
    "response": "```python\ndef max_subarray_sum(arr):\n    max_sum = float('-inf')\n    current_sum = 0\n    for num in arr:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum\n\nn = int(input())\narr = list(map(int, input().split()))\nprint(max_subarray_sum(arr))\n```",
    "ground_truth": {
        "inputs": ["5\n-2 1 -3 4 -1 2 1 -5 4", ...],
        "outputs": ["6", ...]
    },
    "pass_ratio": 0.8,
    "accepted": false,
    "final_status": "wrong_answer",
    "output_tokens": 156,
    "total_judge_time": 1.23,
    "error_breakdown": {
        "passed": 8,
        "wrong_answer": 2,
        "timeout": 0,
        "runtime_error": 0,
        "syntax_error": 0
    },
    "execution_output": {
        "first_failed_case": {
            "input": "10\n1 2 3 -10 5 6 7 -8 9 10",
            "expected": "29",
            "actual": "28",
            "stderr": ""
        }
    },
    "metadata": {
        "model": "Qwen2.5-Coder-7B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 2048,
        "timestamp": "2024-01-31T10:30:00Z"
    }
}
```

### 7.3 分层抽样策略

根据实验设计，建议按 error_type 分层抽样：

| 数据集 | 总采样数 | 抽样策略 |
|--------|---------|---------|
| CodeContests_valid | 50 条 | success: 10, WA: 20, Syntax: 10, Runtime: 5, Timeout: 5 |
| CodeContests_test | 30 条 | success: 6, WA: 12, Syntax: 6, Runtime: 3, Timeout: 3 |
| HumanEval | 20 条 | success: 4, WA: 8, Syntax: 4, Runtime: 2, Timeout: 2 |
| MBPP_reg | 20 条 | success: 4, WA: 8, Syntax: 4, Runtime: 2, Timeout: 2 |

---

## 八、预期产出与验收标准

### 8.1 必须产出清单

| 产出 | 文件路径 | 验收标准 |
|------|----------|---------|
| **数据 Manifest** | `data_manifests/*.jsonl` | 每个 split 有独立 manifest |
| **审计报告** | `data_manifests/audit_report.md` | 无跨 split 重叠，无外部泄漏 |
| **评测指标** | `outputs/phase0/phase0_summary.json` | 4 个数据集全部完成 |
| **问答日志** | `outputs/phase0/qa_logs/*.jsonl` | 总计 120 条 |
| **WandB 面板** | 在线 | 所有指标已记录 |

### 8.2 指标验收标准

Phase 0 是基线评测，不预设性能目标，但需要验证：

1. **评测流水线正常**：所有数据集都能完成评测，无系统性失败
2. **指标完整**：质量、成本、错误分布三类指标全部收集
3. **数据隔离正确**：manifest 和审计报告证明无泄漏
4. **日志可用**：QA 日志包含足够信息用于后续分析

### 8.3 典型基线值参考

基于公开 benchmark 和类似项目经验，7B Base 模型在 CodeContests 上的典型表现：

| 指标 | 典型值范围 | 说明 |
|------|-----------|------|
| accepted@1 | 3% - 10% | 未训练模型通常较低 |
| pass_ratio_mean | 0.10 - 0.25 | 部分测试点能通过 |
| exec_success_rate | 50% - 80% | 能产出可运行代码 |
| syntax_error_rate | 5% - 20% | 偶尔有语法错误 |
| timeout_rate | 5% - 15% | 算法效率问题 |

这些只是参考，实际值可能因模型和数据集不同而异。

---

## 九、时间线规划

### 建议执行顺序

| 步骤 | 内容 | 预估时间 | 产出 |
|------|------|---------|------|
| 1 | 环境搭建与验证 | 2-4 小时 | SandboxFusion 正常运行 |
| 2 | 数据下载与治理 | 4-6 小时 | Manifest + 审计报告 |
| 3 | 评测脚本开发 | 6-8 小时 | 可运行的评测脚本 |
| 4 | CodeContests_valid 评测 | 2-4 小时 | 指标 + 50 条日志 |
| 5 | CodeContests_test 评测 | 2-4 小时 | 指标 + 30 条日志 |
| 6 | HumanEval 评测 | 1-2 小时 | 指标 + 20 条日志 |
| 7 | MBPP_reg 评测 | 1-2 小时 | 指标 + 20 条日志 |
| 8 | 结果汇总与报告 | 2-4 小时 | phase0_summary.json |

**总预估时间**：20-34 小时

---

## 十、附录

### A. verl 关键文件索引

> **⚠️ 注意**：verl 代码库采用双层目录结构，核心代码位于 `verl/verl/` 目录下。

#### Phase 0 评测核心文件（Standalone Rollout）

| 功能 | 文件路径 | 说明 |
|------|----------|------|
| **Rollout Replica 基类** | `verl/verl/workers/rollout/replica.py` | `RolloutReplica`, `RolloutMode`, `get_rollout_replica_class()` |
| **vLLM Replica 实现** | `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `vLLMReplica`, `vLLMHttpServer` |
| **SGLang Replica 实现** | `verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py` | `SGLangReplica` |
| **vLLM Rollout Worker** | `verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py` | `vLLMAsyncRollout` |
| **Rollout 配置** | `verl/verl/workers/config/rollout.py` | `RolloutConfig`, `SamplingConfig` |
| **模型配置** | `verl/verl/workers/config/model.py` | `HFModelConfig` |
| **官方 Standalone 示例** ★ | `verl/verl/trainer/main_generation_server.py` | **推荐参考**：完整的 Standalone 模式实现 |

#### SandboxFusion 集成文件

| 功能 | 文件路径 | 说明 |
|------|----------|------|
| SandboxFusion 评分 | `verl/verl/utils/reward_score/sandbox_fusion/__init__.py` | `compute_score()` 函数 |
| SandboxFusion API 调用 | `verl/verl/utils/reward_score/sandbox_fusion/utils.py` | `check_correctness()`, `call_sandbox_api()` |
| 奖励路由 | `verl/verl/utils/reward_score/__init__.py` | 奖励函数路由 |
| 奖励管理器 | `verl/verl/workers/reward_manager/naive.py`, `prime.py` | 训练时使用 |

#### 参考示例

| 示例 | 文件路径 | 说明 |
|------|----------|------|
| Standalone 测试 | `verl/tests/experimental/agent_loop/test_standalone_rollout.py` | 如何使用 `init_standalone()` |
| 生成服务器 ★ | `verl/verl/trainer/main_generation_server.py` | **首选参考**：官方批量生成示例 |
| 生成脚本 | `verl/verl/trainer/main_generation.py` | 备选参考：另一种生成实现 |

### B. SandboxFusion API 状态码

| 状态码 | 含义 |
|--------|------|
| True | 测试通过 |
| False | Wrong Answer |
| -1 | API/Sandbox 错误 |
| -2 | Runtime Error |
| -3 | Timeout |
| -4 | Compile/Syntax Error |

### C. 常见问题排查

1. **模型输出随机/无意义（最常见错误）**
   - **原因**：`load_format` 参数配置错误
   - **解决**：确保 Standalone 模式使用 `load_format: "auto"`（不是 `"dummy"`）
   - **验证**：检查启动日志是否显示正在加载模型权重

2. **SandboxFusion 504 Gateway Timeout**
   - 降低 `max_concurrent`
   - 增加 `timeout` 值
   - 检查服务端资源

3. **Reward 全为 0**
   - 检查代码提取逻辑（是否正确识别 ```python``` 块）
   - 检查 test_cases 格式是否正确
   - **检查 `load_format` 是否为 `"auto"`**

4. **NCCL 初始化错误**
   - 检查 GPU 可用性：`nvidia-smi`
   - 设置环境变量：`NCCL_DEBUG=INFO`
   - 确保 tensor_parallel_size 不超过可用 GPU 数

5. **评测卡住**
   - 检查网络连接
   - 查看 SandboxFusion 日志
   - 考虑添加单次超时和重试逻辑

> **💡 更多排查指南**：详见 [verl_standalone_rollout_guide.md](./verl_standalone_rollout_guide.md) 第八章「常见问题与排查」。

---

*文档版本：v2.2*
*创建日期：2024-01-31*
*最后更新：2026-01-31*

**v2.2 更新说明**：
- **修正文件路径**：verl 代码库采用双层目录结构（`verl/verl/`），更新所有文件路径引用
- **新增参考文档链接**：关联 `verl_standalone_rollout_guide.md` 提供深入技术解析
- **突出 `load_format` 关键配置**：Standalone 模式必须使用 `"auto"`，这是最常见的配置错误
- **新增 `main_generation_server.py` 参考**：推荐使用 verl 官方 Standalone 模式示例
- **更新常见问题排查**：新增模型输出随机、NCCL 错误等排查指南
- **更新附录文件索引**：添加目录结构说明和推荐参考标记

**v2.1 更新说明**：
- 更新数据获取方式，优先使用 SandboxFusion SDK `get_prompts()` 获取数据
- 新增 `submit()` API 评测方式，简化评测流程
- 评测脚本支持两种方式：`submit()` API（推荐）和 `compute_score()`（与 GRPO 一致）
- 新增 `--use-submit-api` 和 `--use-compute-score` 命令行参数
- 更新架构图，展示两种评测方式选项

**v2.0 更新说明**：
- 修正评测脚本，从纯 HuggingFace 改为使用 verl 的 vLLM/SGLang 分布式推理架构
- 新增 verl Standalone Rollout 模式详解（Section 2.3, 2.4）
- 更新代码执行流程图，展示 Ray 集群协调和 OpenAI-compatible API
- 新增 verl 与 HuggingFace 对比说明（Section 2.6）
- 更新评测脚本框架，使用 `RolloutReplica.init_standalone()` 和 async HTTP 请求
- 新增 verl 配置文件示例（YAML 格式）
- 更新附录中的关键文件索引
