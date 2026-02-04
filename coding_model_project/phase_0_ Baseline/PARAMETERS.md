# Phase 0 评测参数说明

本文档详细说明 `phase0_eval.py` 中的所有参数及其来源。

---

## 参数概览

`phase0_eval.py` 的参数分为两类：
1. **实验常量**：在 `eval_config.py` 中统一管理，所有 Phase 必须保持一致
2. **运行时参数**：每次运行可以调整的参数

---

## 1. 实验常量（来自 eval_config.py）

这些参数**必须在整个实验流程中保持一致**，否则评测结果不可比。

### 1.1 解码参数（EVAL@1 协议）

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| `temperature` | `--temperature` | `0.0` | 采样温度。0.0 = greedy decoding，确保可复现 |
| `top_p` | `--top_p` | `1.0` | Nucleus sampling 参数。temperature=0 时不起作用 |
| `max_new_tokens` | `--max_tokens` | `2048` | 最大生成 token 数。监控截断率，如 >5% 需增加 |

### 1.2 SandboxFusion 配置

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| `run_timeout` | `--run_timeout` | `30` | 代码执行超时（秒）。CodeContests 需要更长时间 |
| `memory_limit_mb` | - | `1024` | 内存限制（MB）。代码中默认值，无命令行参数 |

### 1.3 质量控制阈值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `truncation_warning_threshold` | `0.05` (5%) | 截断率超过此阈值发出警告 |
| `timeout_warning_threshold` | `0.10` (10%) | 超时率超过此阈值发出警告 |

---

## 2. 运行时参数

这些参数可以根据运行环境调整。

### 2.1 模式和连接

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| `mode` | `--mode` | `simple` | 运行模式：`verl`（分布式）或 `simple`（连接已有服务器） |
| `model_path` | `--model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | 模型路径或 HuggingFace ID |
| `vllm_url` | `--vllm_url` | `http://localhost:8000` | vLLM 服务器地址 |
| `sandbox_url` | `--sandbox_url` | `http://localhost:8080` | SandboxFusion 服务器地址 |

### 2.2 数据配置

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| `datasets` | `--datasets` | `humaneval mbpp_reg` | 要评测的数据集列表 |
| `manifest_dir` | `--manifest_dir` | `None` | Manifest 目录（去重后的数据） |

### 2.3 并发和批处理

| 参数 | 命令行 | 默认值 | 推荐值（单卡 4090） | 说明 |
|------|--------|--------|---------------------|------|
| `max_concurrent_requests` | `--max_concurrent` | `64` | `32` | 最大并发请求数 |
| `batch_size` | `--batch_size` | `50` | `50` | 每批处理题目数 |

### 2.4 输出配置

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| `output_dir` | `--output_dir` | `outputs/phase0` | 结果输出目录 |
| `qa_sample_size` | `--qa_sample_size` | `50` | 每个数据集保存的 QA 样本数 |

### 2.5 WandB 配置

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| `use_wandb` | `--use_wandb` | `False` | 是否启用 WandB 日志 |
| `wandb_project` | `--wandb_project` | `rlvr_coding_model` | WandB 项目名 |

### 2.6 评测方式

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| `use_external_tests` | `--use_external_tests` | `True` | 优先使用外部测试用例（从 raw 数据加载） |
| `use_submit_api` | `--use_submit_api` | `True` | 使用 SandboxFusion submit API |

---

## 3. verl 模式专用参数

仅在 `--mode verl` 时使用。

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| `rollout_name` | `--rollout` | `vllm` | 推理引擎：`vllm` 或 `sglang` |
| `tensor_parallel_size` | `--tensor_parallel_size` | `2` | Tensor Parallel 大小 |
| `n_gpus_per_node` | `--n_gpus` | `8` | 每节点 GPU 数量 |
| `gpu_memory_utilization` | `--gpu_memory_utilization` | `0.85` | GPU 显存利用率 |

---

## 4. 典型运行命令

### 4.1 单卡 4090 基线评测

```bash
python src/phase0_eval.py \
    --mode simple \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --vllm_url http://localhost:8000 \
    --sandbox_url http://localhost:8080 \
    --manifest_dir data/manifests \
    --datasets humaneval mbpp_reg codecontests_valid \
    --temperature 0.0 \
    --max_tokens 2048 \
    --run_timeout 30 \
    --max_concurrent 32 \
    --batch_size 50 \
    --output_dir outputs/phase0
```

### 4.2 使用启动脚本（推荐）

```bash
./scripts/run_phase0.sh
```

---

## 5. 输出指标

评测完成后，会输出以下指标：

### 5.1 质量指标

| 指标 | 说明 |
|------|------|
| `accepted_at_1` | 通过率（EVAL@1 协议） |
| `pass_ratio_mean` | 平均通过率 |
| `pass_ratio_p50` | 通过率中位数 |
| `pass_ratio_p90` | 通过率 90 分位 |

### 5.2 成本指标

| 指标 | 说明 |
|------|------|
| `avg_gen_tokens` | 平均生成 token 数 |
| `avg_judge_time` | 平均判题时间 |
| `throughput` | 吞吐量 (problems/sec) |
| `cost_per_solved_tokens` | 每 solved 的 tokens 成本 |
| `cost_per_solved_judge_time` | 每 solved 的判题时间成本 |

### 5.3 质量控制指标

| 指标 | 说明 | 警告阈值 |
|------|------|----------|
| `truncation_rate` | 截断率（finish_reason="length"） | >5% |
| `timeout_rate` | 超时率（error_type="timeout"） | >10% |

---

## 6. 参数来源优先级

参数按以下优先级确定：

1. **命令行参数**：`--temperature 0.0`
2. **环境变量**：`TEMPERATURE=0.0`（启动脚本中）
3. **EvalConfig 默认值**：`phase0_eval.py` 中的 `@dataclass`
4. **eval_config.py 常量**：`EVAL_CONSTANTS` 字典

**建议**：使用启动脚本或直接命令行指定，避免依赖默认值。

---

## 7. 注意事项

1. **参数一致性**：所有 Phase 必须使用相同的 `temperature`、`max_tokens`、`run_timeout`
2. **截断率监控**：如果截断率 >5%，考虑增加 `max_tokens`
3. **超时率监控**：如果超时率 >10%，考虑增加 `run_timeout`
4. **memory_limit_mb**：代码中已自动传递给所有 `run_code` 调用
