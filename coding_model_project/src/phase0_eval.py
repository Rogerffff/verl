#!/usr/bin/env python3
"""
Phase 0 基线评测脚本 - 使用 verl 分布式推理架构
=================================================

核心组件：
- verl RolloutReplica: 管理 vLLM/SGLang 推理服务器
- Ray: 分布式协调
- OpenAI-compatible API: 统一的生成接口
- SandboxFusion: 代码评测

运行方式：
1. GPU 服务器（完整 verl 模式）：
   python src/phase0_eval.py \
       --model Qwen/Qwen2.5-Coder-7B-Instruct \
       --rollout vllm \
       --tensor_parallel_size 2 \
       --n_gpus 8 \
       --datasets humaneval mbpp_reg

2. 本地测试（简化模式，连接已有 vLLM 服务器）：
   python src/phase0_eval.py \
       --mode simple \
       --vllm_url http://localhost:8000 \
       --datasets humaneval mbpp_reg

架构说明：
    本脚本参考 verl/trainer/main_generation_server.py，使用 verl 的
    Standalone Rollout 模式进行分布式推理。
"""

# =============================================================================
# 标准库导入
# =============================================================================
import asyncio      # 异步编程，用于并发处理多个生成/评测请求
import argparse     # 命令行参数解析
import json         # JSON 读写
import os           # 操作系统接口
import sys          # 系统功能（如 sys.exit()）
import time         # 时间测量，用于计算生成/评测耗时

# dataclass: 自动生成数据类
# field: 定义有默认值的可变字段
# asdict: 转换为字典
from dataclasses import dataclass, field, asdict
from datetime import datetime   # 时间戳生成
from pathlib import Path        # 路径操作
from typing import Dict, List, Optional, Tuple, Any  # 类型提示
from collections import defaultdict  # 默认值字典

# =============================================================================
# 第三方库导入
# =============================================================================
import aiohttp      # 异步 HTTP 客户端，用于调用 vLLM OpenAI-compatible API
import numpy as np  # 数值计算（用于统计）

# 本地模块：指标收集和问答日志
from utils.metrics import MetricsCollector, EvalResult
from utils.qa_logger import QALogger

# 评测配置常量（所有 Phase 共用）
try:
    from eval_config import (
        EVAL_CONSTANTS,
        DATASET_CONFIGS,
        get_sampling_params,
        get_sandbox_config,
        get_concurrency_config,
        print_config_summary,
    )
    EVAL_CONFIG_AVAILABLE = True
except ImportError:
    EVAL_CONFIG_AVAILABLE = False
    # 回退默认值
    EVAL_CONSTANTS = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 2048,
        "run_timeout": 30,
        "memory_limit_mb": 1024,
        "truncation_warning_threshold": 0.05,
        "timeout_warning_threshold": 0.10,
    }
    print("Warning: eval_config not found, using fallback defaults")

# =============================================================================
# 可选依赖检测
# =============================================================================

# verl 组件：分布式推理框架
try:
    import ray  # Ray 分布式计算框架
    # get_rollout_replica_class: 获取 vLLM/SGLang 的 Replica 类
    from verl.workers.rollout.replica import get_rollout_replica_class
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    print("Warning: verl not available. Using simple mode only.")

# SandboxFusion SDK：代码沙盒评测
try:
    from sandbox_fusion import (
        get_prompts,          # 获取数据集题目列表
        submit,               # 提交代码评测（使用内置数据）
        submit_safe,          # submit 的安全版本（自动处理异常）
        run_code,             # 执行代码（使用外部测试用例）
        GetPromptsRequest,    # get_prompts 的请求参数
        SubmitRequest,        # submit 的请求参数
        RunCodeRequest,       # run_code 的请求参数
        TestConfig,           # 测试配置（语言、超时等）
        set_endpoint as set_sandbox_endpoint,  # 设置服务地址
    )
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    print("Warning: sandbox_fusion not installed. Run: pip install sandbox-fusion")

# verl compute_score：与 GRPO 训练一致的评测函数
try:
    from verl.utils.reward_score.sandbox_fusion import compute_score
    VERL_COMPUTE_SCORE_AVAILABLE = True
except ImportError:
    VERL_COMPUTE_SCORE_AVAILABLE = False


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class EvalConfig:
    """
    评测配置 - 包含所有可配置参数

    分为以下几组：
    - 运行模式：verl（分布式）或 simple（连接已有服务器）
    - 模型配置：模型路径、推理引擎
    - 生成参数：温度、最大长度等
    - 评测配置：SandboxFusion 设置
    - 数据配置：数据集列表、manifest 目录
    - 输出配置：结果保存位置
    """
    # === 运行模式 ===
    mode: str = "verl"  # "verl"（分布式）或 "simple"（简化模式）

    # === 模型配置 ===
    model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # HuggingFace 模型路径

    # === verl Rollout 配置（verl 模式专用） ===
    rollout_name: str = "vllm"        # 推理引擎："vllm" 或 "sglang"
    tensor_parallel_size: int = 2     # Tensor Parallel 大小（跨 GPU 分割模型）
    n_gpus_per_node: int = 8          # 每节点 GPU 数量
    gpu_memory_utilization: float = 0.85  # GPU 显存利用率

    # === 简化模式配置 ===
    vllm_url: str = "http://localhost:8000"  # 已有 vLLM 服务器地址

    # === 解码参数（EVAL@1 协议：贪婪解码） ===
    # 注意：这些默认值来自 eval_config.EVAL_CONSTANTS，确保所有 Phase 一致
    temperature: float = 0.0  # 0.0 = greedy decoding，确保可复现
    top_p: float = 1.0        # nucleus sampling 参数
    max_new_tokens: int = 2048  # 最大生成 token 数

    # === SandboxFusion 配置 ===
    # 注意：run_timeout 从 10 秒改为 30 秒，适应 CodeContests 竞赛题
    sandbox_url: str = "http://localhost:8080"  # SandboxFusion 服务地址
    run_timeout: int = 30     # 代码执行超时（秒）- CodeContests 需要更长时间
    memory_limit_mb: int = 1024  # 内存限制（MB）

    # === 评测方式选择 ===
    use_submit_api: bool = True      # 使用 submit() API（依赖 SandboxFusion 内置数据）
    use_external_tests: bool = True  # 优先使用外部测试用例（从 raw 数据加载）

    # === 数据配置 ===
    datasets: List[str] = field(default_factory=lambda: ["humaneval", "mbpp_reg"])
    manifest_dir: Optional[str] = None  # Manifest 目录，使用去重后的数据

    # === 并发控制 ===
    max_concurrent_requests: int = 64  # 最大并发请求数
    batch_size: int = 50               # 批处理大小

    # === 输出配置 ===
    output_dir: str = "outputs/phase0"  # 结果输出目录
    qa_sample_size: int = 50            # 每个数据集保存的问答样本数

    # === WandB 配置 ===
    use_wandb: bool = False              # 是否启用 WandB 日志
    wandb_project: str = "rlvr_coding_model"
    wandb_run_name: Optional[str] = None


# =============================================================================
# Prompt 模板配置
# =============================================================================

# System Prompt：指导模型生成 Python 代码
# 注意：去掉了 solve() 示例，避免对 HumanEval/MBPP 的输出产生偏置
SYSTEM_PROMPT = """You are an expert Python programmer.

Output rules:
1. Output Python code only.
2. Include necessary imports only if needed.
3. Wrap the entire code in <code> and </code>.
4. Do not write anything outside the <code> tags.
5. Follow dataset-specific constraints given by the user prompt (function-only vs full program)."""

# 针对不同数据集的 User Prompt 模板
# 注意：
# - HumanEval/MBPP：要求输出完整函数定义，禁止 stdin/main guard
# - MBPP：包含 {entry_point} 占位符，需要在 format_prompt 中替换
# - CodeContests：强调代码执行时必须有输出
PROMPT_TEMPLATES = {
    # HumanEval：补全函数（输出完整函数定义）
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

    # MBPP：实现函数（必须指定函数名和调用形式）
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

    # CodeContests：竞赛题（强调代码执行时必须有输出）
    "codecontests_train": """Solve the following competitive programming problem in Python.

Rules:
- Read from stdin and write to stdout.
- Your program MUST produce output when executed (call solve() under main guard, or execute at top-level).
- Use fast I/O if needed (sys.stdin.buffer).
- Do NOT print anything except the required output.

{prompt}

Output ONLY:
<code>
# python code
</code>""",

    "codecontests_valid": """Solve the following competitive programming problem in Python.

Rules:
- Read from stdin and write to stdout.
- Your program MUST produce output when executed (call solve() under main guard, or execute at top-level).
- Use fast I/O if needed (sys.stdin.buffer).
- Do NOT print anything except the required output.

{prompt}

Output ONLY:
<code>
# python code
</code>""",

    "codecontests_test": """Solve the following competitive programming problem in Python.

Rules:
- Read from stdin and write to stdout.
- Your program MUST produce output when executed (call solve() under main guard, or execute at top-level).
- Use fast I/O if needed (sys.stdin.buffer).
- Do NOT print anything except the required output.

{prompt}

Output ONLY:
<code>
# python code
</code>""",
}


def format_prompt(raw_prompt: str, dataset_key: str, entry_point: str = "", example_call: str = "") -> str:
    """
    格式化原始 prompt，添加指令模板

    Args:
        raw_prompt: 原始题目 prompt
        dataset_key: 数据集名称
        entry_point: 函数入口点（MBPP 需要）
        example_call: 调用形式示例（MBPP 需要，如 remove_Occ("hello","l")）

    Returns:
        格式化后的 user prompt
    """
    template = PROMPT_TEMPLATES.get(dataset_key)
    if not template:
        # 默认模板
        return f"""Solve the following problem in Python.

{raw_prompt}

Output ONLY:
<code>
# python code
</code>"""

    # MBPP 需要 entry_point 和 example_call，显式分支避免 KeyError
    if dataset_key == "mbpp_reg":
        if not entry_point:
            # 报错避免静默训练无效样本
            raise ValueError(f"MBPP entry_point is empty for prompt: {raw_prompt[:50]}...")
        if not example_call:
            raise ValueError(f"MBPP example_call is empty for prompt: {raw_prompt[:50]}...")
        return template.format(prompt=raw_prompt, entry_point=entry_point, example_call=example_call)

    # 其他数据集直接格式化
    return template.format(prompt=raw_prompt)


# =============================================================================
# 数据集配置
# =============================================================================

# 数据集到 SandboxFusion 配置的映射
# 用于将内部数据集名称转换为 SandboxFusion 的数据集名称和配置
DATASET_SANDBOX_CONFIG = {
    "humaneval": {
        "sandbox_dataset": "humaneval_python",  # SandboxFusion 中的名称
        "language": "python",
    },
    "mbpp_reg": {
        "sandbox_dataset": "mbpp",
        "language": "python",
        "id_range": (11, 210),  # MBPP Regular 子集
    },
    "codecontests_train": {
        "sandbox_dataset": "code_contests",
        "language": "python",
    },
    "codecontests_valid": {
        "sandbox_dataset": "code_contests",
        "language": "python",
    },
    "codecontests_test": {
        "sandbox_dataset": "code_contests",
        "language": "python",
    },
}


# =============================================================================
# verl Rollout Server 管理
# =============================================================================

async def start_rollout_servers(config: EvalConfig):
    """
    启动 verl Standalone Rollout 服务器

    Standalone 模式特点：
    - 服务器独立运行，不需要训练进程
    - 自动加载模型权重（load_format="auto"）
    - 暴露 OpenAI-compatible API

    核心步骤：
    1. 创建 RolloutConfig 和 ModelConfig
    2. 计算 replica 数量 = GPU数 / tensor_parallel_size
    3. 创建 RolloutReplica 实例
    4. 调用 init_standalone() 启动服务器
    5. 返回服务器地址列表

    参考: verl/trainer/main_generation_server.py
    """
    if not VERL_AVAILABLE:
        raise RuntimeError("verl not available. Please install verl or use --mode simple")

    from omegaconf import OmegaConf  # Hydra 配置库

    # 构建 RolloutConfig - vLLM/SGLang 推理引擎配置
    rollout_config = OmegaConf.create({
        "name": config.rollout_name,           # "vllm" 或 "sglang"
        "mode": "async",                       # 异步模式
        "tensor_model_parallel_size": config.tensor_parallel_size,  # TP 大小
        "data_parallel_size": 1,               # 数据并行大小
        "pipeline_model_parallel_size": 1,     # 管道并行大小
        "temperature": config.temperature,     # 采样温度
        "top_p": config.top_p,
        "response_length": config.max_new_tokens,  # 最大生成长度
        "prompt_length": 4096,                 # 最大 prompt 长度
        "dtype": "bfloat16",                   # 模型精度
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "load_format": "auto",                 # 关键！Standalone 模式必须用 "auto"
        "enforce_eager": True,                 # 禁用 CUDA Graph（更稳定）
        "enable_prefix_caching": True,         # 前缀缓存优化
        "enable_chunked_prefill": True,        # 分块预填充
        "max_num_seqs": 256,                   # 最大并发序列数
        "max_num_batched_tokens": 8192,        # 最大批量 token 数
        "disable_log_stats": True,             # 禁用统计日志
    })

    # 构建 HFModelConfig - HuggingFace 模型配置
    model_config = OmegaConf.create({
        "path": config.model_path,             # 模型路径
        "trust_remote_code": True,             # 信任远程代码（Qwen 等模型需要）
        "load_tokenizer": True,                # 加载 tokenizer
        "lora_rank": 0,                        # LoRA rank（0 = 不使用 LoRA）
    })

    # 计算 replica 数量
    # 例如：8 GPU / 2 TP = 4 个 replica
    num_replicas = config.n_gpus_per_node // config.tensor_parallel_size

    # 初始化 Ray 分布式计算框架
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {
            "TOKENIZERS_PARALLELISM": "true",  # 启用 tokenizer 并行
            "NCCL_DEBUG": "WARN",              # NCCL 调试级别
        }})

    # 获取 Rollout 类（vLLMReplica 或 SGLangReplica）
    rollout_class = get_rollout_replica_class(config.rollout_name)

    # 创建多个 replica 实例
    rollout_servers = [
        rollout_class(
            replica_rank=replica_rank,         # 当前 replica 的编号
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]

    # 初始化 Standalone 模式服务器
    print(f"Initializing {num_replicas} {config.rollout_name} rollout servers...")
    print(f"  Model: {config.model_path}")
    print(f"  Tensor Parallel Size: {config.tensor_parallel_size}")
    print(f"  GPU Memory Utilization: {config.gpu_memory_utilization}")

    # asyncio.gather: 并发执行所有服务器的初始化
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    # 获取每个服务器的 HTTP 地址
    server_addresses = [server._server_address for server in rollout_servers]
    print(f"Rollout servers ready at: {server_addresses}")

    return rollout_servers, server_addresses


# =============================================================================
# 代码生成
# =============================================================================

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
    通过 OpenAI-compatible API 调用 vLLM/SGLang 生成代码

    API 格式：
    POST /v1/chat/completions
    {
        "model": "模型名",
        "messages": [
            {"role": "system", "content": "system_prompt"},
            {"role": "user", "content": "prompt"}
        ],
        "temperature": 0.0,
        "max_tokens": 2048
    }

    Args:
        session: aiohttp 会话对象（复用连接）
        server_address: 服务器地址（不含 http://）
        model_path: 模型名称
        prompt: 输入 prompt（已格式化）
        sampling_params: 采样参数
        semaphore: 信号量，控制并发数
        system_prompt: 可选的系统 prompt

    Returns:
        (completion, metadata)
        - completion: 生成的代码
        - metadata: 包含 token 数、生成时间等信息
    """
    async with semaphore:  # 控制并发数量
        start_time = time.time()

        # 构建 messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # 发送 POST 请求到 OpenAI-compatible API
            async with session.post(
                url=f"http://{server_address}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer token-abc123",  # vLLM 默认接受任意 token
                },
                json={
                    "model": model_path,
                    "messages": messages,
                    **sampling_params  # temperature, max_tokens 等
                },
                timeout=aiohttp.ClientTimeout(total=300),  # 5 分钟超时
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return "", {"error": f"API error {resp.status}: {error_text}", "gen_time": time.time() - start_time}

                data = await resp.json()

                # 提取生成结果
                completion = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                return completion, {
                    "gen_time": time.time() - start_time,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "finish_reason": data["choices"][0].get("finish_reason", "unknown"),
                }

        except asyncio.TimeoutError:
            return "", {"error": "timeout", "gen_time": time.time() - start_time}
        except Exception as e:
            return "", {"error": str(e), "gen_time": time.time() - start_time}


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

    使用 Round-Robin 策略将请求分发到不同服务器

    Args:
        server_addresses: 服务器地址列表
        model_path: 模型名称
        prompts: prompt 列表（已格式化）
        sampling_params: 采样参数
        max_concurrent: 最大并发数
        system_prompt: 可选的系统 prompt

    Returns:
        [(completion, metadata), ...] 与 prompts 一一对应
    """
    # 信号量：限制同时进行的请求数
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            # Round-Robin 负载均衡：第 i 个请求发到第 i % n 个服务器
            server_idx = i % len(server_addresses)
            server_address = server_addresses[server_idx]

            task = generate_code(
                session, server_address, model_path, prompt,
                sampling_params, semaphore, system_prompt
            )
            tasks.append(task)

        # asyncio.gather: 并发执行所有任务，按顺序返回结果
        results = await asyncio.gather(*tasks)

    return results


# =============================================================================
# 代码评测（SandboxFusion）
# =============================================================================

def evaluate_with_submit_api(
    completion: str,
    sandbox_dataset: str,
    sandbox_id: str,
    config: EvalConfig,
) -> EvalResult:
    """
    使用 SandboxFusion submit() API 评测代码

    submit() API 特点：
    - 依赖 SandboxFusion 内置的测试用例数据
    - 自动处理代码提取、编译、执行
    - 返回详细的测试结果

    Args:
        completion: 模型生成的代码
        sandbox_dataset: SandboxFusion 数据集名称
        sandbox_id: 问题 ID
        config: 评测配置

    Returns:
        EvalResult 包含 accepted、pass_ratio、error_type 等
    """
    if not SANDBOX_AVAILABLE:
        return EvalResult(
            problem_id=sandbox_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="sdk_unavailable",
            judge_time=0.0,
            details={"error": "SandboxFusion SDK not available"},
        )

    start_time = time.time()

    # 空输出处理
    if not completion or not completion.strip():
        return EvalResult(
            problem_id=sandbox_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="empty_output",
            judge_time=time.time() - start_time,
            details={},
        )

    try:
        # 设置 SandboxFusion 服务地址
        set_sandbox_endpoint(config.sandbox_url)

        # submit_safe: submit 的安全版本，自动捕获异常
        # 参数：
        #   dataset: 数据集名称
        #   id: 问题 ID
        #   completion: 提交的代码
        #   config: 测试配置（语言、超时等）
        result = submit_safe(SubmitRequest(
            dataset=sandbox_dataset,
            id=sandbox_id,
            completion=completion,
            config=TestConfig(
                language='python',
                run_timeout=config.run_timeout,
            )
        ))

        judge_time = time.time() - start_time

        # 解析评测结果
        accepted = result.accepted  # 是否全部通过
        tests = result.tests or []  # 每个测试用例的结果

        # 计算 pass_ratio（通过的测试用例比例）
        if tests:
            passed = sum(1 for t in tests if getattr(t, 'status', '') == "success")
            pass_ratio = passed / len(tests)
        else:
            pass_ratio = 1.0 if accepted else 0.0

        # 确定错误类型
        error_type = "success" if accepted else _determine_error_type(tests)

        return EvalResult(
            problem_id=sandbox_id,
            accepted=accepted,
            pass_ratio=pass_ratio,
            error_type=error_type,
            judge_time=judge_time,
            details={
                "extracted_code": result.extracted_code,  # SandboxFusion 提取的代码
                "test_count": len(tests),
            },
        )

    except Exception as e:
        return EvalResult(
            problem_id=sandbox_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="api_error",
            judge_time=time.time() - start_time,
            details={"error": str(e)},
        )


def evaluate_with_compute_score(
    completion: str,
    test_cases: Dict,
    sandbox_id: str,
    config: EvalConfig,
) -> EvalResult:
    """
    使用 verl compute_score() 评测代码

    compute_score 是 verl 框架用于 GRPO 训练的评分函数
    使用它可以确保评测与训练阶段一致

    Args:
        completion: 模型生成的代码
        test_cases: 测试用例字典
        sandbox_id: 问题 ID
        config: 评测配置

    Returns:
        EvalResult
    """
    if not VERL_COMPUTE_SCORE_AVAILABLE:
        return EvalResult(
            problem_id=sandbox_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="compute_score_unavailable",
            judge_time=0.0,
            details={"error": "verl compute_score not available"},
        )

    start_time = time.time()

    if not completion or not completion.strip():
        return EvalResult(
            problem_id=sandbox_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="empty_output",
            judge_time=time.time() - start_time,
            details={},
        )

    try:
        # compute_score: verl 的评分函数
        # 返回 (score, metadata_list)
        #   score: 0.0-1.0 的分数
        #   metadata_list: 每个测试用例的详细结果
        score, metadata_list = compute_score(
            sandbox_fusion_url=f"{config.sandbox_url}/run_code",
            memory_limit_mb=config.memory_limit_mb,
            completion=completion,
            test_cases=test_cases,
            continuous=False,  # False = 二值评分（0 或 1）
            timeout=config.run_timeout,
        )

        judge_time = time.time() - start_time
        accepted = (score == 1.0)

        error_type = _determine_error_type_from_metadata(metadata_list)

        return EvalResult(
            problem_id=sandbox_id,
            accepted=accepted,
            pass_ratio=score,
            error_type=error_type,
            judge_time=judge_time,
            details={"metadata": metadata_list},
        )

    except Exception as e:
        return EvalResult(
            problem_id=sandbox_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="api_error",
            judge_time=time.time() - start_time,
            details={"error": str(e)},
        )


def _determine_error_type(tests) -> str:
    """
    从 submit() 返回的测试结果确定错误类型

    错误类型优先级：
    1. syntax_error / compile_error: 语法错误
    2. runtime_error: 运行时错误
    3. timeout: 超时
    4. wrong_answer: 输出不正确
    """
    for test in tests:
        status = getattr(test, 'status', 'unknown')
        if status == "syntax_error" or status == "compile_error":
            return "syntax_error"
        elif status == "runtime_error":
            return "runtime_error"
        elif status == "timeout":
            return "timeout"
    return "wrong_answer"


def _determine_error_type_from_metadata(metadata_list) -> str:
    """从 compute_score 的 metadata 确定错误类型"""
    if not metadata_list:
        return "unknown"

    for meta in metadata_list:
        status = meta.get('status', 'unknown')
        if status == "compile_error":
            return "syntax_error"
        elif status == "runtime_error":
            return "runtime_error"
        elif status == "timeout":
            return "timeout"
        elif status == "success":
            continue

    return "wrong_answer"


def _parse_run_code_result(result) -> Tuple[str, str]:
    """
    解析 SandboxFusion run_code API 的返回结果

    run_code 返回的对象结构：
    - result.status: RunStatus 枚举（如 RunStatus.Success）
    - result.run_result: 对象，包含：
      - status: CommandRunStatus 枚举（如 CommandRunStatus.Finished）
      - stdout: 标准输出字符串
      - stderr: 标准错误字符串

    Returns:
        (status_str, output_str)
        - status_str: 状态字符串（如 "Success", "Finished"）
        - output_str: 输出内容（stdout + stderr）
    """
    # 获取顶层状态（枚举转字符串）
    status = str(getattr(result, 'status', 'unknown'))

    # 获取 run_result 对象
    run_result = getattr(result, 'run_result', None)

    if run_result is not None and hasattr(run_result, 'stdout'):
        # run_result 是结构化对象
        stdout = run_result.stdout or ""
        stderr = run_result.stderr or ""
        output = stdout + stderr

        # 也检查内部状态
        inner_status = str(getattr(run_result, 'status', ''))
        if "Finished" in inner_status:
            status = "Finished"
    else:
        # 回退：直接转字符串
        output = str(run_result) if run_result else ""

    return status, output


def _extract_code_from_completion(completion: str) -> str:
    """
    从模型输出中提取代码

    提取优先级：
    1. <code>...</code> 标签（取最长的匹配，大小写不敏感）
    2. ```python ... ``` Markdown 代码块（取最长的匹配）
    3. 原始内容（无包装）

    这个函数用于处理模型输出中可能包含的解释文本
    改进：
    - 取最长的 code block，避免取到残缺块
    - 大小写不敏感，支持 <CODE>...</CODE>
    """
    import re

    # 1. 优先尝试 <code>...</code> 标签（大小写不敏感）
    # re.DOTALL: 让 . 匹配换行符
    # re.IGNORECASE: 大小写不敏感
    code_tag_pattern = r'<code>(.*?)</code>'
    matches = re.findall(code_tag_pattern, completion, re.DOTALL | re.IGNORECASE)
    if matches:
        # 取最长的匹配，避免取到残缺块
        return max(matches, key=len).strip()

    # 2. 尝试 markdown 代码块 ```python ... ```
    md_pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(md_pattern, completion, re.DOTALL)
    if matches:
        # 取最长的匹配
        return max(matches, key=len).strip()

    # 3. 返回原始内容
    return completion.strip()


def evaluate_with_run_code(
    completion: str,
    test_cases: Dict[str, Any],
    problem_id: str,
    config: EvalConfig,
) -> EvalResult:
    """
    使用 SandboxFusion run_code API 和外部测试用例评测代码

    run_code API 特点：
    - 直接执行代码，不依赖 SandboxFusion 内置数据
    - 需要自己提供测试用例
    - 更灵活，适用于自定义数据集

    支持三种测试用例格式：
    - humaneval: Python test code with assert statements
    - mbpp: List of assert strings
    - codecontests: List of {input, output} pairs (stdin/stdout)

    Args:
        completion: 模型生成的代码
        test_cases: 测试用例字典，包含 type 和具体测试数据
        problem_id: 问题 ID
        config: 评测配置

    Returns:
        EvalResult
    """
    if not SANDBOX_AVAILABLE:
        return EvalResult(
            problem_id=problem_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="sdk_unavailable",
            judge_time=0.0,
            details={"error": "SandboxFusion SDK not available"},
        )

    start_time = time.time()

    if not completion or not completion.strip():
        return EvalResult(
            problem_id=problem_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="empty_output",
            judge_time=time.time() - start_time,
            details={},
        )

    # 获取测试用例类型
    test_type = test_cases.get("type", "unknown")

    try:
        set_sandbox_endpoint(config.sandbox_url)

        # 根据测试用例类型选择评测函数
        if test_type == "humaneval":
            return _evaluate_humaneval(completion, test_cases, problem_id, config, start_time)
        elif test_type == "mbpp":
            return _evaluate_mbpp(completion, test_cases, problem_id, config, start_time)
        elif test_type == "codecontests":
            return _evaluate_codecontests(completion, test_cases, problem_id, config, start_time)
        else:
            return EvalResult(
                problem_id=problem_id,
                accepted=False,
                pass_ratio=0.0,
                error_type="unknown_test_type",
                judge_time=time.time() - start_time,
                details={"error": f"Unknown test type: {test_type}"},
            )

    except Exception as e:
        return EvalResult(
            problem_id=problem_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="api_error",
            judge_time=time.time() - start_time,
            details={"error": str(e)},
        )


def _evaluate_humaneval(
    completion: str,
    test_cases: Dict[str, Any],
    problem_id: str,
    config: EvalConfig,
    start_time: float,
) -> EvalResult:
    """
    评测 HumanEval 格式的测试用例

    HumanEval 测试用例格式：
    {
        "type": "humaneval",
        "test_code": "def check(candidate):\n    assert candidate(1) == 2\n    ...",
        "entry_point": "function_name"
    }

    评测步骤：
    1. 提取模型生成的代码
    2. 拼接代码 + 测试代码 + check(entry_point) 调用
    3. 执行完整代码
    4. 根据执行结果判断是否通过
    """
    test_code = test_cases.get("test_code", "")      # 测试函数（包含 check 定义）
    entry_point = test_cases.get("entry_point", "")  # 被测函数名

    # 提取代码（处理 <code> 标签等）
    code = _extract_code_from_completion(completion)

    # 组装完整测试代码：
    # 1. 模型生成的函数定义
    # 2. 测试代码（定义 check 函数）
    # 3. 调用 check(entry_point)
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"

    # 调用 run_code API 执行代码
    result = run_code(RunCodeRequest(
        code=full_code,
        language="python",
        run_timeout=config.run_timeout,
        memory_limit_mb=config.memory_limit_mb,  # 添加内存限制
    ))

    judge_time = time.time() - start_time

    # 使用辅助函数解析结果
    status, output = _parse_run_code_result(result)

    # 判断是否通过（检查 Success 或 Finished）
    accepted = "Success" in status or "Finished" in status

    # 确定错误类型
    if accepted:
        error_type = "success"
    elif "SyntaxError" in output:
        error_type = "syntax_error"
    elif "timeout" in status.lower() or "timeout" in output.lower():
        error_type = "timeout"
    elif "Error" in output:
        error_type = "runtime_error"
    else:
        error_type = "wrong_answer"

    return EvalResult(
        problem_id=problem_id,
        accepted=accepted,
        pass_ratio=1.0 if accepted else 0.0,
        error_type=error_type,
        judge_time=judge_time,
        details={
            "status": status,
            "output": output[:500],  # 截取前 500 字符
        },
    )


def _evaluate_mbpp(
    completion: str,
    test_cases: Dict[str, Any],
    problem_id: str,
    config: EvalConfig,
    start_time: float,
) -> EvalResult:
    """
    评测 MBPP 格式的测试用例

    MBPP 测试用例格式：
    {
        "type": "mbpp",
        "test_list": ["assert func(1) == 2", "assert func(3) == 4", ...],
        "test_setup_code": "# 可选的初始化代码"
    }

    评测步骤：
    1. 提取代码
    2. 拼接 setup_code + 代码 + assert 语句
    3. 执行完整代码
    4. 所有 assert 通过 = 成功
    """
    test_list = test_cases.get("test_list", [])          # assert 语句列表
    test_setup_code = test_cases.get("test_setup_code", "")  # 初始化代码

    if not test_list:
        return EvalResult(
            problem_id=problem_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="no_test_cases",
            judge_time=time.time() - start_time,
            details={"error": "No test cases available"},
        )

    # 提取代码
    code = _extract_code_from_completion(completion)

    # 组装完整测试代码：
    # 1. 初始化代码（如 import 语句）
    # 2. 模型生成的函数定义
    # 3. 所有 assert 语句
    test_code = "\n".join(test_list)
    full_code = f"{test_setup_code}\n\n{code}\n\n{test_code}"

    result = run_code(RunCodeRequest(
        code=full_code,
        language="python",
        run_timeout=config.run_timeout,
        memory_limit_mb=config.memory_limit_mb,  # 添加内存限制
    ))

    judge_time = time.time() - start_time

    # 使用辅助函数解析结果
    status, output = _parse_run_code_result(result)

    # 判断是否通过（检查 Success 或 Finished）
    accepted = "Success" in status or "Finished" in status

    if accepted:
        error_type = "success"
    elif "SyntaxError" in output:
        error_type = "syntax_error"
    elif "timeout" in status.lower():
        error_type = "timeout"
    elif "Error" in output:
        error_type = "runtime_error"
    else:
        error_type = "wrong_answer"

    return EvalResult(
        problem_id=problem_id,
        accepted=accepted,
        pass_ratio=1.0 if accepted else 0.0,
        error_type=error_type,
        judge_time=judge_time,
        details={
            "status": status,
            "output": output[:500],
            "test_count": len(test_list),
        },
    )


def _evaluate_codecontests(
    completion: str,
    test_cases: Dict[str, Any],
    problem_id: str,
    config: EvalConfig,
    start_time: float,
) -> EvalResult:
    """
    评测 CodeContests 格式的测试用例（stdin/stdout）

    CodeContests 测试用例格式：
    {
        "type": "codecontests",
        "tests": [
            {"input": "5\n1 2 3 4 5", "output": "15"},
            {"input": "3\n1 2 3", "output": "6"},
            ...
        ]
    }

    评测步骤：
    1. 提取代码
    2. 对每个测试用例：
       - 执行代码，传入 stdin
       - 比较实际输出与期望输出
    3. 统计通过率
    """
    tests = test_cases.get("tests", [])

    if not tests:
        return EvalResult(
            problem_id=problem_id,
            accepted=False,
            pass_ratio=0.0,
            error_type="no_test_cases",
            judge_time=time.time() - start_time,
            details={"error": "No test cases available"},
        )

    # 提取代码
    code = _extract_code_from_completion(completion)

    passed = 0
    total = len(tests)
    error_type = "success"
    last_error = ""

    # 逐个执行测试用例
    for tc in tests:
        stdin_input = tc.get("input", "")
        expected_output = tc.get("output", "").strip()

        # run_code 支持 stdin 参数
        result = run_code(RunCodeRequest(
            code=code,
            language="python",
            run_timeout=config.run_timeout,
            memory_limit_mb=config.memory_limit_mb,  # 添加内存限制
            stdin=stdin_input,  # 传入标准输入
        ))

        # 使用辅助函数解析结果
        status, output = _parse_run_code_result(result)
        actual_output = output.strip()

        # 判断是否通过（检查 Success 或 Finished）
        if "Success" in status or "Finished" in status:
            # 精确比较输出
            if actual_output == expected_output:
                passed += 1
            else:
                error_type = "wrong_answer"
                last_error = f"Expected: {expected_output[:100]}, Got: {actual_output[:100]}"
        elif "SyntaxError" in actual_output:
            error_type = "syntax_error"
            last_error = actual_output[:200]
            break  # 语法错误直接退出
        elif "timeout" in status.lower():
            error_type = "timeout"
            last_error = "Execution timeout"
            break
        else:
            error_type = "runtime_error"
            last_error = actual_output[:200]

    judge_time = time.time() - start_time
    pass_ratio = passed / total if total > 0 else 0.0
    accepted = (passed == total)

    if accepted:
        error_type = "success"

    return EvalResult(
        problem_id=problem_id,
        accepted=accepted,
        pass_ratio=pass_ratio,
        error_type=error_type,
        judge_time=judge_time,
        details={
            "passed": passed,
            "total": total,
            "last_error": last_error,
        },
    )


# =============================================================================
# 数据加载
# =============================================================================

def load_prompts(dataset_key: str, config: EvalConfig) -> List[Dict[str, Any]]:
    """
    加载评测数据

    两种数据源：
    1. manifest_dir: 从本地 manifest + raw 文件加载（包含测试用例）
    2. SandboxFusion: 从在线服务加载（仅 prompt）

    Returns:
        [{problem_id, prompt, sandbox_dataset, test_cases?, ...}, ...]
    """
    if config.manifest_dir:
        return _load_from_manifest(dataset_key, config.manifest_dir)
    else:
        return _load_from_sandbox(dataset_key, config.sandbox_url)


def _load_from_sandbox(dataset_key: str, sandbox_url: str) -> List[Dict[str, Any]]:
    """
    从 SandboxFusion 在线服务加载数据

    注意：这种方式只能获取 prompt，不包含测试用例
    """
    if not SANDBOX_AVAILABLE:
        print(f"  Warning: SandboxFusion SDK not available")
        return []

    cfg = DATASET_SANDBOX_CONFIG.get(dataset_key, {})
    sandbox_dataset = cfg.get("sandbox_dataset", dataset_key)
    id_range = cfg.get("id_range")

    set_sandbox_endpoint(sandbox_url)

    try:
        # get_prompts: 获取数据集的所有题目
        prompts = get_prompts(GetPromptsRequest(
            dataset=sandbox_dataset,
            config={"language": cfg.get("language", "python")}
        ))
    except Exception as e:
        print(f"  Error loading {dataset_key}: {e}")
        return []

    result = []
    for p in prompts:
        pid = str(p.id)

        # ID 范围过滤（用于 MBPP Regular）
        if id_range:
            try:
                id_num = int(pid)
                if id_num < id_range[0] or id_num > id_range[1]:
                    continue
            except ValueError:
                pass

        result.append({
            "problem_id": pid,
            "prompt": p.prompt,
            "sandbox_dataset": sandbox_dataset,
        })

    return result


def _load_from_manifest(dataset_key: str, manifest_dir: str) -> List[Dict[str, Any]]:
    """
    从本地 manifest 和 raw 文件加载数据

    文件结构：
    - {manifest_dir}/{dataset}_manifest.jsonl  # 去重后的 problem_id 列表
    - {manifest_dir}/../raw/{dataset}_raw.jsonl  # 完整数据（含测试用例）

    这种方式可以获取测试用例，用于 run_code API 评测
    """
    manifest_path = Path(manifest_dir) / f"{dataset_key}_manifest.jsonl"
    raw_path = Path(manifest_dir).parent / "raw" / f"{dataset_key}_raw.jsonl"

    if not manifest_path.exists():
        print(f"  Warning: Manifest not found: {manifest_path}")
        return []

    # 从 manifest 加载 problem_id 列表（这些是去重后的）
    problem_ids = set()
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            entry = json.loads(line)
            problem_ids.add(entry["problem_id"])

    if not raw_path.exists():
        print(f"  Warning: Raw data not found: {raw_path}")
        return []

    cfg = DATASET_SANDBOX_CONFIG.get(dataset_key, {})
    sandbox_dataset = cfg.get("sandbox_dataset", dataset_key)

    # 从 raw 文件加载完整数据
    result = []
    has_test_cases = False
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            record = json.loads(line)
            # 只加载在 manifest 中的题目（去重后的）
            if record["problem_id"] in problem_ids:
                item = {
                    "problem_id": record["problem_id"],
                    "prompt": record["prompt"],
                    "sandbox_dataset": sandbox_dataset,
                }

                # 加载测试用例（如果存在）
                if "test_cases" in record:
                    item["test_cases"] = record["test_cases"]
                    has_test_cases = True

                # 加载参考解答（如果存在，用于调试）
                if "canonical_solution" in record:
                    item["canonical_solution"] = record["canonical_solution"]

                result.append(item)

    if has_test_cases:
        print(f"  Loaded test cases for {len(result)} problems")

    return result


# =============================================================================
# 主评测流程
# =============================================================================

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
    2. 批量生成代码（并发请求多个 vLLM replica）
    3. 逐个判题（评测通常是同步的）
    4. 收集指标和日志
    5. 返回数据集级别的统计信息

    Args:
        dataset_key: 数据集名称
        prompts: 题目列表
        server_addresses: vLLM 服务器地址列表
        config: 评测配置
        metrics_collector: 指标收集器
        qa_logger: 问答日志记录器

    Returns:
        数据集级别的统计信息
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_key} ({len(prompts)} problems)")
    print(f"{'='*60}")

    # 采样参数（EVAL@1 协议）
    sampling_params = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_new_tokens,
    }

    results = []
    total_gen_tokens = 0
    total_gen_time = 0.0
    total_judge_time = 0.0
    truncation_count = 0  # 截断计数（finish_reason == "length"）
    timeout_count = 0     # 超时计数

    # 记录开始时间（用于计算 throughput）
    dataset_start_time = time.time()

    # 分批处理
    for batch_start in range(0, len(prompts), config.batch_size):
        batch_end = min(batch_start + config.batch_size, len(prompts))
        batch = prompts[batch_start:batch_end]

        print(f"  Processing batch {batch_start//config.batch_size + 1}/{(len(prompts)-1)//config.batch_size + 1}...")

        # 1. 批量生成代码
        # 格式化 prompt：添加指令模板，要求模型使用 Python 并用 <code> 包裹
        # 注意：只有 MBPP 数据集需要 entry_point 和 example_call
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
        gen_results = await batch_generate(
            server_addresses,
            config.model_path,
            prompt_texts,
            sampling_params,
            config.max_concurrent_requests,
            system_prompt=SYSTEM_PROMPT,  # 使用系统 prompt
        )

        # 2. 逐个判题
        for i, (generated_code, gen_meta) in enumerate(gen_results):
            problem_id = batch[i]["problem_id"]
            prompt = batch[i]["prompt"]
            sandbox_dataset = batch[i]["sandbox_dataset"]
            test_cases = batch[i].get("test_cases")  # 可能为 None

            # 累计生成指标
            gen_tokens = gen_meta.get("completion_tokens", 0)
            gen_time = gen_meta.get("gen_time", 0.0)
            finish_reason = gen_meta.get("finish_reason", "unknown")
            total_gen_tokens += gen_tokens
            total_gen_time += gen_time

            # 检查是否被截断（finish_reason == "length" 表示达到 max_tokens 上限）
            if finish_reason == "length":
                truncation_count += 1

            # 根据配置选择评测方式
            if test_cases and config.use_external_tests:
                # 优先使用外部测试用例 + run_code API
                eval_result = evaluate_with_run_code(
                    generated_code, test_cases, problem_id, config
                )
            elif config.use_submit_api:
                # 使用 submit API（依赖 SandboxFusion 内置数据）
                eval_result = evaluate_with_submit_api(
                    generated_code, sandbox_dataset, problem_id, config
                )
            else:
                # 回退到 submit API
                eval_result = evaluate_with_submit_api(
                    generated_code, sandbox_dataset, problem_id, config
                )

            # 补充 gen_tokens 和 gen_time 到 eval_result
            eval_result.gen_tokens = gen_tokens
            eval_result.gen_time = gen_time

            total_judge_time += eval_result.judge_time

            # 统计超时错误
            if eval_result.error_type == "timeout":
                timeout_count += 1

            # 记录到指标收集器
            metrics_collector.add_result(dataset_key, eval_result)

            # 记录问答日志（用于后续分析）
            qa_logger.log(
                dataset=dataset_key,
                problem_id=problem_id,
                prompt=prompt,
                response=generated_code,
                eval_result=eval_result,
                gen_metadata=gen_meta,
            )

            results.append({
                "problem_id": problem_id,
                "accepted": eval_result.accepted,
                "pass_ratio": eval_result.pass_ratio,
                "error_type": eval_result.error_type,
            })

    # 计算 wall_clock_time 并设置到 metrics_collector
    wall_clock_time = time.time() - dataset_start_time
    metrics_collector.set_wall_clock_time(dataset_key, wall_clock_time)

    # 计算数据集级别的统计信息
    accepted_count = sum(1 for r in results if r["accepted"])
    pass_ratios = np.array([r["pass_ratio"] for r in results]) if results else np.array([])

    # 计算 throughput
    throughput = len(results) / wall_clock_time if wall_clock_time > 0 else 0.0

    # 计算 cost_per_solved
    if accepted_count > 0:
        cost_per_solved_tokens = total_gen_tokens / accepted_count
        cost_per_solved_judge_time = total_judge_time / accepted_count
    else:
        cost_per_solved_tokens = float('inf')
        cost_per_solved_judge_time = float('inf')

    # 计算截断率和超时率
    truncation_rate = truncation_count / len(results) if results else 0.0
    timeout_rate = timeout_count / len(results) if results else 0.0

    dataset_metrics = {
        "total_problems": len(results),
        # 质量指标
        "accepted_at_1": accepted_count / len(results) if results else 0.0,  # 主指标
        "pass_ratio_mean": float(np.mean(pass_ratios)) if len(pass_ratios) > 0 else 0.0,
        "pass_ratio_p50": float(np.median(pass_ratios)) if len(pass_ratios) > 0 else 0.0,
        "pass_ratio_p90": float(np.percentile(pass_ratios, 90)) if len(pass_ratios) > 0 else 0.0,
        # 成本指标
        "total_gen_tokens": total_gen_tokens,
        "avg_gen_tokens": total_gen_tokens / len(results) if results else 0.0,
        "total_gen_time": total_gen_time,
        "avg_gen_time": total_gen_time / len(results) if results else 0.0,
        "total_judge_time": total_judge_time,
        "avg_judge_time": total_judge_time / len(results) if results else 0.0,
        "wall_clock_time": wall_clock_time,
        "throughput": throughput,
        "cost_per_solved_tokens": cost_per_solved_tokens,
        "cost_per_solved_judge_time": cost_per_solved_judge_time,
        # 质量控制指标
        "truncation_count": truncation_count,
        "truncation_rate": truncation_rate,
        "timeout_count": timeout_count,
        "timeout_rate": timeout_rate,
    }

    print(f"\n  Results for {dataset_key}:")
    print(f"    accepted@1: {dataset_metrics['accepted_at_1']:.2%}")
    print(f"    pass_ratio_mean: {dataset_metrics['pass_ratio_mean']:.4f}")
    print(f"    avg_gen_tokens: {dataset_metrics['avg_gen_tokens']:.1f}")
    print(f"    throughput: {throughput:.2f} problems/sec")
    print(f"    truncation_rate: {truncation_rate:.2%} ({truncation_count}/{len(results)})")
    print(f"    timeout_rate: {timeout_rate:.2%} ({timeout_count}/{len(results)})")

    # 质量控制警告
    truncation_threshold = EVAL_CONSTANTS.get("truncation_warning_threshold", 0.05)
    timeout_threshold = EVAL_CONSTANTS.get("timeout_warning_threshold", 0.10)

    if truncation_rate > truncation_threshold:
        print(f"    ⚠️  WARNING: truncation_rate ({truncation_rate:.2%}) > {truncation_threshold:.0%}")
        print(f"       Consider increasing max_new_tokens (current: {config.max_new_tokens})")

    if timeout_rate > timeout_threshold:
        print(f"    ⚠️  WARNING: timeout_rate ({timeout_rate:.2%}) > {timeout_threshold:.0%}")
        print(f"       Consider increasing run_timeout (current: {config.run_timeout}s)")

    return dataset_metrics


async def run_evaluation(config: EvalConfig):
    """
    运行完整评测流程

    主流程：
    1. 初始化 vLLM 服务器（verl 模式）或连接已有服务器（simple 模式）
    2. 初始化组件（指标收集器、问答日志、WandB）
    3. 依次评测每个数据集
    4. 保存结果（metrics.json、summary.json、qa_logs/）
    5. 清理资源
    """
    print("\n" + "="*70)
    print("   Phase 0 Baseline Evaluation (verl Standalone Rollout)")
    print("="*70)
    print(f"Mode: {config.mode}")
    print(f"Model: {config.model_path}")
    print(f"Datasets: {config.datasets}")
    print(f"Output: {config.output_dir}")

    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取服务器地址
    if config.mode == "verl":
        # verl 分布式模式：启动多个 vLLM replica
        print("\n[Starting verl Rollout Servers]")
        rollout_servers, server_addresses = await start_rollout_servers(config)
    else:
        # 简化模式：连接已有的 vLLM 服务器
        print(f"\n[Simple Mode] Connecting to {config.vllm_url}")
        # 去掉 http:// 前缀
        server_addresses = [config.vllm_url.replace("http://", "")]
        rollout_servers = None

    # 初始化组件
    metrics_collector = MetricsCollector()
    qa_logger = QALogger(output_dir / "qa_logs", sample_size=config.qa_sample_size)

    # WandB 初始化（可选）
    if config.use_wandb:
        try:
            import wandb
            run_name = config.wandb_run_name or f"phase0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project=config.wandb_project, name=run_name)
            wandb.config.update(asdict(config))  # 记录配置
        except ImportError:
            print("Warning: wandb not installed")
            config.use_wandb = False

    all_metrics = {}

    try:
        # 评测每个数据集
        for dataset_key in config.datasets:
            print(f"\n[Loading {dataset_key}]")
            prompts = load_prompts(dataset_key, config)

            if not prompts:
                print(f"  No prompts found, skipping...")
                continue

            print(f"  Loaded {len(prompts)} problems")

            dataset_metrics = await evaluate_dataset(
                dataset_key,
                prompts,
                server_addresses,
                config,
                metrics_collector,
                qa_logger,
            )

            all_metrics[dataset_key] = dataset_metrics

            # 记录到 WandB
            if config.use_wandb:
                import wandb
                for key, value in dataset_metrics.items():
                    # 跳过 inf 值（WandB 不支持）
                    if isinstance(value, float) and value == float('inf'):
                        continue
                    wandb.log({f"eval/{dataset_key}/{key}": value})

    finally:
        # 清理 verl 服务器（Ray 会自动清理）
        if rollout_servers:
            print("\n[Shutting down Rollout Servers]")

    # ==========================================================================
    # 保存结果
    # ==========================================================================
    print("\n" + "="*70)
    print("   Saving Results")
    print("="*70)

    # 保存每个数据集的指标（处理 inf 值）
    def handle_inf(obj):
        """将 inf 值转换为 null，便于 JSON 序列化"""
        if isinstance(obj, dict):
            return {k: handle_inf(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [handle_inf(v) for v in obj]
        elif isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')):
            return None  # JSON 不支持 inf
        return obj

    with open(output_dir / "metrics.json", 'w', encoding='utf-8') as f:
        json.dump(handle_inf(all_metrics), f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {output_dir / 'metrics.json'}")

    # 保存问答日志
    qa_logger.save()
    print(f"QA logs saved to: {output_dir / 'qa_logs'}")

    # 保存详细统计（错误分布等）
    summary = metrics_collector.get_summary()
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(handle_inf(summary), f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {output_dir / 'summary.json'}")

    # 打印汇总
    print("\n--- Final Results ---")
    for dataset_key, metrics in all_metrics.items():
        print(f"\n{dataset_key}:")
        print(f"  Quality:")
        print(f"    accepted@1: {metrics['accepted_at_1']:.2%}")
        print(f"    pass_ratio_mean: {metrics['pass_ratio_mean']:.4f}")
        print(f"    pass_ratio_p50: {metrics['pass_ratio_p50']:.4f}")
        print(f"    pass_ratio_p90: {metrics['pass_ratio_p90']:.4f}")
        print(f"  Cost:")
        print(f"    avg_gen_tokens: {metrics['avg_gen_tokens']:.1f}")
        print(f"    avg_judge_time: {metrics['avg_judge_time']:.2f}s")
        print(f"    throughput: {metrics['throughput']:.2f} problems/sec")
        cps_tokens = metrics['cost_per_solved_tokens']
        cps_time = metrics['cost_per_solved_judge_time']
        if cps_tokens != float('inf'):
            print(f"    cost_per_solved_tokens: {cps_tokens:.1f}")
        if cps_time != float('inf'):
            print(f"    cost_per_solved_judge_time: {cps_time:.2f}s")

    if config.use_wandb:
        import wandb
        wandb.finish()

    return all_metrics


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description="Phase 0 Baseline Evaluation (verl Standalone Rollout)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # verl 分布式模式（GPU 服务器）
  python src/phase0_eval.py \\
      --mode verl \\
      --model Qwen/Qwen2.5-Coder-7B-Instruct \\
      --rollout vllm \\
      --tensor_parallel_size 2 \\
      --n_gpus 8 \\
      --datasets humaneval mbpp_reg

  # 简化模式（连接已有 vLLM 服务器）
  python src/phase0_eval.py \\
      --mode simple \\
      --vllm_url http://localhost:8000 \\
      --datasets humaneval mbpp_reg
        """
    )

    # === 模式选择 ===
    parser.add_argument("--mode", type=str, default="simple",
                        choices=["verl", "simple"],
                        help="运行模式: verl (分布式) 或 simple (简化)")

    # === 模型配置 ===
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="模型路径（HuggingFace 格式）")

    # === verl 配置 ===
    parser.add_argument("--rollout", type=str, default="vllm",
                        choices=["vllm", "sglang"],
                        help="Rollout 引擎")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Tensor Parallel 大小（跨 GPU 分割模型）")
    parser.add_argument("--n_gpus", type=int, default=8,
                        help="GPU 数量")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="GPU 显存利用率 (0.0-1.0)")

    # === 简化模式配置 ===
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000",
                        help="vLLM 服务器地址（简化模式使用）")

    # === SandboxFusion 配置 ===
    parser.add_argument("--sandbox_url", type=str, default="http://localhost:8080",
                        help="SandboxFusion 服务地址")
    parser.add_argument("--run_timeout", type=int, default=10,
                        help="代码执行超时（秒）")

    # === 生成配置 ===
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度（0.0 = greedy，用于 EVAL@1）")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="最大生成 token 数")

    # === 评测方式 ===
    parser.add_argument("--use_external_tests", action="store_true", default=True,
                        help="使用外部测试用例（从 raw 数据加载）")
    parser.add_argument("--use_submit_api", action="store_true",
                        help="使用 SandboxFusion submit API（依赖内置数据）")

    # === 数据集配置 ===
    parser.add_argument("--datasets", nargs="+", type=str,
                        default=["humaneval", "mbpp_reg"],
                        help="要评测的数据集列表")
    parser.add_argument("--manifest_dir", type=str, default=None,
                        help="Manifest 目录（使用去重后的数据，包含测试用例）")

    # === 输出配置 ===
    parser.add_argument("--output_dir", type=str, default="outputs/phase0",
                        help="输出目录")
    parser.add_argument("--qa_sample_size", type=int, default=50,
                        help="每个数据集保存的 QA 样本数")

    # === WandB 配置 ===
    parser.add_argument("--use_wandb", action="store_true",
                        help="启用 WandB 日志记录")
    parser.add_argument("--wandb_project", type=str, default="rlvr_coding_model",
                        help="WandB 项目名")

    # === 并发配置 ===
    parser.add_argument("--max_concurrent", type=int, default=64,
                        help="最大并发请求数")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="批处理大小")

    args = parser.parse_args()

    # 创建配置对象
    config = EvalConfig(
        mode=args.mode,
        model_path=args.model,
        rollout_name=args.rollout,
        tensor_parallel_size=args.tensor_parallel_size,
        n_gpus_per_node=args.n_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_url=args.vllm_url,
        sandbox_url=args.sandbox_url,
        run_timeout=args.run_timeout,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        use_submit_api=args.use_submit_api,
        use_external_tests=args.use_external_tests,
        datasets=args.datasets,
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
        qa_sample_size=args.qa_sample_size,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        max_concurrent_requests=args.max_concurrent,
        batch_size=args.batch_size,
    )

    # 检查模式兼容性
    if config.mode == "verl" and not VERL_AVAILABLE:
        print("Error: verl mode requires verl package. Use --mode simple instead.")
        sys.exit(1)

    # 运行评测（异步）
    asyncio.run(run_evaluation(config))


# 程序入口
if __name__ == "__main__":
    main()
