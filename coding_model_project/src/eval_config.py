#!/usr/bin/env python3
"""
评测配置常量 - 所有 Phase 共用
================================

本文件集中管理评测实验的所有配置参数，确保：
1. Baseline 到 SFT 到 RL 阶段使用一致的评测参数
2. 参数变更有据可查，便于复现
3. 避免硬编码散落在各个脚本中

使用方式：
    from eval_config import EVAL_CONSTANTS, get_sampling_params, get_sandbox_config
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# =============================================================================
# 核心评测常量（所有 Phase 共用，禁止随意修改）
# =============================================================================

EVAL_CONSTANTS = {
    # -------------------------------------------------------------------------
    # 解码参数（EVAL@1 协议：贪婪解码）
    # -------------------------------------------------------------------------
    "temperature": 0.0,           # greedy decoding，确保可复现
    "top_p": 1.0,                 # temperature=0 时实际不起作用
    "max_new_tokens": 2048,       # 最大生成 token 数

    # -------------------------------------------------------------------------
    # SandboxFusion 配置
    # -------------------------------------------------------------------------
    "run_timeout": 30,            # 代码执行超时（秒）- CodeContests 需要更长时间
    "memory_limit_mb": 1024,      # 内存限制（MB）- 1GB 足够大多数题目

    # -------------------------------------------------------------------------
    # 训练 Rollout 参数（与评测分开，仅供 GRPO 阶段参考）
    # -------------------------------------------------------------------------
    "rollout_temperature": 0.7,   # 训练时需要探索
    "rollout_top_p": 0.95,        # 训练时的 nucleus sampling
    "rollout_n": 5,               # GRPO 每题采样数

    # -------------------------------------------------------------------------
    # 质量控制阈值
    # -------------------------------------------------------------------------
    "truncation_warning_threshold": 0.05,  # 截断率超过 5% 发出警告
    "timeout_warning_threshold": 0.10,     # 超时率超过 10% 发出警告
}


# =============================================================================
# 数据集特定配置
# =============================================================================

DATASET_CONFIGS = {
    "humaneval": {
        "sandbox_dataset": "humaneval_python",
        "language": "python",
        "expected_problems": 164,
        "typical_accepted_rate": (0.30, 0.45),  # 7B Instruct 模型典型范围
        "run_timeout": 30,  # HumanEval 测试简单，30秒足够
    },
    "mbpp_reg": {
        "sandbox_dataset": "mbpp",
        "language": "python",
        "id_range": (11, 210),  # MBPP Regular 子集
        "expected_problems": 200,
        "typical_accepted_rate": (0.40, 0.55),
        "run_timeout": 30,
    },
    "codecontests_train": {
        "sandbox_dataset": "code_contests",
        "language": "python",
        "run_timeout": 30,  # 竞赛题需要更长时间
    },
    "codecontests_valid": {
        "sandbox_dataset": "code_contests",
        "language": "python",
        "expected_problems": 117,
        "typical_accepted_rate": (0.03, 0.12),
        "run_timeout": 30,
    },
    "codecontests_valid_big": {
        "sandbox_dataset": "code_contests",
        "language": "python",
        "expected_problems": 500,
        "typical_accepted_rate": (0.03, 0.12),
        "run_timeout": 30,
    },
    "codecontests_test": {
        "sandbox_dataset": "code_contests",
        "language": "python",
        "run_timeout": 30,
    },
}


# =============================================================================
# 服务配置
# =============================================================================

SERVICE_CONFIGS = {
    "vllm": {
        "default_port": 8000,
        "default_host": "localhost",
        "tensor_parallel_size": 1,      # 单卡 4090
        "gpu_memory_utilization": 0.85,
        "max_model_len": 6144,          # 4096 prompt + 2048 generation
        "dtype": "bfloat16",
    },
    "sandbox": {
        "default_port": 8080,
        "default_host": "localhost",
        "docker_image": "volcengine/sandbox-fusion:server-20250609",
    },
}


# =============================================================================
# 并发和批处理配置
# =============================================================================

CONCURRENCY_CONFIGS = {
    "single_4090": {
        "max_concurrent_requests": 32,  # 单卡 4090 推荐值
        "batch_size": 50,
        "description": "单卡 RTX 4090 (24GB) 配置",
    },
    "single_a100": {
        "max_concurrent_requests": 64,
        "batch_size": 100,
        "description": "单卡 A100 (40GB/80GB) 配置",
    },
    "multi_gpu": {
        "max_concurrent_requests": 128,
        "batch_size": 200,
        "description": "多卡配置",
    },
}


# =============================================================================
# 辅助函数
# =============================================================================

def get_sampling_params(mode: str = "eval") -> Dict[str, Any]:
    """
    获取采样参数

    Args:
        mode: "eval" (评测，greedy) 或 "rollout" (训练，探索)

    Returns:
        采样参数字典
    """
    if mode == "eval":
        return {
            "temperature": EVAL_CONSTANTS["temperature"],
            "top_p": EVAL_CONSTANTS["top_p"],
            "max_tokens": EVAL_CONSTANTS["max_new_tokens"],
        }
    elif mode == "rollout":
        return {
            "temperature": EVAL_CONSTANTS["rollout_temperature"],
            "top_p": EVAL_CONSTANTS["rollout_top_p"],
            "max_tokens": EVAL_CONSTANTS["max_new_tokens"],
            "n": EVAL_CONSTANTS["rollout_n"],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_sandbox_config(dataset_key: Optional[str] = None) -> Dict[str, Any]:
    """
    获取 SandboxFusion 配置

    Args:
        dataset_key: 数据集名称（可选，用于获取数据集特定的超时设置）

    Returns:
        SandboxFusion 配置字典
    """
    config = {
        "run_timeout": EVAL_CONSTANTS["run_timeout"],
        "memory_limit_mb": EVAL_CONSTANTS["memory_limit_mb"],
    }

    # 如果指定了数据集，使用数据集特定的超时设置
    if dataset_key and dataset_key in DATASET_CONFIGS:
        dataset_config = DATASET_CONFIGS[dataset_key]
        if "run_timeout" in dataset_config:
            config["run_timeout"] = dataset_config["run_timeout"]

    return config


def get_concurrency_config(gpu_type: str = "single_4090") -> Dict[str, Any]:
    """
    获取并发配置

    Args:
        gpu_type: GPU 配置类型

    Returns:
        并发配置字典
    """
    return CONCURRENCY_CONFIGS.get(gpu_type, CONCURRENCY_CONFIGS["single_4090"])


def validate_config() -> List[str]:
    """
    验证配置的合理性

    Returns:
        警告消息列表
    """
    warnings = []

    # 检查解码参数
    if EVAL_CONSTANTS["temperature"] != 0.0:
        warnings.append(
            f"Warning: temperature={EVAL_CONSTANTS['temperature']} != 0.0, "
            "EVAL@1 protocol requires greedy decoding"
        )

    # 检查超时设置
    if EVAL_CONSTANTS["run_timeout"] < 10:
        warnings.append(
            f"Warning: run_timeout={EVAL_CONSTANTS['run_timeout']}s is too short, "
            "may cause false timeout errors"
        )

    if EVAL_CONSTANTS["run_timeout"] > 120:
        warnings.append(
            f"Warning: run_timeout={EVAL_CONSTANTS['run_timeout']}s is very long, "
            "may slow down evaluation"
        )

    # 检查内存限制
    if EVAL_CONSTANTS["memory_limit_mb"] < 256:
        warnings.append(
            f"Warning: memory_limit_mb={EVAL_CONSTANTS['memory_limit_mb']}MB is too low"
        )

    return warnings


def print_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("  Evaluation Configuration Summary")
    print("=" * 60)
    print()
    print("Decoding Parameters (EVAL@1 Protocol):")
    print(f"  temperature:     {EVAL_CONSTANTS['temperature']}")
    print(f"  top_p:           {EVAL_CONSTANTS['top_p']}")
    print(f"  max_new_tokens:  {EVAL_CONSTANTS['max_new_tokens']}")
    print()
    print("SandboxFusion Configuration:")
    print(f"  run_timeout:     {EVAL_CONSTANTS['run_timeout']}s")
    print(f"  memory_limit_mb: {EVAL_CONSTANTS['memory_limit_mb']}MB")
    print()
    print("Quality Control Thresholds:")
    print(f"  truncation_warning: {EVAL_CONSTANTS['truncation_warning_threshold']:.0%}")
    print(f"  timeout_warning:    {EVAL_CONSTANTS['timeout_warning_threshold']:.0%}")
    print()

    # 验证并打印警告
    warnings = validate_config()
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")
        print()


# =============================================================================
# 配置版本（用于追踪配置变更）
# =============================================================================

CONFIG_VERSION = "1.0.0"
CONFIG_DESCRIPTION = """
Phase 0 Baseline 评测配置
- 解码：greedy (temperature=0.0)
- 超时：30秒（适应 CodeContests）
- 内存：1024MB
"""


if __name__ == "__main__":
    # 运行时打印配置摘要
    print_config_summary()
    print(f"Config Version: {CONFIG_VERSION}")
