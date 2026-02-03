#!/usr/bin/env python3
"""
Phase 0 数据治理脚本 (Data Governance)
=======================================

功能：
1. 从 Hugging Face 或 SandboxFusion 获取数据集（HumanEval, MBPP, CodeContests）
2. Prompt 规范化与 SHA256 哈希计算
3. Split 内去重（Intra-split deduplication）
4. 跨 Split 泄漏检查（Cross-split overlap check）
5. 外部基准泄漏检查（HumanEval/MBPP 对 CodeContests 的污染）
6. 生成 Manifest 文件与审计报告

使用方法：
    # 从 Hugging Face 获取完整数据（推荐）
    python src/data_governance.py --source huggingface --output_dir data/

    # 从 SandboxFusion 获取数据（需要先启动服务）
    python src/data_governance.py --source sandbox --output_dir data/

    # 从已有的 raw 文件加载
    python src/data_governance.py --skip_fetch --output_dir data/
"""

# =============================================================================
# 标准库导入
# =============================================================================
import argparse      # 解析命令行参数 (--source, --output_dir 等)
import hashlib       # 计算 SHA256 哈希值，用于数据去重
import json          # 读写 JSON/JSONL 格式文件
import re            # 正则表达式（本文件未使用，保留备用）
import os            # 操作系统路径操作
import sys           # sys.exit() 退出程序

# dataclass: 自动生成数据类的 __init__ 方法
# asdict: 将 dataclass 实例转换为普通字典，方便 JSON 序列化
from dataclasses import dataclass, asdict

# datetime.now(): 获取当前时间，用于生成版本号和报告时间戳
from datetime import datetime

# Path: 现代路径操作，支持 / 运算符拼接路径
from pathlib import Path

# 类型提示，提高代码可读性和 IDE 支持
from typing import Dict, List, Optional, Set, Tuple, Any, Union

# defaultdict: 访问不存在的键时自动创建默认值（如空列表、空集合）
from collections import defaultdict

# =============================================================================
# 第三方库导入（可选依赖）
# =============================================================================

# Hugging Face datasets 库：从 HF Hub 下载公开数据集
try:
    from datasets import load_dataset  # load_dataset("数据集名", split="train/test")
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets not installed. Run: pip install datasets")

# SandboxFusion SDK：与代码沙盒服务交互
try:
    from sandbox_fusion import (
        get_prompts,          # 获取数据集的题目列表
        GetPromptsRequest,    # get_prompts 的请求参数类
        set_endpoint,         # 设置 SandboxFusion 服务地址
    )
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    print("Warning: sandbox_fusion not installed. Run: pip install sandbox-fusion")


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class ManifestEntry:
    """Manifest 单条记录 - 记录每个题目的元数据"""
    dataset: str           # 数据集名称: humaneval, mbpp, code_contests
    split: str             # 数据分割: train, valid, test
    problem_id: str        # 问题 ID（唯一标识）
    prompt_sha256: str     # 规范化 prompt 的 SHA256 哈希（用于去重）
    prompt_length: int     # 原始 prompt 字符数
    canonical_length: int  # 规范化后的 prompt 字符数
    version: str           # 数据版本（获取时间戳）


@dataclass
class AuditResult:
    """审计结果 - 记录去重和泄漏检查的统计信息"""
    dataset: str
    split: str
    total_before_dedup: int
    total_after_dedup: int
    intra_split_duplicates: int
    cross_split_overlaps: Dict[str, int]  # {其他split: 重叠数量}
    external_leakage: Dict[str, int]      # {外部数据集: 泄漏数量}


# =============================================================================
# 数据集配置
# =============================================================================

# 定义需要获取的数据集及其参数
DATASET_CONFIGS = {
    # HumanEval: OpenAI 代码生成基准，164 道 Python 题目
    "humaneval": {
        "dataset": "humaneval_python",  # SandboxFusion 中的数据集名称
        "split": "test",
        "config": {"language": "python"},
        "role": "test_only",  # 仅用于评测，不参与训练
    },

    # MBPP Regular: Google 编程基准，ID 11-210 共 200 道题
    "mbpp_reg": {
        "dataset": "mbpp",
        "split": "test",
        "config": {"is_fewshot": False},
        "id_range": (11, 210),  # 只取这个 ID 范围的题目
        "role": "test_only",
    },

    # CodeContests 训练集：约 13000 道竞赛题，用于 RL 训练
    "codecontests_train": {
        "dataset": "code_contests",
        "split": "train",
        "config": {"language": "python", "locale": "en"},
        "role": "train",
    },

    # CodeContests 验证集：约 117 道，用于超参调优
    "codecontests_valid": {
        "dataset": "code_contests",
        "split": "valid",
        "config": {"language": "python", "locale": "en"},
        "role": "validation",
    },

    # CodeContests 测试集：约 165 道，用于最终评测
    "codecontests_test": {
        "dataset": "code_contests",
        "split": "test",
        "config": {"language": "python", "locale": "en"},
        "role": "test",
    },
}


# =============================================================================
# 规范化函数
# =============================================================================

def canonicalize_prompt(prompt: str) -> str:
    """
    规范化 prompt 文本，确保相同内容产生相同的哈希值

    处理步骤：
    1. 统一换行符 (\\r\\n -> \\n)
    2. 去除每行末尾空白
    3. 压缩连续空行为单个空行
    4. 去除首尾空白
    """
    if not prompt:
        return ""

    # 将 Windows 换行符 (\r\n) 和 Mac 旧式换行符 (\r) 统一为 Unix 换行符 (\n)
    text = prompt.replace('\r\n', '\n').replace('\r', '\n')

    # 按换行符分割成行列表，去除每行末尾的空白字符
    # str.split('\n') 返回按换行符分割的列表
    # str.rstrip() 去除字符串右侧空白
    lines = [line.rstrip() for line in text.split('\n')]

    # 压缩连续空行：连续多个空行只保留一个
    compressed_lines = []
    prev_empty = False  # 记录上一行是否为空

    for line in lines:
        is_empty = len(line.strip()) == 0  # 判断当前行是否为空

        if is_empty:
            if not prev_empty:  # 只有上一行不是空行时才添加空行
                compressed_lines.append('')
            prev_empty = True
        else:
            compressed_lines.append(line)
            prev_empty = False

    # 用换行符连接所有行，并去除首尾空白
    # '\n'.join(列表) 用换行符连接列表中的字符串
    canonical = '\n'.join(compressed_lines).strip()

    return canonical


def compute_sha256(text: str) -> str:
    """
    计算文本的 SHA256 哈希值

    用于数据去重：相同的 prompt 会产生相同的哈希值
    返回 64 字符的十六进制字符串
    """
    # hashlib.sha256() 创建 SHA256 哈希对象
    # .encode('utf-8') 将字符串编码为字节
    # .hexdigest() 返回十六进制格式的哈希值
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# =============================================================================
# SandboxFusion 数据获取（备用方案）
# =============================================================================

def fetch_dataset(
    dataset_key: str,
    config: dict,
    endpoint: str = "http://localhost:8080"
) -> List[Tuple[str, str]]:
    """
    从 SandboxFusion 服务获取单个数据集

    注意：需要先启动 SandboxFusion 服务 (docker run 或 make run)

    Returns:
        [(problem_id, prompt), ...] 题目列表
    """
    if not SANDBOX_AVAILABLE:
        raise RuntimeError("sandbox_fusion SDK not available")

    # 设置 SandboxFusion 服务地址
    set_endpoint(endpoint)

    dataset_name = config["dataset"]
    sdk_config = config.get("config", {})  # 获取额外配置，不存在则返回空字典

    print(f"  Fetching {dataset_key} from SandboxFusion...")
    print(f"    Dataset: {dataset_name}, Config: {sdk_config}")

    try:
        # get_prompts(): SandboxFusion SDK 的核心 API
        # 返回包含 prompts 列表的响应对象
        prompts = get_prompts(GetPromptsRequest(
            dataset=dataset_name,
            config=sdk_config
        ))
    except Exception as e:
        print(f"    Error fetching {dataset_key}: {e}")
        # 尝试不带下划线的数据集名（兼容不同命名约定）
        if "_" in dataset_name:
            alt_name = dataset_name.replace("_", "")
            print(f"    Trying alternative name: {alt_name}")
            try:
                prompts = get_prompts(GetPromptsRequest(
                    dataset=alt_name,
                    config=sdk_config
                ))
            except Exception as e2:
                raise RuntimeError(f"Failed to fetch {dataset_key}: {e}, {e2}")
        else:
            raise

    result = []
    for p in prompts:
        pid = str(p.id)
        prompt_text = p.prompt

        # 如果配置了 ID 范围限制，过滤不在范围内的题目
        if "id_range" in config:
            try:
                id_num = int(pid)
                min_id, max_id = config["id_range"]
                if id_num < min_id or id_num > max_id:
                    continue  # 跳过不在范围内的题目
            except ValueError:
                pass  # 非数字 ID，不做过滤

        result.append((pid, prompt_text))

    print(f"    Retrieved {len(result)} problems")
    return result


def fetch_all_datasets(
    endpoint: str = "http://localhost:8080",
    datasets: Optional[List[str]] = None
) -> Dict[str, List[Tuple[str, str]]]:
    """
    从 SandboxFusion 获取多个数据集

    Args:
        endpoint: SandboxFusion 服务地址
        datasets: 要获取的数据集列表，None 表示获取全部

    Returns:
        {dataset_key: [(problem_id, prompt), ...], ...}
    """
    result = {}

    # 如果未指定数据集列表，则获取配置中的所有数据集
    target_datasets = datasets if datasets else list(DATASET_CONFIGS.keys())

    for key in target_datasets:
        if key not in DATASET_CONFIGS:
            print(f"Warning: Unknown dataset {key}, skipping")
            continue

        config = DATASET_CONFIGS[key]
        try:
            result[key] = fetch_dataset(key, config, endpoint)
        except Exception as e:
            print(f"Error fetching {key}: {e}")
            result[key] = []

    return result


# =============================================================================
# Hugging Face 数据获取（推荐方案）
# =============================================================================

def fetch_humaneval_hf() -> List[Dict[str, Any]]:
    """
    从 Hugging Face 获取 HumanEval 数据集

    数据源：openai_humaneval (https://huggingface.co/datasets/openai_humaneval)

    返回字段：
    - problem_id: 题目 ID (如 "HumanEval/0")
    - prompt: 函数签名和文档字符串
    - test_cases: 测试用例 {type, test_code, entry_point}
    - canonical_solution: 参考解答
    """
    if not HF_AVAILABLE:
        raise RuntimeError("datasets library not available")

    print("  Fetching HumanEval from Hugging Face...")

    # load_dataset(): Hugging Face datasets 库的核心 API
    # 第一个参数是数据集名称，split 指定加载哪个分割
    ds = load_dataset("openai_humaneval", split="test")

    result = []
    for item in ds:
        # item 是一个字典，包含数据集的各个字段
        # .get(key, default) 安全获取字段值，不存在时返回默认值
        pid = item.get("task_id", str(item.get("id", len(result))))
        prompt = item.get("prompt", "")

        # HumanEval 测试用例格式：
        # - test: Python 测试代码，包含 check() 函数
        # - entry_point: 被测函数名
        test_code = item.get("test", "")
        entry_point = item.get("entry_point", "")
        canonical_solution = item.get("canonical_solution", "")

        result.append({
            "problem_id": pid,
            "prompt": prompt,
            "test_cases": {
                "type": "humaneval",
                "test_code": test_code,      # 测试代码
                "entry_point": entry_point,  # 函数入口点
            },
            "canonical_solution": canonical_solution,
        })

    print(f"    Retrieved {len(result)} problems (with test cases)")
    return result


def fetch_mbpp_hf(id_range: Tuple[int, int] = (11, 210)) -> List[Dict[str, Any]]:
    """
    从 Hugging Face 获取 MBPP 数据集

    数据源：mbpp (https://huggingface.co/datasets/mbpp)
    默认只获取 ID 11-210（MBPP Regular 子集，共 200 道题）

    返回字段：
    - problem_id: 题目 ID
    - prompt: 题目描述（text 字段）
    - test_cases: 测试用例 {type, test_list, test_setup_code}
    - canonical_solution: 参考解答（code 字段）
    """
    if not HF_AVAILABLE:
        raise RuntimeError("datasets library not available")

    print("  Fetching MBPP from Hugging Face...")
    ds = load_dataset("mbpp", split="test")

    min_id, max_id = id_range
    result = []

    for item in ds:
        pid = item.get("task_id", len(result))

        # 过滤 ID 范围
        try:
            id_num = int(pid)
            if id_num < min_id or id_num > max_id:
                continue
        except (ValueError, TypeError):
            pass

        # MBPP 的 prompt 来自 text 字段（题目描述）
        prompt = item.get("text", "")

        # MBPP 测试用例格式：
        # - test_list: assert 语句列表，如 ["assert func(1) == 2", ...]
        # - test_setup_code: 测试前的初始化代码
        # - challenge_test_list: 额外的挑战测试用例
        test_list = item.get("test_list", [])
        test_setup_code = item.get("test_setup_code", "")
        challenge_test_list = item.get("challenge_test_list", [])
        code = item.get("code", "")  # 参考解答

        result.append({
            "problem_id": str(pid),
            "prompt": prompt,
            "test_cases": {
                "type": "mbpp",
                "test_list": test_list,
                "test_setup_code": test_setup_code,
                "challenge_test_list": challenge_test_list,
            },
            "canonical_solution": code,
        })

    print(f"    Retrieved {len(result)} problems (ID {min_id}-{max_id}, with test cases)")
    return result


def fetch_codecontests_hf(split: str = "train") -> List[Dict[str, Any]]:
    """
    从 Hugging Face 获取 CodeContests 数据集

    数据源优先级：
    1. sine/FusedCodeContests (更小，下载快)
    2. deepmind/code_contests (官方完整版)

    返回字段：
    - problem_id: 题目 ID
    - prompt: 题目描述
    - test_cases: 测试用例 {type, tests: [{input, output}, ...]}
    - solutions: 参考解答列表（最多保留 3 个）
    """
    if not HF_AVAILABLE:
        raise RuntimeError("datasets library not available")

    print(f"  Fetching CodeContests ({split}) from Hugging Face...")

    ds = None
    dataset_source = None

    # 优先尝试 sine/FusedCodeContests（更小的预处理版本）
    try:
        ds = load_dataset("sine/FusedCodeContests", split=split)
        dataset_source = "sine/FusedCodeContests"
        print(f"    Using {dataset_source}")
    except Exception as e1:
        print(f"    Error loading sine/FusedCodeContests: {e1}")
        # 回退到官方数据集
        try:
            ds = load_dataset("deepmind/code_contests", split=split)
            dataset_source = "deepmind/code_contests"
            print(f"    Using {dataset_source}")
        except Exception as e2:
            raise RuntimeError(f"Failed to load CodeContests: {e1}, {e2}")

    result = []
    for idx, item in enumerate(ds):
        # 获取问题 ID，不同数据源可能使用不同字段名
        pid = item.get("id", item.get("name", str(idx)))

        # 获取问题描述，尝试多个可能的字段名
        prompt = item.get("content", "")
        if not prompt:
            prompt = item.get("description", "")
        if not prompt:
            prompt = item.get("problem_statement", "")

        if not prompt:  # 跳过没有题目描述的记录
            continue

        # 解析测试用例
        # sine/FusedCodeContests 格式: test 是 [{input: {stdin:...}, output: {stdout:...}}, ...]
        test_data = item.get("test", [])

        test_cases_list = []
        if isinstance(test_data, list):
            for tc in test_data:
                if isinstance(tc, dict):
                    input_data = tc.get("input", {})
                    output_data = tc.get("output", {})
                    if isinstance(input_data, dict) and isinstance(output_data, dict):
                        # sine/FusedCodeContests 格式：嵌套字典
                        test_cases_list.append({
                            "input": input_data.get("stdin", ""),
                            "output": output_data.get("stdout", ""),
                        })
                    else:
                        # 其他格式：直接转字符串
                        test_cases_list.append({
                            "input": str(input_data),
                            "output": str(output_data),
                        })

        # 获取解答列表
        solutions = item.get("solutions", [])
        if isinstance(solutions, dict):
            # deepmind/code_contests 格式：solutions 是字典
            solutions = solutions.get("solution", [])

        result.append({
            "problem_id": str(pid),
            "prompt": prompt,
            "test_cases": {
                "type": "codecontests",
                "tests": test_cases_list,  # stdin/stdout 测试对列表
            },
            "solutions": solutions[:3] if solutions else [],  # 只保留前 3 个解答
        })

    print(f"    Retrieved {len(result)} problems (with test cases)")
    return result


def fetch_all_datasets_hf(
    datasets: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    从 Hugging Face 获取所有数据集（推荐入口）

    Returns:
        {dataset_key: [{problem_id, prompt, test_cases, ...}, ...], ...}
    """
    if not HF_AVAILABLE:
        raise RuntimeError("datasets library not available. Run: pip install datasets")

    result = {}
    target_datasets = datasets if datasets else list(DATASET_CONFIGS.keys())

    for key in target_datasets:
        if key not in DATASET_CONFIGS:
            print(f"Warning: Unknown dataset {key}, skipping")
            continue

        try:
            # 根据数据集类型调用对应的获取函数
            if key == "humaneval":
                result[key] = fetch_humaneval_hf()
            elif key == "mbpp_reg":
                config = DATASET_CONFIGS[key]
                id_range = config.get("id_range", (11, 210))
                result[key] = fetch_mbpp_hf(id_range)
            elif key == "codecontests_train":
                result[key] = fetch_codecontests_hf("train")
            elif key == "codecontests_valid":
                result[key] = fetch_codecontests_hf("valid")
            elif key == "codecontests_test":
                result[key] = fetch_codecontests_hf("test")
            else:
                print(f"  Warning: No HuggingFace handler for {key}, skipping")
                result[key] = []
        except Exception as e:
            print(f"Error fetching {key} from HuggingFace: {e}")
            result[key] = []

    return result


def convert_to_legacy_format(
    problems: List[Dict[str, Any]]
) -> List[Tuple[str, str]]:
    """
    将新格式（字典列表）转换为旧格式（元组列表）

    用于兼容只需要 (problem_id, prompt) 的旧代码
    """
    return [(p["problem_id"], p["prompt"]) for p in problems]


# =============================================================================
# 去重与泄漏检查
# =============================================================================

def build_manifest(
    dataset_key: str,
    problems: Union[List[Tuple[str, str]], List[Dict[str, Any]]],
    version: str
) -> List[ManifestEntry]:
    """
    为数据集构建 Manifest（元数据列表）

    每个 ManifestEntry 包含：
    - 数据集和分割信息
    - 问题 ID
    - 规范化后的 prompt 哈希值（用于去重）
    - prompt 长度统计
    """
    config = DATASET_CONFIGS[dataset_key]
    dataset_name = config["dataset"]
    split = config["split"]

    entries = []
    for item in problems:
        # 支持两种输入格式
        if isinstance(item, dict):
            pid = item["problem_id"]
            prompt = item["prompt"]
        else:
            pid, prompt = item  # 元组格式

        # 规范化 prompt 并计算哈希
        canonical = canonicalize_prompt(prompt)
        sha256 = compute_sha256(canonical)

        entry = ManifestEntry(
            dataset=dataset_name,
            split=split,
            problem_id=pid,
            prompt_sha256=sha256,
            prompt_length=len(prompt),
            canonical_length=len(canonical),
            version=version,
        )
        entries.append(entry)

    return entries


def intra_split_dedup(entries: List[ManifestEntry]) -> Tuple[List[ManifestEntry], List[ManifestEntry]]:
    """
    Split 内去重：移除相同 prompt 的重复题目

    基于 prompt_sha256 判断是否重复，保留第一次出现的题目

    Returns:
        (去重后的列表, 被移除的重复项列表)
    """
    seen_hashes: Dict[str, ManifestEntry] = {}  # 已见过的哈希 -> 对应的 entry
    unique = []      # 去重后保留的
    duplicates = []  # 被移除的重复项

    for entry in entries:
        h = entry.prompt_sha256
        if h not in seen_hashes:
            seen_hashes[h] = entry
            unique.append(entry)
        else:
            duplicates.append(entry)

    return unique, duplicates


def cross_split_check(
    manifests: Dict[str, List[ManifestEntry]]
) -> Dict[str, Dict[str, Set[str]]]:
    """
    跨 Split 精确重叠检查

    检查同一数据集不同 split 之间是否有相同的 prompt：
    - train ∩ valid == ∅
    - train ∩ test == ∅
    - valid ∩ test == ∅

    Returns:
        {dataset1: {dataset2: {重叠的哈希值集合}}}
    """
    # defaultdict(lambda: defaultdict(set)) 创建嵌套的默认字典
    overlaps = defaultdict(lambda: defaultdict(set))

    # 为每个数据集构建哈希值集合
    hash_sets: Dict[str, Set[str]] = {}
    for key, entries in manifests.items():
        hash_sets[key] = {e.prompt_sha256 for e in entries}

    # 检查所有数据集配对
    keys = list(manifests.keys())
    for i, key1 in enumerate(keys):
        for key2 in keys[i+1:]:  # 只检查 (A,B)，不重复检查 (B,A)
            # 只检查同一数据集的不同 split
            cfg1 = DATASET_CONFIGS.get(key1, {})
            cfg2 = DATASET_CONFIGS.get(key2, {})

            if cfg1.get("dataset") == cfg2.get("dataset"):
                # 集合交集：找出两个集合中都存在的哈希值
                overlap = hash_sets[key1] & hash_sets[key2]
                if overlap:
                    overlaps[key1][key2] = overlap
                    overlaps[key2][key1] = overlap

    return overlaps


def external_leakage_check(
    manifests: Dict[str, List[ManifestEntry]],
    training_keys: List[str],
    external_keys: List[str]
) -> Dict[str, Dict[str, Set[str]]]:
    """
    外部基准泄漏检查

    检查训练数据中是否包含 HumanEval/MBPP 的题目
    如果包含，说明训练数据存在泄漏，评测结果不可信

    Returns:
        {训练集key: {外部基准key: {泄漏的哈希值集合}}}
    """
    leakage = defaultdict(lambda: defaultdict(set))

    # 构建外部基准的哈希集合
    external_hashes: Dict[str, Set[str]] = {}
    for key in external_keys:
        if key in manifests:
            external_hashes[key] = {e.prompt_sha256 for e in manifests[key]}

    # 检查每个训练集是否包含外部基准的 prompt
    for train_key in training_keys:
        if train_key not in manifests:
            continue
        train_hashes = {e.prompt_sha256 for e in manifests[train_key]}

        for ext_key, ext_hashes in external_hashes.items():
            overlap = train_hashes & ext_hashes  # 集合交集
            if overlap:
                leakage[train_key][ext_key] = overlap

    return leakage


def remove_overlaps(
    manifests: Dict[str, List[ManifestEntry]],
    overlaps: Dict[str, Dict[str, Set[str]]],
    leakage: Dict[str, Dict[str, Set[str]]],
    priority_order: List[str]
) -> Dict[str, List[ManifestEntry]]:
    """
    根据优先级移除重叠样本

    优先级规则（越靠前越优先保留）：
    1. 外部基准 (humaneval, mbpp_reg) - 必须保持完整
    2. 测试集 (codecontests_test)
    3. 验证集 (codecontests_valid)
    4. 训练集 (codecontests_train) - 从这里移除重叠

    Returns:
        清理后的 manifests
    """
    # 构建优先级映射：索引越小优先级越高
    priority = {key: i for i, key in enumerate(priority_order)}

    # 记录每个数据集需要移除的哈希值
    to_remove: Dict[str, Set[str]] = defaultdict(set)

    # 处理跨 split 重叠：优先级低的被移除
    for key1 in overlaps:
        for key2, hashes in overlaps[key1].items():
            p1 = priority.get(key1, 999)
            p2 = priority.get(key2, 999)
            if p1 > p2:  # key1 优先级更低，从 key1 移除
                to_remove[key1].update(hashes)
            elif p2 > p1:  # key2 优先级更低，从 key2 移除
                to_remove[key2].update(hashes)

    # 处理外部泄漏：从训练集移除（保持外部基准完整）
    for train_key in leakage:
        for ext_key, hashes in leakage[train_key].items():
            to_remove[train_key].update(hashes)

    # 执行移除
    cleaned = {}
    for key, entries in manifests.items():
        remove_set = to_remove.get(key, set())
        # 只保留哈希值不在移除集合中的 entry
        cleaned[key] = [e for e in entries if e.prompt_sha256 not in remove_set]

    return cleaned


# =============================================================================
# 输出函数
# =============================================================================

def save_manifest(entries: List[ManifestEntry], output_path: Path):
    """
    保存 Manifest 到 JSONL 文件

    JSONL 格式：每行一个 JSON 对象
    """
    # 创建父目录（如果不存在）
    # parents=True: 递归创建父目录
    # exist_ok=True: 目录已存在时不报错
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入 JSONL 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            # asdict(): 将 dataclass 实例转换为字典
            # json.dumps(): 将字典转换为 JSON 字符串
            # ensure_ascii=False: 允许输出中文等非 ASCII 字符
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + '\n')

    print(f"  Saved {len(entries)} entries to {output_path}")


def save_raw_data(
    dataset_key: str,
    problems: Union[List[Tuple[str, str]], List[Dict[str, Any]]],
    output_dir: Path
):
    """
    保存原始数据到 JSONL 文件

    包含完整的题目信息：prompt、测试用例、参考解答等
    """
    output_path = output_dir / f"{dataset_key}_raw.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in problems:
            if isinstance(item, dict):
                # 新格式：包含测试用例的字典
                pid = item["problem_id"]
                prompt = item["prompt"]
                test_cases = item.get("test_cases", {})
                canonical_solution = item.get("canonical_solution", "")
                solutions = item.get("solutions", [])

                record = {
                    "problem_id": pid,
                    "prompt": prompt,
                    "canonical_prompt": canonicalize_prompt(prompt),  # 规范化后的 prompt
                    "prompt_sha256": compute_sha256(canonicalize_prompt(prompt)),
                    "test_cases": test_cases,
                }

                if canonical_solution:
                    record["canonical_solution"] = canonical_solution
                if solutions:
                    record["solutions"] = solutions
            else:
                # 旧格式：(problem_id, prompt) 元组
                pid, prompt = item
                record = {
                    "problem_id": pid,
                    "prompt": prompt,
                    "canonical_prompt": canonicalize_prompt(prompt),
                    "prompt_sha256": compute_sha256(canonicalize_prompt(prompt)),
                }

            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"  Saved {len(problems)} raw problems to {output_path}")


def generate_audit_report(
    manifests_before: Dict[str, List[ManifestEntry]],
    manifests_after: Dict[str, List[ManifestEntry]],
    intra_duplicates: Dict[str, List[ManifestEntry]],
    cross_overlaps: Dict[str, Dict[str, Set[str]]],
    external_leakage: Dict[str, Dict[str, Set[str]]],
    output_path: Path
):
    """
    生成数据治理审计报告（Markdown 格式）

    报告内容：
    1. 样本数统计（去重前后对比）
    2. 跨 Split 重叠检查结果
    3. 外部基准泄漏检查结果
    4. 最终验证（所有交集是否为空）
    5. 数据集角色说明
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# 数据治理审计报告 (Data Governance Audit Report)")
    lines.append("")
    # datetime.now().strftime(): 格式化当前时间
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 样本数统计表格
    lines.append("## 1. 样本数统计")
    lines.append("")
    lines.append("| 数据集 | 去重前 | 去重后 | Split内重复 | 跨Split移除 |")
    lines.append("|--------|--------|--------|-------------|-------------|")

    for key in manifests_before:
        before = len(manifests_before.get(key, []))
        after = len(manifests_after.get(key, []))
        intra_dup = len(intra_duplicates.get(key, []))
        cross_removed = before - after - intra_dup if before > after else 0

        lines.append(f"| {key} | {before} | {after} | {intra_dup} | {max(0, cross_removed)} |")

    lines.append("")

    # 2. 跨 Split 重叠检查
    lines.append("## 2. 跨 Split 精确重叠检查")
    lines.append("")

    has_overlap = False
    for key1 in cross_overlaps:
        for key2, hashes in cross_overlaps[key1].items():
            if hashes and key1 < key2:  # key1 < key2 避免重复报告 (A,B) 和 (B,A)
                has_overlap = True
                lines.append(f"- `{key1}` ∩ `{key2}`: {len(hashes)} 条重叠")

    if not has_overlap:
        lines.append("**所有 Split 交集均为空 ✓**")

    lines.append("")

    # 3. 外部泄漏检查
    lines.append("## 3. 外部基准泄漏检查 (HumanEval/MBPP)")
    lines.append("")

    has_leakage = False
    for train_key in external_leakage:
        for ext_key, hashes in external_leakage[train_key].items():
            if hashes:
                has_leakage = True
                lines.append(f"- `{train_key}` 与 `{ext_key}` 重叠: {len(hashes)} 条")

    if not has_leakage:
        lines.append("**训练集与外部基准无泄漏 ✓**")

    lines.append("")

    # 4. 最终验证
    lines.append("## 4. 最终验证")
    lines.append("")

    # 构建最终哈希集合
    final_hashes: Dict[str, Set[str]] = {
        key: {e.prompt_sha256 for e in entries}
        for key, entries in manifests_after.items()
    }

    all_clear = True
    checks = [
        ("codecontests_train", "codecontests_valid"),
        ("codecontests_train", "codecontests_test"),
        ("codecontests_valid", "codecontests_test"),
        ("codecontests_train", "humaneval"),
        ("codecontests_train", "mbpp_reg"),
        ("codecontests_valid", "humaneval"),
        ("codecontests_valid", "mbpp_reg"),
    ]

    for key1, key2 in checks:
        if key1 in final_hashes and key2 in final_hashes:
            overlap = final_hashes[key1] & final_hashes[key2]
            status = "✓" if len(overlap) == 0 else f"✗ ({len(overlap)} 条)"
            lines.append(f"- `{key1}` ∩ `{key2}` = {len(overlap)} {status}")
            if overlap:
                all_clear = False

    lines.append("")
    if all_clear:
        lines.append("**所有检查通过 ✓**")
    else:
        lines.append("**存在未解决的重叠问题 ✗**")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. 数据集角色说明
    lines.append("## 5. 数据集角色说明")
    lines.append("")
    lines.append("| 数据集 | 角色 | 说明 |")
    lines.append("|--------|------|------|")
    lines.append("| humaneval | test_only | OpenAI 代码生成基准，仅用于评测 |")
    lines.append("| mbpp_reg | test_only | Google Python 编程基准 (ID 11-210)，仅用于评测 |")
    lines.append("| codecontests_train | train | CodeContests 训练集，用于 RL 训练 |")
    lines.append("| codecontests_valid | validation | CodeContests 验证集，用于超参调优 |")
    lines.append("| codecontests_test | test | CodeContests 测试集，用于最终评测 |")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"  Audit report saved to {output_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Phase 0 数据治理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 从 Hugging Face 获取完整数据（推荐）
    python src/data_governance.py --source huggingface --output_dir data/

    # 从 SandboxFusion 获取数据
    python src/data_governance.py --source sandbox --endpoint http://localhost:8080 --output_dir data/

    # 只获取 HumanEval 和 MBPP
    python src/data_governance.py --source huggingface --datasets humaneval mbpp_reg --output_dir data/

    # 从已有的 raw 文件加载（跳过网络获取）
    python src/data_governance.py --skip_fetch --output_dir data/
        """
    )

    # 添加命令行参数
    parser.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "sandbox", "hf"],
        default="huggingface",
        help="数据源: huggingface/hf (推荐) 或 sandbox (default: huggingface)"
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8080",
        help="SandboxFusion 服务地址，仅在 --source sandbox 时使用"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="输出目录 (default: data/)"
    )

    # nargs="+": 接受一个或多个值，存储为列表
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=None,
        help="要获取的数据集列表，默认获取全部"
    )

    # action="store_true": 布尔标志，有这个参数就是 True
    parser.add_argument(
        "--skip_fetch",
        action="store_true",
        help="跳过数据获取，从已有的 raw 文件加载"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 设置输出目录路径
    output_dir = Path(args.output_dir)
    manifests_dir = output_dir / "manifests"
    raw_dir = output_dir / "raw"
    reports_dir = Path("reports")

    # 创建输出目录
    manifests_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 生成版本号（用于追踪数据版本）
    version = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Phase 0 数据治理 (Data Governance)")
    print("=" * 60)
    print(f"Data source: {'Hugging Face' if args.source in ['huggingface', 'hf'] else 'SandboxFusion'}")
    print(f"Output directory: {output_dir}")
    print(f"Version: {version}")
    print()

    # =========================================================================
    # Step 1: 获取数据
    # =========================================================================
    print("[Step 1] 获取数据集...")

    use_huggingface = args.source in ["huggingface", "hf"]

    if args.skip_fetch:
        # 从已有的 raw 文件加载（跳过网络请求）
        print("  Loading from existing raw files...")
        raw_data = {}
        for key in (args.datasets or DATASET_CONFIGS.keys()):
            raw_path = raw_dir / f"{key}_raw.jsonl"
            if raw_path.exists():
                problems = []
                with open(raw_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # json.loads(): 将 JSON 字符串解析为 Python 对象
                        record = json.loads(line)
                        problems.append((record["problem_id"], record["prompt"]))
                raw_data[key] = problems
                print(f"    Loaded {len(problems)} problems from {raw_path}")
            else:
                print(f"    Warning: {raw_path} not found, skipping")

    elif use_huggingface:
        # 从 Hugging Face 获取数据（推荐）
        print(f"  Data source: Hugging Face (完整数据集)")

        if not HF_AVAILABLE:
            print("Error: datasets library not installed")
            print("Install with: pip install datasets")
            sys.exit(1)  # 退出程序，返回错误码 1

        raw_data = fetch_all_datasets_hf(datasets=args.datasets)

        # 保存原始数据
        print("\n[Step 1.1] 保存原始数据...")
        for key, problems in raw_data.items():
            if problems:
                save_raw_data(key, problems, raw_dir)
    else:
        # 从 SandboxFusion 获取数据
        print(f"  Data source: SandboxFusion ({args.endpoint})")

        if not SANDBOX_AVAILABLE:
            print("Error: sandbox_fusion SDK not installed")
            print("Install with: pip install sandbox-fusion")
            sys.exit(1)

        raw_data = fetch_all_datasets(
            endpoint=args.endpoint,
            datasets=args.datasets
        )

        # 保存原始数据
        print("\n[Step 1.1] 保存原始数据...")
        for key, problems in raw_data.items():
            if problems:
                save_raw_data(key, problems, raw_dir)

    print()

    # =========================================================================
    # Step 2: 构建 Manifest 并进行 Split 内去重
    # =========================================================================
    print("[Step 2] 构建 Manifest 并进行 Split 内去重...")

    manifests_before: Dict[str, List[ManifestEntry]] = {}   # 去重前
    manifests_deduped: Dict[str, List[ManifestEntry]] = {}  # 去重后
    intra_duplicates: Dict[str, List[ManifestEntry]] = {}   # 被移除的重复项

    for key, problems in raw_data.items():
        if not problems:
            continue

        # 构建 manifest
        entries = build_manifest(key, problems, version)
        manifests_before[key] = entries

        # Split 内去重
        unique, dups = intra_split_dedup(entries)
        manifests_deduped[key] = unique
        intra_duplicates[key] = dups

        print(f"  {key}: {len(entries)} -> {len(unique)} (removed {len(dups)} duplicates)")

    print()

    # =========================================================================
    # Step 3: 跨 Split 重叠检查
    # =========================================================================
    print("[Step 3] 跨 Split 精确重叠检查...")
    cross_overlaps = cross_split_check(manifests_deduped)

    for key1 in cross_overlaps:
        for key2, hashes in cross_overlaps[key1].items():
            if hashes and key1 < key2:
                print(f"  Warning: {key1} ∩ {key2} = {len(hashes)} overlaps")

    # any(): 如果任一元素为真返回 True
    if not any(cross_overlaps.values()):
        print("  All cross-split checks passed ✓")

    print()

    # =========================================================================
    # Step 4: 外部泄漏检查
    # =========================================================================
    print("[Step 4] 外部基准泄漏检查 (HumanEval/MBPP)...")

    # 筛选出训练集和验证集
    training_keys = [k for k in manifests_deduped if "train" in k or "valid" in k]
    external_keys = ["humaneval", "mbpp_reg"]

    external_leakage = external_leakage_check(
        manifests_deduped,
        training_keys,
        external_keys
    )

    for train_key in external_leakage:
        for ext_key, hashes in external_leakage[train_key].items():
            if hashes:
                print(f"  Warning: {train_key} contains {len(hashes)} prompts from {ext_key}")

    if not any(external_leakage.values()):
        print("  No external leakage detected ✓")

    print()

    # =========================================================================
    # Step 5: 移除重叠样本
    # =========================================================================
    print("[Step 5] 移除重叠样本...")

    # 优先级顺序：外部基准 > test > valid > train
    priority_order = ["humaneval", "mbpp_reg", "codecontests_test", "codecontests_valid", "codecontests_train"]

    manifests_final = remove_overlaps(
        manifests_deduped,
        cross_overlaps,
        external_leakage,
        priority_order
    )

    for key in manifests_final:
        before = len(manifests_deduped.get(key, []))
        after = len(manifests_final.get(key, []))
        if before != after:
            print(f"  {key}: {before} -> {after} (removed {before - after} overlapping samples)")

    print()

    # =========================================================================
    # Step 6: 保存最终 Manifest
    # =========================================================================
    print("[Step 6] 保存最终 Manifest...")

    for key, entries in manifests_final.items():
        if entries:
            output_path = manifests_dir / f"{key}_manifest.jsonl"
            save_manifest(entries, output_path)

    # 保存重复项记录（用于审计追溯）
    for key, dups in intra_duplicates.items():
        if dups:
            output_path = manifests_dir / f"{key}_duplicates_intrasplit.jsonl"
            save_manifest(dups, output_path)
            print(f"  Saved {len(dups)} intra-split duplicates to {output_path}")

    print()

    # =========================================================================
    # Step 7: 生成审计报告
    # =========================================================================
    print("[Step 7] 生成审计报告...")

    report_path = reports_dir / "data_audit_report.md"
    generate_audit_report(
        manifests_before,
        manifests_final,
        intra_duplicates,
        cross_overlaps,
        external_leakage,
        report_path
    )

    print()
    print("=" * 60)
    print("数据治理完成！")
    print("=" * 60)
    print(f"Manifest 文件: {manifests_dir}/")
    print(f"原始数据: {raw_dir}/")
    print(f"审计报告: {report_path}")
    print()

    # 最终统计
    print("最终样本数统计:")
    for key, entries in manifests_final.items():
        config = DATASET_CONFIGS.get(key, {})
        role = config.get("role", "unknown")
        print(f"  {key} ({role}): {len(entries)} 条")


# 程序入口点
# 当直接运行此文件时 __name__ == "__main__"
# 当被其他文件 import 时 __name__ == "data_governance"
if __name__ == "__main__":
    main()
