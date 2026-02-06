#!/usr/bin/env python3
"""
Phase 0 问答日志模块 (QA Logger)
=================================

功能：
1. 记录问题、模型响应、评测结果
2. 支持分层抽样策略
3. 保存为 JSONL 格式

抽样策略：
- 按数据集分别记录
- 优先记录失败案例（用于错误分析）
- 均匀覆盖不同错误类型
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class QALogEntry:
    """单条问答日志"""
    dataset: str
    problem_id: str
    prompt: str
    response: str
    accepted: bool
    pass_ratio: float
    error_type: str
    judge_time: float
    gen_time: float
    gen_tokens: int
    extra: Dict[str, Any]


# =============================================================================
# QA 日志器
# =============================================================================

class QALogger:
    """
    问答日志器：记录评测过程中的问答对

    使用分层抽样策略保存日志：
    1. 所有失败案例优先保留（用于错误分析）
    2. 成功案例随机抽样
    3. 每个错误类型至少保留一定数量的样本
    """

    def __init__(
        self,
        output_dir: Path,
        sample_size: int = 50,
        min_per_error_type: int = 5,
    ):
        """
        Args:
            output_dir: 输出目录
            sample_size: 每个数据集保存的总样本数
            min_per_error_type: 每种错误类型最少保留的样本数
        """
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size
        self.min_per_error_type = min_per_error_type

        # 按数据集和错误类型存储
        self._entries: Dict[str, Dict[str, List[QALogEntry]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # 统计计数
        self._total_counts: Dict[str, int] = defaultdict(int)

    def log(
        self,
        dataset: str,
        problem_id: str,
        prompt: str,
        response: str,
        eval_result: 'EvalResult',  # 前向引用
        gen_metadata: Dict[str, Any],
    ):
        """
        记录一条问答日志

        Args:
            dataset: 数据集名称
            problem_id: 问题 ID
            prompt: 输入提示
            response: 模型响应
            eval_result: 评测结果（从 metrics.py 导入）
            gen_metadata: 生成元数据（包含 gen_time, completion_tokens 等）
        """
        entry = QALogEntry(
            dataset=dataset,
            problem_id=problem_id,
            prompt=prompt,
            response=response,
            accepted=eval_result.accepted,
            pass_ratio=eval_result.pass_ratio,
            error_type=eval_result.error_type,
            judge_time=eval_result.judge_time,
            gen_time=gen_metadata.get("gen_time", 0.0),
            gen_tokens=gen_metadata.get("completion_tokens", 0),
            extra={
                "finish_reason": gen_metadata.get("finish_reason", "unknown"),
                "details": eval_result.details,
            }
        )

        self._entries[dataset][eval_result.error_type].append(entry)
        self._total_counts[dataset] += 1

    def _sample_entries(self, dataset: str) -> List[QALogEntry]:
        """
        对单个数据集进行分层抽样

        抽样策略：
        1. 每种失败错误类型至少保留 min_per_error_type 个（如不足则全量）
        2. 在剩余配额内优先补足失败样本，其次补足成功样本，直到达到 sample_size

        Args:
            dataset: 数据集名称

        Returns:
            抽样后的日志列表
        """
        entries_by_type = self._entries[dataset]
        all_types = list(entries_by_type.keys())

        sampled: List[QALogEntry] = []
        sampled_problem_ids = set()

        # 1) 覆盖每种失败错误类型
        failure_types = [t for t in all_types if t != "success"]

        for error_type in failure_types:
            type_entries = entries_by_type[error_type]
            if not type_entries:
                continue

            keep_count = min(len(type_entries), self.min_per_error_type)
            chosen = type_entries if len(type_entries) <= keep_count else random.sample(type_entries, keep_count)
            for entry in chosen:
                if entry.problem_id in sampled_problem_ids:
                    continue
                sampled.append(entry)
                sampled_problem_ids.add(entry.problem_id)

        # 2) 罕见成功样本：如果成功样本本身很少，优先全量保留（便于 case study）
        success_entries = entries_by_type.get("success", [])
        if 0 < len(success_entries) <= self.min_per_error_type:
            for entry in success_entries:
                if len(sampled) >= self.sample_size:
                    break
                if entry.problem_id in sampled_problem_ids:
                    continue
                sampled.append(entry)
                sampled_problem_ids.add(entry.problem_id)

        # 3) 补足到 sample_size：优先失败，其次成功
        remaining_quota = max(0, self.sample_size - len(sampled))
        if remaining_quota == 0:
            return sampled

        remaining_failures: List[QALogEntry] = []
        for error_type in failure_types:
            for entry in entries_by_type.get(error_type, []):
                if entry.problem_id not in sampled_problem_ids:
                    remaining_failures.append(entry)

        remaining_successes = [
            entry for entry in entries_by_type.get("success", [])
            if entry.problem_id not in sampled_problem_ids
        ]

        remaining_pool = remaining_failures + remaining_successes
        if not remaining_pool:
            return sampled

        keep_count = min(len(remaining_pool), remaining_quota)
        sampled.extend(remaining_pool if len(remaining_pool) <= keep_count else random.sample(remaining_pool, keep_count))

        return sampled

    def save(self):
        """
        保存所有日志到文件

        输出结构：
        output_dir/
        ├── humaneval_qa.jsonl
        ├── mbpp_reg_qa.jsonl
        ├── codecontests_valid_qa.jsonl
        └── summary.json
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "total_logged": dict(self._total_counts),
            "sampled": {},
            "error_distribution": {},
        }

        for dataset in self._entries:
            # 分层抽样
            sampled = self._sample_entries(dataset)

            # 保存到 JSONL
            output_path = self.output_dir / f"{dataset}_qa.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in sampled:
                    f.write(json.dumps(asdict(entry), ensure_ascii=False) + '\n')

            # 统计
            summary["sampled"][dataset] = len(sampled)

            # 错误分布
            error_dist = {}
            for error_type, entries in self._entries[dataset].items():
                error_dist[error_type] = len(entries)
            summary["error_distribution"][dataset] = error_dist

            print(f"  Saved {len(sampled)} QA logs to {output_path}")

        # 保存汇总
        summary_path = self.output_dir / "qa_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            {
                "total_logged": {dataset: count, ...},
                "by_error_type": {dataset: {error_type: count, ...}, ...},
            }
        """
        stats = {
            "total_logged": dict(self._total_counts),
            "by_error_type": {},
        }

        for dataset in self._entries:
            stats["by_error_type"][dataset] = {
                error_type: len(entries)
                for error_type, entries in self._entries[dataset].items()
            }

        return stats


# =============================================================================
# 辅助函数
# =============================================================================

def format_qa_for_display(entry: QALogEntry, max_prompt_len: int = 200, max_response_len: int = 500) -> str:
    """
    格式化问答日志用于显示

    Args:
        entry: 日志条目
        max_prompt_len: 最大 prompt 显示长度
        max_response_len: 最大 response 显示长度

    Returns:
        格式化的字符串
    """
    prompt = entry.prompt[:max_prompt_len] + "..." if len(entry.prompt) > max_prompt_len else entry.prompt
    response = entry.response[:max_response_len] + "..." if len(entry.response) > max_response_len else entry.response

    return f"""
{'='*60}
Dataset: {entry.dataset} | Problem: {entry.problem_id}
{'='*60}
[PROMPT]
{prompt}

[RESPONSE]
{response}

[RESULT]
Accepted: {entry.accepted} | Pass Ratio: {entry.pass_ratio:.2%}
Error Type: {entry.error_type}
Gen Time: {entry.gen_time:.2f}s | Judge Time: {entry.judge_time:.2f}s
Tokens: {entry.gen_tokens}
{'='*60}
"""


def load_qa_logs(file_path: Path) -> List[QALogEntry]:
    """
    从 JSONL 文件加载问答日志

    Args:
        file_path: JSONL 文件路径

    Returns:
        QALogEntry 列表
    """
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            entries.append(QALogEntry(**data))
    return entries


def analyze_failures(entries: List[QALogEntry]) -> Dict[str, Any]:
    """
    分析失败案例

    Args:
        entries: 日志条目列表

    Returns:
        {
            "total": int,
            "failure_count": int,
            "failure_rate": float,
            "by_error_type": {error_type: count, ...},
            "common_patterns": [...],  # 常见错误模式
        }
    """
    total = len(entries)
    failures = [e for e in entries if not e.accepted]

    error_counts = defaultdict(int)
    for e in failures:
        error_counts[e.error_type] += 1

    return {
        "total": total,
        "failure_count": len(failures),
        "failure_rate": len(failures) / total if total > 0 else 0.0,
        "by_error_type": dict(error_counts),
    }
