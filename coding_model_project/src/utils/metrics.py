#!/usr/bin/env python3
"""
Phase 0 指标收集模块 (Metrics Collection)
==========================================

功能：
1. 收集和计算评测指标
2. 支持四类指标：质量、错误分布、成本效率、系统可靠性
3. 提供统计摘要和 WandB 格式输出

指标分类：
- 质量指标：accepted@1, pass_ratio_mean/p50/p90, exec_success_rate
- 错误分布：syntax_error_rate, runtime_error_rate, timeout_rate, wrong_answer_rate
- 成本效率：avg_gen_tokens, avg_judge_time, throughput, cost_per_solved
- 系统可靠性：api_error_rate, sandbox_error_rate
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class EvalResult:
    """单个问题的评测结果"""
    problem_id: str
    accepted: bool           # 是否通过所有测试用例
    pass_ratio: float        # 通过的测试用例比例 [0, 1]
    error_type: str          # 错误类型：success, syntax_error, runtime_error, timeout, wrong_answer, api_error
    judge_time: float        # 判题耗时（秒）
    gen_tokens: int = 0      # 生成的 token 数
    gen_time: float = 0.0    # 生成耗时（秒）
    details: Dict[str, Any] = field(default_factory=dict)  # 额外信息


@dataclass
class DatasetMetrics:
    """单个数据集的聚合指标"""
    dataset: str
    total_problems: int

    # 质量指标
    accepted_at_1: float           # 主指标
    pass_ratio_mean: float
    pass_ratio_p50: float
    pass_ratio_p90: float
    exec_success_rate: float       # 可执行率（非语法错误、非超时）

    # 错误分布
    syntax_error_rate: float
    runtime_error_rate: float
    timeout_rate: float
    wrong_answer_rate: float
    api_error_rate: float
    unknown_error_rate: float

    # 成本指标
    avg_judge_time: float
    total_judge_time: float
    p50_judge_time: float          # 判题时间中位数
    p95_judge_time: float          # 判题时间第95百分位
    avg_gen_tokens: float          # 平均生成 token 数
    total_gen_tokens: int          # 总生成 token 数
    throughput: float              # 吞吐量（问题/秒）
    cost_per_solved_tokens: float  # 每解决一题的 token 成本
    cost_per_solved_judge_time: float  # 每解决一题的判题时间成本


# =============================================================================
# 指标收集器
# =============================================================================

class MetricsCollector:
    """
    指标收集器：收集评测结果并计算统计指标
    """

    def __init__(self):
        # 按数据集存储结果
        self._results: Dict[str, List[EvalResult]] = defaultdict(list)

        # 错误类型计数
        self._error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # 数据集的 wall_clock_time（用于计算 throughput）
        self._wall_clock_time: Dict[str, float] = {}

    def add_result(self, dataset: str, result: EvalResult):
        """
        添加单个评测结果

        Args:
            dataset: 数据集名称
            result: 评测结果
        """
        self._results[dataset].append(result)
        self._error_counts[dataset][result.error_type] += 1

    def set_wall_clock_time(self, dataset: str, wall_clock_time: float):
        """
        设置数据集的总耗时（用于计算 throughput）

        Args:
            dataset: 数据集名称
            wall_clock_time: 总耗时（秒）
        """
        self._wall_clock_time[dataset] = wall_clock_time

    def get_dataset_metrics(self, dataset: str) -> DatasetMetrics:
        """
        计算单个数据集的聚合指标

        Args:
            dataset: 数据集名称

        Returns:
            DatasetMetrics 对象
        """
        results = self._results.get(dataset, [])
        if not results:
            return DatasetMetrics(
                dataset=dataset,
                total_problems=0,
                accepted_at_1=0.0,
                pass_ratio_mean=0.0,
                pass_ratio_p50=0.0,
                pass_ratio_p90=0.0,
                exec_success_rate=0.0,
                syntax_error_rate=0.0,
                runtime_error_rate=0.0,
                timeout_rate=0.0,
                wrong_answer_rate=0.0,
                api_error_rate=0.0,
                unknown_error_rate=0.0,
                avg_judge_time=0.0,
                total_judge_time=0.0,
                p50_judge_time=0.0,
                p95_judge_time=0.0,
                avg_gen_tokens=0.0,
                total_gen_tokens=0,
                throughput=0.0,
                cost_per_solved_tokens=float('inf'),
                cost_per_solved_judge_time=float('inf'),
            )

        total = len(results)

        # 质量指标
        accepted_count = sum(1 for r in results if r.accepted)
        pass_ratios = np.array([r.pass_ratio for r in results])

        accepted_at_1 = accepted_count / total
        pass_ratio_mean = float(np.mean(pass_ratios))
        pass_ratio_p50 = float(np.median(pass_ratios))
        pass_ratio_p90 = float(np.percentile(pass_ratios, 90))

        # 可执行率：代码能“跑完”（不要求正确），用于衡量格式/运行稳定性
        # 约定：wrong_answer 表示输出不正确但程序完成执行；runtime_error/timeout/syntax_error 不计入可执行
        executable_types = {"success", "wrong_answer"}
        exec_count = sum(1 for r in results if r.error_type in executable_types)
        exec_success_rate = exec_count / total

        # 错误分布
        error_counts = self._error_counts[dataset]
        syntax_error_rate = error_counts.get("syntax_error", 0) / total
        runtime_error_rate = error_counts.get("runtime_error", 0) / total
        timeout_rate = error_counts.get("timeout", 0) / total
        wrong_answer_rate = error_counts.get("wrong_answer", 0) / total
        api_error_rate = error_counts.get("api_error", 0) / total
        unknown_error_rate = error_counts.get("unknown", 0) / total

        # 成本指标 - 判题时间
        judge_times = np.array([r.judge_time for r in results])
        total_judge_time = float(np.sum(judge_times))
        avg_judge_time = float(np.mean(judge_times))
        p50_judge_time = float(np.median(judge_times))
        p95_judge_time = float(np.percentile(judge_times, 95))

        # 成本指标 - 生成 token
        total_gen_tokens = sum(r.gen_tokens for r in results)
        avg_gen_tokens = total_gen_tokens / total if total > 0 else 0.0

        # 吞吐量（问题/秒）
        wall_clock_time = self._wall_clock_time.get(dataset, 0.0)
        throughput = total / wall_clock_time if wall_clock_time > 0 else 0.0

        # cost_per_solved：每解决一题的平均成本
        if accepted_count > 0:
            cost_per_solved_tokens = total_gen_tokens / accepted_count
            cost_per_solved_judge_time = total_judge_time / accepted_count
        else:
            cost_per_solved_tokens = float('inf')
            cost_per_solved_judge_time = float('inf')

        return DatasetMetrics(
            dataset=dataset,
            total_problems=total,
            accepted_at_1=accepted_at_1,
            pass_ratio_mean=pass_ratio_mean,
            pass_ratio_p50=pass_ratio_p50,
            pass_ratio_p90=pass_ratio_p90,
            exec_success_rate=exec_success_rate,
            syntax_error_rate=syntax_error_rate,
            runtime_error_rate=runtime_error_rate,
            timeout_rate=timeout_rate,
            wrong_answer_rate=wrong_answer_rate,
            api_error_rate=api_error_rate,
            unknown_error_rate=unknown_error_rate,
            avg_judge_time=avg_judge_time,
            total_judge_time=total_judge_time,
            p50_judge_time=p50_judge_time,
            p95_judge_time=p95_judge_time,
            avg_gen_tokens=avg_gen_tokens,
            total_gen_tokens=total_gen_tokens,
            throughput=throughput,
            cost_per_solved_tokens=cost_per_solved_tokens,
            cost_per_solved_judge_time=cost_per_solved_judge_time,
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        获取所有数据集的汇总

        Returns:
            {
                "datasets": {dataset: metrics_dict, ...},
                "overall": {aggregated_metrics},
            }
        """
        summary = {"datasets": {}, "overall": {}}

        all_results = []
        for dataset in self._results:
            metrics = self.get_dataset_metrics(dataset)
            summary["datasets"][dataset] = asdict(metrics)
            all_results.extend(self._results[dataset])

        # 计算全局指标
        if all_results:
            total = len(all_results)
            accepted_count = sum(1 for r in all_results if r.accepted)
            pass_ratios = np.array([r.pass_ratio for r in all_results])

            summary["overall"] = {
                "total_problems": total,
                "accepted_at_1": accepted_count / total,
                "pass_ratio_mean": float(np.mean(pass_ratios)),
                "pass_ratio_p50": float(np.median(pass_ratios)),
                "pass_ratio_p90": float(np.percentile(pass_ratios, 90)),
                "total_judge_time": sum(r.judge_time for r in all_results),
            }

        return summary

    def get_wandb_metrics(self, prefix: str = "eval") -> Dict[str, float]:
        """
        获取 WandB 格式的指标

        Args:
            prefix: 指标前缀

        Returns:
            {f"{prefix}/{dataset}/{metric_name}": value, ...}
        """
        metrics = {}

        for dataset in self._results:
            dm = self.get_dataset_metrics(dataset)
            # 质量指标
            metrics[f"{prefix}/{dataset}/accepted_at_1"] = dm.accepted_at_1
            metrics[f"{prefix}/{dataset}/pass_ratio_mean"] = dm.pass_ratio_mean
            metrics[f"{prefix}/{dataset}/pass_ratio_p50"] = dm.pass_ratio_p50
            metrics[f"{prefix}/{dataset}/pass_ratio_p90"] = dm.pass_ratio_p90
            metrics[f"{prefix}/{dataset}/exec_success_rate"] = dm.exec_success_rate
            # 错误分布
            metrics[f"{prefix}/{dataset}/syntax_error_rate"] = dm.syntax_error_rate
            metrics[f"{prefix}/{dataset}/runtime_error_rate"] = dm.runtime_error_rate
            metrics[f"{prefix}/{dataset}/timeout_rate"] = dm.timeout_rate
            metrics[f"{prefix}/{dataset}/wrong_answer_rate"] = dm.wrong_answer_rate
            metrics[f"{prefix}/{dataset}/api_error_rate"] = dm.api_error_rate
            # 成本指标
            metrics[f"{prefix}/{dataset}/avg_judge_time"] = dm.avg_judge_time
            metrics[f"{prefix}/{dataset}/p50_judge_time"] = dm.p50_judge_time
            metrics[f"{prefix}/{dataset}/p95_judge_time"] = dm.p95_judge_time
            metrics[f"{prefix}/{dataset}/avg_gen_tokens"] = dm.avg_gen_tokens
            metrics[f"{prefix}/{dataset}/throughput"] = dm.throughput
            # cost_per_solved (处理 inf 值)
            if dm.cost_per_solved_tokens != float('inf'):
                metrics[f"{prefix}/{dataset}/cost_per_solved_tokens"] = dm.cost_per_solved_tokens
            if dm.cost_per_solved_judge_time != float('inf'):
                metrics[f"{prefix}/{dataset}/cost_per_solved_judge_time"] = dm.cost_per_solved_judge_time

        return metrics

    def get_error_distribution(self, dataset: Optional[str] = None) -> Dict[str, int]:
        """
        获取错误分布

        Args:
            dataset: 可选，指定数据集。如果为 None，返回所有数据集的合计

        Returns:
            {error_type: count, ...}
        """
        if dataset:
            return dict(self._error_counts.get(dataset, {}))

        # 合并所有数据集
        total_counts: Dict[str, int] = defaultdict(int)
        for ds_counts in self._error_counts.values():
            for error_type, count in ds_counts.items():
                total_counts[error_type] += count

        return dict(total_counts)


# =============================================================================
# 辅助函数
# =============================================================================

def compute_pass_ratio_stats(pass_ratios: List[float]) -> Dict[str, float]:
    """
    计算 pass_ratio 统计量

    Args:
        pass_ratios: pass_ratio 列表

    Returns:
        {
            "mean": float,
            "p50": float,
            "p90": float,
            "std": float,
        }
    """
    if not pass_ratios:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "std": 0.0}

    arr = np.array(pass_ratios)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(np.std(arr)),
    }


def is_executable(error_type: str) -> bool:
    """
    判断代码是否可执行

    可执行的定义：代码能够运行（不是语法错误、不是编译错误）

    Args:
        error_type: 错误类型

    Returns:
        是否可执行
    """
    executable_types = {"success", "wrong_answer", "runtime_error"}
    return error_type in executable_types


def compute_cost_metrics(
    total_gen_tokens: int,
    total_judge_time: float,
    accepted_count: int,
    total_problems: int,
) -> Dict[str, float]:
    """
    计算成本效率指标

    Args:
        total_gen_tokens: 总生成 token 数
        total_judge_time: 总判题时间
        accepted_count: 通过的问题数
        total_problems: 总问题数

    Returns:
        {
            "avg_gen_tokens": float,
            "avg_judge_time": float,
            "cost_per_solved_tokens": float,  # 每解决一题的平均 token
            "cost_per_solved_judge_time": float,  # 每解决一题的平均判题时间
        }
    """
    avg_gen_tokens = total_gen_tokens / total_problems if total_problems > 0 else 0
    avg_judge_time = total_judge_time / total_problems if total_problems > 0 else 0

    cost_per_solved_tokens = total_gen_tokens / accepted_count if accepted_count > 0 else float('inf')
    cost_per_solved_judge_time = total_judge_time / accepted_count if accepted_count > 0 else float('inf')

    return {
        "avg_gen_tokens": avg_gen_tokens,
        "avg_judge_time": avg_judge_time,
        "cost_per_solved_tokens": cost_per_solved_tokens,
        "cost_per_solved_judge_time": cost_per_solved_judge_time,
    }
