#!/usr/bin/env python3
"""
Phase 4 Step 1: one-turn verifier-guided repair eval.

This script extends the standalone eval protocol used by phase0_eval.py:

1. First-pass generation + verifier
2. Structured repair feedback for eligible failures
3. Second-pass repair generation + verifier
4. Separate single-turn vs after-repair metrics/logs
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from coding_model_project.src.prompting import SYSTEM_PROMPT, format_prompt
    from coding_model_project.src.repair_feedback import (
        REPAIR_SYSTEM_PROMPT,
        RepairPromptMode,
        build_codecontests_repair_prompt,
        build_repair_feedback_from_eval_result,
    )
except ImportError:
    from prompting import SYSTEM_PROMPT, format_prompt
    from repair_feedback import (
        REPAIR_SYSTEM_PROMPT,
        RepairPromptMode,
        build_codecontests_repair_prompt,
        build_repair_feedback_from_eval_result,
    )


EVAL_CONSTANTS = {"run_timeout": 30}
VERL_AVAILABLE = False
EvalConfig = None
batch_generate = None
evaluate_single_problem_async = None
fetch_openai_models = None
load_prompts = None
start_rollout_servers = None
EvalResult = None
MetricsCollector = None
QALogger = None


def _ensure_phase0_runtime_loaded() -> None:
    global EVAL_CONSTANTS
    global VERL_AVAILABLE
    global EvalConfig
    global batch_generate
    global evaluate_single_problem_async
    global fetch_openai_models
    global load_prompts
    global start_rollout_servers
    global EvalResult
    global MetricsCollector
    global QALogger

    if EvalConfig is not None and MetricsCollector is not None and QALogger is not None:
        return

    try:
        from coding_model_project.src.phase0_eval import (
            EVAL_CONSTANTS as _EVAL_CONSTANTS,
            VERL_AVAILABLE as _VERL_AVAILABLE,
            EvalConfig as _EvalConfig,
            batch_generate as _batch_generate,
            evaluate_single_problem_async as _evaluate_single_problem_async,
            fetch_openai_models as _fetch_openai_models,
            load_prompts as _load_prompts,
            start_rollout_servers as _start_rollout_servers,
        )
    except ImportError:
        from phase0_eval import (
            EVAL_CONSTANTS as _EVAL_CONSTANTS,
            VERL_AVAILABLE as _VERL_AVAILABLE,
            EvalConfig as _EvalConfig,
            batch_generate as _batch_generate,
            evaluate_single_problem_async as _evaluate_single_problem_async,
            fetch_openai_models as _fetch_openai_models,
            load_prompts as _load_prompts,
            start_rollout_servers as _start_rollout_servers,
        )

    EVAL_CONSTANTS = _EVAL_CONSTANTS
    VERL_AVAILABLE = _VERL_AVAILABLE
    EvalConfig = _EvalConfig
    batch_generate = _batch_generate
    evaluate_single_problem_async = _evaluate_single_problem_async
    fetch_openai_models = _fetch_openai_models
    load_prompts = _load_prompts
    start_rollout_servers = _start_rollout_servers

    try:
        from coding_model_project.src.utils.metrics import (
            EvalResult as _EvalResult,
            MetricsCollector as _MetricsCollector,
        )
        from coding_model_project.src.utils.qa_logger import QALogger as _QALogger
    except ImportError:
        from utils.metrics import EvalResult as _EvalResult, MetricsCollector as _MetricsCollector
        from utils.qa_logger import QALogger as _QALogger

    EvalResult = _EvalResult
    MetricsCollector = _MetricsCollector
    QALogger = _QALogger


@dataclass
class RepairOptions:
    repair_max_attempts: int = 1
    repair_error_types: List[str] = None
    repair_min_pass_ratio: float = 0.2
    repair_max_failure_cases: int = 1
    repair_max_feedback_chars: int = 1200
    repair_max_prompt_chars: int = 12000
    prompt_mode: RepairPromptMode = "code_only"
    repair_output_dir_suffix: str = "_repair"

    def __post_init__(self) -> None:
        if self.repair_error_types is None:
            self.repair_error_types = ["wrong_answer", "runtime_error", "timeout"]


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _truncate_text(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    value = str(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _handle_inf(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _handle_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_handle_inf(v) for v in obj]
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return None
    return obj


def _pass_ratio_bucket(pass_ratio: float, *, accepted: bool = False) -> str:
    if accepted or pass_ratio >= 1.0:
        return "accepted"
    if pass_ratio <= 0.0:
        return "bucket_0"
    if pass_ratio < 0.2:
        return "bucket_0_0.2"
    if pass_ratio < 0.6:
        return "bucket_0.2_0.6"
    return "bucket_0.6_1.0"


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_handle_inf(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_first_pass_per_problem_path(first_pass_per_problem: str, dataset_key: str) -> Path:
    path = Path(first_pass_per_problem)
    if path.is_dir():
        return path / f"{dataset_key}.jsonl"
    return path


def _load_first_pass_source_metadata(
    first_pass_per_problem: str,
    dataset_key: str,
) -> Dict[str, Any]:
    jsonl_path = _resolve_first_pass_per_problem_path(first_pass_per_problem, dataset_key)
    candidates = []
    if jsonl_path.parent.name == "per_problem":
        candidates.append(jsonl_path.parent.parent / "run_info.json")
    candidates.append(jsonl_path.parent / "run_info.json")
    candidates.append(jsonl_path.with_suffix(".run_info.json"))

    run_info_path = next((path for path in candidates if path.is_file()), None)
    if run_info_path is None:
        return {
            "run_info_path": None,
            "source_max_response_chars": None,
        }

    try:
        payload = json.loads(run_info_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "run_info_path": str(run_info_path),
            "source_max_response_chars": None,
        }

    config = payload.get("config") or {}
    source_max_response_chars = config.get("max_response_chars")
    try:
        source_max_response_chars = (
            int(source_max_response_chars) if source_max_response_chars is not None else None
        )
    except (TypeError, ValueError):
        source_max_response_chars = None

    return {
        "run_info_path": str(run_info_path),
        "source_max_response_chars": source_max_response_chars,
    }


def _load_first_pass_record_map(
    first_pass_per_problem: str,
    dataset_key: str,
) -> Dict[str, Dict[str, Any]]:
    jsonl_path = _resolve_first_pass_per_problem_path(first_pass_per_problem, dataset_key)
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"Reused first-pass per-problem JSONL not found for {dataset_key}: {jsonl_path}"
        )

    record_map: Dict[str, Dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            problem_id = str(record.get("problem_id", "")).strip()
            if not problem_id:
                raise ValueError(f"Missing problem_id in {jsonl_path}:{line_no}")
            if problem_id in record_map:
                raise ValueError(f"Duplicate problem_id={problem_id} in {jsonl_path}:{line_no}")
            record_map[problem_id] = record

    if not record_map:
        raise ValueError(f"No per-problem rows found in {jsonl_path}")
    return record_map


def _ensure_eval_result_loaded() -> None:
    global EvalResult
    if EvalResult is not None:
        return
    try:
        from coding_model_project.src.utils.metrics import EvalResult as _EvalResult
    except ImportError:
        from utils.metrics import EvalResult as _EvalResult
    EvalResult = _EvalResult


def _eval_result_from_first_pass_record(record: Dict[str, Any]) -> EvalResult:
    _ensure_eval_result_loaded()
    details = dict(record.get("details") or {})

    top_level_fallbacks = {
        "pass_ratio_all": record.get("pass_ratio_all"),
        "passed_tests": record.get("passed_tests"),
        "total_tests": record.get("total_tests"),
        "invalid_for_rl": record.get("invalid_for_rl", False),
        "invalid_reason": record.get("invalid_reason", ""),
        "extraction_status": record.get("extraction_status", "ok"),
    }
    for key, value in top_level_fallbacks.items():
        if key not in details and value is not None:
            details[key] = value

    if "per_case_results" in record and "per_case_results" not in details:
        details["per_case_results"] = record["per_case_results"]

    return EvalResult(
        problem_id=str(record.get("problem_id", "")),
        accepted=bool(record.get("accepted", False)),
        pass_ratio=float(record.get("pass_ratio", 0.0) or 0.0),
        error_type=str(record.get("error_type", "unknown") or "unknown"),
        judge_time=float(record.get("judge_time", 0.0) or 0.0),
        gen_tokens=int(record.get("gen_tokens", 0) or 0),
        gen_time=float(record.get("gen_time", 0.0) or 0.0),
        details=details,
    )


def _first_pass_gen_meta_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "completion_tokens": int(record.get("gen_tokens", 0) or 0),
        "gen_time": float(record.get("gen_time", 0.0) or 0.0),
        "finish_reason": str(record.get("finish_reason", "unknown") or "unknown"),
    }


def _validate_reused_first_pass_record(
    *,
    dataset_key: str,
    batch_item: Dict[str, Any],
    record: Dict[str, Any],
) -> None:
    record_dataset = record.get("dataset")
    if record_dataset and record_dataset != dataset_key:
        raise ValueError(
            f"Dataset mismatch for problem_id={batch_item['problem_id']}: "
            f"expected {dataset_key}, got {record_dataset}"
        )

    record_problem_id = str(record.get("problem_id", "")).strip()
    if record_problem_id != batch_item["problem_id"]:
        raise ValueError(
            f"problem_id mismatch for reused first-pass row: "
            f"expected {batch_item['problem_id']}, got {record_problem_id}"
        )

    prompt_sha256 = record.get("prompt_sha256")
    if prompt_sha256 and prompt_sha256 != _sha256_text(batch_item.get("prompt", "")):
        raise ValueError(
            f"Prompt hash mismatch for reused first-pass row problem_id={batch_item['problem_id']}. "
            "The manifest prompt no longer matches the raw eval artifact."
        )


def _build_prompt_texts_for_batch(batch: List[Dict[str, Any]], dataset_key: str) -> List[str]:
    if dataset_key == "mbpp_reg":
        return [
            format_prompt(
                p["prompt"],
                dataset_key,
                p.get("test_cases", {}).get("entry_point", ""),
                p.get("test_cases", {}).get("example_call", ""),
            )
            for p in batch
        ]
    return [format_prompt(p["prompt"], dataset_key) for p in batch]


def _filter_to_codecontests_datasets(datasets: List[str]) -> None:
    unsupported = [ds for ds in datasets if not ds.startswith("codecontests")]
    if unsupported:
        raise ValueError(
            f"Phase 4 Step 1 currently supports CodeContests only, got unsupported datasets: {unsupported}"
        )


def _repair_reason(eval_result: EvalResult, repair_options: RepairOptions) -> str:
    if eval_result.accepted:
        return "already_accepted"
    if eval_result.error_type not in set(repair_options.repair_error_types):
        return f"error_type_not_eligible:{eval_result.error_type}"
    if eval_result.pass_ratio < repair_options.repair_min_pass_ratio:
        return (
            f"pass_ratio_below_threshold:{eval_result.pass_ratio:.4f}"
            f"<{repair_options.repair_min_pass_ratio:.4f}"
        )
    return "eligible"


def should_trigger_repair(eval_result: EvalResult, repair_options: RepairOptions) -> bool:
    return _repair_reason(eval_result, repair_options) == "eligible"


def _first_pass_record(
    *,
    dataset_key: str,
    problem_id: str,
    prompt: str,
    generated_code: str,
    eval_result: EvalResult,
    gen_tokens: int,
    gen_time: float,
    finish_reason: str,
    max_prompt_chars: int,
    max_response_chars: int,
) -> Dict[str, Any]:
    first_pass_bucket = _pass_ratio_bucket(eval_result.pass_ratio, accepted=eval_result.accepted)
    record = {
        "dataset": dataset_key,
        "problem_id": problem_id,
        "prompt_sha256": _sha256_text(prompt or ""),
        "prompt": _truncate_text(prompt or "", max_prompt_chars),
        "response": _truncate_text(generated_code or "", max_response_chars),
        "accepted": eval_result.accepted,
        "pass_ratio": eval_result.pass_ratio,
        "pass_ratio_bucket": first_pass_bucket,
        "pass_ratio_all": eval_result.details.get("pass_ratio_all", eval_result.pass_ratio),
        "passed_tests": eval_result.details.get("passed_tests"),
        "total_tests": eval_result.details.get("total_tests"),
        "error_type": eval_result.error_type,
        "invalid_for_rl": eval_result.details.get("invalid_for_rl", False),
        "invalid_reason": eval_result.details.get("invalid_reason", ""),
        "extraction_status": eval_result.details.get("extraction_status", "ok"),
        "judge_time": eval_result.judge_time,
        "gen_tokens": gen_tokens,
        "gen_time": gen_time,
        "finish_reason": finish_reason,
        "details": eval_result.details,
    }
    if "per_case_results" in eval_result.details:
        record["per_case_results"] = eval_result.details["per_case_results"]
    return record


def _final_eval_result(
    first_eval: EvalResult,
    first_gen_meta: Dict[str, Any],
    repair_eval: Optional[EvalResult],
    repair_gen_meta: Optional[Dict[str, Any]],
) -> EvalResult:
    if repair_eval is None:
        first_eval.gen_tokens = int(first_gen_meta.get("completion_tokens", 0))
        first_eval.gen_time = float(first_gen_meta.get("gen_time", 0.0))
        return first_eval

    combined_details = dict(repair_eval.details or {})
    combined_details["first_pass"] = first_eval.details
    combined_details["repair_pass"] = repair_eval.details
    return EvalResult(
        problem_id=first_eval.problem_id,
        accepted=repair_eval.accepted,
        pass_ratio=repair_eval.pass_ratio,
        error_type=repair_eval.error_type,
        judge_time=float(first_eval.judge_time + repair_eval.judge_time),
        gen_tokens=int(
            first_gen_meta.get("completion_tokens", 0)
            + repair_gen_meta.get("completion_tokens", 0)
        ),
        gen_time=float(first_gen_meta.get("gen_time", 0.0) + repair_gen_meta.get("gen_time", 0.0)),
        details=combined_details,
    )


def _repair_per_problem_record(
    *,
    dataset_key: str,
    problem_id: str,
    prompt: str,
    first_prompt_text: str,
    first_response: str,
    first_eval: EvalResult,
    first_gen_meta: Dict[str, Any],
    repair_triggered: bool,
    repair_reason: str,
    repair_feedback: Optional[Dict[str, Any]],
    repair_prompt: Optional[str],
    repair_response: Optional[str],
    repair_eval: Optional[EvalResult],
    repair_gen_meta: Optional[Dict[str, Any]],
    repair_prompt_mode: RepairPromptMode,
    max_prompt_chars: int,
    max_response_chars: int,
) -> Dict[str, Any]:
    final_eval = repair_eval if repair_eval is not None else first_eval
    final_accepted_after_repair = final_eval.accepted
    rescued = int((not first_eval.accepted) and final_accepted_after_repair)
    final_response = repair_response if repair_response is not None else first_response
    final_gen_meta = repair_gen_meta if repair_gen_meta is not None else first_gen_meta
    first_pass_bucket = _pass_ratio_bucket(first_eval.pass_ratio, accepted=first_eval.accepted)
    final_pass_bucket = _pass_ratio_bucket(final_eval.pass_ratio, accepted=final_eval.accepted)
    total_gen_tokens = int(
        first_gen_meta.get("completion_tokens", 0)
        + (repair_gen_meta.get("completion_tokens", 0) if repair_gen_meta else 0)
    )
    total_gen_time = float(
        first_gen_meta.get("gen_time", 0.0)
        + (repair_gen_meta.get("gen_time", 0.0) if repair_gen_meta else 0.0)
    )
    total_judge_time = float(first_eval.judge_time + (repair_eval.judge_time if repair_eval else 0.0))

    first_pass = {
        "accepted": first_eval.accepted,
        "pass_ratio": first_eval.pass_ratio,
        "error_type": first_eval.error_type,
        "response": _truncate_text(first_response or "", max_response_chars),
        "details": first_eval.details,
        "gen_tokens": int(first_gen_meta.get("completion_tokens", 0)),
        "gen_time": float(first_gen_meta.get("gen_time", 0.0)),
        "finish_reason": first_gen_meta.get("finish_reason", "unknown"),
    }

    repair_pass = None
    if repair_eval is not None and repair_gen_meta is not None:
        repair_pass = {
            "accepted": repair_eval.accepted,
            "pass_ratio": repair_eval.pass_ratio,
            "error_type": repair_eval.error_type,
            "response": _truncate_text(repair_response or "", max_response_chars),
            "details": repair_eval.details,
            "gen_tokens": int(repair_gen_meta.get("completion_tokens", 0)),
            "gen_time": float(repair_gen_meta.get("gen_time", 0.0)),
            "finish_reason": repair_gen_meta.get("finish_reason", "unknown"),
        }

    return {
        "dataset": dataset_key,
        "problem_id": problem_id,
        "prompt_sha256": _sha256_text(prompt or ""),
        "prompt": _truncate_text(prompt or "", max_prompt_chars),
        "response": _truncate_text(final_response or "", max_response_chars),
        "accepted": final_eval.accepted,
        "pass_ratio": final_eval.pass_ratio,
        "pass_ratio_bucket": final_pass_bucket,
        "first_pass_pass_ratio": first_eval.pass_ratio,
        "first_pass_pass_ratio_bucket": first_pass_bucket,
        "pass_ratio_all": final_eval.details.get("pass_ratio_all", final_eval.pass_ratio),
        "passed_tests": final_eval.details.get("passed_tests"),
        "total_tests": final_eval.details.get("total_tests"),
        "error_type": final_eval.error_type,
        "invalid_for_rl": final_eval.details.get("invalid_for_rl", False),
        "invalid_reason": final_eval.details.get("invalid_reason", ""),
        "extraction_status": final_eval.details.get("extraction_status", "ok"),
        "judge_time": total_judge_time,
        "gen_tokens": total_gen_tokens,
        "gen_time": total_gen_time,
        "finish_reason": final_gen_meta.get("finish_reason", "unknown"),
        "details": final_eval.details,
        "repair_triggered": repair_triggered,
        "repair_reason": repair_reason,
        "repair_prompt_sha256": _sha256_text(repair_prompt or "") if repair_prompt else "",
        "repair_prompt_mode": repair_prompt_mode,
        "repair_prompt": _truncate_text(repair_prompt or "", max_prompt_chars) if repair_prompt else "",
        "repair_response": _truncate_text(repair_response or "", max_response_chars) if repair_response else "",
        "repair_feedback_selection_strategy": (
            repair_feedback.get("selection_strategy", "") if repair_feedback else ""
        ),
        "repair_feedback_extra_hint": (
            repair_feedback.get("extra_hint", "") if repair_feedback else ""
        ),
        "final_accepted_after_repair": final_accepted_after_repair,
        "repair_gain_for_problem": rescued,
        "pass_ratio_gain_for_problem": final_eval.pass_ratio - first_eval.pass_ratio,
        "first_pass": first_pass,
        "repair_pass": repair_pass,
        "protocol_cost": {
            "total_gen_tokens": total_gen_tokens,
            "total_gen_time": total_gen_time,
            "total_judge_time": total_judge_time,
        },
    }


def _summarize_repair_dataset(
    *,
    dataset_key: str,
    repair_prompt_mode: RepairPromptMode,
    first_pass_metrics: Dict[str, Any],
    final_metrics: Dict[str, Any],
    repair_attempt_count: int,
    repair_success_count: int,
    extra_gen_tokens_repair: int,
    extra_judge_time_repair: float,
    repair_trigger_by_error: Dict[str, int],
    failed_problem_count_by_bucket: Dict[str, int],
    repair_attempt_count_by_bucket: Dict[str, int],
    repair_success_count_by_bucket: Dict[str, int],
) -> Dict[str, Any]:
    conditional_repair_success = (
        repair_success_count / repair_attempt_count if repair_attempt_count else 0.0
    )
    extra_gen_tokens_per_success = (
        extra_gen_tokens_repair / repair_success_count if repair_success_count else float("inf")
    )
    extra_judge_time_per_success = (
        extra_judge_time_repair / repair_success_count if repair_success_count else float("inf")
    )
    return {
        "dataset": dataset_key,
        "repair_prompt_mode": repair_prompt_mode,
        "first_pass_accepted_at_1": first_pass_metrics.get("accepted_at_1", 0.0),
        "accepted_after_1_repair": final_metrics.get("accepted_at_1", 0.0),
        "repair_gain": final_metrics.get("accepted_at_1", 0.0) - first_pass_metrics.get("accepted_at_1", 0.0),
        "first_pass_pass_ratio_mean": first_pass_metrics.get("pass_ratio_mean", 0.0),
        "after_repair_pass_ratio_mean": final_metrics.get("pass_ratio_mean", 0.0),
        "repair_attempt_count": repair_attempt_count,
        "repair_success_count": repair_success_count,
        "conditional_repair_success": conditional_repair_success,
        "extra_gen_tokens_repair": extra_gen_tokens_repair,
        "extra_judge_time_repair": extra_judge_time_repair,
        "avg_extra_gen_tokens_repair": (
            extra_gen_tokens_repair / repair_attempt_count if repair_attempt_count else 0.0
        ),
        "avg_extra_judge_time_repair": (
            extra_judge_time_repair / repair_attempt_count if repair_attempt_count else 0.0
        ),
        "extra_gen_tokens_per_repair_success": extra_gen_tokens_per_success,
        "extra_judge_time_per_repair_success": extra_judge_time_per_success,
        "repair_trigger_by_error": repair_trigger_by_error,
        "failed_problem_count_by_bucket": failed_problem_count_by_bucket,
        "repair_attempt_count_by_bucket": repair_attempt_count_by_bucket,
        "repair_success_count_by_bucket": repair_success_count_by_bucket,
        "trigger_rate_on_failed_by_bucket": {
            bucket: (
                repair_attempt_count_by_bucket.get(bucket, 0)
                / failed_problem_count_by_bucket.get(bucket, 0)
                if failed_problem_count_by_bucket.get(bucket, 0)
                else 0.0
            )
            for bucket in sorted(failed_problem_count_by_bucket)
        },
        "bucketed_conditional_repair_success": {
            bucket: (
                repair_success_count_by_bucket.get(bucket, 0)
                / repair_attempt_count_by_bucket.get(bucket, 0)
                if repair_attempt_count_by_bucket.get(bucket, 0)
                else 0.0
            )
            for bucket in sorted(
                set(failed_problem_count_by_bucket)
                | set(repair_attempt_count_by_bucket)
                | set(repair_success_count_by_bucket)
            )
        },
    }


async def evaluate_dataset_with_repair(
    dataset_key: str,
    prompts: List[Dict[str, Any]],
    server_addresses: List[str],
    config: EvalConfig,
    repair_options: RepairOptions,
    first_pass_per_problem: Optional[str],
    first_metrics_collector: MetricsCollector,
    final_metrics_collector: MetricsCollector,
    first_qa_logger: QALogger,
    repair_qa_logger: QALogger,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    print(f"\n{'='*60}")
    print(f"Repair-evaluating: {dataset_key} ({len(prompts)} problems)")
    print(f"{'='*60}")

    if config.max_problems is not None and len(prompts) > config.max_problems:
        import random

        rng = random.Random(config.shuffle_seed)
        prompts = prompts[:]
        rng.shuffle(prompts)
        prompts = prompts[: config.max_problems]
        print(
            f"  NOTE: max_problems enabled -> evaluating {len(prompts)} problems "
            f"(seed={config.shuffle_seed})"
        )

    sampling_params = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_new_tokens,
    }
    judge_semaphore = asyncio.Semaphore(max(1, config.max_concurrent_judges))
    reused_first_pass_records = (
        _load_first_pass_record_map(first_pass_per_problem, dataset_key)
        if first_pass_per_problem
        else None
    )
    reused_first_pass_source_metadata = (
        _load_first_pass_source_metadata(first_pass_per_problem, dataset_key)
        if first_pass_per_problem
        else {"run_info_path": None, "source_max_response_chars": None}
    )
    source_max_response_chars = reused_first_pass_source_metadata.get("source_max_response_chars")

    dataset_start_time = time.time()
    first_pass_wall_clock = 0.0

    repair_attempt_count = 0
    repair_success_count = 0
    extra_gen_tokens_repair = 0
    extra_judge_time_repair = 0.0
    skipped_truncated_reuse_count = 0
    repair_trigger_by_error: Dict[str, int] = {}
    failed_problem_count_by_bucket: Dict[str, int] = {}
    repair_attempt_count_by_bucket: Dict[str, int] = {}
    repair_success_count_by_bucket: Dict[str, int] = {}

    first_pass_file_ctx = contextlib.nullcontext(None)
    final_pass_file_ctx = contextlib.nullcontext(None)
    if config.save_full_results:
        first_pass_dir = Path(config.output_dir) / "per_problem_first_pass"
        final_pass_dir = Path(config.output_dir) / "per_problem"
        first_pass_dir.mkdir(parents=True, exist_ok=True)
        final_pass_dir.mkdir(parents=True, exist_ok=True)
        first_pass_path = first_pass_dir / f"{dataset_key}.jsonl"
        final_pass_path = final_pass_dir / f"{dataset_key}.jsonl"
        first_pass_file_ctx = first_pass_path.open("w", encoding="utf-8")
        final_pass_file_ctx = final_pass_path.open("w", encoding="utf-8")

    with first_pass_file_ctx as first_f, final_pass_file_ctx as final_f:
        for batch_start in range(0, len(prompts), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(prompts))
            batch = prompts[batch_start:batch_end]

            print(
                f"  Processing batch {batch_start // config.batch_size + 1}/"
                f"{(len(prompts) - 1) // config.batch_size + 1}..."
            )

            prompt_texts = _build_prompt_texts_for_batch(batch, dataset_key)
            batch_records: List[Dict[str, Any]] = []
            if reused_first_pass_records is not None:
                for batch_item, prompt_text in zip(batch, prompt_texts):
                    problem_id = batch_item["problem_id"]
                    record = reused_first_pass_records.get(problem_id)
                    if record is None:
                        raise ValueError(
                            f"Missing reused first-pass row for problem_id={problem_id} "
                            f"in dataset={dataset_key}"
                        )
                    _validate_reused_first_pass_record(
                        dataset_key=dataset_key,
                        batch_item=batch_item,
                        record=record,
                    )
                    first_eval = _eval_result_from_first_pass_record(record)
                    gen_meta = _first_pass_gen_meta_from_record(record)
                    batch_records.append(
                        {
                            "problem_id": problem_id,
                            "prompt": batch_item["prompt"],
                            "prompt_text": prompt_text,
                            "generated_code": str(record.get("response", "") or ""),
                            "gen_meta": gen_meta,
                            "first_eval_result": first_eval,
                            "reused_response_maybe_truncated": bool(
                                source_max_response_chars
                                and len(str(record.get("response", "") or "")) >= source_max_response_chars
                            ),
                            "source_response_char_cap": source_max_response_chars,
                        }
                    )
                    first_pass_wall_clock += float(first_eval.judge_time + first_eval.gen_time)
            else:
                first_batch_start = time.time()
                first_gen_results = await batch_generate(
                    server_addresses,
                    config.model_path,
                    prompt_texts,
                    sampling_params,
                    config.max_concurrent_requests,
                    system_prompt=SYSTEM_PROMPT,
                )

                first_judge_tasks = []
                for i, (generated_code, gen_meta) in enumerate(first_gen_results):
                    first_judge_tasks.append(
                        evaluate_single_problem_async(
                            generated_code=generated_code,
                            batch_item=batch[i],
                            config=config,
                            judge_semaphore=judge_semaphore,
                        )
                    )
                    batch_records.append(
                        {
                            "problem_id": batch[i]["problem_id"],
                            "prompt": batch[i]["prompt"],
                            "prompt_text": prompt_texts[i],
                            "generated_code": generated_code,
                            "gen_meta": gen_meta,
                        }
                    )

                first_eval_results = await asyncio.gather(*first_judge_tasks)
                first_pass_wall_clock += time.time() - first_batch_start
                for record, first_eval in zip(batch_records, first_eval_results):
                    record["first_eval_result"] = first_eval

            repair_candidates: List[Dict[str, Any]] = []
            for record in batch_records:
                first_eval = record["first_eval_result"]
                first_eval.gen_tokens = int(record["gen_meta"].get("completion_tokens", 0))
                first_eval.gen_time = float(record["gen_meta"].get("gen_time", 0.0))
                first_bucket = _pass_ratio_bucket(first_eval.pass_ratio, accepted=first_eval.accepted)

                first_metrics_collector.add_result(dataset_key, first_eval)
                if not first_eval.accepted:
                    failed_problem_count_by_bucket[first_bucket] = (
                        failed_problem_count_by_bucket.get(first_bucket, 0) + 1
                    )
                first_qa_logger.log(
                    dataset=dataset_key,
                    problem_id=record["problem_id"],
                    prompt=record["prompt_text"],
                    response=record["generated_code"],
                    eval_result=first_eval,
                    gen_metadata=record["gen_meta"],
                )

                first_record = _first_pass_record(
                    dataset_key=dataset_key,
                    problem_id=record["problem_id"],
                    prompt=record["prompt"],
                    generated_code=record["generated_code"],
                    eval_result=first_eval,
                    gen_tokens=first_eval.gen_tokens,
                    gen_time=first_eval.gen_time,
                    finish_reason=record["gen_meta"].get("finish_reason", "unknown"),
                    max_prompt_chars=config.max_prompt_chars,
                    max_response_chars=config.max_response_chars,
                )
                if first_f is not None:
                    if reused_first_pass_records is not None:
                        first_record["first_pass_source"] = "reused_raw_per_problem"
                        first_record["source_response_char_cap"] = source_max_response_chars
                        first_record["response_maybe_truncated"] = record.get(
                            "reused_response_maybe_truncated", False
                        )
                    first_f.write(json.dumps(first_record, ensure_ascii=False) + "\n")

                if record.get("reused_response_maybe_truncated"):
                    skipped_truncated_reuse_count += 1
                    reason = "source_response_maybe_truncated"
                else:
                    reason = _repair_reason(first_eval, repair_options)
                record["first_eval_result"] = first_eval
                record["first_pass_bucket"] = first_bucket
                record["repair_reason"] = reason
                if reason == "eligible":
                    repair_candidates.append(record)

            repair_outputs_by_problem_id: Dict[str, Dict[str, Any]] = {}
            if repair_candidates:
                repair_prompts = []
                for record in repair_candidates:
                    feedback = build_repair_feedback_from_eval_result(
                        record["first_eval_result"],
                        max_failure_cases=repair_options.repair_max_failure_cases,
                        max_case_chars=repair_options.repair_max_feedback_chars,
                    )
                    repair_prompt = build_codecontests_repair_prompt(
                        problem_prompt=record["prompt"],
                        first_response=record["generated_code"],
                        feedback=feedback,
                        max_prompt_chars=repair_options.repair_max_prompt_chars,
                        prompt_mode=repair_options.prompt_mode,
                    )
                    record["repair_feedback"] = feedback
                    record["repair_prompt"] = repair_prompt
                    repair_prompts.append(repair_prompt)

                repair_gen_results = await batch_generate(
                    server_addresses,
                    config.model_path,
                    repair_prompts,
                    sampling_params,
                    config.max_concurrent_requests,
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                )

                repair_judge_tasks = []
                for record, (repair_response, repair_gen_meta) in zip(repair_candidates, repair_gen_results):
                    record["repair_response"] = repair_response
                    record["repair_gen_meta"] = repair_gen_meta
                    repair_judge_tasks.append(
                        evaluate_single_problem_async(
                            generated_code=repair_response,
                            batch_item=next(item for item in batch if item["problem_id"] == record["problem_id"]),
                            config=config,
                            judge_semaphore=judge_semaphore,
                        )
                    )

                repair_eval_results = await asyncio.gather(*repair_judge_tasks)

                for record, repair_eval in zip(repair_candidates, repair_eval_results):
                    repair_outputs_by_problem_id[record["problem_id"]] = {
                        "repair_eval_result": repair_eval,
                        "repair_response": record["repair_response"],
                        "repair_prompt": record["repair_prompt"],
                        "repair_gen_meta": record["repair_gen_meta"],
                    }

                    repair_eval.gen_tokens = int(record["repair_gen_meta"].get("completion_tokens", 0))
                    repair_eval.gen_time = float(record["repair_gen_meta"].get("gen_time", 0.0))
                    repair_qa_logger.log(
                        dataset=dataset_key,
                        problem_id=record["problem_id"],
                        prompt=record["repair_prompt"],
                        response=record["repair_response"],
                        eval_result=repair_eval,
                        gen_metadata=record["repair_gen_meta"],
                    )

            for record in batch_records:
                problem_id = record["problem_id"]
                first_eval = record["first_eval_result"]
                repair_payload = repair_outputs_by_problem_id.get(problem_id)
                repair_eval = repair_payload["repair_eval_result"] if repair_payload else None
                repair_gen_meta = repair_payload["repair_gen_meta"] if repair_payload else None
                repair_prompt = repair_payload["repair_prompt"] if repair_payload else None
                repair_response = repair_payload["repair_response"] if repair_payload else None

                final_eval = _final_eval_result(
                    first_eval=first_eval,
                    first_gen_meta=record["gen_meta"],
                    repair_eval=repair_eval,
                    repair_gen_meta=repair_gen_meta,
                )
                final_metrics_collector.add_result(dataset_key, final_eval)

                if repair_eval is not None:
                    repair_attempt_count += 1
                    extra_gen_tokens_repair += int(repair_gen_meta.get("completion_tokens", 0))
                    extra_judge_time_repair += float(repair_eval.judge_time)
                    repair_trigger_by_error[first_eval.error_type] = repair_trigger_by_error.get(first_eval.error_type, 0) + 1
                    repair_attempt_count_by_bucket[record["first_pass_bucket"]] = (
                        repair_attempt_count_by_bucket.get(record["first_pass_bucket"], 0) + 1
                    )
                    if (not first_eval.accepted) and repair_eval.accepted:
                        repair_success_count += 1
                        repair_success_count_by_bucket[record["first_pass_bucket"]] = (
                            repair_success_count_by_bucket.get(record["first_pass_bucket"], 0) + 1
                        )

                final_record = _repair_per_problem_record(
                    dataset_key=dataset_key,
                    problem_id=problem_id,
                    prompt=record["prompt"],
                    first_prompt_text=record["prompt_text"],
                    first_response=record["generated_code"],
                    first_eval=first_eval,
                    first_gen_meta=record["gen_meta"],
                    repair_triggered=repair_eval is not None,
                    repair_reason=record["repair_reason"],
                    repair_feedback=record.get("repair_feedback"),
                    repair_prompt=repair_prompt,
                    repair_response=repair_response,
                    repair_eval=repair_eval,
                    repair_gen_meta=repair_gen_meta,
                    repair_prompt_mode=repair_options.prompt_mode,
                    max_prompt_chars=config.max_prompt_chars,
                    max_response_chars=config.max_response_chars,
                )
                if final_f is not None:
                    if reused_first_pass_records is not None:
                        final_record["first_pass_source"] = "reused_raw_per_problem"
                        final_record["source_response_char_cap"] = source_max_response_chars
                        final_record["first_pass_response_maybe_truncated"] = record.get(
                            "reused_response_maybe_truncated", False
                        )
                    final_f.write(json.dumps(final_record, ensure_ascii=False) + "\n")

    total_wall_clock = time.time() - dataset_start_time
    first_metrics_collector.set_wall_clock_time(dataset_key, first_pass_wall_clock)
    final_metrics_collector.set_wall_clock_time(dataset_key, total_wall_clock)

    first_pass_metrics = asdict(first_metrics_collector.get_dataset_metrics(dataset_key))
    final_metrics = asdict(final_metrics_collector.get_dataset_metrics(dataset_key))
    repair_metrics = _summarize_repair_dataset(
        dataset_key=dataset_key,
        repair_prompt_mode=repair_options.prompt_mode,
        first_pass_metrics=first_pass_metrics,
        final_metrics=final_metrics,
        repair_attempt_count=repair_attempt_count,
        repair_success_count=repair_success_count,
        extra_gen_tokens_repair=extra_gen_tokens_repair,
        extra_judge_time_repair=extra_judge_time_repair,
        repair_trigger_by_error=repair_trigger_by_error,
        failed_problem_count_by_bucket=failed_problem_count_by_bucket,
        repair_attempt_count_by_bucket=repair_attempt_count_by_bucket,
        repair_success_count_by_bucket=repair_success_count_by_bucket,
    )
    if first_pass_per_problem:
        repair_metrics["reused_first_pass_source"] = {
            "path": first_pass_per_problem,
            "run_info_path": reused_first_pass_source_metadata.get("run_info_path"),
            "source_max_response_chars": source_max_response_chars,
            "skipped_due_to_truncated_source_response": skipped_truncated_reuse_count,
        }

    print(f"\n  Results for {dataset_key}:")
    print(f"    first_pass accepted@1: {first_pass_metrics['accepted_at_1']:.2%}")
    print(f"    accepted_after_1_repair: {repair_metrics['accepted_after_1_repair']:.2%}")
    print(f"    repair_gain: {repair_metrics['repair_gain']:.2%}")
    print(f"    conditional_repair_success: {repair_metrics['conditional_repair_success']:.2%}")
    print(f"    repair_attempt_count: {repair_metrics['repair_attempt_count']}")
    if first_pass_per_problem:
        print(
            "    skipped_due_to_truncated_source_response: "
            f"{skipped_truncated_reuse_count}"
        )

    return first_pass_metrics, final_metrics, repair_metrics


async def run_repair_evaluation(
    config: EvalConfig,
    repair_options: RepairOptions,
    first_pass_per_problem: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_phase0_runtime_loaded()
    print("\n" + "=" * 70)
    print("   Phase 4 Step 1: One-Turn Verifier-Guided Repair Eval")
    print("=" * 70)
    print(f"Mode: {config.mode}")
    print(f"Model: {config.model_path}")
    print(f"Datasets: {config.datasets}")
    print(f"Output: {config.output_dir}")
    print(f"Repair prompt mode: {repair_options.prompt_mode}")

    _filter_to_codecontests_datasets(config.datasets)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.mode == "verl":
        print("\n[Starting verl Rollout Servers]")
        rollout_servers, server_addresses = await start_rollout_servers(config)
    else:
        print(f"\n[Simple Mode] Connecting to {config.vllm_url}")
        from urllib.parse import urlparse

        parsed = urlparse(config.vllm_url)
        if parsed.scheme:
            server_addr = parsed.netloc or parsed.path
        else:
            server_addr = config.vllm_url
        server_addresses = [server_addr.rstrip("/")]
        rollout_servers = None

    run_info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "config": asdict(config),
        "repair_options": asdict(repair_options),
        "server_addresses": server_addresses,
        "python": sys.version,
        "protocol": "one_turn_repair",
        "first_pass_source": (
            {
                "mode": "reused_raw_per_problem",
                "path": first_pass_per_problem,
            }
            if first_pass_per_problem
            else {"mode": "generated"}
        ),
    }

    try:
        server_models = await fetch_openai_models(server_addresses)
        run_info["openai_models"] = server_models
    except Exception as e:
        run_info["openai_models_error"] = str(e)

    _save_json(output_dir / "run_info.json", run_info)
    print(f"Run info saved to: {output_dir / 'run_info.json'}")

    first_metrics_collector = MetricsCollector()
    final_metrics_collector = MetricsCollector()
    first_qa_logger = QALogger(output_dir / "qa_logs_first_pass", sample_size=config.qa_sample_size)
    repair_qa_logger = QALogger(output_dir / "qa_logs_repair_pass", sample_size=config.qa_sample_size)

    all_first_pass_metrics: Dict[str, Any] = {}
    all_final_metrics: Dict[str, Any] = {}
    all_repair_metrics: Dict[str, Any] = {}

    try:
        for dataset_key in config.datasets:
            print(f"\n[Loading {dataset_key}]")
            prompts = load_prompts(dataset_key, config)
            if not prompts:
                print("  No prompts found, skipping...")
                continue

            print(f"  Loaded {len(prompts)} problems")
            first_pass_metrics, final_metrics, repair_metrics = await evaluate_dataset_with_repair(
                dataset_key=dataset_key,
                prompts=prompts,
                server_addresses=server_addresses,
                config=config,
                repair_options=repair_options,
                first_pass_per_problem=first_pass_per_problem,
                first_metrics_collector=first_metrics_collector,
                final_metrics_collector=final_metrics_collector,
                first_qa_logger=first_qa_logger,
                repair_qa_logger=repair_qa_logger,
            )
            all_first_pass_metrics[dataset_key] = first_pass_metrics
            all_final_metrics[dataset_key] = final_metrics
            all_repair_metrics[dataset_key] = repair_metrics
    finally:
        if rollout_servers:
            print("\n[Shutting down Rollout Servers]")

    first_qa_logger.save()
    repair_qa_logger.save()

    final_summary = final_metrics_collector.get_summary()
    final_summary["protocol"] = "after_one_repair"
    final_summary["repair_prompt_mode"] = repair_options.prompt_mode
    first_pass_summary = first_metrics_collector.get_summary()
    first_pass_summary["protocol"] = "first_pass"
    first_pass_summary["repair_prompt_mode"] = repair_options.prompt_mode
    if first_pass_per_problem:
        first_pass_summary["source"] = "reused_raw_per_problem"
        first_pass_summary["source_path"] = first_pass_per_problem
        first_pass_summary["wall_clock_note"] = "throughput uses sum(gen_time + judge_time) from reused per-problem rows"

    overall_attempts = sum(v["repair_attempt_count"] for v in all_repair_metrics.values())
    overall_successes = sum(v["repair_success_count"] for v in all_repair_metrics.values())
    overall_extra_gen = sum(v["extra_gen_tokens_repair"] for v in all_repair_metrics.values())
    overall_extra_judge = sum(v["extra_judge_time_repair"] for v in all_repair_metrics.values())
    repair_summary = {
        "repair_prompt_mode": repair_options.prompt_mode,
        "datasets": all_repair_metrics,
        "overall": {
            "repair_attempt_count": overall_attempts,
            "repair_success_count": overall_successes,
            "conditional_repair_success": (
                overall_successes / overall_attempts if overall_attempts else 0.0
            ),
            "extra_gen_tokens_repair": overall_extra_gen,
            "extra_judge_time_repair": overall_extra_judge,
            "avg_extra_gen_tokens_repair": (
                overall_extra_gen / overall_attempts if overall_attempts else 0.0
            ),
            "avg_extra_judge_time_repair": (
                overall_extra_judge / overall_attempts if overall_attempts else 0.0
            ),
        },
    }

    _save_json(output_dir / "metrics.json", all_final_metrics)
    _save_json(output_dir / "first_pass_metrics.json", all_first_pass_metrics)
    _save_json(output_dir / "repair_metrics.json", all_repair_metrics)
    _save_json(output_dir / "summary.json", final_summary)
    _save_json(output_dir / "first_pass_summary.json", first_pass_summary)
    _save_json(output_dir / "repair_summary.json", repair_summary)

    print("\n--- Final Results ---")
    for dataset_key in all_final_metrics:
        first_pass_metrics = all_first_pass_metrics[dataset_key]
        final_metrics = all_final_metrics[dataset_key]
        repair_metrics = all_repair_metrics[dataset_key]
        print(f"\n{dataset_key}:")
        print(f"  first_pass accepted@1: {first_pass_metrics['accepted_at_1']:.2%}")
        print(f"  accepted_after_1_repair: {final_metrics['accepted_at_1']:.2%}")
        print(f"  repair_gain: {repair_metrics['repair_gain']:.2%}")
        print(f"  first_pass pass_ratio_mean: {first_pass_metrics['pass_ratio_mean']:.4f}")
        print(f"  after_repair pass_ratio_mean: {final_metrics['pass_ratio_mean']:.4f}")
        print(f"  repair_attempt_count: {repair_metrics['repair_attempt_count']}")
        print(f"  conditional_repair_success: {repair_metrics['conditional_repair_success']:.2%}")

    return {
        "metrics": all_final_metrics,
        "first_pass_metrics": all_first_pass_metrics,
        "repair_metrics": all_repair_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 Step 1: one-turn verifier-guided repair evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--mode", type=str, default="simple", choices=["verl", "simple"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--rollout", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--sandbox_url",
        type=str,
        default="http://localhost:8080",
        help="SandboxFusion 服务地址；shared verifier 路径支持逗号分隔多个 backend 做客户端 round-robin",
    )
    parser.add_argument("--run_timeout", type=int, default=EVAL_CONSTANTS.get("run_timeout", 30))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--use_external_tests", dest="use_external_tests", action="store_true", default=True)
    parser.add_argument("--no_external_tests", dest="use_external_tests", action="store_false")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=["codecontests_valid_big"],
    )
    parser.add_argument("--manifest_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/phase4_repair")
    parser.add_argument("--qa_sample_size", type=int, default=50)
    parser.add_argument("--save_full_results", dest="save_full_results", action="store_true", default=True)
    parser.add_argument("--no_save_full_results", dest="save_full_results", action="store_false")
    parser.add_argument("--max_prompt_chars", type=int, default=4000)
    parser.add_argument("--max_response_chars", type=int, default=12000)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--autofix_codecontests_entrypoint", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rlvr_coding_model")
    parser.add_argument("--max_concurrent", type=int, default=48)
    parser.add_argument("--max_concurrent_judges", type=int, default=12)
    parser.add_argument("--verifier_limiter_budget", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument(
        "--first_pass_per_problem",
        type=str,
        default=None,
        help="Existing raw eval per_problem JSONL path or directory containing <dataset>.jsonl. "
        "When set, skip first-pass generation/judging and reuse fixed responses/details.",
    )

    parser.add_argument("--repair_max_attempts", type=int, default=1)
    parser.add_argument(
        "--repair_error_types",
        nargs="+",
        default=["wrong_answer", "runtime_error", "timeout"],
    )
    parser.add_argument("--repair_min_pass_ratio", type=float, default=0.2)
    parser.add_argument("--repair_max_failure_cases", type=int, default=1)
    parser.add_argument("--repair_max_feedback_chars", type=int, default=1200)
    parser.add_argument("--repair_max_prompt_chars", type=int, default=12000)
    parser.add_argument(
        "--repair_prompt_mode",
        type=str,
        default="code_only",
        choices=["code_only", "short_diagnosis_code"],
    )

    args = parser.parse_args()

    _ensure_phase0_runtime_loaded()

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
        use_external_tests=args.use_external_tests,
        datasets=args.datasets,
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
        qa_sample_size=args.qa_sample_size,
        save_full_results=args.save_full_results,
        max_prompt_chars=args.max_prompt_chars,
        max_response_chars=args.max_response_chars,
        max_problems=args.max_problems,
        shuffle_seed=args.shuffle_seed,
        autofix_codecontests_entrypoint=args.autofix_codecontests_entrypoint,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        max_concurrent_requests=args.max_concurrent,
        max_concurrent_judges=args.max_concurrent_judges,
        verifier_limiter_budget=args.verifier_limiter_budget,
        batch_size=args.batch_size,
    )
    repair_options = RepairOptions(
        repair_max_attempts=args.repair_max_attempts,
        repair_error_types=list(args.repair_error_types),
        repair_min_pass_ratio=args.repair_min_pass_ratio,
        repair_max_failure_cases=args.repair_max_failure_cases,
        repair_max_feedback_chars=args.repair_max_feedback_chars,
        repair_max_prompt_chars=args.repair_max_prompt_chars,
        prompt_mode=args.repair_prompt_mode,
    )

    if config.mode == "verl" and not VERL_AVAILABLE:
        print("Error: verl mode requires verl package. Use --mode simple instead.")
        sys.exit(1)

    asyncio.run(
        run_repair_evaluation(
            config,
            repair_options,
            first_pass_per_problem=args.first_pass_per_problem,
        )
    )


if __name__ == "__main__":
    main()
