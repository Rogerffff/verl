from __future__ import annotations

import math
from typing import Any, Iterable, List

try:
    from coding_model_project.src.verifier import normalize_candidate, verify_candidate_batch
except ImportError:
    from verifier import normalize_candidate, verify_candidate_batch


def _coerce_iterable(values: Iterable[Any]) -> list[Any]:
    if isinstance(values, list):
        return values
    return list(values)


def _is_truncated(extra_info: Any) -> bool:
    if not isinstance(extra_info, dict):
        return False

    finish_reason = extra_info.get("finish_reason")
    return bool(extra_info.get("truncated_by_max_tokens", finish_reason == "length"))


def _apply_invalid_guardrails(summary: dict[str, Any], *, extra_info: Any) -> dict[str, Any]:
    result = dict(summary)
    finish_reason = extra_info.get("finish_reason") if isinstance(extra_info, dict) else None
    truncated_by_max_tokens = _is_truncated(extra_info)

    result["finish_reason"] = finish_reason
    result["truncated_by_max_tokens"] = truncated_by_max_tokens

    if truncated_by_max_tokens:
        result["invalid_for_rl"] = True
        result["invalid_reason"] = "truncated_by_max_tokens"

    return result


def _compute_reward_raw(summary: dict[str, Any], reward_mode: str) -> float:
    if summary["invalid_for_rl"]:
        return math.nan

    error_type = summary["error_type"]

    if reward_mode == "sparse_accepted":
        return 1.0 if summary["accepted"] else 0.0

    if reward_mode == "anchored_dense_v1":
        if error_type in {"syntax_error", "empty_output", "non_code", "extraction_failure"}:
            return -1.0
        return 0.8 * float(summary["pass_ratio_all"]) + 0.2 * float(summary["accepted"])

    if reward_mode == "dense_anchor_v1":
        if error_type in {"syntax_error", "empty_output", "non_code", "extraction_failure"}:
            return -1.0
        reward = -0.2 + 1.0 * float(summary["pass_ratio_all"]) + 0.2 * float(summary["accepted"])
        return max(-1.0, min(1.0, reward))

    if reward_mode == "dense_pass_ratio":
        return float(summary["pass_ratio_all"])

    if reward_mode == "rltf_piecewise":
        if error_type in {"syntax_error", "empty_output", "non_code", "extraction_failure"}:
            return -1.0
        if error_type in {"runtime_error", "timeout"}:
            return -0.6
        reward = -0.3 + 1.3 * float(summary["pass_ratio_all"])
        return max(-1.0, min(1.0, reward))

    raise ValueError(f"Unsupported reward_mode={reward_mode}")


def compute_score(
    *,
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    sandbox_endpoint: str,
    reward_mode: str = "dense_pass_ratio",
    limiter_budget: int = 8,
    run_timeout_s: int = 30,
    memory_limit_mb: int = 1024,
    autofix_codecontests_entrypoint: bool = False,
) -> List[dict[str, Any]]:
    del data_sources  # group/logging uses the original data_source column directly

    solution_list = _coerce_iterable(solution_strs)
    ground_truth_list = _coerce_iterable(ground_truths)
    extra_info_list = _coerce_iterable(extra_infos)

    candidates = [normalize_candidate(solution) for solution in solution_list]
    summaries = verify_candidate_batch(
        candidates=candidates,
        ground_truths=ground_truth_list,
        sandbox_endpoint=sandbox_endpoint,
        run_timeout_s=run_timeout_s,
        memory_limit_mb=memory_limit_mb,
        limiter_budget=limiter_budget,
        include_per_case_results=False,
        autofix_codecontests_entrypoint=autofix_codecontests_entrypoint,
    )

    results: List[dict[str, Any]] = []
    for summary, extra_info in zip(summaries, extra_info_list, strict=True):
        summary_dict = _apply_invalid_guardrails(summary.to_dict(), extra_info=extra_info)
        summary_dict.pop("per_case_results", None)
        reward_raw = _compute_reward_raw(summary_dict, reward_mode=reward_mode)
        summary_dict["reward_raw"] = reward_raw
        summary_dict["score"] = 0.0 if math.isnan(reward_raw) else reward_raw
        summary_dict["problem_id"] = ""
        if isinstance(extra_info, dict):
            summary_dict["problem_id"] = str(extra_info.get("problem_id", ""))
        results.append(summary_dict)
    return results
