from __future__ import annotations

import math
from typing import Any, Iterable, List

try:
    from coding_model_project.src.verifier import normalize_candidate, verify_candidate_batch
except ImportError:
    from verifier import normalize_candidate, verify_candidate_batch


BAD_OUTPUT_ERROR_TYPES = {"syntax_error", "empty_output", "non_code", "extraction_failure"}
SUPPORTED_REWARD_MODES = {"repair_delta_v0"}
_MISSING = object()


def _coerce_iterable(values: Iterable[Any]) -> list[Any]:
    if isinstance(values, list):
        return values
    return list(values)


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return False


def _as_float(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected {field_name} to be numeric, got {value!r}") from exc


def _unit_interval(value: Any, *, field_name: str) -> float:
    number = _as_float(value, field_name=field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"Expected {field_name} in [0, 1], got {number}")
    return number


def _require_field(mapping: dict[str, Any], key: str, *, field_name: str) -> Any:
    value = mapping.get(key, _MISSING)
    if value is _MISSING:
        raise ValueError(f"Repair reward requires {field_name}")
    return value


def _is_truncated(extra_info: Any) -> bool:
    if not isinstance(extra_info, dict):
        return False

    finish_reason = extra_info.get("finish_reason")
    return bool(extra_info.get("truncated_by_max_tokens", finish_reason == "length"))


def _apply_invalid_guardrails(summary: dict[str, Any], *, extra_info: Any) -> dict[str, Any]:
    result = dict(summary)
    finish_reason = _as_str(extra_info.get("finish_reason")) if isinstance(extra_info, dict) else ""
    truncated_by_max_tokens = _is_truncated(extra_info)

    result["finish_reason"] = finish_reason
    result["truncated_by_max_tokens"] = truncated_by_max_tokens

    if truncated_by_max_tokens:
        result["invalid_for_rl"] = True
        result["invalid_reason"] = "truncated_by_max_tokens"

    return result


def _compute_quality(pass_ratio_all: float, accepted: bool) -> float:
    return 0.8 * pass_ratio_all + 0.2 * float(accepted)


def _extract_first_pass_fields(ground_truth: Any, extra_info: Any) -> dict[str, Any]:
    if not isinstance(ground_truth, dict):
        raise ValueError("Repair reward requires reward_model.ground_truth to be a dict")

    first_pass = ground_truth.get("first_pass")
    if not isinstance(first_pass, dict):
        raise ValueError("Repair reward requires ground_truth.first_pass to be a dict")

    repair_metadata = ground_truth.get("repair_metadata")
    if repair_metadata is None:
        repair_metadata = {}
    if not isinstance(repair_metadata, dict):
        raise ValueError("Repair reward requires ground_truth.repair_metadata to be a dict when present")

    problem_id = _as_str(ground_truth.get("problem_id"))
    if not problem_id and isinstance(extra_info, dict):
        problem_id = _as_str(extra_info.get("problem_id"))

    return {
        "problem_id": problem_id,
        "first_pass_pass_ratio_all": _unit_interval(
            _require_field(first_pass, "pass_ratio_all", field_name="ground_truth.first_pass.pass_ratio_all"),
            field_name="first_pass.pass_ratio_all",
        ),
        "first_pass_accepted": _as_bool(
            _require_field(first_pass, "accepted", field_name="ground_truth.first_pass.accepted")
        ),
        "first_pass_bucket": _as_str(first_pass.get("pass_ratio_bucket")),
        "first_pass_error_type": _as_str(first_pass.get("error_type")),
        "first_pass_invalid_for_rl": _as_bool(first_pass.get("invalid_for_rl")),
        "first_pass_finish_reason": _as_str(first_pass.get("finish_reason")),
        "repair_prompt_mode": _as_str(repair_metadata.get("prompt_mode")),
        "teacher_prompt_mode": _as_str(repair_metadata.get("teacher_prompt_mode")),
        "repair_source_protocol": _as_str(repair_metadata.get("source_protocol")),
        "repair_source_run_id": _as_str(repair_metadata.get("source_run_id")),
        "repair_feedback_source": _as_str(repair_metadata.get("feedback_source")),
    }


def _compute_repair_delta_v0(
    *,
    summary: dict[str, Any],
    first_pass: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    p0 = first_pass["first_pass_pass_ratio_all"]
    a0 = first_pass["first_pass_accepted"]
    p1 = _unit_interval(summary.get("pass_ratio_all", 0.0), field_name="pass_ratio_all")
    a1 = _as_bool(summary.get("accepted"))

    q0 = _compute_quality(p0, a0)
    q1 = _compute_quality(p1, a1)
    delta_q = q1 - q0
    delta_pos = max(delta_q, 0.0)
    delta_neg = max(-delta_q, 0.0)
    accepted_gain = float((not a0) and a1)

    if summary["invalid_for_rl"]:
        reward_raw = math.nan
    elif _as_str(summary.get("error_type")) in BAD_OUTPUT_ERROR_TYPES:
        reward_raw = -1.0
    else:
        reward_raw = q1 + 0.25 * delta_pos - 0.75 * delta_neg + 0.25 * accepted_gain

    repair_logs = {
        "q0": q0,
        "q1": q1,
        "delta_q": delta_q,
        "delta_pos": delta_pos,
        "delta_neg": delta_neg,
        "accepted_gain": accepted_gain,
        "first_pass_pass_ratio_all": p0,
        "first_pass_accepted": a0,
        "first_pass_bucket": first_pass["first_pass_bucket"],
        "first_pass_error_type": first_pass["first_pass_error_type"],
        "first_pass_invalid_for_rl": first_pass["first_pass_invalid_for_rl"],
        "first_pass_finish_reason": first_pass["first_pass_finish_reason"],
    }
    return reward_raw, repair_logs


def _validate_flat_result(result: dict[str, Any]) -> None:
    for key, value in result.items():
        if isinstance(value, (dict, list, tuple, set)):
            raise TypeError(f"repair reward result field {key!r} must be flat, got {type(value).__name__}")
        if value is None:
            raise TypeError(f"repair reward result field {key!r} must not be None")


def compute_score(
    *,
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    sandbox_endpoint: str,
    reward_mode: str = "repair_delta_v0",
    limiter_budget: int = 8,
    run_timeout_s: int = 30,
    memory_limit_mb: int = 1024,
    autofix_codecontests_entrypoint: bool = False,
    expected_prompt_mode: str = "",
) -> List[dict[str, Any]]:
    del data_sources  # logging still uses the original data_source column directly

    if reward_mode not in SUPPORTED_REWARD_MODES:
        if reward_mode == "repair_delta_edit_v1":
            raise NotImplementedError("repair_delta_edit_v1 is planned but not implemented yet")
        raise ValueError(f"Unsupported repair reward_mode={reward_mode}")

    solution_list = _coerce_iterable(solution_strs)
    ground_truth_list = _coerce_iterable(ground_truths)
    extra_info_list = _coerce_iterable(extra_infos)
    first_pass_list = [
        _extract_first_pass_fields(ground_truth, extra_info)
        for ground_truth, extra_info in zip(ground_truth_list, extra_info_list, strict=True)
    ]

    if expected_prompt_mode:
        for first_pass in first_pass_list:
            if first_pass["repair_prompt_mode"] != expected_prompt_mode:
                raise ValueError(
                    "Repair prompt mode mismatch: "
                    f"expected {expected_prompt_mode!r}, got {first_pass['repair_prompt_mode']!r}"
                )

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
    for summary, first_pass, extra_info in zip(summaries, first_pass_list, extra_info_list, strict=True):
        summary_dict = _apply_invalid_guardrails(summary.to_dict(), extra_info=extra_info)
        summary_dict.pop("per_case_results", None)
        reward_raw, repair_logs = _compute_repair_delta_v0(summary=summary_dict, first_pass=first_pass)

        result = dict(summary_dict)
        result["reward_raw"] = reward_raw
        result["score"] = 0.0 if math.isnan(reward_raw) else reward_raw
        result["problem_id"] = first_pass["problem_id"]
        result["repair_prompt_mode"] = first_pass["repair_prompt_mode"]
        result["teacher_prompt_mode"] = first_pass["teacher_prompt_mode"]
        result["repair_source_protocol"] = first_pass["repair_source_protocol"]
        result["repair_source_run_id"] = first_pass["repair_source_run_id"]
        result["repair_feedback_source"] = first_pass["repair_feedback_source"]
        result.update(repair_logs)
        _validate_flat_result(result)
        results.append(result)

    return results
