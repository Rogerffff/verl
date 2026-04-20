#!/usr/bin/env python3
"""Helpers for one-turn verifier-guided repair prompts."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:
    from coding_model_project.src.verifier import normalize_candidate
except ImportError:
    from verifier import normalize_candidate


RepairPromptMode = Literal["code_only", "short_diagnosis_code"]


REPAIR_SYSTEM_PROMPT = """You are an expert Python competitive programming repair assistant.

You are given:
1. The original problem statement.
2. A previous Python solution that failed some tests.
3. A short verifier feedback summary.

Your job:
- Repair the solution using the feedback.
- Output a complete Python program.
- The program must read from stdin and write to stdout.
- Return a complete repaired solution, not a patch or diff.
- Follow the exact output format requested by the user prompt.
"""

TIMEOUT_COMPLEXITY_HINT = (
    "Your previous solution timed out on the test case shown below. Even small or moderate "
    "inputs can trigger TLE if the algorithm is super-linear. Re-examine the time complexity: "
    "look for nested loops, repeated string/list scans, recursion without memoization, or "
    "O(n^2) work that should be O(n log n) or O(n). The displayed test case may be truncated; "
    "do not rely on its size as evidence of input scale."
)


def _truncate_text(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    value = str(text)
    if len(value) <= max_chars:
        return value
    if max_chars <= 0:
        return ""
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _extract_first_code(first_response: str) -> str:
    candidate = normalize_candidate(first_response or "")
    if candidate.extraction_status == "ok" and candidate.extracted_code:
        return candidate.extracted_code
    return first_response or ""


def _safe_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _exception_text(case: Dict[str, Any]) -> str:
    for source in (_safe_text(case.get("stderr")), _safe_text(case.get("message"))):
        if not source:
            continue
        lines = [line.strip() for line in source.splitlines() if line.strip()]
        for line in reversed(lines):
            if any(token in line for token in ("Error", "Exception", "Traceback")):
                return line
    return ""


def _text_len(value: Any) -> int:
    return len(_safe_text(value))


def _line_count(value: Any) -> int:
    text = _safe_text(value)
    if not text:
        return 0
    return text.count("\n") + 1


def _test_idx(case: Dict[str, Any]) -> int:
    try:
        return int(case.get("test_idx", 10**9))
    except (TypeError, ValueError):
        return 10**9


def _is_failing_case(case: Dict[str, Any], status: str) -> bool:
    if case.get("status") != status:
        return False
    if case.get("passed") is True:
        return False
    return True


def _has_exception_signal(case: Dict[str, Any]) -> bool:
    haystack = f"{_safe_text(case.get('stderr'))}\n{_safe_text(case.get('message'))}"
    return any(token in haystack for token in ("Error", "Exception", "Traceback"))


def _wrong_answer_sort_key(case: Dict[str, Any]) -> tuple[float, int, int, int]:
    stdin_len = _text_len(case.get("stdin"))
    expected_len = _text_len(case.get("expected"))
    actual_len = _text_len(case.get("actual"))
    output_delta = abs(expected_len - actual_len) + abs(
        _line_count(case.get("expected")) - _line_count(case.get("actual"))
    )
    return (stdin_len + 0.5 * output_delta, stdin_len, expected_len, _test_idx(case))


def _runtime_error_sort_key(case: Dict[str, Any]) -> tuple[int, int, int]:
    return (0 if _has_exception_signal(case) else 1, _text_len(case.get("stdin")), _test_idx(case))


def _timeout_sort_key(case: Dict[str, Any]) -> tuple[int, int]:
    return (_text_len(case.get("stdin")), _test_idx(case))


def _select_cases_for_error_type(
    case_results: List[Dict[str, Any]],
    *,
    error_type: str,
    max_failure_cases: int,
) -> tuple[List[Dict[str, Any]], str]:
    if error_type == "wrong_answer":
        selected = sorted(
            (case for case in case_results if _is_failing_case(case, "wrong_answer")),
            key=_wrong_answer_sort_key,
        )
        return selected[:max_failure_cases], "simplest_counterexample"

    if error_type == "runtime_error":
        selected = sorted(
            (case for case in case_results if _is_failing_case(case, "runtime_error")),
            key=_runtime_error_sort_key,
        )
        return selected[:max_failure_cases], "clearest_exception_then_shortest_stdin"

    if error_type == "timeout":
        selected = sorted(
            (case for case in case_results if _is_failing_case(case, "timeout")),
            key=_timeout_sort_key,
        )
        return selected[:max_failure_cases], "shortest_timeout_stdin"

    return [], "fallback"


def _case_results_from_details(details: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not details:
        return []

    per_case_results = details.get("per_case_results")
    if isinstance(per_case_results, list):
        return per_case_results

    test_case_results = details.get("test_case_results")
    if isinstance(test_case_results, list):
        return test_case_results

    return []


def _select_failure_cases(
    case_results: List[Dict[str, Any]],
    *,
    error_type: str,
    max_failure_cases: int,
    max_case_chars: int,
) -> tuple[List[Dict[str, Any]], str]:
    selected_cases, strategy = _select_cases_for_error_type(
        case_results,
        error_type=error_type,
        max_failure_cases=max_failure_cases,
    )
    selected: List[Dict[str, Any]] = []
    for case in selected_cases:
        status = _safe_text(case.get("status")) or error_type
        selected_case = {
            "status": status,
            "test_idx": case.get("test_idx", -1),
            "stdin": _truncate_text(case.get("stdin", ""), max_case_chars),
        }
        if status == "wrong_answer":
            selected_case.update(
                {
                    "expected": _truncate_text(case.get("expected", ""), max_case_chars),
                    "actual": _truncate_text(case.get("actual", ""), max_case_chars),
                }
            )
        else:
            selected_case.update(
                {
                    "exception": _truncate_text(_exception_text(case), max_case_chars),
                    "stderr": _truncate_text(case.get("stderr", ""), max_case_chars),
                    "message": _truncate_text(case.get("message", ""), max_case_chars),
                }
            )
        selected.append(selected_case)
    return selected, strategy


def build_repair_feedback_from_eval_result(
    eval_result: Any,
    *,
    max_failure_cases: int = 1,
    max_case_chars: int = 400,
) -> Dict[str, Any]:
    """Build short structured feedback from an EvalResult-like object."""
    details = getattr(eval_result, "details", {}) or {}
    passed_tests = details.get("passed_tests", details.get("passed", 0))
    total_tests = details.get("total_tests", details.get("total", 0))
    error_type = getattr(eval_result, "error_type", "unknown")

    failures, selection_strategy = _select_failure_cases(
        _case_results_from_details(details),
        error_type=error_type,
        max_failure_cases=max_failure_cases,
        max_case_chars=max_case_chars,
    )

    first_failure = details.get("first_failure")
    if not failures and isinstance(first_failure, dict):
        selection_strategy = "first_failure_fallback"
        status = error_type
        selected_case = {
            "status": status,
            "test_idx": first_failure.get("test_idx", -1),
            "stdin": _truncate_text(first_failure.get("stdin", ""), max_case_chars),
        }
        if status == "wrong_answer":
            selected_case.update(
                {
                    "expected": _truncate_text(first_failure.get("expected", ""), max_case_chars),
                    "actual": _truncate_text(
                        first_failure.get("actual", first_failure.get("got_stdout", "")),
                        max_case_chars,
                    ),
                }
            )
        else:
            selected_case.update(
                {
                    "exception": _truncate_text(_exception_text(first_failure), max_case_chars),
                    "stderr": _truncate_text(first_failure.get("stderr", ""), max_case_chars),
                    "message": _truncate_text(first_failure.get("message", ""), max_case_chars),
                }
            )
        failures = [selected_case]

    last_error = details.get("last_error")
    if not failures and isinstance(last_error, str) and last_error:
        selection_strategy = "last_error_fallback"
        failures = [
            {
                "status": error_type,
                "test_idx": -1,
                "message": _truncate_text(last_error, max_case_chars),
            }
        ]

    return {
        "passed_tests": int(passed_tests or 0),
        "total_tests": int(total_tests or 0),
        "error_type": str(error_type),
        "selection_strategy": selection_strategy,
        "extra_hint": TIMEOUT_COMPLEXITY_HINT if error_type == "timeout" else "",
        "selected_failures": failures,
    }


def build_codecontests_repair_prompt(
    *,
    problem_prompt: str,
    first_response: str,
    feedback: Dict[str, Any],
    max_prompt_chars: int = 12000,
    prompt_mode: RepairPromptMode = "code_only",
) -> str:
    """Build a compact second-pass repair prompt for CodeContests."""
    problem_text = (problem_prompt or "").strip()
    previous_code = _extract_first_code(first_response).strip()

    feedback_lines: List[str] = []
    feedback_lines.append(
        f"- Passed {feedback.get('passed_tests', 0)} / {feedback.get('total_tests', 0)} tests"
    )
    feedback_lines.append(f"- Primary error type: {feedback.get('error_type', 'unknown')}")
    for idx, failure in enumerate(feedback.get("selected_failures", []), start=1):
        status = failure.get("status", "unknown")
        feedback_lines.append(f"- Representative failure {idx}:")
        stdin = failure.get("stdin")
        if stdin:
            feedback_lines.append(f"  stdin: {stdin}")
        if status == "wrong_answer":
            feedback_lines.append(f"  expected: {failure.get('expected', '')}")
            feedback_lines.append(f"  actual: {failure.get('actual', '')}")
        else:
            exception = failure.get("exception")
            message = failure.get("message")
            stderr = failure.get("stderr")
            if exception:
                feedback_lines.append(f"  exception: {exception}")
            if message:
                feedback_lines.append(f"  message: {message}")
            elif stderr:
                feedback_lines.append(f"  stderr: {stderr}")

    feedback_text = "\n".join(feedback_lines).strip()
    extra_hint = _safe_text(feedback.get("extra_hint")).strip()
    if extra_hint:
        feedback_text = f"{feedback_text}\n- Repair hint: {extra_hint}"

    # Keep the repair prompt safely below the eval wrapper's vLLM max-model-len.
    max_problem_chars = max(1200, min(6000, max_prompt_chars // 2))
    max_code_chars = max(1000, min(4000, max_prompt_chars // 3))
    max_feedback_block_chars = max(600, min(1800, max_prompt_chars // 4))

    problem_text = _truncate_text(problem_text, max_problem_chars)
    previous_code = _truncate_text(previous_code, max_code_chars)
    feedback_text = _truncate_text(feedback_text, max_feedback_block_chars)

    def render(current_problem: str, current_code: str, current_feedback: str) -> str:
        lines: List[str] = []
        lines.append("Original problem statement:")
        lines.append(current_problem)
        lines.append("")
        lines.append("Previous incorrect Python solution:")
        lines.append("```python")
        lines.append(current_code)
        lines.append("```")
        lines.append("")
        lines.append("Verifier feedback:")
        lines.append(current_feedback)
        lines.append("")
        if prompt_mode == "short_diagnosis_code":
            lines.append(
                "Please first give a very short diagnosis of the bug and a very short fix plan, "
                "then provide the repaired full Python solution."
            )
            lines.append("")
            lines.append("Requirements:")
            lines.append("- BUG_SUMMARY must be grounded in the provided feedback.")
            lines.append("- FIX_PLAN must describe only the intended repair direction.")
            lines.append("- BUG_SUMMARY must be at most 1 sentence.")
            lines.append("- FIX_PLAN must be at most 1 sentence.")
            lines.append("- Do not include any other sections.")
            lines.append("- The repaired code must be complete and executable.")
            lines.append("- Put only the final repaired program inside a single <code>...</code> block.")
            lines.append("")
            lines.append("Use exactly this format:")
            lines.append("BUG_SUMMARY: ...")
            lines.append("FIX_PLAN: ...")
            lines.append("<code>")
            lines.append("...your repaired Python program...")
            lines.append("</code>")
        else:
            lines.append("Please repair the solution using the feedback above.")
            lines.append("")
            lines.append("Requirements:")
            lines.append("- Return a complete program that reads from stdin and writes to stdout.")
            lines.append("- Output only the repaired solution.")
            lines.append("- Put the entire final answer inside a single <code>...</code> block.")
            lines.append("")
            lines.append("Format your final answer as:")
            lines.append("<code>")
            lines.append("...your repaired Python program...")
            lines.append("</code>")
        return "\n".join(lines).strip()

    prompt = render(problem_text, previous_code, feedback_text)
    if len(prompt) > max_prompt_chars:
        sections = {
            "code": previous_code,
            "problem": problem_text,
            "feedback": feedback_text,
        }
        minimums = {
            "code": min(len(previous_code), 400),
            "problem": min(len(problem_text), 600),
            "feedback": min(len(feedback_text), 250),
        }
        for name in ("code", "problem", "feedback"):
            if len(prompt) <= max_prompt_chars:
                break
            overflow = len(prompt) - max_prompt_chars
            reducible = max(0, len(sections[name]) - minimums[name])
            if reducible <= 0:
                continue
            target_len = len(sections[name]) - min(reducible, overflow)
            sections[name] = _truncate_text(sections[name], target_len)
            prompt = render(sections["problem"], sections["code"], sections["feedback"])

    return prompt
