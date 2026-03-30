from __future__ import annotations

import ast
import concurrent.futures
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional


_LIMITER_REGISTRY_LOCK = threading.Lock()
_LIMITER_REGISTRY: dict[int, threading.BoundedSemaphore] = {}


@dataclass
class CandidateRecord:
    raw_completion: str
    extracted_code: str
    extraction_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationSummary:
    accepted: bool
    passed_tests: int
    total_tests: int
    pass_ratio_all: float
    error_type: str
    invalid_for_rl: bool
    invalid_reason: str
    judge_time_s: float
    extraction_status: str
    per_case_results: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_limiter(limit: int) -> threading.BoundedSemaphore:
    limit = max(1, int(limit))
    with _LIMITER_REGISTRY_LOCK:
        limiter = _LIMITER_REGISTRY.get(limit)
        if limiter is None:
            limiter = threading.BoundedSemaphore(limit)
            _LIMITER_REGISTRY[limit] = limiter
        return limiter


@contextmanager
def _acquire_limiter(limit: int):
    limiter = _get_limiter(limit)
    limiter.acquire()
    try:
        yield
    finally:
        limiter.release()


def _lazy_import_sandbox():
    from sandbox_fusion import RunCodeRequest, run_code

    return {
        "RunCodeRequest": RunCodeRequest,
        "run_code": run_code,
    }


def _enum_to_value(value: Any) -> str:
    if value is None:
        return ""

    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        text = enum_value
    else:
        text = str(value)

    if text.startswith("RunStatus."):
        return text.split(".", 1)[1]
    if text.startswith("CommandRunStatus."):
        return text.split(".", 1)[1]

    for token in ("Success", "Failed", "SandboxError", "Finished", "TimeLimitExceeded", "Error"):
        if token in text:
            return token
    return text


def _parse_run_code_result_detailed(result: Any) -> Dict[str, Any]:
    overall_status = _enum_to_value(getattr(result, "status", "unknown")) or "unknown"
    message = str(getattr(result, "message", "") or "")

    compile_result = getattr(result, "compile_result", None)
    compile_stdout = str(getattr(compile_result, "stdout", "") or "") if compile_result is not None else ""
    compile_stderr = str(getattr(compile_result, "stderr", "") or "") if compile_result is not None else ""

    run_result = getattr(result, "run_result", None)
    run_status = _enum_to_value(getattr(run_result, "status", "")) if run_result is not None else ""
    return_code = getattr(run_result, "return_code", None) if run_result is not None else None
    run_stdout = str(getattr(run_result, "stdout", "") or "") if run_result is not None else ""
    run_stderr = str(getattr(run_result, "stderr", "") or "") if run_result is not None else ""

    return {
        "overall_status": overall_status,
        "message": message,
        "compile_stdout": compile_stdout,
        "compile_stderr": compile_stderr,
        "run_status": run_status,
        "return_code": return_code,
        "run_stdout": run_stdout,
        "run_stderr": run_stderr,
    }


def _truncate_text(text: str, limit: int) -> str:
    return (text or "")[:limit]


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def _looks_like_python_code(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False

    try:
        ast.parse(stripped)
        return True
    except SyntaxError:
        pass

    code_markers = (
        "def ",
        "class ",
        "import ",
        "from ",
        "return ",
        "print(",
        "if __name__",
        "sys.stdin",
        "sys.stdout",
        "input(",
        "for ",
        "while ",
        "=",
    )
    return any(marker in stripped for marker in code_markers)


def _extract_code_block(completion: str) -> CandidateRecord:
    stripped = (completion or "").strip()
    if not stripped:
        return CandidateRecord(raw_completion=completion or "", extracted_code="", extraction_status="empty_output")

    code_tag_pattern = r"<code>(.*?)</code>"
    matches = re.findall(code_tag_pattern, completion, re.DOTALL | re.IGNORECASE)
    if matches:
        extracted = max(matches, key=len).strip()
        if extracted:
            return CandidateRecord(raw_completion=completion, extracted_code=extracted, extraction_status="ok")
        return CandidateRecord(raw_completion=completion, extracted_code="", extraction_status="extraction_failure")

    md_pattern = r"```(?:python)?[ \t]*\n?(.*?)```"
    matches = re.findall(md_pattern, completion, re.DOTALL | re.IGNORECASE)
    if matches:
        extracted = max(matches, key=len).strip()
        if extracted:
            return CandidateRecord(raw_completion=completion, extracted_code=extracted, extraction_status="ok")
        return CandidateRecord(raw_completion=completion, extracted_code="", extraction_status="extraction_failure")

    if "```" in completion or re.search(r"</?code>", completion, re.IGNORECASE):
        return CandidateRecord(raw_completion=completion, extracted_code="", extraction_status="extraction_failure")

    if _looks_like_python_code(stripped):
        return CandidateRecord(raw_completion=completion, extracted_code=stripped, extraction_status="ok")

    return CandidateRecord(raw_completion=completion, extracted_code="", extraction_status="non_code")


def normalize_candidate(raw_completion: str) -> CandidateRecord:
    return _extract_code_block(raw_completion)


def _codecontests_autofix_entrypoint(code: str) -> tuple[str, Dict[str, Any]]:
    solve_defined = re.search(r"^\s*def\s+solve\s*\(", code, re.MULTILINE) is not None
    solve_called = re.search(r"solve\s*\(", code.split("def solve", 1)[-1] if "def solve" in code else code) is not None

    if not solve_defined:
        return code, {"applied": False, "reason": "solve_not_defined"}
    if solve_called:
        return code, {"applied": False, "reason": "solve_already_called"}

    patched = code.rstrip() + "\n\nif __name__ == '__main__':\n    solve()\n"
    return patched, {"applied": True, "reason": "added_main_guard_for_solve"}


def _empty_or_invalid_summary(
    candidate: CandidateRecord,
    *,
    total_tests: int,
    error_type: str,
    invalid_for_rl: bool,
    invalid_reason: str,
    include_per_case_results: bool,
) -> VerificationSummary:
    per_case_results: Optional[List[Dict[str, Any]]]
    if include_per_case_results and total_tests > 0:
        per_case_results = [
            {
                "test_idx": idx,
                "status": error_type,
                "passed": False,
            }
            for idx in range(total_tests)
        ]
    else:
        per_case_results = None

    return VerificationSummary(
        accepted=False,
        passed_tests=0,
        total_tests=total_tests,
        pass_ratio_all=0.0,
        error_type=error_type,
        invalid_for_rl=invalid_for_rl,
        invalid_reason=invalid_reason,
        judge_time_s=0.0,
        extraction_status=candidate.extraction_status,
        per_case_results=per_case_results,
    )


def _run_code_request(
    *,
    code: str,
    sandbox_endpoint: str,
    run_timeout_s: int,
    memory_limit_mb: int,
    limiter_budget: int,
    stdin: Optional[str] = None,
) -> Any:
    sandbox = _lazy_import_sandbox()
    request_kwargs = {
        "code": code,
        "language": "python",
        "run_timeout": run_timeout_s,
        "memory_limit_MB": memory_limit_mb,
    }
    if stdin is not None:
        request_kwargs["stdin"] = stdin

    with _acquire_limiter(limiter_budget):
        return sandbox["run_code"](sandbox["RunCodeRequest"](**request_kwargs), endpoint=sandbox_endpoint)


def _is_sandbox_error(parsed: Dict[str, Any]) -> bool:
    return parsed.get("overall_status") == "SandboxError"


def _classify_single_run(parsed: Dict[str, Any]) -> tuple[str, bool, str]:
    if _is_sandbox_error(parsed):
        return "sandbox_error", True, "sandbox_error"
    if parsed["run_status"] == "TimeLimitExceeded" or "timelimitexceeded" in parsed["run_stdout"].lower():
        return "timeout", False, ""
    if "SyntaxError" in parsed["compile_stderr"] or "SyntaxError" in parsed["run_stderr"]:
        return "syntax_error", False, ""
    if "AssertionError" in parsed["run_stdout"] or "AssertionError" in parsed["run_stderr"]:
        return "wrong_answer", False, ""
    return "runtime_error", False, ""


def _classify_request_exception(exc: Exception) -> tuple[str, bool, str]:
    message = str(exc).lower()
    if "sandbox responded with error" in message or "sandboxerror" in message or "sandbox error" in message:
        return "sandbox_error", True, "sandbox_error"
    return "api_error", True, "api_error"


def _verify_humaneval_candidate(
    candidate: CandidateRecord,
    test_cases: Dict[str, Any],
    sandbox_endpoint: str,
    run_timeout_s: int,
    memory_limit_mb: int,
    limiter_budget: int,
) -> VerificationSummary:
    total_tests = 1 if test_cases else 0
    if candidate.extraction_status != "ok":
        return _empty_or_invalid_summary(
            candidate,
            total_tests=total_tests,
            error_type=candidate.extraction_status,
            invalid_for_rl=False,
            invalid_reason="",
            include_per_case_results=False,
        )

    test_code = test_cases.get("test_code", "")
    entry_point = test_cases.get("entry_point", "")
    full_code = f"{candidate.extracted_code}\n\n{test_code}\n\ncheck({entry_point})"

    start_time = time.time()
    try:
        parsed = _parse_run_code_result_detailed(
            _run_code_request(
                code=full_code,
                sandbox_endpoint=sandbox_endpoint,
                run_timeout_s=run_timeout_s,
                memory_limit_mb=memory_limit_mb,
                limiter_budget=limiter_budget,
            )
        )
    except Exception as exc:
        error_type, invalid_for_rl, invalid_reason = _classify_request_exception(exc)
        return VerificationSummary(
            accepted=False,
            passed_tests=0,
            total_tests=1,
            pass_ratio_all=0.0,
            error_type=error_type,
            invalid_for_rl=invalid_for_rl,
            invalid_reason=invalid_reason,
            judge_time_s=time.time() - start_time,
            extraction_status=candidate.extraction_status,
            per_case_results=None,
        )

    accepted = parsed["overall_status"] == "Success" and parsed["return_code"] in (0, None)
    if accepted:
        error_type, invalid_for_rl, invalid_reason = "success", False, ""
    else:
        error_type, invalid_for_rl, invalid_reason = _classify_single_run(parsed)

    return VerificationSummary(
        accepted=accepted,
        passed_tests=1 if accepted else 0,
        total_tests=1,
        pass_ratio_all=1.0 if accepted else 0.0,
        error_type=error_type,
        invalid_for_rl=invalid_for_rl,
        invalid_reason=invalid_reason,
        judge_time_s=time.time() - start_time,
        extraction_status=candidate.extraction_status,
        per_case_results=None,
    )


def _verify_mbpp_candidate(
    candidate: CandidateRecord,
    test_cases: Dict[str, Any],
    sandbox_endpoint: str,
    run_timeout_s: int,
    memory_limit_mb: int,
    limiter_budget: int,
) -> VerificationSummary:
    test_list = test_cases.get("test_list", [])
    if not test_list:
        return VerificationSummary(
            accepted=False,
            passed_tests=0,
            total_tests=0,
            pass_ratio_all=0.0,
            error_type="no_test_cases",
            invalid_for_rl=True,
            invalid_reason="no_test_cases",
            judge_time_s=0.0,
            extraction_status=candidate.extraction_status,
            per_case_results=None,
        )

    if candidate.extraction_status != "ok":
        return _empty_or_invalid_summary(
            candidate,
            total_tests=1,
            error_type=candidate.extraction_status,
            invalid_for_rl=False,
            invalid_reason="",
            include_per_case_results=False,
        )

    test_setup_code = test_cases.get("test_setup_code", "")
    test_code = "\n".join(test_list)
    full_code = f"{test_setup_code}\n\n{candidate.extracted_code}\n\n{test_code}"

    start_time = time.time()
    try:
        parsed = _parse_run_code_result_detailed(
            _run_code_request(
                code=full_code,
                sandbox_endpoint=sandbox_endpoint,
                run_timeout_s=run_timeout_s,
                memory_limit_mb=memory_limit_mb,
                limiter_budget=limiter_budget,
            )
        )
    except Exception as exc:
        error_type, invalid_for_rl, invalid_reason = _classify_request_exception(exc)
        return VerificationSummary(
            accepted=False,
            passed_tests=0,
            total_tests=1,
            pass_ratio_all=0.0,
            error_type=error_type,
            invalid_for_rl=invalid_for_rl,
            invalid_reason=invalid_reason,
            judge_time_s=time.time() - start_time,
            extraction_status=candidate.extraction_status,
            per_case_results=None,
        )

    accepted = parsed["overall_status"] == "Success" and parsed["return_code"] in (0, None)
    if accepted:
        error_type, invalid_for_rl, invalid_reason = "success", False, ""
    else:
        error_type, invalid_for_rl, invalid_reason = _classify_single_run(parsed)

    return VerificationSummary(
        accepted=accepted,
        passed_tests=1 if accepted else 0,
        total_tests=1,
        pass_ratio_all=1.0 if accepted else 0.0,
        error_type=error_type,
        invalid_for_rl=invalid_for_rl,
        invalid_reason=invalid_reason,
        judge_time_s=time.time() - start_time,
        extraction_status=candidate.extraction_status,
        per_case_results=None,
    )


def _verify_codecontests_testcase(
    *,
    code: str,
    testcase: Dict[str, Any],
    test_idx: int,
    sandbox_endpoint: str,
    run_timeout_s: int,
    memory_limit_mb: int,
    limiter_budget: int,
) -> Dict[str, Any]:
    stdin_input = testcase.get("input", "")
    expected_output = (testcase.get("output", "") or "").strip()
    try:
        parsed = _parse_run_code_result_detailed(
            _run_code_request(
                code=code,
                stdin=stdin_input,
                sandbox_endpoint=sandbox_endpoint,
                run_timeout_s=run_timeout_s,
                memory_limit_mb=memory_limit_mb,
                limiter_budget=limiter_budget,
            )
        )
    except Exception as exc:
        error_type, _, _ = _classify_request_exception(exc)
        return {
            "test_idx": test_idx,
            "status": error_type,
            "passed": False,
            "stdin": _truncate_text(stdin_input, 300),
            "error": _truncate_text(str(exc), 500),
        }

    actual_stdout = parsed["run_stdout"] or ""
    actual_output = actual_stdout.strip()
    base_result: Dict[str, Any] = {
        "test_idx": test_idx,
        "stdin": _truncate_text(stdin_input, 300),
        "overall_status": parsed["overall_status"],
        "run_status": parsed["run_status"],
        "return_code": parsed["return_code"],
    }

    if parsed["overall_status"] == "Success" and parsed["return_code"] in (0, None):
        if actual_output == expected_output:
            base_result.update({"status": "success", "passed": True})
            return base_result

        base_result.update(
            {
                "status": "wrong_answer",
                "passed": False,
                "expected": _truncate_text(expected_output, 500),
                "actual": _truncate_text(actual_output, 500),
                "stderr": _truncate_text(parsed["run_stderr"], 500),
                "case_insensitive_match": actual_output.lower() == expected_output.lower() and actual_output != expected_output,
                "ws_insensitive_match": _normalize_ws(actual_output) == _normalize_ws(expected_output) and actual_output != expected_output,
                "case_ws_insensitive_match": _normalize_ws(actual_output.lower()) == _normalize_ws(expected_output.lower())
                and actual_output != expected_output,
            }
        )
        return base_result

    if _is_sandbox_error(parsed):
        base_result.update(
            {
                "status": "sandbox_error",
                "passed": False,
                "stderr": _truncate_text(parsed["compile_stderr"] + parsed["run_stderr"], 500),
                "message": _truncate_text(parsed["message"], 300),
            }
        )
        return base_result

    if "SyntaxError" in parsed["compile_stderr"] or "SyntaxError" in parsed["run_stderr"]:
        base_result.update(
            {
                "status": "syntax_error",
                "passed": False,
                "stderr": _truncate_text(parsed["compile_stderr"] + parsed["run_stderr"], 500),
                "message": _truncate_text(parsed["message"], 300),
            }
        )
        return base_result

    if parsed["run_status"] == "TimeLimitExceeded" or "timelimitexceeded" in (parsed["run_stdout"] or "").lower():
        base_result.update({"status": "timeout", "passed": False, "stderr": _truncate_text(parsed["run_stderr"], 300)})
        return base_result

    base_result.update(
        {
            "status": "runtime_error",
            "passed": False,
            "stderr": _truncate_text(parsed["compile_stderr"] + parsed["run_stderr"], 500),
            "message": _truncate_text(parsed["message"], 300),
        }
    )
    return base_result


def _pick_primary_error(error_counts: Dict[str, int]) -> str:
    if not error_counts:
        return "success"
    priority = {
        "sandbox_error": 7,
        "syntax_error": 6,
        "runtime_error": 5,
        "timeout": 4,
        "wrong_answer": 3,
        "api_error": 2,
        "unknown": 1,
    }
    return max(error_counts.items(), key=lambda item: (item[1], priority.get(item[0], 0)))[0]


def _verify_codecontests_candidate(
    candidate: CandidateRecord,
    test_cases: Dict[str, Any],
    sandbox_endpoint: str,
    run_timeout_s: int,
    memory_limit_mb: int,
    limiter_budget: int,
    include_per_case_results: bool,
    autofix_codecontests_entrypoint: bool,
) -> VerificationSummary:
    tests = test_cases.get("tests", [])
    if not tests:
        return VerificationSummary(
            accepted=False,
            passed_tests=0,
            total_tests=0,
            pass_ratio_all=0.0,
            error_type="no_test_cases",
            invalid_for_rl=True,
            invalid_reason="no_test_cases",
            judge_time_s=0.0,
            extraction_status=candidate.extraction_status,
            per_case_results=[] if include_per_case_results else None,
        )

    if candidate.extraction_status != "ok":
        return _empty_or_invalid_summary(
            candidate,
            total_tests=len(tests),
            error_type=candidate.extraction_status,
            invalid_for_rl=False,
            invalid_reason="",
            include_per_case_results=include_per_case_results,
        )

    code = candidate.extracted_code
    if autofix_codecontests_entrypoint:
        code, _ = _codecontests_autofix_entrypoint(code)

    start_time = time.time()
    max_workers = max(1, min(len(tests), int(limiter_budget)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _verify_codecontests_testcase,
                code=code,
                testcase=testcase,
                test_idx=index,
                sandbox_endpoint=sandbox_endpoint,
                run_timeout_s=run_timeout_s,
                memory_limit_mb=memory_limit_mb,
                limiter_budget=limiter_budget,
            )
            for index, testcase in enumerate(tests)
        ]
        per_case_results = [future.result() for future in concurrent.futures.as_completed(futures)]

    per_case_results.sort(key=lambda item: item["test_idx"])
    passed_tests = sum(1 for result in per_case_results if result.get("passed"))
    total_tests = len(per_case_results)
    pass_ratio_all = passed_tests / total_tests if total_tests > 0 else 0.0
    accepted = passed_tests == total_tests and total_tests > 0
    error_counts: Dict[str, int] = {}
    for result in per_case_results:
        status = result.get("status", "unknown")
        if status == "success":
            continue
        error_counts[status] = error_counts.get(status, 0) + 1

    invalid_for_rl = error_counts.get("sandbox_error", 0) > 0 or error_counts.get("api_error", 0) > 0
    if error_counts.get("sandbox_error", 0) > 0:
        invalid_reason = "sandbox_error"
    elif error_counts.get("api_error", 0) > 0:
        invalid_reason = "api_error"
    else:
        invalid_reason = ""
    error_type = "success" if accepted else _pick_primary_error(error_counts)

    return VerificationSummary(
        accepted=accepted,
        passed_tests=passed_tests,
        total_tests=total_tests,
        pass_ratio_all=pass_ratio_all,
        error_type=error_type,
        invalid_for_rl=invalid_for_rl,
        invalid_reason=invalid_reason,
        judge_time_s=time.time() - start_time,
        extraction_status=candidate.extraction_status,
        per_case_results=per_case_results if include_per_case_results else None,
    )


def verify_candidate(
    *,
    candidate: CandidateRecord,
    problem_id: str,
    test_cases: Optional[Dict[str, Any]],
    sandbox_endpoint: str,
    run_timeout_s: int = 30,
    memory_limit_mb: int = 1024,
    limiter_budget: int = 8,
    include_per_case_results: bool = False,
    autofix_codecontests_entrypoint: bool = False,
) -> VerificationSummary:
    del problem_id  # reserved for future trace/log extensions

    if not test_cases:
        return VerificationSummary(
            accepted=False,
            passed_tests=0,
            total_tests=0,
            pass_ratio_all=0.0,
            error_type="no_test_cases",
            invalid_for_rl=True,
            invalid_reason="no_test_cases",
            judge_time_s=0.0,
            extraction_status=candidate.extraction_status,
            per_case_results=None,
        )

    test_type = test_cases.get("type", "unknown")
    if test_type == "humaneval":
        return _verify_humaneval_candidate(
            candidate=candidate,
            test_cases=test_cases,
            sandbox_endpoint=sandbox_endpoint,
            run_timeout_s=run_timeout_s,
            memory_limit_mb=memory_limit_mb,
            limiter_budget=limiter_budget,
        )
    if test_type == "mbpp":
        return _verify_mbpp_candidate(
            candidate=candidate,
            test_cases=test_cases,
            sandbox_endpoint=sandbox_endpoint,
            run_timeout_s=run_timeout_s,
            memory_limit_mb=memory_limit_mb,
            limiter_budget=limiter_budget,
        )
    if test_type == "codecontests":
        return _verify_codecontests_candidate(
            candidate=candidate,
            test_cases=test_cases,
            sandbox_endpoint=sandbox_endpoint,
            run_timeout_s=run_timeout_s,
            memory_limit_mb=memory_limit_mb,
            limiter_budget=limiter_budget,
            include_per_case_results=include_per_case_results,
            autofix_codecontests_entrypoint=autofix_codecontests_entrypoint,
        )

    return VerificationSummary(
        accepted=False,
        passed_tests=0,
        total_tests=0,
        pass_ratio_all=0.0,
        error_type="unknown_test_type",
        invalid_for_rl=True,
        invalid_reason="unknown_test_type",
        judge_time_s=0.0,
        extraction_status=candidate.extraction_status,
        per_case_results=None,
    )


def verify_candidate_batch(
    *,
    candidates: Iterable[CandidateRecord],
    ground_truths: Iterable[Optional[Dict[str, Any]]],
    sandbox_endpoint: str,
    run_timeout_s: int = 30,
    memory_limit_mb: int = 1024,
    limiter_budget: int = 8,
    include_per_case_results: bool = False,
    autofix_codecontests_entrypoint: bool = False,
) -> List[VerificationSummary]:
    candidate_list = list(candidates)
    ground_truth_list = list(ground_truths)
    max_workers = max(1, min(len(candidate_list), int(limiter_budget)))
    indexed_results: List[tuple[int, VerificationSummary]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, (candidate, ground_truth) in enumerate(zip(candidate_list, ground_truth_list, strict=True)):
            if isinstance(ground_truth, dict):
                problem_id = str(ground_truth.get("problem_id", ""))
                test_cases = ground_truth.get("test_cases")
            else:
                problem_id = ""
                test_cases = None
            futures.append(
                (
                    index,
                    executor.submit(
                        verify_candidate,
                        candidate=candidate,
                        problem_id=problem_id,
                        test_cases=test_cases,
                        sandbox_endpoint=sandbox_endpoint,
                        run_timeout_s=run_timeout_s,
                        memory_limit_mb=memory_limit_mb,
                        limiter_budget=limiter_budget,
                        include_per_case_results=include_per_case_results,
                        autofix_codecontests_entrypoint=autofix_codecontests_entrypoint,
                    ),
                )
            )

        for index, future in futures:
            indexed_results.append((index, future.result()))

    indexed_results.sort(key=lambda item: item[0])
    return [result for _, result in indexed_results]
