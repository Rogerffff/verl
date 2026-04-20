from __future__ import annotations

"""离线评测与在线 GRPO reward 共用的共享判题模块。

并发上的要点：本模块**没有**使用单一全局 ThreadPoolExecutor，而是在需要时
在「批次级 / 候选级」与「测试用例级」分别开线程池；真正打到 Sandbox 的请求
则统一受 ``limiter_budget`` 对应的 **BoundedSemaphore** 全局限制。

也就是说：
1. ``verify_candidate_batch()`` 可并发判多个候选；
2. ``_verify_codecontests_candidate()`` 可对单个候选的多个 testcase 并发判题；
3. 每次实际调用 ``run_code()`` 前都必须先 acquire 同一套 limiter，因此
   在途 sandbox 请求数上限由 ``limiter_budget`` 决定。
"""

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
_SANDBOX_RR_LOCK = threading.Lock()
_SANDBOX_RR_INDEX: dict[tuple[str, ...], int] = {}


@dataclass
class CandidateRecord:
    """代码提取后的规范化模型输出。"""

    raw_completion: str
    extracted_code: str
    extraction_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationSummary:
    """评测与 RL 共用的判题结果契约。"""

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
    # 同一数值上限复用同一个信号量，使相同 budget 的调用共享全局并发上限。
    limit = max(1, int(limit))
    with _LIMITER_REGISTRY_LOCK:
        limiter = _LIMITER_REGISTRY.get(limit)
        if limiter is None:
            limiter = threading.BoundedSemaphore(limit)
            _LIMITER_REGISTRY[limit] = limiter
        return limiter


@contextmanager
def _acquire_limiter(limit: int):
    # 限制真实 sandbox RPC 并发；与由哪个线程池提交无关。
    limiter = _get_limiter(limit)
    limiter.acquire()
    try:
        yield
    finally:
        limiter.release()


def _lazy_import_sandbox():
    # 延迟导入 SandboxFusion，仅做提取或离线处理时可避免 upfront 导入开销。
    from sandbox_fusion import RunCodeRequest, run_code

    return {
        "RunCodeRequest": RunCodeRequest,
        "run_code": run_code,
    }


def _normalize_sandbox_endpoints(sandbox_endpoint: str) -> List[str]:
    endpoints = [part.strip() for part in str(sandbox_endpoint or "").split(",") if part.strip()]
    if not endpoints:
        raise ValueError("sandbox_endpoint is empty")
    return endpoints


def primary_sandbox_endpoint(sandbox_endpoint: str) -> str:
    return _normalize_sandbox_endpoints(sandbox_endpoint)[0]


def _choose_sandbox_endpoint_rr(sandbox_endpoint: str) -> str:
    endpoints = tuple(_normalize_sandbox_endpoints(sandbox_endpoint))
    if len(endpoints) == 1:
        return endpoints[0]

    with _SANDBOX_RR_LOCK:
        idx = _SANDBOX_RR_INDEX.get(endpoints, 0)
        chosen = endpoints[idx % len(endpoints)]
        _SANDBOX_RR_INDEX[endpoints] = (idx + 1) % len(endpoints)
    return chosen


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
    # 将 SDK 返回对象压成普通 dict，下游分类逻辑不依赖 SDK 内部结构。
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
    # 提取策略尽量宽松：
    # 1) 优先 <code>...</code>
    # 2) 其次 markdown 代码块
    # 3) 有标记但解析失败则 extraction_failure
    # 4) 最后对 RL 裸输出用「像 Python」启发式兜底
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
    # 部分代码定义了 solve() 但未调用；stdin/stdout 型 CodeContests 可选项补 main 入口。
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
    # 提取失败或无代码时仍返回结构完整的 summary，评测与 RL 侧契约一致。
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
    # 唯一实际调用 SandboxFusion run_code() 的入口；上层并发最终都汇聚到此信号量。
    sandbox = _lazy_import_sandbox()
    chosen_endpoint = _choose_sandbox_endpoint_rr(sandbox_endpoint)
    request_kwargs = {
        "code": code,
        "language": "python",
        "run_timeout": run_timeout_s,
        "memory_limit_MB": memory_limit_mb,
    }
    if stdin is not None:
        request_kwargs["stdin"] = stdin

    with _acquire_limiter(limiter_budget):
        return sandbox["run_code"](sandbox["RunCodeRequest"](**request_kwargs), endpoint=chosen_endpoint)


def _is_sandbox_error(parsed: Dict[str, Any]) -> bool:
    return parsed.get("overall_status") == "SandboxError"


def _classify_single_run(parsed: Dict[str, Any]) -> tuple[str, bool, str]:
    # invalid_for_rl 留给基础设施/服务异常；模型侧错误（语法/运行/超时/WA）仍应参与 reward 学习。
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
    # HumanEval：候选代码与隐藏测试拼成单文件，一次 sandbox 运行，全对或全错。
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
    # MBPP：与 HumanEval 相同，单文件拼测、单次 sandbox、二元通过/失败。
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
    # CodeContests：一个 testcase 对应一次 sandbox 请求；粒度可并行多测，仍受共享 limiter 约束。
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
    # 汇总只暴露一个顶层 error_type：取次数占优者，平局时按固定优先级打破。
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
    # CodeContests：逐 testcase 独立执行再聚合，得到 passed_tests / total_tests / pass_ratio_all（稠密信号）。
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
    # 内层线程池：单候选内 testcase 级调度；可提交 len(tests) 个任务，真实 sandbox 并发仍由 _run_code_request 信号量限制。
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
        # 用 as_completed 尽快收集结果，随后按 test_idx 排序，与下游日志下标一致。
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
    # 按 test_cases.type 分派，评测与 RL 共用同一入口。
    del problem_id  # 预留：后续 trace / 日志扩展

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
    # 外层线程池：候选级并发；CodeContests 内层可能再开 testcase 池，但真实 RPC 仍受 limiter_budget 全局信号量限制。
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
