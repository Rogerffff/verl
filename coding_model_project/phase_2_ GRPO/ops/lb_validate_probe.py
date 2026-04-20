#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


PAYLOADS: list[dict[str, Any]] = [
    {
        "name": "sum_ints_from_stdin",
        "stdin": "10 20 30\n",
        "expected_stdout": "60\n",
        "request": {
            "language": "python",
            "run_timeout": 30,
            "memory_limit_MB": 1024,
            "code": (
                "import sys\n"
                "nums = [int(x) for x in sys.stdin.read().split()]\n"
                "print(sum(nums))\n"
            ),
        },
    },
    {
        "name": "reverse_lines_from_stdin",
        "stdin": "alpha\nbeta\ngamma\n",
        "expected_stdout": "gamma|beta|alpha\n",
        "request": {
            "language": "python",
            "run_timeout": 30,
            "memory_limit_MB": 1024,
            "code": (
                "import sys\n"
                "lines = [line.strip() for line in sys.stdin if line.strip()]\n"
                "print('|'.join(reversed(lines)))\n"
            ),
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw HTTP validation against SandboxFusion or an Nginx LB.")
    parser.add_argument("--endpoint", required=True, help="Base endpoint, e.g. http://localhost:8090")
    parser.add_argument("--requests", type=int, default=12, help="Total number of requests to send.")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of concurrent workers.")
    parser.add_argument("--timeout-seconds", type=float, default=80.0, help="End-to-end HTTP timeout for each request.")
    parser.add_argument("--jsonl-output", help="Optional JSONL file for per-request results.")
    parser.add_argument("--summary-output", help="Optional JSON file for summary results.")
    parser.add_argument(
        "--require-all-success",
        action="store_true",
        help="Exit non-zero if any request fails, returns SandboxError, or produces unexpected stdout.",
    )
    return parser.parse_args()


def build_url(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/run_code"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    lower_value = values[lower]
    upper_value = values[upper]
    weight = index - lower
    return lower_value + (upper_value - lower_value) * weight


def response_excerpt(text: str, limit: int = 400) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def attempt_request(endpoint: str, timeout_seconds: float, index: int) -> dict[str, Any]:
    payload = PAYLOADS[index % len(PAYLOADS)]
    request_body = dict(payload["request"])
    request_body["stdin"] = payload["stdin"]
    request_data = json.dumps(request_body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "grpo-lb-validate/1.0",
    }
    url = build_url(endpoint)

    started_at = time.time()
    base_result: dict[str, Any] = {
        "request_index": index,
        "payload_name": payload["name"],
        "expected_stdout": payload["expected_stdout"],
        "started_at": started_at,
        "ok": False,
        "http_status": None,
        "sandbox_status": "",
        "executor_pod_name": "",
        "upstream_addr": "",
        "request_id": "",
        "stdout": "",
        "unexpected_stdout": False,
        "exception_class": "",
        "exception_message": "",
        "response_excerpt": "",
    }

    request = urllib.request.Request(url, data=request_data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            latency_ms = (time.time() - started_at) * 1000
            parsed = json.loads(body)
            stdout = ((parsed.get("run_result") or {}).get("stdout") or "")
            unexpected_stdout = stdout != payload["expected_stdout"]
            base_result.update(
                {
                    "ok": response.status == 200 and parsed.get("status") != "SandboxError" and not unexpected_stdout,
                    "latency_ms": latency_ms,
                    "http_status": response.status,
                    "sandbox_status": parsed.get("status", ""),
                    "executor_pod_name": parsed.get("executor_pod_name", "") or "",
                    "upstream_addr": response.headers.get("X-Upstream-Addr", ""),
                    "request_id": response.headers.get("X-Request-Id", ""),
                    "stdout": stdout,
                    "unexpected_stdout": unexpected_stdout,
                    "response_excerpt": response_excerpt(body),
                }
            )
            return base_result
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        latency_ms = (time.time() - started_at) * 1000
        parsed: dict[str, Any] | None = None
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        base_result.update(
            {
                "latency_ms": latency_ms,
                "http_status": exc.code,
                "sandbox_status": (parsed or {}).get("status", ""),
                "executor_pod_name": (parsed or {}).get("executor_pod_name", "") or "",
                "upstream_addr": exc.headers.get("X-Upstream-Addr", ""),
                "request_id": exc.headers.get("X-Request-Id", ""),
                "exception_class": exc.__class__.__name__,
                "exception_message": str(exc),
                "response_excerpt": response_excerpt(body),
            }
        )
        return base_result
    except Exception as exc:  # pragma: no cover - defensive path for network/runtime issues
        latency_ms = (time.time() - started_at) * 1000
        base_result.update(
            {
                "latency_ms": latency_ms,
                "exception_class": exc.__class__.__name__,
                "exception_message": str(exc),
            }
        )
        return base_result


def summarize(results: list[dict[str, Any]], endpoint: str) -> dict[str, Any]:
    latencies = [result["latency_ms"] for result in results if result.get("latency_ms") is not None]
    http_failures = sum(1 for result in results if result.get("http_status") not in (200, None))
    sandbox_errors = sum(1 for result in results if result.get("sandbox_status") == "SandboxError")
    unexpected_stdout = sum(1 for result in results if result.get("unexpected_stdout"))
    exceptions = sum(1 for result in results if result.get("exception_class"))
    success_count = sum(1 for result in results if result.get("ok"))

    summary = {
        "endpoint": endpoint,
        "total_requests": len(results),
        "success_count": success_count,
        "failure_count": len(results) - success_count,
        "http_failure_count": http_failures,
        "sandbox_error_count": sandbox_errors,
        "unexpected_stdout_count": unexpected_stdout,
        "exception_count": exceptions,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "executor_pod_name_counts": dict(
            sorted(Counter(result.get("executor_pod_name", "") for result in results if result.get("executor_pod_name")).items())
        ),
        "upstream_addr_counts": dict(
            sorted(Counter(result.get("upstream_addr", "") for result in results if result.get("upstream_addr")).items())
        ),
        "http_status_counts": dict(sorted(Counter(str(result.get("http_status")) for result in results).items())),
        "sandbox_status_counts": dict(sorted(Counter(result.get("sandbox_status", "") for result in results).items())),
    }
    return summary


def write_jsonl(output_path: str | None, results: list[dict[str, Any]]) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=True) + "\n")


def write_summary(output_path: str | None, summary: dict[str, Any]) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    args = parse_args()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(attempt_request, args.endpoint, args.timeout_seconds, index)
            for index in range(args.requests)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    results.sort(key=lambda item: item["request_index"])
    summary = summarize(results, args.endpoint)

    write_jsonl(args.jsonl_output, results)
    write_summary(args.summary_output, summary)
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")

    if args.require_all_success and summary["failure_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
