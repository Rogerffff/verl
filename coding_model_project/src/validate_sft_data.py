#!/usr/bin/env python3
"""
SFT 训练数据验证脚本
====================

对 sft_with_testcases.jsonl 中的解答代码进行沙盒执行验证，
过滤出所有测试用例全部通过的记录，确保 SFT 数据质量。

环境要求：
- conda 环境 `sandbox`（含 sandbox_fusion 0.3.7）
- Docker 运行 SandboxFusion：
  docker run -it --rm --privileged -p 8080:8080 volcengine/sandbox-fusion:server-20250609

用法：
  conda activate sandbox
  cd coding_model_project
  python3 src/validate_sft_data.py --preflight-only   # 先预检
  python3 src/validate_sft_data.py                     # 全量验证
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from sandbox_fusion import (
    RunCodeRequest,
    RunCodeResponse,
    RunStatus,
    set_endpoint,
)
from sandbox_fusion.async_client import run_code as run_code_async
from sandbox_fusion.models import CommandRunStatus

# =============================================================================
# 常量
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent  # coding_model_project/
DATA_DIR = PROJECT_DIR / "phase_1_ SFT" / "bee_hq_python_deduped_filtered"
DEFAULT_INPUT = DATA_DIR / "sft_with_testcases.jsonl"
DEFAULT_SANDBOX_URL = "http://localhost:8080"
DEFAULT_CONCURRENCY = 12
DEFAULT_RUN_TIMEOUT = 30
CLIENT_TIMEOUT = 60.0  # 2x run_timeout, 防止 aiohttp 挂起

SOURCE_NAMES = {2: "Codeforces", 6: "AtCoder", 7: "Aizu"}


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class ValidationResult:
    """单题验证结果"""
    problem_id: str
    source: int
    difficulty: int
    accepted: bool
    pass_ratio: float
    total_tests: int
    passed_tests: int
    error_type: str  # success, wrong_answer, runtime_error, timeout, syntax_error, api_error
    judge_time: float
    first_failure: Optional[Dict[str, Any]]


# =============================================================================
# 预检
# =============================================================================

def check_sandbox_connection(sandbox_url: str) -> bool:
    """检查 sandbox 是否可达"""
    from sandbox_fusion import run_code
    print("[预检 1/3] 检查 sandbox 连接...")
    try:
        result = run_code(RunCodeRequest(
            code='print("hello")',
            language="python",
            run_timeout=10,
        ))
        status = str(result.status)
        stdout = (result.run_result.stdout or "") if result.run_result else ""
        if "Success" in status and "hello" in stdout:
            print(f"  OK - sandbox 正常响应: stdout={stdout.strip()!r}")
            return True
        else:
            print(f"  FAIL - 异常响应: status={status}, stdout={stdout.strip()!r}")
            return False
    except Exception as e:
        print(f"  FAIL - 无法连接 sandbox: {e}")
        print(f"  请确保 Docker 正在运行：")
        print(f"    docker run -it --rm --privileged -p 8080:8080 volcengine/sandbox-fusion:server-20250609")
        return False


def check_stdin_support(sandbox_url: str) -> bool:
    """检查 stdin 功能是否正常（检测 StreamWriter bug）"""
    from sandbox_fusion import run_code
    print("[预检 2/3] 检查 stdin 功能...")
    try:
        result = run_code(RunCodeRequest(
            code='x = input()\nprint(f"got:{x}")',
            language="python",
            run_timeout=10,
            stdin="hello\n",
        ))
        status = str(result.status)
        stdout = (result.run_result.stdout or "") if result.run_result else ""
        stderr = (result.run_result.stderr or "") if result.run_result else ""

        if "Success" in status and "got:hello" in stdout:
            print(f"  OK - stdin 正常工作: stdout={stdout.strip()!r}")
            return True

        # 检查是否是 StreamWriter bug
        if "TimeLimitExceeded" in str(getattr(result.run_result, 'status', '')):
            print(f"  FATAL - stdin 超时，可能存在 StreamWriter bug!")
            print(f"  修复方法：进入 Docker 容器修改 sandbox/runners/base.py")
            print(f"    将 p.stdin.flush() 改为 await p.stdin.drain()")
            print(f"    将 p.stdin.close() 后添加 await p.stdin.wait_closed()")
            return False

        if "flush" in stderr.lower() or "StreamWriter" in stderr:
            print(f"  FATAL - 检测到 StreamWriter flush bug!")
            print(f"  stderr: {stderr[:200]}")
            return False

        print(f"  FAIL - stdin 输出异常: status={status}, stdout={stdout.strip()!r}, stderr={stderr[:100]!r}")
        return False

    except Exception as e:
        error_msg = str(e)
        if "flush" in error_msg or "StreamWriter" in error_msg:
            print(f"  FATAL - StreamWriter bug: {error_msg}")
        else:
            print(f"  FAIL - stdin 检查异常: {e}")
        return False


def smoke_test(records: List[Dict], sandbox_url: str, run_timeout: int) -> bool:
    """小批量烟雾测试：每个来源取一条数据验证"""
    print("[预检 3/3] 小批量烟雾测试...")

    # 每个来源取一条
    samples = {}
    for r in records:
        src = r["source"]
        if src not in samples:
            samples[src] = r
        if len(samples) >= 3:
            break

    from sandbox_fusion import run_code

    all_failed_same = True
    common_error = None
    results_summary = []

    for src, record in samples.items():
        src_name = SOURCE_NAMES.get(src, f"source_{src}")
        problem_id = record["problem_id"]
        code = record["solution"]
        tests = record["test_cases"]["tests"]
        test_count = len(tests)

        # 只测试前 3 个测试用例（节省时间）
        test_subset = tests[:3]
        passed = 0
        error_type = "success"

        for tc in test_subset:
            stdin_input = tc.get("input", "")
            expected_output = tc.get("output", "").strip()

            try:
                result = run_code(RunCodeRequest(
                    code=code,
                    language="python",
                    run_timeout=run_timeout,
                    stdin=stdin_input,
                ))
                status = str(result.status)
                stdout = (result.run_result.stdout or "") if result.run_result else ""
                stderr = (result.run_result.stderr or "") if result.run_result else ""
                actual = stdout.strip()

                if "Success" in status and actual == expected_output:
                    passed += 1
                elif "Success" in status:
                    error_type = "wrong_answer"
                elif result.run_result and result.run_result.status == CommandRunStatus.TimeLimitExceeded:
                    error_type = "timeout"
                    break
                else:
                    error_type = "runtime_error"
            except Exception as e:
                error_type = "api_error"
                break

        status_str = f"{passed}/{len(test_subset)} passed"
        if passed == len(test_subset):
            error_type = "success"
            all_failed_same = False

        print(f"  [{src_name}] {problem_id}: {status_str} (error_type={error_type}, total_tests={test_count})")
        results_summary.append((src_name, error_type))

        if common_error is None:
            common_error = error_type
        elif common_error != error_type:
            all_failed_same = False

    # 如果所有样本都因同一错误失败，可能是 sandbox 问题
    if all_failed_same and common_error and common_error != "success":
        print(f"\n  WARNING: 所有样本都因 '{common_error}' 失败，可能是 sandbox 配置问题")
        return False

    print(f"  烟雾测试完成")
    return True


def run_preflight_checks(sandbox_url: str, records: List[Dict], run_timeout: int) -> bool:
    """运行所有预检"""
    print("=" * 60)
    print("预检阶段")
    print("=" * 60)

    if not check_sandbox_connection(sandbox_url):
        return False

    if not check_stdin_support(sandbox_url):
        return False

    if not smoke_test(records, sandbox_url, run_timeout):
        return False

    print("\n预检全部通过!\n")
    return True


# =============================================================================
# 核心验证
# =============================================================================

async def validate_single_problem(
    record: Dict,
    semaphore: asyncio.Semaphore,
    sandbox_url: str,
    run_timeout: int = 30,
) -> ValidationResult:
    """异步验证单题的所有测试用例"""
    async with semaphore:
        problem_id = record["problem_id"]
        code = record["solution"]
        tests = record["test_cases"]["tests"]
        source = record["source"]
        difficulty = record.get("difficulty", 0)

        passed = 0
        total = len(tests)
        error_type = "success"
        first_failure = None
        start_time = time.time()

        if total == 0:
            return ValidationResult(
                problem_id=problem_id, source=source, difficulty=difficulty,
                accepted=False, pass_ratio=0.0, total_tests=0, passed_tests=0,
                error_type="no_test_cases", judge_time=0.0, first_failure=None,
            )

        for i, tc in enumerate(tests):
            stdin_input = tc.get("input", "")
            expected_output = tc.get("output", "").strip()

            try:
                result = await run_code_async(
                    RunCodeRequest(
                        code=code,
                        language="python",
                        run_timeout=run_timeout,
                        stdin=stdin_input,
                    ),
                    endpoint=sandbox_url,
                    client_timeout=CLIENT_TIMEOUT,
                )

                run_result = result.run_result
                stdout = (run_result.stdout or "") if run_result else ""
                stderr = (run_result.stderr or "") if run_result else ""
                actual_output = stdout.strip()

                if result.status == RunStatus.Success:
                    if actual_output == expected_output:
                        passed += 1
                    else:
                        if error_type == "success":
                            error_type = "wrong_answer"
                        if first_failure is None:
                            first_failure = {
                                "test_idx": i,
                                "status": "wrong_answer",
                                "expected": expected_output[:100],
                                "actual": actual_output[:100],
                            }
                elif run_result and run_result.status == CommandRunStatus.TimeLimitExceeded:
                    error_type = "timeout"
                    if first_failure is None:
                        first_failure = {"test_idx": i, "status": "TimeLimitExceeded"}
                    break  # 后续测试大概率也超时
                elif "SyntaxError" in (stdout + stderr):
                    error_type = "syntax_error"
                    if first_failure is None:
                        first_failure = {"test_idx": i, "status": "syntax_error", "stderr": stderr[:200]}
                    break  # 语法错误无需继续
                else:
                    if error_type == "success":
                        error_type = "runtime_error"
                    if first_failure is None:
                        first_failure = {
                            "test_idx": i,
                            "status": str(result.status),
                            "stderr": stderr[:200],
                        }

            except Exception as e:
                error_type = "api_error"
                if first_failure is None:
                    first_failure = {"test_idx": i, "error": str(e)[:200]}
                break

        pass_ratio = passed / total if total > 0 else 0.0
        accepted = (passed == total)
        if accepted:
            error_type = "success"

        return ValidationResult(
            problem_id=problem_id,
            source=source,
            difficulty=difficulty,
            accepted=accepted,
            pass_ratio=pass_ratio,
            total_tests=total,
            passed_tests=passed,
            error_type=error_type,
            judge_time=time.time() - start_time,
            first_failure=first_failure,
        )


def load_completed_results(results_path: Path) -> Dict[str, Dict]:
    """加载已完成的结果（用于断点续传）"""
    completed = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    completed[data["problem_id"]] = data
    return completed


async def validate_all(
    records: List[Dict],
    results_path: Path,
    sandbox_url: str,
    concurrency: int = 12,
    run_timeout: int = 30,
    force: bool = False,
) -> List[Dict]:
    """并发验证所有记录，增量保存结果"""

    # 加载已完成的结果
    if force and results_path.exists():
        results_path.unlink()
        completed = {}
    else:
        completed = load_completed_results(results_path)

    remaining = [r for r in records if r["problem_id"] not in completed]

    if not remaining:
        print("所有题目已验证完毕。使用 --force 可重新验证。")
        return list(completed.values())

    total = len(remaining)
    print(f"待验证: {total} 题（已完成: {len(completed)}）")
    print(f"并发数: {concurrency}, run_timeout: {run_timeout}s")
    print(f"预估 API 调用: ~{sum(len(r['test_cases']['tests']) for r in remaining):,} 次")
    print()

    semaphore = asyncio.Semaphore(concurrency)
    all_results = list(completed.values())
    done_count = 0
    pass_count = sum(1 for r in completed.values() if r.get("accepted", False))
    start_time = time.time()

    # 创建所有任务
    tasks = [
        asyncio.create_task(
            validate_single_problem(record, semaphore, sandbox_url, run_timeout)
        )
        for record in remaining
    ]

    # 使用 as_completed 获取进度
    with open(results_path, 'a') as f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            result_dict = asdict(result)
            all_results.append(result_dict)

            # 增量保存
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            f.flush()

            done_count += 1
            if result.accepted:
                pass_count += 1

            # 进度输出
            if done_count % 50 == 0 or done_count == total or done_count <= 5:
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / rate if rate > 0 else 0
                current_pass_rate = pass_count / (len(completed) + done_count)
                print(
                    f"  [{done_count}/{total}] "
                    f"pass_rate={current_pass_rate:.1%} "
                    f"rate={rate:.2f}/s "
                    f"ETA={eta/60:.0f}min "
                    f"last={result.problem_id} ({'PASS' if result.accepted else result.error_type})"
                )

    elapsed_total = time.time() - start_time
    print(f"\n验证完成: {done_count} 题, 耗时 {elapsed_total/60:.1f} 分钟")
    return all_results


# =============================================================================
# 报告与输出
# =============================================================================

def generate_report(results: List[Dict]) -> Dict:
    """生成汇总报告"""
    total = len(results)
    if total == 0:
        return {"error": "no results"}

    accepted = sum(1 for r in results if r["accepted"])
    error_dist = Counter(r["error_type"] for r in results)

    # 按来源分组
    by_source = {}
    for src_id, src_name in SOURCE_NAMES.items():
        src_results = [r for r in results if r["source"] == src_id]
        if src_results:
            src_accepted = sum(1 for r in src_results if r["accepted"])
            by_source[src_name] = {
                "total": len(src_results),
                "accepted": src_accepted,
                "pass_rate": round(src_accepted / len(src_results), 4),
                "avg_pass_ratio": round(sum(r["pass_ratio"] for r in src_results) / len(src_results), 4),
                "error_distribution": dict(Counter(r["error_type"] for r in src_results)),
            }

    # 按难度分组
    by_difficulty = {}
    for r in results:
        d = r["difficulty"]
        bucket = f"d{d:02d}"
        if bucket not in by_difficulty:
            by_difficulty[bucket] = {"total": 0, "accepted": 0}
        by_difficulty[bucket]["total"] += 1
        if r["accepted"]:
            by_difficulty[bucket]["accepted"] += 1

    for bucket in by_difficulty:
        t = by_difficulty[bucket]["total"]
        a = by_difficulty[bucket]["accepted"]
        by_difficulty[bucket]["pass_rate"] = round(a / t, 4) if t > 0 else 0

    # 最差 20 题
    sorted_results = sorted(results, key=lambda r: r["pass_ratio"])
    worst_20 = [
        {
            "problem_id": r["problem_id"],
            "source": SOURCE_NAMES.get(r["source"], str(r["source"])),
            "difficulty": r["difficulty"],
            "pass_ratio": r["pass_ratio"],
            "passed_tests": r["passed_tests"],
            "total_tests": r["total_tests"],
            "error_type": r["error_type"],
            "first_failure": r.get("first_failure"),
        }
        for r in sorted_results[:20]
    ]

    report = {
        "summary": {
            "total_problems": total,
            "accepted": accepted,
            "rejected": total - accepted,
            "overall_pass_rate": round(accepted / total, 4),
            "avg_pass_ratio": round(sum(r["pass_ratio"] for r in results) / total, 4),
            "total_judge_time_minutes": round(sum(r["judge_time"] for r in results) / 60, 1),
        },
        "error_distribution": dict(error_dist),
        "by_source": by_source,
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "worst_20_problems": worst_20,
    }

    return report


def write_validated_dataset(
    records: List[Dict],
    results: List[Dict],
    output_path: Path,
):
    """写入通过验证的数据集"""
    accepted_ids = {r["problem_id"] for r in results if r["accepted"]}

    count = 0
    with open(output_path, 'w') as f:
        for record in records:
            if record["problem_id"] in accepted_ids:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"已写入 {count} 条通过验证的记录到 {output_path}")
    return count


# =============================================================================
# 主流程
# =============================================================================

def load_records(input_path: Path) -> List[Dict]:
    """加载数据"""
    records = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="验证 SFT 训练数据：通过 SandboxFusion 执行解答代码并比对测试用例"
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="输入数据文件路径")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR),
                        help="输出目录")
    parser.add_argument("--sandbox-url", type=str, default=DEFAULT_SANDBOX_URL,
                        help="SandboxFusion 服务地址")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="并发数")
    parser.add_argument("--run-timeout", type=int, default=DEFAULT_RUN_TIMEOUT,
                        help="每个测试用例的执行超时（秒）")
    parser.add_argument("--force", action="store_true",
                        help="忽略已有结果，重新验证所有题目")
    parser.add_argument("--preflight-only", action="store_true",
                        help="仅运行预检（连接+stdin+烟雾测试）")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="跳过预检直接验证")
    parser.add_argument("--report-only", action="store_true",
                        help="仅从已有结果生成报告")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    results_path = output_dir / "validation_results.jsonl"
    report_path = output_dir / "validation_report.json"
    validated_path = output_dir / "sft_validated.jsonl"

    # 设置 sandbox endpoint
    set_endpoint(args.sandbox_url)

    # 加载数据
    print(f"加载数据: {input_path}")
    records = load_records(input_path)
    print(f"共 {len(records)} 条记录")
    total_tests = sum(len(r["test_cases"]["tests"]) for r in records)
    print(f"总测试用例数: {total_tests:,}")
    print()

    # 仅生成报告模式
    if args.report_only:
        if not results_path.exists():
            print(f"错误: 结果文件不存在: {results_path}")
            sys.exit(1)
        completed = load_completed_results(results_path)
        results = list(completed.values())
        print(f"从已有结果加载 {len(results)} 条记录")
    else:
        # 预检
        if not args.skip_preflight:
            if not run_preflight_checks(args.sandbox_url, records, args.run_timeout):
                print("\n预检失败，请修复上述问题后重试。")
                sys.exit(1)

            if args.preflight_only:
                print("预检完成，退出。")
                return

        # 全量验证
        print("=" * 60)
        print("全量验证阶段")
        print("=" * 60)
        results = asyncio.run(validate_all(
            records=records,
            results_path=results_path,
            sandbox_url=args.sandbox_url,
            concurrency=args.concurrency,
            run_timeout=args.run_timeout,
            force=args.force,
        ))

    # 生成报告
    print("\n" + "=" * 60)
    print("生成报告")
    print("=" * 60)
    report = generate_report(results)

    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已写入: {report_path}")

    # 打印摘要
    s = report["summary"]
    print(f"\n--- 验证摘要 ---")
    print(f"总题数: {s['total_problems']}")
    print(f"通过:   {s['accepted']} ({s['overall_pass_rate']:.1%})")
    print(f"失败:   {s['rejected']}")
    print(f"平均通过比: {s['avg_pass_ratio']:.1%}")
    print(f"\n错误分布:")
    for err_type, count in sorted(report["error_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {err_type}: {count} ({count/s['total_problems']:.1%})")
    print(f"\n按来源:")
    for src_name, stats in report["by_source"].items():
        print(f"  {src_name}: {stats['accepted']}/{stats['total']} ({stats['pass_rate']:.1%})")

    # 写入过滤后的数据集
    write_validated_dataset(records, results, validated_path)

    print(f"\n完成! 验证耗时: {s['total_judge_time_minutes']:.1f} 分钟")


if __name__ == "__main__":
    main()
