#!/usr/bin/env python3
"""
SandboxFusion 评测流程测试脚本
================================

目的：
- 跳过 vLLM 采样阶段，直接使用参考解答或手写代码
- 验证 SandboxFusion 评测流程是否正确
- 确认能否获取预期的指标

环境要求：
==========
本脚本需要使用 conda 环境 `sandbox`，该环境包含 sandbox_fusion SDK。

conda 环境信息：
- 环境名称: sandbox
- Python 版本: 3.12.12
- Python 路径: /Users/xiaohui/miniconda3/envs/sandbox/bin/python3
- 已安装包: sandbox-fusion 0.3.7

Python 绑定问题（重要）：
========================
在某些 shell 配置中，`python` 命令可能被 alias 到系统 Python：
    $ python --version
    python: aliased to /usr/bin/python3  # 系统 Python 3.9.6

此时即使激活了 conda 环境，`python` 仍然指向系统 Python，导致：
    ModuleNotFoundError: No module named 'sandbox_fusion'

解决方法：
1. 使用 `python3` 代替 `python`（推荐）
2. 或者取消 alias: `unalias python`
3. 或者直接使用完整路径: `/Users/xiaohui/miniconda3/envs/sandbox/bin/python`

使用方法：
==========
1. 启动 SandboxFusion Docker:
   docker run -it --rm --privileged -p 8080:8080 volcengine/sandbox-fusion:server-20250609

2. 激活 conda 环境并运行测试脚本:
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate sandbox
   cd verl/coding_model_project
   python3 src/temp/test_sandbox_eval.py  # 注意使用 python3

3. 查看输出指标

验证环境配置正确：
   which python3  # 应输出: /Users/xiaohui/miniconda3/envs/sandbox/bin/python3
   python3 -c "import sandbox_fusion; print('OK')"  # 应输出: OK
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

# =============================================================================
# SandboxFusion SDK 导入
# =============================================================================
try:
    from sandbox_fusion import (
        run_code,
        submit_safe,
        RunCodeRequest,
        SubmitRequest,
        TestConfig,
        set_endpoint,
    )
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    print("Error: sandbox_fusion not installed. Run: pip install sandbox-fusion")
    exit(1)


# =============================================================================
# 配置
# =============================================================================
SANDBOX_URL = "http://localhost:8080"
# 注意：脚本位于 src/temp/ 目录，数据在 data/ 目录（与 src 同级）
# Path(__file__).parent = src/temp/
# Path(__file__).parent.parent = src/
# Path(__file__).parent.parent.parent = coding_model_project/
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


@dataclass
class TestResult:
    """测试结果"""
    problem_id: str
    dataset: str
    accepted: bool
    pass_ratio: float
    error_type: str
    judge_time: float
    details: Dict[str, Any]


# =============================================================================
# 测试用例加载
# =============================================================================

def load_sample_problems(dataset_key: str, n_samples: int = 3) -> List[Dict]:
    """
    从 raw 数据加载几个样本问题（包含测试用例和参考解答）
    """
    raw_path = RAW_DIR / f"{dataset_key}_raw.jsonl"

    if not raw_path.exists():
        print(f"  Warning: {raw_path} not found")
        return []

    problems = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            problems.append(json.loads(line))

    return problems


# =============================================================================
# 辅助函数：解析 run_code 返回结果
# =============================================================================

def parse_run_code_result(result) -> Tuple[str, str]:
    """
    解析 SandboxFusion run_code API 的返回结果

    run_code 返回的对象结构：
    - result.status: RunStatus 枚举（如 RunStatus.Success, RunStatus.Failed）
    - result.run_result: 对象，包含：
      - status: CommandRunStatus 枚举（如 CommandRunStatus.Finished）
      - stdout: 标准输出字符串
      - stderr: 标准错误字符串

    注意：
    - RunStatus.Success 表示代码执行成功（无错误）
    - RunStatus.Failed 表示代码执行失败（有错误）
    - CommandRunStatus.Finished 只表示命令执行完成，不表示成功

    Returns:
        (status_str, output_str)
        - status_str: 顶层 RunStatus 字符串（如 "Success", "Failed"）
        - output_str: 输出内容（stdout + stderr）
    """
    # 获取顶层状态（枚举转字符串）- 这是判断成功/失败的关键
    status = str(getattr(result, 'status', 'unknown'))

    # 获取 run_result 对象
    run_result = getattr(result, 'run_result', None)

    if run_result is not None and hasattr(run_result, 'stdout'):
        # run_result 是结构化对象
        stdout = run_result.stdout or ""
        stderr = run_result.stderr or ""
        output = stdout + stderr
        # 注意：不再用 inner_status 覆盖 status
        # 因为 CommandRunStatus.Finished 不代表成功，RunStatus 才是
    else:
        # 回退：直接转字符串
        output = str(run_result) if run_result else ""

    return status, output


# =============================================================================
# 评测函数（复用 phase0_eval.py 的逻辑）
# =============================================================================

def evaluate_humaneval(code: str, test_cases: Dict, problem_id: str) -> TestResult:
    """评测 HumanEval 格式"""
    start_time = time.time()

    test_code = test_cases.get("test_code", "")
    entry_point = test_cases.get("entry_point", "")

    # 组装完整代码
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"

    try:
        result = run_code(RunCodeRequest(
            code=full_code,
            language="python",
            run_timeout=10,
        ))

        # 使用辅助函数解析结果
        status, output = parse_run_code_result(result)

        # 判断是否通过（检查 Success 或 Finished）
        accepted = "Success" in status

        if accepted:
            error_type = "success"
        elif "SyntaxError" in output:
            error_type = "syntax_error"
        elif "timeout" in status.lower() or "timeout" in output.lower():
            error_type = "timeout"
        elif "Error" in output:
            error_type = "runtime_error"
        else:
            error_type = "wrong_answer"

        return TestResult(
            problem_id=problem_id,
            dataset="humaneval",
            accepted=accepted,
            pass_ratio=1.0 if accepted else 0.0,
            error_type=error_type,
            judge_time=time.time() - start_time,
            details={"status": status, "output": output[:300]},
        )

    except Exception as e:
        return TestResult(
            problem_id=problem_id,
            dataset="humaneval",
            accepted=False,
            pass_ratio=0.0,
            error_type="api_error",
            judge_time=time.time() - start_time,
            details={"error": str(e)},
        )


def evaluate_mbpp(code: str, test_cases: Dict, problem_id: str) -> TestResult:
    """评测 MBPP 格式"""
    start_time = time.time()

    test_list = test_cases.get("test_list", [])
    test_setup_code = test_cases.get("test_setup_code", "")

    if not test_list:
        return TestResult(
            problem_id=problem_id,
            dataset="mbpp",
            accepted=False,
            pass_ratio=0.0,
            error_type="no_test_cases",
            judge_time=time.time() - start_time,
            details={"error": "No test cases"},
        )

    # 组装完整代码
    test_code = "\n".join(test_list)
    full_code = f"{test_setup_code}\n\n{code}\n\n{test_code}"

    try:
        result = run_code(RunCodeRequest(
            code=full_code,
            language="python",
            run_timeout=10,
        ))

        # 使用辅助函数解析结果
        status, output = parse_run_code_result(result)

        # 判断是否通过（检查 Success 或 Finished）
        accepted = "Success" in status

        if accepted:
            error_type = "success"
        elif "SyntaxError" in output:
            error_type = "syntax_error"
        elif "timeout" in status.lower():
            error_type = "timeout"
        elif "Error" in output:
            error_type = "runtime_error"
        else:
            error_type = "wrong_answer"

        return TestResult(
            problem_id=problem_id,
            dataset="mbpp",
            accepted=accepted,
            pass_ratio=1.0 if accepted else 0.0,
            error_type=error_type,
            judge_time=time.time() - start_time,
            details={"status": status, "output": output[:300], "test_count": len(test_list)},
        )

    except Exception as e:
        return TestResult(
            problem_id=problem_id,
            dataset="mbpp",
            accepted=False,
            pass_ratio=0.0,
            error_type="api_error",
            judge_time=time.time() - start_time,
            details={"error": str(e)},
        )


def evaluate_codecontests(code: str, test_cases: Dict, problem_id: str) -> TestResult:
    """评测 CodeContests 格式（stdin/stdout）"""
    start_time = time.time()

    tests = test_cases.get("tests", [])

    if not tests:
        return TestResult(
            problem_id=problem_id,
            dataset="codecontests",
            accepted=False,
            pass_ratio=0.0,
            error_type="no_test_cases",
            judge_time=time.time() - start_time,
            details={"error": "No test cases"},
        )

    passed = 0
    total = len(tests)
    error_type = "success"
    last_error = ""

    for tc in tests:
        stdin_input = tc.get("input", "")
        expected_output = tc.get("output", "").strip()

        try:
            result = run_code(RunCodeRequest(
                code=code,
                language="python",
                run_timeout=10,
                stdin=stdin_input,
            ))

            # 使用辅助函数解析结果
            status, output = parse_run_code_result(result)
            actual_output = output.strip()

            # 判断是否通过（检查 Success 或 Finished）
            if "Success" in status:
                if actual_output == expected_output:
                    passed += 1
                else:
                    error_type = "wrong_answer"
                    last_error = f"Expected: {expected_output[:50]}, Got: {actual_output[:50]}"
            elif "SyntaxError" in actual_output:
                error_type = "syntax_error"
                last_error = actual_output[:100]
                break
            elif "timeout" in status.lower():
                error_type = "timeout"
                break
            else:
                error_type = "runtime_error"
                last_error = actual_output[:100]

        except Exception as e:
            error_type = "api_error"
            last_error = str(e)
            break

    pass_ratio = passed / total if total > 0 else 0.0
    accepted = (passed == total)

    if accepted:
        error_type = "success"

    return TestResult(
        problem_id=problem_id,
        dataset="codecontests",
        accepted=accepted,
        pass_ratio=pass_ratio,
        error_type=error_type,
        judge_time=time.time() - start_time,
        details={"passed": passed, "total": total, "last_error": last_error},
    )


def evaluate_problem(problem: Dict, dataset_key: str) -> TestResult:
    """根据数据集类型选择评测函数"""
    test_cases = problem.get("test_cases", {})
    test_type = test_cases.get("type", "unknown")
    problem_id = problem["problem_id"]

    # 使用参考解答作为"模型输出"
    # 注意：HumanEval 的 canonical_solution 只是函数体，需要与 prompt 合并
    if test_type == "humaneval" and "canonical_solution" in problem:
        # HumanEval: prompt (函数签名+docstring) + canonical_solution (函数体)
        code = problem["prompt"] + problem["canonical_solution"]
    elif "canonical_solution" in problem:
        code = problem["canonical_solution"]
    elif "solutions" in problem and problem["solutions"]:
        code = problem["solutions"][0]
    else:
        # 没有参考解答，返回失败
        return TestResult(
            problem_id=problem_id,
            dataset=dataset_key,
            accepted=False,
            pass_ratio=0.0,
            error_type="no_solution",
            judge_time=0.0,
            details={"error": "No canonical solution available"},
        )

    # 根据测试用例类型选择评测函数
    if test_type == "humaneval":
        return evaluate_humaneval(code, test_cases, problem_id)
    elif test_type == "mbpp":
        return evaluate_mbpp(code, test_cases, problem_id)
    elif test_type == "codecontests":
        return evaluate_codecontests(code, test_cases, problem_id)
    else:
        return TestResult(
            problem_id=problem_id,
            dataset=dataset_key,
            accepted=False,
            pass_ratio=0.0,
            error_type="unknown_test_type",
            judge_time=0.0,
            details={"error": f"Unknown test type: {test_type}"},
        )


# =============================================================================
# 指标统计
# =============================================================================

def calculate_metrics(results: List[TestResult]) -> Dict[str, Any]:
    """计算评测指标"""
    if not results:
        return {}

    total = len(results)
    accepted_count = sum(1 for r in results if r.accepted)
    pass_ratios = [r.pass_ratio for r in results]
    judge_times = [r.judge_time for r in results]

    # 错误分布
    error_counts = {}
    for r in results:
        error_counts[r.error_type] = error_counts.get(r.error_type, 0) + 1

    return {
        "total_problems": total,
        "accepted_at_1": accepted_count / total,
        "pass_ratio_mean": sum(pass_ratios) / total,
        "avg_judge_time": sum(judge_times) / total,
        "total_judge_time": sum(judge_times),
        "error_distribution": error_counts,
    }


# =============================================================================
# 主测试函数
# =============================================================================

def test_sandbox_connection():
    """测试 SandboxFusion 连接"""
    print("\n[1] 测试 SandboxFusion 连接...")

    try:
        set_endpoint(SANDBOX_URL)

        # 简单测试
        result = run_code(RunCodeRequest(
            code="print('Hello, SandboxFusion!')",
            language="python",
            run_timeout=5,
        ))

        # result.status 可能是枚举类型，转换为字符串比较
        status = str(getattr(result, 'status', 'unknown'))
        run_result = getattr(result, 'run_result', None)

        # run_result 可能是对象，提取 stdout
        if hasattr(run_result, 'stdout'):
            output = run_result.stdout
        else:
            output = str(run_result)

        # 检查是否成功
        if ("Success" in status or "success" in status.lower()) and "Hello" in output:
            print(f"  ✓ 连接成功！Output: {output.strip()}")
            return True
        else:
            print(f"  ✗ 连接失败。Status: {status}, Output: {output}")
            return False

    except Exception as e:
        print(f"  ✗ 连接错误: {e}")
        print(f"  请确保 Docker 容器已启动:")
        print(f"  docker run -it --rm --privileged -p 8080:8080 volcengine/sandbox-fusion:server-20250609")
        return False


def test_dataset(dataset_key: str, n_samples: int = 3):
    """测试单个数据集"""
    print(f"\n[测试 {dataset_key}]")
    print("-" * 50)

    # 加载样本
    problems = load_sample_problems(dataset_key, n_samples)
    if not problems:
        print(f"  ⚠ 没有找到数据，跳过")
        return None

    print(f"  加载了 {len(problems)} 个样本")

    results = []
    for problem in problems:
        problem_id = problem["problem_id"]
        print(f"\n  评测 {problem_id}...")

        result = evaluate_problem(problem, dataset_key)
        results.append(result)

        # 打印单个结果
        status_icon = "✓" if result.accepted else "✗"
        print(f"    {status_icon} accepted={result.accepted}, pass_ratio={result.pass_ratio:.2f}, "
              f"error_type={result.error_type}, judge_time={result.judge_time:.2f}s")

        if not result.accepted and result.details:
            detail_str = str(result.details)[:100]
            print(f"    Details: {detail_str}")

    # 计算指标
    metrics = calculate_metrics(results)

    print(f"\n  === {dataset_key} 指标汇总 ===")
    print(f"  accepted@1: {metrics['accepted_at_1']:.2%}")
    print(f"  pass_ratio_mean: {metrics['pass_ratio_mean']:.4f}")
    print(f"  avg_judge_time: {metrics['avg_judge_time']:.2f}s")
    print(f"  error_distribution: {metrics['error_distribution']}")

    return metrics


# =============================================================================
# 模拟模型输出测试
# =============================================================================

# 正确的 HumanEval 输出
HUMANEVAL_CORRECT = """<code>
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
</code>"""

# 错误：只有函数体（会导致 SyntaxError）
HUMANEVAL_BODY_ONLY = """<code>
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
</code>"""

# 错误：包含 main guard（可能导致问题 - 实际上在当前评测逻辑下可能通过）
HUMANEVAL_WITH_MAIN = """<code>
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                if abs(elem - elem2) < threshold:
                    return True
    return False

if __name__ == "__main__":
    print(has_close_elements([1.0, 2.0], 0.5))
</code>"""

# 正确的 MBPP 输出
MBPP_CORRECT = """<code>
def remove_Occ(s, ch):
    for i in range(len(s)):
        if s[i] == ch:
            s = s[0:i] + s[i+1:]
            break
    for i in range(len(s)-1, -1, -1):
        if s[i] == ch:
            s = s[0:i] + s[i+1:]
            break
    return s
</code>"""

# 错误：函数名错误（会导致 NameError）
MBPP_WRONG_NAME = """<code>
def remove_char(s, ch):
    for i in range(len(s)):
        if s[i] == ch:
            s = s[0:i] + s[i+1:]
            break
    for i in range(len(s)-1, -1, -1):
        if s[i] == ch:
            s = s[0:i] + s[i+1:]
            break
    return s
</code>"""

# 测试多个 code block（应该取最长的）
MULTI_CODE_BLOCKS = """Here's my solution:
<code>
# first attempt
pass
</code>

Actually, here's the correct version:
<code>
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
</code>"""

# 大小写测试
UPPERCASE_CODE_TAG = """<CODE>
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                if abs(elem - elem2) < threshold:
                    return True
    return False
</CODE>"""


def _extract_code_from_completion(completion: str) -> str:
    """
    从模型输出中提取代码（复制自 phase0_eval.py 的最新版本）
    """
    import re

    # 1. 优先尝试 <code>...</code> 标签（大小写不敏感）
    code_tag_pattern = r'<code>(.*?)</code>'
    matches = re.findall(code_tag_pattern, completion, re.DOTALL | re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip()

    # 2. 尝试 markdown 代码块
    md_pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(md_pattern, completion, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # 3. 返回原始内容
    return completion.strip()


def test_code_extraction():
    """测试代码提取函数"""
    print("\n[测试代码提取函数]")
    print("-" * 50)

    # 测试单个 code block
    code1 = _extract_code_from_completion(HUMANEVAL_CORRECT)
    assert "def has_close_elements" in code1, "应该提取到完整函数"
    print("  ✓ 单个 code block 提取正确")

    # 测试多个 code block（取最长）
    code2 = _extract_code_from_completion(MULTI_CODE_BLOCKS)
    assert "def has_close_elements" in code2, "应该取最长的 code block"
    assert "pass" not in code2, "不应该取到短的 block"
    print("  ✓ 多个 code block 取最长正确")

    # 测试大小写不敏感
    code3 = _extract_code_from_completion(UPPERCASE_CODE_TAG)
    assert "def has_close_elements" in code3, "应该支持大写 <CODE>"
    print("  ✓ 大小写不敏感提取正确")


def test_simulated_outputs():
    """测试模拟的模型输出"""
    print("\n[测试模拟模型输出]")
    print("-" * 50)

    # 从 dataset_samples.jsonl 加载测试数据
    samples_path = DATA_DIR / "raw" / "dataset_samples.jsonl"
    samples = {}
    with open(samples_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            data = json.loads(line)
            samples[data["dataset"]] = data

    # 1. 测试 HumanEval 正确输出
    print("\n  1. HumanEval 正确输出:")
    he_sample = samples["humaneval"]
    result = evaluate_humaneval(
        _extract_code_from_completion(HUMANEVAL_CORRECT),
        he_sample["test_cases"],
        he_sample["problem_id"]
    )
    print(f"     accepted={result.accepted}, error_type={result.error_type}")
    if not result.accepted:
        print(f"     details: {result.details}")

    # 2. 测试 HumanEval 只有函数体
    print("\n  2. HumanEval 只有函数体 (应该失败):")
    result = evaluate_humaneval(
        _extract_code_from_completion(HUMANEVAL_BODY_ONLY),
        he_sample["test_cases"],
        he_sample["problem_id"]
    )
    print(f"     accepted={result.accepted}, error_type={result.error_type}")
    if result.accepted:
        print("     WARNING: 只有函数体竟然通过了！需要检查")
    else:
        print("     ✓ 符合预期：只有函数体应该失败")

    # 3. 测试 MBPP 正确输出
    print("\n  3. MBPP 正确输出:")
    mbpp_sample = samples["mbpp_reg"]
    result = evaluate_mbpp(
        _extract_code_from_completion(MBPP_CORRECT),
        mbpp_sample["test_cases"],
        mbpp_sample["problem_id"]
    )
    print(f"     accepted={result.accepted}, error_type={result.error_type}")
    if not result.accepted:
        print(f"     details: {result.details}")

    # 4. 测试 MBPP 函数名错误
    print("\n  4. MBPP 函数名错误 (应该失败):")
    result = evaluate_mbpp(
        _extract_code_from_completion(MBPP_WRONG_NAME),
        mbpp_sample["test_cases"],
        mbpp_sample["problem_id"]
    )
    print(f"     accepted={result.accepted}, error_type={result.error_type}")
    if result.accepted:
        print("     WARNING: 函数名错误竟然通过了！需要检查")
    else:
        print("     ✓ 符合预期：函数名错误应该失败")

    print("\n  模拟输出测试完成")


def test_codecontests_solutions():
    """测试 CodeContests 的手写解答"""
    print("\n[测试 CodeContests 解答]")
    print("-" * 50)

    # 从 dataset_samples.jsonl 加载带解答的测试数据
    samples_path = DATA_DIR / "raw" / "dataset_samples.jsonl"
    samples = {}
    with open(samples_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data["dataset"].startswith("codecontests"):
                samples[data["problem_id"]] = data

    # 测试 Codeforces/1548/C
    if "Codeforces/1548/C" in samples:
        print("\n  1. Codeforces/1548/C (The Three Little Pigs):")
        sample = samples["Codeforces/1548/C"]
        if "solutions" in sample and sample["solutions"]:
            code = sample["solutions"][0]
            result = evaluate_codecontests(code, sample["test_cases"], sample["problem_id"])
            print(f"     accepted={result.accepted}, pass_ratio={result.pass_ratio:.2f}, error_type={result.error_type}")
            if not result.accepted:
                print(f"     details: {result.details}")
        else:
            print("     No solution available")

    # 测试 Codeforces/1575/A
    if "Codeforces/1575/A" in samples:
        print("\n  2. Codeforces/1575/A (Another Sorting Problem):")
        sample = samples["Codeforces/1575/A"]
        if "solutions" in sample and sample["solutions"]:
            code = sample["solutions"][0]
            result = evaluate_codecontests(code, sample["test_cases"], sample["problem_id"])
            print(f"     accepted={result.accepted}, pass_ratio={result.pass_ratio:.2f}, error_type={result.error_type}")
            if not result.accepted:
                print(f"     details: {result.details}")
        else:
            print("     No solution available")

    # 测试错误场景 - 没有调用 solve()
    print("\n  3. CodeContests 没有调用 solve (应该失败):")
    no_call_code = '''
import sys
def solve():
    data = sys.stdin.read().split()
    n = int(data[0])
    q = int(data[1])
    for i in range(q):
        print("42")
# 忘了调用 solve()
'''
    if "Codeforces/1548/C" in samples:
        sample = samples["Codeforces/1548/C"]
        result = evaluate_codecontests(no_call_code, sample["test_cases"], sample["problem_id"])
        print(f"     accepted={result.accepted}, error_type={result.error_type}")
        if result.accepted:
            print("     WARNING: 没有调用 solve 竟然通过了！需要检查")
        else:
            print("     ✓ 符合预期：没有调用 solve 应该失败")

    print("\n  CodeContests 解答测试完成")


def main():
    print("=" * 60)
    print("  SandboxFusion 评测流程测试")
    print("=" * 60)
    print(f"Sandbox URL: {SANDBOX_URL}")
    print(f"Data Dir: {DATA_DIR}")

    # 0. 测试代码提取函数（不需要 SandboxFusion）
    test_code_extraction()

    # 1. 测试连接
    if not test_sandbox_connection():
        print("\n请先启动 SandboxFusion Docker 容器后重试。")
        return

    # 1.5. 测试模拟模型输出
    test_simulated_outputs()

    # 1.6. 测试 CodeContests 解答
    test_codecontests_solutions()

    # 2. 测试各数据集（使用参考解答）
    all_metrics = {}

    datasets_to_test = [
        ("humaneval", 3),         # HumanEval: 3 个样本
        ("mbpp_reg", 3),          # MBPP: 3 个样本
        ("codecontests_valid", 2), # CodeContests: 2 个样本（评测较慢）
    ]

    for dataset_key, n_samples in datasets_to_test:
        metrics = test_dataset(dataset_key, n_samples)
        if metrics:
            all_metrics[dataset_key] = metrics

    # 3. 总结
    print("\n" + "=" * 60)
    print("  测试完成！")
    print("=" * 60)

    if all_metrics:
        print("\n各数据集 accepted@1:")
        for dataset, metrics in all_metrics.items():
            print(f"  {dataset}: {metrics['accepted_at_1']:.2%}")

        # 保存结果
        output_path = Path(__file__).parent.parent / "outputs" / "test_sandbox_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")

    print("\n预期结果说明:")
    print("  - 使用参考解答评测，accepted@1 应该接近 100%")
    print("  - 如果 < 100%，可能是测试用例格式问题或参考解答有误")
    print("  - 这个测试验证了评测流程的正确性")


if __name__ == "__main__":
    main()
