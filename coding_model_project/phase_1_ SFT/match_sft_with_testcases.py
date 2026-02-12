#!/usr/bin/env python3
"""
将去重后的 SFT 数据与 CodeContests 测试用例匹配。

映射规则：
  - Source 2 (Codeforces): BEE name "1000_A. ..." → "Codeforces/1000/A"
  - Source 6 (AtCoder):    BEE name 就是 CodeContests 的 problem_id
  - Source 7 (Aizu):       BEE name "p00001 ..." → "AIZU/p00001"

输入：
  - train_one_per_problem.jsonl (去重后的 SFT 数据)
  - codecontests_train_wo_valid_big_raw.jsonl (含测试用例的原始数据)

输出：
  - sft_with_testcases.jsonl (匹配成功的记录，含 SFT text + 测试用例)
  - match_report.json (匹配统计)
"""

import argparse
import json
import re
from pathlib import Path

CF_NAME_PATTERN = re.compile(r"(\d+)_([A-Za-z0-9]+)")
AIZU_ID_PATTERN = re.compile(r"^(p\d+)")
CODE_BLOCK_PATTERN = re.compile(r"```(?:python3?|Python3?)\s*\n(.*?)```", re.DOTALL)


def extract_solution(text: str) -> str:
    """从 BEE text 字段提取代码解答。

    格式: ### Prompt ... ### Response ```python3 <code> ```
    """
    # 找 ### Response 分割点
    sep = "### Response"
    idx = text.find(sep)
    response_part = text[idx + len(sep):] if idx >= 0 else text

    # 提取 ```python3 ... ``` 代码块
    m = CODE_BLOCK_PATTERN.search(response_part)
    if m:
        return m.group(1).strip()

    # fallback: 取 ### Response 之后的全部文本
    return response_part.strip()


def bee_name_to_problem_id(source: int, name: str) -> str | None:
    """将 BEE 的 (source, name) 转换为 CodeContests problem_id。"""
    if source == 2:  # Codeforces
        m = CF_NAME_PATTERN.search(name)
        if m:
            return f"Codeforces/{m.group(1)}/{m.group(2).upper()}"
    elif source == 6:  # AtCoder: name 就是 problem_id
        return name
    elif source == 7:  # Aizu
        m = AIZU_ID_PATTERN.match(name)
        if m:
            return f"AIZU/{m.group(1)}"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Match deduped SFT data with CodeContests test cases."
    )
    parser.add_argument(
        "--sft_input",
        type=str,
        default="phase_1_ SFT/bee_hq_python_deduped_filtered/train_one_per_problem.jsonl",
    )
    parser.add_argument(
        "--raw_input",
        type=str,
        default="data/raw/codecontests_train_wo_valid_big_raw.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="phase_1_ SFT/bee_hq_python_deduped_filtered/sft_with_testcases.jsonl",
    )
    args = parser.parse_args()

    sft_path = Path(args.sft_input)
    raw_path = Path(args.raw_input)
    output_path = Path(args.output)

    # Step 1: Build test case lookup from CodeContests raw data
    print("[1/3] Loading CodeContests test cases...")
    testcase_lookup: dict[str, dict] = {}
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec.get("problem_id", "")
            testcase_lookup[pid] = {
                "test_cases": rec.get("test_cases", {}),
            }
    print(f"  Loaded test cases for {len(testcase_lookup)} problems")

    # Step 2: Match SFT records with test cases
    print("[2/3] Matching SFT records...")
    total = 0
    matched = 0
    unmatched_no_pid = 0
    unmatched_no_tc = 0
    solution_extracted = 0
    solution_fallback = 0
    by_source = {"matched": {}, "unmatched": {}}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sft_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            rec = json.loads(line)
            source = rec.get("source")
            name = rec.get("name", "")

            pid = bee_name_to_problem_id(source, name)

            if pid is None:
                unmatched_no_pid += 1
                by_source.setdefault("unmatched", {}).setdefault(source, 0)
                by_source["unmatched"][source] = by_source["unmatched"].get(source, 0) + 1
                continue

            tc_data = testcase_lookup.get(pid)
            if tc_data is None:
                unmatched_no_tc += 1
                by_source.setdefault("unmatched", {}).setdefault(source, 0)
                by_source["unmatched"][source] = by_source["unmatched"].get(source, 0) + 1
                continue

            matched += 1
            by_source.setdefault("matched", {}).setdefault(source, 0)
            by_source["matched"][source] = by_source["matched"].get(source, 0) + 1

            text = rec.get("text", "")
            solution = extract_solution(text)
            if CODE_BLOCK_PATTERN.search(text[text.find("### Response"):] if "### Response" in text else text):
                solution_extracted += 1
            else:
                solution_fallback += 1

            out_rec = {
                "problem_id": pid,
                "source": source,
                "name": name,
                "difficulty": rec.get("difficulty"),
                "text": text,
                "solution": solution,
                "test_cases": tc_data["test_cases"],
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    # Step 3: Write report
    print("[3/3] Writing report...")
    source_names = {1: "CodeChef", 2: "Codeforces", 3: "HackerEarth", 6: "AtCoder", 7: "Aizu"}
    report = {
        "input_sft": str(sft_path),
        "input_raw": str(raw_path),
        "output": str(output_path),
        "total_sft_records": total,
        "matched": matched,
        "unmatched_no_problem_id": unmatched_no_pid,
        "unmatched_no_testcase": unmatched_no_tc,
        "matched_by_source": {
            source_names.get(k, str(k)): v
            for k, v in sorted(by_source.get("matched", {}).items())
        },
        "solution_extracted_from_code_block": solution_extracted,
        "solution_fallback_raw_text": solution_fallback,
        "unmatched_by_source": {
            source_names.get(k, str(k)): v
            for k, v in sorted(by_source.get("unmatched", {}).items())
        },
    }

    report_path = output_path.with_name("match_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nResults:")
    print(f"  Total SFT records:  {total}")
    print(f"  Matched:            {matched}")
    print(f"  Unmatched (no pid): {unmatched_no_pid}")
    print(f"  Unmatched (no tc):  {unmatched_no_tc}")
    print(f"\n  By source (matched):")
    for src, count in sorted(by_source.get("matched", {}).items()):
        print(f"    {source_names.get(src, src)}: {count}")
    print(f"\n  Output: {output_path}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
