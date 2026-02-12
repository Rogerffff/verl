#!/usr/bin/env python3
"""
SFT 数据去重：每个题目只保留一条记录。

选择策略：对同一题目的多条解答，保留 text 字段最短的那条
（竞赛编程中更短的解答通常更简洁清晰）。

输入：  bee_hq_python_deduped_filtered/train.jsonl   (95629 条)
输出：  bee_hq_python_deduped_filtered/train_one_per_problem.jsonl
统计：  bee_hq_python_deduped_filtered/dedup_report.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

CF_NAME_PATTERN = re.compile(r"(\d+)_([A-Za-z0-9]+)")


def problem_key(record: dict) -> str:
    """为每条记录生成唯一的题目 key。"""
    source = record.get("source", "?")
    name = record.get("name", "")
    m = CF_NAME_PATTERN.search(name)
    if m:
        return f"{source}_{m.group(1)}_{m.group(2).upper()}"
    return f"{source}_{name}"


def main():
    parser = argparse.ArgumentParser(description="Dedup SFT data: one record per problem.")
    parser.add_argument(
        "--input",
        type=str,
        default="phase_1_ SFT/bee_hq_python_deduped_filtered/train.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="phase_1_ SFT/bee_hq_python_deduped_filtered/train_one_per_problem.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["shortest", "longest", "first"],
        default="shortest",
        help="Which solution to keep per problem (default: shortest)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Pass 1: group records by problem key, keep the best one per strategy
    best_records: dict[str, tuple[int, str]] = {}  # key -> (sort_value, json_line)
    total = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            key = problem_key(rec)
            text_len = len(rec.get("text", ""))

            if args.strategy == "shortest":
                sort_val = text_len
            elif args.strategy == "longest":
                sort_val = -text_len
            else:  # first
                sort_val = total

            if key not in best_records or sort_val < best_records[key][0]:
                best_records[key] = (sort_val, line.strip())

    # Pass 2: write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for key in sorted(best_records.keys()):
            f.write(best_records[key][1] + "\n")

    kept = len(best_records)
    removed = total - kept

    # Write report
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "strategy": args.strategy,
        "total_input": total,
        "unique_problems": kept,
        "duplicates_removed": removed,
    }

    report_path = output_path.parent / "dedup_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Input:    {total} records")
    print(f"Output:   {kept} records (one per problem)")
    print(f"Removed:  {removed} duplicate solutions")
    print(f"Strategy: {args.strategy}")
    print(f"Saved to: {output_path}")
    print(f"Report:   {report_path}")


if __name__ == "__main__":
    main()
