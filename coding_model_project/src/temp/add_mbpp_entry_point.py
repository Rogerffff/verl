#!/usr/bin/env python3
"""
为 MBPP 数据集添加 entry_point 和 example_call 字段
=====================================================

从 test_list 中的 assert 语句提取：
- entry_point: 函数名
- example_call: 调用形式（如 remove_Occ("hello","l")）

使用方法：
    cd verl/coding_model_project
    python3 src/temp/add_mbpp_entry_point.py

注意：
    - 会自动备份原文件
    - 先写入临时文件再替换，确保安全
"""

import json
import re
from pathlib import Path
import shutil
from datetime import datetime


def extract_entry_point(test_cases: dict) -> str:
    """
    从 test_cases 中提取函数名

    更鲁棒的逻辑：扫描 test_list + challenge_test_list，找到第一个匹配就用

    Args:
        test_cases: 包含 test_list 和 challenge_test_list 的字典

    Returns:
        提取的函数名，如果失败返回空字符串
    """
    pattern = re.compile(r'assert\s+(\w+)\s*\(')
    candidates = []
    candidates += test_cases.get("test_list", []) or []
    candidates += test_cases.get("challenge_test_list", []) or []

    for s in candidates:
        m = pattern.search(s)
        if m:
            return m.group(1)
    return ""


def extract_example_call(test_list: list) -> str:
    """
    从 test_list 中提取调用形式

    使用"截断比较运算符"方法，比正则匹配括号更鲁棒。
    可以正确处理嵌套括号、列表、元组等参数。

    示例：
        输入: assert remove_Occ("hello","l") == "heo"
        输出: remove_Occ("hello","l")

        输入: assert sort_matrix([[1,2,3],[2,4,5]]) == [[1,1,1],[1,2,3]]
        输出: sort_matrix([[1,2,3],[2,4,5]])

    Args:
        test_list: assert 语句列表

    Returns:
        调用表达式，如果失败返回空字符串
    """
    if not test_list:
        return ""

    s = test_list[0].strip()

    # 去掉开头 "assert"
    m = re.match(r'^assert\s+(.*)$', s)
    if not m:
        return ""
    expr = m.group(1).strip()

    # 截断到比较运算符之前（优先处理 ==, !=）
    for op in ["==", "!=", ">=", "<=", ">", "<"]:
        idx = expr.find(op)
        if idx != -1:
            expr = expr[:idx].strip()
            break

    # 通常这里就是 func(...)
    return expr


def main():
    # 计算数据文件路径
    # 脚本位置: src/temp/add_mbpp_entry_point.py
    # 数据位置: data/raw/mbpp_reg_raw.jsonl
    script_dir = Path(__file__).parent
    raw_path = script_dir.parent.parent / "data" / "raw" / "mbpp_reg_raw.jsonl"
    temp_path = raw_path.with_suffix('.jsonl.tmp')
    backup_path = raw_path.with_suffix(f'.jsonl.bak.{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    if not raw_path.exists():
        print(f"Error: Data file not found: {raw_path}")
        return

    print(f"Processing: {raw_path}")

    records = []
    missing_entry_points = []
    missing_example_calls = []
    updated_count = 0

    with open(raw_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line)
            test_cases = record.get("test_cases", {})
            test_list = test_cases.get("test_list", [])
            problem_id = record.get("problem_id", "unknown")

            # 提取 entry_point（如果还没有）
            if "entry_point" not in test_cases:
                entry_point = extract_entry_point(test_cases)
                if entry_point:
                    test_cases["entry_point"] = entry_point
                else:
                    missing_entry_points.append((line_num, problem_id))

            # 提取 example_call（总是更新）
            example_call = extract_example_call(test_list)
            if example_call:
                test_cases["example_call"] = example_call
                updated_count += 1
            else:
                missing_example_calls.append((line_num, problem_id))

            record["test_cases"] = test_cases
            records.append(record)

    # 报告提取失败的情况
    if missing_entry_points:
        print(f"\nWARNING: {len(missing_entry_points)} records missing entry_point:")
        for line_num, pid in missing_entry_points[:5]:
            print(f"  Line {line_num}: problem_id={pid}")
        if len(missing_entry_points) > 5:
            print(f"  ... and {len(missing_entry_points) - 5} more")

    if missing_example_calls:
        print(f"\nWARNING: {len(missing_example_calls)} records missing example_call:")
        for line_num, pid in missing_example_calls[:5]:
            print(f"  Line {line_num}: problem_id={pid}")
        if len(missing_example_calls) > 5:
            print(f"  ... and {len(missing_example_calls) - 5} more")

    # 先写到临时文件
    with open(temp_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 备份原文件，再替换
    shutil.copy2(raw_path, backup_path)
    shutil.move(temp_path, raw_path)

    print(f"\nProcessed {len(records)} records")
    print(f"Added/updated example_call for {updated_count} records")
    print(f"Backup saved to: {backup_path}")

    # 显示几个示例
    print("\n示例提取结果:")
    for i, record in enumerate(records[:3]):
        tc = record.get("test_cases", {})
        print(f"  {i+1}. entry_point: {tc.get('entry_point', 'N/A')}")
        print(f"     example_call: {tc.get('example_call', 'N/A')}")


if __name__ == "__main__":
    main()
