#!/usr/bin/env python3
"""
验证 Manifest 去重脚本
======================
检查 data/manifests/ 目录中的 manifest 文件是否已正确去重。

验证项目：
1. 单个 manifest 内部无 prompt_sha256 重复
2. manifest 中无空的 prompt（长度 > 0）
3. problem_id 唯一性检查
4. 显示数据集统计信息
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def verify_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    验证单个 manifest 文件

    Returns:
        {
            "file": str,
            "total_records": int,
            "unique_prompt_sha256": int,
            "unique_problem_id": int,
            "empty_prompts": int,
            "duplicate_sha256": List[str],
            "duplicate_problem_ids": List[str],
            "is_valid": bool,
            "issues": List[str],
        }
    """
    records = load_jsonl(manifest_path)

    result = {
        "file": manifest_path.name,
        "total_records": len(records),
        "unique_prompt_sha256": 0,
        "unique_problem_id": 0,
        "empty_prompts": 0,
        "duplicate_sha256": [],
        "duplicate_problem_ids": [],
        "is_valid": True,
        "issues": [],
    }

    if not records:
        result["is_valid"] = False
        result["issues"].append("Manifest 为空")
        return result

    # 统计 prompt_sha256 和 problem_id
    sha256_counts: Dict[str, int] = defaultdict(int)
    problem_id_counts: Dict[str, int] = defaultdict(int)

    for record in records:
        sha256 = record.get("prompt_sha256", "")
        problem_id = record.get("problem_id", "")
        prompt_length = record.get("prompt_length", 0)

        sha256_counts[sha256] += 1
        problem_id_counts[problem_id] += 1

        # 检查空 prompt
        if prompt_length == 0:
            result["empty_prompts"] += 1

    result["unique_prompt_sha256"] = len(sha256_counts)
    result["unique_problem_id"] = len(problem_id_counts)

    # 找出重复的 prompt_sha256
    for sha256, count in sha256_counts.items():
        if count > 1:
            result["duplicate_sha256"].append(f"{sha256[:16]}... ({count}次)")

    # 找出重复的 problem_id
    for pid, count in problem_id_counts.items():
        if count > 1:
            result["duplicate_problem_ids"].append(f"{pid} ({count}次)")

    # 验证结果
    if result["duplicate_sha256"]:
        result["is_valid"] = False
        result["issues"].append(f"发现 {len(result['duplicate_sha256'])} 个重复的 prompt_sha256")

    if result["duplicate_problem_ids"]:
        result["is_valid"] = False
        result["issues"].append(f"发现 {len(result['duplicate_problem_ids'])} 个重复的 problem_id")

    if result["empty_prompts"] > 0:
        result["is_valid"] = False
        result["issues"].append(f"发现 {result['empty_prompts']} 个空 prompt (prompt_length=0)")

    return result


def verify_cross_split_overlap(manifests: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    检查 CodeContests 各 split 之间的 prompt_sha256 重叠

    Returns:
        {
            "train_vs_valid": int,
            "train_vs_test": int,
            "valid_vs_test": int,
            "overlapping_samples": List[str],
        }
    """
    result = {
        "train_vs_valid": 0,
        "train_vs_test": 0,
        "valid_vs_test": 0,
        "overlapping_samples": [],
    }

    # 获取各 split 的 prompt_sha256 集合
    train_sha256 = set()
    valid_sha256 = set()
    test_sha256 = set()

    for name, records in manifests.items():
        sha256_set = {r.get("prompt_sha256", "") for r in records}
        if "train" in name:
            train_sha256 = sha256_set
        elif "valid" in name:
            valid_sha256 = sha256_set
        elif "test" in name:
            test_sha256 = sha256_set

    # 计算重叠
    if train_sha256 and valid_sha256:
        overlap = train_sha256 & valid_sha256
        result["train_vs_valid"] = len(overlap)
        if overlap:
            result["overlapping_samples"].extend([f"train-valid: {s[:16]}..." for s in list(overlap)[:3]])

    if train_sha256 and test_sha256:
        overlap = train_sha256 & test_sha256
        result["train_vs_test"] = len(overlap)
        if overlap:
            result["overlapping_samples"].extend([f"train-test: {s[:16]}..." for s in list(overlap)[:3]])

    if valid_sha256 and test_sha256:
        overlap = valid_sha256 & test_sha256
        result["valid_vs_test"] = len(overlap)
        if overlap:
            result["overlapping_samples"].extend([f"valid-test: {s[:16]}..." for s in list(overlap)[:3]])

    return result


def main():
    """主函数"""
    # 找到 manifests 目录
    script_dir = Path(__file__).parent
    manifests_dir = script_dir.parent / "data" / "manifests"

    if not manifests_dir.exists():
        print(f"❌ Manifests 目录不存在: {manifests_dir}")
        return

    print("=" * 60)
    print("Manifest 去重验证报告")
    print("=" * 60)
    print(f"Manifests 目录: {manifests_dir}")
    print()

    # 找到所有 manifest 文件
    manifest_files = list(manifests_dir.glob("*_manifest.jsonl"))

    if not manifest_files:
        print("❌ 未找到任何 manifest 文件")
        return

    print(f"找到 {len(manifest_files)} 个 manifest 文件")
    print()

    # 验证每个 manifest
    all_valid = True
    manifests_data: Dict[str, List[Dict[str, Any]]] = {}

    for manifest_file in sorted(manifest_files):
        result = verify_manifest(manifest_file)
        manifests_data[manifest_file.stem] = load_jsonl(manifest_file)

        # 打印结果
        status = "✅" if result["is_valid"] else "❌"
        print(f"{status} {result['file']}")
        print(f"   总记录数: {result['total_records']}")
        print(f"   唯一 prompt_sha256: {result['unique_prompt_sha256']}")
        print(f"   唯一 problem_id: {result['unique_problem_id']}")

        if not result["is_valid"]:
            all_valid = False
            print(f"   问题:")
            for issue in result["issues"]:
                print(f"     - {issue}")
            if result["duplicate_sha256"]:
                print(f"     重复 SHA256 示例: {result['duplicate_sha256'][:3]}")
            if result["duplicate_problem_ids"]:
                print(f"     重复 Problem ID 示例: {result['duplicate_problem_ids'][:3]}")
        print()

    # 检查跨 split 重叠（仅针对 CodeContests）
    cc_manifests = {k: v for k, v in manifests_data.items() if "codecontests" in k.lower()}
    if len(cc_manifests) >= 2:
        print("-" * 60)
        print("CodeContests 跨 Split 重叠检查")
        print("-" * 60)

        overlap_result = verify_cross_split_overlap(cc_manifests)

        has_overlap = any([
            overlap_result["train_vs_valid"] > 0,
            overlap_result["train_vs_test"] > 0,
            overlap_result["valid_vs_test"] > 0,
        ])

        status = "❌" if has_overlap else "✅"
        print(f"{status} train vs valid: {overlap_result['train_vs_valid']} 重叠")
        print(f"{status} train vs test: {overlap_result['train_vs_test']} 重叠")
        print(f"{status} valid vs test: {overlap_result['valid_vs_test']} 重叠")

        if overlap_result["overlapping_samples"]:
            print(f"   重叠样本示例: {overlap_result['overlapping_samples']}")
        print()

    # 总结
    print("=" * 60)
    print("验证总结")
    print("=" * 60)

    if all_valid:
        print("✅ 所有 manifest 文件验证通过！")
        print("   - 无 prompt_sha256 重复")
        print("   - 无空 prompt")
        print("   - 数据已正确去重")
    else:
        print("❌ 部分 manifest 文件存在问题，请检查上方详细报告")
        print()
        print("可能的原因:")
        print("  1. 使用了样本数据 (make run) 而非完整数据 (Docker)")
        print("  2. data_governance.py 未正确执行去重")
        print()
        print("建议:")
        print("  1. 启动 Docker SandboxFusion 服务")
        print("  2. 重新运行 data_governance.py")


if __name__ == "__main__":
    main()
