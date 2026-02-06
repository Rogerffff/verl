#!/usr/bin/env python3
"""
Download BEE-spoke CodeContests Instruct (hq-python-deduped) and filter
overlaps with Codeforces valid/test from sine/FusedCodeContests.

Filtering rule:
- Only for records with source == 2 (Codeforces), match BEE `name` to
  Codeforces IDs in eval splits (e.g., Codeforces/1575/G -> 1575_G).
- If matched, drop the record.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

from datasets import load_dataset


CF_NAME_PATTERN = re.compile(r"(\\d+)_([A-Za-z0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BEE-spoke code_contests_instruct and filter Codeforces overlaps."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hq-python-deduped",
        help="Dataset config for BEE-spoke-data/code_contests_instruct",
    )
    parser.add_argument(
        "--valid_raw",
        type=str,
        default="data/raw/codecontests_valid_raw.jsonl",
        help="Path to codecontests_valid_raw.jsonl",
    )
    parser.add_argument(
        "--test_raw",
        type=str,
        default="data/raw/codecontests_test_raw.jsonl",
        help="Path to codecontests_test_raw.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="phase_1_ SFT/bee_hq_python_deduped_filtered",
        help="Output directory for filtered JSONL files",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=0,
        help="Optional cap for records per split (0 = no cap)",
    )
    return parser.parse_args()


def extract_cf_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    match = CF_NAME_PATTERN.search(name)
    if not match:
        return None
    contest_id, index = match.group(1), match.group(2).upper()
    return f"{contest_id}_{index}"


def cf_problem_id_to_name(problem_id: str) -> Optional[str]:
    if not problem_id or not problem_id.startswith("Codeforces/"):
        return None
    parts = problem_id.split("/")
    if len(parts) < 3:
        return None
    contest_id = parts[1]
    index = parts[2].upper()
    return f"{contest_id}_{index}"


def load_eval_cf_names(valid_raw: Path, test_raw: Path) -> Set[str]:
    eval_cf_names: Set[str] = set()
    for raw_path in (valid_raw, test_raw):
        with raw_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                name = cf_problem_id_to_name(record.get("problem_id", ""))
                if name:
                    eval_cf_names.add(name)
    return eval_cf_names


def record_source(record: Dict) -> Optional[int]:
    source_value = record.get("source")
    if source_value is None:
        return None
    if isinstance(source_value, bool):
        return None
    if isinstance(source_value, int):
        return source_value
    if isinstance(source_value, str) and source_value.isdigit():
        return int(source_value)
    return None


def should_drop_record(
    record: Dict,
    eval_cf_names: Set[str],
) -> Tuple[bool, Optional[str]]:
    source_value = record_source(record)
    if source_value != 2:
        return False, None
    name_value = record.get("name")
    name_key = extract_cf_name(name_value)
    if not name_key:
        return False, None
    if name_key in eval_cf_names:
        return True, name_key
    return False, None


def iter_dataset(
    config: str,
    split: str,
    streaming: bool,
) -> Iterable[Dict]:
    if streaming:
        return load_dataset(
            "BEE-spoke-data/code_contests_instruct",
            config,
            split=split,
            streaming=True,
        )
    return load_dataset(
        "BEE-spoke-data/code_contests_instruct",
        config,
        split=split,
        streaming=False,
    )


def filter_split(
    split: str,
    eval_cf_names: Set[str],
    output_dir: Path,
    config: str,
    streaming: bool,
    max_records: int,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.jsonl"
    overlap_path = output_dir / f"{split}.overlaps.jsonl"

    total = 0
    kept = 0
    dropped = 0
    missing_name = 0

    dataset_iter = iter_dataset(config, split, streaming)
    with output_path.open("w", encoding="utf-8") as output_file, overlap_path.open(
        "w",
        encoding="utf-8",
    ) as overlap_file:
        for record in dataset_iter:
            total += 1
            drop, overlap_name = should_drop_record(record, eval_cf_names)
            if drop:
                dropped += 1
                overlap_file.write(
                    json.dumps({"name": overlap_name, "source": record.get("source")})
                    + "\\n"
                )
            else:
                if record_source(record) == 2 and not extract_cf_name(record.get("name")):
                    missing_name += 1
                output_file.write(json.dumps(record, ensure_ascii=False) + "\\n")
                kept += 1

            if max_records and total >= max_records:
                break

    return {
        "total": total,
        "kept": kept,
        "dropped_overlap": dropped,
        "source2_missing_name": missing_name,
    }


def main() -> None:
    args = parse_args()

    valid_raw = Path(args.valid_raw)
    test_raw = Path(args.test_raw)
    output_dir = Path(args.output_dir)

    if not valid_raw.exists():
        raise FileNotFoundError(f"valid_raw not found: {valid_raw}")
    if not test_raw.exists():
        raise FileNotFoundError(f"test_raw not found: {test_raw}")

    eval_cf_names = load_eval_cf_names(valid_raw, test_raw)
    splits = ["train", "validation", "test"]

    report: Dict[str, Dict[str, int]] = {}
    for split in splits:
        stats = filter_split(
            split=split,
            eval_cf_names=eval_cf_names,
            output_dir=output_dir,
            config=args.config,
            streaming=args.streaming,
            max_records=args.max_records,
        )
        report[split] = stats

    report_path = output_dir / "filter_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)

    print("Filter complete. Report saved to:", report_path)
    for split, stats in report.items():
        print(
            f"{split}: total={stats['total']} kept={stats['kept']} "
            f"dropped_overlap={stats['dropped_overlap']} "
            f"source2_missing_name={stats['source2_missing_name']}"
        )


if __name__ == "__main__":
    main()
