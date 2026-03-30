#!/usr/bin/env python3
"""Build verl RL parquet files from manifest-filtered raw coding datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from coding_model_project.src.prompting import SYSTEM_PROMPT, format_prompt
except ImportError:
    from prompting import SYSTEM_PROMPT, format_prompt


DEFAULT_DATASETS = {
    "codecontests_train_wo_valid_big": {
        "manifest": "manifests/codecontests_train_wo_valid_big_manifest.jsonl",
        "raw": "raw/codecontests_train_wo_valid_big_raw.jsonl",
    },
    "codecontests_valid": {
        "manifest": "manifests/codecontests_valid_manifest.jsonl",
        "raw": "raw/codecontests_valid_raw.jsonl",
    },
    "codecontests_valid_big": {
        "manifest": "manifests/codecontests_valid_big_manifest.jsonl",
        "raw": "raw/codecontests_valid_big_raw.jsonl",
    },
    "codecontests_test": {
        "manifest": "manifests/codecontests_test_manifest.jsonl",
        "raw": "raw/codecontests_test_raw.jsonl",
    },
    "mbpp_reg": {
        "manifest": "manifests/mbpp_reg_manifest.jsonl",
        "raw": "raw/mbpp_reg_raw.jsonl",
    },
    "humaneval": {
        "manifest": "manifests/humaneval_manifest.jsonl",
        "raw": "raw/humaneval_raw.jsonl",
    },
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _filter_raw_by_manifest_ids(manifest_path: Path, raw_path: Path) -> List[Dict[str, Any]]:
    manifest_ids = {record["problem_id"] for record in _read_jsonl(manifest_path)}
    filtered: List[Dict[str, Any]] = []
    found_ids = set()
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["problem_id"] in manifest_ids:
                filtered.append(record)
                found_ids.add(record["problem_id"])

    missing_ids = sorted(manifest_ids - found_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            f"{manifest_path.name} has {len(missing_ids)} problem_id(s) missing from {raw_path.name}: {preview}"
        )
    return filtered


def _build_prompt_messages(dataset_key: str, record: Dict[str, Any]) -> List[Dict[str, str]]:
    test_cases = record.get("test_cases", {})
    formatted_prompt = format_prompt(
        record["prompt"],
        dataset_key=dataset_key,
        entry_point=test_cases.get("entry_point", ""),
        example_call=test_cases.get("example_call", ""),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted_prompt},
    ]


def _convert_record(dataset_key: str, split: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "data_source": dataset_key,
        "prompt": _build_prompt_messages(dataset_key, record),
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "problem_id": record["problem_id"],
                "test_cases": record.get("test_cases", {}),
                "dataset": dataset_key,
            },
        },
        "extra_info": {
            "problem_id": record["problem_id"],
            "split": split,
            "dataset": dataset_key,
        },
    }


def _convert_dataset(dataset_key: str, split: str, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_convert_record(dataset_key, split, record) for record in records]


def _sample_records(records: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if sample_size <= 0 or len(records) <= sample_size:
        return list(records)
    rng = random.Random(seed)
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:sample_size]


def _write_parquet(records: List[Dict[str, Any]], output_path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to build GRPO parquet files. Please install pandas first.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(output_path, index=False)


def build_default_splits(
    *,
    data_root: Path,
    output_dir: Path,
    smoke_train_size: int,
    smoke_val_size: int,
    seed: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = _convert_dataset(
        "codecontests_train_wo_valid_big",
        "train",
        _filter_raw_by_manifest_ids(
            data_root / DEFAULT_DATASETS["codecontests_train_wo_valid_big"]["manifest"],
            data_root / DEFAULT_DATASETS["codecontests_train_wo_valid_big"]["raw"],
        ),
    )

    val_tier1_records: List[Dict[str, Any]] = []
    for dataset_key in ("codecontests_valid", "mbpp_reg"):
        val_tier1_records.extend(
            _convert_dataset(
                dataset_key,
                "val_tier1",
                _filter_raw_by_manifest_ids(
                    data_root / DEFAULT_DATASETS[dataset_key]["manifest"],
                    data_root / DEFAULT_DATASETS[dataset_key]["raw"],
                ),
            )
        )

    val_tier2_records = _convert_dataset(
        "codecontests_valid_big",
        "val_tier2",
        _filter_raw_by_manifest_ids(
            data_root / DEFAULT_DATASETS["codecontests_valid_big"]["manifest"],
            data_root / DEFAULT_DATASETS["codecontests_valid_big"]["raw"],
        ),
    )

    final_eval_records: List[Dict[str, Any]] = []
    for dataset_key in ("codecontests_test", "humaneval"):
        final_eval_records.extend(
            _convert_dataset(
                dataset_key,
                "final_eval",
                _filter_raw_by_manifest_ids(
                    data_root / DEFAULT_DATASETS[dataset_key]["manifest"],
                    data_root / DEFAULT_DATASETS[dataset_key]["raw"],
                ),
            )
        )

    smoke_train_records = _sample_records(train_records, smoke_train_size, seed)
    smoke_val_source = val_tier1_records if val_tier1_records else val_tier2_records
    smoke_val_records = _sample_records(smoke_val_source, smoke_val_size, seed)

    files = {
        "train": output_dir / "train.parquet",
        "val_tier1": output_dir / "val_tier1.parquet",
        "val_tier2": output_dir / "val_tier2.parquet",
        "final_eval": output_dir / "final_eval.parquet",
        "smoke_train": output_dir / "smoke_train.parquet",
        "smoke_val": output_dir / "smoke_val.parquet",
    }
    _write_parquet(train_records, files["train"])
    _write_parquet(val_tier1_records, files["val_tier1"])
    _write_parquet(val_tier2_records, files["val_tier2"])
    _write_parquet(final_eval_records, files["final_eval"])
    _write_parquet(smoke_train_records, files["smoke_train"])
    _write_parquet(smoke_val_records, files["smoke_val"])

    summary = {
        "train": len(train_records),
        "val_tier1": len(val_tier1_records),
        "val_tier2": len(val_tier2_records),
        "final_eval": len(final_eval_records),
        "smoke_train": len(smoke_train_records),
        "smoke_val": len(smoke_val_records),
        "files": {name: str(path) for name, path in files.items()},
    }
    summary_path = output_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build verl RL parquet files for GRPO coding experiments.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("coding_model_project/data"),
        help="Root directory containing manifests/ and raw/ subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("coding_model_project/data/grpo_parquet"),
        help="Directory to write parquet files into.",
    )
    parser.add_argument("--smoke_train_size", type=int, default=64, help="Number of samples in smoke_train.parquet.")
    parser.add_argument("--smoke_val_size", type=int, default=32, help="Number of samples in smoke_val.parquet.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for smoke dataset sampling.")
    args = parser.parse_args()

    summary = build_default_splits(
        data_root=args.data_root,
        output_dir=args.output_dir,
        smoke_train_size=args.smoke_train_size,
        smoke_val_size=args.smoke_val_size,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
