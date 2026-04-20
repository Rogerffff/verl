#!/usr/bin/env python3

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_name(row):
    gts = row.get("gts")
    if isinstance(gts, dict):
        dataset = gts.get("dataset")
        if dataset:
            return dataset
    return row.get("data_source", "<missing>")


def numeric_mean(rows, key):
    vals = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            vals.append(float(value))
        elif isinstance(value, (int, float)) and not math.isnan(value):
            vals.append(float(value))
    return sum(vals) / len(vals) if vals else None


def summarize_rows(rows):
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[dataset_name(row)].append(row)

    summary = {}
    for ds, ds_rows in sorted(by_dataset.items()):
        errors = Counter(row.get("error_type", "") or "ok" for row in ds_rows)
        summary[ds] = {
            "count": len(ds_rows),
            "accepted_mean": numeric_mean(ds_rows, "accepted"),
            "pass_ratio_all_mean": numeric_mean(ds_rows, "pass_ratio_all"),
            "reward_mean": numeric_mean(ds_rows, "reward"),
            "reward_raw_mean": numeric_mean(ds_rows, "reward_raw"),
            "invalid_for_rl_mean": numeric_mean(ds_rows, "invalid_for_rl"),
            "truncated_mean": numeric_mean(ds_rows, "truncated_by_max_tokens"),
            "judge_time_s_mean": numeric_mean(ds_rows, "judge_time_s"),
            "error_type_counts": dict(sorted(errors.items())),
        }
    return summary


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: summarize_checkpoint_eval.py OUTPUT_JSON INPUT_JSONL...")

    output_path = Path(sys.argv[1])
    input_paths = [Path(p) for p in sys.argv[2:]]

    rows = []
    for path in input_paths:
        if path.exists():
            rows.extend(load_jsonl(path))

    if not rows:
        raise SystemExit("no rows found in provided input files")

    summary = summarize_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(output_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
