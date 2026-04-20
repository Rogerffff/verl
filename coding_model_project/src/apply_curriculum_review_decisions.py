#!/usr/bin/env python3
"""Apply manual A/C bucket review decisions and emit a cleaned manifest v2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BUCKET_TARGETS_V2 = {
    "A_retention": 128,
    "B_near_miss": 448,
    "C_hard_partial": 64,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _decision_sort_key(row: dict[str, Any]) -> tuple[int, int, float, str]:
    decision = str(row.get("manual_review_decision", ""))
    explicit = 0 if decision else 1
    critical = 0 if row.get("primary_seed_family") == "only400" else 1
    score = -float(row.get("retrieval_score_max", 0.0))
    return explicit, critical, score, str(row["problem_id"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="step600_v2")
    parser.add_argument("--target-a", type=int, default=BUCKET_TARGETS_V2["A_retention"])
    parser.add_argument("--target-b", type=int, default=BUCKET_TARGETS_V2["B_near_miss"])
    parser.add_argument("--target-c", type=int, default=BUCKET_TARGETS_V2["C_hard_partial"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_rows = _read_jsonl(args.manifest)
    decision_rows = _read_jsonl(args.decisions)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for row in decision_rows:
        key = (str(row["dataset"]), str(row["problem_id"]))
        if key in decisions:
            raise ValueError(f"Duplicate decision row for {key}")
        decisions[key] = row

    updated_rows: list[dict[str, Any]] = []
    missing_decisions: list[str] = []
    decision_counter = Counter()
    dropped_keys: list[str] = []

    for row in manifest_rows:
        key = (str(row["dataset"]), str(row["problem_id"]))
        review_row = decisions.get(key)
        new_row = dict(row)
        if review_row is None:
            if row["initial_bucket"] in {"A_retention", "C_hard_partial"}:
                missing_decisions.append(f"{key[0]}::{key[1]}")
            new_row["manual_review_decision"] = ""
            new_row["manual_review_reason"] = ""
            new_row["manual_review_reviewer"] = ""
            updated_rows.append(new_row)
            continue

        decision = str(review_row.get("decision", "")).strip()
        decision_reason = str(review_row.get("decision_reason", "")).strip()
        reviewer = str(review_row.get("reviewer", "")).strip()
        new_row["manual_review_decision"] = decision
        new_row["manual_review_reason"] = decision_reason
        new_row["manual_review_reviewer"] = reviewer

        if not decision:
            updated_rows.append(new_row)
            continue

        decision_counter[decision] += 1
        if decision == "drop":
            dropped_keys.append(f"{key[0]}::{key[1]}")
            continue
        if decision == "move_to_U":
            dropped_keys.append(f"{key[0]}::{key[1]}")
            continue
        if decision == "move_to_B":
            new_row["initial_bucket"] = "B_near_miss"
        elif decision == "keep_A":
            new_row["initial_bucket"] = "A_retention"
        elif decision == "keep_C":
            new_row["initial_bucket"] = "C_hard_partial"
        else:
            raise ValueError(f"Unsupported decision {decision!r} for {key}")
        updated_rows.append(new_row)

    bucket_targets = {
        "A_retention": args.target_a,
        "B_near_miss": args.target_b,
        "C_hard_partial": args.target_c,
    }
    capped_rows: list[dict[str, Any]] = []
    bucket_before_cap = Counter(row["initial_bucket"] for row in updated_rows)
    bucket_after_cap: Counter[str] = Counter()

    for bucket in ("A_retention", "B_near_miss", "C_hard_partial"):
        rows = [row for row in updated_rows if row["initial_bucket"] == bucket]
        rows.sort(key=_decision_sort_key)
        kept = rows[: bucket_targets[bucket]]
        capped_rows.extend(kept)
        bucket_after_cap[bucket] = len(kept)

    capped_rows.sort(
        key=lambda row: (
            {"A_retention": 0, "B_near_miss": 1, "C_hard_partial": 2}.get(row["initial_bucket"], 9),
            -float(row.get("retrieval_score_max", 0.0)),
            row["problem_id"],
        )
    )

    manifest_out = output_dir / f"curriculum_train_manifest_{args.tag}.jsonl"
    report_out = output_dir / f"curriculum_review_apply_report_{args.tag}.json"

    _write_jsonl(manifest_out, capped_rows)
    _write_json(
        report_out,
        {
            "source_manifest": str(args.manifest),
            "source_decisions": str(args.decisions),
            "bucket_targets": bucket_targets,
            "manifest_rows_before": len(manifest_rows),
            "manifest_rows_after_manual_actions": len(updated_rows),
            "manifest_rows_after_caps": len(capped_rows),
            "bucket_before_cap": dict(bucket_before_cap),
            "bucket_after_cap": dict(bucket_after_cap),
            "decision_counter": dict(decision_counter),
            "missing_decision_count": len(missing_decisions),
            "missing_decision_preview": missing_decisions[:50],
            "dropped_count": len(dropped_keys),
            "dropped_preview": dropped_keys[:50],
        },
    )

    print(f"Wrote cleaned manifest to {manifest_out}")
    print(f"Wrote review apply report to {report_out}")


if __name__ == "__main__":
    main()
