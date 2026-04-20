#!/usr/bin/env python3
"""Prepare focused review assets from light-screen quarantine candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES = Path("coding_model_project/data/quarantine_audit/problem_quarantine_candidates_v2.jsonl")
DEFAULT_OUTPUT_DIR = Path("coding_model_project/data/quarantine_audit/review_assets_v1")

STRUCTURAL_SIGNALS = {
    "query_count_mismatch",
    "opcode_arity_inconsistency",
    "range_violation",
    "second_line_length_mismatch",
}
CLEAR_HARD_SIGNALS = {"marker_contamination", "missing_tests", "malformed_tests"}
EVAL_SPLITS = {"codecontests_valid_big", "codecontests_test", "codecontests_valid"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _signal_names(row: dict[str, Any]) -> set[str]:
    return {signal["name"] for signal in row.get("merged_signals", [])}


def _strip_large_fields(row: dict[str, Any]) -> dict[str, Any]:
    slim = {
        "problem_id": row["problem_id"],
        "seen_splits": row["seen_splits"],
        "existing_quarantine_status": row["existing_quarantine_status"],
        "recommended_decision": row["recommended_decision"],
        "max_risk_score": row["max_risk_score"],
        "prompt_title": row["prompt_title"],
        "tests_count_examples": row["tests_count_examples"],
        "merged_signals": row["merged_signals"],
    }
    if "prompt_excerpt" in row:
        slim["prompt_excerpt"] = row["prompt_excerpt"]
    return slim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = _read_jsonl(args.candidates)

    auto_hard: list[dict[str, Any]] = []
    structural_review: list[dict[str, Any]] = []
    structural_manual_train_only: list[dict[str, Any]] = []
    eval_priority_review: list[dict[str, Any]] = []
    combined_queue: dict[str, dict[str, Any]] = {}
    clear_hard_with_structural_overlap_count = 0

    for row in rows:
        names = _signal_names(row)
        splits = set(row["seen_splits"])
        decision = row["recommended_decision"]
        has_structural = bool(names & STRUCTURAL_SIGNALS)
        has_clear_hard = bool(names & CLEAR_HARD_SIGNALS)

        if decision == "hard_blacklist" and has_structural and has_clear_hard:
            clear_hard_with_structural_overlap_count += 1

        if decision == "hard_blacklist" and has_structural:
            structural_review.append(_strip_large_fields(row))
        elif decision == "hard_blacklist" and has_clear_hard:
            auto_hard.append(_strip_large_fields(row))

        if decision == "manual_review" and has_structural and not (splits & EVAL_SPLITS):
            structural_manual_train_only.append(_strip_large_fields(row))

        if splits & EVAL_SPLITS:
            eval_priority_review.append(_strip_large_fields(row))

    auto_hard.sort(key=lambda item: (-int(item["max_risk_score"]), item["problem_id"]))
    structural_review.sort(key=lambda item: (-int(item["max_risk_score"]), item["problem_id"]))
    structural_manual_train_only.sort(key=lambda item: (-int(item["max_risk_score"]), item["problem_id"]))
    eval_priority_review.sort(key=lambda item: (-int(item["max_risk_score"]), item["problem_id"]))

    for bucket_name, bucket_rows in (
        ("structural_signal_review", structural_review),
        ("structural_manual_train_only_review", structural_manual_train_only),
        ("eval_priority_review", eval_priority_review),
    ):
        for row in bucket_rows:
            problem_id = str(row["problem_id"])
            if problem_id not in combined_queue:
                combined_queue[problem_id] = dict(row)
                combined_queue[problem_id]["review_reasons"] = []
            combined_queue[problem_id]["review_reasons"].append(bucket_name)

    combined_queue_rows = sorted(combined_queue.values(), key=lambda item: (-int(item["max_risk_score"]), item["problem_id"]))

    _write_jsonl(args.output_dir / "problem_quarantine_auto_hard_v2.jsonl", auto_hard)
    _write_jsonl(args.output_dir / "problem_quarantine_structural_review_v1.jsonl", structural_review)
    _write_jsonl(
        args.output_dir / "problem_quarantine_structural_manual_train_only_v1.jsonl",
        structural_manual_train_only,
    )
    _write_jsonl(args.output_dir / "problem_quarantine_eval_priority_review_v1.jsonl", eval_priority_review)
    _write_jsonl(args.output_dir / "problem_quarantine_claude_review_queue_v2.jsonl", combined_queue_rows)

    summary = {
        "candidates_path": str(args.candidates),
        "output_dir": str(args.output_dir),
        "total_candidates": len(rows),
        "auto_hard_count": len(auto_hard),
        "auto_hard_definition": "clear-hard-only (hard_blacklist rows with marker/malformed/missing signals and no structural signal overlap)",
        "clear_hard_with_structural_overlap_count": clear_hard_with_structural_overlap_count,
        "structural_review_count": len(structural_review),
        "structural_manual_train_only_count": len(structural_manual_train_only),
        "eval_priority_review_count": len(eval_priority_review),
        "claude_review_queue_count": len(combined_queue_rows),
        "decision_counts": dict(Counter(row["recommended_decision"] for row in rows)),
    }
    _write_json(args.output_dir / "problem_quarantine_review_summary_v1.json", summary)

    print(f"Wrote {len(auto_hard)} auto-hard rows")
    print(f"Wrote {len(structural_review)} structural-review rows")
    print(f"Wrote {len(structural_manual_train_only)} structural-manual-train-only rows")
    print(f"Wrote {len(eval_priority_review)} eval-priority rows")
    print(f"Wrote {len(combined_queue_rows)} Claude review queue rows")
    print(f"Wrote summary to {args.output_dir / 'problem_quarantine_review_summary_v1.json'}")


if __name__ == "__main__":
    main()
