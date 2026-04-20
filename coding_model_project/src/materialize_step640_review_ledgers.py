#!/usr/bin/env python3
"""Materialize normalized ledgers from step640_v1 review outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _index_queue(rows: list[dict[str, Any]], *, key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        if key in out:
            raise ValueError(f"Duplicate queue key: {key}")
        out[key] = row
    return out


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    review_root = repo_root / "coding_model_project" / "phase_2_ GRPO" / "review_assets" / "step640_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=review_root)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.review_root

    eval_queue = _read_jsonl(root / "eval_review_queue_step640_v1.jsonl")
    eval_decisions = _read_jsonl(root / "problem_quarantine_review_ledger_eval_step640_v1.jsonl")
    a_queue = _read_jsonl(root / "a_bucket_actions_review_queue_step640_v1.jsonl")
    a_decisions = _read_jsonl(root / "a_bucket_actions_decisions_step640_v1.jsonl")
    c_queue = _read_jsonl(root / "c_bucket_actions_review_queue_step640_v1.jsonl")
    c_decisions = _read_jsonl(root / "c_bucket_actions_decisions_step640_v1.jsonl")
    anchor_queue = _read_jsonl(root / "anchor_actions_review_queue_step640_v1.jsonl")
    anchor_decisions = _read_jsonl(root / "anchor_actions_decisions_step640_v1.jsonl")

    eval_by_problem = _index_queue(eval_queue, key_fields=("problem_id",))
    a_by_key = _index_queue(a_queue, key_fields=("dataset", "problem_id"))
    c_by_key = _index_queue(c_queue, key_fields=("dataset", "problem_id"))
    anchor_by_key = _index_queue(anchor_queue, key_fields=("dataset", "problem_id"))

    if len(eval_queue) != len(eval_decisions):
        raise ValueError(f"eval queue/decision mismatch: {len(eval_queue)} vs {len(eval_decisions)}")
    if len(a_queue) != len(a_decisions):
        raise ValueError(f"A queue/decision mismatch: {len(a_queue)} vs {len(a_decisions)}")
    if len(c_queue) != len(c_decisions):
        raise ValueError(f"C queue/decision mismatch: {len(c_queue)} vs {len(c_decisions)}")
    if len(anchor_queue) != len(anchor_decisions):
        raise ValueError(f"anchor queue/decision mismatch: {len(anchor_queue)} vs {len(anchor_decisions)}")

    review_ledger: list[dict[str, Any]] = []
    for row in eval_decisions:
        problem_id = str(row["problem_id"])
        queue_row = eval_by_problem.get((problem_id,))
        if queue_row is None:
            raise ValueError(f"Missing eval queue row for {problem_id}")
        merged = dict(row)
        merged["current_quarantine_state"] = queue_row.get("current_quarantine_state", "")
        merged["current_quarantine_subdecision"] = queue_row.get("current_quarantine_subdecision", "")
        merged["needs_rereview"] = bool(queue_row.get("needs_rereview", False))
        merged["latest_eval_roles"] = list(queue_row.get("latest_eval_roles", []))
        review_ledger.append(merged)

    bucket_actions: list[dict[str, Any]] = []
    for rows, queue_map in ((a_decisions, a_by_key), (c_decisions, c_by_key)):
        for row in rows:
            key = (str(row["dataset"]), str(row["problem_id"]))
            queue_row = queue_map.get(key)
            if queue_row is None:
                raise ValueError(f"Missing bucket queue row for {key}")
            merged = dict(row)
            merged["current_quarantine_state"] = queue_row.get("current_quarantine_state", "")
            merged["current_quarantine_subdecision"] = queue_row.get("current_quarantine_subdecision", "")
            merged["review_priority_v3"] = queue_row.get("review_priority_v3", "")
            merged["needs_rereview"] = bool(queue_row.get("needs_rereview", False))
            bucket_actions.append(merged)

    anchor_actions: list[dict[str, Any]] = []
    for row in anchor_decisions:
        key = (str(row["dataset"]), str(row["problem_id"]))
        queue_row = anchor_by_key.get(key)
        if queue_row is None:
            raise ValueError(f"Missing anchor queue row for {key}")
        merged = dict(row)
        merged["current_quarantine_state"] = queue_row.get("current_quarantine_state", "")
        merged["current_quarantine_subdecision"] = queue_row.get("current_quarantine_subdecision", "")
        merged["review_priority_v3"] = queue_row.get("review_priority_v3", "")
        merged["needs_rereview"] = bool(queue_row.get("needs_rereview", False))
        anchor_actions.append(merged)

    summary = {
        "review_root": str(root),
        "problem_quarantine_review_ledger_count": len(review_ledger),
        "problem_quarantine_review_decision_counts": dict(
            Counter(str(row["decision"]) for row in review_ledger)
        ),
        "bucket_action_count": len(bucket_actions),
        "bucket_action_decision_counts": dict(Counter(str(row["decision"]) for row in bucket_actions)),
        "anchor_action_count": len(anchor_actions),
        "anchor_action_decision_counts": dict(Counter(str(row["decision"]) for row in anchor_actions)),
        "files": {
            "problem_quarantine_review_ledger": str(root / "problem_quarantine_review_ledger_step640_v1.jsonl"),
            "curriculum_bucket_actions": str(root / "curriculum_bucket_actions_step640_v1.jsonl"),
            "curriculum_anchor_actions": str(root / "curriculum_anchor_actions_step640_v1.jsonl"),
        },
    }

    _write_jsonl(root / "problem_quarantine_review_ledger_step640_v1.jsonl", review_ledger)
    _write_jsonl(root / "curriculum_bucket_actions_step640_v1.jsonl", bucket_actions)
    _write_jsonl(root / "curriculum_anchor_actions_step640_v1.jsonl", anchor_actions)
    _write_json(root / "materialized_review_ledgers_step640_v1.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
