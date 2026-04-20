#!/usr/bin/env python3
"""Apply focused manifest v3 review decisions and emit the final manifest."""

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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-v2", type=Path, required=True)
    parser.add_argument("--c-decisions-v3", type=Path, required=True)
    parser.add_argument("--anchor-candidates-v3", type=Path, required=True)
    parser.add_argument("--anchor-decisions-v3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="step600_v3")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_v2 = _read_jsonl(args.manifest_v2)
    c_decisions = {
        (str(row["dataset"]), str(row["problem_id"])): row for row in _read_jsonl(args.c_decisions_v3)
    }
    anchor_candidates = {
        (str(row["dataset"]), str(row["problem_id"])): row for row in _read_jsonl(args.anchor_candidates_v3)
    }
    anchor_decisions = {
        (str(row["dataset"]), str(row["problem_id"])): row for row in _read_jsonl(args.anchor_decisions_v3)
    }

    updated_rows: list[dict[str, Any]] = []
    c_counter = Counter()
    untouched_counter = Counter()
    added_anchor_rows: list[dict[str, Any]] = []
    added_anchor_counter = Counter()

    for row in manifest_v2:
        key = (str(row["dataset"]), str(row["problem_id"]))
        new_row = dict(row)
        bucket = str(row["initial_bucket"])

        if bucket == "C_hard_partial":
            decision_row = c_decisions.get(key)
            if decision_row is None:
                raise ValueError(f"Missing C v3 decision for {key}")
            decision = str(decision_row.get("decision", "")).strip()
            reason = str(decision_row.get("decision_reason", "")).strip()
            reviewer = str(decision_row.get("reviewer", "")).strip()
            if not decision:
                raise ValueError(f"Empty C v3 decision for {key}")
            c_counter[decision] += 1
            if decision == "keep_C":
                new_row["initial_bucket"] = "C_hard_partial"
            elif decision == "move_to_B":
                new_row["initial_bucket"] = "B_near_miss"
            elif decision in {"move_to_U", "drop"}:
                continue
            else:
                raise ValueError(f"Unsupported C v3 decision {decision!r} for {key}")
            new_row["manual_review_v3_decision"] = decision
            new_row["manual_review_v3_reason"] = reason
            new_row["manual_review_v3_reviewer"] = reviewer
            updated_rows.append(new_row)
            continue

        untouched_counter[bucket] += 1
        updated_rows.append(new_row)

    for key, decision_row in anchor_decisions.items():
        decision = str(decision_row.get("decision", "")).strip()
        reason = str(decision_row.get("decision_reason", "")).strip()
        reviewer = str(decision_row.get("reviewer", "")).strip()
        if not decision:
            raise ValueError(f"Empty anchor v3 decision for {key}")
        added_anchor_counter[decision] += 1
        if decision != "add_to_A":
            continue
        candidate = anchor_candidates.get(key)
        if candidate is None:
            raise ValueError(f"Missing anchor candidate row for {key}")
        added_anchor_rows.append(
            {
                "dataset": candidate["dataset"],
                "problem_id": candidate["problem_id"],
                "initial_bucket": "A_retention",
                "primary_seed_family": candidate.get("primary_seed_family", "anchor_common"),
                "matched_seed_ids": list(candidate.get("matched_seed_ids", [])),
                "matched_seed_families": list(candidate.get("matched_seed_families", [])),
                "retrieval_score_max": float(candidate.get("retrieval_score_max", 0.0)),
                "prompt_sha256": candidate.get("prompt_sha256", ""),
                "manual_review_v3_decision": decision,
                "manual_review_v3_reason": reason,
                "manual_review_v3_reviewer": reviewer,
            }
        )

    final_rows = [*updated_rows, *added_anchor_rows]
    final_rows.sort(
        key=lambda row: (
            {"A_retention": 0, "B_near_miss": 1, "C_hard_partial": 2}.get(str(row["initial_bucket"]), 9),
            -float(row.get("retrieval_score_max", 0.0)),
            str(row["problem_id"]),
        )
    )

    report = {
        "source_manifest_v2": str(args.manifest_v2),
        "source_c_decisions_v3": str(args.c_decisions_v3),
        "source_anchor_candidates_v3": str(args.anchor_candidates_v3),
        "source_anchor_decisions_v3": str(args.anchor_decisions_v3),
        "manifest_v2_rows": len(manifest_v2),
        "updated_rows_pre_anchor": len(updated_rows),
        "added_anchor_rows": len(added_anchor_rows),
        "final_manifest_rows": len(final_rows),
        "c_decision_counter": dict(c_counter),
        "anchor_decision_counter": dict(added_anchor_counter),
        "bucket_counts": dict(Counter(str(row["initial_bucket"]) for row in final_rows)),
    }

    manifest_out = output_dir / f"curriculum_train_manifest_{args.tag}.jsonl"
    report_out = output_dir / f"curriculum_apply_report_{args.tag}.json"
    _write_jsonl(manifest_out, final_rows)
    _write_json(report_out, report)

    print(f"Wrote focused manifest v3 to {manifest_out}")
    print(f"Wrote apply report to {report_out}")


if __name__ == "__main__":
    main()
