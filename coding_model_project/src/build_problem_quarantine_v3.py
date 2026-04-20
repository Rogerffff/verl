#!/usr/bin/env python3
"""Apply normalized review ledger(s) to problem_quarantine_v2 and emit v3."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _decision_rank(decision: str) -> int:
    return {
        "hard_blacklist": 0,
        "caution": 1,
        "needs_more_evidence": 2,
        "clean": 3,
    }.get(decision, 9)


def _normalize_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "problem_id",
        "decision",
        "confidence",
        "reason",
        "evidence_summary",
        "review_scope",
        "slice_memberships",
        "decision_source",
        "queue_tag",
        "reviewed_at",
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Review row missing required keys {missing}: {row}")
        out.append(dict(row))
    return out


def _select_latest_reviews(review_files: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    by_problem: dict[str, dict[str, Any]] = {}
    seen_same_tag: dict[tuple[str, str], dict[str, Any]] = {}
    source_counts = Counter()

    for path in review_files:
        rows = _normalize_review_rows(_read_jsonl(path))
        for row in rows:
            problem_id = str(row["problem_id"])
            queue_tag = str(row["queue_tag"])
            key = (problem_id, queue_tag)
            prev_same = seen_same_tag.get(key)
            if prev_same is not None and prev_same["decision"] != row["decision"]:
                raise ValueError(
                    f"Conflicting decisions for {problem_id} within queue_tag={queue_tag}: "
                    f"{prev_same['decision']} vs {row['decision']}"
                )
            seen_same_tag[key] = row

            existing = by_problem.get(problem_id)
            if existing is None:
                by_problem[problem_id] = row
                source_counts[f"ledger::{queue_tag}"] += 1
                continue

            # latest reviewed_at wins across queue tags; tie-break manual decision severity rank.
            reviewed_at = str(row["reviewed_at"])
            existing_reviewed_at = str(existing["reviewed_at"])
            if reviewed_at > existing_reviewed_at or (
                reviewed_at == existing_reviewed_at
                and _decision_rank(str(row["decision"])) < _decision_rank(str(existing["decision"]))
            ):
                by_problem[problem_id] = row
    return by_problem, dict(source_counts)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    review_root = repo_root / "coding_model_project" / "phase_2_ GRPO" / "review_assets" / "step640_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=repo_root / "coding_model_project" / "data" / "problem_quarantine_v2.json",
    )
    parser.add_argument(
        "--review-ledgers",
        type=Path,
        nargs="+",
        default=[review_root / "problem_quarantine_review_ledger_step640_v1.jsonl"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "coding_model_project" / "data" / "problem_quarantine_v3.json",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=review_root / "problem_quarantine_v3_build_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base = _read_json(args.base)
    latest_reviews, review_source_counts = _select_latest_reviews(args.review_ledgers)

    hard_map = {str(row["problem_id"]): dict(row) for row in base.get("hard_blacklist", [])}
    caution_map = {str(row["problem_id"]): dict(row) for row in base.get("caution", [])}
    unresolved_map = {str(row["problem_id"]): dict(row) for row in base.get("unresolved", [])}

    transition_counter = Counter()

    for problem_id, row in latest_reviews.items():
        decision = str(row["decision"])
        prev_state = "unresolved" if problem_id in unresolved_map else "caution" if problem_id in caution_map else "hard_blacklist" if problem_id in hard_map else "clean"

        hard_map.pop(problem_id, None)
        caution_map.pop(problem_id, None)
        unresolved_map.pop(problem_id, None)

        if decision == "hard_blacklist":
            hard_map[problem_id] = {
                "problem_id": problem_id,
                "reason": str(row["reason"]),
                "source": "review_ledger_v3",
                "confidence": row.get("confidence"),
                "evidence_summary": row.get("evidence_summary"),
                "review_scope": row.get("review_scope"),
                "slice_memberships": list(row.get("slice_memberships", [])),
                "decision_source": row.get("decision_source"),
                "queue_tag": row.get("queue_tag"),
                "reviewed_at": row.get("reviewed_at"),
            }
        elif decision == "caution":
            caution_map[problem_id] = {
                "problem_id": problem_id,
                "reason": str(row["reason"]),
                "source": "review_ledger_v3",
                "confidence": row.get("confidence"),
                "evidence_summary": row.get("evidence_summary"),
                "review_scope": row.get("review_scope"),
                "slice_memberships": list(row.get("slice_memberships", [])),
                "decision_source": row.get("decision_source"),
                "queue_tag": row.get("queue_tag"),
                "reviewed_at": row.get("reviewed_at"),
            }
        elif decision in {"clean", "needs_more_evidence"}:
            unresolved_map[problem_id] = {
                "problem_id": problem_id,
                "decision": decision,
                "confidence": row.get("confidence"),
                "reason": str(row.get("reason", "")),
                "evidence_summary": str(row.get("evidence_summary", "")),
                "review_scope": row.get("review_scope"),
                "slice_memberships": list(row.get("slice_memberships", [])),
                "decision_source": row.get("decision_source"),
                "queue_tag": row.get("queue_tag"),
                "reviewed_at": row.get("reviewed_at"),
            }
        else:
            raise ValueError(f"Unsupported decision {decision!r} for {problem_id}")
        transition_counter[f"{prev_state}->{decision}"] += 1

    hard_rows = sorted(hard_map.values(), key=lambda row: str(row["problem_id"]))
    caution_rows = sorted(caution_map.values(), key=lambda row: str(row["problem_id"]))
    unresolved_rows = sorted(
        unresolved_map.values(),
        key=lambda row: (str(row.get("decision", "")), str(row["problem_id"])),
    )

    payload = {
        "version": "problem_quarantine_v3",
        "updated_at": str(date.today()),
        "parent_version": str(base.get("version", "problem_quarantine_v2")),
        "hard_blacklist": hard_rows,
        "caution": caution_rows,
        "unresolved": unresolved_rows,
    }
    _write_json(args.output, payload)

    summary = {
        "base_path": str(args.base),
        "review_ledgers": [str(path) for path in args.review_ledgers],
        "output_path": str(args.output),
        "hard_blacklist_count": len(hard_rows),
        "caution_count": len(caution_rows),
        "unresolved_count": len(unresolved_rows),
        "review_source_counts": review_source_counts,
        "transition_counter": dict(transition_counter),
    }
    _write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
