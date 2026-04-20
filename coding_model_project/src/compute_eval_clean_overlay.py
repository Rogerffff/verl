#!/usr/bin/env python3
"""Compute raw + quarantine-clean overlay summaries from eval per-problem results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from problem_quarantine import caution_ids, hard_blacklist_ids, load_problem_quarantine


def _load_per_problem(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed_rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed_rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_rows.append({"line_no": line_no, "preview": line[:200]})

    by_problem_id: dict[str, dict[str, Any]] = {}
    problem_id_counter = Counter()
    for row in parsed_rows:
        problem_id = str(row.get("problem_id"))
        problem_id_counter[problem_id] += 1
        # Keep the last valid row for a duplicated problem_id.
        by_problem_id[problem_id] = row

    duplicate_problem_ids = sorted(
        problem_id for problem_id, count in problem_id_counter.items() if count > 1
    )
    metadata = {
        "input_row_count": len(parsed_rows) + len(bad_rows),
        "valid_json_row_count": len(parsed_rows),
        "bad_row_count": len(bad_rows),
        "bad_rows_preview": bad_rows[:5],
        "deduped_problem_count": len(by_problem_id),
        "duplicate_problem_ids": duplicate_problem_ids,
    }
    return list(by_problem_id.values()), metadata


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_count = len(rows)
    accepted_count = sum(1 for row in rows if bool(row.get("accepted", False)))
    pass_ratios = [float(row.get("pass_ratio", 0.0)) for row in rows]
    return {
        "count": raw_count,
        "accepted_count": accepted_count,
        "accepted_at_1": accepted_count / raw_count if raw_count else 0.0,
        "pass_ratio_mean": (sum(pass_ratios) / raw_count) if raw_count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute quarantine clean overlay for eval results.")
    parser.add_argument("--per-problem-jsonl", type=Path, required=True)
    parser.add_argument("--quarantine-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--remove-states",
        nargs="+",
        default=["hard_blacklist", "caution"],
        choices=["hard_blacklist", "caution", "unresolved"],
    )
    args = parser.parse_args()

    rows, row_metadata = _load_per_problem(args.per_problem_jsonl)
    quarantine = load_problem_quarantine(args.quarantine_json)

    remove_ids: set[str] = set()
    if "hard_blacklist" in args.remove_states:
        remove_ids |= hard_blacklist_ids(quarantine)
    if "caution" in args.remove_states:
        remove_ids |= caution_ids(quarantine)
    if "unresolved" in args.remove_states:
        remove_ids |= set(quarantine.get("unresolved_ids", set()))

    removed_rows = [row for row in rows if str(row.get("problem_id")) in remove_ids]
    clean_rows = [row for row in rows if str(row.get("problem_id")) not in remove_ids]

    overlay = {
        "quarantine_path": str(args.quarantine_json),
        "remove_states": list(args.remove_states),
        **row_metadata,
        "raw_count": len(rows),
        "removed_count": len(removed_rows),
        "removed_problem_ids": sorted(str(row.get("problem_id")) for row in removed_rows),
        "clean_count": len(clean_rows),
        "raw_metrics": _metrics(rows),
        "clean_metrics": _metrics(clean_rows),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(overlay, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(overlay, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
