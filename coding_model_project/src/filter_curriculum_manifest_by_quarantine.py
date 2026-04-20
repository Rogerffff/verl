#!/usr/bin/env python3
"""Filter a curriculum manifest with shared problem quarantine states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from problem_quarantine import caution_ids, hard_blacklist_ids, load_problem_quarantine


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--quarantine-json", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--remove-states",
        nargs="+",
        default=["hard_blacklist", "caution"],
        choices=["hard_blacklist", "caution", "unresolved"],
        help="Quarantine states to exclude from the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = _read_jsonl(args.input_manifest)
    quarantine = load_problem_quarantine(args.quarantine_json)

    remove_ids: set[str] = set()
    if "hard_blacklist" in args.remove_states:
        remove_ids |= hard_blacklist_ids(quarantine)
    if "caution" in args.remove_states:
        remove_ids |= caution_ids(quarantine)
    if "unresolved" in args.remove_states:
        remove_ids |= set(quarantine.get("unresolved_ids", set()))

    kept_rows: list[dict] = []
    removed_by_state: dict[str, list[str]] = {"hard_blacklist": [], "caution": [], "unresolved": []}

    hard_ids = hard_blacklist_ids(quarantine)
    caution_only_ids = caution_ids(quarantine)
    unresolved_only_ids = set(quarantine.get("unresolved_ids", set()))

    for row in rows:
        problem_id = str(row["problem_id"])
        if problem_id not in remove_ids:
            kept_rows.append(row)
            continue
        if problem_id in hard_ids:
            removed_by_state["hard_blacklist"].append(problem_id)
        elif problem_id in caution_only_ids:
            removed_by_state["caution"].append(problem_id)
        elif problem_id in unresolved_only_ids:
            removed_by_state["unresolved"].append(problem_id)

    _write_jsonl(args.output_manifest, kept_rows)

    summary = {
        "source_manifest": str(args.input_manifest),
        "output_manifest": str(args.output_manifest),
        "quarantine_path": str(args.quarantine_json),
        "quarantine_version": quarantine.get("version"),
        "remove_states": list(args.remove_states),
        "rows_before": len(rows),
        "rows_after": len(kept_rows),
        "before_bucket_counts": dict(Counter(str(row["initial_bucket"]) for row in rows)),
        "after_bucket_counts": dict(Counter(str(row["initial_bucket"]) for row in kept_rows)),
        "removed_hard_count": len(removed_by_state["hard_blacklist"]),
        "removed_hard_problem_ids": sorted(removed_by_state["hard_blacklist"]),
        "removed_caution_count": len(removed_by_state["caution"]),
        "removed_caution_problem_ids": sorted(removed_by_state["caution"]),
        "removed_unresolved_count": len(removed_by_state["unresolved"]),
        "removed_unresolved_problem_ids": sorted(removed_by_state["unresolved"]),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
