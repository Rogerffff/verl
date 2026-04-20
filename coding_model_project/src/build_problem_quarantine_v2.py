#!/usr/bin/env python3
"""Build problem_quarantine_v2 from v1 + auto-hard light screen + Claude review decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_V1 = Path("coding_model_project/data/problem_quarantine_v1.json")
DEFAULT_AUTO_HARD = Path("coding_model_project/data/quarantine_audit/review_assets_v1/problem_quarantine_auto_hard_v2.jsonl")
DEFAULT_CLAUDE = Path("coding_model_project/data/quarantine_audit/review_assets_v1/problem_quarantine_claude_decisions_v1.jsonl")
DEFAULT_OUTPUT = Path("coding_model_project/data/problem_quarantine_v2.json")
DEFAULT_SUMMARY = Path("coding_model_project/data/quarantine_audit/review_assets_v1/problem_quarantine_v2_build_summary.json")
DEFAULT_UNRESOLVED = Path("coding_model_project/data/quarantine_audit/review_assets_v1/problem_quarantine_unresolved_v1.jsonl")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _signal_reason(row: dict[str, Any]) -> str:
    merged = row.get("merged_signals", [])
    parts = []
    for signal in merged[:3]:
        parts.append(f"{signal['name']} x{signal['count']}")
    joined = ", ".join(parts) if parts else "high-confidence structural contamination"
    return f"Auto-screen clear-hard quarantine candidate ({joined})."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1)
    parser.add_argument("--auto-hard", type=Path, default=DEFAULT_AUTO_HARD)
    parser.add_argument("--claude-decisions", type=Path, nargs="+", default=[DEFAULT_CLAUDE])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--unresolved-output", type=Path, default=DEFAULT_UNRESOLVED)
    args = parser.parse_args()

    v1 = _read_json(args.v1)
    auto_hard_rows = _read_jsonl(args.auto_hard)
    claude_rows: list[dict[str, Any]] = []
    for decision_path in args.claude_decisions:
        claude_rows.extend(_read_jsonl(decision_path))

    claude_by_problem = {str(row["problem_id"]): row for row in claude_rows}

    hard_map: dict[str, dict[str, Any]] = {}
    caution_map: dict[str, dict[str, Any]] = {}

    source_counter = Counter()
    unresolved_rows: list[dict[str, Any]] = []

    for row in v1.get("hard_blacklist", []):
        problem_id = str(row["problem_id"])
        hard_map[problem_id] = dict(row)
        source_counter["v1_hard_blacklist"] += 1

    for row in v1.get("caution", []):
        problem_id = str(row["problem_id"])
        if problem_id not in hard_map:
            caution_map[problem_id] = dict(row)
            source_counter["v1_caution"] += 1

    for row in auto_hard_rows:
        problem_id = str(row["problem_id"])
        if problem_id in claude_by_problem:
            continue
        if problem_id in hard_map:
            continue
        hard_map[problem_id] = {
            "problem_id": problem_id,
            "reason": _signal_reason(row),
            "source": "light_screen_auto_hard_v1",
            "max_risk_score": row.get("max_risk_score"),
            "seen_splits": row.get("seen_splits", []),
            "tests_count_examples": row.get("tests_count_examples", []),
            "merged_signals": row.get("merged_signals", []),
        }
        source_counter["auto_hard_v1"] += 1

    for row in claude_rows:
        problem_id = str(row["problem_id"])
        decision = str(row["decision"])
        if decision == "hard_blacklist":
            hard_map[problem_id] = {
                "problem_id": problem_id,
                "reason": str(row["reason"]),
                "source": "claude_review_v1",
                "confidence": row.get("confidence"),
                "evidence_summary": row.get("evidence_summary"),
            }
            caution_map.pop(problem_id, None)
            source_counter["claude_hard_blacklist"] += 1
        elif decision == "caution":
            if problem_id not in hard_map:
                caution_map[problem_id] = {
                    "problem_id": problem_id,
                    "reason": str(row["reason"]),
                    "source": "claude_review_v1",
                    "confidence": row.get("confidence"),
                    "evidence_summary": row.get("evidence_summary"),
                }
                source_counter["claude_caution"] += 1
        elif decision in {"clean", "needs_more_evidence"}:
            unresolved_rows.append(row)
            source_counter[f"claude_{decision}"] += 1
        else:
            raise ValueError(f"Unexpected Claude decision: {decision}")

    hard_rows = sorted(hard_map.values(), key=lambda item: str(item["problem_id"]))
    caution_rows = sorted(caution_map.values(), key=lambda item: str(item["problem_id"]))
    unresolved_rows = sorted(unresolved_rows, key=lambda item: (str(item["decision"]), str(item["problem_id"])))

    payload = {
        "version": "problem_quarantine_v2",
        "updated_at": str(date.today()),
        "parent_version": str(v1.get("version", "problem_quarantine_v1")),
        "hard_blacklist": hard_rows,
        "caution": caution_rows,
        "unresolved": unresolved_rows,
    }
    _write_json(args.output, payload)

    _write_jsonl(args.unresolved_output, unresolved_rows)

    summary = {
        "v1_path": str(args.v1),
        "auto_hard_path": str(args.auto_hard),
        "claude_decisions_paths": [str(path) for path in args.claude_decisions],
        "output_path": str(args.output),
        "hard_blacklist_count": len(hard_rows),
        "caution_count": len(caution_rows),
        "unresolved_count": len(unresolved_rows),
        "source_counts": dict(source_counter),
    }
    _write_json(args.summary_output, summary)

    print(f"Wrote quarantine v2 to {args.output}")
    print(f"hard_blacklist={len(hard_rows)} caution={len(caution_rows)} unresolved={len(unresolved_rows)}")
    print(f"Wrote summary to {args.summary_output}")
    print(f"Wrote unresolved review rows to {args.unresolved_output}")


if __name__ == "__main__":
    main()
