#!/usr/bin/env python3
"""Prepare full-review assets for the CodeContests test split."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_title(prompt: str) -> str:
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    return first[2:].strip() if first.startswith("# ") else first


def _prompt_excerpt(prompt: str, limit: int = 600) -> str:
    text = prompt.strip().replace("\r\n", "\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def _simplify_signals(signals: list[dict[str, Any]], *, keep_examples: int = 3) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for signal in signals:
        out.append(
            {
                "name": signal.get("name"),
                "severity": signal.get("severity"),
                "count": signal.get("count"),
                "raw_count": signal.get("raw_count"),
                "split_count": signal.get("split_count"),
                "split_names": signal.get("split_names"),
                "summary": signal.get("summary"),
                "examples": (signal.get("examples") or [])[:keep_examples],
            }
        )
    return out


def _quarantine_maps(quarantine: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    hard = {row["problem_id"]: row for row in quarantine.get("hard_blacklist", [])}
    caution = {row["problem_id"]: row for row in quarantine.get("caution", [])}
    unresolved = {row["problem_id"]: row for row in quarantine.get("unresolved", [])}
    return hard, caution, unresolved


def _current_state(
    problem_id: str,
    hard: dict[str, dict[str, Any]],
    caution: dict[str, dict[str, Any]],
    unresolved: dict[str, dict[str, Any]],
) -> tuple[str, str | None, dict[str, Any] | None]:
    if problem_id in hard:
        return "hard_blacklist", None, hard[problem_id]
    if problem_id in caution:
        return "caution", None, caution[problem_id]
    if problem_id in unresolved:
        row = unresolved[problem_id]
        return "unresolved", row.get("decision"), row
    return "none", None, None


def _priority(state: str, subdecision: str | None, has_screen_signal: bool, risk_score: int) -> tuple[int, str]:
    if state in {"hard_blacklist", "caution"}:
        return 0, "already_quarantined"
    if subdecision == "needs_more_evidence":
        return 1, "unresolved_needs_more_evidence"
    if has_screen_signal or risk_score > 0:
        return 1, "screen_detected_signal"
    if subdecision == "clean":
        return 2, "recheck_clean_unresolved"
    return 3, "full_test_review_remaining"


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-test-path",
        type=Path,
        default=root / "coding_model_project" / "data" / "raw" / "codecontests_test_raw.jsonl",
    )
    parser.add_argument(
        "--quarantine-path",
        type=Path,
        default=root / "coding_model_project" / "data" / "problem_quarantine_v3.json",
    )
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=root / "coding_model_project" / "data" / "quarantine_audit" / "problem_quarantine_candidates_v2.jsonl",
    )
    parser.add_argument(
        "--screen-rows-path",
        type=Path,
        default=root / "coding_model_project" / "data" / "quarantine_audit" / "problem_contract_screen_rows_v1.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "coding_model_project" / "data" / "quarantine_audit" / "test_review_assets_v1",
    )
    parser.add_argument("--queue-tag", default="test_full_review_v1")
    parser.add_argument("--num-shards", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_jsonl(args.raw_test_path)
    quarantine = json.load(args.quarantine_path.open())
    hard_map, caution_map, unresolved_map = _quarantine_maps(quarantine)
    candidates = {
        row["problem_id"]: row
        for row in _load_jsonl(args.candidates_path)
        if "codecontests_test" in (row.get("seen_splits") or [])
    }
    screen_rows = {
        row["problem_id"]: row
        for row in _load_jsonl(args.screen_rows_path)
        if row.get("split") == "codecontests_test"
    }

    queue: list[dict[str, Any]] = []
    for line_no, row in enumerate(raw_rows, start=1):
        problem_id = row["problem_id"]
        test_cases = row.get("test_cases") or {}
        tests = test_cases.get("tests") or []
        state, subdecision, state_row = _current_state(problem_id, hard_map, caution_map, unresolved_map)
        candidate_row = candidates.get(problem_id)
        screen_row = screen_rows.get(problem_id)
        has_screen_signal = screen_row is not None and bool(screen_row.get("signals"))
        risk_score = int((candidate_row or screen_row or {}).get("max_risk_score") or (screen_row or {}).get("risk_score") or 0)
        priority_rank, priority_reason = _priority(state, subdecision, has_screen_signal, risk_score)
        queue.append(
            {
                "queue_tag": args.queue_tag,
                "review_scope": "test_full_manual_review",
                "split": "codecontests_test",
                "problem_id": problem_id,
                "raw_path": str(args.raw_test_path),
                "raw_line_number": line_no,
                "prompt_title": _prompt_title(row.get("prompt", "")),
                "prompt_excerpt": _prompt_excerpt(row.get("prompt", "")),
                "prompt_sha256": row.get("prompt_sha256"),
                "test_format_type": test_cases.get("type"),
                "tests_count": len(tests),
                "current_quarantine_state": state,
                "current_quarantine_subdecision": subdecision,
                "current_quarantine_reason": (state_row or {}).get("reason"),
                "current_quarantine_source": (state_row or {}).get("source"),
                "current_quarantine_confidence": (state_row or {}).get("confidence"),
                "recommended_decision_from_screen": (candidate_row or screen_row or {}).get("recommended_decision"),
                "risk_score": risk_score,
                "seen_splits": (candidate_row or {}).get("seen_splits") or ["codecontests_test"],
                "test_split_signals": _simplify_signals((screen_row or {}).get("signals") or []),
                "merged_signals": _simplify_signals((candidate_row or {}).get("merged_signals") or []),
                "review_priority_rank": priority_rank,
                "review_priority_reason": priority_reason,
                "needs_rereview": state in {"hard_blacklist", "caution"} or subdecision == "needs_more_evidence",
            }
        )

    queue.sort(
        key=lambda row: (
            row["review_priority_rank"],
            -int(row["risk_score"] or 0),
            row["problem_id"],
        )
    )

    queue_path = output_dir / "problem_quarantine_test_full_review_queue_v1.jsonl"
    _write_jsonl(queue_path, queue)

    # Optional sharding for manual review load balancing.
    num_shards = max(1, int(args.num_shards))
    shard_size = math.ceil(len(queue) / num_shards)
    shard_paths: list[str] = []
    for shard_idx in range(num_shards):
        shard_rows = queue[shard_idx * shard_size : (shard_idx + 1) * shard_size]
        if not shard_rows:
            continue
        shard_path = output_dir / f"problem_quarantine_test_full_review_queue_v1_shard_{shard_idx + 1:02d}_of_{num_shards:02d}.jsonl"
        _write_jsonl(shard_path, shard_rows)
        shard_paths.append(str(shard_path))

    state_counts = Counter(row["current_quarantine_state"] for row in queue)
    subdecision_counts = Counter(
        row["current_quarantine_subdecision"] for row in queue if row["current_quarantine_subdecision"] is not None
    )
    screen_rec_counts = Counter(
        row["recommended_decision_from_screen"] for row in queue if row["recommended_decision_from_screen"] is not None
    )
    priority_counts = Counter(row["review_priority_reason"] for row in queue)
    signal_name_counts = Counter()
    for row in queue:
        for signal in row["test_split_signals"]:
            signal_name_counts[signal["name"]] += 1

    summary = {
        "queue_tag": args.queue_tag,
        "raw_test_path": str(args.raw_test_path),
        "quarantine_path": str(args.quarantine_path),
        "queue_path": str(queue_path),
        "shard_paths": shard_paths,
        "total_problems": len(queue),
        "state_counts": dict(state_counts),
        "subdecision_counts": dict(subdecision_counts),
        "screen_recommended_decision_counts": dict(screen_rec_counts),
        "review_priority_counts": dict(priority_counts),
        "test_split_signal_name_counts": dict(signal_name_counts),
        "problems_with_any_test_signal": sum(1 for row in queue if row["test_split_signals"]),
        "problems_in_existing_quarantine": sum(1 for row in queue if row["current_quarantine_state"] != "none"),
    }
    summary_path = output_dir / "problem_quarantine_test_full_review_summary_v1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
