#!/usr/bin/env python3
"""Export A/C bucket manual-review candidates for curriculum manifest v2 cleanup."""

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


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def _excerpt(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _load_train_raw(train_raw_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(train_raw_path):
        key = (str(row.get("dataset") or "codecontests_train_wo_valid_big"), str(row["problem_id"]))
        prompt = str(row.get("canonical_prompt") or row["prompt"])
        out[key] = {
            "prompt": prompt,
            "prompt_title": _first_nonempty_line(prompt),
        }
    return out


def _load_eval_raw(eval_raw_path: Path, dataset_name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(eval_raw_path):
        prompt = str(row.get("canonical_prompt") or row["prompt"])
        out[str(row["problem_id"])] = {
            "dataset": dataset_name,
            "prompt": prompt,
            "prompt_title": _first_nonempty_line(prompt),
        }
    return out


def _load_snapshot(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("index_states", []):
        key = (str(row["dataset"]), str(row["problem_id"]))
        out[key] = row
    return out


def _bucket_rank(priority: str) -> int:
    return {"p0": 0, "p1": 1, "p2": 2, "p3": 3}.get(priority, 99)


def _load_eval_seed_asset(path: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = {
        "anchor_common": set(payload.get("anchor_common", [])),
        "only400": set(payload.get("only400", [])),
        "only580": set(payload.get("only580", [])),
        "high_partial_wa": set(payload.get("high_partial_wa", [])),
        "medium_partial_wa": set(payload.get("medium_partial_wa", [])),
        "partial_re_tle": set(payload.get("partial_re_tle", [])),
    }
    family_to_bucket = {
        "anchor_common": "A_retention",
        "only400": "A_retention",
        "only580": "B_near_miss",
        "high_partial_wa": "B_near_miss",
        "medium_partial_wa": "B_near_miss",
        "partial_re_tle": "C_hard_partial",
    }
    return groups, family_to_bucket


def _derive_priority(
    row: dict[str, Any],
    snapshot600: dict[str, Any] | None,
    snapshot620: dict[str, Any] | None,
    seed_groups: dict[str, set[str]],
) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    bucket = str(row["initial_bucket"])
    matched_seed_ids = [str(seed_id) for seed_id in row.get("matched_seed_ids", [])]
    matched_set = set(matched_seed_ids)

    step600_bucket = str((snapshot600 or {}).get("bucket", "unknown"))
    step620_bucket = str((snapshot620 or {}).get("bucket", "unknown"))
    visits620 = int((snapshot620 or {}).get("visits", 0) or 0)

    if bucket == "A_retention":
        if matched_set & seed_groups["only400"]:
            reasons.append("matched_only400_seed")
        if step620_bucket in {"C_hard_partial", "D_dead_hard"}:
            reasons.append("demoted_by_620")
        if step600_bucket == "A_retention" and step620_bucket in {"B_near_miss", "C_hard_partial", "D_dead_hard"}:
            reasons.append("lost_retention_bucket")

        if "matched_only400_seed" in reasons and "demoted_by_620" in reasons:
            return "p0", reasons, "move_to_U"
        if "demoted_by_620" in reasons or "lost_retention_bucket" in reasons:
            return "p1", reasons, "move_to_U"
        if "matched_only400_seed" in reasons:
            return "p1", reasons, "keep_A"
        return "p2", reasons, "keep_A"

    if bucket == "C_hard_partial":
        if step620_bucket in {"A_retention", "B_near_miss"}:
            reasons.append("promoted_by_620")
        if visits620 >= 2 and step620_bucket == "D_dead_hard":
            reasons.append("collapsed_to_dead")
        if visits620 >= 2 and step620_bucket == "C_hard_partial":
            reasons.append("stayed_hard_after_visits")

        if "promoted_by_620" in reasons:
            return "p1", reasons, "move_to_B"
        if "collapsed_to_dead" in reasons:
            return "p1", reasons, "drop"
        if "stayed_hard_after_visits" in reasons:
            return "p2", reasons, "move_to_U"
        return "p3", reasons, "keep_C"

    return "p3", reasons, ""


def _build_seed_summaries(
    row: dict[str, Any],
    eval_raw: dict[str, dict[str, Any]],
    seed_groups: dict[str, set[str]],
    *,
    max_prompt_chars: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    families = set(str(x) for x in row.get("matched_seed_families", []))
    for seed_id in row.get("matched_seed_ids", []):
        seed_id = str(seed_id)
        record = eval_raw.get(seed_id, {})
        memberships = [name for name, ids in seed_groups.items() if seed_id in ids]
        family = next((fam for fam in memberships if fam in families), memberships[0] if memberships else "")
        out.append(
            {
                "problem_id": seed_id,
                "family_group": family,
                "prompt_title": record.get("prompt_title", ""),
                "prompt_excerpt": _excerpt(record.get("prompt", ""), max_prompt_chars),
            }
        )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-seed-asset", type=Path, required=True)
    parser.add_argument("--train-raw", type=Path, required=True)
    parser.add_argument("--eval-raw", type=Path, required=True)
    parser.add_argument("--snapshot600", type=Path)
    parser.add_argument("--snapshot620", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="step600_v2")
    parser.add_argument("--review-buckets", default="A_retention,C_hard_partial")
    parser.add_argument("--max-prompt-chars", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    review_buckets = {bucket.strip() for bucket in args.review_buckets.split(",") if bucket.strip()}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_jsonl(args.manifest)
    train_raw = _load_train_raw(args.train_raw)
    eval_seed_groups, _ = _load_eval_seed_asset(args.eval_seed_asset)
    eval_raw = _load_eval_raw(args.eval_raw, dataset_name="codecontests_valid_big")
    snapshot600 = _load_snapshot(args.snapshot600)
    snapshot620 = _load_snapshot(args.snapshot620)

    candidates: list[dict[str, Any]] = []
    bucket_counts = Counter()
    priority_counts = Counter()

    for row in manifest_rows:
        if row["initial_bucket"] not in review_buckets:
            continue

        key = (str(row["dataset"]), str(row["problem_id"]))
        train_record = train_raw.get(key)
        if train_record is None:
            raise ValueError(f"Missing train raw record for {key}")

        state600 = snapshot600.get(key, {})
        state620 = snapshot620.get(key, {})
        priority, reasons, suggested_action = _derive_priority(row, state600, state620, eval_seed_groups)
        bucket_counts[row["initial_bucket"]] += 1
        priority_counts[priority] += 1

        candidates.append(
            {
                "dataset": key[0],
                "problem_id": key[1],
                "review_bucket": row["initial_bucket"],
                "initial_bucket": row["initial_bucket"],
                "primary_seed_family": row.get("primary_seed_family", ""),
                "matched_seed_ids": list(row.get("matched_seed_ids", [])),
                "matched_seed_families": list(row.get("matched_seed_families", [])),
                "retrieval_score_max": row.get("retrieval_score_max", 0.0),
                "prompt_sha256": row.get("prompt_sha256", ""),
                "train_prompt_title": train_record["prompt_title"],
                "train_prompt_excerpt": _excerpt(train_record["prompt"], args.max_prompt_chars),
                "matched_seed_summaries": _build_seed_summaries(
                    row,
                    eval_raw,
                    eval_seed_groups,
                    max_prompt_chars=args.max_prompt_chars,
                ),
                "step600_state": state600,
                "step620_state": state620,
                "review_priority": priority,
                "priority_reasons": reasons,
                "suggested_action": suggested_action,
                "allowed_actions": (
                    ["keep_A", "move_to_B", "move_to_U", "drop"]
                    if row["initial_bucket"] == "A_retention"
                    else ["keep_C", "move_to_B", "move_to_U", "drop"]
                ),
                "decision": "",
                "decision_reason": "",
                "reviewer": "",
            }
        )

    candidates.sort(
        key=lambda row: (
            _bucket_rank(row["review_priority"]),
            0 if row["review_bucket"] == "A_retention" else 1,
            -float(row.get("retrieval_score_max", 0.0)),
            row["problem_id"],
        )
    )

    combined_path = output_dir / f"ac_manual_review_candidates_{args.tag}.jsonl"
    a_path = output_dir / f"a_retention_review_candidates_{args.tag}.jsonl"
    c_path = output_dir / f"c_hard_partial_review_candidates_{args.tag}.jsonl"
    decisions_template_path = output_dir / f"ac_manual_review_decisions_{args.tag}.jsonl"
    summary_path = output_dir / f"ac_manual_review_summary_{args.tag}.json"

    _write_jsonl(combined_path, candidates)
    _write_jsonl(a_path, [row for row in candidates if row["review_bucket"] == "A_retention"])
    _write_jsonl(c_path, [row for row in candidates if row["review_bucket"] == "C_hard_partial"])
    _write_jsonl(
        decisions_template_path,
        [
            {
                "dataset": row["dataset"],
                "problem_id": row["problem_id"],
                "review_bucket": row["review_bucket"],
                "decision": "",
                "decision_reason": "",
                "reviewer": "",
            }
            for row in candidates
        ],
    )

    _write_json(
        summary_path,
        {
            "source_manifest": str(args.manifest),
            "review_buckets": sorted(review_buckets),
            "candidate_count": len(candidates),
            "bucket_counts": dict(bucket_counts),
            "priority_counts": dict(priority_counts),
            "combined_path": str(combined_path),
            "a_path": str(a_path),
            "c_path": str(c_path),
            "decisions_template_path": str(decisions_template_path),
        },
    )

    print(f"Wrote combined review candidates to {combined_path}")
    print(f"Wrote A review candidates to {a_path}")
    print(f"Wrote C review candidates to {c_path}")
    print(f"Wrote decision template to {decisions_template_path}")


if __name__ == "__main__":
    main()
