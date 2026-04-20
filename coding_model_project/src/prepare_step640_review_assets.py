#!/usr/bin/env python3
"""Prepare pre-step640 review assets for quarantine v3 and curriculum actions.

This script freezes the latest complete local assets (currently step600/620),
builds a deduplicated eval review queue, and emits separate bucket/anchor
review queues plus decision templates that match current downstream consumers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from problem_quarantine import load_problem_quarantine


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_per_problem(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["problem_id"]): row for row in _read_jsonl(path)}


def _build_quarantine_lookup(quarantine: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in quarantine.get("hard_blacklist", []):
        pid = str(row["problem_id"])
        out[pid] = {
            "state": "hard_blacklist",
            "subdecision": "hard_blacklist",
            "reason": str(row.get("reason", "")),
            "source": str(row.get("source", "")),
        }
    for row in quarantine.get("caution", []):
        pid = str(row["problem_id"])
        if pid in out:
            continue
        out[pid] = {
            "state": "caution",
            "subdecision": "caution",
            "reason": str(row.get("reason", "")),
            "source": str(row.get("source", "")),
        }
    for row in quarantine.get("unresolved", []):
        pid = str(row["problem_id"])
        if pid in out:
            continue
        out[pid] = {
            "state": "unresolved",
            "subdecision": str(row.get("decision", "")),
            "reason": str(row.get("reason", "")),
            "source": str(row.get("source", "")),
        }
    return out


def _build_delta69_memberships(eval_seed_asset: dict[str, Any]) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = defaultdict(list)
    for name in ("anchor_common", "only400", "only580"):
        for problem_id in eval_seed_asset.get(name, []):
            memberships[str(problem_id)].append(name)
    return memberships


def _delta69_transition(role600: dict[str, Any] | None, role620: dict[str, Any] | None) -> str:
    solved600 = bool((role600 or {}).get("accepted", False))
    solved620 = bool((role620 or {}).get("accepted", False))
    if solved600 and not solved620:
        return "lost_by_620"
    if not solved600 and solved620:
        return "gained_by_620"
    if solved600 and solved620:
        return "stable_solved"
    return "stable_unsolved"


def _eval_priority(label: dict[str, Any]) -> str:
    state = label["current_quarantine_state"]
    subdecision = label["current_quarantine_subdecision"]
    memberships = set(label["slice_memberships"])
    roles = set(label["latest_eval_roles"])
    if state == "caution":
        return "p0"
    if subdecision == "needs_more_evidence":
        return "p0"
    if "eval_priority_review" in memberships:
        return "p1"
    if "delta69" in memberships and ("lost_by_620" in roles or "gained_by_620" in roles):
        return "p1"
    if "delta69" in memberships:
        return "p2"
    return "p3"


def _bucket_priority(row: dict[str, Any], quarantine_label: dict[str, Any], delta69_ids: set[str]) -> tuple[str, bool]:
    reasons = set(str(x) for x in row.get("priority_reasons", []))
    matched_seed_ids = set(str(x) for x in row.get("matched_seed_ids", []))
    quarantine_state = quarantine_label["current_quarantine_state"]
    subdecision = quarantine_label["current_quarantine_subdecision"]

    needs_rereview = False
    if quarantine_state == "caution":
        needs_rereview = True
    if subdecision == "needs_more_evidence":
        needs_rereview = True
    if reasons & {"matched_only400_seed", "demoted_by_620", "lost_retention_bucket", "collapsed_to_dead", "promoted_by_620"}:
        needs_rereview = True
    if matched_seed_ids & delta69_ids:
        needs_rereview = True

    if quarantine_state == "caution" or subdecision == "needs_more_evidence":
        return "p0", needs_rereview
    if reasons & {"demoted_by_620", "lost_retention_bucket", "collapsed_to_dead"}:
        return "p1", needs_rereview
    if reasons & {"matched_only400_seed", "promoted_by_620"}:
        return "p2", needs_rereview
    return "p3", needs_rereview


def _anchor_priority(row: dict[str, Any], quarantine_label: dict[str, Any]) -> tuple[str, bool]:
    seed_summaries = row.get("matched_seed_summaries", [])
    seed_priorities = {str(seed.get("seed_priority", "")) for seed in seed_summaries}
    quarantine_state = quarantine_label["current_quarantine_state"]
    subdecision = quarantine_label["current_quarantine_subdecision"]
    needs_rereview = False
    if quarantine_state == "caution":
        needs_rereview = True
    if subdecision == "needs_more_evidence":
        needs_rereview = True
    if seed_priorities & {"p0", "p1"}:
        needs_rereview = True
    if quarantine_state == "caution" or subdecision == "needs_more_evidence":
        return "p0", needs_rereview
    if "p0" in seed_priorities:
        return "p1", needs_rereview
    if "p1" in seed_priorities:
        return "p2", needs_rereview
    return "p3", needs_rereview


def _review_sort_key(priority: str, score: float, problem_id: str) -> tuple[int, float, str]:
    order = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
    return (order.get(priority, 9), -score, problem_id)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    phase2_root = repo_root / "coding_model_project" / "phase_2_ GRPO"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-tag", default="step640_v1")
    parser.add_argument("--freeze-basis", default="latest_complete_assets_step620_v2")
    parser.add_argument("--is-pre-step640", action="store_true", default=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=phase2_root / "review_assets" / "step640_v1",
    )
    parser.add_argument(
        "--quarantine-path",
        type=Path,
        default=repo_root / "coding_model_project" / "data" / "problem_quarantine_v2.json",
    )
    parser.add_argument(
        "--eval-priority-review",
        type=Path,
        default=repo_root
        / "coding_model_project"
        / "data"
        / "quarantine_audit"
        / "review_assets_v1"
        / "problem_quarantine_eval_priority_review_v1.jsonl",
    )
    parser.add_argument(
        "--eval-seed-asset",
        type=Path,
        default=phase2_root
        / "curriculum_assets"
        / "step600_v3_review_local"
        / "curriculum_eval_seed_asset_step580_v1.json",
    )
    parser.add_argument(
        "--step600-delta69",
        type=Path,
        default=phase2_root
        / "curriculum_assets"
        / "step600_v3_review_local"
        / "step600_delta69_per_problem.jsonl",
    )
    parser.add_argument(
        "--step620-delta69",
        type=Path,
        default=phase2_root
        / "curriculum_assets"
        / "step600_v3_review_local"
        / "step620_v2_delta69_per_problem.jsonl",
    )
    parser.add_argument(
        "--a-candidates",
        type=Path,
        default=phase2_root
        / "curriculum_assets"
        / "step600_v2_review_local"
        / "a_retention_review_candidates_step600_v2.jsonl",
    )
    parser.add_argument(
        "--c-candidates",
        type=Path,
        default=phase2_root
        / "curriculum_assets"
        / "step600_v3_review_local"
        / "c_hard_partial_review_candidates_step600_v3.jsonl",
    )
    parser.add_argument(
        "--anchor-candidates",
        type=Path,
        default=phase2_root
        / "curriculum_assets"
        / "step600_v3_review_local"
        / "anchor_common_stabilizer_candidates_step600_v3.jsonl",
    )
    parser.add_argument(
        "--snapshot600",
        type=Path,
        default=phase2_root
        / "curriculum_assets"
        / "step600_v3_review_local"
        / "curriculum_state_step_600.json",
    )
    parser.add_argument(
        "--snapshot620",
        type=Path,
        default=phase2_root
        / "curriculum_assets"
        / "step600_v3_review_local"
        / "curriculum_state_step_620.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    quarantine = load_problem_quarantine(args.quarantine_path)
    quarantine_lookup = _build_quarantine_lookup(quarantine)
    eval_priority_rows = _read_jsonl(args.eval_priority_review)
    eval_priority_by_problem = {str(row["problem_id"]): row for row in eval_priority_rows}

    eval_seed_asset = _read_json(args.eval_seed_asset)
    delta69_memberships = _build_delta69_memberships(eval_seed_asset)
    delta69_ids = set(delta69_memberships)
    step600_delta69 = _load_per_problem(args.step600_delta69)
    step620_delta69 = _load_per_problem(args.step620_delta69)

    review_freeze_manifest = {
        "queue_tag": args.queue_tag,
        "freeze_basis": args.freeze_basis,
        "is_pre_step640": bool(args.is_pre_step640),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quarantine_version": quarantine.get("version"),
        "inputs": {
            "quarantine_path": str(args.quarantine_path),
            "eval_priority_review": str(args.eval_priority_review),
            "eval_seed_asset": str(args.eval_seed_asset),
            "step600_delta69": str(args.step600_delta69),
            "step620_delta69": str(args.step620_delta69),
            "a_candidates": str(args.a_candidates),
            "c_candidates": str(args.c_candidates),
            "anchor_candidates": str(args.anchor_candidates),
            "snapshot600": str(args.snapshot600),
            "snapshot620": str(args.snapshot620),
        },
        "frozen_slice_versions": {
            "delta69_problem_count": len(delta69_ids),
            "eval_priority_problem_count": len(eval_priority_rows),
        },
        "conflict_resolution_policy": {
            "same_queue_tag_conflict": "error",
            "cross_queue_tag_precedence": "latest_reviewed_at_wins",
            "manual_vs_auto": "manual_review_overrides_auto_inference",
        },
    }

    eval_queue_map: dict[str, dict[str, Any]] = {}
    for problem_id in sorted(delta69_ids | set(eval_priority_by_problem)):
        label = quarantine_lookup.get(
            problem_id,
            {
                "state": "",
                "subdecision": "",
                "reason": "",
                "source": "",
            },
        )
        slice_memberships = list(delta69_memberships.get(problem_id, []))
        latest_eval_roles: list[str] = []
        if problem_id in delta69_ids:
            slice_memberships.append("delta69")
            latest_eval_roles.append(
                _delta69_transition(step600_delta69.get(problem_id), step620_delta69.get(problem_id))
            )
        eval_row = eval_priority_by_problem.get(problem_id)
        if eval_row is not None:
            slice_memberships.append("eval_priority_review")
            latest_eval_roles.extend(f"seen_in::{split}" for split in eval_row.get("seen_splits", []))

        record = {
            "problem_id": problem_id,
            "dataset": "codecontests_valid_big",
            "slice_memberships": sorted(set(slice_memberships)),
            "current_quarantine_state": label["state"],
            "current_quarantine_subdecision": label["subdecision"],
            "needs_rereview": (
                label["state"] == "caution"
                or label["subdecision"] == "needs_more_evidence"
                or "eval_priority_review" in slice_memberships
            ),
            "latest_eval_roles": sorted(set(latest_eval_roles)),
            "review_reasons": [],
            "decision": "",
            "confidence": None,
            "reason": "",
            "evidence_summary": "",
            "queue_tag": args.queue_tag,
            "source_freeze_manifest": "review_freeze_manifest.json",
        }
        if eval_row is not None:
            record["review_reasons"].append("eval_priority_review")
            record["eval_priority_row"] = eval_row
        delta600 = step600_delta69.get(problem_id)
        delta620 = step620_delta69.get(problem_id)
        if delta600 or delta620:
            record["delta69_status"] = {
                "step600_accepted": bool((delta600 or {}).get("accepted", False)),
                "step620_accepted": bool((delta620 or {}).get("accepted", False)),
                "step600_pass_ratio": float((delta600 or {}).get("pass_ratio_all", 0.0) or 0.0),
                "step620_pass_ratio": float((delta620 or {}).get("pass_ratio_all", 0.0) or 0.0),
            }
        record["review_priority"] = _eval_priority(record)
        eval_queue_map[problem_id] = record

    eval_queue_all = sorted(
        eval_queue_map.values(),
        key=lambda row: _review_sort_key(
            str(row["review_priority"]),
            float((row.get("eval_priority_row") or {}).get("max_risk_score", 0.0)),
            str(row["problem_id"]),
        ),
    )
    eval_queue = [row for row in eval_queue_all if row["needs_rereview"]]

    bucket_candidates: list[dict[str, Any]] = []
    for path in (args.a_candidates, args.c_candidates):
        for row in _read_jsonl(path):
            problem_id = str(row["problem_id"])
            label = quarantine_lookup.get(
                problem_id,
                {
                    "state": "",
                    "subdecision": "",
                    "reason": "",
                    "source": "",
                },
            )
            priority, needs_rereview = _bucket_priority(
                row,
                {
                    "current_quarantine_state": label["state"],
                    "current_quarantine_subdecision": label["subdecision"],
                },
                delta69_ids,
            )
            out = dict(row)
            out["current_quarantine_state"] = label["state"]
            out["current_quarantine_subdecision"] = label["subdecision"]
            out["needs_rereview"] = needs_rereview
            out["queue_tag"] = args.queue_tag
            out["review_priority_v3"] = priority
            out["latest_eval_overlap"] = sorted(set(str(x) for x in row.get("matched_seed_ids", [])) & delta69_ids)
            bucket_candidates.append(out)

    bucket_review_queue = [
        row
        for row in bucket_candidates
        if row["current_quarantine_state"] == "caution"
        or row["current_quarantine_subdecision"] == "needs_more_evidence"
        or row["review_priority_v3"] in {"p0", "p1"}
    ]
    bucket_review_queue.sort(
        key=lambda row: _review_sort_key(
            str(row["review_priority_v3"]),
            float(row.get("retrieval_score_max", 0.0)),
            str(row["problem_id"]),
        )
    )

    bucket_actions_template = [
        {
            "dataset": row["dataset"],
            "problem_id": row["problem_id"],
            "review_bucket": row["review_bucket"],
            "decision": "",
            "decision_reason": "",
            "reviewer": "",
            "queue_tag": args.queue_tag,
        }
        for row in bucket_review_queue
    ]
    a_bucket_review_queue = [row for row in bucket_review_queue if row["review_bucket"] == "A_retention"]
    c_bucket_review_queue = [row for row in bucket_review_queue if row["review_bucket"] == "C_hard_partial"]
    a_bucket_actions_template = [row for row in bucket_actions_template if row["review_bucket"] == "A_retention"]
    c_bucket_actions_template = [row for row in bucket_actions_template if row["review_bucket"] == "C_hard_partial"]

    anchor_candidates: list[dict[str, Any]] = []
    for row in _read_jsonl(args.anchor_candidates):
        problem_id = str(row["problem_id"])
        label = quarantine_lookup.get(
            problem_id,
            {
                "state": "",
                "subdecision": "",
                "reason": "",
                "source": "",
            },
        )
        priority, needs_rereview = _anchor_priority(
            row,
            {
                "current_quarantine_state": label["state"],
                "current_quarantine_subdecision": label["subdecision"],
            },
        )
        out = dict(row)
        out["current_quarantine_state"] = label["state"]
        out["current_quarantine_subdecision"] = label["subdecision"]
        out["needs_rereview"] = needs_rereview
        out["queue_tag"] = args.queue_tag
        out["review_priority_v3"] = priority
        anchor_candidates.append(out)

    anchor_review_queue = [
        row
        for row in anchor_candidates
        if row["current_quarantine_state"] == "caution"
        or row["current_quarantine_subdecision"] == "needs_more_evidence"
        or row["review_priority_v3"] in {"p0", "p1"}
    ]
    anchor_review_queue.sort(
        key=lambda row: _review_sort_key(
            str(row["review_priority_v3"]),
            float(row.get("retrieval_score_max", 0.0)),
            str(row["problem_id"]),
        )
    )
    anchor_actions_template = [
        {
            "dataset": row["dataset"],
            "problem_id": row["problem_id"],
            "review_bucket": row["review_bucket"],
            "decision": "",
            "decision_reason": "",
            "reviewer": "",
            "queue_tag": args.queue_tag,
        }
        for row in anchor_review_queue
    ]

    summary = {
        "queue_tag": args.queue_tag,
        "freeze_basis": args.freeze_basis,
        "is_pre_step640": bool(args.is_pre_step640),
        "eval_review_queue_count": len(eval_queue),
        "eval_review_all_count": len(eval_queue_all),
        "eval_priority_counts": dict(Counter(str(row["review_priority"]) for row in eval_queue_all)),
        "eval_quarantine_state_counts": dict(
            Counter(
                f"{row['current_quarantine_state']}::{row['current_quarantine_subdecision']}"
                for row in eval_queue_all
            )
        ),
        "bucket_review_queue_count": len(bucket_review_queue),
        "bucket_priority_counts": dict(Counter(str(row["review_priority_v3"]) for row in bucket_review_queue)),
        "bucket_counts": dict(Counter(str(row["review_bucket"]) for row in bucket_review_queue)),
        "anchor_review_queue_count": len(anchor_review_queue),
        "anchor_priority_counts": dict(Counter(str(row["review_priority_v3"]) for row in anchor_review_queue)),
        "files": {
            "freeze_manifest": str(output_dir / f"review_freeze_manifest_{args.queue_tag}.json"),
            "eval_review_queue": str(output_dir / f"eval_review_queue_{args.queue_tag}.jsonl"),
            "eval_review_all": str(output_dir / f"eval_review_queue_all_{args.queue_tag}.jsonl"),
            "eval_review_template": str(output_dir / f"problem_quarantine_review_ledger_template_{args.queue_tag}.jsonl"),
            "bucket_review_queue": str(output_dir / f"bucket_actions_review_queue_{args.queue_tag}.jsonl"),
            "bucket_review_all": str(output_dir / f"bucket_actions_review_all_{args.queue_tag}.jsonl"),
            "bucket_actions_template": str(output_dir / f"bucket_actions_template_{args.queue_tag}.jsonl"),
            "a_bucket_review_queue": str(output_dir / f"a_bucket_actions_review_queue_{args.queue_tag}.jsonl"),
            "c_bucket_review_queue": str(output_dir / f"c_bucket_actions_review_queue_{args.queue_tag}.jsonl"),
            "a_bucket_actions_template": str(output_dir / f"a_bucket_actions_template_{args.queue_tag}.jsonl"),
            "c_bucket_actions_template": str(output_dir / f"c_bucket_actions_template_{args.queue_tag}.jsonl"),
            "anchor_review_queue": str(output_dir / f"anchor_actions_review_queue_{args.queue_tag}.jsonl"),
            "anchor_review_all": str(output_dir / f"anchor_actions_review_all_{args.queue_tag}.jsonl"),
            "anchor_actions_template": str(output_dir / f"anchor_actions_template_{args.queue_tag}.jsonl"),
        },
    }

    eval_template = [
        {
            "problem_id": row["problem_id"],
            "decision": "",
            "confidence": None,
            "reason": "",
            "evidence_summary": "",
            "review_scope": "eval_critical",
            "slice_memberships": row["slice_memberships"],
            "decision_source": "",
            "queue_tag": args.queue_tag,
            "reviewed_at": "",
        }
        for row in eval_queue
    ]

    _write_json(output_dir / f"review_freeze_manifest_{args.queue_tag}.json", review_freeze_manifest)
    _write_jsonl(output_dir / f"eval_review_queue_{args.queue_tag}.jsonl", eval_queue)
    _write_jsonl(output_dir / f"eval_review_queue_all_{args.queue_tag}.jsonl", eval_queue_all)
    _write_jsonl(output_dir / f"problem_quarantine_review_ledger_template_{args.queue_tag}.jsonl", eval_template)
    _write_jsonl(output_dir / f"bucket_actions_review_queue_{args.queue_tag}.jsonl", bucket_review_queue)
    _write_jsonl(output_dir / f"bucket_actions_review_all_{args.queue_tag}.jsonl", bucket_candidates)
    _write_jsonl(output_dir / f"bucket_actions_template_{args.queue_tag}.jsonl", bucket_actions_template)
    _write_jsonl(output_dir / f"a_bucket_actions_review_queue_{args.queue_tag}.jsonl", a_bucket_review_queue)
    _write_jsonl(output_dir / f"c_bucket_actions_review_queue_{args.queue_tag}.jsonl", c_bucket_review_queue)
    _write_jsonl(output_dir / f"a_bucket_actions_template_{args.queue_tag}.jsonl", a_bucket_actions_template)
    _write_jsonl(output_dir / f"c_bucket_actions_template_{args.queue_tag}.jsonl", c_bucket_actions_template)
    _write_jsonl(output_dir / f"anchor_actions_review_queue_{args.queue_tag}.jsonl", anchor_review_queue)
    _write_jsonl(output_dir / f"anchor_actions_review_all_{args.queue_tag}.jsonl", anchor_candidates)
    _write_jsonl(output_dir / f"anchor_actions_template_{args.queue_tag}.jsonl", anchor_actions_template)
    _write_json(output_dir / f"review_asset_summary_{args.queue_tag}.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
