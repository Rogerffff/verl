#!/usr/bin/env python3
"""Prepare focused manifest v3 review bundles from step600_v2 assets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from problem_quarantine import hard_blacklist_ids, load_problem_quarantine
from step580_curriculum_builder import (
    ProblemKey,
    SeedEntry,
    _load_hash_set,
    _load_train_problems,
    _read_jsonl,
    _top_candidates_for_seed,
    _validate_train_hygiene,
    _write_json,
    _write_jsonl,
    build_idf,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_snapshot(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = _load_json(path)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("index_states", []):
        out[(str(row["dataset"]), str(row["problem_id"]))] = row
    return out


def _load_train_prompt_map(train_raw_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(train_raw_path):
        dataset = str(row.get("dataset") or "codecontests_train_wo_valid_big")
        key = (dataset, str(row["problem_id"]))
        prompt = str(row.get("canonical_prompt") or row["prompt"])
        title = ""
        for line in prompt.splitlines():
            if line.strip():
                title = line.strip()[:200]
                break
        out[key] = {"prompt": prompt, "title": title}
    return out


def _excerpt(text: str, max_chars: int = 600) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _load_eval_raw(eval_raw_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(eval_raw_path):
        prompt = str(row.get("canonical_prompt") or row["prompt"])
        title = ""
        for line in prompt.splitlines():
            if line.strip():
                title = line.strip()[:200]
                break
        out[str(row["problem_id"])] = {"prompt": prompt, "title": title}
    return out


def _load_per_problem(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["problem_id"]): row for row in _read_jsonl(path)}


def _collapse_anchor_candidates(
    key_strs: list[str],
    occurrences: dict[str, list[dict[str, Any]]],
    prompt_sha_by_key: dict[str, str],
    train_prompt_map: dict[tuple[str, str], dict[str, Any]],
    eval_raw: dict[str, dict[str, Any]],
    snapshot600: dict[tuple[str, str], dict[str, Any]],
    snapshot620: dict[tuple[str, str], dict[str, Any]],
    seed_priorities: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key_str in key_strs:
        matches = occurrences[key_str]
        matches_sorted = sorted(matches, key=lambda item: item["retrieval_score"], reverse=True)
        dataset, problem_id = key_str.split("::", 1)
        key = (dataset, problem_id)
        train_record = train_prompt_map[key]
        matched_seed_ids = sorted({item["matched_seed_id"] for item in matches_sorted})
        matched_seed_families = sorted({item["matched_seed_family"] for item in matches_sorted})
        seed_summaries = []
        for seed_id in matched_seed_ids:
            raw = eval_raw.get(seed_id, {})
            seed_summaries.append(
                {
                    "problem_id": seed_id,
                    "family_group": "anchor_common",
                    "seed_priority": seed_priorities.get(seed_id, "p2"),
                    "prompt_title": raw.get("title", ""),
                    "prompt_excerpt": _excerpt(raw.get("prompt", "")),
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "problem_id": problem_id,
                "review_bucket": "anchor_common_stabilizer",
                "proposed_bucket": "A_retention",
                "primary_seed_family": "anchor_common",
                "matched_seed_ids": matched_seed_ids,
                "matched_seed_families": matched_seed_families,
                "retrieval_score_max": matches_sorted[0]["retrieval_score"],
                "prompt_sha256": prompt_sha_by_key[key_str],
                "train_prompt_title": train_record["title"],
                "train_prompt_excerpt": _excerpt(train_record["prompt"]),
                "matched_seed_summaries": seed_summaries,
                "step600_state": snapshot600.get(key, {}),
                "step620_state": snapshot620.get(key, {}),
                "decision": "",
                "decision_reason": "",
                "reviewer": "",
                "allowed_actions": ["add_to_A", "keep_out", "drop"],
            }
        )
    rows.sort(key=lambda row: (-float(row["retrieval_score_max"]), row["problem_id"]))
    return rows


def _round_robin_select(seed_matches: dict[str, list[dict[str, Any]]], target_size: int) -> list[str]:
    positions = {seed_id: 0 for seed_id in seed_matches}
    active = deque(sorted(seed_matches))
    selected: list[str] = []
    selected_set: set[str] = set()

    while active and len(selected) < target_size:
        seed_id = active.popleft()
        matches = seed_matches[seed_id]
        pos = positions[seed_id]
        picked = None
        while pos < len(matches):
            key_str = matches[pos]["key_str"]
            pos += 1
            if key_str in selected_set:
                continue
            picked = key_str
            break
        positions[seed_id] = pos
        if picked is not None:
            selected.append(picked)
            selected_set.add(picked)
        if pos < len(matches):
            active.append(seed_id)
    return selected


def _build_anchor_priority_map(
    anchor_common_ids: list[str],
    step600_delta69: dict[str, dict[str, Any]],
    step620_delta69: dict[str, dict[str, Any]],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for seed_id in anchor_common_ids:
        s600 = bool(step600_delta69.get(seed_id, {}).get("accepted", False))
        s620 = bool(step620_delta69.get(seed_id, {}).get("accepted", False))
        if s600 and not s620:
            out[seed_id] = "p0"
        elif not s620:
            out[seed_id] = "p1"
        else:
            out[seed_id] = "p2"
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-v2", type=Path, required=True)
    parser.add_argument("--c-candidates-v2", type=Path, required=True)
    parser.add_argument("--c-decisions-v2", type=Path, required=True)
    parser.add_argument("--eval-seed-asset", type=Path, required=True)
    parser.add_argument("--step600-delta69", type=Path, required=True)
    parser.add_argument("--step620-delta69-v2", type=Path, required=True)
    parser.add_argument("--snapshot600", type=Path, required=True)
    parser.add_argument("--snapshot620-v2", type=Path, required=True)
    parser.add_argument("--train-raw", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--valid-big-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--eval-raw", type=Path, required=True)
    parser.add_argument("--quarantine-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-c", type=int, default=40)
    parser.add_argument("--anchor-candidate-limit", type=int, default=80)
    parser.add_argument("--anchor-top-k-per-seed", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_v2 = _read_jsonl(args.manifest_v2)
    manifest_keys = {f"{row['dataset']}::{row['problem_id']}" for row in manifest_v2}
    c_candidates_v2 = {
        (str(row["dataset"]), str(row["problem_id"])): row for row in _read_jsonl(args.c_candidates_v2)
    }
    c_decisions_v2 = {
        (str(row["dataset"]), str(row["problem_id"])): row for row in _read_jsonl(args.c_decisions_v2)
    }
    snapshot600 = _load_snapshot(args.snapshot600)
    snapshot620 = _load_snapshot(args.snapshot620_v2)
    eval_seed_asset = _load_json(args.eval_seed_asset)
    step600_delta69 = _load_per_problem(args.step600_delta69)
    step620_delta69 = _load_per_problem(args.step620_delta69_v2)

    # Agent 1 bundle: re-trim current keep_C rows to a smaller v3 target.
    c_rows_v3: list[dict[str, Any]] = []
    for key, decision_row in c_decisions_v2.items():
        if str(decision_row.get("decision")) != "keep_C":
            continue
        candidate = dict(c_candidates_v2[key])
        candidate["current_v2_decision"] = "keep_C"
        candidate["target_keep_c_v3"] = args.target_c
        candidate["decision"] = ""
        candidate["decision_reason"] = ""
        candidate["reviewer"] = ""
        c_rows_v3.append(candidate)
    c_rows_v3.sort(key=lambda row: (-float(row.get("retrieval_score_max", 0.0)), row["problem_id"]))

    c_candidate_path = output_dir / "c_hard_partial_review_candidates_step600_v3.jsonl"
    c_decision_template_path = output_dir / "c_hard_partial_review_decisions_step600_v3.jsonl"
    _write_jsonl(c_candidate_path, c_rows_v3)
    _write_jsonl(
        c_decision_template_path,
        [
            {
                "dataset": row["dataset"],
                "problem_id": row["problem_id"],
                "review_bucket": row["review_bucket"],
                "decision": "",
                "decision_reason": "",
                "reviewer": "",
            }
            for row in c_rows_v3
        ],
    )

    # Agent 2 bundle: anchor_common stabilizer additions for A.
    quarantine = load_problem_quarantine(args.quarantine_path)
    train_problems, _ = _load_train_problems(
        args.train_raw,
        args.train_manifest,
        "codecontests_train_wo_valid_big",
        excluded_problem_ids=hard_blacklist_ids(quarantine),
    )
    train_prompt_map = _load_train_prompt_map(args.train_raw)
    eval_raw = _load_eval_raw(args.eval_raw)
    _, forbidden_hashes = _validate_train_hygiene(
        train_manifest_path=args.train_manifest,
        valid_manifest_path=args.valid_manifest,
        valid_big_manifest_path=args.valid_big_manifest,
        test_manifest_path=args.test_manifest,
    )
    idf = build_idf(train_problems)

    anchor_common_ids = list(eval_seed_asset.get("anchor_common", []))
    anchor_priority = _build_anchor_priority_map(anchor_common_ids, step600_delta69, step620_delta69)
    prioritized_anchor_ids = sorted(anchor_common_ids, key=lambda pid: (anchor_priority.get(pid, "p9"), pid))

    seed_matches: dict[str, list[dict[str, Any]]] = {}
    all_occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_sha_by_key: dict[str, str] = {}

    for seed_id in prioritized_anchor_ids:
        raw = eval_raw.get(seed_id)
        if raw is None:
            continue
        seed = SeedEntry(
            key=ProblemKey(dataset="codecontests_valid_big", problem_id=seed_id),
            family="anchor_common",
            bucket="A_retention",
            prompt=raw["prompt"],
            prompt_sha256="",
            tests_count=0,
            source=seed_id.split("/")[0],
        )
        matches = []
        for match in _top_candidates_for_seed(
            seed,
            train_problems,
            idf,
            top_k=args.anchor_top_k_per_seed,
            forbidden_hashes=forbidden_hashes,
        ):
            key_str = match.key.key_str
            if key_str in manifest_keys:
                continue
            row = {
                "key_str": key_str,
                "dataset": match.key.dataset,
                "problem_id": match.key.problem_id,
                "prompt_sha256": match.prompt_sha256,
                "retrieval_score": match.retrieval_score,
                "matched_seed_id": seed_id,
                "matched_seed_family": "anchor_common",
            }
            matches.append(row)
            all_occurrences[key_str].append(row)
            existing_sha = prompt_sha_by_key.get(key_str)
            if existing_sha is None:
                prompt_sha_by_key[key_str] = match.prompt_sha256
            elif existing_sha != match.prompt_sha256:
                raise ValueError(f"Conflicting prompt_sha256 for {key_str}")
        seed_matches[seed_id] = matches

    selected_anchor_keys = _round_robin_select(seed_matches, args.anchor_candidate_limit)
    anchor_rows = _collapse_anchor_candidates(
        selected_anchor_keys,
        all_occurrences,
        prompt_sha_by_key,
        train_prompt_map,
        eval_raw,
        snapshot600,
        snapshot620,
        anchor_priority,
    )

    anchor_candidate_path = output_dir / "anchor_common_stabilizer_candidates_step600_v3.jsonl"
    anchor_decision_template_path = output_dir / "anchor_common_stabilizer_decisions_step600_v3.jsonl"
    _write_jsonl(anchor_candidate_path, anchor_rows)
    _write_jsonl(
        anchor_decision_template_path,
        [
            {
                "dataset": row["dataset"],
                "problem_id": row["problem_id"],
                "review_bucket": row["review_bucket"],
                "decision": "",
                "decision_reason": "",
                "reviewer": "",
            }
            for row in anchor_rows
        ],
    )

    summary = {
        "source_manifest_v2": str(args.manifest_v2),
        "target_keep_c_v3": args.target_c,
        "c_candidates_v3_count": len(c_rows_v3),
        "anchor_candidate_count": len(anchor_rows),
        "anchor_seed_priority_counts": dict(Counter(anchor_priority.values())),
        "anchor_candidate_priority_counts": dict(
            Counter(
                summary["seed_priority"]
                for row in anchor_rows
                for summary in row.get("matched_seed_summaries", [])
            )
        ),
        "files": {
            "c_candidates_v3": str(c_candidate_path),
            "c_decisions_v3": str(c_decision_template_path),
            "anchor_candidates_v3": str(anchor_candidate_path),
            "anchor_decisions_v3": str(anchor_decision_template_path),
        },
    }
    _write_json(output_dir / "focused_manifest_v3_review_summary.json", summary)

    print(f"Wrote C v3 review candidates to {c_candidate_path}")
    print(f"Wrote anchor shortlist candidates to {anchor_candidate_path}")


if __name__ == "__main__":
    main()
