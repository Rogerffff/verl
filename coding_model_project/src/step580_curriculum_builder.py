#!/usr/bin/env python3
"""Build offline assets for the step580 curriculum RL pilot."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from coding_model_project.src.prompting import SYSTEM_PROMPT, format_prompt
    from coding_model_project.src.problem_quarantine import (
        caution_ids,
        hard_blacklist_ids,
        load_problem_quarantine,
        quarantine_summary,
    )
except ImportError:
    from prompting import SYSTEM_PROMPT, format_prompt
    from problem_quarantine import caution_ids, hard_blacklist_ids, load_problem_quarantine, quarantine_summary


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"

TRAIN_MANIFEST_DEFAULT = DATA_ROOT / "manifests" / "codecontests_train_wo_valid_big_manifest.jsonl"
TRAIN_RAW_DEFAULT = DATA_ROOT / "raw" / "codecontests_train_wo_valid_big_raw.jsonl"
VALID_MANIFEST_DEFAULT = DATA_ROOT / "manifests" / "codecontests_valid_manifest.jsonl"
VALID_BIG_MANIFEST_DEFAULT = DATA_ROOT / "manifests" / "codecontests_valid_big_manifest.jsonl"
VALID_BIG_RAW_DEFAULT = DATA_ROOT / "raw" / "codecontests_valid_big_raw.jsonl"
TEST_MANIFEST_DEFAULT = DATA_ROOT / "manifests" / "codecontests_test_manifest.jsonl"
OUTPUT_DIR_DEFAULT = PROJECT_ROOT / "phase_2_ GRPO" / "curriculum_assets" / "step580_v1"
DEFAULT_QUARANTINE_PATH = PROJECT_ROOT / "data" / "problem_quarantine_v2.json"

TOP_K_PER_SEED = 12
BUCKET_TARGETS = {
    "A_retention": 192,
    "B_near_miss": 384,
    "C_hard_partial": 192,
}
BUCKET_PRIORITY = {
    "A_retention": 3,
    "B_near_miss": 2,
    "C_hard_partial": 1,
}
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "you",
    "are",
    "is",
    "be",
    "by",
    "with",
    "from",
    "that",
    "this",
    "given",
    "find",
    "print",
    "output",
    "input",
    "each",
    "test",
    "case",
    "cases",
    "line",
    "contains",
    "integer",
    "integers",
    "first",
    "second",
    "third",
    "number",
    "numbers",
    "string",
    "strings",
    "array",
    "arrays",
    "sequence",
    "sequences",
    "program",
    "problem",
    "description",
}


@dataclass(frozen=True)
class ProblemKey:
    dataset: str
    problem_id: str

    @property
    def key_str(self) -> str:
        return f"{self.dataset}::{self.problem_id}"


@dataclass
class EvalProblem:
    key: ProblemKey
    prompt: str
    prompt_sha256: str
    tests_count: int
    source: str
    accepted: bool
    pass_ratio_all: float
    error_type: str


@dataclass
class TrainProblem:
    key: ProblemKey
    prompt: str
    prompt_sha256: str
    tests_count: int
    source: str
    tokens: list[str]


@dataclass
class SeedEntry:
    key: ProblemKey
    family: str
    bucket: str
    prompt: str
    prompt_sha256: str
    tests_count: int
    source: str


@dataclass
class CandidateMatch:
    key: ProblemKey
    prompt_sha256: str
    retrieval_score: float
    matched_seed_id: str
    matched_seed_family: str
    bucket: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_prompt_messages(dataset_key: str, record: dict[str, Any]) -> list[dict[str, str]]:
    test_cases = record.get("test_cases", {})
    formatted_prompt = format_prompt(
        record["prompt"],
        dataset_key=dataset_key,
        entry_point=test_cases.get("entry_point", ""),
        example_call=test_cases.get("example_call", ""),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted_prompt},
    ]


def _write_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to build curriculum parquet files.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(output_path, index=False)


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]{2,}", text.lower()) if tok not in STOPWORDS]


def bucket_tests(n: int) -> str:
    if n <= 10:
        return "small"
    if n <= 50:
        return "medium"
    return "large"


def _eval_dataset_key_from_rows(rows: list[dict[str, Any]], default: str) -> str:
    dataset = None
    for row in rows:
        candidate = row.get("dataset")
        if candidate:
            dataset = str(candidate)
            break
    return dataset or default


def _load_eval_problems(
    *,
    per_problem_path: Path,
    raw_path: Path,
    default_dataset_key: str,
    excluded_problem_ids: set[str] | None = None,
) -> tuple[dict[ProblemKey, EvalProblem], list[str]]:
    per_problem_rows = _read_jsonl(per_problem_path)
    raw_rows = _read_jsonl(raw_path)
    dataset_key = _eval_dataset_key_from_rows(per_problem_rows, default_dataset_key)
    raw_by_problem = {str(row["problem_id"]): row for row in raw_rows}
    excluded_problem_ids = excluded_problem_ids or set()

    problems: dict[ProblemKey, EvalProblem] = {}
    excluded_hits: list[str] = []
    for row in per_problem_rows:
        problem_id = str(row["problem_id"])
        if problem_id in excluded_problem_ids:
            excluded_hits.append(problem_id)
            continue
        key = ProblemKey(dataset=dataset_key, problem_id=problem_id)
        if key in problems:
            raise ValueError(f"Duplicate eval row for {key.key_str} in {per_problem_path}")
        raw = raw_by_problem.get(problem_id)
        if raw is None:
            raise ValueError(f"{problem_id} from {per_problem_path} is missing in raw file {raw_path}")
        prompt = str(raw.get("canonical_prompt") or raw["prompt"])
        tests_count = len(raw.get("test_cases", {}).get("tests", []))
        problems[key] = EvalProblem(
            key=key,
            prompt=prompt,
            prompt_sha256=str(row.get("prompt_sha256") or raw.get("prompt_sha256", "")),
            tests_count=tests_count,
            source=problem_id.split("/")[0],
            accepted=bool(row.get("accepted", False)),
            pass_ratio_all=float(row.get("pass_ratio_all", row.get("pass_ratio", 0.0)) or 0.0),
            error_type=str(row.get("error_type", "unknown")),
        )
    return problems, sorted(set(excluded_hits))


def _load_manifest_problem_ids(manifest_path: Path) -> set[str]:
    return {str(row["problem_id"]) for row in _read_jsonl(manifest_path)}


def _load_train_problems(
    train_raw_path: Path,
    train_manifest_path: Path,
    train_dataset_key: str,
    *,
    excluded_problem_ids: set[str] | None = None,
) -> tuple[list[TrainProblem], list[str]]:
    rows = _read_jsonl(train_raw_path)
    manifest_problem_ids = _load_manifest_problem_ids(train_manifest_path)
    excluded_problem_ids = excluded_problem_ids or set()
    excluded_in_manifest = manifest_problem_ids & excluded_problem_ids
    manifest_problem_ids = manifest_problem_ids - excluded_in_manifest
    out: list[TrainProblem] = []
    found_problem_ids: set[str] = set()
    for row in rows:
        problem_id = str(row["problem_id"])
        if problem_id not in manifest_problem_ids:
            continue
        found_problem_ids.add(problem_id)
        prompt = str(row.get("canonical_prompt") or row["prompt"])
        out.append(
            TrainProblem(
                key=ProblemKey(dataset=train_dataset_key, problem_id=problem_id),
                prompt=prompt,
                prompt_sha256=str(row["prompt_sha256"]),
                tests_count=len(row.get("test_cases", {}).get("tests", [])),
                source=problem_id.split("/")[0],
                tokens=tokenize(prompt),
            )
        )
    missing_problem_ids = sorted(manifest_problem_ids - found_problem_ids)
    if missing_problem_ids:
        preview = ", ".join(missing_problem_ids[:10])
        raise ValueError(
            f"{train_manifest_path.name} has {len(missing_problem_ids)} problem_id(s) missing from {train_raw_path.name}: {preview}"
        )
    return out, sorted(excluded_in_manifest)


def _load_hash_set(path: Path) -> set[str]:
    return {str(row["prompt_sha256"]) for row in _read_jsonl(path)}


def build_idf(problems: list[TrainProblem]) -> dict[str, float]:
    doc_freq = Counter()
    for problem in problems:
        for token in set(problem.tokens):
            doc_freq[token] += 1
    total_docs = len(problems)
    return {tok: math.log((total_docs + 1) / (count + 1)) + 1.0 for tok, count in doc_freq.items()}


def score_problem(seed: SeedEntry, problem: TrainProblem, idf: dict[str, float]) -> float:
    seed_tokens = tokenize(seed.prompt)
    seed_tf = Counter(seed_tokens)
    doc_tf = Counter(problem.tokens)
    overlap = set(seed_tf) & set(doc_tf)
    lexical = sum(min(seed_tf[token], doc_tf[token]) * idf.get(token, 1.0) for token in overlap)
    lexical /= math.sqrt(len(problem.tokens) + 1)

    source_bonus = 0.15 * lexical if seed.source == problem.source else 0.0
    prompt_len_similarity = max(
        0.0,
        1.0 - abs(len(seed.prompt) - len(problem.prompt)) / max(len(seed.prompt), len(problem.prompt), 1),
    )
    prompt_bonus = 2.0 * prompt_len_similarity

    seed_bucket = bucket_tests(seed.tests_count)
    doc_bucket = bucket_tests(problem.tests_count)
    tests_bonus = 1.0 if seed_bucket == doc_bucket else 0.0
    return lexical + source_bonus + prompt_bonus + tests_bonus


def _build_seed_groups(
    step400: dict[ProblemKey, EvalProblem],
    step580: dict[ProblemKey, EvalProblem],
) -> dict[str, list[EvalProblem]]:
    solved400 = {key for key, problem in step400.items() if problem.accepted}
    solved580 = {key for key, problem in step580.items() if problem.accepted}

    anchor_common = [step580[key] for key in sorted(solved400 & solved580, key=lambda x: x.problem_id)]
    only400 = [step400[key] for key in sorted(solved400 - solved580, key=lambda x: x.problem_id)]
    only580 = [step580[key] for key in sorted(solved580 - solved400, key=lambda x: x.problem_id)]

    high_partial_wa: list[EvalProblem] = []
    medium_partial_wa: list[EvalProblem] = []
    partial_re_tle: list[EvalProblem] = []

    for problem in step580.values():
        if problem.accepted:
            continue
        if problem.error_type == "wrong_answer" and 0.8 <= problem.pass_ratio_all < 1.0:
            high_partial_wa.append(problem)
        elif problem.error_type == "wrong_answer" and 0.4 <= problem.pass_ratio_all < 0.8:
            medium_partial_wa.append(problem)
        elif problem.error_type in {"runtime_error", "timeout"} and 0.0 < problem.pass_ratio_all < 1.0:
            partial_re_tle.append(problem)

    def by_problem_id(items: list[EvalProblem]) -> list[EvalProblem]:
        return sorted(items, key=lambda item: item.key.problem_id)

    return {
        "anchor_common": by_problem_id(anchor_common),
        "only400": by_problem_id(only400),
        "only580": by_problem_id(only580),
        "high_partial_wa": by_problem_id(high_partial_wa),
        "medium_partial_wa": by_problem_id(medium_partial_wa),
        "partial_re_tle": by_problem_id(partial_re_tle),
    }


def _seed_entries_from_groups(groups: dict[str, list[EvalProblem]]) -> dict[str, list[SeedEntry]]:
    seeds: dict[str, list[SeedEntry]] = {
        "A_retention": [],
        "B_near_miss": [],
        "C_hard_partial": [],
    }
    family_specs = [
        ("A_retention", "anchor_common", groups["anchor_common"]),
        ("A_retention", "only400", groups["only400"]),
        ("B_near_miss", "only580", groups["only580"]),
        ("B_near_miss", "high_partial_wa", groups["high_partial_wa"]),
        ("B_near_miss", "medium_partial_wa", groups["medium_partial_wa"]),
        ("C_hard_partial", "partial_re_tle", groups["partial_re_tle"]),
    ]
    for bucket, family, problems in family_specs:
        for problem in problems:
            seeds[bucket].append(
                SeedEntry(
                    key=problem.key,
                    family=family,
                    bucket=bucket,
                    prompt=problem.prompt,
                    prompt_sha256=problem.prompt_sha256,
                    tests_count=problem.tests_count,
                    source=problem.source,
                )
            )
    return seeds


def _top_candidates_for_seed(
    seed: SeedEntry,
    train_problems: list[TrainProblem],
    idf: dict[str, float],
    *,
    top_k: int,
    forbidden_hashes: set[str],
) -> list[CandidateMatch]:
    scored: list[tuple[float, TrainProblem]] = []
    for problem in train_problems:
        if problem.prompt_sha256 in forbidden_hashes:
            continue
        scored.append((score_problem(seed, problem, idf), problem))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        CandidateMatch(
            key=problem.key,
            prompt_sha256=problem.prompt_sha256,
            retrieval_score=score,
            matched_seed_id=seed.key.problem_id,
            matched_seed_family=seed.family,
            bucket=seed.bucket,
        )
        for score, problem in scored[:top_k]
    ]


def _round_robin_select(
    seed_matches: dict[str, list[CandidateMatch]],
    *,
    target_size: int,
    reserved_keys: set[str],
) -> list[str]:
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
            key_str = matches[pos].key.key_str
            pos += 1
            if key_str in reserved_keys or key_str in selected_set:
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


def _collapse_occurrences(
    key_strs: list[str],
    occurrences: dict[str, list[CandidateMatch]],
    prompt_sha_by_key: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key_str in key_strs:
        matches = occurrences[key_str]
        match_families = sorted({match.matched_seed_family for match in matches})
        match_seed_ids = sorted({match.matched_seed_id for match in matches})
        winning_bucket = max((match.bucket for match in matches), key=lambda item: BUCKET_PRIORITY[item])
        matches_sorted = sorted(matches, key=lambda item: item.retrieval_score, reverse=True)
        winning_bucket_matches = [match for match in matches_sorted if match.bucket == winning_bucket]
        primary_family = winning_bucket_matches[0].matched_seed_family
        dataset, problem_id = key_str.split("::", 1)
        rows.append(
            {
                "dataset": dataset,
                "problem_id": problem_id,
                "initial_bucket": winning_bucket,
                "primary_seed_family": primary_family,
                "matched_seed_ids": match_seed_ids,
                "matched_seed_families": match_families,
                "retrieval_score_max": matches_sorted[0].retrieval_score,
                "prompt_sha256": prompt_sha_by_key[key_str],
            }
        )
    rows.sort(key=lambda row: (-BUCKET_PRIORITY[row["initial_bucket"]], -row["retrieval_score_max"], row["problem_id"]))
    return rows


def _build_curriculum_manifest(
    seed_groups: dict[str, list[SeedEntry]],
    train_problems: list[TrainProblem],
    *,
    forbidden_hashes: set[str],
    top_k_per_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    idf = build_idf(train_problems)
    all_occurrences: dict[str, list[CandidateMatch]] = defaultdict(list)
    prompt_sha_by_key: dict[str, str] = {}
    seed_matches_by_bucket: dict[str, dict[str, list[CandidateMatch]]] = {
        "A_retention": {},
        "B_near_miss": {},
        "C_hard_partial": {},
    }
    selected_summary: dict[str, Any] = {
        "bucket_targets": BUCKET_TARGETS.copy(),
        "bucket_selected_pre_collapse": {},
        "bucket_actual": {},
    }
    reserved_keys: set[str] = set()

    for bucket in ("A_retention", "B_near_miss", "C_hard_partial"):
        for seed in seed_groups[bucket]:
            matches = _top_candidates_for_seed(
                seed,
                train_problems,
                idf,
                top_k=top_k_per_seed,
                forbidden_hashes=forbidden_hashes,
            )
            seed_matches_by_bucket[bucket][seed.key.problem_id] = matches
            for match in matches:
                key_str = match.key.key_str
                all_occurrences[key_str].append(match)
                existing_sha = prompt_sha_by_key.get(key_str)
                if existing_sha is None:
                    prompt_sha_by_key[key_str] = match.prompt_sha256
                elif existing_sha != match.prompt_sha256:
                    raise ValueError(f"Conflicting prompt_sha256 for {key_str}")

    winning_bucket_by_key = {
        key_str: max((match.bucket for match in matches), key=lambda item: BUCKET_PRIORITY[item])
        for key_str, matches in all_occurrences.items()
    }

    manifest_rows: list[dict[str, Any]] = []
    for bucket in ("A_retention", "B_near_miss", "C_hard_partial"):
        filtered_seed_matches: dict[str, list[CandidateMatch]] = {}
        for seed_id, matches in seed_matches_by_bucket[bucket].items():
            filtered_seed_matches[seed_id] = [
                match for match in matches if winning_bucket_by_key.get(match.key.key_str) == bucket
            ]
        selected_key_strs = _round_robin_select(
            filtered_seed_matches,
            target_size=BUCKET_TARGETS[bucket],
            reserved_keys=reserved_keys,
        )
        reserved_keys.update(selected_key_strs)
        selected_summary["bucket_selected_pre_collapse"][bucket] = len(selected_key_strs)
        manifest_rows.extend(_collapse_occurrences(selected_key_strs, all_occurrences, prompt_sha_by_key))

    selected_summary["bucket_actual"] = dict(Counter(row["initial_bucket"] for row in manifest_rows))
    return manifest_rows, selected_summary


def _build_eval_rows(records: list[dict[str, Any]], dataset_key: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "data_source": dataset_key,
                "prompt": _build_prompt_messages(dataset_key, record),
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "problem_id": record["problem_id"],
                        "test_cases": record.get("test_cases", {}),
                        "dataset": dataset_key,
                    },
                },
                "extra_info": {
                    "problem_id": record["problem_id"],
                    "split": split,
                    "dataset": dataset_key,
                },
            }
        )
    return rows


def _build_delta69_eval_parquet(
    valid_big_raw_path: Path,
    *,
    selected_problem_ids: list[str],
    dataset_key: str,
    output_path: Path,
) -> None:
    raw_rows = _read_jsonl(valid_big_raw_path)
    selected_id_set = set(selected_problem_ids)
    selected_rows = [row for row in raw_rows if str(row["problem_id"]) in selected_id_set]
    selected_rows.sort(key=lambda row: selected_problem_ids.index(str(row["problem_id"])))
    if len(selected_rows) != len(selected_problem_ids):
        found = {str(row["problem_id"]) for row in selected_rows}
        missing = sorted(selected_id_set - found)
        raise ValueError(f"delta69 eval parquet is missing valid_big raw rows for: {missing[:10]}")
    eval_rows = _build_eval_rows(selected_rows, dataset_key=dataset_key, split="valid_big")
    _write_parquet(eval_rows, output_path)


def _build_hygiene_report(
    *,
    output_path: Path,
    overlap_stats: dict[str, int],
    seed_groups: dict[str, list[EvalProblem]],
    manifest_rows: list[dict[str, Any]],
    selection_summary: dict[str, Any],
    delta69_count: int,
    quarantine: dict[str, Any],
    quarantine_hits: dict[str, list[str]],
) -> None:
    bucket_counts = Counter(row["initial_bucket"] for row in manifest_rows)
    lines = [
        "# Step580 Curriculum Hygiene Report",
        "",
        "## Quarantine",
        "",
        f"- quarantine path: `{quarantine.get('path')}`",
        f"- hard_blacklist count: `{len(quarantine.get('hard_blacklist_ids', set()))}`",
        f"- caution count: `{len(quarantine.get('caution_ids', set()))}`",
        f"- excluded from train manifest: `{len(quarantine_hits['train'])}`",
        f"- excluded from step400 eval: `{len(quarantine_hits['step400'])}`",
        f"- excluded from step580 eval: `{len(quarantine_hits['step580'])}`",
        "",
        "## Overlap Checks",
        "",
        f"- train vs valid prompt_sha256 overlap: `{overlap_stats['train_vs_valid']}`",
        f"- train vs valid_big prompt_sha256 overlap: `{overlap_stats['train_vs_valid_big']}`",
        f"- train vs test prompt_sha256 overlap: `{overlap_stats['train_vs_test']}`",
        "",
        "## Eval Seed Counts",
        "",
        f"- anchor_common: `{len(seed_groups['anchor_common'])}`",
        f"- only400: `{len(seed_groups['only400'])}`",
        f"- only580: `{len(seed_groups['only580'])}`",
        f"- high_partial_wa: `{len(seed_groups['high_partial_wa'])}`",
        f"- medium_partial_wa: `{len(seed_groups['medium_partial_wa'])}`",
        f"- partial_re_tle: `{len(seed_groups['partial_re_tle'])}`",
        f"- delta69 eval set size: `{delta69_count}`",
        "",
        "## Manifest Coverage",
        "",
        f"- total manifest rows: `{len(manifest_rows)}`",
        f"- A_retention target/actual: `{selection_summary['bucket_targets']['A_retention']}` / `{bucket_counts['A_retention']}`",
        f"- B_near_miss target/actual: `{selection_summary['bucket_targets']['B_near_miss']}` / `{bucket_counts['B_near_miss']}`",
        f"- C_hard_partial target/actual: `{selection_summary['bucket_targets']['C_hard_partial']}` / `{bucket_counts['C_hard_partial']}`",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_train_hygiene(
    *,
    train_manifest_path: Path,
    valid_manifest_path: Path,
    valid_big_manifest_path: Path,
    test_manifest_path: Path,
) -> tuple[dict[str, int], set[str]]:
    train_hashes = _load_hash_set(train_manifest_path)
    valid_hashes = _load_hash_set(valid_manifest_path)
    valid_big_hashes = _load_hash_set(valid_big_manifest_path)
    test_hashes = _load_hash_set(test_manifest_path)
    stats = {
        "train_vs_valid": len(train_hashes & valid_hashes),
        "train_vs_valid_big": len(train_hashes & valid_big_hashes),
        "train_vs_test": len(train_hashes & test_hashes),
    }
    forbidden_hashes = valid_hashes | valid_big_hashes | test_hashes
    return stats, forbidden_hashes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step400-jsonl", type=Path, required=True, help="step400 per-problem jsonl path")
    parser.add_argument("--step580-jsonl", type=Path, required=True, help="step580 per-problem jsonl path")
    parser.add_argument("--train-raw", type=Path, default=TRAIN_RAW_DEFAULT)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST_DEFAULT)
    parser.add_argument("--valid-manifest", type=Path, default=VALID_MANIFEST_DEFAULT)
    parser.add_argument("--valid-big-manifest", type=Path, default=VALID_BIG_MANIFEST_DEFAULT)
    parser.add_argument("--valid-big-raw", type=Path, default=VALID_BIG_RAW_DEFAULT)
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--train-dataset-key", default="codecontests_train_wo_valid_big")
    parser.add_argument("--eval-dataset-key", default="codecontests_valid_big")
    parser.add_argument("--top-k-per-seed", type=int, default=TOP_K_PER_SEED)
    parser.add_argument(
        "--quarantine-path",
        type=Path,
        default=DEFAULT_QUARANTINE_PATH,
        help="Optional shared problem quarantine file. hard_blacklist items are excluded from train assets; hard_blacklist+caution are excluded from eval seed assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    quarantine = load_problem_quarantine(args.quarantine_path)
    train_excluded_problem_ids = hard_blacklist_ids(quarantine)
    eval_excluded_problem_ids = train_excluded_problem_ids | caution_ids(quarantine)

    step400, step400_excluded = _load_eval_problems(
        per_problem_path=args.step400_jsonl,
        raw_path=args.valid_big_raw,
        default_dataset_key=args.eval_dataset_key,
        excluded_problem_ids=eval_excluded_problem_ids,
    )
    step580, step580_excluded = _load_eval_problems(
        per_problem_path=args.step580_jsonl,
        raw_path=args.valid_big_raw,
        default_dataset_key=args.eval_dataset_key,
        excluded_problem_ids=eval_excluded_problem_ids,
    )
    if set(step400) != set(step580):
        only400 = sorted(key.problem_id for key in set(step400) - set(step580))
        only580 = sorted(key.problem_id for key in set(step580) - set(step400))
        raise ValueError(
            f"step400/step580 eval sets differ. only400={only400[:5]}, only580={only580[:5]}"
        )

    seed_groups = _build_seed_groups(step400, step580)
    seed_entries = _seed_entries_from_groups(seed_groups)

    overlap_stats, forbidden_hashes = _validate_train_hygiene(
        train_manifest_path=args.train_manifest,
        valid_manifest_path=args.valid_manifest,
        valid_big_manifest_path=args.valid_big_manifest,
        test_manifest_path=args.test_manifest,
    )

    train_problems, train_excluded = _load_train_problems(
        args.train_raw,
        args.train_manifest,
        args.train_dataset_key,
        excluded_problem_ids=train_excluded_problem_ids,
    )
    manifest_rows, selection_summary = _build_curriculum_manifest(
        seed_entries,
        train_problems,
        forbidden_hashes=forbidden_hashes,
        top_k_per_seed=args.top_k_per_seed,
    )

    manifest_path = output_dir / "curriculum_train_manifest_step580_v1.jsonl"
    _write_jsonl(manifest_path, manifest_rows)

    eval_asset = {
        "eval_dataset": args.eval_dataset_key,
        "train_dataset": args.train_dataset_key,
        "counts": {name: len(items) for name, items in seed_groups.items()},
        "anchor_common": [problem.key.problem_id for problem in seed_groups["anchor_common"]],
        "only400": [problem.key.problem_id for problem in seed_groups["only400"]],
        "only580": [problem.key.problem_id for problem in seed_groups["only580"]],
        "high_partial_wa": [problem.key.problem_id for problem in seed_groups["high_partial_wa"]],
        "medium_partial_wa": [problem.key.problem_id for problem in seed_groups["medium_partial_wa"]],
        "partial_re_tle": [problem.key.problem_id for problem in seed_groups["partial_re_tle"]],
        "delta69_problem_ids": [
            *[problem.key.problem_id for problem in seed_groups["anchor_common"]],
            *[problem.key.problem_id for problem in seed_groups["only400"]],
            *[problem.key.problem_id for problem in seed_groups["only580"]],
        ],
    }
    eval_seed_asset_path = output_dir / "curriculum_eval_seed_asset_step580_v1.json"
    _write_json(eval_seed_asset_path, eval_asset)

    delta69_parquet_path = output_dir / "delta69_eval.parquet"
    _build_delta69_eval_parquet(
        args.valid_big_raw,
        selected_problem_ids=eval_asset["delta69_problem_ids"],
        dataset_key=args.eval_dataset_key,
        output_path=delta69_parquet_path,
    )

    hygiene_report_path = output_dir / "curriculum_hygiene_report_step580_v1.md"
    _build_hygiene_report(
        output_path=hygiene_report_path,
        overlap_stats=overlap_stats,
        seed_groups=seed_groups,
        manifest_rows=manifest_rows,
        selection_summary=selection_summary,
        delta69_count=len(eval_asset["delta69_problem_ids"]),
        quarantine=quarantine,
        quarantine_hits={
            "train": train_excluded,
            "step400": step400_excluded,
            "step580": step580_excluded,
        },
    )

    summary_path = output_dir / "curriculum_manifest_summary_step580_v1.json"
    _write_json(
        summary_path,
        {
            "manifest_path": str(manifest_path),
            "eval_seed_asset_path": str(eval_seed_asset_path),
            "delta69_parquet_path": str(delta69_parquet_path),
            "hygiene_report_path": str(hygiene_report_path),
            "bucket_counts": Counter(row["initial_bucket"] for row in manifest_rows),
            "seed_group_counts": {name: len(items) for name, items in seed_groups.items()},
            "overlap_stats": overlap_stats,
            "problem_quarantine": quarantine_summary(quarantine),
            "quarantine_hits": {
                "train": train_excluded,
                "step400": step400_excluded,
                "step580": step580_excluded,
            },
        },
    )

    print(f"Wrote curriculum manifest to {manifest_path}")
    print(f"Wrote eval seed asset to {eval_seed_asset_path}")
    print(f"Wrote delta69 eval parquet to {delta69_parquet_path}")
    print(f"Wrote hygiene report to {hygiene_report_path}")


if __name__ == "__main__":
    main()
