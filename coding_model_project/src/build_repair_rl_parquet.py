#!/usr/bin/env python3
"""Build repair RL parquet files from cached step1300 first-pass repair assets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

try:
    from coding_model_project.src.repair_feedback import (
        REPAIR_SYSTEM_PROMPT,
        RepairPromptMode,
        build_codecontests_repair_prompt,
        build_repair_feedback_from_eval_result,
    )
except ImportError:
    from repair_feedback import (
        REPAIR_SYSTEM_PROMPT,
        RepairPromptMode,
        build_codecontests_repair_prompt,
        build_repair_feedback_from_eval_result,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PHASE2_DIR = REPO_ROOT / "coding_model_project" / "phase_2_ GRPO"
DATA_ROOT = REPO_ROOT / "coding_model_project" / "data"
STEP1300_REPAIR_DIR = PHASE2_DIR / "sft_repair_data" / "step1300_repair"
STEP1300_SHORTDIAG_DIR = PHASE2_DIR / "sft_repair_data" / "step1300_shortdiag_pure_v1"

DEFAULT_STUDENT_REFERENCES = STEP1300_REPAIR_DIR / "student_references_step1300_repair_cond_v2.jsonl"
DEFAULT_TEACHER_REQUESTS = STEP1300_REPAIR_DIR / "full_teacher_requests_step1300_repair_cond_v2.jsonl"
DEFAULT_RAW_DATA = DATA_ROOT / "raw" / "codecontests_train_wo_valid_big_raw.jsonl"
DEFAULT_BLOCKLIST_SOURCE = STEP1300_SHORTDIAG_DIR / "step1300_shortdiag_pure_keep_set_v1.jsonl"
DEFAULT_BLOCKLIST_FALLBACK = STEP1300_REPAIR_DIR / "final_keep" / "step1300_teacher_keep_set_v2.jsonl"
DEFAULT_BLOCKLIST_ASSET = (
    PHASE2_DIR / "repair_RL" / "assets" / "step1300_probe_v0_audit_suspect_blocklist.jsonl"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "repair_rl_parquet" / "step1300_probe_v0"

AUDIT_KEEP_SOURCES = {
    "remaining_high_pass_audit_0p85_to_0p9",
    "remaining_high_pass_audit_0p7_to_0p85",
    "near_miss_testcase_audit_high_confidence",
    "near_miss_testcase_audit_regen_round1",
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
        return
    except ModuleNotFoundError:
        pass

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("Writing parquet requires pyarrow or pandas.") from exc

    pd.DataFrame.from_records(rows).to_parquet(path, index=False)


def _row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row["problem_id"]), str(row["prompt_sha256"])


def _index_unique(rows: Iterable[Dict[str, Any]], *, name: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key in index:
            raise ValueError(f"Duplicate key in {name}: {key}")
        index[key] = row
    return index


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return False


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _pass_ratio_bucket(pass_ratio: float) -> str:
    if pass_ratio <= 0.0:
        return "bucket_0"
    if pass_ratio < 0.2:
        return "bucket_0_0.2"
    if pass_ratio < 0.6:
        return "bucket_0.2_0.6"
    return "bucket_0.6_1.0"


def _has_feedback_payload(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _materialize_audit_blocklist(
    *,
    preferred_source: Path,
    fallback_source: Path,
    output_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source_path = preferred_source if preferred_source.exists() else fallback_source
    if not source_path.exists():
        raise FileNotFoundError(
            f"Neither preferred nor fallback blocklist source exists: {preferred_source}, {fallback_source}"
        )

    source_rows = _read_jsonl(source_path)
    source_counts = Counter()
    seen_keys: set[Tuple[str, str]] = set()
    blocklist_rows: List[Dict[str, Any]] = []
    duplicate_count = 0

    for row in source_rows:
        keep_source = str(row.get("teacher_keep_source") or "")
        if keep_source not in AUDIT_KEEP_SOURCES:
            continue
        key = _row_key(row)
        if key in seen_keys:
            duplicate_count += 1
            continue
        seen_keys.add(key)
        source_counts[keep_source] += 1
        blocklist_rows.append(
            {
                "problem_id": str(row["problem_id"]),
                "prompt_sha256": str(row["prompt_sha256"]),
                "teacher_keep_source": keep_source,
            }
        )

    blocklist_rows.sort(key=lambda row: (row["teacher_keep_source"], row["problem_id"], row["prompt_sha256"]))
    _write_jsonl(output_path, blocklist_rows)

    meta = {
        "source_path": str(source_path),
        "row_count": len(blocklist_rows),
        "duplicate_count": duplicate_count,
        "source_counts": dict(source_counts),
        "output_path": str(output_path),
    }
    return blocklist_rows, meta


def _load_existing_blocklist(asset_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not asset_path.exists():
        raise FileNotFoundError(f"Materialized blocklist asset does not exist: {asset_path}")

    source_rows = _read_jsonl(asset_path)
    seen_keys: set[Tuple[str, str]] = set()
    source_counts = Counter()
    duplicate_count = 0
    blocklist_rows: List[Dict[str, Any]] = []

    for row in source_rows:
        key = _row_key(row)
        if key in seen_keys:
            duplicate_count += 1
            continue
        seen_keys.add(key)
        keep_source = _as_str(row.get("teacher_keep_source"))
        source_counts[keep_source] += 1
        blocklist_rows.append(
            {
                "problem_id": key[0],
                "prompt_sha256": key[1],
                "teacher_keep_source": keep_source,
            }
        )

    blocklist_rows.sort(key=lambda row: (row["teacher_keep_source"], row["problem_id"], row["prompt_sha256"]))
    meta = {
        "source_path": str(asset_path),
        "row_count": len(blocklist_rows),
        "duplicate_count": duplicate_count,
        "source_counts": dict(source_counts),
        "output_path": str(asset_path),
        "materialization_mode": "reuse_existing_asset",
    }
    return blocklist_rows, meta


def _resolve_audit_blocklist(
    *,
    asset_path: Path,
    preferred_source: Path,
    fallback_source: Path,
    refresh_asset: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if asset_path.exists() and not refresh_asset:
        return _load_existing_blocklist(asset_path)

    blocklist_rows, meta = _materialize_audit_blocklist(
        preferred_source=preferred_source,
        fallback_source=fallback_source,
        output_path=asset_path,
    )
    meta["materialization_mode"] = "generated_from_source"
    return blocklist_rows, meta


def _feedback_from_student_reference(
    student_row: Dict[str, Any],
    *,
    max_failure_cases: int,
    max_case_chars: int,
) -> Dict[str, Any]:
    eval_result = SimpleNamespace(
        error_type=_as_str(student_row.get("error_type")) or "unknown",
        details=student_row.get("details") or {},
    )
    return build_repair_feedback_from_eval_result(
        eval_result,
        max_failure_cases=max_failure_cases,
        max_case_chars=max_case_chars,
    )


def _build_prompt_messages(
    *,
    problem_prompt: str,
    first_pass_code: str,
    feedback: Dict[str, Any],
    prompt_mode: RepairPromptMode,
) -> List[Dict[str, str]]:
    user_prompt = build_codecontests_repair_prompt(
        problem_prompt=problem_prompt,
        first_response=first_pass_code,
        feedback=feedback,
        prompt_mode=prompt_mode,
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _sample_rows(rows: List[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    if count < 0:
        raise ValueError("sample count must be non-negative")
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows but only {len(rows)} are available")
    sampled = list(rows)
    random.Random(seed).shuffle(sampled)
    return sampled[:count]


def _manifest_row(selected_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "problem_id": selected_row["problem_id"],
        "prompt_sha256": selected_row["prompt_sha256"],
        "request_id": selected_row["request_id"],
        "dataset": selected_row["dataset"],
        "source_split": selected_row["source_split"],
        "selected_slice": selected_row["selected_slice"],
        "first_pass_pass_ratio_all": selected_row["first_pass_pass_ratio_all"],
        "first_pass_bucket": selected_row["first_pass_bucket"],
        "first_pass_error_type": selected_row["first_pass_error_type"],
        "curriculum_bucket": selected_row["curriculum_bucket"],
        "repair_stratum": selected_row["repair_stratum"],
        "prompt_mode": selected_row["prompt_mode"],
        "feedback_source": selected_row["feedback_source"],
        "teacher_prompt_mode": selected_row["teacher_prompt_mode"],
    }


def _make_dataset_record(
    *,
    selected_row: Dict[str, Any],
    split: str,
    data_source: str,
    source_run_id: str,
    source_protocol: str,
) -> Dict[str, Any]:
    return {
        "data_source": data_source,
        "prompt": selected_row["prompt_messages"],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "problem_id": selected_row["problem_id"],
                "dataset": selected_row["dataset"],
                "test_cases": selected_row["test_cases"],
                "repair_feedback": selected_row["repair_feedback"],
                "first_pass": {
                    "code": selected_row["first_pass_code"],
                    "pass_ratio_all": selected_row["first_pass_pass_ratio_all"],
                    "accepted": selected_row["first_pass_accepted"],
                    "error_type": selected_row["first_pass_error_type"],
                    "invalid_for_rl": selected_row["first_pass_invalid_for_rl"],
                    "invalid_reason": selected_row["first_pass_invalid_reason"],
                    "finish_reason": selected_row["first_pass_finish_reason"],
                    "pass_ratio_bucket": selected_row["first_pass_bucket"],
                },
                "repair_metadata": {
                    "prompt_mode": selected_row["prompt_mode"],
                    "teacher_prompt_mode": selected_row["teacher_prompt_mode"],
                    "source_run_id": source_run_id,
                    "source_protocol": source_protocol,
                    "feedback_source": selected_row["feedback_source"],
                },
            },
        },
        "extra_info": {
            "problem_id": selected_row["problem_id"],
            "prompt_sha256": selected_row["prompt_sha256"],
            "request_id": selected_row["request_id"],
            "split": split,
            "dataset": selected_row["dataset"],
            "source_split": selected_row["source_split"],
            "repair_prompt_mode": selected_row["prompt_mode"],
            "teacher_prompt_mode": selected_row["teacher_prompt_mode"],
            "selected_slice": selected_row["selected_slice"],
            "curriculum_bucket": selected_row["curriculum_bucket"],
            "repair_stratum": selected_row["repair_stratum"],
            "repair_feedback_selection_strategy": _as_str(
                selected_row["repair_feedback"].get("selection_strategy")
            ),
            "first_pass_pass_ratio_all": selected_row["first_pass_pass_ratio_all"],
            "first_pass_accepted": selected_row["first_pass_accepted"],
            "first_pass_bucket": selected_row["first_pass_bucket"],
            "first_pass_error_type": selected_row["first_pass_error_type"],
            "first_pass_source": "student_references_step1300_repair_cond_v2",
        },
    }


def build_repair_rl_dataset(
    *,
    student_references_path: Path,
    teacher_requests_path: Path,
    raw_data_path: Path,
    blocklist_source_path: Path,
    blocklist_fallback_path: Path,
    blocklist_asset_out: Path,
    output_dir: Path,
    prompt_mode: RepairPromptMode,
    data_source: str,
    source_run_id: str,
    source_protocol: str,
    source_split: str,
    min_pass_ratio: float,
    high_threshold: float,
    target_high_count: int,
    target_mid_count: int,
    smoke_train_size: int,
    smoke_val_size: int,
    max_failure_cases: int,
    max_case_chars: int,
    allowed_error_types: List[str],
    allow_teacher_feedback_fallback: bool,
    refresh_blocklist_asset: bool,
    seed: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    blocklist_rows, blocklist_meta = _resolve_audit_blocklist(
        asset_path=blocklist_asset_out,
        preferred_source=blocklist_source_path,
        fallback_source=blocklist_fallback_path,
        refresh_asset=refresh_blocklist_asset,
    )
    blocklist_keys = {(row["problem_id"], row["prompt_sha256"]) for row in blocklist_rows}
    blocklist_reason_by_key = {
        (row["problem_id"], row["prompt_sha256"]): row["teacher_keep_source"] for row in blocklist_rows
    }

    student_rows = _read_jsonl(student_references_path)
    teacher_rows = _read_jsonl(teacher_requests_path)
    raw_rows = _read_jsonl(raw_data_path)

    teacher_index = _index_unique(teacher_rows, name="teacher_requests")
    raw_index = _index_unique(raw_rows, name="raw_data")

    allowed_error_type_set = {value for value in allowed_error_types if value}
    filtered_rows: List[Dict[str, Any]] = []
    counts = Counter()
    removed_by_error_type = Counter()
    blocklist_removed_rows: List[Dict[str, Any]] = []
    missing_teacher_request_preview: List[Dict[str, Any]] = []
    empty_teacher_feedback_preview: List[Dict[str, Any]] = []

    for student_row in student_rows:
        counts["input_row_count"] += 1
        if _as_bool(student_row.get("accepted")):
            counts["accepted_skip_count"] += 1
            continue
        counts["failed_row_count"] += 1

        if _as_str(student_row.get("finish_reason")) != "stop":
            counts["non_stop_skip_count"] += 1
            continue

        details = student_row.get("details") or {}
        if _as_bool(details.get("invalid_for_rl")):
            counts["invalid_skip_count"] += 1
            continue

        if _as_str(student_row.get("source_split")) != source_split:
            counts["source_split_skip_count"] += 1
            continue

        error_type = _as_str(student_row.get("error_type"))
        if allowed_error_type_set and error_type not in allowed_error_type_set:
            removed_by_error_type[error_type or ""] += 1
            counts["error_type_skip_count"] += 1
            continue

        pass_ratio_all = float(student_row.get("pass_ratio_all", student_row.get("pass_ratio", 0.0)))
        if pass_ratio_all < min_pass_ratio:
            counts["below_min_pass_ratio_skip_count"] += 1
            continue

        key = _row_key(student_row)
        if key in blocklist_keys:
            blocklist_removed_rows.append(
                {
                    "problem_id": key[0],
                    "prompt_sha256": key[1],
                    "teacher_keep_source": blocklist_reason_by_key.get(key, ""),
                    "first_pass_pass_ratio_all": pass_ratio_all,
                    "first_pass_error_type": error_type,
                }
            )
            counts["excluded_audit_suspect_count"] += 1
            continue

        request_row = teacher_index.get(key)
        raw_row = raw_index.get(key)
        if raw_row is None:
            raise ValueError(f"Missing raw row for key={key}")
        if not raw_row.get("test_cases"):
            raise ValueError(f"Missing test_cases for key={key}")

        repair_feedback = {}
        feedback_source = "teacher_requests"
        request_id = ""
        teacher_prompt_mode = ""
        if request_row is None:
            counts["missing_teacher_request_count"] += 1
            if len(missing_teacher_request_preview) < 20:
                missing_teacher_request_preview.append(
                    {
                        "problem_id": key[0],
                        "prompt_sha256": key[1],
                        "first_pass_pass_ratio_all": pass_ratio_all,
                        "first_pass_error_type": error_type,
                    }
                )
        else:
            repair_feedback = request_row.get("repair_feedback") or {}
            request_id = _as_str(request_row.get("request_id"))
            teacher_prompt_mode = _as_str(request_row.get("teacher_prompt_mode"))
            if not _has_feedback_payload(repair_feedback):
                counts["empty_teacher_feedback_count"] += 1
                if len(empty_teacher_feedback_preview) < 20:
                    empty_teacher_feedback_preview.append(
                        {
                            "problem_id": key[0],
                            "prompt_sha256": key[1],
                            "request_id": request_id,
                            "first_pass_pass_ratio_all": pass_ratio_all,
                            "first_pass_error_type": error_type,
                        }
                    )

        if not _has_feedback_payload(repair_feedback):
            if not allow_teacher_feedback_fallback:
                continue
            repair_feedback = _feedback_from_student_reference(
                student_row,
                max_failure_cases=max_failure_cases,
                max_case_chars=max_case_chars,
            )
            if request_row is None:
                feedback_source = "student_reference_fallback_missing_teacher_request"
                counts["teacher_feedback_fallback_missing_request_count"] += 1
            else:
                feedback_source = "student_reference_fallback_empty_teacher_feedback"
                counts["teacher_feedback_fallback_empty_feedback_count"] += 1

        first_pass_code = _as_str(student_row.get("full_extracted_code")).strip()
        if not first_pass_code:
            raise ValueError(f"Missing full_extracted_code for key={key}")
        problem_prompt = _as_str(student_row.get("prompt"))
        if not problem_prompt:
            raise ValueError(f"Missing prompt for key={key}")

        selected_slice = "high" if pass_ratio_all >= high_threshold else "mid"
        selected_row = {
            "problem_id": key[0],
            "prompt_sha256": key[1],
            "dataset": _as_str(student_row.get("dataset")) or "codecontests_train",
            "source_split": _as_str(student_row.get("source_split")) or source_split,
            "request_id": request_id,
            "teacher_prompt_mode": teacher_prompt_mode,
            "problem_prompt": problem_prompt,
            "first_pass_code": first_pass_code,
            "first_pass_pass_ratio_all": pass_ratio_all,
            "first_pass_accepted": _as_bool(student_row.get("accepted")),
            "first_pass_error_type": error_type,
            "first_pass_finish_reason": _as_str(student_row.get("finish_reason")),
            "first_pass_invalid_for_rl": _as_bool(details.get("invalid_for_rl")),
            "first_pass_invalid_reason": _as_str(details.get("invalid_reason")),
            "first_pass_extraction_status": _as_str(details.get("extraction_status")),
            "first_pass_bucket": _pass_ratio_bucket(pass_ratio_all),
            "curriculum_bucket": _as_str(student_row.get("curriculum_bucket")),
            "repair_stratum": _as_str(student_row.get("repair_stratum")),
            "repair_feedback": repair_feedback,
            "feedback_source": feedback_source,
            "test_cases": raw_row.get("test_cases", {}),
            "selected_slice": selected_slice,
            "prompt_mode": str(prompt_mode),
        }
        selected_row["prompt_messages"] = _build_prompt_messages(
            problem_prompt=selected_row["problem_prompt"],
            first_pass_code=selected_row["first_pass_code"],
            feedback=selected_row["repair_feedback"],
            prompt_mode=prompt_mode,
        )
        filtered_rows.append(selected_row)

    if not allow_teacher_feedback_fallback and (
        counts["missing_teacher_request_count"] > 0 or counts["empty_teacher_feedback_count"] > 0
    ):
        raise ValueError(
            "Teacher request contract violated before sampling: "
            f"missing_teacher_request_count={counts['missing_teacher_request_count']}, "
            f"empty_teacher_feedback_count={counts['empty_teacher_feedback_count']}, "
            f"missing_teacher_request_preview={missing_teacher_request_preview[:5]}, "
            f"empty_teacher_feedback_preview={empty_teacher_feedback_preview[:5]}"
        )

    high_candidates = sorted(
        (row for row in filtered_rows if row["selected_slice"] == "high"),
        key=lambda row: (row["problem_id"], row["prompt_sha256"]),
    )
    mid_candidates = sorted(
        (row for row in filtered_rows if row["selected_slice"] == "mid"),
        key=lambda row: (row["problem_id"], row["prompt_sha256"]),
    )

    selected_high = _sample_rows(high_candidates, target_high_count, seed)
    selected_mid = _sample_rows(mid_candidates, target_mid_count, seed + 1)
    selected_rows = [*selected_high, *selected_mid]
    random.Random(seed + 2).shuffle(selected_rows)

    if len(selected_rows) < smoke_train_size + smoke_val_size:
        raise ValueError(
            "Selected repair rows are insufficient for requested smoke splits: "
            f"selected={len(selected_rows)}, smoke_train={smoke_train_size}, smoke_val={smoke_val_size}"
        )

    smoke_train_rows = selected_rows[:smoke_train_size]
    smoke_val_rows = selected_rows[smoke_train_size : smoke_train_size + smoke_val_size]

    train_records = [
        _make_dataset_record(
            selected_row=row,
            split="train",
            data_source=data_source,
            source_run_id=source_run_id,
            source_protocol=source_protocol,
        )
        for row in selected_rows
    ]
    smoke_train_records = [
        _make_dataset_record(
            selected_row=row,
            split="smoke_train_subset",
            data_source=data_source,
            source_run_id=source_run_id,
            source_protocol=source_protocol,
        )
        for row in smoke_train_rows
    ]
    smoke_val_records = [
        _make_dataset_record(
            selected_row=row,
            split="smoke_val_train_subset",
            data_source=data_source,
            source_run_id=source_run_id,
            source_protocol=source_protocol,
        )
        for row in smoke_val_rows
    ]

    files = {
        "train": output_dir / "train.parquet",
        "smoke_train": output_dir / "smoke_train.parquet",
        "smoke_val": output_dir / "smoke_val.parquet",
        "selected_rows": output_dir / "selected_rows.jsonl",
        "audit_suspect_blocklist": output_dir / "audit_suspect_blocklist.jsonl",
        "build_summary": output_dir / "build_summary.json",
    }

    _write_parquet(train_records, files["train"])
    _write_parquet(smoke_train_records, files["smoke_train"])
    _write_parquet(smoke_val_records, files["smoke_val"])
    _write_jsonl(files["selected_rows"], [_manifest_row(row) for row in selected_rows])
    _write_jsonl(files["audit_suspect_blocklist"], blocklist_rows)

    selected_error_type_counts = Counter(row["first_pass_error_type"] for row in selected_rows)
    selected_curriculum_bucket_counts = Counter(row["curriculum_bucket"] for row in selected_rows)
    selected_repair_stratum_counts = Counter(row["repair_stratum"] for row in selected_rows)
    feedback_source_counts = Counter(row["feedback_source"] for row in selected_rows)

    summary = {
        "input_paths": {
            "student_references": str(student_references_path),
            "teacher_requests": str(teacher_requests_path),
            "raw_data": str(raw_data_path),
            "blocklist_source": str(blocklist_source_path if blocklist_source_path.exists() else blocklist_fallback_path),
            "blocklist_asset": str(blocklist_asset_out),
        },
        "builder_config": {
            "prompt_mode": str(prompt_mode),
            "data_source": data_source,
            "source_run_id": source_run_id,
            "source_protocol": source_protocol,
            "source_split": source_split,
            "min_pass_ratio": min_pass_ratio,
            "high_threshold": high_threshold,
            "target_high_count": target_high_count,
            "target_mid_count": target_mid_count,
            "smoke_train_size": smoke_train_size,
            "smoke_val_size": smoke_val_size,
            "max_failure_cases": max_failure_cases,
            "max_case_chars": max_case_chars,
            "allowed_error_types": allowed_error_types,
            "allow_teacher_feedback_fallback": allow_teacher_feedback_fallback,
            "refresh_blocklist_asset": refresh_blocklist_asset,
            "seed": seed,
        },
        "counts": {
            **counts,
            "available_high_count": len(high_candidates),
            "available_mid_count": len(mid_candidates),
            "selected_high_count": len(selected_high),
            "selected_mid_count": len(selected_mid),
            "final_train_count": len(selected_rows),
            "smoke_train_count": len(smoke_train_rows),
            "smoke_val_count": len(smoke_val_rows),
            "upstream_audit_blocklist_count": blocklist_meta["row_count"],
            "join_miss_request_count": counts["missing_teacher_request_count"],
            "empty_teacher_feedback_count": counts["empty_teacher_feedback_count"],
            "join_miss_raw_count": 0,
        },
        "teacher_request_contract": {
            "allow_teacher_feedback_fallback": allow_teacher_feedback_fallback,
            "missing_teacher_request_count": counts["missing_teacher_request_count"],
            "empty_teacher_feedback_count": counts["empty_teacher_feedback_count"],
            "teacher_feedback_fallback_missing_request_count": counts[
                "teacher_feedback_fallback_missing_request_count"
            ],
            "teacher_feedback_fallback_empty_feedback_count": counts[
                "teacher_feedback_fallback_empty_feedback_count"
            ],
            "missing_teacher_request_preview": missing_teacher_request_preview[:20],
            "empty_teacher_feedback_preview": empty_teacher_feedback_preview[:20],
        },
        "available_slice_counts": {
            "high": len(high_candidates),
            "mid": len(mid_candidates),
        },
        "selected_slice_counts": {
            "high": len(selected_high),
            "mid": len(selected_mid),
        },
        "error_type_counts": dict(selected_error_type_counts),
        "curriculum_bucket_counts": dict(selected_curriculum_bucket_counts),
        "repair_stratum_counts": dict(selected_repair_stratum_counts),
        "feedback_source_counts": dict(feedback_source_counts),
        "removed_error_type_counts": dict(removed_by_error_type),
        "smoke_contract": {
            "smoke_train_is_train_subset": True,
            "smoke_val_is_train_subset": True,
            "smoke_val_usage": "trainer_bringup_only_not_heldout_validation",
        },
        "blocklist_meta": blocklist_meta,
        "blocklist_removed_rows_preview": blocklist_removed_rows[:50],
        "files": {name: str(path) for name, path in files.items()},
    }

    _write_json(files["build_summary"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build step1300 repair RL parquet for v0 probe")
    parser.add_argument("--student_references", type=Path, default=DEFAULT_STUDENT_REFERENCES)
    parser.add_argument("--teacher_requests", type=Path, default=DEFAULT_TEACHER_REQUESTS)
    parser.add_argument("--raw_data", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--blocklist_source", type=Path, default=DEFAULT_BLOCKLIST_SOURCE)
    parser.add_argument("--blocklist_fallback", type=Path, default=DEFAULT_BLOCKLIST_FALLBACK)
    parser.add_argument("--blocklist_asset_out", type=Path, default=DEFAULT_BLOCKLIST_ASSET)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt_mode", type=str, default="code_only", choices=["code_only", "short_diagnosis_code"])
    parser.add_argument("--data_source", type=str, default="codecontests_repair_rl")
    parser.add_argument("--source_run_id", type=str, default="step1300_probe_v0")
    parser.add_argument("--source_protocol", type=str, default="cached_first_pass_step1300_repair_cond_v2")
    parser.add_argument("--source_split", type=str, default="codecontests_train_wo_valid_big")
    parser.add_argument("--min_pass_ratio", type=float, default=0.2)
    parser.add_argument("--high_threshold", type=float, default=0.6)
    parser.add_argument("--target_high_count", type=int, default=231)
    parser.add_argument("--target_mid_count", type=int, default=77)
    parser.add_argument("--smoke_train_size", type=int, default=32)
    parser.add_argument("--smoke_val_size", type=int, default=16)
    parser.add_argument("--max_failure_cases", type=int, default=1)
    parser.add_argument("--max_case_chars", type=int, default=400)
    parser.add_argument(
        "--allowed_error_types",
        type=str,
        default="wrong_answer,runtime_error,timeout",
        help="Comma-separated first-pass error types to retain.",
    )
    parser.add_argument(
        "--allow_teacher_feedback_fallback",
        action="store_true",
        help="Allow missing or empty teacher feedback to fall back to feedback rebuilt from student references.",
    )
    parser.add_argument(
        "--refresh_blocklist_asset",
        action="store_true",
        help="Rebuild the materialized audit-suspect blocklist asset from the upstream keep set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    allowed_error_types = [value.strip() for value in args.allowed_error_types.split(",") if value.strip()]

    summary = build_repair_rl_dataset(
        student_references_path=args.student_references,
        teacher_requests_path=args.teacher_requests,
        raw_data_path=args.raw_data,
        blocklist_source_path=args.blocklist_source,
        blocklist_fallback_path=args.blocklist_fallback,
        blocklist_asset_out=args.blocklist_asset_out,
        output_dir=args.output_dir,
        prompt_mode=args.prompt_mode,
        data_source=args.data_source,
        source_run_id=args.source_run_id,
        source_protocol=args.source_protocol,
        source_split=args.source_split,
        min_pass_ratio=args.min_pass_ratio,
        high_threshold=args.high_threshold,
        target_high_count=args.target_high_count,
        target_mid_count=args.target_mid_count,
        smoke_train_size=args.smoke_train_size,
        smoke_val_size=args.smoke_val_size,
        max_failure_cases=args.max_failure_cases,
        max_case_chars=args.max_case_chars,
        allowed_error_types=allowed_error_types,
        allow_teacher_feedback_fallback=args.allow_teacher_feedback_fallback,
        refresh_blocklist_asset=args.refresh_blocklist_asset,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
