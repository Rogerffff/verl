#!/usr/bin/env python3
"""
Build CodeContests valid_big split from train split.

Behavior:
1. Sample `valid_big_size` problems from train manifest with a required prefix
2. Write sampled records to codecontests_valid_big_{manifest,raw}.jsonl
3. Write remaining train records to codecontests_train_wo_valid_big_{manifest,raw}.jsonl
4. Write audit metadata JSON for reproducibility
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_lines(lines: List[str]) -> str:
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _collect_manifest(manifest_path: Path) -> Tuple[List[Dict], Set[str]]:
    rows: List[Dict] = []
    ids: Set[str] = set()

    for obj in _iter_jsonl(manifest_path):
        pid = obj.get("problem_id")
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"Manifest row missing valid problem_id: {obj}")
        if pid in ids:
            raise ValueError(f"Duplicate problem_id in manifest: {pid}")
        ids.add(pid)
        rows.append(obj)

    return rows, ids


def _split_manifest(
    manifest_rows: List[Dict],
    prefix: str,
    sample_size: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict], Set[str], Dict]:
    candidates = [
        row["problem_id"]
        for row in manifest_rows
        if isinstance(row.get("problem_id"), str) and row["problem_id"].startswith(prefix)
    ]
    candidates_sorted = sorted(candidates)

    if len(candidates_sorted) < sample_size:
        raise ValueError(
            f"Not enough candidates for prefix '{prefix}': "
            f"{len(candidates_sorted)} < requested {sample_size}"
        )

    rng = random.Random(seed)
    sampled_ids = set(rng.sample(candidates_sorted, sample_size))

    valid_big_manifest: List[Dict] = []
    train_wo_manifest: List[Dict] = []
    for row in manifest_rows:
        if row["problem_id"] in sampled_ids:
            valid_big_manifest.append(row)
        else:
            train_wo_manifest.append(row)

    info = {
        "candidate_count": len(candidates_sorted),
        "sampled_count": len(sampled_ids),
        "prefix": prefix,
        "seed": seed,
    }
    return valid_big_manifest, train_wo_manifest, sampled_ids, info


def _split_raw(raw_path: Path, sampled_ids: Set[str]) -> Tuple[List[Dict], List[Dict], Set[str]]:
    valid_big_raw: List[Dict] = []
    train_wo_raw: List[Dict] = []
    seen_ids: Set[str] = set()

    for obj in _iter_jsonl(raw_path):
        pid = obj.get("problem_id")
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"Raw row missing valid problem_id: {obj}")
        if pid in seen_ids:
            raise ValueError(f"Duplicate problem_id in raw: {pid}")
        seen_ids.add(pid)

        if pid in sampled_ids:
            valid_big_raw.append(obj)
        else:
            train_wo_raw.append(obj)

    return valid_big_raw, train_wo_raw, seen_ids


def _validate_outputs(
    valid_big_manifest: List[Dict],
    train_wo_manifest: List[Dict],
    valid_big_raw: List[Dict],
    train_wo_raw: List[Dict],
    sampled_ids: Set[str],
    manifest_ids: Set[str],
    raw_ids: Set[str],
    prefix: str,
    requested_size: int,
) -> Dict:
    valid_big_manifest_ids = {r["problem_id"] for r in valid_big_manifest}
    train_wo_manifest_ids = {r["problem_id"] for r in train_wo_manifest}
    valid_big_raw_ids = {r["problem_id"] for r in valid_big_raw}
    train_wo_raw_ids = {r["problem_id"] for r in train_wo_raw}

    checks = {
        "valid_big_manifest_count": len(valid_big_manifest),
        "train_wo_manifest_count": len(train_wo_manifest),
        "valid_big_raw_count": len(valid_big_raw),
        "train_wo_raw_count": len(train_wo_raw),
        "manifest_partition_disjoint": len(valid_big_manifest_ids & train_wo_manifest_ids) == 0,
        "manifest_partition_complete": (valid_big_manifest_ids | train_wo_manifest_ids) == manifest_ids,
        "raw_partition_disjoint": len(valid_big_raw_ids & train_wo_raw_ids) == 0,
        "raw_partition_complete": (valid_big_raw_ids | train_wo_raw_ids) == raw_ids,
        "valid_big_manifest_all_prefix": all(pid.startswith(prefix) for pid in valid_big_manifest_ids),
        "valid_big_manifest_matches_sampled": valid_big_manifest_ids == sampled_ids,
        "valid_big_raw_matches_sampled": valid_big_raw_ids == sampled_ids,
        "sampled_in_manifest": sampled_ids.issubset(manifest_ids),
        "sampled_in_raw": sampled_ids.issubset(raw_ids),
        "requested_size_matched": len(sampled_ids) == requested_size,
    }

    failed = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    if failed:
        raise ValueError(f"Validation failed: {failed}")

    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build codecontests_valid_big split from codecontests_train."
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="coding_model_project/data/manifests/codecontests_train_manifest.jsonl",
        help="Path to codecontests_train_manifest.jsonl",
    )
    parser.add_argument(
        "--train-raw",
        type=str,
        default="coding_model_project/data/raw/codecontests_train_raw.jsonl",
        help="Path to codecontests_train_raw.jsonl",
    )
    parser.add_argument(
        "--valid-big-size",
        type=int,
        default=500,
        help="Number of samples for valid_big.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="Codeforces/",
        help="problem_id prefix filter for candidate pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--out-manifest-dir",
        type=str,
        default="coding_model_project/data/manifests",
        help="Output directory for manifest files.",
    )
    parser.add_argument(
        "--out-raw-dir",
        type=str,
        default="coding_model_project/data/raw",
        help="Output directory for raw files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_manifest_path = Path(args.train_manifest)
    train_raw_path = Path(args.train_raw)
    out_manifest_dir = Path(args.out_manifest_dir)
    out_raw_dir = Path(args.out_raw_dir)

    valid_big_manifest_path = out_manifest_dir / "codecontests_valid_big_manifest.jsonl"
    train_wo_manifest_path = out_manifest_dir / "codecontests_train_wo_valid_big_manifest.jsonl"
    valid_big_raw_path = out_raw_dir / "codecontests_valid_big_raw.jsonl"
    train_wo_raw_path = out_raw_dir / "codecontests_train_wo_valid_big_raw.jsonl"
    meta_path = out_manifest_dir / "codecontests_valid_big_split_meta.json"

    manifest_rows, manifest_ids = _collect_manifest(train_manifest_path)
    valid_big_manifest, train_wo_manifest, sampled_ids, sample_info = _split_manifest(
        manifest_rows=manifest_rows,
        prefix=args.prefix,
        sample_size=args.valid_big_size,
        seed=args.seed,
    )

    valid_big_raw, train_wo_raw, raw_ids = _split_raw(train_raw_path, sampled_ids)

    checks = _validate_outputs(
        valid_big_manifest=valid_big_manifest,
        train_wo_manifest=train_wo_manifest,
        valid_big_raw=valid_big_raw,
        train_wo_raw=train_wo_raw,
        sampled_ids=sampled_ids,
        manifest_ids=manifest_ids,
        raw_ids=raw_ids,
        prefix=args.prefix,
        requested_size=args.valid_big_size,
    )

    _write_jsonl(valid_big_manifest_path, valid_big_manifest)
    _write_jsonl(train_wo_manifest_path, train_wo_manifest)
    _write_jsonl(valid_big_raw_path, valid_big_raw)
    _write_jsonl(train_wo_raw_path, train_wo_raw)

    sampled_ids_sorted = sorted(sampled_ids)
    sampled_prompt_hashes_sorted = sorted(r["prompt_sha256"] for r in valid_big_manifest if "prompt_sha256" in r)
    meta = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "train_manifest": str(train_manifest_path),
            "train_raw": str(train_raw_path),
        },
        "outputs": {
            "valid_big_manifest": str(valid_big_manifest_path),
            "train_wo_valid_big_manifest": str(train_wo_manifest_path),
            "valid_big_raw": str(valid_big_raw_path),
            "train_wo_valid_big_raw": str(train_wo_raw_path),
        },
        "sampling": {
            "prefix": args.prefix,
            "seed": args.seed,
            "requested_valid_big_size": args.valid_big_size,
            **sample_info,
        },
        "counts": {
            "train_manifest_before": len(manifest_rows),
            "train_manifest_after": len(train_wo_manifest),
            "valid_big_manifest": len(valid_big_manifest),
            "train_raw_before": len(raw_ids),
            "train_raw_after": len(train_wo_raw),
            "valid_big_raw": len(valid_big_raw),
        },
        "hashes": {
            "sampled_problem_ids_sha256": _sha256_lines(sampled_ids_sorted),
            "sampled_prompt_sha256_sha256": _sha256_lines(sampled_prompt_hashes_sorted),
        },
        "checks": checks,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Build valid_big split finished.")
    print(f"  valid_big_manifest: {valid_big_manifest_path} ({len(valid_big_manifest)})")
    print(f"  valid_big_raw: {valid_big_raw_path} ({len(valid_big_raw)})")
    print(f"  train_wo_manifest: {train_wo_manifest_path} ({len(train_wo_manifest)})")
    print(f"  train_wo_raw: {train_wo_raw_path} ({len(train_wo_raw)})")
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    main()
