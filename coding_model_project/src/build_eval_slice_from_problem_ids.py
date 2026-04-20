#!/usr/bin/env python3
"""Build a manifest/raw eval slice from a list of problem_ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_problem_ids(path: Path, membership_key: str | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problem_id = str(row["problem_id"])
            if membership_key and membership_key not in row.get("slice_memberships", []):
                continue
            if problem_id in seen:
                continue
            seen.add(problem_id)
            ids.append(problem_id)
    return ids


def _load_raw_records(raw_path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records[str(row["problem_id"])] = row
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a custom eval slice from problem_ids.")
    parser.add_argument("--problem-ids-jsonl", type=Path, required=True)
    parser.add_argument("--membership-key", type=str, default=None)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--dataset-key", type=str, default="codecontests_valid_big")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    problem_ids = _load_problem_ids(args.problem_ids_jsonl, args.membership_key)
    source_raw_path = args.source_manifest_dir.parent / "raw" / f"{args.dataset_key}_raw.jsonl"
    raw_records = _load_raw_records(source_raw_path)

    missing = [pid for pid in problem_ids if pid not in raw_records]
    if missing:
        raise SystemExit(f"Missing {len(missing)} problem_ids from raw: {missing[:10]}")

    manifest_dir = args.output_dir / "manifests"
    raw_dir = args.output_dir / "raw"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f"{args.dataset_key}_manifest.jsonl"
    raw_path = raw_dir / f"{args.dataset_key}_raw.jsonl"
    meta_path = args.output_dir / "slice_meta.json"

    with manifest_path.open("w", encoding="utf-8") as mf, raw_path.open("w", encoding="utf-8") as rf:
        for pid in problem_ids:
            record = raw_records[pid]
            mf.write(json.dumps({"problem_id": pid}, ensure_ascii=False) + "\n")
            rf.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta = {
        "dataset_key": args.dataset_key,
        "problem_count": len(problem_ids),
        "membership_key": args.membership_key,
        "problem_ids_jsonl": str(args.problem_ids_jsonl),
        "source_manifest_dir": str(args.source_manifest_dir),
        "manifest_path": str(manifest_path),
        "raw_path": str(raw_path),
        "problem_ids": problem_ids,
        "missing_problem_ids": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
