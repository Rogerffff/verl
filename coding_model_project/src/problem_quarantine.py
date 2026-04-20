#!/usr/bin/env python3
"""Shared problem quarantine helpers for RL/SFT/curriculum pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_problem_quarantine(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "version": "none",
            "hard_blacklist": [],
            "caution": [],
            "unresolved": [],
            "hard_blacklist_ids": set(),
            "caution_ids": set(),
            "unresolved_ids": set(),
            "all_ids": set(),
        }

    quarantine_path = Path(path)
    if not quarantine_path.exists():
        raise FileNotFoundError(f"Problem quarantine file not found: {quarantine_path}")

    payload = _read_json(quarantine_path)
    hard_rows = list(payload.get("hard_blacklist", []))
    caution_rows = list(payload.get("caution", []))
    unresolved_rows = list(payload.get("unresolved", []))

    hard_ids = {str(row["problem_id"]) for row in hard_rows}
    caution_ids = {str(row["problem_id"]) for row in caution_rows}
    unresolved_ids = {str(row["problem_id"]) for row in unresolved_rows}
    all_ids = hard_ids | caution_ids | unresolved_ids

    return {
        **payload,
        "path": str(quarantine_path),
        "hard_blacklist": hard_rows,
        "caution": caution_rows,
        "unresolved": unresolved_rows,
        "hard_blacklist_ids": hard_ids,
        "caution_ids": caution_ids,
        "unresolved_ids": unresolved_ids,
        "all_ids": all_ids,
    }


def hard_blacklist_ids(quarantine: dict[str, Any]) -> set[str]:
    return set(quarantine.get("hard_blacklist_ids", set()))


def caution_ids(quarantine: dict[str, Any]) -> set[str]:
    return set(quarantine.get("caution_ids", set()))


def unresolved_ids(quarantine: dict[str, Any]) -> set[str]:
    return set(quarantine.get("unresolved_ids", set()))


def quarantine_summary(quarantine: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": quarantine.get("path"),
        "version": quarantine.get("version"),
        "hard_blacklist_count": len(quarantine.get("hard_blacklist_ids", set())),
        "caution_count": len(quarantine.get("caution_ids", set())),
        "unresolved_count": len(quarantine.get("unresolved_ids", set())),
        "all_ids_count": len(quarantine.get("all_ids", set())),
        "hard_blacklist_problem_ids": sorted(quarantine.get("hard_blacklist_ids", set())),
        "caution_problem_ids": sorted(quarantine.get("caution_ids", set())),
        "unresolved_problem_ids": sorted(quarantine.get("unresolved_ids", set())),
    }
