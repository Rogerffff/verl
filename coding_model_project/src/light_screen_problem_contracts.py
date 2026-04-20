#!/usr/bin/env python3
"""Light-screen CodeContests raw data for contract/test issues and quarantine candidates.

The goal is not to fully prove correctness of every problem. Instead, this script
looks for high-signal structural anomalies in hidden tests that are likely to
indicate dirty tasks:

* missing / malformed tests
* obvious marker contamination (e.g. SAMPLE / Explanation leakage)
* query-count mismatch (header says q lines, test provides a different count)
* per-opcode arity inconsistency in query-style tasks
* range violations for interval-query tasks
* first-line N vs next-line token-count mismatch for list-input tasks

It produces:

* a compact candidate JSONL for quarantine review
* an optional manual-review JSONL
* summary JSON
* markdown audit report
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from problem_quarantine import load_problem_quarantine


DEFAULT_SPLITS = {
    "codecontests_train": "coding_model_project/data/raw/codecontests_train_raw.jsonl",
    "codecontests_train_wo_valid_big": "coding_model_project/data/raw/codecontests_train_wo_valid_big_raw.jsonl",
    "codecontests_valid": "coding_model_project/data/raw/codecontests_valid_raw.jsonl",
    "codecontests_valid_big": "coding_model_project/data/raw/codecontests_valid_big_raw.jsonl",
    "codecontests_test": "coding_model_project/data/raw/codecontests_test_raw.jsonl",
}

DEFAULT_OUTPUT_DIR = Path("coding_model_project/data/quarantine_audit")
DEFAULT_QUARANTINE_PATH = Path("coding_model_project/data/problem_quarantine_v1.json")

MARKER_RE = re.compile(
    r"\b(SAMPLE|EXPLANATION|OUTPUT|INPUT|RAMPLE|ELBMPT|EXAMPLE)\b",
    flags=re.IGNORECASE,
)
UPPERCASE_BLOB_RE = re.compile(r"^[A-Z]{5,}$")
QUERY_HINT_RE = re.compile(r"\b(query|queries|operation|operations)\b", flags=re.IGNORECASE)
TYPED_QUERY_HINT_RE = re.compile(
    r"(queries of (one|two|three) types|query of type|type of the query|first digit represents the type|denotes .* query|described in the following format)",
    flags=re.IGNORECASE,
)
QUERY_COUNT_HINT_RE = re.compile(
    r"(number of queries|q\s*\(.*queries.*\)|query_i|i-th query|each of following\s+q\s+lines|next\s+m\s+lines\s+contain.*queries|next\s+q\s+lines)",
    flags=re.IGNORECASE,
)
FIRST_LINE_Q_HINT_RE = re.compile(
    r"(first line contains[^.\n]{0,120}\bq\b|in the first line[^.\n]{0,120}\bq\b|the first line[^.\n]{0,120}\bq\b)",
    flags=re.IGNORECASE,
)
QUERY_RANGE_RE = re.compile(
    r"1\s*[≤<=]+\s*[a-z][a-z_]*\s*[≤<=]+\s*[a-z][a-z_]*\s*[≤<=]+\s*n",
    flags=re.IGNORECASE,
)
NEXT_LINE_N_RE = re.compile(
    r"next line contains[^.\n]{0,120}\bN\b[^.\n]{0,80}space separated",
    flags=re.IGNORECASE,
)
NEXT_LINE_K_RE = re.compile(
    r"next line contains[^.\n]{0,120}\bK\b[^.\n]{0,80}space separated",
    flags=re.IGNORECASE,
)
SECOND_LINE_ARRAY_HINT_RE = re.compile(
    r"(second line contains[^.\n]{0,120}\bintegers\b|second line contains[^.\n]{0,120}\ba1\b|next line contains[^.\n]{0,120}\bn\b[^.\n]{0,80}integers)",
    flags=re.IGNORECASE,
)


@dataclass
class Signal:
    name: str
    severity: str
    count: int
    examples: list[Any]
    summary: str


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def _excerpt(text: str, max_chars: int = 500) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _all_int_tokens(tokens: list[str]) -> bool:
    return bool(tokens) and all(re.fullmatch(r"-?\d+", token) for token in tokens)


def _json_preview(value: Any, max_chars: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _signal(name: str, severity: str, items: list[Any], summary: str) -> Signal | None:
    if not items:
        return None
    return Signal(name=name, severity=severity, count=len(items), examples=items[:8], summary=summary)


def _detect_missing_tests(problem_id: str, tests: list[dict[str, Any]]) -> list[Signal]:
    out: list[Signal] = []
    if not tests:
        out.append(
            Signal(
                name="missing_tests",
                severity="hard",
                count=1,
                examples=[problem_id],
                summary="No hidden tests found under test_cases.tests.",
            )
        )
        return out

    malformed: list[Any] = []
    for idx, test in enumerate(tests):
        if not isinstance(test, dict):
            malformed.append({"test_idx": idx, "value_type": type(test).__name__})
            continue
        if "input" not in test or "output" not in test:
            malformed.append({"test_idx": idx, "keys": sorted(test.keys())})
            continue
        if not isinstance(test["input"], str) or not isinstance(test["output"], str):
            malformed.append(
                {
                    "test_idx": idx,
                    "input_type": type(test.get("input")).__name__,
                    "output_type": type(test.get("output")).__name__,
                }
            )
    malformed_signal = _signal(
        "malformed_tests",
        "hard",
        malformed,
        "Malformed hidden tests: missing input/output or wrong field types.",
    )
    if malformed_signal is not None:
        out.append(malformed_signal)
    return out


def _detect_marker_contamination(tests: list[dict[str, Any]]) -> list[Signal]:
    marker_hits: list[Any] = []
    uppercase_tail_hits: list[Any] = []
    for idx, test in enumerate(tests):
        input_text = str(test.get("input", ""))
        raw_lines = input_text.splitlines()
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if MARKER_RE.search(stripped):
                marker_hits.append({"test_idx": idx, "line": stripped[:160]})
                break
        for line in raw_lines:
            stripped = line.strip()
            if UPPERCASE_BLOB_RE.fullmatch(stripped) and len(set(stripped)) >= 4 and len(stripped) <= 16:
                uppercase_tail_hits.append({"test_idx": idx, "line": stripped})
                break

    out: list[Signal] = []
    marker_signal = _signal(
        "marker_contamination",
        "hard" if len(marker_hits) >= 2 else "manual",
        marker_hits,
        "Hidden tests contain obvious marker text such as SAMPLE/EXPLANATION/OUTPUT.",
    )
    if marker_signal is not None:
        out.append(marker_signal)
    uppercase_signal = _signal(
        "uppercase_blob_tail",
        "manual" if len(uppercase_tail_hits) >= 1 else "manual",
        uppercase_tail_hits,
        "Hidden tests contain standalone uppercase blobs that often indicate prompt/sample contamination.",
    )
    if uppercase_signal is not None:
        out.append(uppercase_signal)
    return out


def _candidate_query_skips(n: int, total_body_lines: int) -> list[int]:
    candidates = [0, 1, 2, 3, max(0, n - 1), n]
    out: list[int] = []
    for value in candidates:
        if 0 <= value <= total_body_lines and value not in out:
            out.append(value)
    return out


def _pick_window(lines: list[str], q: int, skips: list[int], *, width_hint: int | None = None) -> tuple[int, list[str]] | None:
    for skip in skips:
        window = lines[1 + skip : 1 + skip + q]
        if len(window) != q:
            continue
        if width_hint is None:
            return skip, window
        if all(len(entry.split()) == width_hint and _all_int_tokens(entry.split()) for entry in window[: min(len(window), 8)]):
            return skip, window
    return None


def _detect_query_count_mismatch(prompt: str, tests: list[dict[str, Any]]) -> list[Signal]:
    if not QUERY_COUNT_HINT_RE.search(prompt):
        return []
    if not FIRST_LINE_Q_HINT_RE.search(prompt):
        return []

    has_second_line_array = bool(SECOND_LINE_ARRAY_HINT_RE.search(prompt))
    mismatches: list[Any] = []
    for idx, test in enumerate(tests):
        lines = _nonempty_lines(str(test.get("input", "")))
        if len(lines) < 2:
            continue
        header = lines[0].split()
        if len(header) != 2 or not _all_int_tokens(header):
            continue
        n = int(header[0])
        q = int(header[1])
        body_lines = len(lines) - 1
        if q < 0:
            continue
        allowed_skips = _candidate_query_skips(n, body_lines)
        if has_second_line_array and 1 not in allowed_skips:
            allowed_skips.insert(0, 1)
        if any(body_lines - skip == q for skip in allowed_skips):
            continue
        best_skip = min(allowed_skips, key=lambda skip: abs((body_lines - skip) - q))
        body_count = body_lines - best_skip
        if body_count != q:
            mismatches.append(
                {
                    "test_idx": idx,
                    "header": lines[0],
                    "expected_q": q,
                    "actual_query_lines": body_count,
                    "extra_lines_before_queries": best_skip,
                }
            )

    signal = _signal(
        "query_count_mismatch",
        "hard" if len(mismatches) >= 2 else "manual",
        mismatches,
        "Header query count does not match the number of provided query lines.",
    )
    return [signal] if signal is not None else []


def _detect_opcode_arity_inconsistency(prompt: str, tests: list[dict[str, Any]]) -> list[Signal]:
    if not QUERY_HINT_RE.search(prompt):
        return []
    if not TYPED_QUERY_HINT_RE.search(prompt):
        return []

    opcode_arities: dict[str, set[int]] = defaultdict(set)
    examples: dict[str, list[Any]] = defaultdict(list)

    for idx, test in enumerate(tests):
        lines = _nonempty_lines(str(test.get("input", "")))
        if len(lines) < 3:
            continue
        header = lines[0].split()
        if len(header) != 2 or not _all_int_tokens(header):
            continue
        n = int(header[0])
        q = int(header[1])
        if q <= 0:
            continue
        picked = _pick_window(lines, q, _candidate_query_skips(n, len(lines) - 1))
        if picked is None:
            continue
        _, window = picked
        query_like = 0
        for line in window[: min(len(window), 8)]:
            tokens = line.split()
            if len(tokens) >= 2 and len(tokens) <= 5 and re.fullmatch(r"\d+", tokens[0]):
                query_like += 1
        if query_like < max(2, min(len(window), 8) // 2):
            continue
        for line in window:
            tokens = line.split()
            if len(tokens) < 2 or not re.fullmatch(r"\d+", tokens[0]):
                continue
            opcode = tokens[0]
            arity = len(tokens)
            opcode_arities[opcode].add(arity)
            if len(examples[opcode]) < 6:
                examples[opcode].append({"test_idx": idx, "line": line})

    bad: list[Any] = []
    for opcode, arities in opcode_arities.items():
        if len(arities) > 1:
            bad.append({"opcode": opcode, "arities": sorted(arities), "examples": examples[opcode][:4]})

    signal = _signal(
        "opcode_arity_inconsistency",
        "hard",
        bad,
        "Same numeric opcode appears with inconsistent token counts across hidden tests.",
    )
    return [signal] if signal is not None else []


def _detect_range_violations(prompt: str, tests: list[dict[str, Any]]) -> list[Signal]:
    prompt_lower = prompt.lower()
    if not QUERY_HINT_RE.search(prompt):
        return []
    if TYPED_QUERY_HINT_RE.search(prompt):
        return []
    if not (("li" in prompt_lower and "ri" in prompt_lower) or ("l_i" in prompt_lower and "r_i" in prompt_lower)):
        return []
    if not QUERY_RANGE_RE.search(prompt.replace("≤", "<=")):
        return []

    violations: list[Any] = []
    for idx, test in enumerate(tests):
        lines = _nonempty_lines(str(test.get("input", "")))
        if len(lines) < 3:
            continue
        header = lines[0].split()
        if len(header) != 2 or not _all_int_tokens(header):
            continue
        n = int(header[0])
        q = int(header[1])
        picked = _pick_window(lines, q, _candidate_query_skips(n, len(lines) - 1), width_hint=2)
        if picked is None:
            continue
        _, window = picked
        for line in window:
            tokens = line.split()
            if len(tokens) != 2 or not _all_int_tokens(tokens):
                continue
            left, right = map(int, tokens)
            if not (1 <= left <= right <= n):
                violations.append({"test_idx": idx, "header": lines[0], "line": line})
                break

    signal = _signal(
        "range_violation",
        "hard" if len(violations) >= 2 else "manual",
        violations,
        "Hidden tests contain interval queries outside the stated 1 <= l <= r <= n contract.",
    )
    return [signal] if signal is not None else []


def _detect_second_line_length_mismatch(prompt: str, tests: list[dict[str, Any]]) -> list[Signal]:
    prompt_lower = prompt.lower()
    wants_n = bool(NEXT_LINE_N_RE.search(prompt))
    wants_k = bool(NEXT_LINE_K_RE.search(prompt))
    if not wants_n and not wants_k:
        return []

    mismatches: list[Any] = []
    for idx, test in enumerate(tests):
        lines = _nonempty_lines(str(test.get("input", "")))
        if len(lines) < 2:
            continue
        header = lines[0].split()
        if len(header) < 1 or not _all_int_tokens(header):
            continue
        if wants_n:
            expected = int(header[0])
        elif wants_k and len(header) >= 2:
            expected = int(header[1])
        else:
            continue
        actual = len(lines[1].split())
        if expected >= 0 and actual != expected:
            mismatches.append({"test_idx": idx, "header": lines[0], "second_line_tokens": actual, "expected": expected})

    signal = _signal(
        "second_line_length_mismatch",
        "hard" if len(mismatches) >= 2 else "manual",
        mismatches,
        "Header count and second-line token count disagree in a prompt pattern that claims N/K space-separated integers.",
    )
    return [signal] if signal is not None else []


def _collect_signals(problem_id: str, prompt: str, tests: list[dict[str, Any]]) -> list[Signal]:
    signals: list[Signal] = []
    signals.extend(_detect_missing_tests(problem_id, tests))
    signals.extend(_detect_marker_contamination(tests))
    signals.extend(_detect_query_count_mismatch(prompt, tests))
    signals.extend(_detect_opcode_arity_inconsistency(prompt, tests))
    signals.extend(_detect_range_violations(prompt, tests))
    signals.extend(_detect_second_line_length_mismatch(prompt, tests))
    return signals


def _severity_weight(severity: str) -> int:
    return {"hard": 4, "manual": 2, "caution": 1}.get(severity, 1)


def _recommended_decision(signals: list[Signal], *, already_hard: bool, already_caution: bool) -> str:
    if already_hard:
        return "hard_blacklist_existing"
    if already_caution:
        return "caution_existing"
    hard_count = sum(1 for signal in signals if signal.severity == "hard")
    manual_count = sum(1 for signal in signals if signal.severity == "manual")
    if hard_count >= 1:
        return "hard_blacklist"
    if manual_count >= 1:
        return "manual_review"
    return "clean"


def _analyze_split(
    split_name: str,
    path: Path,
    *,
    hard_ids: set[str],
    caution_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_counter = Counter()
    signal_counter = Counter()
    decision_counter = Counter()

    for row in _read_jsonl(path):
        problem_id = str(row["problem_id"])
        prompt = str(row.get("canonical_prompt") or row["prompt"])
        tests = list((row.get("test_cases") or {}).get("tests") or [])
        signals = _collect_signals(problem_id, prompt, tests)
        for signal in signals:
            signal_counter[signal.name] += 1
        split_counter["total_rows"] += 1
        if not signals and problem_id not in hard_ids and problem_id not in caution_ids:
            continue

        score = sum(_severity_weight(signal.severity) * max(1, signal.count) for signal in signals)
        decision = _recommended_decision(
            signals,
            already_hard=problem_id in hard_ids,
            already_caution=problem_id in caution_ids,
        )
        decision_counter[decision] += 1
        rows.append(
            {
                "split": split_name,
                "problem_id": problem_id,
                "prompt_title": _first_nonempty_line(prompt),
                "prompt_excerpt": _excerpt(prompt),
                "tests_count": len(tests),
                "existing_quarantine_status": (
                    "hard_blacklist" if problem_id in hard_ids else "caution" if problem_id in caution_ids else ""
                ),
                "recommended_decision": decision,
                "risk_score": score,
                "signals": [
                    {
                        "name": signal.name,
                        "severity": signal.severity,
                        "count": signal.count,
                        "summary": signal.summary,
                        "examples": signal.examples,
                    }
                    for signal in signals
                ],
            }
        )

    summary = {
        "split": split_name,
        "raw_path": str(path),
        "total_rows": split_counter["total_rows"],
        "candidate_rows": len(rows),
        "signal_counts": dict(signal_counter),
        "decision_counts": dict(decision_counter),
    }
    return rows, summary


def _build_markdown_report(
    *,
    summaries: list[dict[str, Any]],
    problem_candidates: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    total_candidates = len(problem_candidates)
    by_decision = Counter(row["recommended_decision"] for row in problem_candidates)
    by_signal = Counter()
    for row in problem_candidates:
        for signal in row["merged_signals"]:
            by_signal[signal["name"]] += 1

    lines = [
        "# Problem Contract Audit v1",
        "",
        "This is a lightweight structural screen, not a full semantic proof.",
        "For merged signal rows, `count` means the max single-split count for that problem; `raw_count` is the cross-split aggregate.",
        "",
        "## Totals",
        "",
        f"- candidate_count: {total_candidates}",
        f"- hard_blacklist_new: {by_decision.get('hard_blacklist', 0)}",
        f"- manual_review: {by_decision.get('manual_review', 0)}",
        f"- already_hard_blacklist: {by_decision.get('hard_blacklist_existing', 0)}",
        f"- already_caution: {by_decision.get('caution_existing', 0)}",
        "",
        "## Split Summary",
        "",
    ]
    for summary in summaries:
        lines.append(
            f"- {summary['split']}: total_rows={summary['total_rows']}, "
            f"candidate_rows={summary['candidate_rows']}, decision_counts={summary['decision_counts']}"
        )
    lines.extend(["", "## Signal Counts", ""])
    for name, count in by_signal.most_common():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Top Risk Preview", ""])
    for row in sorted(problem_candidates, key=lambda item: (-int(item["max_risk_score"]), item["problem_id"]))[:25]:
        signal_names = ", ".join(
            f"{signal['name']}[{signal['count']}; raw={signal.get('raw_count', signal['count'])}; splits={signal.get('split_count', 1)}]/{signal['severity']}"
            for signal in row["merged_signals"]
        )
        lines.append(
            f"- {row['problem_id']} (splits={','.join(row['seen_splits'])}): decision={row['recommended_decision']}, "
            f"risk_score={row['max_risk_score']}, signals={signal_names}"
        )
    lines.extend(
        [
            "",
            f"Artifacts written under `{output_dir}`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quarantine-path", type=Path, default=DEFAULT_QUARANTINE_PATH)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(DEFAULT_SPLITS.keys()),
        help="Subset of dataset splits to screen.",
    )
    return parser.parse_args()


def _aggregate_problem_candidates(split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_rows:
        grouped[str(row["problem_id"])].append(row)

    aggregated: list[dict[str, Any]] = []
    for problem_id, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda item: (-int(item["risk_score"]), item["split"]))
        merged_signal_counter_raw: Counter[tuple[str, str]] = Counter()
        merged_signal_counter_max: dict[tuple[str, str], int] = defaultdict(int)
        merged_signal_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
        merged_examples: dict[tuple[str, str], list[Any]] = {}
        merged_example_keys: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in rows_sorted:
            for signal in row["signals"]:
                key = (signal["name"], signal["severity"])
                merged_signal_counter_raw[key] += signal["count"]
                merged_signal_counter_max[key] = max(merged_signal_counter_max[key], int(signal["count"]))
                merged_signal_splits[key].add(str(row["split"]))
                merged_examples.setdefault(key, [])
                for example in signal["examples"]:
                    if len(merged_examples[key]) >= 8:
                        break
                    example_key = json.dumps(example, ensure_ascii=False, sort_keys=True)
                    if example_key in merged_example_keys[key]:
                        continue
                    merged_example_keys[key].add(example_key)
                    merged_examples[key].append(example)

        decisions = {row["recommended_decision"] for row in rows_sorted}
        if "hard_blacklist_existing" in decisions:
            decision = "hard_blacklist_existing"
        elif "hard_blacklist" in decisions:
            decision = "hard_blacklist"
        elif "caution_existing" in decisions:
            decision = "caution_existing"
        else:
            decision = "manual_review"

        aggregated.append(
            {
                "problem_id": problem_id,
                "seen_splits": sorted({row["split"] for row in rows_sorted}),
                "existing_quarantine_status": rows_sorted[0]["existing_quarantine_status"],
                "recommended_decision": decision,
                "max_risk_score": max(int(row["risk_score"]) for row in rows_sorted),
                "prompt_title": rows_sorted[0]["prompt_title"],
                "prompt_excerpt": rows_sorted[0]["prompt_excerpt"],
                "tests_count_examples": sorted({int(row["tests_count"]) for row in rows_sorted}),
                "merged_signals": [
                    {
                        "name": name,
                        "severity": severity,
                        "count": count,
                        "raw_count": int(merged_signal_counter_raw[(name, severity)]),
                        "split_count": len(merged_signal_splits[(name, severity)]),
                        "split_names": sorted(merged_signal_splits[(name, severity)]),
                        "examples": merged_examples[(name, severity)][:8],
                    }
                    for (name, severity), count in sorted(
                        merged_signal_counter_max.items(),
                        key=lambda item: (-item[1], item[0][0], item[0][1]),
                    )
                ],
                "split_details": [
                    {
                        "split": row["split"],
                        "recommended_decision": row["recommended_decision"],
                        "risk_score": row["risk_score"],
                        "signals": row["signals"],
                    }
                    for row in rows_sorted
                ],
            }
        )

    aggregated.sort(key=lambda row: (-int(row["max_risk_score"]), row["problem_id"]))
    return aggregated


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    quarantine = load_problem_quarantine(args.quarantine_path)
    hard_ids = set(quarantine.get("hard_blacklist_ids", set()))
    caution_ids = set(quarantine.get("caution_ids", set()))

    split_paths = {name: repo_root / DEFAULT_SPLITS[name] for name in args.splits}
    split_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for split_name, path in split_paths.items():
        rows, summary = _analyze_split(split_name, path, hard_ids=hard_ids, caution_ids=caution_ids)
        split_rows.extend(rows)
        summaries.append(summary)

    split_rows.sort(key=lambda row: (-int(row["risk_score"]), row["problem_id"], row["split"]))
    candidates = _aggregate_problem_candidates(split_rows)
    manual_review_rows = [row for row in candidates if row["recommended_decision"] == "manual_review"]
    hard_rows = [row for row in candidates if row["recommended_decision"] == "hard_blacklist"]

    summary_payload = {
        "version": "problem_contract_screen_v1",
        "quarantine_path": str(args.quarantine_path),
        "splits": summaries,
        "split_candidate_row_count": len(split_rows),
        "candidate_count": len(candidates),
        "manual_review_count": len(manual_review_rows),
        "hard_blacklist_new_count": len(hard_rows),
        "existing_hard_count": sum(1 for row in candidates if row["recommended_decision"] == "hard_blacklist_existing"),
        "existing_caution_count": sum(1 for row in candidates if row["recommended_decision"] == "caution_existing"),
    }

    output_dir = args.output_dir
    _write_jsonl(output_dir / "problem_contract_screen_rows_v1.jsonl", split_rows)
    _write_jsonl(output_dir / "problem_quarantine_candidates_v2.jsonl", candidates)
    _write_jsonl(output_dir / "problem_manual_review_candidates_v1.jsonl", manual_review_rows)
    _write_json(output_dir / "problem_contract_screen_summary_v1.json", summary_payload)
    (output_dir / "problem_contract_audit_report_v1.md").write_text(
        _build_markdown_report(summaries=summaries, problem_candidates=candidates, output_dir=output_dir),
        encoding="utf-8",
    )

    print(f"Wrote {len(split_rows)} split-level rows to {output_dir / 'problem_contract_screen_rows_v1.jsonl'}")
    print(f"Wrote {len(candidates)} quarantine candidates to {output_dir / 'problem_quarantine_candidates_v2.jsonl'}")
    print(f"Wrote {len(manual_review_rows)} manual-review rows to {output_dir / 'problem_manual_review_candidates_v1.jsonl'}")
    print(f"Wrote summary to {output_dir / 'problem_contract_screen_summary_v1.json'}")
    print(f"Wrote markdown report to {output_dir / 'problem_contract_audit_report_v1.md'}")


if __name__ == "__main__":
    main()
