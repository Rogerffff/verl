#!/usr/bin/env python3
"""
Badcase analyzer for Phase-0 eval outputs.

Usage:
  python src/badcase_analyze.py --run_dir outputs/phase0_20260204_121439 --dataset codecontests_valid

Notes:
  - Prefers full per-problem logs at: <run_dir>/per_problem/<dataset>.jsonl
  - Falls back to sampled QA logs at: <run_dir>/qa_logs/<dataset>_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.I | re.S)
MD_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.S)


def _extract_code(text: str) -> str:
    if not text:
        return ""
    matches = CODE_TAG_RE.findall(text)
    if matches:
        return max(matches, key=len).strip()
    matches = MD_FENCE_RE.findall(text)
    if matches:
        return max(matches, key=len).strip()
    return text.strip()


def _wrapper_type(text: str) -> str:
    if not text:
        return "empty"
    if CODE_TAG_RE.search(text):
        return "code_tag"
    if "```" in text:
        return "md_fence"
    return "raw"


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def _guess_exception(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer "ValueError:" style
    m = re.findall(r"\b([A-Za-z_]*Error|Exception)\s*:", text)
    if m:
        return m[-1]
    # Fallback: any "...Error"
    m = re.findall(r"\b([A-Za-z_]*Error|Exception)\b", text)
    if m:
        return m[-1]
    return None


def _parse_expected_got(last_error: str) -> Tuple[Optional[str], Optional[str]]:
    if not last_error:
        return None, None
    m = re.search(r"Expected:\s*(.*?),\s*Got:\s*(.*)$", last_error, flags=re.S)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass
class Record:
    dataset: str
    problem_id: str
    prompt: str
    response: str
    accepted: bool
    pass_ratio: float
    error_type: str
    details: Dict[str, Any]


def _load_records(run_dir: Path, dataset: str) -> Tuple[List[Record], str, Path]:
    per_problem_path = run_dir / "per_problem" / f"{dataset}.jsonl"
    qa_path = run_dir / "qa_logs" / f"{dataset}_qa.jsonl"

    if per_problem_path.exists():
        records: List[Record] = []
        for obj in _iter_jsonl(per_problem_path):
            records.append(
                Record(
                    dataset=obj.get("dataset", dataset),
                    problem_id=str(obj.get("problem_id", "")),
                    prompt=obj.get("prompt", "") or "",
                    response=obj.get("response", "") or "",
                    accepted=bool(obj.get("accepted", False)),
                    pass_ratio=float(obj.get("pass_ratio", 0.0) or 0.0),
                    error_type=str(obj.get("error_type", "unknown")),
                    details=(obj.get("details") or {}) if isinstance(obj.get("details"), dict) else {},
                )
            )
        return records, "per_problem", per_problem_path

    if qa_path.exists():
        records = []
        for obj in _iter_jsonl(qa_path):
            extra = obj.get("extra") or {}
            details = extra.get("details") if isinstance(extra, dict) else {}
            records.append(
                Record(
                    dataset=obj.get("dataset", dataset),
                    problem_id=str(obj.get("problem_id", "")),
                    prompt=obj.get("prompt", "") or "",
                    response=obj.get("response", "") or "",
                    accepted=bool(obj.get("accepted", False)),
                    pass_ratio=float(obj.get("pass_ratio", 0.0) or 0.0),
                    error_type=str(obj.get("error_type", "unknown")),
                    details=details if isinstance(details, dict) else {},
                )
            )
        return records, "qa_logs(sampled)", qa_path

    raise FileNotFoundError(f"Neither {per_problem_path} nor {qa_path} exists")


def _available_datasets(run_dir: Path) -> List[str]:
    datasets = []
    qa_dir = run_dir / "qa_logs"
    if qa_dir.exists():
        for p in qa_dir.glob("*_qa.jsonl"):
            datasets.append(p.name[: -len("_qa.jsonl")])
    per_dir = run_dir / "per_problem"
    if per_dir.exists():
        for p in per_dir.glob("*.jsonl"):
            datasets.append(p.stem)
    return sorted(set(datasets))


def analyze_records(records: List[Record]) -> Dict[str, Any]:
    total = len(records)
    by_error = Counter(r.error_type for r in records)
    accepted = sum(1 for r in records if r.accepted)

    wrap = Counter(_wrapper_type(r.response) for r in records)
    code_features = Counter()

    # For codecontests-focused format heuristics
    wa_mismatch = Counter()
    runtime_exc = Counter()
    empty_stdout_wa = 0

    examples_by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in records:
        code = _extract_code(r.response)
        has_solve_def = bool(re.search(r"\bdef\s+solve\s*\(", code))
        has_main_guard = bool(re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", code))
        has_print = "print(" in code
        reads_stdin = ("sys.stdin" in code) or ("input(" in code)

        if has_solve_def:
            code_features["has_solve_def"] += 1
        if has_main_guard:
            code_features["has_main_guard"] += 1
        if has_print:
            code_features["has_print"] += 1
        if reads_stdin:
            code_features["reads_stdin"] += 1

        # Wrong answer mismatch categorization
        if r.error_type == "wrong_answer":
            fmt = r.details.get("format_match_type")
            if isinstance(fmt, str) and fmt != "strict":
                wa_mismatch[fmt] += 1

            mm = r.details.get("mismatch_summary") if isinstance(r.details, dict) else None
            if isinstance(mm, dict) and isinstance(mm.get("first_mismatch"), dict):
                got_stdout = (mm["first_mismatch"].get("got_stdout") or "").strip()
                if got_stdout == "":
                    empty_stdout_wa += 1
                    wa_mismatch["empty_stdout_possible"] += 1
            else:
                exp, got = _parse_expected_got(r.details.get("last_error", "") if isinstance(r.details, dict) else "")
                if exp is not None and got is not None:
                    exp_s = exp.strip()
                    got_s = got.strip()
                    if got_s == "":
                        empty_stdout_wa += 1
                        wa_mismatch["empty_stdout_possible"] += 1
                    elif got_s.lower() == exp_s.lower() and got_s != exp_s:
                        wa_mismatch["case_insensitive_possible"] += 1
                    elif _normalize_ws(got_s) == _normalize_ws(exp_s) and got_s != exp_s:
                        wa_mismatch["ws_insensitive_possible"] += 1
                    elif _normalize_ws(got_s.lower()) == _normalize_ws(exp_s.lower()) and got_s != exp_s:
                        wa_mismatch["case+ws_insensitive_possible"] += 1
                    else:
                        wa_mismatch["semantic_mismatch"] += 1

        if r.error_type == "runtime_error":
            le = r.details.get("last_error", "") if isinstance(r.details, dict) else ""
            exc = _guess_exception(le) or "unknown"
            runtime_exc[exc] += 1

        # Examples per bucket (cap)
        bucket = r.error_type
        if len(examples_by_bucket[bucket]) < 5:
            examples_by_bucket[bucket].append(
                {
                    "problem_id": r.problem_id,
                    "accepted": r.accepted,
                    "pass_ratio": r.pass_ratio,
                    "prompt_preview": (r.prompt or "")[:240].replace("\n", "\\n"),
                    "last_error": (r.details.get("last_error", "") if isinstance(r.details, dict) else "")[:240].replace(
                        "\n", "\\n"
                    ),
                }
            )

    return {
        "total": total,
        "accepted": accepted,
        "accepted_rate": (accepted / total) if total else 0.0,
        "by_error_type": dict(by_error),
        "wrapper_type": dict(wrap),
        "code_features": dict(code_features),
        "wrong_answer_mismatch": dict(wa_mismatch),
        "wrong_answer_empty_stdout_count": empty_stdout_wa,
        "runtime_error_exceptions": dict(runtime_exc),
        "examples": examples_by_bucket,
    }


def render_markdown(dataset: str, source_kind: str, source_path: Path, stats: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Badcase report: {dataset}")
    lines.append("")
    lines.append(f"- Source: `{source_kind}` -> `{source_path}`")
    lines.append(f"- Total records: {stats['total']}")
    lines.append(f"- Accepted: {stats['accepted']} ({stats['accepted_rate']:.2%})")
    lines.append("")

    lines.append("## Error distribution")
    for k, v in sorted(stats["by_error_type"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Format / structure heuristics")
    lines.append("### Wrapper type")
    for k, v in sorted(stats["wrapper_type"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("### Code features (heuristic)")
    for k, v in sorted(stats["code_features"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")

    if stats["wrong_answer_mismatch"]:
        lines.append("## Wrong-answer mismatch buckets (best-effort)")
        for k, v in sorted(stats["wrong_answer_mismatch"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {k}: {v}")
        lines.append("")

    if stats["runtime_error_exceptions"]:
        lines.append("## Runtime error exceptions")
        for k, v in sorted(stats["runtime_error_exceptions"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("## Examples (up to 5 each)")
    for bucket, examples in stats["examples"].items():
        lines.append(f"### {bucket}")
        for ex in examples:
            lines.append(f"- {ex['problem_id']} (pass_ratio={ex['pass_ratio']}) last_error={ex['last_error']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase-0 badcases from outputs.")
    parser.add_argument("--run_dir", type=str, required=True, help="Phase-0 output dir, e.g. outputs/phase0_YYYYmmdd_HHMMSS")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to analyze (default: all found)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for reports (default: run_dir)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else _available_datasets(run_dir)
    if not datasets:
        raise SystemExit(f"No datasets found under: {run_dir}")

    for ds in datasets:
        records, source_kind, source_path = _load_records(run_dir, ds)
        stats = analyze_records(records)

        md = render_markdown(ds, source_kind, source_path, stats)
        md_path = out_dir / f"badcase_report_{ds}.md"
        json_path = out_dir / f"badcase_report_{ds}.json"
        md_path.write_text(md, encoding="utf-8")
        json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[{ds}] wrote: {md_path}")
        print(f"[{ds}] wrote: {json_path}")


if __name__ == "__main__":
    main()

