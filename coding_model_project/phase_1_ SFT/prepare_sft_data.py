#!/usr/bin/env python3
"""
Phase 1 SFT - Data Preparation Script
======================================

Convert sft_validated.jsonl to parquet format for verl MultiTurnSFTDataset.

Input:  sft_validated.jsonl (2,220 records with problem_id, text, solution, test_cases)
Output: sft_train.parquet + sft_val.parquet with 'messages' column in OpenAI format.

Usage:
    python prepare_sft_data.py \
        --input_jsonl bee_hq_python_deduped_filtered/sft_validated.jsonl \
        --output_dir data/ \
        --val_size 200 \
        --seed 42
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Prompt Templates (must match phase0_eval.py exactly)
# =============================================================================

SYSTEM_PROMPT = """You are an expert Python programmer.

Output rules:
1. Output Python code only.
2. Include necessary imports only if needed.
3. Wrap the entire code in <code> and </code>.
4. Do not write anything outside the <code> tags.
5. Follow dataset-specific constraints given by the user prompt (function-only vs full program)."""

CODECONTESTS_USER_TEMPLATE = """Solve the following competitive programming problem in Python.

Rules:
- Read from stdin and write to stdout.
- Your program MUST produce output when executed (call solve() under main guard, or execute at top-level).
- Use fast I/O if needed (sys.stdin.buffer).
- Do NOT print anything except the required output.

{prompt}

Output ONLY:
<code>
# python code
</code>"""


# =============================================================================
# BEE Instruction Prefix Stripping
# =============================================================================

# All BEE instruction prefixes contain "python3" (case-insensitive) followed by
# a colon, then the actual problem text. Examples:
#   "Please create a solution in Python3 to the following problem:"
#   "Develop a solution in python3 to the problem described below:"
#   "Construct a PYTHON3 code solution to the problem outlined:"
BEE_INSTRUCTION_REGEX = re.compile(
    r'^.*?python\s*3.*?:\s*\n*',
    re.IGNORECASE | re.DOTALL,
)


def extract_raw_problem(text: str) -> str:
    """
    Extract raw problem description from BEE text field.

    BEE text format:
        ### Prompt

        [Instruction prefix ending with colon]

        [Raw problem description with Input/Output/Examples]

        ### Response

        ```python3
        [solution code]
        ```

    Returns the raw problem text with the BEE instruction prefix stripped.
    """
    # Step 1: Extract content between "### Prompt" and "### Response"
    prompt_match = re.search(
        r'###\s*Prompt\s*\n+(.*?)\n+###\s*Response',
        text,
        re.DOTALL,
    )
    if not prompt_match:
        raise ValueError(f"Cannot find ### Prompt ... ### Response pattern in text: {text[:200]}...")

    prompt_section = prompt_match.group(1).strip()

    # Step 2: Strip BEE instruction prefix (everything up to and including the colon + "python3")
    raw_problem = BEE_INSTRUCTION_REGEX.sub('', prompt_section, count=1).strip()

    if not raw_problem:
        raise ValueError(f"Empty problem after stripping instruction prefix: {prompt_section[:200]}...")

    return raw_problem


def build_messages(raw_problem: str, solution: str) -> list:
    """Build OpenAI-format messages list for a single training example."""
    user_content = CODECONTESTS_USER_TEMPLATE.format(prompt=raw_problem)
    assistant_content = f"<code>\n{solution}\n</code>"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def extract_prompt_and_solution(entry: dict) -> tuple[str, str]:
    """Extract prompt/solution from JSONL entry with fallback to text parsing."""
    solution = str(entry.get("solution") or entry.get("response") or "").strip()
    if not solution:
        raise ValueError("empty_solution")

    prompt = entry.get("prompt")
    if prompt is not None and str(prompt).strip():
        raw_problem = str(prompt).strip()
    else:
        text = entry.get("text")
        if text is None or not str(text).strip():
            raise ValueError("missing_prompt_and_text")
        raw_problem = extract_raw_problem(str(text))

    return raw_problem, solution


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for verl MultiTurnSFTDataset")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to sft_validated.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for parquet files")
    parser.add_argument("--val_size", type=int, default=200, help="Number of validation/regression samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # ---- Load and parse ----
    print(f"Loading data from {input_path}...")
    records = []
    parse_errors = []
    total_lines = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines = line_num
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors.append((f"line_{line_num}", f"json_decode_error: {e}"))
                continue

            problem_id = entry.get("problem_id", f"unknown_{line_num}")

            try:
                raw_problem, solution = extract_prompt_and_solution(entry)
            except ValueError as e:
                parse_errors.append((problem_id, str(e)))
                continue

            messages = build_messages(raw_problem, solution)
            records.append({
                "messages": messages,
                "problem_id": problem_id,
            })

    print(f"  Total lines read: {total_lines}")
    print(f"  Successfully parsed: {len(records)}")
    if parse_errors:
        print(f"  Parse errors: {len(parse_errors)}")
        for pid, err in parse_errors[:5]:
            print(f"    - {pid}: {err}")

    if len(records) < args.val_size + 10:
        print(f"Error: Not enough records ({len(records)}) for val_size={args.val_size}")
        sys.exit(1)

    # ---- Split train/val ----
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(records))

    val_indices = set(indices[:args.val_size])
    train_records = [records[i] for i in range(len(records)) if i not in val_indices]
    val_records = [records[i] for i in val_indices]

    print(f"\n  Train: {len(train_records)} samples")
    print(f"  Val:   {len(val_records)} samples")

    # ---- Token length statistics ----
    all_texts = []
    for rec in records:
        total_len = sum(len(m["content"]) for m in rec["messages"])
        all_texts.append(total_len)
    all_texts = np.array(all_texts)
    print(f"\n  Message total char length: min={all_texts.min()}, max={all_texts.max()}, "
          f"mean={all_texts.mean():.0f}, p95={np.percentile(all_texts, 95):.0f}")

    # ---- Save as parquet ----
    train_path = output_dir / "sft_train.parquet"
    val_path = output_dir / "sft_val.parquet"

    # verl MultiTurnSFTDataset expects a 'messages' column
    train_df = pd.DataFrame({"messages": [r["messages"] for r in train_records]})
    val_df = pd.DataFrame({"messages": [r["messages"] for r in val_records]})

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\n  Saved train: {train_path} ({len(train_df)} rows)")
    print(f"  Saved val:   {val_path} ({len(val_df)} rows)")

    # ---- Save problem_id mapping for traceability ----
    mapping_path = output_dir / "split_mapping.json"
    mapping = {
        "seed": args.seed,
        "val_size": args.val_size,
        "total": len(records),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "val_problem_ids": [records[i]["problem_id"] for i in sorted(val_indices)],
    }
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"  Saved split mapping: {mapping_path}")

    # ---- Sanity check: print first example ----
    print("\n--- First Training Example (preview) ---")
    first = train_records[0]["messages"]
    print(f"  System: {first[0]['content'][:80]}...")
    print(f"  User:   {first[1]['content'][:120]}...")
    print(f"  Assist: {first[2]['content'][:120]}...")

    # ---- Verify parquet roundtrip ----
    print("\n--- Parquet Roundtrip Verification ---")
    loaded = pd.read_parquet(train_path)
    print(f"  Loaded {len(loaded)} rows from {train_path}")
    first_msg = loaded.iloc[0]["messages"]
    # parquet roundtrip may return numpy arrays; verl handles this via
    # convert_nested_value_to_list_recursive in MultiTurnSFTDataset
    if isinstance(first_msg, np.ndarray):
        first_msg = first_msg.tolist()
    assert isinstance(first_msg, list), f"Expected list, got {type(first_msg)}"
    assert len(first_msg) == 3, f"Expected 3 messages, got {len(first_msg)}"
    assert first_msg[0]["role"] == "system"
    assert first_msg[1]["role"] == "user"
    assert first_msg[2]["role"] == "assistant"
    assert "<code>" in first_msg[2]["content"]
    print("  Roundtrip check passed!")

    print("\nDone!")


if __name__ == "__main__":
    main()
