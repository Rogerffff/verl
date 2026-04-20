#!/usr/bin/env python3
"""Build verl RL parquet files from manifest-filtered raw coding datasets."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from omegaconf import OmegaConf

try:
    from coding_model_project.src.prompting import SYSTEM_PROMPT, format_prompt
    from coding_model_project.src.problem_quarantine import (
        hard_blacklist_ids,
        load_problem_quarantine,
        quarantine_summary,
    )
except ImportError:
    from prompting import SYSTEM_PROMPT, format_prompt
    from problem_quarantine import hard_blacklist_ids, load_problem_quarantine, quarantine_summary


DEFAULT_DATASETS = {
    "codecontests_train_wo_valid_big": {
        "manifest": "manifests/codecontests_train_wo_valid_big_manifest.jsonl",
        "raw": "raw/codecontests_train_wo_valid_big_raw.jsonl",
    },
    "codecontests_valid": {
        "manifest": "manifests/codecontests_valid_manifest.jsonl",
        "raw": "raw/codecontests_valid_raw.jsonl",
    },
    "codecontests_valid_big": {
        "manifest": "manifests/codecontests_valid_big_manifest.jsonl",
        "raw": "raw/codecontests_valid_big_raw.jsonl",
    },
    "codecontests_test": {
        "manifest": "manifests/codecontests_test_manifest.jsonl",
        "raw": "raw/codecontests_test_raw.jsonl",
    },
    "mbpp_reg": {
        "manifest": "manifests/mbpp_reg_manifest.jsonl",
        "raw": "raw/mbpp_reg_raw.jsonl",
    },
    "humaneval": {
        "manifest": "manifests/humaneval_manifest.jsonl",
        "raw": "raw/humaneval_raw.jsonl",
    },
}

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_QUARANTINE_PATH = Path("coding_model_project/data/problem_quarantine_v2.json")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _filter_raw_by_manifest_ids(
    manifest_path: Path,
    raw_path: Path,
    *,
    excluded_problem_ids: set[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest_ids_all = {str(record["problem_id"]) for record in _read_jsonl(manifest_path)}
    excluded_problem_ids = excluded_problem_ids or set()
    excluded_in_manifest = manifest_ids_all & excluded_problem_ids
    manifest_ids = manifest_ids_all - excluded_in_manifest
    filtered: List[Dict[str, Any]] = []
    found_ids = set()
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["problem_id"] in manifest_ids:
                filtered.append(record)
                found_ids.add(record["problem_id"])

    missing_ids = sorted(manifest_ids - found_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            f"{manifest_path.name} has {len(missing_ids)} problem_id(s) missing from {raw_path.name}: {preview}"
        )
    return filtered, {
        "manifest_path": str(manifest_path),
        "raw_path": str(raw_path),
        "manifest_total": len(manifest_ids_all),
        "excluded_by_quarantine": len(excluded_in_manifest),
        "excluded_problem_ids": sorted(excluded_in_manifest),
        "manifest_after_quarantine": len(manifest_ids),
        "raw_filtered_count": len(filtered),
    }


def _build_prompt_messages(dataset_key: str, record: Dict[str, Any]) -> List[Dict[str, str]]:
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


def _convert_record(dataset_key: str, split: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
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


def _convert_dataset(dataset_key: str, split: str, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_convert_record(dataset_key, split, record) for record in records]


def _sample_records(records: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if sample_size <= 0 or len(records) <= sample_size:
        return list(records)
    sampled = list(records)
    random.Random(seed).shuffle(sampled)
    return sampled[:sample_size]


def _ordered_records(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    return ordered


def _write_parquet(records: List[Dict[str, Any]], output_path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to build GRPO parquet files. Please install pandas first.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(output_path, index=False)


def _load_runtime_tokenizer_and_processor(model_path: str, trust_remote_code: bool):
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(model_path, use_shm=False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    return tokenizer, processor


def _build_runtime_filter_config(
    *,
    cache_dir: Path,
    max_prompt_length: int,
    seed: int,
    filter_overlong_prompts_workers: int,
    tool_config_path: Optional[str],
    apply_chat_template_kwargs: Dict[str, Any],
):
    return OmegaConf.create(
        {
            "cache_dir": str(cache_dir),
            "prompt_key": "prompt",
            "image_key": "images",
            "video_key": "videos",
            "image_patch_size": 14,
            "max_prompt_length": max_prompt_length,
            "return_raw_chat": False,
            "return_full_prompt": False,
            "truncation": "error",
            "filter_overlong_prompts": True,
            "tool_config_path": tool_config_path,
            "filter_overlong_prompts_workers": filter_overlong_prompts_workers,
            "use_shm": False,
            "chat_template_func": None,
            "need_tools_kwargs": False,
            "filter_prompts": True,
            "return_multi_modal_inputs": True,
            "shuffle": False,
            "seed": seed,
            "apply_chat_template_kwargs": apply_chat_template_kwargs,
        }
    )


def _records_from_runtime_filtered_dataset(
    records: List[Dict[str, Any]],
    *,
    tokenizer,
    processor,
    data_config,
) -> List[Dict[str, Any]]:
    from verl.utils.dataset.rl_dataset import RLHFDataset

    with tempfile.TemporaryDirectory(prefix="grpo_fast_val_runtime_filter_") as tmp_dir:
        candidate_path = Path(tmp_dir) / "candidate.parquet"
        _write_parquet(records, candidate_path)
        dataset = RLHFDataset(
            data_files=str(candidate_path),
            tokenizer=tokenizer,
            processor=processor,
            config=data_config,
            max_samples=-1,
        )
        return [dataset.dataframe[i] for i in range(len(dataset.dataframe))]


def _select_fast_val_records(
    ordered_records: List[Dict[str, Any]],
    *,
    target_size: int,
    initial_pool_size: int,
    filter_records_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if target_size <= 0:
        raise ValueError("target_size must be positive for fast_val_codecontests")
    if not ordered_records:
        raise ValueError("ordered_records is empty; cannot build fast_val_codecontests")

    pool_size = min(len(ordered_records), max(initial_pool_size, target_size))
    attempts: List[Dict[str, int]] = []

    while True:
        filtered_records = filter_records_fn(ordered_records[:pool_size])
        attempts.append({"candidate_pool_size": pool_size, "filtered_count": len(filtered_records)})
        if len(filtered_records) >= target_size:
            return filtered_records[:target_size], {
                "candidate_pool_size": pool_size,
                "filtered_count": len(filtered_records),
                "attempts": attempts,
            }

        if pool_size >= len(ordered_records):
            break

        next_pool_size = min(len(ordered_records), max(pool_size * 2, target_size))
        if next_pool_size == pool_size:
            break
        pool_size = next_pool_size

    raise ValueError(
        "Unable to build fast_val_codecontests.parquet with enough post-filter samples. "
        f"target={target_size}, final_filtered={attempts[-1]['filtered_count']}, total_source={len(ordered_records)}"
    )


def build_default_splits(
    *,
    data_root: Path,
    output_dir: Path,
    smoke_train_size: int,
    smoke_val_size: int,
    step_smoke_train_size: int,
    step_smoke_val_size: int,
    seed: int,
    model_path: str,
    trust_remote_code: bool,
    fast_val_codecontests_size: int,
    fast_val_initial_pool_size: int,
    fast_val_seed: int,
    fast_val_max_prompt_length: int,
    fast_val_filter_overlong_prompts_workers: int,
    fast_val_tool_config_path: Optional[str],
    fast_val_apply_chat_template_kwargs: Dict[str, Any],
    quarantine_path: Optional[Path],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    quarantine = load_problem_quarantine(quarantine_path)
    excluded_problem_ids = hard_blacklist_ids(quarantine)
    filter_reports: Dict[str, Dict[str, Any]] = {}

    train_raw_rows, filter_reports["codecontests_train_wo_valid_big"] = _filter_raw_by_manifest_ids(
        data_root / DEFAULT_DATASETS["codecontests_train_wo_valid_big"]["manifest"],
        data_root / DEFAULT_DATASETS["codecontests_train_wo_valid_big"]["raw"],
        excluded_problem_ids=excluded_problem_ids,
    )
    train_records = _convert_dataset(
        "codecontests_train_wo_valid_big",
        "train",
        train_raw_rows,
    )

    codecontests_valid_raw_rows, filter_reports["codecontests_valid"] = _filter_raw_by_manifest_ids(
        data_root / DEFAULT_DATASETS["codecontests_valid"]["manifest"],
        data_root / DEFAULT_DATASETS["codecontests_valid"]["raw"],
        excluded_problem_ids=excluded_problem_ids,
    )
    codecontests_valid_records = _convert_dataset(
        "codecontests_valid",
        "val_tier1",
        codecontests_valid_raw_rows,
    )
    mbpp_reg_raw_rows, filter_reports["mbpp_reg"] = _filter_raw_by_manifest_ids(
        data_root / DEFAULT_DATASETS["mbpp_reg"]["manifest"],
        data_root / DEFAULT_DATASETS["mbpp_reg"]["raw"],
        excluded_problem_ids=excluded_problem_ids,
    )
    mbpp_reg_records = _convert_dataset(
        "mbpp_reg",
        "val_tier1",
        mbpp_reg_raw_rows,
    )
    val_tier1_records = [*codecontests_valid_records, *mbpp_reg_records]

    val_tier2_raw_rows, filter_reports["codecontests_valid_big"] = _filter_raw_by_manifest_ids(
        data_root / DEFAULT_DATASETS["codecontests_valid_big"]["manifest"],
        data_root / DEFAULT_DATASETS["codecontests_valid_big"]["raw"],
        excluded_problem_ids=excluded_problem_ids,
    )
    val_tier2_records = _convert_dataset(
        "codecontests_valid_big",
        "val_tier2",
        val_tier2_raw_rows,
    )

    final_eval_records: List[Dict[str, Any]] = []
    for dataset_key in ("codecontests_test", "humaneval"):
        raw_rows, filter_reports[dataset_key] = _filter_raw_by_manifest_ids(
            data_root / DEFAULT_DATASETS[dataset_key]["manifest"],
            data_root / DEFAULT_DATASETS[dataset_key]["raw"],
            excluded_problem_ids=excluded_problem_ids,
        )
        final_eval_records.extend(
            _convert_dataset(
                dataset_key,
                "final_eval",
                raw_rows,
            )
        )

    smoke_train_records = _sample_records(train_records, smoke_train_size, seed)
    smoke_val_source = val_tier1_records if val_tier1_records else val_tier2_records
    smoke_val_records = _sample_records(smoke_val_source, smoke_val_size, seed)
    step_smoke_train_records = _sample_records(train_records, step_smoke_train_size, seed)
    step_smoke_val_records = _sample_records(smoke_val_source, step_smoke_val_size, seed)

    fast_val_codecontests_records: List[Dict[str, Any]] = []
    fast_val_meta: Dict[str, Any] = {}
    if fast_val_codecontests_size > 0:
        tokenizer, processor = _load_runtime_tokenizer_and_processor(model_path, trust_remote_code)
        fast_val_config = _build_runtime_filter_config(
            cache_dir=output_dir / ".fast_val_cache",
            max_prompt_length=fast_val_max_prompt_length,
            seed=fast_val_seed,
            filter_overlong_prompts_workers=fast_val_filter_overlong_prompts_workers,
            tool_config_path=fast_val_tool_config_path,
            apply_chat_template_kwargs=fast_val_apply_chat_template_kwargs,
        )
        ordered_codecontests_valid = _ordered_records(
            [
                {
                    **record,
                    "extra_info": {
                        **record.get("extra_info", {}),
                        "split": "fast_val_codecontests",
                    },
                }
                for record in codecontests_valid_records
            ],
            fast_val_seed,
        )
        fast_val_codecontests_records, fast_val_meta = _select_fast_val_records(
            ordered_codecontests_valid,
            target_size=fast_val_codecontests_size,
            initial_pool_size=fast_val_initial_pool_size,
            filter_records_fn=lambda candidate_records: _records_from_runtime_filtered_dataset(
                candidate_records,
                tokenizer=tokenizer,
                processor=processor,
                data_config=fast_val_config,
            ),
        )

    files = {
        "train": output_dir / "train.parquet",
        "val_tier1": output_dir / "val_tier1.parquet",
        "val_tier2": output_dir / "val_tier2.parquet",
        "final_eval": output_dir / "final_eval.parquet",
        "smoke_train": output_dir / "smoke_train.parquet",
        "smoke_val": output_dir / "smoke_val.parquet",
        "step_smoke_train": output_dir / "step_smoke_train.parquet",
        "step_smoke_val": output_dir / "step_smoke_val.parquet",
    }
    if fast_val_codecontests_size > 0:
        files["fast_val_codecontests"] = output_dir / "fast_val_codecontests.parquet"

    _write_parquet(train_records, files["train"])
    _write_parquet(val_tier1_records, files["val_tier1"])
    _write_parquet(val_tier2_records, files["val_tier2"])
    _write_parquet(final_eval_records, files["final_eval"])
    _write_parquet(smoke_train_records, files["smoke_train"])
    _write_parquet(smoke_val_records, files["smoke_val"])
    _write_parquet(step_smoke_train_records, files["step_smoke_train"])
    _write_parquet(step_smoke_val_records, files["step_smoke_val"])

    if fast_val_codecontests_size > 0:
        _write_parquet(fast_val_codecontests_records, files["fast_val_codecontests"])
        fast_val_parity_records = _records_from_runtime_filtered_dataset(
            fast_val_codecontests_records,
            tokenizer=tokenizer,
            processor=processor,
            data_config=fast_val_config,
        )
        if len(fast_val_parity_records) != fast_val_codecontests_size:
            raise ValueError(
                "fast_val_codecontests parity check failed after write. "
                f"expected={fast_val_codecontests_size}, actual={len(fast_val_parity_records)}"
            )
        fast_val_meta = {
            **fast_val_meta,
            "target_size": fast_val_codecontests_size,
            "final_size": len(fast_val_codecontests_records),
            "parity_filtered_count": len(fast_val_parity_records),
            "model_path": model_path,
            "max_prompt_length": fast_val_max_prompt_length,
            "seed": fast_val_seed,
        }

    summary = {
        "train": len(train_records),
        "val_tier1": len(val_tier1_records),
        "val_tier2": len(val_tier2_records),
        "final_eval": len(final_eval_records),
        "smoke_train": len(smoke_train_records),
        "smoke_val": len(smoke_val_records),
        "step_smoke_train": len(step_smoke_train_records),
        "step_smoke_val": len(step_smoke_val_records),
        "files": {name: str(path) for name, path in files.items()},
        "problem_quarantine": quarantine_summary(quarantine),
        "filter_reports": filter_reports,
    }
    if fast_val_codecontests_size > 0:
        summary["fast_val_codecontests"] = len(fast_val_codecontests_records)
        summary["fast_val_codecontests_meta"] = fast_val_meta

    summary_path = output_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build verl RL parquet files for GRPO coding experiments.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("coding_model_project/data"),
        help="Root directory containing manifests/ and raw/ subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("coding_model_project/data/grpo_parquet"),
        help="Directory to write parquet files into.",
    )
    parser.add_argument("--smoke_train_size", type=int, default=64, help="Number of samples in smoke_train.parquet.")
    parser.add_argument("--smoke_val_size", type=int, default=32, help="Number of samples in smoke_val.parquet.")
    parser.add_argument(
        "--step_smoke_train_size",
        type=int,
        default=16,
        help="Number of samples in step_smoke_train.parquet.",
    )
    parser.add_argument(
        "--step_smoke_val_size",
        type=int,
        default=8,
        help="Number of samples in step_smoke_val.parquet.",
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Training MODEL_PATH tokenizer source.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the runtime-parity tokenizer/processor.",
    )
    parser.add_argument(
        "--fast_val_codecontests_size",
        type=int,
        default=16,
        help="Target size for fast_val_codecontests.parquet after runtime-equivalent prompt filtering.",
    )
    parser.add_argument(
        "--fast_val_initial_pool_size",
        type=int,
        default=32,
        help="Initial candidate pool size before runtime-equivalent filtering for fast_val_codecontests.parquet.",
    )
    parser.add_argument(
        "--fast_val_seed",
        type=int,
        default=0,
        help="Seed used for dedicated CodeContests fast-val sampling.",
    )
    parser.add_argument(
        "--fast_val_max_prompt_length",
        type=int,
        default=1024,
        help="max_prompt_length used for runtime-parity filtering of fast_val_codecontests.parquet.",
    )
    parser.add_argument(
        "--fast_val_filter_overlong_prompts_workers",
        type=int,
        default=1,
        help="Number of workers used by runtime-parity prompt filtering.",
    )
    parser.add_argument(
        "--fast_val_tool_config_path",
        type=str,
        default=None,
        help="Optional tool config path forwarded into runtime-parity prompt filtering.",
    )
    parser.add_argument(
        "--fast_val_apply_chat_template_kwargs",
        type=str,
        default="{}",
        help="JSON dict forwarded as apply_chat_template_kwargs for runtime-parity prompt filtering.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for smoke dataset sampling.")
    parser.add_argument(
        "--quarantine_path",
        type=Path,
        default=DEFAULT_QUARANTINE_PATH,
        help="Optional shared problem quarantine file. hard_blacklist items are excluded during parquet construction.",
    )
    args = parser.parse_args()

    fast_val_apply_chat_template_kwargs = json.loads(args.fast_val_apply_chat_template_kwargs)
    if not isinstance(fast_val_apply_chat_template_kwargs, dict):
        raise ValueError("--fast_val_apply_chat_template_kwargs must decode to a JSON object")

    summary = build_default_splits(
        data_root=args.data_root,
        output_dir=args.output_dir,
        smoke_train_size=args.smoke_train_size,
        smoke_val_size=args.smoke_val_size,
        step_smoke_train_size=args.step_smoke_train_size,
        step_smoke_val_size=args.step_smoke_val_size,
        seed=args.seed,
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        fast_val_codecontests_size=args.fast_val_codecontests_size,
        fast_val_initial_pool_size=args.fast_val_initial_pool_size,
        fast_val_seed=args.fast_val_seed,
        fast_val_max_prompt_length=args.fast_val_max_prompt_length,
        fast_val_filter_overlong_prompts_workers=args.fast_val_filter_overlong_prompts_workers,
        fast_val_tool_config_path=args.fast_val_tool_config_path,
        fast_val_apply_chat_template_kwargs=fast_val_apply_chat_template_kwargs,
        quarantine_path=args.quarantine_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
