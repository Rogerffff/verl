from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import datasets
from omegaconf import DictConfig

from verl.utils.dataset.rl_dataset import RLHFDataset


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    extra_info = row.get("extra_info") or {}
    dataset = str(extra_info.get("dataset") or row.get("data_source") or "")
    problem_id = str(extra_info.get("problem_id") or row.get("problem_id") or "")
    return dataset, problem_id


def _duplicate_count(counter: Counter[tuple[str, str]]) -> int:
    return sum(count - 1 for count in counter.values() if count > 1)


class Step580CurriculumDataset(RLHFDataset):
    """RL dataset that injects curriculum metadata after runtime filtering."""

    def __init__(self, *args, **kwargs):
        config: DictConfig = kwargs["config"]
        data_files = kwargs.get("data_files")
        self.curriculum_data_files = self._normalize_data_files(data_files)
        self.is_train_split = self._infer_is_train_split(self.curriculum_data_files)
        curriculum_cfg = config.get("curriculum", {})
        manifest_path = curriculum_cfg.get("manifest_path")
        if self.is_train_split and not manifest_path:
            raise ValueError("data.curriculum.manifest_path must be set for Step580CurriculumDataset")

        self.curriculum_manifest_path = Path(str(manifest_path)) if manifest_path else None
        self.curriculum_state_dir = Path(str(curriculum_cfg.get("state_dir") or Path.cwd() / "curriculum_state"))
        self.curriculum_state_dir.mkdir(parents=True, exist_ok=True)
        coverage_name = "runtime_curriculum_coverage.json" if self.is_train_split else "runtime_curriculum_coverage_val.json"
        self.curriculum_coverage_report_path = self.curriculum_state_dir / coverage_name
        self.curriculum_thresholds = {
            "A_retention": int(curriculum_cfg.get("min_runtime_a", 154)),
            "B_near_miss": int(curriculum_cfg.get("min_runtime_b", 308)),
            "C_hard_partial": int(curriculum_cfg.get("min_runtime_c", 154)),
        }

        self.curriculum_manifest_rows = (
            self._load_manifest_rows(self.curriculum_manifest_path) if self.curriculum_manifest_path else []
        )
        self.curriculum_metadata_by_key = {
            (str(row["dataset"]), str(row["problem_id"])): dict(row) for row in self.curriculum_manifest_rows
        }

        self._pre_filter_key_counts: Counter[tuple[str, str]] = Counter()
        self._post_filter_key_counts: Counter[tuple[str, str]] = Counter()
        self.curriculum_metadata_by_index: dict[int, dict[str, Any]] = {}
        self.curriculum_initial_bucket_by_index: list[str] = []
        self.curriculum_key_by_index: list[tuple[str, str]] = []

        super().__init__(*args, **kwargs)
        self._attach_curriculum_metadata()

    @staticmethod
    def _normalize_data_files(data_files: Any) -> list[str]:
        if data_files is None:
            return []
        if isinstance(data_files, str):
            return [data_files]
        if isinstance(data_files, (list, tuple)):
            return [str(path) for path in data_files]
        return [str(data_files)]

    @staticmethod
    def _infer_is_train_split(data_files: list[str]) -> bool:
        names = [Path(path).name.lower() for path in data_files]
        if any("train" in name for name in names):
            return True
        if any(("val" in name) or ("valid" in name) or ("test" in name) for name in names):
            return False
        return True

    @staticmethod
    def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (str(row["dataset"]), str(row["problem_id"]))
                if key in seen_keys:
                    raise ValueError(f"Duplicate curriculum manifest row for {key}")
                seen_keys.add(key)
                rows.append(row)
        if not rows:
            raise ValueError(f"Curriculum manifest is empty: {path}")
        return rows

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        if dataframe is None:
            dataframe = self.dataframe
        self._pre_filter_key_counts = self._count_keys(dataframe)
        filtered = super().maybe_filter_out_long_prompts(dataframe)
        self._post_filter_key_counts = self._count_keys(filtered)
        return filtered

    def _count_keys(self, dataframe: datasets.Dataset) -> Counter[tuple[str, str]]:
        counter: Counter[tuple[str, str]] = Counter()
        for idx in range(len(dataframe)):
            counter[_row_key(dataframe[idx])] += 1
        return counter

    def _attach_curriculum_metadata(self) -> None:
        runtime_bucket_counts: Counter[str] = Counter()
        missing_due_to_runtime_filter: Counter[str] = Counter()
        missing_not_found_post_filter: Counter[str] = Counter()
        missing_preview: dict[str, list[str]] = {
            "A_retention": [],
            "B_near_miss": [],
            "C_hard_partial": [],
        }

        post_filter_counts = self._count_keys(self.dataframe)
        duplicate_key_count_post_filter = _duplicate_count(post_filter_counts)

        for idx in range(len(self.dataframe)):
            row = self.dataframe[idx]
            key = _row_key(row)
            metadata = dict(self.curriculum_metadata_by_key.get(key, {}))
            if metadata:
                bucket = str(metadata["initial_bucket"])
            else:
                bucket = "U_unseen"
                metadata = {
                    "dataset": key[0],
                    "problem_id": key[1],
                    "initial_bucket": bucket,
                    "primary_seed_family": "",
                    "matched_seed_ids": [],
                    "matched_seed_families": [],
                    "retrieval_score_max": 0.0,
                    "prompt_sha256": "",
                }

            self.curriculum_metadata_by_index[idx] = metadata
            self.curriculum_initial_bucket_by_index.append(bucket)
            self.curriculum_key_by_index.append(key)
            runtime_bucket_counts[bucket] += 1

        for key, metadata in self.curriculum_metadata_by_key.items():
            if key in post_filter_counts:
                continue
            bucket = str(metadata["initial_bucket"])
            if self._pre_filter_key_counts.get(key, 0) > 0:
                missing_due_to_runtime_filter[bucket] += 1
            else:
                missing_not_found_post_filter[bucket] += 1
            if len(missing_preview[bucket]) < 10:
                missing_preview[bucket].append(f"{key[0]}::{key[1]}")

        target_bucket_counts = {
            "A_retention": sum(1 for row in self.curriculum_manifest_rows if row["initial_bucket"] == "A_retention"),
            "B_near_miss": sum(1 for row in self.curriculum_manifest_rows if row["initial_bucket"] == "B_near_miss"),
            "C_hard_partial": sum(1 for row in self.curriculum_manifest_rows if row["initial_bucket"] == "C_hard_partial"),
        }
        retention_ratio_by_bucket = {
            bucket: (
                float(runtime_bucket_counts.get(bucket, 0)) / float(target_bucket_counts[bucket])
                if target_bucket_counts[bucket] > 0
                else 0.0
            )
            for bucket in target_bucket_counts
        }

        coverage_report = {
            "split_role": "train" if self.is_train_split else "validation",
            "data_files": self.curriculum_data_files,
            "filtered_dataset_size": len(self.dataframe),
            "manifest_row_count": len(self.curriculum_manifest_rows),
            "runtime_bucket_counts": {
                "A_retention": runtime_bucket_counts.get("A_retention", 0),
                "B_near_miss": runtime_bucket_counts.get("B_near_miss", 0),
                "C_hard_partial": runtime_bucket_counts.get("C_hard_partial", 0),
                "U_unseen": runtime_bucket_counts.get("U_unseen", 0),
            },
            "target_bucket_counts": target_bucket_counts,
            "retention_ratio_by_bucket": retention_ratio_by_bucket,
            "missing_due_to_runtime_filter_by_bucket": dict(missing_due_to_runtime_filter),
            "missing_not_found_post_filter_by_bucket": dict(missing_not_found_post_filter),
            "missing_preview_by_bucket": missing_preview,
            "duplicate_key_count_post_filter": duplicate_key_count_post_filter,
        }
        self.curriculum_coverage_report_path.write_text(
            json.dumps(coverage_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if not self.is_train_split:
            return

        if duplicate_key_count_post_filter > 0:
            raise RuntimeError(
                "Filtered dataset contains duplicate (dataset, problem_id) keys; "
                f"see {self.curriculum_coverage_report_path}"
            )

        threshold_failures = {
            bucket: {
                "actual": runtime_bucket_counts.get(bucket, 0),
                "minimum": threshold,
            }
            for bucket, threshold in self.curriculum_thresholds.items()
            if runtime_bucket_counts.get(bucket, 0) < threshold
        }
        if threshold_failures:
            raise RuntimeError(
                "Runtime curriculum coverage failed thresholds "
                f"{threshold_failures}; see {self.curriculum_coverage_report_path}"
            )

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        metadata = self.curriculum_metadata_by_index[item]
        extra_info = dict(row_dict.get("extra_info") or {})
        extra_info["index"] = item
        extra_info["dataset"] = metadata["dataset"]
        extra_info["problem_id"] = metadata["problem_id"]
        extra_info["initial_bucket"] = metadata["initial_bucket"]
        extra_info["primary_seed_family"] = metadata["primary_seed_family"]

        row_dict["extra_info"] = extra_info
        row_dict["index"] = item
        row_dict["dataset"] = metadata["dataset"]
        row_dict["problem_id"] = metadata["problem_id"]
        row_dict["initial_bucket"] = metadata["initial_bucket"]
        row_dict["primary_seed_family"] = metadata["primary_seed_family"]
        return row_dict
