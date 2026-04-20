from __future__ import annotations

import importlib.util
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "coding_model_project" / "phase_2_ GRPO" / "sft_repair_data" / "scripts"
SPEC_PATH = REPO_ROOT / "coding_model_project" / "phase_2_ GRPO" / "sft_repair_data" / "v2a" / "repair_v2a_spec.json"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Phase2SftV2aTests(unittest.TestCase):
    def test_timeout_tail_style_winner_prefers_fast_valid_teacher(self) -> None:
        v2a_spec = _load_module("v2a_spec_module", SCRIPT_DIR / "v2a_spec.py")
        repair_module = _load_module("prepare_repair_module", SCRIPT_DIR / "prepare_repair_sft_parquet.py")
        spec = v2a_spec.load_spec(SPEC_PATH)

        rows = [
            {
                "teacher_group_id": "g1",
                "teacher_style": "concise_standard",
                "seed_problem_id": "Codeforces/1569/C",
                "train_problem_id": "train/timeout_case",
                "bucket": "timeout_tail",
                "prompt_sha256": "sha-timeout",
                "judge_time": 92.0,
                "code_char_count": 160,
                "student_accepted": False,
                "student_pass_ratio_all": 0.4,
                "student_judge_time_s": 50.0,
                "extracted_code": "print(1)",
                "raw_prompt": "timeout prompt",
            },
            {
                "teacher_group_id": "g1",
                "teacher_style": "runtime_optimized",
                "seed_problem_id": "Codeforces/1569/C",
                "train_problem_id": "train/timeout_case",
                "bucket": "timeout_tail",
                "prompt_sha256": "sha-timeout",
                "judge_time": 70.0,
                "code_char_count": 180,
                "student_accepted": False,
                "student_pass_ratio_all": 0.4,
                "student_judge_time_s": 50.0,
                "extracted_code": "print(2)",
                "raw_prompt": "timeout prompt",
            },
        ]

        winners, summary = repair_module._select_style_winners(rows, spec)
        self.assertEqual(summary["accepted_group_count"], 1)
        self.assertEqual(winners[0]["teacher_style"], "runtime_optimized")

    def test_mixed_builder_materializes_two_repeats_per_anchor(self) -> None:
        v2a_spec = _load_module("v2a_spec_module_2", SCRIPT_DIR / "v2a_spec.py")
        mix_module = _load_module("mixed_builder_module", SCRIPT_DIR / "build_phase2_mixed_sft.py")
        spec = v2a_spec.load_spec(SPEC_PATH)

        anchor_rows = []
        for idx in range(6):
            anchor_rows.append(
                {
                    "problem_id": f"stable/{idx}",
                    "prompt_sha256": f"stable-{idx}",
                    "anchor_bucket": "stable_protocol",
                    "anchor_source": "reused_v1",
                    "source_row_id": f"reused_v1:stable/{idx}:stable-{idx}",
                }
            )
        for idx in range(3):
            anchor_rows.append(
                {
                    "problem_id": f"easy-r/{idx}",
                    "prompt_sha256": f"easy-r-{idx}",
                    "anchor_bucket": "easy_medium_cp",
                    "anchor_source": "reused_v1",
                    "source_row_id": f"reused_v1:easy-r/{idx}:easy-r-{idx}",
                }
            )
        for idx in range(3):
            anchor_rows.append(
                {
                    "problem_id": f"general-r/{idx}",
                    "prompt_sha256": f"general-r-{idx}",
                    "anchor_bucket": "general_coding",
                    "anchor_source": "reused_v1",
                    "source_row_id": f"reused_v1:general-r/{idx}:general-r-{idx}",
                }
            )
        for idx in range(5):
            anchor_rows.append(
                {
                    "problem_id": f"easy-n/{idx}",
                    "prompt_sha256": f"easy-n-{idx}",
                    "anchor_bucket": "easy_medium_cp",
                    "anchor_source": "new_v2a",
                    "source_row_id": f"new_v2a:easy-n/{idx}:easy-n-{idx}",
                }
            )
        for idx in range(7):
            anchor_rows.append(
                {
                    "problem_id": f"general-n/{idx}",
                    "prompt_sha256": f"general-n-{idx}",
                    "anchor_bucket": "general_coding",
                    "anchor_source": "new_v2a",
                    "source_row_id": f"new_v2a:general-n/{idx}:general-n-{idx}",
                }
            )

        mix_module._validate_v2a_anchor_contract(anchor_rows, spec)
        materialized = mix_module._materialize_anchor_oversample(anchor_rows, 48)

        self.assertEqual(len(materialized), 48)
        self.assertEqual(Counter(row["anchor_source"] for row in anchor_rows), {"reused_v1": 12, "new_v2a": 12})
        self.assertEqual(
            Counter(row["anchor_bucket"] for row in anchor_rows),
            {"stable_protocol": 6, "easy_medium_cp": 8, "general_coding": 10},
        )
        repeated_ids = Counter(row["source_row_id"] for row in materialized)
        self.assertEqual(set(repeated_ids.values()), {2})
        self.assertEqual(set(row["oversample_repeat_idx"] for row in materialized), {0, 1})

    def test_student_reference_profile_validation_checks_expected_fields(self) -> None:
        v2a_spec = _load_module("v2a_spec_module_3", SCRIPT_DIR / "v2a_spec.py")
        refs_module = _load_module("student_refs_module", SCRIPT_DIR / "build_student_references.py")
        spec = v2a_spec.load_spec(SPEC_PATH)
        profile = v2a_spec.student_reference_profile(spec)

        run_info = {
            "config": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 2048,
                "sandbox_url": "http://localhost:8090",
                "run_timeout": 30,
                "max_concurrent_judges": 24,
                "verifier_limiter_budget": 24,
                "use_external_tests": True,
                "use_submit_api": False,
                "datasets": ["codecontests_train"],
            },
            "timestamp": "2026-04-04T00:00:00Z",
        }
        summary = refs_module._validate_eval_run_info(run_info, profile)
        self.assertEqual(summary["dataset_key"], "codecontests_train")
        self.assertIn("temperature", summary["checked_fields"])

    def test_student_reference_shortlist_validation_requires_exact_match(self) -> None:
        refs_module = _load_module("student_refs_module_2", SCRIPT_DIR / "build_student_references.py")

        normalized_rows = [
            {
                "prompt_sha256": "sha-1",
                "train_problem_id": "Codeforces/1000/A",
            },
            {
                "prompt_sha256": "sha-2",
                "train_problem_id": "Codeforces/1001/B",
            },
        ]
        expected_pairs = {
            ("sha-1", "Codeforces/1000/A"),
            ("sha-2", "Codeforces/1001/B"),
        }
        summary = refs_module._validate_shortlist_coverage(normalized_rows, expected_pairs)
        self.assertEqual(summary["status"], "validated")
        self.assertEqual(summary["expected_count"], 2)

        with self.assertRaises(SystemExit):
            refs_module._validate_shortlist_coverage(
                normalized_rows[:-1],
                expected_pairs,
            )

    def test_materialize_student_reference_eval_subset_preserves_shortlist_order(self) -> None:
        subset_module = _load_module(
            "student_ref_subset_module",
            SCRIPT_DIR / "materialize_student_reference_eval_subset.py",
        )

        shortlist = [
            {
                "train_problem_id": "Codeforces/1001/A",
                "prompt_sha256": subset_module._prompt_sha256("prompt-a"),
                "raw_prompt": "prompt-a",
            },
            {
                "train_problem_id": "Codeforces/1002/B",
                "prompt_sha256": subset_module._prompt_sha256("prompt-b"),
                "raw_prompt": "prompt-b",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "train_raw.jsonl"
            rows = [
                {"problem_id": "Codeforces/1002/B", "prompt": "prompt-b-raw", "test_cases": {"inputs": [], "outputs": []}},
                {"problem_id": "Codeforces/999/X", "prompt": "prompt-x", "test_cases": {"inputs": [], "outputs": []}},
                {"problem_id": "Codeforces/1001/A", "prompt": "prompt-a", "test_cases": {"inputs": [], "outputs": []}},
            ]
            with raw_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            selected, prompt_overrides = subset_module._load_selected_raw_rows(raw_path=raw_path, shortlist=shortlist)
            self.assertEqual(
                [row["problem_id"] for row in selected],
                ["Codeforces/1001/A", "Codeforces/1002/B"],
            )
            self.assertEqual(prompt_overrides, ["Codeforces/1002/B"])
            self.assertEqual(selected[1]["prompt"], "prompt-b")
            self.assertEqual(selected[1]["prompt_source"], "student_reference_shortlist_v2a")


if __name__ == "__main__":
    unittest.main()
