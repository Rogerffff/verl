from pathlib import Path

import pytest

from coding_model_project.src import build_grpo_parquet


def _fake_raw_record(problem_id: str, *, entry_point: str = "", example_call: str = "") -> dict:
    return {
        "problem_id": problem_id,
        "prompt": f"solve {problem_id}",
        "test_cases": {"entry_point": entry_point, "example_call": example_call},
    }


def test_select_fast_val_records_expands_until_target():
    ordered_records = [{"problem_id": f"p{i}"} for i in range(12)]

    def filter_records(records):
        if len(records) <= 8:
            return list(records[:5])
        return list(records[:9])

    selected, meta = build_grpo_parquet._select_fast_val_records(
        ordered_records,
        target_size=8,
        initial_pool_size=8,
        filter_records_fn=filter_records,
    )

    assert len(selected) == 8
    assert meta["candidate_pool_size"] == 12
    assert meta["filtered_count"] == 9
    assert meta["attempts"] == [
        {"candidate_pool_size": 8, "filtered_count": 5},
        {"candidate_pool_size": 12, "filtered_count": 9},
    ]


def test_select_fast_val_records_raises_when_source_cannot_survive_filter():
    ordered_records = [{"problem_id": f"p{i}"} for i in range(6)]

    with pytest.raises(ValueError, match="Unable to build fast_val_codecontests.parquet"):
        build_grpo_parquet._select_fast_val_records(
            ordered_records,
            target_size=5,
            initial_pool_size=4,
            filter_records_fn=lambda records: list(records[:2]),
        )


def test_build_default_splits_writes_fast_val_summary(monkeypatch, tmp_path: Path):
    source_records = {
        "codecontests_train_wo_valid_big_manifest.jsonl": [_fake_raw_record(f"train-{i}") for i in range(20)],
        "codecontests_valid_manifest.jsonl": [_fake_raw_record(f"valid-{i}") for i in range(40)],
        "mbpp_reg_manifest.jsonl": [
            _fake_raw_record(f"mbpp-{i}", entry_point="solve", example_call="solve(1)") for i in range(5)
        ],
        "codecontests_valid_big_manifest.jsonl": [_fake_raw_record(f"valid-big-{i}") for i in range(7)],
        "codecontests_test_manifest.jsonl": [_fake_raw_record(f"test-{i}") for i in range(3)],
        "humaneval_manifest.jsonl": [_fake_raw_record(f"human-{i}") for i in range(2)],
    }

    monkeypatch.setattr(
        build_grpo_parquet,
        "_filter_raw_by_manifest_ids",
        lambda manifest_path, raw_path: source_records[manifest_path.name],
    )
    monkeypatch.setattr(build_grpo_parquet, "_load_runtime_tokenizer_and_processor", lambda *args, **kwargs: ("tok", None))

    call_count = {"value": 0}

    def fake_runtime_filter(records, *, tokenizer, processor, data_config):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return list(records[:12])
        if call_count["value"] == 2:
            return list(records[:-4])
        return list(records)

    monkeypatch.setattr(build_grpo_parquet, "_records_from_runtime_filtered_dataset", fake_runtime_filter)

    def fake_write_parquet(records, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(len(records)), encoding="utf-8")

    monkeypatch.setattr(build_grpo_parquet, "_write_parquet", fake_write_parquet)

    summary = build_grpo_parquet.build_default_splits(
        data_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        smoke_train_size=4,
        smoke_val_size=3,
        step_smoke_train_size=2,
        step_smoke_val_size=2,
        seed=42,
        model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        trust_remote_code=False,
        fast_val_codecontests_size=16,
        fast_val_initial_pool_size=16,
        fast_val_seed=0,
        fast_val_max_prompt_length=1024,
        fast_val_filter_overlong_prompts_workers=1,
        fast_val_tool_config_path=None,
        fast_val_apply_chat_template_kwargs={},
    )

    assert summary["fast_val_codecontests"] == 16
    assert summary["fast_val_codecontests_meta"]["candidate_pool_size"] == 32
    assert summary["fast_val_codecontests_meta"]["parity_filtered_count"] == 16
    assert "fast_val_codecontests" in summary["files"]
    assert (tmp_path / "out" / "fast_val_codecontests.parquet").exists()
    assert (tmp_path / "out" / "build_summary.json").exists()
