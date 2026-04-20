import math

import pytest

from coding_model_project.src import repair_grpo_batch_reward


class _DummySummary:
    def __init__(self, **payload):
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


def _base_summary(**overrides):
    payload = {
        "accepted": False,
        "passed_tests": 2,
        "total_tests": 4,
        "pass_ratio_all": 0.5,
        "error_type": "wrong_answer",
        "invalid_for_rl": False,
        "invalid_reason": "",
        "judge_time_s": 0.4,
        "extraction_status": "ok",
    }
    payload.update(overrides)
    return _DummySummary(**payload)


def _base_ground_truth(**first_pass_overrides):
    first_pass = {
        "code": "print('old')",
        "pass_ratio_all": 0.5,
        "accepted": False,
        "error_type": "wrong_answer",
        "invalid_for_rl": False,
        "invalid_reason": "",
        "finish_reason": "stop",
        "pass_ratio_bucket": "bucket_0.2_0.6",
    }
    first_pass.update(first_pass_overrides)
    return {
        "problem_id": "p1",
        "dataset": "codecontests_train",
        "test_cases": {"type": "stdin_stdout", "tests": [{"input": "1\n", "output": "1\n"}]},
        "first_pass": first_pass,
        "repair_metadata": {
            "prompt_mode": "code_only",
            "teacher_prompt_mode": "short_diagnosis_code",
            "source_protocol": "cached_first_pass_step1300_repair_cond_v2",
            "source_run_id": "step1300_probe_v0",
            "feedback_source": "teacher_requests",
        },
    }


def _patch_verifier(monkeypatch, summaries):
    monkeypatch.setattr(repair_grpo_batch_reward, "normalize_candidate", lambda solution: solution)
    monkeypatch.setattr(repair_grpo_batch_reward, "verify_candidate_batch", lambda **kwargs: summaries)


def test_compute_score_uses_repair_delta_v0_formula(monkeypatch):
    _patch_verifier(monkeypatch, [_base_summary(accepted=True, pass_ratio_all=1.0, passed_tests=4, total_tests=4, error_type="success")])

    result = repair_grpo_batch_reward.compute_score(
        data_sources=["codecontests_repair_rl"],
        solution_strs=["<code>print('new')</code>"],
        ground_truths=[_base_ground_truth()],
        extra_infos=[{"problem_id": "p1", "finish_reason": "stop", "truncated_by_max_tokens": False}],
        sandbox_endpoint="http://sandbox",
        reward_mode="repair_delta_v0",
    )[0]

    assert result["q0"] == pytest.approx(0.4)
    assert result["q1"] == pytest.approx(1.0)
    assert result["delta_q"] == pytest.approx(0.6)
    assert result["delta_pos"] == pytest.approx(0.6)
    assert result["delta_neg"] == pytest.approx(0.0)
    assert result["accepted_gain"] == pytest.approx(1.0)
    assert result["reward_raw"] == pytest.approx(1.4)
    assert result["score"] == pytest.approx(1.4)
    assert result["problem_id"] == "p1"
    assert result["repair_prompt_mode"] == "code_only"
    assert result["teacher_prompt_mode"] == "short_diagnosis_code"


def test_compute_score_marks_truncated_samples_invalid(monkeypatch):
    _patch_verifier(monkeypatch, [_base_summary(accepted=True, pass_ratio_all=1.0, passed_tests=4, total_tests=4, error_type="success")])

    result = repair_grpo_batch_reward.compute_score(
        data_sources=["codecontests_repair_rl"],
        solution_strs=["<code>print('new')</code>"],
        ground_truths=[_base_ground_truth()],
        extra_infos=[{"problem_id": "p1", "finish_reason": "length"}],
        sandbox_endpoint="http://sandbox",
        reward_mode="repair_delta_v0",
    )[0]

    assert result["invalid_for_rl"] is True
    assert result["invalid_reason"] == "truncated_by_max_tokens"
    assert result["truncated_by_max_tokens"] is True
    assert result["finish_reason"] == "length"
    assert result["score"] == 0.0
    assert math.isnan(result["reward_raw"])
    assert result["q1"] == pytest.approx(1.0)


def test_compute_score_penalizes_bad_output(monkeypatch):
    _patch_verifier(monkeypatch, [_base_summary(pass_ratio_all=0.0, passed_tests=0, total_tests=4, error_type="non_code", extraction_status="non_code")])

    result = repair_grpo_batch_reward.compute_score(
        data_sources=["codecontests_repair_rl"],
        solution_strs=["hello world"],
        ground_truths=[_base_ground_truth(pass_ratio_all=0.75, pass_ratio_bucket="bucket_0.6_1.0")],
        extra_infos=[{"problem_id": "p1", "finish_reason": "stop", "truncated_by_max_tokens": False}],
        sandbox_endpoint="http://sandbox",
        reward_mode="repair_delta_v0",
    )[0]

    assert result["reward_raw"] == pytest.approx(-1.0)
    assert result["score"] == pytest.approx(-1.0)
    assert result["q0"] == pytest.approx(0.6)
    assert result["q1"] == pytest.approx(0.0)
    assert result["delta_neg"] == pytest.approx(0.6)


def test_compute_score_requires_first_pass_contract(monkeypatch):
    _patch_verifier(monkeypatch, [_base_summary()])

    with pytest.raises(ValueError, match="ground_truth.first_pass"):
        repair_grpo_batch_reward.compute_score(
            data_sources=["codecontests_repair_rl"],
            solution_strs=["<code>print('new')</code>"],
            ground_truths=[{"problem_id": "p1"}],
            extra_infos=[{"problem_id": "p1", "finish_reason": "stop", "truncated_by_max_tokens": False}],
            sandbox_endpoint="http://sandbox",
            reward_mode="repair_delta_v0",
        )


@pytest.mark.parametrize(
    ("missing_key", "error_match"),
    [
        ("pass_ratio_all", "ground_truth.first_pass.pass_ratio_all"),
        ("accepted", "ground_truth.first_pass.accepted"),
    ],
)
def test_compute_score_requires_frozen_first_pass_baseline_fields(monkeypatch, missing_key, error_match):
    _patch_verifier(monkeypatch, [_base_summary()])
    ground_truth = _base_ground_truth()
    ground_truth["first_pass"].pop(missing_key)

    with pytest.raises(ValueError, match=error_match):
        repair_grpo_batch_reward.compute_score(
            data_sources=["codecontests_repair_rl"],
            solution_strs=["<code>print('new')</code>"],
            ground_truths=[ground_truth],
            extra_infos=[{"problem_id": "p1", "finish_reason": "stop", "truncated_by_max_tokens": False}],
            sandbox_endpoint="http://sandbox",
            reward_mode="repair_delta_v0",
        )


def test_compute_score_returns_flat_fields_only(monkeypatch):
    _patch_verifier(monkeypatch, [_base_summary()])

    result = repair_grpo_batch_reward.compute_score(
        data_sources=["codecontests_repair_rl"],
        solution_strs=["<code>print('new')</code>"],
        ground_truths=[_base_ground_truth()],
        extra_infos=[{"problem_id": "p1", "finish_reason": "stop", "truncated_by_max_tokens": False}],
        sandbox_endpoint="http://sandbox",
        reward_mode="repair_delta_v0",
    )[0]

    for key, value in result.items():
        assert not isinstance(value, (dict, list, tuple, set)), key
        assert value is not None, key


def test_compute_score_checks_expected_prompt_mode(monkeypatch):
    monkeypatch.setattr(repair_grpo_batch_reward, "normalize_candidate", lambda solution: solution)
    monkeypatch.setattr(
        repair_grpo_batch_reward,
        "verify_candidate_batch",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("verifier should not be called on prompt mismatch")),
    )

    with pytest.raises(ValueError, match="Repair prompt mode mismatch"):
        repair_grpo_batch_reward.compute_score(
            data_sources=["codecontests_repair_rl"],
            solution_strs=["<code>print('new')</code>"],
            ground_truths=[_base_ground_truth()],
            extra_infos=[{"problem_id": "p1", "finish_reason": "stop", "truncated_by_max_tokens": False}],
            sandbox_endpoint="http://sandbox",
            reward_mode="repair_delta_v0",
            expected_prompt_mode="short_diagnosis_code",
        )
