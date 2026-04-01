import math

import pytest

from coding_model_project.src import grpo_batch_reward


class _DummySummary:
    def __init__(self, **payload):
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


def _base_summary(**overrides):
    payload = {
        "accepted": False,
        "passed_tests": 1,
        "total_tests": 4,
        "pass_ratio_all": 0.25,
        "error_type": "wrong_answer",
        "invalid_for_rl": False,
        "invalid_reason": "",
        "judge_time_s": 0.3,
        "extraction_status": "ok",
    }
    payload.update(overrides)
    return _DummySummary(**payload)


def _patch_verifier(monkeypatch, summaries):
    monkeypatch.setattr(grpo_batch_reward, "normalize_candidate", lambda solution: solution)
    monkeypatch.setattr(grpo_batch_reward, "verify_candidate_batch", lambda **kwargs: summaries)


def test_compute_score_supports_formal_reward_modes(monkeypatch):
    _patch_verifier(monkeypatch, [_base_summary(accepted=True, pass_ratio_all=0.5)])

    anchored = grpo_batch_reward.compute_score(
        data_sources=["src"],
        solution_strs=["print('ok')"],
        ground_truths=[{}],
        extra_infos=[{"problem_id": "p1", "finish_reason": "stop", "truncated_by_max_tokens": False}],
        sandbox_endpoint="http://sandbox",
        reward_mode="anchored_dense_v1",
    )[0]
    dense_anchor = grpo_batch_reward.compute_score(
        data_sources=["src"],
        solution_strs=["print('ok')"],
        ground_truths=[{}],
        extra_infos=[{"problem_id": "p1", "finish_reason": "stop", "truncated_by_max_tokens": False}],
        sandbox_endpoint="http://sandbox",
        reward_mode="dense_anchor_v1",
    )[0]
    sparse = grpo_batch_reward.compute_score(
        data_sources=["src"],
        solution_strs=["print('ok')"],
        ground_truths=[{}],
        extra_infos=[{"problem_id": "p1", "finish_reason": "stop", "truncated_by_max_tokens": False}],
        sandbox_endpoint="http://sandbox",
        reward_mode="sparse_accepted",
    )[0]

    assert anchored["score"] == pytest.approx(0.6)
    assert anchored["reward_raw"] == pytest.approx(0.6)
    assert dense_anchor["score"] == pytest.approx(0.5)
    assert dense_anchor["reward_raw"] == pytest.approx(0.5)
    assert sparse["score"] == pytest.approx(1.0)
    assert sparse["reward_raw"] == pytest.approx(1.0)


def test_compute_score_marks_truncated_samples_invalid(monkeypatch):
    _patch_verifier(monkeypatch, [_base_summary(pass_ratio_all=1.0, accepted=True)])

    result = grpo_batch_reward.compute_score(
        data_sources=["src"],
        solution_strs=["print('ok')"],
        ground_truths=[{}],
        extra_infos=[{"finish_reason": "length"}],
        sandbox_endpoint="http://sandbox",
        reward_mode="anchored_dense_v1",
    )[0]

    assert result["invalid_for_rl"] is True
    assert result["invalid_reason"] == "truncated_by_max_tokens"
    assert result["truncated_by_max_tokens"] is True
    assert result["finish_reason"] == "length"
    assert result["score"] == 0.0
    assert math.isnan(result["reward_raw"])


def test_compute_score_handles_non_code_and_legacy_modes(monkeypatch):
    _patch_verifier(monkeypatch, [_base_summary(error_type="non_code", extraction_status="non_code")])

    anchored = grpo_batch_reward.compute_score(
        data_sources=["src"],
        solution_strs=["hello world"],
        ground_truths=[{}],
        extra_infos=[{}],
        sandbox_endpoint="http://sandbox",
        reward_mode="anchored_dense_v1",
    )[0]

    _patch_verifier(monkeypatch, [_base_summary(pass_ratio_all=0.25)])
    dense_pass_ratio = grpo_batch_reward.compute_score(
        data_sources=["src"],
        solution_strs=["print('ok')"],
        ground_truths=[{}],
        extra_infos=[{}],
        sandbox_endpoint="http://sandbox",
        reward_mode="dense_pass_ratio",
    )[0]

    assert anchored["score"] == pytest.approx(-1.0)
    assert anchored["reward_raw"] == pytest.approx(-1.0)
    assert dense_pass_ratio["score"] == pytest.approx(0.25)
    assert dense_pass_ratio["reward_raw"] == pytest.approx(0.25)


def test_compute_score_keeps_dense_pass_ratio_legacy_behavior_for_non_code(monkeypatch):
    _patch_verifier(monkeypatch, [_base_summary(error_type="non_code", extraction_status="non_code", pass_ratio_all=0.0)])

    dense_pass_ratio = grpo_batch_reward.compute_score(
        data_sources=["src"],
        solution_strs=["hello world"],
        ground_truths=[{}],
        extra_infos=[{}],
        sandbox_endpoint="http://sandbox",
        reward_mode="dense_pass_ratio",
    )[0]

    assert dense_pass_ratio["score"] == pytest.approx(0.0)
    assert dense_pass_ratio["reward_raw"] == pytest.approx(0.0)
