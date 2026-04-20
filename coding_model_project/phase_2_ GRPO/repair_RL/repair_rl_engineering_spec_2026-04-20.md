# Repair RL Engineering Spec (2026-04-20)

## 0. Purpose

This document translates the current formal repair reward design into a small engineering spec.

It focuses only on three questions:

1. what input fields the reward adapter must receive,
2. what columns / nested fields the repair RL parquet must contain,
3. what the minimum code changes are in the current `verl` + `coding_model_project` pipeline.

This document does **not** cover:

- repair data selection policy,
- complete second-pass repair RL probe planning,
- experiment scheduling,
- hyperparameter search.

The formal reward definitions themselves live in:

- [repair_reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_RL/repair_reward_design.md)

---

## 1. Current Formal Scope

The current reward family is:

- `repair_delta_v0`
- `repair_delta_edit_v1`

The rollout order is fixed as:

1. first implement and run `repair_delta_v0`
2. only after `v0` shows signal, add `repair_delta_edit_v1`

So this engineering spec is written with the following priority:

- **v0 must be easy to implement**
- **v1 must be enabled with minimal incremental changes**

---

## 2. Current Pipeline Baseline

### 2.1 Current single-turn RL data path

The current general GRPO parquet builder emits records like:

- `data_source`
- `prompt`
- `ability`
- `reward_model.ground_truth`
- `extra_info`

Current builder reference:

- [build_grpo_parquet.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_grpo_parquet.py#L118)

Current reward adapter reference:

- [grpo_batch_reward.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/grpo_batch_reward.py#L75)

Current batch reward manager behavior:

- decodes rollout responses,
- reads `reward_model.ground_truth`,
- reads `extra_info`,
- calls `compute_score(...)`,
- writes returned fields into `reward_extra_info`.

Reference:

- [batch.py](/Users/roger/Desktop/coding_RL_project/verl/verl/workers/reward_manager/batch.py#L47)

### 2.2 Why repair RL needs a parallel data path

Repair RL is not just "ordinary coding RL with a different reward mode".

It changes the sample semantics from:

- `problem -> code`

to:

- `problem + first-pass buggy code + verifier feedback -> repaired code`

So the current generic builder is not enough.  
The cleanest low-risk path is:

- keep the existing general builder untouched,
- add a dedicated repair RL parquet builder,
- add a dedicated repair reward adapter module,
- keep trainer-side changes as small as possible.

---

## 3. Recommended Minimal File Strategy

To minimize blast radius, the recommended implementation layout is:

### 3.1 New builder

Add a dedicated builder, for example:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_repair_rl_parquet.py`

Responsibility:

- read cached first-pass repair assets,
- build second-pass repair prompts,
- emit parquet records in the same outer schema shape as current RL parquet,
- but with repair-specific `ground_truth` and `extra_info`.

### 3.2 New reward adapter

Add a dedicated reward adapter, for example:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_grpo_batch_reward.py`

Responsibility:

- compute `repair_delta_v0`,
- later extend to `repair_delta_edit_v1`,
- reuse existing shared verifier,
- return the same style of compact `reward_extra_info` dicts as current `grpo_batch_reward.py`.

### 3.3 Keep current shared verifier

Do not fork the verifier if avoidable.

Continue using:

- `normalize_candidate`
- `verify_candidate_batch`

from:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py`

### 3.4 Keep current GRPO trainer logic

For `repair_delta_v0`, no core GRPO algorithm change should be required.

The current trainer already supports:

- `invalid_for_rl`
- groupwise GRPO advantage by `uid`
- reward extra info logging

References:

- [ray_trainer.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/ppo/ray_trainer.py#L237)
- [metric_utils.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/ppo/metric_utils.py#L240)

---

## 4. Reward Adapter Input Contract

This section answers:

- what data `compute_score(...)` must receive for repair RL.

There are two categories:

1. repaired rollout verdict fields
2. frozen first-pass context fields

### 4.1 Required repaired rollout verdict fields

These come from the repaired candidate after passing through the existing shared verifier.

They are required for both `repair_delta_v0` and `repair_delta_edit_v1`.

Required fields:

- `pass_ratio_all`
- `accepted`
- `error_type`
- `invalid_for_rl`
- `invalid_reason`
- `extraction_status`
- `judge_time_s`

Required guardrail context:

- `finish_reason`
- `truncated_by_max_tokens`

These are already part of the current verifier / reward contract.

### 4.2 Required frozen first-pass fields for `repair_delta_v0`

These must be present per sample before reward computation starts.

Minimum required fields:

- `problem_id`
- `dataset`
- `test_cases`
- `first_pass_pass_ratio_all`
- `first_pass_accepted`

Strongly recommended:

- `first_pass_error_type`
- `first_pass_invalid_for_rl`
- `first_pass_invalid_reason`
- `first_pass_finish_reason`
- `first_pass_pass_ratio_bucket`

Why these are needed:

- `repair_delta_v0` needs `q0`, which is computed from:
  - `first_pass_pass_ratio_all`
  - `first_pass_accepted`
- filtering / debugging benefits from knowing whether the first-pass artifact was itself invalid or truncated

### 4.3 Additional required fields for `repair_delta_edit_v1`

`repair_delta_edit_v1` needs everything from `v0`, plus:

- `first_pass_code`

because edit metrics compare:

- frozen first-pass code
- repaired rollout code

If `first_pass_code` is not available, `repair_delta_edit_v1` cannot be computed.

### 4.4 Group information requirement for `repair_delta_edit_v1`

This is the one place where `v1` has a real interface requirement beyond `v0`.

`repair_delta_edit_v1` uses:

- `candidate_good_gate`
- `group_good_gate`

The group gate is defined over all rollouts from the same prompt group.

So the reward adapter must be able to identify which repaired samples belong to the same rollout group.

Current issue:

- the existing `compute_score(...)` call gets:
  - `data_sources`
  - `solution_strs`
  - `ground_truths`
  - `extra_infos`
- but it does **not** explicitly receive `uid`

Current GRPO grouping id exists in the trainer:

- `batch.non_tensor_batch["uid"]`

but that value is not currently merged into `extra_info` before `compute_score(...)`.

Therefore:

- `repair_delta_v0` does **not** need any trainer-side change here
- `repair_delta_edit_v1` **does** need access to per-sample group id

Minimal fix for `v1`:

- in [batch.py](/Users/roger/Desktop/coding_RL_project/verl/verl/workers/reward_manager/batch.py#L67)
- when constructing `merged_extra`,
- also inject:
  - `merged_extra["uid"] = data.non_tensor_batch["uid"][i]`

This is the smallest way to let the reward adapter compute group-level edit gates.

---

## 5. Parquet Schema: Recommended Repair RL Record Shape

The outer parquet schema should remain compatible with current RL dataset conventions:

- `data_source`
- `prompt`
- `ability`
- `reward_model`
- `extra_info`

### 5.1 Top-level record

Recommended shape:

```python
{
    "data_source": "codecontests_repair_rl",
    "prompt": [...chat messages for second-pass repair...],
    "ability": "code",
    "reward_model": {
        "style": "rule",
        "ground_truth": {...}
    },
    "extra_info": {...}
}
```

### 5.2 `prompt`

For repair RL, `prompt` should be the already-built second-pass repair prompt, not the original first-pass coding prompt.

Preferred source:

- reuse the existing repair prompt construction logic from:
  - [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)

The builder should emit:

- system prompt
- user repair prompt

matching the repair mode chosen for training.

### 5.3 `reward_model.ground_truth`

Recommended minimum structure:

```python
{
    "problem_id": str,
    "dataset": str,
    "test_cases": {...},
    "first_pass": {
        "code": str,                  # required for v1, optional but recommended for v0
        "pass_ratio_all": float,
        "accepted": bool,
        "error_type": str,
        "invalid_for_rl": bool,
        "invalid_reason": str,
        "finish_reason": str,
        "pass_ratio_bucket": str,
    },
    "repair_metadata": {
        "prompt_mode": str,
        "source_run_id": str,
        "source_protocol": str,       # recommended, e.g. cached-first-pass / reuse-step1300
    },
}
```

Required today for `v0`:

- `problem_id`
- `dataset`
- `test_cases`
- `first_pass.pass_ratio_all`
- `first_pass.accepted`

Required later for `v1`:

- `first_pass.code`

Why place these in `ground_truth`:

- these are stable per-sample repair inputs,
- they are semantically part of the "truth contract" for reward computation,
- the current reward manager already reads `ground_truths` directly.

### 5.4 `extra_info`

Recommended minimum structure:

```python
{
    "problem_id": str,
    "dataset": str,
    "split": str,
    "repair_prompt_mode": str,
    "first_pass_pass_ratio_all": float,
    "first_pass_accepted": bool,
    "first_pass_bucket": str,
    "first_pass_source": str,
}
```

Recommended optional fields:

- `first_pass_error_type`
- `first_pass_invalid_for_rl`
- `first_pass_finish_reason`
- `first_pass_prompt_sha256`
- `first_pass_response_sha256`

Why keep these in `extra_info`:

- they are useful for logging and reward debug,
- they are lightweight,
- they can be surfaced into `reward_extra_info` with minimal effort.

### 5.5 Required new parquet columns summary

Compared with current `build_grpo_parquet.py`, repair RL parquet must newly supply:

- second-pass repair prompt in `prompt`
- frozen first-pass summary in `reward_model.ground_truth.first_pass`
- repair task metadata in `reward_model.ground_truth.repair_metadata`
- lightweight duplicated first-pass debug info in `extra_info`

If only `repair_delta_v0` is being implemented, the absolute minimum addition is:

- `first_pass.pass_ratio_all`
- `first_pass.accepted`

If `repair_delta_edit_v1` is planned, also add:

- `first_pass.code`

from day one, so the data format does not need to change again later.

---

## 6. Reward Adapter Spec

### 6.1 Adapter entrypoint

Recommended new module:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_grpo_batch_reward.py`

Recommended entrypoint shape:

```python
def compute_score(
    *,
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    sandbox_endpoint: str,
    reward_mode: str,
    limiter_budget: int,
    run_timeout_s: int,
    memory_limit_mb: int,
    autofix_codecontests_entrypoint: bool = False,
) -> List[dict[str, Any]]:
    ...
```

This matches the current custom batch reward interface closely, minimizing integration work.

### 6.2 `repair_delta_v0` computation requirements

The adapter must:

1. normalize repaired rollout code
2. verify repaired rollout using shared verifier
3. extract:
   - `p1 = pass_ratio_all`
   - `a1 = accepted`
4. read frozen first-pass values from `ground_truth.first_pass`
5. compute:
   - `q0 = 0.8 * p0 + 0.2 * a0`
   - `q1 = 0.8 * p1 + 0.2 * a1`
   - `delta_pos`
   - `delta_neg`
   - `accepted_gain`
6. apply invalid and bad-output guardrails
7. return:
   - `score`
   - `reward_raw`
   - standard verifier fields
   - repair-specific logs

### 6.3 Repair-specific logs recommended for `reward_extra_info`

At minimum for `v0`, return:

- `q0`
- `q1`
- `delta_q`
- `delta_pos`
- `delta_neg`
- `accepted_gain`
- `first_pass_pass_ratio_all`
- `first_pass_accepted`
- `first_pass_bucket`

These are not required for reward computation once score is produced, but they are highly useful for:

- debugging,
- checkpoint comparison,
- downstream metric aggregation.

### 6.4 `repair_delta_edit_v1` incremental requirements

For `v1`, the same adapter can be extended to also compute:

- line-level edit ratio
- token-level edit ratio
- hybrid edit ratio
- dynamic edit budget
- candidate gate
- group gate
- edit penalty

Additional returned log fields for `v1`:

- `line_lcs_edit_ratio`
- `token_levenshtein_ratio`
- `hybrid_edit_ratio`
- `edit_budget`
- `edit_over`
- `candidate_good`
- `group_good_count`
- `group_good_rate`
- `group_gate`
- `edit_penalty`

### 6.5 Guardrail compatibility requirement

The new repair reward adapter should preserve the current invalid sample semantics:

- infra / sandbox / API / no-test / truncation
  - `invalid_for_rl = True`
  - `score = 0.0`
  - `reward_raw = NaN`

and should preserve the current bad-output negative path:

- `empty_output`
- `non_code`
- `extraction_failure`
- `syntax_error`
  - `reward = -1.0`

This keeps repair RL aligned with the current project-wide verifier contract.

---

## 7. Minimal Code Changes

This section is the practical answer to:

- what files need to change, and how much.

### 7.1 New file: repair parquet builder

Add:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_repair_rl_parquet.py`

Why:

- current general builder only knows how to format first-pass coding prompts,
- repair RL needs second-pass prompts plus frozen first-pass context.

This file is required.

### 7.2 New file: repair reward adapter

Add:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_grpo_batch_reward.py`

Why:

- keeps repair-specific logic separate from current mainline coding reward adapter,
- reduces risk of disturbing normal GRPO experiments,
- makes `v0 -> v1` upgrade simpler.

This file is required.

### 7.3 Config wiring

Need one config-level change so trainer loads the repair reward adapter instead of the normal coding reward adapter for repair RL experiments.

Exact path depends on your existing reward-fn import setup, but at minimum:

- repair RL run config must point to the new reward adapter module,
- reward kwargs must include:
  - `reward_mode=repair_delta_v0` or `repair_delta_edit_v1`
  - sandbox endpoint
  - concurrency / timeout settings

This change is required.

### 7.4 No trainer core change required for `repair_delta_v0`

For `v0`, the current trainer core is already enough.

No change should be required in:

- `ray_trainer.py`
- GRPO advantage code
- shared verifier core

because:

- reward remains outcome-style scalar reward,
- invalid masking already exists,
- groupwise GRPO already uses `uid`.

### 7.5 One minimal trainer-side change required for `repair_delta_edit_v1`

If and only if `repair_delta_edit_v1` is enabled:

modify:

- `/Users/roger/Desktop/coding_RL_project/verl/verl/workers/reward_manager/batch.py`

Current code builds:

```python
merged_extra = dict(extra_info)
merged_extra["rollout_reward_scores"] = rollout_reward_scores[i]
```

For `v1`, minimally add:

```python
merged_extra["uid"] = data.non_tensor_batch["uid"][i]
```

Why:

- reward adapter then knows prompt-group membership,
- group-gated edit penalty can be computed without changing the trainer core API.

This is the smallest required trainer-side change for `v1`.

### 7.6 Metric logging changes are optional, not required

You may optionally extend:

- `/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/ppo/metric_utils.py`

to aggregate:

- `q0/q1/delta_q`
- regression rate
- edit metrics
- group gate rate

But this is not required for the first runnable version.

The first runnable version can rely on:

- raw `reward_extra_info` dumps
- rollout data inspection

---

## 8. Recommended Implementation Order

### 8.1 Order for `repair_delta_v0`

1. build repair RL parquet builder
2. emit second-pass prompts + frozen first-pass fields
3. add repair reward adapter
4. implement `repair_delta_v0`
5. return compact repair-specific logs
6. run first small probe

### 8.2 Order for `repair_delta_edit_v1`

Only after `v0` is working:

1. add `first_pass.code` consumption in reward adapter
2. add edit metric helpers
3. add `uid` pass-through in batch reward manager
4. implement group-gated edit penalty
5. run upgrade probe

---

## 9. Minimal Required Fields Checklist

### 9.1 `repair_delta_v0` required to run

Required in parquet / reward input:

- `problem_id`
- `dataset`
- `test_cases`
- `first_pass.pass_ratio_all`
- `first_pass.accepted`
- repair prompt in `prompt`

Required at reward runtime:

- repaired rollout verifier summary:
  - `pass_ratio_all`
  - `accepted`
  - `error_type`
  - `invalid_for_rl`
  - `invalid_reason`
  - `extraction_status`
  - `judge_time_s`
  - truncation info

### 9.2 `repair_delta_edit_v1` additional required fields

Add:

- `first_pass.code`
- `uid` passed into reward-time extra info

Without those two additions, `v1` cannot be computed correctly.

---

## 10. Recommended Final Folder State

This `repair_RL/` folder should contain at least:

- [repair_reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_RL/repair_reward_design.md)
- [repair_rl_engineering_spec_2026-04-20.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_RL/repair_rl_engineering_spec_2026-04-20.md)

Recommended future additions:

- repair parquet schema doc
- minimal probe plan
- implementation checklist

---

## 11. Final Working Summary

The smallest viable path is:

- new repair parquet builder,
- new repair reward adapter,
- no trainer core changes for `repair_delta_v0`.

The smallest additional change needed for `repair_delta_edit_v1` is:

- pass `uid` into reward-time `extra_info`,
- plus provide frozen `first_pass.code`.

That is the minimum engineering shape needed to make the current formal reward design executable inside the existing project.
