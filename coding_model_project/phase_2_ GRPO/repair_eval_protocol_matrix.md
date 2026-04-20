# Repair Eval Protocol Matrix

## Goal

This note fixes the evaluation contract for repair-SFT so we do not mix:

- stronger raw first-pass ability
- stronger one-turn repair ability

The project should report both, but with different protocols.

## Protocols

### Protocol A: Fixed-input repair eval

Definition:

- Freeze a canonical first-pass artifact from one source checkpoint.
- Reuse the same:
  - problem statement
  - first-pass code
  - verifier feedback
- Let different candidate models perform only the repair pass.

This is the primary metric for repair-SFT.

What it answers:

- Did the model learn to repair code better when given the same bad solution and the same feedback?

Recommended canonical source:

- `step900` canonical raw `per_problem`

Recommended datasets:

- `codecontests_valid_big`
- `codecontests_test`

### Protocol B: Self-first-pass repair eval

Definition:

- Each model first generates its own first-pass solution.
- Each model then repairs its own code.

This is a secondary, end-to-end usability metric.

What it answers:

- If we deploy this model as a practical one-turn repair system, how good is the full pipeline?

## Reporting Priority

### Primary repair-SFT headline

Use Protocol A.

Recommended comparison set:

- `step900`
- `step40`
- `step1300`

on:

- `codecontests_valid_big`
- `codecontests_test`

### Secondary deployment headline

Use Protocol B.

Recommended comparison set:

- `step900`
- `step40`
- `step1300`

on:

- `codecontests_valid_big`
- `codecontests_test`

## Why This Split Matters

`step1300` is currently the strongest raw RL checkpoint. If we only compare self-first-pass repair runs, then a better first-pass model can look like a better repair model even when its repair skill is not actually stronger.

Protocol A removes that confound and is therefore the correct main metric for repair-SFT.

## Current Canonical Choices

### ValidBig500 Protocol A source

Current canonical fixed source:

- `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step620_v3q3_to900_seed0_step900_codecontests_validbig500_raw_patched_rr828x_vastai3/per_problem`

Current canonical repair settings for the existing comparable runs:

- `prompt_mode = short_diagnosis_code`
- `repair_error_types = [wrong_answer, runtime_error, timeout]`
- `repair_min_pass_ratio = 0.2`
- `repair_max_failure_cases = 1`
- `repair_max_feedback_chars = 1200`
- `repair_max_prompt_chars = 12000`
- patched sandbox
- direct-backend client RR

For apples-to-apples comparison, any missing Protocol A run on `valid_big500` should use this same contract.

### CodeContests Test Protocol A source

Recommended canonical fixed source:

- a dedicated `step900` raw run on `codecontests_test`

This source should be materialized once and then reused by:

- `step900`
- `step40`
- `step1300`

## Current Asset Status

### Already present on remote `vastai5`

Comparable `Protocol A / valid_big500 / short_diagnosis_code / reuse_step900 / p>=0.2` runs:

- `step900`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolA_full/step900_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5`
- `step40`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolA_full/step40_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5`

Non-comparable `step1300` valid_big500 repair assets also exist, but they are not the same protocol:

- `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_ablation/grpo_a1_step1300_validbig500_repair_reuse_ablate_short_diag_p06_wa_re`

Why non-comparable:

- it is an ablation namespace
- it uses a narrower trigger (`p>=0.6`)
- it is not the same “Protocol A full” contract as the `step900` and `step40` runs above

### Missing or still needed

To close the main valid_big500 triad:

- `step1300` Protocol A on `valid_big500`, using the same reused `step900` source and same settings as the existing `step900` and `step40` runs

To close the main held-out test triad:

- a canonical `step900` raw `codecontests_test` first-pass source
- `step900` Protocol A on `codecontests_test`
- `step40` Protocol A on `codecontests_test`
- `step1300` Protocol A on `codecontests_test`

## Interpretation Rules

If Protocol A says:

- `step40 > step900`

then repair-SFT improved repair skill.

If Protocol B says:

- `step1300 >= step40`

then the stronger RL checkpoint may still be the better deployed end-to-end system.

These are not contradictory findings. They answer different questions.

## Recommended Next Queue

### Priority 1

- Run `step1300` Protocol A on `valid_big500` with the same reused `step900` source and the same settings as the existing `step900` and `step40` valid_big500 runs.

### Priority 2

- Materialize canonical `step900` raw on `codecontests_test`.

### Priority 3

- Run Protocol A on `codecontests_test` for:
  - `step900`
  - `step40`
  - `step1300`

### Priority 4

- Only after Protocol A is complete, use Protocol B as the end-to-end supporting table.
