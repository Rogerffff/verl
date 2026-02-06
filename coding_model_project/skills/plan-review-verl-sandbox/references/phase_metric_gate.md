# Phase Metric Gate (Plan Review)

Apply this gate together with:
- `coding_model_project/experiment_design/eval_protocol.md`
- `coding_model_project/experiment_design/metric_templates.md`

## Global Must-Haves

- Plan must include `EVAL@1` as main protocol.
- If plan includes extra sampling or multi-round generation, plan must include `EVAL@budget`.
- Quality metrics and cost metrics must be reported together under same protocol.

## Core Metrics Required in Plan

- `accepted@1` (or dataset-equivalent `pass@1`)
- `pass_ratio_mean`
- `pass_ratio_p50`
- `pass_ratio_p90`
- `exec_success_rate`
- error breakdown:
  - `syntax_error_rate`
  - `runtime_error_rate`
  - `timeout_rate`
  - `wrong_answer_rate`
- cost panel:
  - `avg_total_gen_tokens`
  - `avg_total_judge_time`
  - `throughput`
  - `cost_per_solved_tokens`
  - `cost_per_solved_judge_time`

## Phase-Specific Plan Requirements

### Phase 0

- include full core metrics + protocol statement
- include reliability signals where available (`api_error_rate`, `sandbox_error_rate`)

### Phase 1 (SFT)

- include training metrics (`train/loss`, `train/grad_norm`, `train/lr`, `val/loss`)
- include core evaluation metrics for checkpoint comparison

### Phase 3 (GRPO)

- include GRPO training metrics (at least):
  - `critic/score/mean`
  - `actor/loss`
  - `actor/grad_norm`
  - `actor/clip_ratio`
  - `actor/kl_loss`
- include custom reward health metrics:
  - `pass_ratio_mean`
  - `accepted_rate_train`
  - `timeout_rate_train`
  - `zero_reward_rate`
- include phase eval metrics with protocol separation (`EVAL@1`, `EVAL@budget`)

### Phase 4 (Multi-round)

- include repair effectiveness metrics:
  - `accepted@multi`
  - `recovery_rate`
  - `delta_pass_ratio_mean`
  - `avg_rounds_used`
- include round-level cost decomposition

## Severity Rule

If a plan claims phase readiness but misses any mandatory metric in this gate, mark as at least `High` severity unless explicitly justified.
