# Metric Requirements (Phase-Aware)

Use this file with:
- `coding_model_project/experiment_design/eval_protocol.md`
- `coding_model_project/experiment_design/metric_templates.md`

## Global Protocol Requirements

- Always include `EVAL@1` outputs.
- For methods using extra sampling/rounds, include `EVAL@budget` outputs.
- Keep quality and cost reported together under same protocol.

## Core Quality Metrics

- `accepted@1` (or dataset-equivalent `pass@1`)
- `pass_ratio_mean`
- `pass_ratio_p50`
- `pass_ratio_p90`
- `exec_success_rate`
- error breakdown rates:
  - `syntax_error_rate`
  - `runtime_error_rate`
  - `timeout_rate`
  - `wrong_answer_rate`

## Core Cost Metrics

- `avg_total_gen_tokens`
- `avg_total_judge_time`
- `throughput`
- `cost_per_solved_tokens`
- `cost_per_solved_judge_time`

## Phase-Specific Must-Haves

### Phase 0

- Quality metrics + cost metrics above
- reliability metrics where available:
  - `api_error_rate`
  - `sandbox_error_rate`

### Phase 1 (SFT)

- training metrics:
  - `train/loss`
  - `train/grad_norm`
  - `train/lr`
  - `val/loss`
- evaluation metrics: same core quality/cost set for checkpoints

### Phase 3 (GRPO)

- training metrics (at least):
  - `critic/score/mean`
  - `actor/loss`
  - `actor/grad_norm`
  - `actor/clip_ratio`
  - `actor/kl_loss`
- custom reward/eval monitoring:
  - `pass_ratio_mean`
  - `accepted_rate_train`
  - `timeout_rate_train`
  - `zero_reward_rate`
- phase eval metrics: same core quality/cost set with protocol separation

### Phase 4 (Multi-round)

- `accepted@multi`
- `recovery_rate`
- `delta_pass_ratio_mean`
- `avg_rounds_used`
- round-level cost decomposition metrics

## Review Rule

If plan claims a phase-complete implementation, missing any phase must-have metric is at least `High` severity unless explicitly justified and approved.
