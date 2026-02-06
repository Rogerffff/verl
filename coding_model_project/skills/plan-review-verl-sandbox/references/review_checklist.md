# Review Checklist (verl + SandboxFusion)

## 0. Document Type Gate (must pass first)

- Confirm the target document is an implementation plan, not a strategy/vision document.
- Confirm it includes concrete commands, code touch-points, and deliverable paths.
- If missing, stop strict API review and return the missing implementation details list.

## 1. API and Symbol Correctness

- Verify every mentioned class/function exists in code.
- Verify config keys match actual dataclass/config definitions.
- Verify phase-specific algorithm selection fields (for example `adv_estimator`, `loss_type`) are real and compatible.

## 2. Trainer and Worker Flow

- Verify rollout -> reward -> ref logprob -> value -> advantage -> update flow aligns with trainer implementation.
- Verify assumptions about PPO/GRPO differences are correct (for example critic usage conditions).
- Verify worker backend assumptions (FSDP vs Megatron vs rollout backend) are backed by code paths.

## 3. SandboxFusion Integration

- Verify SandboxFusion-related claims using files under `/Users/xiaohui/Desktop/verl/SandboxFusion`, not only current repo.
- Verify execution API contract used by the plan matches actual sandbox input/output schema.
- Verify fields used for metrics exist (for example pass ratio, error type, execution duration/time).
- Verify timeout/runtime/syntax classification logic is implementable with available artifacts.

## 4. Metrics and Logging Feasibility

- Load and apply `references/phase_metric_gate.md`.
- Cross-check against:
  - `coding_model_project/experiment_design/eval_protocol.md`
  - `coding_model_project/experiment_design/metric_templates.md`
- Verify each metric in plan is computable from existing logs or can be added clearly.
- Verify naming consistency between planned metrics and existing tracking code.
- Verify cost metrics do not mix incompatible units.
- Flag missing phase-mandatory metrics as `High` severity by default.

## 5. Data Split and Evaluation Hygiene

- Verify train/val/test isolation rules are enforceable by data pipeline.
- Verify benchmark usage constraints are respected (no leakage to training/tuning).
- Verify checkpoint selection and evaluation cadence are operationally feasible.

## 6. Deliverables and Handoff

- Verify each phase has explicit outputs consumed by next phase.
- Verify failure criteria and rollback path are defined.
- Verify open assumptions are listed and testable.

## 7. Implementation Readiness

- Verify run commands are executable in current repo layout.
- Verify every "to-be-implemented" step maps to concrete files/modules.
- Verify acceptance criteria can be checked automatically or by deterministic manual checks.
