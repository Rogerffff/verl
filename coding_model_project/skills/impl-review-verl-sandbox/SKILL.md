---
name: impl-review-verl-sandbox
description: Review implemented code against concrete phase implementation plans in RL coding-model projects and verify requirement traceability, verl API usage, SandboxFusion API/field correctness, and metric instrumentation completeness. Use after coding is done for Phase 0/1/3/4 before merge or experiment runs.
---

# Impl Review Verl Sandbox

Review whether code implementation actually satisfies a phase implementation plan.

## Inputs

- One implementation plan document.
- Implemented code paths (or repo scope).
- Optional explicit commit/diff range.

## Execute Workflow

1. Validate document type.
Accept only implementation plan docs with concrete steps, commands, files, and measurable outputs.

2. Build requirement list from the plan.
Split into atomic requirements: behavior, API usage, config keys, output artifacts, metrics.

3. Verify traceability requirement-by-requirement.
For each requirement, classify as `Implemented`, `Partially Implemented`, `Missing`, or `Unverifiable` with file evidence.

4. Verify API correctness.
- verl APIs/configs: class/function names, config keys, expected data flow.
- SandboxFusion APIs/fields: endpoint/use-mode, response fields, status mapping, duration/time fields.

5. Verify metric instrumentation completeness.
Load and apply:
- `references/metric_requirements.md`
- `coding_model_project/experiment_design/eval_protocol.md`
- `coding_model_project/experiment_design/metric_templates.md`
Check that required phase metrics are actually computed and logged.

6. Produce strict review report using `references/output_format.md`.
Findings first, severity ordered, with concrete fixes.

## Cross-Repo Evidence Rules

- verl repo path: `/Users/xiaohui/Desktop/verl/verl`
- SandboxFusion repo path: `/Users/xiaohui/Desktop/verl/SandboxFusion`
- Any SandboxFusion claim must be verified in SandboxFusion repo files.
- Do not mark correctness without file evidence.

## Use Bundled Script

Use `scripts/verify_claims.sh` for fast claim/symbol existence checks across both repos.

```bash
bash scripts/verify_claims.sh \
  --roots /Users/xiaohui/Desktop/verl/verl,/Users/xiaohui/Desktop/verl/SandboxFusion \
  --claims /tmp/claims.txt
```

## Output Requirements

- Findings first (Critical -> High -> Medium -> Low).
- Requirement traceability matrix summary.
- Phase metric coverage summary (`required / implemented / missing`).
- Minimal executable fix plan.
