---
name: plan-review-verl-sandbox
description: Review concrete RL implementation-plan documents for coding-model projects and strictly verify that claimed verl and SandboxFusion APIs, file paths, commands, metrics, and workflow assumptions match the repository. Use for phase execution plans (for example Phase 0/1/3 implementation plans), not high-level design proposals.
---

# Plan Review Verl Sandbox

Review implementation-plan documents (for example `phase0_implementation_plan.md`, `phase1_sft_implementation_plan.md`, `phase3_grpo_implementation_plan.md`) and produce a strict feasibility and API-accuracy report before coding starts.

## Qualify Input Document

1. Confirm the target is an implementation plan, not a high-level design.
Implementation plan signals:
- explicit file paths to edit/create
- runnable commands/scripts
- concrete phase deliverables and acceptance gates
- measurable metric collection plan

2. If the document is high-level only, return `Not Reviewable Yet` and list missing implementation details required for strict API review.

## Execute Review Workflow

1. Read the target implementation-plan file and split it into concrete claims.
Claims include API names, classes/functions, config keys, file paths, runtime assumptions, and metric definitions.

2. Verify each claim against source code, not memory.
Use repository search (`rg` preferred, fallback `grep`) and open the relevant files for evidence.
For cross-repo plans, search all required repositories, not only current working directory.

3. Treat unverified claims as risks.
If a claim cannot be proven from code, mark it as `Unverified` instead of assuming it is correct.

4. Prioritize verl and SandboxFusion integration correctness.
Validate:
- verl trainer/worker APIs and config key names
- rollout/reward/ref-logprob/value flow assumptions
- sandbox execution interface, outputs, and timing fields
- metric computability from available logs/artifacts

5. Run phase metric gate checks.
Load and enforce:
- `references/phase_metric_gate.md`
- `coding_model_project/experiment_design/eval_protocol.md`
- `coding_model_project/experiment_design/metric_templates.md`
Confirm the plan covers required global metrics and phase-specific metrics.

6. Check execution feasibility end-to-end.
Confirm required inputs, checkpoints, datasets, and stage handoff artifacts exist and are compatible.

7. Produce a strict report using `references/output_format.md`.
Every key conclusion must cite concrete file evidence (`path:line`).

## Required Standards

- Never accept vague wording like "调用某 API" without exact symbol/path evidence.
- Separate `Fact`, `Inference`, and `Recommendation`.
- If a better implementation path exists, provide it with migration steps.
- Flag hidden blockers early: missing API, mismatched schema, unavailable metrics, undefined stage outputs.
- Enforce cross-repo evidence rules for this project:
  - verl repo: `/Users/xiaohui/Desktop/verl/verl`
  - SandboxFusion repo: `/Users/xiaohui/Desktop/verl/SandboxFusion`
  - Any SandboxFusion claim must be validated in the SandboxFusion repo path above.

## Review Scope

Use `references/review_checklist.md` as the default checklist. Expand only when the plan introduces extra systems.
For metrics, always apply `references/phase_metric_gate.md`.

## Use Bundled Script

Use `scripts/verify_symbols.sh` to quickly check whether claimed symbols or config keys exist in the repo.

Example:

```bash
bash scripts/verify_symbols.sh --root /path/to/repo --symbols /path/to/symbols.txt
```

`symbols.txt` should contain one symbol per line (class/function/config key/file fragment).

For this project, run symbol checks for both repos when the plan includes both systems:

```bash
bash scripts/verify_symbols.sh --root /Users/xiaohui/Desktop/verl/verl --symbols /tmp/symbols.txt
bash scripts/verify_symbols.sh --root /Users/xiaohui/Desktop/verl/SandboxFusion --symbols /tmp/sandbox_symbols.txt
```

## Output Requirements

- Provide severity-ordered findings first.
- Include a pass/fail summary for each implementation block (setup/run/eval/handoff).
- End with a minimal fix plan that can be directly executed.
