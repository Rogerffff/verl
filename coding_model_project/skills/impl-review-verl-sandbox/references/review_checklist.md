# Implementation Review Checklist

## 1. Requirement Traceability

- Extract atomic requirements from the implementation plan.
- Map each requirement to concrete code evidence (`path:line`).
- Mark status: Implemented / Partially Implemented / Missing / Unverifiable.

## 2. verl API Correctness

- Verify referenced trainer/worker/reward APIs exist and are used correctly.
- Verify config keys are valid and wired to runtime behavior.
- Verify assumptions about PPO/GRPO/SFT flow match real code paths.

## 3. SandboxFusion API and Field Correctness

- Verify calls/SDK usage against SandboxFusion repo implementation and docs.
- Verify response fields used by code actually exist.
- Verify error/status mapping is consistent with available statuses.
- Verify time/cost fields (`duration`, wall time, etc.) are correctly sourced.

## 4. Metrics Coverage

- Verify all required phase metrics are computed.
- Verify metrics are logged to expected sinks (WandB/files).
- Verify metric names and formulas follow protocol documents.

## 5. Operability and Acceptance

- Verify commands/scripts in plan are executable with current repo layout.
- Verify output artifacts required by plan are produced at expected paths.
- Verify acceptance gates are machine-checkable or deterministic.
