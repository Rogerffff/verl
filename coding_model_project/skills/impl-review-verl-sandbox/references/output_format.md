# Output Format

## 1) Findings (ordered by severity)

For each finding include:
- ID
- Severity
- Requirement/Claim
- Verdict: Implemented / Partially Implemented / Missing / Unverifiable / Incorrect
- Evidence (`path:line`)
- Impact
- Fix

## 2) Requirement Traceability Summary

- Total requirements
- Implemented count
- Partially implemented count
- Missing count
- Unverifiable count

## 3) Metric Coverage Summary

- Phase
- Required metrics count
- Implemented metrics count
- Missing metrics list

## 4) API Correctness Summary

- verl API correctness: Pass/Fail + blockers
- SandboxFusion API correctness: Pass/Fail + blockers

## 5) Minimal Fix Plan

Numbered smallest executable changes.
