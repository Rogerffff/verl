# Output Format

## 0) Reviewability Gate

- Document type: Implementation Plan / High-Level Design
- Gate result: Reviewable / Not Reviewable Yet
- If not reviewable: required missing details (commands, file paths, acceptance checks, outputs)

## 1) Findings (highest severity first)

For each finding:
- ID: short identifier
- Severity: Critical / High / Medium / Low
- Claim: exact plan statement being checked
- Verdict: Correct / Incorrect / Unverified / Partially Correct
- Evidence: file references with line numbers
- Impact: what breaks or becomes unreliable
- Fix: concrete correction

## 2) Phase Gate Summary

- Phase name
- Gate status: Pass / Fail / Conditional
- Blocking issues
- Non-blocking improvements

## 3) Minimal Fix Plan

Numbered list with smallest executable edits to make the plan implementation-ready.
