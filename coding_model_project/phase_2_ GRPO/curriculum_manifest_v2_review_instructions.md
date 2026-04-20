# Curriculum Manifest v2 Review Instructions

## Goal
Clean `A_retention` and `C_hard_partial` for the next curriculum pilot.

The review objective is:
- make `A_retention` smaller but much cleaner
- keep `B_near_miss` as the main training bucket
- shrink `C_hard_partial` to only structurally relevant hard neighbors

## Files
- `ac_manual_review_candidates_<tag>.jsonl`
  Combined review queue.
- `a_retention_review_candidates_<tag>.jsonl`
  Full `A_retention` review queue.
- `c_hard_partial_review_candidates_<tag>.jsonl`
  Full `C_hard_partial` review queue.
- `ac_manual_review_decisions_<tag>.jsonl`
  Decision template to fill.

## Review Contract
Each line is one train problem keyed by:
- `dataset`
- `problem_id`

The reviewer should only modify:
- `decision`
- `decision_reason`
- `reviewer`

Do not edit the identity fields.

## Allowed Decisions

### For `A_retention`
- `keep_A`
- `move_to_B`
- `move_to_U`
- `drop`

### For `C_hard_partial`
- `keep_C`
- `move_to_B`
- `move_to_U`
- `drop`

## Heuristics

### `A_retention`
Prefer `keep_A` only when the candidate is a strong anti-regression neighbor:
- same series / same family
- same IO structure
- same algorithmic motif
- likely to preserve a known canary

Prefer `move_to_U` or `drop` when:
- similarity is mostly lexical
- title overlap is high but task structure is different
- runtime state drifted from `A` toward `C/D`

### `C_hard_partial`
Prefer `move_to_B` when:
- the candidate is structurally relevant
- runtime state drifted toward `A/B`

Prefer `move_to_U` or `drop` when:
- it stayed hard without meaningful relation to the seed
- it collapsed toward dead-hard behavior
- it looks like a weak lexical match

## Output Quality Bar
- `A_retention` should become smaller but cleaner
- `C_hard_partial` should become much smaller
- `B_near_miss` should absorb only genuinely useful neighbors

## Suggested Reviewer Workflow
1. Review all `A_retention` rows.
2. Review all `C_hard_partial` rows.
3. If time is limited, start from:
   - `review_priority = p0`
   - then `p1`

## Notes
- `matched_seed_summaries` contains the eval seed context.
- `step600_state` and `step620_state` provide runtime evidence from the first curriculum pilot.
- `suggested_action` is only a heuristic, not a command.
