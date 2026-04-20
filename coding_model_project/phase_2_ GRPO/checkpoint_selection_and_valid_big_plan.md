# Continuation Checkpoint Selection And `valid_big` Plan

## Scope

This note compares the strict same-protocol full eval results already available on the new machine for:

- baseline
- `step100`
- `step120`
- `step160`
- `step180`
- `step200`

The goal is to pick the best continuation checkpoint and define the next `valid_big` full-output runs.

## Strict Full-Eval Summary

All metrics below are from the same `phase0_eval.py` protocol over:

- `humaneval`
- `mbpp_reg`
- `codecontests_valid`

| checkpoint | CC accepted@1 | CC pass_ratio_mean | CC runtime_error_rate | CC timeout_rate | CC wrong_answer_rate | CC avg_judge_time | HumanEval | MBPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0256 | 0.1422 | 0.2051 | 0.0684 | 0.6923 | 143.36 | 0.8720 | 0.5850 |
| step100 | 0.0513 | 0.1576 | 0.1624 | 0.0684 | 0.7179 | 150.06 | 0.8537 | 0.5700 |
| step120 | 0.0171 | 0.1283 | 0.1538 | 0.0769 | 0.7521 | 73.65 | 0.8537 | 0.5850 |
| step160 | 0.0427 | 0.1613 | 0.1282 | 0.0855 | 0.7436 | 79.86 | 0.8537 | 0.5900 |
| step180 | 0.0342 | 0.1503 | 0.1368 | 0.0855 | 0.7436 | 88.94 | 0.8598 | 0.5900 |
| step200 | 0.0513 | 0.1867 | 0.1282 | 0.1197 | 0.7009 | 106.02 | 0.8537 | 0.6250 |

## Main Read

- `step100` is the first checkpoint that clearly beats baseline on the main `codecontests_valid` target.
- `step120` is a true regression point and should be treated as a negative trajectory for repair analysis.
- `step160` partially recovers and is a decent continuation checkpoint, but it does not beat `step100` on accepted@1.
- `step180` is not the best continuation point.
- `step200` is the strongest continuation checkpoint:
  - matches `step100` on `codecontests_valid accepted@1`
  - clearly exceeds `step100` on `codecontests_valid pass_ratio_mean`
  - improves `codecontests_valid runtime_error_rate`
  - has the best `MBPP_reg` score among all strict checkpoints
  - keeps `HumanEval` tied with `step100`

## Per-Problem Read

### `step100` vs baseline

- gained solves:
  - `Codeforces/1553/H`
  - `Codeforces/1559/A`
  - `Codeforces/1560/A`
  - `Codeforces/1560/F2`
- lost solves:
  - `Codeforces/1548/E`

### `step200` vs baseline

- gained solves:
  - `Codeforces/1552/B`
  - `Codeforces/1553/H`
  - `Codeforces/1554/C`
  - `Codeforces/1560/F2`
- lost solves:
  - `Codeforces/1554/A`

### `step200` vs `step100`

- newly solved relative to `step100`:
  - `Codeforces/1548/E`
  - `Codeforces/1552/B`
  - `Codeforces/1554/C`
- lost solves relative to `step100`:
  - `Codeforces/1554/A`
  - `Codeforces/1559/A`
  - `Codeforces/1560/A`

### Large `pass_ratio_all` gains at `step200` vs `step100`

- `Codeforces/1554/C`: `0.00 -> 1.00`
- `Codeforces/1552/B`: `0.00 -> 1.00`
- `Codeforces/1551/C`: `0.00 -> 0.98`
- `Codeforces/1552/F`: `0.00 -> 0.80`
- `Codeforces/1553/I`: `0.02 -> 0.82`
- `Codeforces/1560/E`: `0.00 -> 0.66`
- `Codeforces/1569/D`: `0.36 -> 0.78`

### Large `pass_ratio_all` drops at `step200` vs `step100`

- `Codeforces/1554/A`: `1.00 -> 0.00`
- `Codeforces/1569/C`: `0.72 -> 0.04`
- `Codeforces/1555/F`: `0.56 -> 0.04`
- `Codeforces/1574/D`: `0.36 -> 0.00`
- `Codeforces/1561/A`: `0.33 -> 0.00`

## Selection

### Best continuation checkpoint

- **Pick `step200` as the best continuation checkpoint.**

Reason:

- best `codecontests_valid pass_ratio_mean`
- ties `step100` on `codecontests_valid accepted@1`
- best `MBPP_reg`
- lower `runtime_error_rate` than `step100`
- no `HumanEval` collapse relative to `step100`

### Runner-up

- `step160` is the runner-up continuation checkpoint.

Reason:

- better than `step120` and `step180`
- lower timeout burden than `step200`
- but weaker than `step200` on the main target

## `valid_big` Full-Output Recommendation

Do not rely on `codecontests_valid (117)` alone for repair-SFT planning or checkpoint selection.

### Minimum useful `valid_big` set

Run full-output `valid_big` for:

1. **baseline**
2. **step100**
3. **step200**

Why this set:

- baseline is required to anchor headline improvement on a larger dev set
- `step100` is the best pre-continuation RL checkpoint
- `step200` is the best continuation checkpoint

### Optional extra run

- `step160`

Only run this if:

- `valid_big` shows `step100` and `step200` are very close
- or you want a lower-timeout continuation alternative

## Data Hygiene For Repair-SFT

- Use `codecontests_valid` and `codecontests_valid_big` to identify failure modes.
- Do **not** train repair-SFT directly on `valid` or `valid_big` tasks.
- Build repair data from:
  - `codecontests_train_wo_valid_big`
  - or external tasks with the same failure patterns

## Logging Note

`phase0_eval.py` currently truncates per-problem prompt logs to `4000` chars and response logs to `12000` chars.

- `coding_model_project/src/phase0_eval.py:202`
- `coding_model_project/src/phase0_eval.py:203`
- `coding_model_project/src/phase0_eval.py:1788`
- `coding_model_project/src/phase0_eval.py:1789`

For `valid_big` behavior analysis, consider increasing the prompt logging cap before running the full-output jobs.
