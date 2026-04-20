# RL Canary Suite v1

This suite is the recommended external canary layer for the next `step200 -> +60` RL continuation.

Use it at save-points (`220 / 240 / 260`) while keeping the trainer's `fast_val_codecontests.parquet`
online validation as a smoke signal only.

## Design

Routine save-point canary is `CodeContests`-only:

- `retention_canary_codecontests` (`10`)
- `timeout_canary_codecontests` (`6`)
- `structural_hard_watchlist` (`6`)
- `mid_val_codecontests_32` (`32`)

Total routine external canary size: `54` problems.

`HumanEval` and `MBPP_reg` are intentionally **not** part of the routine save-point canary.
They should be used only for:

- low-frequency spot checks, or
- final strict shortlist evaluation

## Why This Layout

For the current mainline objective, checkpoint selection should be dominated by `CodeContests`
signal, not benchmark mini-slices.

- `fast_val_codecontests.parquet` stays online for smoke / trend / invalid-truncation checks
- the external canary gives a cleaner checkpoint selection signal at each save-point
- `mid_val_codecontests_32` adds a broader dev slice so selection is not dominated by only
  `retention / timeout / structural` specialty buckets

## Recommended Usage

- Keep `fast_val_codecontests.parquet` as trainer internal validation
- Run this external suite at:
  - `pre-SFT step200` once as anchor baseline
  - `step220`
  - `step240`
  - `step260`
- Only after shortlist selection, consider strict full eval on:
  - `codecontests_valid`
  - `codecontests_valid_big`
  - optionally `humaneval` / `mbpp_reg`

## Selection Rule

1. Filter out checkpoints that materially regress `retention_canary_codecontests`
2. Filter out checkpoints that materially worsen `timeout_canary_codecontests`
3. Filter out checkpoints that collapse `structural_hard_watchlist`
4. Among remaining checkpoints, select the best `mid_val_codecontests_32`

