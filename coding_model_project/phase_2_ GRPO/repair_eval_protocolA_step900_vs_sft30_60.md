# Protocol A Repair Eval: step900 vs SFT step30/40/50/60

## Scope

This note summarizes the fixed-input one-turn repair evaluation run completed on `vastai5` for:

- `step900` repair baseline
- SFT checkpoints `step30 / step40 / step50 / step60`

Evaluation protocol:

- fixed first-pass input: `step900` canonical raw `per_problem`
- dataset: `codecontests_valid_big` (`500` problems)
- prompt mode: `short_diagnosis_code`
- trigger:
  - `repair_min_pass_ratio = 0.2`
  - `repair_error_types = wrong_answer runtime_error timeout`
- sandbox:
  - patched sandbox
  - direct-backend client RR over `8081..8088`

Reference first-pass source:

- [/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step620_v3q3_to900_seed0_step900_codecontests_validbig500_raw_patched_rr828x_vastai3/per_problem](/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step620_v3q3_to900_seed0_step900_codecontests_validbig500_raw_patched_rr828x_vastai3/per_problem)

Output root:

- [/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolA_full](/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolA_full)

Run chain logs:

- [/workspace/eval_logs/run_protocolA_full_repair_eval_step900_sft.log](/workspace/eval_logs/run_protocolA_full_repair_eval_step900_sft.log)
- [/workspace/eval_logs/run_protocolA_full_repair_eval_sft_only.log](/workspace/eval_logs/run_protocolA_full_repair_eval_sft_only.log)

## Main Metrics

All runs share the same fixed first-pass:

- `first_pass accepted@1 = 0.128`
- `first_pass pass_ratio_mean = 0.3472177471`
- `repair_attempt_count = 193`
- trigger buckets:
  - `bucket_0.2_0.6 = 123`
  - `bucket_0.6_1.0 = 70`

### Summary Table

| ckpt | final accepted@1 | repair gain | repair successes | conditional success | final pass_ratio_mean |
|---|---:|---:|---:|---:|---:|
| step900 | 0.146 | 0.018 | 9 | 0.04663 | 0.329985 |
| step30 | 0.148 | 0.020 | 10 | 0.05181 | 0.331393 |
| step40 | 0.148 | 0.020 | 10 | 0.05181 | 0.334616 |
| step50 | 0.148 | 0.020 | 10 | 0.05181 | 0.333373 |
| step60 | 0.148 | 0.020 | 10 | 0.05181 | 0.334463 |

### Bucketed Repair Success

| ckpt | bucket 0.2-0.6 | bucket 0.6-1.0 |
|---|---:|---:|
| step900 | 0.01626 | 0.10000 |
| step30 | 0.01626 | 0.11429 |
| step40 | 0.01626 | 0.11429 |
| step50 | 0.01626 | 0.11429 |
| step60 | 0.01626 | 0.11429 |

Interpretation:

- the SFT line does **not** improve the medium bucket (`0.2-0.6`)
- the gain comes entirely from the high near-miss bucket (`0.6-1.0`)

### Repair Cost per Success

| ckpt | extra gen tokens / success | extra judge time / success |
|---|---:|---:|
| step900 | 6581.22 | 146.73 |
| step30 | 5992.80 | 154.57 |
| step40 | 5987.70 | 141.87 |
| step50 | 5965.30 | 136.40 |
| step60 | 5956.40 | 139.23 |

Interpretation:

- SFT checkpoints are modestly more token-efficient than `step900`
- `step50` is the cheapest by judge time
- `step40` and `step60` are the best by final `pass_ratio_mean`

## Per-Problem Delta vs step900

Accepted-set comparison between each SFT checkpoint and the `step900` repair baseline is identical.

Every SFT checkpoint (`30/40/50/60`) shows:

- `2` gains vs `step900`
- `1` loss vs `step900`

Gains:

- `Codeforces/1147/B`: `0.88 -> 1.0`
- `Codeforces/981/G`: `0.4651 -> 1.0`

Loss:

- `Codeforces/493/A`: `1.0 -> 0.56`

So the net solve difference is:

- `+1 solved` relative to `step900`

Also important:

- accepted-set mismatch across `step30 / step40 / step50 / step60` is exactly `0`
- they solve the same final set of problems
- the only differences across these four checkpoints are in non-AC pass ratios and repair cost

## Recommendation

If the objective is the strongest repair checkpoint under Protocol A fixed-input evaluation:

- choose `step40` as the primary repair base

Why:

- it ties the best accepted@1 (`0.148`)
- it ties the best repair gain (`+0.020`)
- it matches the best conditional repair success (`0.05181`)
- it has the highest final `pass_ratio_mean` among all evaluated SFT checkpoints (`0.334616`)

If the objective is the cheapest checkpoint among the tied SFT options:

- `step50` is slightly better on extra repair cost

## Overall Conclusion

The `short_diagnosis_code` SFT line produced a **real but small** repair gain over the `step900` baseline under fixed-input evaluation:

- `73 solved -> 74 solved`
- `9 repair successes -> 10 repair successes`
- improvement is concentrated in the `0.6-1.0` near-miss bucket
- no evidence that SFT meaningfully expands repair coverage into lower-pass-ratio failures

This supports a narrow conclusion:

- the SFT run improved repair reliability on already-near-correct code
- it did **not** fundamentally broaden the repair horizon

For follow-up evaluation:

1. use `step40` as the selected SFT checkpoint
2. run end-to-end repair evaluation for `step40`:
   - first generate its own raw valid_big500 outputs
   - then reuse that raw `per_problem` for repair
3. if time is limited, skip `step30/50/60` for any further end-to-end work
