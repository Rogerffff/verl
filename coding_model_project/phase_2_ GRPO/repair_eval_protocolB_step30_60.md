# Protocol B Repair Eval: step30/40/50/60

## Scope

This note summarizes end-to-end Protocol B repair evaluation for:

- `step30`
- `step40`
- `step50`
- `step60`

Protocol B means:

1. merge the SFT checkpoint to HF
2. run that checkpoint's own raw `valid_big500`
3. reuse that raw `per_problem` as the first-pass source
4. run one-turn repair with `short_diagnosis_code`

Judge protocol:

- patched sandbox
- direct-backend client RR over `8081..8088`

Output roots:

- raw: [/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_protocolB](/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_protocolB)
- repair: [/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolB](/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolB)

Logs:

- [/workspace/eval_logs/run_protocolB_step40.log](/workspace/eval_logs/run_protocolB_step40.log)
- [/workspace/eval_logs/run_protocolB_extra_steps.log](/workspace/eval_logs/run_protocolB_extra_steps.log)

## Summary Table

| ckpt | raw accepted@1 | raw pass_ratio_mean | after-repair accepted@1 | after-repair pass_ratio_mean | repair gain | repair successes | conditional repair success |
|---|---:|---:|---:|---:|---:|---:|---:|
| step30 | 0.132 | 0.347269 | 0.150 | 0.330056 | 0.018 | 9 | 0.04787 |
| step40 | 0.132 | 0.349763 | 0.156 | 0.336154 | 0.024 | 12 | 0.06349 |
| step50 | 0.134 | 0.352640 | 0.154 | 0.338582 | 0.020 | 10 | 0.05181 |
| step60 | 0.134 | 0.348789 | 0.152 | 0.337546 | 0.018 | 9 | 0.04787 |

## Main Takeaways

### Best checkpoint by final solved count

- `step40` is the best end-to-end repair checkpoint
- it reaches `0.156` accepted@1 (`78/500`)
- it also has the largest repair gain (`+0.024`) and the most repair successes (`12`)

### Best checkpoint by pass-ratio quality

- `step50` has the highest:
  - raw `pass_ratio_mean`
  - final `pass_ratio_mean`
- but it still trails `step40` on final solved count

### Ranking

If the objective is repair benchmark headline performance:

1. `step40`
2. `step50`
3. `step60`
4. `step30`

## Comparison vs Protocol A

Protocol A used a fixed `step900` canonical first-pass for all checkpoints.

### Final accepted@1

| ckpt | Protocol A | Protocol B |
|---|---:|---:|
| step30 | 0.148 | 0.150 |
| step40 | 0.148 | 0.156 |
| step50 | 0.148 | 0.154 |
| step60 | 0.148 | 0.152 |

Interpretation:

- all checkpoints benefit from using their own raw first-pass instead of the fixed `step900` source
- `step40` benefits the most

### Repair gain

| ckpt | Protocol A gain | Protocol B gain |
|---|---:|---:|
| step30 | 0.020 | 0.018 |
| step40 | 0.020 | 0.024 |
| step50 | 0.020 | 0.020 |
| step60 | 0.020 | 0.018 |

Interpretation:

- `step40` is the only checkpoint that clearly improves both:
  - first-pass quality
  - repair conversion itself
- `step30` and `step60` get slightly better end-to-end results mainly because their own raw first-pass is stronger than the fixed `step900` source, not because the repair stage itself improves

## Step40 Special Note

`step40` raw-to-repair transition is especially clean:

- raw `66/500`
- repaired `78/500`
- `12` fail-to-AC gains
- `0` AC-to-fail losses

Representative repaired gains include:

- `Codeforces/1088/B`
- `Codeforces/1147/B`
- `Codeforces/1165/B`
- `Codeforces/1183/F`
- `Codeforces/1331/H`
- `Codeforces/172/B`
- `Codeforces/242/A`
- `Codeforces/447/A`
- `Codeforces/519/A`
- `Codeforces/57/B`
- `Codeforces/616/A`
- `Codeforces/981/G`

## Final Recommendation

Use `step40` as the primary repair base for any follow-up work.

Why:

- best final accepted@1 under Protocol B
- best repair gain under Protocol B
- best repair success count under Protocol B
- also already the preferred checkpoint under the earlier Protocol A comparison

`step50` is still useful as a secondary checkpoint if you want a higher-pass-ratio alternate view, but it is not the best headline repair model.
