# Repair Protocol B Results

## Scope

This note summarizes the completed self-first-pass repair runs under the current patched environment:

- patched sandbox
- direct-backend client RR
- `short_diagnosis_code`

Protocol B means:

- each model uses its own first-pass raw output
- then repairs its own code

This is the practical end-to-end deployment metric, not the pure fixed-input repair metric.

## Sources

### ValidBig500

- `step900`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolB/step900_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5`
- `step40`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolB/step40_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5`
- `step1300`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolB/step1300_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5`

### CodeContests Test

- `step900`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_test_repair/step900_codecontests_test_repair_reuse_selfraw_rr808x_vastai5`
- `step40`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_test_repair/step40_codecontests_test_repair_reuse_selfraw_rr808x_vastai5`
- `step1300`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_test_repair/step1300_codecontests_test_repair_reuse_selfraw_rr808x_vastai5`

## Results

### ValidBig500

| model | first-pass acc@1 | final acc@1 | repair gain | first-pass prm | final prm | attempts | conditional repair success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `step900` | `0.128` | `0.142` | `+0.014` | `0.3472` | `0.3284` | `193` | `0.0363` |
| `step40` | `0.132` | `0.156` | `+0.024` | `0.3498` | `0.3362` | `189` | `0.0635` |
| `step1300` | `0.118` | `0.130` | `+0.012` | `0.3458` | `0.3285` | `199` | `0.0302` |

Bucketed conditional repair success:

- `step900`
  - `bucket_0.6_1.0 = 0.0857`
  - `bucket_0.2_0.6 = 0.0081`
- `step40`
  - `bucket_0.6_1.0 = 0.1286`
  - `bucket_0.2_0.6 = 0.0252`
- `step1300`
  - `bucket_0.6_1.0 = 0.0441`
  - `bucket_0.2_0.6 = 0.0229`

### CodeContests Test

| model | first-pass acc@1 | final acc@1 | repair gain | first-pass prm | final prm | attempts | conditional repair success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `step900` | `0.0727` | `0.0788` | `+0.0061` | `0.2308` | `0.2426` | `151` | `0.0066` |
| `step40` | `0.0667` | `0.0788` | `+0.0121` | `0.2280` | `0.2463` | `152` | `0.0132` |
| `step1300` | `0.0788` | `0.1091` | `+0.0303` | `0.2501` | `0.2879` | `150` | `0.0333` |

Bucketed conditional repair success:

- `step900`
  - `bucket_0.6_1.0 = 0.0556`
- `step40`
  - `bucket_0 = 0.0167`
  - `bucket_0_0.2 = 0.0196`
- `step1300`
  - `bucket_0.6_1.0 = 0.1053`
  - `bucket_0 = 0.0492`

## Comparison With Protocol A

### ValidBig500

- Protocol A fixed-input winner is still `step40`, but only slightly.
- Protocol B also prefers `step40`, and the margin is clearer:
  - `step40 final acc@1 = 15.6%`
  - `step900 final acc@1 = 14.2%`
  - `step1300 final acc@1 = 13.0%`

This is the strongest evidence that the current repair-SFT line did learn some repair skill on the development-side slice.

### CodeContests Test

- Protocol A on the narrow `p>=0.2` fixed-input test slice produced no solved gains for any model.
- Protocol B on the broader self-first-pass test setting does show gains:
  - `step900`: `+1` solved
  - `step40`: `+2` solved
  - `step1300`: `+5` solved

This means the held-out test story is:

- the repair line is not yet strong enough to dominate fixed-input repair on the current narrow Protocol A test slice
- but end-to-end self-repair does help on the full practical test pipeline
- and the strongest overall deployed model is still `step1300`

## Important Caveat

`codecontests_test` Protocol A and Protocol B are not using the same trigger:

- Protocol A test runs here use `repair_min_pass_ratio = 0.2`, `repair_max_failure_cases = 1`
- Protocol B test runs here use `repair_min_pass_ratio = 0.0`, `repair_max_failure_cases = 3`

So the difference between “Protocol A no gains” and “Protocol B some gains” is partly:

- a skill difference
- and partly a trigger / prompt-budget difference

This does not invalidate the result, but it matters for interpretation.

## Current `step1300_sft_v1` Follow-up

These newer runs use the current `step1300`-base repair-SFT v1 lineage:

- `valid_big500`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolB_step1300sftv1/step1300sftv1_step20_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolB_step1300sftv1/step1300sftv1_step40_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolB_step1300sftv1/step1300sftv1_step60_validbig500_repair_protocolB_shortdiag_p02_selfraw_rr808x_vastai5`
- `codecontests_test`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_test_repair_step1300sftv1/step1300sftv1_step60_codecontests_test_repair_reuse_selfraw_rr808x_vastai5`

### ValidBig500 Follow-up

| model | first-pass acc@1 | final acc@1 | repair gain | first-pass prm | final prm | attempts | conditional repair success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `step1300_rl` | `0.118` | `0.130` | `+0.012` | `0.3458` | `0.3285` | `199` | `0.0302` |
| `step1300_sft_v1_step20` | `0.126` | `0.142` | `+0.016` | `0.3526` | `0.3435` | `197` | `0.0406` |
| `step1300_sft_v1_step40` | `0.126` | `0.142` | `+0.016` | `0.3595` | `0.3430` | `201` | `0.0398` |
| `step1300_sft_v1_step60` | `0.126` | `0.144` | `+0.018` | `0.3531` | `0.3386` | `199` | `0.0452` |

Interpretation:

- The current `step1300`-base repair-SFT v1 line does improve end-to-end repair on the development slice.
- `step1300_sft_v1_step60` is the best current-lineage Protocol B checkpoint on `valid_big500`.

### CodeContests Test Follow-up

| model | first-pass acc@1 | final acc@1 | repair gain | first-pass prm | final prm | attempts | conditional repair success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `step1300_rl` | `0.0788` | `0.1091` | `+0.0303` | `0.2501` | `0.2879` | `150` | `0.0333` |
| `step1300_sft_v1_step60` | `0.0788` | `0.0970` | `+0.0182` | `0.2600` | `0.2761` | `151` | `0.0199` |

Interpretation:

- `step1300_sft_v1_step60` does have held-out self-repair gains.
- But it does **not** beat `step1300_rl` on `codecontests_test`.
- So `step60` is best read as the strongest v1 repair-SFT checkpoint for comparison, not as a replacement deployed model.

## Recommendation

### Short answer

Yes to a larger `step1300 repair-SFT v2`, but only as a **changed recipe**.

No to a naive “just scale up v1” continuation.

### Why

The current evidence now shows two things at once:

1. a real fixed-input gain on `valid_big500`
2. a real end-to-end gain on `valid_big500`
3. non-zero self-repair gains on `codecontests_test`

But it also clearly shows a ceiling:

1. the newer `step1300_sft_v1_step60` still does not beat `step1300_rl` as the held-out deployed test model
2. fixed-input `codecontests_test` repair is still weak under the narrow Protocol A trigger
3. the current v1 data mix is still narrow enough that the gains look specialized rather than robust

### What to change in the next repair-SFT round

If the goal is stronger real-world repair ability, the next round should not just be “more of the same” for either the old `step900` line or the current `step1300 shortdiag pure v1` line.

Recommended changes:

- expand the repair-SFT corpus substantially
  - target at least `3x` to `5x` the current accepted keep set
- rebalance the corpus
  - reduce the dominance of hard `C`-style partial repairs
  - include more `Core / ExpansionB` style repairs
- keep some anchor / general-code rows if the run becomes materially larger
  - avoid turning a stronger `step1300` base into a too-narrow repair specialist
- keep high-precision QC
  - patched sandbox
  - direct-backend RR
  - explicit rejudge / double-check on kept teacher outputs
- keep `short_diagnosis_code` as the main format
  - current evidence is already enough to prioritize data recipe and scale over more prompt-format branching

### Base checkpoint choice

If the goal is the best deployed repair model, the next repair-SFT round should continue training on top of a stronger base such as `step1300`, not revert to an older weaker base.

Current evidence:

- `step1300_sft_v1_step60` adds repair skill on the dev side
- `step1300_rl` keeps the strongest held-out end-to-end test performance

So the most promising next-line hypothesis is:

- preserve the stronger `step1300` first-pass ability
- add a more balanced repair-SFT recipe on top of that stronger base

### Practical next move

The highest-ROI next move is:

1. build a larger repair-conditioned SFT dataset
2. train a new repair-SFT model from the stronger `step1300` base with a more balanced recipe
3. re-run both Protocol A and Protocol B

If time is limited, that is still a better use of effort than more tiny prompt ablations, but the key is **recipe change**, not just more steps on the current v1 mix.
