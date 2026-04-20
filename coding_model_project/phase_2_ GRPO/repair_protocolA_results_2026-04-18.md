# Repair Protocol A Results

## Scope

This note summarizes the completed fixed-input repair runs under the current primary repair-SFT evaluation contract:

- Protocol A
- `prompt_mode = short_diagnosis_code`
- `repair_error_types = [wrong_answer, runtime_error, timeout]`
- patched sandbox
- direct-backend client RR

For `valid_big500`, the current comparable runs use:

- `repair_min_pass_ratio = 0.2`
- canonical reused first pass from `step900`

For `codecontests_test`, the current completed Protocol A runs also use:

- `repair_min_pass_ratio = 0.2`
- canonical reused first pass from `step900`

## Sources

### ValidBig500 Protocol A

- `step900`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolA_full/step900_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5`
- `step40`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolA_full/step40_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5`
- `step1300`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500_repair_protocolA_full/step1300_validbig500_repair_protocolA_full_shortdiag_p02_reuse_step900_rr808x_vastai5`

### CodeContests Test Protocol A

- `step900`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_test_repair_protocolA/step900_codecontests_test_repair_protocolA_shortdiag_p02_reuse_step900_rr808x_vastai5`
- `step40`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_test_repair_protocolA/step40_codecontests_test_repair_protocolA_shortdiag_p02_reuse_step900_rr808x_vastai5`
- `step1300`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_test_repair_protocolA/step1300_codecontests_test_repair_protocolA_shortdiag_p02_reuse_step900_rr808x_vastai5`
- `step1300_sft_v1_step60`
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_test_repair_protocolA_step1300sftv1/step1300sftv1_step60_codecontests_test_repair_protocolA_shortdiag_p02_reuse_step900_rr808x_vastai5`

## Results

### ValidBig500

Canonical first pass is identical across all three models:

- `first_pass accepted@1 = 0.128`
- `first_pass pass_ratio_mean = 0.3472`


| model      | final accepted@1 | repair gain | final pass_ratio_mean | attempts | conditional repair success |
| ---------- | ---------------- | ----------- | --------------------- | -------- | -------------------------- |
| `step900`  | `0.146`          | `+0.018`    | `0.3300`              | `193`    | `0.0466`                   |
| `step40`   | `0.148`          | `+0.020`    | `0.3346`              | `193`    | `0.0518`                   |
| `step1300` | `0.146`          | `+0.018`    | `0.3285`              | `193`    | `0.0466`                   |


Bucketed conditional repair success:

- `step900`
  - `bucket_0.6_1.0 = 0.1000`
  - `bucket_0.2_0.6 = 0.0163`
- `step40`
  - `bucket_0.6_1.0 = 0.1143`
  - `bucket_0.2_0.6 = 0.0163`
- `step1300`
  - `bucket_0.6_1.0 = 0.1143`
  - `bucket_0.2_0.6 = 0.0081`

### CodeContests Test

Canonical first pass is identical across all three models:

- `first_pass accepted@1 = 0.0727`
- `first_pass pass_ratio_mean = 0.2308`


| model      | final accepted@1 | repair gain | final pass_ratio_mean | attempts | conditional repair success |
| ---------- | ---------------- | ----------- | --------------------- | -------- | -------------------------- |
| `step900`  | `0.0727`         | `+0.0000`   | `0.2144`              | `38`     | `0.0000`                   |
| `step40`   | `0.0727`         | `+0.0000`   | `0.2145`              | `38`     | `0.0000`                   |
| `step1300` | `0.0727`         | `+0.0000`   | `0.2260`              | `38`     | `0.0000`                   |


All three models produced:

- `repair_gain = 0`
- `conditional_repair_success = 0`

### CodeContests Test Follow-up: `step1300_sft_v1_step60`

This follow-up uses the same fixed-input Protocol A contract:

- `reuse_step900`
- `repair_min_pass_ratio = 0.2`
- `repair_max_failure_cases = 1`

| model | first-pass acc@1 | final acc@1 | repair gain | final pass_ratio_mean | attempts | conditional repair success |
|---|---:|---:|---:|---:|---:|---:|
| `step1300_rl` | `0.0727` | `0.0727` | `+0.0000` | `0.2260` | `38` | `0.0000` |
| `step1300_sft_v1_step60` | `0.0727` | `0.0727` | `+0.0000` | `0.2206` | `38` | `0.0000` |

Interpretation:

- `step1300_sft_v1_step60` does not improve held-out fixed-input repair under the current narrow Protocol A test contract.
- Relative to `step1300_rl`, it matches solved count but lands slightly lower on final `pass_ratio_mean`.

## Interpretation

### Main takeaways

1. `step40` is the best completed Protocol A result on `valid_big500`, but only slightly.
2. `step1300` does not beat `step900` on fixed-input repair for `valid_big500`.
3. On `codecontests_test`, this Protocol A setting currently shows no successful repairs for any of the three models.
4. The newer `step1300_sft_v1_step60` follow-up also shows no held-out Protocol A gain.

### What this means

Under fixed-input repair evaluation:

- the current repair-SFT line shows a small but real gain on the development-side `valid_big500` slice
- the same gain does not yet transfer to held-out `codecontests_test` under the current trigger and prompt contract
- this remains true even for the newer `step1300`-base repair-SFT v1 checkpoint `step60`

### Caution

The completed `codecontests_test` Protocol A runs use a relatively narrow trigger:

- `repair_min_pass_ratio = 0.2`

That means this table answers:

- whether the models can repair the subset of test failures selected by the current `p>=0.2` gate

It does not answer:

- whether one-turn repair helps on the full test failure pool under a broader self-first-pass deployment protocol

## Current Reading

The fixed-input conclusion is now stable:

- `Protocol A` is still useful for fair, frozen-input repair comparison.
- But on held-out `codecontests_test`, the current trigger remains too narrow to show meaningful solved gains.
- The new `step1300_sft_v1_step60` follow-up does not change that conclusion.

So the practical read is:

- use `Protocol A` mainly to compare repair skill on development-side slices such as `valid_big500`
- do not use current `codecontests_test / Protocol A` alone to argue that repair-SFT has generalized
