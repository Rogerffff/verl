# ValidBig500 Checkpoint Review (2026-04-09)

## Scope

This note compares the current `valid_big500` raw evaluations for:

- `step1300`
- `step1200`
- `step1100`

against the historical baselines:

- `step1000`
- `step900`
- `step600`
- `step400`

The main question is whether the current `1200 -> 1400` continuation is still improving exact solve count on `valid_big500`, or whether it has shifted toward better partial credit without converting that into more ACs.

## Run Selection

### Current protocol-matched runs used

For `step1100/1200/1300`, I use the later `*_rerun1` runs on `vastai3`, because the earlier same-day non-rerun runs drifted materially and should not be mixed with them.

- `step1100`:
[run_info.json](/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step1000_overnight_u5a4b4c3_to1200_lb240_seed0_vastai3_step1100_codecontests_validbig500_raw_patched_rr808x_vastai3_rerun1/run_info.json)
- `step1200`:
[run_info.json](/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step1000_overnight_u5a4b4c3_to1200_lb240_seed0_vastai3_step1200_codecontests_validbig500_raw_patched_rr818x_vastai3_rerun1/run_info.json)
- `step1300`:
[run_info.json](/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3_step1300_codecontests_validbig500_raw_patched_rr828x_vastai3_rerun1/run_info.json)

For `step900/1000`, I use the latest patched-sandbox runs on `2026-04-08`:

- `step900`:
[run_info.json](/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step620_v3q3_to900_seed0_step900_codecontests_validbig500_raw_patched_rr818x_vastai3_retry2/run_info.json)
- `step1000`:
[run_info.json](/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_codecontests_validbig500_raw_patched_rr828x_vastai3/run_info.json)

### Historical baselines with caveat

For `step600/400`, the latest available local archived outputs are:

- `step600`:
[run_info.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/未命名/rl_codecontests_validbig500/grpo_a1_curriculum_step580_to600_seed0_step600_codecontests_validbig500/run_info.json)
- `step400`:
[run_info.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/output/未命名/rl_codecontests_validbig500/grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0_step400_codecontests_validbig500_retry3/run_info.json)

These are useful as historical anchors, but they are older than the explicit patched-sandbox reruns used for `step900/1000/1100/1200/1300`. So the strictest apples-to-apples comparison is among `step900 -> step1300`.

## Headline Table


| checkpoint | eval timestamp        | solved | accepted@1 | pass_ratio_mean | avg_judge_time | timeout_rate |
| ---------- | --------------------- | ------ | ---------- | --------------- | -------------- | ------------ |
| `step400`  | `2026-04-04T03:43:37` | 53     | 0.106      | 0.331626        | archived       | archived     |
| `step600`  | `2026-04-05T18:26:05` | 57     | 0.114      | 0.331616        | archived       | archived     |
| `step900`  | `2026-04-08T11:35:45` | 66     | 0.132      | 0.348440        | 11.48s         | 6.2%         |
| `step1000` | `2026-04-08T11:28:01` | 69     | 0.138      | 0.344402        | 12.60s         | 6.8%         |
| `step1100` | `2026-04-08T23:51:25` | 63     | 0.126      | 0.345914        | 14.87s         | 6.4%         |
| `step1200` | `2026-04-08T23:51:25` | 62     | 0.124      | 0.340210        | 19.45s         | 7.0%         |
| `step1300` | `2026-04-08T23:51:25` | 59     | 0.118      | 0.345778        | 17.43s         | 5.4%         |


Main read:

- `step1000` is still the solve-count winner on `valid_big500`.
- `step900` remains the strongest alternative if we also value slightly better mean partial credit than `step1000`.
- `step1100/1200/1300` do not continue the exact-solve upward trend.
- `step1300` partially recovers from `step1200`, and later fixed-response rejudge shows this recovery is real on pass-ratio quality even though it still trails both `step900` and `step1000` on solved count.

## Pass-Ratio Buckets

Buckets below are computed from `per_problem/codecontests_valid_big.jsonl` using `pass_ratio_all`:

- `zero`: `0`
- `low`: `(0, 0.2)`
- `mid`: `[0.2, 0.6)`
- `high`: `[0.6, 1.0)`
- `full`: `1.0`


| checkpoint | zero | low | mid | high | full |
| ---------- | ---- | --- | --- | ---- | ---- |
| `step400`  | 124  | 135 | 113 | 75   | 53   |
| `step600`  | 123  | 140 | 107 | 73   | 57   |
| `step900`  | 117  | 133 | 115 | 69   | 66   |
| `step1000` | 113  | 141 | 114 | 63   | 69   |
| `step1100` | 112  | 140 | 119 | 66   | 63   |
| `step1200` | 112  | 132 | 133 | 61   | 62   |
| `step1300` | 107  | 135 | 131 | 68   | 59   |


Interpretation:

- `step1000` pushes the most samples into `full`, but it does so with fewer `high` partials and a slightly worse `pass_ratio_mean` than `step900`.
- `step1100` loses `6` full solves vs `step1000`, but those problems mostly fall into `mid/high` rather than all the way to `zero`.
- `step1200` is the weakest late checkpoint: full solves fall again, `mid` spikes to `133`, and judge time becomes the worst.
- `step1300` improves the non-zero distribution relative to `step1200`:
  - `zero` drops from `112 -> 107`
  - `high` rises from `61 -> 68`
  - `pass_ratio_mean` recovers from `0.340210 -> 0.345778`
  but `full` still drops from `62 -> 59`

This is the clearest sign in the full rerun that the continuation is still making some programs more partially correct, but is not converting that into more ACs. The later fixed-response rejudge strengthens the first half of that statement: `step1300` really does look better on partial credit, not just on a lucky single rerun.

## Solve-Set Churn

### `step400 -> step600`

- solve gain: `+17`
- solve loss: `-13`
- net: `+4`

Representative gains:

- `Codeforces/1114/A`
- `Codeforces/1151/A`
- `Codeforces/1170/I`
- `Codeforces/1202/C`
- `Codeforces/204/E`

Representative losses:

- `Codeforces/1154/G`
- `Codeforces/1175/G`
- `Codeforces/371/D`
- `Codeforces/679/C`
- `Codeforces/847/G`

This was still a meaningful improvement stage, but even here solve-set churn was already large.

### `step900 -> step1000`

- solve gain: `+22`
- solve loss: `-19`
- net: `+3`

Representative gains:

- `Codeforces/100/F`
- `Codeforces/1202/C`
- `Codeforces/1250/N`
- `Codeforces/1331/H`

Representative losses:

- `Codeforces/1088/B`
- `Codeforces/1327/D`
- `Codeforces/1424/G`
- `Codeforces/185/D`

Key interpretation:

- `step1000` is the solve-count winner.
- But it wins narrowly, with heavy churn and lower `pass_ratio_mean` than `step900`.
- This is more “trade full solves across families” than “strictly dominates”.

### `step1000 -> step1100`

- solve gain: `+14`
- solve loss: `-20`
- net: `-6`

Representative gains:

- `Codeforces/1088/B`
- `Codeforces/1424/G`
- `Codeforces/362/C`
- `Codeforces/447/A`

Representative losses:

- `Codeforces/100/F`
- `Codeforces/1156/D`
- `Codeforces/1250/N`
- `Codeforces/1438/E`

Key interpretation:

- `step1100` already stops the solve-count climb.
- It does keep decent partial quality, which is why `pass_ratio_mean` stays slightly above `step1000`, but it loses too many exact solves.

### `step1100 -> step1200`

- solve gain: `+12`
- solve loss: `-13`
- net: `-1`

Representative gains:

- `Codeforces/1011/F`
- `Codeforces/1156/D`
- `Codeforces/1250/N`
- `Codeforces/679/C`

Representative losses:

- `Codeforces/1088/B`
- `Codeforces/1424/G`
- `Codeforces/362/C`
- `Codeforces/371/D`

Key interpretation:

- `step1200` is not just slightly worse on solves.
- It is also the slowest eval by judge-time profile and has the worst `pass_ratio_mean` among the late checkpoints.
- This looks like the weakest point in the `1000 -> 1300` continuation.

### `step1200 -> step1300`

- solve gain: `+13`
- solve loss: `-16`
- net: `-3`

Representative gains:

- `Codeforces/1088/B`
- `Codeforces/1170/I`
- `Codeforces/362/C`
- `Codeforces/371/D`

Representative losses:

- `Codeforces/1011/F`
- `Codeforces/1156/D`
- `Codeforces/1250/N`
- `Codeforces/616/A`

Key interpretation:

- `step1300` clearly recovers from the `step1200` dip in partial credit.
- But it still loses more exact solves than it gains.
- So `step1300` is not a new exact-solve winner, but it does emerge as the strongest late partial-credit checkpoint once fixed-response rejudge is taken into account.

### `step1000 -> step1300`

- solve gain: `+12`
- solve loss: `-22`
- net: `-10`
- improved problems: `139`
- regressed problems: `106`

Representative gains:

- `Codeforces/1088/B`
- `Codeforces/1253/A`
- `Codeforces/242/A`
- `Codeforces/447/A`
- `Codeforces/591/B`

Representative losses:

- `Codeforces/100/F`
- `Codeforces/1033/A`
- `Codeforces/1156/D`
- `Codeforces/1250/N`
- `Codeforces/1438/E`
- `Codeforces/802/M`

This is the most important comparison. It says:

- `step1300` still changes many problems in a positive direction.
- But the positive drift is mostly into `mid/high` partial territory rather than all the way to `full`.
- On `valid_big500`, the continuation from `1000 -> 1300` is therefore not improving the main win criterion.

## Log Review

The selected runs are log-clean.

Representative launch logs:

- `step900`:
[launch.log](/workspace/eval_logs/grpo_a1_curriculum_step620_v3q3_to900_seed0_step900_codecontests_validbig500_raw_patched_rr818x_vastai3_retry2.launch.log)
- `step1000`:
[launch.log](/workspace/eval_logs/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_codecontests_validbig500_raw_patched_rr828x_vastai3.launch.log)
- `step1100`:
[launch.log](/workspace/eval_logs/grpo_a1_curriculum_step1000_overnight_u5a4b4c3_to1200_lb240_seed0_vastai3_step1100_codecontests_validbig500_raw_patched_rr808x_vastai3_rerun1.launch.log)
- `step1200`:
[launch.log](/workspace/eval_logs/grpo_a1_curriculum_step1000_overnight_u5a4b4c3_to1200_lb240_seed0_vastai3_step1200_codecontests_validbig500_raw_patched_rr818x_vastai3_rerun1.launch.log)
- `step1300`:
[launch.log](/workspace/eval_logs/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3_step1300_codecontests_validbig500_raw_patched_rr828x_vastai3_rerun1.launch.log)

Observed behavior:

- All selected runs reached `DONE_EVAL`.
- `api_error_rate = 0`
- `sandbox_error_rate = 0`
- no traceback-level failure in eval logs
- only non-fatal merge/vLLM warnings:
  - tokenizer/processor warnings
  - `TORCH_CUDA_ARCH_LIST` warning

Runtime differences are real:

- `step900`: about `5m30s`
- `step1000`: about `5m51s`
- `step1100`: about `7m02s`
- `step1200`: about `7m38s`
- `step1300`: about `7m03s`

This matches the judge-time rise in the summaries. The later checkpoints are not just weaker in solve count; they are also more expensive to evaluate.

## Important Protocol Note

The earlier same-day non-rerun runs for `step1100/1200/1300` differ from the later `rerun1` scores.

Most important drift:

- `step1100`: `0.348484 -> 0.345914`
- `step1300`: `0.357553 -> 0.345778`
- `step1200`: roughly stable

So for any serious checkpoint comparison, the `rerun1` set should be treated as the official current set.

This means:

- the patched sandbox improved reliability
- but evaluation variance is not fully eliminated
- winner selection on exact solves should continue to rely on consistent rerun groups, not single ad hoc runs

## Fixed-Response Rejudge

To separate judge drift from generation drift, the fixed-response rejudge sweep was also run for `step1100/1200/1300` using the same saved responses and only re-running the verifier.

Report:

- `/tmp/fixed_response_rejudge_report.json`

Outputs:

- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1100_rejudge1`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1100_rejudge2`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1200_rejudge1`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1200_rejudge2`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1300_rejudge1`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1300_rejudge2`

Judge-only drift is real, but clearly smaller than the earlier full-rerun drift:

- `source -> rejudge1` accepted mismatches:
  - `step1100 = 4`
  - `step1200 = 4`
  - `step1300 = 2`
- `rejudge1 -> rejudge2` accepted mismatches:
  - `step1100 = 2`
  - `step1200 = 6`
  - `step1300 = 3`

This is much smaller than the previously observed full-rerun response drift:

- `step1100 response_diff_count = 84`
- `step1200 response_diff_count = 111`
- `step1300 response_diff_count = 134`

So the best current decomposition is:

- the dominant source of full-run inconsistency is still generation drift
- judge drift remains, but it is not the main driver

Most importantly, `step1300`'s high `pass_ratio_mean` survives fixed-response rejudge almost unchanged:

- source first run: `0.357553`
- `rejudge1`: `0.357475`
- `rejudge2`: `0.357354`

For comparison:

- `step1100`: `0.348049 / 0.347537`
- `step1200`: `0.339792 / 0.339607`

This is strong evidence that `step1300` is not the exact-solve winner, but it is the current strongest partial-credit / pass-ratio checkpoint among the late checkpoints.

## Final Judgment

1. `step1000` is still the `valid_big500` exact-solve winner.
  It has the highest exact solve count: `69/500`, and the gap over `step1300` is much larger than the observed judge-only drift.
2. `step900` remains the strongest alternative exact-solve anchor.
  It trails `step1000` by only `3` solves and keeps slightly better full-run `pass_ratio_mean`.
3. `step1100/1200/1300` do not justify replacing `step1000` as the solve-count winner.
  The trend after `step1000` is not “exact solves keep improving”.
4. `step1300` should be reclassified.
  It is not the exact-solve winner, but fixed-response rejudge now supports calling it the strongest current partial-credit / pass-ratio checkpoint.
5. The continuation after `step1000` appears to be optimizing partial correctness more than exact completion.
  That is the best single explanation for:
  - `full` bucket falling in full reruns
  - `mid/high` buckets staying healthy or recovering
  - `step1300` keeping the best fixed-response `pass_ratio_mean`

## Recommendation

### For checkpoint selection

- Keep `step1000` as the main exact-solve `valid_big500` checkpoint winner.
- Keep `step900` as the main comparison anchor.
- Do not promote `step1300` as the main exact-solve winner based on `valid_big500`.
- Do treat `step1300` as the best current checkpoint for partial-credit analysis, near-miss mining, and repair-oriented data collection.

### For training diagnosis

The `1000 -> 1300` continuation should be treated as a warning sign that the current regime may be trading exactness for partial reward. The most likely next questions are:

- whether `limiter_budget=240` and the current curriculum mix are amplifying noisy partial-reward updates
- whether the continuation is drifting toward “more partially correct outputs” without enough pressure to preserve already-solved families

### For future eval practice

- Continue using patched-sandbox rerun groups.
- When checkpoint gaps are small, prefer repeated reruns before changing the exact-solve winner.
- For late-checkpoint comparisons, report both:
  - full-run exact solves
  - fixed-response rejudge `pass_ratio_mean`
- If `step1400` is evaluated, compare it directly against the same official set:
  - `step900 patched retry2`
  - `step1000 patched`
  - `step1100/1200/1300 patched rerun1`
