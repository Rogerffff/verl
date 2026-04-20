# A1 Formal RL Run Metrics Summary

## Scope

This document summarizes the completed formal observation run:

- Experiment: `grpo_a1_formal_observe_lb24_multi2_resume10_seed0`
- Algorithm: `A1 + anchored_dense_v1 + guardrails + filter_groups=false`
- Resume source: `global_step_10` from the final readiness checkpoint

Primary evidence sources:

- Old-machine training log mirror: `/workspace/eval_logs/old_machine/grpo_a1_formal_observe_lb24_multi2_resume10_seed0.log`
- New-machine full eval summaries:
  - `/workspace/eval_logs/grpo_a1_eval_step60_combined_summary.json`
  - `/workspace/eval_logs/grpo_a1_eval_step80_combined_summary.json`
  - `/workspace/eval_logs/grpo_a1_eval_step100_combined_summary.json`
- Phase-0 baseline:
  - `/workspace/verl/coding_model_project/outputs/phase0_fullval_20260331/summary.json`
  - `/workspace/verl/coding_model_project/outputs/phase0_fullval_20260331/metrics.json`

## Run Context

Effective training setup for this run:

- `train_batch_size=8`
- `ppo_mini_batch_size=8`
- `ppo_micro_batch_size_per_gpu=4`
- `rollout.n=8`
- `actor.optim.lr=1e-6`
- `ppo_epochs=1`
- `run_timeout_s=30`
- `limiter_budget=24`
- `sandbox_endpoint=http://localhost:8090`
- `2` sandbox backends behind nginx LB
- `fast_val_codecontests.parquet` for online validation
- `test_freq=10`
- `save_freq=20`
- `max_actor_ckpt_to_keep=1`
- `max_critic_ckpt_to_keep=1`

The launcher scripts used for this configuration are:

- [run_grpo_a1_formal_observation_resume10.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_grpo_a1_formal_observation_resume10.sh)
- [run_grpo_formal.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/run_grpo_formal.sh)

## Training Dynamics

### High-level outcome

- The run completed successfully through `step100`.
- `update_actor` stayed stable throughout the run.
- The main systems bottleneck remained `reward` wall time and its long-tail variance.
- `invalid_for_rl` and `truncated_by_max_tokens` stayed low after the early unstable period.
- Checkpoint retention worked as intended; only the latest full checkpoint was preserved.

### Windowed training summary

10-step rolling windows from the training log:

| Window | reward mean (s) | reward median (s) | update_actor mean (s) | step mean (s) | reward_valid_rate | invalid_for_rl | trunc | timeout | accepted_rate | pass_ratio_all | grad_norm | entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11-20 | 235.63 | 271.51 | 43.94 | 301.65 | 0.9922 | 0.0078 | 0.0078 | 0.0516 | 0.0906 | 0.2360 | 0.7398 | 0.1680 |
| 21-30 | 111.63 | 92.06 | 43.27 | 177.51 | 0.9703 | 0.0297 | 0.0172 | 0.0281 | 0.0641 | 0.2036 | 0.7213 | 0.1694 |
| 31-40 | 150.89 | 65.96 | 42.77 | 216.13 | 0.9984 | 0.0016 | 0.0016 | 0.0297 | 0.1047 | 0.3080 | 0.7948 | 0.1608 |
| 41-50 | 151.91 | 117.28 | 43.24 | 217.58 | 0.9938 | 0.0063 | 0.0063 | 0.0391 | 0.1297 | 0.2831 | 0.8267 | 0.1467 |
| 51-60 | 131.96 | 86.29 | 43.43 | 198.26 | 0.9969 | 0.0031 | 0.0031 | 0.0438 | 0.1438 | 0.3538 | 0.8221 | 0.1261 |
| 61-70 | 120.95 | 38.41 | 44.06 | 185.77 | 0.9984 | 0.0016 | 0.0016 | 0.0266 | 0.1797 | 0.3396 | 0.8848 | 0.1221 |
| 71-80 | 147.95 | 104.29 | 43.68 | 213.45 | 0.9984 | 0.0016 | 0.0016 | 0.0344 | 0.1438 | 0.3160 | 0.8590 | 0.1242 |
| 81-90 | 165.83 | 123.57 | 43.30 | 230.21 | 0.9953 | 0.0047 | 0.0047 | 0.0375 | 0.1672 | 0.3243 | 0.9347 | 0.1212 |
| 91-100 | 181.52 | 193.36 | 43.78 | 247.70 | 0.9953 | 0.0047 | 0.0047 | 0.0453 | 0.1406 | 0.3179 | 0.8388 | 0.1134 |

### Training interpretation

- `update_actor` was stable for the entire run at roughly `43-44s`.
- `reward` variance never fully disappeared; late windows still had heavy tails.
- The training signal was real:
  - `reward_raw_valid_rate` stayed high
  - `all_invalid_group_rate` did not become a persistent problem
  - `accepted_rate` and `pass_ratio_all_mean` improved over the early windows
- Entropy trended down from roughly `0.168` into the `0.11-0.12` range, which is consistent with policy sharpening rather than immediate collapse.

## Checkpoints

Observed checkpoint behavior:

- `global_step_20`: pruned to a tiny stub
- `global_step_40`: pruned to a tiny stub
- `global_step_60`: pruned to a tiny stub
- `global_step_80`: pruned to a tiny stub
- `global_step_100`: retained as the final full checkpoint

Approximate full checkpoint size:

- `global_step_100`: about `86G`

Operational conclusion:

- `max_actor_ckpt_to_keep=1` and `max_critic_ckpt_to_keep=1` worked correctly.
- The disk strategy was just enough for this run, but future long runs still need careful space management.

## Full Eval Comparison

### Dataset-level comparison

| Dataset | Metric | Baseline | Step60 | Step80 | Step100 |
|---|---|---:|---:|---:|---:|
| `codecontests_valid` | accepted | 0.0256 | 0.0256 | 0.0000 | 0.0171 |
| `codecontests_valid` | pass_ratio_all | 0.1422 | 0.1252 | 0.1442 | 0.1419 |
| `codecontests_valid` | invalid_for_rl | 0.0000 | 0.0171 | 0.0171 | 0.0085 |
| `codecontests_valid` | truncated | 0.0000 | 0.0171 | 0.0171 | 0.0085 |
| `codecontests_valid` | judge_time_s | 143.36 | 22.62 | 19.19 | 28.32 |
| `humaneval` | accepted | 0.8720 | 0.8415 | 0.8415 | 0.8476 |
| `humaneval` | pass_ratio_all | 0.8720 | 0.8415 | 0.8415 | 0.8476 |
| `humaneval` | judge_time_s | 0.1923 | 0.1087 | 0.1791 | 0.1724 |
| `mbpp_reg` | accepted | 0.5850 | 0.6300 | 0.6300 | 0.6300 |
| `mbpp_reg` | pass_ratio_all | 0.5850 | 0.6300 | 0.6300 | 0.6300 |
| `mbpp_reg` | judge_time_s | 0.2126 | 0.1954 | 0.1464 | 0.1070 |

### Error distribution shifts

#### CodeContests

Baseline:

- `success=3`
- `wrong_answer=81`
- `runtime_error=24`
- `timeout=8`
- `syntax_error=1`

Step60:

- `success=3`
- `wrong_answer=87`
- `runtime_error=17`
- `timeout=6`
- `syntax_error=1`
- `extraction_failure=3`

Step80:

- `success=0`
- `wrong_answer=93`
- `runtime_error=13`
- `timeout=7`
- `syntax_error=1`
- `extraction_failure=3`

Step100:

- `success=2`
- `wrong_answer=88`
- `runtime_error=16`
- `timeout=8`
- `syntax_error=1`
- `extraction_failure=2`

Interpretation:

- The run reduced some `runtime_error` mass, but a meaningful portion of that shift moved into `wrong_answer`.
- `accepted` did not improve on the hardest dataset.
- Lower judge time does not mean the model found uniformly better algorithms; in many cases it simply failed faster.

#### HumanEval

- Baseline `accepted`: `0.8720`
- Step100 `accepted`: `0.8476`

Interpretation:

- HumanEval regressed slightly but did not collapse.
- The model stayed in the same general capability band, with a small negative delta.

#### MBPP_reg

- Baseline `accepted`: `0.5850`
- Step60/80/100 `accepted`: `0.6300`

Interpretation:

- RL consistently improved the smaller regression-style coding set.
- This is the clearest positive signal in the full evals.

## Checkpoint Selection

If one checkpoint must be selected from this run, `step100` is the most balanced choice.

Why not `step60`:

- It matched baseline `accepted` on CodeContests but had worse `pass_ratio_all`.

Why not `step80`:

- It had the best CodeContests `pass_ratio_all`, but `accepted=0` on full `117`-problem evaluation.

Why `step100`:

- Best overall balance across the three datasets
- Slight HumanEval recovery relative to `step60/80`
- CodeContests almost back to baseline aggregate, though still not clearly above it

## Main Conclusions

### What this run proved

- The A1 RL setup is trainable and operationally stable.
- Shared verifier + multi-sandbox infra is good enough for a real 100-step run.
- Actor update is not the bottleneck anymore.
- RL improves some easier or more structured coding tasks (`MBPP_reg`).

### What this run did not prove

- It did not show a clear aggregate gain on the hardest target set, `codecontests_valid`.
- It did not show that pure RL, with the current data and reward setup alone, is enough to reliably improve full algorithmic correctness.

### Best reading of the result

This was a successful formal observation run, but not a decisive quality win over the phase-0 baseline on the hardest coding benchmark.

## Recommendations

### Keep these core RL hyperparameters unchanged for now

Do not make the next decision by immediately changing:

- `ppo_epochs`
- `learning_rate`
- `train_batch_size`
- `rollout.n`

Reason:

- The current issue is not obvious optimizer instability.
- The main limitation looks more like data/ability coverage on hard algorithmic problems than a clearly wrong RL hyperparameter.

### Improve checkpoint and eval strategy first

For the next long run:

- Keep a cheap high-frequency fast-val for smoke/trend only
- Add a medium-frequency eval after saves
- Use a stronger checkpoint selection rule than fast-val alone

Recommended hierarchy:

- High frequency: small CodeContests fast-val
- Medium frequency: `val_tier1` or an equivalent balanced eval slice
- Low frequency: full `codecontests_valid + humaneval + mbpp_reg`

### Most promising next training change

The most promising next move is not another blind long RL run with the same setup.

Instead, prepare a small targeted repair-SFT or teacher-SFT set built from:

- persistent `wrong_answer` cases in CodeContests
- tasks that oscillate between partial pass and failure
- problems where RL reduced runtime errors but did not convert them into success

This would target the actual weakness revealed by the full evals:

- the model is often closer to correct
- but not consistently crossing the last correctness threshold on hard algorithmic problems

### If another RL run is still desired

If a follow-up RL run is launched before adding repair SFT:

- keep the core training hyperparameters fixed
- start from `step100`
- strengthen eval-based checkpoint selection
- avoid reading too much into small fast-val fluctuations

## Bottom Line

- Systems result: successful
- Training result: real learning signal, no collapse
- Quality result: mixed
- Strongest gain: `MBPP_reg`
- Weakest area: full `CodeContests_valid`
- Best current checkpoint: `step100`
- Best next move: stronger eval discipline plus targeted repair/teacher SFT before major RL hyperparameter changes
