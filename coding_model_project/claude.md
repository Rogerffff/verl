# Claude AI Assistant Context - RLVR Coding Model Project

---

## Project Overview

This is an **end-to-end LLM post-training project** using Reinforcement Learning with Verifiable Rewards (RLVR) for code generation. The goal is to train a model that can solve competitive programming problems better through GRPO with code execution feedback.

This is a **resume project** designed to demonstrate:
- Understanding of LLM post-training pipelines (SFT → GRPO)
- Practical experience with RL training (GRPO/PPO in verl framework)
- Engineering skills (shared verifier, distributed training, async evaluation, data governance)
- Rigorous experimental methodology (ablations, multiple seeds, proper train/test splits)

---

## Current Project Status

| Phase | Status | Notes |
|---|---|---|
| Phase 0: Baseline Eval | **Done** | Eval infra complete, baseline established |
| Phase 1: SFT | **Shelved** | Training results worse than base model, skipped |
| Phase 2: DPO | **Skipped** | Optional, not planned |
| **Phase 3: GRPO** | **In Progress** | Infra done, reward frozen, training starting |
| Phase 4: Multi-round Repair | Pending | Optional agentic extension |

**Key decision**: Skip SFT, go directly to GRPO. Justified by Phase 0 baseline showing pass_ratio_mean=14% on CodeContests (sufficient dense reward signal for RL).

---

## Technical Stack

| Component | Technology |
|---|---|
| Training Framework | **verl** (Ray-based distributed RL) |
| Inference Engine | **vLLM** (hybrid engine in verl) |
| Code Evaluation | **SandboxFusion** (safe code execution) |
| Base Model | **Qwen2.5-Coder-7B-Instruct** |
| Hardware | **4x RTX 5090 32GB** |
| Container | `verlai/verl:vllm011.latest` |
| Experiment Tracking | **WandB** |

---

## Directory Structure

```
coding_model_project/
├── phase_2_ GRPO/                  # ★ GRPO phase docs & scripts
│   ├── formal_reward_design.md     # ★ Frozen reward v1 design (R3: dense_anchor_v1)
│   ├── algorithm_decision_guide.md # ★ A0/A1/A2 algorithm variants & adoption rules
│   ├── experiment_handoff.md       # ★ Remote experiment runbook & handoff doc
│   ├── full_code_path_guide.md     # Complete RL + Eval code path trace
│   ├── shared_verifier_infra_guide.md  # Shared verifier architecture doc
│   ├── README.md                   # GRPO phase overview & data processing details
│   ├── run_grpo_smoke.sh           # GRPO smoke test script (4 GPU)
│   └── run_grpo_step_smoke.sh      # GRPO single-step smoke script
│
├── experiment_design/              # Phase 0 design docs
│   ├── final_experiment_design.md  # Original 5-phase experiment design
│   ├── eval_protocol.md            # EVAL@1/k/budget protocols
│   ├── reward_design.md            # Reward research notes (pre-formal)
│   ├── data_governance.md          # Dedup pipeline design
│   ├── guardrails.md               # Safety guardrails
│   └── metric_templates.md         # Metric definitions
│
├── src/                            # ★ Core implementation code
│   ├── verifier/                   # ★ Shared verifier (eval + RL共用)
│   │   ├── shared.py               #   Core: normalize_candidate, verify_candidate, verify_candidate_batch
│   │   └── __init__.py             #   Exports
│   ├── grpo_batch_reward.py        # ★ RL reward function (plugs into verl BatchRewardManager)
│   ├── prompting.py                # ★ Prompt templates (shared by eval + parquet builder)
│   ├── build_grpo_parquet.py       # ★ Data builder: manifest+raw → verl Parquet
│   ├── phase0_eval.py              # Main evaluation script (2600+ lines)
│   ├── eval_config.py              # Eval constants and configs
│   ├── data_governance.py          # Deduplication logic
│   └── utils/                      # Metrics, QA logging utilities
│       ├── metrics.py              #   MetricsCollector, EvalResult, DatasetMetrics
│       └── qa_logger.py            #   Stratified QA sampling
│
├── scripts/
│   └── run_phase0.sh               # Eval run script (requires manifest + sandbox)
│
├── data/                           # Dataset manifests and raw data
│   ├── manifests/                  #   Deduplicated problem_id lists (*.jsonl)
│   ├── raw/                        #   Full records with test_cases (*.jsonl)
│   └── grpo_parquet/               #   Built Parquet files for verl training
│
├── outputs/                        # Evaluation results
│   └── phase0_fullval_20260331/    # ★ Current GRPO-branch baseline
│       ├── eval_analysis.md        #   Detailed analysis with reward simulations
│       ├── metrics.json            #   Per-dataset metrics
│       ├── summary.json            #   Aggregated summary
│       └── per_problem/            #   Per-problem JSONL with per_case_results
│
├── PROGRESS.md                     # Project progress log
└── CLAUDE.md                       # This file
```

### verl framework modifications (minimal, 2 files)

```
verl/trainer/ppo/
├── metric_utils.py     # Added compute_verifier_metrics() — aggregates verifier fields for WandB
└── ray_trainer.py      # Added one line: metrics.update(compute_verifier_metrics(batch=batch))
```

---

## Key Design Documents (read in this order)

### For understanding the GRPO phase:
1. **`phase_2_ GRPO/algorithm_decision_guide.md`** — Algorithm variants A0/A1/A2, adoption rules, rescue config
2. **`phase_2_ GRPO/formal_reward_design.md`** — Frozen reward v1 (R3: dense_anchor_v1), reward formula, ablation plan
3. **`phase_2_ GRPO/experiment_handoff.md`** — Remote experiment runbook, GPU topology, step-by-step instructions
4. **`phase_2_ GRPO/full_code_path_guide.md`** — Complete code path trace for RL training + eval
5. **`phase_2_ GRPO/shared_verifier_infra_guide.md`** — Shared verifier architecture, data flow, integration

### For understanding Phase 0 baseline:
6. **`experiment_design/final_experiment_design.md`** — Original 5-phase design
7. **`outputs/phase0_fullval_20260331/eval_analysis.md`** — Current baseline analysis with reward simulations
8. **`PROGRESS.md`** — Project progress and key decisions log

---

## Key Architecture: Shared Verifier

The project's core infra decision is a **shared verifier** that both eval and RL training use for judging:

```
shared verifier (src/verifier/shared.py)
├── normalize_candidate() — extract code from model output
├── verify_candidate()    — judge single problem
└── verify_candidate_batch() — concurrent batch judging
         │                              │
    Eval path                      RL training path
    phase0_eval.py                 grpo_batch_reward.py
    evaluate_with_run_code()       compute_score()
         │                              │
    Terminal / JSONL              verl BatchRewardManager
                                        │
                                  ray_trainer.py → WandB
```

**Why**: verl's built-in `sandbox_fusion.compute_score(continuous=True)` only uses first 10 testcases. Our verifier runs ALL testcases for trustworthy pass_ratio.

---

## Algorithm Decision Summary

**Headline**: GRPO with DAPO-style stabilizations (A1 mainline)

| Variant | Key Differences | Role |
|---|---|---|
| A0 | Symmetric clip (0.2/0.2), with KL | Stable baseline control |
| **A1** | **Asymmetric clip (0.2/0.28), no KL, token-mean** | **Recommended mainline** |
| A2 | A1 + Dr.GRPO debiasing (no std norm, seq-mean-token-sum-norm) | Strong ablation |

Rescue config: if A1 drifts (timeout_rate 2x baseline), add back `use_kl_loss=True, kl_loss_coef=0.001, kl_loss_type=low_var_kl`.

---

## Reward Design Summary

**Frozen reward v1 mainline**: `anchored_dense_v1`

```
if invalid_for_rl or truncated_by_max_tokens:
    reward = INVALID_FOR_RL
elif error_type in {empty_output, extraction_failure, non_code, syntax_error}:
    reward = -1.0
else:
    reward = 0.8 * pass_ratio_all + 0.2 * accepted
```

Strong contrast reward: `dense_anchor_v1 = clip(-0.2 + 1.0 * pass_ratio_all + 0.2 * accepted, -1, 1)`.

---

## Dataset Roles (Critical — data isolation)

| Dataset | File | Size | Role | Rules |
|---|---|---|---|---|
| CodeContests_train | `codecontests_train_wo_valid_big` | 11,785 | Training | **Must use `_wo_valid_big` variant** |
| CodeContests_valid | `codecontests_valid` | 117 | High-freq val (tier1) | Hyperparameter tuning allowed |
| CodeContests_valid_big | `codecontests_valid_big` | 500 | Low-freq val (tier2) | Main checkpoint selection |
| CodeContests_test | `codecontests_test` | 165 | Final eval only | **NO training/tuning** |
| HumanEval | `humaneval` | 164 | Test only | **NO training/tuning** |
| MBPP_reg | `mbpp_reg` | 200 | Quick regression | ID range 11-210 |

---

## Running Things

### Build GRPO Parquet data
```bash
python coding_model_project/src/build_grpo_parquet.py \
    --data_root coding_model_project/data \
    --output_dir coding_model_project/data/grpo_parquet
```

### Run Eval (Phase 0)
```bash
cd coding_model_project && bash scripts/run_phase0.sh
```
Requires: vLLM at :8001, SandboxFusion at :8080, manifest files in data/manifests/

### Run GRPO Smoke Test
```bash
bash "coding_model_project/phase_2_ GRPO/run_grpo_smoke.sh"
```
Requires: 4 GPUs, SandboxFusion at :8080, smoke Parquet files built

### Check Results
- Eval metrics: `outputs/phase0_*/metrics.json`
- Eval per-problem: `outputs/phase0_*/per_problem/*.jsonl`
- Training metrics: WandB dashboard (`verifier/*` panel)

---

## Important Notes for AI Assistants

1. **Data Isolation is Critical**: Never mix train/test. Use `_wo_valid_big` (11,785), NOT the 12,285 variant.
2. **Shared Verifier is the single source of truth**: Both eval and RL reward must go through `src/verifier/shared.py`. Never use verl's built-in `default_compute_score` for CodeContests.
3. **Evaluation Protocol Consistency**: EVAL@1 = T=0.0, top_p=1.0, max_new_tokens=2048, run_timeout=30s.
4. **Error Types**: Track syntax_error, runtime_error, timeout, wrong_answer separately — they have different reward implications.
5. **Reward is frozen**: `anchored_dense_v1` is the v1 mainline reward. `dense_anchor_v1` is the strongest explicit contrast.
6. **Algorithm-Reward boundary**: Algorithm design (A0/A1/A2) and reward design (R1/R2/R3) are independent axes. Don't conflate them.
7. **Trainer-side metadata merge is required**: `finish_reason` / `truncated_by_max_tokens` must be merged into per-sample `extra_info` before BatchRewardManager runs.
8. **verl framework changes are still small but not zero**: `ray_trainer.py`, `metric_utils.py`, `core_algos.py`, rollout metadata plumbing, and the project reward module all changed together.
