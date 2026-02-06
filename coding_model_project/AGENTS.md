# Claude AI Assistant Context - RLVR Coding Model Project

---

## Project Overview for AI Assistants

This document provides context for AI assistants (Claude, GPT, etc.) working on this project.

## Cross-Repo Path Note (Important)

- Primary project repo: `/Users/xiaohui/Desktop/verl/verl`
- SandboxFusion repo: `/Users/xiaohui/Desktop/verl/SandboxFusion`
- When reviewing or implementing anything related to SandboxFusion APIs, code execution behavior, judge metadata, or docs, inspect files under the SandboxFusion repo path above instead of only searching in this repo.

### What is this project?

This is an **end-to-end LLM post-training project** using Reinforcement Learning with Verifiable Rewards (RLVR) for code generation. The goal is to train a model that can solve competitive programming problems better through:

1. **SFT (Supervised Fine-Tuning)** - Reduce syntax/runtime errors
2. **DPO (Direct Preference Optimization)** - Optional preference alignment
3. **GRPO (Group Relative Policy Optimization)** - Online RL with code execution feedback
4. **Multi-round Repair** - Agentic code fixing based on execution feedback

### Project Purpose

This is a **resume project** designed to demonstrate:
- Understanding of LLM post-training pipelines
- Practical experience with RL training (GRPO/PPO)
- Engineering skills (distributed training, async evaluation, data governance)
- Rigorous experimental methodology (ablations, multiple seeds, proper train/test splits)

---

## Key Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Training Framework | **verl** | Distributed RL training |
| Inference Engine | **vLLM** | High-performance LLM inference |
| Code Evaluation | **SandboxFusion** | Safe code execution sandbox |
| Base Model | **Qwen2.5-Coder-7B-Instruct** | Starting checkpoint |
| Experiment Tracking | **WandB** | Metrics logging |

---

## Directory Structure Quick Reference

```
coding_model_project/
├── experiment_design/          # Design docs (START HERE for understanding)
│   └── final_experiment_design.md   # ★ MOST IMPORTANT FILE
├── phase_0_ Baseline/          # Phase 0 implementation docs
├── src/                        # Core implementation code
│   ├── phase0_eval.py          # Main evaluation script
│   ├── eval_config.py          # Constants and configs
│   ├── data_governance.py      # Data deduplication
│   └── utils/                  # Metrics, logging utilities
├── data/                       # Dataset manifests and raw data
├── outputs/                    # Evaluation results
├── scripts/                    # Shell scripts
└── agent.md                    # Full project introduction
```

---

## When Helping with This Project

### 1. For Understanding the Project
- Start with `experiment_design/final_experiment_design.md` - contains complete 5-phase design
- Read `agent.md` for file structure overview
- Check `phase_0_ Baseline/phase0_implementation_plan.md` for implementation details

### 2. For Code Implementation
- Core evaluation logic: `src/phase0_eval.py`
- Configuration constants: `src/eval_config.py`
- Metrics collection: `src/utils/metrics.py`
- Data loading: `src/data_governance.py`

### 3. For Experiment Design Questions
- Evaluation protocols: `experiment_design/eval_protocol.md`
- Reward design (Dense vs Sparse): `experiment_design/reward_design.md`
- Data governance: `experiment_design/data_governance.md`
- Guardrails: `experiment_design/guardrails.md`

### 4. For verl Framework Questions
- Architecture guide: `phase_0_ Baseline/verl_standalone_rollout_guide.md`
- Learning materials: `verl基础讲解/` directory

---

## Key Concepts

### Evaluation Protocols
| Protocol | Definition | When to Use |
|----------|------------|-------------|
| **EVAL@1** | Best-of-1 (greedy decoding) | Main comparison metric |
| **EVAL@k** | Best-of-k samples | Inference budget curves |
| **EVAL@budget** | Fixed token budget | Cost-controlled comparison |

### Reward Types
| Type | Definition | Value Range |
|------|------------|-------------|
| **Dense** | `reward = pass_ratio` | [0, 1] continuous |
| **Sparse** | `reward = 1[accepted]` | {0, 1} discrete |

### Dataset Roles
| Dataset | Role | Rule |
|---------|------|------|
| CodeContests_train | Training | For SFT/GRPO training |
| CodeContests_valid | Validation | Hyperparameter selection |
| CodeContests_test | Test | Final evaluation only |
| HumanEval | Test only | Industry benchmark, NO training |
| MBPP_reg | Validation | Quick regression checks |

---

## Current Project Status

### Completed
- [x] Phase 0: Baseline evaluation completed
- [x] Data governance (deduplication, leakage checks)
- [x] Core evaluation infrastructure

### In Progress / TODO
- [ ] Phase 1: SFT implementation
- [ ] Phase 2: DPO (optional)
- [ ] Phase 3: GRPO training
- [ ] Phase 4: Multi-round repair (optional)

---

## Common Tasks

### Running Phase 0 Evaluation
```bash
python src/phase0_eval.py \
    --mode simple \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --datasets humaneval mbpp_reg codecontests_valid
```

### Checking Results
- Metrics: `outputs/phase0_*/metrics.json`
- QA logs: `outputs/phase0_*/qa_logs/`
- Run info: `outputs/phase0_*/run_info.json`

---

## Important Notes for AI Assistants

1. **Data Isolation is Critical**: Never mix train/test data. HumanEval and CodeContests_test are TEST ONLY.

2. **Evaluation Protocol Consistency**: Always use the same temperature/top_p/max_tokens across comparisons.

3. **Error Types**: Track syntax_error, runtime_error, timeout, wrong_answer separately.

4. **Cost Metrics**: Always report both tokens AND judge_time, not just accuracy.

5. **Reproducibility**: Use 2 seeds for key experiments, report mean±std.

6. **verl Framework**: This project uses verl's Standalone Rollout for inference, which is different from the training rollout workers.

---

## Files to Read First

If you're new to this project, read these files in order:

1. `experiment_design/final_experiment_design.md` - Complete design


