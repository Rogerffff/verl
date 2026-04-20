# Repair RL Background And Consulting Brief (2026-04-20)

## 0. Purpose

This document is a self-contained background brief for other agents or collaborators who need to quickly understand:

1. what this coding RL project is,
2. what has already been built and validated,
3. what the current repair-related bottleneck is,
4. why repair RL is now being considered,
5. what external work is most relevant,
6. what kinds of advice are currently most useful.

This is not meant to replace the canonical project docs. It is a consultation-oriented summary that packages the current state into one place so other agents can give higher-quality recommendations without re-reading the whole repo.

---

## 1. One-Paragraph Executive Summary

This project is a coding RL post-training project built on `verl`, using verifier-based GRPO to improve code generation ability on competitive-programming style tasks. The formal Phase 3 mainline is `GRPO with DAPO-style stabilizations`, using `anchored_dense + guardrails` reward and a shared verifier infra so that eval and RL training consume the same ground-truth execution signal. The main raw RL story is already strong: held-out raw coding ability improved substantially over the base model, and the project has already accumulated a large amount of systems, eval, data-governance, repair, and QC infrastructure. However, the current repair-SFT line has not yet become a held-out headline win: dev-side repair gains are real, but the best current repair-SFT checkpoint still does not beat `step1300_rl` on held-out deployment-style repair evaluation. The present question is whether a focused repair RL round is a better next move than trying to expand repair-SFT again, especially because repair-SFT data appears hard to scale further while repair RL may be cheaper and less data-preparation-heavy.

---

## 2. Repo And Branch Context

### 2.1 Main repos

- Main project repo:
  - `/Users/roger/Desktop/coding_RL_project/verl`
- Sandbox repo used for execution / judging:
  - `/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion`

Important note for any agent:

- `verl` and `SandboxFusion` are separate git repos.
- If a question touches sandbox API behavior, execution semantics, return fields, or runtime issues, check `SandboxFusion`, not only `verl`.

### 2.2 Current branch-level framing

- The current working line is the GRPO development line for this coding RL project.
- The current formal algorithm choice is:
  - `GRPO with DAPO-style stabilizations`
- `filter_groups` is intentionally not part of the current mainline definition.

Canonical local references:

- `README.md`
- `algorithm_decision_guide.md`
- `formal_reward_design.md`
- `shared_verifier_infra_guide.md`
- `experiment_handoff.md`

---

## 3. Current Project Narrative

The cleanest honest external narrative is:

1. Start from a strong coding base model.
2. Observe that the target failure mode is mainly wrong-answer / partial-correctness, not syntax.
3. Build verifier-based RL infra in `verl` with a shared execution-based truth source.
4. Run curriculum RL and get real held-out raw gains.
5. Build one-turn repair eval, repair feedback prompts, repair-conditioned SFT, teacher/QC, quarantine, and rejudge machinery to understand whether partial gains can be converted into final AC.
6. Discover that repair helps on dev slices and in some deployment-style cases, but current repair-SFT still does not beat the strongest held-out RL checkpoint as the deployed model.

In other words:

- the project already has a strong raw RL story,
- a strong ML systems / eval rigor story,
- and a strong data quality / debugging / infra story,
- but the repair branch is not yet a clean held-out headline success.

---

## 4. Canonical Local Docs To Read First

If another agent only has time to read a few files, these are the most useful.

### 4.1 Current authority docs

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/experiment_handoff.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_eval_current_contract_2026-04-18.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_protocolA_results_2026-04-18.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_protocolB_results_2026-04-18.md`

### 4.2 Mainline design docs

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/README.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/algorithm_decision_guide.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/formal_reward_design.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/shared_verifier_infra_guide.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_phase4_design.md`

### 4.3 Resume / asset summary docs

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/resume_materials/project_asset_inventory_2026-04-18.md`

---

## 5. Current Technical Mainline

### 5.1 Training objective and algorithm

The formal Phase 3 headline is:

- `GRPO with DAPO-style stabilizations`

Included in the current mainline:

- `clip-higher`
- `no-KL`
- `token-mean`
- verifier-side / execution-side batching and stability improvements

Explicitly not part of the current mainline:

- `filter_groups / dynamic sampling`
- reward-side overlong shaping
- full DAPO reproduction

Source:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/algorithm_decision_guide.md`

### 5.2 Reward

The formal reward is:

- `anchored_dense + guardrails`
- default name: `anchored_dense_v1`

Formula:

- if infra/sandbox/API/no-test/truncation issues happen:
  - `INVALID_FOR_RL`
- if `error_type` is extraction / empty / non-code / syntax type:
  - `-1.0`
- else:
  - `0.8 * pass_ratio_all + 0.2 * accepted`

Important semantic choice:

- reward is based on full external tests from the shared verifier,
- `accepted` is kept as an explicit anchor,
- invalid samples get zeroed from RL updates.

Source:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/formal_reward_design.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/grpo_batch_reward.py`

### 5.3 Shared verifier infra

This project already has a strong shared verifier design:

- eval and RL both use the same verifier layer,
- the core execution truth contract lives in `verifier/shared.py`,
- `phase0_eval.py` and RL reward both consume it,
- verifier metrics are logged through the trainer path,
- invalid-for-RL semantics are explicitly integrated into trainer-side advantage handling.

Main code / docs:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/grpo_batch_reward.py`
- `/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/ppo/ray_trainer.py`
- `/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/ppo/metric_utils.py`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/shared_verifier_infra_guide.md`

---

## 6. Current Experimental Status

### 6.1 Strongest current raw deployed model

The current strongest held-out deployed base is still:

- `step1300_rl`

This is the main result anchor right now, not the repair-SFT line.

Relevant local references:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/experiment_handoff.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/resume_materials/resume_claim_evidence_map_2026-04-18.md`

### 6.2 Strongest raw held-out story

The strongest currently resume-ready outcome story is raw held-out RL gain:

- base `codecontests_test accepted@1 = 0.01818`
- `step1300_rl codecontests_test accepted@1 = 0.07879`
- `HumanEval accepted@1` also improved
- `MBPP_reg accepted@1` also improved

This supports the claim that the main RL line already produced meaningful external gains.

### 6.3 Checkpoint readout nuance

Checkpoint interpretation is not single-metric:

- `step1000` is the more exact-solve-friendly point
- `step1300` is stronger on partial-credit / pass-ratio quality

This matters because the current repair question is partly about whether stronger partial checkpoints can be converted into more final AC.

### 6.4 Repair current status

Repair is real but still qualified.

Current status:

- fixed-input repair skill is real on dev
- end-to-end self-repair is real in deployment-style eval
- current-lineage repair-SFT v1 improved dev-side repair
- but current repair-SFT still does not beat `step1300_rl` as the held-out deployed model

Specifically:

- best current-lineage repair-SFT checkpoint:
  - `step1300_sft_v1_step60`
- strongest held-out deployed base:
  - `step1300_rl`

Held-out key read:

- `step1300_rl` on `codecontests_test / Protocol B`:
  - `0.1091`
- `step1300_sft_v1_step60` on the same held-out Protocol B:
  - `0.0970`

Canonical references:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/experiment_handoff.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_protocolB_results_2026-04-18.md`
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_eval_current_contract_2026-04-18.md`

### 6.5 Current positioning of one-turn repair

One-turn repair is still useful, but internally it should currently be read as:

- a real capability,
- a good diagnosis / fallback tool,
- not yet the main engine of project-level headline improvement.

The current internal summary has already narrowed its role to:

- targeted fallback,
- repair-prior diagnosis tool,
- not the new main training engine.

Source:

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/experiment_handoff.md`

---

## 7. Existing Assets Relevant To Repair RL

This project already has many assets that make repair RL feasible.

### 7.1 Eval and repair pipeline assets

- one-turn repair eval implementation:
  - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py`
- repair prompt / feedback builder:
  - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py`
- current repair prompt design doc:
  - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_prompt_design.md`

### 7.2 Reward and verifier assets

- shared verifier
- batch reward adapter
- trainer integration for verifier metrics and invalid masking

These mean that repair RL does not need to start from zero on infra.

### 7.3 Existing repair evaluation assets

There are already complete Protocol A / Protocol B result trees and summaries in local outputs, including:

- `valid_big500`
- `codecontests_test`
- delta slices
- per-problem logs
- first-pass and after-repair summaries

### 7.4 Existing repair-SFT assets

The project already contains:

- repair-conditioned SFT planning docs
- repair SFT data prep assets
- teacher generation and QC assets
- anchor/general-code mix planning

But the current concern is that this repair-SFT data line is becoming hard to expand significantly from the current `step1300`-lineage traces.

### 7.5 Resume / consulting assets

The project already has unusually strong "meta-assets" for agent consultation:

- claim-evidence map
- project asset inventory
- handoff docs
- protocol docs

This makes it possible for other agents to reason about the project quickly, if given the right background document.

---

## 8. Current Bottleneck

The practical bottleneck is:

1. current repair-SFT is not yet a held-out headline win,
2. current repair-SFT data derived from `step1300` / current ABCD-bucket traces looks hard to scale much further,
3. another small repair-SFT pass may suffer from the same limited data-support problem,
4. repair RL may be cheaper because it can potentially reuse existing buggy-code + verifier-feedback assets and avoid another major teacher/SFT data construction round.

This changes the decision framing.

The question is no longer only:

- "Is repair RL elegant in theory?"

It is now also:

- "Given current local constraints, is repair RL the cheapest credible path to test whether second-pass training can create a held-out repair gain?"

---

## 9. Motivation For Considering Repair RL Now

### 9.1 High-level motivation

The motivation is to test whether current near-miss / partial-correctness gains can be converted into final AC through targeted second-pass learning, without requiring a large new repair-SFT dataset.

### 9.2 Why not just continue the current repair-SFT path?

Because the current concern is:

- the useful repair-SFT data pool from current traces may be close to saturation,
- expanding it much further may be difficult or expensive,
- current repair-SFT already shows dev gains but still does not overtake the strongest held-out RL checkpoint.

### 9.3 Why repair RL is attractive in this specific situation

Repair RL is attractive here because it may:

- reuse existing buggy-code and verifier-feedback assets,
- avoid another large teacher-generation cycle,
- preserve more general raw coding ability than narrow repair-SFT,
- directly optimize for second-pass repair success under the same verifier contract,
- provide a sharper yes/no answer about whether second-pass training is worth pursuing further.

### 9.4 What exact hypothesis repair RL would test

The concrete hypothesis is:

- first-pass RL has already created meaningful partial correctness,
- those failures are not uniformly hopeless,
- one-turn verifier-guided repair can be trained directly via RL,
- and a second-pass RL policy can improve held-out deployment-style repair enough to beat the current `step1300_rl` repair baseline.

In plain terms:

- can second-pass learning create a real held-out gain where current repair-SFT has not yet crossed the line?

---

## 10. Why Repair RL Is Not Automatically Easy

This section expands the main cautions in detail.

### 10.1 Reward distribution is more skewed than single-turn coding RL

In ordinary single-turn coding RL, the reward only asks:

- how good is the final code?

In repair RL, the task changes:

- given a buggy first-pass solution and verifier feedback, how well does the model repair it?

This creates a more skewed reward distribution:

1. many samples are still nearly hopeless and remain near zero,
2. a small subset of high-partial near-miss samples jump sharply upward,
3. some samples regress because the model over-edits and destroys correct parts.

So a repair-RL group may look like:

- many low or zero-ish values,
- a few very large wins,
- a few misleading medium values from degraded edits.

This makes group-relative estimation noisier than in ordinary single-turn RL.

### 10.2 Trigger policy is highly sensitive

Repair is not equally useful on all failures.

If repair candidate selection is too broad:

- too many hopeless samples enter training,
- reward variance collapses,
- judge cost rises,
- training becomes inefficient.

If it is too narrow:

- training only covers trivial near-miss repairs,
- apparent repair success looks high,
- but coverage and generalization stay weak.

This means repair RL depends strongly on:

- which buckets are allowed,
- what minimum pass-ratio threshold is used,
- which error types are included,
- how many failure cases are shown in feedback.

### 10.3 Judge cost is higher

A fully online repair pipeline can require:

1. first-pass generation,
2. first-pass judge,
3. feedback construction,
4. second-pass generation,
5. second-pass judge.

That is much heavier than a one-pass RL sample.

This can be partly reduced by using cached first-pass traces and only training the second pass, but repair RL still tends to create higher execution cost pressure than ordinary single-turn RL.

### 10.4 Plain `anchored_dense` is probably not sufficient

The current reward only looks at final solution quality.

For repair, that misses three important facts:

1. whether the second pass improved relative to the first pass,
2. whether it regressed and damaged already-correct behavior,
3. whether the edit was unnecessarily large or destructive.

Example failure mode:

- first pass: `0.9`
- second pass: `0.7`

Under plain final-score reward, this can still look like a decent positive sample.
But in repair terms, it is a bad repair because the model made the solution worse.

This is why second-pass repair may require a more edit-aware reward than current plain final-score `anchored_dense`.

### 10.5 Over-editing is a real risk

Repair is not only about "make final tests pass".
It is also about:

- not overwriting correct code unnecessarily,
- not turning a local bug fix into a full unstable rewrite,
- preserving already-correct structure when possible.

A reward that only values final pass-ratio can unintentionally encourage:

- rewrite-heavy behavior,
- unstable repair style,
- higher regression risk.

---

## 11. External Work Reviewed

Below is a compact review of external work that is most relevant to this decision.

### 11.1 CodeRL (2022)

- Title:
  - `CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning`
- Link:
  - https://arxiv.org/abs/2207.01780
- Relevance:
  - shows that RL with execution / functional-correctness signal can improve code generation,
  - introduces a critic-based dense-feedback design for general code generation.
- Transferable lesson:
  - verifier/test feedback is a legitimate RL signal for coding,
  - but this is still mainly a generate-from-scratch coding RL paper, not a repair-specific recipe.

### 11.2 RLTF (2023)

- Title:
  - `RLTF: Reinforcement Learning from Unit Test Feedback`
- Link:
  - https://arxiv.org/abs/2307.04349
- Relevance:
  - argues that online RL with unit-test feedback is useful,
  - emphasizes multi-granularity feedback instead of only coarse final outcome.
- Transferable lesson:
  - online verifier-guided RL is a valid direction,
  - but again this is general coding RL evidence, not directly a second-pass repair design.

### 11.3 ChatRepair / Conversational APR (2023)

- Title:
  - `Conversational Automated Program Repair`
- Link:
  - https://arxiv.org/abs/2301.13246
- Relevance:
  - a key repair reference using validation feedback in a repair loop,
  - alternates patch generation and feedback conversationally.
- Transferable lesson:
  - repair pipelines based on validation feedback are mature and sensible,
  - public repair literature is more mature on feedback-loop repair pipelines than on repair RL as a main training line.

### 11.4 Self-Debugging (2023)

- Title:
  - `Teaching Large Language Models to Self-Debug`
- Link:
  - https://arxiv.org/abs/2304.05128
- Relevance:
  - shows iterative debugging / repair can materially improve code performance.
- Transferable lesson:
  - second-pass improvement is real,
  - but the paper is more about prompting / self-debugging behavior than repair-specific RL.

### 11.5 VRpilot / Validation-feedback repair (2024)

- Title:
  - `A Case Study of LLM for Automated Vulnerability Repair: Assessing Impact of Reasoning and Patch Validation Feedback`
- Link:
  - https://arxiv.org/abs/2405.15690
- Relevance:
  - shows reasoning + validation feedback materially help repair in security settings.
- Transferable lesson:
  - external tool feedback is highly useful in repair,
  - but this again supports repair pipelines more directly than it supplies a repair RL recipe.

### 11.6 PSGPO (2024/2025)

- Title:
  - `Process Supervision-Guided Policy Optimization for Code Generation`
- Link:
  - https://arxiv.org/abs/2410.17621
- Relevance:
  - highlights that pure final unit-test reward can be too sparse,
  - proposes line-level / process reward modeling for coding RL.
- Transferable lesson:
  - if repair RL suffers from sparse or badly distributed reward, richer process signals may help,
  - but this is still general coding RL rather than direct repair RL.

### 11.7 SWE-RL (2025)

- Title:
  - `SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution`
- Link:
  - https://arxiv.org/abs/2502.18449
- Relevance:
  - an important sign that RL over software engineering data can transfer beyond narrow coding benchmarks.
- Transferable lesson:
  - software-engineering-flavored RL can improve general capability,
  - but SWE-RL is much broader than single-turn repair and uses different data assumptions.

### 11.8 ReCode (2025)

- Title:
  - `ReCode: Updating Code API Knowledge with Reinforcement Learning`
- Link:
  - https://arxiv.org/abs/2506.20495
- Relevance:
  - edit-style code RL rather than bug repair,
  - compares favorably to SFT in retaining general code ability.
- Transferable lesson:
  - edit-style RL may be less damaging to broad general coding ability than narrow SFT,
  - this is one of the strongest external reasons repair RL is attractive here.

### 11.9 GAPO (2025/2026)

- Title:
  - `GAPO: Robust Advantage Estimation for Real-World Code LLMs`
- Link:
  - https://arxiv.org/abs/2510.21830
- Relevance:
  - directly studies group-relative RL in real-world code editing.
- Transferable lesson:
  - real code-edit reward distributions are skewed and noisy,
  - repair RL should expect group reward instability,
  - robust advantage estimation or cleaner sample filtering may be needed.

### 11.10 PRepair (2026)

- Title:
  - `QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization`
- Link:
  - https://arxiv.org/abs/2604.05963
- Relevance:
  - the closest currently visible direct precedent for repair-specific GRPO-style RL.
- Why it matters:
  - explicitly frames over-editing as a central failure mode,
  - uses edit-aware GRPO reward rather than only final pass/fail.
- Transferable lesson:
  - the strongest direct external support for repair RL is not "just use final test reward",
  - it is "use repair-aware reward, especially edit-aware reward".

---

## 12. External Takeaways Most Relevant To This Project

The literature review suggests the following:

1. verifier / unit-test feedback for code RL is well-supported,
2. iterative repair with validation feedback is well-supported,
3. public repair pipeline work is more mature than public repair-RL-as-mainline work,
4. edit-style RL can preserve broad ability better than narrow SFT in some settings,
5. real code-edit RL has skewed / noisy reward distributions,
6. the strongest direct repair-RL signal points toward edit-aware reward design rather than plain final-score reward.

So the literature does not say:

- "repair RL is obviously the standard next step."

Instead it says:

- repair RL is plausible and increasingly supported,
- but it should be approached carefully,
- especially on reward design and sample filtering.

---

## 13. Current Recommendation

### 13.1 Updated recommendation under current constraints

Given the newly stated constraint that current repair-SFT data is hard to expand significantly, the recommendation is:

- **repair RL is worth trying**
- but as a **small, controlled second-pass RL probe**
- not yet as a new large project mainline

This is more positive than a generic "stay with repair-SFT" recommendation because the local bottleneck is now clearly data-scale and prep-cost related.

### 13.2 Why this recommendation changed

If repair-SFT data were easy to grow, the higher-ROI move would still likely be:

- larger changed-recipe repair-SFT v2

But under the current condition:

- repair-SFT data seems close to saturation,
- extra teacher/QC work may not enlarge it much,
- and repair RL may test the key hypothesis more cheaply.

That makes repair RL a reasonable next experiment.

### 13.3 What should not be done

Do not immediately launch:

- a huge repair RL campaign over all repair-eligible failures,
- a reward that only copies plain `anchored_dense`,
- a broad trigger that includes many hopeless `bucket_0` / dead-hard cases,
- a setup that changes too many variables at once.

---

## 14. Proposed Shape Of A Minimal Repair RL Probe

This is not yet a final experimental plan, but it is the current recommended shape.

### 14.1 Scope

Define the task as:

- `problem statement + buggy first-pass code + verifier feedback -> repaired code`

This should be treated as a second-pass task, not ordinary single-turn coding.

### 14.2 Base model

Prefer:

- `step1300_rl`

Possible comparison checkpoint:

- `step1300_sft_v1_step60`

### 14.3 Candidate sample policy

Start narrow.

Likely initial candidate slices:

- higher-partial `A/B` style samples,
- near-miss failures,
- likely `wrong_answer` and `runtime_error`,
- avoid making `timeout` a primary focus at first,
- avoid flooding with `bucket_0` hopeless failures.

### 14.4 Data source strategy

Prefer reusing:

- existing first-pass traces,
- cached buggy code,
- existing verifier feedback logic,
- current repair prompt builder.

This keeps cost lower and avoids a new large teacher-data build.

### 14.5 Reward direction

Do not use only final-score `anchored_dense`.

A better starting direction is:

- final correctness term,
- plus improvement-vs-first-pass term,
- plus optional edit-preservation / edit-size regularization.

In rough terms:

- reward good repairs,
- penalize regressions,
- discourage destructive full rewrites when unnecessary.

### 14.6 Success criterion

The main success criterion should be:

- beat the current `step1300_rl` held-out repair baseline on deployment-style repair eval

Concretely, the key target is:

- `codecontests_test / Protocol B`

At the same time, check:

- raw general regression should not collapse.

### 14.7 When to stop early

If a small repair RL probe shows:

- unstable reward,
- no held-out gain,
- large judge cost for little improvement,
- or obvious general-ability regression,

then it should be stopped early and not promoted into a new long campaign.

---

## 15. Questions To Ask Other Agents

If sharing this doc with other agents, the most useful questions are probably:

1. Given the current local constraints, is a small repair RL probe higher ROI than another repair-SFT round?
2. What repair-specific reward design would be most reasonable on top of the current verifier contract?
3. What first-pass bucket / trigger policy is most likely to give stable repair RL signal?
4. Should repair RL be trained purely as second-pass RL using cached first-pass traces, or should it remain fully online?
5. How should success be measured so that the result is resume-relevant rather than only academically interesting?
6. If repair RL fails, what is the best fallback:
   - repair-SFT v2,
   - repair-distilled single-turn SFT,
   - or no further repair training at all?

---

## 16. Final Working Position

The current working position behind this brief is:

- the project is already strong enough to be resume-worthy without new repair-RL results,
- but a small repair RL probe is now justified because repair-SFT data scale appears constrained,
- and repair RL may be the cheapest remaining way to test whether second-pass learning can become a genuine held-out improvement rather than only a dev-side or narrow-slice effect.

The main caveat is that repair RL should not be treated as "ordinary coding RL but with a repair prompt".
It likely needs:

- repair-aware sample selection,
- repair-aware reward shaping,
- and careful control of judge cost.

That is the main thing other agents should keep in mind when giving advice.
