# Step900 Teacher Shards (v2 Backend RR)

当前发给 Claude Code 的是 **audited local shard materialization**，
不是最早那份只带 `review_status` 的旧 shard。

这版 shard 已经补上：

- `source_truncated`
- `source_char_cap`
- `source_provenance`
- `selection_mode`
- `c_trust_class`
- `c_source_tag`
- `c_decision_reason`
- `c_reviewer`

并且这轮 `SelectiveC` 的真实语义已经明确为：

- `legacy_reviewed_c + heuristic_supplemental_c`
- 不是“纯 reviewed-C 三层 corpus”
- 这里两类标签表示的是来源差异：
  - `legacy_reviewed_c` = `step600` 冷启动 bootstrap reviewed-C
  - `heuristic_supplemental_c` = `step600 -> step900` 训练过程中真实进入 `C` 的 retained C
- 不应直接解释成质量高低排序

当前 canonical teacher queue 已经切成 9 个互不冲突的 shard。

输入 shard 目录：
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr`

输出目录根：
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr`

manifest：
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/manifest.json`

audited source requests：
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/teacher_generation_requests_step900_candidate_v2_backend_rr.jsonl`

audited source student references：
- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/student_references_step900_candidate_v2_backend_rr_enriched.jsonl`

Shard summary:

- `core_01`: Core, 45 requests, 45 generation units
  input: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/core_01.jsonl`
  output: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/core_01/core_01_raw_responses.jsonl`
  prompt: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/prompt_core_01.md`
- `core_02`: Core, 44 requests, 44 generation units
  input: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/core_02.jsonl`
  output: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/core_02/core_02_raw_responses.jsonl`
  prompt: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/prompt_core_02.md`
- `core_03`: Core, 44 requests, 44 generation units
  input: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/core_03.jsonl`
  output: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/core_03/core_03_raw_responses.jsonl`
  prompt: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/prompt_core_03.md`
- `exp_01`: Expansion, 26 requests, 52 generation units
  input: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/exp_01.jsonl`
  output: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/exp_01/exp_01_raw_responses.jsonl`
  prompt: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/prompt_exp_01.md`
- `exp_02`: Expansion, 26 requests, 52 generation units
  input: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/exp_02.jsonl`
  output: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/exp_02/exp_02_raw_responses.jsonl`
  prompt: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/prompt_exp_02.md`
- `exp_03`: Expansion, 26 requests, 52 generation units
  input: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/exp_03.jsonl`
  output: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/exp_03/exp_03_raw_responses.jsonl`
  prompt: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/prompt_exp_03.md`
- `exp_04`: Expansion, 26 requests, 52 generation units
  input: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/exp_04.jsonl`
  output: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/exp_04/exp_04_raw_responses.jsonl`
  prompt: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/prompt_exp_04.md`
- `selc_01`: SelectiveC, 22 requests, 44 generation units
  input: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/selc_01.jsonl`
  output: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/selc_01/selc_01_raw_responses.jsonl`
  prompt: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/prompt_selc_01.md`
- `selc_02`: SelectiveC, 22 requests, 44 generation units
  input: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/selc_02.jsonl`
  output: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/selc_02/selc_02_raw_responses.jsonl`
  prompt: `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/prompt_selc_02.md`
