# Step900 Pure ShortDiag SFT Training Guide

## 0. Purpose

这份文档是当前 `step900 repair-conditioned SFT` 的**审核入口页**。

目标有两个：

1. 让另一个 agent 可以快速审核：
   - 当前最终 SFT 数据是否来自正确的数据处理链
   - 当前训练脚本和参数口径是否自洽
2. 固定这条线当前准备开跑的训练口径：
   - 使用 `182-row` 纯 `short_diagnosis+code` 数据
   - 先不做内部 `train/val/test split`
   - 训练时只使用单个 train parquet
   - checkpoint 选择依赖外部 `valid_big500 repair eval`

这份文档不替代更早的规划文档；它把它们串起来，并补上当前已经物化的最终数据资产与训练入口。

---

## 1. Current Decision

当前准备进入 SFT 的数据集是：

- **student anchor**：`step900`
- **init model**：必须显式固定到 `step900` 的 merged HF 目录，不能使用 launcher 的 step200 fallback
- **instruction family**：`short_diagnosis+code`
- **final train artifact**：`182-row pure shortdiag parquet`
- **source split**：全部来自 `codecontests_train_wo_valid_big`
- **verification level**：全部来自 `accept_2_of_2`
- **internal validation**：当前关闭
- **model selection**：用外部 `valid_big500 repair eval` 做 checkpoint 比较

一句话版本：

**当前不是训练“混合 instruction-family”的 request-unique 193 行版本，而是训练其中更纯净的 182 行 `short_diagnosis_code` 子集。**

---

## 2. Canonical References

当前这条线需要一起看的上游文档：

- 数据规划：
  - [step900_repair_conditioned_sft_data_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_repair_conditioned_sft_data_plan.md)
- teacher generation / QC：
  - [step900_teacher_generation_qc_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_teacher_generation_qc_plan.md)
- teacher prompt / shard 协议：
  - [step900_teacher_prompt_and_schema_v2_backend_rr.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_teacher_prompt_and_schema_v2_backend_rr.md)
  - [step900_teacher_shards_v2_backend_rr.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_teacher_shards_v2_backend_rr.md)
- repair prompt 设计：
  - [repair_prompt_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_prompt_design.md)

当前这份文档只补充：

- 已经实际产出的最终 SFT 数据资产
- 数据从 verified keep set 到 pure shortdiag parquet 的最后两步
- 训练 launcher 的当前可用口径

---

## 3. Final Artifacts To Audit

### 3.1 Final SFT dataset

当前建议直接用于 SFT 的数据资产：

- request-unique mixed keep set：
  - [step900_request_unique_keep_set_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_request_unique_v1/step900_request_unique_keep_set_v1.jsonl)
- request-unique mixed SFT jsonl：
  - [step900_request_unique_sft_dataset_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_request_unique_v1/step900_request_unique_sft_dataset_v1.jsonl)
- request-unique mixed summary：
  - [step900_request_unique_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_request_unique_v1/step900_request_unique_sft_dataset_v1.summary.json)

最终纯 `short_diagnosis_code` 版本：

- pure shortdiag keep set：
  - [step900_shortdiag_pure_keep_set_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_keep_set_v1.jsonl)
- pure shortdiag SFT jsonl：
  - [step900_shortdiag_pure_sft_dataset_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.jsonl)
- pure shortdiag train parquet：
  - [step900_shortdiag_pure_train_v1.parquet](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_train_v1.parquet)
- pure shortdiag summary：
  - [step900_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.summary.json)

### 3.2 Current final counts

当前 final pure shortdiag 数据集的统计是：

- `row_count = 182`
- `distinct_request_count = 182`
- `distinct_problem_count = 182`
- `teacher_prompt_mode = short_diagnosis_code` for all rows
- `stability_label = accept_2_of_2` for all rows
- `repair_stratum`：
  - `Core = 87`
  - `Expansion = 70`
  - `SelectiveC = 25`
- `curriculum_bucket`：
  - `B_near_miss = 157`
  - `C_hard_partial = 25`
- `source_split`：
  - `codecontests_train_wo_valid_big = 182`
- `student_error_type`：
  - `wrong_answer = 179`
  - `runtime_error = 3`

---

## 4. End-To-End Data Processing Lineage

这里按“当前审核最重要的链路”压缩成 6 段。

### 4.1 Stage A-C: Candidate queue and student references

这一段的规划和边界条件在：

- [step900_repair_conditioned_sft_data_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_repair_conditioned_sft_data_plan.md)

关键口径：

- 主来源是 `codecontests_train_wo_valid_big`
- 不混入 eval slices
- 排除 quarantine 问题题
- 使用 `step900` 的 curriculum bucket 骨架
- 三层 repair strata：
  - `Core`
  - `Expansion`
  - `SelectiveC`

当前进入 teacher 阶段的 canonical queue 资产是：

- teacher requests：
  - [teacher_generation_requests_step900_candidate_v2_backend_rr.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/teacher_generation_requests_step900_candidate_v2_backend_rr.jsonl)
- teacher request summary：
  - [teacher_generation_requests_step900_candidate_v2_backend_rr.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/teacher_generation_requests_step900_candidate_v2_backend_rr.summary.json)
- enriched student references：
  - [student_references_step900_candidate_v2_backend_rr_enriched.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/student_references_step900_candidate_v2_backend_rr_enriched.jsonl)

当前 canonical queue 规模是：

- `281 requests`
- `429 generation units`

并且已经固定 judge 协议为：

- `patched sandbox`
- `client RR`
- `direct backend URL`
- `8081..8088`

### 4.2 Stage D-F: Teacher generation, QC, stability rejudge

这一段的主文档在：

- [step900_teacher_generation_qc_plan.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/step900_teacher_generation_qc_plan.md)

当前这一段的关键输出是：

- stable keep set：
  - [teacher_qc_v2_full.stable_keep_set.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/final_keep/teacher_qc_v2_full.stable_keep_set.jsonl)

这个 stable keep set 的统计是：

- `267 rows`
- 但不是 request-unique
- 对应 `182 distinct requests`

原因不是脏数据，而是：

- `Core` 用 `best_of_n = 1`
- `Expansion / SelectiveC` 用 `best_of_n = 2`
- 因此部分 request 会有两个都通过 QC 的 accepted attempts

teacher requests 的 best-of-n 来源脚本：

- [build_repair_conditioned_teacher_requests.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_repair_conditioned_teacher_requests.py)
- [build_teacher_generation_shards.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_teacher_generation_shards.py)

### 4.3 Reject salvage: Claude actionable repair + one-pass QC

在 stable keep set 之外，又做了一轮 reject salvage：

1. 从 reject 里抽出 `repair_candidate`
2. Claude 直接给 `repaired_code`
3. 只对这批 repair candidate 再跑一次 QC
4. 将 accepted salvage 样本并回 keep pool

这段新增的脚本：

- 构建 salvage QC candidates：
  - [build_step900_actionable_repair_qc_candidates.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_step900_actionable_repair_qc_candidates.py)
- 合并 salvage accepted 与原 keep：
  - [merge_step900_actionable_repair_keep_set.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/merge_step900_actionable_repair_keep_set.py)

salvage 合并后的 summary：

- [teacher_qc_v2_full.stable_keep_set_plus_actionable_repair_once.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/final_keep/teacher_qc_v2_full.stable_keep_set_plus_actionable_repair_once.summary.json)

当前这步的关键数字：

- 原 keep：`267`
- salvage candidates：`32`
- salvage accepted：`16`
- 合并后的 accepted attempt pool：`283`

### 4.4 Request-unique selection

为了进入真正的 SFT 数据集，我们不直接用 `283 accepted attempts`，而是先压成每个 `request_id` 只保留一个 target。

脚本：

- [build_step900_request_unique_sft_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_step900_request_unique_sft_dataset.py)

输入：

- [teacher_qc_v2_full.stable_keep_set_plus_actionable_repair_once.all.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/final_keep/teacher_qc_v2_full.stable_keep_set_plus_actionable_repair_once.all.jsonl)

输出：

- [step900_request_unique_keep_set_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_request_unique_v1/step900_request_unique_keep_set_v1.jsonl)
- [step900_request_unique_sft_dataset_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_request_unique_v1/step900_request_unique_sft_dataset_v1.jsonl)
- [step900_request_unique_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_request_unique_v1/step900_request_unique_sft_dataset_v1.summary.json)

当前 request-unique 选择规则是：

- higher `stability_label`
- then prefer `short_diagnosis_code`
- then lower `attempt_index`
- then shorter code
- then lower judge time

结果：

- `193 request-unique rows`
- 其中：
  - `182 short_diagnosis_code`
  - `11 code_only`

### 4.5 Pure shortdiag filtering

为了这轮先做更纯净的 `short_diagnosis+code` SFT，又从 `193 request-unique` 里取纯 shortdiag 子集。

脚本：

- [build_step900_shortdiag_pure_sft_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_step900_shortdiag_pure_sft_dataset.py)

输入：

- [step900_request_unique_keep_set_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_request_unique_v1/step900_request_unique_keep_set_v1.jsonl)
- [step900_request_unique_sft_dataset_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_request_unique_v1/step900_request_unique_sft_dataset_v1.jsonl)

输出：

- [step900_shortdiag_pure_keep_set_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_keep_set_v1.jsonl)
- [step900_shortdiag_pure_sft_dataset_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.jsonl)
- [step900_shortdiag_pure_train_v1.parquet](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_train_v1.parquet)
- [step900_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.summary.json)

结果：

- `182 rows`
- `182 distinct requests`
- `182 distinct problems`
- dropped `11` non-shortdiag requests

### 4.6 Prompt construction into training `messages`

request-unique SFT 构建时，user prompt 不是重新写死模板，而是直接复用：

- [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)

具体调用函数在：

- [build_step900_request_unique_sft_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_step900_request_unique_sft_dataset.py)

即：

- `build_codecontests_repair_prompt(...)`

因此训练样本里的 `messages` 与我们当前 repair eval / repair prompt 线保持一致，不是另起一套 SFT-only prompt。

---

## 5. Training Script And Current Parameters

### 5.1 Launcher

当前建议使用的 launcher：

- [run_phase2_repair_sft.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_phase2_repair_sft.sh)

这个脚本当前已经补成：

- 支持 `row_count` 风格的 summary / manifest
- 支持 `DISABLE_INTERNAL_VAL=true`
- 在 `DISABLE_INTERNAL_VAL=true` 时自动传：
  - `data.val_files=[]`
  - `trainer.test_freq=0`

这样可以直接用于当前这份单 train parquet 数据，不需要先伪造一个很薄的内部 val。

这里有一个前提：

- 当前仓库里的 trainer 代码已经补过“空 val + `test_freq <= 0`”路径

对应代码：

- [fsdp_sft_trainer.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/fsdp_sft_trainer.py)
- [sft_trainer.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/sft_trainer.py)
- [sft_trainer_ray.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/sft_trainer_ray.py)

### 5.2 Underlying trainer behavior

底层 trainer 当前都已经支持没有 `val_files`：

- [fsdp_sft_trainer.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/fsdp_sft_trainer.py)
- [sft_trainer.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/sft_trainer.py)
- [sft_trainer_ray.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/sft_trainer_ray.py)

这里之所以还要在 launcher 里把 `trainer.test_freq` 兜成 `0`，
是因为我们想明确关闭内部 validation；
当前 trainer 已经补过对 `test_freq <= 0` 的安全处理，不会再走到原先的 modulo-by-zero / 空 val 路径。

### 5.3 Current launcher defaults

脚本当前的默认训练参数是：

- `NPROC_PER_NODE=4`
- `TOTAL_TRAINING_STEPS=20`
- `TRAIN_BATCH_SIZE=8`
- `MICRO_BATCH_SIZE_PER_GPU=1`
- `DATA_MAX_LENGTH=4096`
- `TRUNCATION=error`
- `LR=5e-7`
- `WARMUP_RATIO=0.0`
- `SAVE_FREQ=4`
- `TEST_FREQ=2`
- `MAX_CKPT_TO_KEEP=4`
- `SAVE_HF_MODEL=true`
- `USE_WANDB=false`
- `SEED=42`

这些是 launcher 默认值，不等于这轮 pure shortdiag 训练必须完全照搬。

### 5.4 Recommended current run mode

对当前 `182-row pure shortdiag` 数据，我建议的当前运行口径是：

- `MODEL_PATH` 显式指向 `step900` 的 merged HF model
- `DISABLE_INTERNAL_VAL=true`
- `MIX_TRAIN_FILE` 指向 pure shortdiag parquet
- `MIX_MANIFEST_FILE` 指向 pure shortdiag summary
- `MIX_VAL_FILE` 不需要真实存在
- `TRAIN_BATCH_SIZE=8`
- `MICRO_BATCH_SIZE_PER_GPU=1`
- `ULYSSES_SEQUENCE_PARALLEL_SIZE=2`
- `USE_REMOVE_PADDING=true`
- `TOTAL_TRAINING_STEPS=60`
- `SAVE_FREQ=10`
- `MAX_CKPT_TO_KEEP=4`
- `TEST_FREQ` 在 `DISABLE_INTERNAL_VAL=true` 下会被 launcher 自动改成 `0`
- `LR`：
  - 单跑一条时，先用 `1e-6`
  - 如果预算允许，再补一条 `5e-7` 对照
- `DATA_MAX_LENGTH` 不应在审计前写死；必须先过 tokenizer-length audit

推荐原因：

- 这批数据本身只有 `182` 条，内部再切一个很薄的 val，统计意义有限
- 当前 global batch size=`8`、4 卡口径下，launcher 自己的采样语义约等于 `22 steps / epoch`
- 对 `6144` 长序列，优先先开 trainer 已内置的 `Ulysses SP + remove padding`，而不是先把 global batch size 降掉
- `60 steps` 约等于 `2.7 epochs`，比旧的 `20 steps` 更接近“正式首轮实验”，而不是 canary
- `SAVE_FREQ=10` 仍然能留下足够的 checkpoint 对比粒度，同时比 `5` 更省磁盘
- `MAX_CKPT_TO_KEEP=4` 更贴近当前远端可用磁盘空间上限
- 当前更关心：
  - 训练能否稳定收敛
  - checkpoint 在 `valid_big500 repair eval` 上的外部表现

### 5.4B Mandatory preflight: tokenizer-length audit

在正式启动前，必须先跑一次 tokenizer-level 长度审计。

审计脚本：

- [audit_step900_shortdiag_token_lengths.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/audit_step900_shortdiag_token_lengths.py)

作用：

- 使用训练时同一套 `MODEL_PATH` tokenizer
- 对当前 `182-row` pure shortdiag jsonl 逐条计算真实 token 长度
- 输出：
  - `over_max_length_count`
  - `token_count_total` 的 `max / p90 / p95 / p99`
  - 最长样本列表

推荐执行口径：

```bash
cd /Users/roger/Desktop/coding_RL_project/verl

python3 "/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/audit_step900_shortdiag_token_lengths.py" \
  --input_jsonl "/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.jsonl" \
  --model_path "/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_curriculum_step620_v3q3_to900_seed0/global_step_900/actor_merged_hf_phase4_repair" \
  --max_length 4096 \
  --output "/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_token_length_audit_v1.json"
```

如果审计结果显示：

- `over_max_length_count = 0`
  - 可以继续用 `DATA_MAX_LENGTH=4096`
- 有少量超长样本
  - 优先考虑把 `DATA_MAX_LENGTH` 提到 `6144`
- 超长样本占比明显
  - 不要直接硬开跑；应先决定：
    - 提升 `DATA_MAX_LENGTH`
    - 或单独过滤 / 复写过长样本

也就是说：

- **`DATA_MAX_LENGTH=4096` 在这条线里是 provisional，不是未经审计的固定常量**

### 5.4C Mandatory preflight: 1-sample dataset canary

在长度审计之后、正式训练之前，再跑一次 `1-sample canary`。

canary 脚本：

- [canary_step900_shortdiag_sft_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/canary_step900_shortdiag_sft_dataset.py)

作用：

- 使用训练时同一套 `MODEL_PATH`
- 直接实例化 `MultiTurnSFTDataset`
- 只取 `max_samples=1`
- 读取 `dataset[0]`
- 输出：
  - `input_ids_len`
  - `attention_mask_sum`
  - `loss_mask_sum`
  - 各张量 shape

推荐执行口径：

```bash
cd /Users/roger/Desktop/coding_RL_project/verl

python3 "/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/canary_step900_shortdiag_sft_dataset.py" \
  --model_path "/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_curriculum_step620_v3q3_to900_seed0/global_step_900/actor_merged_hf_phase4_repair" \
  --train_parquet "/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_train_v1.parquet" \
  --max_length 6144 \
  --pad_mode no_padding \
  --truncation error
```

如果这条 canary 正常返回 `status=ok`，再进入正式 SFT。

### 5.4A Init model must be explicit

这轮 guide 里的训练命令必须显式写死 `MODEL_PATH`。

原因：

- 当前 launcher 的默认回退逻辑仍然会去找旧的 `step200` checkpoint
- 如果不显式传 `MODEL_PATH` / `MERGED_MODEL_DIR` / `STEP200_CKPT_DIR`，
  就可能静默训练到错误的 base model

因此这轮当前推荐的口径是：

- `MODEL_PATH` 指向与当前 `step900` repair 受控评测一致的 merged HF 目录

推荐路径：

```text
/workspace/verl_repo/checkpoints/global_step_900/actor_merged_hf_step900_candidate_v2_backend_rr
```

如果目标机器上该目录不存在，应先从同一个 `global_step_900` checkpoint merge 出来并冻结路径，
再开始 SFT；不要依赖 launcher 的历史 fallback。

### 5.5 Example launch command

下面是一条适配当前 pure shortdiag 数据的**正式首轮实验**示意命令：

```bash
cd /Users/roger/Desktop/coding_RL_project/verl

MODEL_PATH="/workspace/verl_repo/checkpoints/global_step_900/actor_merged_hf_step900_candidate_v2_backend_rr" \
DISABLE_INTERNAL_VAL=true \
MIX_TRAIN_FILE="/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_train_v1.parquet" \
MIX_MANIFEST_FILE="/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step900_shortdiag_pure_v1/step900_shortdiag_pure_sft_dataset_v1.summary.json" \
EXPERIMENT_NAME="phase2_step900_shortdiag_pure_sft_v1" \
TOTAL_TRAINING_STEPS=60 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 \
ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
USE_REMOVE_PADDING=true \
DATA_MAX_LENGTH=6144 \
LR=1e-6 \
SAVE_FREQ=10 \
MAX_CKPT_TO_KEEP=4 \
TEST_FREQ=0 \
bash "/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/ops/run_phase2_repair_sft.sh" 4
```

如果 `SP=2` 仍然有显存压力，再按这个顺序往上加：

1. 保持 `TRAIN_BATCH_SIZE=8` 不变，先尝试 `ULYSSES_SEQUENCE_PARALLEL_SIZE=4`
2. 仍不够时，再考虑 `FSDP_CPU_OFFLOAD=true`
3. 只有这些 trainer 内建显存旋钮都不足时，才回退到更小的 global batch

注意：

- `USE_REMOVE_PADDING=true` 单独打开并不能走到真正的变长序列路径；要和 `ULYSSES_SEQUENCE_PARALLEL_SIZE>1` 一起开才会进入 trainer 的 SP branch
- 因此这轮真正优先的显存控制组合是：
  - `ULYSSES_SEQUENCE_PARALLEL_SIZE=2`
  - `USE_REMOVE_PADDING=true`

这里 `DATA_MAX_LENGTH=6144` 是基于当前风险评估给出的更稳妥默认值；
如果 tokenizer-length audit 证明 `4096` 已足够，则可以调回 `4096`。

这里 `TEST_FREQ=0` 即使不显式传，
在 `DISABLE_INTERNAL_VAL=true` 下 launcher 也会自动兜成 `0`；
这里写出来只是为了让审核更直观。

如果在远端执行：

- `cd`
- `MIX_TRAIN_FILE`
- `MIX_MANIFEST_FILE`
- 启动脚本路径

都应替换成远端绝对路径；
但 `MODEL_PATH` 继续必须显式保留。

---

## 6. Why No Internal Train/Val/Test Split For This Iteration

当前不做内部 split 是一个**工程上有意识的选择**，不是漏做。

主要理由：

1. 当前 pure shortdiag 数据只有 `182` 条
2. 这轮主要目标是快速验证：
   - repair-conditioned SFT 是否有正增益
   - `short_diagnosis+code` 这条 instruction family 是否值得继续做
3. 更关键的模型比较会放到外部：
   - `valid_big500 repair eval`

因此当前最自然的第一轮口径是：

- 用全部 `182` 条做 train
- 保存多个 checkpoint
- 用外部 `valid_big500` 做 repair eval 比较 checkpoint

### 6.1 Caveat

这个口径的代价也必须写清楚：

- `valid_big500` 会成为当前 SFT 线的开发集
- 因此它不再是严格独立的最终测试集

这不影响当前“先把 repair-conditioned SFT 跑通”的目标，
但后面如果要写正式结论，
需要额外准备独立的 holdout 或更严格的测试协议。

---

## 7. Audit Checklist For Another Agent

另一个 agent 审这条线时，建议按下面顺序检查。

### 7.1 Data lineage

确认以下链条是否闭合：

1. `step900_candidate_v2_backend_rr` 是 canonical queue
2. teacher QC 使用的是 `patched sandbox + direct-backend client RR`
3. salvage `32 -> 16 accepted` 已正确并回 keep pool
4. request-unique 选择逻辑与 summary 一致
5. pure shortdiag 过滤后剩 `182` 行

### 7.2 Data integrity

重点核对：

- final parquet 是否真的是 `182` rows
- 是否全部 `teacher_prompt_mode = short_diagnosis_code`
- 是否全部 `stability_label = accept_2_of_2`
- 是否全部来自 `codecontests_train_wo_valid_big`

### 7.3 Training launcher

重点核对：

- `run_phase2_repair_sft.sh` 是否支持当前 `row_count` 风格 manifest
- `DISABLE_INTERNAL_VAL=true` 时是否会：
  - 传 `data.val_files=[]`
  - 把 `trainer.test_freq` 置成 `0`
- 当前 command 是否确实指向 pure shortdiag parquet，而不是旧的 `final_v1` 混合数据

---

## 8. Minimal Reviewer Verdict Target

如果另一个 agent 要快速给结论，我建议他至少回答这 4 个问题：

1. 当前 `182-row pure shortdiag` parquet 是否来自正确的数据处理链？
2. 当前数据是否满足：
   - request-unique
   - pure shortdiag
   - accept_2_of_2
   - no eval contamination
3. 当前 launcher 是否真的支持“无内部 val”训练？
4. 以这份数据做第一轮 train-only SFT，再用 `valid_big500 repair eval` 选 ckpt，这个实验设计是否自洽？

如果这 4 个问题答案都为“是”，当前这条 SFT 线就可以进入训练执行阶段。
