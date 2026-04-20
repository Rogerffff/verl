# Step1300 Shortdiag Pure SFT Training Guide

## 1. Goal

这份文档用于在 **正式启动 `step1300 short_diagnosis_code pure SFT` 之前**，把：

- 当前训练数据资产
- 选定的初始模型
- 预检步骤
- 建议训练参数
- 远端磁盘空间与可删 checkpoint

一次性写清楚，方便 review 后再开跑。

当前状态：

- `step1300 shortdiag pure` 数据已经构建完成
- 远端 `vastai5` 上 **尚未启动正式 SFT**
- 预检的 `tokenizer-length audit / 1-sample canary` 也**故意暂停**，等待这份 guide 先被 review

---

## 2. Final Data Artifacts

本轮最终使用的是 **request-unique, pure `short_diagnosis_code`** 数据，而不是原始 keep set 直接进 trainer。

### 2.1 Keep Set

当前最新 keep set：

- [step1300_teacher_keep_set_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.jsonl)
- [step1300_teacher_keep_set_v2.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.summary.json)

关键统计：

- `553` rows
- `553` unique `request_id`
- 全部 `teacher_prompt_mode = short_diagnosis_code`
- 全部 `source_split = codecontests_train_wo_valid_big`

来源分布：

- `primary_qc = 389`
- `regen_round1_qc = 96`
- `regen_round2_qc = 12`
- `near_miss_testcase_audit_high_confidence = 13`
- `near_miss_testcase_audit_regen_round1 = 11`
- `remaining_high_pass_audit_0p85_to_0p9 = 11`
- `remaining_high_pass_audit_0p7_to_0p85 = 21`

repair strata 分布：

- `CoreNearMiss = 119`
- `ExpansionB = 118`
- `CurrentCRecoverable = 316`

curriculum bucket 分布：

- `B_near_miss = 237`
- `C_hard_partial = 316`

### 2.2 Trainer-Ready SFT Dataset

trainer 直接使用的数据资产：

- pure keep set:
  - [step1300_shortdiag_pure_keep_set_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_keep_set_v1.jsonl)
- SFT jsonl:
  - [step1300_shortdiag_pure_sft_dataset_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.jsonl)
- train parquet:
  - [step1300_shortdiag_pure_train_v1.parquet](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_train_v1.parquet)
- summary:
  - [step1300_shortdiag_pure_sft_dataset_v1.summary.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.summary.json)

长度审计后，实际训练采用的是 **`len<=6144` 的过滤版**：

- filtered train parquet:
  - `/workspace/verl_repo/coding_model_project/phase_2_GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_train_len6144_v1.parquet`
- filtered summary:
  - `/workspace/verl_repo/coding_model_project/phase_2_GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_len6144_v1.summary.json`

已经本地验证：

- keep jsonl `553` 行
- SFT jsonl `553` 行
- parquet `553` 行
- `messages` 为标准两轮：
  - `user`
  - `assistant`
- assistant 格式为：
  - `BUG_SUMMARY: ...`
  - `FIX_PLAN: ...`
  - `<code>...</code>`

实际训练前又额外做了一层长度过滤：

- 原始 `553` rows
- `>6144 tokens` 的样本 `7` 条
- 过滤后训练集 `546` rows

生成这批资产的脚本：

- [build_step1300_shortdiag_pure_sft_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/build_step1300_shortdiag_pure_sft_dataset.py)

---

## 3. Chosen Init Model

这轮 SFT 的推荐初始化模型是远端 `step1300` 的 merged HF 目录：

```bash
/workspace/verl_repo/checkpoints/rlvr_coding_model/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3/global_step_1300/actor_merged_hf_step1300_repair_cond_v2
```

选择理由：

1. 这是当前 `step1300` 主线上已经用于 repair 条线的 HF artifact，和现有 Step 4 repair 资产口径一致
2. 远端已确认是完整 HF 模型目录，包含：
   - `config.json`
   - `tokenizer_config.json`
   - `tokenizer.json`
   - `special_tokens_map.json`
   - `model.safetensors.index.json`
3. 比直接用 `actor/huggingface` 更明确，避免不小心切回别的 export 目录

因此，这轮 guide 里的训练命令必须显式写死：

- `MODEL_PATH=/workspace/verl_repo/checkpoints/.../actor_merged_hf_step1300_repair_cond_v2`

不要依赖 launcher 的 fallback。

---

## 4. Remote Paths On `vastai5`

远端 repo 根目录：

```bash
/workspace/verl_repo
```

远端 step1300 SFT 数据目录：

```bash
/workspace/verl_repo/coding_model_project/phase_2_GRPO/sft_repair_data/step1300_shortdiag_pure_v1
```

关键文件：

- train parquet
  - `/workspace/verl_repo/coding_model_project/phase_2_GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_train_v1.parquet`
- dataset summary
  - `/workspace/verl_repo/coding_model_project/phase_2_GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_v1.summary.json`

---

## 5. Recommended Training Recipe

当前推荐先沿用 `step900 shortdiag pure` 那轮已经验证过的训练思路，只把：

- init model
- train parquet
- experiment name

换成 `step1300` 版本。

### 5.1 Core Settings

建议值：

- `DISABLE_INTERNAL_VAL=true`
- `TOTAL_TRAINING_STEPS=60`
- `TRAIN_BATCH_SIZE=8`
- `MICRO_BATCH_SIZE_PER_GPU=1`
- `ULYSSES_SEQUENCE_PARALLEL_SIZE=2`
- `USE_REMOVE_PADDING=true`
- `LR=1e-6`
- `SAVE_FREQ=10`
- `MAX_CKPT_TO_KEEP=6`
- `TEST_FREQ=0`

说明：

- `60 steps` 对 `553-row` 语料已经不是 canary，而是第一轮正式实验
- `SAVE_FREQ=10` 可以留下足够对比点，同时明显省磁盘
- `MAX_CKPT_TO_KEEP=6` 对应 `10/20/30/40/50/60` 六个 checkpoint，便于后续对照
- `ULYSSES_SEQUENCE_PARALLEL_SIZE=2 + USE_REMOVE_PADDING=true` 仍然是第一优先级的显存控制旋钮
- 暂不做内部 val split，继续走外部 repair eval / valid500 评估

### 5.2 Data Max Length

这里 **不要先拍死成 `4096`**。

建议口径：

- 先跑 tokenizer-length audit
- 如果 `6144` 内整体安全，就直接用 `6144`
- 如果只有极少数样本超过 `6144`，优先构建 `len<=6144` 的过滤版 parquet，再保持 `DATA_MAX_LENGTH=6144`
- 只有当审计证明 `4096` 足够时，再收回到 `4096`

---

## 6. Mandatory Preflight Before Launch

### 6.1 Tokenizer-Length Audit

先在远端用**训练时同一个 `MODEL_PATH`** 对 `553-row parquet` 做 token-length 审计。

推荐命令：

```bash
cd /workspace/verl_repo

python3 - <<'PY'
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

model_path = "/workspace/verl_repo/checkpoints/rlvr_coding_model/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3/global_step_1300/actor_merged_hf_step1300_repair_cond_v2"
parquet_path = "/workspace/verl_repo/coding_model_project/phase_2_GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_train_v1.parquet"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
table = pq.read_table(parquet_path)
rows = table.to_pylist()

totals = []
for row in rows:
    ids = tokenizer.apply_chat_template(
        row["messages"],
        add_generation_prompt=False,
        tokenize=True,
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    totals.append(len(ids))

totals_sorted = sorted(totals)
def pct(p):
    idx = min(len(totals_sorted) - 1, max(0, round((len(totals_sorted) - 1) * p)))
    return int(totals_sorted[idx])

report = {
    "sample_count": len(totals),
    "p50": pct(0.50),
    "p90": pct(0.90),
    "p95": pct(0.95),
    "max": int(max(totals_sorted)),
    "over_4096": int(sum(x > 4096 for x in totals)),
    "over_6144": int(sum(x > 6144 for x in totals)),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
```

本轮实际审计结果是：

- `sample_count = 553`
- `p50 = 1263`
- `p90 = 1999`
- `p95 = 2467`
- `p99 = 6579`
- `max = 13241`
- `over_4096 = 12`
- `over_6144 = 7`

因此本轮没有直接用原始 `553-row` parquet，而是额外构建了：

- `step1300_shortdiag_pure_train_len6144_v1.parquet`

也就是**只丢掉那 `7` 条超长样本**，其他 `546` 条完整保留。

### 6.2 1-Sample Canary

长度审计之后，再跑一次 `1-sample canary`，确认 trainer dataset 真能吃进去。

当前复用的 canary 脚本是：

- [canary_step900_shortdiag_sft_dataset.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/sft_repair_data/scripts/canary_step900_shortdiag_sft_dataset.py)

虽然文件名里是 `step900`，但它本质是通用脚本，可以通过 `--train_parquet` 指向 `step1300` 数据。

远端命令：

```bash
cd /workspace/verl_repo

python3 "/workspace/verl_repo/coding_model_project/phase_2_GRPO/sft_repair_data/scripts/canary_step900_shortdiag_sft_dataset.py" \
  --model_path "/workspace/verl_repo/checkpoints/rlvr_coding_model/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3/global_step_1300/actor_merged_hf_step1300_repair_cond_v2" \
  --train_parquet "/workspace/verl_repo/coding_model_project/phase_2_GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_train_len6144_v1.parquet" \
  --max_length 6144 \
  --truncation error
```

通过标准：

- 输出 `status = ok`
- `dataset_len_after_max_samples = 1`
- `loss_mask_sum > 0`

本轮实际 canary 已通过：

- `dataset_len_after_max_samples = 1`
- `input_ids_len = 1253`
- `loss_mask_sum = 376`
- `status = ok`

只有这两步都过，才进入正式训练。

---

## 7. Example Launch Command

这条是 **review 通过后** 才执行的正式训练命令。

```bash
cd /workspace/verl_repo

MODEL_PATH="/workspace/verl_repo/checkpoints/rlvr_coding_model/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3/global_step_1300/actor_merged_hf_step1300_repair_cond_v2" \
DISABLE_INTERNAL_VAL=true \
MIX_TRAIN_FILE="/workspace/verl_repo/coding_model_project/phase_2_GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_train_len6144_v1.parquet" \
MIX_MANIFEST_FILE="/workspace/verl_repo/coding_model_project/phase_2_GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_sft_dataset_len6144_v1.summary.json" \
EXPERIMENT_NAME="phase2_step1300_shortdiag_pure_sft_v1_len6144_keep6" \
TOTAL_TRAINING_STEPS=60 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 \
ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
USE_REMOVE_PADDING=true \
DATA_MAX_LENGTH=6144 \
LR=1e-6 \
SAVE_FREQ=10 \
MAX_CKPT_TO_KEEP=6 \
TEST_FREQ=0 \
bash "/workspace/verl_repo/coding_model_project/phase_2_GRPO/ops/run_phase2_repair_sft.sh" 4
```

注意：

- `run_phase2_repair_sft.sh` 当前在 `DISABLE_INTERNAL_VAL=true` 下会把：
  - `data.val_files=[]`
  - `trainer.test_freq=0`
- 这在**当前仓库版本**里是安全的，因为 `verl/trainer/fsdp_sft_trainer.py` 已经先判断 `test_freq > 0` 再做 modulo

---

## 8. Why No Internal Train/Val/Test Split For This Iteration

这轮延续 `step900 shortdiag pure` 的决策：

- 不在 `553-row` 里再切内部 val/test
- 训练过程中只存 checkpoint
- checkpoint 优劣继续通过外部评测判断：
  - `valid_big500`
  - `repair Protocol A / Protocol B`
  - 必要时再看 `codecontests_test`

这样做的原因：

1. 这批 repair-conditioned 样本本来就不大
2. 你当前更关心的是“repair skill shaping 是否有效”，而不是内部 val curve 漂不漂亮
3. 当前项目的主要可信评估还是 external eval，不是内部 split

---

## 9. Remote Space Review

`vastai5` 在清理完成后的空间：

```text
Filesystem: /workspace
Size: 700G
Used: 214G
Avail: 487G
```

最大的 checkpoint 目录：

- `114G`
  - `/workspace/verl_repo/checkpoints/rlvr_coding_model/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3/global_step_1300`
- `57G`
  - `/workspace/verl_repo/checkpoints/rlvr_coding_model/step900_sft/global_step_40`

其他占用：

- `/workspace/verl_repo/coding_model_project/outputs` 总共约 `2.4G`
- 不是当前主要瓶颈

### 9.1 What Looks Safe To Delete

**已删除：`step1000_import/global_step_1000`**

- 已回收：
  - 约 `86G`

**已删除：`/workspace/verl_repo/checkpoints/global_step_900`**

- 已回收：
  - 约 `114G`

**已删除：`/root/.cache/huggingface`**

- 已回收：
  - 约 `24G`

**仍可删除候选：`step900_sft/global_step_40`**

- 路径：
  - `/workspace/verl_repo/checkpoints/rlvr_coding_model/step900_sft/global_step_40`
- 可回收：
  - 约 `57G`
- 适合删除的前提：
  - 你已经不打算继续做 `step40` 的额外 repair eval / regression compare

这块目前更像“可删但不如 `step1000_import` 那么优先”。

### 9.2 What Should Not Be Deleted

不建议删：

- `/workspace/verl_repo/checkpoints/rlvr_coding_model/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3/global_step_1300`

因为这就是本轮 `step1300 shortdiag pure SFT` 的初始化 checkpoint 根目录。

### 9.3 Space Risk Judgment

当前 `487G` 空闲，在已经完成上述清理的前提下，`MAX_CKPT_TO_KEEP=6` 是可承受的。

原因是：

仍需关注的风险是：

- 每个 checkpoint 都会保存 `hf_model`
- `save_freq=10` 且 `total_training_steps=60` 会落满 `6` 个 checkpoint
- 如果训练后还要立刻接 repair eval，则最好继续保留额外 `50G+` 余量

---

## 10. Sandbox Status

当前 `vastai5` 上：

- `8081..8088` 这 8 个 sandbox backend 进程都在运行
- 它们对后续 repair eval 是够用的

但这轮 **SFT 本身不依赖 sandbox**。  
所以 sandbox 的存在主要是为了训练完成后的 eval，不是 SFT 启动前的 blocker。

---

## 11. Recommended Next Actions

在这份 guide review 通过之前，不再启动正式 SFT。

推荐顺序：

1. review 这份 guide 中的：
   - init model
   - dataset path
   - `TOTAL_TRAINING_STEPS / SAVE_FREQ / MAX_CKPT_TO_KEEP`
   - 删除策略
2. 如果同意，先删除：
   - `step1000_import/global_step_1000`
3. 然后执行：
   - tokenizer-length audit
   - 1-sample canary
4. 两步都过后，再正式启动：
   - `phase2_step1300_shortdiag_pure_sft_v1`
