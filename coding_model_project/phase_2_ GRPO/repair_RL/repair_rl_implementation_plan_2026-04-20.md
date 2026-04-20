# Repair RL Implementation Plan (2026-04-20)

这份文档的目标是：

- 把当前已经确认的 `data design + reward design + engineering spec`
- 合并成一份可以直接执行的最终实现计划

相关文档：

- [repair_data_design_2026-04-20.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_RL/repair_data_design_2026-04-20.md)
- [repair_reward_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_RL/repair_reward_design.md)
- [repair_rl_engineering_spec_2026-04-20.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_RL/repair_rl_engineering_spec_2026-04-20.md)

---

## 1. Final Goal

当前要落地的不是完整大规模 second-pass repair RL 体系，
而是一轮：

- 数据口径明确
- reward 口径明确
- 实现改动面尽量小
- 能快速判断是否有 signal

的 `repair RL v0 probe`。

本轮正式目标写死为：

1. 构建一份可直接喂给 `verl` 的 repair RL parquet
2. 用共享 verifier 跑 `repair_delta_v0`
3. 启动一轮小规模 repair RL probe
4. 用已有 repair eval 协议判断是否存在正信号
5. 只有在 `v0` 有信号后，才升级到 `repair_delta_edit_v1`

---

## 2. Final Scope

### 2.1 当前实现只覆盖 `repair_delta_v0`

当前正式执行顺序是：

```text
Phase A:
    先实现 repair_delta_v0

Phase B:
    只有在 v0 确认有信号后
    再实现 repair_delta_edit_v1
```

因此这份 implementation plan 的主线默认是：

- `repair_delta_v0` first

### 2.2 当前 probe 的正式数据范围

当前 data design 已确认：

- base table:
  - [student_references_step1300_repair_cond_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/student_references_step1300_repair_cond_v2.jsonl)
- join table #1:
  - [full_teacher_requests_step1300_repair_cond_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/full_teacher_requests_step1300_repair_cond_v2.jsonl)
- join table #2:
  - [codecontests_train_wo_valid_big_raw.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/raw/codecontests_train_wo_valid_big_raw.jsonl)

当前正式推荐的 probe 数据规模是：

```text
75 / 25 clean mix
= 231 high + 77 mid
= 308 rows
```

其中：

- high:
  - `first_pass_pass_ratio_all >= 0.6`
- mid:
  - `0.2 <= first_pass_pass_ratio_all < 0.6`

默认排除：

- audit-suspect rows
- `< 0.2` low / zero slice

### 2.3 冻结起始 checkpoint

在进入 Phase 1 之前，首轮 probe 的 init checkpoint 先写死：

- 主 probe init:
  - `step1300_rl`
- 可选对照 init:
  - `step1300_sft_v1_step60`

当前正式执行口径是：

- **Phase 1 ~ Phase 5 先只保证 `step1300_rl` 跑通**
- `step1300_sft_v1_step60` 只作为可选对照，不阻塞首轮实现

原因：

- 当前 held-out deployed base 仍是 `step1300_rl`
- `step1300_sft_v1_step60` 虽然有开发侧 repair 增益，
- 但还没有在 held-out `Protocol B` 上替代 `step1300_rl`

因此首轮 repair RL 主 probe 先回答：

- **从 `step1300_rl` 出发，是否能做出真实可解释的 held-out repair 增益**

为了避免实现时口径漂移，run script 必须满足：

- `MODEL_PATH` 显式传入
- 不使用模糊默认值来决定起始模型

### 2.4 冻结评测协议

首轮 repair RL probe 的评测协议先写死为两层。

#### Protocol A

用途：

- fixed-input repair-only diagnostic
- 观察“同一份坏代码 / 同一份 verifier feedback”下 second-pass 纯修复能力是否提升

当前正式口径：

- dataset:
  - `valid_big500`
- source first-pass:
  - `reuse_step900`
- trigger contract:
  - 延续当前 canonical Protocol A 主表

在首轮 repair RL 里，`Protocol A` 的角色是：

- **开发侧诊断表**
- **checkpoint 选择辅助表**

它不是首轮 headline gate。

#### Protocol B

用途：

- self-first-pass deployment metric
- 观察模型上线后“先自己生成 first pass，再自己 repair”的 end-to-end 效果

当前正式口径：

- dataset:
  - `codecontests_test`
- trigger:
  - 延续当前 Protocol B 部署式口径

在首轮 repair RL 里，`Protocol B` 的角色是：

- **主 gate**
- **最终 go / no-go 判据**

所以首轮 probe 的正式评测优先级写死为：

```text
primary gate:
    Protocol B / codecontests_test

secondary diagnostic:
    Protocol A / valid_big500 / reuse_step900
```

补充说明：

- `Protocol A + reuse_step1300`
  - 可以作为与 step1300-style cached failures 更贴近的 secondary analysis
  - 但不是 headline 表

### 2.5 冻结 prompt mode 口径

首轮 plan 不再把 `prompt_mode` 硬编码成 `short_diagnosis_code`。

当前正式要求改为：

- `prompt_mode` 必须是显式配置
- 支持值：
  - `code_only`
  - `short_diagnosis_code`

builder 和 run script 都应暴露这个配置，例如：

```bash
REPAIR_PROMPT_MODE=code_only
```

结合当前 `step1300` 相关证据，首轮主 probe 的正式默认值建议是：

- `prompt_mode = code_only`

原因：

- `short_diagnosis_code` 在 `step900` 上有正信号
- 但在 `step1300` 上不能外推成默认最优

因此更稳的执行口径是：

- 首轮主 probe:
  - `code_only`
- 如果 `v0` 流水线稳定，再做：
  - `short_diagnosis_code` prompt ablation

---

## 3. Final Deliverables

本轮实现完成后，应该产出以下东西。

### 3.1 代码产物

新增：

- [build_repair_rl_parquet.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_repair_rl_parquet.py)
- [repair_grpo_batch_reward.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_grpo_batch_reward.py)
- 建议新增一份 repair RL run script
  - 例如：
    - [run_repair_rl_probe.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/run_repair_rl_probe.sh)

按需修改：

- [batch.py](/Users/roger/Desktop/coding_RL_project/verl/verl/workers/reward_manager/batch.py)
  - `v0` 不一定需要改
  - `v1` 需要透传 `uid`

### 3.2 数据产物

建议输出目录：

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/data/repair_rl_parquet/step1300_probe_v0`

建议至少包含：

- `train.parquet`
- `smoke_train.parquet`
- `smoke_val.parquet`
- `build_summary.json`
- `selected_rows.jsonl` 或 `selected_manifest.jsonl`
- materialized `audit_suspect_blocklist.jsonl`
  - builder 输入 asset，可同时复制到输出目录留档

### 3.3 实验产物

至少需要：

- 一轮 step-smoke 训练日志
- 一轮正式 probe 训练日志
- 对应 checkpoint
- repair eval 结果
- 同时观察 raw code eval 是否被拖坏

---

## 4. Code Change Map

### 4.1 新建 builder

新增文件：

- [build_repair_rl_parquet.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_repair_rl_parquet.py)

职责：

1. 读取 `student_references`
2. 按 `(problem_id, prompt_sha256)` join `full_teacher_requests`
3. 按 `(problem_id, prompt_sha256)` join raw dataset
4. 做数据过滤与 high/mid 配比采样
5. 构造 second-pass repair prompt
6. 输出 repair RL parquet

### 4.2 新建 reward adapter

新增文件：

- [repair_grpo_batch_reward.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_grpo_batch_reward.py)

职责：

1. 接收 repair RL 样本
2. 对 repaired rollout 走 shared verifier
3. 读取 frozen first-pass fields
4. 计算 `repair_delta_v0`
5. 返回 `score + reward_raw + verifier summary + repair-specific logs`

### 4.2.1 Reward adapter 返回结构必须是扁平字段

这里的“verifier summary + repair-specific logs”必须收窄成：

- **只返回扁平标量字段**

允许的 per-sample value 类型应限制为：

- `float`
- `int`
- `bool`
- `str`

不允许通过 `reward_extra_info` 往 trainer 链路里塞：

- `dict`
- `list`
- `per_case_results`
- `test_cases`
- 完整 verifier summary 对象
- 任何嵌套 debug blob

原因：

- 当前 `BatchRewardManager` 会把这些字段直接塞进 `reward_extra_info`
- trainer / metric aggregation 会对非字符串字段做均值聚合
- 如果塞嵌套对象，验证和日志路径很容易直接炸

因此首轮正式口径写死为：

```text
reward_extra_info:
    flat scalar/string keys only
```

如果需要 richer debug dump，应走：

- builder summary
- standalone smoke script
- sidecar JSONL

而不是走 `reward_extra_info`

### 4.3 训练入口 wiring

当前已有 wiring 模式已经足够：

- [run_grpo_step_smoke.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/run_grpo_step_smoke.sh)
- [reward.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/ppo/reward.py)

当前正式建议是：

- 不改 trainer 主干配置机制
- 继续使用：
  - `reward_manager.name=batch`
  - `custom_reward_function.path=...`
  - `custom_reward_function.name=compute_score`

也就是说 repair RL 训练脚本只需要把 custom reward function 指到：

- `coding_model_project/src/repair_grpo_batch_reward.py`

### 4.4 `v1` 的唯一 trainer-side plumbing

只有当后续升级到 `repair_delta_edit_v1` 时，才需要最小改动：

- [batch.py](/Users/roger/Desktop/coding_RL_project/verl/verl/workers/reward_manager/batch.py)

要加的东西很小：

```python
merged_extra["uid"] = data.non_tensor_batch["uid"][i]
```

当前 `repair_delta_v0` 阶段不依赖这一步。

---

## 5. Phase Plan

## 5.1 Phase 0: Freeze Inputs

目标：

- 把本轮 probe 的输入口径固定下来

本阶段结论已经有了：

- 数据用 `student_references + full_teacher_requests + raw`
- 数据规模默认 `308`
- reward 先跑 `repair_delta_v0`
- init checkpoint 先固定为 `step1300_rl`
- 主 gate 先固定为 `Protocol B / codecontests_test`
- `prompt_mode` 改成显式配置，不再写死 `short_diagnosis_code`

本阶段无需再改代码。

### 5.1.1 Materialize audit-suspect blocklist

在 builder 真正开始写之前，先把 audit-suspect 排除口径从“描述性规则”变成“可复现文件”。

建议产出：

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_RL/assets/step1300_probe_v0_audit_suspect_blocklist.jsonl`

推荐生成来源：

- [step1300_shortdiag_pure_keep_set_v1.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_shortdiag_pure_v1/step1300_shortdiag_pure_keep_set_v1.jsonl)

或等价上游 keep set：

- [step1300_teacher_keep_set_v2.jsonl](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/step1300_repair/final_keep/step1300_teacher_keep_set_v2.jsonl)

blocklist 规则写死为：

```text
teacher_keep_source in {
    remaining_high_pass_audit_0p85_to_0p9,
    remaining_high_pass_audit_0p7_to_0p85,
    near_miss_testcase_audit_high_confidence,
    near_miss_testcase_audit_regen_round1,
}
```

materialized blocklist 至少保留：

- `problem_id`
- `prompt_sha256`
- `teacher_keep_source`

当前已知参考计数：

- upstream audit-derived rows:
  - `56`
- 与首轮 high/mid selected slice 交叉后应排除：
  - `37`

builder 默认应读取这个 materialized blocklist，
而不是每次在运行时临时猜规则。

### 5.2 Phase 1: Build Repair RL Parquet

目标：

- 先把数据变成能被 trainer 直接读的 parquet

#### Phase 1.1 Builder 需要实现的逻辑

在 [build_repair_rl_parquet.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_repair_rl_parquet.py) 中实现：

1. 加载 `student_references`
2. 过滤：
   - `accepted == false`
   - `finish_reason == "stop"`
   - `details.invalid_for_rl == false`
   - `first_pass_pass_ratio_all >= 0.2`
   - `source_split == "codecontests_train_wo_valid_big"`
3. 读取 materialized `audit_suspect_blocklist.jsonl`
4. 按 `(problem_id, prompt_sha256)` 排除 audit-suspect rows
5. 记录：
   - upstream blocklist row count
   - removed row count
6. 分成：
   - high slice
   - mid slice
7. 采样成：
   - `231 high + 77 mid`
8. join `full_teacher_requests` 补：
   - `repair_feedback`
   - `request_id`
   - `teacher_prompt_mode`
9. join raw dataset 补：
   - `test_cases`
10. 用显式配置的 `prompt_mode` 构造 repair prompt
11. 把实际使用的 `prompt_mode` 写入 metadata
12. 写 parquet 与 summary

#### Phase 1.2 Prompt 构造策略

优先方案：

- 直接复用：
  - [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)

具体是：

- 优先直接使用 join 到的 `repair_feedback`
- 再用 `build_codecontests_repair_prompt(...)` 构造 second-pass prompt

注意：

- `prompt_mode` 必须是 builder 参数
- 不能在 schema 中写死为 `short_diagnosis_code`

这样做的好处是：

- 与当前 repair eval 协议保持一致
- 不会凭空再造一套 repair prompt 口径

#### Phase 1.3 输出 record 结构

builder 输出的 parquet record 应保持和当前 RL outer schema 一致：

```python
{
    "data_source": "codecontests_repair_rl",
    "prompt": [...],
    "ability": "code",
    "reward_model": {
        "style": "rule",
        "ground_truth": {...}
    },
    "extra_info": {...}
}
```

其中 `ground_truth` 至少包含：

```python
{
    "problem_id": str,
    "dataset": str,
    "test_cases": {...},
    "first_pass": {
        "code": str,
        "pass_ratio_all": float,
        "accepted": bool,
        "error_type": str,
        "invalid_for_rl": bool,
        "invalid_reason": str,
        "finish_reason": str,
        "pass_ratio_bucket": str,
    },
    "repair_metadata": {
        "prompt_mode": str,
        "source_run_id": str,
        "source_protocol": str,
    },
}
```

### 5.3 Phase 2: Implement `repair_delta_v0`

目标：

- 让 reward adapter 跑通，并与现有 shared verifier 对齐

在 [repair_grpo_batch_reward.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_grpo_batch_reward.py) 中实现：

1. 保持 `compute_score(...)` 签名尽量贴近 [grpo_batch_reward.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/grpo_batch_reward.py)
2. 调用：
   - `normalize_candidate`
   - `verify_candidate_batch`
3. 从 repaired rollout 读取：
   - `pass_ratio_all`
   - `accepted`
   - `error_type`
   - `invalid_for_rl`
   - `invalid_reason`
   - `extraction_status`
   - `judge_time_s`
4. 从 `ground_truth.first_pass` 读取：
   - `p0`
   - `a0`
   - 其他 first-pass metadata
5. 计算：
   - `q0 = 0.8 * p0 + 0.2 * a0`
   - `q1 = 0.8 * p1 + 0.2 * a1`
   - `delta_pos`
   - `delta_neg`
   - `accepted_gain`
6. 应用 guardrails：
   - infra invalid -> `reward_raw = NaN`, `score = 0`
   - bad output -> `reward = -1`
7. 返回：
   - `score`
   - `reward_raw`
   - 扁平 verifier fields
   - `q0/q1/delta_q/...`

#### Phase 2.0 `repair_delta_v0` 的输出范围

在进入实现前，把 `v0` 的输出口径也写死：

- valid sample 上：
  - 直接使用 `base_reward`
  - **不再额外做第二层 clip**
- analytic valid range:
  - `[-0.75, 1.5]`
- 连同 guardrail 后的 emitted range:
  - invalid sample:
    - `reward_raw = NaN`, `score = 0.0`
  - bad output:
    - `reward = -1.0`
  - 所以整体输出口径可读成：
    - `[-1.0, 1.5] + INVALID_FOR_RL`

这样首轮 probe 的 reward 含义更清楚：

- `repair_delta_v0` 是有界但不再二次压平的 improvement reward
- `repair_delta_edit_v1` 再单独考虑后续 clip / edit penalty 口径

#### Phase 2.1 当前不做的事情

当前 `v0` 不做：

- edit ratio
- edit budget
- candidate gate
- group gate
- uid plumbing

这些全部留给 `repair_delta_edit_v1`。

### 5.4 Phase 3: Wire Training Entry

目标：

- 用最小配置改动把 repair RL 跑起来

建议新增：

- [run_repair_rl_probe.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/run_repair_rl_probe.sh)

这个脚本直接参考：

- [run_grpo_step_smoke.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/run_grpo_step_smoke.sh)

但改成：

1. `DATA_DIR` 指向 repair RL parquet 输出目录
2. `MODEL_PATH` 必须显式传入
3. `REPAIR_PROMPT_MODE` 必须显式传入
4. `custom_reward_function.path` 指向：
   - `coding_model_project/src/repair_grpo_batch_reward.py`
5. `reward_mode` 默认设成：
   - `repair_delta_v0`
6. `data.train_files` 使用：
   - `train.parquet`
7. `data.val_files` 使用：
   - `smoke_val.parquet` 或一份小型 repair val parquet
8. 首轮主 probe 默认：
   - `MODEL_PATH = step1300_rl`
   - `REPAIR_PROMPT_MODE = code_only`

### 5.5 Phase 4: Smoke and Builder Validation

目标：

- 先确认数据和 reward contract 是通的

本阶段必须先完成下面几件事。

#### Phase 4.1 Builder smoke

检查：

1. parquet 可成功写出
2. 行数符合预期：
   - `train = 308`
   - smoke splits 有内容
3. `ground_truth.first_pass.code` 非空
4. `ground_truth.test_cases` 非空
5. `extra_info.problem_id` / `dataset` / `split` 正常
6. `repair_metadata.prompt_mode` 等于显式传入的 `prompt_mode`
7. blocklist 相关计数与预期一致：
   - upstream blocklist rows = `56`
   - selected-slice removed rows = `37`

#### Phase 4.2 Reward adapter smoke

检查：

1. `compute_score(...)` 在小 batch 上可运行
2. 返回结构可被 [batch.py](/Users/roger/Desktop/coding_RL_project/verl/verl/workers/reward_manager/batch.py) 接受
3. `reward_extra_info` 中能看到：
   - `q0`
   - `q1`
   - `delta_q`
   - `accepted_gain`
4. invalid / bad-output 路径符合预期
5. `reward_extra_info` 只包含扁平标量 / 字符串键

#### Phase 4.3 Step-smoke train

建议先跑一个极小 smoke：

- very small train parquet
- `ROLLOUT_N=1`
- 小 batch
- 短步数

主要检查：

- trainer 能否读 parquet
- reward function 能否稳定跑完
- 不出现大量 schema / join / verifier contract 错误

### 5.6 Phase 5: Run `repair_delta_v0` Probe

目标：

- 跑第一轮正式 repair RL probe

本轮 probe 的正式建议：

```text
data:
    308 rows

reward:
    repair_delta_v0

training:
    one short but real GRPO run
```

这轮 probe 的判断标准不是“直接做成最终主线”，而是：

1. repair eval 是否比当前 base 有正向移动
2. reward 是否稳定
3. group invalid / truncation / verifier noise 是否可控
4. raw code accuracy 是否被明显拖坏

### 5.6.1 这轮 probe 的固定评测表

首轮 probe 跑完后，至少要出下面这三张表：

1. `Protocol B / codecontests_test`
   - 这是主 gate
   - 比较对象默认是：
     - `step1300_rl` baseline
     - `repair_rl_from_step1300_rl`
2. `Protocol A / valid_big500 / reuse_step900`
   - 这是修复能力诊断表
   - 用来看 second-pass repair skill 是否更强
3. raw code eval / canary panel
   - 用来看 raw code ability 是否被明显拖坏

如果后面再跑对照 init：

- `step1300_sft_v1_step60 -> repair RL`

则这三张表再追加对照列，但不阻塞首轮主 probe。

### 5.7 Phase 6: Decide on `repair_delta_edit_v1`

只有当 `v0` 满足下面条件时，才进入 `v1`：

1. repair eval 有正向 signal
2. reward 日志看起来合理
3. 没出现明显的 reward collapse
4. 训练中不是主要被格式错误 / truncation 驱动

进入 `v1` 后才做：

1. 在 [batch.py](/Users/roger/Desktop/coding_RL_project/verl/verl/workers/reward_manager/batch.py) 透传 `uid`
2. 在 reward adapter 中加入：
   - line edit ratio
   - token edit ratio
   - hybrid edit ratio
   - dynamic budget
   - candidate gate
   - weak group gate

---

## 6. Builder Implementation Checklist

builder 落地时建议按下面顺序写。

### 6.1 数据加载与 join

- 读 `student_references`
- 建立 `(problem_id, prompt_sha256)` 索引
- 读 `full_teacher_requests`
- 读 raw dataset
- 校验 join miss 是否为 `0`

### 6.2 过滤与采样

- 先做 clean base filter
- 再做 audit-suspect 排除
- 再按 `first_pass_pass_ratio_all` 分 high/mid
- 再按 `75/25` 目标采样

### 6.3 prompt 与 record 生成

- 构造 repair prompt
- 生成 RL record
- 生成 summary
- 输出 parquet

### 6.4 builder summary 必须记录的字段

`build_summary.json` 至少应包含：

- `input_row_count`
- `failed_row_count`
- `selected_high_count`
- `selected_mid_count`
- `excluded_audit_suspect_count`
- `upstream_audit_blocklist_count`
- `final_train_count`
- `join_miss_request_count`
- `join_miss_raw_count`
- `error_type_counts`
- `curriculum_bucket_counts`
- `repair_stratum_counts`
- `prompt_mode`

---

## 7. Reward Adapter Implementation Checklist

### 7.1 先复制当前适合复用的骨架

直接以：

- [grpo_batch_reward.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/grpo_batch_reward.py)

为骨架开始写，复用：

- `compute_score(...)` 结构
- truncated guardrail 逻辑
- verifier batch 调用模式

### 7.2 `repair_delta_v0` 至少返回这些字段

建议至少返回：

- `score`
- `reward_raw`
- `accepted`
- `pass_ratio_all`
- `error_type`
- `invalid_for_rl`
- `invalid_reason`
- `finish_reason`
- `truncated_by_max_tokens`
- `q0`
- `q1`
- `delta_q`
- `delta_pos`
- `delta_neg`
- `accepted_gain`
- `first_pass_pass_ratio_all`
- `first_pass_accepted`
- `first_pass_bucket`
- `problem_id`

并明确禁止返回：

- `dict`
- `list`
- `per_case_results`
- `test_cases`
- 任何嵌套 verifier / debug 结构

### 7.3 当前 reward adapter 不要扩大 scope

当前不要顺手做：

- filter_groups
- 新 verifier fork
- 多轮 repair
- process reward
- extra teacher-based shaping

保持 reward adapter 足够窄，便于判断 signal。

---

## 8. Validation Plan

## 8.1 数据验证

必须检查：

1. `train.parquet` 条数是否对
2. high/mid 比例是否对
3. 每条样本都有：
   - `first_pass.code`
   - `first_pass.pass_ratio_all`
   - `repair_feedback`
   - `test_cases`
4. `prompt_mode` 是否与本轮配置一致
5. audit-suspect 排除是否来自固定 blocklist 文件

## 8.2 reward 验证

必须检查：

1. `q0` 与 `first_pass_pass_ratio_all` 一致
2. repaired rollout 变好时 `delta_q > 0`
3. repaired rollout 变坏时 `delta_neg > 0`
4. AC 时 `accepted_gain == 1`
5. invalid / truncation 返回 `score = 0`
6. `reward_raw` 的输出口径符合：
   - valid sample `[-0.75, 1.5]`
   - bad output `-1.0`

## 8.3 训练验证

必须检查：

1. trainer 不报 schema 错
2. reward extra info 正常写出
3. invalid group rate 不异常
4. 训练 early steps 没有明显 reward explode / collapse

## 8.4 probe 成功判据

本轮 probe 至少要回答：

1. 是否比当前 base 有 repair 指标改善？
2. 是否没有明显伤害 raw code ability？
3. 是否说明 second-pass RL 比继续堆 repair-SFT 更有希望？

其中正式主 gate 写死为：

- `Protocol B / codecontests_test`

`Protocol A / valid_big500 / reuse_step900` 是必须出的诊断表，
但不是首轮 headline gate。

---

## 9. What Not To Do

为了控制 blast radius，本轮明确不做：

1. 不把 `< 0.2` slice 加进首轮 probe
2. 不把 audit-suspect 样本混进首轮 probe
3. 不直接上 `repair_delta_edit_v1`
4. 不改 shared verifier truth contract
5. 不改 trainer 核心 advantage 逻辑
6. 不把这轮 probe 扩成多轮 repair RL 主线

---

## 10. Final Execution Order

当前正式执行顺序建议写死为：

1. 实现 [build_repair_rl_parquet.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/build_repair_rl_parquet.py)
2. 生成 `308` 行 repair RL parquet
3. 实现 [repair_grpo_batch_reward.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_grpo_batch_reward.py)
4. 新增 [run_repair_rl_probe.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/run_repair_rl_probe.sh)
5. 跑 builder smoke
6. 跑 reward smoke
7. 跑 step-smoke training
8. 跑正式 `repair_delta_v0` probe
9. 做 `Protocol B / codecontests_test` + `Protocol A / valid_big500 / reuse_step900` + raw code eval
10. 只有确认有 signal 后，再进入 `repair_delta_edit_v1`

---

## 11. Final Recommendation

当前最合理的实现路径不是一次性把 repair RL 做成完整体系，
而是先用：

- 已确认的数据口径
- 已确认的 shared verifier contract
- 最小新增 builder
- 最小新增 reward adapter

快速做出一轮 `repair_delta_v0` probe。

这条路径的优点是：

- 工程改动面小
- 与现有 GRPO 主线兼容
- 可以最快验证“repair RL 是否真的比继续扩 repair-SFT 更值”

如果这轮 probe 没信号，
你损失的是一轮受控实现成本；
如果它有信号，
下一步再把 `repair_delta_edit_v1` 接上就很顺。
