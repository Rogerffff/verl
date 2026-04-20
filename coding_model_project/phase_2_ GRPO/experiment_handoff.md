# GRPO 实验接手文档（2026-04-18 更新）

本文档面向“直接接手本项目继续做远端实验”的 agent / 合作者。目标不是重复所有设计背景，而是把：

1. 当前代码和分支状态
2. 已经验证过的环境与命令
3. 训练所需组件在 GPU 机器上的搭建过程
4. 当前已经确认的链路状态、瓶颈和坑
5. 下一步最推荐的实验顺序

一次性交代清楚。

如果某块内容已经在已有文档里写得足够清楚，本文档会直接引用，不再重复展开。

---

## 0. 当前接手结论（2026-04-18）

这份 handoff 下面的大段内容仍然保留了更早期 `step400/520/580` 和最初 `step580 curriculum pilot` 的历史背景；这些内容还有效，但**已经不是当前最新进度**。真正需要先看的，是这一节。

当前最新状态可以一句话概括成：

- 当前最强的 **held-out deployed base** 仍然是 `step1300_rl`
- 当前这轮 `step1300 shortdiag pure SFT v1` 的最佳 checkpoint 是：
  - 开发集 repair 主看 `step1300_sft_v1_step60`
  - 它也是这轮 current-lineage `Protocol B valid_big500` 的最佳点
- 但 `step1300_sft_v1_step60` **还不能替代** `step1300_rl` 作为主 deployed model：
  - 在 `codecontests_test / Protocol A` 上没有净 gain
  - 在 `codecontests_test / Protocol B` 上也没有超过 `step1300_rl`
- `step900` 继续主要承担两件事：
  - Protocol A 的 canonical frozen first-pass source
  - 历史 repair fair-comparison 的固定输入基座
- 后续引用 repair 结论时，应优先使用：
  - Protocol B 结果做“真实部署式 / end-to-end”主判断
  - Protocol A 结果做“固定输入、公平比较 checkpoint repairability”判断

### 0.1 先读这些当前真正需要的文档

这几份文档比下面的旧背景更重要：

- `Curriculum RL` 的完整设计和**后续实际推进历史**：
  - [curriculum_rl_pilot_v8_explainer.md](curriculum_rl_pilot_v8_explainer.md)
- `Phase 4 repair` 的最新设计草案（当前只真正落地 Step 1 开发评测版）：
  - [repair_phase4_design.md](repair_phase4_design.md)
  - [phase4_step1_repair_eval_implementation_plan.md](phase4_step1_repair_eval_implementation_plan.md)
  - [repair_eval_current_contract_2026-04-18.md](repair_eval_current_contract_2026-04-18.md)
  - [repair_protocolA_results_2026-04-18.md](repair_protocolA_results_2026-04-18.md)
  - [repair_protocolB_results_2026-04-18.md](repair_protocolB_results_2026-04-18.md)
  - [repair_eval_protocolA_step900_vs_sft30_60.md](repair_eval_protocolA_step900_vs_sft30_60.md)
  - [repair_eval_protocolB_step30_60.md](repair_eval_protocolB_step30_60.md)
- quarantine v3 说明：
  - [problem_quarantine_v3_guide.md](problem_quarantine_v3_guide.md)
- shared verifier / eval 主链：
  - [shared_verifier_infra_guide.md](shared_verifier_infra_guide.md)
  - [full_code_path_guide.md](full_code_path_guide.md)

如果要快速接手，这一节 + 上面的 `curriculum_rl_pilot_v8_explainer.md` 已经足够覆盖当前 RL 主线。

### 0.2 2026-04-18 repair 最新覆盖结论

这里先给当前最该引用的 repair 结论，下面更早的 `step600/900/1300`、`step400/580` 等段落继续保留作历史背景。

#### 当前 `step1300 shortdiag pure SFT v1` 的 best ckpt

- 当前这轮 current-lineage repair-SFT v1 的 best ckpt 是：`step1300_sft_v1_step60`
- 这个结论来自：
  - `valid_big500 / Protocol A + reuse_step900`
  - `valid_big500 / Protocol B + self-first-pass`

`valid_big500` 当前结果：

| model | Protocol A final acc@1 | Protocol B final acc@1 | 结论 |
|---|---:|---:|---|
| `step1300_rl` | `0.146` | `0.130` | current RL baseline |
| `step1300_sft_v1_step20` | `0.150` | `0.142` | dev 上有真实 repair gain |
| `step1300_sft_v1_step40` | `0.146` | `0.142` | 不如 `step20/60` |
| `step1300_sft_v1_step60` | `0.150` | `0.144` | current-lineage best |

解释：

- 这轮小规模 `step1300`-base repair-SFT v1 在开发集 repair 上是有效的
- `step60` 同时拿到了：
  - 与最优并列的 `Protocol A`
  - 单独最优的 `Protocol B`

#### 但 `step60` 还不能替代 `step1300_rl`

在 held-out `codecontests_test` 上：

| model | Protocol A final acc@1 | Protocol B final acc@1 | 结论 |
|---|---:|---:|---|
| `step1300_rl` | `0.0727` | `0.1091` | strongest held-out deployed model |
| `step1300_sft_v1_step60` | `0.0727` | `0.0970` | repair 有增益，但没超过 baseline |

解释：

- `Protocol A`：
  - `step1300_rl` 和 `step60` 都没有救回任何题
- `Protocol B`：
  - `step60` 确实有 self-repair gain
  - 但它只到 `9.70%`
  - 仍然低于 `step1300_rl` 的 `10.91%`

#### 当前正式拍板

- current-lineage repair-SFT v1 的主比较 ckpt：`step1300_sft_v1_step60`
- 当前最强 held-out deployed base：`step1300_rl`
- canonical frozen first-pass source：`step900`
- 如果继续做 `step1300 repair-SFT v2`：
  - **可以做**
  - 但只能按“改配方的 v2”推进
  - **不应**把当前 v1 当成可以直接放大复制的成功配方

#### v2 该怎么改，而不是简单放大

- 降低当前纯 `C_hard_partial` 风格样本的占比
- 提高 `Core / ExpansionB` 的占比
- 必要时加入少量 anchor/general code rows，降低 pure repair-only drift 风险
- 继续以 `step1300` 作为 base，而不是退回旧 base

### 0.3 历史 RL checkpoint 结果总表（保留作背景）

#### `delta69` 结果

| checkpoint | raw solved | raw pass_ratio_mean | clean 结论 |
|---|---:|---:|---|
| `step600` | `50/69` | `0.8892` | raw = clean |
| `step620` | `55/69` | `0.9023` | raw = clean |
| `step700` | `47/69` | `0.8234` | raw = clean |
| `step800` | `43/69` | `0.7699` | raw = clean |
| `step900` | `48/69` | `0.8549` | raw = clean |

说明：

- `step620` 是这条长跑前半程在 `delta69` 上最强的点
- 但这部分提升没有完整迁移到 `valid_big500`
- `step800` 是一次明显的坏点
- `step900` 比 `step800` 回来了，但仍然没有超过 `step600`

#### `valid_big500` 结果

| checkpoint | raw solved | raw pass_ratio_mean | clean solved | clean pass_ratio_mean |
|---|---:|---:|---:|---:|
| `step600` | `57/500` | `0.3316` | `57/498` | `0.3319` |
| `step620` | `50/500` | `0.3335` | `50/498` | `0.3329` |
| `step700` | `51/500` | `0.3312` | `51/498` | `0.3314` |
| `step800` | `40/500` | `0.3103` | `39/498` | `0.3094` |
| `step900` | `54/500` | `0.3473` | `53/498` | `0.3467` |

当前正式判断：

- `step600`：当前 **solve-count winner**
- `step900`：当前 **overall quality / partial correctness winner**
- `step800`：明确 bad checkpoint
- `step700`：介于两者之间，没有形成超过 `step600` 的 solved 增益

这轮 clean overlay 只在 `valid_big500` 里去掉了同样 2 题：

- `Codeforces/1425/I`
- `Codeforces/641/C`

所以 clean 口径没有改变排序，这一点很重要。

### 0.4 这轮 RL 学到了什么

把 `step620 -> 900` 的 sampler snapshot、`delta69` 和 `valid_big500` 放在一起看，当前最可靠的结论是：

- 这轮 RL 的收益是**真实的**
- 但收益更偏向 **partial correctness / broad gain**
- 还没有稳定转成 solve-count 提升

从 curriculum snapshot 看：

- `step620`：`U=9616, A=119, B=360, C=52, D=5`
- `step700`：`U=9376, A=129, B=393, C=157, D=97`
- `step800`：`U=9076, A=278, B=237, C=341, D=220`
- `step900`：`U=8776, A=364, B=272, C=455, D=285`

到 `step900` 的累计 `U` 流向是：

- `U -> A = 207`
- `U -> B = 164`
- `U -> C = 277`
- `U -> D = 192`

也就是：

- `U -> A/B = 371`
- `U -> C/D = 469`

这说明：

- `U` 图谱确实在建立
- 但新定级出来的题，hard/dead-hard 比例偏高
- `step800` 的坏点，本质上是 `B` 被明显吃进了 `C/D`

所以当前不应该把 `step900` 的收益理解成“solve 已经稳涨”，更准确的读法是：

- 这条探索分支扩大了已知难度图谱
- 提高了 broad partial
- 但 solve-set churn 仍然明显

### 0.5 当前主线已经切到哪一步

在 `step900` 评测结束之后，主线又开了一条新的 exploitation 分支：

- 实验名：
  - `grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0`
- resume 基座：
  - `global_step_900`
  - `curriculum_state_step_900.json`
- 当前策略目标：
  - 优先把 `u_visit1_backlog` 定级完
  - 明显提高 `A/B`
  - 持续压低 `C/D`

当前配置是：

- `TOTAL_TRAINING_STEPS=1000`
- `LIMITER_BUDGET=150`
- `SAVE_FREQ=35`
- `snapshot_steps=[910,945,980,1000]`
- `901..940`：`U/A/B/C/D = 4/5/6/1/0`，`U_revisit_quota=3`
- `941..1000`：`U/A/B/C/D = 2/6/7/1/0`，`U_revisit_quota=2`

当前状态：

- 这条 run 已经完成 dataset/curriculum 初始化
- `runtime_curriculum_coverage.json` 已生成
- 它是从前一条 `limiter_budget=200` 版本失败后重启出来的更稳版本

需要特别记住：

- `limiter_budget=200` 那条失败，不是数据坏了，也不是 GPU OOM
- 根因是 **Ray 的 CPU/RAM memory pressure 保护**
- 报错是在 `_compute_old_log_prob()` 暴露出来，但真正先死的是 Ray worker

### 0.6 当前远端 reward infra 与评测脚本

当前远端已经稳定搭好了 3 套 `8 sandbox + 1 LB`：

- 训练池：`8090 -> 8081..8088`
- eval 池 A：`8091 -> 8181..8188`
- eval 池 B：`8092 -> 8281..8288`

3 套池的 probe 都是：

- `32/32 success`

新增并已经实际用过的评测脚本：

- [ops/run_grpo_codecontests_delta69_checkpoint.sh](ops/run_grpo_codecontests_delta69_checkpoint.sh)
- [ops/launch_three_delta69_checkpoints.sh](ops/launch_three_delta69_checkpoints.sh)
- [ops/launch_remaining_validbig_serial.sh](ops/launch_remaining_validbig_serial.sh)

并发经验值：

- 三池并行跑 `delta69` 时，稳定配置是：
  - `MAX_CONCURRENT=69`
  - `MAX_CONCURRENT_JUDGES=72`
  - `VERIFIER_LIMITER_BUDGET=72`
  - `BATCH_SIZE=69`
- 单池跑 `valid_big500` 时，稳定配置是：
  - `MAX_CONCURRENT=200`
  - `MAX_CONCURRENT_JUDGES=180`
  - `VERIFIER_LIMITER_BUDGET=180`
  - `BATCH_SIZE=200`

原因见：

- [`../src/verifier/shared.py`](../src/verifier/shared.py)

这里的 `limiter_budget` 约束的是**真实在途 sandbox RPC 数**，不是“题目数”。单题内部还会 fan-out 到 testcase 级并发，所以不能按“500 题就可以直接把并发也设到 500”来理解。

### 0.7 当前代码状态（接手时务必注意）

当前分支仍然是：

- `feature/grpo-development`

当前 `HEAD` 是：

- `7d4fceea871f3b71dbf8109662b062ed8d786f4b`

但本地 working tree **明显是 dirty 的**，很多这轮 RL / quarantine / eval 脚本和文档都还没有整理成干净提交。所以接手时要记住：

- 不要把 `HEAD commit` 当成完整真相
- 以当前文件树里的真实脚本、资产和文档为准
- 尤其是 `curriculum_assets/`、`review_assets/`、`src/problem_quarantine.py`、`src/step580_curriculum_sampler.py`、`phase_2_ GRPO/ops/` 这几块

---

## 1. 先读这些已有文档

以下文档已经覆盖了“为什么这么设计”和“代码链路怎么走”：

- 项目总进度：[../PROGRESS.md](../PROGRESS.md)
- Phase 2 总背景与实验计划：[README.md](README.md)
- 奖励设计与算法决策：[algorithm_decision_guide.md](algorithm_decision_guide.md)
- shared verifier 的详细设计：[shared_verifier_infra_guide.md](shared_verifier_infra_guide.md)
- RL / Eval 完整代码链路：[full_code_path_guide.md](full_code_path_guide.md)
- 数据脏题治理最新说明（quarantine v3）：[problem_quarantine_v3_guide.md](problem_quarantine_v3_guide.md)
- 数据脏题初筛历史说明（quarantine v2）：[problem_quarantine_v2_guide.md](problem_quarantine_v2_guide.md)
- `Curriculum RL` 设计与这轮实际推进历史：[curriculum_rl_pilot_v8_explainer.md](curriculum_rl_pilot_v8_explainer.md)

本文档重点补的是：远端机器怎么搭、哪些命令已经跑过、当前 smoke 到了哪一步、哪里还卡着，以及这轮 `step600 -> 900 -> 1000` 的最新执行状态。

补充说明：

- 当前最新的共享 quarantine 主清单是：
  - [../data/problem_quarantine_v3.json](../data/problem_quarantine_v3.json)
- 这轮 `v3` 的策略、review 资产、结果和接入建议，不在本文档重复展开，统一看：
  - [problem_quarantine_v3_guide.md](problem_quarantine_v3_guide.md)
- 如需回看最初一轮全库轻筛与 `v2` 的形成过程，再看：
  - [problem_quarantine_v2_guide.md](problem_quarantine_v2_guide.md)

---

## 2. 当前代码快照

### 2.1 分支与提交

- 当前工作分支：`feature/grpo-development`
- 当前 `HEAD`：
  - `7d4fceea871f3b71dbf8109662b062ed8d786f4b`

注意：

- 当前 working tree 是 dirty 的
- 很多这轮 RL / quarantine / eval 的关键文件仍是“本地文件状态”，不是单靠 `HEAD` 就能恢复
- 所以真正接手时，优先看：
  - 本文档 `0. 当前接手结论`
  - [curriculum_rl_pilot_v8_explainer.md](curriculum_rl_pilot_v8_explainer.md)
  - 以及当前文件树里已经存在的 `ops/`、`curriculum_assets/`、`review_assets/`、`src/` 修改

这个 commit 已包含：

- shared verifier 主链接入
- eval 切换到 shared verifier
- batch reward 接入 verl
- trainer verifier metrics hook
- parquet 构建脚本
- `run_grpo_smoke.sh`
- `run_grpo_step_smoke.sh`

### 2.2 本轮新增的两个关键提交

- `6a42039ce154881fac1eca72bbe5a6283e3eb66a`
  - 新增更快的单步 smoke 脚本
  - parquet builder 增加 `step_smoke_*`
- `882026442725e44da49a8e94b9e56a1826f405a9`
  - 修正 tiny step smoke 的数据规模，避免 prompt 过滤后 dataloader 为空

---

## 3. 当前主链实现位置

shared truth / reward / eval 主链已经不再依赖 `default_compute_score` 或 `submit()`。

关键文件如下：

- shared quarantine helper：
  - [`../src/problem_quarantine.py`](../src/problem_quarantine.py)
- curriculum builder：
  - [`../src/step580_curriculum_builder.py`](../src/step580_curriculum_builder.py)
- curriculum dataset：
  - [`../src/step580_curriculum_dataset.py`](../src/step580_curriculum_dataset.py)
- curriculum sampler：
  - [`../src/step580_curriculum_sampler.py`](../src/step580_curriculum_sampler.py)
- shared verifier：
  - [`../src/verifier/shared.py`](../src/verifier/shared.py)
- RL batch reward：
  - [`../src/grpo_batch_reward.py`](../src/grpo_batch_reward.py)
- eval 主入口：
  - [`../src/phase0_eval.py`](../src/phase0_eval.py)
- parquet 构建：
  - [`../src/build_grpo_parquet.py`](../src/build_grpo_parquet.py)
- manifest quarantine filter：
  - [`../src/filter_curriculum_manifest_by_quarantine.py`](../src/filter_curriculum_manifest_by_quarantine.py)
- 标准 smoke：
  - [`run_grpo_smoke.sh`](run_grpo_smoke.sh)
- 快速单步 smoke：
  - [`run_grpo_step_smoke.sh`](run_grpo_step_smoke.sh)
- curriculum launcher：
  - [`ops/run_grpo_a1_resume580_to660_curriculum.sh`](ops/run_grpo_a1_resume580_to660_curriculum.sh)
- trainer verifier metrics：
  - [`../../verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py)
  - [`../../verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py)

如果需要理解这些文件之间如何调用，请直接看：

- [shared_verifier_infra_guide.md](shared_verifier_infra_guide.md)
- [full_code_path_guide.md](full_code_path_guide.md)

---

## 4. 重要原则与禁止事项

这些是当前主链必须坚持的约束：

- 不使用 `default_compute_score` 作为 shared truth / code reward 主链。
- 不使用 sandbox 的 `submit()` API。
- 所有 reward 与 eval 都只使用 `coding_model_project/data` 自带的 external test cases。
- `CodeContests` 必须按 full-test 聚合 `pass_ratio_all`。
- `HumanEval / MBPP` v1 只要求 normalized single-case summary，不承诺 testcase fan-out。

已经确认过的坑：

1. `default_compute_score` 会丢 metadata，不适合作为 shared truth。
2. `submit()` 在当前本地部署的 sandbox 下无法提供完整内置测试集。
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 和 vLLM memory pool 不兼容，不要和当前 rollout/vLLM 组合一起用。

---

## 5. 远端机器基线

下面是已经实际验证过的一台远端机器基线。未来换新机器，也建议尽量对齐。

### 5.1 机器与镜像

- GPU：`4 x NVIDIA GeForce RTX 5090`
- 单卡显存：约 `32GB`
- 镜像：`verlai/verl:vllm011.latest`

### 5.2 已验证版本

在这台机器上实际打印并确认过：

- `Python 3.12.11`
- `torch 2.8.0+cu128`
- `vllm 0.11.0`
- `ray 2.49.2`
- `pandas 2.3.3`
- `pyarrow 22.0.0`

结论：`verlai/verl:vllm011.latest` 和当前分支是兼容的，至少 eval + shared verifier + GRPO smoke 初始化这一层没有版本级 blocker。

---

## 6. 远端目录约定

下面这些路径是已经跑通过的一套约定，后续继续用最省心：

- repo 根目录：`/root/verl`
- raw 数据目录：`/root/verl/coding_model_project/data/raw`
- parquet 输出目录：`/root/verl/coding_model_project/data/grpo_parquet`
- sandbox server venv：`/root/sandboxfusion-venv`
- sandbox runtime venv：`/root/sandbox-runtime`
- sandbox server 日志：`/root/sandboxfusion-run/server.log`
- Ray 日志：`/tmp/ray/session_latest/logs`
- checkpoint 目录：`/root/verl/checkpoints/rlvr_coding_model/<experiment_name>`

---

## 7. 数据准备

### 7.1 raw 数据必须准备齐全

至少需要把以下 raw 文件放到 `coding_model_project/data/raw/`：

- `codecontests_train_wo_valid_big_raw.jsonl`
- `codecontests_valid_raw.jsonl`
- `codecontests_valid_big_raw.jsonl`
- `codecontests_test_raw.jsonl`
- `humaneval_raw.jsonl`
- `mbpp_reg_raw.jsonl`

`dataset_samples.jsonl` 保留也没问题，但不是主实验所必需。

### 7.2 重要说明

训练和评测都依赖这些 raw + manifest 生成的 external tests。

不要尝试改成依赖 sandbox 内置 dataset tables；当前这条路没有被验证，也不符合本项目当前 shared truth 的设计。

### 7.3 当前真正使用的数据入口（2026-04-07）

当前这轮 RL 主线已经不是最早的 `grpo_parquet`，而是 quarantine v3 接入后的版本：

- 训练 parquet：
  - `/workspace/verl/coding_model_project/data/grpo_parquet_qv3/train.parquet`
- fast val parquet：
  - `/workspace/verl/coding_model_project/data/grpo_parquet_qv3/fast_val_codecontests.parquet`
- build summary：
  - `/workspace/verl/coding_model_project/data/grpo_parquet_qv3/build_summary.json`

当前语义是：

- `train parquet`：只硬过滤 `hard_blacklist`
- `curriculum manifest / 高价值 seed`：过滤 `hard_blacklist + caution`
- `unresolved`：保留，不硬过滤

其中 `qv3` 的主清单是：

- [../data/problem_quarantine_v3.json](../data/problem_quarantine_v3.json)

`qv3` 当前已知计数：

- `hard_blacklist = 780`
- `caution = 22`
- `unresolved = 83`

当前实际使用的 curriculum manifest 是：

- [curriculum_assets/step600_v3_qv3/curriculum_train_manifest_step600_v3_qv3.jsonl](curriculum_assets/step600_v3_qv3/curriculum_train_manifest_step600_v3_qv3.jsonl)

它来自：

- `focused manifest v3` 成品：`593` 行
- 再经过 `qv3` 过滤后变成：`587` 行

当前最终桶大小是：

- `A = 139`
- `B = 409`
- `C = 39`

对应过滤报告：

- [curriculum_assets/step600_v3_qv3/quarantine_filter_report_step600_v3_qv3.json](curriculum_assets/step600_v3_qv3/quarantine_filter_report_step600_v3_qv3.json)

---

## 8. GPU 机器上的完整搭建过程

下面是已经在远端机器上实际用过的一套搭建流程。

### 8.1 拉代码与子模块

如果是新机器，建议：

```bash
git clone --recurse-submodules <repo-url> /root/verl
cd /root/verl
git checkout feature/grpo-development
git submodule update --init --recursive
```

如果 repo 已存在：

```bash
cd /root/verl
git fetch origin feature/grpo-development
git checkout feature/grpo-development
git pull --ff-only origin feature/grpo-development
git submodule update --init --recursive
```

### 8.2 安装项目本体与 sandbox client

在官方镜像容器里执行：

```bash
cd /root/verl
python3 -m pip install --no-deps -e /root/verl
python3 -m pip install --no-deps -e /root/verl/SandboxFusion/scripts/client
python3 -m pip install tenacity
```

说明：

- `--no-deps` 是有意的，避免覆盖镜像自带的 PyTorch/vLLM 栈。
- `tenacity` 是 client 运行时实际缺的一个依赖，已补过。

### 8.3 搭建 SandboxFusion server 环境

```bash
python3 -m venv /root/sandboxfusion-venv
/root/sandboxfusion-venv/bin/pip install "pydantic<2.7" fastapi "uvicorn[standard]==0.25.0" structlog psutil aiofiles aiohttp tenacity "databases[aiomysql,aiosqlite]" "transformers>=4.44.0"
mkdir -p /root/verl/SandboxFusion/docs/build
mkdir -p /root/sandboxfusion-run
```

启动 server：

```bash
cd /root/verl/SandboxFusion
nohup /root/sandboxfusion-venv/bin/python -m uvicorn sandbox.server.server:app --host 0.0.0.0 --port 8080 >/root/sandboxfusion-run/server.log 2>&1 &
```

验活：

```bash
curl -sf http://localhost:8080/v1/ping
```

预期返回：

```text
"pong"
```

### 8.4 搭建 sandbox runtime 兼容层

当前 SandboxFusion local runner 会尝试：

```bash
source /opt/miniconda3/bin/activate sandbox-runtime
```

如果机器上没有这套 conda 环境，需要做一个轻量兼容层。

1. 创建 runtime venv：

```bash
python3 -m venv /root/sandbox-runtime
```

2. 创建兼容目录：

```bash
mkdir -p /opt/miniconda3/bin
mkdir -p /opt/miniconda3/condabin
```

3. 写一个最小 `activate` shim：

```bash
cat >/opt/miniconda3/bin/activate <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "sandbox-runtime" ]; then
  export VIRTUAL_ENV=/root/sandbox-runtime
  export PATH="/root/sandbox-runtime/bin:$PATH"
  return 0 2>/dev/null || exit 0
fi
echo "Unsupported env: ${1:-}" >&2
return 1 2>/dev/null || exit 1
EOF
chmod +x /opt/miniconda3/bin/activate
```

4. 验证：

```bash
bash -c 'source /opt/miniconda3/bin/activate sandbox-runtime; which python'
```

预期输出：

```text
/root/sandbox-runtime/bin/python
```

如果这一步不通，`/run_code` 会失败。

---

## 9. parquet 构建

### 9.1 标准构建命令

```bash
cd /root/verl
python3 coding_model_project/src/build_grpo_parquet.py \
  --data_root coding_model_project/data \
  --output_dir coding_model_project/data/grpo_parquet
```

### 9.2 当前建议的 step smoke 构建命令

为了避免 tiny train 被 prompt 过滤后直接空掉，建议显式指定：

```bash
cd /root/verl
python3 coding_model_project/src/build_grpo_parquet.py \
  --data_root coding_model_project/data \
  --output_dir coding_model_project/data/grpo_parquet \
  --step_smoke_train_size 16 \
  --step_smoke_val_size 8
```

最新一次已验证输出：

```json
{
  "train": 11785,
  "val_tier1": 317,
  "val_tier2": 500,
  "final_eval": 329,
  "smoke_train": 64,
  "smoke_val": 32,
  "step_smoke_train": 16,
  "step_smoke_val": 8
}
```

---

## 10. 已验证通过的命令

### 10.1 `/run_code` smoke

成功样例：

```bash
python3 - <<'PY'
from sandbox_fusion import RunCodeRequest, run_code
req = RunCodeRequest(language='python', code='print(123)', run_timeout=3, compile_timeout=3)
resp = run_code(req, endpoint='http://localhost:8080')
print(resp.status)
print(resp.run_result.status)
print((resp.run_result.stdout or '').strip())
PY
```

已验证结果：

- `RunStatus.Success`
- `CommandRunStatus.Finished`
- `stdout = 123`

超时样例：

```bash
python3 - <<'PY'
from sandbox_fusion import RunCodeRequest, run_code
req = RunCodeRequest(language='python', code='while True:\n    pass', run_timeout=1, compile_timeout=3)
resp = run_code(req, endpoint='http://localhost:8080')
print(resp.status)
print(resp.run_result.status if resp.run_result else None)
PY
```

已验证结果：

- `RunStatus.Failed`
- `CommandRunStatus.TimeLimitExceeded`

### 10.2 eval smoke

命令：

```bash
cd /root/verl/coding_model_project
python3 src/phase0_eval.py \
  --mode simple \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --vllm_url http://localhost:8001 \
  --sandbox_url http://localhost:8080 \
  --manifest_dir data/manifests \
  --datasets codecontests_valid \
  --temperature 0.0 \
  --max_tokens 512 \
  --run_timeout 30 \
  --max_concurrent 4 \
  --verifier_limiter_budget 4 \
  --batch_size 1 \
  --max_problems 1 \
  --output_dir outputs/phase0_smoke_reboot
```

已验证结果：

- `accepted@1 = 0.00%`
- `pass_ratio_mean = 0.2200`

并且 per-problem 输出已确认：

- 顶层存在 `per_case_results`
- `len(per_case_results) = 50`
- `passed_tests = 11`
- `total_tests = 50`
- `pass_ratio_all = 0.22`

结论：eval + shared verifier + CodeContests full-test 聚合已经通。

---

## 11. GRPO smoke / step-smoke 的现状

### 11.1 标准 smoke 脚本

文件：

- [run_grpo_smoke.sh](run_grpo_smoke.sh)

这条链已经验证过能够：

- 完成 Ray 初始化
- 完成 rollout / vLLM server 初始化
- 进入 validation generation
- 在训练主循环里触发 shared verifier + sandbox judge

但默认 smoke 墙钟时间很长，不适合快速看“能不能打出 step”。

### 11.2 快速单步 smoke 脚本

文件：

- [run_grpo_step_smoke.sh](run_grpo_step_smoke.sh)

默认策略：

- 用 `step_smoke_train.parquet`
- 用 `step_smoke_val.parquet`
- `train_batch_size = 8`
- `ppo_mini_batch_size = 8`
- `val_before_train = False`
- `test_freq = 1000`
- `total_training_steps = 1`（由 dataloader 大小实际收敛成 1）
- `max_response_length = 512`
- `actor_rollout_ref.actor.fsdp_config.offload_policy = True`
- `custom_reward_function.reward_kwargs.run_timeout_s = 15`

推荐启动命令：

```bash
cd /root/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled
export ROLLOUT_TP_SIZE=2
export ROLLOUT_GPU_MEM_UTIL=0.5
export MAX_RESPONSE_LENGTH=512
export ACTOR_OFFLOAD_POLICY=True
export RUN_TIMEOUT_S=15

bash 'coding_model_project/phase_2_ GRPO/run_grpo_step_smoke.sh' \
  trainer.experiment_name=grpo_step_smoke_shared_verifier_probe
```

---

## 12. 本轮真机验证结论

### 12.1 已确认打通的部分

下面这些已经不再是 blocker：

1. `Hydra` 自定义 reward kwargs 注入
2. `reward_model.use_reward_loop=False` 之后误入 experimental reward loop
3. shared verifier 接入 RL 主链
4. batched reward 触发 sandbox judge
5. CodeContests full-test reward/eval 聚合

### 12.2 我实际观察到的 RL 链路状态

在 `step_smoke_train=16 / step_smoke_val=8` 的单步 smoke 下，远端日志已经出现：

- `Size of train dataloader: 1`
- `Total training steps: 1`
- `Training Progress: 0/1`

这说明单步 smoke 已经真的进入训练循环，不再停留在 init 阶段。

### 12.3 reward judge 的并发证据

在 step 过程中，sandbox server 日志中出现了同一秒内多条 judge 请求，例如：

- 多次 `start processing python request ...`
- 多次 `running command python /tmp/...`

这说明：

- reward 已经真的调用 shared verifier
- verifier 已经把多个 testcase / 样本并发发给 sandbox
- sandbox 侧已经不是串行评测

### 12.4 早期最稳定复现的失败点

在没有打开 `offload_policy=True` 之前，最清楚、最稳定复现的失败点是：

- **actor update 阶段的显存 OOM**

具体出现在：

- `actor_rollout_update_actor`
- `self.actor_optimizer.step()`
- `torch._foreach_sqrt(device_exp_avg_sqs)`

错误核心：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 520.00 MiB
```

这说明在 4x5090、当前 FSDP2 + actor update 配置下：

- rollout / reward / verifier 已经能走到
- 但 optimizer step 还不够稳

### 12.5 已验证通过的一步 smoke 参数

下面这组参数已经在远端 `4 x RTX 5090` 上成功跑通单步训练：

- `actor_rollout_ref.actor.fsdp_config.offload_policy=True`
- `custom_reward_function.reward_kwargs.run_timeout_s=15`
- `train_batch_size=8`
- `ppo_mini_batch_size=8`
- `max_response_length=512`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2`

最先跑通的是 `rollout.n=1` 的 step smoke，关键观察是：

- `timing_s/reward ≈ 6.31s`
- `timing_s/update_actor ≈ 39.62s`
- `timing_s/step ≈ 59.42s`
- `perf/max_memory_reserved_gb ≈ 21.92`
- `verifier/judge_time_s_mean ≈ 6.09s`
- `verifier/judge_time_s_p95 ≈ 6.30s`
- `verifier/timeout_rate = 0.0`

这说明：

- `run_timeout_s=15` 在这批样本上没有造成显性 timeout 惩罚
- `offload_policy=True` 才是把一步 smoke 从 OOM 拉到可完成状态的关键开关

### 12.6 rollout.n=8 的正式单步 probe 结果

后续又在同一台机器上跑了更接近正式 GRPO 的一轮：

```bash
cd /root/verl
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
MAX_RESPONSE_LENGTH=512 \
ROLLOUT_TP_SIZE=2 \
ROLLOUT_N=8 \
LIMITER_BUDGET=8 \
bash 'coding_model_project/phase_2_ GRPO/run_grpo_step_smoke.sh' \
  actor_rollout_ref.actor.fsdp_config.offload_policy=True \
  custom_reward_function.reward_kwargs.run_timeout_s=15 \
  trainer.experiment_name=codex_probe_rollout8_rt15_offload
```

这轮已经真实完成：

- 训练退出码：`0`
- 日志出现：`Training Progress: 100%|...| 1/1`
- 训练结束后 4 张 GPU 都回到空闲

关键指标如下：

- `timing_s/gen = 12.52s`
- `timing_s/reward = 51.61s`
- `timing_s/old_log_prob = 16.01s`
- `timing_s/update_actor = 156.54s`
- `timing_s/step = 236.70s`
- `perf/max_memory_allocated_gb = 15.75`
- `perf/max_memory_reserved_gb = 21.75`
- `perf/cpu_memory_used_gb = 130.33`
- `verifier/pass_ratio_all_mean = 0.3322`
- `verifier/accepted_rate = 0.1406`
- `verifier/invalid_for_rl_rate = 0.0`
- `verifier/judge_time_s_mean = 6.10s`
- `verifier/judge_time_s_p95 = 10.40s`
- `verifier/timeout_rate = 0.0`
- `verifier/runtime_error_rate = 0.1406`
- `verifier/wrong_answer_rate = 0.6875`

validation 侧同时返回了：

- `val-aux/codecontests_valid/pass_ratio_all/mean@1 = 0.25`
- `val-aux/codecontests_valid/accepted/mean@1 = 0.0`
- `val-aux/codecontests_valid/judge_time_s/mean@1 ≈ 2.49s`
- `val-aux/mbpp_reg/pass_ratio_all/mean@1 = 0.25`
- `val-aux/mbpp_reg/accepted/mean@1 = 0.25`

### 12.7 关于“吞吐瓶颈是不是主要在 sandbox”

当前结论是：

- **reward judge 确实是 step 内的重要耗时段**
- 但**当前主瓶颈不是 sandbox，而是 actor update**

基于 `rollout.n=8` 这轮已完成的单步 probe，时间占比非常清楚：

- `update_actor ≈ 156.54s`
- `reward ≈ 51.61s`
- `old_log_prob ≈ 16.01s`
- `gen ≈ 12.52s`

所以更准确的结论是：

- shared verifier + sandbox 并发链路已经通，而且 `rollout.n=8` 已可完成
- reward judge 是 step 内第二大耗时段，后续仍值得做横向扩展
- 但在当前参数下，首先值得优化的仍然是 actor update

此外，sandbox 日志也证明“并发不足”不是当前主问题。以这轮 `rollout.n=8` 为例：

- `start processing python request` 峰值约 `80 次/秒`
- `running command python` 峰值约 `160 次/秒`
- 未观察到 `process killed`

因此：

- 不建议把当前 wall time 先归咎为 sandbox 并发太低
- `run_timeout_s=15` 在这轮里也没有表现出明显的 reward 偏置

---

## 13. 已踩过的坑

### 13.1 不要再走 `default_compute_score`

原因见：

- [shared_verifier_infra_guide.md](shared_verifier_infra_guide.md)

简述：

- 只看前 10 个 testcase
- 会丢 metadata
- 不适合 shared truth

### 13.2 不要用 `submit()`

当前本地部署的 sandbox 无法提供完整内置测试集，因此 `submit()` 不适合作为主链。

### 13.3 不要给当前 rollout/vLLM 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

会直接报：

```text
Expandable segments are not compatible with memory pool
```

这是 vLLM memory pool 层面的不兼容，不要继续尝试这条路。

### 13.4 tiny step smoke 数据一定要显式指定训练集大小

如果只取 `8` 条，经过 prompt 过滤后可能掉到 `7`，进而触发：

```text
AssertionError: Train dataloader is empty!
```

所以当前建议始终显式传：

```bash
--step_smoke_train_size 16 --step_smoke_val_size 8
```

---

## 14. 接手后的推荐顺序

### 第一优先级：把一步 smoke 稳定跑完

当前最合适的入口就是：

- [run_grpo_step_smoke.sh](run_grpo_step_smoke.sh)

截至本文档更新时，这个目标已经达成。建议把下面这组参数视为当前默认 smoke 基线：

- `actor_rollout_ref.actor.fsdp_config.offload_policy=True`
- `custom_reward_function.reward_kwargs.run_timeout_s=15`
- `train_batch_size=8`
- `ppo_mini_batch_size=8`
- `max_response_length=512`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2`

推荐继续试的方向：

1. 在当前基线之上做多 step 稳定性验证
2. 优先继续压 actor update 耗时 / 显存
3. 不改 shared verifier 主链
4. 先保持 `run_timeout_s=15` 做 smoke；正式 eval 仍建议保留 `30s`

优先可尝试的方向：

- `actor_rollout_ref.actor.ppo_mini_batch_size=4`
- 如有必要，再把 `train_batch_size` 从 `8` 降到 `4`
- 如果目标是更快而不是更稳，再考虑更多 GPU，而不是优先扩 sandbox

正式主线补充口径：

- formal 入口默认 `run_timeout_s=30`
- formal 入口显式固定 `reward_model.use_reward_loop=False`
- formal 入口显式固定 `reward_model.launch_reward_fn_async=False`
- trainer 会在 reward 前做 `extra_info` 副本合并，把 `finish_reason` / `truncated_by_max_tokens` 注入给 BatchRewardManager

### 第二优先级：在“一步能稳定完成”后再做 sandbox 吞吐归因

建议做两组对比：

1. 单 sandbox，改 `limiter_budget`
   - `1 / 2 / 4 / 8`
2. 多 sandbox 实例
   - 先服务侧做多个实例
   - 训练侧仍然只认一个入口 URL

### 第三优先级：只有在上一步完成后，才开始做 sandbox 水平扩展

当前不建议一上来就改成多 sandbox client 逻辑。

先确认：

- 单步 smoke 能稳定结束
- reward judge 的 wall time 占比足够高
- `limiter_budget` 提升确实带来吞吐改善

再去做多个 sandbox 实例，会更好归因，也更好答辩。

---

## 15. 建议保留的日志与证据

接手实验时，建议优先保留这些证据：

- parquet 构建输出：
  - `coding_model_project/data/grpo_parquet/build_summary.json`
- eval 结果：
  - `coding_model_project/outputs/phase0_*`
- sandbox server 日志：
  - `/root/sandboxfusion-run/server.log`
- Ray 日志：
  - `/tmp/ray/session_latest/logs`
- 每次实验的最终 commit：
  - `git rev-parse HEAD`

这些信息足够支撑：

- shared verifier 主链已接通
- reward judge 已发生
- 当前失败点具体在哪个阶段

---

## 16. 最后一句话总结

当前项目已经从“reward / eval 真值不一致、sandbox 主链不可信”的状态，推进到了：

- eval 可用
- shared verifier 可用
- RL reward 已真实接入
- sandbox judge 已在训练时并发执行

当前剩余最现实的问题不是 reward 设计，而是：

- 先让 `4x5090` 上的一步 GRPO smoke **稳定跑完**
- 然后再量化 `sandbox` 是否是主要吞吐瓶颈
- 最后再做多 sandbox 水平扩展

---

## 17. 2026-04-01 最新接手更新（以本节为准）

如果本节与前文有冲突，以本节为准。本节记录的是：

- formal 主线代码已经落地后的最新代码基线
- 新租 `4 x RTX 5090` 机器上的真实 bring-up 状态
- `A1` formal 主线的 `2-step RL probe` 实测结果
- 当前已经证明了什么、还没有证明什么

### 17.1 当前代码基线

当前建议接手基线 commit 更新为：

- `7d4fceea871f3b71dbf8109662b062ed8d786f4b`

最近几次关键提交：

- `6468ecd1c787f292e91176cfd78a5bb248b99946`
  - formal GRPO reward contract
  - trainer-side `extra_info` 合并
  - `invalid_for_rl` / `all-invalid group` 语义
  - verifier / validation 新指标
  - formal 入口脚本
- `d8bf31a514ea9b3e5634c0647edd43de56cc4bb9`
  - formal 脚本 seed override 修正
- `7d4fceea871f3b71dbf8109662b062ed8d786f4b`
  - formal `filter_groups` override 路径修正

formal 主线的关键文件现在是：

- formal 脚本：
  - [`run_grpo_formal.sh`](run_grpo_formal.sh)
  - [`run_grpo_a0.sh`](run_grpo_a0.sh)
  - [`run_grpo_a1.sh`](run_grpo_a1.sh)
  - [`run_grpo_a2.sh`](run_grpo_a2.sh)
- reward contract：
  - [`../src/grpo_batch_reward.py`](../src/grpo_batch_reward.py)
- trainer reward metadata merge：
  - [`../../verl/trainer/ppo/reward.py`](../../verl/trainer/ppo/reward.py)
- GRPO invalid / all-invalid 语义：
  - [`../../verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py)
- verifier / validation / throughput 指标：
  - [`../../verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py)
- trainer 主循环：
  - [`../../verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py)

如果需要完整设计背景，不要在本节里重复找，直接回看：

- [README.md](README.md)
- [algorithm_decision_guide.md](algorithm_decision_guide.md)
- [formal_reward_design.md](formal_reward_design.md)
- [shared_verifier_infra_guide.md](shared_verifier_infra_guide.md)
- [full_code_path_guide.md](full_code_path_guide.md)

### 17.2 当前远端机器与目录

当前活跃机器：

```bash
ssh -p 44433 root@69.176.92.133 -L 8080:localhost:8080
```

本地 `~/.ssh/config` 中已可直接使用：

```bash
ssh vastai
```

当前远端约定：

- repo：`/root/verl`
- 数据根目录：`/workspace/data`
- repo 内数据软链：
  - `/root/verl/coding_model_project/data/raw -> /workspace/data/raw`
  - `/root/verl/coding_model_project/data/manifests -> /workspace/data/manifests`
- parquet 输出：
  - `/root/verl/coding_model_project/data/grpo_parquet`
- sandbox server：
  - `http://localhost:8080`
- sandbox 日志：
  - `/root/sandboxfusion-run/server.log`
- 训练主日志：
  - `/root/grpo_a1_formal_pilot_fallback.log`
  - `/root/grpo_a1_rl_chain.log`
- Ray 日志：
  - `/tmp/ray/session_latest/logs`

当前这台机器上已经确认：

- `SandboxFusion` server 正常，`curl -sf http://localhost:8080/v1/ping` 返回 `"pong"`
- repo 已 pull 到上述 commit
- formal parquet 已构建完成，最近一次构建数量为：

```json
{
  "train": 11785,
  "val_tier1": 317,
  "val_tier2": 500,
  "final_eval": 329
}
```

### 17.3 formal 10-step pilot 为什么没有直接跑完

第一轮真正的 formal A1 pilot 是按 `10-step + val_before_train=True` 启动的，但没有进入训练 step，而是长时间卡在 pre-train validation。

关键原因不是 reward 或 sandbox 主链报错，而是：

- `trainer.val_before_train=True` 时，trainer 会先完整调用 `_validate()`
- 当前验证集是：
  - `val_tier1 = 317`
  - `val_tier2 = 500`
- 过滤过长 prompt 之后，远端实际进入验证的数据量是 `757`
- 当前实现里，如果没有单独设置 `data.val_batch_size`，默认直接取 `len(val_dataset)`，见 [`../../verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py)
- validation 默认 `val_kwargs.n=1`，不是 `rollout.n=8`，见 `../../verl/trainer/config/rollout/rollout.yaml`

这意味着：

- formal pilot 的第一眼 wall time，主要花在 “完整 validation + sandbox 判题”
- 不是先打一两个 training step 再验证

这也是后面专门切出 `2-step RL probe` 的原因：先验证“训练主链是否真正打通”，不要把时间耗在 pre-train full validation 上。

### 17.4 已完成的 A1 2-step RL probe

这轮 probe 不是最终正式实验，而是“只验证 formal 训练主链是否正确”。

入口脚本：

- [`run_grpo_a1.sh`](run_grpo_a1.sh)
- [`run_grpo_formal.sh`](run_grpo_formal.sh)

基线 formal 配置见：

- [`run_grpo_formal.sh`](run_grpo_formal.sh)

这轮 probe 在 formal 基线之上额外加了以下 override：

- `trainer.total_training_steps=2`
- `trainer.test_freq=0`
- `trainer.save_freq=0`
- `trainer.val_before_train=False`
- `actor_rollout_ref.actor.use_torch_compile=False`
- `actor_rollout_ref.ref.use_torch_compile=False`
- `actor_rollout_ref.rollout.enforce_eager=True`

其中最后三条只是 infra fallback，用来避开这台新机器在 compile / CUDA graph capture 阶段的长启动时间；它们**不改变** reward / GRPO / formal 语义。

远端实际启动命令已经在日志中完整展开，见：

- `/root/grpo_a1_rl_chain.log`

对应的核心配置可以整理成：

- 算法：`A1`
- `algorithm.adv_estimator=grpo`
- `algorithm.norm_adv_by_std_in_grpo=True`
- `actor.loss_agg_mode=token-mean`
- `actor.clip_ratio_low=0.2`
- `actor.clip_ratio_high=0.28`
- `actor.use_kl_loss=False`
- 模型：`Qwen/Qwen2.5-Coder-7B-Instruct`
- `train_batch_size=8`
- `ppo_mini_batch_size=8`
- `rollout.n=8`
- `rollout.tensor_model_parallel_size=2`
- `actor.fsdp_config.offload_policy=True`
- `max_response_length=512`
- `reward_mode=anchored_dense_v1`
- `reward_model.use_reward_loop=False`
- `reward_model.launch_reward_fn_async=False`
- `run_timeout_s=30`
- `limiter_budget=8`

formal reward 的实际公式与 invalid 语义以：

- [`../src/grpo_batch_reward.py`](../src/grpo_batch_reward.py)

为准。

### 17.5 A1 2-step RL probe 的真实结果

这轮 run 已经自然结束，不是中途挂掉。日志证据：

- `Training Progress: 100%|...| 2/2`
- 日志中已经完整打印 `step:1` 和 `step:2`
- run 结束后 Ray 退出，4 张 GPU 回到空闲

step 指标在日志中的位置：

- `step:1`：`/root/grpo_a1_rl_chain.log` 第 `848` 行
- `step:2`：`/root/grpo_a1_rl_chain.log` 第 `851` 行

关键指标如下：

| 指标 | Step 1 | Step 2 |
|---|---:|---:|
| `training/global_step` | 1 | 2 |
| `verifier/pass_ratio_all_mean` | 0.3178 | 0.0972 |
| `verifier/accepted_rate` | 0.15625 | 0.03125 |
| `verifier/invalid_for_rl_rate` | 0.015625 | 0.0 |
| `verifier/truncated_by_max_tokens_rate` | 0.015625 | 0.0 |
| `verifier/reward_raw_valid_count` | 63 | 64 |
| `verifier/reward_raw_valid_rate` | 0.984375 | 1.0 |
| `verifier/reward_raw_mean` | 0.2742 | 0.0059 |
| `verifier/judge_time_s_mean` | 4.27s | 12.77s |
| `verifier/judge_time_s_p95` | 8.69s | 36.14s |
| `verifier/extraction_failure_rate` | 0.015625 | 0.046875 |
| `verifier/syntax_error_rate` | 0.015625 | 0.03125 |
| `verifier/runtime_error_rate` | 0.0625 | 0.25 |
| `verifier/timeout_rate` | 0.0 | 0.015625 |
| `verifier/wrong_answer_rate` | 0.75 | 0.625 |
| `grpo/all_invalid_group_count` | 0 | 0 |
| `response_length/mean` | 223.84 | 234.63 |
| `response_length/clip_ratio` | 0.015625 | 0.046875 |
| `timing_s/gen` | 19.20s | 14.20s |
| `timing_s/reward` | 34.66s | 123.02s |
| `timing_s/old_log_prob` | 11.55s | 10.86s |
| `timing_s/update_actor` | 146.95s | 139.16s |
| `timing_s/step` | 212.39s | 287.26s |
| `perf/total_num_tokens` | 54,902 | 49,816 |
| `perf/throughput` | 64.63 | 43.35 |

补充说明：

- 每个 step 实际是 `8 prompt groups x 8 samples = 64 samples`
- `perf/throughput` 的定义是“每秒每卡 token 数”，见 [`../../verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py)

### 17.6 对 probe 结果的解释

#### 17.6.1 已经证明了什么

这轮 `A1` probe 已经可以证明 formal 主链在远端真实跑通了：

- rollout
- shared verifier reward
- trainer-side `extra_info` metadata merge
- truncation / `invalid_for_rl` 语义
- GRPO advantage
- actor update

而且不是“只过初始化”，而是已经真实完成了 `2` 个 training step。

#### 17.6.2 奖励信号是否足够

当前判断是：**足够，而且比 sparse reward 更适合作为早期 A1 主线信号**。

理由：

- step 1 / step 2 的 `reward_raw_valid_rate` 分别是 `0.984` 和 `1.0`，说明几乎所有 sample 都进入了 RL
- `invalid_for_rl_rate` 很低，没有出现大面积 invalid
- `truncated_by_max_tokens_rate` 分别只有 `1/64` 和 `0/64`，说明 `max_response_length=512` 目前没有把大部分样本直接掐死
- `accepted_rate` 虽然低，但 `pass_ratio_all_mean` 明显高于 0，这说明大量样本仍有 partial credit
- 这正是 [`../src/grpo_batch_reward.py`](../src/grpo_batch_reward.py) 中 `anchored_dense_v1` 想要提供的 dense signal

因此：

- 这轮结果支持“当前阶段继续用 `anchored_dense_v1`”
- 不支持退回 `sparse_accepted`

#### 17.6.3 当前吞吐是否合理

对“correctness probe”来说合理；对“正式长跑”来说偏保守。

当前 wall time 的主要分布不是 rollout，而是：

- `update_actor`
- `reward`

step 1：

- `gen ≈ 19.2s`
- `reward ≈ 34.7s`
- `update_actor ≈ 147.0s`

step 2：

- `gen ≈ 14.2s`
- `reward ≈ 123.0s`
- `update_actor ≈ 139.2s`

这说明：

- 当前主瓶颈不是 “模型不会生成”
- 更像是 “actor update 偏重 + sandbox verifier 长尾”

另外，step 2 的 `reward` 明显变慢，和 `judge_time_s_mean / p95` 的上升高度一致：

- step 2 `timeout_rate` 只有 `1/64`
- 不能简单归因为 “被 timeout 卡住”
- 更准确的解释是：这一批样本整体判题更慢，叠加极少量 timeout 长尾

#### 17.6.4 当前还没证明什么

这轮 probe **没有**证明以下事情：

- `A0` 已远端实跑通过
- `A2` 已远端实跑通过
- `val_before_train=True` 的完整 formal 10-step pilot 已完成
- `test_freq/save_freq` 打开的完整正式流程已完成
- `torch_compile=True` / 非 eager rollout 的最终高性能配置已稳定

其中：

- `A0` 风险较低，因为它与 `A1` 共用绝大多数 formal 主链，只是多了 KL 和更保守的 clip
- `A2` 风险更高，因为它改的是更敏感的训练语义：
  - `algorithm.norm_adv_by_std_in_grpo=False`
  - `actor.loss_agg_mode=seq-mean-token-sum-norm`
  - `actor.loss_scale_factor=2048`

所以如果由其他 agent 接手，**不能**把 “A1 2-step probe 通过” 直接当成 “A0/A2 全部证明完毕”。

### 17.7 目前最值得关注的剩余问题

当前我会把剩余问题分成 4 类：

1. formal 训练主链正确性
- `A1`：已证明
- `A0`：未远端实跑，但风险低
- `A2`：未远端实跑，优先级高于 A0

2. 完整正式流程
- 还没有一轮真正完成的：
  - `val_before_train=True`
  - `test_freq>0`
  - `save_freq>0`
  - `10-step`

3. 性能配置
- 当前 probe 为了先验证主线正确，使用了：
  - `actor.use_torch_compile=False`
  - `ref.use_torch_compile=False`
  - `rollout.enforce_eager=True`
- 这能降低初始化成本，但不是最终性能配置

4. 吞吐瓶颈
- 当前更像是：
  - `update_actor` 偏慢
  - `reward` 长尾明显
- 并不能简单归咎为 sandbox “并发不足”

### 17.8 其他 agent 接手时建议的实验顺序

如果目标是“直接接住当前实验，不重复踩坑”，建议按这个顺序：

1. 先把当前远端状态对齐到本节
- 确认 repo commit
- 确认 sandbox `"pong"`
- 确认 parquet 已就绪
- 保存当前 `git rev-parse HEAD`

2. 优先跑 `A2` 的 `1-2 step RL probe`
- 原因：这是三条算法里当前剩余风险最高的一条
- 保持与本节 `A1 probe` 同样的 infra fallback

3. 再回到 `A1` 跑一轮带 validation/save 的短 formal pilot
- 推荐仍从 `2-10 step` 小规模开始
- 先验证完整正式流程，再谈长跑

4. 最后才尝试恢复更激进的性能配置
- 如：
  - `actor.use_torch_compile=True`
  - `ref.use_torch_compile=True`
  - 关闭 `enforce_eager`

当前不建议的顺序是：

- 先恢复高性能 compile / cudagraph 路径
- 先做多 sandbox 扩展
- 先做长 validation + 10-step formal 跑满

因为这些都会让问题归因变得更乱。

### 17.9 一句话 handoff

截至 `2026-04-01`：

- formal 代码主线已经落地并推到 `feature/grpo-development`
- 新租 `4x5090` 机器已完成 repo / data / sandbox / parquet bring-up
- `A1` formal 主线已经在远端真实完成 `2-step RL probe`
- 当前最重要的未完成项不是“reward 是否可用”，而是：
  - `A2` 是否同样能跑通
  - 完整 `formal pilot` 是否能在保留 validation/save 的情况下稳定完成

### 17.10 `2026-04-02` reward-first 阶段总览

`2026-04-02` 之后，这条线的优先级已经明确调整为：

1. 先修 `reward` 吞吐
2. 再修在线 `validation/testing`
3. 最后才做 `actor update` 提速

这个调整不是拍脑袋，而是来自已经完成的 `10-step baseline` 和后续一系列 probe：

- 非 validation 步上，`reward` 已经比 `update_actor` 更常成为主瓶颈
- validation 步上，`testing` 是额外且独立的高成本链路
- `actor update` 在调完 batching 后已经不再是 launch blocker

本阶段不改 reward 语义，继续冻结：

- `A1`
- `anchored_dense_v1 + guardrails`
- `filter_groups=false`
- `RUN_TIMEOUT_S=30`
- `SEED=0`

reward 设计、算法选择与 verifier infra 的背景不在这里重复展开，统一引用：

- [algorithm_decision_guide.md](./algorithm_decision_guide.md)
- [formal_reward_design.md](./formal_reward_design.md)
- [shared_verifier_infra_guide.md](./shared_verifier_infra_guide.md)
- [eval_validation_strategy.md](./eval_validation_strategy.md)

### 17.11 当前有效机器与环境状态

当前主实验机已经切到新机器，状态如下：

- SSH：
  - `ssh -i ~/.ssh/vastai_ed25519 -p 40616 root@74.2.96.34 -L 8080:localhost:8080`
- repo 根目录：
  - `/root/verl`
- 数据软链：
  - `/root/verl/coding_model_project/data/raw -> /workspace/data/raw`
  - `/root/verl/coding_model_project/data/manifests -> /workspace/data/manifests`
- parquet 根目录：
  - `/root/verl/coding_model_project/data/grpo_parquet`
- sandbox：
  - `http://localhost:8080/v1/ping -> "pong"`
- 模型缓存：
  - `Qwen/Qwen2.5-Coder-7B-Instruct` 已预热

旧 readiness 前的 checkpoint 目录已经清理，释放磁盘空间：

- 删除：
  - `/root/verl/checkpoints/rlvr_coding_model/grpo_a1_fallback_baseline_seed0_host74`
- 删除后可用空间恢复到约 `172G`

### 17.12 dedicated fast-val 资产已经实现并验证

`fast_val_codecontests.parquet` 已经不是概念，而是落地完成的资产生成路径。

本地代码改动与验证：

- builder：
  - [build_grpo_parquet.py](../src/build_grpo_parquet.py)
- CPU 测试：
  - [test_build_grpo_parquet_on_cpu.py](../../tests/utils/test_build_grpo_parquet_on_cpu.py)

这条 builder 路径的关键点是：

- 不手写复制 prompt 过滤逻辑
- 构建时复用 runtime validation 的 tokenizer / processor 初始化路径
- tokenizer / processor 来源固定为训练时的 `MODEL_PATH`
- 使用与 [rl_dataset.py](../../verl/utils/dataset/rl_dataset.py) runtime 过滤一致的 chat template / processor / `max_prompt_length=1024` / `filter_overlong_prompts=True` 语义
- 写出后再做 parity check，保证 post-filter 最终长度仍然是 `16`

这意味着后续其他 agent 不需要再怀疑 `fast_val_codecontests.parquet` 是否和 runtime 过滤口径漂移；当前 builder 已经专门为这个问题做了对齐。

### 17.13 reward 并发的代码级结论

这部分是后续 reward infra 优化的核心背景，必须知道。

结论先说：

- 当前 RL reward 链路的真实并发主阀门不在 SandboxFusion server
- 当前 first-order bottleneck 在 verifier 侧的全局 `limiter_budget`

代码级依据如下：

- verifier 端虽然有两层线程池：
  - 候选级并发见 [shared.py](../src/verifier/shared.py)
  - CodeContests 单题 testcase 级并发也在 [shared.py](../src/verifier/shared.py)
- 但所有线程最后都要经过同一个进程级 `BoundedSemaphore(limit)`，因此有效并发先被 verifier 自己卡住
  - 参考 [shared.py](../src/verifier/shared.py)
- client 端当前仍是同步阻塞 `requests.post`，接口也只吃单个 endpoint string
  - 参考 [client.py](../../SandboxFusion/scripts/client/src/sandbox_fusion/client.py)
- server 端 `/run_code` 路径本身是 async handler，runner 也是 async 启 subprocess，并没有看到这条通用 Python 路径上的全局串行锁
  - 参考 [sandbox_api.py](../../SandboxFusion/sandbox/server/sandbox_api.py)
- `sandbox.max_concurrency` 虽然出现在配置里，但没有证据表明它是当前这条 RL reward 主路径上的真实 cap
  - 参考 [local.yaml](../../SandboxFusion/sandbox/configs/local.yaml)

这对后续优化顺序的含义是：

1. 第一刀先调 `LIMITER_BUDGET`
2. 如果单实例 limiter sweep 仍不够，再做“一个 LB URL 后挂多个 sandbox 实例”
3. 当前阶段不优先做 multi-endpoint client 改造
4. 当前阶段不优先降 `RUN_TIMEOUT_S`

### 17.14 已完成实验：reward-only limiter sweep

这轮 sweep 是在冻结 actor / inference baseline 的前提下完成的：

- `actor.use_torch_compile=False`
- `rollout.enforce_eager=True`
- `actor.fsdp_config.offload_policy=True`
- `model.enable_gradient_checkpointing=True`
- `ppo_micro_batch_size_per_gpu=1`
- `ppo_mini_batch_size=8`
- `trainer.total_training_steps=4`
- `trainer.test_freq=0`
- `trainer.save_freq=0`
- `trainer.val_before_train=False`
- `RUN_TIMEOUT_S=30`
- `SEED=0`

reward-only probe 日志：

- `lb=8`：
  - `/root/grpo_a1_reward_probe_lb8_seed0.log`
- `lb=12`：
  - `/root/grpo_a1_reward_probe_lb12_seed0.log`
- `lb=16`：
  - `/root/grpo_a1_reward_probe_lb16_seed0.log`

以 `steps 2-4` 的 median 为口径：

| LIMITER_BUDGET | median `timing_s/reward` | median `timing_s/update_actor` | max `invalid_for_rl_rate` | max `truncated_by_max_tokens_rate` | min `reward_raw_valid_rate` | max `timeout_rate` |
|---|---:|---:|---:|---:|---:|---:|
| `8`  | `272.38s` | `134.27s` | `0.0156` | `0.0156` | `0.9844` | `0.0313` |
| `12` | `101.13s` | `135.13s` | `0.0156` | `0.0156` | `0.9844` | `0.0156` |
| `16` | `142.70s` | `136.70s` | `0.0156` | `0.0156` | `0.9844` | `0.0313` |

结论：

- 当前单实例 sandbox 下，`LIMITER_BUDGET=12` 是明确 winner
- 这也支持了上一节的代码级结论：reward 第一优先级旋钮就是 verifier 侧 limiter

### 17.15 已完成实验：dedicated fast-val baseline vs candidate

这轮比较用于验证：

- dedicated `fast_val_codecontests.parquet` 是否可作为 launch-quality gate
- reward winner `LIMITER_BUDGET=12` 是否会破坏在线 validation

配置：

- `VAL_FILES=fast_val_codecontests.parquet`
- `data.val_batch_size=16`
- `trainer.total_training_steps=4`
- `trainer.test_freq=2`
- `trainer.save_freq=0`
- `trainer.val_before_train=False`
- `RUN_TIMEOUT_S=30`
- `SEED=0`

日志：

- baseline `lb=8`：
  - `/root/grpo_a1_fastval_baseline_lb8_seed0.log`
- candidate `lb=12`：
  - `/root/grpo_a1_fastval_candidate_lb12_seed0.log`

按 validation 步的 median 统计：

| Config | median `timing_s/testing` | median `pass_ratio_all/mean@1` | median `accepted/mean@1` | median `invalid_for_rl/mean@1` | median `truncated_by_max_tokens/mean@1` |
|---|---:|---:|---:|---:|---:|
| baseline `lb=8` | `45.04s` | `0.2779` | `0.1563` | `0.0000` | `0.0000` |
| candidate `lb=12` | `32.95s` | `0.2710` | `0.1250` | `0.0313` | `0.0313` |

结论：

- `LIMITER_BUDGET=12` 的在线 validation 更快
- `pass_ratio_all/accepted` 的回退没有超过当时设定的 gate
- `invalid_for_rl` 和 `truncated_by_max_tokens` 有小幅上升，但仍在 gate 之内

因此：

- reward winner `lb=12` 通过了 dedicated fast-val promotion

### 17.16 已完成实验：actor sweep

在 reward winner `LIMITER_BUDGET=12` 固定后，actor sweep 已经完成。

公共配置：

- `trainer.total_training_steps=4`
- `trainer.test_freq=0`
- `trainer.save_freq=0`
- `trainer.val_before_train=False`
- `RUN_TIMEOUT_S=30`
- `SEED=0`
- `LIMITER_BUDGET=12`

日志：

- `micro=2, mini=8`：
  - `/root/grpo_a1_actor_probe_micro2_lb12_seed0.log`
- `micro=4, mini=8`：
  - `/root/grpo_a1_actor_probe_micro4_lb12_seed0.log`
- `micro=4, mini=4`：
  - `/root/grpo_a1_actor_probe_micro4_minib4_lb12_seed0.log`

按 `steps 2-4` 的 median `timing_s/update_actor` 统计：

| `ppo_micro_batch_size_per_gpu` | `ppo_mini_batch_size` | median `timing_s/update_actor` | `max_memory_reserved_gb` 量级 | OOM |
|---|---:|---:|---:|---:|
| `2` | `8` | `74.81s` | `~22.31 GB` | `No` |
| `4` | `8` | `45.16s` | `~22.37 GB` | `No` |
| `4` | `4` | `53.06s` | `~22.37 GB` | `No` |

结论：

- actor winner 已经明确：
  - `ppo_micro_batch_size_per_gpu=4`
  - `ppo_mini_batch_size=8`
- 相对 actor baseline `micro=1, mini=8`，`update_actor` 中位数已经下降约 `66.6%`
- 当前新机器上，actor update 已经不再是 formal launch 的主阻塞项

额外说明：

- 在当前配置路径下，不要把 `actor/pg_loss`、`actor/pg_clipfrac`、`actor/ppo_kl` 接近 `0` 当作“policy 没更新”的证据
- 这里命中的是当前 actor 的 on-policy 分支，这些值接近 `0` 基本是按定义发生

### 17.17 当前 final readiness pilot

当前进行中的 readiness pilot 配置已经固定为：

- `LIMITER_BUDGET=12`
- `ppo_micro_batch_size_per_gpu=4`
- `ppo_mini_batch_size=8`
- `VAL_FILES=fast_val_codecontests.parquet`
- `data.val_batch_size=16`
- `trainer.total_training_steps=10`
- `trainer.test_freq=5`
- `trainer.save_freq=5`
- `trainer.val_before_train=False`
- `RUN_TIMEOUT_S=30`
- `SEED=0`
- 继续保留稳定 fallback：
  - `actor.use_torch_compile=False`
  - `ref.use_torch_compile=False`
  - `rollout.enforce_eager=True`
  - `actor.fsdp_config.offload_policy=True`
  - `model.enable_gradient_checkpointing=True`

运行名与日志：

- run name：
  - `grpo_a1_final_readiness_lb12_micro4_minib8_seed0`
- log：
  - `/root/grpo_a1_final_readiness_lb12_micro4_minib8_seed0.log`

截至目前，前 `5` 个 step 已经打出，且：

- `step 5` 的 validation 已完成
- `step 5` 的 save 已完成
- checkpoint 已经成功写出到：
  - `/root/verl/checkpoints/rlvr_coding_model/grpo_a1_final_readiness_lb12_micro4_minib8_seed0/global_step_5`

前 `5` 步关键指标如下：

| Step | `timing_s/reward` | `timing_s/update_actor` | `verifier/judge_time_s_p95` | `timeout_rate` | `invalid_for_rl_rate` | `truncated_by_max_tokens_rate` | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| `1` | `29.64s`  | `51.80s` | `12.35s`  | `0.0000` | `0.0156` | `0.0156` | 快步 |
| `2` | `511.92s` | `44.74s` | `219.79s` | `0.0781` | `0.0000` | `0.0000` | 极慢步 |
| `3` | `113.03s` | `45.47s` | `59.33s`  | `0.0000` | `0.0000` | `0.0000` | 中等 |
| `4` | `315.21s` | `43.36s` | `160.49s` | `0.0313` | `0.0000` | `0.0000` | 慢步 |
| `5` | `377.20s` | `44.70s` | `237.10s` | `0.0156` | `0.0156` | `0.0156` | 带 validation/save |

`step 5` 的 dedicated fast-val 指标：

- `val-aux/codecontests_valid/pass_ratio_all/mean@1 = 0.2360`
- `val-aux/codecontests_valid/accepted/mean@1 = 0.1250`
- `val-aux/codecontests_valid/invalid_for_rl/mean@1 = 0.0000`
- `val-aux/codecontests_valid/truncated_by_max_tokens/mean@1 = 0.0000`
- `timing_s/testing = 22.85s`
- `timing_s/save_checkpoint = 18.32s`

当前结论非常明确：

- `actor update` 已经稳定，不是主要问题
- `reward` 仍然是 readiness pilot 里最主要的成本风险

### 17.18 当前 reward 仍然存在的问题

现在这条 readiness pilot 透露出的核心问题不是“actor 慢”，而是：

- 题目批次方差很大
- CodeContests judge 长尾仍然明显
- reward time 仍然会被少数慢题 / 慢 testcase 批次拖爆

当前最有代表性的观测：

- `update_actor` 已经很稳：
  - `51.8s / 44.7s / 45.5s / 43.4s / 44.7s`
- `reward` 仍然波动巨大：
  - `29.6s / 511.9s / 113.0s / 315.2s / 377.2s`
- 对应的 `judge_time_s_p95` 也在同步大幅波动：
  - `12.4s / 219.8s / 59.3s / 160.5s / 237.1s`
- `timeout_rate` 不是绝对主导，但在慢步上确实抬起来了：
  - `0 / 7.8% / 0 / 3.1% / 1.6%`
- `invalid_for_rl_rate` 在 `step 2-4` 为 `0`
- `truncated_by_max_tokens_rate` 在 `step 2-4` 为 `0`

因此：

- 当前主要问题是“题目批次方差 + reward judge 长尾”
- 不是 actor update
- 也不是 reward 语义本身坏了

这也解释了为什么当前**不建议**先把 `RUN_TIMEOUT_S=30` 往下压：

- timeout 目前不是唯一主因
- 在这条 reward 设计里，timeout 属于正常失败语义的一部分
- 提前改 timeout 会直接改 reward 分布和训练目标，而不只是改吞吐

reward / timeout 语义参考：

- [shared.py](../src/verifier/shared.py)
- [grpo_batch_reward.py](../src/grpo_batch_reward.py)

### 17.19 当前最推荐的下一步

如果目标是“让正式 RL 更便宜”，接下来最值得做的不是先降 timeout，而是并行推进 reward infra。

建议顺序：

1. 继续让当前 readiness pilot 自然跑完
- 当前配置已经足够稳定，值得拿完整 `10-step` 结果

2. 并行准备“一个 LB URL 后挂多个 sandbox 实例”的实现
- 不改 verifier / client 的多 endpoint 接口
- 仍然让上层只看到单个 `sandbox_endpoint: str`
- 这样最符合当前代码结构，也最容易和现有 shared verifier infra 对接

3. LB + 多 sandbox 准备好之后，重新做 reward-only probe
- 保持 `RUN_TIMEOUT_S=30`
- 保持当前 winner actor 配置不动
- 新一轮优先 sweep：
  - `LIMITER_BUDGET=12`
  - `LIMITER_BUDGET=16`
  - `LIMITER_BUDGET=20`
- 目标是观察在多实例后，limiter sweet spot 是否上移

4. 只有当“多 sandbox + limiter resweep”之后 reward 长尾仍不可接受，才做 pilot-only timeout ablation
- 可以再试：
  - `RUN_TIMEOUT_S=25`
  - `RUN_TIMEOUT_S=20`
- 但这一步必须明确标成 “pilot-only”，因为它已经不是纯系统调优，而是在改任务 / reward 口径

### 17.20 给下一位 agent 的一句话交接

如果你现在接手这条线，请不要再回到“actor 是第一优先级”的旧判断。

当前已经知道的事实是：

- `LIMITER_BUDGET=12` 是单实例 sandbox 下的 reward winner
- `ppo_micro_batch_size_per_gpu=4, ppo_mini_batch_size=8` 是当前机器上的 actor winner
- dedicated `fast_val_codecontests.parquet` 已经可用，并且通过了 runtime parity 设计
- 当前 readiness pilot 已经证明：
  - actor 稳了
  - validation/save 路径通了
  - reward 仍然因为题目批次方差和 judge 长尾而抖动很大

所以接下来最重要的工作不是再调 actor，而是：

- 一边继续收完 readiness pilot
- 一边准备 `LB + 多 sandbox`
- 然后重新做 reward-only probe 与 limiter resweep

### 17.21 更新：LB + 多 sandbox 已经落地，以上旧判断已被后续实验覆盖

从这一节开始，以这里的结论为准；上面 `17.19 / 17.20` 里“下一步还需要准备 LB + 多 sandbox”的描述已经过时。

已经新增并验证的 ops 资产：

- `coding_model_project/phase_2_ GRPO/ops/sandbox_backend_start.sh`
- `coding_model_project/phase_2_ GRPO/ops/sandbox_backend_stop.sh`
- `coding_model_project/phase_2_ GRPO/ops/sandbox_backend_status.sh`
- `coding_model_project/phase_2_ GRPO/ops/capture_host_baseline.sh`
- `coding_model_project/phase_2_ GRPO/ops/render_nginx_sandbox_lb.sh`
- `coding_model_project/phase_2_ GRPO/ops/apply_nginx_sandbox_lb.sh`
- `coding_model_project/phase_2_ GRPO/ops/lb_validate_probe.py`
- `coding_model_project/phase_2_ GRPO/ops/run_grpo_a1_reward_probe.sh`
- `coding_model_project/phase_2_ GRPO/ops/run_grpo_a1_fastval_gate.sh`
- `coding_model_project/phase_2_ GRPO/ops/run_grpo_a1_formal_observation_resume10.sh`
- `coding_model_project/phase_2_ GRPO/ops/monitor_first_save.py`

远端当前采用的 sandbox / LB 拓扑：

- Nginx LB：`http://localhost:8090`
- backend `sandbox-8081`：`127.0.0.1:8081`
- backend `sandbox-8082`：`127.0.0.1:8082`
- `8083` 仅作为救援型 bring-up 做过验证，最终没有保留在正式配置里

关键原则没有变：

- 上层 verifier / reward / client 仍然只看到单个 `sandbox_endpoint: str`
- 没有引入 multi-endpoint client 逻辑
- 仍然是 shared verifier 主链

### 17.22 已完成实验：LB + multi-sandbox reward-only sweep

第一轮短 reward-only sweep 的结果如下：

- `control_lb12_single`
  - `timing_s/reward` steps `2-4` median：`213.82s`
- `lb12_multi2`
  - `timing_s/reward` steps `2-4` median：`90.17s`
- `lb16_multi2`
  - `timing_s/reward` steps `2-4` median：`95.36s`
- `lb20_multi2`
  - `timing_s/reward` steps `2-4` median：`45.33s`
- `control_lb20_single`
  - `timing_s/reward` steps `2-4` median：`48.32s`

这轮结果说明：

- `LIMITER_BUDGET=20` 明显优于旧的单实例 `lb12`
- `multi2` 在短 sweep 上是成立的
- 但它更像“短样本下的 best sample”，不是已经完全稳定的 formal baseline

对应的短 fast-val：

- `grpo_a1_fastval_gate_lb20_multi2`
  - `reward/mean@1 = 0.1803`
  - `accepted/mean@1 = 0.125`
  - `pass_ratio_all/mean@1 = 0.2723`
  - `judge_time_s/mean@1 = 5.14`

### 17.23 已完成实验：rebaseline + rescue + batch probe

后续同口径 rebaseline 说明，`lb20 + multi2` 不能直接当“稳定基线”：

- `b8_s6` rebaseline 里，reward tail 再次暴露
- 慢步上 `timeout_rate` 会抬头
- 所以 `lb20_multi2 = 45.33s` 更准确地说是“历史最好样本”，不是当前这套 infra 的可靠点估计

因此后面先做了 infra rescue，再做 batch probe：

1. `multi2 + lb24 + b8_s6`
- `timing_s/reward` steps `2-6` median：`122.82s`
- `timing_s/step` steps `2-6` median：`192.02s`
- 结论：比 `lb20 + multi2 + b8_s6` 更适合作为正式 RL 候选基线

2. `multi3 + lb20 + b8`
- 没有比 `multi2 + lb24` 更稳
- step 波动更大
- 最终未保留为 formal 配置

3. `b12` preflight on `lb24 + multi2`
- reward 没有直接爆炸
- 但 `timing_s/update_actor` 中位数约 `64.34s`
- 对比 `b8` 的约 `45.25s`，恶化约 `42%`
- 说明下一步瓶颈已经转向 actor update，而不是 reward infra
- 因此没有继续推进 `b12_s6` / `b16`

结论：

- 当前最优可用 formal 候选不是“更大 batch”
- 而是：
  - `SANDBOX_URL=http://localhost:8090`
  - `2 backend`
  - `LIMITER_BUDGET=24`
  - `train_batch_size=8`
  - `ppo_mini_batch_size=8`
  - `ppo_micro_batch_size_per_gpu=4`
  - `rollout.n=8`
  - `RUN_TIMEOUT_S=30`

候选配置的 fast-val gate：

- `grpo_a1_fastval_gate_lb24_multi2_b8`
  - `reward/mean@1 = 0.15781967109069228`
  - `accepted/mean@1 = 0.0625`
  - `pass_ratio_all/mean@1 = 0.25977459016393445`
  - `invalid_for_rl/mean@1 = 0.0`
  - `truncated_by_max_tokens/mean@1 = 0.0`
  - `judge_time_s/mean@1 = 4.696976989507675`

### 17.24 当前正式 RL run（这是目前最重要的现场）

正式 observation run 已经启动，并且是从 readiness 的 `global_step_10` checkpoint 成功续跑的。

脚本：

- `coding_model_project/phase_2_ GRPO/ops/run_grpo_a1_formal_observation_resume10.sh`

实验名：

- `grpo_a1_formal_observe_lb24_multi2_resume10_seed0`

当前正式 run 的关键参数：

- `SANDBOX_URL=http://localhost:8090`
- `LIMITER_BUDGET=24`
- `train_batch_size=8`
- `ppo_mini_batch_size=8`
- `ppo_micro_batch_size_per_gpu=4`
- `rollout.n=8`
- `RUN_TIMEOUT_S=30`
- `test_freq=10`
- `save_freq=20`
- `trainer.max_actor_ckpt_to_keep=1`
- `trainer.max_critic_ckpt_to_keep=1`
- `val_before_train=False`
- `VAL_FILES=fast_val_codecontests.parquet`
- `resume_from_path=/root/verl/checkpoints/rlvr_coding_model/grpo_a1_final_readiness_lb12_micro4_minib8_seed0/global_step_10`
- `trainer.default_local_dir=/root/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0`
- `trainer.validation_data_dir=/root/verl/validation_dumps/grpo_a1_formal_observe_lb24_multi2_resume10_seed0`

磁盘策略：

- readiness 旧 checkpoint 在确认 resume 成功后已经删除
- 目的就是避免 `save step 20` 再次因为磁盘爆掉而失败
- 当前只保留一个 actor/critic checkpoint

截至本次 handoff 的最新现场：

- 训练已推进到 `step:30`
- 首个 save 已成功完成：
  - `latest_checkpointed_iteration.txt = 20`
  - checkpoint 目录中已有 `global_step_20`
- 当前外层日志：
  - `/root/grpo_a1_formal_observe_lb24_multi2_resume10_seed0.log`
- 当前 checkpoint 目录：
  - `/root/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0`
- 当前 validation dump 目录：
  - `/root/verl/validation_dumps/grpo_a1_formal_observe_lb24_multi2_resume10_seed0`

`step30` 同时触发了 fast-val，当前可见指标：

- `val-core/codecontests_valid/reward/mean@1 = 0.23181967076379806`
- `val-aux/codecontests_valid/accepted/mean@1 = 0.125`
- `val-aux/codecontests_valid/pass_ratio_all/mean@1 = 0.2585245901639344`
- `val-aux/codecontests_valid/invalid_for_rl/mean@1 = 0.0`
- `val-aux/codecontests_valid/judge_time_s/mean@1 = 7.010471522808075`
- `verifier/reward_raw_valid_rate = 0.953125`
- `verifier/invalid_for_rl_rate = 0.046875`
- `verifier/truncated_by_max_tokens_rate = 0.046875`
- `timing_s/reward = 42.89s`
- `timing_s/update_actor = 42.62s`
- `timing_s/testing = 44.65s`

### 17.25 关于 monitor / backend anomaly 的最新结论

第一次“监控到异常后停止”的结论，现在需要按下面的更正理解：

- 初版 first-save monitor 把 SandboxFusion backend 日志里的
  - `Failed to write to stdin ... handler is closed`
  - `Broken pipe`
  当作硬异常
- 这会造成误报

后续排查已经确认：

- 这类日志出自 `SandboxFusion/sandbox/runners/base.py`
- 它更像“子进程过早退出时，父进程给 stdin 写入失败”的 noisy backend log
- verifier 不会自动把它记成 `sandbox_error`
- 它在 `8081` / `8082` 都会出现，不是单 backend 独有
- 真正更值得盯的是：
  - Nginx non-200
  - `SandboxError`
  - `500`
  - `invalid_for_rl_rate`
  - `truncated_by_max_tokens_rate`
  - `timeout_rate`
  - reward / judge 长尾

因此已经做了两件事：

1. 更新 `ops/monitor_first_save.py`
- 把这类日志改为 `backend_benign_hits`
- 不再把它直接当成 stopper

2. 本地代码库中同步修改了 `SandboxFusion/sandbox/runners/base.py`
- 对 `handler is closed` / `Broken pipe` 做降噪
- 从 `logger.exception(...)` 降为普通 warning
- 这个改动已经进本地代码库，但不会影响当前远端已启动的 sandbox 进程，除非后续重启 backend

修正后的 monitor：

- 目录：`/root/monitoring/grpo_a1_formal_observe_lb24_multi2_resume10_seed0_v2`
- 当前状态：first-save 目标已经完成，`status.json` 显示 `success=true`

### 17.26 给下一位 agent 的当前接手点

如果你现在接手，请直接基于下面这些事实继续：

- 不要再把 `LIMITER_BUDGET=12` 当作当前 formal winner
- 单实例 `lb12`、短 sweep 的 `lb20_multi2`，都已经被后续 rebaseline / rescue / formal run 覆盖
- 当前正式主线配置是：
  - `multi2 + lb24 + b8`
- 当前正式 run 不是“待启动”，而是已经在跑：
  - 最新已见 `step30`
  - `global_step_20` 已成功落盘

后续接手时优先检查：

- `/root/grpo_a1_formal_observe_lb24_multi2_resume10_seed0.log`
- `/root/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0`
- `/root/verl/validation_dumps/grpo_a1_formal_observe_lb24_multi2_resume10_seed0`
- `/root/sandboxfusion-multi/nginx/sandbox_lb.access.log`
- `/root/sandboxfusion-multi/logs/sandbox_8081.log`
- `/root/sandboxfusion-multi/logs/sandbox_8082.log`
- `/root/monitoring/grpo_a1_formal_observe_lb24_multi2_resume10_seed0_v2/status.json`

如果要继续 formal 监控，优先关注：

- `timing_s/reward`
- `timing_s/testing`
- `verifier/judge_time_s_p95`
- `verifier/reward_raw_valid_rate`
- `verifier/invalid_for_rl_rate`
- `verifier/truncated_by_max_tokens_rate`
- `verifier/timeout_rate`
- `val-aux/codecontests_valid/accepted/mean@1`
- `val-aux/codecontests_valid/pass_ratio_all/mean@1`

一句话最新版 handoff：

- LB + multi-sandbox 已经实现并上线
- 当前 winner 已经不是单实例 `lb12`，而是 `multi2 + lb24 + b8`
- formal RL 已经从 `step10` checkpoint 成功续跑到 `step30`
- `global_step_20` 已保存成功
- 监控口径已经修正，不再把 `handler is closed` 这种 noisy log 当成硬 blocker

### 17.27 2026-04-02 晚些时候的最新进展：old host 已到 step80，new host 已完成一次真实 save+val 验证

这部分覆盖上面 `step30` 的旧现场，当前请以这里为准。

#### A. 旧机器 formal RL 最新状态

当前主线 formal run 仍然是：

- `grpo_a1_formal_observe_lb24_multi2_resume10_seed0`

当前真实现场已经推进到：

- `step:80`
- `latest_checkpointed_iteration.txt = 80`
- `global_step_80` 已成功落盘

现场路径：

- 外层日志：
  - `/root/grpo_a1_formal_observe_lb24_multi2_resume10_seed0.log`
- checkpoint 目录：
  - `/root/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0/global_step_80`
- validation dump 目录：
  - `/root/verl/validation_dumps/grpo_a1_formal_observe_lb24_multi2_resume10_seed0`

`step80` 的关键指标：

- `val-core/codecontests_valid/reward/mean@1 = 0.20798633829690516`
- `val-aux/codecontests_valid/accepted/mean@1 = 0.0625`
- `val-aux/codecontests_valid/pass_ratio_all/mean@1 = 0.24435792349726776`
- `val-aux/codecontests_valid/invalid_for_rl/mean@1 = 0.0`
- `val-aux/codecontests_valid/truncated_by_max_tokens/mean@1 = 0.0`
- `val-aux/codecontests_valid/judge_time_s/mean@1 = 4.5724134892225266`
- `verifier/reward_raw_valid_rate = 1.0`
- `verifier/invalid_for_rl_rate = 0.0`
- `verifier/truncated_by_max_tokens_rate = 0.0`
- `verifier/timeout_rate = 0.0`
- `timing_s/reward = 20.04s`
- `timing_s/update_actor = 43.38s`
- `timing_s/testing = 14.08s`
- `timing_s/save_checkpoint = 16.22s`
- `timing_s/step = 87.12s`

这说明：

- 当前 formal RL 并没有在 `step80` 附近出现新的 save 崩溃
- `multi2 + lb24 + b8` 在这一段样本上表现是健康的
- 后续切机时，`global_step_80` 是当前应优先迁移的 resume 基线

#### B. 新机器 `/workspace` 的独立验证结果

为了在旧机器跑到 `step80` 期间验证新机器环境，已经在新机器上单独跑了一轮短 integrated pilot：

- 实验名：
  - `grpo_a1_newhost_integrated_pilot_resume60_to65_seed0`
- resume 基线：
  - `/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0/global_step_60`
- 运行目录：
  - `/workspace/verl`

新机器实际验证到的内容，不只是“能初始化”，而是：

- 成功从 `global_step_60` resume
- 真正推进到 `step65`
- 成功写出 `global_step_65`
- 成功写出 validation dump
- 成功完成一次 fast-val

新机器关键路径：

- log：
  - `/workspace/grpo_a1_newhost_integrated_pilot_resume60_to65_seed0.log`
- checkpoint：
  - `/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_newhost_integrated_pilot_resume60_to65_seed0/global_step_65`
- validation dump：
  - `/workspace/verl/validation_dumps/grpo_a1_newhost_integrated_pilot_resume60_to65_seed0/65.jsonl`
- verifier endpoint：
  - `http://localhost:8090`

新机器 `step65` 的关键指标：

- `val-core/codecontests_valid/reward/mean@1 = 0.22381967189721763`
- `val-aux/codecontests_valid/accepted/mean@1 = 0.0`
- `val-aux/codecontests_valid/pass_ratio_all/mean@1 = 0.27977459016393447`
- `val-aux/codecontests_valid/invalid_for_rl/mean@1 = 0.0`
- `val-aux/codecontests_valid/truncated_by_max_tokens/mean@1 = 0.0`
- `val-aux/codecontests_valid/judge_time_s/mean@1 = 6.008014559745789`
- `verifier/reward_raw_valid_rate = 1.0`
- `verifier/invalid_for_rl_rate = 0.0`
- `verifier/truncated_by_max_tokens_rate = 0.0`
- `verifier/timeout_rate = 0.0`
- `timing_s/reward = 17.83s`
- `timing_s/update_actor = 56.09s`
- `timing_s/testing = 43.51s`
- `timing_s/save_checkpoint = 29.47s`
- `timing_s/step = 111.71s`
- `perf/max_memory_reserved_gb = 21.916`

这轮新机器验证的用途不是要替代正式 run 曲线，而是回答两个运维问题：

1. `/workspace` 路径下的新环境是否可用  
- 答案是可用

2. 新机器上的 sandbox + nginx LB + resume + save + fast-val 是否整链路打通  
- 答案是已经打通

#### C. 切机前后的建议

如果后续要把旧机器的 `step80` checkpoint 搬到新机器继续训练，请优先使用：

- checkpoint：
  - `/root/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0/global_step_80`

而不是继续用旧的 `global_step_60`。

切机时建议保留当前 formal 配置不变：

- `SANDBOX_URL=http://localhost:8090`
- `LIMITER_BUDGET=24`
- `train_batch_size=8`
- `ppo_mini_batch_size=8`
- `ppo_micro_batch_size_per_gpu=4`
- `rollout.n=8`
- `RUN_TIMEOUT_S=30`
- `test_freq=10`
- `save_freq=20`
- `trainer.max_actor_ckpt_to_keep=1`
- `trainer.max_critic_ckpt_to_keep=1`
- `VAL_FILES=fast_val_codecontests.parquet`

一句话最新 handoff：

- 旧机器 formal run 已经健康推进到 `step80`，`global_step_80` 已成功保存
- 新机器 `/workspace` 环境已经通过一次真实的 `resume -> train -> save -> fast-val` 验证
- 因此下一步切机时，不需要重新做 infra bring-up，只需要把 `global_step_80` 迁到新机器并从那里继续跑

### 17.28 2026-04-03 的最新进展：formal continuation 已完成到 `step200`，strict full-eval 队列正在跑 `step200`

这部分覆盖上面 `step80/new host bring-up` 的旧现场。当前请优先以这里为准。

#### A. 当前主线实验已经推进到哪里

当前 A1 formal 主线已经不是“准备切机”，而是已经完成了下面这条完整链路：

1. old host 上的主线 formal run：
- `grpo_a1_formal_observe_lb24_multi2_resume10_seed0`
- 已经从 `step10` 正式跑完到 `step100`

2. `step100` 已做过一次 strict baseline 同口径 full eval
- 结果目录：
  - `/workspace/verl/coding_model_project/outputs/phase0_fullval_step100_same_protocol`

3. new host 上的 continuation run：
- `grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0`
- 实际上已经从 `global_step_100` 继续跑完到 `step200`

当前最权威的训练完成证据是：

- 外层 continuation 日志：
  - `/workspace/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0_resume120_keep3_to200.log`
- 其中已经明确出现：
  - `Training Progress: 100%|...| 200/200`
  - `step:200`

#### B. 当前可用 checkpoint 的真实情况

现在后续 agent 不要再把 `step80` 当成最新 resume 基线。当前真正需要关注的是：

1. strict winner / continuation 起点：
- `/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0/global_step_100`

2. continuation 目录：
- `/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0`

当前 continuation 目录里 checkpoint 的真实状态是：

- `global_step_120`：完整 checkpoint，可评测
- `global_step_140`：只有占位目录，不是完整 checkpoint，不要拿来评测
- `global_step_160`：完整 checkpoint，可评测
- `global_step_180`：完整 checkpoint，可评测
- `global_step_200`：完整 checkpoint，可评测

因此当前严格可评测的 checkpoint shortlist 是：

- `step100`
- `step120`
- `step160`
- `step180`
- `step200`

不要把 `step140` 加进后续评测队列。

#### C. `step100` strict full eval 的当前结论

`step100` 已经用 phase0 fullval 同口径完成过一次真正 apples-to-apples 的比较。当前应保留的结论是：

- 在主任务 `CodeContests_valid (117)` 上，`step100` 相对 baseline 是正向：
  - baseline：`accepted@1 = 0.0256`, `pass_ratio_all = 0.1422`
  - strict `step100`：`accepted@1 = 0.0513`, `pass_ratio_all = 0.1576`
- 因此这轮 A1 formal run 不能再简单归类为“没有赢主任务”
- 但约束项有轻微回退：
  - `HumanEval`：baseline `0.8720` -> `step100` `0.8537`
  - `MBPP_reg`：baseline `0.5850` -> `step100` `0.5700`

这也是为什么后面选择了先做 `100 -> 200` continuation，而不是立刻改 reward / 改 PPO 配方。

#### D. `121 -> 200` continuation 训练侧总结

这段 continuation 没有 collapse。训练主链路一直是通的：

- `reward_raw_valid_rate` 整体均值约 `0.9982`
- `invalid_for_rl_rate` 整体均值约 `0.0018`
- `truncated_by_max_tokens_rate` 整体均值约 `0.0018`
- `response_length/clip_ratio` 整体均值约 `0.0082`

训练侧最明显的问题不是 actor update，而是 reward / judge 长尾：

- `121-140`：
  - `accepted_rate mean ≈ 0.1227`
  - `pass_ratio_all_mean ≈ 0.3034`
  - `timeout_rate mean ≈ 0.0320`
  - `timing_s/reward mean ≈ 162.47s`

- `141-160`：
  - `accepted_rate mean ≈ 0.1445`
  - `pass_ratio_all_mean ≈ 0.3351`
  - `timeout_rate mean ≈ 0.0297`
  - `timing_s/reward mean ≈ 133.00s`

- `161-180`：
  - `accepted_rate mean ≈ 0.1250`
  - `pass_ratio_all_mean ≈ 0.3422`
  - `timeout_rate mean ≈ 0.0680`
  - `timing_s/reward mean ≈ 250.41s`

- `181-200`：
  - `accepted_rate mean ≈ 0.1203`
  - `pass_ratio_all_mean ≈ 0.3155`
  - `timeout_rate mean ≈ 0.0664`
  - `timing_s/reward mean ≈ 250.98s`

因此：

- continuation 后半段不是“训练学崩了”
- 更像是 verifier / judge 长尾重新变重，导致线上观测方差变大

当前已知的后半段高风险现象主要是：

- 多个 step 的 `timeout_rate > 0.1`
- 多个 step 的 `timing_s/reward > 300s`
- 但 invalid / truncation 并没有同步扩散

#### E. continuation fast-val 的当前读法

当前 `110 -> 200` 的 fast-val 轨迹是“中后期震荡，但 `step200` 比 `160/180/190` 有回升”：

- `step110`：
  - `reward = 0.1808`
  - `accepted = 0.0`
  - `pass_ratio_all = 0.2260`

- `step120`：
  - `reward = 0.1768`
  - `accepted = 0.0`
  - `pass_ratio_all = 0.2210`

- `step130`：
  - `reward = 0.1993`
  - `accepted = 0.0625`
  - `pass_ratio_all = 0.2335`

- `step160`：
  - `reward = 0.1308`
  - `accepted = 0.0`
  - `pass_ratio_all = 0.1635`

- `step180`：
  - `reward = 0.1585`
  - `accepted = 0.0625`
  - `pass_ratio_all = 0.1825`

- `step190`：
  - `reward = 0.1423`
  - `accepted = 0.0625`
  - `pass_ratio_all = 0.1623`

- `step200`：
  - `reward = 0.1878`
  - `accepted = 0.125`
  - `pass_ratio_all = 0.2035`

当前更稳妥的解释是：

- `step200` 至少不是一个明显比 `step160/180/190` 更差的 continuation 终点
- 但是否真正优于 `step100`，不要只看 fast-val，必须等 strict full eval

#### F. 当前 strict checkpoint eval 队列进度

当前 new host 上已经启动了一条新的 strict full-protocol checkpoint eval 队列，用来评测：

- `step120`
- `step160`
- `step180`
- `step200`

评测脚本：

- `/workspace/verl/coding_model_project/phase_2_ GRPO/ops/run_phase0_full_protocol_checkpoint_queue.sh`

队列日志：

- `/workspace/eval_logs/phase0_full_protocol_eval_queue_lb24_multi2.out`

当前这条队列使用的设置是：

- `phase0_eval.py`
- `sandbox_url = http://localhost:8090`
- `verifier_limiter_budget = 24`
- `max_concurrent_judges = 24`
- `max_tokens = 2048`
- `datasets = humaneval mbpp_reg codecontests_valid`
- `save_full_results = true`

截至本次 handoff 最后一次核查（`2026-04-02 19:46 PDT`）：

- `step120`：`DONE_EVAL`
- `step160`：`DONE_EVAL`
- `step180`：`DONE_EVAL`
- `step200`：已经 `START_EVAL`，仍在运行

当时远端仍可见活跃进程：

- `vllm.entrypoints.openai.api_server` for `step200`
- `phase0_eval.py` for `step200`

同时 `step200` 的输出目录已经创建并开始写文件：

- `/workspace/verl/coding_model_project/outputs/phase0_fullval_step200_lb24_multi2`

其中已经可见：

- `run_info.json`
- `per_problem/humaneval.jsonl`
- `per_problem/mbpp_reg.jsonl`
- `per_problem/codecontests_valid.jsonl`

截至更细的一次现场核查：

- `humaneval.jsonl` 已经到 `164/164`
- `mbpp_reg.jsonl` 已经到 `200/200`
- `codecontests_valid.jsonl` 正在写，当前已到 `50/117`

但在 `DONE_EVAL step=200` 真正写入队列日志之前，不要把 `step200` 当成已经评完。

#### G. 当前 full-output 日志保存情况

这轮 strict eval 现在已经在保存全量 CodeContests 输出日志，不需要再额外补开开关。

真正用于行为分析的是：

- `per_problem/codecontests_valid.jsonl`
- `qa_logs/codecontests_valid_qa.jsonl`

而不是：

- 只看 `summary.json`
- 或只看 `qa_logs/qa_summary.json`

当前集中拷贝目录在：

- `/workspace/eval_logs/codecontests_full_outputs`

截至本次 handoff 已确认：

- `step100_codecontests_valid_full.jsonl`：已就绪
- `step120_codecontests_valid_full.jsonl`：已就绪
- `step160_codecontests_valid_full.jsonl`：已就绪
- `step180_codecontests_valid_full.jsonl`：已就绪
- `step200_codecontests_valid_full.jsonl`：当前仍是占位，需等 `step200` eval 完成后再确认

这些 full-output 日志已经足够支持：

- failure mode taxonomy
- badcase 深挖
- repair 候选模式初筛

但当前 `phase0_eval.py` 里仍然有日志截断上限：

- `max_prompt_chars = 4000`
- `max_response_chars = 12000`

对当前 `117` 题分析来说问题不大，但如果后续要跑 `valid_big` full-output 做 repair 行为分析，建议先把 prompt 日志上限放宽。

#### H. 本地 repair analysis 目录

另一个 agent 已经基于 `117` 题 strict traces 做了第一版 repair / failure taxonomy。当前本地参考目录是：

- `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/repair_analysis`

其中当前最值得直接读的文件有：

- `codecontests_repair_shortlist.md`
- `codecontests_repair_candidates.json`
- `analyze_codecontests_repair_candidates.py`

raw 对照输入也已经保存在：

- `raw/baseline_codecontests_valid.jsonl`
- `raw/step100_codecontests_valid.jsonl`
- `raw/step120_codecontests_valid.jsonl`

这批分析当前的用途是：

- 建立 `117` 题 failure taxonomy
- 找回归样本 / 稳定 hard failure / near-miss 样本
- 指导后续去训练集或外部题源里找“同类题”

不要把它误用成“直接从验证集抽题做 repair-SFT 数据”。

#### I. 当前已经采纳的后续策略

当前已经采纳、后续 agent 应继续遵守的边界是：

1. `117` 题 `codecontests_valid` 的用途
- 快速诊断
- failure mode taxonomy
- repair 候选模式初筛

2. `117` 题不适合单独决定 targeted repair / teacher-SFT 的最终训练题单

3. `valid_big (500)` 更适合作为主开发验证集
- 后续应在当前 checkpoint eval 收尾后，选：
  - `step100`
  - continuation 里 strict 表现最强的一个 checkpoint
- 对 shortlist 跑 `valid_big` full-output

4. 不要直接用 `117 valid` 或 `valid_big` 里的题构造 repair-SFT / teacher-SFT 训练样本
- 应该用 `valid + valid_big` 识别失败模式
- 再去：
  - `codecontests_train_wo_valid_big_raw.jsonl`
  - 或外部题源
- 构造真正训练用的 repair / teacher 数据

5. 最终 headline 不应停留在 `valid`
- 开发判断看 `valid` / `valid_big`
- 方案冻结后再跑 `codecontests_test`

一句话最新版 handoff：

- `step100` 已经在 strict full eval 上确认主任务优于 baseline
- continuation 已经正式跑完到 `step200`
- 当前 strict checkpoint eval 队列已经完成 `120/160/180`，`step200` 仍在跑
- `117` 题 full-output 与 repair taxonomy 已经具备，可继续做 badcase / failure mode 分析
- 但 targeted repair / teacher-SFT 的正式样本构造，应等待 shortlist checkpoint 的 `valid_big` full-output 之后再从 `train_wo_valid_big` 或外部题源生成
- 在对 `baseline / step100 / step200` 跑过 `valid_big (500)` 后，可以确认 `step200` 是当时最好的 diagnosis anchor：
  - `baseline`：`solved=36`，`accepted@1=0.072`，`pass_ratio_mean=0.2460`
  - `step100`：`solved=31`，`accepted@1=0.062`，`pass_ratio_mean=0.2547`
  - `step200`：`solved=46`，`accepted@1=0.092`，`pass_ratio_mean=0.2933`
  - 也就是说，`step200` 是第一个同时在 `solve count` 和 `mean partial pass` 上都明确超过 baseline 的 continuation checkpoint
- 但 `step200` 不是“已经足够稳的最终解”，而是“最适合拿来做 repair diagnosis 的锚点”：
  - `timeout` 和 `judge_time` 尾部明显变重
  - solve-set churn 依然很大：`baseline -> step200` 有 `24` 个 solve gains，但也有 `14` 个 solve drops
  - 这说明 continuation 不是在一个稳定底座上单调变强，而是在“得到一些新解法”的同时“丢掉一些旧能力”
- `strict codecontests_valid (117)` 也给出相同信号：
  - `baseline`：`accepted@1=0.0256`，`pass_ratio_all=0.1422`
  - `step100`：`accepted@1=0.0513`，`pass_ratio_all=0.1576`
  - `step120`：`accepted@1=0.0171`，`pass_ratio_all=0.1283`
  - `step200`：`accepted@1=0.0513`，`pass_ratio_all=0.1867`
  - `step200` 的意义主要是：更高的 partial correctness、更低的 `runtime_error`，但并没有从根本上消灭 hardest set 的 `wrong_answer` 主体
- 这也是当时决定转去做 targeted repair / teacher-SFT 的根本原因：
  - `step200` 已经证明同配方 RL 继续训练“能学到东西”
  - 但它同时暴露出 solve-set 漂移、late timeout tail、high-partial unsolved 池很大
  - 这类信号更像“需要一个 stabilizing prior / repair prior”，而不是“继续同配方长跑 RL 就会自然收敛”

### 17.29 2026-04-04 的最新进展：Phase 2 repair + anchor SFT v1 已完整跑通，但当前不应取代 `pre-SFT step200` 作为 RL 主线起点

这部分覆盖上面对“Phase 2 SFT 可能作为下一阶段起点”的旧预期。当前请优先以这里为准。

先补一条背景，说明为什么当时会在 `step200` 之后专门分出一条 Phase 2 repair + anchor SFT 支线。

#### 为什么当时要围绕 `step200` 做 targeted repair / teacher-SFT

`step200` 不是因为“已经足够好”才拿来做 SFT，而是因为它同时满足了两件事：

- 它是当时最强的 continuation checkpoint
- 它也最完整地暴露了当前纯 RL 线路的核心问题

也就是说，`step200` 更像“最佳 diagnosis anchor”，不是“最终可以直接替代后续一切策略的终点”。

当时从 `100 -> 200` continuation 和后续 strict eval / `valid_big` full-output 里，已经比较明确地看到下面这些问题：

1. solve-set churn 明显，稳定性不够
- `valid_big (500)` 上：
  - `baseline` solves `36`
  - `step100` solves `31`
  - `step200` solves `46`
- 但 solve overlap 并不稳定：
  - `baseline ∩ step200 = 22`
  - `step100 ∩ step200 = 21`
  - `solved by all three = 16`
- 这意味着 continuation 不是“保住旧能力再叠加新能力”，而是“新增一些解，同时丢失一些旧解”

2. aggregate 指标更好，但 hardest set 的主失败模式没变
- `strict codecontests_valid (117)` 上，`step200` 的 headline 确实是当时 continuation 最优：
  - `accepted@1=0.0513`
  - `pass_ratio_all=0.1867`
  - `runtime_error_rate=0.1282`
- 但 dominant terminal state 依旧是 `wrong_answer`
  - baseline `wrong_answer_rate=0.6923`
  - `step200 wrong_answer_rate=0.7009`
- 这说明 `step200` 主要是在“减少 RE、提升 partial pass”，而不是已经把 hardest tasks 大规模推到 AC

3. late continuation 的 timeout / 慢尾更重
- `valid_big` 上：
  - baseline `timeout=0.030`
  - `step100 timeout=0.040`
  - `step200 timeout=0.066`
  - `avg_judge_time` 也从 `41.88s -> 69.48s`
- `strict 117` 上：
  - baseline `timeout_rate=0.0684`
  - `step200 timeout_rate=0.1197`
- 也就是：`step200` 不是简单“更快更强”，而是经常“更接近正确，但更慢、更容易拖到 timeout tail”

4. 训练过程本身没有崩，但后半程已经能看到 plateau + long-tail 加重
- continuation `100 -> 200` 是稳定跑完的，没有 invalid/truncation 崩坏
- 但训练窗口里：
  - 后半程 `timing_s/reward` 明显变重
  - `fast_val_16` 只能看到强波动，看不出稳定单调提升
  - 很多收益体现为 `runtime_error -> wrong_answer` 或 `high partial but unsolved`
- 这类信号说明：问题已经不太像“trainer 没学到”，而更像“纯 RL 缺一个 stabilizing / repair prior”

5. 模型输出日志直接显示 hardest failure motifs 没有被 continuation 自然修掉
- `Codeforces/1572/C`
  - `step200` 仍然反复写 counting / frequency heuristic
- `Codeforces/1574/D`
  - 仍然能看到 `cartesian product` / 枚举式错误结构
- `Codeforces/1553/F`
  - 仍然是 nested-loop brute force 家族
- `Codeforces/1569/C`
  - 则更像“局部更接近正确，但容易掉进 timeout-heavy collapse”

6. 也正因为 `step200` 已经有大量 near-miss / high-partial 样本，所以它非常适合拿来做 repair diagnosis
- `strict 117` 上，`step200` 相比 `step100` 的代表性新 partial gains 包括：
  - `Codeforces/1551/C`: `0.00 -> 0.98`
  - `Codeforces/1552/F`: `0.00 -> 0.80`
  - `Codeforces/1553/I`: `0.02 -> 0.82`
  - `Codeforces/1560/E`: `0.00 -> 0.66`
  - `Codeforces/1569/D`: `0.36 -> 0.78`
- 这些题说明模型已经进入“正确算法邻域”，但还没稳定收口，非常适合 teacher-SFT / repair-SFT 转化

所以当时决定做 targeted repair / teacher-SFT，不是因为觉得 RL 已经失败，而是因为：

- `step200` 已经证明这条 RL 线是 trainable 的
- 但它也明确暴露出：
  - solve-set 不稳
  - timeout tail 上升
  - structural hard motifs 仍顽固
  - near-miss pool 足够大，值得用 teacher 信号把它们往 AC 推

Phase 2 SFT 就是在这个背景下被启动的：它本质上是一次“给纯 RL 补一个 repair prior / stabilizing prior”的实验，而不是“否定 RL，转向 SFT 主线”。

当前已经完成的是：

1. Phase 2 repair teacher 数据准备
2. anchor teacher 数据准备
3. `step200 -> short continual SFT` 训练
4. `pre-SFT step200` 与 `SFT step8/12/16/20` 的同口径 full-suite 外部评测

结论先写在前面：

- 当前这轮 teacher-SFT `v1` **没有产生足够强的净收益**
- 当前 overall 最强的仍然是 **`pre-SFT step200`**
- 因此 **当前不建议从 SFT checkpoint 继续 RL**
- 如果主线要继续推进，当前更合理的起点仍然是 **`pre-SFT step200`**
- 但这**不影响**在 RL 继续跑的同时，并行扩展下一版更强的 teacher-SFT 数据资产

#### A. Phase 2 SFT 数据资产已经冻结

这轮数据准备的设计、约束和构造策略，不要在 handoff 里重复推导，直接看这些文档：

- repair 数据默认策略：
  - [sft_repair_data/repair_sft_training_defaults.md](sft_repair_data/repair_sft_training_defaults.md)
- anchor 数据策略：
  - [sft_repair_data/anchor_data_strategy.md](sft_repair_data/anchor_data_strategy.md)
- anchor v1 构建计划：
  - [sft_repair_data/anchor_v1_build_plan.md](sft_repair_data/anchor_v1_build_plan.md)
- Phase 2 SFT runbook：
  - [sft_repair_data/phase2_sft_runbook.md](sft_repair_data/phase2_sft_runbook.md)

当前已经落盘并可复用的数据资产有三层：

1. repair patch set
2. anchor pool
3. final mixed SFT parquet

关键文件位置：

- repair：
  - [sft_repair_data/v1/repair_sft_train_v1.parquet](sft_repair_data/v1/repair_sft_train_v1.parquet)
  - [sft_repair_data/v1/repair_sft_val_v1.parquet](sft_repair_data/v1/repair_sft_val_v1.parquet)
  - [sft_repair_data/v1/repair_sft_split_manifest_v1.json](sft_repair_data/v1/repair_sft_split_manifest_v1.json)
  - [sft_repair_data/v1/repair_sft_qc_report_v1.md](sft_repair_data/v1/repair_sft_qc_report_v1.md)
- anchor：
  - [sft_repair_data/anchor_v1/anchor_train_v1.parquet](sft_repair_data/anchor_v1/anchor_train_v1.parquet)
  - [sft_repair_data/anchor_v1/anchor_qc_report_v1.md](sft_repair_data/anchor_v1/anchor_qc_report_v1.md)
- final mix：
  - [sft_repair_data/final_v1/phase2_sft_train_v1.parquet](sft_repair_data/final_v1/phase2_sft_train_v1.parquet)
  - [sft_repair_data/final_v1/phase2_sft_val_v1.parquet](sft_repair_data/final_v1/phase2_sft_val_v1.parquet)
  - [sft_repair_data/final_v1/phase2_mix_manifest_v1.json](sft_repair_data/final_v1/phase2_mix_manifest_v1.json)

当前这版数据的真实规模和口径是：

- repair strict accepted：`27`
- repair train / val：`19 / 8`
- repair 经过：
  - `prompt_sha256` 去重
  - compatibility pruning
- repair train bucket 分布：
  - `high_partial_conversion = 10`
  - `anti_regression = 4`
  - `structural_hard = 5`
- 当前 repair train **不包含** `timeout_tail`
- anchor strict accepted：`53`
- final mix 默认采用：
  - `repair : anchor = 1 : 1`
- final mixed train / val：`38 / 8`

这里有一个非常重要的口径，后续不要忘：

- 当前 `timeout_tail` 在 `final_v1` 里是 **guardrail canary**
- 它**不是**这轮 train parquet 里的直接监督目标

这也意味着：如果后面发现这轮 SFT 没修 timeout，不要把这个结果误读成“trainer 没学到应该学的内容”，因为当前训练数据本来就没有真正教它 timeout-tail。

#### B. 这轮 SFT 的训练方式

训练入口脚本：

- [ops/run_phase2_repair_sft.sh](ops/run_phase2_repair_sft.sh)

训练器主入口仍然是：

- [`../../verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py)

数据格式是：

- `MultiTurnSFTDataset`
- parquet `messages` 列
- assistant-only loss mask

这部分代码路径和数据读取逻辑已经专门核过；当前 `final_v1` 的 `messages` parquet 能被 trainer 正常读取，loss mask 也正常，不是“数据根本没训进去”的问题。

这轮实际训练 run 的关键口径如下：

- init checkpoint：
  - `pre-SFT step200`
- run 名称：
  - `phase2_repair_anchor_sft_step200_r1a1_v1`
- 默认 mix：
  - `repair : anchor = 1 : 1`
- 学习率：
  - `5e-7`
- `train_batch_size = 8`
- `micro_batch_size_per_gpu = 1`
- `total_training_steps = 20`
- `save_freq = 4`
- `test_freq = 2`
- `warmup_ratio = 0.0`
- `data.max_length = 4096`

当前应保留的训练产物位于远端：

```text
/workspace/verl/checkpoints/rlvr_coding_model/phase2_repair_anchor_sft_step200_r1a1_v1
```

这轮训练实际保留下来的 SFT checkpoints 是：

- `global_step_8`
- `global_step_12`
- `global_step_16`
- `global_step_20`

其中 `global_step_4` 曾经保存过，但已被 `max_ckpt_to_keep=4` 正常清理，不要把它当成缺失异常。

#### C. 这轮 SFT 的外部 full-suite 评测已经真实完成

完整评测方案已经冻结在：

- [sft_repair_data/final_v1/phase2_sft_eval_suite_v1.md](sft_repair_data/final_v1/phase2_sft_eval_suite_v1.md)
- [sft_repair_data/final_v1/phase2_sft_eval_suite_v1.json](sft_repair_data/final_v1/phase2_sft_eval_suite_v1.json)

实际评测结果根目录在远端：

```text
/workspace/verl/coding_model_project/outputs/phase2_sft_eval_suite_v1
```

当前已经完成的 5 个同口径评测对象是：

- `phase2_sft_pre_sft_step200_fullsuite_v1_retry1`
- `phase2_sft_step8_fullsuite_v1`
- `phase2_sft_step12_fullsuite_v1`
- `phase2_sft_step16_fullsuite_v1`
- `phase2_sft_step20_fullsuite_v1_retry1`

这些 full-suite run 当前都已经核对过：

- `summary.json`
- `metrics.json`
- `run_info.json`
- `per_problem/*.jsonl`
- `qa_logs/*.jsonl`

并且 5 个 checkpoint 的 eval config 是一致的：

- `temperature = 0.0`
- `top_p = 1.0`
- `max_new_tokens = 2048`
- `sandbox_url = http://localhost:8090`
- `max_concurrent_requests = 32`
- `max_concurrent_judges = 24`
- `verifier_limiter_budget = 24`
- `batch_size = 24`

`8090` 后面挂的是 `8081/8082` 双 backend LB，这条链在 full-suite 运行时也已经真实打通过。

#### D. full-suite 的 checkpoint 级结论

当前最应该保留的对比表是：

| slice | pre-SFT step200 | step8 | step12 | step16 | step20 |
|---|---:|---:|---:|---:|---:|
| `repair_val` `accepted@1 / pass_ratio_mean` | `0.125 / 0.440` | `0.125 / 0.425` | `0.125 / 0.408` | `0.125 / 0.338` | `0.125 / 0.408` |
| `retention_canary_codecontests` | `0.200 / 0.575` | `0.200 / 0.541` | `0.100 / 0.387` | `0.200 / 0.481` | `0.200 / 0.537` |
| `timeout_canary_codecontests` | `0.000 / 0.731` | `0.000 / 0.718` | `0.000 / 0.731` | `0.000 / 0.724` | `0.000 / 0.721` |
| `structural_hard_watchlist` | `0.000 / 0.270` | `0.000 / 0.270` | `0.000 / 0.270` | `0.000 / 0.270` | `0.000 / 0.270` |
| `humaneval_mini` | `0.750 / 0.750` | `0.625 / 0.625` | `0.750 / 0.750` | `0.625 / 0.625` | `0.625 / 0.625` |
| `mbpp_reg_mini` | `0.750 / 0.750` | `0.750 / 0.750` | `0.750 / 0.750` | `0.750 / 0.750` | `0.750 / 0.750` |

直接读表就够得出结论：

- 当前 **overall 最强的仍然是 `pre-SFT step200`**
- 在 SFT checkpoints 里：
  - `step20` 可以视为“最不差”的一个
  - 但仍然没有超过 `pre-SFT step200`

更具体一点：

1. `repair_val`
- 没有任何一个 SFT checkpoint 带来新的 solve
- 只是 `pass_ratio_mean` 有不同程度回退
- `step16` 的回退最重

2. `retention_canary_codecontests`
- 当前最重要的 durable regression 是：
  - `Codeforces/909/A`
- pre-SFT 是 AC
- SFT 后所有 checkpoints 都回退到 `0.546875` 的 WA

3. `timeout_canary_codecontests`
- 当前没有任何 solve gain
- `step16` 只是 judge time 更低，不等于真正修复 timeout-tail

4. `structural_hard_watchlist`
- 完全没动
- 这轮 SFT 对结构性 hard cases 没有实质性改善

5. benchmark mini
- `HumanEval` 只有 `pre-SFT` 和 `step12` 维持住了 `0.75`
- `MBPP_reg` 基本平

#### E. 逐题行为分析后的真实读法

逐题行为不要只看 aggregate。当前更可靠的读法是：

- 这轮 SFT **不是完全没学到任何东西**
- 但它带来的局部 gain 太少，且被 regression 抵消了

当前最重要的逐题结论：

1. `repair_val` 没有新增 solve
- `Codeforces/1545/B` 是最清楚的 repair regression：
  - `0.86 -> 0.80 -> 0.68 -> 0.04 -> 0.64`
- `Codeforces/1420/C2` 有波动，但没有超过 pre

2. `retention_canary_codecontests`
- `Codeforces/909/A` 是最清楚的 anti-regression failure
- `Codeforces/371/D` 看上去像 gain，但不要高估：
  - 我已经核过 `qa_logs` 的 response hash
  - `pre / step8 / step16 / step20` 这几个 checkpoint 在这题上生成的是**同一份代码**
  - 但 verdict 却从 `48/50` 变成 `50/50`
  - 这更像评测噪声或边界波动，不该算成一个稳定、可归因到 SFT 的 gain

3. `timeout_canary_codecontests`
- `Codeforces/1538/B` 说明 SFT 没有真正改善 timeout-tail：
  - pre：`0.74` 的 WA，约 `35.6s`
  - `step8`：退化成 timeout，约 `122.6s`
  - `step12/16/20`：又回到同样的 `0.74` WA，但更慢

4. `structural_hard_watchlist`
- 六题基本完全平
- 当前没有证据表明这轮 SFT 动到了 hardest algorithmic family

因此更准确的结论是：

- 这轮 SFT **不是 crash**
- 但也**没有形成值得接管主线的净收益**

#### F. 当前是否应从 SFT checkpoint 继续 RL

当前建议是：

- **不要**

更明确一点：

- 当前不建议从 `step8/12/16/20` 任何一个 SFT checkpoint 继续主线 RL
- 当前更合理的主线起点仍然是：
  - `pre-SFT step200`

原因很简单：

1. `pre-SFT step200` 在当前同口径 full-suite 上是 overall 最强
2. 当前 SFT checkpoints 没有带来足够强的 repair gain
3. 当前 SFT checkpoints 还带来了可观察的 retention regression
4. 这轮 SFT 的训练数据本身就没有真正覆盖 timeout-tail train bucket，因此不应把它当成“继续 RL 前必须先走的一步”

如果主线现在要继续推进，推荐口径是：

- **主线继续用 `pre-SFT step200` 跑下一段 RL**
- 然后再做一轮 checkpoint shortlist 评测与行为分析

当前比较自然的做法是：

1. 从 `pre-SFT step200` 再继续一段 A1 formal RL
2. 训练期间保留当前 fast-val / reward-side 监控
3. 到一个新的 shortlist 后，再做 strict apples-to-apples eval

至于这“一段”到底取多长，当前建议优先保持和已有 continuation 粒度一致，不要一上来跑太长无人看守的区间。一个安全口径是：

- 先再推进一段中等长度 continuation
- 中途按固定 checkpoint 间隔保留可评测点
- 让后续 shortlist 决策仍然基于 strict eval，而不是只看 fast-val

#### G. 当前是否值得补做更大的 SFT full eval

当前我的建议是：

- **不需要为了决定 RL 主线而额外补更多 SFT 评测，当前证据已经足够**

如果只是为了“决定下一段 RL 从哪里起跑”，现在的 canary full-suite 已经够用了，结论非常清楚：

- `pre-SFT step200 > SFT checkpoints`

如果后面要把这轮 SFT 写成更正式的归档结论，可以低优先级补两件事：

- `pre-SFT step200` vs `SFT step20`
- 在：
  - `codecontests_valid (117)`
  - `valid_big (500)`
  - full `humaneval`
  - full `mbpp_reg`
  上再做一轮最终归档对比

但这一步更像“论文式补证据”或“简历项目整理”，不是主线实验推进的 blocker。

#### H. RL 继续跑的同时，是否值得并行继续准备更多 SFT 数据

当前建议是：

- **完全值得，而且应该并行推进**

更明确一点：

- 主线 RL 可以继续从 `pre-SFT step200` 往下跑
- 与此同时，可以充分利用空闲的 Claude Code 额度，继续扩下一版 teacher 数据

这两条线是并行而不是互斥的。

当前最值得扩的数据方向不是“再做一版很像现在的 patch set”，而是补当前 `v1` 的明显缺口：

1. `timeout_tail` train bucket
- 当前 `final_v1` 里它几乎没有真正进入 train
- 下一版应补：
  - 更短
  - 更 student-compatible
  - 更明确针对 slow-tail / complexity 的 teacher 样本

2. anti-regression guardrail 数据
- 当前 `909/A` 这种简单但敏感的 regression 说明：
  - 还需要一批更稳的 must-keep style / easy-CP / lexical-boundary 类数据
- 这类数据应该继续保持：
  - train-only
  - teacher-generated
  - strict QC

3. 更强的 repair v2
- 当前 `1553/I` 是 underfilled seed
- 还可以补更多同类 train-only analogs
- 同时也可以把 `valid_big` 上已经识别出的 hard / near-miss seeds 再扩一轮 retrieval + teacher generation

4. 更强的 anchor 资产
- 当前 `anchor_v1` 已经够支撑 first pass
- 但如果后面还要再试 SFT v2，可以继续扩 teacher-only anchor：
  - stable protocol
  - general coding
  - easy / medium competitive programming

需要强调的边界仍然不变：

- 继续只用 `train_wo_valid_big` 做 teacher 数据
- 不要把 `valid / valid_big / test / humaneval / mbpp_reg` 放进训练集
- teacher 数据继续走 strict QC
- 新增资产要和 `prompt_sha256` hygiene 一起维护

#### I. 当前最推荐的主线顺序

如果下一位 agent 现在接手，推荐顺序是：

1. 把主线 RL continuation 从 `pre-SFT step200` 继续推进
2. 保持当前 reward / verifier / eval 主链不变
3. 在 RL 跑着的时候，并行准备下一版更强的 SFT 数据：
   - timeout-tail
   - anti-regression guardrails
   - additional repair analogs
   - stronger anchors
4. 等下一轮 RL shortlist 出来后，再决定：
   - 是直接继续 RL
   - 还是先插入一版更强的 SFT v2

当前最不推荐的动作是：

- 因为这轮 SFT 跑通了，就直接把 `step20` 当成新主线起点继续 RL

#### J. 给下一位 agent 的一句话交接

当前 Phase 2 SFT `v1` 已经把“数据准备 -> short continual SFT -> external full-suite eval”整条链跑通了，但当前结果显示它**还不值得替代 `pre-SFT step200` 成为主线 RL 起点**；主线继续从 `pre-SFT step200` 往下跑，同时并行扩充下一版更强的 teacher 数据，是当前最稳也最高效的策略。

---

## 最新更新（2026-04-05）：step400 已接管当前 SFT 主线

这一节是当前最新状态。

如果与前文任何结论冲突，以本节为准。

### 当前最终判断

- `step460` 的 `valid_big(500)` 已完成评测，但**效果不如 `step400`**
- 因此当前用于下一版 SFT 的锚点已经从 `step200` 改为：
  - `grpo step400`
- 当前应继续推进的 SFT 版本是：
  - `repair_v2b_step400`
  - `anchor_v2b_step400`
  - `final_v2b_step400`
- 旧的 `repair_v2a / step200` 线现在只保留为历史准备资产，不再作为主线 SFT 锚点

### 为什么最终选 step400，而不是继续沿用 step200 / step460

当前最关键的三份外部评测结果是：

- `valid_big 500`：
  - `step400`
    - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0_step400_codecontests_validbig500_retry3`
  - `step460`
    - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_formal_observe_lb64_multi4_resume400_to460_seed0_step460_codecontests_validbig500`
- `valid117`：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_valid117/grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0_step400_codecontests_valid117_retry2`
- `canary_v1`：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_canary_v1/grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0_step400_rl_canary_v1_retry1`

核心结论：

1. `step400` 是当前最好的 `valid_big 500` checkpoint
- `accepted@1 = 0.106`
- `pass_ratio_mean = 0.3316`
- `runtime_error_rate = 0.116`
- `timeout_rate = 0.048`
- `avg_judge_time = 33.39s`

2. `step460` 明显回退
- `accepted@1 = 0.096`
- `pass_ratio_mean = 0.3200`
- `runtime_error_rate = 0.126`
- `timeout_rate = 0.068`
- `avg_judge_time = 32.00s`

因此当前正确读法不是“RL 越往后越好”，而是：

- `400 -> 460` 这段 continuation 没有稳定带来真实净收益
- 当前 checkpoint shortlist winner 是 `step400`
- 下一版 SFT 应锚定 `step400`，而不是再回头做 `step200`，也不是贸然切到 `step460`

### step400 在 valid117 / canary_v1 上的正式读法

#### A. `valid117`

`step400` 对应 summary：

- `accepted@1 = 0.05128`，也就是 `6 / 117`
- `pass_ratio_mean = 0.19850`
- `runtime_error_rate = 0.14530`
- `timeout_rate = 0.07692`
- `avg_judge_time = 83.52s`

关键读法：

- solve 数并没有超过旧的 `step200`
- 但执行效率和 partial-credit frontier 更强
- 这说明当前更值得补的是：
  - `high_partial -> final correctness`
  - `anti_regression / stability`
- 而不是把预算过多押在“泛 timeout 修复”

#### B. `canary_v1`

`step400` 的四个 canary slice：

- `retention_canary_codecontests`
  - `accepted@1 = 0.50`
  - `pass_ratio_mean = 0.8007`
- `timeout_canary_codecontests`
  - `accepted@1 = 0.00`
  - `pass_ratio_mean = 0.8377`
  - `timeout_rate = 0.6667`
- `structural_hard_watchlist`
  - `accepted@1 = 0.00`
  - `pass_ratio_mean = 0.4333`
- `mid_val_codecontests_32`
  - `accepted@1 = 0.21875`
  - `pass_ratio_mean = 0.6467`

关键读法：

- retention 比旧的 `step200` repair 时代更稳，但依然不能放松 anti-regression
- timeout-case 已经有明显高 partial，但仍然没真正 solve
- structural-hard 仍未解决，但已经不是“完全没有信号”

因此 step400 repair 的核心目标已经非常明确：

1. `high_partial_conversion`
2. `anti_regression / stability`
3. 保留一条真实 `timeout_tail`
4. 保留一条更小但精的 `structural_hard`

### 当前已经落地的 step400 SFT 资产

当前 step400 主线 SFT 资产已经真实 materialize 完成，下面这些文件都可以直接继续使用：

- 计划与 spec
  - `phase_2_ GRPO/sft_repair_data/v2b_step400/step400_repair_plan.md`
  - `phase_2_ GRPO/sft_repair_data/v2b_step400/repair_v2b_step400_spec.json`
- shortlist / subset
  - `phase_2_ GRPO/sft_repair_data/v2b_step400/seed_manifest_v2b_step400.json`
  - `phase_2_ GRPO/sft_repair_data/v2b_step400/retrieval_candidates_v2b_step400.json`
  - `phase_2_ GRPO/sft_repair_data/v2b_step400/student_reference_eval_shortlist_v2b_step400.jsonl`
  - `phase_2_ GRPO/sft_repair_data/v2b_step400/student_reference_eval_subset_v2b_step400/slice_meta.json`
- student-reference / teacher-request
  - `phase_2_ GRPO/sft_repair_data/v2b_step400/student_reference_profile_v2b_step400.json`
  - `phase_2_ GRPO/sft_repair_data/v2b_step400/student_references_step400.jsonl`
  - `phase_2_ GRPO/sft_repair_data/v2b_step400/teacher_generation_requests_v2b_step400.jsonl`

当前关键规模：

- `student_reference_eval_shortlist_v2b_step400.jsonl`
  - `494` 行
- `student_references_step400.jsonl`
  - `494` 行
- `teacher_generation_requests_v2b_step400.jsonl`
  - `624` 行
  - `312` 条 `concise_standard`
  - `312` 条 `runtime_optimized`
  - `300` 个 unique `teacher_group_id`

按 unique teacher group 统计的 bucket 覆盖：

- `high_partial_conversion = 96`
- `anti_regression = 84`
- `timeout_tail = 72`
- `structural_hard = 48`

这点很重要：

- 当前 step400 repair 的前置覆盖面已经明显大于之前 `step200` 那版 repair-v1
- “raw 数据太小导致 SFT 没法形成净收益”这个老问题，现在已经明显缓解
- 后续真正需要守住的是：
  - teacher QC 后的 strict-accepted 数量
  - 以及 final mix 的 bucket 平衡

当前 step400 spec 的核心合同是：

- repair bucket 配额
  - `high_partial_conversion = 18 train / 4 val`
  - `anti_regression = 14 train / 4 val`
  - `timeout_tail = 8 train / 2 val`
  - `structural_hard = 8 train / 2 val`
- `minimum_unique_accepted_total = 68`
- mixed SFT 配比
  - `repair_weight = 3`
  - `anchor_weight = 2`
  - `repair_train_target = 48`
  - `repair_val_target = 12`
  - `anchor_unique_target = 24`
  - `anchor_effective_target = 32`

anchor 合同：

- bucket
  - `stable_protocol = 6`
  - `easy_medium_cp = 10`
  - `general_coding = 8`
- source
  - `reused_v1 = 10`
  - `new_v2a = 14`
- source-bucket split
  - `reused_v1`
    - `stable_protocol = 6`
    - `easy_medium_cp = 2`
    - `general_coding = 2`
  - `new_v2a`
    - `stable_protocol = 0`
    - `easy_medium_cp = 8`
    - `general_coding = 6`

### step400 student-reference strict eval 已经打通

本轮 step400 student-reference strict eval 使用的是隔离的 reward infra：

- isolated `4 sandbox + 1 LB`
- `SANDBOX_URL = http://localhost:8096`
- backend ports
  - `8097`
  - `8098`
  - `8099`
  - `8100`
- `MAX_CONCURRENT = 64`
- `MAX_CONCURRENT_JUDGES = 64`
- `VERIFIER_LIMITER_BUDGET = 64`
- `BATCH_SIZE = 64`
- `GPU_DEVICE = 2`
- `VLLM_PORT = 8004`

对应 remote eval 输出目录：

- `/workspace/verl/coding_model_project/outputs/repair_v2b_step400_student_reference_eval/global_step_400_stepref_v2b_step400`

对应 checkpoint：

- `/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0/global_step_400`

对应 strict eval summary：

- `total_problems = 494`
- `accepted@1 = 0.11740890688259109`
- `pass_ratio_mean = 0.3509789243094511`
- `runtime_error_rate = 0.09716599190283401`
- `timeout_rate = 0.07894736842105263`
- `avg_judge_time = 39.77960769608919`

一句话结论：

- step400 已经不只是“理论上适合作为新 anchor”
- 而是 student-reference strict eval / teacher-request 这条链已经真的落好了

### 远端实例停机前的环境快照

下面这份快照是停机前确认过的状态，后面如果要重启实例，可以直接照着恢复。

#### A. 系统与 GPU

remote snapshot 时间：

- `2026-04-04T10:02:04-07:00`

环境：

- `4 x NVIDIA GeForce RTX 5090`
- `Python 3.12.11`
- `uv 0.8.22`

停机前 GPU 状态是：

- `0/1/2/3` 四张卡都空闲
- 显存占用约 `2 MiB`

#### B. 当前没有残留训练 / vLLM / eval 进程

停机前核过：

- `vllm` 进程：无
- `phase0_eval.py`：无
- `model_merger`：无
- RL trainer 主进程：无

也就是说：

- 当前实例里没有“还在后台偷偷跑”的训练或评测
- 只有 sandbox backend 和 nginx LB 还活着

#### C. 当前 sandbox / reward infra 状态

停机前端口状态：

- `8001 CLOSED`
- `8002 CLOSED`
- `8004 CLOSED`
- `8016 CLOSED`
- `8080 CLOSED`
- `8090 OPEN`
- `8094 OPEN`
- `8096 OPEN`
- `8097 OPEN`
- `8098 OPEN`
- `8099 OPEN`
- `8100 OPEN`

当前 state root：

- `/root/sandboxfusion-multi`
- `/root/sandboxfusion-step200`
- `/root/sandboxfusion-step400`
- `/root/sandboxfusion-venv`

当前几套 pool 的用途：

1. `sandboxfusion-multi`
- 历史共用池
- 主要对应老的 shared pool
- 当前 LB 主口径还是 `8090`
- backend 是 `8081/8082/8083/8084`
- 这套还在，但更偏“历史/共用”

2. `sandboxfusion-step200`
- 旧的 `step200 student-reference` 独立池
- LB：
  - `8094`
- backend：
  - `8085`
  - `8086`
- state env 仍在：
  - `backend_8085.env`
  - `backend_8086.env`
  - `stage_state.env`

3. `sandboxfusion-step400`
- 当前最重要的 step400 SFT 专用池
- LB：
  - `8096`
- backend：
  - `8097`
  - `8098`
  - `8099`
  - `8100`
- 当前 state env：
  - `backend_8097.env`
  - `backend_8098.env`
  - `backend_8099.env`
  - `backend_8100.env`
  - `stage_state.env`

step400 backend 的探针结果记录在 env 里：

- `8097` `SMOKE_P95_MS = 60.77057123184204`
- `8098` `SMOKE_P95_MS = 66.82050228118896`
- `8099` `SMOKE_P95_MS = 66.49088859558105`
- `8100` `SMOKE_P95_MS = 61.90979480743408`

step400 日志路径：

- `/root/sandboxfusion-step400/logs/sandbox_8097.log`
- `/root/sandboxfusion-step400/logs/sandbox_8098.log`
- `/root/sandboxfusion-step400/logs/sandbox_8099.log`
- `/root/sandboxfusion-step400/logs/sandbox_8100.log`
- `/root/sandboxfusion-step400/nginx/sandbox_lb_8096.access.log`
- `/root/sandboxfusion-step400/nginx/sandbox_lb_8096.error.log`

#### D. SSH 的一个小坑

当前 `ssh config` 会尝试做本地 `8080` 转发。

如果本地 `8080` 已经被占用，SSH 时会看到类似：

- `bind [::1]:8080: Address already in use`
- `Could not request local forwarding`

这个警告**不会阻止远端命令执行**，只是本地转发失败。

如果只是做远端检查，不依赖本地 `8080` 转发，可以：

- 直接忽略这个 warning
- 或者本地改用 `ssh -o ClearAllForwardings=yes vastai2 ...`

### 之后重启实例时，优先恢复哪些服务

#### A. step400 isolated sandbox

最重要的是先恢复当前 step400 的专用 reward infra：

remote 机器上执行：

```bash
bash "/workspace/verl/coding_model_project/phase_2_ GRPO/ops/setup_repair_v2b_step400_isolated_sandbox.sh"
```

这条脚本会调用：

- `phase_2_ GRPO/ops/setup_eval_sandbox_4x2.sh`

并按如下默认值拉起：

- `STATE_ROOT=/root/sandboxfusion-step400`
- `BASE_PORT=8097`
- `BACKEND_COUNT=4`
- `LB_BASE_PORT=8096`
- `LB_COUNT=1`
- `BACKENDS_PER_LB=4`

#### B. 如果需要重跑 step400 student-reference strict eval

remote 机器上执行：

```bash
cd /workspace/verl/coding_model_project
SETUP_SANDBOX=false \
PREPARE_STATIC_ASSETS=false \
BUILD_TEACHER_REQUESTS=false \
GPU_DEVICE=2 \
VLLM_PORT=8004 \
SANDBOX_URL=http://localhost:8096 \
MAX_CONCURRENT=64 \
MAX_CONCURRENT_JUDGES=64 \
VERIFIER_LIMITER_BUDGET=64 \
BATCH_SIZE=64 \
bash "/workspace/verl/coding_model_project/phase_2_ GRPO/ops/run_repair_v2b_step400_student_reference_eval.sh"
```

这条脚本默认 checkpoint 是：

- `/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0/global_step_400`

默认输出目录是：

- `/workspace/verl/coding_model_project/outputs/repair_v2b_step400_student_reference_eval/global_step_400_stepref_v2b_step400`

#### C. 相关脚本

当前和 step400 SFT 最相关的 remote ops 脚本是：

- `phase_2_ GRPO/ops/setup_eval_sandbox_4x2.sh`
- `phase_2_ GRPO/ops/setup_repair_v2b_step400_isolated_sandbox.sh`
- `phase_2_ GRPO/ops/run_repair_v2b_step400_student_reference_eval.sh`
- `phase_2_ GRPO/ops/run_repair_v2a_student_reference_eval.sh`

### 给下一位 agent 的一句话交接

当前最该继续推进的不是旧的 `step200/v2a`，而是已经落好 student-reference / request 资产的 `step400/v2b_step400`：`step460` 的 `valid_big` 回退已经说明当前 RL shortlist winner 仍然是 `step400`，因此下一步应直接围绕 `teacher_generation_requests_v2b_step400.jsonl` 生成 Claude teacher 数据，完成 QC、parquet 和 final mix，再启动一版真正锚定 `step400` 的 repair-SFT。

## 最新补充（2026-04-05 晚些时候）：`step520` probe 已完成，当前正在跑 `step550 -> 580` 的 `b16 + micro8` RL probe

如果与上面任何“接下来应立即转入 SFT”的表述冲突，以本节为准：

- 先明确 batch 时间线，避免误会：
  - `step400`、`step460`、`step520` 以及更早那批 continuation checkpoint，全部都是 `train_batch_size = 8 / ppo_mini_batch_size = 8`
  - `b16` 是 **`step520 -> 580`** 才首次引入的后续 probe
  - 因此 `step520 valid_big` 必须按 **`b8/mini8` checkpoint** 解读，不能误认为它已经受益于后面的 `b16`
- `step400` 仍然是当前 **正式 `valid_big(500)` winner**
- 但在真正停止 RL 主线前，又追加做了两条 probe：
  1. `step460 -> 520`：在 **`train_batch_size = 8 / ppo_mini_batch_size = 8` 不变** 的前提下，验证 `6 sandbox + 96 limiter` 下继续同配 RL 是否还能涨
  2. `step550 -> 580`：验证 `train_batch_size=16 / ppo_mini_batch_size=16 / ppo_micro_batch_size_per_gpu=8 / 8 sandbox + 128 limiter` 是否能进一步改善 update 吞吐

### 1. `step520 valid_big(500)` 已完成，结论仍然是不如 `step400`

正式结果目录：

- `step520`
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_formal_observe_lb96_multi6_resume460_to520_seed0_step520_codecontests_validbig500`
- `step400`
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0_step400_codecontests_validbig500_retry3`

关键指标：

- `step400`
  - `53 / 500`
  - `accepted@1 = 0.106`
  - `pass_ratio_mean = 0.3316`
  - `runtime_error_rate = 0.116`
  - `timeout_rate = 0.048`
- `step520`
  - `46 / 500`
  - `accepted@1 = 0.092`
  - `pass_ratio_mean = 0.3193`
  - `runtime_error_rate = 0.142`
  - `timeout_rate = 0.080`

正确读法：

- `step520` 是 **最后一个 `b8/mini8` 的 probe checkpoint**
- `step520` 不是运行事故，也不是之前 `step400 retry2` 那种 `empty_output collapse`
- 它是一次**干净但回退**的 probe
- 因此当前 `valid_big` winner 仍然是 `step400`

对应 watcher / 评测触发日志：

- `/workspace/eval_logs/grpo_a1_formal_observe_lb96_multi6_resume460_to520_seed0_step520_validbig500_then_b16_wait.out`

### 2. `step520 -> 580` 的 `b16 on-policy` probe 已真实启动，并在 `550` 处切到下一条 `micro8` probe

先做的 probe 是：

- 训练脚本：
  - `phase_2_ GRPO/ops/run_grpo_a1_resume520_to580_b16.sh`
- 核心配置：
  - `train_batch_size = 16`
  - `ppo_mini_batch_size = 16`
  - `rollout_n = 8`
  - `limiter_budget = 96`
  - `reward pool = 6 sandbox`
- 实验名：
  - `grpo_a1_formal_observe_lb96_multi6_b16_onpolicy_resume520_to580_seed0`
- 主日志：
  - `/workspace/eval_logs/grpo_a1_formal_observe_lb96_multi6_b16_onpolicy_resume520_to580_seed0.out`

也就是说，**`b16` 是从 `step520` 继续训练之后才开始生效**；它不会回溯改变 `step400/460/520` 这些 checkpoint 的解释口径。

这条 `b16` probe 已经真实跑到 `550` 并完成保存。

其中一个有用的内部读数是 `540 fast-val`：

- dump：
  - `/workspace/verl/validation_dumps/grpo_a1_formal_observe_lb96_multi6_b16_onpolicy_resume520_to580_seed0/540.jsonl`
- 摘要：
  - `accepted_rate = 0.0`
  - `pass_ratio_mean = 0.2720`

所以到 `550` 为止，`b16 + micro4` 的收益**仍未被 fast-val 证明**，因此又做了下一层 probe：

- 在 `550` 停旧 run
- 改成 `ppo_micro_batch_size_per_gpu = 8`
- 同时把 reward pool 从 `6 backend` 扩到 `8 backend`

### 3. `550` 自动切换脚本曾失败一次，当前已经手工修复

相关脚本：

- watcher：
  - `phase_2_ GRPO/ops/run_grpo_step550_switch_to_b16_micro8_wait.sh`
- sandbox / LB 扩容：
  - `phase_2_ GRPO/ops/setup_reward_sandbox_lb8.sh`
- 新训练脚本：
  - `phase_2_ GRPO/ops/run_grpo_a1_resume550_to580_b16_micro8.sh`

最初 watcher 的执行结果是：

- 成功等到 `global_step_550`
- 成功停止旧的 `b16/micro4` run
- 但在 `SETUP_REWARD_POOL` 阶段失败

失败根因有两个：

1. `setup_reward_sandbox_lb8.sh` 的 staging guardrail 过于保守
- `8087/8088` 在 staging 阶段被误杀
- 触发条件包括：
  - `MemAvailable` 相对下降过大
  - `swap usage` 相对 clean-host baseline 上升

2. 新 run 第一次被拉起时，`8090` 实际没有服务
- 主日志里出现大量：
  - `Failed to establish a new connection: [Errno 111] Connection refused`

已做的修复：

- 本地脚本 `setup_reward_sandbox_lb8.sh` 已补充更宽松的 guardrail 默认值
- 之后手工恢复了：
  - `8081..8088`
  - `8090 -> 8081..8088`
- 再从 `global_step_550` 手工重拉当前 run

注意：

- `run_grpo_step550_switch_to_b16_micro8_wait.sh` 的历史日志仍保留在：
  - `/workspace/eval_logs/grpo_step550_switch_to_b16_micro8_wait.out`
- 但**它不是当前现场的唯一权威来源**
- 当前应以“新的训练进程 + 当前主日志 + 当前 `8090` access log”为准

### 4. 当前 live reward infra 状态：`8090` 已经从旧的 `4 backend` 升到 `8 backend`

这点非常重要，因为它已经覆盖了上面较早章节里的旧状态。

当前确认活着并被 RL 主线实际使用的共享池是：

- LB：
  - `8090`
- backend：
  - `8081`
  - `8082`
  - `8083`
  - `8084`
  - `8085`
  - `8086`
  - `8087`
  - `8088`

最新真实 `/run_code` 分流（最近约 `800` 条）是：

- `8081: 100`
- `8082: 101`
- `8083: 99`
- `8084: 99`
- `8085: 102`
- `8086: 99`
- `8087: 101`
- `8088: 99`

而且：

- `run_code_ok = 800`
- `run_code_bad = 0`

所以不要再按前文“`8090` 只有 `8081..8084` 四个 backend”的旧口径理解当前现场；
**对当前 RL 主线来说，`8090` 现在已经是 `8 backend`。**

当前最重要的日志路径：

- access log：
  - `/root/sandboxfusion-multi/nginx/sandbox_lb_8090.access.log`
- error log：
  - `/root/sandboxfusion-multi/nginx/sandbox_lb_8090.error.log`
- backend logs：
  - `/root/sandboxfusion-multi/logs/sandbox_8081.log`
  - `/root/sandboxfusion-multi/logs/sandbox_8082.log`
  - `/root/sandboxfusion-multi/logs/sandbox_8083.log`
  - `/root/sandboxfusion-multi/logs/sandbox_8084.log`
  - `/root/sandboxfusion-multi/logs/sandbox_8085.log`
  - `/root/sandboxfusion-multi/logs/sandbox_8086.log`
  - `/root/sandboxfusion-multi/logs/sandbox_8087.log`
  - `/root/sandboxfusion-multi/logs/sandbox_8088.log`

### 5. 当前正在跑的 `micro8` probe：`step550 -> 580`

当前 live run：

- 脚本：
  - `phase_2_ GRPO/ops/run_grpo_a1_resume550_to580_b16_micro8.sh`
- 实验名：
  - `grpo_a1_formal_observe_lb128_multi8_b16_micro8_resume550_to580_seed0`
- checkpoint 输出目录：
  - `/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb128_multi8_b16_micro8_resume550_to580_seed0`
- 主日志：
  - `/workspace/eval_logs/grpo_a1_formal_observe_lb128_multi8_b16_micro8_resume550_to580_seed0.out`

它的配置意图是：

- `train_batch_size = 16`
- `ppo_mini_batch_size = 16`
- `ppo_micro_batch_size_per_gpu = 8`
- `rollout_n = 8`
- `limiter_budget = 128`
- `reward pool = 8 sandbox`

一个容易误读的小点：

- 启动命令里同时会看到一条较早的默认值：
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`
- 以及后面附加的一条 override：
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8`

Hydra 以最后一条 override 为准，所以当前 live run 的最终值是：

- `ppo_micro_batch_size_per_gpu = 8`

### 6. 当前 `micro8` probe 的早期训练读数

当前最有价值的日志不在主训练 log，而在这次 Ray session 的 worker log：

- `/tmp/ray/session_2026-04-05_06-08-53_694796_930211/logs/worker-bb815f4c74c25ffac4f54f49eb27cecc95ec3663daad71ef012f6c07-01000000-938599.out`

目前已经确认：

- 训练已从 `550` 真正推进到 `551`、`552`
- 没有 OOM
- 没有格式侧坏信号

`step551`：

- `training/global_step = 551`
- `verifier/reward_raw_valid_count = 128`
- `verifier/invalid_for_rl_rate = 0.0`
- `verifier/truncated_by_max_tokens_rate = 0.0`
- `verifier/empty_output_rate = 0.0`
- `verifier/timeout_rate = 0.1641`
- `verifier/accepted_rate = 0.0625`
- `verifier/reward_raw_mean = 0.2559`
- `timing_s/reward = 140.07s`
- `timing_s/update_actor = 82.83s`
- `perf/time_per_step = 266.53s`
- `perf/throughput = 94.97`
- `perf/max_memory_reserved_gb = 25.65`

`step552`：

- `training/global_step = 552`
- `verifier/reward_raw_valid_count = 128`
- `verifier/invalid_for_rl_rate = 0.0`
- `verifier/truncated_by_max_tokens_rate = 0.0`
- `verifier/empty_output_rate = 0.0`
- `verifier/timeout_rate = 0.0234`
- `verifier/accepted_rate = 0.125`
- `verifier/reward_raw_mean = 0.2431`
- `timing_s/reward = 68.76s`
- `timing_s/update_actor = 50.85s`
- `perf/time_per_step = 173.28s`
- `perf/throughput = 165.40`
- `perf/max_memory_reserved_gb = 26.74`

当前正确读法：

- `micro=8` 没有引起 OOM
- `max_memory_reserved_gb` 提高到了约 `25.6 -> 26.7GB`
- 但仍然低于 `32GB` 卡的危险区
- `step551` 像是恢复后的第一步重 batch
- `step552` 更接近 steady-state

也就是说：

- `update_actor` 确实从 `82.8s` 降到了 `50.8s`
- `reward` 也从 `140.1s` 降到了 `68.8s`
- `throughput` 从 `95` 提到了 `165`

所以目前这条 `micro8` probe 至少在**前两步**看起来是有希望的，但样本还太少，还不能直接下最终结论。

### 7. 当前磁盘与 checkpoint 状态

为了给后续 probe 腾空间，已经清掉：

- `/workspace/global_step_200`
- `/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb96_multi6_resume460_to520_seed0/global_step_490`

因此当前不要再假设：

- `step490` 仍在本地
- 旧的 stray `global_step_200` 仍在 `/workspace`

### 8. 给下一位 agent 的当前一句话交接

这句交接只适用于 **`step580 valid_big` 落盘前**。最新结论已经被下一小节覆盖：`micro8` probe 最终完成外部 `valid_big(500)` 后，当前 `valid_big` winner 已从 `step400` 更新为 `step580`。

## 最新补充（2026-04-06）：`step580 valid_big(500)` 已完成，当前 `valid_big` winner 更新为 `step580`

正式结果目录：

- `step580`
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_formal_observe_lb128_multi8_b16_micro8_resume550_to580_seed0_step580_codecontests_validbig500`
- `step520`
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_formal_observe_lb96_multi6_resume460_to520_seed0_step520_codecontests_validbig500`
- `step400`
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0_step400_codecontests_validbig500_retry3`

触发与收尾日志：

- `/workspace/eval_logs/grpo_a1_formal_observe_lb128_multi8_b16_micro8_resume550_to580_seed0_step580_codecontests_validbig500_wait.out`

这次 `step580` 的 `valid_big` 是**干净跑完**的：

- `CHECKPOINT_READY -> START_MERGE -> DONE_MERGE -> START_VLLM -> DONE_VLLM_HEALTH -> START_EVAL -> DONE_EVAL`
- `api_error_rate = 0.0`
- `sandbox_error_rate = 0.0`
- 没有出现 `empty_output collapse`

### 1. headline 指标对比

- `step400`
  - `53 / 500`
  - `accepted@1 = 0.106`
  - `pass_ratio_mean = 0.3316`
  - `runtime_error_rate = 0.116`
  - `timeout_rate = 0.048`
- `step520`
  - `46 / 500`
  - `accepted@1 = 0.092`
  - `pass_ratio_mean = 0.3193`
  - `runtime_error_rate = 0.142`
  - `timeout_rate = 0.080`
- `step580`
  - `56 / 500`
  - `accepted@1 = 0.112`
  - `pass_ratio_mean = 0.3268`
  - `runtime_error_rate = 0.110`
  - `timeout_rate = 0.058`

正确读法：

- `step580` **明显好于 `step520`**
- `step580` 在 exact solve / `accepted@1` 上**小幅超过 `step400`**
- 但 `step400` 的 `pass_ratio_mean` 仍然略高于 `step580`
- 因此 `step580` 是新的 `valid_big` winner，但不是“全面碾压 `step400`”

### 2. solve-set churn 仍然存在，但方向优于 `step520`

相对 `step400`：

- `step580 gained solves = 16`
- `step580 lost solves = 13`

代表性 gain：

- `Codeforces/1170/I`
- `Codeforces/1183/F`
- `Codeforces/1331/D`
- `Codeforces/1354/B`
- `Codeforces/204/E`
- `Codeforces/402/C`
- `Codeforces/799/E`

代表性 loss：

- `Codeforces/1157/A`
- `Codeforces/1175/G`
- `Codeforces/1406/B`
- `Codeforces/371/D`
- `Codeforces/522/D`
- `Codeforces/679/C`
- `Codeforces/847/G`

相对 `step520`：

- `step580 gained solves = 16`
- `step580 lost solves = 6`

所以：

- `step520` 更像一次明显回退的 `b8/mini8` probe
- `step580` 则是一次**真正把一部分 near-miss 推过线**的 `b16 + micro8` probe
- 但它仍然没有完全解决 solve-set churn

### 3. 行为分析：`step580` 的主要收益来自“收口 near-miss”

几个代表题：

- `Codeforces/1183/F`
  - `step400: 0.98 WA`
  - `step520: 0.94 WA`
  - `step580: 1.0 AC`
- `Codeforces/1170/I`
  - `step400: 0.0`
  - `step520: 0.33`
  - `step580: 1.0 AC`
- `Codeforces/402/C`
  - `step400: 0.82`
  - `step520: 0.98`
  - `step580: 1.0 AC`

这说明 `step580` 的提升不是“空洞的 partial 变高”，而是确实把一部分已经进入正确邻域的题收口到了 AC。

但 regression 也同样真实：

- `Codeforces/1157/A`
  - `step400/520: 1.0 AC`
  - `step580: 0.0 WA`
- `Codeforces/679/C`
  - `step400: 1.0 AC`
  - `step520: 0.88 WA`
  - `step580: 0.94 WA`
- `Codeforces/371/D`
  - `step400: 1.0 AC`
  - `step520/580: 0.02 RE`

所以 `step580` 更像是：

- `step400` 之上的**更 aggressive winner**
- exact solve 更强
- 但并没有比 `step400` 更稳

### 4. 当前结论与建议

截至这次 `step580 valid_big`：

- 当前 `valid_big` winner 更新为 **`step580`**
- `step400` 应保留为 **rollback / 对照 checkpoint**
- `step520` 不再是主候选

如果下一步要继续主线：

- 首选继续围绕 `step580` 设计后续 probe / teacher 数据
- 同时保留 `step400` 作为更保守的 fallback
- 不要再把 handoff 中较早的“`step400` 仍是当前 winner”当作最新结论

---

## 补充：`step400` SFT v1 外部复盘（2026-04-06）

这轮基于 `step400` 的 dataset-clean mixed SFT 已经完整做完，并完成了：

- 训练：`20` steps
- shortlist checkpoint：`global_step_10`、`global_step_20`
- cheap-screen：`valid117 + canary_v1`
- 外部拍板：两个 checkpoint 并行跑 `valid_big500`

### 1. 训练与 shortlist

实验名：

- `phase2_repair_sft_step400_r3a2_clean_v1`

保存的两个 checkpoint：

- `/workspace/verl/checkpoints/rlvr_coding_model/phase2_repair_sft_step400_r3a2_clean_v1/global_step_10`
- `/workspace/verl/checkpoints/rlvr_coding_model/phase2_repair_sft_step400_r3a2_clean_v1/global_step_20`

训练日志：

- `/workspace/eval_logs/phase2_repair_sft_step400_r3a2_clean_v1.out`

小验证集 loss：

- `step10 = 0.89951`
- `step20 = 0.89802`

注意：

- tiny val loss 略偏向 `step20`
- 但后续外部评测证明，它**不能**单独决定最终 winner

### 2. Cheap-screen 结果

`valid117`

- `step400`: `6/117`, `pass_ratio_mean = 0.1985`
- `step10`: `8/117`, `pass_ratio_mean = 0.1967`
- `step20`: `7/117`, `pass_ratio_mean = 0.2008`

`canary_v1`

- retention：
  - `step400 = 0.5000 / 0.8007`
  - `step10 = 0.4000 / 0.7987`
  - `step20 = 0.5000 / 0.8007`
- timeout：
  - `step400 = 0.0000 / 0.8377`
  - `step10 = 0.0000 / 0.8377`
  - `step20 = 0.0000 / 0.7044`
- structural：
  - `step400 = 0.0000 / 0.4333`
  - `step10 = 0.0000 / 0.5633`
  - `step20 = 0.0000 / 0.5633`
- mid32：
  - `step400 = 0.2188 / 0.6467`
  - `step10 = 0.3125 / 0.6473`
  - `step20 = 0.2500 / 0.6461`

cheap-screen 当时更偏向：

- `step10` 更亮眼
- `step20` retention 更稳

### 3. `valid_big500` 最终结果

输出目录：

- `step10`：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/phase2_repair_sft_step400_r3a2_clean_v1_step10_codecontests_validbig500`
- `step20`：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/phase2_repair_sft_step400_r3a2_clean_v1_step20_codecontests_validbig500`
- baseline `step400`：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_formal_observe_lb24_multi2_resume240_to400_seed0_step400_codecontests_validbig500_retry3`

总指标：

- `step400`: `53/500`, `accepted@1 = 0.106`, `pass_ratio_mean = 0.3316`
- `step10`: `49/500`, `accepted@1 = 0.098`, `pass_ratio_mean = 0.3305`
- `step20`: `51/500`, `accepted@1 = 0.102`, `pass_ratio_mean = 0.3278`

结论：

- 两个 SFT checkpoint 都更快
- 但两个都没有超过 `step400`
- 如果只在两个 SFT checkpoint 里选，`step20` 略优于 `step10`
- 但正式 winner 仍然是 **pre-SFT `step400`**

### 4. 代表性 case

gain：

- `Codeforces/1151/A`
  - `step400`: `WA 0.38`
  - `step10/20`: `success 1.0`
- `Codeforces/676/B`
  - `step400`: `WA 0.9835`
  - `step10/20`: `success 1.0`
- `Codeforces/204/E`
  - `step400`: `WA 0.98`
  - `step20`: `success 1.0`
- `Codeforces/680/A`
  - `step400`: `WA 0.52`
  - `step20`: `success 1.0`

regression：

- 两个 checkpoint 都掉的 solved case：
  - `Codeforces/115/E`
  - `Codeforces/1154/G`
  - `Codeforces/1253/A`
- `step20` 特别明显的坏点：
  - `Codeforces/1406/B`: `success -> wrong_answer`, `1.0 -> 0.2`
  - `Codeforces/981/G`: `success -> wrong_answer`, `1.0 -> 0.3488`
- timeout / efficiency 坏点：
  - `Codeforces/934/B`: `WA 0.76 -> timeout 0.2`
  - `Codeforces/723/F`: `WA 0.34 -> timeout 0.12`

### 5. 对下一轮 SFT 的结论

当前证据不支持把这轮结果解释成“只是训练步数不够”。

更准确的读法是：

- 这轮 SFT 修掉了一批 high-partial
- 但没有形成对 `step400` 的净收益
- 问题更像 solve-set churn / anti-regression 不足

因此下一轮如果继续做 SFT，建议优先改：

- 更强的 anti-regression / must-keep stabilizer
- 更密的 early checkpoint 保存
- 更强的 external gate

而不是简单把当前这版继续往更多步数推。

完整复盘文档见：

- `phase_2_ GRPO/sft_repair_data/v2b_step400/step400_sft_v1_postmortem.md`

## 补充：2026-04-08 的最新进展：Step 1 one-turn repair 已切到 `reuse-first-pass` 协议，当前 repair base 结论已基本稳定

这轮最重要的变化不是“又多跑了几条 repair eval”，而是**先修正了 repair 评测协议本身**。

### 1. 协议修复：repair 不再重跑 first-pass generation

之前 raw eval 和 repair first-pass 是两次独立 vLLM 生成，因此即便：

- checkpoint 相同
- `temperature=0.0`
- prompt 模板相同

first-pass 结果仍会漂移，导致：

- repair first-pass 不能和历史 raw summary 严格 apples-to-apples
- `step900` / `step1000` 的 repair gain 很容易被 first-pass 差异污染

这轮已经把 `phase4_repair_eval.py` 改成支持：

- 直接读取已有 raw eval 的 `per_problem/*.jsonl`
- 固定复用 raw response / first-pass verifier 结果
- 只新增第二轮 repair generation + verifier

也就是当前的正式 repair 协议已经变成：

1. raw eval 先产生 canonical `per_problem`
2. repair eval 通过 `FIRST_PASS_PER_PROBLEM=...` 复用这些 first-pass 产物
3. repair gain 只衡量“在固定 first-pass 上多加一轮 repair”的净变化

当前 repair 阶段的最新文档与实现参考：

- `phase_2_ GRPO/repair_phase4_design.md`
- `phase_2_ GRPO/phase4_step1_repair_eval_implementation_plan.md`
- `phase_2_ GRPO/repair_analysis/testcase_selection_rule_audit_v1.md`
- `phase_2_ GRPO/repair_analysis/testcase_selection_rule_recommendation_v1.json`

其中 testcase selection 规则当前已固定为：

- `wrong_answer -> simplest_counterexample`
- `runtime_error -> clearest_exception_then_shortest_stdin`
- `timeout -> shortest_timeout_stdin + fixed complexity hint`

### 2. 运行层补充：并行 repair 失败根因已定位，串行 `180` 配置已验证稳定

这轮中间还踩到一个运行层问题：

- 最初并行拉起 4 条 repair run 时，3 条卡在 `START_VLLM`
- 根因是多个 vLLM 同时启动时，内部 `TCPStore` 自动抢同一个端口，报 `EADDRINUSE`
- 只有 `step900 delta69` 那条较早完成，其余 3 条都需要重启

随后已经改成：

- 同一时间只允许 1 条 repair eval 占用 sandbox 池
- 串行跑完剩余 repair jobs
- 并固定 `VLLM_INTERNAL_PORT_BASE`

当前稳定串行链日志：

- `/workspace/eval_logs/run_repair_serial_reuse_180.log`

当前串行配置：

- `MAX_CONCURRENT=180`
- `MAX_CONCURRENT_JUDGES=180`
- `VERIFIER_LIMITER_BUDGET=180`
- `BATCH_SIZE=180`

这套配置已完整跑通：

- `step900 valid_big500`
- `step1000 valid_big500`
- `step1000 delta69`

因此当前可以把：

- `reuse-first-pass`
- 单 run 独占 sandbox 池
- `180/180/180`

视为 Step 1 repair 的稳定执行口径。

### 3. `valid_big500`：修正协议后，`step900` 仍然是更强的 repair base

输出目录：

- `step900`：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_repair/grpo_a1_curriculum_step620_v3q3_to900_seed0_step900_codecontests_validbig500_repair_reuse_firstpass_full`
- `step1000`：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_repair/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_codecontests_validbig500_repair_reuse_firstpass_full`

#### `step900 valid_big500`

- first pass:
  - `54/500`
  - `accepted@1 = 0.108`
  - `pass_ratio_mean = 0.3473`
- after repair:
  - `67/500`
  - `accepted@1 = 0.134`
  - `pass_ratio_mean = 0.3454`

repair 指标：

- `repair_attempt_count = 198`
- `repair_success_count = 13`
- `conditional_repair_success = 6.57%`
- `repair_gain = +2.6 pct`
- `net solve gain = +13`

clean 口径：

- first-pass clean:
  - `54/498`
  - `0.1084`
  - `pass_ratio_mean = 0.3473`
- after-repair clean:
  - `66/498`
  - `0.1325`
  - `pass_ratio_mean = 0.3448`

#### `step1000 valid_big500`

- first pass:
  - `53/500`
  - `accepted@1 = 0.106`
  - `pass_ratio_mean = 0.3442`
- after repair:
  - `65/500`
  - `accepted@1 = 0.130`
  - `pass_ratio_mean = 0.3401`

repair 指标：

- `repair_attempt_count = 193`
- `repair_success_count = 12`
- `conditional_repair_success = 6.22%`
- `repair_gain = +2.4 pct`
- `net solve gain = +12`

clean 口径：

- first-pass clean:
  - `53/498`
  - `0.1064`
  - `pass_ratio_mean = 0.3445`
- after-repair clean:
  - `65/498`
  - `0.1305`
  - `pass_ratio_mean = 0.3404`

#### 对 `valid_big500` 的结论

修正协议后，之前“`step1000` 似乎更 repairable”的印象已经明显减弱。

当前更可靠的读法是：

- `step900` first pass 本身就略优于 `step1000`
- repair 后 `step900` 仍然略优于 `step1000`
- 两者的 conditional repair success 已经很接近
- 因此 `step900` 仍是当前更好的 `valid_big500` repair base

### 4. `delta69`：`step900` 对高 through-rate near-miss 的 repair 更明显

输出目录：

- `step900`：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_delta69_repair/grpo_a1_curriculum_step620_v3q3_to900_seed0_step900_delta69_repair_reuse_firstpass_full`
- `step1000`：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_delta69_repair/grpo_a1_curriculum_step900_abfocus_lb150_s35_to1000_seed0_step1000_delta69_repair_reuse_firstpass_full`

#### `step900 delta69`

- first pass:
  - `48/69`
  - `accepted@1 = 0.6957`
  - `pass_ratio_mean = 0.8549`
- after repair:
  - `52/69`
  - `accepted@1 = 0.7536`
  - `pass_ratio_mean = 0.8672`

repair 指标：

- `repair_attempt_count = 16`
- `repair_success_count = 4`
- `conditional_repair_success = 25.0%`
- `repair_gain = +5.80 pct`

#### `step1000 delta69`

- first pass:
  - `48/69`
  - `accepted@1 = 0.6957`
  - `pass_ratio_mean = 0.8605`
- after repair:
  - `49/69`
  - `accepted@1 = 0.7101`
  - `pass_ratio_mean = 0.8593`

repair 指标：

- `repair_attempt_count = 16`
- `repair_success_count = 1`
- `conditional_repair_success = 6.25%`
- `repair_gain = +1.45 pct`

#### 对 `delta69` 的结论

`delta69` 上的结论比 `valid_big500` 更鲜明：

- `step900` 明显比 `step1000` 更会修
- `step900` 不只是 solve 增益更大，`pass_ratio_mean` 也同步提高
- `step1000` 则更像只救回极少数高 partial 样本

所以如果把 `delta69` 视为当前 repair-ready slice，`step900` 是明确更好的底座。

### 5. repair 行为模式：当前更像 high-precision fallback，而不是主增益引擎

这轮 4 条 run 有一个一致特征：

- solve 层面没有出现 `AC -> fail`
- 但 unsolved 区域里仍有明显的 `pass_ratio` 上下波动

例如：

- `step900 valid_big500`：
  - `gained solves = 13`
  - `lost solves = 0`
  - `pass_ratio up = 44`
  - `pass_ratio down = 32`
- `step1000 valid_big500`：
  - `gained solves = 12`
  - `lost solves = 0`
  - `pass_ratio up = 38`
  - `pass_ratio down = 31`

这说明当前 one-turn repair 的行为更像：

- 对最终 solved 数相对安全
- 但并不保证整体 partial correctness 变得更稳

换句话说，它更像：

- **high-precision fallback**
- 适合从 high-partial / near-miss 里捞 solve
- 还不适合作为“整体提升 pass_ratio_mean”的主手段

### 6. 当前最重要的 bucket 结论仍然成立，而且变得更清楚

`valid_big500` 上：

- `step900`
  - `bucket_0.6_1.0`: `11 / 81 = 13.58%`
  - `bucket_0.2_0.6`: `2 / 117 = 1.71%`
- `step1000`
  - `bucket_0.6_1.0`: `12 / 79 = 15.19%`
  - `bucket_0.2_0.6`: `0 / 114 = 0%`

`delta69` 上：

- `step900`
  - `bucket_0.6_1.0`: `3 / 11 = 27.27%`
  - `bucket_0.2_0.6`: `1 / 5 = 20.0%`
- `step1000`
  - `bucket_0.6_1.0`: `1 / 11 = 9.09%`
  - `bucket_0.2_0.6`: `0 / 5 = 0%`

因此当前很明确：

- Step 1 repair 的甜点区仍是 `bucket_0.6_1.0`
- `bucket_0.2_0.6` 在 `step900` 上还有少量价值，但性价比明显低于高 partial
- `timeout` 当前没有体现出明确收益，不应继续作为默认重点 repair trigger

### 7. 对当前 repair 主线的最终结论

截至这轮 `reuse-first-pass` 正式评测：

1. `phase4` Step 1 repair 评测协议已经可信
   - 以后必须优先复用 canonical raw `per_problem`
   - 不要再把“独立重跑 first-pass”的结果当作正式 repair 增益依据

2. 当前 repair base 结论已经基本稳定
   - 这条 2026-04-09 结论只适用于当时的 pre-SFT frontier 比较
   - 后续在 patched sandbox + direct-backend RR 下，已被 `step40/50/60/30` 的 Protocol A / B 结果覆盖

3. one-turn repair 值得保留，但定位应收窄
   - 它确实能稳定捞回一批 high-partial near-miss
   - 但它不是当前主线能力提升的核心引擎
   - 更合理的定位是：**targeted fallback / repair prior diagnosis tool**

### 8. 建议的下一步

当前更推荐的后续动作：

1. 把 `step40` 作为当前 Step 1 repair 主 base
2. `step50` 作为更偏 repaired partial-quality 的次优备选
3. 后续 repair eval 默认继续区分：
   - Protocol A：`FIRST_PASS_PER_PROBLEM=...`
   - Protocol B：各 checkpoint 自己 raw first-pass + self-repair
4. 下一轮如果追求效率，可优先试：
   - `repair_min_pass_ratio = 0.6`
5. `timeout` 可从默认 repair trigger 中降级
6. 如果需要绝对同环境口径，可以补一条：
   - 用当前 patched sandbox + direct-backend RR，把 `step40 delta69` 补成与 `valid_big500` 同口径的 follow-up

但即使不补这条复跑，当前大方向结论也已经足够清楚：

- `step40` 是当前更优的 one-turn repair-ready checkpoint
- `step50` 是当前更偏 partial-quality 的 repair 备选
- repair 有价值，但应该被当作受控使用的窄工具，而不是新的主训练引擎

## 补充：2026-04-09 的最新进展：`step900 valid_big500` prompt ablation 已完成，旧 `code_only` 结果确认受 sandbox 污染

### 1. 这轮 prompt ablation 的背景

在把 `repair_prompt_design.md` 里的两组 prompt 真正接进代码后，基于：

- `reuse-first-pass`
- `step900`
- `valid_big500`
- `repair_min_pass_ratio = 0.6`
- `repair_error_types = {wrong_answer, runtime_error}`

做了一轮小规模 prompt ablation，比较：

- `code_only`
- `short_diagnosis_code`

其中：

- `code_only` 初次运行目录：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_repair_ablation/grpo_a1_step900_validbig500_repair_reuse_ablate_code_only_p06_wa_re`
- `short_diagnosis_code` 目录：
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_repair_ablation/grpo_a1_step900_validbig500_repair_reuse_ablate_short_diag_p06_wa_re`

### 2. 旧 `code_only` run 已确认污染，应作废

在 `vastai2` 的 sandbox 池异常后，对这轮 ablation 做了日志与逐题核对，确认：

- 旧 `code_only` run 出现了明确的 sandbox/API 污染
- `summary.json` 中：
  - `api_error_rate = 0.036`
- `per_problem/codecontests_valid_big.jsonl` 中：
  - 有 `18` 条 `error_type = "api_error"`
- eval log 中有大量：
  - `Connection refused`
  - 少量 `502`

因此：

- 旧 `code_only` 结果**不能**作为正式 prompt ablation 基线
- `short_diagnosis_code` 那条 run 没有对应污染：
  - `api_error_rate = 0.0`
  - `sandbox_error_rate = 0.0`
  - 可继续保留

### 3. clean `code_only` rerun 已完成，正式对照结果如下

在恢复 `8090` sandbox LB 和 8 个 backend 后，只重跑了 `code_only`。

clean rerun 输出当前在：

- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_repair/codecontests_validbig500_repair_eval`

这次与 `short_diagnosis_code` 对照时，两者共享：

- 完全相同的 `FIRST_PASS_PER_PROBLEM`
- 完全相同的 checkpoint
- 完全相同的 trigger / error type slice
- 完全相同的并发设置

因此这次是严格可比的 prompt ablation。

#### fixed first-pass（两组完全一致）

- `54 / 500`
- `accepted@1 = 0.108`
- `pass_ratio_mean = 0.3473`

#### clean `code_only`

- after repair:
  - `63 / 500`
  - `accepted@1 = 0.126`
  - `pass_ratio_mean = 0.3425`
- repair 指标：
  - `repair_attempt_count = 73`
  - `repair_success_count = 9`
  - `conditional_repair_success = 12.33%`
  - `api_error_rate = 0.0`
  - `sandbox_error_rate = 0.0`

#### clean `short_diagnosis_code`

- after repair:
  - `66 / 500`
  - `accepted@1 = 0.132`
  - `pass_ratio_mean = 0.3336`
- repair 指标：
  - `repair_attempt_count = 73`
  - `repair_success_count = 12`
  - `conditional_repair_success = 16.44%`
  - `api_error_rate = 0.0`
  - `sandbox_error_rate = 0.0`

### 4. 这轮 prompt ablation 的结论

当前可以正式下结论：

1. `short_diagnosis_code` 在这条受控切片上有正信号
   - 相比 clean `code_only`：
     - 多 `3` 个 solve
     - 多 `3` 次 repair success
     - `conditional_repair_success` 更高

2. 这条信号是可信的
   - 因为 first-pass 已固定复用 canonical raw `per_problem`
   - 两组 prompt 的比较不再被 first-pass 漂移污染
   - 旧 `code_only` 污染 run 已被识别并剔除

3. 这仍然只是“小规模 prompt ablation 正信号”
   - 还不能直接推出“所有 repair / 所有 checkpoint / 所有 bucket 都应该默认带 diagnosis”
   - 但已经足够支持：
     - 后续 Step 3 数据收集优先保留 `short_diagnosis_code` 这种短诊断格式

### 5. 当前对后续 repair prompt 主线的建议

截至 2026-04-09，建议更新为：

1. Step 1 / Step 3 相关 prompt 设计默认优先：
   - `short_diagnosis_code`
2. `code_only` 不删除
   - 继续作为对照组 / simpler baseline 保留
3. 后续 prompt 对比仍然必须继续满足：
   - `reuse-first-pass`
   - canonical raw `per_problem`
   - 同一 checkpoint / 同一 slice / 同一并发口径

这意味着当前 prompt 方向已经从：

- “是否要做 `<think>`”

进一步收敛成：

- “是否要保留一个**极短、结构化、可控**的 diagnosis scaffold”

而这轮结果给出的答案是：

- **值得保留，并应优先进入下一阶段 repair-conditioned data design**

## 补充：2026-04-09 的最新进展：`step1300` 不是 exact-solve winner，但 fixed-response rejudge 支持它是当前最强的 partial-credit checkpoint

这轮对 `step1100/1200/1300` 又补做了一次 fixed-response rejudge，目的是把：

- 生成漂移
- judge 漂移

拆开看清楚。

报告与输出：

- `/tmp/fixed_response_rejudge_report.json`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1100_rejudge1`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1100_rejudge2`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1200_rejudge1`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1200_rejudge2`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1300_rejudge1`
- `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_fixed_rejudge/step1300_rejudge2`

这轮的核心结论需要把之前的说法收紧成两层：

1. `step1000` 仍然是 `valid_big500` 的 exact-solve winner
   - 当前 patched full eval 还是：
     - `step1000 = 69 / 500`
     - `step900 = 66 / 500`
     - `step1100 = 63 / 500`
     - `step1200 = 62 / 500`
     - `step1300 = 59 / 500`
   - judge-only 漂移只有 `2~6` 题量级，不足以抹掉 `step1000` 相对 `step1300` 的 solve gap

2. `step1300` 不该再简单归类为“不是 winner”
   - 更准确的说法是：
     - **它不是 exact-solve winner**
     - **但它是当前最强的 partial-credit / pass_ratio_mean winner**

fixed-response rejudge 的关键证据是：

- `step1300` source first run:
  - `pass_ratio_mean = 0.357553`
- `step1300 rejudge1`:
  - `accepted@1 = 0.122`
  - `pass_ratio_mean = 0.357475`
- `step1300 rejudge2`:
  - `accepted@1 = 0.116`
  - `pass_ratio_mean = 0.357354`

对比：

- `step1100` rejudge:
  - `pass_ratio_mean = 0.348049 / 0.347537`
- `step1200` rejudge:
  - `pass_ratio_mean = 0.339792 / 0.339607`

这说明：

- patched sandbox 修掉了之前的大规模输出截断问题
- judge 漂移仍然存在，但明显小于 full rerun 的生成漂移
- `step1300` 的高 `pass_ratio_mean` 不是一次偶然 judge artifact，而是在 fixed-response 下也基本稳定

因此当前对 late checkpoint 的推荐读法更新为：

- `step1000`：当前 `valid_big500` 的主 winner，用于 exact-solve 结论
- `step1300`：当前最强的 partial-credit checkpoint，更适合：
  - near-miss / repair case harvesting
  - repair-SFT / diagnosis-conditioned data collection
  - 分析 continuation 是否在“partial correctness frontier”上继续前进
- `step1200`：仍然是这段 continuation 里的低点

这也意味着，后续对 `1100/1200/1300` 的汇报不应再只报单次 full run，而应该同时报：

- full-run solved / accepted@1
- fixed-response rejudge `pass_ratio_mean`

同时，对 repair 方向的实验建议也应随之更新：

- `step900` 继续保留为当前**已验证过**的 one-turn repair 主 base
- `step1300` 现在值得补一轮：
  - `valid_big500`
  - `reuse-first-pass`
  - `code-only` vs `short_diagnosis_code`

这轮实验的目的不是重选 exact-solve winner，而是验证：

- 当前最强 partial-credit checkpoint 是否也表现出更高的 repairability
- `short_diagnosis_code` 的正信号是否能迁移到 `step1300` 这类 near-miss 更重的 checkpoint

## 补充：2026-04-09 的最新进展：`step1300` 的 fixed-first-pass repair probe 已完成，但没有取代 `step900` 成为更好的 repair base

这轮 `step1300` prompt ablation 已经按受控协议跑完，口径是：

- checkpoint:
  - `step1300`
- dataset:
  - `valid_big500`
- protocol:
  - `reuse-first-pass`
- source first-pass:
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3_step1300_codecontests_validbig500_raw_patched_rr828x_vastai3/per_problem`
- sandbox:
  - patched sandbox + client RR
- trigger:
  - `wrong_answer + runtime_error`
  - `pass_ratio >= 0.6`

输出目录：

- `step1300 code_only`
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_repair_ablation/grpo_a1_step1300_validbig500_repair_reuse_ablate_code_only_p06_wa_re`
- `step1300 short_diagnosis_code`
  - `/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500_repair_ablation/grpo_a1_step1300_validbig500_repair_reuse_ablate_short_diag_p06_wa_re`

结果如下：

### `step1300 code_only`

- first pass:
  - `61 / 500`
  - `accepted@1 = 0.122`
  - `pass_ratio_mean = 0.357553`
- after repair:
  - `65 / 500`
  - `accepted@1 = 0.130`
  - `pass_ratio_mean = 0.348504`
- repair summary:
  - `repair_attempt_count = 66`
  - `repair_success_count = 4`
  - `conditional_repair_success = 6.06%`

### `step1300 short_diagnosis_code`

- first pass:
  - `61 / 500`
  - `accepted@1 = 0.122`
  - `pass_ratio_mean = 0.357553`
- after repair:
  - `65 / 500`
  - `accepted@1 = 0.130`
  - `pass_ratio_mean = 0.336978`
- repair summary:
  - `repair_attempt_count = 66`
  - `repair_success_count = 4`
  - `conditional_repair_success = 6.06%`

与当前主线 `step900` 的同切片受控结果对比如下：

- `step900 code_only`
  - `54 -> 64 / 500`
  - `conditional_repair_success = 13.70%`
- `step900 short_diagnosis_code`
  - `54 -> 66 / 500`
  - `conditional_repair_success = 16.44%`

因此这轮实验给出的答案已经比较清楚：

1. `step1300` 的 raw first-pass 仍然更强
   - 它继续支持“当前最强 partial-credit / near-miss checkpoint”这一定位

2. 但 `step1300` 的 repairability 并不更强
   - 在同样的 `WA/RE + pass_ratio >= 0.6 + reuse-first-pass` 切片上
   - 它只救回 `4` 题
   - 明显弱于 `step900` 的 `10~12` 题 rescue

3. `short_diagnosis_code` 在 `step1300` 上没有复现 `step900` 的明显优势
   - solve 数与 `code_only` 持平
   - `pass_ratio_mean` 还更低
   - 因此“短诊断格式有正信号”目前仍应理解成：
     - 对 `step900` 这类已验证 repair-ready checkpoint 成立
     - 不应外推成对所有 high-partial checkpoint 都成立

当前对 repair base 的正式结论应再收紧成：

- 这条 `step900 vs step1300` 结论现在只保留为 **pre-SFT frontier 历史结论**
- 当前最新 repair base 结论已经更新为：
  - `step40`：当前更好的 one-turn repair 主 base
  - `step50`：当前更偏 repaired partial-quality 的次优备选
  - `step900`：当前 Protocol A 的 frozen first-pass source / 历史 validated base
  - `step1300`：当前更强的 partial-credit / near-miss probe checkpoint，但不再当主 repair base

额外注意：

- `step1300 code_only` 的 final `per_problem/codecontests_valid_big.jsonl` 有 `4` 条坏行
- `summary.json` 与 `repair_summary.json` 可用
- 但这条 run 不适合直接做细粒度 badcase 资产沉淀

## 补充：2026-04-09 的当前正式评测口径约定：默认使用 patched sandbox + client RR（不走 nginx LB）

上面已经补充了：

- full rerun 里同时存在：
  - 生成漂移
  - judge 漂移
- fixed-response rejudge 把这两层拆开之后，可以确认：
  - judge 漂移仍然存在
  - 但量级明显小于 full rerun 的生成漂移

这一节把**当前正式评测应该怎么跑**单独定成明确约定，避免后续 agent 又误回到旧的 LB 口径。

### 1. 当前 active 机器与路径口径

当前 active 机器已经切到：

- `vastai3`

当前真正使用的 repo 根目录是：

- `/workspace/verl`

因此，本文档前面大量出现的：

- `/root/verl`
- `/root/sandboxfusion-run`

要视为**历史 bring-up 口径**。在 `vastai3` 上继续实验时，应优先以：

- `/workspace/verl`
- `/root/sandboxfusion-multi`
- `/workspace/eval_logs`

这套现场路径为准。

### 2. “patched sandbox” 的含义

当前所谓 patched sandbox，指的是远端实际运行的 SandboxFusion server 已经包含了这两处修复：

- `SandboxFusion/sandbox/runners/base.py`
- `SandboxFusion/sandbox/utils/execution.py`

修复目标是：

1. 不再使用旧版极短超时的输出读取方式
2. 不再在输出读取前把进程树清掉
3. 用 request-scoped process group / guarded drain 避免：
   - 空 stdout
   - 截断 stdout
   - kill-before-drain 竞态

这解决的是之前最严重的那类：

- 同一 response
- 同一 testcase
- 有时 `success`
- 有时 `wrong_answer`
- 且 `actual` 经常是空串或截断

### 3. “client RR” 的准确含义

当前正式评测默认**不是**：

- `http://localhost:8090`
- `http://localhost:8091`
- `http://localhost:8092`

这类 nginx LB 单入口。

当前正式评测默认应该传的是**逗号分隔的 backend URL 列表**，例如：

- `http://localhost:8081,...,http://localhost:8088`
- `http://localhost:8181,...,http://localhost:8188`
- `http://localhost:8281,...,http://localhost:8288`

这条链在 verifier 里会走：

- `coding_model_project/src/verifier/shared.py`
- `_normalize_sandbox_endpoints()`
- `_choose_sandbox_endpoint_rr()`

也就是说：

- 上层仍然只传一个 `sandbox_url: str`
- 但只要里面是逗号分隔多 URL，verifier 就会在**客户端本地做 round-robin 选 backend**

因此当前约定里的：

- `patched sandbox + client RR`

准确含义就是：

- patched 的 `SandboxFusion server`
- + `shared.py` 里的本地 RR 分发
- + 直接打 backend URL
- **而不是** nginx multi-upstream LB

### 4. 这条约定已经在 `1100/1200/1300` 的 live eval 中被实际使用

以下三条 `valid_big500` 运行的 `run_info.json` 已确认使用的是多 URL 直连：

- [step1100 run_info.json](/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step1000_overnight_u5a4b4c3_to1200_lb240_seed0_vastai3_step1100_codecontests_validbig500_raw_patched_rr808x_vastai3/run_info.json)
- [step1200 run_info.json](/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step1000_overnight_u5a4b4c3_to1200_lb240_seed0_vastai3_step1200_codecontests_validbig500_raw_patched_rr818x_vastai3/run_info.json)
- [step1300 run_info.json](/workspace/verl/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3_step1300_codecontests_validbig500_raw_patched_rr828x_vastai3/run_info.json)

三者的 `sandbox_url` 都是：

- `8081..8088`
- `8181..8188`
- `8281..8288`

而不是：

- `8090`
- `8091`
- `8092`

所以这批 `1100/1200/1300` 的评测与 fixed-response rejudge，本身就已经是在当前推荐口径下完成的。

### 5. 为什么不能再把 `step900` 的 `0 / 4 / 1` 当成“全局保证”

之前在 `step900 patched baseline` 上观察到的：

- `accepted_mismatch_count = 0`
- `pass_ratio_mismatch_count = 4`
- `error_type_mismatch_count = 1`

是一个**非常好的信号**，说明 patched sandbox 确实把旧的大规模并发输出截断问题压掉了。

但这组数字只能理解成：

- `step900`
- 在那一批固定 response 上
- 在那次 fixed-response probe 下

的一个低噪声 floor。

它**不能**直接外推成：

- “从今以后任何 checkpoint、任何 response 集合、任何 full eval rerun 都只会有 `0/4/1` 的 judge 漂移”

这次 `1100/1200/1300` 的 fixed-response rejudge 已经证明：

- 仍会出现 `accepted mismatch = 2~6`
- `pass_ratio mismatch` 更高，常在 `11~17`

也就是说：

- 当前 patched sandbox + client RR 已经明显优于旧链路
- 但 judge 仍然**不是严格 deterministic**
- judge-only floor 会随 response 集合变化，不是单个常数

### 6. 当前正式比较 checkpoint 的推荐口径

后续如果要做 checkpoint 排序或对外汇报，建议默认遵守：

1. full eval 结果用于看：
   - solve / accepted@1
   - 生成侧整体表现

2. fixed-response rejudge 用于看：
   - judge-only 漂移
   - partial-credit / `pass_ratio_mean`
   - 某个 checkpoint 的高分是否只是判题偶然

3. 若两个 checkpoint 的 solved 差距很小：
   - 尤其是 `<= 2~4` 题
   - 不要只凭单次 full rerun 直接拍板

4. 若目的是分析 repair / near-miss：
   - 优先看 fixed canonical first-pass
   - 尽量避免把 first-pass 生成漂移和 repair 增益混在一起

### 7. 当前正式运行 checklist

后续 agent 接手跑正式评测时，至少确认下面几项：

1. `run_info.json` 里的 `sandbox_url` 是：
   - 逗号分隔 backend URL 列表
   - 不是 `8090/8091/8092`

2. `run_info.json` / launch log 里的并发口径明确记录：
   - `max_concurrent`
   - `max_concurrent_judges`
   - `verifier_limiter_budget`

3. rerun 必须写入**新 output_dir**
   - 不能覆盖 first run

4. 输出里确认：
   - `api_error_rate = 0.0`
   - `sandbox_error_rate = 0.0`

5. 如果要研究“到底是 judge 漂还是生成漂”：
   - 必须额外做 fixed-response rejudge
   - 不要只比较两次 full rerun
