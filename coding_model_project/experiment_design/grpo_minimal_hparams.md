# RLVR Coding Model - GRPO 关键超参最小可复现集

目标：把“能复现实验结果”从口号变成清单；面试时也能快速回答“你到底把哪些超参固定了？”。

---

## 1) 必须写死的 run 元信息（没有这些就不可复现）

- `base_model`：例如 `Qwen2.5-Coder-7B-Instruct`
- `init_checkpoint`：SFT（或 DPO）初始化 checkpoint 路径/ID
- `ref_model`：reference model 来源（通常为 `init_checkpoint` 的 frozen copy）
- `dataset_manifest`：训练/验证/测试所用 manifest 文件名与版本（见 `data_governance.md`）
- `code_version`：git commit/hash（或 release tag）
- `seed`：至少 2 个 seeds；记录 python/numpy/torch 与 dataloader seed

---

## 2) GRPO/PPO 关键超参（最小集）

> 建议把下表按 `dense` / `sparse` 两套配置分别落盘，并在 WandB 的 config 里原样保存。

### 2.1 Rollout / Sampling（决定“采样分布”和训练成本）

| 字段 | 说明 |
|------|------|
| `rollout_n` | 每个 prompt 采样条数（GRPO 的 group size 核心来源） |
| `temperature` / `top_p` | 采样策略（必须固定） |
| `max_new_tokens` | 每条样本最大生成长度（强烈建议固定） |
| `stop_sequences`（如有） | 防止输出多余文本/重复 |

### 2.2 Reward（决定学习信号）

| 字段 | 说明 |
|------|------|
| `reward_type` | `dense=pass_ratio` 或 `sparse=1[accepted]` |
| `reward_norm` | 是否做归一化（均值/方差/分位数），以及在哪个粒度做（per-batch/per-group） |
| `reward_clip` | 是否 clip，以及阈值（防极端 outlier） |

### 2.3 Advantage / GRPO 细节（决定方差与稳定性）

| 字段 | 说明 |
|------|------|
| `adv_estimator` | `grpo` |
| `group_size` | group 内样本数（通常应与 `rollout_n` 一致；如果不一致必须解释） |
| `norm_adv_by_std_in_grpo` | 是否按 std 归一化 advantage（你文档中已有） |

### 2.4 KL / Reference（决定“离谱输出”与训练发散风险）

| 字段 | 说明 |
|------|------|
| `use_kl_loss` | 是否启用 KL 约束 |
| `kl_loss_coef` 或 `kl_target` | 二选一：固定系数或 target-KL（必须写死） |
| `kl_clip`（如有） | 防止 KL 爆炸的裁剪 |

### 2.5 Optimizer / PPO 更新（决定收敛速度）

| 字段 | 说明 |
|------|------|
| `lr` | actor 学习率 |
| `clip_ratio` | PPO clip ratio |
| `train_batch_size` | 每次更新的样本量（以 sequences 计） |
| `mini_batch_size` | PPO mini-batch |
| `ppo_epochs` | 每批数据做几轮优化 |
| `grad_accum_steps` | 梯度累积 |
| `max_grad_norm` | 梯度裁剪（强烈建议固定） |

---

## 3) 必须在报告里回答的 3 个复现性问题

1. `rollout_n/group_size` 变化会不会导致结果改变？（至少做一次敏感性 sanity-check 或解释为什么不做）
2. ref model 是谁、是否冻结、KL 口径是什么？
3. `EVAL@1`/`EVAL@k`/`EVAL@budget` 三种评测口径下结论是否一致？（见 `eval_protocol.md`）

