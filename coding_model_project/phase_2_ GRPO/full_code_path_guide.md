# 完整代码链路指南：RL 训练 + Eval 评估

本文档从两个 shell 脚本入口出发，逐层追踪每一个函数调用、数据流向、关键代码位置。

---

# Part A：RL 训练完整链路

## A1. 入口：`run_grpo_smoke.sh`

**文件**: `coding_model_project/phase_2_ GRPO/run_grpo_smoke.sh`

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \                    # 选择 GRPO 优势估计器
    algorithm.norm_adv_by_std_in_grpo=True \           # 用组内标准差归一化优势
    algorithm.use_kl_in_reward=False \                 # 不在 reward 中加 KL 惩罚
    data.train_files=$DATA_DIR/smoke_train.parquet \   # 训练数据
    data.val_files=$DATA_DIR/smoke_val.parquet \        # 验证数据
    data.train_batch_size=16 \                          # 每步取 16 个 prompt
    actor_rollout_ref.rollout.n=4 \                    # 每个 prompt 生成 4 个回复（GRPO 核心）
    actor_rollout_ref.actor.strategy=fsdp2 \           # 用 FSDP2 做分布式
    actor_rollout_ref.actor.clip_ratio_low=0.2 \       # PPO clip 下界
    actor_rollout_ref.actor.clip_ratio_high=0.28 \     # PPO clip 上界（DAPO 风格非对称）
    actor_rollout_ref.actor.use_kl_loss=False \        # actor 损失中不加 KL（DAPO 风格）
    actor_rollout_ref.rollout.name=vllm \              # 用 vLLM 做推理
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    reward_manager.source=register \                   # 从 verl 注册表加载 reward manager
    reward_manager.name=batch \                        # 用 BatchRewardManager
    reward_model.launch_reward_fn_async=False \        # 同步计算 reward
    custom_reward_function.path=.../grpo_batch_reward.py \  # 自定义 reward 函数路径
    custom_reward_function.name=compute_score \             # 函数名
    custom_reward_function.reward_kwargs.sandbox_endpoint=$SANDBOX_URL \
    custom_reward_function.reward_kwargs.reward_mode=$REWARD_MODE \
    custom_reward_function.reward_kwargs.limiter_budget=$LIMITER_BUDGET \
    trainer.n_gpus_per_node=4 \                        # 4 卡
    trainer.total_epochs=1 \
    trainer.test_freq=1 \                              # 每步都验证
    trainer.val_before_train=True                      # 训练前先验证一次
```

**配置如何传递**：Hydra 把所有 `key=value` 解析成 `OmegaConf DictConfig`，传给 `main_ppo.main(config)`。

---

## A2. Python 入口：`main_ppo.py`

**文件**: `verl/trainer/main_ppo.py`

```
main(config)                          # Line 36
  └── run_ppo(config)                 # Line 45
        ├── ray.init(...)             # Line 77 — 初始化 Ray 分布式运行时
        └── ray.get(runner.run.remote(config))  # Line 99
              └── TaskRunner.run()    # Line 256
```

### `TaskRunner.run()` 做了什么（Line 256-362）

```
1. 创建 Worker：
   actor_rollout_cls = add_actor_rollout_worker(config)   # Line 277
   # → 根据 strategy=fsdp2 选择 ActorRolloutRefWorker

2. 加载 Tokenizer：
   tokenizer = hf_tokenizer(model_path)                   # Line 308

3. 加载 Reward 函数：
   reward_fn = load_reward_manager(config, tokenizer)      # Line 313
   val_reward_fn = load_reward_manager(config, tokenizer)  # Line 317
   # → 这两行决定了 RL 训练用什么 reward（详见 A4）

4. 创建数据集：
   train_dataset = create_rl_dataset(config, "train")      # Line 325
   val_dataset = create_rl_dataset(config, "val")           # Line 330
   # → 从 Parquet 读取 prompt + ground_truth

5. 创建 Trainer：
   trainer = RayPPOTrainer(
       config, tokenizer, reward_fn, val_reward_fn,
       train_dataset, val_dataset, ...
   )                                                         # Line 344

6. 启动训练：
   trainer.init_workers()                                    # Line 358
   trainer.fit()                                             # Line 361
```

**要修改什么**：
- 换模型 → 改 `actor_rollout_ref.model.path`
- 换数据 → 改 `data.train_files` / `data.val_files`
- 换 reward 函数 → 改 `custom_reward_function.path` 和 `.name`

---

## A3. Reward 函数加载

**文件**: `verl/trainer/ppo/reward.py`

### `load_reward_manager()` (Line 99-176)

```python
def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):

    # 步骤 1：加载自定义 compute_score 函数
    compute_score = get_custom_reward_fn(config)
    # → 从 custom_reward_function.path 动态 import compute_score 函数
    # → 用 partial() 把 reward_kwargs 预绑定进去
    #   预绑定的参数 = {sandbox_endpoint, reward_mode, limiter_budget, run_timeout_s, memory_limit_mb}

    # 步骤 2：选择 RewardManager 类
    reward_manager_cls = get_reward_manager_cls("batch")
    # → 从注册表拿到 BatchRewardManager

    # 步骤 3：实例化
    return BatchRewardManager(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=预绑定后的函数,    # ← 关键：自定义函数注入到这里
        reward_fn_key="data_source",
    )
```

### `get_custom_reward_fn()` (Line 60-96)

```python
def get_custom_reward_fn(config):
    raw_fn = load_extern_object(
        module_path="coding_model_project/src/grpo_batch_reward.py",
        object_name="compute_score"
    )
    reward_kwargs = {
        "sandbox_endpoint": "http://localhost:8080",
        "reward_mode": "anchored_dense_v1",
        "limiter_budget": 8,
        ...
    }
    return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    # 返回一个函数，调用时会自动合并 reward_kwargs
```

### `BatchRewardManager` 运行时行为

**文件**: `verl/workers/reward_manager/batch.py`

```python
class BatchRewardManager(AbstractRewardManager):

    def __call__(self, data: DataProto, return_dict=False):
        # 1. 检查是否已有预计算的 rm_scores
        # 2. 调用 self.verify(data) 获取评分
        scores = self.verify(data)   # Line 94
        # 3. 把 score 写到 reward_tensor 的最后一个有效 token 位置
        for i in range(len(data)):
            reward_tensor[i, valid_length - 1] = score
        # 4. 收集 reward_extra_info
        return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}

    def verify(self, data):                               # Line 47

```

当前正式主链多了一层 trainer-side merge：

- rollout / agent loop 把 `finish_reason`、`truncated_by_max_tokens` 作为独立 `non_tensor_batch` 字段返回
- `ray_trainer.py::_compute_or_extract_reward()` 在真正调用 reward_fn 之前，为每个 sample 复制一份 `extra_info`
- 然后把这两个字段写入副本，再回填到 `batch.non_tensor_batch["extra_info"]`
- 这样既能让 custom reward 读到截断信息，也不会因为 `DataProto.union()` 的对象相等约束而冲突
        # 1. 解码所有 response token → 文本
        responses_str = [tokenizer.decode(ids) for ids in response_ids]
        # 2. 从 batch 中提取 ground_truth
        ground_truths = [item["reward_model"]["ground_truth"] for item in data]
        # 3. 调用自定义 compute_score（预绑定了 sandbox_endpoint 等参数）
        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
        )
        # scores 的实际调用链：
        #   → grpo_batch_reward.compute_score()
        #     → normalize_candidate() + verify_candidate_batch()
        #       → SandboxFusion 判题
        return scores  # List[dict]，每个 dict 有 "score" 和 verifier 字段
```

**要修改什么**：
- 换 reward 公式 → 改 `grpo_batch_reward.py` 的 `_map_reward()`
- 换判题逻辑 → 改 `verifier/shared.py`
- 换 sandbox → 改 `sandbox_endpoint` 参数

---

## A4. 训练主循环：`ray_trainer.py` 的 `fit()` 方法

**文件**: `verl/trainer/ppo/ray_trainer.py`, Line 1319-1966

这是整个训练的核心。每一步（step）包含 13 个阶段：

### 阶段 1：数据准备 (Line 1430-1459)

```python
# 从 dataloader 取一个 batch（16 个 prompt）
batch = DataProto.from_single_dict(batch_dict)

# 给每个 prompt 分配唯一 ID（GRPO 分组用）
batch.non_tensor_batch["uid"] = [uuid4(), uuid4(), ...]  # 16 个

# 把每个 prompt 复制 n=4 次（生成 4 个回复用）
gen_batch = batch.repeat(repeat_times=4, interleave=True)
# interleave=True → [p1, p1, p1, p1, p2, p2, p2, p2, ...]
# 现在 gen_batch 有 16×4 = 64 条
```

**理解 `interleave=True`**：
```
原始 batch: [prompt_A, prompt_B, prompt_C, ...]
repeat(n=4, interleave=True):
  → [A, A, A, A, B, B, B, B, C, C, C, C, ...]
  同一个 prompt 的 4 份相邻排列，uid 相同
  GRPO 后面按 uid 分组就能把同一 prompt 的回复放一起
```

### 阶段 2：Rollout 生成 (Line 1467-1486)

```python
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
# → 调用 ActorRolloutRefWorker.generate_sequences()
# → 内部调用 vLLM 做 token 生成
# → 返回: responses (token IDs), rollout_log_probs, attention_mask
```

**ActorRolloutRefWorker 内部**（`verl/workers/fsdp_workers.py`）：
```
generate_sequences(prompts):
  1. await rollout_mode()
     → 把 FSDP 模型权重提取出来
     → DTensor → 普通 Tensor
     → 同步到 vLLM 引擎（update_weights）
  2. self.rollout.generate_sequences(prompts)
     → vLLM 用 PagedAttention 做高效推理
     → 返回 generated tokens + log probs
  3. await trainer_mode()
     → 释放 vLLM 的 KV cache
     → 恢复 FSDP 训练模式
```

**Hybrid Engine 的意义**：同一组 GPU 既做训练又做推理。训练时权重在 FSDP 里，推理时临时同步到 vLLM，推理完释放内存回来训练。这就是 `hybrid_engine=True` 的核心。

### 阶段 3：数据合并 (Line 1537-1569)

```python
# 把原始 batch 也复制 n=4 次（让 ground_truth 和 response 对齐）
batch = batch.repeat(repeat_times=4, interleave=True)

# 合并生成结果
batch = batch.union(gen_batch_output)
# 现在 batch 包含：prompts, responses, rollout_log_probs, ground_truths, ...

# 计算 response_mask（标记哪些 token 是 response 而不是 prompt）
batch.batch["response_mask"] = compute_response_mask(batch)
# response_mask: [0,0,...,0, 1,1,1,...,1, 0]
#                 prompt部分  response部分  padding
```

### 阶段 4：Reward 计算 (Line 1590-1604)

```python
# 同步模式（launch_reward_fn_async=False）
reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
    batch, reward_fn=self.reward_fn, return_dict=False
)
# → 调用 BatchRewardManager.__call__(batch)
# → 内部调用 grpo_batch_reward.compute_score()
# → 内部调用 shared verifier 批量判题
# → 返回 reward_tensor: shape (64, response_length)
#          reward 只在每个 response 的最后一个有效 token 处有值
```

**reward_tensor 的形状理解**：
```
response:    [token1, token2, ..., tokenN, pad, pad]
reward:      [  0,      0,   ...,  0.6,    0,   0 ]
                                    ↑ 奖励只在最后一个有效 token
```

**reward_extra_infos_dict 写入 batch**（Line 1715-1716）：
```python
if reward_extra_infos_dict:
    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
# 现在 batch.non_tensor_batch 有了：
#   "accepted": np.array([False, True, ...])
#   "pass_ratio_all": np.array([0.6, 1.0, ...])
#   "error_type": np.array(["wrong_answer", "success", ...])
#   "judge_time_s": np.array([2.3, 1.1, ...])
#   "invalid_for_rl": np.array([False, False, ...])
#   等等
```

### 阶段 5：Old Log Probability (Line 1637-1662)

```python
old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
# → 调用 actor_rollout_wg.compute_log_prob(batch)
# → DataParallelPPOActor._forward_micro_batch()
# → 用当前策略前向传播一次
# → 输出 old_log_probs: shape (64, response_length)
# → 还输出 entropy（用于监控策略探索度）

batch = batch.union(old_log_prob)
# 现在 batch.batch["old_log_probs"] 已就绪
```

**为什么需要 old_log_probs？** PPO 的重要性采样比率：
```
ratio = exp(log_prob_new - log_prob_old)
```
`old_log_probs` 是 mini-batch 更新开始前的快照，`log_prob_new` 在每次 gradient step 后变化。

### 阶段 6：Reference Policy Log Prob (Line 1681-1684)

```python
if self.use_reference_policy:
    ref_log_prob = self._compute_ref_log_prob(batch)
    batch = batch.union(ref_log_prob)
# → batch.batch["ref_log_prob"] 就绪
# 用于 KL 散度计算（虽然 GRPO smoke 中 use_kl_loss=False）
```

### 阶段 7：Value 计算 (Line 1692-1695)

```python
if self.use_critic:      # GRPO 不用 critic → 跳过
    values = self._compute_values(batch)
    batch = batch.union(values)
```

**GRPO vs PPO 的区别**：PPO 需要 critic 来估计 V(s) 做 GAE；GRPO 不需要，直接用组内奖励归一化。

### 阶段 8：KL 惩罚 (Line 1718-1730)

```python
if self.config.algorithm.use_kl_in_reward:  # smoke 中 = False → 跳过
    batch = apply_kl_penalty(batch, kl_ctrl=..., kl_penalty=...)
    # token_level_rewards = token_level_scores - β * KL(π_θ || π_ref)
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    # 直接用原始 reward，不加 KL
```

### 阶段 9：Advantage 计算 — GRPO 核心算法 ⭐

**文件**: `verl/trainer/ppo/ray_trainer.py` Line 188-277 的 `compute_advantage()`
**文件**: `verl/trainer/ppo/core_algos.py` 的 `compute_grpo_outcome_advantage()`

```python
batch = compute_advantage(
    batch,
    adv_estimator=AdvantageEstimator.GRPO,
    num_repeat=4,                          # 每个 prompt 的回复数
    norm_adv_by_std_in_grpo=True,
)
```

**`compute_grpo_outcome_advantage()` 的逻辑**（core_algos.py）：

```python
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    # 步骤 1：把 token-level reward 加总为 scalar
    scores = token_level_rewards.sum(dim=-1)  # shape: (64,)
    # 因为 reward 只在最后一个 token 有值，sum 后就是那个值

    # 步骤 2：按 uid 分组
    # index = ["uuid-A", "uuid-A", "uuid-A", "uuid-A",  ← prompt A 的 4 个回复
    #          "uuid-B", "uuid-B", "uuid-B", "uuid-B",  ← prompt B 的 4 个回复
    #          ...]
    id2score = defaultdict(list)
    for i in range(batch_size):
        id2score[index[i]].append(scores[i])

    # 步骤 3：计算组内均值和标准差
    for uid in id2score:
        group_scores = torch.stack(id2score[uid])
        id2mean[uid] = group_scores.mean()   # μ
        id2std[uid] = group_scores.std()     # σ

    # 步骤 4：归一化
    for i in range(batch_size):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + ε)
            # A_i = (R_i - μ) / σ
        else:
            scores[i] = scores[i] - id2mean[index[i]]
            # Dr.GRPO 变体: A_i = R_i - μ（不除以 σ）

    # 步骤 5：广播到 token 级别
    advantages = scores.unsqueeze(-1) * response_mask
    # shape: (64, response_length)
    # 同一个 response 的所有 token 共享同一个 advantage 值

    return advantages, advantages  # GRPO 中 returns = advantages
```

**数值示例**：
```
Prompt A 的 4 个回复：
  回复 1: reward = 0.8  → advantage = (0.8 - 0.6) / 0.15 = +1.33  ← 好回复，增加概率
  回复 2: reward = 0.7  → advantage = (0.7 - 0.6) / 0.15 = +0.67
  回复 3: reward = 0.5  → advantage = (0.5 - 0.6) / 0.15 = -0.67
  回复 4: reward = 0.4  → advantage = (0.4 - 0.6) / 0.15 = -1.33  ← 差回复，降低概率
  组内均值 μ = 0.6, 标准差 σ = 0.15
```

### 阶段 10：Critic 更新 (Line 1782-1792)

```python
if self.use_critic:  # GRPO 跳过
    critic_output = self._update_critic(batch)
```

### 阶段 11：Actor 更新 ⭐ (Line 1794-1816)

```python
actor_output = self._update_actor(batch)
# → 调用 actor_rollout_wg.update_actor(batch)
# → DataParallelPPOActor.update_policy()
```

**`update_policy()` 内部**（`verl/workers/actor/dp_actor.py`）：

```
update_policy(data):
  # 分成 mini-batch，每个 mini-batch 做一次梯度更新
  for mini_batch in split(data, ppo_mini_batch_size):
      # 分成 micro-batch（显存优化）
      for micro_batch in split(mini_batch, ppo_micro_batch_size_per_gpu):
          # 1. 前向传播：用当前策略计算 new_log_prob
          new_log_prob = self._forward_micro_batch(micro_batch)

          # 2. 计算 PPO loss
          ratio = exp(new_log_prob - old_log_prob)
          surr1 = ratio * advantage
          surr2 = clip(ratio, 1 - clip_low, 1 + clip_high) * advantage
          # 注意：clip_low=0.2, clip_high=0.28 是非对称的（DAPO 风格）
          loss = -min(surr1, surr2)

          # 3. loss 聚合方式
          if loss_agg_mode == "token-mean":
              loss = (loss * response_mask).sum() / response_mask.sum()

          # 4. 反向传播
          loss.backward()  # 梯度累积

      # 5. 梯度裁剪 + 优化器更新
      grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
      optimizer.step()
      optimizer.zero_grad()
```

**PPO clip 的直觉理解**：
```
如果 advantage > 0（好回复）：
  ratio > 1 → 策略更倾向这个回复了 → 好
  但 clip 不让 ratio 超过 1 + clip_high = 1.28 → 防止更新太猛

如果 advantage < 0（差回复）：
  ratio < 1 → 策略不那么倾向这个回复了 → 好
  但 clip 不让 ratio 低于 1 - clip_low = 0.8 → 防止更新太猛

DAPO 风格：clip_high > clip_low → 允许好回复的概率增加幅度 > 差回复的概率降低幅度
→ 鼓励探索新的好代码模式
```

### 阶段 12：Metrics 收集 (Line 1896-1928)

```python
metrics.update(compute_data_metrics(batch=batch))       # reward 统计、response 长度
metrics.update(compute_verifier_metrics(batch=batch))   # ← 我们新增的 verifier 指标
metrics.update(compute_timing_metrics(batch, timing))   # 各阶段耗时
metrics.update(compute_throughout_metrics(batch, ...))   # 吞吐量
```

**`compute_verifier_metrics()`** (metric_utils.py Line 234-266)：
```
从 batch.non_tensor_batch 读取 reward_extra_info 写入的字段：
  verifier/pass_ratio_all_mean   ← np.mean(batch.non_tensor_batch["pass_ratio_all"])
  verifier/accepted_rate         ← np.mean(batch.non_tensor_batch["accepted"])
  verifier/invalid_for_rl_rate
  verifier/judge_time_s_mean, judge_time_s_p95
  verifier/syntax_error_rate, runtime_error_rate, timeout_rate, wrong_answer_rate
  verifier/empty_output_rate, non_code_rate, extraction_failure_rate
```

### 阶段 13：Checkpoint 保存 + 验证 (Line 1823-1840)

```python
# 验证（每 test_freq 步做一次）
val_metrics = self._validate()
# → 用 val_reward_fn 在验证集上生成 + 判题 + 计算 reward
# → 返回 per-data-source 的 acc/reward 指标

# 保存 checkpoint（如果 save_freq > 0）
self._save_checkpoint()
```

---

## A5. 训练循环数据流全景

```
Parquet 文件
  │  (prompt, ground_truth, data_source, extra_info)
  ▼
DataLoader → batch_dict (16 个 prompt)
  │
  ▼
DataProto.from_single_dict(batch_dict)
  │  .batch = {input_ids, attention_mask}
  │  .non_tensor_batch = {reward_model: {ground_truth: ...}, data_source, extra_info, uid}
  │
  ▼ repeat(n=4, interleave=True)
64 条 (每个 prompt 4 份)
  │
  ▼ generate_sequences() → vLLM
  │  .batch += {responses, rollout_log_probs}
  │
  ▼ compute_reward() → BatchRewardManager → grpo_batch_reward → shared verifier → SandboxFusion
  │  .batch += {token_level_scores}     ← reward 只在最后一个 token
  │  .non_tensor_batch += {accepted, pass_ratio_all, error_type, judge_time_s, ...}
  │
  ▼ compute_old_log_prob() → Actor forward
  │  .batch += {old_log_probs, entropys}
  │
  ▼ compute_ref_log_prob() → Reference forward
  │  .batch += {ref_log_prob}
  │
  ▼ token_level_rewards = token_level_scores  (因为 use_kl_in_reward=False)
  │
  ▼ compute_advantage(GRPO)
  │  → 按 uid 分组，组内 (R_i - μ) / σ
  │  .batch += {advantages, returns}
  │
  ▼ update_actor() → PPO clipped loss → gradient → optimizer.step()
  │
  ▼ compute_verifier_metrics() → WandB
```

---

# Part B：Eval 评估完整链路

## B1. 入口：`run_phase0.sh`

**文件**: `coding_model_project/scripts/run_phase0.sh`

```bash
# 1. 服务检查（Line 74-87）
curl vLLM /v1/models          # 确认推理服务在线
curl SandboxFusion /health    # 确认判题服务在线

# 2. manifest 检查（Line 91-98）
# 强制要求 data/manifests/*.jsonl 存在（因为 shared verifier 只走 external tests）

# 3. 运行评测（Line 105-118）
python src/phase0_eval.py \
    --mode simple \                    # 连接已有 vLLM，不启分布式
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --vllm_url http://localhost:8001 \
    --sandbox_url http://localhost:8080 \
    --manifest_dir data/manifests \    # 本地去重后的数据
    --datasets humaneval mbpp_reg codecontests_valid \
    --temperature 0.0 \                # EVAL@1 贪心解码
    --max_tokens 2048 \
    --run_timeout 30 \
    --max_concurrent 32 \              # 最大并发生成请求数
    --verifier_limiter_budget 8 \      # sandbox 并发上限
    --batch_size 50                    # 每批处理 50 题
```

## B2. Eval 主脚本：`phase0_eval.py`

**文件**: `coding_model_project/src/phase0_eval.py` (2600+ 行)

### 入口函数 `main()` (Line 2082)

```
main()
  ├── argparse 解析参数 → EvalConfig
  └── asyncio.run(run_evaluation(config))
```

### `run_evaluation()` (Line 1894-2075)

```
run_evaluation(config):
  │
  ├── 1. 获取服务地址
  │   simple 模式: 直接解析 vllm_url → ["localhost:8001"]
  │   verl 模式: start_rollout_servers() → 启动 Ray + vLLM replicas
  │
  ├── 2. 记录 run_info（审计用）
  │   config, 命令行参数, 模型信息 → outputs/run_info.json
  │
  ├── 3. 初始化组件
  │   MetricsCollector (指标收集)
  │   QALogger (问答日志采样)
  │
  ├── 4. 对每个数据集:
  │   ├── load_prompts(dataset_key, config)     # 加载题目
  │   └── evaluate_dataset(dataset_key, ...)     # 评测
  │
  └── 5. 保存结果
      metrics.json, qa_logs/, per_problem/*.jsonl, summary.json
```

### `load_prompts()` — 数据加载 (Line 1433-1563)

```
load_prompts(dataset_key, config):
  │
  ├── 如果有 manifest_dir:
  │   _load_from_manifest()
  │   ├── 读 manifests/humaneval_manifest.jsonl  → problem_id 白名单
  │   ├── 读 raw/humaneval_raw.jsonl             → 完整数据（含 test_cases）
  │   └── 按 problem_id 过滤 → [{problem_id, prompt, test_cases}, ...]
  │
  └── 否则:
      _load_from_sandbox()
      ├── 调用 SandboxFusion get_prompts API
      └── 只有 prompt，没有 test_cases → 无法用 shared verifier
```

**test_cases 的格式**（根据数据集不同）：
```python
# HumanEval
{"type": "humaneval", "test_code": "def check(candidate):\n    assert ...", "entry_point": "func_name"}

# MBPP
{"type": "mbpp", "test_list": ["assert func(1) == 2", ...], "test_setup_code": "", "entry_point": "func", "example_call": "func(1)"}

# CodeContests
{"type": "codecontests", "tests": [{"input": "5\n1 2 3 4 5", "output": "15"}, ...]}
```

### `evaluate_dataset()` — 评测主循环 (Line 1616-1891)

```
evaluate_dataset(dataset_key, prompts, server_addresses, config, ...):
  │
  ├── 对每批 50 题:
  │   │
  │   ├── 1. 格式化 prompt
  │   │   format_prompt(raw_prompt, dataset_key)  # 从 prompting.py
  │   │   → 加上 system prompt 和 dataset-specific 指令
  │   │
  │   ├── 2. 批量生成 (async)
  │   │   batch_generate(server_addresses, prompts, ...)
  │   │   → 对每个 prompt: POST /v1/chat/completions 到 vLLM
  │   │   → 并发上限 = max_concurrent_requests (64)
  │   │   → Round-Robin 负载均衡到多个 server
  │   │   → 返回 [(completion_text, {gen_tokens, gen_time, finish_reason}), ...]
  │   │
  │   ├── 3. 批量判题 (async)
  │   │   对每题: evaluate_single_problem_async()
  │   │     └── evaluate_with_run_code(completion, test_cases, problem_id, config)
  │   │           ├── candidate = normalize_candidate(completion)   # shared verifier
  │   │           ├── summary = verify_candidate(candidate, ...)   # shared verifier
  │   │           └── return _summary_to_eval_result(summary)
  │   │   → 并发上限 = max_concurrent_judges (16)
  │   │
  │   └── 4. 收集结果
  │       metrics_collector.add_result(dataset_key, eval_result)
  │       qa_logger.log(...)
  │       per_problem.jsonl 写入每题详情
  │
  └── 计算数据集级别统计:
      accepted_at_1, pass_ratio_mean/p50/p90, throughput, cost_per_solved, ...
```

### Eval 与 RL 共用的代码提取 + 判题流程

```
                    ┌──────────────────────────┐
                    │  normalize_candidate()   │
                    │  (verifier/shared.py:187) │
   模型输出文本 ──→ │                          │ ──→ CandidateRecord
                    │  <code>标签 > markdown >  │      {extracted_code,
                    │  raw code > non_code     │       extraction_status}
                    └──────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────┐
                    │   verify_candidate()     │
                    │  (verifier/shared.py:645) │
                    │                          │
                    │  按 test_cases["type"]    │
                    │  分派到:                  │ ──→ VerificationSummary
                    │  - humaneval (Line 286)   │      {accepted, passed_tests,
                    │  - mbpp     (Line 355)   │       total_tests, pass_ratio_all,
                    │  - codecontests (Line 553)│       error_type, invalid_for_rl,
                    │                          │       judge_time_s, ...}
                    └──────────────────────────┘
                                │
                    (codecontests 路径详解)
                                │
                    ┌──────────────────────────┐
                    │ _verify_codecontests_    │
                    │ candidate()   (Line 553)  │
                    │                          │
                    │  1. autofix entrypoint   │
                    │  2. ThreadPoolExecutor   │
                    │     每个 testcase 一个线程│
                    │  3. 每个线程:             │
                    │     _run_code_request()  │
                    │     → BoundedSemaphore   │
                    │     → sandbox run_code   │
                    │     → 比较 stdout/expect │
                    │  4. 聚合 per_case_results│
                    │     pass_ratio = passed/N│
                    └──────────────────────────┘
```

---

# Part C：共享组件详解

## C1. `verifier/shared.py` — 共享判题模块

**关键函数定位表**：

| 函数 | 行号 | 作用 | 调用者 |
|---|---|---|---|
| `normalize_candidate()` | 187 | 从模型输出提取代码 | eval + RL |
| `_extract_code_block()` | 157 | 代码提取核心逻辑 | normalize_candidate |
| `_looks_like_python_code()` | 128 | 裸代码检测 | _extract_code_block |
| `verify_candidate()` | 645 | 单题判定（分派器） | eval |
| `verify_candidate_batch()` | 718 | 批量并发判定 | RL reward |
| `_verify_humaneval_candidate()` | 286 | HumanEval 判题 | verify_candidate |
| `_verify_mbpp_candidate()` | 355 | MBPP 判题 | verify_candidate |
| `_verify_codecontests_candidate()` | 553 | CodeContests 判题 | verify_candidate |
| `_verify_codecontests_testcase()` | 438 | 单个 testcase 判定 | _verify_codecontests |
| `_run_code_request()` | 240 | 调用 SandboxFusion API | 所有 _verify_* |
| `_pick_primary_error()` | 538 | 聚合错误类型 | _verify_codecontests |
| `_codecontests_autofix_entrypoint()` | 191 | 自动补 `solve()` 调用 | _verify_codecontests |
| `_acquire_limiter()` | 54 | 全局并发控制 | _run_code_request |

## C2. `grpo_batch_reward.py` — RL Reward 函数

**关键函数定位表**：

| 函数 | 行号 | 作用 |
|---|---|---|
| `compute_score()` | 38 | 入口函数，被 BatchRewardManager 调用 |
| `_map_reward()` | 17 | reward 映射：VerificationSummary → float |
| `_coerce_iterable()` | 11 | 类型转换辅助 |

**修改 reward 公式的位置**：`_map_reward()` 函数，Line 17-35。

## C3. `prompting.py` — Prompt 模板

**修改 prompt 的位置**：`PROMPT_TEMPLATES` dict，Line 14-103。
**修改 system prompt 的位置**：`SYSTEM_PROMPT`，Line 4-11。

## C4. `build_grpo_parquet.py` — 数据构建

**关键函数定位表**：

| 函数 | 行号 | 作用 |
|---|---|---|
| `build_default_splits()` | 134 | 主函数：读 manifest + raw → 生成 6 个 Parquet |
| `_filter_raw_by_manifest_ids()` | 56 | 按 manifest 过滤 raw 数据 |
| `_convert_record()` | 90 | 把原始记录转成 verl Parquet 行格式 |
| `_build_prompt_messages()` | 76 | 构建 prompt message list |
| `_write_parquet()` | 124 | 写 Parquet 文件 |

---

# Part D：verl 框架核心组件

## D1. DataProto — 统一数据格式

**文件**: `verl/protocol.py`

```python
@dataclass
class DataProto:
    batch: TensorDict         # GPU tensor 数据 (batch_size, ...)
    non_tensor_batch: dict    # CPU numpy 数据 (metadata, ground_truth, ...)
    meta_info: dict           # 全局元数据 (temperature, timing, ...)
```

**训练循环中 batch 的内容演变**：

| 阶段 | batch (tensor) 新增 | non_tensor_batch 新增 |
|---|---|---|
| 初始 | input_ids, attention_mask | reward_model, data_source, extra_info, uid |
| rollout 后 | responses, rollout_log_probs | |
| reward 后 | token_level_scores | accepted, pass_ratio_all, error_type, ... |
| old_log_prob 后 | old_log_probs, entropys | |
| ref_log_prob 后 | ref_log_prob | |
| KL 处理后 | token_level_rewards | |
| advantage 后 | advantages, returns | |
| response_mask | response_mask | |

## D2. Worker 体系

```
ActorRolloutRefWorker (fsdp_workers.py)
├── 角色：actor + rollout + ref（三合一，hybrid engine）
├── 训练模式 → FSDP 分布式训练
├── 推理模式 → 权重同步到 vLLM
│
├── generate_sequences()     # rollout: vLLM 推理
│   ├── rollout_mode()       # 权重 FSDP → vLLM
│   ├── vllm.generate()      # vLLM 生成
│   └── trainer_mode()       # 切回 FSDP
│
├── compute_log_prob()       # actor: 前向传播算 log prob
│   └── DataParallelPPOActor._forward_micro_batch()
│
├── update_actor()           # actor: PPO 梯度更新
│   └── DataParallelPPOActor.update_policy()
│
└── compute_ref_log_prob()   # ref: 冻结模型前向传播
```

## D3. Hybrid Engine 权重同步

```
训练模式（FSDP）:
  模型权重分散在 4 张 GPU 上（FSDP 分片）
  ├── GPU 0: shard_0
  ├── GPU 1: shard_1
  ├── GPU 2: shard_2
  └── GPU 3: shard_3

rollout_mode() 切换:
  1. AllGather 收集完整权重到每张 GPU
  2. DTensor → 普通 Tensor
  3. 生成器逐层传给 vLLM: yield (name, tensor)
  4. vLLM 调用 model.load_weights() 加载

trainer_mode() 切回:
  1. vLLM 释放 KV cache（如果 free_cache_engine=True）
  2. 恢复 FSDP 训练状态
  3. 模型切回 training mode
```

## D4. vLLM Rollout

**关键参数**：
```yaml
rollout.name: vllm
rollout.tensor_model_parallel_size: 1      # TP=1，每张 GPU 一个完整模型
rollout.gpu_memory_utilization: 0.35       # vLLM 用 35% 显存（剩下给训练）
rollout.n: 4                               # 每个 prompt 生成 4 个回复
```

**显存分配**（4×5090 32GB，TP=1）：
```
每张 GPU 32GB:
  FSDP 训练: ~14GB (模型参数 + 优化器状态 + 梯度, gradient checkpoint 开启)
  vLLM 推理: 32 × 0.35 = ~11.2GB (模型副本 + KV cache)
  剩余: ~6.8GB (激活值等)
```

**如果 OOM**：第一选择是改 `tensor_model_parallel_size: 1 → 2`（2 张 GPU 合力跑推理）。

---

# Part E：关键修改位置速查表

| 想修改什么 | 改哪个文件 | 具体位置 |
|---|---|---|
| Reward 映射公式 | `grpo_batch_reward.py` | `_map_reward()` Line 17 |
| 新增 reward mode | `grpo_batch_reward.py` | `_map_reward()` 加 elif |
| Prompt 模板 | `prompting.py` | `PROMPT_TEMPLATES` Line 14 |
| System Prompt | `prompting.py` | `SYSTEM_PROMPT` Line 4 |
| 代码提取逻辑 | `verifier/shared.py` | `_extract_code_block()` Line 157 |
| CodeContests 判题 | `verifier/shared.py` | `_verify_codecontests_candidate()` Line 553 |
| 错误分类 | `verifier/shared.py` | `_classify_single_run()` Line 267 |
| invalid_for_rl 规则 | `verifier/shared.py` | `_verify_codecontests_candidate()` Line 622 |
| WandB 训练指标 | `metric_utils.py` | `compute_verifier_metrics()` Line 234 |
| GRPO 算法参数 | `run_grpo_smoke.sh` | clip_ratio, kl_loss, rollout.n 等 |
| 训练数据 | `run_grpo_smoke.sh` | `DATA_DIR` 和 `data.train_files` |
| Sandbox 并发度 | `run_grpo_smoke.sh` | `LIMITER_BUDGET` |
| Eval 数据集 | `run_phase0.sh` | `DATASETS` 变量 |
| Eval 并发度 | `run_phase0.sh` | `MAX_CONCURRENT`, `VERIFIER_LIMITER_BUDGET` |
| 训练拓扑（GPU 数等） | `run_grpo_smoke.sh` | `trainer.n_gpus_per_node`, `tensor_model_parallel_size` |
| PPO 超参 | `run_grpo_smoke.sh` | `clip_ratio_low/high`, `ppo_mini_batch_size`, `lr` |
| 切换算法变体 | `run_grpo_smoke.sh` | `algorithm.*` 参数 |
| A0→A1→A2 切换 | 见 `algorithm_decision_guide.md` | 改 clip_ratio, use_kl_loss, norm_adv 等 |
