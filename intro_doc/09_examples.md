# 示例代码解读

## 1. 示例目录结构

verl 提供了丰富的训练示例：

```
examples/
├── ppo_trainer/           # PPO 训练示例
│   ├── run_qwen2-7b_rm.sh
│   ├── run_deepseek7b_llm.sh
│   └── ...
├── grpo_trainer/          # GRPO 训练示例
│   ├── run_qwen2-7b.sh
│   ├── run_qwen3-8b.sh
│   └── ...
├── remax_trainer/         # ReMax 训练示例
├── rloo_trainer/          # RLOO 训练示例
├── sft/                   # 监督微调示例
│   ├── gsm8k/
│   └── multiturn/
├── sglang_multiturn/      # SGLang 多轮对话示例
├── tuning/                # 超参调优示例
│   ├── 0.5b/
│   ├── 7b/
│   └── 32b/
└── data_preprocess/       # 数据预处理脚本
```

---

## 2. GRPO 训练示例

### 2.1 基础 GRPO 示例

**文件**：`examples/grpo_trainer/run_qwen2-7b.sh`

```bash
python3 -m verl.trainer.main_ppo \
    # ===== 算法配置 =====
    algorithm.adv_estimator=grpo \           # 使用 GRPO 优势估计

    # ===== 数据配置 =====
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \

    # ===== 模型配置 =====
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \

    # ===== Actor 配置 =====
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \

    # ===== FSDP 配置 =====
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \

    # ===== Rollout 配置 =====
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \   # 每个 prompt 生成 5 个响应

    # ===== Reference 配置 =====
    actor_rollout_ref.ref.fsdp_config.param_offload=True \

    # ===== Trainer 配置 =====
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.total_epochs=15
```

### 2.2 关键参数解释

| 参数 | 说明 |
|-----|------|
| `algorithm.adv_estimator=grpo` | 使用 GRPO 优势估计（无需 Critic） |
| `actor_rollout_ref.rollout.n=5` | 每个 prompt 生成 5 个响应用于组内比较 |
| `actor_rollout_ref.actor.use_kl_loss=True` | 使用 KL 损失作为正则化 |
| `actor_rollout_ref.ref.fsdp_config.param_offload=True` | Reference 模型参数卸载到 CPU |

---

## 3. PPO 训练示例

### 3.1 使用奖励模型的 PPO

**文件**：`examples/ppo_trainer/run_qwen2-7b_rm.sh`

```bash
python3 -m verl.trainer.main_ppo \
    # ===== 算法配置 =====
    algorithm.adv_estimator=gae \            # PPO 使用 GAE
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=True \        # 在奖励中使用 KL 惩罚

    # ===== KL 控制配置 =====
    algorithm.kl_ctrl.type=adaptive \        # 自适应 KL 控制
    algorithm.kl_ctrl.kl_coef=0.02 \
    algorithm.kl_ctrl.target_kl=0.01 \

    # ===== 模型配置 =====
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \

    # ===== Actor 配置 =====
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_epochs=4 \   # PPO 多轮更新
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \

    # ===== Critic 配置 =====
    critic.optim.lr=1e-5 \
    critic.ppo_epochs=4 \
    critic.cliprange_value=0.2 \

    # ===== 奖励模型配置 =====
    reward_model.enable=True \
    reward_model.model.path=weqweasdas/RM-Gemma-2B \
    reward_model.micro_batch_size_per_gpu=16 \

    # ===== Trainer 配置 =====
    trainer.critic_warmup=10 \               # Critic 预热步数
    trainer.total_epochs=1
```

### 3.2 使用函数奖励的 PPO

对于可验证的任务（如数学），可以使用函数奖励：

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \

    # 禁用奖励模型，使用函数奖励
    reward_model.enable=False \

    # 函数奖励在 trainer 中配置
    trainer.reward_fn.type=function \
    trainer.reward_fn.path=verl.utils.reward_score.gsm8k_reward
```

---

## 4. Megatron 后端示例

### 4.1 使用 Megatron 训练大模型

**文件**：`examples/grpo_trainer/run_deepseek671b_math_megatron_80gb.sh`

```bash
python3 -m verl.trainer.main_ppo \
    # ===== 模型配置 =====
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-V3 \

    # ===== 使用 Megatron 后端 =====
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron_config.tensor_parallel_size=8 \
    actor_rollout_ref.actor.megatron_config.pipeline_parallel_size=4 \
    actor_rollout_ref.actor.megatron_config.expert_parallel_size=8 \  # MoE 专家并行

    # ===== Rollout 配置 =====
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \

    # ===== 多节点训练 =====
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4
```

---

## 5. LoRA 微调示例

### 5.1 GRPO + LoRA

**文件**：`examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh`

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \

    # ===== LoRA 配置 =====
    actor_rollout_ref.actor.peft_config.peft_type=lora \
    actor_rollout_ref.actor.peft_config.lora_r=16 \
    actor_rollout_ref.actor.peft_config.lora_alpha=32 \
    actor_rollout_ref.actor.peft_config.target_modules='["q_proj","v_proj"]' \

    # ===== Rollout 也需要 LoRA =====
    actor_rollout_ref.rollout.enable_lora=True
```

---

## 6. 多轮对话示例

### 6.1 SGLang 多轮训练

**文件**：`examples/sglang_multiturn/run_qwen2-7b_search.sh`

```bash
python3 -m verl.trainer.main_ppo \
    # ===== 使用 SGLang =====
    actor_rollout_ref.rollout.name=sglang \

    # ===== 多轮配置 =====
    actor_rollout_ref.rollout.multi_turn=True \
    actor_rollout_ref.rollout.max_turns=5 \

    # ===== 工具配置 =====
    actor_rollout_ref.rollout.tools='["search"]' \
    actor_rollout_ref.rollout.tool_config.search.api_key=$SEARCH_API_KEY
```

---

## 7. 监督微调 (SFT) 示例

### 7.1 基础 SFT

**文件**：`examples/sft/gsm8k/run_qwen_05.sh`

```bash
python3 -m verl.trainer.fsdp_sft_trainer \
    # ===== 数据配置 =====
    data.train_files=data/gsm8k_train.parquet \
    data.val_files=data/gsm8k_test.parquet \

    # ===== 模型配置 =====
    model.path=Qwen/Qwen2-0.5B \
    model.enable_gradient_checkpointing=True \

    # ===== 训练配置 =====
    training.learning_rate=2e-5 \
    training.num_train_epochs=3 \
    training.per_device_train_batch_size=8 \
    training.gradient_accumulation_steps=4 \

    # ===== FSDP 配置 =====
    fsdp.strategy=fsdp2
```

---

## 8. 数据预处理

### 8.1 准备训练数据

```python
# examples/data_preprocess/gsm8k.py

import pandas as pd
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("openai/gsm8k", "main")

# 转换为 parquet 格式
def process_example(example):
    return {
        "prompt": example["question"],
        "response": example["answer"],
        "reward": 1.0,  # 可选的奖励标签
    }

train_data = [process_example(ex) for ex in dataset["train"]]
test_data = [process_example(ex) for ex in dataset["test"]]

pd.DataFrame(train_data).to_parquet("train.parquet")
pd.DataFrame(test_data).to_parquet("test.parquet")
```

### 8.2 数据格式要求

```python
# 必需字段
{
    "prompt": str,      # 输入提示
}

# 可选字段
{
    "response": str,    # 期望的响应（用于 SFT 或验证）
    "reward": float,    # 奖励标签
    "ground_truth": str,  # 用于函数奖励的真实答案
}
```

---

## 9. 自定义奖励函数

### 9.1 实现自定义奖励

```python
# my_reward.py

import re

def gsm8k_reward_fn(response: str, ground_truth: str) -> float:
    """GSM8K 数学问题的奖励函数"""

    # 提取答案
    def extract_answer(text):
        match = re.search(r'#### (\d+)', text)
        if match:
            return int(match.group(1))
        return None

    pred_answer = extract_answer(response)
    true_answer = extract_answer(ground_truth)

    # 比较答案
    if pred_answer is not None and pred_answer == true_answer:
        return 1.0
    else:
        return 0.0
```

### 9.2 在配置中使用

```bash
python3 -m verl.trainer.main_ppo \
    trainer.reward_fn.path=my_reward.gsm8k_reward_fn
```

---

## 10. 自定义算法扩展

### 10.1 注册新的优势估计器

```python
# my_algorithm.py

from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("my_estimator")
def compute_my_advantage(
    token_level_rewards,
    response_mask,
    **kwargs
):
    """自定义优势估计器"""
    # 实现自定义逻辑
    advantages = token_level_rewards * response_mask
    return advantages, advantages
```

### 10.2 注册新的策略损失

```python
from verl.trainer.ppo.core_algos import register_policy_loss

@register_policy_loss("my_loss")
def compute_my_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    **kwargs
):
    """自定义策略损失"""
    # 实现自定义损失
    ratio = torch.exp(log_prob - old_log_prob)
    loss = -advantages * ratio
    loss = (loss * response_mask).sum() / response_mask.sum()
    return loss, {"my_metric": loss.item()}
```

### 10.3 使用自定义算法

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=my_estimator \
    actor_rollout_ref.actor.policy_loss_fn=my_loss
```

---

## 11. 运行示例的最佳实践

### 11.1 环境准备

```bash
# 1. 安装 verl
pip install verl

# 2. 安装推理引擎
pip install vllm  # 或 sglang

# 3. 准备数据
python examples/data_preprocess/gsm8k.py

# 4. 设置环境变量
export WANDB_API_KEY=your_key
export HF_TOKEN=your_token
```

### 11.2 单机多卡训练

```bash
# 8 GPU 单机
bash examples/grpo_trainer/run_qwen2-7b.sh
```

### 11.3 多机分布式训练

```bash
# 节点 0 (主节点)
RAY_HEAD_ADDRESS=auto \
bash examples/grpo_trainer/run_qwen2-7b.sh \
    trainer.nnodes=2 \
    trainer.node_rank=0

# 节点 1
RAY_HEAD_ADDRESS=<master_ip>:6379 \
bash examples/grpo_trainer/run_qwen2-7b.sh \
    trainer.nnodes=2 \
    trainer.node_rank=1
```

---

## 12. 示例代码路径汇总

| 任务 | 示例路径 |
|-----|---------|
| GRPO 训练 | `examples/grpo_trainer/` |
| PPO 训练 | `examples/ppo_trainer/` |
| ReMax 训练 | `examples/remax_trainer/` |
| RLOO 训练 | `examples/rloo_trainer/` |
| 监督微调 | `examples/sft/` |
| 多轮对话 | `examples/sglang_multiturn/` |
| 数据预处理 | `examples/data_preprocess/` |
| 超参调优 | `examples/tuning/` |

---

## 13. 总结

通过这些示例，您可以快速上手：

1. **基础训练**：使用 GRPO 或 PPO 进行强化学习训练
2. **大模型训练**：使用 Megatron 后端训练超大模型
3. **高效训练**：使用 LoRA 进行参数高效微调
4. **复杂任务**：实现多轮对话、工具调用等高级功能
5. **自定义扩展**：注册自定义的优势估计器和损失函数

如需更多帮助，请参考 [verl 官方文档](https://verl.readthedocs.io/)。
