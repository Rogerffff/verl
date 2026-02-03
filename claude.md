# verl 代码库快速导航

verl (Volcano Engine Reinforcement Learning for LLMs) 是大型语言模型强化学习训练框架，支持 PPO、GRPO、DAPO 等算法，基于 Hybrid-Controller 编程模型。

---

## 顶层目录

```
verl/
├── verl/                 # 核心 Python 包（见下方详细说明）
├── examples/             # 训练示例脚本（PPO/GRPO/SFT 等）
├── docs/                 # Sphinx 文档源文件
├── tests/                # 测试套件
├── recipe/               # 训练配方（verl-recipe 子模块）
├── docker/               # Docker 配置
├── scripts/              # 实用脚本
├── intro_doc/            # 代码库介绍文档（中文，共9篇）
├── doc_index.md          # 文档索引（快速查找所有 docs/ 下文档）
├── setup.py / pyproject.toml  # 包安装配置
└── find_doc/             # 文档查找工具
```

---

## 核心包 `verl/verl/`

### trainer/ — 训练器与算法

训练主循环和 RL 算法实现。

| 路径 | 说明 |
|------|------|
| `trainer/ppo/ray_trainer.py` | Ray 分布式训练器主循环（PPO/GRPO 共用） |
| `trainer/ppo/core_algos.py` | 核心算法：13+ 优势估计器、10+ 策略损失函数（2200+ 行） |
| `trainer/ppo/reward.py` | 奖励处理 |
| `trainer/ppo/metric_utils.py` | 训练指标计算 |
| `trainer/ppo/rollout_corr_helper.py` | 离策略校正辅助 |
| `trainer/config/` | 配置定义（algorithm.py, actor/, critic/ 等） |
| `trainer/main_ppo.py` | 命令行入口 |

### workers/ — Worker 进程

分布式训练中各角色的实现。

| 路径 | 说明 |
|------|------|
| `workers/actor/base.py` | Actor 基类（compute_log_prob, update_policy） |
| `workers/actor/dp_actor.py` | FSDP Actor 实现 |
| `workers/actor/megatron_actor.py` | Megatron Actor 实现 |
| `workers/critic/base.py` | Critic 基类（compute_values, update_critic） |
| `workers/critic/dp_critic.py` | FSDP Critic 实现 |
| `workers/critic/megatron_critic.py` | Megatron Critic 实现 |
| `workers/rollout/base.py` | Rollout 基类 |
| `workers/rollout/vllm_rollout/` | vLLM 推理后端集成 |
| `workers/rollout/sglang_rollout/` | SGLang 推理后端集成 |
| `workers/reward_manager/naive.py` | 基础奖励管理器 |
| `workers/reward_manager/dapo.py` | DAPO 奖励管理器（动态采样过滤） |
| `workers/reward_manager/prime.py` | PRIME 奖励管理器 |
| `workers/engine/fsdp/` | FSDP 训练引擎后端 |
| `workers/engine/megatron/` | Megatron 训练引擎后端 |
| `workers/sharding_manager/` | 模型分片管理（训练↔推理权重同步） |
| `workers/fsdp_workers.py` | FSDP Worker 组合类 |
| `workers/megatron_workers.py` | Megatron Worker 组合类 |

### single_controller/ — 分布式控制器

| 路径 | 说明 |
|------|------|
| `single_controller/base/worker.py` | Worker 基类（分布式 rank 信息） |
| `single_controller/base/worker_group.py` | WorkerGroup 管理（数据分发/收集） |
| `single_controller/ray/base.py` | Ray 集成实现 |

### protocol.py — 数据协议

`DataProto` 统一数据交换格式（48KB），包含：
- `batch`: TensorDict 张量数据
- `non_tensor_batch`: 非张量数据
- `meta_info`: 元数据
- 支持自动填充、切分、合并

### utils/ — 工具模块

| 路径 | 说明 |
|------|------|
| `utils/checkpoint/` | 检查点保存/加载 |
| `utils/dataset/` | 数据集处理 |
| `utils/model.py` | 模型加载 |
| `utils/tokenizer.py` | 分词器 |
| `utils/fsdp_utils.py` | FSDP 工具函数 |
| `utils/torch_functional.py` | PyTorch 操作 |
| `utils/seqlen_balancing.py` | 序列长度平衡 |
| `utils/ulysses.py` | Ulysses 序列并行 |
| `utils/tracking.py` | 实验追踪（wandb 等） |
| `utils/memory_utils.py` | 内存管理 |
| `utils/config.py` | 配置工具 |
| `utils/profiler/` | 性能分析 |
| `utils/sglang/` | SGLang 集成工具 |
| `utils/vllm/` | vLLM 集成工具 |
| `utils/kernel/` | 自定义 CUDA 内核 |

### models/ — 模型实现

模型注册和自定义模型实现。

### experimental/ — 实验性功能

| 路径 | 说明 |
|------|------|
| `experimental/fully_async_policy/` | 全异步策略训练 |
| `experimental/one_step_off_policy/` | 单步离策略学习 |
| `experimental/agent_loop/` | Agent 循环（多轮对话 RL） |
| `experimental/reward_loop/` | 奖励循环架构 |
| `experimental/transfer_queue/` | 异步数据传输队列 |
| `experimental/dynamic_dataset/` | 动态数据集 |
| `experimental/vla/` | Vision-Language Action 模型 |

### 其他模块

| 路径 | 说明 |
|------|------|
| `tools/` | 工具集成（搜索、代码执行） |
| `interactions/` | 环境交互接口 |
| `checkpoint_engine/` | 检查点引擎 |
| `third_party/` | 第三方集成 |
| `base_config.py` | 基础配置类 |

---

## examples/ — 训练示例

| 目录 | 说明 |
|------|------|
| `examples/ppo_trainer/` | PPO 训练（27+ 配置脚本） |
| `examples/grpo_trainer/` | GRPO 训练（50+ 配置脚本） |
| `examples/remax_trainer/` | ReMax 训练 |
| `examples/rloo_trainer/` | RLOO 训练 |
| `examples/reinforce_plus_plus_trainer/` | REINFORCE++ 训练 |
| `examples/gmpo_trainer/` | GMPO 训练 |
| `examples/gspo_trainer/` | GSPO 训练 |
| `examples/sapo_trainer/` | SAPO 训练 |
| `examples/gpg_trainer/` | GPG 训练 |
| `examples/cispo_trainer/` | CISPO 训练 |
| `examples/otb_trainer/` | OTB 训练 |
| `examples/sft/` | 监督微调（gsm8k / multiturn / vlm） |
| `examples/sglang_multiturn/` | SGLang 多轮对话 |
| `examples/split_placement/` | 分离式 GPU 部署 |
| `examples/tuning/` | 超参调优（0.5B-70B） |
| `examples/data_preprocess/` | 数据预处理 |
| `examples/tutorial/` | 入门教程 |

---

## 架构概览

```
训练主循环 (ray_trainer.py)
  │
  ├─ 1. Rollout: 生成序列 → rollout worker (vLLM/SGLang)
  ├─ 2. Reward: 计算奖励 → reward_manager
  ├─ 3. Ref LogProb: 参考模型概率 → actor worker
  ├─ 4. Values: 价值估计 → critic worker (仅 PPO)
  ├─ 5. Advantage: 优势计算 → core_algos.py
  ├─ 6. Update Policy → actor worker
  └─ 7. Update Critic → critic worker (仅 PPO)
```

Worker 层次：
- **Actor**: BasePPOActor → DataParallelPPOActor (FSDP) / MegatronPPOActor
- **Critic**: BasePPOCritic → DataParallelPPOCritic (FSDP) / MegatronPPOCritic
- **Rollout**: BaseRollout → vLLMAsyncRollout / ServerAdapter (SGLang)

---

## 详细文档索引

- `doc_index.md` — 所有 docs/ 目录下文档的分类索引
- `intro_doc/01_overview.md` — 项目总览与目录结构
- `intro_doc/02_architecture.md` — 系统架构详解
- `intro_doc/03_ppo_deep_dive.md` — PPO 算法深入
- `intro_doc/04_grpo_deep_dive.md` — GRPO 算法深入
- `intro_doc/05_ppo_trainer.md` — 训练器工作流程
- `intro_doc/06_workers.md` — Worker 实现详解
- `intro_doc/07_distributed_training.md` — 分布式训练
- `intro_doc/08_configuration.md` — 配置参数说明
- `intro_doc/09_examples.md` — 示例代码说明

---

## 关键算法定位

所有算法实现集中在 `verl/trainer/ppo/core_algos.py`：

**优势估计器**：GAE, GRPO, REINFORCE++, RLOO, ReMax, DAPO, OPO 等

**策略损失函数**：PPO-clip, GRPO, DAPO (token-level), DPO, GPG, CollabLLM 等

**配置入口**：`trainer/config/algorithm.py` 中的 `adv_estimator` 和 `loss_type` 字段选择算法组合。
