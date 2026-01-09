# verl 代码库总览

## 1. 项目简介

verl (Volcano Engine Reinforcement Learning for LLMs) 是由字节跳动 Seed 团队发起并由 verl 社区维护的大型语言模型强化学习训练库。它是 **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** 论文的开源实现。

### 核心特性

1. **灵活的算法表达**：通过 Hybrid-Controller 编程模型，可以用几行代码构建 GRPO、PPO 等 RL 数据流
2. **无缝集成现有 LLM 基础设施**：解耦计算和数据依赖，支持 FSDP、Megatron-LM、vLLM、SGLang 等
3. **灵活的设备映射**：支持将模型放置到不同 GPU 集合上，实现高效资源利用
4. **高性能**：SOTA 的训练和推理引擎集成，3D-HybridEngine 消除内存冗余

### 支持的算法

- **PPO** (Proximal Policy Optimization)
- **GRPO** (Group Relative Policy Optimization)
- **REINFORCE++**
- **RLOO** (Rejection Sampling Least-to-Most Optimization)
- **ReMax**
- **DAPO** (Data Augmented Policy Optimization)
- **GSPO**, **SAPO**, **GPG**, **CISPO** 等

---

## 2. 顶层目录结构

```
verl/
├── verl/                 # 核心 Python 包
├── examples/             # 训练示例脚本
├── docs/                 # 文档
├── tests/                # 测试套件
├── recipe/               # 配方（已迁移至 verl-recipe 子模块）
├── docker/               # Docker 配置
├── scripts/              # 实用脚本
├── .github/              # GitHub CI/CD 工作流
├── setup.py              # 包安装配置
├── pyproject.toml        # 项目配置
└── README.md             # 项目说明
```

---

## 3. 核心包结构 (`verl/`)

核心代码位于 `verl/verl/` 目录，包含以下主要模块：

```
verl/verl/
├── trainer/              # 训练器和 RL 算法实现
├── workers/              # Worker 进程（Actor, Critic, Rollout, Reward）
├── single_controller/    # 单机/分布式控制器
├── models/               # 模型实现和注册
├── utils/                # 工具函数
├── experimental/         # 实验性功能
├── tools/                # 工具集成（搜索、代码执行等）
├── interactions/         # 环境交互接口
├── checkpoint_engine/    # 检查点引擎
├── third_party/          # 第三方集成
├── protocol.py           # 数据协议定义
└── base_config.py        # 基础配置类
```

### 3.1 trainer/ - 训练器模块

训练的核心逻辑所在：

```
trainer/
├── ppo/                  # PPO 训练器
│   ├── core_algos.py     # ⭐ 核心算法实现（2200+ 行）
│   ├── ray_trainer.py    # Ray 分布式训练器
│   ├── reward.py         # 奖励处理
│   ├── metric_utils.py   # 指标计算
│   └── rollout_corr_helper.py  # 离策略校正
├── config/               # 配置定义
│   ├── algorithm.py      # 算法配置
│   ├── actor/            # Actor 配置
│   ├── critic/           # Critic 配置
│   └── ...
├── main_ppo.py           # 命令行入口
└── constants_ppo.py      # 常量定义
```

**关键文件**：`core_algos.py` 包含所有优势估计器和策略损失函数的实现。

### 3.2 workers/ - Worker 模块

处理分布式训练中不同角色的进程：

```
workers/
├── actor/                # Actor 模型 Worker
│   ├── base.py           # 基类定义
│   ├── dp_actor.py       # FSDP Actor（32KB）
│   └── megatron_actor.py # Megatron Actor（38KB）
├── critic/               # Critic 模型 Worker
│   ├── base.py           # 基类定义
│   ├── dp_critic.py      # FSDP Critic
│   └── megatron_critic.py # Megatron Critic
├── rollout/              # 生成/Rollout Worker
│   ├── base.py           # 基类定义
│   ├── vllm_rollout/     # vLLM 集成
│   ├── sglang_rollout/   # SGLang 集成
│   └── naive/            # 简单实现
├── reward_manager/       # 奖励管理
│   ├── naive.py          # 基础奖励管理
│   ├── dapo.py           # DAPO 奖励管理
│   └── prime.py          # PRIME 奖励管理
├── engine/               # 训练引擎后端
│   ├── fsdp/             # FSDP 引擎
│   └── megatron/         # Megatron 引擎
├── sharding_manager/     # 模型分片管理
├── config/               # Worker 配置
├── fsdp_workers.py       # FSDP Worker 组合类
└── megatron_workers.py   # Megatron Worker 组合类
```

### 3.3 single_controller/ - 控制器模块

分布式训练的协调层：

```
single_controller/
├── base/
│   ├── worker.py         # Worker 基类
│   └── worker_group.py   # WorkerGroup 管理
└── ray/
    └── base.py           # Ray 集成实现
```

### 3.4 utils/ - 工具模块

丰富的工具函数：

```
utils/
├── checkpoint/           # 检查点保存/加载
├── dataset/              # 数据集处理
├── debug/                # 调试工具
├── kernel/               # 自定义 CUDA 内核
├── logger/               # 日志
├── metric/               # 指标计算
├── profiler/             # 性能分析
├── reward_score/         # 奖励评分
├── sglang/               # SGLang 集成工具
├── vllm/                 # vLLM 集成工具
├── megatron/             # Megatron 工具
├── config.py             # 配置工具
├── model.py              # 模型加载
├── tokenizer.py          # 分词器
├── fsdp_utils.py         # FSDP 工具
├── torch_functional.py   # PyTorch 操作
├── seqlen_balancing.py   # 序列长度平衡
├── ulysses.py            # Ulysses 序列并行
├── tracking.py           # 实验追踪（wandb 等）
└── memory_utils.py       # 内存管理
```

### 3.5 experimental/ - 实验性模块

前沿研究功能：

```
experimental/
├── fully_async_policy/   # 全异步策略训练
├── one_step_off_policy/  # 单步离策略学习
├── agent_loop/           # Agent 循环集成
├── reward_loop/          # 奖励循环架构
├── transfer_queue/       # 传输队列
├── dynamic_dataset/      # 动态数据集
└── vla/                  # Vision-Language Action 模型
```

---

## 4. examples/ 目录结构

丰富的训练示例：

```
examples/
├── ppo_trainer/          # PPO 训练示例（27+ 配置）
├── grpo_trainer/         # GRPO 训练示例（50+ 配置）
├── remax_trainer/        # ReMax 训练示例
├── rloo_trainer/         # RLOO 训练示例
├── reinforce_plus_plus_trainer/  # REINFORCE++ 训练
├── sft/                  # 监督微调示例
│   ├── gsm8k/            # 数学 SFT
│   ├── multiturn/        # 多轮对话
│   └── vlm/              # 视觉语言模型
├── sglang_multiturn/     # SGLang 多轮示例
├── tuning/               # 超参调优示例（0.5B-70B）
├── data_preprocess/      # 数据预处理
├── split_placement/      # 分离式 GPU 部署
└── tutorial/             # 入门教程
```

---

## 5. Hybrid-Controller 编程模型

verl 的核心设计理念是 **Hybrid-Controller 编程模型**，它将复杂的 RL 训练流程解耦为：

```
┌─────────────────────────────────────────────────────────────┐
│                      Controller (协调器)                      │
│  - 管理训练流程                                               │
│  - 调度各 Worker 执行                                         │
│  - 处理数据流转                                               │
└─────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │   Rollout   │     │    Actor    │     │   Critic    │
   │   Worker    │     │   Worker    │     │   Worker    │
   └─────────────┘     └─────────────┘     └─────────────┘
   - vLLM/SGLang       - 策略更新          - 价值估计
   - 生成序列          - FSDP/Megatron     - FSDP/Megatron
```

### 核心概念

1. **Worker**：执行具体计算任务的进程
   - Actor Worker：训练策略模型
   - Critic Worker：训练价值函数（PPO 需要）
   - Rollout Worker：使用推理引擎生成序列
   - Reward Worker：计算奖励

2. **WorkerGroup**：管理一组 Worker 的抽象
   - 通过装饰器绑定方法
   - 自动处理数据分发和收集

3. **DataProto**：统一的数据协议
   - 定义在 `protocol.py`
   - 支持自动填充、切分、合并

4. **Dispatch/Collect 模式**：
   - `dispatch_fn`：将数据分发到各 Worker
   - `collect_fn`：收集 Worker 的结果
   - 支持多种模式：`ONE_TO_ALL`、`DP_COMPUTE` 等

---

## 6. 关键依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                        verl.trainer                          │
│  - ppo/ray_trainer.py (训练主循环)                           │
│  - ppo/core_algos.py (算法实现)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        verl.workers                          │
│  - actor/ (策略模型)                                         │
│  - critic/ (价值模型)                                        │
│  - rollout/ (序列生成)                                       │
│  - reward_manager/ (奖励计算)                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     verl.single_controller                   │
│  - ray/base.py (Ray 分布式协调)                              │
│  - base/worker.py (Worker 抽象)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         verl.utils                           │
│  - 检查点、数据集、日志、指标等工具                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 快速导航

根据您想了解的内容，请参阅相应的文档：

| 想了解的内容 | 推荐文档 |
|------------|---------|
| 系统整体架构 | [02_architecture.md](02_architecture.md) |
| PPO 算法实现 | [03_ppo_deep_dive.md](03_ppo_deep_dive.md) |
| GRPO 算法实现 | [04_grpo_deep_dive.md](04_grpo_deep_dive.md) |
| 训练器工作流程 | [05_ppo_trainer.md](05_ppo_trainer.md) |
| Worker 实现细节 | [06_workers.md](06_workers.md) |
| 分布式训练 | [07_distributed_training.md](07_distributed_training.md) |
| 配置参数 | [08_configuration.md](08_configuration.md) |
| 示例代码 | [09_examples.md](09_examples.md) |

---

## 8. 总结

verl 是一个功能完备的 LLM 强化学习训练框架，具有以下特点：

- **模块化设计**：清晰的 Worker 分离，易于扩展
- **多算法支持**：13+ 种优势估计器，10+ 种策略损失函数
- **多后端支持**：FSDP、Megatron-LM 训练后端；vLLM、SGLang 推理后端
- **生产级别**：支持大规模模型（671B）和数百 GPU 集群
- **活跃社区**：由字节跳动和众多机构贡献和维护

后续文档将深入探讨各个模块的具体实现。
