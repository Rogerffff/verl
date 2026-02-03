# verl 文档索引

本索引帮助你快速查找 verl 框架的相关文档。

---

## 🚀 快速开始 (Quickstart)

| 文档 | 描述 |
|------|------|
| [install.rst](docs/start/install.rst) | 安装指南，包括 Docker 镜像、自定义环境安装、CUDA/cuDNN 配置，以及 AMD GPU (ROCM) 支持说明。 |
| [quickstart.rst](docs/start/quickstart.rst) | 快速入门教程，演示如何使用 GSM8K 数据集进行 PPO 训练，包括数据准备、模型下载和训练脚本。 |
| [multinode.rst](docs/start/multinode.rst) | 多节点训练指南，涵盖手动启动 Ray 集群、SkyPilot、Slurm 和 dstack 等多种部署方式。 |
| [ray_debug_tutorial.rst](docs/start/ray_debug_tutorial.rst) | Ray 调试教程，介绍如何使用 Ray 分布式调试器进行问题排查。 |
| [agentic_rl.rst](docs/start/agentic_rl.rst) | Agentic RL 训练指南，介绍服务器异步 rollout、多轮对话、工具调用和 LangGraph Agent 框架。 |

---

## 📖 编程指南 (Programming Guide)

| 文档 | 描述 |
|------|------|
| [hybrid_flow.rst](docs/hybrid_flow.rst) | HybridFlow 编程指南，解释 verl 的核心设计理念、控制流与计算流分离、以及 PPO 代码架构。 |
| [single_controller.rst](docs/single_controller.rst) | Single Controller 设计文档，详细介绍 WorkerGroup、ResourcePool 的实现原理和方法绑定机制。 |

---

## 📦 数据准备 (Data Preparation)

| 文档 | 描述 |
|------|------|
| [prepare_data.rst](docs/preparation/prepare_data.rst) | 数据准备指南，介绍如何将数据集转换为 parquet 格式，以及 `make_map_fn` 函数的实现方法。 |
| [reward_function.rst](docs/preparation/reward_function.rst) | 奖励函数实现指南，介绍 RewardManager 的使用方法和自定义奖励函数的实现。 |

---

## ⚙️ 配置说明 (Configuration)

| 文档 | 描述 |
|------|------|
| [config.rst](docs/examples/config.rst) | 完整配置说明，包括数据、Actor/Rollout/Ref、Critic、Reward Model、Algorithm 和 Trainer 的所有配置项。 |

---

## 📚 算法 (Algorithms)

| 文档 | 描述 |
|------|------|
| [ppo.md](docs/algo/ppo.md) | PPO (Proximal Policy Optimization) 算法说明，包括 KL 散度控制、Dual-clip PPO 等高级扩展。 |
| [grpo.md](docs/algo/grpo.md) | GRPO (Group Relative Policy Optimization) 算法说明，无需 Critic 模型的高效 RL 算法。 |
| [dapo.md](docs/algo/dapo.md) | DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) 算法，支持动态采样和 Token 级别损失。 |
| [spin.md](docs/algo/spin.md) | SPIN (Self-Play fine-tuning) 算法说明。 |
| [sppo.md](docs/algo/sppo.md) | SPPO (Self-Play Preference Optimization) 算法说明。 |
| [entropy.md](docs/algo/entropy.md) | 熵正则化相关配置和说明。 |
| [opo.md](docs/algo/opo.md) | OPO (Online Policy Optimization) 算法说明。 |
| [baseline.md](docs/algo/baseline.md) | 算法基线和性能基准，提供各模型在不同数据集上的训练结果参考。 |
| [gpg.md](docs/algo/gpg.md) | GPG (Generalized Policy Gradient) 算法说明。 |
| [collabllm.md](docs/algo/collabllm.md) | CollabLLM 协作学习方法说明。 |
| [otb.md](docs/algo/otb.md) | OTB (On-Policy Training Budget) 相关说明。 |
| [rollout_corr.md](docs/algo/rollout_corr.md) | Rollout Correction 技术文档，解决 rollout 和训练间的分布不匹配问题。 |
| [rollout_corr_math.md](docs/algo/rollout_corr_math.md) | Rollout Correction 的数学推导和理论分析。 |

---

## 👷 Workers 说明 (PPO Trainer and Workers)

| 文档 | 描述 |
|------|------|
| [ray_trainer.rst](docs/workers/ray_trainer.rst) | Ray Trainer 架构说明，介绍 PPO 训练器的工作原理和配置。 |
| [fsdp_workers.rst](docs/workers/fsdp_workers.rst) | FSDP Workers 说明，介绍基于 PyTorch FSDP 的分布式训练 Worker 实现。 |
| [megatron_workers.rst](docs/workers/megatron_workers.rst) | Megatron Workers 说明，介绍基于 Megatron-LM 的大规模模型并行训练 Worker。 |
| [sglang_worker.rst](docs/workers/sglang_worker.rst) | SGLang Worker 说明，介绍如何使用 SGLang 作为推理后端进行 rollout。 |
| [model_engine.rst](docs/workers/model_engine.rst) | Model Engine 说明，介绍模型引擎的抽象接口和实现。 |

---

## 🎯 示例 (Examples)

| 文档 | 描述 |
|------|------|
| [gsm8k_example.rst](docs/examples/gsm8k_example.rst) | GSM8K 完整示例，包括 SFT 预训练和 PPO 后训练的完整流程。 |
| [ppo_code_architecture.rst](docs/examples/ppo_code_architecture.rst) | PPO 代码架构解析，帮助理解训练代码的组织结构。 |
| [multi_modal_example.rst](docs/examples/multi_modal_example.rst) | 多模态训练示例，介绍如何进行视觉语言模型的 RL 训练。 |
| [skypilot_examples.rst](docs/examples/skypilot_examples.rst) | SkyPilot 云端部署示例，介绍如何在云平台上运行训练任务。 |
| [sandbox_fusion_example.rst](docs/examples/sandbox_fusion_example.rst) | Sandbox Fusion 示例，介绍代码执行环境的集成。 |

---

## 🔧 性能调优 (Performance Tuning)

| 文档 | 描述 |
|------|------|
| [best_practices.rst](docs/perf/best_practices.rst) | 最佳实践指南，介绍提高训练效率的各种技巧和配置建议。 |
| [perf_tuning.rst](docs/perf/perf_tuning.rst) | 性能调优指南，详细说明如何优化训练速度和资源利用率。 |
| [device_tuning.rst](docs/perf/device_tuning.rst) | 设备调优指南，针对不同 GPU 型号的优化建议。 |
| [dpsk.md](docs/perf/dpsk.md) | DeepSeek 模型相关的性能优化说明。 |
| [verl_profiler_system.md](docs/perf/verl_profiler_system.md) | verl Profiler 系统说明，介绍如何使用内置 profiler 分析性能瓶颈。 |
| [nsight_profiling.md](docs/perf/nsight_profiling.md) | Nsight 性能分析指南，介绍如何使用 NVIDIA Nsight 工具进行深度性能分析。 |
| [README_vllm0.8.md](docs/README_vllm0.8.md) | vLLM 0.8 版本的使用说明和兼容性说明。 |

---

## 🚧 高级功能 (Advanced Features)

| 文档 | 描述 |
|------|------|
| [checkpoint.rst](docs/advance/checkpoint.rst) | Checkpoint 系统说明，介绍容错训练的检查点保存和恢复机制。 |
| [rope.rst](docs/advance/rope.rst) | RoPE (Rotary Position Embedding) 相关配置和扩展说明。 |
| [attention_implementation.rst](docs/advance/attention_implementation.rst) | 注意力实现配置，介绍不同注意力机制的选择和配置。 |
| [ppo_lora.rst](docs/advance/ppo_lora.rst) | PPO + LoRA 训练指南，介绍如何使用 LoRA 进行高效参数微调。 |
| [placement.rst](docs/advance/placement.rst) | 模型放置策略说明，介绍如何配置模型在不同 GPU 上的放置。 |
| [dpo_extension.rst](docs/advance/dpo_extension.rst) | DPO (Direct Preference Optimization) 扩展说明。 |
| [fsdp_extension.rst](docs/advance/fsdp_extension.rst) | FSDP 扩展指南，介绍如何添加新模型的 FSDP 支持。 |
| [megatron_extension.rst](docs/advance/megatron_extension.rst) | Megatron 扩展指南，介绍如何添加新模型的 Megatron 支持。 |
| [rollout_trace.rst](docs/advance/rollout_trace.rst) | Rollout Trace 功能说明，介绍如何追踪和调试 rollout 过程。 |
| [rollout_skip.rst](docs/advance/rollout_skip.rst) | Rollout Skip 功能说明，介绍如何跳过不必要的 rollout 步骤。 |
| [one_step_off.md](docs/advance/one_step_off.md) | One-Step Off-Policy 相关说明。 |
| [agent_loop.rst](docs/advance/agent_loop.rst) | Agent Loop 内部设计文档，详细介绍异步 rollout 系统的架构。 |
| [reward_loop.rst](docs/advance/reward_loop.rst) | Reward Loop 说明，介绍奖励计算循环的实现。 |
| [fully_async.md](docs/advance/fully_async.md) | 完全异步训练模式说明，介绍如何实现全异步的 RL 训练。 |
| [fp8.md](docs/advance/fp8.md) | FP8 训练支持说明，介绍如何使用 FP8 精度进行训练。 |
| [async-on-policy-distill.md](docs/advance/async-on-policy-distill.md) | 异步 On-Policy 蒸馏说明。 |
| [grafana_prometheus.md](docs/advance/grafana_prometheus.md) | Grafana + Prometheus 监控集成说明，介绍如何配置训练监控。 |

---

## 🔄 多轮对话 (Multi-turn & SGLang)

| 文档 | 描述 |
|------|------|
| [multiturn.rst](docs/sglang_multiturn/multiturn.rst) | 多轮对话支持说明，介绍如何进行多轮对话场景的 RL 训练。 |
| [interaction_system.rst](docs/sglang_multiturn/interaction_system.rst) | 交互系统说明，介绍 Agent 与环境交互的系统设计。 |
| [sandbox_fusion.rst](docs/sglang_multiturn/sandbox_fusion.rst) | Sandbox Fusion 开发说明，介绍代码执行沙箱的集成。 |
| [search_tool_example.rst](docs/sglang_multiturn/search_tool_example.rst) | 搜索工具示例，介绍如何集成搜索工具进行 Agent 训练。 |

---

## 📊 数据传输 (Data Transfer)

| 文档 | 描述 |
|------|------|
| [transfer_queue.md](docs/data/transfer_queue.md) | Transfer Queue 说明，介绍数据传输队列的实现和使用（英文版）。 |
| [transfer_queue_zh.md](docs/data/transfer_queue_zh.md) | Transfer Queue 说明，介绍数据传输队列的实现和使用（中文版）。 |

---

## 🖥️ 硬件支持 (Hardware Support)

### AMD GPU (ROCm)

| 文档 | 描述 |
|------|------|
| [amd_build_dockerfile_page.rst](docs/amd_tutorial/amd_build_dockerfile_page.rst) | AMD GPU Docker 构建指南，详细介绍如何为 MI300 等 AMD GPU 构建 Docker 镜像。 |
| [amd_vllm_page.rst](docs/amd_tutorial/amd_vllm_page.rst) | AMD GPU + vLLM 使用说明。 |

### 华为昇腾 (Ascend NPU)

| 文档 | 描述 |
|------|------|
| [ascend_quick_start.rst](docs/ascend_tutorial/ascend_quick_start.rst) | 昇腾 NPU 快速开始指南。 |
| [ascend_consistency.rst](docs/ascend_tutorial/ascend_consistency.rst) | 昇腾 NPU 一致性说明。 |
| [ascend_profiling_zh.rst](docs/ascend_tutorial/ascend_profiling_zh.rst) | 昇腾 NPU 性能分析指南（中文）。 |
| [ascend_profiling_en.rst](docs/ascend_tutorial/ascend_profiling_en.rst) | 昇腾 NPU 性能分析指南（英文）。 |
| [dockerfile_build_guidance.rst](docs/ascend_tutorial/dockerfile_build_guidance.rst) | 昇腾 NPU Docker 镜像构建指南。 |
| [ascend_sglang_quick_start.rst](docs/ascend_tutorial/ascend_sglang_quick_start.rst) | 昇腾 NPU + SGLang 快速开始指南。 |

---

## 📚 API 参考 (API Reference)

| 文档 | 描述 |
|------|------|
| [data.rst](docs/api/data.rst) | 数据相关 API 文档。 |
| [single_controller.rst](docs/api/single_controller.rst) | Single Controller API 文档。 |
| [trainer.rst](docs/api/trainer.rst) | Trainer API 文档。 |
| [utils.rst](docs/api/utils.rst) | 工具函数 API 文档。 |

---

## 📝 博客 (Blog)

| 文档 | 描述 |
|------|------|
| [v0.7.md](docs/blog/v0.7.md) | verl v0.7 版本发布说明，介绍新功能和改进。 |

---

## ❓ 常见问题 (FAQ)

| 文档 | 描述 |
|------|------|
| [faq.rst](docs/faq/faq.rst) | 常见问题解答，包括 Ray 相关问题、分布式训练、安装问题、内存问题等。 |

---

## 🔧 其他文件

| 文档 | 描述 |
|------|------|
| [README.md](docs/README.md) | 文档构建说明，介绍如何本地构建和预览文档。 |
| [README_vllm0.7.md](docs/README_vllm0.7.md) | vLLM 0.7 版本的使用说明。 |
| [index.rst](docs/index.rst) | 文档主页，Sphinx 文档的入口页面。 |
