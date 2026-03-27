# Phase 3: GRPO 训练 — 前置文档

> 在开始 GRPO 实验之前，你需要知道的一切。
>
> 最后更新：2026-03-19

---

## 目录

1. [项目总览与当前决策](#1-项目总览与当前决策)
2. [Phase 0 基线评测结果](#2-phase-0-基线评测结果)
3. [数据文件完整说明](#3-数据文件完整说明)
4. [评测脚本设计](#4-评测脚本设计)
5. [奖励函数设计](#5-奖励函数设计)
6. [GRPO 超参最小集](#6-grpo-超参最小集)
7. [实验设计与实验矩阵](#7-实验设计与实验矩阵)
8. [关键文件导航](#8-关键文件导航)

---

## 1. 项目总览与当前决策

### 1.1 五阶段训练流水线

```
Phase 0        Phase 1       Phase 2        Phase 3         Phase 4
Baseline  →     SFT     →  DPO (可选)  →    GRPO      →  多轮修复 (可选)
  ✅            ⚠️ 搁置       ⏭️ 跳过      🔜 当前阶段        📋 待定
```

### 1.2 为什么跳过 SFT，直接进入 GRPO

**Phase 1 SFT 结论**：

- SFT 训练基础设施已全部就绪（脚本、数据、评测流水线）
- 但由于选取的 SFT 数据（BEE CodeContests 子集，2,220 题）质量和分布与目标任务不匹配，**训练后模型效果不如 base model**
- SFT 的初衷是"降低语法/运行时错误"，但 Phase 0 数据显示 base model 的 syntax_error_rate 本来就极低（<1%），主要问题是 wrong_answer（57-64%），这恰恰是 RL 擅长优化的

**直接 GRPO 的依据**：

- CodeContests 上 **pass_ratio_mean = 14-26%**，远高于 0，说明模型能部分解题但不够好 — 这是 dense reward 的理想工作区间
- pass_ratio_p90 高达 0.46-0.96，说明部分题目模型已经接近正确，RL 有明确的优化空间
- 错误以 wrong_answer 为主（70%+），模型代码能执行但逻辑不对 → 适合用 reward signal 优化

### 1.3 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 训练框架 | **verl** | 分布式 GRPO/PPO 训练（Ray + FSDP/Megatron） |
| 推理引擎 | **vLLM** | Rollout 阶段高性能推理 |
| 代码评测 | **SandboxFusion** | 安全沙盒执行代码，计算 pass_ratio 和 accepted |
| 基础模型 | **Qwen2.5-Coder-7B-Instruct** | 起始 checkpoint（不经过 SFT，直接 GRPO） |
| 实验追踪 | **WandB** | 训练曲线、评测指标记录 |

### 1.4 仓库结构

| 路径 | 分支 | 说明 |
|------|------|------|
| `verl/verl` | main | verl 框架主分支 |
| `verl/verl-sft` | feature/sft-development | SFT 开发分支（保留 Phase 0 outputs） |
| `verl/verl-grpo` | feature/grpo-development | **GRPO 开发分支（当前工作分支）** |

子模块：
- `recipe/` → verl-recipe 训练配方
- `SandboxFusion/` → 代码执行沙盒（https://github.com/Rogerffff/sandbox.git）

---

## 2. Phase 0 基线评测结果

> 数据来源：`outputs/phase0_20260206_154702/`（仅存在于 verl-sft worktree，未提交到 git）
>
> 模型：Qwen2.5-Coder-7B-Instruct | 评测协议：EVAL@1 (T=0.0, greedy)

### 2.1 质量指标

| 数据集 | 题目数 | accepted@1 | pass_ratio_mean | pass_ratio_p50 | pass_ratio_p90 | exec_success_rate |
|--------|--------|-----------|-----------------|----------------|----------------|-------------------|
| HumanEval | 164 | **87.2%** | 87.2% | 1.0 | 1.0 | 97.6% |
| MBPP_reg | 200 | **58.5%** | 58.5% | 1.0 | 1.0 | 99.0% |
| CodeContests_valid_big | 500 | **9.2%** | **25.7%** | 0.08 | 0.96 | 67.0% |
| CodeContests_valid | 117 | **1.7%** | **14.3%** | 0.0 | 0.468 | 65.8% |
| CodeContests_test | 165 | **3.0%** | **13.8%** | 0.0 | 0.46 | 60.0% |

### 2.2 错误分布

| 数据集 | syntax_error | runtime_error | timeout | wrong_answer | api_error |
|--------|-------------|---------------|---------|--------------|-----------|
| HumanEval | 0% | 2.4% | 0% | 10.4% | 0% |
| MBPP_reg | 0% | 1.0% | 0% | 40.5% | 0% |
| CodeContests_valid_big | 0.6% | **26.2%** | **6.2%** | **57.8%** | 0% |
| CodeContests_valid | 0.9% | **23.9%** | **9.4%** | **64.1%** | 0% |
| CodeContests_test | 0.6% | **32.7%** | **6.7%** | **57.0%** | 0% |

**关键观察**：
- syntax_error 极低（<1%）→ SFT 对此几乎无改善空间
- wrong_answer 是主导错误（57-64%）→ 这是 RL 的核心优化目标
- runtime_error 23-33% → 可能通过学习更好的边界条件处理来改善
- timeout 6-9% → 可能通过学习更高效的算法来改善

### 2.3 成本指标

| 数据集 | avg_gen_tokens | avg_judge_time | throughput | cost_per_solved_tokens |
|--------|---------------|----------------|------------|----------------------|
| HumanEval | 124.2 | 0.048s | 5.63 prob/s | 142.5 |
| MBPP_reg | 53.8 | 0.059s | 8.35 prob/s | 91.9 |
| CodeContests_valid_big | 238.8 | 4.75s | 0.20 prob/s | 2,596.0 |
| CodeContests_valid | 264.0 | 7.69s | 0.13 prob/s | 15,445.5 |
| CodeContests_test | 273.6 | 5.92s | 0.16 prob/s | 9,028.2 |

### 2.4 奖励密度分析（对 GRPO 的意义）

**为什么这些数字说明可以直接 GRPO？**

1. **pass_ratio_mean 14-26%**：模型不是完全乱猜（那会接近 0%），也不是已经很好（那 RL 没什么空间）。这是 RL 的"甜区"——有足够的正向 reward signal 来引导策略梯度。

2. **pass_ratio_p90 = 0.46-0.96**：说明分布的上尾有相当多"接近正确"的生成，GRPO 的 group-relative advantage 可以有效区分好坏样本。

3. **exec_success_rate 60-67%**：大部分代码能执行到出结果（而不是崩溃），reward function 能产生有意义的 pass_ratio 值。

4. **与 RLTF 论文的 reward 设计对齐**：`-0.3 + 1.3 * pass_ratio` 中，正反馈阈值是 pass_ratio > 23.1%。当前 mean = 14-26%，正好处于"部分样本能拿到正反馈、部分拿不到"的区间，advantage 有区分度。

### 2.5 运行配置（复现基线用）

```bash
python src/phase0_eval.py \
    --mode simple \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --vllm_url http://localhost:8001 \
    --sandbox_url http://localhost:8080 \
    --manifest_dir data/manifests \
    --temperature 0.0 \
    --max_tokens 2048 \
    --run_timeout 30 \
    --max_concurrent 32 \
    --batch_size 50 \
    --datasets humaneval mbpp_reg codecontests_valid_big codecontests_valid codecontests_test
```

---

## 3. 数据文件完整说明

> 这是最容易混乱的部分。下面用流程图和精确数字说清楚每个文件的来龙去脉。

### 3.1 数据处理全流程

```
原始 CodeContests 数据
│
│  Step 1: 从 SandboxFusion 下载
│  ├─ codecontests_train_raw.jsonl   (13,328 条)   785 MB
│  ├─ codecontests_valid_raw.jsonl   (117 条)       12 MB
│  └─ codecontests_test_raw.jsonl    (165 条)       5.8 MB
│
│  Step 2: 去重 (intra-split deduplication, 基于 prompt_sha256)
│  删除 1,043 条精确重复 → 保留 12,285 条
│  ├─ codecontests_train_manifest.jsonl    (12,285 条)   2.9 MB
│  └─ codecontests_train_duplicates_intrasplit.jsonl  (1,043 条，记录被删除的)
│
│  Step 3: 抽取 valid_big (seed=42, 仅 Codeforces 题目, 6776 候选中抽 500)
│  ├─ codecontests_valid_big_manifest.jsonl        (500 条)    120 KB   ← 验证集
│  ├─ codecontests_valid_big_raw.jsonl             (500 条)    31 MB    ← 验证集原始数据
│  ├─ codecontests_valid_big_split_meta.json       (审计元数据)
│  │
│  └─ codecontests_train_wo_valid_big_manifest.jsonl (11,785 条) 2.8 MB  ← ★ GRPO 训练集
│     codecontests_train_wo_valid_big_raw.jsonl      (12,828 条) 754 MB  ← ★ GRPO 训练集原始
│
│  另外，独立下载的外部基准（与 CodeContests 不相关，零重叠）：
│  ├─ humaneval_raw.jsonl / humaneval_manifest.jsonl       (164 条)
│  └─ mbpp_reg_raw.jsonl / mbpp_reg_manifest.jsonl         (200 条)

SFT 数据（完全独立的处理链，与上面无关）：
│
│  Step A: 从 BEE 数据集下载 (95,629 条原始记录)
│  Step B: 去重 — 每道题保留最短解 → 2,220 条
│  Step C: 与 CodeContests test cases 匹配
│  Step D: 80/20 分割 (seed=42) → sft_train.parquet (2,020) + sft_val.parquet (200)
│
│  这些数据已不再使用（SFT 效果不佳，已搁置）
```

### 3.2 GRPO 训练必须使用的文件

| 用途 | 文件（相对 coding_model_project/） | 题目数 |
|------|------|--------|
| **训练集** | `data/manifests/codecontests_train_wo_valid_big_manifest.jsonl` | 11,785 |
| 训练集原始数据 | `data/raw/codecontests_train_wo_valid_big_raw.jsonl` | 12,828 |
| **高频验证集** | `data/manifests/codecontests_valid_big_manifest.jsonl` | 500 |
| 高频验证原始 | `data/raw/codecontests_valid_big_raw.jsonl` | 500 |
| 回归验证 | `data/manifests/mbpp_reg_manifest.jsonl` | 200 |
| 辅助验证 | `data/manifests/codecontests_valid_manifest.jsonl` | 117 |
| **最终测试**（仅阶段结束） | `data/manifests/codecontests_test_manifest.jsonl` | 165 |
| **最终测试**（仅阶段结束） | `data/manifests/humaneval_manifest.jsonl` | 164 |

> **注意 manifest 和 raw 的行数差异**：manifest 是去重后的（12,285 → 抽走 500 = 11,785），raw 文件是基于 manifest 从原始 13,328 条中筛选出的（13,328 - 500 = 12,828），两者题目 ID 集合相同，raw 保留了完整的题目描述和 test cases。

### 3.3 为什么不能用 codecontests_train_manifest.jsonl

**千万不要用 `codecontests_train_manifest.jsonl` (12,285 条) 做 GRPO 训练！**

原因：这个文件包含 valid_big 的 500 题。如果用它训练，然后在 valid_big 上评测，就造成了**数据泄漏** — 训练集和验证集有 500 题重叠，评测指标无意义。

正确的训练文件是 `codecontests_train_wo_valid_big_manifest.jsonl` (11,785 条)，其中 `wo` = without。

### 3.4 SFT 数据 vs GRPO 数据

| 维度 | SFT 数据 | GRPO 数据 |
|------|----------|-----------|
| 来源 | BEE 数据集（Codeforces/AtCoder/Aizu 的人类解法） | CodeContests_train（竞赛题目 + test cases） |
| 格式 | parquet (OpenAI message 格式: system/user/assistant) | JSONL manifest + raw (problem_id, prompt, test_cases) |
| 题目数 | 2,220（去重后） | 11,785（去重 + 去 valid_big 后） |
| 用途 | 监督学习（模仿人类解法） | 在线 RL（生成代码 → 执行 → reward） |
| 当前状态 | **已搁置**（效果不佳） | **即将使用** |
| 文件路径 | `phase_1_ SFT/data/sft_train.parquet` | `data/manifests/codecontests_train_wo_valid_big_manifest.jsonl` |

### 3.5 数据集角色分配

| 数据集 | 角色 | GRPO 中的用途 | 使用频率 | 禁忌 |
|--------|------|-------------|----------|------|
| CodeContests_train_wo_valid_big | **Train** | Rollout 生成 + Reward 计算 | 每步 | - |
| CodeContests_valid_big | **Val (主)** | 高频评测，选超参，早停 | 每 50-100 步 | 不能训练 |
| CodeContests_valid | **Val (辅)** | 辅助验证 | 可选 | 不能训练 |
| MBPP_reg | **Val (回归)** | 监控模型不退化 | 每 50-100 步 | 不能训练 |
| CodeContests_test | **Test** | 阶段结束最终评测 | 仅阶段结束 | 不能训练，不能选超参 |
| HumanEval | **Test** | 行业对标 | 仅阶段结束 | 不能训练，不能选超参 |

### 3.6 完整文件清单

#### Manifest 文件 (`data/manifests/`)

| 文件名 | 行数 | 大小 | 用途 |
|--------|------|------|------|
| `codecontests_train_manifest.jsonl` | 12,285 | 2.9 MB | 原始去重训练集（**不要用于 GRPO**） |
| `codecontests_train_wo_valid_big_manifest.jsonl` | 11,785 | 2.8 MB | **GRPO 训练集** |
| `codecontests_valid_big_manifest.jsonl` | 500 | 120 KB | GRPO 高频验证集 |
| `codecontests_valid_big_split_meta.json` | - | 1.8 KB | valid_big 抽取审计元数据 |
| `codecontests_valid_manifest.jsonl` | 117 | 28 KB | 辅助验证集 |
| `codecontests_test_manifest.jsonl` | 165 | 39 KB | 最终测试集 |
| `codecontests_train_duplicates_intrasplit.jsonl` | 1,043 | 250 KB | 去重记录（仅供审计） |
| `humaneval_manifest.jsonl` | 164 | 43 KB | HumanEval 测试集 |
| `mbpp_reg_manifest.jsonl` | 200 | 39 KB | MBPP 回归测试集 |

#### Raw 数据文件 (`data/raw/`)

| 文件名 | 行数 | 大小 | 用途 |
|--------|------|------|------|
| `codecontests_train_raw.jsonl` | 13,328 | 785 MB | 原始训练数据（去重前） |
| `codecontests_train_wo_valid_big_raw.jsonl` | 12,828 | 754 MB | **GRPO 训练原始数据** |
| `codecontests_valid_big_raw.jsonl` | 500 | 31 MB | 验证集原始数据 |
| `codecontests_valid_raw.jsonl` | 117 | 12 MB | 辅助验证原始数据 |
| `codecontests_test_raw.jsonl` | 165 | 5.8 MB | 测试集原始数据 |
| `humaneval_raw.jsonl` | 164 | 309 KB | HumanEval 原始数据 |
| `mbpp_reg_raw.jsonl` | 200 | 177 KB | MBPP 原始数据 |
| `dataset_samples.jsonl` | 5 | 8.8 KB | 数据格式示例 |

#### Raw 数据条目格式

每行是一个 JSON 对象：

```json
{
  "problem_id": "Codeforces/1000/A",
  "prompt": "题目描述全文...",
  "canonical_prompt": "标准化后的题目描述...",
  "prompt_sha256": "bdee1721...",
  "test_cases": {
    "type": "codecontests",
    "tests": [
      {"input": "3\n1 2 3\n", "output": "6\n"},
      ...
    ]
  }
}
```

`test_cases` 是 GRPO 计算 reward 的关键 — SandboxFusion 用这些 test cases 判题，返回 pass_ratio。

#### SFT 数据文件（已搁置，仅供参考）

| 路径 | 大小 | 说明 |
|------|------|------|
| `phase_1_ SFT/data/sft_train.parquet` | 2.1 MB | 2,020 条训练数据 |
| `phase_1_ SFT/data/sft_val.parquet` | 244 KB | 200 条验证数据 |
| `phase_1_ SFT/data/split_mapping.json` | 5.7 KB | 分割元数据 (seed=42, val_size=200) |

### 3.7 数据完整性保障

- **去重审计**：`reports/data_audit_report.md` 记录了所有去重决策
- **跨集零重叠**：train ∩ valid = 0, train ∩ test = 0, valid ∩ test = 0, train ∩ HumanEval = 0
- **valid_big 可复现**：`codecontests_valid_big_split_meta.json` 记录了 seed=42, prefix="Codeforces/", 从 6776 候选中抽 500
- **数据治理脚本**：`src/data_governance.py` 可以从零重新执行全部去重流程

---

## 4. 评测脚本设计

### 4.1 架构概览

`src/phase0_eval.py`（2,600+ 行）是核心评测脚本，分 6 层：

```
第 1 层：配置与常量
  ├─ EvalConfig (数据类)
  ├─ SYSTEM_PROMPT (统一系统提示)
  └─ PROMPT_TEMPLATES (各数据集的 User Prompt 模板)

第 2 层：服务器管理
  ├─ start_rollout_servers()    # 启动 vLLM (verl mode)
  └─ fetch_openai_models()      # 查询服务端模型 ID

第 3 层：代码生成
  ├─ generate_code()            # 单条异步生成
  └─ batch_generate()           # 批量并发 + Round-Robin 负载均衡

第 4 层：代码评测
  ├─ evaluate_with_submit_api() # SandboxFusion submit() API
  └─ evaluate_with_run_code()   # SandboxFusion run_code() API

第 5 层：数据加载
  └─ load_prompts()             # 从 manifest 或 SandboxFusion 直接加载

第 6 层：主评测流程
  ├─ evaluate_dataset()         # 评测单个数据集
  └─ run_evaluation()           # 完整评测闭环（多数据集）
```

### 4.2 Prompt 构造

**System Prompt**（所有数据集共用）：

```
You are an expert Python programmer.

Output rules:
1. Output Python code only.
2. Include necessary imports only if needed.
3. Wrap the entire code in <code> and </code>.
4. Do not write anything outside the <code> tags.
5. Follow dataset-specific constraints given by the user prompt.
```

**User Prompt 模板**（按数据集区分）：

- **HumanEval**：要求补全函数，禁止 stdin/stdout/main guard
- **MBPP_reg**：要求实现指定函数名，包含 `{entry_point}` 和 `{example_call}` 占位符
- **CodeContests_***（train/valid/valid_big/test 共用模板）：
  ```
  Solve the following competitive programming problem in Python.

  Rules:
  - Read from stdin and write to stdout.
  - Your program MUST produce output when executed.
  - Use fast I/O if needed (sys.stdin.buffer).
  - Do NOT print anything except the required output.

  {prompt}

  Output ONLY:
  <code>
  # python code
  </code>
  ```

**代码提取**：优先解析 `<code>...</code>` 标签，回退到 markdown 代码块，最后使用原始输出。

### 4.3 SandboxFusion 评测方式

两种 API 可选：

| 方式 | API | 适用场景 | 说明 |
|------|-----|----------|------|
| submit | `submit_safe(SubmitRequest)` | 有内置 test cases 的数据集 | 直接使用 SandboxFusion 服务端的题库和测试用例 |
| run_code | `run_code(RunCodeRequest)` | 外部测试用例 | 手动传入 test cases，逐个执行和比较 |

**submit API 流程**：
1. 发送：dataset_name (如 "code_contests") + problem_id + 代码 + 配置 (timeout, memory_limit)
2. 返回：accepted (bool) + 每个 test case 的结果 + pass_ratio

### 4.4 错误分类逻辑

按优先级分类（从严到宽）：

```
1. empty_output     → 没有生成代码
2. syntax_error     → 编译/语法错误 (SandboxFusion 返回 -4)
3. runtime_error    → 运行时崩溃 (返回 -2)
4. timeout          → 超时 (返回 -3)
5. wrong_answer     → 能执行但输出不对 (返回 False, pass_ratio < 1.0)
6. success          → 全部测试通过 (accepted = True)
```

### 4.5 EVAL@1 协议常量

| 参数 | 值 | 说明 |
|------|-----|------|
| temperature | 0.0 | Greedy decoding（确定性） |
| top_p | 1.0 | 不启用 nucleus sampling |
| max_new_tokens | 2048 | 最大生成 token 数 |
| run_timeout | 30s | SandboxFusion 执行超时 |
| memory_limit_MB | 1024 | 内存限制 |

**这些常量在所有阶段的评测中必须保持一致**，否则指标不可比较。

### 4.6 辅助脚本

| 脚本 | 说明 |
|------|------|
| `src/eval_config.py` | 集中管理评测常量、数据集配置、并发配置 |
| `src/utils/metrics.py` | MetricsCollector 指标收集器（EvalResult → DatasetMetrics） |
| `src/utils/qa_logger.py` | 问答日志分层抽样保存 |
| `src/badcase_analyze.py` | 错误案例分析工具（错误分布、代码提取成功率、异常类型统计） |
| `scripts/run_phase0.sh` | Phase 0 一键启动脚本（含服务健康检查） |
| `phase_1_ SFT/phase1_eval.py` | Phase 1 评测脚本（支持检查点评测、三级 Tier、最佳检查点选择） |

---

## 5. 奖励函数设计

> 详细设计文档：`experiment_design/reward_design.md`
>
> 出处：RLTF (Reinforcement Learning from Unit Test Feedback) + VeRPO + DAPO + DeepCoder

### 5.1 主线方案：Dense-linear (Piecewise)

```python
if truncated_by_max_tokens:
    reward = MASK(advantage=0)     # 截断 → 不学习（DAPO/DeepCoder 实践）
elif syntax_error:
    reward = -1.0                   # 语法错误 → 最重惩罚
elif runtime_error or timeout:
    reward = -0.6                   # 运行时/超时 → 中等惩罚
else:
    pass_ratio = passed_tests / total_tests
    reward = -0.3 + 1.3 * pass_ratio   # RLTF Adaptive Feedback
reward = clip(reward, -1, 1)
```

**关键性质**：
- `pass_ratio = 0` → reward = -0.3（跑了但全错，轻惩罚）
- `pass_ratio = 0.231` → reward = 0（正反馈阈值：需过 ~23% 测试才给正分）
- `pass_ratio = 1.0` → reward = 1.0（全部通过）
- 正反馈阈值 23.1% 能抑制"乱猜也能拿正分"的噪声学习

### 5.2 四种消融方案

| 方案 | 定义 | 信号 | 6/10 测试通过时的 reward |
|------|------|------|-------------------------|
| **(A) Sparse** | `r = 1[accepted]` | 离散 {0,1} | 0（非 AC 无奖励） |
| **(B) Dense-linear** (主线) | Piecewise + pass_ratio | 连续 [-1,1] | 0.48 |
| **(C) Dense-weighted** | VeRPO：按测试难度加权 pass_ratio | 连续 [-1,1] | 取决于权重分配 |
| **(D) Dense + Outcome Anchor** | VeRPO：dense + AC bonus | 连续 [-1,1] | 0.3（加 anchor headroom） |

**核心消融**：(A) Sparse vs (B) Dense-linear 是必做的，对应实验设计中的"Dense vs Sparse reward signal"对比。

### 5.3 截断/超长处理

- 被 max_tokens 截断的样本，代码不完整，会表现为语法错误
- **不应该**当成正常语法错误来学习（会误导策略）
- 推荐：将截断样本的 advantage 置 0（mask 掉），或使用 DAPO 的 soft punish/filter
- 依据：RLTF、DeepCoder、DAPO 均有此实践

---

## 6. GRPO 超参最小集

> 详细文档：`experiment_design/grpo_minimal_hparams.md`

### 6.1 必须固定的 Run 元信息

| 字段 | GRPO 训练的值 |
|------|-------------|
| base_model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| init_checkpoint | base_model（不经过 SFT） |
| ref_model | init_checkpoint 的 frozen copy |
| dataset_manifest | `codecontests_train_wo_valid_big_manifest.jsonl` (v20260206) |
| code_version | git commit hash |
| seed | 至少 2 个 seeds |

### 6.2 五类关键超参

#### Rollout / Sampling

| 参数 | verl 配置键 | 建议值 | 说明 |
|------|------------|--------|------|
| group_size | `actor_rollout_ref.rollout.n` | 5 | 每个 prompt 采样条数 |
| temperature | rollout 配置 | 0.7 | Rollout 采样温度（评测仍用 0.0） |
| top_p | rollout 配置 | 0.95 | Nucleus sampling |
| max_new_tokens | `data.max_response_length` | 2048 | 最大生成长度 |

#### Reward

| 参数 | 说明 |
|------|------|
| reward_type | `dense` (pass_ratio) 或 `sparse` (1[accepted]) — 消融变量 |
| reward_clip | [-1, 1] |

#### Advantage / GRPO

| 参数 | verl 配置键 | 建议值 |
|------|------------|--------|
| adv_estimator | `algorithm.adv_estimator` | `grpo` |
| norm_adv_by_std | `algorithm.norm_adv_by_std_in_grpo` | `True` |

#### KL / Reference

| 参数 | verl 配置键 | 建议值 |
|------|------------|--------|
| use_kl_loss | `actor_rollout_ref.actor.use_kl_loss` | `True` |
| kl_loss_coef | `actor_rollout_ref.actor.kl_loss_coef` | `0.001` |
| kl_loss_type | `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` |

#### Optimizer / PPO

| 参数 | verl 配置键 | 建议值 |
|------|------------|--------|
| lr | `actor_rollout_ref.actor.optim.lr` | `1e-6` |
| clip_ratio | `actor_rollout_ref.actor.clip_ratio` | `0.2` |
| ppo_mini_batch_size | `actor_rollout_ref.actor.ppo_mini_batch_size` | `256` |
| ppo_micro_batch_size | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 16-40 |
| entropy_coeff | `actor_rollout_ref.actor.entropy_coeff` | `0` |

### 6.3 verl GRPO 配置示例

参考 `examples/grpo_trainer/run_qwen2-7b.sh`：

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-7B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=rlvr_coding_model \
    trainer.experiment_name=grpo_dense_seed0
```

### 6.4 GRPO vs PPO 的关键区别

| 维度 | PPO | GRPO |
|------|-----|------|
| Critic | 需要单独训练的 Value Network | **不需要** |
| Advantage | GAE (基于 V(s)) | Group-relative (基于组内 reward 均值) |
| KL 约束 | KL 加在 reward 里 | **KL 加在 loss 里** |
| 采样 | 通常 n=1 | n>1 (如 n=5)，每个 prompt 多条采样 |
| 配置 | 需要 critic 相关参数 | 不需要 critic 参数 |

### 6.5 三个必答复现性问题

1. **rollout_n / group_size 敏感性**：group_size 变了结果会不会变？至少做一次 sanity check。
2. **ref_model 设置**：谁是 ref_model？是否冻结？KL 的计算口径？
3. **三种评测口径一致性**：EVAL@1 / EVAL@k / EVAL@budget 下结论是否一致？

---

## 7. 实验设计与实验矩阵

> 详细设计：`experiment_design/final_experiment_design.md`（Phase 3 章节）

### 7.1 核心消融

**必做**：Dense reward vs Sparse reward（2 种 × 2 seeds = 4 runs 最少）

| 实验 | reward_type | 预期 |
|------|------------|------|
| GRPO-Dense-seed0 | pass_ratio (连续) | 收敛更快，pass_ratio ↑ |
| GRPO-Dense-seed1 | pass_ratio (连续) | 稳定性验证 |
| GRPO-Sparse-seed0 | 1[accepted] (离散) | 信号稀疏，收敛慢 |
| GRPO-Sparse-seed1 | 1[accepted] (离散) | 稳定性验证 |

### 7.2 训练监控指标（每步记录到 WandB）

| 类别 | 指标 |
|------|------|
| Reward | score_mean, score_std, score_max, score_min, advantages_mean |
| Actor | loss, grad_norm, clip_ratio |
| KL | kl_loss, kl_mean, kl_max |
| Response | length_mean, length_max |

### 7.3 评测频率

| 评测 | 数据集 | 频率 | 预计耗时 |
|------|--------|------|----------|
| Tier 1 | codecontests_valid (117) + mbpp_reg (200) | 每 50-100 步 | ~10 min |
| Tier 2 | + codecontests_valid_big (500) | 每 200-500 步 | ~20 min |
| Tier 3 | + codecontests_test + humaneval | 仅阶段结束 | ~35 min |

### 7.4 检查点策略

- 至少保存 **5 个**检查点
- **Best accepted@1**：主要最佳模型
- **Best pass_ratio_mean**：备选最佳
- 每 100 步：恢复点
- 两个 seed 都要保存

### 7.5 预期结果（Phase 3 目标）

根据实验设计文档，Phase 3 的目标是：

| 指标 | Phase 0 基线 | Phase 3 目标 |
|------|-------------|-------------|
| CodeContests_test accepted@1 | 3.0% | **+3~10pp** (6-13%) |
| pass_ratio_mean | 13.8% | ↑ 显著提升 |
| HumanEval pass@1 | 87.2% | ≈ 保持（±2pp，不退化） |
| MBPP_reg pass@1 | 58.5% | ≈ 保持（±2pp） |

### 7.6 防 reward hacking 要点

> 详细文档：`experiment_design/guardrails.md`

- **空输出/非代码**：直接 judge_score=0，不进沙盒
- **超长输出**：截断并标记 `truncated=True`，advantage 置 0
- **重复代码**：hash cache，复用上次结果
- **沙盒约束**：禁止网络、限制文件系统、限制 syscalls
- **监控面板**：输出长度分布、short-circuit 比例、cache 命中率、timeout 趋势

---

## 8. 关键文件导航

### 评测代码

| 文件（相对 `coding_model_project/`） | 说明 |
|------|------|
| `src/phase0_eval.py` | 核心评测脚本（2,600+ 行） |
| `src/eval_config.py` | 评测常量和配置（280+ 行） |
| `src/utils/metrics.py` | 指标收集器 |
| `src/utils/qa_logger.py` | 问答日志 |
| `src/badcase_analyze.py` | 错误案例分析 |
| `src/data_governance.py` | 数据去重/治理（45K+ 行） |
| `src/build_valid_big_split.py` | valid_big 抽取脚本 |
| `src/validate_sft_data.py` | SFT 数据验证 |
| `scripts/run_phase0.sh` | Phase 0 启动脚本 |

### 实验设计文档

| 文件（相对 `coding_model_project/`） | 说明 |
|------|------|
| `experiment_design/final_experiment_design.md` | **完整五阶段实验设计**（最权威参考） |
| `experiment_design/grpo_minimal_hparams.md` | GRPO 超参最小可复现集 |
| `experiment_design/reward_design.md` | 奖励函数设计（含论文溯源） |
| `experiment_design/eval_protocol.md` | EVAL@1/k/@budget 评测协议 |
| `experiment_design/data_governance.md` | 数据治理原则 |
| `experiment_design/guardrails.md` | 防 reward hacking |
| `experiment_design/project_plan.md` | 简历版项目计划 |
| `experiment_design/resource_plan.md` | GPU 资源规划 |

### Phase 0 文档

| 文件（相对 `coding_model_project/`） | 说明 |
|------|------|
| `phase_0_ Baseline/phase0_implementation_plan.md` | Phase 0 详细实施计划（1,900+ 行） |
| `phase_0_ Baseline/verl_standalone_rollout_guide.md` | verl 架构深度讲解 |
| `phase_0_ Baseline/data_governance_guide.md` | 数据治理完整指南 |
| `phase_0_ Baseline/metrics_collection_spec.md` | 指标收集规范 |
| `reports/data_audit_report.md` | 数据去重审计报告 |

### Phase 1 SFT 文档（已搁置，但含 GRPO 衔接说明）

| 文件（相对 `coding_model_project/`） | 说明 |
|------|------|
| `phase_1_ SFT/SFT_impl_doc/05_end_of_phase_and_grpo_handoff.md` | SFT → GRPO 衔接 |
| `phase_1_ SFT/SFT_impl_doc/explain/07_eval_and_handoff.md` | 评测与衔接详解 |
| `phase_1_ SFT/phase1_eval.py` | Phase 1 评测脚本（含 Tier 系统） |

### verl 框架参考

| 文件（相对仓库根） | 说明 |
|------|------|
| `examples/grpo_trainer/` | 50+ GRPO 训练示例脚本 |
| `verl/trainer/main_ppo.py` | GRPO/PPO 训练入口 |
| `verl/trainer/ppo/ray_trainer.py` | 训练主循环 |
| `verl/trainer/ppo/core_algos.py` | 算法实现（13+ 优势估计器） |
| `verl/trainer/config/` | 配置定义 |

### 数据文件

| 文件（相对 `coding_model_project/`） | 用途 |
|------|------|
| `data/manifests/codecontests_train_wo_valid_big_manifest.jsonl` | **GRPO 训练集 manifest** |
| `data/raw/codecontests_train_wo_valid_big_raw.jsonl` | **GRPO 训练集原始数据** |
| `data/manifests/codecontests_valid_big_manifest.jsonl` | GRPO 高频验证集 |
| `data/manifests/codecontests_valid_big_split_meta.json` | valid_big 审计元数据 |
| `data/manifests/codecontests_test_manifest.jsonl` | 最终测试集 |
| `data/manifests/humaneval_manifest.jsonl` | HumanEval 测试集 |
| `data/manifests/mbpp_reg_manifest.jsonl` | MBPP 回归测试集 |

### 外部依赖

| 组件 | 位置 | 说明 |
|------|------|------|
| SandboxFusion | `SandboxFusion/` (git submodule) | 代码执行沙盒服务 |
| verl-recipe | `recipe/` (git submodule) | 训练配方 |

---

## 附录：快速启动清单

开始 GRPO 实验前的 checklist：

- [ ] 确认在 `feature/grpo-development` 分支上
- [ ] 确认 SandboxFusion 子模块已初始化：`git submodule update --init`
- [ ] 确认训练数据存在：`wc -l data/manifests/codecontests_train_wo_valid_big_manifest.jsonl` → 11,785
- [ ] 确认验证数据存在：`wc -l data/manifests/codecontests_valid_big_manifest.jsonl` → 500
- [ ] 启动 SandboxFusion 服务
- [ ] 确认 GPU 资源（推荐 8×RTX 4090 或等效）
- [ ] 准备 WandB 项目：`rlvr_coding_model`
- [ ] 确认 base model 可访问：`Qwen/Qwen2.5-Coder-7B-Instruct`
- [ ] 复现 Phase 0 基线（可选，验证环境一致性）
- [ ] 编写 GRPO 训练脚本（配置 reward function + 数据路径 + 超参）
