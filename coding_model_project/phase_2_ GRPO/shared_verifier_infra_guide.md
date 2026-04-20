# Shared Verifier Infra 修改详解

本文档完整介绍本次 infra 修改的**动机、架构、每个文件的作用、数据流向、以及后续使用方式**。

---



### 1.2 修改的目标

用一句话概括：**抽出一个 shared verifier 模块，eval 和 RL 训练都从它获取判题结果，保证真值唯一。**

具体来说：
1. 新增一个项目级共享判题模块 (`verifier/shared.py`)
2. `phase0_eval.py` 改为调用它
3. 新增一个 batch reward 函数 (`grpo_batch_reward.py`) 也调用它，通过 verl 的 `BatchRewardManager` 接入训练
4. 新增数据构建脚本 (`build_grpo_parquet.py`) 生成 verl 可消费的 Parquet
5. 在 verl 的 `metric_utils.py` 和 `ray_trainer.py` 中接入 verifier 指标

---

## 二、修改后的架构全景

```
┌─────────────────────────────────────────────────────────────┐
│                     共享判题层                                │
│            coding_model_project/src/verifier/shared.py       │
│                                                              │
│  normalize_candidate()  →  提取代码                           │
│  verify_candidate()     →  单题判定                           │
│  verify_candidate_batch() → 批量并发判定                      │
│                                                              │
│  输入: raw_completion + test_cases + sandbox_endpoint         │
│  输出: VerificationSummary (accepted, pass_ratio_all, ...)   │
└────────────┬──────────────────────────────┬──────────────────┘
             │                              │
     ┌───────▼───────┐             ┌───────▼────────┐
     │  Eval 路径     │             │  RL 训练路径    │
     │  phase0_eval.py│             │  grpo_batch_   │
     │                │             │  reward.py     │
     │ evaluate_with_ │             │                │
     │ run_code()     │             │ compute_score()│
     └───────┬────────┘             └───────┬────────┘
             │                              │
             ▼                              ▼
     终端输出 / JSONL 日志          verl BatchRewardManager
                                            │
                                            ▼
                                   ray_trainer.py 训练循环
                                            │
                                            ▼
                                   WandB 指标面板
```

---

## 三、每个文件的详细说明

### 3.1 `coding_model_project/src/verifier/shared.py` — 核心判题模块

这是本次修改最重要的文件。所有判题逻辑都汇聚在这里。

#### 3.1.1 数据结构

**`CandidateRecord`** — 代码提取结果
```python
@dataclass
class CandidateRecord:
    raw_completion: str      # 模型原始输出
    extracted_code: str      # 提取出的纯代码
    extraction_status: str   # "ok" / "empty_output" / "extraction_failure" / "non_code"
```

**`VerificationSummary`** — 判题结果（eval 和 RL 共用的真值契约）
```python
@dataclass
class VerificationSummary:
    accepted: bool            # 是否全部通过
    passed_tests: int         # 通过的 testcase 数
    total_tests: int          # 总 testcase 数
    pass_ratio_all: float     # passed_tests / total_tests（基于全量 testcase）
    error_type: str           # "success" / "syntax_error" / "runtime_error" / "timeout" / "wrong_answer" / ...
    invalid_for_rl: bool      # 是否为 RL 无效样本（True = 不应该学习）
    invalid_reason: str       # 无效原因
    judge_time_s: float       # 判题耗时（秒）
    extraction_status: str    # 代码提取状态
    per_case_results: list    # 每个 testcase 的详细结果（eval 用，RL 不需要）
```

#### 3.1.2 代码提取流程 (`normalize_candidate`)

这个函数解决了问题 B。提取策略按优先级：

```
1. 尝试匹配 <code>...</code> 标签 ← prompt 模板要求模型用这个格式
2. 尝试匹配 ```python ... ``` 或 ``` ... ``` markdown 代码块
3. 如果文本中有 ``` 或 <code> 标记但匹配失败 → extraction_failure
4. 用 ast.parse() 检测是否是合法 Python → 是则直接当代码用（raw code fallback）
5. 用关键字启发式（def/class/import/for/while/=）检测 → 是则当代码用
6. 以上都不是 → non_code
```

**关键设计**：RL 训练中模型可能不输出 markdown 标记，步骤 4-5 保证正确的裸代码不会被误判为 0 分。

#### 3.1.3 三种数据集的判题路径

shared verifier 通过 `test_cases["type"]` 字段分派到不同路径：

| test_cases.type | 调用函数 | 判题方式 | pass_ratio 含义 |
|---|---|---|---|
| `"humaneval"` | `_verify_humaneval_candidate` | 把代码和所有 assert 拼成一个文件执行一次 | 0 或 1（all-or-nothing）|
| `"mbpp"` | `_verify_mbpp_candidate` | 同上 | 0 或 1 |
| `"codecontests"` | `_verify_codecontests_candidate` | 每个 testcase 独立执行，并发 | 0~1 连续值（这才是 RL 的 dense reward 信号）|

**CodeContests 路径详解**（这是最重要的路径）：

```
输入: 代码 + N 个 {input, output} testcase

1. 创建线程池（max_workers = min(testcase数, limiter_budget)）
2. 每个 testcase 并发提交到 SandboxFusion：
   - 把代码 + stdin 发给 sandbox run_code API
   - 比较 stdout 和 expected_output
   - 返回 {status, passed, ...}
3. 全局 BoundedSemaphore 控制最大并发数（limiter_budget）
4. 收集所有结果后聚合：
   - passed_tests = 通过的 testcase 数
   - pass_ratio_all = passed_tests / total_tests（基于全量 testcase！）
   - accepted = (passed_tests == total_tests)
   - error_type = 出现最多的错误类型
```

#### 3.1.4 `invalid_for_rl` 语义

这个字段决定 RL 训练是否应该从该样本学习：

| 场景 | invalid_for_rl | 理由 |
|---|---|---|
| syntax_error / runtime_error / timeout / wrong_answer | `False` | 有效 RL 信号 — 模型写的代码有问题，reward 反映了问题程度 |
| success | `False` | 有效 RL 信号 — 正向奖励 |
| sandbox_error（sandbox 服务异常） | `True` | 不是模型的问题，是基础设施问题 |
| api_error（网络/HTTP 错误） | `True` | 同上 |
| no_test_cases（数据缺失） | `True` | 没有 testcase 就没有判题依据 |
| empty_output（模型没输出） | `False` | 模型确实没生成代码，这是有效的负反馈 |

在当前 formal 契约里：

- `invalid_for_rl=True` 的样本在 reward 侧会被记录为 `reward_raw=NaN`、`score=0.0`
- 在 GRPO advantage 计算中，这类样本**不参与**组内 mean/std 统计，且其自身 `advantages/returns` 被强制置 0
- 如果某个 prompt-group 全部都是 invalid，则整组 `advantages/returns` 全置 0，并单独记录 `grpo/all_invalid_group_*` 指标

#### 3.1.5 并发控制机制

```
BoundedSemaphore(limiter_budget)  ← 全局唯一，限制对 SandboxFusion 的最大并发请求数
         │
         ├── verify_candidate_batch: 多个问题并发（线程池）
         │       └── 每个问题的 _verify_codecontests_candidate: 多个 testcase 并发（线程池）
         │               └── 每个 testcase 的 _run_code_request: 先 acquire semaphore 再调 API
         │
         └── 所有层级共享同一个 semaphore，确保全局不超限
```

`limiter_budget=8` 意味着同时最多 8 个 SandboxFusion API 调用。

---

### 3.2 `coding_model_project/src/grpo_batch_reward.py` — RL 训练的 reward 函数

这个文件是 shared verifier 和 verl 训练框架之间的桥梁。

#### 3.2.1 它的角色

verl 框架有一个 `BatchRewardManager`（在 `verl/workers/reward_manager/batch.py`），它需要一个 `compute_score` 函数来计算奖励。`grpo_batch_reward.py` 提供的就是这个函数。

#### 3.2.2 调用链

```
verl 训练循环
  → BatchRewardManager.__call__(data)
    → BatchRewardManager.verify(data)
      → self.compute_score(data_sources, solution_strs, ground_truths, extra_infos, **reward_kwargs)
        ↓
        grpo_batch_reward.compute_score()
          → normalize_candidate()       # 从 shared verifier 导入
          → verify_candidate_batch()    # 从 shared verifier 导入，批量并发判题
          → _compute_reward_raw()       # 把 VerificationSummary 映射为 formal/legacy reward
        ↓
        返回 List[dict]，每个 dict 包含 "score" 和 verifier 指标
      ↓
    BatchRewardManager 把 "score" 写入 reward_tensor，其他字段写入 reward_extra_info
  ↓
ray_trainer.py 把 reward_extra_info 写入 batch.non_tensor_batch
  ↓
compute_verifier_metrics() 从 non_tensor_batch 读取并聚合为 WandB 指标
```

#### 3.2.3 formal / legacy reward 模式

正式实现里，`compute_score(..., reward_mode=...)` 支持三条 formal reward 和两条 legacy smoke alias：

**`sparse_accepted`** — 稀疏奖励
```
if invalid_for_rl or truncated_by_max_tokens:
    reward = INVALID_FOR_RL
else:
    reward = 1.0 if accepted else 0.0
```

**`anchored_dense_v1`** — 当前正式主线 reward
```
if invalid_for_rl or truncated_by_max_tokens:
    reward = INVALID_FOR_RL
elif error_type in {empty_output, extraction_failure, non_code, syntax_error}:
    reward = -1.0
else:
    reward = 0.8 * pass_ratio_all + 0.2 * accepted
```

**`dense_anchor_v1`** — 正式强对照 reward
```
if invalid_for_rl or truncated_by_max_tokens:
    reward = INVALID_FOR_RL
elif error_type in {empty_output, extraction_failure, non_code, syntax_error}:
    reward = -1.0
else:
    reward = clip(-0.2 + 1.0 * pass_ratio_all + 0.2 * accepted, -1.0, 1.0)
```

legacy smoke alias `dense_pass_ratio` / `rltf_piecewise` 继续保留，只用于 infra bring-up，不作为正式主线。

当前 A0 / A1 / A2 三条正式算法路线，共用完全相同的 verifier infra 与 reward contract：

- 共用 `BatchRewardManager`
- 共用 `grpo_batch_reward.compute_score`
- 共用 shared verifier (`verifier/shared.py`)
- 共用 trainer-side metadata merge 与 `invalid_for_rl` 语义

A0 / A1 / A2 的差异只发生在 GRPO / actor 更新配置层，不发生在 verifier infra 层。

#### 3.2.4 返回值格式

每个样本返回一个 dict：
```python
{
    "score": 0.48,               # BatchRewardManager 实际写入 reward_tensor 的值
    "reward_raw": 0.48,          # formal reward 原始标量；invalid 样本为 NaN
    "accepted": False,           # 以下都是 verifier 的原始字段
    "passed_tests": 6,
    "total_tests": 10,
    "pass_ratio_all": 0.6,
    "error_type": "wrong_answer",
    "invalid_for_rl": False,
    "invalid_reason": "",
    "judge_time_s": 2.3,
    "extraction_status": "ok",
    "finish_reason": "stop",
    "truncated_by_max_tokens": False,
    "problem_id": "CF_1234",
}
```

当前正式主链固定为：

- rollout / agent loop 只回传独立字段 `finish_reason`、`truncated_by_max_tokens`
- trainer 在 `_compute_or_extract_reward()` 前复制每个 sample 的 `extra_info`，再把这两个字段合并进去
- `BatchRewardManager` 继续只读 `data.non_tensor_batch["extra_info"]`
- validation 侧对 `reward_raw` 采用 finite-only 聚合，避免 invalid 样本把 `val-aux/*/reward_raw/*` 污染成 `NaN`

`BatchRewardManager` 会把除 `score` 外的所有字段收集到 `reward_extra_info` dict 中。

#### 3.2.5 formal 主线的固定 infra 约束

在完成 A0 / A1 / A2 正式算法实现后，shared verifier infra 需要把下面这些约束视为“冻结口径”：

- formal 入口固定 `reward_manager.name=batch`
- formal 入口固定 `reward_model.use_reward_loop=False`
- formal 入口固定 `reward_model.launch_reward_fn_async=False`
- formal 入口固定 `algorithm.adv_estimator=grpo`
- formal 入口显式关闭 `algorithm.filter_groups.enable`

其中前两条尤其重要：当前 trainer-side `extra_info` 合并 helper 运行在同步 reward 主链上，因此不应在 formal 入口里随意切回 reward loop 或 async reward。

---

### 3.3 `coding_model_project/src/prompting.py` — Prompt 模板

从 `phase0_eval.py` 中抽出的独立模块，避免构建 Parquet 时 import 整个 eval 脚本。

**关键设计**：所有 prompt 模板要求模型用 `<code>...</code>` 格式输出，这与 `shared.py` 中代码提取的**第一优先级**匹配。

```
Output ONLY:
<code>
# python code
</code>
```

不同数据集的 prompt 差异：
- **HumanEval**：要求补全函数，不读 stdin，不 print
- **MBPP**：要求实现函数，指定函数名和调用方式
- **CodeContests**：要求完整程序，读 stdin 写 stdout

---

### 3.4 `coding_model_project/src/build_grpo_parquet.py` — 数据构建脚本

将 manifest + raw JSONL 转换为 verl 可消费的 Parquet 文件。

#### 3.4.1 输入输出

```
输入:
  data/manifests/codecontests_train_wo_valid_big_manifest.jsonl  (11785 个 problem_id)
  data/raw/codecontests_train_wo_valid_big_raw.jsonl             (原始数据，包含 prompt + test_cases)
  ... (其他数据集同理)

输出:
  data/grpo_parquet/
  ├── train.parquet          # 训练集（11785 条）
  ├── val_tier1.parquet      # 高频验证（codecontests_valid + mbpp_reg）
  ├── val_tier2.parquet      # 低频验证（codecontests_valid_big，500 条）
  ├── final_eval.parquet     # 最终评测（codecontests_test + humaneval）
  ├── smoke_train.parquet    # 烟测训练（64 条，固定 seed）
  ├── smoke_val.parquet      # 烟测验证（32 条，固定 seed）
  └── build_summary.json     # 构建摘要
```

#### 3.4.2 Parquet 行格式

每行是一个 dict，verl 的 `DataProto` 会消费这些字段：

```python
{
    "data_source": "codecontests_train_wo_valid_big",    # verl 用来分派 reward 函数
    "prompt": [                                           # verl 用来做 rollout 生成
        {"role": "system", "content": "You are an expert..."},
        {"role": "user", "content": "Solve the following..."}
    ],
    "ability": "code",
    "reward_model": {
        "style": "rule",
        "ground_truth": {                                 # BatchRewardManager 传给 compute_score
            "problem_id": "CF_1234",
            "test_cases": {
                "type": "codecontests",
                "tests": [
                    {"input": "5\n1 2 3 4 5", "output": "15"},
                    {"input": "3\n1 2 3", "output": "6"},
                    ...
                ]
            },
            "dataset": "codecontests_train_wo_valid_big"
        }
    },
    "extra_info": {                                       # 传给 compute_score 的附加信息
        "problem_id": "CF_1234",
        "split": "train",
        "dataset": "codecontests_train_wo_valid_big"
    }
}
```

#### 3.4.3 数据安全

- manifest 作为过滤器：只有 manifest 中的 problem_id 才会进入 Parquet
- 如果 manifest 中有 id 但 raw 中找不到，脚本会**报错退出**（不是静默跳过）
- 训练集用的是 `codecontests_train_wo_valid_big`（11785 条），已剔除 valid_big 的 500 条，避免数据泄露

---

### 3.5 `verl/trainer/ppo/metric_utils.py` 新增 — `compute_verifier_metrics()`

这个函数从 `batch.non_tensor_batch` 中读取 verifier 的输出字段，聚合为 WandB 可记录的指标。

```python
def compute_verifier_metrics(batch: DataProto) -> dict[str, Any]:
    # 读取 reward_extra_info 写入的字段，计算：
    # verifier/pass_ratio_all_mean  — 平均通过率
    # verifier/accepted_rate        — 全部通过（AC）的比例
    # verifier/invalid_for_rl_rate  — 无效样本比例
    # verifier/judge_time_s_mean    — 平均判题耗时
    # verifier/judge_time_s_p95     — P95 判题耗时
    # verifier/empty_output_rate    — 空输出比例
    # verifier/non_code_rate        — 非代码输出比例
    # verifier/extraction_failure_rate — 代码提取失败比例
    # verifier/syntax_error_rate    — 语法错误比例
    # verifier/runtime_error_rate   — 运行时错误比例
    # verifier/timeout_rate         — 超时比例
    # verifier/wrong_answer_rate    — 答案错误比例
```

---

### 3.6 `verl/trainer/ppo/ray_trainer.py` 修改 — 接入 verifier 指标

在训练循环的 Metrics 阶段（阶段 12）新增一行：

```python
metrics.update(compute_verifier_metrics(batch=batch))
```

**数据流关键步骤**（在 ray_trainer.py 1715-1716 行）：

```python
if reward_extra_infos_dict:
    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
```

这行代码把 `grpo_batch_reward.compute_score` 返回的所有 verifier 字段（`accepted`, `pass_ratio_all`, `error_type` 等）写入 `batch.non_tensor_batch`，后续 `compute_verifier_metrics` 才能读到它们。

---

### 3.7 `phase0_eval.py` 修改 — eval 主链切到 shared verifier

**核心改动**：`evaluate_with_run_code()` 函数现在的实现是：

```python
def evaluate_with_run_code(completion, test_cases, problem_id, config):
    candidate = normalize_candidate(completion)           # 共享代码提取
    summary = verify_candidate(                           # 共享判题
        candidate=candidate,
        problem_id=problem_id,
        test_cases=test_cases,
        sandbox_endpoint=config.sandbox_url,
        ...
    ).to_dict()
    return _summary_to_eval_result(problem_id, summary)   # 适配到旧 EvalResult 格式
```

**评测主循环** `evaluate_single_problem_async()` 现在只有一个路径：
```python
if test_cases and config.use_external_tests:
    return evaluate_with_run_code(...)  # 走 shared verifier
else:
    return "no_test_cases" error        # 没有 testcase 就报错
```

**旧路径已废弃**：
- `evaluate_with_submit_api()` → 调用直接抛 RuntimeError
- `evaluate_with_compute_score()` → 调用直接抛 RuntimeError

**per-problem 输出新增字段**：
```python
{
    "pass_ratio_all": ...,       # 基于全量 testcase 的通过率
    "passed_tests": ...,         # 通过的 testcase 数
    "total_tests": ...,          # 总 testcase 数
    "invalid_for_rl": ...,       # 是否为 RL 无效样本
    "extraction_status": ...,    # 代码提取状态
}
```

---

### 3.8 运行脚本

**`coding_model_project/scripts/run_phase0.sh`** — Eval 运行脚本
- 强制要求 manifest 目录存在（因为 shared verifier 只用 external tests）
- 传入 `--verifier_limiter_budget` 控制并发

**`coding_model_project/phase_2_ GRPO/run_grpo_smoke.sh` / `run_grpo_step_smoke.sh`** — GRPO 烟测脚本
- 使用 `reward_manager.source=register, name=batch` → verl 的 `BatchRewardManager`
- 使用 `custom_reward_function.path=.../grpo_batch_reward.py` → 自定义 reward
- 使用 `custom_reward_function.reward_kwargs.*` 传入 sandbox_endpoint, reward_mode 等
- 使用 `reward_model.use_reward_loop=False` → 继续走 trainer-side reward 主链
- 使用 `reward_model.launch_reward_fn_async=False` → 同步计算

**`coding_model_project/phase_2_ GRPO/run_grpo_formal.sh` + `run_grpo_a0.sh` / `run_grpo_a1.sh` / `run_grpo_a2.sh`** — 当前正式主线入口
- A0 / A1 / A2 都复用同一套 shared verifier infra
- 默认 `reward_mode=anchored_dense_v1`
- 默认 `reward_model.use_reward_loop=False`
- 默认 `reward_model.launch_reward_fn_async=False`
- 默认 `algorithm.adv_estimator=grpo`

---

## 四、verl 框架的 reward 接入机制详解

这一节解释 verl 的 reward 系统是如何加载和使用自定义 reward 函数的，因为这是理解整个集成的关键。

### 4.1 两个独立的配置维度

verl 的 reward 系统有两个**独立**的配置项：

| 配置项 | 作用 | 本项目的值 |
|---|---|---|
| `reward_manager` | 选择哪个 RewardManager 类来管理 reward 计算 | `source=register, name=batch` → `BatchRewardManager` |
| `custom_reward_function` | 选择具体的评分函数（被注入到 RewardManager 中） | `path=.../grpo_batch_reward.py, name=compute_score` |

它们的关系：`RewardManager` 是容器，`compute_score` 是被装进容器的评分逻辑。

### 4.2 加载流程（`verl/trainer/ppo/reward.py`）

```
1. get_custom_reward_fn(config)
   → 从 custom_reward_function.path 加载 grpo_batch_reward.compute_score
   → 用 partial() 把 reward_kwargs (sandbox_endpoint, reward_mode, ...) 预绑定
   → 返回一个"已绑定参数的 compute_score"

2. get_reward_manager_cls("batch")
   → 从 verl 的 registry 拿到 BatchRewardManager 类

3. BatchRewardManager(
       tokenizer=tokenizer,
       num_examine=0,
       compute_score=上面的预绑定函数,   ← 关键！自定义评分函数被注入到这里
       reward_fn_key="data_source",
   )
```

### 4.3 运行时调用链

```
训练循环每一步:
  batch = rollout 生成的数据（含 prompt, response, ground_truth 等）

  trainer-side metadata merge
    → 逐 sample 复制 extra_info
    → 把 finish_reason / truncated_by_max_tokens 合并进副本
    → 回填到 batch.non_tensor_batch["extra_info"]

  compute_reward(batch, reward_fn)
    → reward_fn(batch, return_dict=True)
      → BatchRewardManager.__call__(batch)
        → BatchRewardManager.verify(batch)
          → 解码所有 response 为文本
          → 从 batch 中提取 ground_truths
          → 调用 self.compute_score(
                data_sources=[...],
                solution_strs=[解码后的文本...],
                ground_truths=[test_cases...],
                extra_infos=[...],
                sandbox_endpoint="http://...",    ← 来自预绑定参数
                reward_mode="anchored_dense_v1",  ← 当前 formal 默认；也可切到其他 formal/legacy 模式
                limiter_budget=8,                 ← 来自预绑定参数
            )
          → 这就是 grpo_batch_reward.compute_score()
            → 调用 shared verifier 批量判题
            → 返回 [{score: 0.5, accepted: False, ...}, ...]
        → 把 score 写到 reward_tensor[i, last_token_pos]
        → 把其他字段收集到 reward_extra_info
      → 返回 {reward_tensor, reward_extra_info}
    → 返回 (reward_tensor, reward_extra_infos_dict)

  batch.non_tensor_batch.update(reward_extra_infos_dict)  ← 写入 batch，供 metrics 读取
  batch.batch["token_level_scores"] = reward_tensor       ← 写入 batch，供 advantage 计算
```

### 4.4 为什么选 `BatchRewardManager` 而不是 `NaiveRewardManager`

- `NaiveRewardManager`：逐条调用 `compute_score(data_source, solution_str, ground_truth, extra_info)` — 每次只传一个样本
- `BatchRewardManager`：一次性传入整个 batch — `compute_score(data_sources=[...], solution_strs=[...], ...)`

我们需要批量判题才能用 `verify_candidate_batch` 做并发，`NaiveRewardManager` 的逐条调用无法利用批量并发优势。

---

## 五、后续使用方式

### 5.1 构建 formal 训练数据

```bash
cd /path/to/verl

# 确保 data/ 目录下有 manifests/ 和 raw/ 子目录
ls coding_model_project/data/manifests/
ls coding_model_project/data/raw/

# 构建 formal parquet
python coding_model_project/src/build_grpo_parquet.py \
    --data_root coding_model_project/data \
    --output_dir coding_model_project/data/grpo_parquet \
    --seed 42
```

如果只是做 infra bring-up，可以再额外构建 smoke / step-smoke 数据；如果目标是正式 A0 / A1 / A2 实验，则优先使用：

- `train.parquet`
- `val_tier1.parquet`
- `val_tier2.parquet`
- `final_eval.parquet`

### 5.2 运行 Eval（Phase 0 Baseline）

```bash
cd coding_model_project

# 确保 SandboxFusion 已启动
curl -sf http://localhost:8080/v1/ping

# 运行评测
bash scripts/run_phase0.sh
```

eval 结果中每题都会有 `pass_ratio_all`, `passed_tests`, `total_tests`, `invalid_for_rl` 字段。

### 5.3 运行 formal GRPO（A0 / A1 / A2）

```bash
cd /path/to/verl

# 确保 SandboxFusion 已启动
# 确保已构建 formal parquet

# 运行 A1 formal
bash "coding_model_project/phase_2_ GRPO/run_grpo_a1.sh"
```

如果要切换算法路线：

```bash
bash "coding_model_project/phase_2_ GRPO/run_grpo_a0.sh"
bash "coding_model_project/phase_2_ GRPO/run_grpo_a1.sh"
bash "coding_model_project/phase_2_ GRPO/run_grpo_a2.sh"
```

当前推荐理解是：

- `run_grpo_formal.sh` 是共享 formal 入口
- `run_grpo_a0.sh` / `run_grpo_a1.sh` / `run_grpo_a2.sh` 只是三条算法薄封装
- smoke 脚本只保留给 infra bring-up，不再代表正式主线

### 5.4 切换 Reward Mode

在 formal 主线中，推荐只在这三条模式之间切换：

```bash
REWARD_MODE=anchored_dense_v1  # 当前正式主线
REWARD_MODE=dense_anchor_v1    # 正式强对照
REWARD_MODE=sparse_accepted    # 稀疏 ablation
```

legacy 模式仍然可以保留给 smoke / bring-up：

```bash
REWARD_MODE=dense_pass_ratio   # legacy smoke alias
REWARD_MODE=rltf_piecewise     # legacy smoke alias
```

例如：

```bash
REWARD_MODE=dense_anchor_v1 bash "coding_model_project/phase_2_ GRPO/run_grpo_a1.sh"
```

### 5.5 调整并发度与 formal 默认

```bash
LIMITER_BUDGET=16  # 增大并发（需要 SandboxFusion 能承受）
LIMITER_BUDGET=4   # 减小并发（如果 sandbox 不稳定）
```

当前推荐默认需要分成“历史单实例默认”和“当前 4x5090 正式 deployed 默认”两层理解：

- 历史单实例 formal 默认：`LIMITER_BUDGET=8`
- 当前 4x5090 正式 deployed 配置：
  - `SANDBOX_URL=http://localhost:8090`
  - `LIMITER_BUDGET=24`
  - `RUN_TIMEOUT_S=30`
- formal：`RUN_TIMEOUT_S=30`
- smoke / step-smoke：`RUN_TIMEOUT_S=15`

不建议在 formal 入口中随意改动：

- `reward_model.use_reward_loop=False`
- `reward_model.launch_reward_fn_async=False`

#### 5.5.1 当前线上使用的 Nginx LB + multi-sandbox 形态

shared verifier / reward 侧完全没有改成 multi-endpoint client；仍然只看到一个 `sandbox_endpoint: str`。

当前 deployed 拓扑是：

```text
shared verifier / grpo_batch_reward
  -> sandbox_endpoint = http://localhost:8090
  -> Nginx upstream sandbox_pool
     -> 127.0.0.1:8081
     -> 127.0.0.1:8082
```

当前没有保留 `8083+` 作为正式默认；`8083` 只在 rescue 型 bring-up / 对照时用过。

这套 infra 的直接收益是：

- 不需要改 verifier / client 的接口
- 可以通过单一 `sandbox_endpoint` 透明切换 single / multi backend
- 可以在 Nginx 层拿到 upstream 分布、request time、status 等运维观测

#### 5.5.2 当前 repo 中已经落地的 ops 资产

位于 `coding_model_project/phase_2_ GRPO/ops/`：

- `sandbox_backend_start.sh`
- `sandbox_backend_stop.sh`
- `sandbox_backend_status.sh`
- `capture_host_baseline.sh`
- `render_nginx_sandbox_lb.sh`
- `apply_nginx_sandbox_lb.sh`
- `lb_validate_probe.py`
- `run_grpo_a1_reward_probe.sh`
- `run_grpo_a1_fastval_gate.sh`
- `run_grpo_a1_formal_observation_resume10.sh`
- `monitor_first_save.py`

其中：

- backend bring-up 复用现有 `make run-online PORT=...`
- Nginx config 会挂 `/run_code`、`/run_jupyter`、`/v1/ping`
- LB probe 走 raw HTTP，不依赖 SDK `run_code()`，这样失败请求也能保留：
  - HTTP status
  - response body excerpt
  - `X-Upstream-Addr`
  - `X-Request-Id`

#### 5.5.3 multi-sandbox bring-up / validate 的标准顺序

推荐顺序：

1. 停掉 legacy `:8080`
2. 采 clean-host baseline
3. 逐个 bring-up `8081` / `8082` / `8083`
4. 只保留通过 guardrail 的 backend 数量
5. 渲染并应用 `:8090` Nginx config
6. 先做 direct smoke，再做 LB raw HTTP probe
7. 再跑 reward-only probe / fast-val / 正式 RL

当前 guardrail 要点：

- backend RSS 不超过 `4 GiB`
- `MemAvailable` 不要相对前一阶段跌太多
- `/tmp` 空间不能过低
- swap 不应继续抬升
- 新 backend 的 direct smoke `p95` 不应显著恶化

更细的执行方式参考：

- `coding_model_project/phase_2_ GRPO/ops/README.md`

#### 5.5.4 当前对“Failed to write to stdin ... handler is closed” 的判断

这个现象在当前 formal run 中已经实际出现过，需要明确口径：

- 它出自 `SandboxFusion/sandbox/runners/base.py`
- 更像“子进程过早退出后，父进程给 stdin 写入失败”的 noisy backend log
- verifier 不会因此自动把样本归成 `sandbox_error`
- 当前更应该用它做“benign signal / log hygiene”处理，而不是直接当基础设施崩坏

因此运维上要这样看：

- 真正的硬异常：
  - Nginx non-200
  - `SandboxError`
  - backend `500`
  - training log traceback / OOM / save failure
- 需要持续观察但不应单独 stopper 的 noisy log：
  - `Failed to write to stdin ... handler is closed`
  - `Broken pipe`

现在 `monitor_first_save.py` 已经把这类日志单独归到 `backend_benign_hits`，不再直接触发 first-save 监控失败。

### 5.6 在 WandB 中查看的关键指标

训练开始后，在 WandB 面板中关注这些指标：

| 指标名 | 含义 | 预期趋势 |
|---|---|---|
| `verifier/pass_ratio_all_mean` | 批次平均通过率 | 应随训练上升 |
| `verifier/accepted_rate` | AC 率 | 应随训练上升 |
| `verifier/invalid_for_rl_rate` | 无效样本率 | 应保持低位（<5%）|
| `verifier/truncated_by_max_tokens_rate` | 截断导致的无效率 | 不应持续升高 |
| `verifier/reward_raw_mean` | finite reward 的均值 | 应与 pass_ratio/accepted 同方向变化 |
| `verifier/reward_raw_valid_rate` | 有效 reward 样本比例 | 应保持接近 1 |
| `verifier/judge_time_s_p95` | P95 判题耗时 | 应稳定（否则 sandbox 有瓶颈）|
| `verifier/syntax_error_rate` | 语法错误率 | 不应上升（否则模型在退化）|
| `verifier/runtime_error_rate` | 运行时错误率 | 不应持续上升 |
| `verifier/timeout_rate` | 超时率 | 不应持续上升（否则模型在写死循环）|
| `verifier/empty_output_rate` | 空输出率 | 不应上升 |
| `grpo/all_invalid_group_rate` | 全 invalid prompt-group 比例 | 应保持在 0 或极低 |
| `timing_s/reward` | reward 阶段 wall time | 不应无缘由暴涨 |
| `timing_s/update_actor` | actor update wall time | 当前常是主要瓶颈之一 |
| `perf/throughput` | 每秒每卡 token 数 | 用于横向比较不同配置 |

---

## 六、与 verl 框架原有代码的关系

### 6.1 绕过了什么

本次修改**绕过**了 verl 自带的以下代码，不再使用：

| 被绕过的代码 | 原因 |
|---|---|
| `verl/utils/reward_score/sandbox_fusion/__init__.py` 的 `compute_score()` | 只用前 10 个 testcase |
| `verl/utils/reward_score/sandbox_fusion/utils.py` 的 `check_correctness()` | 依赖 requests 直接调 API，我们用 sandbox_fusion SDK |
| `verl/utils/reward_score/__init__.py` 的 `default_compute_score()` | 丢失 per_case 元数据 |
| `NaiveRewardManager` | 逐条处理，无法批量并发 |

### 6.2 复用了什么

| 复用的代码 | 方式 |
|---|---|
| `BatchRewardManager` (`verl/workers/reward_manager/batch.py`) | 通过 `reward_manager.name=batch` 使用 |
| `load_reward_manager` + `get_custom_reward_fn` (`verl/trainer/ppo/reward.py`) | 用于加载自定义 compute_score 函数 |
| `ray_trainer.py` 的 reward_extra_info 写入机制 | 利用已有的 `batch.non_tensor_batch.update()` |
| `AbstractRewardManager` 接口 | grpo_batch_reward 的 compute_score 符合 BatchRewardManager 要求的签名 |

### 6.3 formal 主线落地后，verl 框架内实际涉及的改动

初版 shared verifier 只需要很少的框架改动；但在 formal A0 / A1 / A2 完整落地后，涉及 verl 框架本体的地方已经不止两处。

当前需要一起理解的框架内文件有：

1. `verl/trainer/ppo/reward.py`
   - trainer-side `extra_info` merge helper
   - 在 reward 前把 `finish_reason` / `truncated_by_max_tokens` 合并回 per-sample `extra_info`
2. `verl/trainer/ppo/core_algos.py`
   - `invalid_for_rl` 样本不参与 GRPO 组均值/方差
   - all-invalid group 整组置 0
3. `verl/trainer/ppo/metric_utils.py`
   - verifier 指标
   - validation `reward_raw` finite-only 聚合
   - `grpo/all_invalid_group_*`
4. `verl/trainer/ppo/ray_trainer.py`
   - 接入 verifier metrics
   - 接入 trainer-side metadata merge
5. `verl/workers/reward_manager/batch.py`
   - reward manager 读取 `extra_info` 时的对象复制与稳定处理
6. `verl/workers/rollout/replica.py`
   - `TokenOutput` 保留 `finish_reason`
7. `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
   - rollout 保留原始 `finish_reason`
8. `verl/experimental/agent_loop/single_turn_agent_loop.py`
   - agent loop 只回传独立 metadata 字段，不直接改 `extra_info`

所以现在更准确的说法是：

- shared verifier 的判题逻辑仍主要在 `coding_model_project/`
- 但为了把它稳定接到 formal A0 / A1 / A2 训练主链，verl 框架内也有一小组必要改动

---

## 七、面试常见问题

### Q: 为什么不直接用 verl 自带的 sandbox_fusion reward？

A: 两个原因。第一，verl 自带的 `compute_score(continuous=True)` 只用前 10 个 testcase 算 pass_ratio，CodeContests 有 50-200+ 个 testcase，子集估算不可信。第二，`default_compute_score` 会丢弃 per-case 元数据，只返回一个 float，无法追踪错误类型分布和 invalid 样本。

### Q: 为什么要做 shared verifier 而不是 eval 和 RL 各自实现？

A: 确保评测和训练使用完全相同的判题标准。如果 eval 说某题通过率 30%，训练时的 reward 也必须是 0.3。两套代码判题标准不一致会导致：(1) baseline 数字和训练曲线无法对比；(2) 面试时无法解释为什么 eval 提升了但 reward 曲线没反映。

### Q: `invalid_for_rl` 的设计依据是什么？

A: 区分"模型的问题"和"基础设施的问题"。如果 sandbox 挂了（api_error/sandbox_error），模型得到 reward=0 会让 GRPO 认为这个方向不好，但实际上模型的代码可能是正确的。所以这类样本不应参与学习。相反，syntax_error 是模型写出了不合法的代码，这是有效的负反馈。

### Q: A0 / A1 / A2 需要三套不同的 verifier infra 吗？

A: 不需要。三条算法路线共享完全相同的 shared verifier infra，包括 `BatchRewardManager`、`grpo_batch_reward.compute_score`、trainer-side `extra_info` merge、`invalid_for_rl` 语义和 verifier metrics。它们的差异在 GRPO / actor 更新配置层，不在判题 infra 层。

### Q: 为什么 formal 入口固定 `reward_model.use_reward_loop=False` 和 `reward_model.launch_reward_fn_async=False`？

A: 因为当前 truncation metadata 的正式接法是：rollout / agent loop 先回传独立字段，trainer 再在 reward 前把它们合并进 per-sample `extra_info` 副本。这个 helper 工作在同步 reward 主链上。如果随意切回 reward loop 或 async reward，会绕开当前已经验证过的主链假设，后续接手者很难定位问题。

### Q: 为什么选 `BatchRewardManager` 而不是自己写一个 `AbstractRewardManager`？

A: 工程上最小侵入。`BatchRewardManager` 已经实现了 token 解码、reward_tensor 写入、reward_extra_info 收集等所有通用逻辑，我们只需要提供 `compute_score` 函数。如果自己写 RewardManager，需要把这些逻辑全部重新实现，而且框架升级时容易 break。
