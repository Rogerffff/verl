# Phase 4 Step 1 实施方案：开发评测版 One-Turn Verifier-Guided Repair

本文档是 [repair_phase4_design.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_phase4_design.md) 的**实现级补充**。  
它只回答当前真正要落地的那一步：

- `valid_big500`
- `delta69`

上的 **开发评测版 one-turn repair**

也就是：

1. 先按当前单轮协议生成代码
2. 跑 verifier
3. 对选中的失败样本构造一条短反馈
4. 再让模型修一次
5. 再跑一次 verifier
6. 分开汇报：
   - 单轮指标
   - repair 指标

---

## 0. 2026-04-08 审计修订

基于全量 dev-eval testcase 代表性审计：

- [testcase_selection_rule_audit_v1.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_analysis/testcase_selection_rule_audit_v1.md)
- [testcase_selection_rule_recommendation_v1.json](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_analysis/testcase_selection_rule_recommendation_v1.json)

Step 1 的实现口径补充修订如下：

1. 正式对照 checkpoint 不再只看 `step900 / step1000`
   - Step 1 的正式对照组改为：`step600 / step900 / step1000`
   - 目的不是只看 partial winner，而是同时比较：
     - single-shot solve winner
     - partial / repair-ready winner

2. testcase 选择规则不再使用“第一条 matching failure”
   - `wrong_answer`：
     - 使用 `simplest_counterexample`
     - 核心思想是优先更短、更干净、更像 isolated counterexample 的 failing case
   - `runtime_error`：
     - 使用 `clearest_exception_then_shortest_stdin`
     - 优先带明确异常关键词的 case，再在其中选更短输入
   - `timeout`：
     - 使用 `shortest_timeout_stdin`
     - 并固定附加一条复杂度提示

3. Step 1 的结果分析必须按 first-pass `pass_ratio` 分桶
   - `bucket_0`
   - `bucket_0_0.2`
   - `bucket_0.2_0.6`
   - `bucket_0.6_1.0`

4. 审计确认当前 `per_case_results` 的截断发生在 verifier 层，而不是 sandbox 或落盘层
   - 位置在：
     - [verifier/shared.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py) 的 `_verify_codecontests_testcase()`
     - 以及 `_truncate_text()`
   - 当前主要上限是：
     - `stdin` 约在 `300` chars 截断
     - `expected / actual / stderr` 大多约在 `500` chars 截断
     - `timeout` 分支的 `stderr` 是 `300` chars
   - SandboxFusion 返回的是完整 `stdout/stderr`，JSONL 只是把 verifier 已截断好的 dict 原样序列化
   - 因此选“第一条 fail”经常会把已经被 verifier 截断的大 case 塞进 repair prompt
   - 这不会改变 Step 1 的 testcase selection 规则结论，但说明未来如果要保留代表性 case 的完整输入，最合适的改动点在 verifier 层

5. Step 1 的主 trigger 先保持不变
   - 仍然默认：
     - `accepted == False`
     - `error_type in {wrong_answer, runtime_error, timeout}`
     - `pass_ratio >= 0.2`
   - 但 repair 报告必须附带按 bucket 的条件成功率

## 0.1 2026-04-17 结果修订

最初这份实现计划是围绕 `step600 / step900 / step1000` 的 pre-SFT 对照写的；这部分历史设计仍然有效，但当前真正落地后的结论已经更新为：

- Protocol A：
  - 固定 `step900 valid_big500 raw patched_rr828x` 的 canonical `per_problem`
  - 比较 `step900` 与 SFT `step30 / 40 / 50 / 60` 的 fixed-input repairability
  - 结果是四个 SFT checkpoint 全部 `74/500`，`step900` 为 `73/500`
  - 其中 `step40` 的 final `pass_ratio_mean` 最高
- Protocol B：
  - 各 checkpoint 自己 raw first-pass，再 self-repair
  - 结果排序为：
    - `step40 > step50 > step60 > step30`
  - `step40` 的 final `accepted_after_1_repair = 78/500`

因此当前实现级拍板应改成：

- 主 one-turn repair base：`step40`
- repaired partial-quality 次优备选：`step50`
- frozen first-pass source / 历史 validated base：`step900`
- `step1300` 继续只保留为 partial-credit probe

---

## 1. 当前拍板

### 1.1 Step 1 不改 PPO trainer

当前这一步**不要**改：

- [ray_trainer.py](/Users/roger/Desktop/coding_RL_project/verl/verl/trainer/ppo/ray_trainer.py)
- rollout worker
- `verl.experimental.agent_loop`

原因不是这些能力不存在，而是：

1. 当前 `valid_big500` / `delta69` 评测本来就不是走 PPO trainer 主循环
2. 当前 Step 1 是**开发评测协议**，不是训练协议
3. 当前最小、最稳、最不容易引入回归的路径，是直接扩展现有 standalone eval

所以这一步的工程策略是：

- **复用现有 rollout server**
- **不碰 PPO trainer**
- **不碰 rollout worker base interface**
- **不先上 agent loop**
- **直接在 `coding_model_project/src` 的 standalone eval 链实现两次生成**

### 1.2 `agent_loop` 当前只作为后续扩展参考

`verl` 里确实已经有：

- [single_turn_agent_loop.py](/Users/roger/Desktop/coding_RL_project/verl/verl/experimental/agent_loop/single_turn_agent_loop.py)
- [tool_agent_loop.py](/Users/roger/Desktop/coding_RL_project/verl/verl/experimental/agent_loop/tool_agent_loop.py)
- [agent_loop.py](/Users/roger/Desktop/coding_RL_project/verl/verl/experimental/agent_loop/agent_loop.py)

但它们当前的定位是：

- 训练时 rollout
- 多轮轨迹
- `DataProto` / Ray worker / request scheduling

这对 Step 1 来说过重。  
如果将来真的做 repair-aware RL，再考虑单独加：

- `RepairOneTurnAgentLoop`

但这不是当前 Step 1 的正确落点。

---

## 2. 当前 baseline eval 的真实代码路径

### 2.1 launcher

当前 checkpoint eval 的真实入口是：

- [run_grpo_codecontests_validbig_checkpoint.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_grpo_codecontests_validbig_checkpoint.sh)
- [run_grpo_codecontests_delta69_checkpoint.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_grpo_codecontests_delta69_checkpoint.sh)

它们做的事是：

1. merge checkpoint 成 HF 权重
2. 启动一个独立的 vLLM OpenAI-compatible server
3. 调 [phase0_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py)

所以当前 Step 1 的最佳挂接点，不在 trainer，而在：

- [phase0_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py)

### 2.2 当前 baseline eval 的核心函数

当前 `phase0_eval.py` 已经完整拥有：

- 配置入口：
  - [EvalConfig](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py#L138)
- 生成：
  - [generate_code()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py#L359)
  - [batch_generate()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py#L445)
- 判题：
  - [evaluate_with_run_code()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py#L770)
  - [evaluate_single_problem_async()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py#L1566)
- 单 dataset 主流程：
  - [evaluate_dataset()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py#L1616)
- 全局入口：
  - [run_evaluation()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py#L1894)
  - [main()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py#L2084)

当前 verifier 路径也已经足够支撑 repair：

- [verify_candidate()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py#L682)
- CodeContests testcase 结果结构：
  - [\_verify_codecontests_testcase()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py#L468)

可直接拿到：

- `passed_tests`
- `total_tests`
- `pass_ratio_all`
- `error_type`
- `per_case_results`
  - `stdin`
  - `expected`
  - `actual`
  - `stderr`
  - `message`
  - `status`

这已经足够生成 one-turn repair feedback。

---

## 3. 实现策略总览

### 3.1 推荐方案

推荐实现路径是：

1. **保持 `phase0_eval.py` 的 baseline 协议不变**
2. 把能复用的生成/判题 helper 继续复用
3. 新增一个并列入口：
   - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py`
4. 新增一个 repair prompt/helper 模块：
   - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py`
   - 或 `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_prompting.py`
5. 新增一个 ops launcher：
   - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/ops/run_phase4_repair_eval_checkpoint.sh`

并且 Step 1 从第一天就沿用当前稳定评测链的双账本口径：

- `raw`
- `clean overlay (problem_quarantine_v3)`

### 3.2 为什么不用“直接往 `phase0_eval.py` 里塞参数”

技术上可以给 `phase0_eval.py` 增加：

- `--repair_mode one_turn`

然后在同一个文件里做分支。

但从当前项目维护角度，我更推荐：

- baseline eval 保持干净
- repair eval 是一个并列协议

因为这两条线之后会长期并存：

- baseline 单轮评测
- repair 评测

如果把 repair 逻辑直接塞进 baseline 主文件，很容易让后续 agent 混淆：

- baseline 指标
- repair 指标
- 单轮日志
- 二轮日志

所以推荐结构是：

- `phase0_eval.py`：保持单轮
- `phase4_repair_eval.py`：承担 one-turn repair

但代码层面应尽量复用 `phase0_eval.py` 的函数，而不是重新复制。

---

## 4. Step 1 的协议定义

### 4.1 输入

输入 checkpoint：

- `step600`
- `step900`
- `step1000`

当前推荐的解释方式是：

- `step600`：single-shot solve winner 对照
- `step900`：broad / partial winner 对照
- `step1000`：A/B-focused 后续分支对照

补充更新（2026-04-17）：

- 上面这组 checkpoint 现在主要保留作 pre-SFT 历史对照
- 当前实际执行 Step 1 时，推荐分成两类：
  - Protocol A：
    - 固定 `step900` canonical raw `per_problem`
    - 比较 `step900` 与 `step30 / 40 / 50 / 60`
  - Protocol B：
    - 直接比较 `step30 / 40 / 50 / 60` 的 self-first-pass + self-repair 结果
- 当前 winner：
  - `step40` 是主 one-turn repair base
  - `step50` 是更偏 repaired partial-quality 的次优点
  - `step900` 不再是当前最佳 base，而是 frozen source / 历史 validated base

Step 1 不是只验证某个单点能不能 repair，而是要回答：

- 哪个 checkpoint 最适合做 repair base
- `accepted_after_1_repair` 的 winner 和 `accepted@1` 的 winner 是否一致

输入 eval slice：

- `codecontests_valid_big`
- `delta69` 对应的 manifest

### 4.2 单题流程

对每个 problem：

1. 构造 baseline prompt
2. first-pass 生成代码
3. first-pass verifier
4. 判断是否触发 repair
5. 如果触发：
   - 生成短 feedback
   - second-pass repair prompt
   - repair generation
   - repair verifier
6. 输出：
   - first-pass 结果
   - second-pass 结果
   - final after-repair 结果

### 4.3 repair trigger

开发评测版先使用保守策略，避免算力和 judge 成本爆炸。

默认 repair 条件建议：

- `accepted == False`
- `error_type in {"wrong_answer", "runtime_error", "timeout"}`
- `pass_ratio >= 0.2`

可选更严格优先级：

- `pass_ratio >= 0.6`

不做 repair：

- `syntax_error`
- `api_error`
- `sandbox_error`
- `no_test_cases`
- `pass_ratio == 0` 的明显坏样本

补充说明：

- `syntax_error` 当前默认不 repair，是开发期的保守默认值
- 后续可以通过 CLI 显式打开，作为一个低成本高收益桶单独实验

原因：

- 当前 Step 1 的目标不是“尽可能多补第二轮”
- 而是判断 repair 对 near-miss 是否真的有开发价值

---

## 5. feedback 设计

### 5.1 设计原则

对 7B coder 模型，repair feedback 应该：

- 短
- 结构化
- 面向修复
- 不要灌太长原始 judge log

### 5.2 推荐字段

建议反馈只包含：

1. `passed_tests / total_tests`
2. `error_type`
3. 一条最具代表性的失败摘要
4. 初轮代码
5. 固定指令：只输出修正后的完整 Python 代码

### 5.3 失败摘要选择规则

当前已经废弃“第一条 matching failure”这条近似规则。  
Step 1 之后统一改成**按错误类型选择代表性 testcase**。

#### `wrong_answer`

规则：

- `simplest_counterexample`

实现语义：

- 在 `per_case_results` 里筛出 `status == "wrong_answer"` 的失败 case
- 计算：
  - `stdin_len`
  - `output_delta`
    - 近似使用 `expected / actual` 的长度差和行数差
- 选择：
  - 更短 `stdin`
  - 更小输出差异
  - 更低 `test_idx`

目标：

- 优先给模型一个更短、更干净、更像 isolated bug 的 counterexample
- 避免把已经被截断的大输入直接塞进 repair prompt

#### `runtime_error`

规则：

- `clearest_exception_then_shortest_stdin`

实现语义：

- 在 `per_case_results` 里筛出 `status == "runtime_error"` 的失败 case
- 优先保留 `stderr` 中出现：
  - `Error`
  - `Exception`
  - `Traceback`
  的 case
- 在这些 case 里选最短 `stdin`
- 再按 `test_idx` 打平

目标：

- 先给模型最清楚的异常信息
- 再给尽可能短、容易 mental execution 的输入

#### `timeout`

规则：

- `shortest_timeout_stdin`

实现语义：

- 在 `per_case_results` 里筛出 `status == "timeout"` 的失败 case
- 直接选最短 `stdin`
- 再按 `test_idx` 打平

并且始终附加固定复杂度提示：

- 程序在部分测试上超时，优先检查时间复杂度、重复计算、嵌套循环、无 memo 的递归等

目标：

- timeout 不再依赖某个巨大的、可能已经被截断的 case 作为唯一信号
- 明确把模型的注意力导向复杂度修复

#### 通用约束

repair feedback 里不要放：

- 所有失败测试
- 长 traceback 全文
- 多条重复 testcase

每次只放：

- 一条代表性 testcase summary
- 再配错误类型对应的固定短提示（必要时）

### 5.4 prompt 结构

建议新增：

- [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)

最少提供两个 helper：

1. `build_repair_feedback_from_eval_result(eval_result: EvalResult) -> dict`
2. `build_codecontests_repair_prompt(problem_prompt: str, first_code: str, feedback: dict) -> str`

当前实现建议进一步固定为：

- `build_codecontests_repair_prompt(..., prompt_mode=\"code_only\")`
- `build_codecontests_repair_prompt(..., prompt_mode=\"short_diagnosis_code\")`

并把 `prompt_mode` 显式记到：

- `run_info.json`
- `first_pass_summary.json`
- `summary.json`
- `repair_summary.json`
- `per_problem/*.jsonl`

原因是 2026-04-09 的 `step900 + valid_big500 + reuse-first-pass` 小消融已经显示：

- clean `short_diagnosis_code`
  - `66 / 500`
  - `conditional_repair_success = 16.44%`
- clean `code_only`
  - `63 / 500`
  - `conditional_repair_success = 12.33%`

因此当前更推荐：

- `short_diagnosis_code` 作为默认优先 prompt mode
- `code_only` 继续保留为基线对照

同时需要特别注意：

- 首轮 `code_only` prompt ablation 曾受 sandbox 掉线污染
- 正式结论只能基于 clean rerun
- 所有后续 prompt 对比都必须继续使用：
  - `reuse-first-pass`
  - 同一份 canonical raw `per_problem`

这里的 `problem_prompt` 口径明确固定为：

- **原始题面**

不要传已经经过 [format_prompt()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/prompting.py#L94) 包装过的 baseline prompt，避免双层 instruction 干扰 repair。

建议新增专用系统 prompt：

- `REPAIR_SYSTEM_PROMPT`

和 baseline 的 [SYSTEM_PROMPT](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/prompting.py#L4) 分开。

推荐 repair 系统提示语义：

- 你正在修复一份先前失败的 Python 竞赛解答
- 必须根据反馈修复
- 输出完整可执行 Python 程序
- 只输出 `<code>...</code>`

---

## 6. 文件级实现方案

### 6.1 新增文件

#### A. `phase4_repair_eval.py`

路径：

- [phase4_repair_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py)

职责：

- 承接 one-turn repair eval 的主流程
- 复用 `phase0_eval.py` 的生成与判题函数
- 做 second pass orchestration
- 单独输出 repair 指标

建议实现的核心函数：

1. `should_trigger_repair(eval_result, config) -> bool`
2. `run_first_pass_batch(...)`
3. `run_repair_pass_batch(...)`
4. `evaluate_dataset_with_repair(...)`
5. `run_repair_evaluation(...)`

#### B. `repair_feedback.py`

路径：

- [repair_feedback.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/repair_feedback.py)

职责：

- 从 `EvalResult.details` 提取结构化 repair feedback
- 生成 repair prompt
- 做 prompt sha256 和截断处理

### 6.2 最小改动文件

#### A. `phase0_eval.py`

建议只做“抽 helper / 提升复用性”的最小改动，不改 baseline 语义。

建议抽出的 helper：

1. `build_prompt_texts_for_batch(...)`
2. `write_per_problem_record(...)`
3. `summarize_dataset_metrics(...)`

这样 `phase4_repair_eval.py` 不需要复制大段逻辑。

#### B. `prompting.py`

可以不改。

如果你希望把 repair prompt 也放到统一 prompt 模块里，可以加：

- `REPAIR_SYSTEM_PROMPT`

但我更推荐先把 repair 相关逻辑留在 `repair_feedback.py`，避免 baseline prompt 文件被混杂。

#### C. `src/utils/qa_logger.py`

当前可先不改。

更稳的做法是在 repair eval 里单独写：

- `qa_logs_first_pass`
- `qa_logs_repair_pass`

或者写一份新的：

- `repair_qa.jsonl`

#### D. `src/utils/metrics.py`

当前也可以先不改。

原因：

- `MetricsCollector` 当前是单轮指标对象
- repair 指标最初可以先在 `phase4_repair_eval.py` 单独聚合并写：
  - `repair_metrics.json`
  - `repair_summary.json`

等 Step 1 跑通后，再决定是否把 repair 指标抽象进共享 metrics 层。

---

## 7. Step 1 下的 server 选择策略

### 7.1 当前结论

当前 [batch_generate()](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py#L445) 是 round-robin 分发。

但 Step 1 当前 checkpoint eval 的真实运行方式是：

- wrapper 脚本起一个本地 vLLM
- `phase0_eval.py` 在 simple mode 下拿到的是单元素 `server_addresses`

所以 Step 1 **不要把 sticky-server 设计成必做项**。

当前结论：

- simple mode 下默认只有一个 server，repair second pass 自然会打回同一个实例
- 因此不需要为了 Step 1 先改 `generate_code()` / `batch_generate()`
- 如果将来要支持多 replica repair eval，再把 sticky-server 作为可选优化加进去

也就是说，Step 1 的最小实现里：

- 不改 `server_address` metadata
- 不改 `preferred_server_addresses`
- 不引入 trainer-side sticky session 语义

---

## 8. 输出与目录结构

### 8.1 输出目录

建议新输出目录和 baseline 并列：

- `outputs/rl_codecontests_validbig500_repair/...`
- `outputs/rl_codecontests_delta69_repair/...`

或者保持同一 `output_dir` 下新增 repair 文件。

更推荐的是：

- 单独的 repair 输出目录

原因：

- raw baseline 和 repair eval 不要混

### 8.2 per-problem 记录

每条记录建议至少包含：

- `dataset`
- `problem_id`
- `accepted`
- `pass_ratio`
- `error_type`
- `prompt_sha256`
- `repair_triggered`
- `repair_reason`
- `final_accepted_after_repair`
- `repair_gain_for_problem`
- `first_pass`
  - `accepted`
  - `pass_ratio`
  - `error_type`
  - `response`
  - `details`
- `repair_pass`
  - `accepted`
  - `pass_ratio`
  - `error_type`
  - `response`
  - `details`

建议路径：

- `output_dir/per_problem/<dataset>.jsonl`

这里的关键约束是：

- **顶层 `accepted / pass_ratio / error_type` 一律表示 after-repair 最终结果**
- first-pass / second-pass 细节放进嵌套字段

这样当前的 clean overlay 工具：

- [compute_eval_clean_overlay.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/compute_eval_clean_overlay.py)

就可以继续直接吃 repair eval 的 per-problem 文件，而不需要先改 overlay 协议。

但 Step 1 还需要同时保留 **first-pass clean 对照**。

因此建议额外再写一份单轮 per-problem 文件：

- `output_dir/per_problem_first_pass/<dataset>.jsonl`

这份文件的 top-level 字段继续沿用当前 baseline 口径：

- `problem_id`
- `accepted`
- `pass_ratio`
- `error_type`

然后对两份 per-problem 都跑：

1. first-pass clean overlay
2. after-repair clean overlay

建议产物：

- `first_pass_clean_overlay_problem_quarantine_v3.json`
- `clean_overlay_problem_quarantine_v3.json`

另外，Phase 4 也必须继续沿用 baseline 当前的截断策略：

- `max_prompt_chars`
- `max_response_chars`

不要默认无上限保存完整：

- `first_pass.prompt`
- `repair_prompt`
- `response`
- `per_case_results`

否则 CodeContests repair 结果会比当前单轮日志大很多。

### 8.3 汇总指标

必须同时保存两套指标：

#### A. 单轮账本

- `accepted_at_1`
- `pass_ratio_mean`
- `pass_ratio_p50`
- `pass_ratio_p90`

#### B. repair 账本

- `accepted_after_1_repair`
- `repair_gain`
- `conditional_repair_success`
- `bucketed_conditional_repair_success`
- `repair_attempt_count`
- `repair_success_count`
- `avg_extra_gen_tokens_repair`
- `avg_extra_judge_time_repair`

并且 Step 1 之后必须补充 bucket 级分析：

- `bucket_0`
- `bucket_0_0.2`
- `bucket_0.2_0.6`
- `bucket_0.6_1.0`

至少输出：

- `repair_attempt_count_by_bucket`
- `repair_success_count_by_bucket`
- `conditional_repair_success_by_bucket`
- `trigger_rate_on_failed_by_bucket`

推荐文件：

- `summary.json`
  - 直接表示 **after-repair 最终结果**
- `repair_summary.json`
  - 专门放 repair 增益、条件成功率、额外成本
- `first_pass_summary.json`
  - 单轮 baseline 对照
- `first_pass_metrics.json`
  - 单轮数据集级指标
- `metrics.json`
  - after-repair 最终指标
- `repair_metrics.json`
  - 数据集级别 repair 指标

---

## 9. CLI / Config 设计

### 9.1 Step 1 直接沿用 `EvalConfig`

Step 1 不建议再包一层：

- `RepairEvalConfig(base: EvalConfig)`

因为这样很容易让 CLI 默认值和现有：

- [phase0_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py#L2084)

漂开。

更稳的做法是：

- 继续直接使用 `EvalConfig`
- 在 `phase4_repair_eval.py` 的 CLI 上增加少量 repair-only 参数

新增 repair 字段建议是：

- `enable_repair: bool = True`
- `repair_max_attempts: int = 1`
- `repair_error_types: List[str] = ["wrong_answer", "runtime_error", "timeout"]`
- `repair_min_pass_ratio: float = 0.2`
- `repair_max_failure_cases: int = 1`
- `repair_max_feedback_chars: int = 1200`
- `repair_prompt_mode: str = "short_diagnosis_code"`
- `repair_output_dir_suffix: str = "_repair"`

### 9.2 CLI 新增参数

建议 `phase4_repair_eval.py` 新增：

- `--repair_min_pass_ratio`
- `--repair_error_types`
- `--repair_max_failure_cases`
- `--repair_max_feedback_chars`
- `--repair_prompt_mode`
- `--first_pass_per_problem`

并继续沿用现有：

- `--max_prompt_chars`
- `--max_response_chars`
- `--run_timeout`
- `--verifier_limiter_budget`

同时保留 `phase0_eval.py` 里已有的：

- `--mode`
- `--model`
- `--vllm_url`
- `--manifest_dir`
- `--datasets`
- `--output_dir`
- `--max_concurrent`
- `--max_concurrent_judges`
- `--batch_size`

### 9.3 Step 1 推荐默认并发

repair eval 的真实成本是：

- 两次生成
- 最多两次 verifier

所以 Step 1 的默认并发应比当前单轮高并发 runner 更保守，避免一开始就把：

- vLLM
- SandboxFusion
- shared verifier limiter

全部打满。

推荐初始值：

- `max_concurrent = 48`
- `max_concurrent_judges = 12`
- `verifier_limiter_budget = 8`
- `batch_size = 24`
- `run_timeout = 30`

如果运行稳定，再逐步向上调。

不要直接照搬当前某些单轮 wrapper 里常见的：

- `180`
- `200`

这类高并发默认值。

---

## 10. Ops launcher 设计

建议新增：

- [run_phase4_repair_eval_checkpoint.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_phase4_repair_eval_checkpoint.sh)

复用当前：

- [run_grpo_codecontests_validbig_checkpoint.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_grpo_codecontests_validbig_checkpoint.sh)

的主要套路：

1. merge checkpoint
2. 拉起 vLLM
3. 改为调：
   - `python3 .../phase4_repair_eval.py`

然后和当前稳定链路一样，紧跟着执行：

- [compute_eval_clean_overlay.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/compute_eval_clean_overlay.py)

生成：

- raw repair summary
- clean repair overlay
- raw first-pass summary
- clean first-pass overlay

再为 `delta69` 提供一个小 wrapper：

- `run_phase4_repair_delta69_checkpoint.sh`

和当前：

- [run_grpo_codecontests_delta69_checkpoint.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/run_grpo_codecontests_delta69_checkpoint.sh)

一样，只替换：

- manifest 路径
- output base

---

## 11. 实施顺序

### Milestone 1：最小可运行骨架

目标：

- 对 `valid_big500` 跑通 one-turn repair 协议
- 不要求先把输出格式做到最漂亮

动作：

1. 新增 `repair_feedback.py`
2. 新增 `phase4_repair_eval.py`
3. 从 `phase0_eval.py` 复用生成和判题
4. 增加简单 `repair_summary.json`
5. 同时落 first-pass per-problem 和 first-pass clean overlay

### Milestone 2：输出与指标补齐

目标：

- 补齐 per-problem 双阶段记录
- 补齐 repair metrics

动作：

1. 扩充 per-problem schema
2. 保存：
   - `metrics.json`
   - `repair_metrics.json`
   - `summary.json`
   - `repair_summary.json`

### Milestone 3：delta69 + valid_big500 双跑

目标：

- 对同一 checkpoint 跑：
  - baseline 单轮
  - one-turn repair

并输出最终对照：

- `accepted@1`
- `accepted after 1 repair`
- `repair gain`
- `conditional repair success`

---

## 12. 当前不做的事

Step 1 当前明确**不做**：

- trainer-side repair rollout
- `AgentLoopManager` 扩展
- `RepairOneTurnAgentLoop`
- repair-aware RL
- repair-conditioned SFT
- repair-distilled single-turn SFT
- test split repair eval

原因很简单：

- 当前只需要回答 repair 在开发评测上有没有价值

---

## 13. 最终建议

当前 one-turn repair 的正确实现路径是：

1. 继续使用当前 standalone eval 的 rollout server
2. 不修改 PPO trainer 和 rollout worker
3. 新增一个 repair eval 入口
4. 复用当前 verifier 和 per-problem logging
5. 先在：
   - `valid_big500`
   - `delta69`
   上完成开发评测
6. 从 Day 1 就同时保留：
   - raw
   - clean overlay

一句话总结：

**Step 1 是一个“评测协议扩展”，不是一个 trainer/rollout 改造项目。**
