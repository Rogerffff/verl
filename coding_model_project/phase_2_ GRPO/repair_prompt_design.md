# Repair Prompt Design

本文档把当前 Step 1 one-turn repair 的 prompt 设计重新整理成**可直接落地**的版本。

目标不是讨论所有可能的 prompt trick，而是回答当前最实际的问题：

1. 另一位 agent 基于 `FeedbackEval` / `LeDex` 给出的建议，哪些值得吸收
2. 哪些建议不能直接照搬到当前代码库
3. 当前应该如何设计两组 prompt
   - `code-only repair`
   - `short-diagnosis+code`

这份文档默认面向当前项目的实际约束：

- base model: `Qwen2.5-Coder-7B-Instruct`
- task: `CodeContests` 风格完整程序修复
- verifier: 结构化 testcase feedback
- extraction: 当前项目**强依赖 `<code>...</code>`**
- Step 1 评测协议: **reuse-first-pass**

---

## 1. 结论先行

当前建议可以参考，但要做三层收紧：

1. 可以吸收“先诊断再修”的思路
2. 不建议把 `<think>...</think>` 当主格式
3. 必须保留当前工程链最稳定的 `<code>...</code>` 输出约束

所以当前最合理的设计不是：

- `long free-form reflection`
- `hidden thinking tag`

而是：

- `code-only repair`
- `short-diagnosis+code`

其中：

- `code-only repair` 作为主线评测模板
- `short-diagnosis+code` 作为小规模 ablation 模板

---

## 2. 哪些建议值得吸收

### 2.1 来自 `FeedbackEval` 的可吸收点

当前最值得吸收的不是某条“神奇 prompt”，而是这几个方向：

- repair prompt 需要保留任务语义，不应只剩错误日志
- 结构化、grounded 的 feedback 有价值
- 结构化 reasoning 比自由发挥式长反思更可控
- one-turn repair 本身是合理的第一阶段

对应到当前项目，就是：

- repair prompt 里必须保留原始题面
- feedback 继续以 testcase / error_type 为主
- 如果要加“反思”，就加短结构化诊断，不加长自由文本

### 2.2 来自 `LeDex` 的可吸收点

`LeDex` 最值得吸收的是：

- `wrong solution + execution feedback -> refinement` 这条训练目标是成立的
- explanation + refinement 可以作为一种独立 instruction family
- direct refinement 和 explanation+refinement 最好视作两类数据，而不是二选一

对应到当前项目，就是：

- Step 3 的方向本身没有问题
- 当前先做 `code-only` vs `short-diagnosis+code` 是合理的
- 如果后面真的做 repair-conditioned SFT，最好保留两种格式，而不是只收一种

---

## 3. 哪些建议不能直接照搬

### 3.1 不直接采用 `<think>...</think>`

不建议把 `<think>` 作为主格式，原因不是“模型完全不会”，而是：

- 当前项目真正要验证的是“短诊断是否有帮助”
- 不是“特殊标签是否有帮助”
- 对当前 extraction / judge pipeline，`<think>` 会平白增加输出不稳风险

而且当前模型是 `Qwen2.5-Coder-7B-Instruct`，不是带明确 thinking-mode 控制的 `Qwen3` 系列。

所以当前应该把问题收敛成：

- 是否需要短诊断字段

而不是：

- 是否需要 `<think>` 标签

### 3.2 不直接去掉 repair 身份 framing

另一位 agent 提醒“不要复用 baseline prompt 包装”，这个方向是对的。  
但不能走到另一个极端：把 repair prompt 退化成“只有错代码 + 测试反馈”。

当前 repair prompt 仍然应该保留：

- 题目语义
- repair assistant 身份
- 输出格式约束

否则容易损失当前 baseline prompt 里已经比较稳的任务 framing。

### 3.3 不改变 `<code>` 主输出协议

虽然很多论文示例用 markdown code fence，但当前代码库最稳的提取规则仍是：

- 优先 `<code>...</code>`

位置见：

- [phase0_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py)
- [verifier/shared.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py)

因此 prompt 设计必须服从当前工程现实：

- repaired code 继续包在 `<code>...</code>` 里
- 任何额外文本都必须在 `<code>` 外，且结构固定
- `<code>` 应尽量保留给最终输出专用

也就是说：

- 待修的旧代码，优先放在普通 fenced code block 里
- 最终答案，继续要求放在 `<code>...</code>` 里

这样更容易把“待修对象”和“最终要抽取的答案”分离开。

---

## 4. 当前设计原则

### 4.1 保留原始题面

当前 repair prompt 用：

- 原始题面
- 错误代码
- verifier feedback

不要用：

- baseline `format_prompt()` 包完后的整段指令化 prompt

原因：

- 避免双层 instruction
- 但又保留足够的任务语义

### 4.2 feedback 只保留 grounded signal

当前 feedback 继续优先保留：

- `passed_tests / total_tests`
- `error_type`
- `representative_failure`
- `timeout complexity hint`

不建议额外加入：

- 臆测式算法建议
- 模型自己猜的高层错误定位
- 太长的自然语言点评

另外：

- `selection_strategy` 对实验记录有用，但默认不放进模型可见 prompt
- 它应保留在日志、JSONL、summary 里，用于分析 testcase 选择规则
- 模型真正需要看的，是 failure content 本身，而不是“这条 case 是怎么选出来的”

### 4.3 诊断字段必须短、可控、可抽取

如果要做 “diagnosis + code”：

- `BUG_SUMMARY` 最多 `1` 句
- `FIX_PLAN` 最多 `1` 句
- 必须 grounded in feedback
- 不允许长链式推理

这样做的目的不是让模型“展示思考”，而是：

- 给一个短中间变量
- 看它是否能帮助 repair
- 同时不大幅增加 extraction 风险

### 4.4 当前优先支持的错误类型

基于当前 Step 1 结果，prompt 设计的小消融应优先看：

- `wrong_answer`
- `runtime_error`

`timeout` 先保留在主协议里，但不建议让它主导这轮 prompt ablation 结论。

对应到 feedback 呈现形式，建议默认使用：

- `wrong_answer`: `stdin + expected + actual`
- `runtime_error`: `stdin + exception / stderr 关键信息 + message`
- `timeout`: `stdin + timeout message + complexity hint`

其中 `wrong_answer` 默认不展示 `stderr`，除非后续单独证明它能稳定带来增益。

---

## 5. 推荐的系统提示

当前建议把 repair system prompt 收敛成下面这版。

```text
You are an expert Python competitive-programming repair assistant.

You will be given:
1. the original problem statement,
2. a previously generated incorrect Python solution,
3. structured verifier feedback.

Your job is to repair the solution so that it satisfies the problem requirements and passes more tests.

Rules:
- Focus on fixing the logic using the given feedback.
- Keep the solution self-contained and executable in Python.
- The program must read from stdin and write to stdout.
- Return a complete repaired solution, not a patch or diff.
- Follow the exact output format requested by the user prompt.
```

设计动机：

- 保留 repair assistant 的角色 framing
- 明确当前是“修复完整程序”，不是写 patch
- 不在 system prompt 里硬塞“不要解释”，因为这件事应由不同 user prompt 组控制

---

## 6. Group A: `code-only repair`

### 6.1 适用定位

这组作为：

- 当前 Step 1 repair 主线模板
- extraction 最稳的默认模板
- 后续 fallback repair 的默认生产格式候选

### 6.2 推荐 user prompt

```text
Original problem statement:
{problem_statement}

Previous incorrect Python solution:
```python
{first_code}
```

Verifier feedback:
- passed_tests: {passed_tests}/{total_tests}
- error_type: {error_type}
{failure_summary_block}
{optional_error_specific_hint_block}

Please repair the solution using the feedback above.

Requirements:
- Return a complete program that reads from stdin and writes to stdout.
- Output only the repaired solution.
- Put the entire final answer inside a single <code>...</code> block.

Format your final answer as:
<code>
...your repaired Python program...
</code>
```

### 6.3 设计要点

这版最重要的是：

- 保留原始题面
- 明确指出 previous incorrect solution
- feedback 保持结构化
- 输出约束极强

这里的 `failure_summary_block` 应按当前 error type 分开填：

`wrong_answer`

```text
- representative_failure:
  - stdin: ...
  - expected: ...
  - actual: ...
```

`runtime_error`

```text
- representative_failure:
  - stdin: ...
  - exception: ...
  - message: ...
```

`timeout`

```text
- representative_failure:
  - stdin: ...
  - message: ...
```

`optional_error_specific_hint_block` 只保留 grounded hint，例如 timeout 的复杂度提醒。

### 6.4 这组的作用

这组的目标是回答：

- 在固定 first-pass 上，模型是否能仅靠 verifier feedback 修代码

它是当前最应该优先保留的 repair prompt 版本。

---

## 7. Group B: `short-diagnosis+code`

### 7.1 适用定位

这组不是主线替代，而是：

- 一个小规模 prompt ablation
- 用来判断 Step 3 是否值得显式保留诊断字段

### 7.2 推荐 user prompt

```text
Original problem statement:
{problem_statement}

Previous incorrect Python solution:
```python
{first_code}
```

Verifier feedback:
- passed_tests: {passed_tests}/{total_tests}
- error_type: {error_type}
{failure_summary_block}
{optional_error_specific_hint_block}

Please first give a very short diagnosis of the bug and a very short fix plan, then provide the repaired full Python solution.

Requirements:
- BUG_SUMMARY must be grounded in the provided feedback.
- FIX_PLAN must describe only the intended repair direction.
- BUG_SUMMARY must be at most 1 sentence.
- FIX_PLAN must be at most 1 sentence.
- Do not include any other sections.
- The repaired code must be complete and executable.
- Put only the final repaired program inside a single <code>...</code> block.

Use exactly this format:
BUG_SUMMARY: ...
FIX_PLAN: ...
<code>
...your repaired Python program...
</code>
```

### 7.3 设计要点

这组和 `code-only` 的区别，不是让模型写长 reasoning，而是只多两小段：

- `BUG_SUMMARY`
- `FIX_PLAN`

要求必须非常短，原因有 3 个：

1. 避免变成长自由反思
2. 避免污染 extraction / judge
3. 让它将来更容易直接转成 Step 3 数据格式

### 7.4 诊断字段的好坏标准

好的 `BUG_SUMMARY / FIX_PLAN` 应该像：

- `BUG_SUMMARY: The solution is close, but it mishandles a boundary case where equal values should be treated differently.`
- `FIX_PLAN: I will correct the boundary condition and keep the rest of the logic unchanged.`

不好的形式是：

- 长篇推理
- 猜测性很强的算法分析
- 和给定 feedback 无关的泛泛建议

---

## 8. 两组 prompt 的最终关系

当前不建议二选一拍脑袋决定，而建议这样使用：

### 主线

- `code-only repair`

### 小消融

- `short-diagnosis+code`

推荐先只在下面这个切片上比较：

- checkpoint: `step900`
- dataset: `valid_big500`
- error types: `wrong_answer + runtime_error`
- trigger: `pass_ratio >= 0.6`

这样最容易回答“短诊断是否真的有净收益”。

补充：

- `step900` 是这轮 prompt ablation 最初使用的**历史 validated base**
- 但在 2026-04-17 的 Protocol A / Protocol B 更新后，当前主 repair base 已经切到 `step40`
- `step50` 是当前更偏 repaired partial-quality 的次优点
- `step1300` 继续只保留为 **partial-credit probe checkpoint**
- 因此后续很值得再加一轮更小的 follow-up：
  - checkpoint: `step1300`
  - protocol: `reuse-first-pass`
  - prompts: `code-only` vs `short-diagnosis+code`
  - 目的不是重选 exact-solve winner，而是回答：
    - “当前最强 near-miss / pass-ratio checkpoint 上，短诊断是否仍然有净收益？”

---

## 9. 当前的判定指标

这轮 prompt ablation 不要只看最终 solve 数，还要同时看：

- `accepted_after_1_repair`
- `repair_gain`
- `conditional_repair_success`
- `bucket_0.6_1.0` 的成功率
- `pass_ratio_mean_delta`
- `extraction_status`
- 平均额外 token 成本

如果 `short-diagnosis+code` 满足下面任一条件，就值得继续保留到 Step 3 设计里：

- 比 `code-only` 多救回 `2-3` 题
- 或在高 partial bucket 上成功率更高
- 或能明显减少大幅 regression / over-repair

如果没有明显收益，就不要为了“看起来更聪明”而保留它。

---

## 10. 当前项目的最终拍板

### 当前可以参考的部分

- 参考 `FeedbackEval` 的结构化 repair prompt 思路
- 参考 `LeDex` 的 explanation + refinement instruction family

### 当前不直接采用的部分

- 不把 `<think>...</think>` 作为主格式
- 不把长自由反思引入 Step 1
- 不去掉 `<code>` 输出协议

### 当前应该执行的版本

1. 保留 `code-only repair` 作为主线模板
2. 新增 `short-diagnosis+code` 作为小消融模板
3. 如果后续进入 Step 3 repair-conditioned SFT，优先考虑同时保留两类样本

checkpoint 口径也应区分成两层：

- `step40`：当前已验证的主 one-turn repair base
- `step50`：当前 repaired partial-quality 次优备选
- `step900`：Protocol A frozen source / 历史 validated base
- `step1300`：当前最强 partial-credit checkpoint，用于 near-miss / repairability probe

---

## 11. 2026-04-09 最新 prompt ablation 结果

在两组 prompt 真正接入代码后，基于下面这条固定切片做了一轮小消融：

- checkpoint: `step900`
- dataset: `valid_big500`
- protocol: `reuse-first-pass`
- trigger: `pass_ratio >= 0.6`
- error types: `wrong_answer + runtime_error`

### 11.1 一个需要先记住的事实

首轮 `code-only` ablation run 后来确认受 sandbox 掉线污染：

- `api_error_rate = 0.036`
- `per_problem` 中有 `18` 条 `api_error`
- eval log 中出现大量 `Connection refused`

因此旧的 `code-only` 首轮结果应作废，不能拿来和 `short_diagnosis+code` 做正式比较。

之后在 sandbox 恢复后，只重跑了 `code-only`，得到 clean rerun。

### 11.2 当前正式可比的结果

两组正式对照都使用同一份 canonical first-pass：

- first pass:
  - `54 / 500`
  - `accepted@1 = 0.108`
  - `pass_ratio_mean = 0.3473`

#### clean `code-only`

- after repair:
  - `63 / 500`
  - `accepted@1 = 0.126`
  - `pass_ratio_mean = 0.3425`
- repair:
  - `repair_attempt_count = 73`
  - `repair_success_count = 9`
  - `conditional_repair_success = 12.33%`

#### clean `short-diagnosis+code`

- after repair:
  - `66 / 500`
  - `accepted@1 = 0.132`
  - `pass_ratio_mean = 0.3336`
- repair:
  - `repair_attempt_count = 73`
  - `repair_success_count = 12`
  - `conditional_repair_success = 16.44%`

### 11.3 对这轮结果的解释

这轮结果说明：

1. `short-diagnosis+code` 在当前切片上是有正信号的
   - 相比 clean `code-only`
   - 多 `3` 个 solve
   - 多 `3` 次 repair success

2. 这个信号不是由 first-pass 漂移造成的
   - 因为两组都复用了同一份 raw `per_problem`
   - first-pass 已完全固定

3. 这个信号目前仍然是“局部成立”
   - 只在：
     - `step900`
     - `valid_big500`
     - `WA/RE`
     - `pass_ratio >= 0.6`
     这一切片上被验证

所以更准确的说法不是：

- `short-diagnosis+code` 已经全面优于 `code-only`

而是：

- `short-diagnosis+code` 已经在当前最关键的 Step 1 repair slice 上显示出足够明确的正信号，值得作为后续 Step 3 数据格式优先候选

### 11.4 当前 prompt 主线建议

截至这轮 clean ablation，建议更新为：

- 默认优先格式：
  - `short-diagnosis+code`
- 保留的基线格式：
  - `code-only`

也就是说：

- `code-only` 不删
- 但后续 Step 3 数据设计、repair-conditioned 样本收集、以及下一轮 repair prompt 实验，应优先围绕 `short-diagnosis+code` 展开

### 11.5 基于 2026-04-09 fixed-response rejudge 的后续实验建议

由于 `step1300` 现在已被重新定性为：

- 不是 exact-solve winner
- 但大概率是当前最强的 `pass_ratio_mean` / partial-credit checkpoint

所以下一轮最值得补的 prompt 实验是：

- checkpoint: `step1300`
- dataset: `valid_big500`
- protocol: `reuse-first-pass`
- source first-pass: 固定使用单一 canonical raw `per_problem`
- prompts:
  - `code-only`
  - `short-diagnosis+code`

这里必须继续坚持：

- 不重跑 first-pass generation
- 不混用多个 `step1300` raw run 的 response
- 先固定一份 source `per_problem`，再只比较 repair prompt 本身

推荐做法是二选一并固定：

- 如果目标是测试“当前最强 partial-credit frontier 的 repairability”，优先固定到 `step1300` 那条高 `pass_ratio_mean` 的 first-run raw `per_problem`
- 如果目标是和晚期 checkpoint 的 official rerun 口径严格对齐，则固定到 `step1300 ... rerun1` 的 `per_problem`

但两者不要混在同一轮 prompt ablation 里。

如果目标是验证“partial-credit winner 是否更 repairable”，那么更推荐把 `step1300` 的 canonical source 选成：

- 那个 `pass_ratio_mean` 最高、并且已被 fixed-response rejudge 支持其 partial 质量真实存在的 raw run

而不是再让一次新的 full rerun 决定输入样本。

### 11.6 `step1300` prompt ablation 已完成：它没有取代 `step900` 成为更好的 repair base

这轮后续实验现在已经完成，口径是：

- checkpoint:
  - `step1300`
- dataset:
  - `valid_big500`
- protocol:
  - `reuse-first-pass`
- source:
  - 固定使用高 `pass_ratio_mean` 的 first-run raw `per_problem`
- trigger:
  - `wrong_answer + runtime_error`
  - `pass_ratio >= 0.6`

结果如下：

#### `step1300 code-only`

- first pass:
  - `61 / 500`
  - `accepted@1 = 0.122`
  - `pass_ratio_mean = 0.357553`
- after repair:
  - `65 / 500`
  - `accepted@1 = 0.130`
  - `pass_ratio_mean = 0.348504`
- `repair_attempt_count = 66`
- `repair_success_count = 4`
- `conditional_repair_success = 6.06%`

#### `step1300 short-diagnosis+code`

- first pass:
  - `61 / 500`
  - `accepted@1 = 0.122`
  - `pass_ratio_mean = 0.357553`
- after repair:
  - `65 / 500`
  - `accepted@1 = 0.130`
  - `pass_ratio_mean = 0.336978`
- `repair_attempt_count = 66`
- `repair_success_count = 4`
- `conditional_repair_success = 6.06%`

对比当前 `step900` 同切片结果：

- `step900 code-only`
  - `54 -> 64 / 500`
  - `conditional_repair_success = 13.70%`
- `step900 short-diagnosis+code`
  - `54 -> 66 / 500`
  - `conditional_repair_success = 16.44%`

因此当前 prompt 与 checkpoint 的正式建议更新为：

1. `step1300` 的 first-pass 仍然更强
   - 它继续支持“partial-credit winner”这一定位

2. 但 `step1300` 没有更强的 repairability
   - 它在这条受控 slice 上只救回 `4` 题
   - 明显弱于 `step900` 的 `10~12` 题 rescue

3. `short-diagnosis+code` 的优势没有迁移到 `step1300`
   - solve 数与 `code-only` 持平
   - `pass_ratio_mean` 还更低

4. 这条 `step900 vs step1300` 结论现在只保留为 pre-SFT 历史对照
   - 当前正式拍板应更新为：
   - repair 主 base：`step40`
   - repaired partial-quality 次优备选：`step50`
   - 默认优先 prompt mode：`short-diagnosis+code`
   - `step900`：Protocol A frozen source / 历史 validated base
   - `step1300`：partial-credit / near-miss probe checkpoint，不是新的 one-turn repair 主 base

### 11.7 2026-04-17 最新 checkpoint-base 更新

在 short-diagnosis SFT 后，使用 patched sandbox + direct-backend client RR 又补完了两套新评测：

#### Protocol A：固定输入 repair eval

- frozen source：
  - `step900 valid_big500 raw patched_rr828x` canonical `per_problem`
- 结果：
  - `step900`：`73/500`
  - `step30 / step40 / step50 / step60`：全部 `74/500`
- 解释：
  - 四个 SFT checkpoint 的 final accepted set 完全一致
  - 在这一组并列里，`step40` 的 final `pass_ratio_mean` 最高

#### Protocol B：end-to-end repair eval

- 每个 checkpoint 先自己 raw first-pass，再 self-repair
- 结果：
  - `step40 = 78/500`
  - `step50 = 77/500`
  - `step60 = 76/500`
  - `step30 = 75/500`

因此当前 prompt 设计文档里的正式 base 口径应更新为：

- 主 one-turn repair base：`step40`
- repaired partial-quality 次优备选：`step50`
- Protocol A frozen source / 历史 validated base：`step900`
- 默认优先 prompt mode 继续保持：
  - `short-diagnosis+code`

---

## 12. 后续实现建议

当需要把这两组 prompt 接进代码时，建议直接拆成两个 builder：

```python
def build_code_only_repair_prompt(...) -> str:
    ...

def build_diagnosis_then_code_repair_prompt(...) -> str:
    ...
```

两者共用同一个 feedback dict：

```python
{
    "passed_tests": int,
    "total_tests": int,
    "error_type": str,
    "selection_strategy": str,
    "selected_failures": list,
    "extra_hint": str,
}
```

其中：

- `selection_strategy` 保留给日志、per-problem 和 summary
- 默认 builder 不把 `selection_strategy` 渲染进模型可见 prompt

当前最重要的不是先把两组都接入主线，而是：

- 先保留当前 `code-only` 稳定实现
- 再把 `short-diagnosis+code` 做成一个可控 ablation 分支

这样最符合当前项目阶段，也最容易解释后续结果。
