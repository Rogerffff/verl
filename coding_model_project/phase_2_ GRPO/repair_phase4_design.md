# Phase 4 Repair 设计方案（草案，当前只落地第 1 步）

本文档描述 Phase 4 的整体设计，但**当前真正需要落地的只有第 1 步：开发评测版 one-turn verifier-guided repair**。  
后续的 repair data mining、repair-conditioned SFT、repair-distilled single-turn SFT 都先停留在设计层，不在当前实现范围内。

Step 1 的实现级方案见：

- [phase4_step1_repair_eval_implementation_plan.md](phase4_step1_repair_eval_implementation_plan.md)

---

## 1. 当前判断

经过当前这轮 curriculum RL：

- 单轮 RL 的 broad gain 已经出现
- `valid_big500` 上能看到整体 `pass_ratio_mean` 提升
- 但 solve 增益不够稳定，存在明显 churn
- `delta69` 上还能看到 retention 漂移

这意味着下一阶段的主矛盾已经不再是“继续看更多 unseen”，而是：

**如何把当前 RL 拉起来的 partial correctness / near-miss，进一步转成最终 AC。**

因此，最自然的下一步不是直接切到重型 multi-turn RL，而是先做：

**one-turn verifier-guided repair**

---

## 2. 总体路线

Phase 4 整体分成 4 层，但当前只做第 1 层。

### Step 1. 开发评测版 one-turn repair

用途：
- 在开发验证集上判断 repair 值不值得做

数据：
- `valid_big500`
- `delta69`

目标：
- 计算 `accepted after 1 repair`
- 计算 `repair gain`
- 计算 `conditional repair success`
- 看 repair 对哪些 failure bucket 最有效

**这是当前唯一需要真正实现的部分。**

### Step 2. 训练集 repair data mining

用途：
- 从训练集里挖出“初轮失败，但一轮 repair 后成功”的样本

注意：
- 这一步不是开发评测
- 这一步也不是最终要立刻上线
- 只是后续如果 Step 1 证明 repair 值得做时，作为 SFT 数据生成流程

### Step 3. repair-conditioned SFT

训练目标：
- `题目 + 错代码 + verifier 反馈 -> 修正后的完整代码`

主要提升：
- 看到反馈后修错代码的能力

### Step 4. repair-distilled single-turn SFT

训练目标：
- `题目 -> repair 后最终 verified AC 代码`

主要提升：
- 单轮 `accepted@1`
- 单轮一次做对能力

---

## 3. 最重要的概念区分

这一部分必须先讲清楚，否则后面所有 agent 都容易把口径搞混。

### 3.1 one-turn repair 评测

这是**评测协议**，不是训练。

形式：
1. 给模型题目
2. 生成第一版代码
3. 跑 verifier
4. 如果失败，给一段短反馈
5. 再让模型修一次
6. 再跑 verifier

这里我们看的是：
- `accepted@1`
- `accepted after 1 repair`
- `repair gain`

这一步可以用：
- `valid_big500`
- `delta69`

### 3.2 repair-conditioned SFT

这是**训练任务**。

输入：
- 题目
- 初始错误代码
- verifier 反馈

输出：
- 修正后的正确代码

它主要训练的是：
- **收到反馈后修代码**

它不等于单轮一次做对能力。

### 3.3 repair-distilled single-turn SFT

这也是**训练任务**，但目标不同。

输入：
- 题目

输出：
- repair 后最终 verified AC 的代码

它更像：
- 把 `generate -> verify -> repair -> verify` 产生的高质量最终答案
- 蒸馏回单轮模型

它更直接优化：
- `accepted@1`

---

## 4. 数据 split 的职责

### 4.1 训练集

训练集的职责是：
- 未来如果 Step 1 有效，用来做 repair data mining
- 再用这些 mined 数据构造 SFT 数据

训练集**不能**直接替代开发验证集。

### 4.2 开发验证集

开发验证集的职责是：
- 评估 repair 值不值得做
- 评估 repair 给当前 checkpoint 带来的真实增益

当前优先用：
- `valid_big500`
- `delta69`

### 4.3 最终 test

最终 test 的职责是：
- 在 pipeline 定型之后做冻结评测

当前不应该拿 test 来决定：
- 继续 RL
- 还是转 repair

---

## 5. 当前只落地第 1 步的原因

当前不应该立刻上 repair-SFT 或 multi-turn RL，原因有 3 个：

1. 还没证明 one-turn repair 本身有足够收益
2. 还没回答“当前 RL 拉起来的 partial gain，有多少能通过一次修复转成 AC”
3. 如果在这个问题没回答前就上训练，后面收益归因会很混乱

所以当前最稳的顺序是：

1. 先做 Step 1：开发评测版 one-turn repair
2. 如果收益明显，再做 Step 2：训练集 data mining
3. 再决定是否进入 Step 3 / Step 4

---

## 6. Step 1 的具体设计（当前要实现的部分）

### 6.1 输入 checkpoint

Step 1 的输入应该是：
- 当前选中的 RL checkpoint

建议：
- Step 1 正式对照改为：
  - `step600`
  - `step900`
  - `step1000`

目的：

- `step600` 代表 single-shot solve winner
- `step900 / step1000` 代表更偏 partial / repair-ready 的候选点
- repair 阶段要比较的是：
  - `accepted@1`
  - `accepted after 1 repair`
  - `conditional repair success`
  而不是只看 partial winner

补充更新（2026-04-17）：

- `step900 / step1000 / step1300` 的比较现在主要保留为 **pre-SFT frontier 历史背景**
- 在 patched sandbox + direct-backend client RR 下，最新 Step 1 repair base 已经按两套协议重算：
  - Protocol A：固定 `step900` canonical raw `per_problem` 做 `reuse-first-pass`
  - Protocol B：各 checkpoint 自己 raw first-pass，再 self-repair

最新结论：

- Protocol A：
  - `step30 / 40 / 50 / 60` 全部 `74/500`
  - `step900` 为 `73/500`
  - 四个 SFT checkpoint 的 final accepted set 完全一致
  - 其中 `step40` 的 final `pass_ratio_mean` 最高
- Protocol B：
  - `step40 = 78/500`
  - `step50 = 77/500`
  - `step60 = 76/500`
  - `step30 = 75/500`

因此当前更合理的 Step 1 checkpoint 分层是：

- 主 one-turn repair base：
  - `step40`
- repaired partial-quality 次优备选：
  - `step50`
- frozen first-pass source / 历史 validated base：
  - `step900`
- partial-credit probe：
  - `step1300`

口径上：

- 如果目的是公平比较 repairability，优先看 Protocol A
- 如果目的是做部署式 / end-to-end base 选择，优先看 Protocol B

这里也要注意 source 口径：

- `step1300` 的 one-turn repair probe 必须绑定到**一份固定的 canonical raw `per_problem`**
- 可以选：
  - 高 `pass_ratio_mean` 的 first-run source
  - 或 `rerun1` source
- 但不能在同一轮实验里混用多个 `step1300` raw response 集

### 6.2 开发评测集

当前优先：
- `valid_big500`
- `delta69`

不要用：
- `test`

### 6.3 one-turn repair 的流程

对每道评测题：

1. 单轮生成代码
2. 跑 verifier
3. 如果第一轮已经 AC，直接记单轮成功
4. 如果失败，构造一条短 feedback
5. 再生成一次完整代码
6. 再跑 verifier

### 6.4 feedback 格式

只保留短而结构化的反馈，不要给太长的日志。

推荐保留：
- `passed_tests / total_tests`
- `error_type`
  - `wrong_answer`
  - `runtime_error`
  - `timeout`
- 一条代表性失败摘要
  - `wrong_answer`：优先更短、更干净的 counterexample
  - `runtime_error`：优先更清楚异常类型的 case，再选更短输入
  - `timeout`：优先最短 timeout case，并补固定复杂度提示
- 一句固定指令：
  - `Please output only the repaired full Python solution.`

### 6.5 repair 触发范围

不是所有失败题都值得 repair。

优先 repair：
- `pass_ratio > 0.2`
- near-miss
- mixed-case
- partial + RE/TLE

低优先级：
- `pass_ratio = 0`
- 明显思路错误
- 已经属于 `D_dead_hard` 的题

### 6.6 当前要汇报的指标

对每个 eval slice，都要同时报：

#### 单轮指标
- `accepted@1`
- `pass_ratio_mean`
- `wrong_answer_rate`
- `runtime_error_rate`
- `timeout_rate`

#### repair 指标
- `accepted after 1 repair`
- `repair gain`
  - `accepted_after_1_repair - accepted@1`
- `conditional repair success`
  - 初轮失败样本中，被一次 repair 救回来的比例
- `extra judge cost per repaired solve`

### 6.7 当前输出形式

Step 1 最终应该产出：

1. `valid_big500` 的单轮 vs repair 对照结果
2. `delta69` 的单轮 vs repair 对照结果
3. 一份按 failure bucket 的 repair 诊断

在 2026-04-09 之后，建议再额外补一类受控输出：

4. `step1300` 的 fixed-first-pass prompt ablation
   - `code-only`
   - `short-diagnosis+code`

它的目标不是重做 base selection，而是回答：

- “如果 first-pass 换成当前最强 partial-credit checkpoint，repair success 会不会进一步提高？”

---

## 7. 后续设计：Step 2 训练集 data mining（先不实现）

如果 Step 1 证明 repair 值得做，下一步才进入训练集 data mining。

### 7.1 数据来源

来源必须是：
- **训练集**

不能直接拿：
- `valid_big500`
- `delta69`
- `test`

### 7.2 只挖哪类题

优先从训练集里挖：
- `B_near_miss`
- promising `C`
- `pass_ratio > 0.2`
- 初轮失败但 repair 后成功

不优先：
- `0-pass`
- 明显 `D`

### 7.3 输出两类样本

#### A. repair-conditioned 样本

输入：
- 题目
- 初始错误代码
- verifier 反馈

输出：
- 短诊断 + 修正后的完整代码

当前更推荐的具体格式是：

- `BUG_SUMMARY`：最多 1 句
- `FIX_PLAN`：最多 1 句
- `<code>...</code>`：最终完整修复代码

原因是 2026-04-09 的 `step900 + valid_big500 + reuse-first-pass` prompt ablation 显示：

- clean `short_diagnosis_code`:
  - `66 / 500`
  - `conditional_repair_success = 16.44%`
- clean `code_only`:
  - `63 / 500`
  - `conditional_repair_success = 12.33%`

所以当前 Step 3 数据收集更适合优先保留：

- `short_diagnosis+code`

同时继续保留：

- `code-only`

作为对照 instruction family，而不是完全删除。

最初的假设是：

- 如果 `step1300` 上的 fixed-first-pass one-turn repair 也继续支持这一点
- 那么 `short_diagnosis+code` 就不再只是 `step900` 局部切片上的正信号
- 而会更接近对 “repair-ready / near-miss heavy checkpoint” 普遍成立的格式偏好

但 2026-04-09 的 `step1300 + valid_big500 + reuse-first-pass` probe 已经给出了相反结果：

- `step1300 code_only`
  - `61 -> 65 / 500`
  - `conditional_repair_success = 6.06%`
- `step1300 short_diagnosis_code`
  - `61 -> 65 / 500`
  - `conditional_repair_success = 6.06%`

所以当前更准确的说法是：

- `short_diagnosis+code` 在 `step900` 这类已验证 repair-ready checkpoint 上有明确正信号
- 但这个优势目前还没有迁移到 `step1300` 这种 high-partial checkpoint
- 因此它仍然适合作为默认优先 prompt mode
- 但要继续把它视为 **checkpoint-dependent** 的工程选择，而不是普适结论

#### B. repair-distilled single-turn 样本

输入：
- 题目

输出：
- repair 后最终 verified AC 的代码

---

## 8. 为什么要同时设计两种 SFT

因为它们优化的是两个不同目标。

### repair-conditioned SFT

优化的是：
- 收到反馈后的修复能力

适合汇报成：
- verifier-guided self-repair

### repair-distilled single-turn SFT

优化的是：
- 单轮一次做对能力
- `accepted@1`

适合汇报成：
- 利用 repair pipeline 产生的 verified AC 数据，反蒸馏回单轮模型

---

## 9. 最终项目指标该怎么分

后续如果 Step 2/3/4 真的做起来，项目应该保留两套指标。

### 9.1 单轮账本

看：
- 单轮 `accepted@1`
- 单轮 `pass_ratio_mean`

### 9.2 repair 账本

看：
- `accepted after 1 repair`
- `conditional repair success`
- `repair gain`

不要把两套口径混成一个指标，否则后面很难解释：
- 模型本身是不是更强了
- 还是 repair pipeline 更强了

---

## 10. 当前拍板

当前只拍板以下内容：

### 立即进入实现
- Step 1：开发评测版 one-turn verifier-guided repair

在当前阶段，这条实现线应继续细化成两类：

- 主线：
  - `step40` 为当前主 one-turn repair base
  - `step50` 为当前更偏 repaired partial-quality 的次优备选
- 新增受控 probe：
  - Protocol A 下继续固定 `step900` 的 `FIRST_PASS_PER_PROBLEM`
  - Protocol B 下继续比较各新 checkpoint 的 end-to-end repair 能力
  - `step1300` 仍只保留为 partial-credit / near-miss probe

补充更新（2026-04-17）：

- `step900` 的历史优势已经被更晚的 SFT checkpoint 修正
- 当前正式结论应更新为：
  - `step40` 是新的 one-turn repair 主 base
  - `step50` 更适合作为 repaired partial-quality / pass-ratio 备选
  - `step900` 保留为 frozen source 与历史 validated base
  - `step1300` 继续只作为 partial-credit / near-miss 数据源候选

### 只做设计，不实现
- Step 2：训练集 repair data mining
- Step 3：repair-conditioned SFT
- Step 4：repair-distilled single-turn SFT
- multi-turn RL / multi-turn GRPO

---

## 11. 一句话总结

Phase 4 当前的正确推进顺序是：

**先证明 one-turn repair 值不值得做，再考虑用训练集 repair 成功样本去做 SFT。**

并且之后要明确区分：

- **repair-conditioned SFT**：提升“看到反馈后修代码”的能力
- **repair-distilled single-turn SFT**：提升“单轮一次做对”的能力

当前还不应该直接跳到 multi-turn RL。
