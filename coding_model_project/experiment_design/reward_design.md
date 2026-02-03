
---

## 1) `rratio = -0.3 + 1.3*pass_ratio` 的来源：RLTF 的 Adaptive Feedback（不是你 PDF 自己发明的）

在 **RLTF（Reinforcement Learning from Unit Test Feedback）**里，作者定义了一个 **Adaptive Feedback**，形式就是：

* 令 `pass_ratio = N_pass / (N_pass + N_fail)`
* `F_adaptive = -0.3 + 1.3 * pass_ratio` 

所以你现在用的 `-0.3` 和 `1.3` **是有论文原型的**：它保证

* `pass_ratio = 0` 时 reward = `-0.3`
* `pass_ratio = 1` 时 reward = `+1.0` 

额外一个很有用的性质（你面试讲得出来）：
`F_adaptive > 0` 的阈值是 `pass_ratio > 0.3/1.3 ≈ 0.2308`，也就是说 **必须过掉约 23% 的测试点才开始给正反馈**，这能抑制“瞎写也能拿到一点点正分”的噪声学习（尤其在 GRPO 的 group normalization 下很关键）。

---

## 2) “错误类型 piecewise 惩罚”的依据：RLTF 的 Coarse-Grained Feedback（直接给了那组 -0.3/-0.6/-1）

RLTF 同时给了一套 **粗粒度的错误分段奖励**（Coarse-grained feedback）：

* **Pass**：`+1`
* **Failure（能跑但没通过测试）**：`-0.3`
* **Error（非语法类错误，如 runtime/timeout 等）**：`-0.6`
* **Syntax Error**：`-1.0`

这就是你想做的 “error-type piecewise 惩罚” 最硬的依据：**论文原文就给了这个 piecewise 标尺**。

> 你要“避免有问题的奖励函数”，这里的关键点是：**别把 rratio 和 rerr 简单相加导致重复惩罚**（下面我给你推荐写法）。

---

## 3) `rlen`（长度惩罚）是否有依据？——“一般 token 长度惩罚”在 coding RL 里并不主流，但“截断/超长要特殊处理”非常有依据

### 3.1 你写的 `rlen`：如果是“按 token 越长越扣分”，**论文里并没有一个通用共识模板**

你能在一些 RLHF/对齐研究里看到“长度/verbosity 导致 reward hacking，需要 reward shaping 约束”的讨论（比如强调 *reward 应当有界、并警惕通过冗长输出钻空子*）([arXiv][1])，但这类结论**更多来自 RM 场景**，并不是“coding unit-test reward 必须扣 token 长度”。

更接近你场景的“大厂/大规模 RL”经验是：**不是对正常长度做惩罚，而是对“超长导致截断/不完整”做特殊处理**。

### 3.2 “截断/超长要特殊处理”——这点我强烈建议你做，而且有多篇工作支撑

**(a) RLTF 直接点名：max token 截断导致的“不完整代码”会表现成 Syntax Error，需要特别对待**
RLTF 里专门提到：由于**最大 token 限制**导致生成被截断，代码“不完整”，会触发语法错误；他们把这类情况单独归类到全局反馈里处理（不要当成普通语法错乱来学）。

**(b) DeepCoder（工业博客）明确写：他们做了 overlong filtering + truncation masking（而且说灵感来自 DAPO）**
DeepCoder 的训练实践里，专门强调对 **overlong responses** 做过滤，并对截断样本做 **masking**（避免把“截断副作用”当成学习信号）。([Together AI][2])

**(c) DAPO 给了系统化方案：Overlong Reward Shaping + Overlong Filtering**
DAPO 明确把“过长/截断”当成训练噪声来源，提出 **Overlong Reward Shaping** 和 **Overlong Filtering** 来稳定训练。([arXiv][3])

**(d) 另一个训练系统论文给出非常直接的工程规则：Truncation masking（截断样本 advantage 置 0）**
有工作直接建议：对“被截断的样本”做 **truncation masking**，把它们的 advantage 设为 0，避免产生误导梯度。

> 所以回答你之前那句：“(4.6 截断/超长要特殊处理) 要不要做？”
> **要做，而且优先级比“rlen 扣分”更高。**

---

## 4) 我建议你把 Dense-linear 写成“论文对齐版”：用 **piecewise override**，别 `rratio+rerr` 叠加重复惩罚

你现在写的是：
`rdense-linear = clip(rratio + rerr + rlen, [-1,1])`

更稳、更对齐 RLTF 的写法是（核心思想：**遇到严重错误就直接覆盖 reward**）：

```text
if truncated_by_max_tokens:
    reward = MASK(advantage=0)   # 或 soft punish（DAPO）
elif syntax_error:
    reward = -1.0
elif runtime_error_or_timeout:
    reward = -0.6
else:
    pass_ratio = passed_tests / total_tests
    reward = -0.3 + 1.3*pass_ratio
reward = clip(reward, -1, 1)
```

* `-0.3 + 1.3*pass_ratio`：来自 RLTF adaptive feedback 
* `-1/-0.6` 的错误分段：来自 RLTF coarse feedback
* `truncation`：建议 masking/过滤/soft punish，来自 RLTF + DeepCoder + DAPO + truncation masking 实践 ([Together AI][2])
* `clip[-1,1]`：RLTF 本身 reward 就落在 [-1,1] 这类有界范围内（-1 到 +1），你 clip 是一致的；且“reward 有界”也是 reward shaping 的常见原则（避免极端值/过优化导致 hacking）。([arXiv][1])

---

## 5) 你要做的 4 个 reward 消融：逐个给“计算过程例子”（按同一题的不同输出）

假设某题 `total_tests=10`。

### (A) Sparse：`r = 1[AC]`

**例 1：过了 6/10（非 AC）**

* `AC=0` → `r=0`

**例 2：10/10（AC）**

* `AC=1` → `r=1`

> DeepCoder 类方案偏好这种 outcome-only sparse，以减少 partial reward 被“钻空子”的风险。([Together AI][2])

---

### (B) Dense-linear（主线）：`r = piecewise(...)`（上面那套）

**例 1：过了 6/10，正常运行，无异常**

* `pass_ratio = 6/10 = 0.6`
* `r = -0.3 + 1.3*0.6`

  * `1.3*0.6 = 0.78`
  * `-0.3 + 0.78 = 0.48`
* `clip` 后还是 `0.48`
  （这正是 RLTF adaptive feedback 的数值逻辑）

**例 2：语法错误（比如少括号）**

* 触发 `syntax_error` → `r = -1.0`

**例 3：运行时错误 / 超时（TLE）**

* 触发 `runtime_error_or_timeout` → `r = -0.6`

**例 4：截断导致的不完整代码（max tokens hit）**

* 触发 `truncated_by_max_tokens`
* 推荐：**mask advantage=0** 或 DAPO 风格 soft punish/过滤，而不是当成普通 syntax=-1 来学 ([arXiv][3])

---

### (C) Dense-weighted（VeRPO 思路）：把 `pass_ratio` 换成 `weighted_pass_ratio`

VeRPO 的核心点是：**给不同 unit tests 分配不同权重（按通过率等统计），让“更有判别力/更难”的测试贡献更大**，并用这种 weighted reward 改善收敛与方差。([arXiv][4])

一个简单可实现的例子（你可以写进 repo，面试也好讲）：

* 设 10 个测试权重 `w_i`（总和归一到 1），比如：

  * 3 个“难测”（历史通过率低）各 `0.2` → 共 `0.6`
  * 7 个“易测”各 `~0.057142` → 共 `0.4`
* 某输出通过：2 个难测 + 5 个易测

  * `weighted_pass_ratio = 2*0.2 + 5*0.057142 = 0.4 + 0.28571 = 0.68571`
* 然后套 RLTF 的 affine：

  * `r = -0.3 + 1.3*0.68571`
  * `1.3*0.68571 ≈ 0.891423`
  * `r ≈ 0.591423`（clip 后不变）

> 这类“按测试判别力加权”就是 VeRPO 主张的方向（具体权重函数你可以自定义，但**理念有出处**）。([arXiv][4])

---

### (D) + Outcome anchor：dense + “全局结果锚点”（VeRPO 的 global outcome anchors 思路）

VeRPO 提到 **global execution outcome anchors**：在 weighted test reward 之外，再用全局 outcome 引导模型更偏向“完全正确”。([arXiv][4])

为了让 anchor 真正“起作用”，你需要给它留出 headroom（否则 AC 时 `r` 已经 1 了再加也被 clip 吃掉）。一个可解释、可控的实现是：

* `r_dense = -0.3 + 1.0*pass_ratio`（范围 [-0.3, 0.7]）
* `r_anchor = 0.3 * 1[AC]`
* `r = clip(r_dense + r_anchor, [-1,1])`

**例：6/10（非 AC）**

* `r_dense = -0.3 + 1.0*0.6 = 0.3`
* `r_anchor = 0`
* `r=0.3`

**例：10/10（AC）**

* `r_dense = -0.3 + 1.0*1 = 0.7`
* `r_anchor = 0.3`
* `r=1.0`

这就实现了“**部分正确有密集信号，但完全正确额外加速**”，而且理念上对齐 VeRPO 的“global anchors”动机。([arXiv][4])

---

## 6) 你问的核心：`rlen` 到底该不该做？我给你一个“不会出大问题”的结论

**不要一上来就做“按 token 长度扣分”的 rlen（很可能误伤需要长代码的题）**。
你真正必须做的是：**截断/超长的特殊处理**（mask/过滤/soft punish），这在 RLTF、DeepCoder、DAPO、以及 truncation masking 经验里都有明确依据。([Together AI][2])

如果你想把 “成本/效率” 纳入 reward（作为第 5 个可选消融），我建议你做 **late-only efficiency penalty**，并且只在 “已经接近正确” 时启用（比如 `pass_ratio>=0.8` 或 `AC=1`），更安全；VeRPO 也有“效率衰减奖励”的思想（他们是在多轮交互里按 turn decay）。([arXiv][4])

---

## 7) 给你一句能写进报告里的“证据链总结”

* `-0.3 + 1.3*pass_ratio` 与 `(-1/-0.6/-0.3)` 的错误分段：来自 RLTF 的 multi-granularity unit-test feedback 设计 
* weighted pass reward + outcome anchors：对应 VeRPO 的 verified reward 设计方向 ([arXiv][4])
* 截断/超长：不要当成普通语法错学习，优先做 masking/过滤/soft punish（RLTF + DeepCoder + DAPO + truncation masking）([Together AI][2])
* pass_ratio 作为连续 reward 在 GRPO/代码执行反馈场景非常常见（例如多轮 GRPO 的 Murphy 直接把 reward 定义为通过测试比例）([arXiv][5])

---

如果你愿意，我下一步可以把你这 4 个 reward 消融 **写成一段“可直接放进 repo 的 reward.py 伪代码 + 记录字段规范（error_type / truncated / exec_time / tokens）”**，并且把每个 reward 的日志指标设计成你面板里那套（error breakdown、timeout rate、cost-per-solved），这样你实现起来不会散。

[1]: https://arxiv.org/pdf/2502.18770?utm_source=chatgpt.com "Reward Shaping to Mitigate Reward Hacking in RLHF"
[2]: https://www.together.ai/blog/deepcoder "DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level"
[3]: https://arxiv.org/html/2503.14476v1 "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
[4]: https://arxiv.org/html/2601.03525v1 "VeRPO: Verifiable Dense Reward Policy Optimization for Code Generation"
[5]: https://arxiv.org/html/2511.07833v1 "Murphy: Multi-Turn GRPO for Self Correcting Code Generation"
