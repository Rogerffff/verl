# Repair RL Reward Design

这份文档只负责一件事：

- 固定当前 repair RL 的 **正式 reward 设计**

它**不负责**展开：

- 数据筛选设计
- 完整 second-pass repair RL probe 计划
- 训练排期与算力安排

当前正式采用的是一个 **两阶段 reward family**：

```text
Stage 1:
    repair_delta_v0
    = 修复后质量
    + 相对 first-pass 的提升奖励
    - 相对 first-pass 的回退惩罚
    + first-fail → repair-AC 的额外奖励

Stage 2:
    repair_delta_edit_v1
    = repair_delta_v0
    - gated edit penalty
```

其中 edit penalty 采用：

```text
hybrid edit ratio
+ dynamic edit budget
+ candidate-level q0/q1 gate
+ weak group-level good-rate gate
```

当前正式 rollout 顺序写死为：

```text
先跑 repair_delta_v0
确认 repair RL 有信号后
再跑 repair_delta_edit_v1
```

原因是：

- `repair_delta_v0` 更接近最小可行版本，
- 更适合先判断 second-pass RL 是否有效，
- 也更符合当前项目“小规模 repair RL probe 先看信号”的定位。

在此基础上，再吸收 PRepair 的 edit-aware 思想升级到 `repair_delta_edit_v1`。PRepair 的核心贡献是指出 correctness-only GRPO 会导致 over-editing，并提出在模型达到足够 repair correctness 后再加 edit-aware penalty；它的 edit cost 是 line-level Levenshtein distance，奖励里有 group-level accuracy threshold。这个思想应该吸收，但你的 CodeContests repair 场景有 `pass_ratio` 这样的 dense signal，而不是纯二值 correctness，所以直接照搬 PRepair 会浪费你的 verifier 信息。PRepair 论文也明确说 correctness-only repair 目标忽略了修改幅度，over-editing 会覆盖原本正确的代码；它用 EA-GRPO 在达到足够 group-level accuracy 后才施加 edit penalty，目标是兼顾正确性和最小修改。([arXiv][1])

---

## 1. 是否采纳另一个 agent 的建议

我的判断是：

| 建议                                               |     是否采纳 | 理由                                                         |
| ------------------------------------------------ | -------: | ---------------------------------------------------------- |
| 不直接照搬 PRepair 的纯 line-level Levenshtein          |       采纳 | CodeContests 中很多 bug 是操作符、边界、表达式级小修；纯 line-level 太粗        |
| 用 line + token hybrid edit metric                |       采纳 | line 负责检测结构性重写，token 负责避免把 `<` 改成 `<=` 这种小修放大              |
| edit penalty 权重要小                                |       采纳 | repair correctness 仍然是主目标，edit penalty 只用于同等质量下偏好少改        |
| edit budget 跟 q0 动态变化                            |       采纳 | first-pass 越接近正确，越应该小改；first-pass 越差，越允许大改                 |
| 保留 PRepair-style group gate                      | 采纳，但做弱化版 | group gate 可以防止早期模型还没学会修时就被 edit penalty 误导                |
| 用 PRepair 的 binary group accuracy gate 作为唯一 gate |      不采纳 | 你的场景有 pass_ratio 和 q0/q1，candidate-level dense gate 更有信息量  |
| 用 PRepair 的 group 内标准化 sigmoid edit penalty      |   第一版不采纳 | 小 group 下标准化可能不稳；你的 reward 已经足够复杂，先用可解释的 bounded edit_over |

所以最终方案不是“我的方案 vs PRepair”，而是：

```text
PRepair 的思想：
    edit-aware repair
    group-level correctness gate
    防止 over-editing

你的项目适配：
    dense q0/q1 improvement reward
    line + token hybrid edit ratio
    q0-dependent dynamic edit budget
    candidate gate + weak group gate
```

---

## 2. 正式 reward 家族命名

当前正式 reward family 包含两个名字：

```text
repair_delta_v0
repair_delta_edit_v1
```

含义是：

```text
repair:
    用于 one-step / second-pass repair RL

delta:
    显式奖励 q1 - q0 的提升，惩罚 q1 < q0 的回退

edit:
    在 v1 中额外对高质量 repair 的过度编辑做小惩罚
```

当前正式版本顺序：

```text
第一版先跑：
    repair_delta_v0

确认有信号后再跑：
    repair_delta_edit_v1
```

因此：

- `repair_delta_v0` 是当前第一版正式 reward
- `repair_delta_edit_v1` 是当前预定的下一版升级 reward

---

## 3. 基础定义

对每个 repair prompt，你有：

```text
c0 = first-pass buggy code
ci = 第 i 个 repaired rollout code
```

对应 verifier 结果：

```text
p0 = first-pass pass_ratio_all
a0 = first-pass accepted, 0 or 1

pi = repaired pass_ratio_all
ai = repaired accepted, 0 or 1
```

定义质量分数：

```text
q0 = 0.8 * p0 + 0.2 * a0
qi = 0.8 * pi + 0.2 * ai
```

这和你当前主线 `anchored_dense` 保持一致。

一般来说，repair RL 训练样本应该主要来自 first-pass failed cases，所以大部分时候：

```text
a0 = 0
```

但公式保留 `a0`，方便你以后扩展到 broader repair setting。

---

## 4. 输出异常处理

先继承你当前 reward 的 guardrail 语义。

### 4.1 基础设施问题

如果 repaired rollout 出现：

```text
sandbox error
API error
no-test
timeout caused by infra
truncation
verifier invalid
```

则：

```text
return INVALID_FOR_RL
```

这类样本不应该参与 advantage。

### 4.2 模型输出格式错误

如果 repaired output 是：

```text
empty_output
non-code
extraction_failure
syntax error
```

则：

```text
reward = -1.0
```

这类是模型行为错误，不是基础设施错误，应该参与训练并被惩罚。

---

## 5. Base repair reward (`repair_delta_v0`)

`repair_delta_v0` 的核心 reward 是：

```text
delta_pos = max(qi - q0, 0)
delta_neg = max(q0 - qi, 0)

accepted_gain = 1 if a0 == 0 and ai == 1 else 0

base_reward =
    qi
  + 0.25 * delta_pos
  - 0.75 * delta_neg
  + 0.25 * accepted_gain
```

解释一下每一项：

```text
qi:
    修复后代码本身的最终质量。

0.25 * delta_pos:
    如果修复后比 first-pass 更好，额外奖励。

-0.75 * delta_neg:
    如果修复后比 first-pass 更差，强惩罚。

0.25 * accepted_gain:
    如果从 first-pass failed 变成 repair accepted，额外奖励。
```

为什么 regression penalty 比 improvement bonus 大？

因为 repair model 不应该“乱改”。对于 first-pass 已经有 partial correctness 的代码，最坏的行为不是“没修好”，而是把原本正确的部分破坏掉。

### 5.1 当前正式落地版本

当前正式建议是：

```text
第一版 repair RL 先只使用这一段 base reward
也就是先跑 repair_delta_v0
```

原因：

```text
1. 它已经保留了你当前 anchored_dense 的主语义；
2. 它已经显式奖励 q1 - q0 的提升；
3. 它已经显式惩罚 regression；
4. 它已经对 first-fail -> repair-AC 给额外奖励；
5. 它更容易判断当前 repair RL 是否真的有信号；
6. 它避免第一版就把 edit-aware 复杂度一起引入。
```

所以从文档口径上，`repair_delta_v0` 不是临时 debug reward，而是：

```text
当前第一版正式 reward
```

### 5.2 `repair_delta_v0` 的输出范围

为了避免实现时再出现“公式已经定了，但最终是否 clip 没写死”的歧义，
当前正式口径补充为：

```text
对于 valid sample:
    repair_delta_v0 直接输出 base_reward
    不再额外做第二层 clip
```

因此：

```text
q0, q1 in [0, 1]
delta_pos in [0, 1]
delta_neg in [0, 1]
accepted_gain in {0, 1}
```

可推出：

```text
valid sample 上的 analytic range:
    base_reward in [-0.75, 1.5]
```

再加上 guardrail 口径：

```text
invalid sample:
    reward_raw = NaN
    score = 0.0

bad output:
    reward = -1.0
```

所以第一版 `repair_delta_v0` 的正式输出口径可以读成：

```text
valid sample:
    [-0.75, 1.5]

bad output:
    -1.0

invalid sample:
    INVALID_FOR_RL
```

这意味着：

- `repair_delta_v0` 已经是有界 reward，
- 但首轮 probe 不再额外把它压平成更窄区间，
- 这样更方便直接观察“相对 first-pass 改善 / 回退”到底有多强。

---

## 6. Edit ratio 设计

最终不要只用 line-level LCS，也不要只用 line-level Levenshtein。

我建议：

```text
hybrid_edit_ratio =
    0.4 * line_lcs_edit_ratio
  + 0.6 * token_levenshtein_ratio
```

如果你暂时不想实现 token-level Levenshtein，也可以用：

```text
hybrid_edit_ratio =
    0.4 * line_lcs_edit_ratio
  + 0.6 * token_lcs_edit_ratio
```

但我更推荐 token-level Levenshtein，因为它更接近“实际 token 编辑操作数”。

---

### 6.1 line_lcs_edit_ratio

```text
line_lcs_edit_ratio =
    1 - LCS_lines(c0, ci) / max(num_lines(c0), num_lines(ci))
```

它主要检测结构是否被保留。

如果模型把整份代码重写了，这个值会高。

---

### 6.2 token_levenshtein_ratio

```text
token_levenshtein_ratio =
    Levenshtein_tokens(c0, ci) / max(num_tokens(c0), num_tokens(ci))
```

它主要检测细粒度修改量。

例如：

```python
if x <= y:
```

改成：

```python
if x < y:
```

line-level 会认为整行不同，但 token-level 只会认为一个 operator token 变了。

---

### 6.3 为什么 token 权重更高

我建议：

```text
line weight = 0.4
token weight = 0.6
```

原因是 CodeContests 里很多 repair 是：

```text
边界条件
下标
取模
排序方向
比较符
初始化
循环范围
```

这些通常是 token-level 或 expression-level 修复。如果 line 权重太高，会把很多小修误判成大修。

如果你后续发现模型依然大面积重写，可以改成：

```text
0.5 line + 0.5 token
```

第一版我会用：

```text
0.4 line + 0.6 token
```

---

## 7. Dynamic edit budget

不要固定 `edit_budget = 0.25`。我建议让 budget 随 `q0` 变化：

```text
edit_budget(q0) = clip(0.35 - 0.20 * q0, 0.15, 0.35)
```

展开看：

```text
q0 = 0.00 → edit_budget = 0.35
q0 = 0.50 → edit_budget = 0.25
q0 = 1.00 → edit_budget = 0.15
```

直觉是：

```text
first-pass 很差：
    可能算法方向都错了，允许较大改动。

first-pass 中等：
    允许中等修改。

first-pass 很接近正确：
    更希望局部修复，不希望整段重写。
```

然后定义：

```text
edit_over =
    max(hybrid_edit_ratio - edit_budget(q0), 0)
    /
    max(1 - edit_budget(q0), 1e-6)
```

所以只有超过 budget 的修改才被惩罚。

---

## 8. Candidate-level edit gate

edit penalty 不能对所有样本都开。否则模型还没学会修时，会被迫少改，最后学成 copy 原代码。

我建议 candidate gate 写成：

```text
candidate_good_i =
    qi >= q0
    and
    (
        ai == 1
        or
        pi >= 0.75
    )
```

也就是说，只有当这个 repaired candidate：

```text
1. 没有比 first-pass 更差；
2. 并且已经 accepted，或者 pass_ratio 至少达到 0.75；
```

才有资格被 edit penalty 约束。

为什么这里用 `pi >= 0.75`，而不是 `qi >= 0.6`？

因为：

```text
qi = 0.8 * pi + 0.2 * ai
```

当 `ai = 0` 时：

```text
qi >= 0.6
等价于
pi >= 0.75
```

所以直接写 `pi >= 0.75` 更清楚。

如果你想更早启用 edit penalty，可以把它降到：

```text
pi >= 0.70
```

第一版我建议用：

```text
pi >= 0.75
```

更稳。

---

## 9. Group-level good-rate gate

这里采纳 PRepair 的思想，但不照搬 binary accuracy gate。

对同一个 prompt 的 K 个 rollout，先计算每个 rollout 的：

```text
candidate_good_i
```

然后：

```text
group_good_count = sum(candidate_good_i)
group_good_rate  = group_good_count / num_active_rollouts
```

定义：

```text
group_gate =
    group_good_rate >= 0.25
```

如果你的 rollout group size 是 8，那么：

```text
0.25 ≈ 至少 2 个 good candidates
```

我建议实际实现时写成：

```text
group_gate =
    group_good_count >= max(1, ceil(0.25 * num_active_rollouts))
```

如果你的 group size 固定是 8，可以更硬一点：

```text
group_gate =
    group_good_count >= 2
```

这个 gate 的作用是：

```text
如果这一组 rollout 里几乎没有高质量 repair：
    先不要启用 edit penalty，让模型优先学会修对。

如果这一组里已经有一些高质量 repair：
    再在这些高质量 repair 之间偏好少改。
```

这和 PRepair 的“达到足够 group-level correctness 后才加 edit penalty”的精神一致；PRepair 的论文也强调，edit penalty 是动态触发的，过低或过高的 threshold 都可能损害 correctness 和 precise repair 的平衡。([arXiv][1])

---

## 10. 最终 edit penalty

最终 edit penalty 是：

```text
edit_penalty_i =
    lambda_edit
    * candidate_good_i
    * group_gate
    * edit_over_i
```

`repair_delta_edit_v1` 建议：

```text
lambda_edit = 0.08
```

所以最终最多大约扣 `0.08` 左右。

这很重要：edit penalty 不是主 reward。它只是告诉模型：

```text
如果修复质量差不多，少改更好。
```

不是告诉模型：

```text
永远不要大改。
```

---

## 11. `repair_delta_edit_v1` reward 公式

`repair_delta_edit_v1` 的完整公式如下：

```text
q0 = 0.8 * p0 + 0.2 * a0
qi = 0.8 * pi + 0.2 * ai

delta_pos = max(qi - q0, 0)
delta_neg = max(q0 - qi, 0)

accepted_gain = I[a0 == 0 and ai == 1]

base_reward_i =
    qi
  + 0.25 * delta_pos
  - 0.75 * delta_neg
  + 0.25 * accepted_gain

hybrid_edit_ratio_i =
    0.4 * line_lcs_edit_ratio_i
  + 0.6 * token_levenshtein_ratio_i

edit_budget_i =
    clip(0.35 - 0.20 * q0, 0.15, 0.35)

edit_over_i =
    max(hybrid_edit_ratio_i - edit_budget_i, 0)
    /
    max(1 - edit_budget_i, 1e-6)

candidate_good_i =
    I[
        qi >= q0
        and
        (
            ai == 1
            or
            pi >= 0.75
        )
    ]

group_gate =
    I[group_good_rate >= 0.25]

edit_penalty_i =
    0.08
    * candidate_good_i
    * group_gate
    * edit_over_i

reward_i =
    base_reward_i - edit_penalty_i

reward_i =
    clip(reward_i, -1.0, 1.4)
```

因此：

```text
repair_delta_edit_v1 =
    repair_delta_v0
  - gated dynamic hybrid edit penalty
```

---

## 12. 伪代码实现

下面是接近你可以直接落地的版本。

```python
import math
from typing import List


BAD_OUTPUT_ERRORS = {
    "empty_output",
    "non_code",
    "extraction_failure",
    "syntax_error",
}


def compute_quality(pass_ratio: float, accepted: bool) -> float:
    return 0.8 * float(pass_ratio) + 0.2 * float(accepted)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_edit_budget(q0: float) -> float:
    # q0 low  -> allow larger rewrite
    # q0 high -> expect more local repair
    return clip(0.35 - 0.20 * q0, 0.15, 0.35)


def compute_edit_over(edit_ratio: float, q0: float) -> float:
    budget = compute_edit_budget(q0)
    return max(edit_ratio - budget, 0.0) / max(1.0 - budget, 1e-6)


def compute_base_repair_reward(
    p0: float,
    a0: bool,
    p1: float,
    a1: bool,
) -> tuple[float, dict]:
    q0 = compute_quality(p0, a0)
    q1 = compute_quality(p1, a1)

    delta_pos = max(q1 - q0, 0.0)
    delta_neg = max(q0 - q1, 0.0)

    accepted_gain = float((not a0) and a1)

    base_reward = (
        q1
        + 0.25 * delta_pos
        - 0.75 * delta_neg
        + 0.25 * accepted_gain
    )

    info = {
        "q0": q0,
        "q1": q1,
        "delta_q": q1 - q0,
        "delta_pos": delta_pos,
        "delta_neg": delta_neg,
        "accepted_gain": accepted_gain,
        "base_reward": base_reward,
    }

    return base_reward, info


def candidate_edit_gate(
    *,
    q0: float,
    q1: float,
    pass_ratio_repair: float,
    accepted_repair: bool,
    pass_ratio_gate: float = 0.75,
) -> bool:
    return (
        q1 >= q0
        and (
            accepted_repair
            or pass_ratio_repair >= pass_ratio_gate
        )
    )


def compute_hybrid_edit_ratio(
    first_code: str,
    repaired_code: str,
) -> tuple[float, dict]:
    """
    You can implement these two functions with the LCS / token edit code
    discussed earlier.

    line_lcs_edit_ratio:
        1 - LCS_lines / max(num_lines_old, num_lines_new)

    token_levenshtein_ratio:
        Levenshtein_tokens / max(num_tokens_old, num_tokens_new)
    """
    line_ratio = line_lcs_edit_ratio(first_code, repaired_code)
    token_ratio = token_levenshtein_ratio(first_code, repaired_code)

    hybrid = 0.4 * line_ratio + 0.6 * token_ratio
    hybrid = clip(hybrid, 0.0, 1.0)

    info = {
        "line_lcs_edit_ratio": line_ratio,
        "token_levenshtein_ratio": token_ratio,
        "hybrid_edit_ratio": hybrid,
    }

    return hybrid, info


def compute_group_repair_rewards(
    first_result,
    repair_results: List,
    first_code: str,
    repaired_codes: List[str],
    *,
    expected_group_size: int = 8,
    group_good_rate_threshold: float = 0.25,
    edit_lambda: float = 0.08,
    pass_ratio_gate: float = 0.75,
):
    """
    first_result fields:
        pass_ratio_all: float
        accepted: bool

    repair_result fields:
        pass_ratio_all: float
        accepted: bool
        invalid_for_rl: bool
        error_type: str

    Returns:
        rewards:
            list of float or "INVALID_FOR_RL"
        logs:
            list of per-rollout debug dicts
    """

    p0 = first_result.pass_ratio_all
    a0 = bool(first_result.accepted)
    q0 = compute_quality(p0, a0)

    rewards = []
    logs = []
    active_indices = []
    candidate_good_flags = []

    # First pass: compute base rewards and candidate gates.
    for i, repair_result in enumerate(repair_results):
        log = {"rollout_idx": i}

        if repair_result.invalid_for_rl:
            rewards.append("INVALID_FOR_RL")
            log["invalid_for_rl"] = True
            log["preliminary_reward"] = "INVALID_FOR_RL"
            logs.append(log)
            continue

        active_indices.append(i)

        if repair_result.error_type in BAD_OUTPUT_ERRORS:
            rewards.append(-1.0)
            log.update({
                "invalid_for_rl": False,
                "bad_output": True,
                "q0": q0,
                "q1": None,
                "candidate_good": False,
                "preliminary_reward": -1.0,
            })
            logs.append(log)
            candidate_good_flags.append(False)
            continue

        p1 = repair_result.pass_ratio_all
        a1 = bool(repair_result.accepted)

        base_reward, base_info = compute_base_repair_reward(
            p0=p0,
            a0=a0,
            p1=p1,
            a1=a1,
        )

        edit_ratio, edit_info = compute_hybrid_edit_ratio(
            first_code,
            repaired_codes[i],
        )

        candidate_good = candidate_edit_gate(
            q0=base_info["q0"],
            q1=base_info["q1"],
            pass_ratio_repair=p1,
            accepted_repair=a1,
            pass_ratio_gate=pass_ratio_gate,
        )

        rewards.append(base_reward)

        log.update(base_info)
        log.update(edit_info)
        log.update({
            "invalid_for_rl": False,
            "bad_output": False,
            "candidate_good": candidate_good,
            "preliminary_reward": base_reward,
        })

        logs.append(log)
        candidate_good_flags.append(candidate_good)

    # Compute group gate.
    active_n = len(active_indices)
    good_count = sum(candidate_good_flags)

    if active_n == 0:
        return rewards, logs

    # For fixed group size 8, this is effectively >= 2.
    min_good_count = max(
        1,
        math.ceil(group_good_rate_threshold * active_n),
    )

    # Optional: if you want stricter PRepair-style behavior for group_size=8:
    # min_good_count = max(min_good_count, 2)

    group_good_rate = good_count / active_n
    group_gate = good_count >= min_good_count

    # Second pass: apply edit penalty where appropriate.
    for i in active_indices:
        if rewards[i] == "INVALID_FOR_RL":
            continue

        log = logs[i]

        if log.get("bad_output", False):
            log.update({
                "group_good_count": good_count,
                "group_good_rate": group_good_rate,
                "group_gate": group_gate,
                "edit_penalty": 0.0,
                "final_reward": rewards[i],
            })
            continue

        q0_i = log["q0"]
        edit_ratio_i = log["hybrid_edit_ratio"]

        edit_over = compute_edit_over(edit_ratio_i, q0_i)

        edit_penalty = (
            edit_lambda
            * float(log["candidate_good"])
            * float(group_gate)
            * edit_over
        )

        final_reward = log["base_reward"] - edit_penalty
        final_reward = clip(final_reward, -1.0, 1.4)

        rewards[i] = final_reward

        log.update({
            "edit_budget": compute_edit_budget(q0_i),
            "edit_over": edit_over,
            "group_good_count": good_count,
            "group_good_rate": group_good_rate,
            "group_gate": group_gate,
            "edit_penalty": edit_penalty,
            "final_reward": final_reward,
        })

    return rewards, logs
```

---

## 13. 为什么这个版本比 PRepair 原版更适合你

PRepair 的任务更偏“precise repair”：给一个 buggy program，目标是尽量少改并修对。它的设置强调 binary correctness 和 edit cost，论文里用 line sequence 的 Levenshtein distance 来衡量 edit cost。([arXiv][1])

你的任务不同：

```text
CodeContests repair RL
已有 pass_ratio_all
已有 accepted
已有 first-pass q0
已有 repaired q1
目标既包括 final AC，也包括 partial correctness improvement
```

CodeContests 本身是 competitive programming dataset，包含来自 Aizu、AtCoder、CodeChef、Codeforces、HackerEarth 等来源的问题，并包含测试用例、正确和错误的人类解法；这意味着你的 verifier 可以提供比二值 pass/fail 更丰富的训练信号。([GitHub][2])

所以你的 reward 应该利用：

```text
q1
q1 - q0
regression q1 < q0
pass_ratio_repair
accepted transition
```

而不是只看：

```text
accepted or not
edit distance
```

---

## 14. 为什么不用纯 PRepair group gate

PRepair 的 group gate 是：

```text
如果 group-level accuracy 足够高，才启用 edit penalty
```

这在 binary correctness 场景里合理。

但在你的场景里，可能出现：

```text
8 个 rollout 里没有一个 accepted，
但有 3 个从 pass_ratio 0.20 提升到 0.75。
```

PRepair-style binary group accuracy 会认为：

```text
group accuracy = 0
不要启用 edit penalty
```

但你的 dense reward 明确知道这些 candidate 已经是高质量 partial repair。

因此我建议用：

```text
candidate_good = q1 >= q0 and (accepted or pass_ratio >= 0.75)
```

这会比 binary accepted 更合理。

同时再加弱 group gate，是为了保留 PRepair 的安全机制：

```text
只有这一组已经出现一些高质量 repair 时，才开始比较谁改得更少。
```

---

## 15. 为什么 edit penalty 只扣很小

你的主任务仍然是：

```text
提升 held-out Protocol B final accepted@1
```

不是为了追求最小 diff。

如果 first-pass 算法方向错了，正确 repair 可能需要大改。尤其 CodeContests 是算法题，不是普通小 bug patch；有些 failure 不是局部 typo，而是状态设计、贪心策略、复杂度、边界处理整个方向有问题。

所以 edit penalty 必须是：

```text
小权重
有 budget
有 gate
动态触发
```

而不是：

```text
改得越少 reward 越高
```

这里的 `lambda_edit = 0.08` 只是 `repair_delta_edit_v1` 的初始建议：在 correctness 差不多时，把少改的 candidate 排在前面。它不会压过 AC bonus 和 regression penalty。

---

## 16. 最重要的日志项

为了能判断 reward 是否工作，建议每个 rollout 都 log：

```text
q0
q1
delta_q
delta_pos
delta_neg
accepted_gain

line_lcs_edit_ratio
token_levenshtein_ratio
hybrid_edit_ratio
edit_budget
edit_over

candidate_good
group_good_count
group_good_rate
group_gate
edit_penalty

base_reward
final_reward
```

每个 checkpoint 汇总：

```text
mean_delta_q
regression_rate
severe_regression_rate
mean_hybrid_edit_ratio
edit_ratio_on_success
edit_ratio_on_regression
candidate_good_rate
group_gate_rate
```

特别要看：

```text
group_gate_rate
```

如果它几乎永远是 0，说明 edit penalty 基本没生效。

如果它很早就接近 1，说明 gate 太松，可能会过早惩罚 edit。

---

## 17. `repair_delta_edit_v1` 推荐初始超参数

`repair_delta_edit_v1` 的初始超参数建议固定为：

```text
q score:
    q = 0.8 * pass_ratio + 0.2 * accepted

base reward:
    improvement bonus alpha = 0.25
    regression penalty beta = 0.75
    accepted_gain bonus gamma = 0.25

edit metric:
    line_lcs weight = 0.4
    token_levenshtein weight = 0.6

edit budget:
    budget(q0) = clip(0.35 - 0.20 * q0, 0.15, 0.35)

edit gate:
    candidate_good = q1 >= q0 and (accepted or pass_ratio >= 0.75)
    group_good_rate_threshold = 0.25

edit penalty:
    lambda_edit = 0.08

reward clip:
    [-1.0, 1.4]
```

注意：

```text
如果当前只跑一个正式版本，应先跑 repair_delta_v0
这里这组超参数只用于后续升级到 repair_delta_edit_v1 时的初始设置
```

---

## 18. 什么情况下调参

### 如果模型还是大面积重写

现象：

```text
after_repair accepted 小涨
hybrid_edit_ratio 大涨
edit_ratio_on_success 很高
regression_rate 也升高
```

调法：

```text
lambda_edit: 0.08 → 0.10
line weight: 0.4 → 0.5
edit_budget 上限: 0.35 → 0.30
pass_ratio_gate: 0.75 → 0.70
```

---

### 如果模型不敢改，repair success 不涨

现象：

```text
hybrid_edit_ratio 下降
conditional_repair_success 不涨
mean_delta_q 不涨
no-op / copy rate 上升
```

调法：

```text
lambda_edit: 0.08 → 0.04 或 0.05
edit_budget 上限: 0.35 → 0.40
pass_ratio_gate: 0.75 → 0.80
group threshold: 0.25 → 0.375
```

---

### 如果 edit penalty 几乎没生效

现象：

```text
group_gate_rate 接近 0
candidate_good_rate 很低
```

调法：

```text
pass_ratio_gate: 0.75 → 0.70
group threshold: 0.25 → 0.125
或者先去掉 group_gate，只保留 candidate_gate 做一个 debug run
```

---

### 如果 raw coding ability 下降

现象：

```text
Protocol B repair 上升
但 raw CodeContests / HumanEval / MBPP_reg 明显下降
```

调法：

```text
减少 repair RL steps
降低学习率
checkpoint early stop
加入少量 raw coding anchor batch
或者把 repair model 和 first-pass model 分离
```

---

## 19. 最终正式推荐

当前这份文档的正式结论写死如下。

### 19.1 第一版正式 reward

第一版 repair RL 先跑：

```text
repair_delta_v0
=
q1
+ 0.25 * max(q1 - q0, 0)
- 0.75 * max(q0 - q1, 0)
+ 0.25 * I[first not AC and repair AC]
```

它的核心优点是：

```text
1. 保留你现有 anchored_dense verifier 信号；
2. 显式优化 repair 前后提升；
3. 强惩罚 regression；
4. 对 first-fail → repair-AC 给足够奖励；
5. 第一版更简单，更适合先判断 repair RL 是否有信号。
```

这就是当前**第一版正式落地 reward**。

### 19.2 第二版升级 reward

如果 `repair_delta_v0` 已经显示出稳定正信号，再升级到：

```text
repair_delta_edit_v1
=
q1
+ 0.25 * max(q1 - q0, 0)
- 0.75 * max(q0 - q1, 0)
+ 0.25 * I[first not AC and repair AC]
- 0.08
  * candidate_good_gate
  * group_good_gate
  * edit_over
```

其中：

```text
candidate_good_gate =
    I[q1 >= q0 and (repair accepted or repair pass_ratio >= 0.75)]

group_good_gate =
    I[group_good_rate >= 0.25]

edit_over =
    max(hybrid_edit_ratio - dynamic_budget(q0), 0)
    /
    (1 - dynamic_budget(q0))

hybrid_edit_ratio =
    0.4 * line_lcs_edit_ratio
  + 0.6 * token_levenshtein_ratio

dynamic_budget(q0) =
    clip(0.35 - 0.20 * q0, 0.15, 0.35)
```

它的作用是：

```text
1. 吸收 PRepair 的 edit-aware 思想；
2. 但不被 PRepair 的 binary reward / line-level metric 限制；
3. 更适配 CodeContests 这种有 dense pass_ratio 的 competitive programming repair 场景；
4. 在确认 repair correctness 已有信号后，再额外抑制 over-editing。
```

### 19.3 正式 rollout 顺序

最终正式顺序：

```text
Stage 1:
    repair_delta_v0

Stage 2:
    repair_delta_edit_v1
    only after v0 shows signal
```

这就是当前文档固定下来的正式 reward 设计。

[1]: https://arxiv.org/html/2604.05963 "QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization"
[2]: https://github.com/google-deepmind/code_contests "GitHub - google-deepmind/code_contests · GitHub"
