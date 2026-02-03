# RLVR Coding Model - Guardrails（防 Reward Hacking / 质量控制）

目标：让 RLVR coding 的训练与评测更接近工业可用形态，减少“投机取巧”“无效输出”“超长输出”“利用环境漏洞”等问题，并保证成本可控。

---

## 1) 常见风险（面试高频追问）

- **输出超长/重复**：通过堆模板、重复代码、或输出非代码文本来“搏”部分奖励或拖垮判题成本。
- **无效输出**：空输出、非代码、混入解释文本导致判题失败但浪费资源。
- **利用不确定性**：随机性、时间差、未固定种子导致偶然通过。
- **环境/系统利用**：访问文件、网络、系统调用、或依赖未声明环境。
- **多轮反馈泄漏**：把隐藏测试或过多 stderr/stdout 直接喂回模型，变相“教答案”。

---

## 2) 必须启用的 Guardrails（建议作为默认配置）

### 2.1 输出与判题短路（省钱 + 防投机）

- **空输出/非代码**：直接判为 0 分（或最小奖励），不进入 sandbox（记录 short-circuit reason）。
- **超长输出**：设置 `max_new_tokens` 上限；超出直接截断并记录 `truncated=True`（评测必须计入成本）。
- **重复样本缓存**：对候选代码做 `code_hash`（去空白规范化后 hash），命中缓存则复用判题结果（记录 cache hit rate）。

### 2.2 沙盒约束（安全 + 公平）

- 禁网、限制文件系统、限制系统调用（由 SandboxFusion/沙盒层保证）
- 固定 time/mem limit（并把 limit 写入评测报告）
- 运行环境版本固定（python 版本/依赖/编译器版本），并随结果一起记录

### 2.3 反馈摘要（多轮修复必须做“脱敏”）

多轮修复只允许喂回**短、结构化、脱敏**信息（避免泄漏与 prompt 膨胀）：

- `error_type`（SYNTAX/RUNTIME/TIMEOUT/WA）
- `pass_ratio = passed/total`
- `failed_cases`（最多 K 条，必要时做内容截断/脱敏）
- `traceback_tail`（最后 N 行）
- 禁止直接回传完整隐藏测试、或完整 stdout/stderr 原文

---

## 3) 建议加分的 Guardrails（能显著提升工业可信度）

- **reward clipping / outlier handling**：避免极端样本（如异常快/异常慢）主导梯度
- **timeout 防爆**：timeout rate 连续上升触发 early stop 或降低并发/缩短 max_new_tokens
- **长度惩罚（可选）**：在 reward 中加入轻微长度成本（或只在 `EVAL@budget` 中用预算约束）
- **determinism**：对允许随机性的题目，固定随机种子或在判题脚本里覆盖随机源

---

## 4) 报告中要展示的“防投机证据”（建议 1 页）

- 输出长度分布（训练中与评测时）
- short-circuit 分布（空/非代码/超长等）
- cache hit rate
- sandbox error / timeout 率曲线（异常上升要能解释）

