# RLVR Coding Model - 数据治理与泄漏防护

目标：把“train/valid/test 严格隔离”变成**可审计的事实**，并显式处理 CodeContests 这类题库常见的重复题/近似题问题，避免面试时被质疑 benchmark 被污染。

---

## 1) 风险模型（为什么必须做）

CodeContests / Online Judge 类数据常见风险：

- **重复题**：同一题在不同 split 重复出现（ID 变了、或 statement 文本略有差异）。
- **近似题/同源题**：题面改写、变量名替换、样例变动但本质相同。
- **外部基准污染**：HumanEval/MBPP 的题面或同源题出现在训练数据里，导致对标失真。

---

## 2) 数据清单（manifest）是最小可复现单元

对每个 split 固定一份 manifest（推荐落盘到 `data_manifests/`，并进 repo），每行至少包含：

- `dataset`：CodeContests / HumanEval / MBPP_reg
- `split`：train / valid / test
- `problem_id`：原始 ID（如有）
- `prompt_sha256`：对 canonical prompt 求 SHA256（用于精确去重/交叉检查）
- `prompt_simhash`（可选）：用于近似去重
- `version`：数据版本标识（下载日期/commit/hash）

> 原则：任何一次训练/评测，都必须能回溯到“用了哪一份 manifest”。

---

## 3) Canonicalization（计算 hash 前的规范化）

统一把题面转换为 canonical 文本再 hash，建议规则：

- 统一换行符与空白：`\r\n → \n`、多空格压缩（保留必要换行）
- 去除显然无关字段：多余的版权声明、网页噪声（如有）
- 保留题面关键信息：题目描述、输入输出格式、约束、样例 I/O

输出：`canonical_prompt`，并计算 `prompt_sha256 = sha256(canonical_prompt)`。

---

## 4) 去重与泄漏检查（必须做，顺序固定）

### Step 1：split 内去重（intra-split）

- 对同一 split 内按 `prompt_sha256` 去重
- 保留 1 条作为主样本，其余进入 `duplicates_intrasplit.jsonl`（用于审计）

### Step 2：跨 split 精确交叉检查（inter-split exact）

检查集合交集：

- `train ∩ valid == ∅`
- `train ∩ test == ∅`
- `valid ∩ test == ∅`

处理策略（推荐）：

- 若 `train` 与 `test/valid` 冲突：**从 train 删除冲突样本**（保留 test/valid 的完整性）
- 若 `valid` 与 `test` 冲突：从 valid 删除冲突样本

### Step 3：HumanEval/MBPP 对标泄漏检查（exact）

- 计算 HumanEval/MBPP 的 `prompt_sha256`
- 检查是否出现在 CodeContests_train/valid

处理策略：

- 若发现冲突：从训练侧删除冲突样本，并在报告中显式列出冲突 ID（审计要求）

### Step 4（可选加分）：近似去重（near-dup）

如果题库规模较大且你担心“改写泄漏”，建议加一层近似检测：

- `prompt_simhash` 或 MinHash（基于 token 3-gram/5-gram）
- 设定阈值（如 Hamming distance ≤ 3）人工 spot-check

输出：`duplicates_near.jsonl` + 手工复核结论（抽样即可）。

---

## 5) 报告中必须出现的审计结果（建议 1 页）

至少包含：

- 每个 split 的样本数（去重前/后）
- 跨 split exact overlap 计数（应为 0）
- HumanEval/MBPP 与训练集 overlap 计数（应为 0）
- 若发生删除：列出删除策略与受影响样本数

