# 数据治理审计报告 (Data Governance Audit Report)

生成时间: 2026-02-02 21:38:21

---

## 1. 样本数统计

| 数据集 | 去重前 | 去重后 | Split内重复 | 跨Split移除 |
|--------|--------|--------|-------------|-------------|
| humaneval | 164 | 164 | 0 | 0 |
| mbpp_reg | 200 | 200 | 0 | 0 |
| codecontests_train | 13328 | 12285 | 1043 | 0 |
| codecontests_valid | 117 | 117 | 0 | 0 |
| codecontests_test | 165 | 165 | 0 | 0 |

## 2. 跨 Split 精确重叠检查

**所有 Split 交集均为空 ✓**

## 3. 外部基准泄漏检查 (HumanEval/MBPP)

**训练集与外部基准无泄漏 ✓**

## 4. 最终验证

- `codecontests_train` ∩ `codecontests_valid` = 0 ✓
- `codecontests_train` ∩ `codecontests_test` = 0 ✓
- `codecontests_valid` ∩ `codecontests_test` = 0 ✓
- `codecontests_train` ∩ `humaneval` = 0 ✓
- `codecontests_train` ∩ `mbpp_reg` = 0 ✓
- `codecontests_valid` ∩ `humaneval` = 0 ✓
- `codecontests_valid` ∩ `mbpp_reg` = 0 ✓

**所有检查通过 ✓**

---

## 5. 数据集角色说明

| 数据集 | 角色 | 说明 |
|--------|------|------|
| humaneval | test_only | OpenAI 代码生成基准，仅用于评测 |
| mbpp_reg | test_only | Google Python 编程基准 (ID 11-210)，仅用于评测 |
| codecontests_train | train | CodeContests 训练集，用于 RL 训练 |
| codecontests_valid | validation | CodeContests 验证集，用于超参调优 |
| codecontests_test | test | CodeContests 测试集，用于最终评测 |