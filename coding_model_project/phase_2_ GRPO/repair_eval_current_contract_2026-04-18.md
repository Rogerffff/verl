# Current Repair Eval Contract (2026-04-18)

这份文档只做一件事：

- 把当前 repair 评测口径重新梳理清楚
- 明确 `Protocol A / Protocol B`
- 明确 `reuse_step900 / reuse_step1300`
- 明确 `valid_big500 / codecontests_test`
- 明确哪些结果已经存在，哪些还需要补

它不是历史背景文档，也不是实验总结文档。它的目标是避免后面继续把不同协议、不同固定输入源、不同 lineage 的 checkpoint 混在一张表里。

---

## 1. 先给结论

如果只记一条，记这一条：

- **主 repair-SFT headline 一律看 `Protocol A + reuse_step900`**
- **真实部署能力 headline 一律看 `Protocol B`**

换句话说：

1. `Protocol A` 用来回答：
   - 在**同一份坏代码、同一份 verifier feedback** 上，谁的 repair 能力更强
2. `Protocol B` 用来回答：
   - 如果模型上线，先自己生成 first pass，再自己修，**整条 end-to-end pipeline** 谁更强

这两张表都需要，但回答的问题不同，不能互相替代。

---

## 2. 术语定义

### 2.1 Protocol A

固定输入 repair eval：

- 冻结一份 canonical first-pass artifact
- 所有候选模型都复用同一份：
  - problem statement
  - first-pass code
  - verifier feedback
- 只比较 second-pass repair

这是当前 **repair-SFT 主评测口径**。

### 2.2 Protocol B

self-first-pass repair eval：

- 每个模型先自己生成 raw first pass
- 再对自己的 first pass 做 repair

这是当前 **部署式 / end-to-end 口径**。

### 2.3 `reuse_step900`

固定输入源来自 `step900` 的 canonical raw `per_problem`。

当前历史主表默认的 canonical source 是：

- `valid_big500`:
  - `/workspace/verl_repo/coding_model_project/outputs/rl_codecontests_validbig500/grpo_a1_curriculum_step620_v3q3_to900_seed0_step900_codecontests_validbig500_raw_patched_rr828x_vastai3/per_problem`

对于 `codecontests_test`，当前文档中已经把 completed Protocol A 结果也统一写成：

- canonical reused first pass from `step900`

所以**如果目标是 apples-to-apples 历史可比表，默认就是 `reuse_step900`**。

### 2.4 `reuse_step1300`

固定输入源来自 `step1300` 的 canonical raw `per_problem`。

这个口径可以做，但它回答的是另一类问题：

- `step1300` 风格的 first-pass failure，谁更会修？

它是**探索性 secondary probe**，不是当前主表口径。

所以：

- `reuse_step1300` 结果不能直接和 `reuse_step900` 主表混成一个 headline 表

---

## 3. 数据集各自回答什么问题

### 3.1 `valid_big500`

这是当前 repair-SFT 的开发集主舞台。

它适合做：

- checkpoint 选择
- prompt mode 比较
- Protocol A 主表
- Protocol B 开发侧验证

### 3.2 `codecontests_test`

这是 held-out test set，当前规模是 `165` 题。

引用：

- [README.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/README.md)

它适合做：

- held-out repair skill check
- end-to-end deployment check

当前推荐读法：

- `Protocol A / codecontests_test`
  - 看“固定输入 repair skill 能不能迁移到 test”
- `Protocol B / codecontests_test`
  - 看“真实上线时，这个模型整条 repair pipeline 是否更强”

---

## 4. 当前必须避免的混淆

### 4.1 不要把 `reuse_step900` 和 `reuse_step1300` 混成一张表

这两个问题不同：

- `reuse_step900`
  - 更适合历史主表、repair-SFT 公平比较
- `reuse_step1300`
  - 更适合 probing：当 failure 风格换成 `step1300` 时，repair 会不会更强

因此：

- `Protocol A + reuse_step900`
  - 是当前 **canonical primary table**
- `Protocol A + reuse_step1300`
  - 只能当 secondary analysis

### 4.2 不要把旧 `step40` 和新 `step40` 混掉

现在至少有两套不同 lineage 会产生“`step40`”这个名字：

1. 旧 repair-SFT 线里的 `step40`
   - 已经写进：
     - [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)
     - [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)
     - [repair_eval_protocolA_step900_vs_sft30_60.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_protocolA_step900_vs_sft30_60.md)

2. 当前这轮 `step1300` base SFT canary 的 `global_step_40`
   - 路径：
     - `/workspace/verl_repo/checkpoints/rlvr_coding_model/phase2_step1300_shortdiag_pure_sft_v1_len6144_keep6/global_step_40/huggingface`

从现在开始，建议命名明确写成：

- `legacy_sft_step40`
- `step1300_sft_v1_step20`
- `step1300_sft_v1_step40`
- `step1300_sft_v1_step60`

不要再单独写裸的 `step40` / `step60`。

---

## 5. 当前已经完成的结果

### 5.1 已完成：历史主表（旧 repair-SFT 线）

这些结果已经存在，而且可以直接引用。

#### Protocol A + `reuse_step900`

文档：

- [repair_protocolA_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolA_results_2026-04-18.md)

已完成比较：

- `step900`
- `legacy_sft_step40`
- `step1300`

数据集：

- `valid_big500`
- `codecontests_test`

#### Protocol B

文档：

- [repair_protocolB_results_2026-04-18.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_protocolB_results_2026-04-18.md)

已完成比较：

- `step900`
- `legacy_sft_step40`
- `step1300`

数据集：

- `valid_big500`
- `codecontests_test`

### 5.2 已完成：旧 `step900` repair-SFT 线内部 checkpoint 对比

文档：

- [repair_eval_protocolA_step900_vs_sft30_60.md](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/repair_eval_protocolA_step900_vs_sft30_60.md)

注意：

- 这里的 `step30 / step40 / step50 / step60`
- 指的是**旧 repair-SFT 线**
- 不是当前 `step1300` base canary

---

## 6. 当前这轮 `step1300` base SFT canary 应该怎么评

当前新 checkpoint 是：

- `step1300_sft_v1_step20`
  - `/workspace/verl_repo/checkpoints/rlvr_coding_model/phase2_step1300_shortdiag_pure_sft_v1_len6144_keep6/global_step_20/huggingface`
- `step1300_sft_v1_step40`
  - `/workspace/verl_repo/checkpoints/rlvr_coding_model/phase2_step1300_shortdiag_pure_sft_v1_len6144_keep6/global_step_40/huggingface`
- `step1300_sft_v1_step60`
  - `/workspace/verl_repo/checkpoints/rlvr_coding_model/phase2_step1300_shortdiag_pure_sft_v1_len6144_keep6/global_step_60/huggingface`

它们的直接 baseline 是：

- `step1300_rl`
  - `/workspace/verl_repo/checkpoints/rlvr_coding_model/grpo_a1_curriculum_step1200_followup_u6a2b4c3d1_to1400_lb240_seed0_vastai3/global_step_1300/actor_merged_hf_step1300_repair_cond_v2`

### 6.1 当前最该先跑的主表

先跑：

- `Protocol A`
- `reuse_step900`
- `valid_big500`

比较对象：

- `step1300_rl`
- `step1300_sft_v1_step20`
- `step1300_sft_v1_step40`
- `step1300_sft_v1_step60`

原因：

- 这是和历史主表最可比的口径
- 能最干净回答：
  - 在 canonical `step900` failure slice 上
  - 这轮 `step1300` base SFT 到底有没有学到更强 repair skill

### 6.2 接下来再跑什么

如果上面这张 `valid_big500 / Protocol A / reuse_step900` 表里：

- `step1300_sft_v1_step40` 或 `step1300_sft_v1_step60`
- 相对 `step1300_rl` 有明确 gain

那么下一步再跑：

- `Protocol B`
- `valid_big500`
- 比较：
  - `step1300_rl`
  - 最优的 `step1300_sft_v1_stepX`

然后再跑 held-out：

- `Protocol A`
- `codecontests_test`
- 只补：
  - `step1300_rl`
  - 最优的 `step1300_sft_v1_stepX`

最后再跑：

- `Protocol B`
- `codecontests_test`
- 比较：
  - `step1300_rl`
  - 最优的 `step1300_sft_v1_stepX`

### 6.3 什么时候才跑 `reuse_step1300`

只有在下面这个问题明确值得问的时候：

- “如果固定输入源改成 `step1300` 自己的 raw failure，SFT 会不会更有优势？”

这时才跑：

- `Protocol A`
- `reuse_step1300`
- `valid_big500`

它是有价值的，但它是 **secondary probe**，不是当前 headline。

---

## 7. 当前推荐的评测优先级

### Priority 1

主评测：

- `Protocol A`
- `reuse_step900`
- `valid_big500`
- 对比：
  - `step1300_rl`
  - `step1300_sft_v1_step20`
  - `step1300_sft_v1_step40`
  - `step1300_sft_v1_step60`

### Priority 2

从 `20/40/60` 中挑出 best checkpoint。

### Priority 3

部署验证：

- `Protocol B`
- `valid_big500`
- 只跑：
  - `step1300_rl`
  - best `step1300_sft_v1_stepX`

### Priority 4

held-out fixed-input：

- `Protocol A`
- `reuse_step900`
- `codecontests_test`
- 只跑：
  - `step1300_rl`
  - best `step1300_sft_v1_stepX`

### Priority 5

held-out deployment：

- `Protocol B`
- `codecontests_test`
- 只跑：
  - `step1300_rl`
  - best `step1300_sft_v1_stepX`

### Priority 6

如果前面有正信号，再做：

- `Protocol A`
- `reuse_step1300`
- `valid_big500`

作为 supplementary probe。

---

## 8. 一句话执行规则

后面任何人继续跑 repair eval 时，先问自己：

1. 我要比较的是 **repair skill**，还是 **部署 end-to-end 能力**？
   - repair skill -> `Protocol A`
   - 部署能力 -> `Protocol B`

2. 我要的是 **历史主表可比**，还是 **step1300-style failure probe**？
   - 历史主表可比 -> `reuse_step900`
   - step1300-style probe -> `reuse_step1300`

3. 我是在做 **开发集选 ckpt**，还是 **held-out final check**？
   - 选 ckpt -> 先 `valid_big500`
   - final check -> 再 `codecontests_test`

只要这三问答清楚，当前评测合同就不会再混。

---

## 9. 2026-04-18 当前已完成答案

上面第 6、7 节讲的是“应该怎么跑”。到 2026-04-18 这一轮结束时，当前 `step1300`-base repair-SFT v1 的关键问题已经实际跑完，结论如下。

### 9.1 `valid_big500` 已完成

#### Protocol A + `reuse_step900`

已完成比较：

- `step1300_rl`
- `step1300_sft_v1_step20`
- `step1300_sft_v1_step40`
- `step1300_sft_v1_step60`

结论：

- `step20` 和 `step60` 并列最好
- 两者都比 `step1300_rl` 多救回 `2` 题
- 所以从 fixed-input repair skill 看，这轮 v1 **确实学到了东西**

#### Protocol B + self-first-pass

已完成比较：

- `step1300_rl`
- `step1300_sft_v1_step20`
- `step1300_sft_v1_step40`
- `step1300_sft_v1_step60`

结论：

- `step60` 是当前 current-lineage 的 best checkpoint
- 所有 v1 checkpoint 都优于 `step1300_rl` baseline
- 所以从开发集端到端 repair 看，`step60` 是当前最该继续追踪的 v1 ckpt

### 9.2 `codecontests_test` 已完成到 `step60`

当前 held-out follow-up 已完成：

- `Protocol A + reuse_step900`
  - `step1300_rl`
  - `step1300_sft_v1_step60`
- `Protocol B + self-first-pass`
  - `step1300_rl`
  - `step1300_sft_v1_step60`

结论：

- `Protocol A`
  - 两者都没有净 gain
  - `step60` 没有把 dev-side fixed-input gain 迁移到 held-out test
- `Protocol B`
  - `step60` 有 self-repair gain
  - 但最终仍然低于 `step1300_rl`

所以：

- `step1300_sft_v1_step60` 是当前 **最佳 v1 repair-SFT comparison checkpoint**
- `step1300_rl` 仍然是当前 **最强 held-out deployed model**

### 9.3 对 `v2` 的当前拍板

如果继续做 `step1300 repair-SFT v2`，当前答案是：

- **可以继续做**
- 但必须按“改配方的 v2”来做
- 不能把当前 v1 当成一个只要继续放大就会稳定赢过 `step1300_rl` 的配方

当前最关键的 recipe change 方向是：

- 降低 `C_hard_partial` 的主导占比
- 增加 `Core / ExpansionB` 的占比
- 如果训练步数和数据规模明显扩大，加入少量 anchor/general code rows
- 继续用 `step1300` 作为 stronger base，而不是退回旧 base
