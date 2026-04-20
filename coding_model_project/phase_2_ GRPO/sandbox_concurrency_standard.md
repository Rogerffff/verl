# Sandbox 并发启动规范

## 目标

这份文档用于统一 `SandboxFusion + direct-backend client RR` 的正式评测 / teacher QC 并发口径，避免再次误用 `limiter_budget=8` 一类只适合 smoke test 的低并发配置。

适用范围：

- `valid_big500` / `codecontests_test` 正式评测
- repair eval
- teacher primary QC / rejudge
- 其他需要跑完整 CodeContests testcase 的大批量判题任务

不适用范围：

- 1~10 条样本的 smoke test
- 新机刚起时的 backend 健康检查

## 当前标准环境

- backend 直连 RR：
  - `http://localhost:8081`
  - `http://localhost:8082`
  - `http://localhost:8083`
  - `http://localhost:8084`
  - `http://localhost:8085`
  - `http://localhost:8086`
  - `http://localhost:8087`
  - `http://localhost:8088`
- 统一使用 `phase_2_ GRPO/ops/setup_eval_sandbox_4x2.sh` 启 8 个 backend
- 正式评测 / QC 默认不要依赖 LB；优先直接打 `8081..8088`

## 历史依据

来自现有 handoff 与实际 run 记录，已经验证过的经验值是：

- 单池跑 `valid_big500` 的稳定大并发：
  - `MAX_CONCURRENT=200`
  - `MAX_CONCURRENT_JUDGES=180`
  - `VERIFIER_LIMITER_BUDGET=180`
  - `BATCH_SIZE=200`
- 三池并行跑 `delta69` 的稳定配置：
  - `MAX_CONCURRENT=69`
  - `MAX_CONCURRENT_JUDGES=72`
  - `VERIFIER_LIMITER_BUDGET=72`
  - `BATCH_SIZE=69`

关键理解：

- `limiter_budget` 约束的是**真实在途 sandbox RPC 数**
- 不是“题目数”
- 单题内部还会 fan-out 到 testcase 级请求
- 因此过小的 `limiter_budget` 会显著拉长尾部耗时

## 正式任务默认并发

### 1. 正式评测 / full QC 默认值

对于 8 backend 的 direct RR，默认用：

- `workers=96`
- `limiter_budget=180`

这是当前推荐的正式大并发口径。

### 2. 保守但仍然是大并发的回退值

如果新机器 CPU 偏弱，或者观察到 backend 明显拥堵，再回退到：

- `workers=64`
- `limiter_budget=180`

注意：

- 这里仍然保留 `limiter_budget=180`
- 不要为了“保守”把 `limiter_budget` 直接降到个位数或十几

### 3. 明确禁止的正式配置

以下配置只允许用于 smoke test，不允许用于 full QC / full eval：

- `workers=8`
- `limiter_budget=8`
- 任意 `limiter_budget <= 32`

原因：

- 这类配置会严重放大长尾
- 对 500 题、1000+ teacher candidates 这类任务会明显拖慢整体 wall-clock
- 容易造成“GPU 空闲但判题迟迟不结束”的假瓶颈

## 启动前检查

正式任务前必须确认：

1. 8 个 backend 都活着
2. 日志路径都存在
3. backend 端口都能响应
4. 不存在旧的低并发 full QC 进程残留

建议检查项：

```bash
bash /workspace/verl_repo/coding_model_project/phase_2_GRPO/ops/sandbox_backend_status.sh
ps -eo pid,etime,cmd | grep qc_repair_conditioned_teacher_candidates.py | grep -v grep
```

## 标准 URL 写法

```text
http://localhost:8081,http://localhost:8082,http://localhost:8083,http://localhost:8084,http://localhost:8085,http://localhost:8086,http://localhost:8087,http://localhost:8088
```

## 推荐命令模板

### teacher primary QC

```bash
python3 -u coding_model_project/phase_2_GRPO/sft_repair_data/scripts/qc_repair_conditioned_teacher_candidates.py \
  --input_candidates <input_jsonl> \
  --output_results <output_jsonl> \
  --summary_out <summary_json> \
  --sandbox_url http://localhost:8081,http://localhost:8082,http://localhost:8083,http://localhost:8084,http://localhost:8085,http://localhost:8086,http://localhost:8087,http://localhost:8088 \
  --workers 96 \
  --limiter_budget 180 \
  --run_timeout 30 \
  --memory_limit_mb 1024
```

### accepted-only rejudge / full rejudge

优先仍用：

- `workers=96`
- `limiter_budget=180`

如果机器负载明显偏高，再回退到：

- `workers=64`
- `limiter_budget=180`

## 执行规则

1. 正式任务默认先用 `96 / 180`
2. 如果确实需要回退，先回退到 `64 / 180`
3. 只有 smoke test 才允许 `8 / 8`
4. 如果有人想改成更小并发，必须先说明原因，并明确这是 smoke 还是正式 run
5. 正式 run 的命令、日志、summary 中都要把并发参数写清楚

## 结论

从现在开始，`8 backend direct RR` 的正式判题口径默认就是：

- `workers=96`
- `limiter_budget=180`

保守回退值是：

- `workers=64`
- `limiter_budget=180`

`limiter_budget=8` 这类配置不再用于正式 full QC / full eval。
