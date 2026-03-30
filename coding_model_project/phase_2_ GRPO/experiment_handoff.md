# GRPO 实验接手文档（2026-03-30）

本文档面向“直接接手本项目继续做远端实验”的 agent / 合作者。目标不是重复所有设计背景，而是把：

1. 当前代码和分支状态
2. 已经验证过的环境与命令
3. 训练所需组件在 GPU 机器上的搭建过程
4. 当前已经确认的链路状态、瓶颈和坑
5. 下一步最推荐的实验顺序

一次性交代清楚。

如果某块内容已经在已有文档里写得足够清楚，本文档会直接引用，不再重复展开。

---

## 1. 先读这些已有文档

以下文档已经覆盖了“为什么这么设计”和“代码链路怎么走”：

- 项目总进度：[../PROGRESS.md](../PROGRESS.md)
- Phase 2 总背景与实验计划：[README.md](README.md)
- 奖励设计与算法决策：[algorithm_decision_guide.md](algorithm_decision_guide.md)
- shared verifier 的详细设计：[shared_verifier_infra_guide.md](shared_verifier_infra_guide.md)
- RL / Eval 完整代码链路：[full_code_path_guide.md](full_code_path_guide.md)

本文档重点补的是：远端机器怎么搭、哪些命令已经跑过、当前 smoke 到了哪一步、哪里还卡着。

---

## 2. 当前代码快照

### 2.1 分支与提交

- 当前工作分支：`feature/grpo-development`
- 当前建议接手基线 commit：
  - `882026442725e44da49a8e94b9e56a1826f405a9`

这个 commit 已包含：

- shared verifier 主链接入
- eval 切换到 shared verifier
- batch reward 接入 verl
- trainer verifier metrics hook
- parquet 构建脚本
- `run_grpo_smoke.sh`
- `run_grpo_step_smoke.sh`

### 2.2 本轮新增的两个关键提交

- `6a42039ce154881fac1eca72bbe5a6283e3eb66a`
  - 新增更快的单步 smoke 脚本
  - parquet builder 增加 `step_smoke_*`
- `882026442725e44da49a8e94b9e56a1826f405a9`
  - 修正 tiny step smoke 的数据规模，避免 prompt 过滤后 dataloader 为空

---

## 3. 当前主链实现位置

shared truth / reward / eval 主链已经不再依赖 `default_compute_score` 或 `submit()`。

关键文件如下：

- shared verifier：
  - [`../src/verifier/shared.py`](../src/verifier/shared.py)
- RL batch reward：
  - [`../src/grpo_batch_reward.py`](../src/grpo_batch_reward.py)
- eval 主入口：
  - [`../src/phase0_eval.py`](../src/phase0_eval.py)
- parquet 构建：
  - [`../src/build_grpo_parquet.py`](../src/build_grpo_parquet.py)
- 标准 smoke：
  - [`run_grpo_smoke.sh`](run_grpo_smoke.sh)
- 快速单步 smoke：
  - [`run_grpo_step_smoke.sh`](run_grpo_step_smoke.sh)
- trainer verifier metrics：
  - [`../../verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py)
  - [`../../verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py)

如果需要理解这些文件之间如何调用，请直接看：

- [shared_verifier_infra_guide.md](shared_verifier_infra_guide.md)
- [full_code_path_guide.md](full_code_path_guide.md)

---

## 4. 重要原则与禁止事项

这些是当前主链必须坚持的约束：

- 不使用 `default_compute_score` 作为 shared truth / code reward 主链。
- 不使用 sandbox 的 `submit()` API。
- 所有 reward 与 eval 都只使用 `coding_model_project/data` 自带的 external test cases。
- `CodeContests` 必须按 full-test 聚合 `pass_ratio_all`。
- `HumanEval / MBPP` v1 只要求 normalized single-case summary，不承诺 testcase fan-out。

已经确认过的坑：

1. `default_compute_score` 会丢 metadata，不适合作为 shared truth。
2. `submit()` 在当前本地部署的 sandbox 下无法提供完整内置测试集。
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 和 vLLM memory pool 不兼容，不要和当前 rollout/vLLM 组合一起用。

---

## 5. 远端机器基线

下面是已经实际验证过的一台远端机器基线。未来换新机器，也建议尽量对齐。

### 5.1 机器与镜像

- GPU：`4 x NVIDIA GeForce RTX 5090`
- 单卡显存：约 `32GB`
- 镜像：`verlai/verl:vllm011.latest`

### 5.2 已验证版本

在这台机器上实际打印并确认过：

- `Python 3.12.11`
- `torch 2.8.0+cu128`
- `vllm 0.11.0`
- `ray 2.49.2`
- `pandas 2.3.3`
- `pyarrow 22.0.0`

结论：`verlai/verl:vllm011.latest` 和当前分支是兼容的，至少 eval + shared verifier + GRPO smoke 初始化这一层没有版本级 blocker。

---

## 6. 远端目录约定

下面这些路径是已经跑通过的一套约定，后续继续用最省心：

- repo 根目录：`/root/verl`
- raw 数据目录：`/root/verl/coding_model_project/data/raw`
- parquet 输出目录：`/root/verl/coding_model_project/data/grpo_parquet`
- sandbox server venv：`/root/sandboxfusion-venv`
- sandbox runtime venv：`/root/sandbox-runtime`
- sandbox server 日志：`/root/sandboxfusion-run/server.log`
- Ray 日志：`/tmp/ray/session_latest/logs`
- checkpoint 目录：`/root/verl/checkpoints/rlvr_coding_model/<experiment_name>`

---

## 7. 数据准备

### 7.1 raw 数据必须准备齐全

至少需要把以下 raw 文件放到 `coding_model_project/data/raw/`：

- `codecontests_train_wo_valid_big_raw.jsonl`
- `codecontests_valid_raw.jsonl`
- `codecontests_valid_big_raw.jsonl`
- `codecontests_test_raw.jsonl`
- `humaneval_raw.jsonl`
- `mbpp_reg_raw.jsonl`

`dataset_samples.jsonl` 保留也没问题，但不是主实验所必需。

### 7.2 重要说明

训练和评测都依赖这些 raw + manifest 生成的 external tests。

不要尝试改成依赖 sandbox 内置 dataset tables；当前这条路没有被验证，也不符合本项目当前 shared truth 的设计。

---

## 8. GPU 机器上的完整搭建过程

下面是已经在远端机器上实际用过的一套搭建流程。

### 8.1 拉代码与子模块

如果是新机器，建议：

```bash
git clone --recurse-submodules <repo-url> /root/verl
cd /root/verl
git checkout feature/grpo-development
git submodule update --init --recursive
```

如果 repo 已存在：

```bash
cd /root/verl
git fetch origin feature/grpo-development
git checkout feature/grpo-development
git pull --ff-only origin feature/grpo-development
git submodule update --init --recursive
```

### 8.2 安装项目本体与 sandbox client

在官方镜像容器里执行：

```bash
cd /root/verl
python3 -m pip install --no-deps -e /root/verl
python3 -m pip install --no-deps -e /root/verl/SandboxFusion/scripts/client
python3 -m pip install tenacity
```

说明：

- `--no-deps` 是有意的，避免覆盖镜像自带的 PyTorch/vLLM 栈。
- `tenacity` 是 client 运行时实际缺的一个依赖，已补过。

### 8.3 搭建 SandboxFusion server 环境

```bash
python3 -m venv /root/sandboxfusion-venv
/root/sandboxfusion-venv/bin/pip install "pydantic<2.7" fastapi "uvicorn[standard]==0.25.0" structlog psutil aiofiles aiohttp tenacity "databases[aiomysql,aiosqlite]" "transformers>=4.44.0"
mkdir -p /root/verl/SandboxFusion/docs/build
mkdir -p /root/sandboxfusion-run
```

启动 server：

```bash
cd /root/verl/SandboxFusion
nohup /root/sandboxfusion-venv/bin/python -m uvicorn sandbox.server.server:app --host 0.0.0.0 --port 8080 >/root/sandboxfusion-run/server.log 2>&1 &
```

验活：

```bash
curl -sf http://localhost:8080/v1/ping
```

预期返回：

```text
"pong"
```

### 8.4 搭建 sandbox runtime 兼容层

当前 SandboxFusion local runner 会尝试：

```bash
source /opt/miniconda3/bin/activate sandbox-runtime
```

如果机器上没有这套 conda 环境，需要做一个轻量兼容层。

1. 创建 runtime venv：

```bash
python3 -m venv /root/sandbox-runtime
```

2. 创建兼容目录：

```bash
mkdir -p /opt/miniconda3/bin
mkdir -p /opt/miniconda3/condabin
```

3. 写一个最小 `activate` shim：

```bash
cat >/opt/miniconda3/bin/activate <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "sandbox-runtime" ]; then
  export VIRTUAL_ENV=/root/sandbox-runtime
  export PATH="/root/sandbox-runtime/bin:$PATH"
  return 0 2>/dev/null || exit 0
fi
echo "Unsupported env: ${1:-}" >&2
return 1 2>/dev/null || exit 1
EOF
chmod +x /opt/miniconda3/bin/activate
```

4. 验证：

```bash
bash -c 'source /opt/miniconda3/bin/activate sandbox-runtime; which python'
```

预期输出：

```text
/root/sandbox-runtime/bin/python
```

如果这一步不通，`/run_code` 会失败。

---

## 9. parquet 构建

### 9.1 标准构建命令

```bash
cd /root/verl
python3 coding_model_project/src/build_grpo_parquet.py \
  --data_root coding_model_project/data \
  --output_dir coding_model_project/data/grpo_parquet
```

### 9.2 当前建议的 step smoke 构建命令

为了避免 tiny train 被 prompt 过滤后直接空掉，建议显式指定：

```bash
cd /root/verl
python3 coding_model_project/src/build_grpo_parquet.py \
  --data_root coding_model_project/data \
  --output_dir coding_model_project/data/grpo_parquet \
  --step_smoke_train_size 16 \
  --step_smoke_val_size 8
```

最新一次已验证输出：

```json
{
  "train": 11785,
  "val_tier1": 317,
  "val_tier2": 500,
  "final_eval": 329,
  "smoke_train": 64,
  "smoke_val": 32,
  "step_smoke_train": 16,
  "step_smoke_val": 8
}
```

---

## 10. 已验证通过的命令

### 10.1 `/run_code` smoke

成功样例：

```bash
python3 - <<'PY'
from sandbox_fusion import RunCodeRequest, run_code
req = RunCodeRequest(language='python', code='print(123)', run_timeout=3, compile_timeout=3)
resp = run_code(req, endpoint='http://localhost:8080')
print(resp.status)
print(resp.run_result.status)
print((resp.run_result.stdout or '').strip())
PY
```

已验证结果：

- `RunStatus.Success`
- `CommandRunStatus.Finished`
- `stdout = 123`

超时样例：

```bash
python3 - <<'PY'
from sandbox_fusion import RunCodeRequest, run_code
req = RunCodeRequest(language='python', code='while True:\n    pass', run_timeout=1, compile_timeout=3)
resp = run_code(req, endpoint='http://localhost:8080')
print(resp.status)
print(resp.run_result.status if resp.run_result else None)
PY
```

已验证结果：

- `RunStatus.Failed`
- `CommandRunStatus.TimeLimitExceeded`

### 10.2 eval smoke

命令：

```bash
cd /root/verl/coding_model_project
python3 src/phase0_eval.py \
  --mode simple \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --vllm_url http://localhost:8001 \
  --sandbox_url http://localhost:8080 \
  --manifest_dir data/manifests \
  --datasets codecontests_valid \
  --temperature 0.0 \
  --max_tokens 512 \
  --run_timeout 30 \
  --max_concurrent 4 \
  --verifier_limiter_budget 4 \
  --batch_size 1 \
  --max_problems 1 \
  --output_dir outputs/phase0_smoke_reboot
```

已验证结果：

- `accepted@1 = 0.00%`
- `pass_ratio_mean = 0.2200`

并且 per-problem 输出已确认：

- 顶层存在 `per_case_results`
- `len(per_case_results) = 50`
- `passed_tests = 11`
- `total_tests = 50`
- `pass_ratio_all = 0.22`

结论：eval + shared verifier + CodeContests full-test 聚合已经通。

---

## 11. GRPO smoke / step-smoke 的现状

### 11.1 标准 smoke 脚本

文件：

- [run_grpo_smoke.sh](run_grpo_smoke.sh)

这条链已经验证过能够：

- 完成 Ray 初始化
- 完成 rollout / vLLM server 初始化
- 进入 validation generation
- 在训练主循环里触发 shared verifier + sandbox judge

但默认 smoke 墙钟时间很长，不适合快速看“能不能打出 step”。

### 11.2 快速单步 smoke 脚本

文件：

- [run_grpo_step_smoke.sh](run_grpo_step_smoke.sh)

默认策略：

- 用 `step_smoke_train.parquet`
- 用 `step_smoke_val.parquet`
- `train_batch_size = 8`
- `ppo_mini_batch_size = 8`
- `val_before_train = False`
- `test_freq = 1000`
- `total_training_steps = 1`（由 dataloader 大小实际收敛成 1）
- `max_response_length = 512`

推荐启动命令：

```bash
cd /root/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled
export ROLLOUT_TP_SIZE=2
export ROLLOUT_GPU_MEM_UTIL=0.5
export MAX_RESPONSE_LENGTH=512

bash 'coding_model_project/phase_2_ GRPO/run_grpo_step_smoke.sh' \
  trainer.experiment_name=grpo_step_smoke_shared_verifier_probe
```

---

## 12. 本轮真机验证结论

### 12.1 已确认打通的部分

下面这些已经不再是 blocker：

1. `Hydra` 自定义 reward kwargs 注入
2. `reward_model.use_reward_loop=False` 之后误入 experimental reward loop
3. shared verifier 接入 RL 主链
4. batched reward 触发 sandbox judge
5. CodeContests full-test reward/eval 聚合

### 12.2 我实际观察到的 RL 链路状态

在 `step_smoke_train=16 / step_smoke_val=8` 的单步 smoke 下，远端日志已经出现：

- `Size of train dataloader: 1`
- `Total training steps: 1`
- `Training Progress: 0/1`

这说明单步 smoke 已经真的进入训练循环，不再停留在 init 阶段。

### 12.3 reward judge 的并发证据

在 step 过程中，sandbox server 日志中出现了同一秒内多条 judge 请求，例如：

- 多次 `start processing python request ...`
- 多次 `running command python /tmp/...`

这说明：

- reward 已经真的调用 shared verifier
- verifier 已经把多个 testcase / 样本并发发给 sandbox
- sandbox 侧已经不是串行评测

### 12.4 目前最主要的真实失败点

当前最清楚、最稳定复现的失败点是：

- **actor update 阶段的显存 OOM**

具体出现在：

- `actor_rollout_update_actor`
- `self.actor_optimizer.step()`
- `torch._foreach_sqrt(device_exp_avg_sqs)`

错误核心：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 520.00 MiB
```

这说明在 4x5090、当前 FSDP2 + actor update 配置下：

- rollout / reward / verifier 已经能走到
- 但 optimizer step 还不够稳

### 12.5 关于“吞吐瓶颈是不是主要在 sandbox”

当前结论是：

- **reward judge 确实是 step 内的重要耗时段**
- 但**不能说总吞吐瓶颈只在 sandbox**

因为已观察到两类重耗时：

1. **冷启动成本**
   - vLLM / CUDA graph capture / AgentLoop / worker 初始化
2. **actor update 成本**
   - 当前会直接 OOM，导致一步跑不完

所以更准确的结论是：

- shared verifier + sandbox 并发链路已经通
- reward judge 是后续值得横向扩展的对象
- 但在做多 sandbox 之前，最好先把“一步 smoke 稳定完成”解决掉，否则很难干净地归因吞吐

---

## 13. 已踩过的坑

### 13.1 不要再走 `default_compute_score`

原因见：

- [shared_verifier_infra_guide.md](shared_verifier_infra_guide.md)

简述：

- 只看前 10 个 testcase
- 会丢 metadata
- 不适合 shared truth

### 13.2 不要用 `submit()`

当前本地部署的 sandbox 无法提供完整内置测试集，因此 `submit()` 不适合作为主链。

### 13.3 不要给当前 rollout/vLLM 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

会直接报：

```text
Expandable segments are not compatible with memory pool
```

这是 vLLM memory pool 层面的不兼容，不要继续尝试这条路。

### 13.4 tiny step smoke 数据一定要显式指定训练集大小

如果只取 `8` 条，经过 prompt 过滤后可能掉到 `7`，进而触发：

```text
AssertionError: Train dataloader is empty!
```

所以当前建议始终显式传：

```bash
--step_smoke_train_size 16 --step_smoke_val_size 8
```

---

## 14. 接手后的推荐顺序

### 第一优先级：把一步 smoke 稳定跑完

当前最合适的入口就是：

- [run_grpo_step_smoke.sh](run_grpo_step_smoke.sh)

推荐继续试的方向：

1. 优先处理 actor update OOM
2. 不要动 reward timeout
3. 不要改 shared verifier 主链
4. 尽量只做 smoke 级别的内存兜底

优先可尝试的方向：

- `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True`
- 如果还不稳，再看 actor/FSDP 侧更保守的 smoke 配置

### 第二优先级：在“一步能稳定完成”后再做 sandbox 吞吐归因

建议做两组对比：

1. 单 sandbox，改 `limiter_budget`
   - `1 / 2 / 4 / 8`
2. 多 sandbox 实例
   - 先服务侧做多个实例
   - 训练侧仍然只认一个入口 URL

### 第三优先级：只有在上一步完成后，才开始做 sandbox 水平扩展

当前不建议一上来就改成多 sandbox client 逻辑。

先确认：

- 单步 smoke 能稳定结束
- reward judge 的 wall time 占比足够高
- `limiter_budget` 提升确实带来吞吐改善

再去做多个 sandbox 实例，会更好归因，也更好答辩。

---

## 15. 建议保留的日志与证据

接手实验时，建议优先保留这些证据：

- parquet 构建输出：
  - `coding_model_project/data/grpo_parquet/build_summary.json`
- eval 结果：
  - `coding_model_project/outputs/phase0_*`
- sandbox server 日志：
  - `/root/sandboxfusion-run/server.log`
- Ray 日志：
  - `/tmp/ray/session_latest/logs`
- 每次实验的最终 commit：
  - `git rev-parse HEAD`

这些信息足够支撑：

- shared verifier 主链已接通
- reward judge 已发生
- 当前失败点具体在哪个阶段

---

## 16. 最后一句话总结

当前项目已经从“reward / eval 真值不一致、sandbox 主链不可信”的状态，推进到了：

- eval 可用
- shared verifier 可用
- RL reward 已真实接入
- sandbox judge 已在训练时并发执行

当前剩余最现实的问题不是 reward 设计，而是：

- 先让 `4x5090` 上的一步 GRPO smoke **稳定跑完**
- 然后再量化 `sandbox` 是否是主要吞吐瓶颈
- 最后再做多 sandbox 水平扩展

