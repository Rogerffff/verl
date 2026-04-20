# 新机器迁移与 Teacher QC 运行手册（2026-04-15）

本文档用于把当前 `step900 repair-conditioned teacher generation -> QC` 这条线，从暂时失联的旧机器迁移到新的 `4 x 5090` 机器上继续执行。

默认前提：

- 新机器镜像仍然使用 `verlai/verl:vllm011.latest`
- 新机器工作目录仍然使用 `/workspace/verl`
- 当前正式评测 / QC 口径仍然是：
  - `patched sandbox`
  - `direct-backend client RR`
  - **不走** nginx LB 单入口作为 canonical eval/QC 路径

相关背景文档：

- [experiment_handoff.md](experiment_handoff.md)
- [step900_teacher_generation_qc_plan.md](step900_teacher_generation_qc_plan.md)
- [step900_teacher_prompt_and_schema_v2_backend_rr.md](step900_teacher_prompt_and_schema_v2_backend_rr.md)
- [step900_teacher_shards_v2_backend_rr.md](step900_teacher_shards_v2_backend_rr.md)

---

## 1. 当前本地资产状态

截至 2026-04-15，本地已经确认：

- 9/9 Claude shard 全部返回
- `281 requests`
- `429 generation units`
- 全量 preflight 通过：
  - `finish_reason = stop` 共 `429`
  - `extraction_status = ok` 共 `429`
  - `BUG_SUMMARY / FIX_PLAN / single <code>` 全部齐全

当前权威产物：

- teacher request：
  - `coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/teacher_generation_requests_step900_candidate_v2_backend_rr.jsonl`
- shard manifest：
  - `coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/manifest.json`
- teacher raw outputs 根目录：
  - `coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/`
- 全量 preflight summary：
  - `coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_generation_preflight_v2_full.summary.json`
- 全量 assembled candidates：
  - `coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_candidates_v2_full.jsonl`
- 全量 assembled summary：
  - `coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_candidates_v2_full.summary.json`
- repair-conditioned QC 脚本：
  - `coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/qc_repair_conditioned_teacher_candidates.py`

当前三层分布：

- `Core = 133`
- `Expansion = 208`
- `SelectiveC = 88`

当前 `SelectiveC` provenance 分布：

- `heuristic_supplemental_c = 70`
- `legacy_reviewed_c = 18`

---

## 2. 迁移结论：是否只迁移 `/workspace/verl`

### 2.1 结论

如果你的目标是：

- 继续 `teacher QC`
- 后续继续 `repair-conditioned SFT`
- 以及保留当前 repo、checkpoint、teacher 资产

那么：

- **迁移整个 `/workspace/verl` 就足够了**

原因：

- 当前 repo 根 `/workspace/verl` 内已经包含：
  - `SandboxFusion/`
  - `coding_model_project/`
  - `checkpoints/`
  - `coding_model_project/data/raw`
  - `coding_model_project/data/manifests`
  - `phase_2_ GRPO` 文档、脚本、teacher assets
  - 这轮 `teacher_generation_outputs` 与 `teacher_generation_shards`

### 2.2 额外推荐但非必须迁移

如果你想保留历史运行日志，建议额外迁移：

- `/workspace/eval_logs`

如果你只关心继续执行，不关心旧日志，则这一步不是必须。

### 2.3 不建议迁移的机器态目录

这些目录不值得从旧机器硬搬，直接在新机器重建即可：

- `/root/sandboxfusion-venv`
- `/root/sandbox-runtime`
- `/root/sandboxfusion-multi`
- `/tmp/ray`

这些内容要么是 venv，要么是 sandbox 运行态和临时日志，不适合作为跨机器 source-of-truth。

---

## 3. 如果不整棵迁移，至少要带哪些目录

如果你不是整棵复制 `/workspace/verl`，而是做 selective transfer，那么至少要包含下面这些路径：

- `/workspace/verl/SandboxFusion`
- `/workspace/verl/coding_model_project/data/raw`
- `/workspace/verl/coding_model_project/data/manifests`
- `/workspace/verl/coding_model_project/data/problem_quarantine_v3.json`
- `/workspace/verl/coding_model_project/phase_2_ GRPO`
- `/workspace/verl/coding_model_project/outputs/repair_conditioned_student_reference_eval/step900_candidate_v2_backend_rr_student_ref`
- `/workspace/verl/coding_model_project/phase_2_ GRPO/sft_repair_data/provisional_repair_conditioned/step900_candidate_v2_backend_rr`
- `/workspace/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr`
- `/workspace/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr`

如果后续还要继续 RL / checkpoint eval，则还需要：

- `/workspace/verl/checkpoints`
- `/workspace/verl/coding_model_project/curriculum_states`

---

## 4. 新机器第一阶段：落盘与完整性检查

假设你已经把本地或旧远端内容传到新机器，并落在：

- `/workspace/verl`

先做最小完整性检查：

```bash
cd /workspace/verl

test -d SandboxFusion
test -d coding_model_project
test -d checkpoints
test -f "coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/manifest.json"
test -f "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_candidates_v2_full.jsonl"
test -f "coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/qc_repair_conditioned_teacher_candidates.py"
```

然后建议再看一眼当前全量 summary：

```bash
cat "/workspace/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_candidates_v2_full.summary.json"
cat "/workspace/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_generation_preflight_v2_full.summary.json"
```

期望看到：

- `assembled_candidate_count = 429`
- `missing_generation_unit_count = 0`
- `Core = 133`
- `Expansion = 208`
- `SelectiveC = 88`

---

## 5. 新机器第二阶段：环境安装

### 5.1 Python / repo editable install

```bash
cd /workspace/verl

python3 -m pip install --no-deps -e /workspace/verl
python3 -m pip install --no-deps -e /workspace/verl/SandboxFusion/scripts/client
python3 -m pip install tenacity
```

### 5.2 系统依赖

`setup_eval_sandbox_4x2.sh` / `sandbox_backend_start.sh` 这条链依赖：

- `nginx`
- `make`
- `curl`
- `lsof`（没有也能运行，但建议装上）

建议直接装：

```bash
apt-get update
apt-get install -y nginx make curl lsof
```

### 5.3 sandbox server venv

```bash
python3 -m venv /root/sandboxfusion-venv

/root/sandboxfusion-venv/bin/pip install \
  "pydantic<2.7" \
  fastapi \
  "uvicorn[standard]==0.25.0" \
  structlog \
  psutil \
  aiofiles \
  aiohttp \
  tenacity \
  "databases[aiomysql,aiosqlite]" \
  "transformers>=4.44.0"
```

### 5.4 sandbox runtime shim

这一步保留和旧 bring-up 口径一致，避免 SandboxFusion 内部代码执行链找不到 `sandbox-runtime`：

```bash
python3 -m venv /root/sandbox-runtime
mkdir -p /opt/miniconda3/bin /opt/miniconda3/condabin

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

---

## 6. 新机器第三阶段：启动 patched sandbox backend

当前 canonical QC 口径是：

- backend 直接地址
- client-side RR
- 即：
  - `http://localhost:8081,...,http://localhost:8088`

虽然 canonical eval/QC 不走 LB，但建议仍然使用现有 bring-up 脚本：

- 它会顺手把 `8081..8088` 这 8 个 backend 全部带起来
- LB 只是附带产物，不影响后续 direct-backend QC

启动命令：

```bash
cd /workspace/verl

STATE_ROOT=/root/sandboxfusion-multi \
SANDBOX_VENV=/root/sandboxfusion-venv \
BASE_PORT=8081 \
BACKEND_COUNT=8 \
LB_BASE_PORT=8090 \
LB_COUNT=4 \
BACKENDS_PER_LB=2 \
bash "coding_model_project/phase_2_ GRPO/ops/setup_eval_sandbox_4x2.sh"
```

状态检查：

```bash
bash "coding_model_project/phase_2_ GRPO/ops/sandbox_backend_status.sh"
```

最小健康检查：

```bash
for p in 8081 8082 8083 8084 8085 8086 8087 8088; do
  curl -sf "http://127.0.0.1:${p}/v1/ping" && echo " backend ${p} ok"
done
```

如需做 HTTP probe：

```bash
python3 "coding_model_project/phase_2_ GRPO/ops/lb_validate_probe.py" \
  --endpoint "http://127.0.0.1:8081" \
  --requests 8 \
  --workers 4 \
  --require-all-success
```

---

## 7. 新机器第四阶段：先跑 QC metadata preflight

正式打 sandbox 之前，先确认输入文件本身没坏：

```bash
cd /workspace/verl

python3 "coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/qc_repair_conditioned_teacher_candidates.py" \
  --input_candidates "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_candidates_v2_full.jsonl" \
  --output_results "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_qc_v2_full.jsonl" \
  --summary_out "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_qc_v2_full.summary.json" \
  --sandbox_url "http://localhost:8081,http://localhost:8082,http://localhost:8083,http://localhost:8084,http://localhost:8085,http://localhost:8086,http://localhost:8087,http://localhost:8088" \
  --metadata_preflight_only
```

期望结果：

- `row_count = 429`
- `finish_reason_counts.stop = 429`
- `source_truncated_true_count = 0`

---

## 8. 新机器第五阶段：运行 primary QC

当前建议先用保守并发起跑：

- `workers = 8`
- `limiter_budget = 8`
- `run_timeout = 30`
- `memory_limit_mb = 1024`

命令：

```bash
cd /workspace/verl

python3 "coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/qc_repair_conditioned_teacher_candidates.py" \
  --input_candidates "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_candidates_v2_full.jsonl" \
  --output_results "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_qc_v2_full.jsonl" \
  --summary_out "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_qc_v2_full.summary.json" \
  --sandbox_url "http://localhost:8081,http://localhost:8082,http://localhost:8083,http://localhost:8084,http://localhost:8085,http://localhost:8086,http://localhost:8087,http://localhost:8088" \
  --workers 8 \
  --limiter_budget 8 \
  --run_timeout 30 \
  --memory_limit_mb 1024
```

跑完后检查 summary：

```bash
cat "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_qc_v2_full.summary.json"
```

---

## 9. 新机器第六阶段：稳定性复判

当前项目里，patched sandbox 已经明显比旧版稳定，但仍然不把单次 `accepted=true` 直接视为最终 keep。

因此 primary QC 后，建议按下面顺序继续：

### 9.1 抽出第一次 accepted 样本

```bash
python3 - <<'PY'
import json
from pathlib import Path

inp = Path("/workspace/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_qc_v2_full.jsonl")
out = Path("/workspace/verl/coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_qc_v2_full.accepted_only.jsonl")

kept = 0
with inp.open("r", encoding="utf-8") as fi, out.open("w", encoding="utf-8") as fo:
    for line in fi:
        row = json.loads(line)
        if row.get("accepted"):
            fo.write(json.dumps(row, ensure_ascii=False) + "\\n")
            kept += 1

print({"accepted_rows": kept, "output": str(out)})
PY
```

### 9.2 用同一个 QC 脚本再重判一次

```bash
python3 "coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/qc_repair_conditioned_teacher_candidates.py" \
  --input_candidates "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_qc_v2_full.accepted_only.jsonl" \
  --output_results "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_qc_v2_full.rejudge1.jsonl" \
  --summary_out "coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_qc_v2_full.rejudge1.summary.json" \
  --sandbox_url "http://localhost:8081,http://localhost:8082,http://localhost:8083,http://localhost:8084,http://localhost:8085,http://localhost:8086,http://localhost:8087,http://localhost:8088" \
  --workers 8 \
  --limiter_budget 8 \
  --run_timeout 30 \
  --memory_limit_mb 1024
```

### 9.3 最终 keep 规则

建议仍按旧计划执行：

- `2/2 accepted`：直接进入 keep set
- 如果第一次和第二次不一致：
  - 再跑第 3 次
  - 用 `2-of-3` 决定是否保留

这一步等新机器就绪后，再由我继续执行即可。

---

## 10. 当前最推荐的迁移操作

如果你是从本地把当前状态推到新机器，最简单稳妥的做法是：

1. 把本地整个 repo 根传到新机器：
   - 本地：`/Users/roger/Desktop/coding_RL_project/verl`
   - 远端：`/workspace/verl`
2. 如果你还想保留旧运行日志，再单独传：
   - `/workspace/eval_logs`
3. 新机器起来后，严格按本文档执行：
   - 环境安装
   - sandbox bring-up
   - QC preflight
   - primary QC
   - accepted rejudge

---

## 11. 这份文档对应的当前真实文件

新机器起好后，第一时间优先检查这些文件是否存在：

- `coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_generation_preflight_v2_full.summary.json`
- `coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_candidates_v2_full.jsonl`
- `coding_model_project/phase_2_ GRPO/teacher_generation_outputs/step900_candidate_v2_backend_rr/preflight/teacher_candidates_v2_full.summary.json`
- `coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/manifest.json`
- `coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/assemble_repair_conditioned_teacher_candidates.py`
- `coding_model_project/phase_2_ GRPO/sft_repair_data/scripts/qc_repair_conditioned_teacher_candidates.py`
- `coding_model_project/phase_2_ GRPO/ops/setup_eval_sandbox_4x2.sh`

如果这些都在，新机器这条线就已经具备“继续 teacher QC”的必要条件。
