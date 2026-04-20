# Step900 Teacher Prompt And Return Schema (v2 Backend RR)

## 0. Canonical Queue

当前真正应该发给 Claude Code 的，不是最早那份 remote `teacher_generation_requests`，
而是本地这份 **audited local materialization**：

- audited teacher requests:
  - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/teacher_generation_requests_step900_candidate_v2_backend_rr.jsonl`
- audited teacher request summary:
  - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/teacher_generation_requests_step900_candidate_v2_backend_rr.summary.json`
- enriched student references:
  - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/student_references_step900_candidate_v2_backend_rr_enriched.jsonl`
- shard manifest:
  - `/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_ GRPO/teacher_generation_shards/step900_candidate_v2_backend_rr/manifest.json`

这份 local audited queue 的来源是：

- upstream Stage B/C：
  - `patched sandbox + direct-backend client RR`
  - `8081..8088`
- remote canonical v2 student refs：
  - `step900_candidate_v2_backend_rr`
- local audit fix：
  - 追加 `source-integrity`
  - 追加 `SelectiveC provenance`
  - 明确区分：
    - `legacy_reviewed_c`
    - `heuristic_supplemental_c`

这里的含义是：

- `legacy_reviewed_c`
  - 指 `step600` 冷启动时，为了给空桶启动课程训练，
    从训练集按评测信号启发式搜出后再人工 review 通过的 bootstrap C
- `heuristic_supplemental_c`
  - 指 `step600 -> step900` 训练过程中真实进入 `C_hard_partial` 的题里，
    再按当前信号筛出的 retained C

所以这两个值首先是 **来源标签**，
不是简单的“reviewed 一定更高质量”。

当前真实规模：

- `request_count = 281`
- `Core = 133`
- `Expansion = 104`
- `SelectiveC = 44`
- `SelectiveC legacy_reviewed_c = 9`
- `SelectiveC heuristic_supplemental_c = 35`
- `teacher_best_of_n = 1` for `Core`
- `teacher_best_of_n = 2` for `Expansion / SelectiveC`
- total generation units:
  - `133 + 2 * (104 + 44) = 429`

当前 request / shard JSONL 已经显式携带这些审计字段：

- `source_truncated`
- `source_char_cap`
- `source_provenance`
- `review_status`
- `selection_mode`
- `c_trust_class`
- `c_source_tag`
- `c_decision_reason`
- `c_reviewer`

注意：

- 字段名 `c_trust_class` 沿用当前资产命名
- 但文档解释上应把它视为 `C provenance label`
- 不应把它当成质量等级

所以发给 Claude Code 时，推荐做法是：

- 你给 Claude 的输入可以是：
  - 整个 audited `teacher_generation_requests_*.jsonl`
  - 或者你手工切出来的一个 shard
- Claude 需要自己把每条 request 展开成 `attempt_index = 1..teacher_best_of_n`
- 每个 attempt 只能返回 **一个** completion

---

## 1. Teacher Task

Claude Code 在这一阶段只做一件事：

- 读取 repair-conditioned teacher requests
- 对每条 request 生成 one-turn repair teacher completion
- 返回规范 JSONL

Claude Code **不负责**：

- judge
- verifier
- extraction
- 后处理
- 去重
- QC 结论

---

## 2. Teacher Prompt

下面这段就是建议你直接发给 Claude Code 的主 prompt。

```text
You are generating repair-conditioned teacher completions for a Python competitive-programming SFT dataset.

I will give you a JSONL file. Each line is one teacher request. The key fields you should use are:

- request_id
- problem_id
- dataset
- source_split
- curriculum_bucket
- repair_stratum
- teacher_prompt_mode
- teacher_best_of_n
- prompt
- full_extracted_code
- error_type
- pass_ratio
- pass_ratio_all
- passed_tests
- total_tests
- repair_feedback

The request file may also carry audit-only provenance fields such as:

- source_truncated
- source_char_cap
- source_provenance
- review_status
- selection_mode
- c_trust_class
- c_source_tag
- c_decision_reason
- c_reviewer

You do not need to copy those provenance fields into your output. They are for audit only.

Your job:

1. Read each JSON object.
2. For each request, generate exactly `teacher_best_of_n` repaired outputs.
3. Each repaired output must be returned as one JSON object line.
4. Use:
   - `generation_unit_id = "{request_id}__attempt{attempt_index}"`
   - `attempt_index` starts from 1
   - `attempt_count = teacher_best_of_n`
5. Preserve input order. For a request with `teacher_best_of_n = 2`, emit attempt 1 first, then attempt 2.

The repair target is:

- original problem statement: `prompt`
- previous incorrect student code: `full_extracted_code`
- grounded verifier feedback: `repair_feedback`

The output style is fixed:

- `teacher_prompt_mode` is `short_diagnosis_code`
- so each completion must have exactly this visible format:

BUG_SUMMARY: one short sentence grounded in the feedback
FIX_PLAN: one short sentence describing the intended fix
<code>
...complete repaired Python program...
</code>

Hard requirements for `teacher_completion`:

- return a complete Python program
- read from stdin and write to stdout
- do not return a patch or diff
- do not include markdown fences like ``` outside the `<code>...</code>` block
- include exactly one final `<code>...</code>` block
- do not include extra sections beyond:
  - `BUG_SUMMARY: ...`
  - `FIX_PLAN: ...`
  - one `<code>...</code>` block

Hard requirements for the JSONL you return:

- output JSONL only
- one JSON object per generation attempt
- no prose before the first line
- no prose after the last line
- no markdown code fence wrapping the JSONL
- every input request must be fully covered
- do not skip rows silently
- if you truly cannot produce a valid completion for one attempt, still emit a row with:
  - empty `teacher_completion`
  - `generation_finish_reason = "error"`
  - a short explanation in `notes`

Return each JSON object with exactly these keys:

- generation_unit_id
- request_id
- attempt_index
- attempt_count
- teacher_prompt_mode
- teacher_model
- teacher_completion
- generation_finish_reason
- notes

Use:

- `teacher_prompt_mode = "short_diagnosis_code"`
- `teacher_model = "claude_code"`
- `generation_finish_reason = "stop"` for normal successful generations
- `notes = ""` unless there is something exceptional to record

Important:

- The JSONL is the final deliverable.
- Do not summarize your work.
- Do not explain anything outside the JSONL.
```

---

## 3. Required Raw Return Schema

Claude Code 返回的 raw JSONL 每行必须是一个对象，字段固定如下：

```json
{
  "generation_unit_id": "string, required, format: {request_id}__attempt{attempt_index}",
  "request_id": "string, required, must match input request_id",
  "attempt_index": "integer, required, 1-based",
  "attempt_count": "integer, required, equal to teacher_best_of_n for that request",
  "teacher_prompt_mode": "string, required, fixed to short_diagnosis_code",
  "teacher_model": "string, required, recommended value: claude_code",
  "teacher_completion": "string, required, raw teacher text in BUG_SUMMARY + FIX_PLAN + <code>...</code> format",
  "generation_finish_reason": "string, required, usually stop; use error only if generation truly failed",
  "notes": "string, required, default empty string"
}
```

补充约束：

- `generation_unit_id` 必须全局唯一
- `request_id + attempt_index` 必须唯一
- `teacher_completion` 必须保留完整原文，不要再做二次提取
- `generation_finish_reason` 不能缺失
- `notes` 不能省略，默认写空串

---

## 4. Example Return Rows

单 attempt request 的返回示例：

```json
{"generation_unit_id":"step900_rc_codeforces_123_a__attempt1","request_id":"step900_rc_codeforces_123_a","attempt_index":1,"attempt_count":1,"teacher_prompt_mode":"short_diagnosis_code","teacher_model":"claude_code","teacher_completion":"BUG_SUMMARY: The previous solution updates the answer with the wrong boundary case and misses one parity branch.\nFIX_PLAN: Recompute the branch conditions directly from the statement and keep the linear scan logic unchanged.\n<code>\nimport sys\n\ndef solve():\n    data = sys.stdin.read().strip().split()\n    if not data:\n        return\n    n = int(data[0])\n    arr = list(map(int, data[1:1+n]))\n    print(sum(arr))\n\nif __name__ == \"__main__\":\n    solve()\n</code>","generation_finish_reason":"stop","notes":""}
```

双 attempt request 的返回示例：

```json
{"generation_unit_id":"step900_rc_codeforces_456_b__attempt1","request_id":"step900_rc_codeforces_456_b","attempt_index":1,"attempt_count":2,"teacher_prompt_mode":"short_diagnosis_code","teacher_model":"claude_code","teacher_completion":"BUG_SUMMARY: The previous solution uses the right DP state but transitions with the wrong index offset.\nFIX_PLAN: Keep the same DP idea and correct the transition and initialization.\n<code>\n# repaired program here\n</code>","generation_finish_reason":"stop","notes":""}
{"generation_unit_id":"step900_rc_codeforces_456_b__attempt2","request_id":"step900_rc_codeforces_456_b","attempt_index":2,"attempt_count":2,"teacher_prompt_mode":"short_diagnosis_code","teacher_model":"claude_code","teacher_completion":"BUG_SUMMARY: The failure comes from an off-by-one transition that corrupts the DP table near the base case.\nFIX_PLAN: Rewrite the recurrence carefully and keep the rest of the solution structure minimal.\n<code>\n# repaired program here\n</code>","generation_finish_reason":"stop","notes":""}
```

失败占位示例：

```json
{"generation_unit_id":"step900_rc_codeforces_789_c__attempt1","request_id":"step900_rc_codeforces_789_c","attempt_index":1,"attempt_count":1,"teacher_prompt_mode":"short_diagnosis_code","teacher_model":"claude_code","teacher_completion":"","generation_finish_reason":"error","notes":"Could not produce a valid completion for this row."}
```

---

## 5. Downstream Normalized Candidate Schema

raw JSONL 回收后，建议组装成 normalized teacher candidates。下游 QC 最少需要这些字段：

```json
{
  "generation_unit_id": "string",
  "request_id": "string",
  "attempt_index": "integer",
  "attempt_count": "integer",
  "problem_id": "string",
  "dataset": "string",
  "source_split": "string",
  "curriculum_bucket": "string",
  "repair_stratum": "string",
  "teacher_prompt_mode": "string",
  "prompt_sha256": "string",
  "student_completion": "string",
  "full_extracted_code_student": "string",
  "source_truncated": "boolean",
  "source_char_cap": "integer|null",
  "source_provenance": "string",
  "error_type": "string",
  "pass_ratio": "number",
  "pass_ratio_all": "number",
  "passed_tests": "integer",
  "total_tests": "integer",
  "judge_time_s": "number",
  "review_status": "string",
  "selection_mode": "string|null",
  "c_trust_class": "string|null",
  "c_source_tag": "string|null",
  "c_decision_reason": "string|null",
  "c_reviewer": "string|null",
  "repair_feedback": "object",
  "teacher_model": "string",
  "teacher_completion_raw": "string",
  "generation_finish_reason": "string",
  "notes": "string"
}
```

注意：

- 当前这条 repair-conditioned 线的 normalized candidate，应该以：
  - `problem_id`
  - `curriculum_bucket`
  - `repair_stratum`
  为主键语义
- `SelectiveC` 在这一轮是混合 provenance：
  - `legacy_reviewed_c + heuristic_supplemental_c`
- `legacy_reviewed_c` 对应 bootstrap reviewed-C
- `heuristic_supplemental_c` 对应 current-training retained-C
- 下游 QC / keep accounting 应区分来源，但不应自动假设前者质量更高
- 不要硬套旧的：
  - `seed_problem_id`
  - `train_problem_id`
  - `bucket`
  那套 retrieval-SFT schema

---

## 6. Practical Send Pattern

实际发给 Claude Code 时，建议这样做：

1. 不要一次塞全部 `281 requests`
2. 每次发一个 shard
3. 每个 shard 控制在大约 `40 ~ 70 generation units`
4. 先跑：
   - `Core`
   - 再跑 `Expansion`
   - 最后跑 `SelectiveC`

如果你暂时还没物化 `generation_units` shard，也可以直接把 request shard 发给 Claude，
让它按照 `teacher_best_of_n` 自己展开 attempt rows。

---

## 7. Short Final Reminder

这轮 teacher generation 的最重要约束只有 4 条：

- 用当前 canonical v2 queue
- one attempt = one JSONL row
- 输出严格是 `short_diagnosis_code`
- Claude 返回 JSONL only，不要额外解释
