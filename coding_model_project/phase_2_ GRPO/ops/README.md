# Multi-Sandbox Ops Assets

These scripts stage the post-pilot Nginx LB + multi-sandbox reward probe flow without changing the RL reward/verifier contract.

## Order of Operations

1. Wait for the current readiness pilot to finish.
2. Stop the old `:8080` sandbox before bringing up staged backends:

```bash
STOP_LEGACY_8080=true BACKEND_COUNT_TARGET=4 \
  bash "coding_model_project/phase_2_ GRPO/ops/sandbox_backend_stop.sh"
```

3. Capture a clean-host baseline after the legacy sandbox is down:

```bash
bash "coding_model_project/phase_2_ GRPO/ops/capture_host_baseline.sh"
```

4. Start staged backends one at a time, with guardrails:

```bash
BACKEND_COUNT_TARGET=2 \
  bash "coding_model_project/phase_2_ GRPO/ops/sandbox_backend_start.sh"
```

5. Inspect backend status:

```bash
bash "coding_model_project/phase_2_ GRPO/ops/sandbox_backend_status.sh"
```

6. Render the Nginx LB config for the currently active backends:

```bash
NGINX_TEST=true \
  bash "coding_model_project/phase_2_ GRPO/ops/render_nginx_sandbox_lb.sh"
```

7. Apply the rendered Nginx config:

```bash
bash "coding_model_project/phase_2_ GRPO/ops/apply_nginx_sandbox_lb.sh"
```

8. Validate direct backends or the staged LB with the raw HTTP probe:

```bash
python3 "coding_model_project/phase_2_ GRPO/ops/lb_validate_probe.py" \
  --endpoint http://127.0.0.1:8081 \
  --requests 6 \
  --workers 2 \
  --require-all-success
```

9. Run the same-harness reward-only control or candidate sweep:

```bash
SANDBOX_URL=http://localhost:8090 LIMITER_BUDGET=12 \
  bash "coding_model_project/phase_2_ GRPO/ops/run_grpo_a1_reward_probe.sh"
```

10. Run the fast-val promotion gate:

```bash
SANDBOX_URL=http://localhost:8090 LIMITER_BUDGET=12 \
  bash "coding_model_project/phase_2_ GRPO/ops/run_grpo_a1_fastval_gate.sh"
```

## Guardrails Enforced by `sandbox_backend_start.sh`

- newest backend RSS after warmup must stay at or below `4 GiB`
- if the clean-host baseline had at least `128 GiB` `MemAvailable`, the staged host must also remain at or above `128 GiB`
- `MemAvailable` cannot drop by more than `16 GiB` versus the previous accepted stage
- `/tmp` free space must remain at or above `40 GiB`
- swap usage cannot increase above the clean-host baseline
- direct smoke must pass
- newest backend smoke `p95` cannot exceed `2x` the previous accepted stage

If fewer than `2` staged backends pass, the script exits non-zero and the host should stay on the single-backend path.
