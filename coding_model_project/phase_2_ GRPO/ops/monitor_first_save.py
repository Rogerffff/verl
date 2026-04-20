#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"step:(\d+)")
PROGRESS_RE = re.compile(r"Training Progress:\s+\d+%\|.*?\|\s+(\d+)/(\d+)")
FLOAT_METRICS = (
    "timing_s/reward",
    "timing_s/update_actor",
    "timing_s/step",
    "verifier/judge_time_s_p95",
    "verifier/timeout_rate",
    "verifier/reward_raw_valid_rate",
    "verifier/invalid_for_rl_rate",
    "verifier/truncated_by_max_tokens_rate",
)
TRAINING_ERROR_PATTERNS = (
    "PytorchStreamWriter failed writing file",
    "No space left on device",
    "unexpected pos",
    "RayTaskError",
    "Unhandled error",
    "torch.OutOfMemoryError",
    "CUDA out of memory",
    "RuntimeError:",
    "Traceback (most recent call last):",
)
BACKEND_ERROR_PATTERNS = (
    "SandboxError",
    "500 Internal Server Error",
)
BACKEND_BENIGN_PATTERNS = (
    "Failed to write to stdin:",
    "handler is closed",
    "Broken pipe",
)
RAY_LOG_PATTERNS = (
    "torch.OutOfMemoryError",
    "CUDA out of memory",
    "Traceback (most recent call last):",
    "ray.exceptions.RayActorError",
    "WorkerCrashedError",
    "The actor died because",
    "A worker died or was killed",
    "No space left on device",
    "PytorchStreamWriter failed writing file",
    "unexpected pos",
)


def run_cmd(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def get_ps_lines(experiment_name: str) -> list[str]:
    output = run_cmd(
        [
            "bash",
            "-lc",
            f"ps -eo pid,etimes,pcpu,pmem,cmd | grep -F {json.dumps(experiment_name)} | grep -v grep || true",
        ]
    )
    return [line for line in output.splitlines() if line.strip()]


def get_gpu_snapshot() -> list[dict[str, Any]]:
    try:
        output = run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "utilization_gpu": float(parts[1]),
                "memory_used_mb": float(parts[2]),
                "memory_total_mb": float(parts[3]),
            }
        )
    return rows


def parse_latest_step_metrics(log_text: str) -> dict[str, Any]:
    latest_step = None
    latest_line = None
    for line in log_text.splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        if latest_step is None or step >= latest_step:
            latest_step = step
            latest_line = line

    if latest_step is None or latest_line is None:
        return {}

    payload: dict[str, Any] = {"step": latest_step, "raw_line": latest_line}
    for metric in FLOAT_METRICS:
        match = re.search(re.escape(metric) + r":([^\s]+)", latest_line)
        payload[metric] = float(match.group(1)) if match else None
    return payload


def parse_latest_progress_step(log_text: str) -> int | None:
    latest_step = None
    for line in log_text.splitlines():
        match = PROGRESS_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        if latest_step is None or step >= latest_step:
            latest_step = step
    return latest_step


def tracker_step(tracker_path: Path) -> int | None:
    if not tracker_path.exists():
        return None
    try:
        return int(tracker_path.read_text().strip())
    except Exception:
        return None


def checkpoint_save_complete(ckpt_root: Path, step: int) -> bool:
    tracker = tracker_step(ckpt_root / "latest_checkpointed_iteration.txt")
    step_dir = ckpt_root / f"global_step_{step}"
    actor_dir = step_dir / "actor"
    return (
        tracker is not None
        and tracker >= step
        and step_dir.exists()
        and (step_dir / "data.pt").exists()
        and actor_dir.exists()
        and any(actor_dir.glob("model_world_size_*_rank_0.pt"))
        and any(actor_dir.glob("optim_world_size_*_rank_0.pt"))
    )


def disk_free_gb() -> float:
    usage = shutil.disk_usage("/")
    return usage.free / (1024**3)


class IncrementalReader:
    def __init__(self, path: Path):
        self.path = path
        self.position = path.stat().st_size if path.exists() else 0

    def read_new_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.position:
            self.position = 0
        with self.path.open("r", errors="ignore") as handle:
            handle.seek(self.position)
            data = handle.read()
            self.position = handle.tell()
        return data.splitlines()


def interesting_ray_log_files(ray_log_dir: Path) -> list[Path]:
    if not ray_log_dir.exists():
        return []
    paths: list[Path] = []
    for pattern in ("worker-*.err", "worker-*.out", "raylet.err", "raylet.out"):
        paths.extend(sorted(ray_log_dir.glob(pattern)))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


@dataclass
class MonitorState:
    experiment_name: str
    first_save_step: int
    latest_step: int | None = None
    latest_progress_step: int | None = None
    latest_metrics_step: int | None = None
    latest_metrics: dict[str, Any] | None = None
    lb_non_200_count: int = 0
    lb_slow_ge_25s: int = 0
    lb_upstream_counts: dict[str, int] | None = None
    backend_error_hits: dict[str, int] | None = None
    backend_benign_hits: dict[str, int] | None = None
    backend_kill_hits: dict[str, int] | None = None
    disk_free_gb: float = 0.0
    gpu_snapshot: list[dict[str, Any]] | None = None
    ps_lines: list[str] | None = None
    success: bool = False
    anomaly: str | None = None
    anomaly_details: list[str] | None = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--first-save-step", type=int, default=20)
    parser.add_argument("--nginx-log", required=True)
    parser.add_argument("--backend-log", action="append", default=[])
    parser.add_argument("--ray-log-dir")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--disk-min-free-gb", type=float, default=40.0)
    parser.add_argument("--timeout-rate-max", type=float, default=0.25)
    parser.add_argument("--invalid-rate-max", type=float, default=0.05)
    parser.add_argument("--truncated-rate-max", type=float, default=0.05)
    parser.add_argument("--reward-valid-rate-min", type=float, default=0.95)
    parser.add_argument("--stall-seconds", type=float, default=1800.0)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--status-jsonl")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    ckpt_root = Path(args.ckpt_root)
    nginx_log = Path(args.nginx_log)
    backend_logs = [Path(path) for path in args.backend_log]
    ray_log_dir = Path(args.ray_log_dir) if args.ray_log_dir else None
    status_json = Path(args.status_json)
    summary_json = Path(args.summary_json)
    status_jsonl = Path(args.status_jsonl) if args.status_jsonl else None

    log_reader = IncrementalReader(log_path)
    nginx_reader = IncrementalReader(nginx_log)
    backend_readers = {path.name: IncrementalReader(path) for path in backend_logs}
    ray_log_readers: dict[str, IncrementalReader] = {}

    lb_upstream_counts: dict[str, int] = {}
    lb_non_200_count = 0
    lb_slow_ge_25s = 0
    backend_error_hits = {path.name: 0 for path in backend_logs}
    backend_benign_hits = {path.name: 0 for path in backend_logs}
    backend_kill_hits = {path.name: 0 for path in backend_logs}
    last_step_change_time = time.time()
    previous_step: int | None = None

    while True:
        log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
        latest_metrics = parse_latest_step_metrics(log_text)
        latest_metrics_step = latest_metrics.get("step")
        latest_progress_step = parse_latest_progress_step(log_text)
        latest_step_candidates = [step for step in (latest_metrics_step, latest_progress_step) if step is not None]
        latest_step = max(latest_step_candidates) if latest_step_candidates else None
        if latest_step != previous_step and latest_step is not None:
            previous_step = latest_step
            last_step_change_time = time.time()

        new_log_lines = log_reader.read_new_lines()
        training_error_lines = [
            line for line in new_log_lines if any(pattern in line for pattern in TRAINING_ERROR_PATTERNS)
        ]

        for line in nginx_reader.read_new_lines():
            status_match = re.search(r"status=(\d+)", line)
            upstream_match = re.search(r"upstream_addr=([^ ]+)", line)
            request_time_match = re.search(r"request_time=([0-9.]+)", line)
            if status_match and status_match.group(1) != "200":
                lb_non_200_count += 1
            if upstream_match:
                upstream = upstream_match.group(1)
                lb_upstream_counts[upstream] = lb_upstream_counts.get(upstream, 0) + 1
            if request_time_match and float(request_time_match.group(1)) >= 25.0:
                lb_slow_ge_25s += 1

        backend_bad_lines: list[str] = []
        for name, reader in backend_readers.items():
            new_lines = reader.read_new_lines()
            for line in new_lines:
                if any(pattern in line for pattern in BACKEND_BENIGN_PATTERNS):
                    backend_benign_hits[name] += 1
                if any(pattern in line for pattern in BACKEND_ERROR_PATTERNS):
                    backend_error_hits[name] += 1
                    backend_bad_lines.append(f"{name}: {line}")
                if "process killed:" in line:
                    backend_kill_hits[name] += 1

        ray_bad_lines: list[str] = []
        if ray_log_dir is not None:
            for path in interesting_ray_log_files(ray_log_dir):
                reader = ray_log_readers.get(path.name)
                if reader is None:
                    reader = IncrementalReader(path)
                    ray_log_readers[path.name] = reader
                for line in reader.read_new_lines():
                    if any(pattern in line for pattern in RAY_LOG_PATTERNS):
                        ray_bad_lines.append(f"{path.name}: {line}")

        current_disk_free = disk_free_gb()
        ps_lines = get_ps_lines(args.experiment_name)
        gpu_snapshot = get_gpu_snapshot()

        anomaly: str | None = None
        anomaly_details: list[str] = []

        if not ps_lines and not checkpoint_save_complete(ckpt_root, args.first_save_step):
            anomaly = "training_process_missing_before_first_save"
            anomaly_details.append("No process with the experiment name is still running.")

        if training_error_lines:
            anomaly = "training_log_error_detected"
            anomaly_details.extend(training_error_lines[-10:])

        if lb_non_200_count > 0:
            anomaly = "lb_non_200_detected"
            anomaly_details.append(f"LB observed {lb_non_200_count} non-200 responses since monitor start.")

        if backend_bad_lines:
            anomaly = "backend_error_detected"
            anomaly_details.extend(backend_bad_lines[-10:])

        if ray_bad_lines:
            anomaly = "ray_log_error_detected"
            anomaly_details.extend(ray_bad_lines[-10:])

        if current_disk_free < args.disk_min_free_gb:
            anomaly = "disk_free_below_threshold"
            anomaly_details.append(
                f"Disk free dropped to {current_disk_free:.2f} GiB, below threshold {args.disk_min_free_gb:.2f} GiB."
            )

        if latest_metrics:
            reward_valid = latest_metrics.get("verifier/reward_raw_valid_rate")
            invalid_rate = latest_metrics.get("verifier/invalid_for_rl_rate")
            truncated_rate = latest_metrics.get("verifier/truncated_by_max_tokens_rate")
            timeout_rate = latest_metrics.get("verifier/timeout_rate")
            if reward_valid is not None and reward_valid < args.reward_valid_rate_min:
                anomaly = "reward_valid_rate_too_low"
                anomaly_details.append(
                    f"reward_raw_valid_rate={reward_valid:.4f} < {args.reward_valid_rate_min:.4f} at step {latest_step}"
                )
            if invalid_rate is not None and invalid_rate > args.invalid_rate_max:
                anomaly = "invalid_for_rl_rate_too_high"
                anomaly_details.append(
                    f"invalid_for_rl_rate={invalid_rate:.4f} > {args.invalid_rate_max:.4f} at step {latest_step}"
                )
            if truncated_rate is not None and truncated_rate > args.truncated_rate_max:
                anomaly = "truncated_rate_too_high"
                anomaly_details.append(
                    f"truncated_by_max_tokens_rate={truncated_rate:.4f} > {args.truncated_rate_max:.4f} at step {latest_step}"
                )
            if timeout_rate is not None and timeout_rate > args.timeout_rate_max:
                anomaly = "timeout_rate_too_high"
                anomaly_details.append(
                    f"timeout_rate={timeout_rate:.4f} > {args.timeout_rate_max:.4f} at step {latest_step}"
                )

        if time.time() - last_step_change_time > args.stall_seconds and not checkpoint_save_complete(
            ckpt_root, args.first_save_step
        ):
            anomaly = "step_progress_stalled"
            anomaly_details.append(
                f"No new logged step for {(time.time() - last_step_change_time):.0f}s before first save completion."
            )

        state = MonitorState(
            experiment_name=args.experiment_name,
            first_save_step=args.first_save_step,
            latest_step=latest_step,
            latest_progress_step=latest_progress_step,
            latest_metrics_step=latest_metrics_step,
            latest_metrics=latest_metrics or None,
            lb_non_200_count=lb_non_200_count,
            lb_slow_ge_25s=lb_slow_ge_25s,
            lb_upstream_counts=dict(sorted(lb_upstream_counts.items())),
            backend_error_hits=dict(backend_error_hits),
            backend_benign_hits=dict(backend_benign_hits),
            backend_kill_hits=dict(backend_kill_hits),
            disk_free_gb=current_disk_free,
            gpu_snapshot=gpu_snapshot,
            ps_lines=ps_lines,
            success=False,
            anomaly=anomaly,
            anomaly_details=anomaly_details or None,
        )

        status_json.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2))
        if status_jsonl is not None:
            with status_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"ts": time.time(), **asdict(state)}, ensure_ascii=False) + "\n")

        if anomaly is not None:
            summary_json.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2))
            print(json.dumps(asdict(state), ensure_ascii=False, indent=2))
            return 2

        if checkpoint_save_complete(ckpt_root, args.first_save_step):
            state.success = True
            status_json.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2))
            summary_json.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2))
            print(json.dumps(asdict(state), ensure_ascii=False, indent=2))
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
