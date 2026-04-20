#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_LINE_RE = re.compile(r"\bstep:(\d+)\s+-\s+actor/entropy:")
PROGRESS_RE = re.compile(r"Training Progress:\s+\d+%\|.*?\|\s+(\d+)/(\d+)")

FLOAT_METRICS = (
    "timing_s/reward",
    "timing_s/update_actor",
    "timing_s/step",
    "verifier/pass_ratio_all_mean",
    "verifier/accepted_rate",
    "verifier/reward_raw_valid_rate",
    "verifier/invalid_for_rl_rate",
    "verifier/truncated_by_max_tokens_rate",
    "verifier/judge_time_s_mean",
    "verifier/judge_time_s_p95",
    "verifier/timeout_rate",
    "response_length/clip_ratio",
)

HARD_ERROR_PATTERNS = (
    "No space left on device",
    "PytorchStreamWriter failed writing file",
    "unexpected pos",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "Traceback (most recent call last):",
    "RayTaskError",
)

SOFT_ERROR_PATTERNS = (
    "SIGABRT",
    "Fatal Python error",
    "WorkerCrashedError",
    "The actor died because",
    "A worker died or was killed",
)


def run_cmd(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def disk_free_gb(path: str = "/") -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


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


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def read_tail_text(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        data = handle.read()
    return data.decode("utf-8", errors="ignore")


def extract_latest_metrics(log_text: str) -> dict[str, Any]:
    latest_step = None
    latest_line = None
    for raw_line in log_text.splitlines():
        line = strip_ansi(raw_line)
        match = STEP_LINE_RE.search(line)
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


def extract_latest_progress_step(log_text: str) -> int | None:
    latest_step = None
    for raw_line in log_text.splitlines():
        line = strip_ansi(raw_line)
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
        return [strip_ansi(line) for line in data.splitlines()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_name(row: dict[str, Any]) -> str:
    gts = row.get("gts")
    if isinstance(gts, dict):
        dataset = gts.get("dataset")
        if dataset:
            return str(dataset)
    return str(row.get("data_source", "<missing>"))


def numeric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            vals.append(float(value))
        elif isinstance(value, (int, float)) and not math.isnan(value):
            vals.append(float(value))
    return sum(vals) / len(vals) if vals else None


def summarize_validation_file(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(dataset_name(row), []).append(row)

    datasets: dict[str, Any] = {}
    for name, ds_rows in sorted(by_dataset.items()):
        error_counts: dict[str, int] = {}
        for row in ds_rows:
            error = row.get("error_type", "") or "ok"
            error_counts[error] = error_counts.get(error, 0) + 1
        datasets[name] = {
            "count": len(ds_rows),
            "accepted_mean": numeric_mean(ds_rows, "accepted"),
            "pass_ratio_all_mean": numeric_mean(ds_rows, "pass_ratio_all"),
            "reward_mean": numeric_mean(ds_rows, "reward"),
            "reward_raw_mean": numeric_mean(ds_rows, "reward_raw"),
            "invalid_for_rl_mean": numeric_mean(ds_rows, "invalid_for_rl"),
            "truncated_mean": numeric_mean(ds_rows, "truncated_by_max_tokens"),
            "judge_time_s_mean": numeric_mean(ds_rows, "judge_time_s"),
            "error_type_counts": dict(sorted(error_counts.items())),
        }
    return {"path": str(path), "datasets": datasets}


@dataclass
class MonitorState:
    experiment_name: str
    target_step: int
    latest_step: int | None = None
    latest_progress_step: int | None = None
    latest_metrics_step: int | None = None
    latest_metrics: dict[str, Any] | None = None
    latest_validation_step: int | None = None
    latest_validation_summary: dict[str, Any] | None = None
    disk_free_gb: float = 0.0
    gpu_snapshot: list[dict[str, Any]] | None = None
    ps_lines: list[str] | None = None
    watcher_tail: list[str] | None = None
    warnings: list[str] | None = None
    soft_error_lines: list[str] | None = None
    hard_error_lines: list[str] | None = None
    success: bool = False
    anomaly: str | None = None
    anomaly_details: list[str] | None = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--disk-path", default="/workspace")
    parser.add_argument("--disk-min-free-gb", type=float, default=80.0)
    parser.add_argument("--stall-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--tail-bytes", type=int, default=400000)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--status-jsonl")
    parser.add_argument("--validation-summary-json")
    parser.add_argument("--log-path", action="append", default=[])
    parser.add_argument("--watch-log")
    parser.add_argument("--validation-dir")
    parser.add_argument("--timeout-rate-warn", type=float, default=0.12)
    parser.add_argument("--invalid-rate-warn", type=float, default=0.05)
    parser.add_argument("--truncated-rate-warn", type=float, default=0.05)
    parser.add_argument("--reward-valid-rate-min", type=float, default=0.98)
    parser.add_argument("--response-clip-warn", type=float, default=0.05)
    args = parser.parse_args()

    ckpt_root = Path(args.ckpt_root)
    log_paths = [Path(path) for path in args.log_path]
    watch_log = Path(args.watch_log) if args.watch_log else None
    validation_dir = Path(args.validation_dir) if args.validation_dir else None
    status_json = Path(args.status_json)
    summary_json = Path(args.summary_json)
    status_jsonl = Path(args.status_jsonl) if args.status_jsonl else None
    validation_summary_json = Path(args.validation_summary_json) if args.validation_summary_json else None

    log_readers = {str(path): IncrementalReader(path) for path in log_paths}
    watch_reader = IncrementalReader(watch_log) if watch_log else None
    validation_cache: dict[str, dict[str, Any]] = {}
    validation_index: dict[int, dict[str, Any]] = {}
    last_step_change_time = time.time()
    previous_step: int | None = None
    soft_error_lines: list[str] = []
    hard_error_lines: list[str] = []

    while True:
        combined_text = "\n".join(read_tail_text(path, args.tail_bytes) for path in log_paths if path.exists())
        latest_metrics = extract_latest_metrics(combined_text)
        latest_metrics_step = latest_metrics.get("step")
        latest_progress_step = extract_latest_progress_step(combined_text)
        latest_step_candidates = [step for step in (latest_metrics_step, latest_progress_step) if step is not None]
        latest_step = max(latest_step_candidates) if latest_step_candidates else None

        if latest_step is not None and latest_step != previous_step:
            previous_step = latest_step
            last_step_change_time = time.time()

        for path, reader in log_readers.items():
            for line in reader.read_new_lines():
                if any(pattern in line for pattern in HARD_ERROR_PATTERNS):
                    hard_error_lines.append(f"{path}: {line}")
                elif any(pattern in line for pattern in SOFT_ERROR_PATTERNS):
                    soft_error_lines.append(f"{path}: {line}")
        soft_error_lines = soft_error_lines[-20:]
        hard_error_lines = hard_error_lines[-20:]

        watcher_tail: list[str] = []
        if watch_log is not None and watch_log.exists():
            watch_text = read_tail_text(watch_log, 10000)
            watcher_tail = [line for line in watch_text.splitlines() if line.strip()][-10:]
            if watch_reader is not None:
                for line in watch_reader.read_new_lines():
                    if any(pattern in line for pattern in HARD_ERROR_PATTERNS):
                        hard_error_lines.append(f"{watch_log}: {line}")
                    elif "Traceback" in line:
                        hard_error_lines.append(f"{watch_log}: {line}")

        if validation_dir is not None and validation_dir.exists():
            for path in sorted(validation_dir.glob("*.jsonl")):
                stat = path.stat()
                cache_key = f"{stat.st_mtime_ns}:{stat.st_size}"
                if validation_cache.get(path.name, {}).get("cache_key") == cache_key:
                    continue
                summary = summarize_validation_file(path)
                validation_cache[path.name] = {"cache_key": cache_key, "summary": summary}
                try:
                    step = int(path.stem)
                except ValueError:
                    continue
                validation_index[step] = summary

        latest_validation_step = max(validation_index) if validation_index else None
        latest_validation_summary = validation_index.get(latest_validation_step) if latest_validation_step else None

        if validation_summary_json is not None:
            serializable = {str(step): summary for step, summary in sorted(validation_index.items())}
            validation_summary_json.write_text(json.dumps(serializable, ensure_ascii=False, indent=2))

        warnings: list[str] = []
        if latest_metrics:
            reward_valid = latest_metrics.get("verifier/reward_raw_valid_rate")
            invalid_rate = latest_metrics.get("verifier/invalid_for_rl_rate")
            truncated_rate = latest_metrics.get("verifier/truncated_by_max_tokens_rate")
            timeout_rate = latest_metrics.get("verifier/timeout_rate")
            response_clip = latest_metrics.get("response_length/clip_ratio")
            if reward_valid is not None and reward_valid < args.reward_valid_rate_min:
                warnings.append(
                    f"reward_raw_valid_rate={reward_valid:.4f} below {args.reward_valid_rate_min:.4f} at step {latest_step}"
                )
            if invalid_rate is not None and invalid_rate > args.invalid_rate_warn:
                warnings.append(
                    f"invalid_for_rl_rate={invalid_rate:.4f} above {args.invalid_rate_warn:.4f} at step {latest_step}"
                )
            if truncated_rate is not None and truncated_rate > args.truncated_rate_warn:
                warnings.append(
                    f"truncated_by_max_tokens_rate={truncated_rate:.4f} above {args.truncated_rate_warn:.4f} at step {latest_step}"
                )
            if timeout_rate is not None and timeout_rate > args.timeout_rate_warn:
                warnings.append(f"timeout_rate={timeout_rate:.4f} above {args.timeout_rate_warn:.4f} at step {latest_step}")
            if response_clip is not None and response_clip > args.response_clip_warn:
                warnings.append(
                    f"response_length/clip_ratio={response_clip:.4f} above {args.response_clip_warn:.4f} at step {latest_step}"
                )

        current_disk_free = disk_free_gb(args.disk_path)
        ps_lines = get_ps_lines(args.experiment_name)
        gpu_snapshot = get_gpu_snapshot()

        anomaly: str | None = None
        anomaly_details: list[str] = []

        if current_disk_free < args.disk_min_free_gb:
            anomaly = "disk_free_below_threshold"
            anomaly_details.append(
                f"Disk free dropped to {current_disk_free:.2f} GiB, below threshold {args.disk_min_free_gb:.2f} GiB."
            )

        if hard_error_lines:
            newest = hard_error_lines[-5:]
            if any("No space left on device" in line or "OutOfMemory" in line or "CUDA out of memory" in line for line in newest):
                anomaly = "hard_error_detected"
                anomaly_details.extend(newest)

        target_complete = checkpoint_save_complete(ckpt_root, args.target_step)
        if not target_complete and not ps_lines:
            anomaly = "training_process_missing_before_target_step"
            anomaly_details.append("No process with the experiment name is still running.")

        if not target_complete and time.time() - last_step_change_time > args.stall_seconds:
            anomaly = "step_progress_stalled"
            anomaly_details.append(
                f"No new logged step for {(time.time() - last_step_change_time):.0f}s before target step completion."
            )

        state = MonitorState(
            experiment_name=args.experiment_name,
            target_step=args.target_step,
            latest_step=latest_step,
            latest_progress_step=latest_progress_step,
            latest_metrics_step=latest_metrics_step,
            latest_metrics=latest_metrics or None,
            latest_validation_step=latest_validation_step,
            latest_validation_summary=latest_validation_summary,
            disk_free_gb=current_disk_free,
            gpu_snapshot=gpu_snapshot,
            ps_lines=ps_lines,
            watcher_tail=watcher_tail or None,
            warnings=warnings or None,
            soft_error_lines=soft_error_lines or None,
            hard_error_lines=hard_error_lines or None,
            success=False,
            anomaly=anomaly,
            anomaly_details=anomaly_details or None,
        )

        status_json.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2))
        if status_jsonl is not None:
            with status_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"ts": time.time(), **asdict(state)}, ensure_ascii=False) + "\n")

        if target_complete:
            state.success = True
            status_json.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2))
            summary_json.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2))
            print(json.dumps(asdict(state), ensure_ascii=False, indent=2))
            return 0

        if anomaly is not None:
            summary_json.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2))
            print(json.dumps(asdict(state), ensure_ascii=False, indent=2))
            return 2

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
