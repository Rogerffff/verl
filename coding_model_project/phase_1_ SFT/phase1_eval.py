#!/usr/bin/env python3
"""
Phase 1 SFT - Checkpoint Evaluation Script
============================================

Evaluate SFT checkpoints using phase0_eval.py in simple mode.

Workflow:
  1. Locate HF model in checkpoint directory
  2. Start vLLM server loading that model (or connect to existing one)
  3. Wait for health check (/v1/models returns 200)
  4. Call phase0_eval.py --mode simple with specified datasets
  5. Save all results to output directory for later comparison
  6. Shut down vLLM server

All checkpoints and evaluation results are preserved. After training,
compare results across steps and download the best checkpoint.

Usage:
    # Tier 1 evaluation (every checkpoint)
    python phase1_eval.py \
        --checkpoint_dir checkpoints/rlvr_coding_model/phase1_sft_qwen7b_coder/global_step_100 \
        --tier 1 \
        --sandbox_url http://localhost:8080

    # Full evaluation (final checkpoint)
    python phase1_eval.py \
        --checkpoint_dir checkpoints/rlvr_coding_model/phase1_sft_qwen7b_coder/global_step_378 \
        --tier 3 \
        --sandbox_url http://localhost:8080

    # Custom datasets
    python phase1_eval.py \
        --checkpoint_dir checkpoints/.../global_step_100 \
        --datasets codecontests_valid mbpp_reg \
        --sandbox_url http://localhost:8080

    # Connect to existing vLLM server (skip auto-start)
    python phase1_eval.py \
        --checkpoint_dir checkpoints/.../global_step_100 \
        --vllm_url http://localhost:8000 \
        --tier 1

Tiers:
    1: codecontests_valid (117) + mbpp_reg (200)           ~10 min
    2: + codecontests_valid_big (500)                       ~20 min
    3: + codecontests_test (165) + humaneval (164)          ~35 min
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Constants
# =============================================================================

TIER_DATASETS = {
    1: ["codecontests_valid", "mbpp_reg"],
    2: ["codecontests_valid", "mbpp_reg", "codecontests_valid_big"],
    3: ["codecontests_valid", "mbpp_reg", "codecontests_valid_big",
        "codecontests_test", "humaneval"],
}

VLLM_DEFAULT_PORT = 8000
VLLM_HEALTH_TIMEOUT = 300  # 5 minutes max to wait for vLLM startup
VLLM_HEALTH_INTERVAL = 5   # seconds between health checks


# =============================================================================
# vLLM Server Management
# =============================================================================

def find_hf_model(checkpoint_dir: str) -> str:
    """Locate the HuggingFace model directory within a checkpoint.

    Supports two patterns:
      - checkpoint_dir/huggingface/   (direct global_step_N directory)
      - checkpoint_dir itself         (if it already contains config.json)
    """
    ckpt_path = Path(checkpoint_dir)

    # Pattern 1: checkpoint_dir/huggingface/
    hf_dir = ckpt_path / "huggingface"
    if (hf_dir / "config.json").exists():
        return str(hf_dir)

    # Pattern 2: checkpoint_dir itself has config.json (already an HF model dir)
    if (ckpt_path / "config.json").exists():
        return str(ckpt_path)

    raise FileNotFoundError(
        f"Cannot find HF model in {checkpoint_dir}. "
        f"Expected config.json in {hf_dir} or {ckpt_path}"
    )


def start_vllm_server(
    model_path: str,
    port: int = VLLM_DEFAULT_PORT,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 6144,
) -> subprocess.Popen:
    """Start a vLLM OpenAI-compatible server as a subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ]

    print(f"Starting vLLM server...")
    print(f"  Model: {model_path}")
    print(f"  Port: {port}")
    print(f"  TP: {tensor_parallel_size}")
    print(f"  Command: {' '.join(cmd)}")

    # Inherit stdout/stderr to avoid deadlocks when pipe buffers are full.
    proc = subprocess.Popen(cmd)
    return proc


def wait_for_vllm(
    port: int = VLLM_DEFAULT_PORT,
    timeout: int = VLLM_HEALTH_TIMEOUT,
    interval: int = VLLM_HEALTH_INTERVAL,
) -> bool:
    """Wait for vLLM server to become ready via /v1/models health check."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/v1/models"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read())
                    models = [m.get("id", "") for m in data.get("data", [])]
                    print(f"  vLLM ready! Serving models: {models}")
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        except Exception as e:
            print(f"  Health check error: {e}")

        elapsed = int(time.time() - start_time)
        print(f"  Waiting for vLLM... ({elapsed}s / {timeout}s)")
        time.sleep(interval)

    return False


def stop_vllm_server(proc: subprocess.Popen):
    """Gracefully stop the vLLM server."""
    if proc is None or proc.poll() is not None:
        return

    print("Shutting down vLLM server...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
        print("  vLLM server stopped.")
    except subprocess.TimeoutExpired:
        print("  Force killing vLLM server...")
        proc.kill()
        proc.wait()


# =============================================================================
# Evaluation
# =============================================================================

def run_phase0_eval(
    vllm_url: str,
    model_path: str,
    datasets: List[str],
    output_dir: str,
    sandbox_url: str,
    manifest_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run phase0_eval.py in simple mode and return (metrics, summary)."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    eval_script = project_dir / "src" / "phase0_eval.py"

    if not eval_script.exists():
        raise FileNotFoundError(f"Eval script not found: {eval_script}")

    cmd = [
        sys.executable, str(eval_script),
        "--mode", "simple",
        "--vllm_url", vllm_url,
        "--model", model_path,
        "--datasets", *datasets,
        "--manifest_dir", manifest_dir,
        "--output_dir", output_dir,
        "--sandbox_url", sandbox_url,
        "--save_full_results",
    ]

    print(f"\nRunning evaluation...")
    print(f"  Datasets: {datasets}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(project_dir),
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"phase0_eval.py exited with code {result.returncode}")

    # Load metrics + summary from output.
    metrics_path = Path(output_dir) / "metrics.json"
    summary_path = Path(output_dir) / "summary.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    return metrics, summary


# =============================================================================
# Helpers
# =============================================================================

def extract_step_number(checkpoint_dir: str) -> int:
    """Extract step number from checkpoint directory name.

    Supports: global_step_100, global_step_378, etc.
    """
    ckpt_name = Path(checkpoint_dir).name
    if ckpt_name.startswith("global_step_"):
        try:
            return int(ckpt_name.split("global_step_")[1])
        except (ValueError, IndexError):
            pass

    # Fallback: try parent directory (in case we're pointing at huggingface/)
    parent_name = Path(checkpoint_dir).parent.name
    if parent_name.startswith("global_step_"):
        try:
            return int(parent_name.split("global_step_")[1])
        except (ValueError, IndexError):
            pass

    return -1


def build_dataset_scores(
    dataset: str,
    metrics: Dict[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Extract score fields used for history/best-checkpoint selection."""
    dataset_metrics = metrics.get(dataset, {}) if isinstance(metrics.get(dataset), dict) else {}
    dataset_summary = summary.get("datasets", {}).get(dataset, {})

    return {
        "accepted_at_1": dataset_metrics.get("accepted_at_1"),
        "pass_ratio_mean": dataset_metrics.get("pass_ratio_mean"),
        "exec_success_rate": dataset_summary.get("exec_success_rate"),
    }


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def update_best_checkpoint(output_base: Path, history_path: Path):
    """Update best_checkpoint.json by codecontests_valid exec_success_rate."""
    if not history_path.exists():
        return

    candidates = []
    with open(history_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            try:
                step = int(record.get("step"))
            except (TypeError, ValueError):
                continue

            cc_scores = record.get("scores", {}).get("codecontests_valid", {})
            exec_success = _to_float(cc_scores.get("exec_success_rate"))
            if exec_success is None:
                continue

            accepted = _to_float(cc_scores.get("accepted_at_1"))
            candidates.append(
                {
                    "step": step,
                    "checkpoint_dir": record.get("checkpoint_dir"),
                    "exec_success_rate": exec_success,
                    "accepted_at_1": accepted,
                }
            )

    if not candidates:
        return

    best = max(
        candidates,
        key=lambda x: (
            x["exec_success_rate"],
            x["accepted_at_1"] if x["accepted_at_1"] is not None else float("-inf"),
            x["step"],
        ),
    )

    best_ckpt_dir = Path(str(best["checkpoint_dir"]))
    best_model_path = best_ckpt_dir / "huggingface"
    if not best_model_path.exists():
        best_model_path = best_ckpt_dir

    all_checkpoints = {}
    for item in sorted(candidates, key=lambda x: x["step"]):
        all_checkpoints[str(item["step"])] = {
            "exec_success_rate": item["exec_success_rate"],
            "accepted_at_1": item["accepted_at_1"],
            "checkpoint_dir": item["checkpoint_dir"],
        }

    best_payload = {
        "best_step": best["step"],
        "best_exec_success_rate": best["exec_success_rate"],
        "best_accepted_at_1": best["accepted_at_1"],
        "model_path": str(best_model_path),
        "selection_metric": "summary.datasets.codecontests_valid.exec_success_rate",
        "all_checkpoints": all_checkpoints,
    }

    best_path = output_base / "best_checkpoint.json"
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(best_payload, f, indent=2)
    print(f"  Best checkpoint updated: {best_path} (step={best['step']})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 SFT - Checkpoint Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint directory (global_step_N/)")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                        help="Evaluation tier (1=quick, 2=medium, 3=full)")
    parser.add_argument("--datasets", nargs="+", type=str, default=None,
                        help="Override: specific datasets to evaluate (ignores --tier)")
    parser.add_argument("--output_base", type=str, default=None,
                        help="Base output directory (default: phase_1_ SFT/outputs/phase1)")
    parser.add_argument("--sandbox_url", type=str, default="http://localhost:8080",
                        help="SandboxFusion server URL")
    parser.add_argument("--manifest_dir", type=str, default=None,
                        help="Manifest directory (default: auto-detect)")

    # vLLM options
    parser.add_argument("--vllm_url", type=str, default=None,
                        help="Use existing vLLM server (skip starting one)")
    parser.add_argument("--vllm_port", type=int, default=VLLM_DEFAULT_PORT,
                        help="vLLM server port (when starting a new server)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--max_model_len", type=int, default=6144,
                        help="Max model length for vLLM (prompt + generation)")

    args = parser.parse_args()

    # --- Resolve paths ---
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Resolve manifest_dir
    manifest_dir = args.manifest_dir
    if manifest_dir is None:
        manifest_dir = str(project_dir / "data" / "manifests")
    if not Path(manifest_dir).exists():
        print(f"Error: Manifest directory not found: {manifest_dir}")
        sys.exit(1)

    # Resolve output directory
    step = extract_step_number(str(checkpoint_dir))
    output_base = args.output_base or str(script_dir / "outputs" / "phase1")
    output_base_path = Path(output_base)
    step_label = f"step_{step}" if step >= 0 else checkpoint_dir.name
    output_dir = str(output_base_path / step_label)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine datasets
    datasets = args.datasets or TIER_DATASETS[args.tier]

    # --- Find HF model ---
    try:
        hf_model_path = find_hf_model(str(checkpoint_dir))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("=" * 60)
    print("  Phase 1 SFT - Checkpoint Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  HF Model:   {hf_model_path}")
    print(f"  Step:        {step}")
    print(f"  Tier:        {args.tier}")
    print(f"  Datasets:    {datasets}")
    print(f"  Output:      {output_dir}")
    print(f"  Sandbox:     {args.sandbox_url}")
    print(f"  Manifest:    {manifest_dir}")

    # --- Start or connect vLLM ---
    vllm_proc = None
    if args.vllm_url:
        vllm_url = args.vllm_url
        print(f"\n  Using existing vLLM: {vllm_url}")
    else:
        vllm_url = f"http://localhost:{args.vllm_port}"
        vllm_proc = start_vllm_server(
            model_path=hf_model_path,
            port=args.vllm_port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

        if not wait_for_vllm(port=args.vllm_port):
            print("Error: vLLM server failed to start within timeout")
            stop_vllm_server(vllm_proc)
            sys.exit(1)

    # --- Run evaluation ---
    eval_error = None
    try:
        metrics, summary = run_phase0_eval(
            vllm_url=vllm_url,
            model_path=hf_model_path,
            datasets=datasets,
            output_dir=output_dir,
            sandbox_url=args.sandbox_url,
            manifest_dir=manifest_dir,
        )

        # --- Print summary ---
        print("\n" + "=" * 60)
        print("  Evaluation Summary")
        print("=" * 60)
        for ds in datasets:
            ds_metrics = metrics.get(ds, {})
            if not isinstance(ds_metrics, dict):
                ds_metrics = {}
            ds_summary = summary.get("datasets", {}).get(ds, {})
            accepted = ds_metrics.get("accepted_at_1", 0)
            pass_ratio = ds_metrics.get("pass_ratio_mean", 0)
            exec_success = _to_float(ds_summary.get("exec_success_rate"))
            syntax_err = ds_metrics.get("syntax_error_rate", 0)
            runtime_err = ds_metrics.get("runtime_error_rate", 0)
            timeout_rate = ds_metrics.get("timeout_rate", 0)
            print(f"\n  {ds}:")
            if exec_success is not None:
                print(f"    exec_success:     {exec_success:.2%}")
            if accepted is not None:
                print(f"    accepted@1:       {accepted:.2%}")
            if pass_ratio is not None:
                print(f"    pass_ratio_mean:  {pass_ratio:.4f}")
            if syntax_err:
                print(f"    syntax_error:     {syntax_err:.2%}")
            if runtime_err:
                print(f"    runtime_error:    {runtime_err:.2%}")
            if timeout_rate:
                print(f"    timeout:          {timeout_rate:.2%}")

        # --- Save eval config for traceability ---
        eval_info = {
            "checkpoint_dir": str(checkpoint_dir),
            "hf_model_path": hf_model_path,
            "step": step,
            "tier": args.tier,
            "datasets": datasets,
            "vllm_url": vllm_url,
            "sandbox_url": args.sandbox_url,
            "manifest_dir": manifest_dir,
        }
        with open(Path(output_dir) / "eval_info.json", 'w') as f:
            json.dump(eval_info, f, indent=2)

        # --- Append to eval history (one line per run) ---
        history_entry = {
            "step": step,
            "tier": args.tier,
            "checkpoint_dir": str(checkpoint_dir),
            "datasets": datasets,
            "scores": {ds: build_dataset_scores(ds, metrics, summary) for ds in datasets},
        }
        history_path = output_base_path / "eval_history.jsonl"
        with open(history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(history_entry) + "\n")
        print(f"\n  Results appended to: {history_path}")
        update_best_checkpoint(output_base_path, history_path)

    except Exception as e:
        eval_error = e
        print(f"\nError: Evaluation failed: {e}")

    finally:
        # --- Cleanup ---
        if vllm_proc:
            stop_vllm_server(vllm_proc)

    if eval_error is not None:
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
