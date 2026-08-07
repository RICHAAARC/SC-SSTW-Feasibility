#!/usr/bin/env python3
"""Run the formal real-Wan learned saved-MP4 observation GPU L1 Gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.gpu_learned_observation_l1 import finalize_l1_success_package, run_gpu_learned_observation_l1, write_l1_failure_package  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    metrics = None
    failure_reason = None
    with tempfile.TemporaryFile(mode="w+b") as stdout_capture, tempfile.TemporaryFile(mode="w+b") as stderr_capture:
        try:
            os.dup2(stdout_capture.fileno(), 1)
            os.dup2(stderr_capture.fileno(), 2)
            try:
                metrics = run_gpu_learned_observation_l1(args.config, args.output_dir, args.expected_commit)
                print(json.dumps(metrics, sort_keys=True), flush=True)
            except Exception as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"
                print(failure_reason, file=sys.stderr, flush=True)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
        stdout_capture.seek(0)
        stderr_capture.seek(0)
        stdout_text = stdout_capture.read().decode("utf-8", errors="replace")
        stderr_text = stderr_capture.read().decode("utf-8", errors="replace")
    if failure_reason is not None:
        write_l1_failure_package(args.config, args.output_dir, args.expected_commit, failure_reason, stdout_text=stdout_text, stderr_text=stderr_text)
        sys.stdout.write(stdout_text)
        sys.stderr.write(stderr_text)
        return 1
    assert metrics is not None
    finalize_l1_success_package(args.output_dir, stdout_text=stdout_text, stderr_text=stderr_text)
    sys.stdout.write(stdout_text)
    sys.stderr.write(stderr_text)
    return 0 if metrics["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
