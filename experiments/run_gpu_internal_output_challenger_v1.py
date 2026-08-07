#!/usr/bin/env python3
"""Run the formal real-Wan internal attention-output challenger v1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.gpu_internal_challenger import run_internal_challenger, write_internal_failure_package  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    try:
        metrics = run_internal_challenger(args.config, args.output_dir, args.expected_commit)
        print(__import__("json").dumps(metrics, sort_keys=True))
        return 0 if metrics["gate_pass"] else 2
    except Exception as exc:
        write_internal_failure_package(args.config, args.output_dir, args.expected_commit, f"{type(exc).__name__}: {exc}")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
