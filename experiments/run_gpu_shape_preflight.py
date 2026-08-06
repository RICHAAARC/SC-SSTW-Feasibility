#!/usr/bin/env python3
"""Run the real-Wan, no-injection GPU shape/interface preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.gpu_shape_preflight import ShapePreflightError, run_shape_preflight, write_failure_package  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    try:
        metrics = run_shape_preflight(args.config, args.output_dir, args.expected_commit)
    except Exception as exc:
        write_failure_package(args.config, args.output_dir, args.expected_commit, str(exc))
        print(json.dumps({"gate": "GPU shape/interface preflight", "gate_pass": False, "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(metrics, sort_keys=True))
    return 0 if metrics["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
