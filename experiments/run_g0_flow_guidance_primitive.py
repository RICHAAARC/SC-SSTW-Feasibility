#!/usr/bin/env python3
"""Run the single SSTW-v2 real-Wan Flow-guidance primitive diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sstw.flow_guidance_embedder import (  # noqa: E402
    G0InstrumentationError,
    g0_exit_code,
    run_g0_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run_g0_once(repo_root=ROOT, output=args.output)
    except (G0InstrumentationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        report = {
            "status": "INSTRUMENTATION_INSUFFICIENT",
            "diagnostic_class": "DIAGNOSTIC_ONLY",
            "reason": type(exc).__name__,
            "message": str(exc),
            "formal_result": False,
            "stage_progression_allowed": False,
        }
        print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 2
    report = {
        "status": result["status"],
        "diagnostic_class": "DIAGNOSTIC_ONLY",
        "output": str(args.output.resolve()),
        "transformer_calls": result["execution"]["transformer_calls"],
        "vae_decodes": result["execution"]["vae_decodes"],
        "formal_result": False,
        "stage_progression_allowed": False,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return g0_exit_code(result["status"])


if __name__ == "__main__":
    raise SystemExit(main())
