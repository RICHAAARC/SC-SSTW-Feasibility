#!/usr/bin/env python3
"""Run the sole frozen Step-4 candidate-wise calibration CPU diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.rc0_candidate_self_calibration_fast_cpu import (  # noqa: E402
    SelfCalibrationDiagnosticError,
    run_step4_once,
)


def main() -> int:
    argv = [str(Path(__file__).resolve())]
    try:
        response = run_step4_once(repo_root=ROOT, argv=argv, cwd=Path.cwd())
    except (SelfCalibrationDiagnosticError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "DIAGNOSTIC_INSUFFICIENT", "diagnostic_class": "DIAGNOSTIC_ONLY",
            "question4": "INSUFFICIENT_TO_DECIDE", "route": "DIAGNOSTIC_INSUFFICIENT",
            "reason": type(exc).__name__, "message": str(exc),
            "formal_result": False, "stage_progression_allowed": False,
        }, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
