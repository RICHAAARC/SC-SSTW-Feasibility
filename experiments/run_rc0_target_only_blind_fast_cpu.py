#!/usr/bin/env python3
"""Run the sole frozen Phase-A target-only CPU diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.rc0_target_only_blind_fast_cpu import TargetOnlyDiagnosticError, run_phase_a_once  # noqa: E402


def main() -> int:
    argv = [str(Path(__file__).resolve())]
    try:
        response = run_phase_a_once(repo_root=ROOT, argv=argv, cwd=Path.cwd())
    except (TargetOnlyDiagnosticError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "DIAGNOSTIC_INSUFFICIENT", "diagnostic_class": "DIAGNOSTIC_ONLY",
            "phase_a": "INSUFFICIENT_TO_DECIDE", "route": "DIAGNOSTIC_INSUFFICIENT",
            "reason": type(exc).__name__, "message": str(exc),
            "formal_result": False, "stage_progression_allowed": False,
        }, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return int(response["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
