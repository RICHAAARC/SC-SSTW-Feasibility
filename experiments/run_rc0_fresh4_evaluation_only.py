#!/usr/bin/env python3
"""Run the sole zero-parameter read-only fresh4 evaluation closure."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.rc0_fresh4_evaluation_only import EvaluationOnlyError, run_evaluation_only_once  # noqa: E402


def main() -> int:
    argv = [str(Path(__file__).resolve())]
    try:
        response = run_evaluation_only_once(repo_root=ROOT, argv=argv, cwd=Path.cwd())
    except (EvaluationOnlyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INSUFFICIENT", "diagnostic_class": "DIAGNOSTIC_ONLY",
                          "phase_c": "INSUFFICIENT_TO_DECIDE", "route": "INSUFFICIENT",
                          "reason": type(exc).__name__, "message": str(exc),
                          "formal_result": False, "stage_progression_allowed": False},
                         sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return 0 if response["phase_c"] == "FEASIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
