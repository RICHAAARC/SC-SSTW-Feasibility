#!/usr/bin/env python3
"""Read-only evaluation closure for the one frozen pixel-chroma run1."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.rc0_pixel_chroma_fast_cpu import (  # noqa: E402
    DiagnosticError,
    evaluate_existing_run_once,
)


def main() -> int:
    argv = [str(Path(__file__).resolve())]
    try:
        response = evaluate_existing_run_once(repo_root=ROOT, argv=argv, cwd=Path.cwd())
    except (DiagnosticError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "DIAGNOSTIC_INSUFFICIENT", "diagnostic_class": "DIAGNOSTIC_ONLY",
            "reason": type(exc).__name__, "message": str(exc),
            "formal_result": False, "stage_progression_allowed": False,
        }, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return 0 if response["status"] == "CPU_DIAGNOSTIC_RESULT_READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
