#!/usr/bin/env python3
"""Generate the frozen fresh exact-four carrier-free base videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.rc0_fresh_exact4_base_gpu import FreshBaseDiagnosticError, run_exact4_base_once  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    argv = [str(Path(__file__).resolve()), "--output", str(args.output)]
    try:
        response = run_exact4_base_once(repo_root=ROOT, output=args.output, argv=argv, cwd=Path.cwd())
    except (FreshBaseDiagnosticError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "DIAGNOSTIC_INSUFFICIENT", "diagnostic_class": "DIAGNOSTIC_ONLY",
                          "reason": type(exc).__name__, "message": str(exc), "formal_result": False,
                          "stage_progression_allowed": False}, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
