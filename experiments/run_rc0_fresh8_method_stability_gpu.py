#!/usr/bin/env python3
"""Generate the frozen eight additional carrier-free fresh12 bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.rc0_fresh8_method_stability_gpu import (  # noqa: E402
    Fresh8DiagnosticError,
    run_fresh8_base_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    argv = [str(Path(__file__).resolve()), "--output", str(args.output)]
    try:
        response = run_fresh8_base_once(repo_root=ROOT, output=args.output, argv=argv, cwd=Path.cwd())
    except (Fresh8DiagnosticError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "DIAGNOSTIC_INSUFFICIENT", "diagnostic_class": "DIAGNOSTIC_ONLY",
                          "reason": type(exc).__name__, "message": str(exc), "formal_result": False,
                          "stage_progression_allowed": False}, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
