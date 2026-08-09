#!/usr/bin/env python3
"""Execute the frozen one-shot CPU pixel-chroma diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.rc0_pixel_chroma_fast_cpu import DiagnosticError, run_once  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    actual_argv = [str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]
    try:
        response = run_once(repo_root=ROOT, output=args.output, argv=actual_argv, cwd=Path.cwd())
    except (DiagnosticError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "DIAGNOSTIC_INSUFFICIENT", "diagnostic_class": "DIAGNOSTIC_ONLY",
            "reason": type(exc).__name__, "message": str(exc), "actual_package_path": str(args.output.resolve()),
            "formal_result": False, "stage_progression_allowed": False,
        }, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
