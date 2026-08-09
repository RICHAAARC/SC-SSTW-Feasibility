#!/usr/bin/env python3
"""Run the exact20 S1 real-DiT relation primitive diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sstw.s1_real_dit_relation_primitive import S1InstrumentationError, run_s1_once  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    argv = [str(Path(__file__).resolve()), "--output", str(args.output)]
    try:
        result = run_s1_once(repo_root=ROOT, output=args.output, argv=argv, cwd=Path.cwd())
    except (S1InstrumentationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INSTRUMENTATION_INSUFFICIENT", "reason": type(exc).__name__, "message": str(exc),
                          "formal_result": False, "stage_progression_allowed": False}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "S1_GO" else 3


if __name__ == "__main__":
    raise SystemExit(main())
