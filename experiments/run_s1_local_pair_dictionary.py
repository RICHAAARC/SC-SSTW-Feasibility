#!/usr/bin/env python3
"""Run the exact10 S1 local single-pair dictionary screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sstw.s1_local_pair_dictionary import (  # noqa: E402
    S1InstrumentationError,
    dictionary_exit_code,
    run_dictionary_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    argv = [str(Path(__file__).resolve()), "--output", str(args.output)]
    try:
        result = run_dictionary_once(repo_root=ROOT, output=args.output, argv=argv, cwd=Path.cwd())
    except (S1InstrumentationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "INSTRUMENTATION_INSUFFICIENT",
                    "reason": type(exc).__name__,
                    "message": str(exc),
                    "formal_result": False,
                    "stage_progression_allowed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return dictionary_exit_code(result["status"])


if __name__ == "__main__":
    raise SystemExit(main())
