#!/usr/bin/env python3
"""Run protocol v2 Gate 1A input-admissibility preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.gate1a_preflight import Gate1AError, run_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run_preflight(args.config, args.output_dir)
    except (Gate1AError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"gate": "Gate 1A", "implementer_decision": "GATE_FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
