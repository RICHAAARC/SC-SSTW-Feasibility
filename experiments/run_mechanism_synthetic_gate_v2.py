#!/usr/bin/env python3
"""Run frozen protocol v2 Gate 1B after verifying Gate 1A admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.mechanism_synthetic_gate_v2 import ProtocolError, run_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gate1a-evidence", required=True, type=Path)
    parser.add_argument("--admission-lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        metrics = run_gate(args.config, args.gate1a_evidence, args.admission_lock, args.output_dir)
    except (ProtocolError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"gate": "Gate 1B", "gate_pass": False, "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(metrics, sort_keys=True))
    return 0 if metrics["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
