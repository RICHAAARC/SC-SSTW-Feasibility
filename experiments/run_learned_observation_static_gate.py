#!/usr/bin/env python3
"""Run only the learned-observation CPU/static contract Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sc_sstw_feasibility.learned_observation import static_contract_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "learned_observation_frontend.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = static_contract_report(config)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
