#!/usr/bin/env python3
"""Run the frozen two-dimensional SC-SSTW synthetic mechanism Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.mechanism_synthetic_gate import ProtocolError, run_gate  # noqa: E402


def _write_failure(output_dir: Path, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "evidence_kind": "frozen_2d_synthetic_only_no_video_no_gpu",
        "gate_pass": False,
        "failure_type": "ProtocolError",
        "failure_reason": reason,
        "paper_claim": False,
        "gpu_claim": False,
        "saved_video_claim": False,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (output_dir / "metrics.json").write_text(encoded, encoding="utf-8")
    decision = {
        "gate": "Gate 1",
        "implementer_decision": "GATE_FAIL",
        "auditor_decision": "PENDING",
        "reason": reason,
    }
    (output_dir / "gate_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="complete frozen mechanism configuration; no defaults are supplied",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        metrics = run_gate(args.config, args.output_dir)
    except ProtocolError as exc:
        _write_failure(args.output_dir, str(exc))
        print(json.dumps({"gate_pass": False, "failure_type": "ProtocolError", "failure_reason": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0 if metrics["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
