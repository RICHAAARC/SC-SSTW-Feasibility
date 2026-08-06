"""Protocol v2 Gate 1A: input admissibility only.

This module intentionally depends only on the standard library and never imports
acquisition, calibration, synchronization, or scoring code.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


class Gate1AError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _matrix(case: dict[str, Any]) -> list[list[float]]:
    c1, s1 = math.cos(case["theta_1"]), math.sin(case["theta_1"])
    c2, s2 = math.cos(case["theta_2"]), math.sin(case["theta_2"])
    d1, d2 = case["s_1"], case["s_2"]
    return [
        [c1 * d1 * c2 - s1 * d2 * s2, -c1 * d1 * s2 - s1 * d2 * c2],
        [s1 * d1 * c2 + c1 * d2 * s2, -s1 * d1 * s2 + c1 * d2 * c2],
    ]


def _numeric_condition(a: list[list[float]]) -> float:
    x = a[0][0] ** 2 + a[1][0] ** 2
    y = a[0][0] * a[0][1] + a[1][0] * a[1][1]
    z = a[0][1] ** 2 + a[1][1] ** 2
    root = math.sqrt(max(0.0, (x - z) ** 2 + 4.0 * y * y))
    largest = math.sqrt(max(0.0, (x + z + root) / 2.0))
    smallest = math.sqrt(max(0.0, (x + z - root) / 2.0))
    return math.inf if smallest == 0.0 else largest / smallest


def run_preflight(config_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if config.get("protocol_id") != "mechanism_feasibility_v2":
        raise Gate1AError("protocol_id must be mechanism_feasibility_v2")
    protocol = config["synthetic_protocol"]
    gate = config["gate_1a"]
    channel = protocol["channel"]
    cases = channel["case_parameters"]
    edits = protocol["case_protocols"]
    if len(cases) != gate["case_count"] or len(edits) != gate["case_count"]:
        raise Gate1AError("exactly eight channel and edit cases are required")
    if protocol["null_protocol"]["case_count"] != gate["null_case_count"]:
        raise Gate1AError("exactly eight null cases are required")
    if [item["case_index"] for item in cases] != list(range(gate["case_count"])):
        raise Gate1AError("channel case indices must be complete and ordered")
    if [item["case_index"] for item in edits] != list(range(gate["case_count"])):
        raise Gate1AError("edit case indices must be complete and ordered")

    records = []
    for case in cases:
        values = [case["theta_1"], case["theta_2"], case["s_1"], case["s_2"], *case["bias"]]
        if len(case["bias"]) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise Gate1AError(f"case {case['case_index']} is not finite two-dimensional input")
        if case["s_1"] <= 0.0 or case["s_2"] <= 0.0:
            raise Gate1AError(f"case {case['case_index']} has non-positive scale")
        a = _matrix(case)
        determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        condition = _numeric_condition(a)
        declared = max(case["s_1"], case["s_2"]) / min(case["s_1"], case["s_2"])
        checks = {
            "finite": all(math.isfinite(v) for row in a for v in row),
            "full_rank": math.isfinite(determinant) and abs(determinant) > 0.0,
            "condition_at_most_10": math.isfinite(condition) and condition <= gate["maximum_condition_number"],
            "condition_matches_constructed_scales": math.isclose(condition, declared, rel_tol=1e-12, abs_tol=1e-12),
        }
        if not all(checks.values()):
            raise Gate1AError(f"case {case['case_index']} failed admissibility: {checks}")
        records.append({"case_index": case["case_index"], "A": a, "b": case["bias"], "determinant": determinant, "condition_number": condition, "checks": checks})

    result = {
        "gate": "Gate 1A",
        "evidence_kind": "v2_input_admissibility_only_no_method_metrics",
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "channel_sha256": hashlib.sha256(_canonical(channel)).hexdigest(),
        "edit_case_count": len(edits),
        "null_case_count": protocol["null_protocol"]["case_count"],
        "cases": records,
        "forbidden_method_calls_executed": [],
        "implementer_decision": "GATE_PASS",
        "auditor_decision": "PENDING",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate1a_input_admissibility.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
