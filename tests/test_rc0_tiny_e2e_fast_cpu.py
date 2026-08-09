from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from src.sc_sstw_feasibility.rc0_causal_localization_v2 import schedule_a
from src.sc_sstw_feasibility.rc0_tiny_e2e_fast_cpu import (
    STEP_AUDITS,
    TinyE2EDiagnosticError,
    _truth_audit,
    capture_candidates_without_truth,
    load_and_validate_config,
    run_condition_pipeline,
    run_step6_once,
)


ROOT = Path(__file__).resolve().parents[1]


def _affine_observation() -> np.ndarray:
    q = np.asarray(schedule_a(), dtype=np.float64)
    matrix = np.arange(60, dtype=np.float64).reshape(2, 30) / 31.0 + 0.2
    bias = np.linspace(-0.3, 0.4, 30)
    return q @ matrix + bias


def test_config_binds_exact_tiny_sample_audits_and_stage_order() -> None:
    config = load_and_validate_config(ROOT / "configs/rc0_tiny_e2e_fast_cpu.json")
    assert config["success"] == {
        "original_condition_count": 8,
        "off_negative_count": 4,
        "positive_condition_count": 4,
        "phase_cell_count": 16,
        "aggregation": "all_conditions_and_all_phase_cells_no_average_no_majority",
    }
    assert set(STEP_AUDITS) == {"step12", "step3", "step4", "step5"}
    assert all(len(digest) == 64 for _, digest in STEP_AUDITS.values())


def test_capture_set_is_invariant_to_truth_audit_label() -> None:
    candidates = capture_candidates_without_truth(_affine_observation())
    assert len(candidates) == 8
    assert sorted(candidate["start_index"] for candidate in candidates) == list(range(8))


def test_off_presence_rejection_short_circuits_all_downstream() -> None:
    frames = np.zeros((49, 320, 512, 3), dtype=np.uint8)
    features = np.zeros((13, 30), dtype=np.float64)
    result = run_condition_pipeline(frames, frames, frames, features, features, features)
    assert result["presence"]["cell_pass"] is False
    assert result["downstream_executed"] is False
    assert result["paired_30D_sha256"] is None
    assert result["capture"] is None and result["calibration"] is None and result["phase"] is None


def test_pipeline_signature_has_no_expected_truth_and_truth_is_separate() -> None:
    assert "expected_truth" not in inspect.signature(run_condition_pipeline).parameters
    assert "condition" not in inspect.signature(run_condition_pipeline).parameters
    source = inspect.getsource(run_step6_once)
    assert source.index("outputs[f\"{group}:{condition}\"] = run_condition_pipeline") < source.index("_truth_audit(outputs)")


def test_truth_audit_requires_off_short_circuit_and_all_positive_phase_cells() -> None:
    outputs = {}
    for group in ("orbital_glass", "articulated_paper"):
        for condition in ("OFF_R1", "OFF_R2"):
            outputs[f"{group}:{condition}"] = {"presence": {"cell_pass": False}, "downstream_executed": False, "calibration": None, "phase": None}
        for condition in ("A", "B"):
            outputs[f"{group}:{condition}"] = {
                "presence": {"cell_pass": True}, "downstream_executed": True,
                "calibration": {"selected_candidates": [{"start_index": 0, "template_id": condition}]},
                "phase": {name: {"passed": True} for name in ("identity", "delete6_duplicate12", "local_phase_plus1", "local_phase_minus1")},
            }
    truth, first = _truth_audit(outputs)
    assert first is None and len(truth) == 8 and all(item["passed"] for item in truth.values())
    outputs["orbital_glass:OFF_R1"]["downstream_executed"] = True
    _, first = _truth_audit(outputs)
    assert first == "capture"


def test_config_drift_and_scientific_reimplementation_are_absent(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/rc0_tiny_e2e_fast_cpu.json").read_text(encoding="utf-8"))
    config["pipeline"]["capture_K"] = 9
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    try:
        load_and_validate_config(path)
    except TinyE2EDiagnosticError:
        pass
    else:
        raise AssertionError("pipeline budget drift was accepted")
    source = inspect.getsource(run_condition_pipeline)
    for required in ("level_p_metrics(", "capture_candidates_without_truth(", "calibrate_capture_candidates(", "equalize_observations(", "evaluate_phase_cell("):
        assert required in source
    for forbidden in ("encode_mp4(", "dynamic_time_sync(", "affine_burst_residual(", "calibrate_from_pilot_pairs(", "generation("):
        assert forbidden not in source
