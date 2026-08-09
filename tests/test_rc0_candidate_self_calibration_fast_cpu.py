from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from src.sc_sstw_feasibility.rc0_candidate_self_calibration_fast_cpu import (
    CANDIDATE_BUDGET_K,
    EQUALIZATION_RIDGE,
    FIT_INDICES,
    HELD_OUT_INDICES,
    K2,
    LEAST_SQUARES_IMPLEMENTATION_RIDGE,
    SelfCalibrationDiagnosticError,
    audit_ranked_cell,
    calibrate_capture_candidates,
    capture_set_sha256,
    load_and_validate_config,
    run_step4_once,
)
from src.sc_sstw_feasibility.rc0_causal_localization_v2 import schedule_a, schedule_b


ROOT = Path(__file__).resolve().parents[1]


def _candidate(start: int, template: str, rank: int) -> dict[str, object]:
    return {
        "start_index": start, "template_id": template, "residual": float(rank) / 100.0,
        "observed_length": 6, "missing_template_index": None,
        "full_rank": rank, "selected_rank": rank,
    }


def _candidates(truth_template: str = "A") -> list[dict[str, object]]:
    other = "B" if truth_template == "A" else "A"
    return [_candidate(0, truth_template, 1)] + [_candidate(start, other if start % 2 else truth_template, start + 1) for start in range(1, 8)]


def _affine_observation(schedule: tuple[tuple[float, float], ...]) -> np.ndarray:
    points = np.asarray(schedule, dtype=np.float64)
    matrix = np.arange(60, dtype=np.float64).reshape(2, 30) / 29.0 + 0.3
    bias = np.linspace(-0.4, 0.6, 30, dtype=np.float64)
    return points @ matrix + bias


def test_config_freezes_separation_formula_and_k2() -> None:
    config = load_and_validate_config(ROOT / "configs/rc0_candidate_self_calibration_fast_cpu.json")
    assert CANDIDATE_BUDGET_K == 8 and K2 == 1
    assert FIT_INDICES == (0, 1, 2, 3) and HELD_OUT_INDICES == (4, 5)
    assert EQUALIZATION_RIDGE == 1e-4
    assert LEAST_SQUARES_IMPLEMENTATION_RIDGE == 1e-8
    assert config["calibration"]["fit_parameter_dimension"] == 90
    assert config["calibration"]["least_squares_regularization_argument"] is None
    assert config["calibration"]["least_squares_implementation_default_ridge"] == 1e-8
    assert config["calibration"]["acceptance_threshold"] is None
    assert config["separation"]["truth_available_to_fit_or_ranking"] is False
    assert config["selection"]["winner_rule"] == "strictly_lower_held_out_MSE_than_runner_up"


def test_every_capture_candidate_is_independently_fit_and_truth_is_not_an_argument() -> None:
    observation = _affine_observation(schedule_a())
    candidates = _candidates("A")
    before = capture_set_sha256(candidates)
    ranked = calibrate_capture_candidates(observation, candidates)
    after = capture_set_sha256(sorted((record["capture_identity"] for record in ranked), key=lambda item: item["selected_rank"]))
    assert before == after
    assert len(ranked) == 8
    assert all(record["fit"]["parameter_dimension"] == 90 for record in ranked)
    assert all(record["fit"]["fit_indices"] == [0, 1, 2, 3] for record in ranked)
    assert "own_template" not in inspect.signature(calibrate_capture_candidates).parameters
    result = audit_ranked_cell(ranked, "A")
    assert result["selected_candidates"] == [candidates[0]]
    assert result["target_calibration_rank"] == 1
    assert result["passed"] is True


def test_candidate_specific_affine_nuisance_is_removed_without_cross_candidate_state() -> None:
    for template, schedule in (("A", schedule_a()), ("B", schedule_b())):
        observation = _affine_observation(schedule)
        ranked = calibrate_capture_candidates(observation, _candidates(template))
        target = next(record for record in ranked if record["capture_identity"]["start_index"] == 0)
        assert target["capture_identity"]["template_id"] == template
        # The frozen 1e-4 inverse ridge intentionally leaves a tiny bias.
        assert target["held_out_mse"] < 1e-7
        assert target["fit"]["pilot_reconstruction_mse"] < 1e-12


def test_strict_score_tie_fails_instead_of_using_serialization_tie_break() -> None:
    ranked = [
        {"held_out_mse": 0.1, "capture_identity": _candidate(0, "A", 1), "calibration_rank": 1},
        {"held_out_mse": 0.1, "capture_identity": _candidate(1, "B", 2), "calibration_rank": 2},
    ] + [
        {"held_out_mse": float(index), "capture_identity": _candidate(index, "A", index + 1), "calibration_rank": index + 1}
        for index in range(2, 8)
    ]
    result = audit_ranked_cell(ranked, "A")
    assert result["unique_winner"] is False
    assert result["selected_candidates"] == []
    assert result["passed"] is False


def test_capture_tamper_and_nonfinite_observation_fail_closed() -> None:
    candidates = _candidates()
    candidates[1]["start_index"] = 0
    try:
        calibrate_capture_candidates(np.zeros((13, 30)), candidates)
    except SelfCalibrationDiagnosticError:
        pass
    else:
        raise AssertionError("duplicate capture identity was accepted")
    invalid = np.zeros((13, 30))
    invalid[0, 0] = np.nan
    try:
        calibrate_capture_candidates(invalid, _candidates())
    except SelfCalibrationDiagnosticError:
        pass
    else:
        raise AssertionError("nonfinite observation was accepted")


def test_runner_is_read_only_no_generation_encoding_or_capture_recompute() -> None:
    source = inspect.getsource(run_step4_once)
    assert "decode_saved_mp4" in source and "extract_feature_matrix" in source
    assert "step3[\"cells\"]" in source
    for forbidden in ("encode_mp4(", "apply_carrier(", "scan_burst_candidates(", "run_step3_once(", "generation("):
        assert forbidden not in source
    cli = (ROOT / "experiments/run_rc0_candidate_self_calibration_fast_cpu.py").read_text(encoding="utf-8")
    assert "argparse" not in cli and "add_argument" not in cli


def test_config_rejects_threshold_or_k2_drift(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/rc0_candidate_self_calibration_fast_cpu.json").read_text(encoding="utf-8"))
    config["calibration"]["acceptance_threshold"] = 0.02
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    try:
        load_and_validate_config(path)
    except SelfCalibrationDiagnosticError:
        pass
    else:
        raise AssertionError("threshold injection was accepted")
    config["calibration"]["acceptance_threshold"] = None
    config["selection"]["K2"] = 2
    path.write_text(json.dumps(config), encoding="utf-8")
    try:
        load_and_validate_config(path)
    except SelfCalibrationDiagnosticError:
        pass
    else:
        raise AssertionError("K2 drift was accepted")
