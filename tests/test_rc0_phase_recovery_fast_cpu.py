from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from src.sc_sstw_feasibility.rc0_causal_localization_v2 import schedule_a, schedule_b
from src.sc_sstw_feasibility.rc0_phase_recovery_fast_cpu import (
    MAX_EDIT_BUDGET,
    PERTURBATION_ORDER,
    PERTURBATION_SOURCE_INDICES,
    REPEAT_PENALTY,
    SCORE_TOLERANCE,
    SKIP_PENALTY,
    PhaseRecoveryDiagnosticError,
    apply_time_perturbation,
    evaluate_phase_cell,
    expected_operation_path,
    load_and_validate_config,
    recover_complete_path,
    run_step5_once,
)
from src.sc_sstw_feasibility.sync import dynamic_time_sync


ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_existing_dtw_and_minimal_perturbations() -> None:
    config = load_and_validate_config(ROOT / "configs/rc0_phase_recovery_fast_cpu.json")
    assert SKIP_PENALTY == 0.22 and REPEAT_PENALTY == 0.14
    assert config["dp"]["band"] is None
    assert config["dp"]["transition_tie_order"] == ["MATCH", "SKIP_TEMPLATE", "REPEAT_TEMPLATE"]
    assert MAX_EDIT_BUDGET == 2 and SCORE_TOLERANCE == 1e-12
    assert PERTURBATION_ORDER == ("identity", "delete6_duplicate12", "local_phase_plus1", "local_phase_minus1")
    assert all(len(indices) == 13 for indices in PERTURBATION_SOURCE_INDICES.values())


def test_exact_truth_paths_and_edit_budgets_are_preregistered() -> None:
    identity = expected_operation_path("identity")
    assert len(identity) == 13 and all(item["operation"] == "MATCH" for item in identity)
    for perturbation in PERTURBATION_ORDER[1:]:
        path = expected_operation_path(perturbation)
        assert sum(item["operation"] == "SKIP_TEMPLATE" for item in path) == 1
        assert sum(item["operation"] == "REPEAT_TEMPLATE" for item in path) == 1
        assert sum(item["operation"] != "MATCH" for item in path) == 2


def test_synthetic_exact_schedule_recovers_all_paths_and_beats_cross() -> None:
    for own, schedule in (("A", schedule_a()), ("B", schedule_b())):
        for perturbation in PERTURBATION_ORDER:
            result = evaluate_phase_cell(schedule, own, perturbation)
            assert result["path_exact"] is True
            assert result["edit_exact"] is True
            assert result["score_consistent"] is True
            assert result["bounded_abandoned"] is False
            assert result["own_beats_cross"] is True
            assert result["passed"] is True


def test_complete_backtracker_matches_existing_primary_on_nontrivial_values() -> None:
    observed = apply_time_perturbation(np.asarray(schedule_a(), dtype=np.float64) + 0.003, "local_phase_plus1")
    template = [tuple(map(float, point)) for point in schedule_a()]
    primary = dynamic_time_sync(observed, template, skip_penalty=SKIP_PENALTY, repeat_penalty=REPEAT_PENALTY)
    complete = recover_complete_path(observed, template)
    assert abs(primary.score - complete["score"]) <= SCORE_TOLERANCE
    assert [list(pair) for pair in primary.path] == complete["diagonal_path"]


def test_invalid_shape_unknown_perturbation_and_config_drift_fail_closed(tmp_path: Path) -> None:
    for states, perturbation in ((np.zeros((12, 2)), "identity"), (np.zeros((13, 2)), "unknown")):
        try:
            apply_time_perturbation(states, perturbation)
        except PhaseRecoveryDiagnosticError:
            pass
        else:
            raise AssertionError("invalid perturbation input was accepted")
    config = json.loads((ROOT / "configs/rc0_phase_recovery_fast_cpu.json").read_text(encoding="utf-8"))
    config["dp"]["skip_penalty"] = 0.23
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    try:
        load_and_validate_config(path)
    except PhaseRecoveryDiagnosticError:
        pass
    else:
        raise AssertionError("DP parameter drift was accepted")


def test_runner_consumes_frozen_calibration_and_has_no_generation_encoding_or_refit() -> None:
    source = inspect.getsource(run_step5_once)
    assert "_frozen_calibration" in source and "equalize_observations" in source
    for forbidden in ("encode_mp4(", "apply_carrier(", "scan_burst_candidates(", "calibrate_from_pilot_pairs(", "generation("):
        assert forbidden not in source
    cli = (ROOT / "experiments/run_rc0_phase_recovery_fast_cpu.py").read_text(encoding="utf-8")
    assert "argparse" not in cli and "add_argument" not in cli
