from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sc_sstw_feasibility.rc0_causal_localization_v2 import (
    ALLOWED_STATES,
    ATTEMPT_ORDER,
    CONDITION_ORDER,
    CROSS_RESIDUAL_EXPECTED,
    EVIDENCE_LABEL,
    FEATURE_NOISE_MULTIPLIER,
    GROUP_ORDER,
    PIXEL_ABSOLUTE_MIN,
    PIXEL_NOISE_MAX,
    ProtocolViolation,
    RELATION_MAX_RESIDUAL,
    START_INDICES,
    STATUS_INSUFFICIENT,
    STATUS_INVALID,
    STATUS_P_FAIL,
    STATUS_P_PASS_R_FAIL,
    STATUS_P_PASS_R_PASS,
    level_p_metrics,
    level_r_metrics,
    schedule_a,
    schedule_b,
    schedule_preflight,
    terminal_state,
    validate_config,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rc0_causal_localization_v2.json"
PLAN_PATH = ROOT / "plans/rc0_matched_quartets_v2.json"
PROTOCOL_PATH = ROOT / "protocols/rc0_causal_localization_v2.md"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cells(value: bool = True) -> dict[str, bool]:
    return {f"{group}:{condition}": value for group in GROUP_ORDER for condition in ("A", "B")}


def _embedded_schedule(name: str) -> np.ndarray:
    points = np.asarray(schedule_a() if name == "A" else schedule_b(), dtype=np.float64)
    matrix = np.zeros((13, 30), dtype=np.float64)
    for index in range(15):
        matrix[:, 2 * index] = (index + 1) * points[:, 0] + 0.1 * index
        matrix[:, 2 * index + 1] = (16 - index) * points[:, 1] - 0.05 * index
    return matrix


def test_config_plan_and_frozen_rc1_g0_guards() -> None:
    config = _load(CONFIG_PATH)
    plan = _load(PLAN_PATH)
    validate_config(config)
    validate_plan(plan)

    rc1_config = _load(ROOT / "configs/rc1_method_validation.json")
    rc1_plan = _load(ROOT / "plans/rc1_matched_triplets.json")
    assert config["carrier"] == rc1_config["carrier"]
    assert config["model"] == rc1_config["model"]
    assert config["generation"] == rc1_config["generation"]
    assert config["encoding"] == rc1_config["encoding"]
    assert [(group["group_id"], group["content_grammar"], group["prompt"], group["seed"]) for group in plan["groups"]] == [
        (group["group_id"], group["content_grammar"], group["prompt"], group["seed"])
        for group in rc1_plan["groups"]
    ]
    for group in plan["groups"]:
        assert hashlib.sha256(group["prompt"].encode("utf-8")).hexdigest() == group["prompt_sha256"]

    guards = config["source_guards"]
    assert _sha(ROOT / "configs/rc1_method_validation.json") == guards["rc1_config_raw_sha256"]
    assert _sha(ROOT / "plans/rc1_matched_triplets.json") == guards["rc1_plan_raw_sha256"]
    assert _sha(ROOT / "protocols/rc1_method_validation.md") == guards["rc1_protocol_raw_sha256"]
    assert _sha(ROOT / "configs/learned_observation_l1_v2_development.json") == guards["g0_config_raw_sha256"]


def test_exact_two_by_four_budget_and_only_condition_difference() -> None:
    plan = _load(PLAN_PATH)
    assert tuple(plan["group_order"]) == GROUP_ORDER
    assert tuple(plan["condition_order"]) == CONDITION_ORDER
    assert plan["attempt_budget"] == 8
    assert tuple((attempt["group_id"], attempt["condition"]) for attempt in plan["attempts"]) == ATTEMPT_ORDER
    assert plan["only_allowed_difference"] == "condition_label_and_corresponding_frozen_carrier_schedule"
    assert plan["retry_policy"] == "no_retry_no_replacement_no_additional_attempts"
    assert plan["stop_on_attempt_failure"] is True
    assert plan["formal_data_paths"] == []


def test_schedule_geometry_and_every_analytic_window() -> None:
    result = schedule_preflight()
    assert result["evidence_label"] == EVIDENCE_LABEL
    assert result["passed"] is True
    assert result["A_observation_by_B_template"] == pytest.approx(CROSS_RESIDUAL_EXPECTED, abs=1e-9)
    assert result["B_observation_by_A_template"] == pytest.approx(CROSS_RESIDUAL_EXPECTED, abs=1e-9)
    for observation in ("A", "B"):
        for template in ("A", "B"):
            for start, residual in enumerate(result["wrong_window_residuals"][observation][template]):
                assert (residual <= RELATION_MAX_RESIDUAL) is (observation == template and start == 0)
    a, b = schedule_a(), schedule_b()
    assert a[:3] == b[:3]
    assert b[4] == a[5] and b[5] == a[4]
    assert a[6:] == b[6:]


def test_level_p_symmetric_formula_and_inclusive_boundaries() -> None:
    shape = (3, 2, 2, 3)
    base = np.full(shape, 0.5)
    at_bound = level_p_metrics(base - PIXEL_NOISE_MAX, base + PIXEL_NOISE_MAX, base + 3.0 * PIXEL_NOISE_MAX)
    assert at_bound["off_repeat_floor"] == pytest.approx(PIXEL_NOISE_MAX)
    assert at_bound["effect"] == pytest.approx(3.0 * PIXEL_NOISE_MAX)
    assert at_bound["cell_pass"] is True

    absolute_bound = level_p_metrics(base, base, base + PIXEL_ABSOLUTE_MIN)
    assert absolute_bound["cell_pass"] is True
    below_absolute = level_p_metrics(base, base, base + PIXEL_ABSOLUTE_MIN - 2e-12)
    assert below_absolute["checks"]["absolute_effect_at_least_two_code_values"] is False
    noisy_off = level_p_metrics(base - PIXEL_NOISE_MAX - 2e-12, base + PIXEL_NOISE_MAX + 2e-12, base + 0.1)
    assert noisy_off["checks"]["off_repeat_noise_at_most_one_code_value"] is False
    assert noisy_off["off_repeat_control_valid"] is False
    assert noisy_off["cell_pass"] is True


def test_level_p_rejects_metric_shape_and_range_switching() -> None:
    good = np.full((2, 1, 1, 3), 0.5)
    with pytest.raises(ProtocolViolation):
        level_p_metrics(good[..., :2], good[..., :2], good[..., :2])
    with pytest.raises(ProtocolViolation):
        level_p_metrics(good, good, np.full_like(good, 1.1))
    with pytest.raises(ProtocolViolation):
        level_p_metrics(good, good[:1], good)


@pytest.mark.parametrize("condition", ["A", "B"])
def test_level_r_fixed_no_training_formula_accepts_only_own_start0(condition: str) -> None:
    off1 = np.zeros((13, 30), dtype=np.float64)
    off2 = np.zeros((13, 30), dtype=np.float64)
    metrics = level_r_metrics(off1, off2, _embedded_schedule(condition), condition)
    assert metrics["readout"] == "fixed_affine_relation_projection_no_training"
    assert metrics["signal_norm"] > FEATURE_NOISE_MULTIPLIER * metrics["off_repeat_floor_norm"]
    assert metrics["cell_pass"] is True
    assert metrics["residuals"][condition][0] <= RELATION_MAX_RESIDUAL
    assert all(value > RELATION_MAX_RESIDUAL for value in metrics["residuals"][condition][1:])
    cross = "B" if condition == "A" else "A"
    assert all(value > RELATION_MAX_RESIDUAL for value in metrics["residuals"][cross])


def test_level_r_time_or_content_only_and_off_noise_fail() -> None:
    zeros = np.zeros((13, 30), dtype=np.float64)
    content_only = np.ones((13, 30), dtype=np.float64)
    degenerate = level_r_metrics(zeros, zeros, content_only, "A")
    assert degenerate["checks"]["feature_signal_non_degenerate"] is False
    assert degenerate["cell_pass"] is False

    off_pattern = _embedded_schedule("A")
    weak_signal = 2.0 * off_pattern
    noisy = level_r_metrics(off_pattern, -off_pattern, weak_signal, "A")
    assert noisy["checks"]["feature_signal_at_least_three_times_off_floor"] is False
    assert noisy["cell_pass"] is False

    shifted = np.roll(_embedded_schedule("A"), 1, axis=0)
    wrong_time = level_r_metrics(zeros, zeros, shifted, "A")
    assert wrong_time["checks"]["own_template_start0_accept"] is False
    assert wrong_time["cell_pass"] is False


def test_state_priority_all_cells_and_level_r_short_circuit() -> None:
    p = _cells(True)
    r = _cells(True)
    assert terminal_state(integrity_valid=False, evidence_complete=True, off_repeat_control_valid=True, p_cells=p, r_was_executed=False) == STATUS_INVALID
    assert terminal_state(integrity_valid=True, evidence_complete=False, off_repeat_control_valid=True, p_cells=p, r_was_executed=False) == STATUS_INSUFFICIENT
    assert terminal_state(integrity_valid=True, evidence_complete=True, off_repeat_control_valid=False, p_cells=p, r_was_executed=False) == STATUS_INSUFFICIENT

    p_fail = dict(p)
    p_fail["orbital_glass:A"] = False
    assert terminal_state(integrity_valid=True, evidence_complete=True, off_repeat_control_valid=True, p_cells=p_fail, r_was_executed=False) == STATUS_P_FAIL
    assert terminal_state(integrity_valid=True, evidence_complete=True, off_repeat_control_valid=True, p_cells=p_fail, r_was_executed=True, r_cells=r) == STATUS_INVALID

    assert terminal_state(integrity_valid=True, evidence_complete=True, off_repeat_control_valid=True, p_cells=p, r_was_executed=False) == STATUS_INSUFFICIENT
    r_fail = dict(r)
    r_fail["articulated_paper:B"] = False
    assert terminal_state(integrity_valid=True, evidence_complete=True, off_repeat_control_valid=True, p_cells=p, r_was_executed=True, r_cells=r_fail) == STATUS_P_PASS_R_FAIL
    assert terminal_state(integrity_valid=True, evidence_complete=True, off_repeat_control_valid=True, p_cells=p, r_was_executed=True, r_cells=r) == STATUS_P_PASS_R_PASS
    assert tuple(ALLOWED_STATES) == (
        STATUS_P_FAIL,
        STATUS_P_PASS_R_FAIL,
        STATUS_P_PASS_R_PASS,
        STATUS_INVALID,
        STATUS_INSUFFICIENT,
    )


def test_threshold_condition_retry_and_budget_injection_fail_closed() -> None:
    config = _load(CONFIG_PATH)
    changed = copy.deepcopy(config)
    changed["level_p"]["absolute_effect_min"] = 0.0
    with pytest.raises(ProtocolViolation):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["level_r"]["maximum_relation_residual"] = 1.0
    with pytest.raises(ProtocolViolation):
        validate_config(changed)

    plan = _load(PLAN_PATH)
    changed_plan = copy.deepcopy(plan)
    changed_plan["groups"][0]["conditions"].append("C")
    with pytest.raises(ProtocolViolation):
        validate_plan(changed_plan)
    changed_plan = copy.deepcopy(plan)
    changed_plan["attempts"].append({"attempt_index": 9, "group_id": "orbital_glass", "condition": "A"})
    with pytest.raises(ProtocolViolation):
        validate_plan(changed_plan)
    changed_plan = copy.deepcopy(plan)
    changed_plan["retry_policy"] = "retry_once"
    with pytest.raises(ProtocolViolation):
        validate_plan(changed_plan)


def test_protocol_boundary_is_nonformal_and_formula_module_has_no_execution_io() -> None:
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    module = (ROOT / "src/sc_sstw_feasibility/rc0_causal_localization_v2.py").read_text(encoding="utf-8")
    config = _load(CONFIG_PATH)
    plan = _load(PLAN_PATH)
    assert "formal_result=false" in protocol
    assert "stage_progression_allowed=false" in protocol
    assert "oracle-paired" in protocol
    assert "41007–41008 remain" in protocol
    assert config["formal_result"] is False and config["stage_progression_allowed"] is False
    assert plan["formal_result"] is False and plan["stage_progression_allowed"] is False
    for forbidden in ("import subprocess", "subprocess.run", "decode_saved_mp4", "run_gpu_generation", "authorization_manifest", "google drive"):
        assert forbidden not in module.lower()
