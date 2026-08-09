from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

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
    VIDEO_SHAPE,
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


def _video(value: float, shape: tuple[int, ...] = VIDEO_SHAPE) -> np.ndarray:
    scalar = np.asarray([[[[value]]]], dtype=np.float64)
    return np.broadcast_to(scalar, shape)


def _set_path(document: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    current = document
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


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
    extractor_path = "src/sc_sstw_feasibility/learned_observation.py"
    extractor_commit = guards["extractor_source_commit"]
    blob = subprocess.check_output(
        ["git", "rev-parse", f"{extractor_commit}:{extractor_path}"],
        cwd=ROOT,
        text=True,
    ).strip()
    raw = subprocess.check_output(["git", "show", f"{extractor_commit}:{extractor_path}"], cwd=ROOT)
    assert blob == "6288d954a1bdaded5fd2f92ed78b463bc11a6a18" == guards["extractor_git_blob"]
    assert hashlib.sha256(raw).hexdigest() == guards["extractor_raw_sha256"]
    assert guards["extractor_source_tree"] == "90e50f107685c768600686239c922242e660d20a"
    assert guards["extractor_source_path"] == extractor_path
    assert guards["extractor_symbol"] == "extract_feature_matrix"


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
    at_bound = level_p_metrics(
        _video(0.5 - PIXEL_NOISE_MAX),
        _video(0.5 + PIXEL_NOISE_MAX),
        _video(0.5 + 3.0 * PIXEL_NOISE_MAX),
    )
    assert at_bound["off_repeat_floor"] == pytest.approx(PIXEL_NOISE_MAX)
    assert at_bound["effect"] == pytest.approx(3.0 * PIXEL_NOISE_MAX)
    assert at_bound["cell_pass"] is True

    absolute_bound = level_p_metrics(_video(0.5), _video(0.5), _video(0.5 + PIXEL_ABSOLUTE_MIN))
    assert absolute_bound["cell_pass"] is True
    below_absolute = level_p_metrics(_video(0.5), _video(0.5), _video(0.5 + PIXEL_ABSOLUTE_MIN - 2e-12))
    assert below_absolute["checks"]["absolute_effect_at_least_two_code_values"] is False
    noisy_off = level_p_metrics(
        _video(0.5 - PIXEL_NOISE_MAX - 2e-12),
        _video(0.5 + PIXEL_NOISE_MAX + 2e-12),
        _video(0.6),
    )
    assert noisy_off["checks"]["off_repeat_noise_at_most_one_code_value"] is False
    assert noisy_off["off_repeat_control_valid"] is False
    assert noisy_off["cell_pass"] is True


def test_level_p_rejects_metric_shape_and_range_switching() -> None:
    good = _video(0.5)
    bad_shapes = (
        (48, 1, 1, 3),
        (49, 1, 1, 3),
        (49, 320, 511, 3),
        (49, 320, 512, 1),
        (49, 320, 512, 4),
    )
    for shape in bad_shapes:
        with pytest.raises(ProtocolViolation):
            level_p_metrics(_video(0.5, shape), good, good)
    for bad_value in (float("nan"), float("inf"), -1e-12, 1.0 + 1e-12):
        with pytest.raises(ProtocolViolation):
            level_p_metrics(good, good, _video(bad_value))


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


@pytest.mark.parametrize(
    "p_mutation",
    [
        lambda cells: cells.pop("orbital_glass:A"),
        lambda cells: cells.__setitem__("extra:A", True),
        lambda cells: cells.__setitem__("orbital_glass:A", 1),
        lambda cells: cells.__setitem__("orbital_glass:A", "true"),
    ],
)
def test_state_rejects_bad_p_schema_before_noise_sufficiency(p_mutation) -> None:
    p = _cells(True)
    p_mutation(p)
    assert terminal_state(
        integrity_valid=True,
        evidence_complete=True,
        off_repeat_control_valid=False,
        p_cells=p,
        r_was_executed=False,
    ) == STATUS_INVALID


@pytest.mark.parametrize(
    "r_mutation",
    [
        lambda cells: cells.pop("articulated_paper:B"),
        lambda cells: cells.__setitem__("extra:B", True),
        lambda cells: cells.__setitem__("articulated_paper:B", 0),
        lambda cells: cells.__setitem__("articulated_paper:B", "false"),
    ],
)
def test_state_rejects_bad_r_schema_and_execution_consistency(r_mutation) -> None:
    r = _cells(True)
    r_mutation(r)
    assert terminal_state(
        integrity_valid=True,
        evidence_complete=True,
        off_repeat_control_valid=False,
        p_cells=_cells(True),
        r_was_executed=True,
        r_cells=r,
    ) == STATUS_INVALID
    assert terminal_state(
        integrity_valid=True,
        evidence_complete=True,
        off_repeat_control_valid=True,
        p_cells=_cells(True),
        r_was_executed=False,
        r_cells=_cells(True),
    ) == STATUS_INVALID
    assert terminal_state(
        integrity_valid=True,
        evidence_complete=True,
        off_repeat_control_valid=True,
        p_cells=_cells(True),
        r_was_executed=True,
        r_cells=None,
    ) == STATUS_INVALID


def test_state_rejects_r_after_p_fail_and_nonboolean_flags() -> None:
    p = _cells(True)
    p["orbital_glass:A"] = False
    assert terminal_state(
        integrity_valid=True,
        evidence_complete=True,
        off_repeat_control_valid=False,
        p_cells=p,
        r_was_executed=True,
        r_cells=_cells(True),
    ) == STATUS_INVALID
    assert terminal_state(
        integrity_valid=1,
        evidence_complete=True,
        off_repeat_control_valid=True,
        p_cells=_cells(True),
        r_was_executed=False,
    ) == STATUS_INVALID


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


def test_config_top_level_extra_missing_reorder_and_audit_pocs_reject() -> None:
    config = _load(CONFIG_PATH)
    changed = copy.deepcopy(config)
    changed["extra_claim"] = "method"
    with pytest.raises(ProtocolViolation):
        validate_config(changed)
    changed = copy.deepcopy(config)
    del changed["question_order"]
    with pytest.raises(ProtocolViolation):
        validate_config(changed)
    changed = dict(reversed(list(config.items())))
    with pytest.raises(ProtocolViolation):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["question_order"][0] = "renamed_question"
    with pytest.raises(ProtocolViolation):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["evidence_class"] = "formal_result"
    with pytest.raises(ProtocolViolation):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["state_priority"][0], changed["state_priority"][1] = changed["state_priority"][1], changed["state_priority"][0]
    with pytest.raises(ProtocolViolation):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["condition_order"].append("C")
    with pytest.raises(ProtocolViolation):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["source_guards"]["baseline_tree"] = "0" * 40
    with pytest.raises(ProtocolViolation):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["source_guards"]["extractor_git_blob"] = "0" * 40
    with pytest.raises(ProtocolViolation):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["states"]["renamed"] = changed["states"].pop("p_fail")
    with pytest.raises(ProtocolViolation):
        validate_config(changed)


@pytest.mark.parametrize(
    "object_name",
    ["schedule", "carrier", "model", "generation", "encoding", "decode", "level_p", "level_r", "states", "source_guards"],
)
@pytest.mark.parametrize("mutation", ["extra", "missing", "reorder"])
def test_every_config_nested_object_has_exact_keys_and_order(object_name: str, mutation: str) -> None:
    config = _load(CONFIG_PATH)
    nested = config[object_name]
    if mutation == "extra":
        nested["unexpected"] = None
    elif mutation == "missing":
        del nested[next(iter(nested))]
    else:
        config[object_name] = dict(reversed(list(nested.items())))
    with pytest.raises(ProtocolViolation):
        validate_config(config)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("attempt_budget", 8.0),
        ("formal_result", 0),
        ("question_order", ("H_P_saved_mp4_pixel_bridge", "H_R_paired_public_relation")),
        ("schedule.B_swapped_indices", (4, 5)),
        ("carrier.block_index", 29.0),
        ("generation.inference_steps", True),
        ("encoding.quality", 5),
        ("encoding.bitrate", 0),
        ("encoding.fallback_permitted", 0),
        ("decode.integer_range", (0, 255)),
        ("level_p.relative_floor_multiplier", 3),
        ("level_r.maximum_relation_residual", "0.25"),
        ("source_guards.extractor_git_blob", None),
    ],
)
def test_config_exact_json_types_reject_python_loose_equality(path: str, replacement: object) -> None:
    config = _load(CONFIG_PATH)
    _set_path(config, path, replacement)
    with pytest.raises(ProtocolViolation):
        validate_config(config)


def test_plan_top_group_attempt_and_audit_pocs_reject() -> None:
    plan = _load(PLAN_PATH)
    changed = copy.deepcopy(plan)
    changed["unexpected"] = None
    with pytest.raises(ProtocolViolation):
        validate_plan(changed)
    changed = copy.deepcopy(plan)
    del changed["plan_id"]
    with pytest.raises(ProtocolViolation):
        validate_plan(changed)
    changed = dict(reversed(list(plan.items())))
    with pytest.raises(ProtocolViolation):
        validate_plan(changed)
    changed = copy.deepcopy(plan)
    changed["plan_id"] = "renamed_plan"
    with pytest.raises(ProtocolViolation):
        validate_plan(changed)

    for collection in ("groups", "attempts"):
        changed = copy.deepcopy(plan)
        changed[collection][0]["unexpected"] = "retry" if collection == "attempts" else None
        with pytest.raises(ProtocolViolation):
            validate_plan(changed)
        changed = copy.deepcopy(plan)
        del changed[collection][0][next(iter(changed[collection][0]))]
        with pytest.raises(ProtocolViolation):
            validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed[collection][0] = dict(reversed(list(changed[collection][0].items())))
        with pytest.raises(ProtocolViolation):
            validate_plan(changed)

    changed = copy.deepcopy(plan)
    changed["groups"][0]["unexpected"] = "group_extra"
    with pytest.raises(ProtocolViolation):
        validate_plan(changed)
    changed = copy.deepcopy(plan)
    changed["attempts"][0]["retry"] = True
    with pytest.raises(ProtocolViolation):
        validate_plan(changed)
    changed = copy.deepcopy(plan)
    changed["failure_semantics"] = "retry_and_continue"
    with pytest.raises(ProtocolViolation):
        validate_plan(changed)


@pytest.mark.parametrize("mutation", ["remove", "extra", "reorder"])
def test_plan_matched_fields_are_exact_and_ordered(mutation: str) -> None:
    plan = _load(PLAN_PATH)
    fields = plan["matched_fields"]
    if mutation == "remove":
        fields.remove("actual_initial_latent_sha256")
    elif mutation == "extra":
        fields.append("candidate")
    else:
        fields[3], fields[4] = fields[4], fields[3]
    with pytest.raises(ProtocolViolation):
        validate_plan(plan)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("attempt_budget", 8.0),
        ("stop_on_attempt_failure", 1),
        ("formal_result", 0),
        ("group_order", ("orbital_glass", "articulated_paper")),
    ],
)
def test_plan_top_exact_json_types(path: str, replacement: object) -> None:
    plan = _load(PLAN_PATH)
    _set_path(plan, path, replacement)
    with pytest.raises(ProtocolViolation):
        validate_plan(plan)


def test_plan_nested_exact_json_types() -> None:
    plan = _load(PLAN_PATH)
    changed = copy.deepcopy(plan)
    changed["groups"][0]["seed"] = 52001.0
    with pytest.raises(ProtocolViolation):
        validate_plan(changed)
    changed = copy.deepcopy(plan)
    changed["attempts"][0]["attempt_index"] = True
    with pytest.raises(ProtocolViolation):
        validate_plan(changed)
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
