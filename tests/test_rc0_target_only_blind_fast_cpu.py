from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from sc_sstw_feasibility.rc0_causal_localization_v2 import schedule_a, schedule_b
from sc_sstw_feasibility.rc0_target_only_blind_fast_cpu import (
    ABSOLUTE_PRESENCE_THRESHOLD,
    FRAME_GROUPS,
    HEIGHT,
    PROJECTION_EPSILON,
    RELATION_RESIDUAL_THRESHOLD,
    WIDTH,
    aggregate_frame_projections,
    load_and_validate_config,
    project_centered_frame,
    run_target_pipeline,
    truth_audit_after_outputs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_and_formula_constants() -> None:
    config = load_and_validate_config(ROOT / "configs/rc0_target_only_blind_fast_cpu.json")
    assert config["observation"]["denominator_epsilon"] == PROJECTION_EPSILON == 1e-15
    assert config["presence"]["absolute_lower_bound"] == ABSOLUTE_PRESENCE_THRESHOLD == 2 / 255
    assert config["presence"]["relation_upper_bound"] == RELATION_RESIDUAL_THRESHOLD == 0.25
    assert FRAME_GROUPS[0] == (0,)
    assert FRAME_GROUPS[-1] == (45, 46, 47, 48)
    assert sorted(index for group in FRAME_GROUPS for index in group) == list(range(49))


def test_analytic_chroma_projection_sign_and_scale() -> None:
    x = np.arange(WIDTH, dtype=np.float64)
    y = np.arange(HEIGHT, dtype=np.float64)
    u = np.asarray((1.0, -1.0, 0.0)) / np.sqrt(2.0)
    v = np.asarray((1.0, 1.0, -2.0)) / np.sqrt(6.0)
    bx = np.cos(2 * np.pi * x / WIDTH)[None, :, None] * u[None, None, :]
    by = np.cos(2 * np.pi * y / HEIGHT)[:, None, None] * v[None, None, :]
    frame = 0.5 + 0.0125 * bx + 0.02 * by
    qx, qy = project_centered_frame(frame)
    assert abs(qx - 0.0125) < 1e-12
    assert abs(qy - 0.02) < 1e-12


def test_frame_to_symbol_mapping_is_exact_arithmetic_mean() -> None:
    projections = np.column_stack((np.arange(49, dtype=np.float64), -np.arange(49, dtype=np.float64)))
    observed = aggregate_frame_projections(projections)
    assert observed.shape == (13, 2)
    assert observed[0].tolist() == [0.0, 0.0]
    assert observed[1].tolist() == [2.5, -2.5]
    assert observed[12].tolist() == [46.5, -46.5]


def _affine_schedule(which: str) -> np.ndarray:
    source = np.asarray(schedule_a() if which == "A" else schedule_b(), dtype=np.float64)
    matrix = np.asarray(((0.031, -0.007), (0.004, 0.027)), dtype=np.float64)
    return source @ matrix.T + np.asarray((0.002, -0.003), dtype=np.float64)


def test_truth_free_pipeline_closes_synthetic_a_and_b() -> None:
    for condition in ("A", "B"):
        output = run_target_pipeline(_affine_schedule(condition))
        assert output["presence"]["presence_pass"] is True
        assert output["calibration"]["unique_winner"] is True
        selected = output["calibration"]["selected_candidates"]
        assert [(item["start_index"], item["template_id"]) for item in selected] == [(0, condition)]
        assert all(cell["passed"] for cell in output["phase"].values())


def test_zero_target_rejects_presence_and_short_circuits() -> None:
    output = run_target_pipeline(np.zeros((13, 2), dtype=np.float64))
    assert output["presence"]["energy_pass"] is False
    assert output["presence"]["presence_pass"] is False
    assert output["downstream_executed"] is False
    assert output["calibration"] is None and output["phase"] is None


def test_truth_is_only_a_post_output_audit_argument() -> None:
    assert list(inspect.signature(run_target_pipeline).parameters) == ["observation"]
    outputs = {
        "orbital_glass:OFF_R1": run_target_pipeline(np.zeros((13, 2))),
        "orbital_glass:OFF_R2": run_target_pipeline(np.zeros((13, 2))),
        "orbital_glass:A": run_target_pipeline(_affine_schedule("A")),
        "orbital_glass:B": run_target_pipeline(_affine_schedule("B")),
        "articulated_paper:OFF_R1": run_target_pipeline(np.zeros((13, 2))),
        "articulated_paper:OFF_R2": run_target_pipeline(np.zeros((13, 2))),
        "articulated_paper:A": run_target_pipeline(_affine_schedule("A")),
        "articulated_paper:B": run_target_pipeline(_affine_schedule("B")),
    }
    audit, first_failure = truth_audit_after_outputs(outputs)
    assert first_failure is None
    assert len(audit) == 8 and all(item["passed"] for item in audit.values())


def test_config_declares_no_off_reference_or_training() -> None:
    config = json.loads((ROOT / "configs/rc0_target_only_blind_fast_cpu.json").read_text(encoding="utf-8"))
    assert config["input"]["off_reference"] == "forbidden"
    assert config["input"]["paired_difference"] == "forbidden"
    assert config["input"]["training"] == "forbidden"
