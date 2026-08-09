from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from sc_sstw_feasibility.rc0_causal_localization_v2 import schedule_a, schedule_b
from sc_sstw_feasibility.rc0_fresh4_effectiveness_cpu import (
    CONDITION_ORDER,
    GROUP_ORDER,
    SOURCE_COMMIT,
    SOURCE_TREE,
    TRANSFORM_ORDER,
    _phase_already_transformed,
    apply_frozen_frame_transform,
    evaluate_transformed_target,
    load_frozen_phase_c,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_phase_c_identity_and_zero_science_diff() -> None:
    config, phase_c = load_frozen_phase_c(ROOT)
    assert tuple(config["attempt_order"]) == GROUP_ORDER
    assert tuple(phase_c["condition_order"]) == CONDITION_ORDER
    assert tuple(item["id"] for item in phase_c["transforms"]) == TRANSFORM_ORDER
    assert phase_c["pixel_chroma_relative_amplitude"] == 6 / 255
    assert SOURCE_COMMIT == "add1e966e9bf8778a502e40532558185d5f64d3a"
    assert SOURCE_TREE == "0b48705929038f9769adf3ba3b53af1b29d1fded"


def test_frame_array_transforms_follow_exact_frozen_indices() -> None:
    _config, phase_c = load_frozen_phase_c(ROOT)
    frames = np.zeros((49, 320, 512, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(49, dtype=np.uint8)
    for item in phase_c["transforms"]:
        transformed = apply_frozen_frame_transform(frames, phase_c, item["id"])
        assert transformed[:, 0, 0, 0].tolist() == item["frame_source_indices"]


def _affine_schedule(name: str) -> np.ndarray:
    source = np.asarray(schedule_a() if name == "A" else schedule_b(), dtype=np.float64)
    matrix = np.asarray(((0.031, -0.007), (0.004, 0.027)), dtype=np.float64)
    return source @ matrix.T + np.asarray((0.002, -0.003))


def test_already_transformed_dp_closes_all_frozen_truth_paths() -> None:
    _config, phase_c = load_frozen_phase_c(ROOT)
    for template in ("A", "B"):
        states = np.asarray(schedule_a() if template == "A" else schedule_b(), dtype=np.float64)
        for item in phase_c["transforms"]:
            edited = states[np.asarray(item["symbol_source_indices"])]
            cell = _phase_already_transformed(edited, template, item["id"])
            assert cell["passed"] is True


def test_transformed_pipeline_receives_no_condition_truth() -> None:
    assert list(inspect.signature(evaluate_transformed_target).parameters) == ["decoded_frames", "transform_id"]
    runner_source = (ROOT / "experiments/run_rc0_fresh4_effectiveness_cpu.py").read_text(encoding="utf-8")
    module_source = inspect.getsource(__import__("sc_sstw_feasibility.rc0_fresh4_effectiveness_cpu", fromlist=["*"]))
    assert "generation" not in runner_source.lower()
    assert "torch" not in runner_source and "WanPipeline" not in module_source
    assert "retry_index\": 0" in module_source
    assert "/tmp" not in module_source and "TemporaryDirectory" not in module_source
