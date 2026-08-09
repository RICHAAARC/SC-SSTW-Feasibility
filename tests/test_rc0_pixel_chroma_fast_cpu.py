from __future__ import annotations

import json
import inspect
import math
from pathlib import Path

import numpy as np

from src.sc_sstw_feasibility.learned_observation import decode_saved_mp4
from src.sc_sstw_feasibility.rc0_causal_localization_v2 import schedule_a, schedule_b
from src.sc_sstw_feasibility.rc0_pixel_chroma_fast_cpu import (
    ATTEMPT_ORDER,
    BOUND_GENERATED_ARTIFACTS,
    CONDITION_ORDER,
    FRAME_POINT_INDICES,
    GROUP_ORDER,
    TARGET_RMS,
    VIDEO_SHAPE,
    apply_carrier,
    compare_with_independent_recompute,
    construct_unit_residual,
    encode_mp4,
    evaluate_existing_run_once,
    load_and_validate_config,
    probe_mp4,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/rc0_pixel_chroma_fast_cpu.json"


def test_frozen_config_and_exact_attempt_order() -> None:
    config = load_and_validate_config(CONFIG)
    assert config["diagnostic_class"] == "DIAGNOSTIC_ONLY"
    assert GROUP_ORDER == ("orbital_glass", "articulated_paper")
    assert CONDITION_ORDER == ("OFF_R1", "OFF_R2", "A", "B")
    assert ATTEMPT_ORDER == tuple((g, c) for g in GROUP_ORDER for c in CONDITION_ORDER)
    assert len(ATTEMPT_ORDER) == 8
    assert config["formal_result"] is False
    assert config["stage_progression_allowed"] is False


def test_exact_frame_mapping_and_frozen_schedules() -> None:
    assert len(FRAME_POINT_INDICES) == 49
    assert FRAME_POINT_INDICES[0] == 0
    for point in range(1, 13):
        assert FRAME_POINT_INDICES[1 + 4 * (point - 1) : 1 + 4 * point] == (point,) * 4
    assert sorted(set(FRAME_POINT_INDICES)) == list(range(13))
    assert len(schedule_a()) == len(schedule_b()) == 13
    assert schedule_a()[4] == schedule_b()[5]
    assert schedule_a()[5] == schedule_b()[4]


def test_basis_is_zero_sum_orthonormal_and_unit_residual() -> None:
    u = np.asarray((1.0, -1.0, 0.0)) / math.sqrt(2.0)
    v = np.asarray((1.0, 1.0, -2.0)) / math.sqrt(6.0)
    assert abs(float(u.sum())) < 1e-15
    assert abs(float(v.sum())) < 1e-15
    assert abs(float(np.dot(u, v))) < 1e-15
    assert math.isclose(float(np.linalg.norm(u)), 1.0, abs_tol=1e-15)
    assert math.isclose(float(np.linalg.norm(v)), 1.0, abs_tol=1e-15)
    for schedule in (schedule_a(), schedule_b()):
        unit, identity = construct_unit_residual(schedule)
        assert unit.shape == VIDEO_SHAPE
        assert unit.dtype == np.float32
        assert abs(float(unit.mean(dtype=np.float64))) < 1e-7
        assert math.isclose(float(np.sqrt(np.mean(np.square(unit, dtype=np.float32), dtype=np.float64))), 1.0, abs_tol=2e-6)
        assert abs(identity["unit_global_mean"]) < 1e-7
        assert math.isclose(identity["unit_global_rms"], 1.0, abs_tol=2e-6)


def test_apply_carrier_has_fixed_target_and_no_input_mutation() -> None:
    source = np.full(VIDEO_SHAPE, 128, dtype=np.uint8)
    before = source.copy()
    treated, identity = apply_carrier(source, schedule_a())
    assert np.array_equal(source, before)
    assert treated.shape == VIDEO_SHAPE and treated.dtype == np.uint8
    assert identity["preclip_target_residual_rms"] == TARGET_RMS == 6.0 / 255.0
    assert identity["clipped_element_fraction"] == 0.0
    assert math.isclose(identity["postclip_actual_residual_rms"], TARGET_RMS, abs_tol=2e-6)


def test_encoder_is_one_fixed_h264_path(tmp_path: Path) -> None:
    frames = np.full(VIDEO_SHAPE, 128, dtype=np.uint8)
    path = tmp_path / "condition" / "saved.mp4"
    encode_mp4(frames, path)
    assert probe_mp4(path) == {
        "codec_name": "h264", "pix_fmt": "yuv420p", "width": 512, "height": 320,
        "avg_frame_rate": "8/1", "nb_frames": "49",
    }
    decoded = decode_saved_mp4(path)
    assert decoded.shape == VIDEO_SHAPE and decoded.dtype == np.uint8


def test_no_tuning_or_alternate_carrier_fields() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    text = CONFIG.read_text(encoding="utf-8")
    assert config["carrier"]["target_rms_expression"] == "6/255"
    assert "options" not in text
    assert "candidate" not in text
    assert "retry_allowed" not in text
    runner = (ROOT / "experiments/run_rc0_pixel_chroma_fast_cpu.py").read_text(encoding="utf-8")
    assert runner.count("add_argument") == 1
    assert '"--output"' in runner
    assert "--strength" not in runner and "--schedule" not in runner and "--source" not in runner


def test_protocol_freezes_p_short_circuit_and_routes() -> None:
    protocol = (ROOT / "protocols/rc0_pixel_chroma_fast_cpu.md").read_text(encoding="utf-8")
    assert "Only then is the frozen 30D extractor" in protocol
    assert "CURRENT_CARRIER_NOT_FEASIBLE" in protocol
    assert "STEP3_AISB_CPU" in protocol
    assert "KEEP_CARRIER_SWITCH_TO_VAE_READOUT" in protocol
    assert "no retry" in protocol.lower()


def test_generation_path_registration_bug_is_closed() -> None:
    source = inspect.getsource(__import__(
        "src.sc_sstw_feasibility.rc0_pixel_chroma_fast_cpu",
        fromlist=["run_once"],
    ).run_once)
    assert "video_paths[group][condition] = path" in source
    assert source.index("video_paths[group][condition] = path") < source.index("evaluation = evaluate_saved_videos(video_paths)")


def test_evaluation_only_binds_all_eight_artifact_hashes() -> None:
    assert tuple(BOUND_GENERATED_ARTIFACTS) == GROUP_ORDER
    assert sum(len(group) for group in BOUND_GENERATED_ARTIFACTS.values()) == 8
    for group in GROUP_ORDER:
        assert tuple(BOUND_GENERATED_ARTIFACTS[group]) == CONDITION_ORDER
        for condition, (path, digest) in BOUND_GENERATED_ARTIFACTS[group].items():
            assert path.is_absolute()
            assert f"/{group}/{condition}/saved.mp4" in str(path)
            assert len(digest) == 64


def test_evaluation_only_has_no_encode_or_generation_path() -> None:
    source = inspect.getsource(evaluate_existing_run_once)
    assert "evaluate_saved_videos(video_paths)" in source
    assert "encode_mp4" not in source
    assert "apply_carrier" not in source
    assert "validate_source_zip" not in source
    cli = (ROOT / "experiments/evaluate_rc0_pixel_chroma_run1.py").read_text(encoding="utf-8")
    assert "argparse" not in cli
    assert "add_argument" not in cli
    assert "response = run_once(" not in cli


def test_independent_comparison_is_exact_and_detects_drift() -> None:
    independent = json.loads(Path(
        "/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-pixel-chroma-fast-cpu-9e19d31-run1/independent_recompute.json"
    ).read_text(encoding="utf-8"))
    evaluation = {
        key: independent[key]
        for key in ("step1", "step2", "route", "next_question", "p_cells", "r_executed", "r_cells")
    }
    matched = compare_with_independent_recompute(evaluation, independent)
    assert matched["all_compared_values_match"] is True
    assert matched["maximum_absolute_numeric_difference"] == 0.0
    changed = json.loads(json.dumps(evaluation))
    changed["p_cells"]["orbital_glass:A"]["effect"] += 1e-8
    rejected = compare_with_independent_recompute(changed, independent)
    assert rejected["all_compared_values_match"] is False
    assert rejected["mismatch_paths"] == ["evaluation.p_cells.orbital_glass:A.effect"]
