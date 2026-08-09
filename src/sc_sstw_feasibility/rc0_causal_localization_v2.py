"""Pure-formula preflight for the RC0 causal-localization V2 protocol.

This module does not decode video, run generation, load manifests, or evaluate
an experiment package.  It only makes the preregistered arithmetic and state
priority executable on synthetic in-memory arrays.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .learned_observation_l1_v2 import TEMPORAL_POINTS


PROTOCOL_ID = "sc_sstw_rc0_causal_localization_v2"
CONFIG_SCHEMA = "sc_sstw_rc0_causal_localization_config_v2"
PLAN_SCHEMA = "sc_sstw_rc0_matched_quartets_plan_v2"
EVIDENCE_LABEL = "synthetic_formula_preflight_only_non_evidence"

CONDITION_ORDER = ("OFF_R1", "OFF_R2", "A", "B")
GROUP_ORDER = ("orbital_glass", "articulated_paper")
ATTEMPT_ORDER = tuple(
    (group_id, condition)
    for group_id in GROUP_ORDER
    for condition in CONDITION_ORDER
)
START_INDICES = tuple(range(8))
TEMPLATES = ("A", "B")

STATUS_P_FAIL = "P_FAIL_CURRENT_CARRIER_STOP"
STATUS_P_PASS_R_FAIL = "P_PASS_R_FAIL_READOUT_SWITCH_REQUIRED"
STATUS_P_PASS_R_PASS = "P_PASS_R_PASS_PAIRED_RELATION_PRESENT"
STATUS_INVALID = "INVALID_EXPERIMENT"
STATUS_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
ALLOWED_STATES = (
    STATUS_P_FAIL,
    STATUS_P_PASS_R_FAIL,
    STATUS_P_PASS_R_PASS,
    STATUS_INVALID,
    STATUS_INSUFFICIENT,
)

PIXEL_EPSILON = 1e-12
PIXEL_ABSOLUTE_MIN = 2.0 / 255.0
PIXEL_NOISE_MAX = 1.0 / 255.0
PIXEL_NOISE_MULTIPLIER = 3.0
FEATURE_NORM_EPSILON = 1e-12
FEATURE_NOISE_MULTIPLIER = 3.0
RELATION_DENOMINATOR_EPSILON = 1e-9
RELATION_MAX_RESIDUAL = 0.25
CROSS_RESIDUAL_EXPECTED = 0.761311945935564
CROSS_RESIDUAL_TOLERANCE = 1e-9
VIDEO_SHAPE = (49, 320, 512, 3)

FROZEN_CONFIG: dict[str, Any] = {
    "schema": CONFIG_SCHEMA,
    "protocol_id": PROTOCOL_ID,
    "evidence_class": "protocol_design_and_synthetic_formula_preflight_only",
    "question_order": ["H_P_saved_mp4_pixel_bridge", "H_R_paired_public_relation"],
    "state_machine": [
        "integrity_and_evidence_sufficiency",
        "level_p_all_four_cells",
        "level_r_only_after_level_p_all_pass",
        "terminal_state",
    ],
    "condition_order": list(CONDITION_ORDER),
    "attempt_budget": 8,
    "start_indices": list(START_INDICES),
    "schedule": {
        "A_source_module": "sc_sstw_feasibility.learned_observation_l1_v2",
        "A_source_symbol": "TEMPORAL_POINTS",
        "B_derivation": "schedule_A_with_indices_4_and_5_swapped",
        "B_swapped_indices": [4, 5],
        "cross_residual_expected": CROSS_RESIDUAL_EXPECTED,
        "cross_residual_absolute_tolerance": CROSS_RESIDUAL_TOLERANCE,
    },
    "carrier": {
        "kind": "dit_internal_self_attention_output_residual",
        "module_path": "transformer.blocks[29].attn1",
        "block_index": 29,
        "target_relative_rms": 0.03,
        "target_relative_rms_absolute_tolerance": 0.00005,
        "apply_to": "8_steps_each_cond_then_uncond_16_calls",
    },
    "model": {
        "id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
    },
    "generation": {
        "scheduler": "WanPipeline_frozen_default_for_revision",
        "sampler": "WanPipeline_frozen_default_for_revision",
        "inference_steps": 8,
        "guidance_scale": 5.0,
        "height": 320,
        "width": 512,
        "frame_count": 49,
        "fps": 8,
        "dtype": "torch.bfloat16",
        "negative_prompt": "text, watermark, logo, camera motion, cuts, multiple scenes, flicker",
    },
    "encoding": {
        "call": "diffusers.utils.export_to_video",
        "container": "mp4",
        "codec": "h264",
        "pixel_format": "yuv420p",
        "quality": 5.0,
        "bitrate": None,
        "macro_block_size": 16,
        "fallback_permitted": False,
    },
    "decode": {
        "source": "saved_mp4_only",
        "decoder": "ffmpeg_rgb24_reference_path",
        "frame_order": "decoded_presentation_order_indices_0_through_48",
        "color_space": "ffmpeg_rgb24_srgb_nonlinear",
        "integer_range": [0, 255],
        "channel_order": "RGB",
        "resize_policy": "forbidden",
        "alignment_policy": "index_identity_only_no_search",
        "required_geometry": list(VIDEO_SHAPE),
    },
    "level_p": {
        "metric": "normalized_decoded_rgb_rms",
        "off_reference": "per_pixel_elementwise_mean_OFF_R1_OFF_R2",
        "off_floor": "half_normalized_rgb_rms_OFF_R1_minus_OFF_R2",
        "per_frame_formula": "sqrt_mean_H_W_RGB_squared_difference",
        "video_aggregation": "sqrt_mean_49_frames_of_squared_per_frame_rms",
        "epsilon": PIXEL_EPSILON,
        "absolute_effect_min": PIXEL_ABSOLUTE_MIN,
        "off_repeat_floor_max": PIXEL_NOISE_MAX,
        "relative_floor_multiplier": PIXEL_NOISE_MULTIPLIER,
    },
    "level_r": {
        "feature_shape": [13, 30],
        "off_reference": "elementwise_mean_OFF_R1_OFF_R2",
        "readout": "fixed_affine_relation_projection_no_training",
        "feature_norm_epsilon": FEATURE_NORM_EPSILON,
        "feature_noise_multiplier": FEATURE_NOISE_MULTIPLIER,
        "relation_denominator_epsilon": RELATION_DENOMINATOR_EPSILON,
        "maximum_relation_residual": RELATION_MAX_RESIDUAL,
        "templates": list(TEMPLATES),
        "start_indices": list(START_INDICES),
    },
    "states": {
        "p_fail": STATUS_P_FAIL,
        "p_pass_r_fail": STATUS_P_PASS_R_FAIL,
        "p_pass_r_pass": STATUS_P_PASS_R_PASS,
        "invalid": STATUS_INVALID,
        "insufficient": STATUS_INSUFFICIENT,
    },
    "state_priority": [STATUS_INVALID, STATUS_INSUFFICIENT, STATUS_P_FAIL, STATUS_P_PASS_R_FAIL, STATUS_P_PASS_R_PASS],
    "source_guards": {
        "baseline_commit": "51b4d4d1e5c0a52139b3f8a61748decff6d52931",
        "baseline_tree": "60426085872a2a3dbb82d2ec53962b1d3adb9dda",
        "rc1_config_raw_sha256": "8accea693798e2dd2ad4451ea14df71651116b89e3d8707cfe344886a57bcf15",
        "rc1_plan_raw_sha256": "d5396107f00e68eba8f972a08996608421f02a4c205339a34b3616636e41a8c7",
        "rc1_protocol_raw_sha256": "ccac64d53c1a9b87bfdab8316e963ed5bdfe415e5d95fdc553e3ae8f09438b2a",
        "g0_config_raw_sha256": "4584ed9d016ddf98c9a01408622b284b6e87724f33f1452b7cc7412ee1b88007",
        "extractor_source_commit": "fe8bc36461fdf40db917a3772a30ce6969a6c3a8",
        "extractor_source_tree": "90e50f107685c768600686239c922242e660d20a",
        "extractor_source_path": "src/sc_sstw_feasibility/learned_observation.py",
        "extractor_symbol": "extract_feature_matrix",
        "extractor_git_blob": "6288d954a1bdaded5fd2f92ed78b463bc11a6a18",
        "extractor_raw_sha256": "9c7fd37995d49344c2200a4855eaf6a547ea27336fdfe4fc652c62ac327b9866",
    },
    "formal_result": False,
    "stage_progression_allowed": False,
}

FROZEN_PLAN: dict[str, Any] = {
    "schema": PLAN_SCHEMA,
    "protocol_id": PROTOCOL_ID,
    "plan_id": "rc0_two_heterogeneous_same_latent_quartets_v2",
    "group_order": list(GROUP_ORDER),
    "condition_order": list(CONDITION_ORDER),
    "groups": [
        {
            "group_id": "orbital_glass",
            "content_grammar": "multi_object_counter_rotation",
            "prompt": "locked overhead camera, three small colored glass beads counter-rotating around fixed triangular table marks, continuous periodic motion, matte charcoal surface, stable soft lighting, no text, no cuts",
            "prompt_sha256": "9ce9b1b78f8cf26d8ec83417caf1fa84803dc7c619da43132d94b78a058d0b98",
            "seed": 52001,
            "initial_latent_policy": "one_actual_tensor_created_once_then_four_clones_before_condition_execution",
            "conditions": list(CONDITION_ORDER),
        },
        {
            "group_id": "articulated_paper",
            "content_grammar": "articulated_surface_deformation",
            "prompt": "locked oblique camera, a striped paper accordion repeatedly unfolds and refolds in place while its cast shadow changes shape, pale wooden surface, stable studio lighting, no text, no cuts",
            "prompt_sha256": "0b81e777d223a9502e0cb7bec519844500869e08e6c006c58e993fc93e06f9ee",
            "seed": 52002,
            "initial_latent_policy": "one_actual_tensor_created_once_then_four_clones_before_condition_execution",
            "conditions": list(CONDITION_ORDER),
        },
    ],
    "attempts": [
        {"attempt_index": index, "group_id": group_id, "condition": condition}
        for index, (group_id, condition) in enumerate(ATTEMPT_ORDER, start=1)
    ],
    "matched_fields": [
        "prompt",
        "prompt_sha256",
        "seed",
        "actual_initial_latent_sha256",
        "actual_initial_latent_shape",
        "actual_initial_latent_dtype",
        "model_id",
        "model_revision",
        "scheduler_class",
        "scheduler_config_sha256",
        "sampler",
        "inference_steps",
        "guidance_scale",
        "height",
        "width",
        "frame_count",
        "fps",
        "dtype",
        "negative_prompt",
        "encoder_call",
        "container",
        "codec",
        "pixel_format",
        "quality",
        "bitrate",
        "macro_block_size",
        "runtime_identity",
        "execution_path",
    ],
    "only_allowed_difference": "condition_label_and_corresponding_frozen_carrier_schedule",
    "off_repeat_semantics": "OFF_R1_and_OFF_R2_are_two_real_attempts_of_the_exact_same_OFF_condition",
    "attempt_budget": 8,
    "retry_policy": "no_retry_no_replacement_no_additional_attempts",
    "stop_on_attempt_failure": True,
    "failure_semantics": "started_attempt_consumes_budget_then_stop_and_report_INVALID_or_INSUFFICIENT",
    "formal_data_paths": [],
    "formal_result": False,
    "stage_progression_allowed": False,
}


class ProtocolViolation(ValueError):
    """A fail-closed protocol or formula input violation."""


def _require_exact_json(actual: Any, expected: Any, path: str) -> None:
    """Require exact JSON shape, order, scalar type, and frozen value."""

    if type(actual) is not type(expected):
        raise ProtocolViolation(f"{path} JSON type changed")
    if type(expected) is dict:
        if tuple(actual) != tuple(expected):
            raise ProtocolViolation(f"{path} keys or key order changed")
        for key in expected:
            _require_exact_json(actual[key], expected[key], f"{path}.{key}")
        return
    if type(expected) is list:
        if len(actual) != len(expected):
            raise ProtocolViolation(f"{path} list length changed")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _require_exact_json(actual_item, expected_item, f"{path}[{index}]")
        return
    if actual != expected:
        raise ProtocolViolation(f"{path} frozen value changed")


def schedule_a() -> tuple[tuple[float, float], ...]:
    """Read schedule A from the existing frozen burst-alpha source."""

    return tuple((float(x), float(y)) for x, y in TEMPORAL_POINTS)


def schedule_b() -> tuple[tuple[float, float], ...]:
    """Derive B by the sole authorized A-index exchange."""

    points = list(schedule_a())
    points[4], points[5] = points[5], points[4]
    return tuple(points)


def affine_relation_residual(
    observation_window: Sequence[Sequence[float]],
    template_points: Sequence[Sequence[float]],
) -> float:
    """Return the fixed six-point affine relation residual in any feature dimension."""

    observation = np.asarray(observation_window, dtype=np.float64)
    template = np.asarray(template_points, dtype=np.float64)
    if observation.ndim != 2 or observation.shape[0] != 6 or observation.shape[1] < 1:
        raise ProtocolViolation("observation window must be finite 6xd")
    if template.shape != (6, 2) or not np.isfinite(observation).all() or not np.isfinite(template).all():
        raise ProtocolViolation("template must be finite 6x2 and observation must be finite")
    augmented_anchors = np.column_stack((template[:3], np.ones(3)))
    try:
        weights = np.linalg.solve(
            augmented_anchors.T,
            np.column_stack((template[3:], np.ones(3))).T,
        ).T
    except np.linalg.LinAlgError as exc:
        raise ProtocolViolation("template anchors are degenerate") from exc
    predicted = weights @ observation[:3]
    centered = observation - observation.mean(axis=0)
    return float(
        np.sum((observation[3:] - predicted) ** 2)
        / (np.sum(centered**2) + RELATION_DENOMINATOR_EPSILON)
    )


def schedule_preflight() -> dict[str, Any]:
    """Verify the frozen A/B geometry without claiming saved-MP4 evidence."""

    a = schedule_a()
    b = schedule_b()
    if len(a) != 13 or len(b) != 13 or a[:3] != b[:3] or a[6:] != b[6:]:
        raise ProtocolViolation("A/B schedule structure changed")
    differing = tuple(index for index, pair in enumerate(zip(a, b, strict=True)) if pair[0] != pair[1])
    if differing != (4, 5) or b[4] != a[5] or b[5] != a[4]:
        raise ProtocolViolation("B must only exchange A indices 4 and 5")
    residual_ab = affine_relation_residual(a[:6], b[:6])
    residual_ba = affine_relation_residual(b[:6], a[:6])
    if not math.isclose(residual_ab, CROSS_RESIDUAL_EXPECTED, abs_tol=CROSS_RESIDUAL_TOLERANCE, rel_tol=0.0):
        raise ProtocolViolation("A-by-B analytic residual changed")
    if not math.isclose(residual_ba, CROSS_RESIDUAL_EXPECTED, abs_tol=CROSS_RESIDUAL_TOLERANCE, rel_tol=0.0):
        raise ProtocolViolation("B-by-A analytic residual changed")
    wrong_windows: dict[str, dict[str, list[float]]] = {}
    for observation_name, observation in (("A", a), ("B", b)):
        wrong_windows[observation_name] = {}
        for template_name, template in (("A", a), ("B", b)):
            values = [affine_relation_residual(observation[start : start + 6], template[:6]) for start in START_INDICES]
            wrong_windows[observation_name][template_name] = values
            for start, value in enumerate(values):
                expected_accept = observation_name == template_name and start == 0
                if expected_accept != (value <= RELATION_MAX_RESIDUAL):
                    raise ProtocolViolation("analytic template/window separation changed")
    return {
        "evidence_label": EVIDENCE_LABEL,
        "A_observation_by_B_template": residual_ab,
        "B_observation_by_A_template": residual_ba,
        "wrong_window_residuals": wrong_windows,
        "passed": True,
    }


def _normalized_video(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != VIDEO_SHAPE or not np.isfinite(array).all():
        raise ProtocolViolation("formula video must be a finite 49x320x512x3 array")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ProtocolViolation("formula video values must be normalized RGB in [0,1]")
    return array


def level_p_metrics(off_r1: Any, off_r2: Any, carrier: Any) -> dict[str, Any]:
    """Compute the symmetric preregistered normalized-RGB RMS quantities."""

    off1 = _normalized_video(off_r1)
    off2 = _normalized_video(off_r2)
    treated = _normalized_video(carrier)
    if off1.shape != off2.shape or off1.shape != treated.shape:
        raise ProtocolViolation("Level P arrays must have identical decoded geometry")
    effect_squared: list[float] = []
    noise_squared: list[float] = []
    for frame_index in range(VIDEO_SHAPE[0]):
        off_ref_frame = (off1[frame_index] + off2[frame_index]) / 2.0
        effect_squared.append(float(np.mean((treated[frame_index] - off_ref_frame) ** 2)))
        noise_squared.append(float(0.25 * np.mean((off1[frame_index] - off2[frame_index]) ** 2)))
    effect_per_frame = np.sqrt(np.asarray(effect_squared, dtype=np.float64))
    noise_per_frame = np.sqrt(np.asarray(noise_squared, dtype=np.float64))
    effect = float(math.sqrt(sum(effect_squared) / VIDEO_SHAPE[0]))
    noise = float(math.sqrt(sum(noise_squared) / VIDEO_SHAPE[0]))
    checks = {
        "off_repeat_noise_at_most_one_code_value": noise <= PIXEL_NOISE_MAX + PIXEL_EPSILON,
        "absolute_effect_at_least_two_code_values": effect + PIXEL_EPSILON >= PIXEL_ABSOLUTE_MIN,
        "effect_at_least_three_times_off_floor": effect + PIXEL_EPSILON >= PIXEL_NOISE_MULTIPLIER * noise,
    }
    effect_cell_pass = checks["absolute_effect_at_least_two_code_values"] and checks["effect_at_least_three_times_off_floor"]
    return {
        "metric": "normalized_decoded_rgb_rms",
        "effect_per_frame": effect_per_frame.tolist(),
        "off_repeat_floor_per_frame": noise_per_frame.tolist(),
        "effect": effect,
        "off_repeat_floor": noise,
        "checks": checks,
        "off_repeat_control_valid": checks["off_repeat_noise_at_most_one_code_value"],
        "cell_pass": effect_cell_pass,
    }


def _feature_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (13, 30) or not np.isfinite(array).all():
        raise ProtocolViolation("Level R features must be finite 13x30 matrices")
    return array


def level_r_metrics(off_r1: Any, off_r2: Any, carrier: Any, own_template: str) -> dict[str, Any]:
    """Compute the fixed no-training relation screen for one carrier condition."""

    if own_template not in TEMPLATES:
        raise ProtocolViolation("own template must be A or B")
    off1 = _feature_matrix(off_r1)
    off2 = _feature_matrix(off_r2)
    treated = _feature_matrix(carrier)
    off_ref = (off1 + off2) / 2.0
    paired = treated - off_ref
    paired_centered = paired - paired.mean(axis=0, keepdims=True)
    off_delta = (off1 - off2) / 2.0
    off_delta_centered = off_delta - off_delta.mean(axis=0, keepdims=True)
    signal_norm = float(np.linalg.norm(paired_centered))
    noise_norm = float(np.linalg.norm(off_delta_centered))
    signal_present = signal_norm > FEATURE_NORM_EPSILON
    noise_separated = signal_norm + FEATURE_NORM_EPSILON >= FEATURE_NOISE_MULTIPLIER * noise_norm
    normalized = paired_centered / signal_norm if signal_present else np.zeros_like(paired_centered)
    schedules = {"A": schedule_a(), "B": schedule_b()}
    residuals = {
        template: [
            affine_relation_residual(normalized[start : start + 6], schedules[template][:6])
            if signal_present
            else math.inf
            for start in START_INDICES
        ]
        for template in TEMPLATES
    }
    own_start0 = residuals[own_template][0] <= RELATION_MAX_RESIDUAL
    own_wrong_reject = all(value > RELATION_MAX_RESIDUAL for value in residuals[own_template][1:])
    cross_template = "B" if own_template == "A" else "A"
    cross_reject = all(value > RELATION_MAX_RESIDUAL for value in residuals[cross_template])
    checks = {
        "feature_signal_non_degenerate": signal_present,
        "feature_signal_at_least_three_times_off_floor": noise_separated,
        "own_template_start0_accept": own_start0,
        "own_template_wrong_windows_reject": own_wrong_reject,
        "cross_template_all_windows_reject": cross_reject,
    }
    return {
        "readout": "fixed_affine_relation_projection_no_training",
        "signal_norm": signal_norm,
        "off_repeat_floor_norm": noise_norm,
        "residuals": residuals,
        "checks": checks,
        "cell_pass": all(checks.values()),
    }


def terminal_state(
    *,
    integrity_valid: bool,
    evidence_complete: bool,
    off_repeat_control_valid: bool,
    p_cells: Mapping[str, bool],
    r_was_executed: bool,
    r_cells: Mapping[str, bool] | None = None,
) -> str:
    """Apply the preregistered priority without averaging or majority vote."""

    expected_cells = {f"{group}:{condition}" for group in GROUP_ORDER for condition in TEMPLATES}
    flags = (integrity_valid, evidence_complete, off_repeat_control_valid, r_was_executed)
    if any(type(flag) is not bool for flag in flags):
        return STATUS_INVALID
    if type(p_cells) is not dict or set(p_cells) != expected_cells or any(type(value) is not bool for value in p_cells.values()):
        return STATUS_INVALID
    if r_was_executed:
        if type(r_cells) is not dict or set(r_cells) != expected_cells or any(type(value) is not bool for value in r_cells.values()):
            return STATUS_INVALID
    elif r_cells is not None:
        return STATUS_INVALID
    p_all_pass = all(p_cells[cell] for cell in expected_cells)
    if not p_all_pass and r_was_executed:
        return STATUS_INVALID
    if not integrity_valid:
        return STATUS_INVALID
    if not evidence_complete or not off_repeat_control_valid:
        return STATUS_INSUFFICIENT
    if not p_all_pass:
        return STATUS_P_FAIL
    if not r_was_executed:
        return STATUS_INSUFFICIENT
    assert r_cells is not None
    if not all(r_cells[cell] for cell in expected_cells):
        return STATUS_P_PASS_R_FAIL
    return STATUS_P_PASS_R_PASS


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject any config key, order, JSON type, or frozen-value drift."""

    _require_exact_json(config, FROZEN_CONFIG, "config")


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Reject any plan key, order, JSON type, or frozen-value drift."""

    _require_exact_json(plan, FROZEN_PLAN, "plan")
