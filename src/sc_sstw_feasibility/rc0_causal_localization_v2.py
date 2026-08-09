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


class ProtocolViolation(ValueError):
    """A fail-closed protocol or formula input violation."""


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
    if array.ndim != 4 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ProtocolViolation("formula video must be a finite TxHxWx3 array")
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
    off_ref = (off1 + off2) / 2.0
    effect_per_frame = np.sqrt(np.mean((treated - off_ref) ** 2, axis=(1, 2, 3)))
    noise_per_frame = 0.5 * np.sqrt(np.mean((off1 - off2) ** 2, axis=(1, 2, 3)))
    effect = float(np.sqrt(np.mean(effect_per_frame**2)))
    noise = float(np.sqrt(np.mean(noise_per_frame**2)))
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
    if not integrity_valid:
        return STATUS_INVALID
    if not evidence_complete or not off_repeat_control_valid:
        return STATUS_INSUFFICIENT
    if set(p_cells) != expected_cells:
        return STATUS_INVALID
    if not all(bool(p_cells[cell]) for cell in expected_cells):
        if r_was_executed:
            return STATUS_INVALID
        return STATUS_P_FAIL
    if not r_was_executed or r_cells is None:
        return STATUS_INSUFFICIENT
    if set(r_cells) != expected_cells:
        return STATUS_INVALID
    if not all(bool(r_cells[cell]) for cell in expected_cells):
        return STATUS_P_PASS_R_FAIL
    return STATUS_P_PASS_R_PASS


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject metric, threshold, schedule, budget, or state injection."""

    if config.get("schema") != CONFIG_SCHEMA or config.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolViolation("config identity changed")
    expected_schedule = {
        "A_source_module": "sc_sstw_feasibility.learned_observation_l1_v2",
        "A_source_symbol": "TEMPORAL_POINTS",
        "B_derivation": "schedule_A_with_indices_4_and_5_swapped",
        "B_swapped_indices": [4, 5],
        "cross_residual_expected": CROSS_RESIDUAL_EXPECTED,
        "cross_residual_absolute_tolerance": CROSS_RESIDUAL_TOLERANCE,
    }
    if config.get("schedule") != expected_schedule:
        raise ProtocolViolation("schedule identity or geometry changed")
    expected_carrier = {
        "kind": "dit_internal_self_attention_output_residual",
        "module_path": "transformer.blocks[29].attn1",
        "block_index": 29,
        "target_relative_rms": 0.03,
        "target_relative_rms_absolute_tolerance": 0.00005,
        "apply_to": "8_steps_each_cond_then_uncond_16_calls",
    }
    if config.get("carrier") != expected_carrier:
        raise ProtocolViolation("carrier identity changed")
    expected_model = {
        "id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
    }
    if config.get("model") != expected_model:
        raise ProtocolViolation("model identity changed")
    expected_generation = {
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
    }
    if config.get("generation") != expected_generation:
        raise ProtocolViolation("generation settings changed")
    expected_encoding = {
        "call": "diffusers.utils.export_to_video",
        "container": "mp4",
        "codec": "h264",
        "pixel_format": "yuv420p",
        "quality": 5.0,
        "bitrate": None,
        "macro_block_size": 16,
        "fallback_permitted": False,
    }
    if config.get("encoding") != expected_encoding:
        raise ProtocolViolation("encoding settings changed")
    expected_decode = {
        "source": "saved_mp4_only",
        "decoder": "ffmpeg_rgb24_reference_path",
        "frame_order": "decoded_presentation_order_indices_0_through_48",
        "color_space": "ffmpeg_rgb24_srgb_nonlinear",
        "integer_range": [0, 255],
        "channel_order": "RGB",
        "resize_policy": "forbidden",
        "alignment_policy": "index_identity_only_no_search",
        "required_geometry": [49, 320, 512, 3],
    }
    if config.get("decode") != expected_decode:
        raise ProtocolViolation("decode or alignment policy changed")
    if tuple(config.get("condition_order", ())) != CONDITION_ORDER:
        raise ProtocolViolation("condition order changed")
    if config.get("attempt_budget") != 8 or tuple(config.get("start_indices", ())) != START_INDICES:
        raise ProtocolViolation("budget or scan starts changed")
    level_p = config.get("level_p", {})
    expected_p = {
        "metric": "normalized_decoded_rgb_rms",
        "off_reference": "per_pixel_elementwise_mean_OFF_R1_OFF_R2",
        "off_floor": "half_normalized_rgb_rms_OFF_R1_minus_OFF_R2",
        "per_frame_formula": "sqrt_mean_H_W_RGB_squared_difference",
        "video_aggregation": "sqrt_mean_49_frames_of_squared_per_frame_rms",
        "epsilon": PIXEL_EPSILON,
        "absolute_effect_min": PIXEL_ABSOLUTE_MIN,
        "off_repeat_floor_max": PIXEL_NOISE_MAX,
        "relative_floor_multiplier": PIXEL_NOISE_MULTIPLIER,
    }
    if level_p != expected_p:
        raise ProtocolViolation("Level P metric or thresholds changed")
    level_r = config.get("level_r", {})
    expected_r = {
        "feature_shape": [13, 30],
        "off_reference": "elementwise_mean_OFF_R1_OFF_R2",
        "readout": "fixed_affine_relation_projection_no_training",
        "feature_norm_epsilon": FEATURE_NORM_EPSILON,
        "feature_noise_multiplier": FEATURE_NOISE_MULTIPLIER,
        "relation_denominator_epsilon": RELATION_DENOMINATOR_EPSILON,
        "maximum_relation_residual": RELATION_MAX_RESIDUAL,
        "templates": ["A", "B"],
        "start_indices": list(START_INDICES),
    }
    if level_r != expected_r:
        raise ProtocolViolation("Level R formula or thresholds changed")
    states = config.get("states", {})
    if tuple(states.values()) != ALLOWED_STATES:
        raise ProtocolViolation("state set or priority changed")
    expected_state_machine = (
        "integrity_and_evidence_sufficiency",
        "level_p_all_four_cells",
        "level_r_only_after_level_p_all_pass",
        "terminal_state",
    )
    if tuple(config.get("state_machine", ())) != expected_state_machine:
        raise ProtocolViolation("state-machine order changed")
    expected_priority = (STATUS_INVALID, STATUS_INSUFFICIENT, STATUS_P_FAIL, STATUS_P_PASS_R_FAIL, STATUS_P_PASS_R_PASS)
    if tuple(config.get("state_priority", ())) != expected_priority:
        raise ProtocolViolation("terminal-state priority changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise ProtocolViolation("evidence boundary changed")


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Reject extra groups, conditions, retries, or attempt-order drift."""

    if plan.get("schema") != PLAN_SCHEMA or plan.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolViolation("plan identity changed")
    if tuple(plan.get("group_order", ())) != GROUP_ORDER or tuple(plan.get("condition_order", ())) != CONDITION_ORDER:
        raise ProtocolViolation("declared group or condition order changed")
    groups = plan.get("groups")
    if not isinstance(groups, list) or tuple(group.get("group_id") for group in groups) != GROUP_ORDER:
        raise ProtocolViolation("group identity or order changed")
    expected_groups = {
        "orbital_glass": {
            "content_grammar": "multi_object_counter_rotation",
            "prompt": "locked overhead camera, three small colored glass beads counter-rotating around fixed triangular table marks, continuous periodic motion, matte charcoal surface, stable soft lighting, no text, no cuts",
            "prompt_sha256": "9ce9b1b78f8cf26d8ec83417caf1fa84803dc7c619da43132d94b78a058d0b98",
            "seed": 52001,
        },
        "articulated_paper": {
            "content_grammar": "articulated_surface_deformation",
            "prompt": "locked oblique camera, a striped paper accordion repeatedly unfolds and refolds in place while its cast shadow changes shape, pale wooden surface, stable studio lighting, no text, no cuts",
            "prompt_sha256": "0b81e777d223a9502e0cb7bec519844500869e08e6c006c58e993fc93e06f9ee",
            "seed": 52002,
        },
    }
    for group in groups:
        expected = expected_groups[group["group_id"]]
        if any(group.get(key) != value for key, value in expected.items()):
            raise ProtocolViolation("group content identity changed")
        if tuple(group.get("conditions", ())) != CONDITION_ORDER:
            raise ProtocolViolation("quartet conditions changed")
        if group.get("initial_latent_policy") != "one_actual_tensor_created_once_then_four_clones_before_condition_execution":
            raise ProtocolViolation("same-latent policy changed")
    attempts = plan.get("attempts")
    observed_attempts = tuple((item.get("group_id"), item.get("condition")) for item in attempts or ())
    if observed_attempts != ATTEMPT_ORDER or tuple(item.get("attempt_index") for item in attempts or ()) != tuple(range(1, 9)):
        raise ProtocolViolation("attempt order or budget changed")
    if plan.get("retry_policy") != "no_retry_no_replacement_no_additional_attempts":
        raise ProtocolViolation("retry policy changed")
    if plan.get("stop_on_attempt_failure") is not True:
        raise ProtocolViolation("failure stop rule changed")
    if plan.get("attempt_budget") != 8 or plan.get("formal_data_paths") != []:
        raise ProtocolViolation("budget or formal-data isolation changed")
    if plan.get("only_allowed_difference") != "condition_label_and_corresponding_frozen_carrier_schedule":
        raise ProtocolViolation("allowed-difference rule changed")
    if plan.get("off_repeat_semantics") != "OFF_R1_and_OFF_R2_are_two_real_attempts_of_the_exact_same_OFF_condition":
        raise ProtocolViolation("OFF repeat semantics changed")
    if plan.get("formal_result") is not False or plan.get("stage_progression_allowed") is not False:
        raise ProtocolViolation("plan evidence boundary changed")
