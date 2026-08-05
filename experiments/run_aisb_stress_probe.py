"""Run CPU-only AISB stress diagnostics under harder synthetic edits.

This probe stays inside the synthetic affine-channel sandbox. It does not use
video observations, GPU, Wan, fixed-FPR calibration, or paper-claim evidence.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.aisb import (
    BurstCandidate,
    best_non_overlapping_sequence,
    make_redundant_templates,
    scan_burst_candidates,
    template_observation_pairs,
)
from sc_sstw_feasibility.calibration import calibrate_from_pilot_pairs, equalize_observations
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from sc_sstw_feasibility.sync import dynamic_time_sync


def _unit_state(key: str, index: int) -> tuple[float, float]:
    digest = hashlib.sha256(f"{key}:{index}".encode("utf-8")).digest()
    angle = 2.0 * math.pi * (int.from_bytes(digest[:8], "big") / float(1 << 64))
    return (math.cos(angle), math.sin(angle))


def _template_by_id() -> dict[str, object]:
    return {template.template_id: template for template in make_redundant_templates()}


def _candidate_key(candidate: BurstCandidate) -> tuple[int, str, int | None]:
    return (candidate.start_index, candidate.template_id, candidate.missing_template_index)


def _states_with_bursts(key: str, length: int, burst_plan: list[tuple[int, str]]) -> list[tuple[float, float]]:
    templates = _template_by_id()
    states = [_unit_state(key, index) for index in range(length)]
    for start, template_id in burst_plan:
        for offset, point in enumerate(templates[template_id].points):
            states[start + offset] = point
    return states


def _build_stress_sequence(case_index: int) -> tuple[str, list[tuple[float, float]], list[tuple[int, str]]]:
    """Build variable-spacing public bursts interleaved with secret states."""

    owner_key = f"owner_stress_{case_index}"
    templates = _template_by_id()
    template_ids = [
        "redundant_alpha",
        "redundant_beta",
        "redundant_gamma",
        "redundant_alpha",
        "redundant_beta",
        "redundant_gamma",
    ]
    states: list[tuple[float, float]] = []
    burst_plan: list[tuple[int, str]] = []
    for ordinal, template_id in enumerate(template_ids):
        filler_count = 1 + ((case_index + 2 * ordinal) % 5)
        for _ in range(filler_count):
            states.append(_unit_state(owner_key, len(states)))
        start = len(states)
        states.extend(templates[template_id].points)
        burst_plan.append((start, template_id))
    for _ in range(4):
        states.append(_unit_state(owner_key, len(states)))
    return owner_key, states, burst_plan


def _apply_stress_edits[T](
    values: list[T],
    burst_plan: list[tuple[int, str]],
    *,
    case_index: int,
) -> tuple[list[T], list[int], list[tuple[int, str, int]]]:
    """Apply crop, non-burst deletions/repeats, and one deletion in every burst."""

    crop_start = case_index % 3
    crop_end = len(values) - (case_index % 4)
    missing_by_start = {
        start: (case_index + 3 * ordinal) % 9
        for ordinal, (start, _) in enumerate(burst_plan)
    }
    protected_sources = set()
    for start, _ in burst_plan:
        for offset in range(9):
            if offset != missing_by_start[start]:
                protected_sources.add(start + offset)

    edited: list[T] = []
    source_indices: list[int] = []
    source_to_first_observed: dict[int, int] = {}
    for source_index, value in enumerate(values[crop_start:crop_end], start=crop_start):
        if any(source_index == start + missing for start, missing in missing_by_start.items()):
            continue
        if source_index not in protected_sources and (source_index * 7 + case_index) % 10 in {0, 1}:
            continue
        source_to_first_observed.setdefault(source_index, len(edited))
        edited.append(value)
        source_indices.append(source_index)
        if source_index not in protected_sources and (source_index + case_index) % 17 == 0:
            edited.append(value)
            source_indices.append(source_index)

    edited_truth: list[tuple[int, str, int]] = []
    for start, template_id in burst_plan:
        missing_template_index = missing_by_start[start]
        present_offsets = [offset for offset in range(9) if offset != missing_template_index]
        if all(start + offset in source_to_first_observed for offset in present_offsets):
            edited_truth.append(
                (
                    source_to_first_observed[start + present_offsets[0]],
                    template_id,
                    missing_template_index,
                )
            )
    return edited, source_indices, edited_truth


def _calibrate_from_candidates(observations: list[list[float]], accepted: list[BurstCandidate]):
    pilot_pairs = []
    templates = _template_by_id()
    for candidate in accepted:
        pilot_pairs.extend(template_observation_pairs(candidate, observations, templates[candidate.template_id]))
    return calibrate_from_pilot_pairs(pilot_pairs)


def _stress_case(case_index: int, *, noise_std: float) -> dict[str, object]:
    owner_key, states, burst_plan = _build_stress_sequence(case_index)
    channel = make_random_channel(33000 + case_index, relation_count=14, noise_std=noise_std)
    observations = generate_observations(states, channel, seed=34000 + case_index)
    edited, source_indices, edited_truth = _apply_stress_edits(
        observations,
        burst_plan,
        case_index=case_index,
    )
    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=0.006)
    truth_set = set(edited_truth)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}

    calibration = _calibrate_from_candidates(edited, accepted)
    equalized = equalize_observations(edited, calibration)
    state_reconstruction_mse = sum(
        (estimated[0] - states[source_index][0]) ** 2
        + (estimated[1] - states[source_index][1]) ** 2
        for estimated, source_index in zip(equalized, source_indices, strict=True)
    ) / len(equalized)
    owner_score = dynamic_time_sync(equalized, _states_with_bursts(owner_key, len(states), burst_plan)).score
    wrong_scores = [
        dynamic_time_sync(equalized, _states_with_bursts(f"wrong_stress_{case_index}_{index}", len(states), burst_plan)).score
        for index in range(10)
    ]
    score_margin = owner_score - max(wrong_scores)
    true_positive = len(truth_set & accepted_set)
    false_positive = len(accepted_set - truth_set)
    false_negative = len(truth_set - accepted_set)
    return {
        "case_index": case_index,
        "noise_std": noise_std,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "alignment_accuracy": true_positive / max(1, len(truth_set)),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "owner_score": owner_score,
        "best_wrong_score": max(wrong_scores),
        "score_margin": score_margin,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "calibration_condition_number": calibration.condition_number,
        "state_reconstruction_mse": state_reconstruction_mse,
        "best_residual": min(candidate.residual for candidate in candidates),
        "pass": (
            accepted_set == truth_set
            and score_margin > 0.02
            and calibration.pilot_reconstruction_mse < 0.01
            and state_reconstruction_mse < 0.02
        ),
    }


def _random_non_burst_case(case_index: int, *, noise_std: float) -> dict[str, object]:
    rng = random.Random(36000 + case_index)
    states = [(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(88)]
    channel = make_random_channel(37000 + case_index, relation_count=14, noise_std=noise_std)
    observations = generate_observations(states, channel, seed=38000 + case_index)
    edited = [
        observation
        for index, observation in enumerate(observations[case_index % 4 : len(observations) - (case_index % 5)])
        if (index + case_index) % 9 not in {0, 1}
    ]
    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=0.006)
    return {
        "case_index": case_index,
        "noise_std": noise_std,
        "accepted_count": len(accepted),
        "best_residual": min(candidate.residual for candidate in candidates),
        "pass": len(accepted) == 0,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize_stress_cases(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "alignment_accuracy_mean": _mean([float(case["alignment_accuracy"]) for case in cases]),
        "false_positive": sum(int(case["false_positive"]) for case in cases),
        "false_negative": sum(int(case["false_negative"]) for case in cases),
        "score_margin_mean": _mean([float(case["score_margin"]) for case in cases]),
        "state_reconstruction_mse_mean": _mean([float(case["state_reconstruction_mse"]) for case in cases]),
        "best_residual_max": max(float(case["best_residual"]) for case in cases),
        "cases": cases,
    }


def _summarize_false_cases(cases: list[dict[str, object]]) -> dict[str, object]:
    residuals = sorted(float(case["best_residual"]) for case in cases)
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "accepted_count_total": sum(int(case["accepted_count"]) for case in cases),
        "best_residual_min": residuals[0],
        "best_residual_median": residuals[len(residuals) // 2],
        "best_residual_max": residuals[-1],
        "cases": cases,
    }


def main() -> None:
    stress_cases = [_stress_case(index, noise_std=0.016) for index in range(12)]
    false_cases = [_random_non_burst_case(index, noise_std=0.016) for index in range(64)]
    noise_margin_cases = {
        "noise_0.012": [_stress_case(index, noise_std=0.012) for index in range(6)],
        "noise_0.016": [_stress_case(index, noise_std=0.016) for index in range(6)],
        "noise_0.020": [_stress_case(index, noise_std=0.020) for index in range(6)],
    }
    noise_margin_summary = {
        key: {
            "case_count": len(cases),
            "pass_count": sum(1 for case in cases if case["pass"]),
            "false_negative": sum(int(case["false_negative"]) for case in cases),
            "score_margin_mean": _mean([float(case["score_margin"]) for case in cases]),
            "diagnostic_pass": all(case["pass"] for case in cases),
        }
        for key, cases in noise_margin_cases.items()
    }
    report = {
        "status": "aisb_stress_synthetic_only_no_video_no_gpu_no_claim",
        "stress_edit_contract": {
            "redundant_burst_length": 9,
            "crop": "deterministic start/end crop",
            "inside_burst_deletion": "one arbitrary template point deleted from every fully retained burst",
            "outside_burst_edits": "deterministic non-burst deletions and occasional repeats",
            "residual_threshold": 0.006,
        },
        "stress_cases": _summarize_stress_cases(stress_cases),
        "random_non_burst_cases": _summarize_false_cases(false_cases),
        "noise_margin_diagnostic": {
            "diagnostic_kind": "stress_noise_margin_not_fixed_fpr",
            "summaries": noise_margin_summary,
            "interpretation": (
                "noise_0.016 is the current passing stress tier; noise_0.020 is reported as a margin diagnostic "
                "and is not used to tune the threshold."
            ),
            "fixed_fpr_claim": False,
        },
        "synthetic_construction_pass": (
            all(case["pass"] for case in stress_cases)
            and all(case["pass"] for case in false_cases)
            and noise_margin_summary["noise_0.012"]["diagnostic_pass"]
            and noise_margin_summary["noise_0.016"]["diagnostic_pass"]
        ),
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
