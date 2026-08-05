"""Run CPU-only AISB payload diagnostics under edit stress and mismatch.

This combines hard local edits, burst-internal deletion, deterministic
non-affine observation mismatch, and fixed message-space owner/wrong-key
scoring. It remains synthetic only: no video, GPU, fixed-FPR, or paper claim.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from sc_sstw_feasibility.aisb import (
    best_non_overlapping_sequence,
    make_redundant_templates,
    scan_burst_candidates,
    template_observation_pairs,
)
from sc_sstw_feasibility.calibration import calibrate_from_pilot_pairs, equalize_observations
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from run_aisb_channel_mismatch_probe import _apply_quadratic_mismatch
from run_aisb_payload_probe import (
    _best_message_score,
    _candidate_key,
    _payload_state,
)


def _template_by_id() -> dict[str, object]:
    return {template.template_id: template for template in make_redundant_templates()}


def _build_sequence(
    case_index: int,
    *,
    owner_key: str,
    message: str,
) -> tuple[list[tuple[float, float]], list[tuple[int, str]]]:
    template_ids = [
        "redundant_alpha",
        "redundant_beta",
        "redundant_gamma",
        "redundant_alpha",
        "redundant_beta",
        "redundant_gamma",
    ]
    templates = _template_by_id()
    states: list[tuple[float, float]] = []
    burst_plan: list[tuple[int, str]] = []
    for ordinal, template_id in enumerate(template_ids):
        filler_count = 1 + ((case_index + 2 * ordinal) % 5)
        for _ in range(filler_count):
            states.append(_payload_state(owner_key, message, len(states)))
        start = len(states)
        states.extend(templates[template_id].points)
        burst_plan.append((start, template_id))
    for _ in range(5):
        states.append(_payload_state(owner_key, message, len(states)))
    return states, burst_plan


def _states_with_public_bursts(
    key: str,
    message: str,
    length: int,
    burst_plan: list[tuple[int, str]],
) -> list[tuple[float, float]]:
    templates = _template_by_id()
    states = [_payload_state(key, message, index) for index in range(length)]
    for start, template_id in burst_plan:
        for offset, point in enumerate(templates[template_id].points):
            states[start + offset] = point
    return states


def _apply_stress_edits[T](
    values: list[T],
    burst_plan: list[tuple[int, str]],
    *,
    case_index: int,
) -> tuple[list[T], list[int], list[tuple[int, str, int]]]:
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

    truth: list[tuple[int, str, int]] = []
    for start, template_id in burst_plan:
        missing = missing_by_start[start]
        present_offsets = [offset for offset in range(9) if offset != missing]
        if all(start + offset in source_to_first_observed for offset in present_offsets):
            truth.append((source_to_first_observed[start + present_offsets[0]], template_id, missing))
    return edited, source_indices, truth


def _stress_mismatch_payload_case(
    case_index: int,
    *,
    message_space_size: int,
    wrong_key_count: int,
    gamma: float,
    noise_std: float,
) -> dict[str, object]:
    owner_key = f"owner_stress_mismatch_{message_space_size}_{case_index}"
    message_space = [f"message_{index}" for index in range(message_space_size)]
    true_message = message_space[case_index % len(message_space)]
    wrong_keys = [
        f"wrong_stress_mismatch_{message_space_size}_{case_index}_{index}"
        for index in range(wrong_key_count)
    ]
    states, burst_plan = _build_sequence(case_index, owner_key=owner_key, message=true_message)
    observations = generate_observations(
        states,
        make_random_channel(121000 + 100 * message_space_size + case_index, relation_count=16, noise_std=noise_std),
        seed=122000 + 100 * message_space_size + case_index,
    )
    observations = _apply_quadratic_mismatch(observations, states, gamma=gamma)
    edited, source_indices, truth = _apply_stress_edits(observations, burst_plan, case_index=case_index)
    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=0.006)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    truth_set = set(truth)
    if not accepted:
        return {
            "case_index": case_index,
            "message_space_size": message_space_size,
            "wrong_key_count": wrong_key_count,
            "gamma": gamma,
            "truth_count": len(truth_set),
            "accepted_count": 0,
            "alignment_exact": False,
            "owner_message_recovered": False,
            "score_margin": None,
            "pass": False,
        }

    templates = _template_by_id()
    pilot_pairs = []
    for candidate in accepted:
        pilot_pairs.extend(template_observation_pairs(candidate, edited, templates[candidate.template_id]))
    calibration = calibrate_from_pilot_pairs(pilot_pairs)
    equalized = equalize_observations(edited, calibration)
    state_reconstruction_mse = sum(
        (estimated[0] - states[source_index][0]) ** 2
        + (estimated[1] - states[source_index][1]) ** 2
        for estimated, source_index in zip(equalized, source_indices, strict=True)
    ) / len(equalized)
    owner_best_message, owner_best_score = _best_message_score(
        equalized,
        key=owner_key,
        message_space=message_space,
        sequence_length=len(states),
        burst_plan=burst_plan,
    )
    wrong_results = [
        _best_message_score(
            equalized,
            key=wrong_key,
            message_space=message_space,
            sequence_length=len(states),
            burst_plan=burst_plan,
        )
        for wrong_key in wrong_keys
    ]
    best_wrong_message, best_wrong_score = max(wrong_results, key=lambda item: item[1])
    score_margin = owner_best_score - best_wrong_score
    return {
        "case_index": case_index,
        "message_space_size": message_space_size,
        "wrong_key_count": wrong_key_count,
        "gamma": gamma,
        "noise_std": noise_std,
        "true_message": true_message,
        "owner_best_message": owner_best_message,
        "best_wrong_message": best_wrong_message,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "alignment_exact": accepted_set == truth_set,
        "owner_message_recovered": owner_best_message == true_message,
        "owner_best_score": owner_best_score,
        "best_wrong_score": best_wrong_score,
        "score_margin": score_margin,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "state_reconstruction_mse": state_reconstruction_mse,
        "pass": (
            accepted_set == truth_set
            and owner_best_message == true_message
            and score_margin > 0.02
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    margins = [
        float(case["score_margin"])
        for case in cases
        if case["score_margin"] is not None
    ]
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "alignment_exact_count": sum(1 for case in cases if case["alignment_exact"]),
        "owner_message_recovery_count": sum(1 for case in cases if case["owner_message_recovered"]),
        "score_margin_mean": _mean(margins),
        "score_margin_min": min(margins) if margins else None,
        "false_negative_count": sum(1 for case in cases if not case["alignment_exact"]),
        "cases": cases,
    }


def main() -> None:
    tiers = {
        "gamma_0.5_messages_16_wrong_24": [
            _stress_mismatch_payload_case(index, message_space_size=16, wrong_key_count=24, gamma=0.5, noise_std=0.012)
            for index in range(6)
        ],
        "gamma_0.5_messages_32_wrong_24": [
            _stress_mismatch_payload_case(index, message_space_size=32, wrong_key_count=24, gamma=0.5, noise_std=0.012)
            for index in range(6)
        ],
        "gamma_0.8_messages_16_wrong_24": [
            _stress_mismatch_payload_case(index, message_space_size=16, wrong_key_count=24, gamma=0.8, noise_std=0.012)
            for index in range(6)
        ],
    }
    summaries = {name: _summarize(cases) for name, cases in tiers.items()}
    report = {
        "status": "aisb_stress_mismatch_payload_synthetic_only_no_video_no_gpu_no_claim",
        "stress_mismatch_contract": {
            "edit_model": "crop + non-burst deletions/repeats + one missing point in every burst",
            "owner_and_wrong_keys_share_same_message_space": True,
            "fixed_fpr_claim": False,
        },
        "tiers": summaries,
        "synthetic_construction_pass": (
            summaries["gamma_0.5_messages_16_wrong_24"]["pass_count"] == 6
            and summaries["gamma_0.5_messages_32_wrong_24"]["pass_count"] == 6
        ),
        "gamma_0.8_is_margin_diagnostic_not_required_pass": True,
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
