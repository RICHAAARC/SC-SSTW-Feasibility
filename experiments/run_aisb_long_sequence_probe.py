"""Run CPU-only AISB long-sequence payload diagnostics.

This probe stresses whether the AISB-frozen state-trajectory scorer still
separates owner/wrong keys when the sequence has more public bursts, longer
secret-state spans, larger message search, and more wrong keys. It remains
synthetic only: no video, GPU, fixed-FPR, or paper claim.
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


def _build_long_sequence(
    case_index: int,
    *,
    owner_key: str,
    message: str,
    burst_count: int,
    filler_multiplier: int = 1,
) -> tuple[list[tuple[float, float]], list[tuple[int, str]]]:
    """Build a longer synthetic sequence with predeclared public AISB bursts."""

    if filler_multiplier <= 0:
        raise ValueError("filler_multiplier must be positive")
    template_cycle = ["redundant_alpha", "redundant_beta", "redundant_gamma"]
    templates = _template_by_id()
    states: list[tuple[float, float]] = []
    burst_plan: list[tuple[int, str]] = []
    for ordinal in range(burst_count):
        template_id = template_cycle[(case_index + ordinal) % len(template_cycle)]
        filler_count = (4 + ((case_index + 3 * ordinal) % 8)) * filler_multiplier
        for _ in range(filler_count):
            states.append(_payload_state(owner_key, message, len(states)))
        start = len(states)
        states.extend(templates[template_id].points)
        burst_plan.append((start, template_id))
    for _ in range(12):
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


def _apply_long_edits[T](
    values: list[T],
    burst_plan: list[tuple[int, str]],
    *,
    case_index: int,
) -> tuple[list[T], list[int], list[tuple[int, str, int]]]:
    """Apply crop, non-burst deletion/repeat, and one deletion per retained burst."""

    crop_start = case_index % 4
    crop_end = len(values) - (case_index % 5)
    missing_by_start = {
        start: (case_index + 2 * ordinal) % 9
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
        if source_index not in protected_sources and (source_index * 11 + case_index) % 17 in {0, 1, 2}:
            continue
        source_to_first_observed.setdefault(source_index, len(edited))
        edited.append(value)
        source_indices.append(source_index)
        if source_index not in protected_sources and (source_index + 3 * case_index) % 23 == 0:
            edited.append(value)
            source_indices.append(source_index)

    truth: list[tuple[int, str, int]] = []
    for start, template_id in burst_plan:
        missing = missing_by_start[start]
        present_offsets = [offset for offset in range(9) if offset != missing]
        if all(start + offset in source_to_first_observed for offset in present_offsets):
            truth.append((source_to_first_observed[start + present_offsets[0]], template_id, missing))
    return edited, source_indices, truth


def _long_sequence_case(
    case_index: int,
    *,
    burst_count: int,
    message_space_size: int,
    wrong_key_count: int,
    gamma: float,
    noise_std: float,
) -> dict[str, object]:
    owner_key = f"owner_long_{burst_count}_{message_space_size}_{case_index}"
    message_space = [f"message_{index}" for index in range(message_space_size)]
    true_message = message_space[case_index % len(message_space)]
    wrong_keys = [
        f"wrong_long_{burst_count}_{message_space_size}_{case_index}_{index}"
        for index in range(wrong_key_count)
    ]
    states, burst_plan = _build_long_sequence(
        case_index,
        owner_key=owner_key,
        message=true_message,
        burst_count=burst_count,
    )
    observations = generate_observations(
        states,
        make_random_channel(131000 + 100 * message_space_size + case_index, relation_count=16, noise_std=noise_std),
        seed=132000 + 100 * message_space_size + case_index,
    )
    if gamma != 0.0:
        observations = _apply_quadratic_mismatch(observations, states, gamma=gamma)
    edited, source_indices, truth = _apply_long_edits(observations, burst_plan, case_index=case_index)
    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=0.006)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    truth_set = set(truth)
    if not accepted:
        return {
            "case_index": case_index,
            "burst_count": burst_count,
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
        "burst_count": burst_count,
        "message_space_size": message_space_size,
        "wrong_key_count": wrong_key_count,
        "gamma": gamma,
        "noise_std": noise_std,
        "sequence_length": len(states),
        "edited_length": len(edited),
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
        "bursts_10_gamma_0.0_messages_32_wrong_16": [
            _long_sequence_case(index, burst_count=10, message_space_size=32, wrong_key_count=16, gamma=0.0, noise_std=0.014)
            for index in range(2)
        ],
        "bursts_10_gamma_0.5_messages_32_wrong_16": [
            _long_sequence_case(index, burst_count=10, message_space_size=32, wrong_key_count=16, gamma=0.5, noise_std=0.012)
            for index in range(2)
        ],
        "bursts_12_gamma_0.5_messages_32_wrong_16": [
            _long_sequence_case(index, burst_count=12, message_space_size=32, wrong_key_count=16, gamma=0.5, noise_std=0.012)
            for index in range(2)
        ],
        "bursts_12_gamma_0.8_messages_16_wrong_16": [
            _long_sequence_case(index, burst_count=12, message_space_size=16, wrong_key_count=16, gamma=0.8, noise_std=0.012)
            for index in range(2)
        ],
    }
    summaries = {name: _summarize(cases) for name, cases in tiers.items()}
    report = {
        "status": "aisb_long_sequence_synthetic_only_no_video_no_gpu_no_claim",
        "long_sequence_contract": {
            "owner_and_wrong_keys_share_same_message_space": True,
            "edit_model": "crop + non-burst deletions/repeats + one missing point in every burst",
            "fixed_fpr_claim": False,
        },
        "tiers": summaries,
        "synthetic_construction_pass": (
            summaries["bursts_10_gamma_0.0_messages_32_wrong_16"]["pass_count"] == 2
            and summaries["bursts_10_gamma_0.5_messages_32_wrong_16"]["pass_count"] == 2
            and summaries["bursts_12_gamma_0.5_messages_32_wrong_16"]["pass_count"] == 2
        ),
        "gamma_0.8_is_margin_diagnostic_not_required_pass": True,
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
