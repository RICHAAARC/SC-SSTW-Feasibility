"""Run CPU-only AISB + secret payload scoring diagnostics.

The probe tests whether, after public AISB alignment is frozen, a fixed
candidate message set can be scored fairly for owner and wrong keys. It remains
synthetic affine-channel evidence only: no video, GPU, fixed-FPR, or paper claim.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
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


def _bit(domain: str, *parts: object) -> int:
    payload = ":".join(str(part) for part in (domain, *parts))
    return 1 if hashlib.sha256(payload.encode("utf-8")).digest()[0] & 1 else -1


def _phase(domain: str, *parts: object) -> float:
    payload = ":".join(str(part) for part in (domain, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return 2.0 * math.pi * (int.from_bytes(digest[:8], "big") / float(1 << 64))


def _payload_state(key: str, message: str, index: int) -> tuple[float, float]:
    """Return a key/message-conditioned 2D state.

    This is a toy PRC/state generator for feasibility only. It is not a final
    code design and does not imply detection calibration.
    """

    angle = (
        _phase("payload.init", key, message)
        + 0.39 * index
        + 0.58 * _bit("payload.key", key, index)
        + 0.83 * _bit("payload.message", message, index)
        + 0.31 * _bit("payload.key_message", key, message, index)
    )
    return (math.cos(angle), math.sin(angle))


def _template_by_id() -> dict[str, object]:
    return {template.template_id: template for template in make_redundant_templates()}


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


def _build_payload_sequence(
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
    ]
    states: list[tuple[float, float]] = []
    burst_plan: list[tuple[int, str]] = []
    templates = _template_by_id()
    for ordinal, template_id in enumerate(template_ids):
        filler_count = 3 + ((case_index + ordinal) % 6)
        for _ in range(filler_count):
            states.append(_payload_state(owner_key, message, len(states)))
        start = len(states)
        states.extend(templates[template_id].points)
        burst_plan.append((start, template_id))
    for _ in range(6):
        states.append(_payload_state(owner_key, message, len(states)))
    return states, burst_plan


def _apply_payload_edits[T](
    values: list[T],
    burst_plan: list[tuple[int, str]],
    *,
    case_index: int,
) -> tuple[list[T], list[int], list[tuple[int, str, int]]]:
    crop_start = case_index % 3
    crop_end = len(values) - (case_index % 4)
    missing_by_start = {
        start: (case_index + 2 * ordinal) % 9
        for ordinal, (start, _) in enumerate(burst_plan)
    }
    protected = set()
    for start, _ in burst_plan:
        for offset in range(9):
            if offset != missing_by_start[start]:
                protected.add(start + offset)

    edited: list[T] = []
    source_indices: list[int] = []
    source_to_first_observed: dict[int, int] = {}
    for source_index, value in enumerate(values[crop_start:crop_end], start=crop_start):
        if any(source_index == start + missing for start, missing in missing_by_start.items()):
            continue
        if source_index not in protected and (source_index * 5 + case_index) % 12 in {0, 1}:
            continue
        source_to_first_observed.setdefault(source_index, len(edited))
        edited.append(value)
        source_indices.append(source_index)
        if source_index not in protected and (source_index + case_index) % 19 == 0:
            edited.append(value)
            source_indices.append(source_index)

    truth: list[tuple[int, str, int]] = []
    for start, template_id in burst_plan:
        missing = missing_by_start[start]
        present_offsets = [offset for offset in range(9) if offset != missing]
        if all(start + offset in source_to_first_observed for offset in present_offsets):
            truth.append((source_to_first_observed[start + present_offsets[0]], template_id, missing))
    return edited, source_indices, truth


def _candidate_key(candidate: BurstCandidate) -> tuple[int, str, int | None]:
    return (candidate.start_index, candidate.template_id, candidate.missing_template_index)


def _best_message_score(
    equalized: list[tuple[float, float]],
    *,
    key: str,
    message_space: list[str],
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> tuple[str, float]:
    scores = [
        (
            message,
            dynamic_time_sync(
                equalized,
                _states_with_public_bursts(key, message, sequence_length, burst_plan),
            ).score,
        )
        for message in message_space
    ]
    return max(scores, key=lambda item: item[1])


def _payload_case(case_index: int, *, noise_std: float) -> dict[str, object]:
    owner_key = f"owner_payload_{case_index}"
    message_space = [f"message_{index}" for index in range(8)]
    true_message = message_space[case_index % len(message_space)]
    wrong_keys = [f"wrong_payload_{case_index}_{index}" for index in range(12)]
    states, burst_plan = _build_payload_sequence(case_index, owner_key=owner_key, message=true_message)
    channel = make_random_channel(41000 + case_index, relation_count=16, noise_std=noise_std)
    observations = generate_observations(states, channel, seed=42000 + case_index)
    edited, source_indices, truth = _apply_payload_edits(observations, burst_plan, case_index=case_index)

    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=0.006)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    truth_set = set(truth)
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
    true_positive = len(truth_set & accepted_set)
    false_positive = len(accepted_set - truth_set)
    false_negative = len(truth_set - accepted_set)
    score_margin = owner_best_score - best_wrong_score
    return {
        "case_index": case_index,
        "noise_std": noise_std,
        "message_space_size": len(message_space),
        "wrong_key_count": len(wrong_keys),
        "true_message": true_message,
        "owner_best_message": owner_best_message,
        "best_wrong_message": best_wrong_message,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "alignment_accuracy": true_positive / max(1, len(truth_set)),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "owner_best_score": owner_best_score,
        "best_wrong_score": best_wrong_score,
        "score_margin": score_margin,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "calibration_condition_number": calibration.condition_number,
        "state_reconstruction_mse": state_reconstruction_mse,
        "pass": (
            accepted_set == truth_set
            and owner_best_message == true_message
            and score_margin > 0.02
            and state_reconstruction_mse < 0.02
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize_cases(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "alignment_accuracy_mean": _mean([float(case["alignment_accuracy"]) for case in cases]),
        "false_positive": sum(int(case["false_positive"]) for case in cases),
        "false_negative": sum(int(case["false_negative"]) for case in cases),
        "owner_message_recovery_count": sum(
            1 for case in cases if case["owner_best_message"] == case["true_message"]
        ),
        "score_margin_mean": _mean([float(case["score_margin"]) for case in cases]),
        "score_margin_min": min(float(case["score_margin"]) for case in cases),
        "state_reconstruction_mse_mean": _mean([float(case["state_reconstruction_mse"]) for case in cases]),
        "cases": cases,
    }


def main() -> None:
    payload_cases = [_payload_case(index, noise_std=0.016) for index in range(12)]
    noise_margin_cases = {
        "noise_0.012": [_payload_case(index, noise_std=0.012) for index in range(6)],
        "noise_0.016": [_payload_case(index, noise_std=0.016) for index in range(6)],
        "noise_0.020": [_payload_case(index, noise_std=0.020) for index in range(6)],
    }
    noise_margin_summary = {
        key: {
            "case_count": len(cases),
            "pass_count": sum(1 for case in cases if case["pass"]),
            "false_negative": sum(int(case["false_negative"]) for case in cases),
            "owner_message_recovery_count": sum(
                1 for case in cases if case["owner_best_message"] == case["true_message"]
            ),
            "score_margin_mean": _mean([float(case["score_margin"]) for case in cases]),
            "diagnostic_pass": all(case["pass"] for case in cases),
        }
        for key, cases in noise_margin_cases.items()
    }
    report = {
        "status": "aisb_payload_synthetic_only_no_video_no_gpu_no_claim",
        "payload_scoring_contract": {
            "message_space_size": 8,
            "wrong_key_count": 12,
            "candidate_message_search_space": "identical for owner and every wrong key",
            "public_alignment": "AISB accepted before affine calibration",
            "fixed_fpr_claim": False,
        },
        "payload_cases": _summarize_cases(payload_cases),
        "noise_margin_diagnostic": {
            "diagnostic_kind": "payload_noise_margin_not_fixed_fpr",
            "summaries": noise_margin_summary,
            "interpretation": (
                "noise_0.016 is the current passing payload tier; noise_0.020 is reported as a margin diagnostic "
                "and is not used to tune the threshold."
            ),
            "fixed_fpr_claim": False,
        },
        "synthetic_construction_pass": (
            all(case["pass"] for case in payload_cases)
            and noise_margin_summary["noise_0.012"]["diagnostic_pass"]
            and noise_margin_summary["noise_0.016"]["diagnostic_pass"]
        ),
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
