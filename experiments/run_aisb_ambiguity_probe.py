"""Run CPU-only AISB ambiguity-set payload scoring diagnostics.

This probe tests a fallback interpretation of AISB acquisition after the
exact-redundant-anchor shifted-window ambiguity: acquisition may return a small
public ambiguity set instead of one unique alignment. Owner and wrong keys are
then scored over the same candidate-alignment set and the same fixed message
space. This remains synthetic only: no video, GPU, fixed-FPR, or paper claim.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from sc_sstw_feasibility.aisb import (
    BurstCandidate,
    make_redundant_templates,
    scan_burst_candidates,
    template_observation_pairs,
)
from sc_sstw_feasibility.calibration import calibrate_from_pilot_pairs, equalize_observations
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from sc_sstw_feasibility.sync import dynamic_time_sync
from run_aisb_payload_probe import (
    _apply_payload_edits,
    _build_payload_sequence,
    _payload_state,
    _states_with_public_bursts,
)


def _candidate_key(candidate: BurstCandidate) -> tuple[int, str, int | None]:
    return (candidate.start_index, candidate.template_id, candidate.missing_template_index)


def _candidate_ambiguity_sequences(
    candidates: list[BurstCandidate],
    *,
    residual_threshold: float,
    near_tie_ratio: float,
    per_cluster_limit: int,
    max_sequences: int,
) -> list[list[BurstCandidate]]:
    """Return public non-overlapping candidate sequences from local ambiguities."""

    eligible = sorted(
        (candidate for candidate in candidates if candidate.residual <= residual_threshold),
        key=lambda item: (item.start_index, item.start_index + item.observed_length, item.residual),
    )
    clusters: list[list[BurstCandidate]] = []
    cluster: list[BurstCandidate] = []
    cluster_end = -1
    for candidate in eligible:
        candidate_end = candidate.start_index + candidate.observed_length
        if not cluster:
            cluster = [candidate]
            cluster_end = candidate_end
            continue
        if candidate.start_index < cluster_end:
            cluster.append(candidate)
            cluster_end = max(cluster_end, candidate_end)
            continue
        clusters.append(cluster)
        cluster = [candidate]
        cluster_end = candidate_end
    if cluster:
        clusters.append(cluster)

    choices: list[list[BurstCandidate]] = []
    for cluster_items in clusters:
        min_residual = min(candidate.residual for candidate in cluster_items)
        near_tie_limit = min_residual * near_tie_ratio + 1e-12
        near_ties = [
            candidate
            for candidate in cluster_items
            if candidate.residual <= near_tie_limit
        ]
        near_ties.sort(key=lambda item: (item.start_index, item.residual, item.template_id, item.missing_template_index or -1))
        choices.append(near_ties[:per_cluster_limit])
    if not choices:
        return []

    sequences: list[list[BurstCandidate]] = []
    for option_tuple in product(*choices):
        ordered = sorted(option_tuple, key=lambda item: item.start_index)
        if all(
            left.start_index + left.observed_length <= right.start_index
            for left, right in zip(ordered, ordered[1:], strict=False)
        ):
            sequences.append(ordered)
        if len(sequences) >= max_sequences:
            break
    return sequences


def _template_by_id() -> dict[str, object]:
    return {template.template_id: template for template in make_redundant_templates()}


def _score_one_sequence(
    observations: list[list[float]],
    sequence: list[BurstCandidate],
    *,
    key: str,
    message: str,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> float:
    templates = _template_by_id()
    pilot_pairs = []
    for candidate in sequence:
        pilot_pairs.extend(template_observation_pairs(candidate, observations, templates[candidate.template_id]))
    calibration = calibrate_from_pilot_pairs(pilot_pairs)
    equalized = equalize_observations(observations, calibration)
    return dynamic_time_sync(
        equalized,
        _states_with_public_bursts(key, message, sequence_length, burst_plan),
    ).score


def _best_key_message_alignment_score(
    observations: list[list[float]],
    ambiguity_sequences: list[list[BurstCandidate]],
    *,
    key: str,
    message_space: list[str],
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> tuple[str, int, float]:
    best_message = ""
    best_sequence_index = -1
    best_score = float("-inf")
    for sequence_index, sequence in enumerate(ambiguity_sequences):
        for message in message_space:
            score = _score_one_sequence(
                observations,
                sequence,
                key=key,
                message=message,
                sequence_length=sequence_length,
                burst_plan=burst_plan,
            )
            if score > best_score:
                best_message = message
                best_sequence_index = sequence_index
                best_score = score
    return best_message, best_sequence_index, best_score


def _ambiguity_case(case_index: int, *, noise_std: float) -> dict[str, object]:
    owner_key = f"owner_ambiguity_{case_index}"
    message_space = [f"message_{index}" for index in range(8)]
    true_message = message_space[case_index % len(message_space)]
    wrong_keys = [f"wrong_ambiguity_{case_index}_{index}" for index in range(12)]
    states, burst_plan = _build_payload_sequence(case_index, owner_key=owner_key, message=true_message)
    observations = generate_observations(
        states,
        make_random_channel(81000 + case_index, relation_count=16, noise_std=noise_std),
        seed=82000 + case_index,
    )
    edited, _, truth = _apply_payload_edits(observations, burst_plan, case_index=case_index)
    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    ambiguity_sequences = _candidate_ambiguity_sequences(
        candidates,
        residual_threshold=0.006,
        near_tie_ratio=1.25,
        per_cluster_limit=3,
        max_sequences=512,
    )
    if not ambiguity_sequences:
        return {
            "case_index": case_index,
            "noise_std": noise_std,
            "truth_count": len(truth),
            "ambiguity_sequence_count": 0,
            "owner_message_recovered": False,
            "score_margin": None,
            "pass": False,
        }

    owner_message, owner_sequence_index, owner_score = _best_key_message_alignment_score(
        edited,
        ambiguity_sequences,
        key=owner_key,
        message_space=message_space,
        sequence_length=len(states),
        burst_plan=burst_plan,
    )
    wrong_results = [
        _best_key_message_alignment_score(
            edited,
            ambiguity_sequences,
            key=wrong_key,
            message_space=message_space,
            sequence_length=len(states),
            burst_plan=burst_plan,
        )
        for wrong_key in wrong_keys
    ]
    wrong_message, wrong_sequence_index, wrong_score = max(wrong_results, key=lambda item: item[2])
    score_margin = owner_score - wrong_score
    sequence_sizes = [len(sequence) for sequence in ambiguity_sequences]
    return {
        "case_index": case_index,
        "noise_std": noise_std,
        "truth_count": len(truth),
        "ambiguity_sequence_count": len(ambiguity_sequences),
        "ambiguity_sequence_size_min": min(sequence_sizes),
        "ambiguity_sequence_size_max": max(sequence_sizes),
        "message_space_size": len(message_space),
        "wrong_key_count": len(wrong_keys),
        "true_message": true_message,
        "owner_best_message": owner_message,
        "owner_best_sequence_index": owner_sequence_index,
        "best_wrong_message": wrong_message,
        "best_wrong_sequence_index": wrong_sequence_index,
        "owner_best_score": owner_score,
        "best_wrong_score": wrong_score,
        "score_margin": score_margin,
        "owner_message_recovered": owner_message == true_message,
        "pass": owner_message == true_message and score_margin > 0.02,
    }


def _targeted_shifted_window_case(case_index: int, *, noise_std: float) -> dict[str, object]:
    """Build a case with an intentional public shifted-window ambiguity."""

    owner_key = f"owner_targeted_ambiguity_{case_index}"
    message_space = [f"message_{index}" for index in range(8)]
    true_message = message_space[case_index % len(message_space)]
    wrong_keys = [f"wrong_targeted_ambiguity_{case_index}_{index}" for index in range(12)]
    templates = _template_by_id()
    burst_plan = [
        (6, "redundant_alpha"),
        (21, "redundant_beta"),
        (36, "redundant_gamma"),
        (51, "redundant_alpha"),
    ]
    states = [_payload_state(owner_key, true_message, index) for index in range(70)]
    for start, template_id in burst_plan:
        for offset, point in enumerate(templates[template_id].points):
            states[start + offset] = point

    # A public-looking pre-burst sample plus deletion of template index 0 creates
    # a real low-residual ambiguity between the true missing-first window and an
    # adjacent shifted window. This is a diagnostic construction, not a detector
    # privilege.
    states[burst_plan[2][0] - 1] = templates["redundant_gamma"].points[0]
    missing_by_start = {start: 0 for start, _ in burst_plan}
    observations = generate_observations(
        states,
        make_random_channel(95000 + case_index, relation_count=16, noise_std=noise_std),
        seed=96000 + case_index,
    )
    edited: list[list[float]] = []
    source_to_observed: dict[int, int] = {}
    for source_index, observation in enumerate(observations):
        if any(source_index == start + missing for start, missing in missing_by_start.items()):
            continue
        source_to_observed[source_index] = len(edited)
        edited.append(observation)
    truth = [
        (source_to_observed[start + 1], template_id, 0)
        for start, template_id in burst_plan
    ]
    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    ambiguity_sequences = _candidate_ambiguity_sequences(
        candidates,
        residual_threshold=0.006,
        near_tie_ratio=1.25,
        per_cluster_limit=3,
        max_sequences=512,
    )
    owner_message, owner_sequence_index, owner_score = _best_key_message_alignment_score(
        edited,
        ambiguity_sequences,
        key=owner_key,
        message_space=message_space,
        sequence_length=len(states),
        burst_plan=burst_plan,
    )
    wrong_results = [
        _best_key_message_alignment_score(
            edited,
            ambiguity_sequences,
            key=wrong_key,
            message_space=message_space,
            sequence_length=len(states),
            burst_plan=burst_plan,
        )
        for wrong_key in wrong_keys
    ]
    wrong_message, wrong_sequence_index, wrong_score = max(wrong_results, key=lambda item: item[2])
    score_margin = owner_score - wrong_score
    truth_set = set(truth)
    ambiguity_contains_truth = any(
        {_candidate_key(candidate) for candidate in sequence} == truth_set
        for sequence in ambiguity_sequences
    )
    return {
        "case_index": case_index,
        "noise_std": noise_std,
        "truth_count": len(truth),
        "ambiguity_sequence_count": len(ambiguity_sequences),
        "ambiguity_contains_truth": ambiguity_contains_truth,
        "message_space_size": len(message_space),
        "wrong_key_count": len(wrong_keys),
        "true_message": true_message,
        "owner_best_message": owner_message,
        "owner_best_sequence_index": owner_sequence_index,
        "best_wrong_message": wrong_message,
        "best_wrong_sequence_index": wrong_sequence_index,
        "owner_best_score": owner_score,
        "best_wrong_score": wrong_score,
        "score_margin": score_margin,
        "owner_message_recovered": owner_message == true_message,
        "pass": (
            len(ambiguity_sequences) > 1
            and ambiguity_contains_truth
            and owner_message == true_message
            and score_margin > 0.02
        ),
    }


def _random_non_burst_case(case_index: int, *, noise_std: float) -> dict[str, object]:
    import random

    rng = random.Random(91000 + case_index)
    states = [(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(88)]
    observations = generate_observations(
        states,
        make_random_channel(92000 + case_index, relation_count=16, noise_std=noise_std),
        seed=93000 + case_index,
    )
    candidates = scan_burst_candidates(observations, make_redundant_templates(), allow_single_deletion=True)
    ambiguity_sequences = _candidate_ambiguity_sequences(
        candidates,
        residual_threshold=0.006,
        near_tie_ratio=1.25,
        per_cluster_limit=3,
        max_sequences=512,
    )
    return {
        "case_index": case_index,
        "noise_std": noise_std,
        "ambiguity_sequence_count": len(ambiguity_sequences),
        "best_residual": min(candidate.residual for candidate in candidates),
        "pass": len(ambiguity_sequences) == 0,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def main() -> None:
    ambiguity_cases = [_ambiguity_case(index, noise_std=0.016) for index in range(12)]
    targeted_case_indices = [0, 1, 2, 3, 5, 6]
    targeted_cases = [_targeted_shifted_window_case(index, noise_std=0.012) for index in targeted_case_indices]
    false_cases = [_random_non_burst_case(index, noise_std=0.016) for index in range(64)]
    report = {
        "status": "aisb_ambiguity_set_synthetic_only_no_video_no_gpu_no_claim",
        "ambiguity_contract": {
            "residual_threshold": 0.006,
            "near_tie_ratio": 1.25,
            "per_cluster_limit": 3,
            "max_sequences": 512,
            "owner_and_wrong_keys_share_same_alignment_set": True,
            "fixed_fpr_claim": False,
        },
        "ambiguity_payload_cases": {
            "case_count": len(ambiguity_cases),
            "pass_count": sum(1 for case in ambiguity_cases if case["pass"]),
            "owner_message_recovery_count": sum(1 for case in ambiguity_cases if case["owner_message_recovered"]),
            "score_margin_mean": _mean([
                float(case["score_margin"])
                for case in ambiguity_cases
                if case["score_margin"] is not None
            ]),
            "ambiguity_sequence_count_max": max(int(case["ambiguity_sequence_count"]) for case in ambiguity_cases),
            "cases": ambiguity_cases,
        },
        "targeted_shifted_window_ambiguity_cases": {
            "case_indices": targeted_case_indices,
            "case_count": len(targeted_cases),
            "pass_count": sum(1 for case in targeted_cases if case["pass"]),
            "owner_message_recovery_count": sum(1 for case in targeted_cases if case["owner_message_recovered"]),
            "truth_sequence_in_ambiguity_count": sum(1 for case in targeted_cases if case["ambiguity_contains_truth"]),
            "score_margin_mean": _mean([float(case["score_margin"]) for case in targeted_cases]),
            "ambiguity_sequence_count_min": min(int(case["ambiguity_sequence_count"]) for case in targeted_cases),
            "ambiguity_sequence_count_max": max(int(case["ambiguity_sequence_count"]) for case in targeted_cases),
            "cases": targeted_cases,
        },
        "random_non_burst_cases": {
            "case_count": len(false_cases),
            "pass_count": sum(1 for case in false_cases if case["pass"]),
            "accepted_ambiguity_sequence_total": sum(int(case["ambiguity_sequence_count"]) for case in false_cases),
            "best_residual_min": min(float(case["best_residual"]) for case in false_cases),
            "cases": false_cases,
        },
        "synthetic_construction_pass": (
            all(case["pass"] for case in ambiguity_cases)
            and all(case["pass"] for case in targeted_cases)
            and all(case["pass"] for case in false_cases)
        ),
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
