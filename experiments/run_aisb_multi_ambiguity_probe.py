"""Run CPU-only AISB multi-ambiguity-set diagnostics.

This probe creates two targeted shifted-window ambiguity clusters in one longer
sequence. Owner and wrong keys search the same public alignment hypotheses and
the same message set. It remains synthetic only: no video, GPU, fixed-FPR, or
paper claim.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from sc_sstw_feasibility.aisb import make_redundant_templates, scan_burst_candidates
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from run_aisb_ambiguity_probe import (
    _best_key_message_alignment_score,
    _candidate_ambiguity_sequences,
)
from run_aisb_channel_mismatch_probe import _apply_quadratic_mismatch
from run_aisb_long_sequence_probe import _build_long_sequence


def _multi_ambiguity_case(
    case_index: int,
    *,
    burst_count: int,
    message_space_size: int,
    wrong_key_count: int,
    gamma: float,
    noise_std: float,
) -> dict[str, object]:
    owner_key = f"owner_multi_ambiguity_{burst_count}_{message_space_size}_{case_index}"
    message_space = [f"message_{index}" for index in range(message_space_size)]
    true_message = message_space[case_index % len(message_space)]
    wrong_keys = [
        f"wrong_multi_ambiguity_{burst_count}_{message_space_size}_{case_index}_{index}"
        for index in range(wrong_key_count)
    ]
    states, burst_plan = _build_long_sequence(
        case_index,
        owner_key=owner_key,
        message=true_message,
        burst_count=burst_count,
    )
    templates = {template.template_id: template for template in make_redundant_templates()}
    ambiguous_ordinals = [min(3, len(burst_plan) - 1), min(7, len(burst_plan) - 1)]
    for ordinal in sorted(set(ambiguous_ordinals)):
        start, template_id = burst_plan[ordinal]
        if start > 0:
            states[start - 1] = templates[template_id].points[0]

    observations = generate_observations(
        states,
        make_random_channel(161000 + 100 * message_space_size + case_index, relation_count=16, noise_std=noise_std),
        seed=162000 + 100 * message_space_size + case_index,
    )
    if gamma != 0.0:
        observations = _apply_quadratic_mismatch(observations, states, gamma=gamma)

    missing_by_start = {start: 0 for start, _ in burst_plan}
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
    truth_set = set(truth)
    ambiguity_contains_truth = any(
        {
            (candidate.start_index, candidate.template_id, candidate.missing_template_index)
            for candidate in sequence
        } == truth_set
        for sequence in ambiguity_sequences
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
    return {
        "case_index": case_index,
        "burst_count": burst_count,
        "message_space_size": message_space_size,
        "wrong_key_count": wrong_key_count,
        "gamma": gamma,
        "true_message": true_message,
        "owner_best_message": owner_message,
        "best_wrong_message": wrong_message,
        "owner_best_sequence_index": owner_sequence_index,
        "best_wrong_sequence_index": wrong_sequence_index,
        "ambiguity_sequence_count": len(ambiguity_sequences),
        "ambiguity_contains_truth": ambiguity_contains_truth,
        "owner_best_score": owner_score,
        "best_wrong_score": wrong_score,
        "score_margin": score_margin,
        "pass": (
            len(ambiguity_sequences) > 1
            and ambiguity_contains_truth
            and owner_message == true_message
            and score_margin > 0.02
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "truth_sequence_in_ambiguity_count": sum(1 for case in cases if case["ambiguity_contains_truth"]),
        "owner_message_recovery_count": sum(1 for case in cases if case["owner_best_message"] == case["true_message"]),
        "ambiguity_sequence_count_min": min(int(case["ambiguity_sequence_count"]) for case in cases),
        "ambiguity_sequence_count_max": max(int(case["ambiguity_sequence_count"]) for case in cases),
        "multi_cluster_like_count": sum(1 for case in cases if int(case["ambiguity_sequence_count"]) >= 4),
        "score_margin_mean": _mean([float(case["score_margin"]) for case in cases]),
        "score_margin_min": min(float(case["score_margin"]) for case in cases),
        "cases": cases,
    }


def main() -> None:
    tiers = {
        "bursts_10_gamma_0.0_messages_8_wrong_8": [
            _multi_ambiguity_case(index, burst_count=10, message_space_size=8, wrong_key_count=8, gamma=0.0, noise_std=0.012)
            for index in range(3)
        ],
        "bursts_10_gamma_0.5_messages_8_wrong_8": [
            _multi_ambiguity_case(index, burst_count=10, message_space_size=8, wrong_key_count=8, gamma=0.5, noise_std=0.012)
            for index in range(3)
        ],
    }
    summaries = {name: _summarize(cases) for name, cases in tiers.items()}
    report = {
        "status": "aisb_multi_ambiguity_synthetic_only_no_video_no_gpu_no_claim",
        "multi_ambiguity_contract": {
            "targeted_shifted_window_cluster_count": 2,
            "owner_and_wrong_keys_share_same_alignment_set": True,
            "owner_and_wrong_keys_share_same_message_space": True,
            "fixed_fpr_claim": False,
        },
        "tiers": summaries,
        "synthetic_construction_pass": all(
            summary["pass_count"] == summary["case_count"]
            for summary in summaries.values()
        ),
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
