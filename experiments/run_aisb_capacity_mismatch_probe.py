"""Run CPU-only AISB capacity diagnostics under non-affine mismatch.

This combines the payload-capacity probe with the deterministic quadratic
observation-space perturbation used by the channel-mismatch diagnostic. It
remains synthetic only: no video, GPU, fixed-FPR, or paper claim.
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
    _apply_payload_edits,
    _best_message_score,
    _build_payload_sequence,
    _candidate_key,
    _template_by_id,
)


def _capacity_mismatch_case(
    case_index: int,
    *,
    message_space_size: int,
    wrong_key_count: int,
    gamma: float,
    noise_std: float,
) -> dict[str, object]:
    owner_key = f"owner_capacity_mismatch_{message_space_size}_{case_index}"
    message_space = [f"message_{index}" for index in range(message_space_size)]
    true_message = message_space[case_index % len(message_space)]
    wrong_keys = [
        f"wrong_capacity_mismatch_{message_space_size}_{case_index}_{index}"
        for index in range(wrong_key_count)
    ]
    states, burst_plan = _build_payload_sequence(case_index, owner_key=owner_key, message=true_message)
    observations = generate_observations(
        states,
        make_random_channel(111000 + 100 * message_space_size + case_index, relation_count=16, noise_std=noise_std),
        seed=112000 + 100 * message_space_size + case_index,
    )
    observations = _apply_quadratic_mismatch(observations, states, gamma=gamma)
    edited, source_indices, truth = _apply_payload_edits(observations, burst_plan, case_index=case_index)
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
            "noise_std": noise_std,
            "accepted_count": 0,
            "truth_count": len(truth_set),
            "alignment_exact": False,
            "owner_message_recovered": False,
            "score_margin": None,
            "best_residual": min(candidate.residual for candidate in candidates),
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
        "accepted_count": len(accepted_set),
        "truth_count": len(truth_set),
        "alignment_exact": accepted_set == truth_set,
        "owner_message_recovered": owner_best_message == true_message,
        "owner_best_score": owner_best_score,
        "best_wrong_score": best_wrong_score,
        "score_margin": score_margin,
        "best_residual": min(candidate.residual for candidate in candidates),
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
        "gamma_0.5_messages_8_wrong_12": [
            _capacity_mismatch_case(index, message_space_size=8, wrong_key_count=12, gamma=0.5, noise_std=0.012)
            for index in range(6)
        ],
        "gamma_0.5_messages_16_wrong_24": [
            _capacity_mismatch_case(index, message_space_size=16, wrong_key_count=24, gamma=0.5, noise_std=0.012)
            for index in range(6)
        ],
        "gamma_0.5_messages_32_wrong_24": [
            _capacity_mismatch_case(index, message_space_size=32, wrong_key_count=24, gamma=0.5, noise_std=0.012)
            for index in range(6)
        ],
        "gamma_1.0_messages_16_wrong_24": [
            _capacity_mismatch_case(index, message_space_size=16, wrong_key_count=24, gamma=1.0, noise_std=0.012)
            for index in range(6)
        ],
    }
    summaries = {name: _summarize(cases) for name, cases in tiers.items()}
    report = {
        "status": "aisb_capacity_mismatch_synthetic_only_no_video_no_gpu_no_claim",
        "capacity_mismatch_contract": {
            "owner_and_wrong_keys_share_same_message_space": True,
            "perturbation": "deterministic quadratic observation-space residual after affine channel",
            "fixed_fpr_claim": False,
        },
        "tiers": summaries,
        "synthetic_construction_pass": (
            summaries["gamma_0.5_messages_8_wrong_12"]["pass_count"] == 6
            and summaries["gamma_0.5_messages_16_wrong_24"]["pass_count"] == 6
            and summaries["gamma_0.5_messages_32_wrong_24"]["pass_count"] == 6
        ),
        "gamma_1.0_is_margin_diagnostic_not_required_pass": True,
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
