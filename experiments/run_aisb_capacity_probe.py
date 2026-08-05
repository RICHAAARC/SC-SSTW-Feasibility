"""Run CPU-only AISB payload capacity diagnostics.

This probe varies the fixed candidate message-set size after public AISB
alignment is frozen. Owner and wrong keys search the same message space. The
probe remains synthetic only: no video, GPU, fixed-FPR, or paper claim.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from sc_sstw_feasibility.aisb import (
    make_redundant_templates,
    scan_burst_candidates,
    best_non_overlapping_sequence,
    template_observation_pairs,
)
from sc_sstw_feasibility.calibration import calibrate_from_pilot_pairs, equalize_observations
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from run_aisb_payload_probe import (
    _apply_payload_edits,
    _best_message_score,
    _build_payload_sequence,
    _candidate_key,
    _template_by_id,
)


def _capacity_case(
    case_index: int,
    *,
    message_space_size: int,
    wrong_key_count: int,
    noise_std: float,
) -> dict[str, object]:
    owner_key = f"owner_capacity_{message_space_size}_{case_index}"
    message_space = [f"message_{index}" for index in range(message_space_size)]
    true_message = message_space[case_index % len(message_space)]
    wrong_keys = [
        f"wrong_capacity_{message_space_size}_{case_index}_{index}"
        for index in range(wrong_key_count)
    ]
    states, burst_plan = _build_payload_sequence(case_index, owner_key=owner_key, message=true_message)
    observations = generate_observations(
        states,
        make_random_channel(101000 + 100 * message_space_size + case_index, relation_count=16, noise_std=noise_std),
        seed=102000 + 100 * message_space_size + case_index,
    )
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
    score_margin = owner_best_score - best_wrong_score
    return {
        "case_index": case_index,
        "message_space_size": message_space_size,
        "wrong_key_count": wrong_key_count,
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
        "state_reconstruction_mse": state_reconstruction_mse,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "pass": (
            accepted_set == truth_set
            and owner_best_message == true_message
            and score_margin > 0.02
            and state_reconstruction_mse < 0.02
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "alignment_exact_count": sum(1 for case in cases if case["alignment_exact"]),
        "owner_message_recovery_count": sum(1 for case in cases if case["owner_message_recovered"]),
        "score_margin_mean": _mean([float(case["score_margin"]) for case in cases]),
        "score_margin_min": min(float(case["score_margin"]) for case in cases),
        "state_reconstruction_mse_mean": _mean([float(case["state_reconstruction_mse"]) for case in cases]),
        "cases": cases,
    }


def main() -> None:
    tiers = {
        "messages_8_wrong_12": [
            _capacity_case(index, message_space_size=8, wrong_key_count=12, noise_std=0.016)
            for index in range(8)
        ],
        "messages_16_wrong_24": [
            _capacity_case(index, message_space_size=16, wrong_key_count=24, noise_std=0.016)
            for index in range(8)
        ],
        "messages_32_wrong_24": [
            _capacity_case(index, message_space_size=32, wrong_key_count=24, noise_std=0.016)
            for index in range(8)
        ],
    }
    summaries = {name: _summarize(cases) for name, cases in tiers.items()}
    report = {
        "status": "aisb_capacity_synthetic_only_no_video_no_gpu_no_claim",
        "capacity_contract": {
            "owner_and_wrong_keys_share_same_message_space": True,
            "wrong_key_search_is_not_freer_than_owner_search": True,
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
