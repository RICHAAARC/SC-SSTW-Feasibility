"""Run CPU-only AISB diagnostics under synthetic non-affine channel mismatch.

This is a robustness diagnostic for the affine-channel assumption. It uses a
known synthetic quadratic perturbation after the affine relation channel and
does not constitute real video, fixed-FPR, or paper-claim evidence.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import random
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
from run_aisb_payload_probe import (
    _apply_payload_edits,
    _best_message_score,
    _build_payload_sequence,
    _candidate_key,
    _template_by_id,
)


def _apply_quadratic_mismatch(
    observations: list[list[float]],
    states: list[tuple[float, float]],
    *,
    gamma: float,
) -> list[list[float]]:
    """Add a deterministic non-affine perturbation in the observation space."""

    distorted: list[list[float]] = []
    for observation, (x_value, y_value) in zip(observations, states, strict=True):
        quadratic = 0.7 * x_value * x_value - 0.4 * y_value * y_value + 0.3 * x_value * y_value
        distorted.append([
            value + gamma * quadratic * math.sin(0.37 * (index + 1))
            for index, value in enumerate(observation)
        ])
    return distorted


def _mismatch_case(case_index: int, *, gamma: float, noise_std: float = 0.012) -> dict[str, object]:
    owner_key = f"owner_mismatch_{case_index}"
    message_space = [f"message_{index}" for index in range(8)]
    true_message = message_space[case_index % len(message_space)]
    wrong_keys = [f"wrong_mismatch_{case_index}_{index}" for index in range(12)]
    states, burst_plan = _build_payload_sequence(case_index, owner_key=owner_key, message=true_message)
    observations = generate_observations(
        states,
        make_random_channel(51000 + case_index, relation_count=16, noise_std=noise_std),
        seed=52000 + case_index,
    )
    observations = _apply_quadratic_mismatch(observations, states, gamma=gamma)
    edited, _, truth = _apply_payload_edits(observations, burst_plan, case_index=case_index)
    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=0.006)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    truth_set = set(truth)
    if not accepted:
        return {
            "case_index": case_index,
            "gamma": gamma,
            "truth_count": len(truth_set),
            "accepted_count": 0,
            "alignment_accuracy": 0.0,
            "false_positive": 0,
            "false_negative": len(truth_set),
            "best_residual": min(candidate.residual for candidate in candidates),
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
    owner_best_message, owner_best_score = _best_message_score(
        equalized,
        key=owner_key,
        message_space=message_space,
        sequence_length=len(states),
        burst_plan=burst_plan,
    )
    wrong_best = max(
        (
            _best_message_score(
                equalized,
                key=wrong_key,
                message_space=message_space,
                sequence_length=len(states),
                burst_plan=burst_plan,
            )
            for wrong_key in wrong_keys
        ),
        key=lambda item: item[1],
    )
    true_positive = len(truth_set & accepted_set)
    false_positive = len(accepted_set - truth_set)
    false_negative = len(truth_set - accepted_set)
    score_margin = owner_best_score - wrong_best[1]
    return {
        "case_index": case_index,
        "gamma": gamma,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "alignment_accuracy": true_positive / max(1, len(truth_set)),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "best_residual": min(candidate.residual for candidate in candidates),
        "owner_message_recovered": owner_best_message == true_message,
        "owner_best_score": owner_best_score,
        "best_wrong_score": wrong_best[1],
        "score_margin": score_margin,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "pass": (
            accepted_set == truth_set
            and owner_best_message == true_message
            and score_margin > 0.02
        ),
    }


def _random_non_burst_case(case_index: int, *, gamma: float, noise_std: float = 0.012) -> dict[str, object]:
    rng = random.Random(70000 + case_index)
    states = [(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(88)]
    observations = generate_observations(
        states,
        make_random_channel(71000 + case_index, relation_count=16, noise_std=noise_std),
        seed=72000 + case_index,
    )
    observations = _apply_quadratic_mismatch(observations, states, gamma=gamma)
    candidates = scan_burst_candidates(observations, make_redundant_templates(), allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=0.006)
    return {
        "case_index": case_index,
        "gamma": gamma,
        "accepted_count": len(accepted),
        "best_residual": min(candidate.residual for candidate in candidates),
        "pass": len(accepted) == 0,
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
        "owner_message_recovery_count": sum(1 for case in cases if case["owner_message_recovered"]),
        "score_margin_mean": _mean([
            float(case["score_margin"])
            for case in cases
            if case["score_margin"] is not None
        ]),
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
    }


def main() -> None:
    mismatch_cases = [_mismatch_case(index, gamma=0.5) for index in range(12)]
    false_cases = [_random_non_burst_case(index, gamma=0.5) for index in range(64)]
    gamma_margin_cases = {
        "gamma_0.0": [_mismatch_case(index, gamma=0.0) for index in range(6)],
        "gamma_0.3": [_mismatch_case(index, gamma=0.3) for index in range(6)],
        "gamma_0.5": [_mismatch_case(index, gamma=0.5) for index in range(6)],
        "gamma_1.0": [_mismatch_case(index, gamma=1.0) for index in range(6)],
    }
    gamma_margin_summary = {
        key: {
            "case_count": len(cases),
            "pass_count": sum(1 for case in cases if case["pass"]),
            "false_negative": sum(int(case["false_negative"]) for case in cases),
            "owner_message_recovery_count": sum(1 for case in cases if case["owner_message_recovered"]),
            "diagnostic_pass": all(case["pass"] for case in cases),
        }
        for key, cases in gamma_margin_cases.items()
    }
    report = {
        "status": "aisb_channel_mismatch_synthetic_only_no_video_no_gpu_no_claim",
        "mismatch_contract": {
            "perturbation": "deterministic quadratic observation-space residual after affine channel",
            "passing_gamma": 0.5,
            "residual_threshold": 0.006,
            "fixed_fpr_claim": False,
        },
        "mismatch_cases": _summarize_cases(mismatch_cases),
        "random_non_burst_cases": _summarize_false_cases(false_cases),
        "gamma_margin_diagnostic": {
            "diagnostic_kind": "quadratic_mismatch_margin_not_fixed_fpr",
            "summaries": gamma_margin_summary,
            "interpretation": "gamma_0.5 is the current passing mismatch tier; gamma_1.0 exposes false negatives.",
            "fixed_fpr_claim": False,
        },
        "synthetic_construction_pass": (
            all(case["pass"] for case in mismatch_cases)
            and all(case["pass"] for case in false_cases)
            and gamma_margin_summary["gamma_0.3"]["diagnostic_pass"]
            and gamma_margin_summary["gamma_0.5"]["diagnostic_pass"]
        ),
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
