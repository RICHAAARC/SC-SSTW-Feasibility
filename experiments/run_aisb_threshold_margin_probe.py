"""Run CPU-only AISB residual-threshold margin diagnostics.

This diagnostic does not tune a fixed-FPR threshold. It checks whether the
current synthetic high-mismatch false negative is a narrow residual-margin issue
or a broader public-acquisition breakdown. It remains synthetic only: no video,
GPU, fixed-FPR, or paper claim.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from sc_sstw_feasibility.aisb import best_non_overlapping_sequence, make_redundant_templates, scan_burst_candidates
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from run_aisb_channel_mismatch_probe import _apply_quadratic_mismatch
from run_aisb_long_sequence_probe import _apply_long_edits, _build_long_sequence
from run_aisb_payload_probe import _candidate_key


THRESHOLDS = [0.006, 0.00625, 0.0065, 0.0075, 0.01]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-only AISB residual-threshold margin diagnostics.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Quadratic mismatch strength for the synthetic channel.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.012,
        help="Synthetic affine-channel observation noise standard deviation.",
    )
    parser.add_argument(
        "--high-case-count",
        type=int,
        default=4,
        help="Number of deterministic high-mismatch cases to run.",
    )
    parser.add_argument(
        "--random-case-count",
        type=int,
        default=64,
        help="Number of deterministic random non-burst cases to run.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(str(threshold) for threshold in THRESHOLDS),
        help="Comma-separated diagnostic residual thresholds to scan.",
    )
    args = parser.parse_args()
    if args.gamma < 0:
        raise SystemExit("--gamma must be non-negative")
    if args.noise_std < 0:
        raise SystemExit("--noise-std must be non-negative")
    if args.high_case_count <= 0:
        raise SystemExit("--high-case-count must be positive")
    if args.random_case_count <= 0:
        raise SystemExit("--random-case-count must be positive")
    thresholds = []
    for raw_threshold in args.thresholds.split(","):
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise SystemExit("--thresholds must contain comma-separated numbers") from exc
        if threshold <= 0:
            raise SystemExit("--thresholds entries must be positive")
        thresholds.append(threshold)
    if not thresholds:
        raise SystemExit("--thresholds must not be empty")
    args.thresholds = thresholds
    return args


def _high_mismatch_case(
    case_index: int,
    *,
    gamma: float,
    noise_std: float,
    thresholds: list[float] | None = None,
) -> dict[str, object]:
    if thresholds is None:
        thresholds = THRESHOLDS
    owner_key = f"owner_pruned_12_64_{case_index}"
    true_message = f"message_{case_index % 64}"
    states, burst_plan = _build_long_sequence(
        case_index,
        owner_key=owner_key,
        message=true_message,
        burst_count=12,
    )
    observations = generate_observations(
        states,
        make_random_channel(141000 + 100 * 64 + case_index, relation_count=16, noise_std=noise_std),
        seed=142000 + 100 * 64 + case_index,
    )
    observations = _apply_quadratic_mismatch(observations, states, gamma=gamma)
    edited, _, truth = _apply_long_edits(observations, burst_plan, case_index=case_index)
    truth_set = set(truth)
    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    threshold_results = []
    for threshold in thresholds:
        accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=threshold)
        accepted_set = {_candidate_key(candidate) for candidate in accepted}
        threshold_results.append(
            {
                "threshold": threshold,
                "accepted_count": len(accepted_set),
                "false_positive": len(accepted_set - truth_set),
                "false_negative": len(truth_set - accepted_set),
                "alignment_exact": accepted_set == truth_set,
            }
        )
    truth_residuals = [
        candidate.residual
        for candidate in candidates
        if _candidate_key(candidate) in truth_set
    ]
    best_false_residual = min(
        (
            candidate.residual
            for candidate in candidates
            if _candidate_key(candidate) not in truth_set
        ),
        default=None,
    )
    return {
        "case_index": case_index,
        "gamma": gamma,
        "noise_std": noise_std,
        "truth_count": len(truth_set),
        "truth_residual_max": max(truth_residuals),
        "truth_residual_min": min(truth_residuals),
        "best_false_residual": best_false_residual,
        "threshold_results": threshold_results,
    }


def _random_non_burst_case(
    case_index: int,
    *,
    gamma: float,
    noise_std: float,
    thresholds: list[float] | None = None,
) -> dict[str, object]:
    if thresholds is None:
        thresholds = THRESHOLDS
    rng = random.Random(161000 + case_index)
    states = [(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(190)]
    observations = generate_observations(
        states,
        make_random_channel(162000 + case_index, relation_count=16, noise_std=noise_std),
        seed=163000 + case_index,
    )
    observations = _apply_quadratic_mismatch(observations, states, gamma=gamma)
    candidates = scan_burst_candidates(observations, make_redundant_templates(), allow_single_deletion=True)
    threshold_results = []
    for threshold in thresholds:
        accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=threshold)
        threshold_results.append(
            {
                "threshold": threshold,
                "accepted_count": len(accepted),
                "false_positive": len(accepted),
            }
        )
    return {
        "case_index": case_index,
        "gamma": gamma,
        "noise_std": noise_std,
        "best_residual": min(candidate.residual for candidate in candidates),
        "threshold_results": threshold_results,
    }


def _summarize_high_mismatch(cases: list[dict[str, object]], *, thresholds: list[float]) -> dict[str, object]:
    by_threshold = {}
    for threshold in thresholds:
        rows = [
            result
            for case in cases
            for result in case["threshold_results"]
            if result["threshold"] == threshold
        ]
        by_threshold[str(threshold)] = {
            "alignment_exact_count": sum(1 for row in rows if row["alignment_exact"]),
            "false_positive_total": sum(int(row["false_positive"]) for row in rows),
            "false_negative_total": sum(int(row["false_negative"]) for row in rows),
        }
    return {
        "case_count": len(cases),
        "truth_residual_max": max(float(case["truth_residual_max"]) for case in cases),
        "truth_residual_min": min(float(case["truth_residual_min"]) for case in cases),
        "best_false_residual_min": min(float(case["best_false_residual"]) for case in cases),
        "by_threshold": by_threshold,
        "cases": cases,
    }


def _summarize_random(cases: list[dict[str, object]], *, thresholds: list[float]) -> dict[str, object]:
    by_threshold = {}
    for threshold in thresholds:
        rows = [
            result
            for case in cases
            for result in case["threshold_results"]
            if result["threshold"] == threshold
        ]
        by_threshold[str(threshold)] = {
            "accepted_total": sum(int(row["accepted_count"]) for row in rows),
            "false_positive_total": sum(int(row["false_positive"]) for row in rows),
        }
    min_case = min(cases, key=lambda case: float(case["best_residual"]))
    return {
        "case_count": len(cases),
        "best_residual_min": min(float(case["best_residual"]) for case in cases),
        "best_residual_min_case_index": int(min_case["case_index"]),
        "by_threshold": by_threshold,
    }


def main() -> None:
    args = _parse_args()
    high_mismatch_cases = [
        _high_mismatch_case(index, gamma=args.gamma, noise_std=args.noise_std, thresholds=args.thresholds)
        for index in range(args.high_case_count)
    ]
    random_cases = [
        _random_non_burst_case(index, gamma=args.gamma, noise_std=args.noise_std, thresholds=args.thresholds)
        for index in range(args.random_case_count)
    ]
    high_summary = _summarize_high_mismatch(high_mismatch_cases, thresholds=args.thresholds)
    random_summary = _summarize_random(random_cases, thresholds=args.thresholds)
    threshold_00625 = high_summary["by_threshold"].get("0.00625")
    random_00625 = random_summary["by_threshold"].get("0.00625")
    report = {
        "status": "aisb_threshold_margin_synthetic_only_no_video_no_gpu_no_claim",
        "threshold_margin_contract": {
            "threshold_sweep_is_diagnostic_not_fixed_fpr": True,
            "uses_owner_key_or_message": False,
            "estimates_affine_channel_during_acquisition": False,
            "paper_claim": False,
        },
        "thresholds": args.thresholds,
        "gamma": args.gamma,
        "noise_std": args.noise_std,
        "high_mismatch": high_summary,
        "random_non_burst": random_summary,
        "synthetic_margin_candidate_threshold_0.00625_pass": (
            threshold_00625 is not None
            and random_00625 is not None
            and threshold_00625["alignment_exact_count"] == len(high_mismatch_cases)
            and threshold_00625["false_positive_total"] == 0
            and threshold_00625["false_negative_total"] == 0
            and random_00625["accepted_total"] == 0
        ),
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
