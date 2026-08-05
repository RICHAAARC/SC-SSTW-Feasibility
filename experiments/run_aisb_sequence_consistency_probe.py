"""Run CPU-only public AISB sequence-consistency diagnostics.

Single-burst residual thresholds can admit isolated random non-burst windows at
high synthetic mismatch. This probe adds a second public-only acquisition
constraint: accepted low-residual bursts must support a predeclared template
cycle over a minimum number of non-overlapping bursts. It does not estimate
the affine channel, use owner keys/messages, run video/GPU code, calibrate
fixed-FPR thresholds, or make paper claims.
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

from sc_sstw_feasibility.aisb import BurstCandidate, best_non_overlapping_sequence, make_redundant_templates, scan_burst_candidates
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from run_aisb_channel_mismatch_probe import _apply_quadratic_mismatch
from run_aisb_long_sequence_probe import _apply_long_edits, _build_long_sequence
from run_aisb_payload_probe import _candidate_key


PUBLIC_TEMPLATE_CYCLE = ("redundant_alpha", "redundant_beta", "redundant_gamma")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-only public AISB sequence-consistency diagnostics.",
    )
    parser.add_argument("--gamma", type=float, default=50.0)
    parser.add_argument("--noise-std", type=float, default=0.012)
    parser.add_argument("--high-case-count", type=int, default=4)
    parser.add_argument("--random-case-count", type=int, default=256)
    parser.add_argument("--thresholds", type=str, default="0.01,0.0105,0.0125")
    parser.add_argument("--burst-count", type=int, default=12)
    parser.add_argument("--min-sequence-support", type=int, default=12)
    args = parser.parse_args()
    if args.gamma < 0:
        raise SystemExit("--gamma must be non-negative")
    if args.noise_std < 0:
        raise SystemExit("--noise-std must be non-negative")
    if args.high_case_count <= 0:
        raise SystemExit("--high-case-count must be positive")
    if args.random_case_count <= 0:
        raise SystemExit("--random-case-count must be positive")
    if args.burst_count <= 0:
        raise SystemExit("--burst-count must be positive")
    if args.min_sequence_support <= 0:
        raise SystemExit("--min-sequence-support must be positive")
    thresholds: list[float] = []
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


def _cycle_support_count(candidates: list[BurstCandidate]) -> int:
    """Return the longest public template-cycle subsequence support count."""

    sorted_candidates = sorted(candidates, key=lambda candidate: (candidate.start_index, candidate.template_id))
    best = 0
    for phase in range(len(PUBLIC_TEMPLATE_CYCLE)):
        count = 0
        for candidate in sorted_candidates:
            expected = PUBLIC_TEMPLATE_CYCLE[(phase + count) % len(PUBLIC_TEMPLATE_CYCLE)]
            if candidate.template_id == expected:
                count += 1
        best = max(best, count)
    return best


def _has_sequence_support(candidates: list[BurstCandidate], *, min_sequence_support: int) -> bool:
    return _cycle_support_count(candidates) >= min_sequence_support


def _high_mismatch_sequence_case(
    case_index: int,
    *,
    gamma: float,
    noise_std: float,
    thresholds: list[float],
    burst_count: int,
    min_sequence_support: int,
) -> dict[str, object]:
    owner_key = f"owner_pruned_12_64_{case_index}"
    true_message = f"message_{case_index % 64}"
    states, burst_plan = _build_long_sequence(
        case_index,
        owner_key=owner_key,
        message=true_message,
        burst_count=burst_count,
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
        support_count = _cycle_support_count(accepted)
        sequence_supported = support_count >= min_sequence_support
        threshold_results.append(
            {
                "threshold": threshold,
                "accepted_count": len(accepted_set),
                "sequence_support_count": support_count,
                "sequence_supported": sequence_supported,
                "residual_alignment_exact": accepted_set == truth_set,
                "sequence_alignment_exact": sequence_supported and accepted_set == truth_set,
                "residual_false_positive": len(accepted_set - truth_set),
                "residual_false_negative": len(truth_set - accepted_set),
                "sequence_false_positive": 0 if sequence_supported and accepted_set <= truth_set else len(accepted_set - truth_set),
                "sequence_false_negative": 0 if sequence_supported and accepted_set == truth_set else len(truth_set - accepted_set),
            }
        )
    return {
        "case_index": case_index,
        "gamma": gamma,
        "noise_std": noise_std,
        "truth_count": len(truth_set),
        "threshold_results": threshold_results,
    }


def _random_non_burst_sequence_case(
    case_index: int,
    *,
    gamma: float,
    noise_std: float,
    thresholds: list[float],
    min_sequence_support: int,
) -> dict[str, object]:
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
        support_count = _cycle_support_count(accepted)
        sequence_supported = support_count >= min_sequence_support
        threshold_results.append(
            {
                "threshold": threshold,
                "accepted_count": len(accepted),
                "sequence_support_count": support_count,
                "residual_false_positive": len(accepted),
                "sequence_false_positive": 1 if sequence_supported else 0,
            }
        )
    return {
        "case_index": case_index,
        "gamma": gamma,
        "noise_std": noise_std,
        "threshold_results": threshold_results,
    }


def _summarize_high(cases: list[dict[str, object]], *, thresholds: list[float]) -> dict[str, object]:
    by_threshold = {}
    for threshold in thresholds:
        rows = [
            result
            for case in cases
            for result in case["threshold_results"]
            if result["threshold"] == threshold
        ]
        by_threshold[str(threshold)] = {
            "residual_alignment_exact_count": sum(1 for row in rows if row["residual_alignment_exact"]),
            "sequence_alignment_exact_count": sum(1 for row in rows if row["sequence_alignment_exact"]),
            "residual_false_positive_total": sum(int(row["residual_false_positive"]) for row in rows),
            "residual_false_negative_total": sum(int(row["residual_false_negative"]) for row in rows),
            "sequence_false_positive_total": sum(int(row["sequence_false_positive"]) for row in rows),
            "sequence_false_negative_total": sum(int(row["sequence_false_negative"]) for row in rows),
            "sequence_support_min": min(int(row["sequence_support_count"]) for row in rows),
        }
    return {"case_count": len(cases), "by_threshold": by_threshold, "cases": cases}


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
            "residual_false_positive_total": sum(int(row["residual_false_positive"]) for row in rows),
            "sequence_false_positive_total": sum(int(row["sequence_false_positive"]) for row in rows),
            "sequence_support_max": max(int(row["sequence_support_count"]) for row in rows),
        }
    return {"case_count": len(cases), "by_threshold": by_threshold}


def main() -> None:
    args = _parse_args()
    high_cases = [
        _high_mismatch_sequence_case(
            index,
            gamma=args.gamma,
            noise_std=args.noise_std,
            thresholds=args.thresholds,
            burst_count=args.burst_count,
            min_sequence_support=args.min_sequence_support,
        )
        for index in range(args.high_case_count)
    ]
    random_cases = [
        _random_non_burst_sequence_case(
            index,
            gamma=args.gamma,
            noise_std=args.noise_std,
            thresholds=args.thresholds,
            min_sequence_support=args.min_sequence_support,
        )
        for index in range(args.random_case_count)
    ]
    high_summary = _summarize_high(high_cases, thresholds=args.thresholds)
    random_summary = _summarize_random(random_cases, thresholds=args.thresholds)
    report = {
        "status": "aisb_sequence_consistency_synthetic_only_no_video_no_gpu_no_claim",
        "sequence_consistency_contract": {
            "uses_owner_key_or_message": False,
            "estimates_affine_channel_during_acquisition": False,
            "threshold_sweep_is_diagnostic_not_fixed_fpr": True,
            "sequence_support_is_public_template_order_only": True,
            "paper_claim": False,
        },
        "gamma": args.gamma,
        "noise_std": args.noise_std,
        "thresholds": args.thresholds,
        "burst_count": args.burst_count,
        "min_sequence_support": args.min_sequence_support,
        "high_mismatch": high_summary,
        "random_non_burst": random_summary,
        "synthetic_sequence_consistency_pass": any(
            high_summary["by_threshold"][str(threshold)]["sequence_alignment_exact_count"] == args.high_case_count
            and high_summary["by_threshold"][str(threshold)]["sequence_false_positive_total"] == 0
            and high_summary["by_threshold"][str(threshold)]["sequence_false_negative_total"] == 0
            and random_summary["by_threshold"][str(threshold)]["sequence_false_positive_total"] == 0
            for threshold in args.thresholds
        ),
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
