"""Run CPU-only exact scoring under the AISB threshold-margin candidate.

This probe checks whether a diagnostic residual threshold candidate preserves
exact owner/wrong/message separation. The default tier remains 64-message /
64-wrong-key, while CLI parameters allow smaller CPU diagnostic tiers. It is
synthetic only: no video, GPU, fixed-FPR, or paper claim.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_aisb_exact_search_scale_probe import _exact_scale_case


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    margins = [float(case["score_margin"]) for case in cases]
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "alignment_exact_count": sum(1 for case in cases if case["alignment_exact"]),
        "owner_message_recovery_count": sum(1 for case in cases if case["owner_best_message"] == case["true_message"]),
        "score_margin_mean": _mean(margins),
        "score_margin_min": min(margins),
        "candidate_count_max": max(int(case["candidate_count"]) for case in cases),
        "total_exact_score_count_max": max(int(case["total_exact_score_count"]) for case in cases),
        "cases": cases,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-only exact scoring under the AISB residual-threshold candidate.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First deterministic synthetic case index to run.",
    )
    parser.add_argument(
        "--case-count",
        type=int,
        default=3,
        help="Number of deterministic synthetic cases to run.",
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
        "--residual-threshold",
        type=float,
        default=0.00625,
        help="Diagnostic AISB residual threshold candidate.",
    )
    parser.add_argument(
        "--burst-count",
        type=int,
        default=12,
        help="Number of public AISB bursts in each synthetic case.",
    )
    parser.add_argument(
        "--filler-multiplier",
        type=int,
        default=1,
        help="Multiplier for secret-state filler spans between public AISB bursts.",
    )
    parser.add_argument(
        "--message-space-size",
        type=int,
        default=64,
        help="Number of synthetic messages to score exactly.",
    )
    parser.add_argument(
        "--wrong-key-count",
        type=int,
        default=64,
        help="Number of wrong keys to score exactly.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU worker processes for exact candidate scoring. Default 1 preserves serial behavior.",
    )
    args = parser.parse_args()
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.case_count <= 0:
        raise SystemExit("--case-count must be positive")
    if args.gamma < 0:
        raise SystemExit("--gamma must be non-negative")
    if args.noise_std < 0:
        raise SystemExit("--noise-std must be non-negative")
    if args.residual_threshold <= 0:
        raise SystemExit("--residual-threshold must be positive")
    if args.burst_count <= 0:
        raise SystemExit("--burst-count must be positive")
    if args.filler_multiplier <= 0:
        raise SystemExit("--filler-multiplier must be positive")
    if args.message_space_size <= 0:
        raise SystemExit("--message-space-size must be positive")
    if args.wrong_key_count <= 0:
        raise SystemExit("--wrong-key-count must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    return args


def main() -> None:
    args = _parse_args()
    cases = [
        _exact_scale_case(
            index,
            burst_count=args.burst_count,
            message_space_size=args.message_space_size,
            wrong_key_count=args.wrong_key_count,
            gamma=args.gamma,
            noise_std=args.noise_std,
            residual_threshold=args.residual_threshold,
            filler_multiplier=args.filler_multiplier,
            workers=args.workers,
        )
        for index in range(args.start_index, args.start_index + args.case_count)
    ]
    summary = _summarize(cases)
    report = {
        "status": "aisb_threshold_candidate_exact_search_synthetic_only_no_video_no_gpu_no_claim",
        "threshold_candidate_contract": {
            "residual_threshold": args.residual_threshold,
            "threshold_is_diagnostic_not_fixed_fpr": True,
            "screening_or_pruning": False,
            "owner_and_wrong_keys_share_message_search": True,
            "paper_claim": False,
        },
        "tier": {
            "bursts": args.burst_count,
            "message_space_size": args.message_space_size,
            "wrong_key_count": args.wrong_key_count,
            "workers": args.workers,
            "filler_multiplier": args.filler_multiplier,
            "gamma": args.gamma,
            "noise_std": args.noise_std,
            "start_index": args.start_index,
            "case_count_requested": args.case_count,
            "summary": summary,
        },
        "synthetic_construction_pass": summary["pass_count"] == summary["case_count"],
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
