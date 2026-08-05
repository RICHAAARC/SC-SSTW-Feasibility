"""Run CPU-only AISB stress grid diagnostics.

This probe sweeps deterministic non-affine mismatch and observation noise over
the existing edit-stress payload case. It is diagnostic only: synthetic
relation channel, no video, no GPU, no fixed-FPR calibration, and no paper
claim.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_aisb_stress_mismatch_payload_probe import _stress_mismatch_payload_case
from run_aisb_sequence_ambiguity_exact_probe import _sequence_ambiguity_exact_case


def _parse_float_csv(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AISB CPU-only stress grid diagnostics.")
    parser.add_argument("--gammas", type=_parse_float_csv, default=[0.5, 0.8, 1.0])
    parser.add_argument("--noise-stds", type=_parse_float_csv, default=[0.012, 0.016])
    parser.add_argument("--message-space-size", type=int, default=16)
    parser.add_argument("--wrong-key-count", type=int, default=24)
    parser.add_argument("--case-count", type=int, default=4)
    parser.add_argument("--ambiguity-message-space-size", type=int, default=None)
    parser.add_argument("--ambiguity-wrong-key-count", type=int, default=None)
    parser.add_argument("--ambiguity-case-count", type=int, default=None)
    parser.add_argument("--ambiguity-workers", type=int, default=1)
    args = parser.parse_args()
    if any(gamma < 0 for gamma in args.gammas):
        raise SystemExit("--gammas must be non-negative")
    if any(noise < 0 for noise in args.noise_stds):
        raise SystemExit("--noise-stds must be non-negative")
    if args.message_space_size <= 0:
        raise SystemExit("--message-space-size must be positive")
    if args.wrong_key_count <= 0:
        raise SystemExit("--wrong-key-count must be positive")
    if args.case_count <= 0:
        raise SystemExit("--case-count must be positive")
    if args.ambiguity_message_space_size is not None and args.ambiguity_message_space_size <= 0:
        raise SystemExit("--ambiguity-message-space-size must be positive")
    if args.ambiguity_wrong_key_count is not None and args.ambiguity_wrong_key_count <= 0:
        raise SystemExit("--ambiguity-wrong-key-count must be positive")
    if args.ambiguity_case_count is not None and args.ambiguity_case_count <= 0:
        raise SystemExit("--ambiguity-case-count must be positive")
    if args.ambiguity_workers <= 0:
        raise SystemExit("--ambiguity-workers must be positive")
    return args


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize_grid_cell(cases: list[dict[str, object]]) -> dict[str, object]:
    margins = [
        float(case["score_margin"])
        for case in cases
        if case.get("score_margin") is not None
    ]
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "alignment_exact_count": sum(1 for case in cases if case["alignment_exact"]),
        "owner_message_recovery_count": sum(1 for case in cases if case["owner_message_recovered"]),
        "score_margin_min": min(margins) if margins else None,
        "score_margin_mean": _mean(margins),
        "cases": cases,
    }


def _stress_grid(
    *,
    gammas: list[float],
    noise_stds: list[float],
    message_space_size: int,
    wrong_key_count: int,
    case_count: int,
) -> dict[str, object]:
    cells: dict[str, dict[str, object]] = {}
    for gamma in gammas:
        for noise_std in noise_stds:
            cases = [
                _stress_mismatch_payload_case(
                    case_index,
                    message_space_size=message_space_size,
                    wrong_key_count=wrong_key_count,
                    gamma=gamma,
                    noise_std=noise_std,
                )
                for case_index in range(case_count)
            ]
            cells[f"gamma_{gamma:g}_noise_{noise_std:g}"] = _summarize_grid_cell(cases)
    return cells


def _ambiguity_scoring_grid(
    *,
    gammas: list[float],
    noise_stds: list[float],
    message_space_size: int,
    wrong_key_count: int,
    case_count: int,
    workers: int,
) -> dict[str, object]:
    cells: dict[str, dict[str, object]] = {}
    for gamma in gammas:
        for noise_std in noise_stds:
            cases = [
                _sequence_ambiguity_exact_case(
                    case_index,
                    burst_count=12,
                    message_space_size=message_space_size,
                    wrong_key_count=wrong_key_count,
                    gamma=gamma,
                    noise_std=noise_std,
                    residual_threshold=0.0125,
                    near_tie_ratio=5.0,
                    per_cluster_limit=3,
                    max_sequences=256,
                    filler_multiplier=5,
                    min_sequence_support=12,
                    workers=workers,
                    scoring_mode="ordered_bounded_global_c",
                )
                for case_index in range(case_count)
            ]
            cells[f"gamma_{gamma:g}_noise_{noise_std:g}"] = {
                "case_count": len(cases),
                "pass_count": sum(1 for case in cases if case["pass"]),
                "truth_sequence_covered_count": sum(
                    1 for case in cases if case["truth_sequence_covered_by_ambiguity"]
                ),
                "truth_sequence_exact_count": sum(
                    1 for case in cases if case["truth_sequence_in_ambiguity"]
                ),
                "owner_message_recovery_count": sum(
                    1 for case in cases if case["owner_best_message"] == case["true_message"]
                ),
                "global_owner_recovery_count": sum(
                    1
                    for case in cases
                    if case["global_best_role"] == "owner"
                    and case["global_best_message"] == case["true_message"]
                ),
                "score_margin_min": min(float(case["score_margin"]) for case in cases),
                "score_margin_mean": _mean([float(case["score_margin"]) for case in cases]),
                "cases": cases,
            }
    return cells


def main() -> None:
    args = _parse_args()
    ambiguity_message_space_size = (
        args.ambiguity_message_space_size
        if args.ambiguity_message_space_size is not None
        else args.message_space_size
    )
    ambiguity_wrong_key_count = (
        args.ambiguity_wrong_key_count
        if args.ambiguity_wrong_key_count is not None
        else args.wrong_key_count
    )
    ambiguity_case_count = (
        args.ambiguity_case_count
        if args.ambiguity_case_count is not None
        else args.case_count
    )
    cells = _stress_grid(
        gammas=args.gammas,
        noise_stds=args.noise_stds,
        message_space_size=args.message_space_size,
        wrong_key_count=args.wrong_key_count,
        case_count=args.case_count,
    )
    ambiguity_cells = _ambiguity_scoring_grid(
        gammas=args.gammas,
        noise_stds=args.noise_stds,
        message_space_size=ambiguity_message_space_size,
        wrong_key_count=ambiguity_wrong_key_count,
        case_count=ambiguity_case_count,
        workers=args.ambiguity_workers,
    )
    report = {
        "status": "aisb_stress_grid_synthetic_only_no_video_no_gpu_no_claim",
        "stress_grid_contract": {
            "source_case": "run_aisb_stress_mismatch_payload_probe",
            "edit_model": "crop + non-burst deletions/repeats + one missing point in every burst",
            "owner_and_wrong_keys_share_same_message_space": True,
            "threshold_is_diagnostic_not_fixed_fpr": True,
            "paper_claim": False,
        },
        "grid": {
            "gammas": args.gammas,
            "noise_stds": args.noise_stds,
            "message_space_size": args.message_space_size,
            "wrong_key_count": args.wrong_key_count,
            "case_count": args.case_count,
            "cells": cells,
        },
        "ambiguity_set_scoring_grid": {
            "message_space_size": ambiguity_message_space_size,
            "wrong_key_count": ambiguity_wrong_key_count,
            "case_count": ambiguity_case_count,
            "workers": args.ambiguity_workers,
            "scoring_mode": "ordered_bounded_global_c",
            "cells": ambiguity_cells,
        },
        "diagnostic_complete": True,
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
