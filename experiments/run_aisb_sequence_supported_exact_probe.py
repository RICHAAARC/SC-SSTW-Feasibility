"""Run CPU-only sequence-supported AISB exact scoring diagnostics.

This probe composes the current active feasibility chain:

public AISB residual scan -> public template-cycle support -> freeze alignment
-> estimate affine channel -> equalize -> exact owner/wrong/message scoring.

It is synthetic only: no video, GPU, fixed-FPR calibration, or paper claim.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from sc_sstw_feasibility.aisb import best_non_overlapping_sequence, make_redundant_templates, scan_burst_candidates, template_observation_pairs
from sc_sstw_feasibility.calibration import calibrate_from_pilot_pairs, equalize_observations
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from run_aisb_channel_mismatch_probe import _apply_quadratic_mismatch
from run_aisb_exact_search_scale_probe import _best_roles_score_only_parallel, _score_only, _summarize
from run_aisb_long_sequence_probe import _apply_long_edits, _build_long_sequence
from run_aisb_payload_probe import _candidate_key
from run_aisb_pruned_search_probe import _candidate_space, _candidate_states
from run_aisb_sequence_consistency_probe import _cycle_support_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-only sequence-supported AISB exact scoring diagnostics.",
    )
    parser.add_argument("--gamma", type=float, default=50.0)
    parser.add_argument("--noise-std", type=float, default=0.012)
    parser.add_argument("--residual-threshold", type=float, default=0.0125)
    parser.add_argument("--burst-count", type=int, default=12)
    parser.add_argument("--filler-multiplier", type=int, default=3)
    parser.add_argument("--min-sequence-support", type=int, default=12)
    parser.add_argument("--message-space-size", type=int, default=24)
    parser.add_argument("--wrong-key-count", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
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
    if args.min_sequence_support <= 0:
        raise SystemExit("--min-sequence-support must be positive")
    if args.message_space_size <= 0:
        raise SystemExit("--message-space-size must be positive")
    if args.wrong_key_count <= 0:
        raise SystemExit("--wrong-key-count must be positive")
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.case_count <= 0:
        raise SystemExit("--case-count must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    return args


def _prepare_sequence_supported_case(
    case_index: int,
    *,
    burst_count: int,
    message_space_size: int,
    gamma: float,
    noise_std: float,
    residual_threshold: float,
    filler_multiplier: int,
    min_sequence_support: int,
) -> tuple[
    list[tuple[float, float]],
    str,
    str,
    list[str],
    int,
    list[tuple[int, str]],
    bool,
    int,
    int,
]:
    owner_key = f"owner_pruned_{burst_count}_{message_space_size}_{case_index}"
    message_space = [f"message_{index}" for index in range(message_space_size)]
    true_message = message_space[case_index % len(message_space)]
    states, burst_plan = _build_long_sequence(
        case_index,
        owner_key=owner_key,
        message=true_message,
        burst_count=burst_count,
        filler_multiplier=filler_multiplier,
    )
    observations = generate_observations(
        states,
        make_random_channel(141000 + 100 * message_space_size + case_index, relation_count=16, noise_std=noise_std),
        seed=142000 + 100 * message_space_size + case_index,
    )
    if gamma != 0.0:
        observations = _apply_quadratic_mismatch(observations, states, gamma=gamma)
    edited, _, truth = _apply_long_edits(observations, burst_plan, case_index=case_index)
    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=residual_threshold)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    truth_set = set(truth)
    support_count = _cycle_support_count(accepted)
    alignment_exact = accepted_set == truth_set and support_count >= min_sequence_support
    templates = {template.template_id: template for template in make_redundant_templates()}
    pilot_pairs = []
    for candidate in accepted:
        pilot_pairs.extend(template_observation_pairs(candidate, edited, templates[candidate.template_id]))
    calibration = calibrate_from_pilot_pairs(pilot_pairs)
    equalized = equalize_observations(edited, calibration)
    return (
        equalized,
        owner_key,
        true_message,
        message_space,
        len(states),
        burst_plan,
        alignment_exact,
        len(accepted_set),
        support_count,
    )


def _sequence_supported_exact_case(
    case_index: int,
    *,
    burst_count: int,
    message_space_size: int,
    wrong_key_count: int,
    gamma: float,
    noise_std: float,
    residual_threshold: float,
    filler_multiplier: int,
    min_sequence_support: int,
    workers: int,
) -> dict[str, object]:
    (
        equalized,
        owner_key,
        true_message,
        message_space,
        sequence_length,
        burst_plan,
        alignment_exact,
        accepted_count,
        sequence_support_count,
    ) = _prepare_sequence_supported_case(
        case_index,
        burst_count=burst_count,
        message_space_size=message_space_size,
        gamma=gamma,
        noise_std=noise_std,
        residual_threshold=residual_threshold,
        filler_multiplier=filler_multiplier,
        min_sequence_support=min_sequence_support,
    )
    wrong_keys = [
        f"wrong_sequence_supported_{burst_count}_{message_space_size}_{case_index}_{index}"
        for index in range(wrong_key_count)
    ]
    candidates = _candidate_space(owner_key, wrong_keys, message_space)
    global_best, global_score, owner_best, owner_score, wrong_best, wrong_score, scored_count, abandoned_count = _best_roles_score_only_parallel(
        equalized,
        candidates,
        sequence_length=sequence_length,
        burst_plan=burst_plan,
        workers=workers,
    )
    owner_full_score = _score_only(
        equalized,
        owner_best,
        sequence_length=sequence_length,
        burst_plan=burst_plan,
    )
    score_only_matches_full_owner = abs(owner_full_score - owner_score) < 1e-12
    margin = owner_score - wrong_score
    return {
        "case_index": case_index,
        "burst_count": burst_count,
        "message_space_size": message_space_size,
        "wrong_key_count": wrong_key_count,
        "gamma": gamma,
        "noise_std": noise_std,
        "residual_threshold": residual_threshold,
        "filler_multiplier": filler_multiplier,
        "min_sequence_support": min_sequence_support,
        "accepted_count": accepted_count,
        "sequence_support_count": sequence_support_count,
        "alignment_exact": alignment_exact,
        "true_message": true_message,
        "owner_best_message": owner_best[2],
        "owner_best_score": owner_score,
        "best_wrong_message": wrong_best[2],
        "best_wrong_score": wrong_score,
        "global_best_role": global_best[0],
        "global_best_message": global_best[2],
        "global_best_score": global_score,
        "score_margin": margin,
        "candidate_count": len(candidates),
        "total_exact_score_count": scored_count,
        "bounded_abandoned_count": abandoned_count,
        "parallel_workers": workers,
        "score_only_matches_full_owner": score_only_matches_full_owner,
        "pass": (
            alignment_exact
            and owner_best[2] == true_message
            and global_best[0] == "owner"
            and global_best[2] == true_message
            and score_only_matches_full_owner
            and margin > 0.02
        ),
    }


def main() -> None:
    args = _parse_args()
    cases = [
        _sequence_supported_exact_case(
            index,
            burst_count=args.burst_count,
            message_space_size=args.message_space_size,
            wrong_key_count=args.wrong_key_count,
            gamma=args.gamma,
            noise_std=args.noise_std,
            residual_threshold=args.residual_threshold,
            filler_multiplier=args.filler_multiplier,
            min_sequence_support=args.min_sequence_support,
            workers=args.workers,
        )
        for index in range(args.start_index, args.start_index + args.case_count)
    ]
    summary = _summarize(cases)
    report = {
        "status": "aisb_sequence_supported_exact_scoring_synthetic_only_no_video_no_gpu_no_claim",
        "sequence_supported_exact_contract": {
            "uses_owner_key_or_message_for_acquisition": False,
            "estimates_affine_channel_during_acquisition": False,
            "screening_or_pruning": False,
            "owner_and_wrong_keys_share_message_search": True,
            "threshold_is_diagnostic_not_fixed_fpr": True,
            "paper_claim": False,
        },
        "tier": {
            "bursts": args.burst_count,
            "message_space_size": args.message_space_size,
            "wrong_key_count": args.wrong_key_count,
            "filler_multiplier": args.filler_multiplier,
            "min_sequence_support": args.min_sequence_support,
            "gamma": args.gamma,
            "noise_std": args.noise_std,
            "residual_threshold": args.residual_threshold,
            "start_index": args.start_index,
            "case_count_requested": args.case_count,
            "workers": args.workers,
            "summary": summary,
        },
        "synthetic_construction_pass": summary["pass_count"] == summary["case_count"],
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
