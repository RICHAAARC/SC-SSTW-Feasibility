"""Run CPU-only pruned owner/wrong-key search diagnostics.

The exhaustive synthetic scorer scales with
`key_count * message_space_size * sequence_length`. This probe adds a diagnostic
two-stage search: cheap decimated dynamic-time-sync screening followed by full
dynamic-time-sync on the screened candidates. A small tier checks the pruned
winner against exhaustive search; larger tiers report feasibility but are not
claimed exhaustive.
"""

from __future__ import annotations

from pathlib import Path
import argparse
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
from sc_sstw_feasibility.sync import dynamic_time_sync, dynamic_time_sync_score
from run_aisb_channel_mismatch_probe import _apply_quadratic_mismatch
from run_aisb_long_sequence_probe import (
    _apply_long_edits,
    _build_long_sequence,
    _states_with_public_bursts,
)
from run_aisb_payload_probe import _candidate_key


Candidate = tuple[str, str, str]


def _prepare_equalized_case(
    case_index: int,
    *,
    burst_count: int,
    message_space_size: int,
    gamma: float,
    noise_std: float,
    residual_threshold: float = 0.006,
    filler_multiplier: int = 1,
) -> tuple[list[tuple[float, float]], str, str, list[str], int, list[tuple[int, str]], bool]:
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
    templates = {template.template_id: template for template in make_redundant_templates()}
    pilot_pairs = []
    for candidate in accepted:
        pilot_pairs.extend(template_observation_pairs(candidate, edited, templates[candidate.template_id]))
    calibration = calibrate_from_pilot_pairs(pilot_pairs)
    equalized = equalize_observations(edited, calibration)
    return equalized, owner_key, true_message, message_space, len(states), burst_plan, accepted_set == truth_set


def _candidate_states(candidate: Candidate, sequence_length: int, burst_plan: list[tuple[int, str]]) -> list[tuple[float, float]]:
    _, key, message = candidate
    return _states_with_public_bursts(key, message, sequence_length, burst_plan)


def _cheap_decimated_score(
    equalized: list[tuple[float, float]],
    candidate: Candidate,
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
    stride: int,
) -> float:
    states = _candidate_states(candidate, sequence_length, burst_plan)
    return dynamic_time_sync_score(equalized[::stride], states[::stride])


def _full_score(
    equalized: list[tuple[float, float]],
    candidate: Candidate,
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> float:
    states = _candidate_states(candidate, sequence_length, burst_plan)
    return dynamic_time_sync_score(equalized, states)


def _candidate_space(owner_key: str, wrong_keys: list[str], message_space: list[str]) -> list[Candidate]:
    return [
        ("owner", owner_key, message)
        for message in message_space
    ] + [
        ("wrong", wrong_key, message)
        for wrong_key in wrong_keys
        for message in message_space
    ]


def _best_exhaustive(
    equalized: list[tuple[float, float]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> tuple[Candidate, float, int]:
    scored = [
        (
            candidate,
            _full_score(equalized, candidate, sequence_length=sequence_length, burst_plan=burst_plan),
        )
        for candidate in candidates
    ]
    best_candidate, best_score = max(scored, key=lambda item: item[1])
    return best_candidate, best_score, len(scored)


def _best_pruned(
    equalized: list[tuple[float, float]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
    stride: int,
    top_k_global: int,
    top_k_owner: int,
    top_k_per_wrong_key: int,
) -> tuple[Candidate, float, int, int]:
    cheap_scored = [
        (
            candidate,
            _cheap_decimated_score(
                equalized,
                candidate,
                sequence_length=sequence_length,
                burst_plan=burst_plan,
                stride=stride,
            ),
        )
        for candidate in candidates
    ]
    selected: dict[Candidate, None] = {}
    for candidate, _ in sorted(cheap_scored, key=lambda item: item[1], reverse=True)[:top_k_global]:
        selected[candidate] = None
    owner_candidates = [item for item in cheap_scored if item[0][0] == "owner"]
    for candidate, _ in sorted(owner_candidates, key=lambda item: item[1], reverse=True)[:top_k_owner]:
        selected[candidate] = None
    wrong_keys = sorted({candidate[1] for candidate, _ in cheap_scored if candidate[0] == "wrong"})
    for wrong_key in wrong_keys:
        wrong_candidates = [item for item in cheap_scored if item[0][0] == "wrong" and item[0][1] == wrong_key]
        for candidate, _ in sorted(wrong_candidates, key=lambda item: item[1], reverse=True)[:top_k_per_wrong_key]:
            selected[candidate] = None

    full_scored = [
        (
            candidate,
            _full_score(equalized, candidate, sequence_length=sequence_length, burst_plan=burst_plan),
        )
        for candidate in selected
    ]
    best_candidate, best_score = max(full_scored, key=lambda item: item[1])
    return best_candidate, best_score, len(cheap_scored), len(full_scored)


def _selected_from_cheap_scores(
    cheap_scored: list[tuple[Candidate, float]],
    *,
    top_k_global: int,
    top_k_owner: int,
    top_k_per_wrong_key: int,
) -> dict[Candidate, None]:
    selected: dict[Candidate, None] = {}
    for candidate, _ in sorted(cheap_scored, key=lambda item: item[1], reverse=True)[:top_k_global]:
        selected[candidate] = None
    if top_k_owner > 0:
        owner_candidates = [item for item in cheap_scored if item[0][0] == "owner"]
        for candidate, _ in sorted(owner_candidates, key=lambda item: item[1], reverse=True)[:top_k_owner]:
            selected[candidate] = None
    if top_k_per_wrong_key > 0:
        wrong_keys = sorted({candidate[1] for candidate, _ in cheap_scored if candidate[0] == "wrong"})
        for wrong_key in wrong_keys:
            wrong_candidates = [item for item in cheap_scored if item[0][0] == "wrong" and item[0][1] == wrong_key]
            for candidate, _ in sorted(wrong_candidates, key=lambda item: item[1], reverse=True)[:top_k_per_wrong_key]:
                selected[candidate] = None
    return selected


def _best_from_full_scores(
    full_scored: list[tuple[Candidate, float]],
    *,
    role: str | None = None,
) -> tuple[Candidate, float]:
    candidates = [item for item in full_scored if role is None or item[0][0] == role]
    if not candidates:
        raise ValueError("screened candidate set is missing a required role")
    return max(candidates, key=lambda item: item[1])


def _role_margin(best_owner_score: float, best_wrong_score: float) -> float:
    return best_owner_score - best_wrong_score


def _pruned_case(
    case_index: int,
    *,
    burst_count: int,
    message_space_size: int,
    wrong_key_count: int,
    gamma: float,
    noise_std: float,
    residual_threshold: float = 0.006,
    filler_multiplier: int = 1,
    compare_exhaustive: bool,
) -> dict[str, object]:
    equalized, owner_key, true_message, message_space, sequence_length, burst_plan, alignment_exact = _prepare_equalized_case(
        case_index,
        burst_count=burst_count,
        message_space_size=message_space_size,
        gamma=gamma,
        noise_std=noise_std,
        residual_threshold=residual_threshold,
        filler_multiplier=filler_multiplier,
    )
    wrong_keys = [f"wrong_pruned_{burst_count}_{message_space_size}_{case_index}_{index}" for index in range(wrong_key_count)]
    candidates = _candidate_space(owner_key, wrong_keys, message_space)
    cheap_scored = [
        (
            candidate,
            _cheap_decimated_score(
                equalized,
                candidate,
                sequence_length=sequence_length,
                burst_plan=burst_plan,
                stride=4,
            ),
        )
        for candidate in candidates
    ]
    selected = _selected_from_cheap_scores(
        cheap_scored,
        top_k_global=64,
        top_k_owner=8,
        top_k_per_wrong_key=1,
    )
    owner_cheap_scored = [item for item in cheap_scored if item[0][0] == "owner"]
    wrong_cheap_scored = [item for item in cheap_scored if item[0][0] == "wrong"]
    selected.update(
        _selected_from_cheap_scores(
            owner_cheap_scored,
            top_k_global=16,
            top_k_owner=8,
            top_k_per_wrong_key=0,
        )
    )
    selected.update(
        _selected_from_cheap_scores(
            wrong_cheap_scored,
            top_k_global=64,
            top_k_owner=0,
            top_k_per_wrong_key=1,
        )
    )
    full_scored = [
        (
            candidate,
            _full_score(equalized, candidate, sequence_length=sequence_length, burst_plan=burst_plan),
        )
        for candidate in selected
    ]
    pruned_best, pruned_best_score = _best_from_full_scores(full_scored)
    pruned_owner_best, pruned_owner_score = _best_from_full_scores(full_scored, role="owner")
    pruned_wrong_best, pruned_wrong_score = _best_from_full_scores(full_scored, role="wrong")
    exhaustive_best = None
    exhaustive_best_score = None
    exhaustive_count = None
    pruned_matches_exhaustive = None
    if compare_exhaustive:
        exhaustive_best, exhaustive_best_score, exhaustive_count = _best_exhaustive(
            equalized,
            candidates,
            sequence_length=sequence_length,
            burst_plan=burst_plan,
        )
        pruned_matches_exhaustive = pruned_best == exhaustive_best and abs(pruned_best_score - exhaustive_best_score) < 1e-12
    margin = _role_margin(pruned_owner_score, pruned_wrong_score)
    return {
        "case_index": case_index,
        "burst_count": burst_count,
        "message_space_size": message_space_size,
        "wrong_key_count": wrong_key_count,
        "gamma": gamma,
        "residual_threshold": residual_threshold,
        "filler_multiplier": filler_multiplier,
        "alignment_exact": alignment_exact,
        "true_message": true_message,
        "pruned_best_role": pruned_best[0],
        "pruned_owner_best_message": pruned_owner_best[2],
        "pruned_owner_best_score": pruned_owner_score,
        "pruned_wrong_best_score": pruned_wrong_score,
        "score_margin": margin,
        "candidate_count": len(candidates),
        "cheap_screened_count": len(cheap_scored),
        "full_scored_count": len(full_scored),
        "exhaustive_count": exhaustive_count,
        "pruned_matches_exhaustive": pruned_matches_exhaustive,
        "pass": (
            alignment_exact
            and pruned_owner_best[2] == true_message
            and margin > 0.02
            and (pruned_matches_exhaustive is not False)
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-only pruned AISB owner/wrong-key search diagnostics.",
    )
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--case-count", type=int, default=None)
    parser.add_argument("--burst-count", type=int, default=12)
    parser.add_argument("--message-space-size", type=int, default=64)
    parser.add_argument("--wrong-key-count", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.012)
    parser.add_argument("--residual-threshold", type=float, default=0.006)
    parser.add_argument("--filler-multiplier", type=int, default=1)
    parser.add_argument("--compare-exhaustive", action="store_true")
    args = parser.parse_args()
    if (args.start_index is None) != (args.case_count is None):
        raise SystemExit("--start-index and --case-count must be supplied together")
    if args.start_index is not None and args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.case_count is not None and args.case_count <= 0:
        raise SystemExit("--case-count must be positive")
    if args.burst_count <= 0:
        raise SystemExit("--burst-count must be positive")
    if args.message_space_size <= 0:
        raise SystemExit("--message-space-size must be positive")
    if args.wrong_key_count <= 0:
        raise SystemExit("--wrong-key-count must be positive")
    if args.gamma < 0:
        raise SystemExit("--gamma must be non-negative")
    if args.noise_std < 0:
        raise SystemExit("--noise-std must be non-negative")
    if args.residual_threshold <= 0:
        raise SystemExit("--residual-threshold must be positive")
    if args.filler_multiplier <= 0:
        raise SystemExit("--filler-multiplier must be positive")
    return args


def _run_parameterized(args: argparse.Namespace) -> dict[str, object]:
    cases = [
        _pruned_case(
            index,
            burst_count=args.burst_count,
            message_space_size=args.message_space_size,
            wrong_key_count=args.wrong_key_count,
            gamma=args.gamma,
            noise_std=args.noise_std,
            residual_threshold=args.residual_threshold,
            filler_multiplier=args.filler_multiplier,
            compare_exhaustive=args.compare_exhaustive,
        )
        for index in range(args.start_index, args.start_index + args.case_count)
    ]
    summary = _summarize(cases)
    return {
        "status": "aisb_pruned_search_synthetic_only_no_video_no_gpu_no_claim",
        "pruned_search_contract": {
            "cheap_stage": "decimated_dynamic_time_sync_stride_4",
            "full_stage": "full_dynamic_time_sync_on_screened_candidates",
            "compare_exhaustive": args.compare_exhaustive,
            "larger_tiers_not_exhaustive_claim": not args.compare_exhaustive,
            "fixed_fpr_claim": False,
            "paper_claim": False,
        },
        "tier": {
            "bursts": args.burst_count,
            "message_space_size": args.message_space_size,
            "wrong_key_count": args.wrong_key_count,
            "gamma": args.gamma,
            "noise_std": args.noise_std,
            "residual_threshold": args.residual_threshold,
            "filler_multiplier": args.filler_multiplier,
            "start_index": args.start_index,
            "case_count_requested": args.case_count,
            "summary": summary,
        },
        "synthetic_construction_pass": summary["pass_count"] == summary["case_count"],
        "paper_claim": False,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "alignment_exact_count": sum(1 for case in cases if case["alignment_exact"]),
        "owner_message_recovery_count": sum(1 for case in cases if case["pruned_owner_best_message"] == case["true_message"]),
        "score_margin_mean": _mean([float(case["score_margin"]) for case in cases]),
        "score_margin_min": min(float(case["score_margin"]) for case in cases),
        "candidate_count_max": max(int(case["candidate_count"]) for case in cases),
        "full_scored_count_max": max(int(case["full_scored_count"]) for case in cases),
        "cases": cases,
    }


def main() -> None:
    args = _parse_args()
    if args.start_index is not None:
        print(json.dumps(_run_parameterized(args), indent=2, sort_keys=True))
        return

    tiers = {
        "exhaustive_check_bursts_10_messages_32_wrong_16": [
            _pruned_case(index, burst_count=10, message_space_size=32, wrong_key_count=16, gamma=0.5, noise_std=0.012, compare_exhaustive=True)
            for index in range(1)
        ],
        "pruned_bursts_12_messages_48_wrong_48": [
            _pruned_case(index, burst_count=12, message_space_size=48, wrong_key_count=48, gamma=0.5, noise_std=0.012, compare_exhaustive=False)
            for index in range(1)
        ],
        "pruned_bursts_12_messages_48_wrong_48_gamma_0.8": [
            _pruned_case(index, burst_count=12, message_space_size=48, wrong_key_count=48, gamma=0.8, noise_std=0.012, compare_exhaustive=False)
            for index in range(1)
        ],
    }
    summaries = {name: _summarize(cases) for name, cases in tiers.items()}
    exhaustive_ok = all(
        case["pruned_matches_exhaustive"] is True
        for case in tiers["exhaustive_check_bursts_10_messages_32_wrong_16"]
    )
    report = {
        "status": "aisb_pruned_search_synthetic_only_no_video_no_gpu_no_claim",
        "pruned_search_contract": {
            "cheap_stage": "decimated_dynamic_time_sync_stride_4",
            "full_stage": "full_dynamic_time_sync_on_screened_candidates",
            "small_tier_compared_to_exhaustive": True,
            "larger_tiers_not_exhaustive_claim": True,
            "fixed_fpr_claim": False,
        },
        "tiers": summaries,
        "synthetic_construction_pass": (
            exhaustive_ok
            and summaries["pruned_bursts_12_messages_48_wrong_48"]["pass_count"] == 1
            and summaries["pruned_bursts_12_messages_48_wrong_48_gamma_0.8"]["pass_count"] == 1
        ),
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
