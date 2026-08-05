"""Diagnose top scoring AISB owner/wrong-key competition for one synthetic case.

This is CPU-only diagnostic tooling. It reuses the same public AISB ambiguity
set and exact owner/wrong-key scoring contract as
`run_aisb_sequence_ambiguity_exact_probe.py`, but emits top-k ranking details
instead of only the best owner and best wrong scores.

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

from sc_sstw_feasibility.aisb import make_redundant_templates, scan_burst_candidates
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from sc_sstw_feasibility.sync import flatten_state_pairs
from sc_sstw_feasibility.sync_fast import (
    as_c_flat_sequence,
    dynamic_time_sync_score_bounded_prepared_workspace_c,
    make_c_dtw_workspace,
)
from run_aisb_channel_mismatch_probe import _apply_quadratic_mismatch
from run_aisb_long_sequence_probe import _apply_long_edits, _build_long_sequence
from run_aisb_payload_probe import _candidate_key
from run_aisb_pruned_search_probe import Candidate, _candidate_space
from run_aisb_sequence_ambiguity_exact_probe import (
    _candidate_states_by_candidate,
    _equalized_for_sequence,
    _supported_ambiguity_sequences,
)
from run_aisb_sequence_consistency_probe import _cycle_support_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose CPU-only AISB case scoring competition.",
    )
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=100.0)
    parser.add_argument("--noise-std", type=float, default=0.20)
    parser.add_argument("--residual-threshold", type=float, default=0.0125)
    parser.add_argument("--near-tie-ratio", type=float, default=5.0)
    parser.add_argument("--per-cluster-limit", type=int, default=3)
    parser.add_argument("--max-sequences", type=int, default=512)
    parser.add_argument("--burst-count", type=int, default=12)
    parser.add_argument("--filler-multiplier", type=int, default=7)
    parser.add_argument("--min-sequence-support", type=int, default=12)
    parser.add_argument("--message-space-size", type=int, default=64)
    parser.add_argument("--wrong-key-count", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    if args.case_index < 0:
        raise SystemExit("--case-index must be non-negative")
    if args.gamma < 0:
        raise SystemExit("--gamma must be non-negative")
    if args.noise_std < 0:
        raise SystemExit("--noise-std must be non-negative")
    if args.residual_threshold <= 0:
        raise SystemExit("--residual-threshold must be positive")
    if args.near_tie_ratio < 1.0:
        raise SystemExit("--near-tie-ratio must be at least 1")
    if args.per_cluster_limit <= 0:
        raise SystemExit("--per-cluster-limit must be positive")
    if args.max_sequences <= 0:
        raise SystemExit("--max-sequences must be positive")
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
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    return args


def _score_candidate(
    *,
    sequence_index: int,
    candidate: Candidate,
    native_equalized: object,
    native_candidate_states: object,
    workspace: object,
) -> dict[str, object]:
    score, abandoned = dynamic_time_sync_score_bounded_prepared_workspace_c(
        native_equalized,
        native_candidate_states,
        workspace,
        min_score_to_beat=-float("inf"),
    )
    if abandoned:
        raise RuntimeError("exact diagnostic scoring unexpectedly abandoned")
    return {
        "sequence_index": sequence_index,
        "role": candidate[0],
        "key": candidate[1],
        "message": candidate[2],
        "score": score,
    }


def _top_by_message(scored: list[dict[str, object]], *, role: str) -> list[dict[str, object]]:
    best_by_message: dict[str, dict[str, object]] = {}
    for item in scored:
        if item["role"] != role:
            continue
        message = str(item["message"])
        if message not in best_by_message or float(item["score"]) > float(best_by_message[message]["score"]):
            best_by_message[message] = item
    return sorted(best_by_message.values(), key=lambda item: float(item["score"]), reverse=True)


def _candidate_summary(sequence: object) -> list[dict[str, object]]:
    return [
        {
            "start_index": candidate.start_index,
            "template_id": candidate.template_id,
            "residual": candidate.residual,
            "observed_length": candidate.observed_length,
            "missing_template_index": candidate.missing_template_index,
            "candidate_key": _candidate_key(candidate),
        }
        for candidate in sequence
    ]


def main() -> None:
    args = _parse_args()

    owner_key = f"owner_pruned_{args.burst_count}_{args.message_space_size}_{args.case_index}"
    message_space = [f"message_{index}" for index in range(args.message_space_size)]
    true_message = message_space[args.case_index % len(message_space)]
    states, burst_plan = _build_long_sequence(
        args.case_index,
        owner_key=owner_key,
        message=true_message,
        burst_count=args.burst_count,
        filler_multiplier=args.filler_multiplier,
    )
    observations = generate_observations(
        states,
        make_random_channel(
            141000 + 100 * args.message_space_size + args.case_index,
            relation_count=16,
            noise_std=args.noise_std,
        ),
        seed=142000 + 100 * args.message_space_size + args.case_index,
    )
    if args.gamma != 0.0:
        observations = _apply_quadratic_mismatch(observations, states, gamma=args.gamma)
    edited, _, truth = _apply_long_edits(observations, burst_plan, case_index=args.case_index)
    scanned = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    ambiguity_sequences = _supported_ambiguity_sequences(
        scanned,
        residual_threshold=args.residual_threshold,
        near_tie_ratio=args.near_tie_ratio,
        per_cluster_limit=args.per_cluster_limit,
        max_sequences=args.max_sequences,
        min_sequence_support=args.min_sequence_support,
    )
    truth_set = set(truth)
    truth_sequence_indices = [
        index
        for index, sequence in enumerate(ambiguity_sequences)
        if {_candidate_key(candidate) for candidate in sequence} == truth_set
    ]
    truth_covered_sequence_indices = [
        index
        for index, sequence in enumerate(ambiguity_sequences)
        if truth_set <= {_candidate_key(candidate) for candidate in sequence}
    ]
    equalized_by_sequence = [
        _equalized_for_sequence(edited, sequence)
        for sequence in ambiguity_sequences
    ]
    wrong_keys = [
        f"wrong_sequence_ambiguity_{args.burst_count}_{args.message_space_size}_{args.case_index}_{index}"
        for index in range(args.wrong_key_count)
    ]
    candidates = _candidate_space(owner_key, wrong_keys, message_space)
    states_by_candidate = _candidate_states_by_candidate(
        candidates,
        sequence_length=len(states),
        burst_plan=burst_plan,
    )
    native_states_by_candidate = {
        candidate: as_c_flat_sequence(candidate_states)
        for candidate, candidate_states in states_by_candidate.items()
    }
    native_equalized_by_sequence = [
        as_c_flat_sequence(flatten_state_pairs(equalized))
        for equalized in equalized_by_sequence
    ]
    workspace_by_sequence = [
        make_c_dtw_workspace(len(states))
        for _ in equalized_by_sequence
    ]

    scored: list[dict[str, object]] = []
    for sequence_index, native_equalized in enumerate(native_equalized_by_sequence):
        for candidate in candidates:
            scored.append(
                _score_candidate(
                    sequence_index=sequence_index,
                    candidate=candidate,
                    native_equalized=native_equalized,
                    native_candidate_states=native_states_by_candidate[candidate],
                    workspace=workspace_by_sequence[sequence_index],
                )
            )

    top_global = sorted(scored, key=lambda item: float(item["score"]), reverse=True)[: args.top_k]
    top_owner = [item for item in top_global if item["role"] == "owner"]
    top_wrong = sorted(
        (item for item in scored if item["role"] == "wrong"),
        key=lambda item: float(item["score"]),
        reverse=True,
    )[: args.top_k]
    top_owner_messages = _top_by_message(scored, role="owner")[: args.top_k]
    top_wrong_messages = _top_by_message(scored, role="wrong")[: args.top_k]
    best_owner = top_owner_messages[0]
    best_wrong = top_wrong[0]

    report = {
        "status": "aisb_case_competition_diagnostic_synthetic_only_no_video_no_gpu_no_claim",
        "paper_claim": False,
        "case": {
            "case_index": args.case_index,
            "gamma": args.gamma,
            "noise_std": args.noise_std,
            "residual_threshold": args.residual_threshold,
            "near_tie_ratio": args.near_tie_ratio,
            "message_space_size": args.message_space_size,
            "wrong_key_count": args.wrong_key_count,
            "filler_multiplier": args.filler_multiplier,
            "true_message": true_message,
            "owner_key": owner_key,
            "truth_sequence_indices": truth_sequence_indices,
            "truth_covered_sequence_indices": truth_covered_sequence_indices,
            "ambiguity_sequence_count": len(ambiguity_sequences),
            "sequence_support_count_min": min(_cycle_support_count(sequence) for sequence in ambiguity_sequences)
            if ambiguity_sequences
            else 0,
            "sequence_support_count_max": max(_cycle_support_count(sequence) for sequence in ambiguity_sequences)
            if ambiguity_sequences
            else 0,
            "candidate_count": len(candidates),
            "exact_score_count": len(scored),
        },
        "competition_summary": {
            "owner_best_message": best_owner["message"],
            "owner_best_sequence_index": best_owner["sequence_index"],
            "owner_best_score": best_owner["score"],
            "best_wrong_message": best_wrong["message"],
            "best_wrong_key": best_wrong["key"],
            "best_wrong_sequence_index": best_wrong["sequence_index"],
            "best_wrong_score": best_wrong["score"],
            "score_margin": float(best_owner["score"]) - float(best_wrong["score"]),
            "owner_global_winner": top_global[0]["role"] == "owner",
            "owner_message_recovered": best_owner["message"] == true_message,
            "diagnostic_pass": best_owner["message"] == true_message
            and top_global[0]["role"] == "owner"
            and top_global[0]["message"] == true_message
            and float(best_owner["score"]) - float(best_wrong["score"]) > 0.02,
        },
        "ambiguity_sequences": [
            {
                "sequence_index": index,
                "support_count": _cycle_support_count(sequence),
                "is_exact_truth": index in truth_sequence_indices,
                "covers_truth": index in truth_covered_sequence_indices,
                "candidates": _candidate_summary(sequence),
            }
            for index, sequence in enumerate(ambiguity_sequences)
        ],
        "top_global": top_global,
        "top_wrong": top_wrong,
        "top_owner_messages": top_owner_messages,
        "top_wrong_messages": top_wrong_messages,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
