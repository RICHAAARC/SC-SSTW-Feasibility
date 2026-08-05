"""Run CPU-only exact larger-search AISB diagnostics.

The previous pruned-search diagnostic showed that cheap candidate screening can
change the exact winner. This probe keeps exhaustive key/message scoring and
only replaces the path-producing DTW implementation with an exact score-only
variant. It is a CPU-cost diagnostic, not a GPU, video, fixed-FPR, or paper
claim.
"""

from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from sc_sstw_feasibility.sync import dynamic_time_sync, dynamic_time_sync_score, dynamic_time_sync_score_bounded
from run_aisb_pruned_search_probe import (
    Candidate,
    _candidate_space,
    _candidate_states,
    _prepare_equalized_case,
)


def _score_only(
    equalized: list[tuple[float, float]],
    candidate: Candidate,
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> float:
    return dynamic_time_sync_score(
        equalized,
        _candidate_states(candidate, sequence_length, burst_plan),
    )


def _score_candidate_for_pool(
    payload: tuple[Candidate, list[tuple[float, float]], int, list[tuple[int, str]]],
) -> tuple[Candidate, float]:
    candidate, equalized, sequence_length, burst_plan = payload
    return (
        candidate,
        _score_only(
            equalized,
            candidate,
            sequence_length=sequence_length,
            burst_plan=burst_plan,
        ),
    )


def _bounded_score_only(
    equalized: list[tuple[float, float]],
    candidate: Candidate,
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
    min_score_to_beat: float,
) -> tuple[float, bool]:
    return dynamic_time_sync_score_bounded(
        equalized,
        _candidate_states(candidate, sequence_length, burst_plan),
        min_score_to_beat=min_score_to_beat,
    )


def _best_score_only(
    equalized: list[tuple[float, float]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> tuple[Candidate, float, int]:
    best_candidate = candidates[0]
    best_score = _score_only(
        equalized,
        best_candidate,
        sequence_length=sequence_length,
        burst_plan=burst_plan,
    )
    scored_count = 1
    for candidate in candidates[1:]:
        score = _score_only(
            equalized,
            candidate,
            sequence_length=sequence_length,
            burst_plan=burst_plan,
        )
        scored_count += 1
        if score > best_score:
            best_candidate = candidate
            best_score = score
    return best_candidate, best_score, scored_count


def _best_roles_score_only(
    equalized: list[tuple[float, float]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> tuple[Candidate, float, Candidate, float, Candidate, float, int]:
    best_global: Candidate | None = None
    best_owner: Candidate | None = None
    best_wrong: Candidate | None = None
    best_global_score = -float("inf")
    best_owner_score = -float("inf")
    best_wrong_score = -float("inf")
    scored_count = 0
    for candidate in candidates:
        score = _score_only(
            equalized,
            candidate,
            sequence_length=sequence_length,
            burst_plan=burst_plan,
        )
        scored_count += 1
        if score > best_global_score:
            best_global = candidate
            best_global_score = score
        if candidate[0] == "owner" and score > best_owner_score:
            best_owner = candidate
            best_owner_score = score
        if candidate[0] == "wrong" and score > best_wrong_score:
            best_wrong = candidate
            best_wrong_score = score
    if best_global is None or best_owner is None or best_wrong is None:
        raise ValueError("exact search candidate space must include global, owner, and wrong candidates")
    return best_global, best_global_score, best_owner, best_owner_score, best_wrong, best_wrong_score, scored_count


def _best_roles_score_only_bounded(
    equalized: list[tuple[float, float]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> tuple[Candidate, float, Candidate, float, Candidate, float, int, int]:
    best_global: Candidate | None = None
    best_owner: Candidate | None = None
    best_wrong: Candidate | None = None
    best_global_score = -float("inf")
    best_owner_score = -float("inf")
    best_wrong_score = -float("inf")
    scored_count = 0
    abandoned_count = 0
    for candidate in candidates:
        if candidate[0] == "owner":
            score, abandoned = _bounded_score_only(
                equalized,
                candidate,
                sequence_length=sequence_length,
                burst_plan=burst_plan,
                min_score_to_beat=best_owner_score,
            )
        else:
            score, abandoned = _bounded_score_only(
                equalized,
                candidate,
                sequence_length=sequence_length,
                burst_plan=burst_plan,
                min_score_to_beat=best_wrong_score,
            )
        scored_count += 1
        abandoned_count += int(abandoned)
        if abandoned:
            continue
        if score > best_global_score:
            best_global = candidate
            best_global_score = score
        if candidate[0] == "owner" and score > best_owner_score:
            best_owner = candidate
            best_owner_score = score
        if candidate[0] == "wrong" and score > best_wrong_score:
            best_wrong = candidate
            best_wrong_score = score
    if best_global is None or best_owner is None or best_wrong is None:
        raise ValueError("exact search candidate space must include global, owner, and wrong candidates")
    return best_global, best_global_score, best_owner, best_owner_score, best_wrong, best_wrong_score, scored_count, abandoned_count


def _best_roles_score_only_parallel(
    equalized: list[tuple[float, float]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
    workers: int,
) -> tuple[Candidate, float, Candidate, float, Candidate, float, int, int]:
    if workers <= 1:
        return _best_roles_score_only_bounded(
            equalized,
            candidates,
            sequence_length=sequence_length,
            burst_plan=burst_plan,
        )
    payloads = [
        (candidate, equalized, sequence_length, burst_plan)
        for candidate in candidates
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        scored = list(executor.map(_score_candidate_for_pool, payloads, chunksize=8))
    best_global, best_global_score = max(scored, key=lambda item: item[1])
    owner_scored = [item for item in scored if item[0][0] == "owner"]
    wrong_scored = [item for item in scored if item[0][0] == "wrong"]
    if not owner_scored or not wrong_scored:
        raise ValueError("exact search candidate space must include owner and wrong candidates")
    best_owner, best_owner_score = max(owner_scored, key=lambda item: item[1])
    best_wrong, best_wrong_score = max(wrong_scored, key=lambda item: item[1])
    return best_global, best_global_score, best_owner, best_owner_score, best_wrong, best_wrong_score, len(scored), 0


def _exact_scale_case(
    case_index: int,
    *,
    burst_count: int,
    message_space_size: int,
    wrong_key_count: int,
    gamma: float,
    noise_std: float,
    residual_threshold: float = 0.006,
    filler_multiplier: int = 1,
    workers: int = 1,
) -> dict[str, object]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    equalized, owner_key, true_message, message_space, sequence_length, burst_plan, alignment_exact = _prepare_equalized_case(
        case_index,
        burst_count=burst_count,
        message_space_size=message_space_size,
        gamma=gamma,
        noise_std=noise_std,
        residual_threshold=residual_threshold,
        filler_multiplier=filler_multiplier,
    )
    wrong_keys = [
        f"wrong_exact_scale_{burst_count}_{message_space_size}_{case_index}_{index}"
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

    owner_full_score = dynamic_time_sync(
        equalized,
        _candidate_states(owner_best, sequence_length, burst_plan),
    ).score
    score_only_matches_full_owner = abs(owner_full_score - owner_score) < 1e-12
    margin = owner_score - wrong_score
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


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "alignment_exact_count": sum(1 for case in cases if case["alignment_exact"]),
        "owner_message_recovery_count": sum(1 for case in cases if case["owner_best_message"] == case["true_message"]),
        "score_margin_mean": _mean([float(case["score_margin"]) for case in cases]),
        "score_margin_min": min(float(case["score_margin"]) for case in cases),
        "candidate_count_max": max(int(case["candidate_count"]) for case in cases),
        "total_exact_score_count_max": max(int(case["total_exact_score_count"]) for case in cases),
        "cases": cases,
    }


def main() -> None:
    tiers = {
        "exact_bursts_10_messages_16_wrong_12_gamma_0.5": [
            _exact_scale_case(
                index,
                burst_count=10,
                message_space_size=16,
                wrong_key_count=12,
                gamma=0.5,
                noise_std=0.012,
            )
            for index in range(2)
        ],
        "exact_bursts_12_messages_24_wrong_12_gamma_0.5": [
            _exact_scale_case(
                index,
                burst_count=12,
                message_space_size=24,
                wrong_key_count=12,
                gamma=0.5,
                noise_std=0.012,
            )
            for index in range(1)
        ],
        "exact_bursts_12_messages_16_wrong_12_gamma_0.8": [
            _exact_scale_case(
                index,
                burst_count=12,
                message_space_size=16,
                wrong_key_count=12,
                gamma=0.8,
                noise_std=0.012,
            )
            for index in range(1)
        ],
    }
    summaries = {name: _summarize(cases) for name, cases in tiers.items()}
    report = {
        "status": "aisb_exact_score_only_search_synthetic_only_no_video_no_gpu_no_claim",
        "exact_search_contract": {
            "screening_or_pruning": False,
            "score_only_dtw_exact_equivalence_checked": True,
            "owner_and_wrong_keys_share_message_search": True,
            "fixed_fpr_claim": False,
            "paper_claim": False,
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
