"""Run CPU-only sequence-ambiguity AISB exact scoring diagnostics.

At high synthetic mismatch, public AISB residual and template-cycle constraints
can retain a small public alignment ambiguity set instead of one unique
alignment. This probe freezes that public ambiguity set, estimates one affine
channel per candidate alignment, and scores owner and wrong keys over the same
alignment set and message set.

It is synthetic only: no video, GPU, fixed-FPR calibration, or paper claim.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from sc_sstw_feasibility.aisb import (
    BurstCandidate,
    make_redundant_templates,
    scan_burst_candidates,
    template_observation_pairs,
)
from sc_sstw_feasibility.calibration import calibrate_from_pilot_pairs, equalize_observations
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from sc_sstw_feasibility.sync import (
    dynamic_time_sync_score_bounded_flat,
    flatten_state_pairs,
)
from sc_sstw_feasibility.sync_fast import dynamic_time_sync_score_bounded_flat_c
from sc_sstw_feasibility.sync_fast import dynamic_time_sync_score_bounded_prepared_c
from sc_sstw_feasibility.sync_fast import dynamic_time_sync_score_bounded_prepared_workspace_c
from sc_sstw_feasibility.sync_fast import as_c_flat_sequence
from sc_sstw_feasibility.sync_fast import as_c_flat_candidate_matrix
from sc_sstw_feasibility.sync_fast import make_c_dtw_workspace
from sc_sstw_feasibility.sync_fast import score_candidates_margin_proof_workspace_c
from run_aisb_ambiguity_probe import _candidate_ambiguity_sequences
from run_aisb_channel_mismatch_probe import _apply_quadratic_mismatch
from run_aisb_exact_search_scale_probe import _score_only
from run_aisb_long_sequence_probe import _apply_long_edits, _build_long_sequence
from run_aisb_payload_probe import _candidate_key
from run_aisb_pruned_search_probe import Candidate, _candidate_space
from run_aisb_sequence_consistency_probe import _cycle_support_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-only sequence-ambiguity AISB exact scoring diagnostics.",
    )
    parser.add_argument("--gamma", type=float, default=50.0)
    parser.add_argument("--noise-std", type=float, default=0.012)
    parser.add_argument("--residual-threshold", type=float, default=0.0125)
    parser.add_argument("--near-tie-ratio", type=float, default=2.0)
    parser.add_argument("--per-cluster-limit", type=int, default=3)
    parser.add_argument("--top-k-per-start", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=512)
    parser.add_argument("--burst-count", type=int, default=12)
    parser.add_argument("--filler-multiplier", type=int, default=3)
    parser.add_argument("--min-sequence-support", type=int, default=12)
    parser.add_argument("--message-space-size", type=int, default=24)
    parser.add_argument("--wrong-key-count", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file for CPU progress diagnostics; does not change stdout report.",
    )
    parser.add_argument(
        "--case-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file receiving each completed case before the final stdout report.",
    )
    parser.add_argument("--progress-interval", type=int, default=1024)
    parser.add_argument(
        "--scoring-mode",
        choices=(
            "candidate_parallel",
            "ordered_bounded",
            "ordered_bounded_c",
            "ordered_bounded_global",
            "ordered_bounded_global_c",
            "diagnostic_pruned_c",
            "margin_proof_c",
        ),
        default="candidate_parallel",
    )
    parser.add_argument("--diagnostic-top-k-global", type=int, default=512)
    parser.add_argument("--diagnostic-top-k-owner", type=int, default=64)
    parser.add_argument("--diagnostic-top-k-per-wrong-key", type=int, default=2)
    args = parser.parse_args()
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
    if args.top_k_per_start <= 0:
        raise SystemExit("--top-k-per-start must be positive")
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
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.case_count <= 0:
        raise SystemExit("--case-count must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.progress_interval <= 0:
        raise SystemExit("--progress-interval must be positive")
    if args.diagnostic_top_k_global < 0:
        raise SystemExit("--diagnostic-top-k-global must be non-negative")
    if args.diagnostic_top_k_owner < 0:
        raise SystemExit("--diagnostic-top-k-owner must be non-negative")
    if args.diagnostic_top_k_per_wrong_key < 0:
        raise SystemExit("--diagnostic-top-k-per-wrong-key must be non-negative")
    return args


ProgressCallback = Callable[[dict[str, object]], None]


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")))
        handle.write("\n")


def _make_progress_callback(path: Path | None) -> ProgressCallback | None:
    if path is None:
        return None

    def callback(payload: dict[str, object]) -> None:
        _append_jsonl(path, payload)

    return callback


def _score_alignment_candidate_for_pool(
    payload: tuple[int, Candidate, list[tuple[float, float]], int, list[tuple[int, str]]],
) -> tuple[int, Candidate, float]:
    sequence_index, candidate, equalized, sequence_length, burst_plan = payload
    return (
        sequence_index,
        candidate,
        _score_only(
            equalized,
            candidate,
            sequence_length=sequence_length,
            burst_plan=burst_plan,
        ),
    )


def _supported_ambiguity_sequences(
    candidates: list[BurstCandidate],
    *,
    residual_threshold: float,
    near_tie_ratio: float,
    per_cluster_limit: int,
    max_sequences: int,
    min_sequence_support: int,
) -> list[list[BurstCandidate]]:
    sequences = _candidate_ambiguity_sequences(
        candidates,
        residual_threshold=residual_threshold,
        near_tie_ratio=near_tie_ratio,
        per_cluster_limit=per_cluster_limit,
        max_sequences=max_sequences,
    )
    return [
        sequence
        for sequence in sequences
        if _cycle_support_count(sequence) >= min_sequence_support
    ]


def _equalized_for_sequence(
    observations: list[list[float]],
    sequence: list[BurstCandidate],
) -> list[tuple[float, float]]:
    templates = {template.template_id: template for template in make_redundant_templates()}
    pilot_pairs = []
    for candidate in sequence:
        pilot_pairs.extend(template_observation_pairs(candidate, observations, templates[candidate.template_id]))
    calibration = calibrate_from_pilot_pairs(pilot_pairs)
    return equalize_observations(observations, calibration)


def _score_public_alignment_set(
    equalized_by_sequence: list[list[tuple[float, float]]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
    workers: int,
    scoring_mode: str,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 1024,
    diagnostic_top_k_global: int = 512,
    diagnostic_top_k_owner: int = 64,
    diagnostic_top_k_per_wrong_key: int = 2,
) -> tuple[
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    int,
    int,
]:
    if scoring_mode in {"ordered_bounded", "ordered_bounded_c"}:
        return _score_public_alignment_set_ordered_bounded(
            equalized_by_sequence,
            candidates,
            sequence_length=sequence_length,
            burst_plan=burst_plan,
            workers=workers,
            use_native_c=scoring_mode == "ordered_bounded_c",
        )
    if scoring_mode in {"ordered_bounded_global", "ordered_bounded_global_c"}:
        return _score_public_alignment_set_ordered_bounded_global(
            equalized_by_sequence,
            candidates,
            sequence_length=sequence_length,
            burst_plan=burst_plan,
            use_native_c=scoring_mode == "ordered_bounded_global_c",
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
    if scoring_mode == "margin_proof_c":
        return _score_public_alignment_set_margin_proof_c(
            equalized_by_sequence,
            candidates,
            sequence_length=sequence_length,
            burst_plan=burst_plan,
        )
    if scoring_mode == "diagnostic_pruned_c":
        return _score_public_alignment_set_diagnostic_pruned_c(
            equalized_by_sequence,
            candidates,
            sequence_length=sequence_length,
            burst_plan=burst_plan,
            workers=workers,
            top_k_global=diagnostic_top_k_global,
            top_k_owner=diagnostic_top_k_owner,
            top_k_per_wrong_key=diagnostic_top_k_per_wrong_key,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
    if scoring_mode != "candidate_parallel":
        raise ValueError("unknown scoring mode")
    payloads = [
        (sequence_index, candidate, equalized, sequence_length, burst_plan)
        for sequence_index, equalized in enumerate(equalized_by_sequence)
        for candidate in candidates
    ]
    if workers <= 1:
        scored = [_score_alignment_candidate_for_pool(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            scored = list(executor.map(_score_alignment_candidate_for_pool, payloads, chunksize=8))
    best_global = max(scored, key=lambda item: item[2])
    owner_scored = [item for item in scored if item[1][0] == "owner"]
    wrong_scored = [item for item in scored if item[1][0] == "wrong"]
    if not owner_scored or not wrong_scored:
        raise ValueError("exact search candidate space must include owner and wrong candidates")
    best_owner = max(owner_scored, key=lambda item: item[2])
    best_wrong = max(wrong_scored, key=lambda item: item[2])
    return best_global, best_owner, best_wrong, len(scored), 0


def _candidate_states_by_candidate(
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> dict[Candidate, list[float]]:
    from run_aisb_pruned_search_probe import _candidate_states

    return {
        candidate: flatten_state_pairs(_candidate_states(candidate, sequence_length, burst_plan))
        for candidate in candidates
    }


def _score_one_alignment_sequence_ordered_bounded(
    payload: tuple[
        int,
        list[tuple[float, float]],
        list[Candidate],
        dict[Candidate, list[float]],
        dict[Candidate, object],
        bool,
    ],
) -> tuple[
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    int,
    int,
]:
    sequence_index, equalized, candidates, states_by_candidate, native_states_by_candidate, use_native_c = payload
    equalized_flat = flatten_state_pairs(equalized)
    native_equalized = as_c_flat_sequence(equalized_flat) if use_native_c else None
    native_workspace = make_c_dtw_workspace(len(next(iter(states_by_candidate.values()))) // 2) if use_native_c else None
    ordered = sorted(
        candidates,
        key=lambda candidate: _cheap_score_from_states(equalized_flat, states_by_candidate[candidate]),
        reverse=True,
    )
    best_global: tuple[int, Candidate, float] | None = None
    best_owner: tuple[int, Candidate, float] | None = None
    best_wrong: tuple[int, Candidate, float] | None = None
    scored_count = 0
    abandoned_count = 0
    for candidate in ordered:
        current_role_best = best_owner if candidate[0] == "owner" else best_wrong
        min_score_to_beat = current_role_best[2] if current_role_best is not None else -float("inf")
        if use_native_c:
            if native_equalized is None:
                raise ValueError("native equalized sequence not prepared")
            if native_workspace is None:
                raise ValueError("native DTW workspace not prepared")
            score, abandoned = dynamic_time_sync_score_bounded_prepared_workspace_c(
                native_equalized,
                native_states_by_candidate[candidate],
                native_workspace,
                min_score_to_beat=min_score_to_beat,
            )
        else:
            score, abandoned = dynamic_time_sync_score_bounded_flat(
                equalized_flat,
                states_by_candidate[candidate],
                min_score_to_beat=min_score_to_beat,
            )
        scored_count += 1
        abandoned_count += int(abandoned)
        if abandoned:
            continue
        item = (sequence_index, candidate, score)
        if best_global is None or score > best_global[2]:
            best_global = item
        if candidate[0] == "owner" and (best_owner is None or score > best_owner[2]):
            best_owner = item
        if candidate[0] == "wrong" and (best_wrong is None or score > best_wrong[2]):
            best_wrong = item
    if best_global is None or best_owner is None or best_wrong is None:
        raise ValueError("exact search candidate space must include global, owner, and wrong candidates")
    return best_global, best_owner, best_wrong, scored_count, abandoned_count


def _cheap_score_from_states(
    equalized: list[float],
    states: list[float],
    *,
    stride: int = 8,
) -> float:
    """Cheap ordering heuristic only; exact/bounded scoring remains authoritative.

    The previous heuristic ran a Python DTW for every public-sequence/candidate
    pair before exact scoring. Hard ambiguity cases can spend minutes there with
    no useful scientific signal. This sparse aligned distance is intentionally
    cheaper: it only orders candidates before the exact bounded scorer verifies
    owner/wrong scores.
    """
    total = 0.0
    limit = min(len(equalized), len(states))
    for index in range(0, limit, 2 * stride):
        dx = equalized[index] - states[index]
        dy = equalized[index + 1] - states[index + 1]
        total += dx * dx + dy * dy
    return -total


def _score_public_alignment_set_ordered_bounded(
    equalized_by_sequence: list[list[tuple[float, float]]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
    workers: int,
    use_native_c: bool,
) -> tuple[
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    int,
    int,
]:
    states_by_candidate = _candidate_states_by_candidate(
        candidates,
        sequence_length=sequence_length,
        burst_plan=burst_plan,
    )
    native_states_by_candidate = {
        candidate: as_c_flat_sequence(states)
        for candidate, states in states_by_candidate.items()
    } if use_native_c else {}
    payloads = [
        (sequence_index, equalized, candidates, states_by_candidate, native_states_by_candidate, use_native_c)
        for sequence_index, equalized in enumerate(equalized_by_sequence)
    ]
    if workers > 1 and len(payloads) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            sequence_results = list(executor.map(_score_one_alignment_sequence_ordered_bounded, payloads))
        best_global: tuple[int, Candidate, float] | None = None
        best_owner: tuple[int, Candidate, float] | None = None
        best_wrong: tuple[int, Candidate, float] | None = None
        scored_count = 0
        abandoned_count = 0
        for sequence_global, sequence_owner, sequence_wrong, sequence_scored, sequence_abandoned in sequence_results:
            scored_count += sequence_scored
            abandoned_count += sequence_abandoned
            if best_global is None or sequence_global[2] > best_global[2]:
                best_global = sequence_global
            if best_owner is None or sequence_owner[2] > best_owner[2]:
                best_owner = sequence_owner
            if best_wrong is None or sequence_wrong[2] > best_wrong[2]:
                best_wrong = sequence_wrong
        if best_global is None or best_owner is None or best_wrong is None:
            raise ValueError("exact search candidate space must include global, owner, and wrong candidates")
        return best_global, best_owner, best_wrong, scored_count, abandoned_count

    best_global: tuple[int, Candidate, float] | None = None
    best_owner: tuple[int, Candidate, float] | None = None
    best_wrong: tuple[int, Candidate, float] | None = None
    scored_count = 0
    abandoned_count = 0
    for payload in payloads:
        sequence_global, sequence_owner, sequence_wrong, sequence_scored, sequence_abandoned = (
            _score_one_alignment_sequence_ordered_bounded(payload)
        )
        scored_count += sequence_scored
        abandoned_count += sequence_abandoned
        if best_global is None or sequence_global[2] > best_global[2]:
            best_global = sequence_global
        if best_owner is None or sequence_owner[2] > best_owner[2]:
            best_owner = sequence_owner
        if best_wrong is None or sequence_wrong[2] > best_wrong[2]:
            best_wrong = sequence_wrong
    if best_global is None or best_owner is None or best_wrong is None:
        raise ValueError("exact search candidate space must include global, owner, and wrong candidates")
    return best_global, best_owner, best_wrong, scored_count, abandoned_count


def _score_public_alignment_set_ordered_bounded_global(
    equalized_by_sequence: list[list[tuple[float, float]]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
    use_native_c: bool,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 1024,
) -> tuple[
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    int,
    int,
]:
    states_by_candidate = _candidate_states_by_candidate(
        candidates,
        sequence_length=sequence_length,
        burst_plan=burst_plan,
    )
    native_states_by_candidate = {
        candidate: as_c_flat_sequence(states)
        for candidate, states in states_by_candidate.items()
    } if use_native_c else {}
    equalized_flat_by_sequence = [
        flatten_state_pairs(equalized)
        for equalized in equalized_by_sequence
    ]
    native_equalized_by_sequence = [
        as_c_flat_sequence(equalized_flat)
        for equalized_flat in equalized_flat_by_sequence
    ] if use_native_c else []
    workspace_by_sequence = [
        make_c_dtw_workspace(len(next(iter(states_by_candidate.values()))) // 2)
        for _ in equalized_by_sequence
    ] if use_native_c else []
    ordered = sorted(
        (
            (
                sequence_index,
                candidate,
                _cheap_score_from_states(equalized_flat, states_by_candidate[candidate]),
            )
            for sequence_index, equalized_flat in enumerate(equalized_flat_by_sequence)
            for candidate in candidates
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    best_global: tuple[int, Candidate, float] | None = None
    best_owner: tuple[int, Candidate, float] | None = None
    best_wrong: tuple[int, Candidate, float] | None = None
    scored_count = 0
    abandoned_count = 0
    for sequence_index, candidate, _cheap_score in ordered:
        if progress_callback is not None and scored_count % progress_interval == 0:
            progress_callback(
                {
                    "event": "scoring_candidate_start",
                    "scoring_mode": "ordered_bounded_global_c" if use_native_c else "ordered_bounded_global",
                    "sequence_index": sequence_index,
                    "candidate_index": scored_count,
                    "candidate_role": candidate[0],
                    "candidate_key": candidate[1],
                    "candidate_message": candidate[2],
                    "scored_count": scored_count,
                    "ordered_candidate_count": len(ordered),
                }
            )
        current_role_best = best_owner if candidate[0] == "owner" else best_wrong
        min_score_to_beat = current_role_best[2] if current_role_best is not None else -float("inf")
        if use_native_c:
            score, abandoned = dynamic_time_sync_score_bounded_prepared_workspace_c(
                native_equalized_by_sequence[sequence_index],
                native_states_by_candidate[candidate],
                workspace_by_sequence[sequence_index],
                min_score_to_beat=min_score_to_beat,
            )
        else:
            score, abandoned = dynamic_time_sync_score_bounded_flat(
                equalized_flat_by_sequence[sequence_index],
                states_by_candidate[candidate],
                min_score_to_beat=min_score_to_beat,
            )
        scored_count += 1
        abandoned_count += int(abandoned)
        if progress_callback is not None and scored_count % progress_interval == 0:
            progress_callback(
                {
                    "event": "scoring_candidate_finish",
                    "scoring_mode": "ordered_bounded_global_c" if use_native_c else "ordered_bounded_global",
                    "scored_count": scored_count,
                    "ordered_candidate_count": len(ordered),
                    "abandoned_count": abandoned_count,
                }
            )
        if abandoned:
            continue
        item = (sequence_index, candidate, score)
        if best_global is None or score > best_global[2]:
            best_global = item
        if candidate[0] == "owner" and (best_owner is None or score > best_owner[2]):
            best_owner = item
        if candidate[0] == "wrong" and (best_wrong is None or score > best_wrong[2]):
            best_wrong = item
    if best_global is None or best_owner is None or best_wrong is None:
        raise ValueError("exact search candidate space must include global, owner, and wrong candidates")
    return best_global, best_owner, best_wrong, scored_count, abandoned_count


def _score_public_alignment_set_margin_proof_c(
    equalized_by_sequence: list[list[tuple[float, float]]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
) -> tuple[
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    int,
    int,
]:
    """Prove the diagnostic owner margin without exact-ranking all wrong keys.

    This keeps the same candidate space and exact bounded native DTW recurrence.
    It first computes the owner best exactly. Wrong-key candidates are then
    scored only until they either beat the required diagnostic margin threshold
    or are safely abandoned by the exact lower-bound rule. The returned wrong
    score is therefore an upper bound when `best_wrong_message` is
    `wrong_score_upper_bound_placeholder`.
    """

    states_by_candidate = _candidate_states_by_candidate(
        candidates,
        sequence_length=sequence_length,
        burst_plan=burst_plan,
    )
    equalized_flat_by_sequence = [
        flatten_state_pairs(equalized)
        for equalized in equalized_by_sequence
    ]
    native_equalized_by_sequence = [
        as_c_flat_sequence(equalized_flat)
        for equalized_flat in equalized_flat_by_sequence
    ]
    native_candidate_matrix = as_c_flat_candidate_matrix(
        [states_by_candidate[candidate] for candidate in candidates],
        [0 if candidate[0] == "owner" else 1 for candidate in candidates],
    )
    workspace = make_c_dtw_workspace(len(next(iter(states_by_candidate.values()))) // 2)
    if not any(candidate[0] == "owner" for candidate in candidates) or not any(candidate[0] == "wrong" for candidate in candidates):
        raise ValueError("margin proof candidate space must include owner and wrong candidates")

    best_global: tuple[int, Candidate, float] | None = None
    best_owner: tuple[int, Candidate, float] | None = None
    best_wrong: tuple[int, Candidate, float] | None = None
    scored_count = 0
    abandoned_count = 0
    for sequence_index, native_equalized in enumerate(native_equalized_by_sequence):
        result = score_candidates_margin_proof_workspace_c(
            native_equalized,
            native_candidate_matrix,
            workspace,
        )
        scored_count += result.scored_count
        abandoned_count += result.abandoned_count
        owner_item = (sequence_index, candidates[result.best_owner_index], result.best_owner_score)
        wrong_item = (
            sequence_index if result.best_wrong_index >= 0 else -1,
            candidates[result.best_wrong_index]
            if result.best_wrong_index >= 0
            else ("wrong", "wrong_score_upper_bound_placeholder", "wrong_score_upper_bound_placeholder"),
            result.best_wrong_score,
        )
        sequence_global = wrong_item if result.best_wrong_index >= 0 and wrong_item[2] > owner_item[2] else owner_item
        if best_global is None or sequence_global[2] > best_global[2]:
            best_global = sequence_global
        if best_wrong is None or wrong_item[2] > best_wrong[2]:
            best_wrong = wrong_item
        item = owner_item
        if best_owner is None or owner_item[2] > best_owner[2]:
            best_owner = item
    if best_global is None or best_owner is None or best_wrong is None:
        raise ValueError("margin proof did not compute required candidates")
    return best_global, best_owner, best_wrong, scored_count, abandoned_count


def _select_diagnostic_pruned_candidates(
    cheap_scored: list[tuple[Candidate, float]],
    *,
    top_k_global: int,
    top_k_owner: int,
    top_k_per_wrong_key: int,
) -> list[Candidate]:
    """Select a diagnostic subset before exact scoring.

    This is a CPU feasibility diagnostic, not exhaustive wrong-key evidence:
    candidates not retained by this public cheap screen are not exact-scored.
    """

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
    return list(selected)


def _score_public_alignment_set_diagnostic_pruned_c(
    equalized_by_sequence: list[list[tuple[float, float]]],
    candidates: list[Candidate],
    *,
    sequence_length: int,
    burst_plan: list[tuple[int, str]],
    workers: int,
    top_k_global: int,
    top_k_owner: int,
    top_k_per_wrong_key: int,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 1024,
) -> tuple[
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    int,
    int,
]:
    states_by_candidate = _candidate_states_by_candidate(
        candidates,
        sequence_length=sequence_length,
        burst_plan=burst_plan,
    )
    native_states_by_candidate = {
        candidate: as_c_flat_sequence(states)
        for candidate, states in states_by_candidate.items()
    }
    payloads = [
        (
            sequence_index,
            equalized,
            candidates,
            states_by_candidate,
            native_states_by_candidate,
            top_k_global,
            top_k_owner,
            top_k_per_wrong_key,
        )
        for sequence_index, equalized in enumerate(equalized_by_sequence)
    ]
    if workers > 1 and len(payloads) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            sequence_results = list(executor.map(_score_one_alignment_sequence_diagnostic_pruned_c, payloads))
        best_global: tuple[int, Candidate, float] | None = None
        best_owner: tuple[int, Candidate, float] | None = None
        best_wrong: tuple[int, Candidate, float] | None = None
        scored_count = 0
        abandoned_count = 0
        for sequence_global, sequence_owner, sequence_wrong, sequence_scored, sequence_abandoned in sequence_results:
            scored_count += sequence_scored
            abandoned_count += sequence_abandoned
            if progress_callback is not None and scored_count % progress_interval == 0:
                progress_callback(
                    {
                        "event": "diagnostic_pruned_scoring_progress",
                        "scored_count": scored_count,
                        "abandoned_count": abandoned_count,
                        "workers": workers,
                    }
                )
            if best_global is None or sequence_global[2] > best_global[2]:
                best_global = sequence_global
            if best_owner is None or sequence_owner[2] > best_owner[2]:
                best_owner = sequence_owner
            if best_wrong is None or sequence_wrong[2] > best_wrong[2]:
                best_wrong = sequence_wrong
        if best_global is None or best_owner is None or best_wrong is None:
            raise ValueError("diagnostic pruned candidate space must include global, owner, and wrong candidates")
        return best_global, best_owner, best_wrong, scored_count, abandoned_count

    best_global: tuple[int, Candidate, float] | None = None
    best_owner: tuple[int, Candidate, float] | None = None
    best_wrong: tuple[int, Candidate, float] | None = None
    scored_count = 0
    abandoned_count = 0
    for payload in payloads:
        sequence_global, sequence_owner, sequence_wrong, sequence_scored, sequence_abandoned = (
            _score_one_alignment_sequence_diagnostic_pruned_c(payload)
        )
        scored_count += sequence_scored
        abandoned_count += sequence_abandoned
        if progress_callback is not None and scored_count % progress_interval == 0:
            progress_callback(
                {
                    "event": "diagnostic_pruned_scoring_progress",
                    "scored_count": scored_count,
                    "abandoned_count": abandoned_count,
                    "workers": workers,
                }
            )
        if best_global is None or sequence_global[2] > best_global[2]:
            best_global = sequence_global
        if best_owner is None or sequence_owner[2] > best_owner[2]:
            best_owner = sequence_owner
        if best_wrong is None or sequence_wrong[2] > best_wrong[2]:
            best_wrong = sequence_wrong
    if best_global is None or best_owner is None or best_wrong is None:
        raise ValueError("diagnostic pruned candidate space must include global, owner, and wrong candidates")
    return best_global, best_owner, best_wrong, scored_count, abandoned_count


def _score_one_alignment_sequence_diagnostic_pruned_c(
    payload: tuple[
        int,
        list[tuple[float, float]],
        list[Candidate],
        dict[Candidate, list[float]],
        dict[Candidate, object],
        int,
        int,
        int,
    ],
) -> tuple[
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    tuple[int, Candidate, float],
    int,
    int,
]:
    (
        sequence_index,
        equalized,
        candidates,
        states_by_candidate,
        native_states_by_candidate,
        top_k_global,
        top_k_owner,
        top_k_per_wrong_key,
    ) = payload
    equalized_flat = flatten_state_pairs(equalized)
    native_equalized = as_c_flat_sequence(equalized_flat)
    workspace = make_c_dtw_workspace(len(next(iter(states_by_candidate.values()))) // 2)
    cheap_scored = [
        (
            candidate,
            _cheap_score_from_states(equalized_flat, states_by_candidate[candidate]),
        )
        for candidate in candidates
    ]
    cheap_score_by_candidate = dict(cheap_scored)
    selected = _select_diagnostic_pruned_candidates(
        cheap_scored,
        top_k_global=top_k_global,
        top_k_owner=top_k_owner,
        top_k_per_wrong_key=top_k_per_wrong_key,
    )
    if not any(candidate[0] == "owner" for candidate in selected) or not any(candidate[0] == "wrong" for candidate in selected):
        raise ValueError("diagnostic pruned candidate set must include owner and wrong candidates")
    best_global: tuple[int, Candidate, float] | None = None
    best_owner: tuple[int, Candidate, float] | None = None
    best_wrong: tuple[int, Candidate, float] | None = None
    scored_count = 0
    abandoned_count = 0
    for candidate in sorted(
        selected,
        key=lambda item: cheap_score_by_candidate[item],
        reverse=True,
    ):
        current_role_best = best_owner if candidate[0] == "owner" else best_wrong
        min_score_to_beat = current_role_best[2] if current_role_best is not None else -float("inf")
        score, abandoned = dynamic_time_sync_score_bounded_prepared_workspace_c(
            native_equalized,
            native_states_by_candidate[candidate],
            workspace,
            min_score_to_beat=min_score_to_beat,
        )
        scored_count += 1
        abandoned_count += int(abandoned)
        if abandoned:
            continue
        item = (sequence_index, candidate, score)
        if best_global is None or score > best_global[2]:
            best_global = item
        if candidate[0] == "owner" and (best_owner is None or score > best_owner[2]):
            best_owner = item
        if candidate[0] == "wrong" and (best_wrong is None or score > best_wrong[2]):
            best_wrong = item
    if best_global is None or best_owner is None or best_wrong is None:
        raise ValueError("diagnostic pruned candidate space must include global, owner, and wrong candidates")
    return best_global, best_owner, best_wrong, scored_count, abandoned_count


def _sequence_ambiguity_exact_case(
    case_index: int,
    *,
    burst_count: int,
    message_space_size: int,
    wrong_key_count: int,
    gamma: float,
    noise_std: float,
    residual_threshold: float,
    near_tie_ratio: float,
    per_cluster_limit: int,
    top_k_per_start: int = 1,
    max_sequences: int,
    filler_multiplier: int,
    min_sequence_support: int,
    workers: int,
    scoring_mode: str = "candidate_parallel",
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 1024,
    diagnostic_top_k_global: int = 512,
    diagnostic_top_k_owner: int = 64,
    diagnostic_top_k_per_wrong_key: int = 2,
) -> dict[str, object]:
    if progress_callback is not None:
        progress_callback(
            {
                "event": "case_start",
                "case_index": case_index,
                "gamma": gamma,
                "noise_std": noise_std,
                "message_space_size": message_space_size,
                "wrong_key_count": wrong_key_count,
                "scoring_mode": scoring_mode,
            }
        )
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
    scanned = scan_burst_candidates(
        edited,
        make_redundant_templates(),
        allow_single_deletion=True,
        top_k_per_start=top_k_per_start,
    )
    ambiguity_sequences = _supported_ambiguity_sequences(
        scanned,
        residual_threshold=residual_threshold,
        near_tie_ratio=near_tie_ratio,
        per_cluster_limit=per_cluster_limit,
        max_sequences=max_sequences,
        min_sequence_support=min_sequence_support,
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
    if not ambiguity_sequences:
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "case_no_ambiguity_sequences",
                    "case_index": case_index,
                    "candidate_count": len(scanned),
                }
            )
        return {
            "case_index": case_index,
            "burst_count": burst_count,
            "message_space_size": message_space_size,
            "wrong_key_count": wrong_key_count,
            "gamma": gamma,
            "noise_std": noise_std,
            "residual_threshold": residual_threshold,
            "near_tie_ratio": near_tie_ratio,
            "per_cluster_limit": per_cluster_limit,
            "top_k_per_start": top_k_per_start,
            "max_sequences": max_sequences,
            "filler_multiplier": filler_multiplier,
            "min_sequence_support": min_sequence_support,
            "truth_sequence_in_ambiguity": False,
            "truth_sequence_covered_by_ambiguity": False,
            "ambiguity_sequence_count": 0,
            "sequence_support_count_min": 0,
            "true_message": true_message,
            "owner_best_message": "",
            "global_best_role": "",
            "score_margin": float("-inf"),
            "parallel_workers": workers,
            "pass": False,
        }
    equalized_by_sequence = [
        _equalized_for_sequence(edited, sequence)
        for sequence in ambiguity_sequences
    ]
    if progress_callback is not None:
        progress_callback(
            {
                "event": "case_scoring_start",
                "case_index": case_index,
                "ambiguity_sequence_count": len(ambiguity_sequences),
                "truth_sequence_in_ambiguity": bool(truth_sequence_indices),
                "truth_sequence_covered_by_ambiguity": bool(truth_covered_sequence_indices),
                "candidate_count": (1 + wrong_key_count) * message_space_size,
                "scoring_mode": scoring_mode,
            }
        )
    wrong_keys = [
        f"wrong_sequence_ambiguity_{burst_count}_{message_space_size}_{case_index}_{index}"
        for index in range(wrong_key_count)
    ]
    candidates = _candidate_space(owner_key, wrong_keys, message_space)
    best_global, best_owner, best_wrong, scored_count, abandoned_count = _score_public_alignment_set(
        equalized_by_sequence,
        candidates,
        sequence_length=len(states),
        burst_plan=burst_plan,
        workers=workers,
        scoring_mode=scoring_mode,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        diagnostic_top_k_global=diagnostic_top_k_global,
        diagnostic_top_k_owner=diagnostic_top_k_owner,
        diagnostic_top_k_per_wrong_key=diagnostic_top_k_per_wrong_key,
    )
    margin = best_owner[2] - best_wrong[2]
    best_wrong_is_upper_bound = best_wrong[1][1] == "wrong_score_upper_bound_placeholder"
    return {
        "case_index": case_index,
        "burst_count": burst_count,
        "message_space_size": message_space_size,
        "wrong_key_count": wrong_key_count,
        "gamma": gamma,
        "noise_std": noise_std,
        "residual_threshold": residual_threshold,
        "near_tie_ratio": near_tie_ratio,
        "per_cluster_limit": per_cluster_limit,
        "max_sequences": max_sequences,
        "filler_multiplier": filler_multiplier,
        "min_sequence_support": min_sequence_support,
        "truth_sequence_in_ambiguity": bool(truth_sequence_indices),
        "truth_sequence_indices": truth_sequence_indices,
        "truth_sequence_covered_by_ambiguity": bool(truth_covered_sequence_indices),
        "truth_covered_sequence_indices": truth_covered_sequence_indices,
        "ambiguity_sequence_count": len(ambiguity_sequences),
        "sequence_support_count_min": min(_cycle_support_count(sequence) for sequence in ambiguity_sequences),
        "sequence_support_count_max": max(_cycle_support_count(sequence) for sequence in ambiguity_sequences),
        "true_message": true_message,
        "owner_best_message": best_owner[1][2],
        "owner_best_sequence_index": best_owner[0],
        "owner_best_score": best_owner[2],
        "best_wrong_message": best_wrong[1][2],
        "best_wrong_sequence_index": best_wrong[0],
        "best_wrong_score": best_wrong[2],
        "best_wrong_score_is_upper_bound": best_wrong_is_upper_bound,
        "global_best_role": best_global[1][0],
        "global_best_message": best_global[1][2],
        "global_best_sequence_index": best_global[0],
        "global_best_score": best_global[2],
        "score_margin": margin,
        "score_margin_is_lower_bound": best_wrong_is_upper_bound,
        "candidate_count": len(candidates),
        "total_exact_score_count": scored_count,
        "bounded_abandoned_count": abandoned_count,
        "parallel_workers": workers,
        "scoring_mode": scoring_mode,
        "diagnostic_top_k_global": diagnostic_top_k_global,
        "diagnostic_top_k_owner": diagnostic_top_k_owner,
        "diagnostic_top_k_per_wrong_key": diagnostic_top_k_per_wrong_key,
        "diagnostic_pruned_search": scoring_mode == "diagnostic_pruned_c",
        "pass": (
            bool(truth_covered_sequence_indices)
            and best_owner[1][2] == true_message
            and best_global[1][0] == "owner"
            and best_global[1][2] == true_message
            and margin > 0.02
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize_ambiguity_cases(cases: list[dict[str, object]]) -> dict[str, object]:
    margins = [float(case["score_margin"]) for case in cases if case["score_margin"] != float("-inf")]
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "truth_sequence_in_ambiguity_count": sum(1 for case in cases if case["truth_sequence_in_ambiguity"]),
        "truth_sequence_covered_by_ambiguity_count": sum(
            1 for case in cases if case["truth_sequence_covered_by_ambiguity"]
        ),
        "owner_message_recovery_count": sum(1 for case in cases if case["owner_best_message"] == case["true_message"]),
        "global_owner_recovery_count": sum(
            1
            for case in cases
            if case["global_best_role"] == "owner" and case["global_best_message"] == case["true_message"]
        ),
        "ambiguity_sequence_count_min": min(int(case["ambiguity_sequence_count"]) for case in cases),
        "ambiguity_sequence_count_max": max(int(case["ambiguity_sequence_count"]) for case in cases),
        "score_margin_mean": _mean(margins),
        "score_margin_min": min(margins) if margins else None,
        "total_exact_score_count_max": max(int(case.get("total_exact_score_count", 0)) for case in cases),
        "cases": cases,
    }


def main() -> None:
    args = _parse_args()
    progress_callback = _make_progress_callback(args.progress_jsonl)
    cases = []
    for index in range(args.start_index, args.start_index + args.case_count):
        case = _sequence_ambiguity_exact_case(
            index,
            burst_count=args.burst_count,
            message_space_size=args.message_space_size,
            wrong_key_count=args.wrong_key_count,
            gamma=args.gamma,
            noise_std=args.noise_std,
            residual_threshold=args.residual_threshold,
            near_tie_ratio=args.near_tie_ratio,
            per_cluster_limit=args.per_cluster_limit,
            top_k_per_start=args.top_k_per_start,
            max_sequences=args.max_sequences,
            filler_multiplier=args.filler_multiplier,
            min_sequence_support=args.min_sequence_support,
            workers=args.workers,
            scoring_mode=args.scoring_mode,
            progress_callback=progress_callback,
            progress_interval=args.progress_interval,
            diagnostic_top_k_global=args.diagnostic_top_k_global,
            diagnostic_top_k_owner=args.diagnostic_top_k_owner,
            diagnostic_top_k_per_wrong_key=args.diagnostic_top_k_per_wrong_key,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "case_finish",
                    "case_index": index,
                    "pass": case["pass"],
                    "score_margin": case["score_margin"],
                    "truth_sequence_covered_by_ambiguity": case["truth_sequence_covered_by_ambiguity"],
                }
            )
        if args.case_jsonl is not None:
            _append_jsonl(args.case_jsonl, {"event": "case_result", "case": case})
        cases.append(case)
    summary = _summarize_ambiguity_cases(cases)
    report = {
        "status": "aisb_sequence_ambiguity_exact_scoring_synthetic_only_no_video_no_gpu_no_claim",
        "sequence_ambiguity_exact_contract": {
            "uses_owner_key_or_message_for_acquisition": False,
            "estimates_affine_channel_during_acquisition": False,
            "owner_and_wrong_keys_share_alignment_set": True,
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
            "near_tie_ratio": args.near_tie_ratio,
            "per_cluster_limit": args.per_cluster_limit,
            "top_k_per_start": args.top_k_per_start,
            "max_sequences": args.max_sequences,
            "start_index": args.start_index,
            "case_count_requested": args.case_count,
            "workers": args.workers,
            "scoring_mode": args.scoring_mode,
            "diagnostic_top_k_global": args.diagnostic_top_k_global,
            "diagnostic_top_k_owner": args.diagnostic_top_k_owner,
            "diagnostic_top_k_per_wrong_key": args.diagnostic_top_k_per_wrong_key,
            "diagnostic_pruned_search": args.scoring_mode == "diagnostic_pruned_c",
            "summary": summary,
        },
        "synthetic_construction_pass": summary["pass_count"] == summary["case_count"],
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
