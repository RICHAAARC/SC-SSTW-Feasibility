"""Optional local C hot path for exact CPU-only DTW scoring.

This module compiles a tiny C helper into `/tmp` on demand. It is an
implementation accelerator only: the recurrence, tie-break order, score
normalization, and bounded-abandon rule match `sync.py`.
"""

from __future__ import annotations

from ctypes import CDLL, POINTER, byref, c_double, c_int
from dataclasses import dataclass
from pathlib import Path
import hashlib
import subprocess
import tempfile


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "src" / "sc_sstw_feasibility" / "sync_fast.c"
_BUILD_DIR = Path(tempfile.gettempdir()) / "sc_sstw_feasibility_native"


def _source_digest() -> str:
    return hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]


def _library_path() -> Path:
    return _BUILD_DIR / f"sync_fast_{_source_digest()}.so"


def _compile_library() -> Path:
    output = _library_path()
    if output.exists():
        return output
    _BUILD_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.so")
    subprocess.run(
        [
            "gcc",
            "-O3",
            "-fPIC",
            "-shared",
            "-o",
            str(temporary),
            str(_SOURCE),
        ],
        check=True,
    )
    temporary.replace(output)
    return output


_LIBRARY: CDLL | None = None


def _library() -> CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = CDLL(str(_compile_library()))
        _LIBRARY.sc_sstw_dynamic_time_sync_score_bounded_flat.argtypes = [
            POINTER(c_double),
            c_int,
            POINTER(c_double),
            c_int,
            c_double,
            c_double,
            c_double,
            POINTER(c_double),
            POINTER(c_int),
        ]
        _LIBRARY.sc_sstw_dynamic_time_sync_score_bounded_flat.restype = c_int
        _LIBRARY.sc_sstw_dynamic_time_sync_score_bounded_flat_workspace.argtypes = [
            POINTER(c_double),
            c_int,
            POINTER(c_double),
            c_int,
            c_double,
            c_double,
            c_double,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_int),
            POINTER(c_int),
            POINTER(c_double),
            POINTER(c_int),
        ]
        _LIBRARY.sc_sstw_dynamic_time_sync_score_bounded_flat_workspace.restype = c_int
        _LIBRARY.sc_sstw_score_candidates_margin_proof_workspace.argtypes = [
            POINTER(c_double),
            c_int,
            POINTER(c_double),
            POINTER(c_int),
            c_int,
            c_int,
            c_double,
            c_double,
            c_double,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_int),
            POINTER(c_int),
            POINTER(c_int),
            POINTER(c_double),
            POINTER(c_int),
            POINTER(c_double),
            POINTER(c_int),
            POINTER(c_int),
        ]
        _LIBRARY.sc_sstw_score_candidates_margin_proof_workspace.restype = c_int
    return _LIBRARY


@dataclass(frozen=True)
class CFlatSequence:
    """Prepared C array for repeated exact native DTW scoring."""

    values: object
    pair_count: int


@dataclass(frozen=True)
class CFlatCandidateMatrix:
    """Prepared row-major candidate matrix for native batch scoring."""

    values: object
    roles: object
    candidate_count: int
    pair_count: int


@dataclass(frozen=True)
class CBatchMarginProofResult:
    best_owner_index: int
    best_owner_score: float
    best_wrong_index: int
    best_wrong_score: float
    scored_count: int
    abandoned_count: int


@dataclass(frozen=True)
class CDtwWorkspace:
    """Reusable native DP workspace for same-thread repeated scoring."""

    previous_costs: object
    current_costs: object
    previous_counts: object
    current_counts: object


def as_c_flat_sequence(flattened_xy: list[float]) -> CFlatSequence:
    """Convert flattened `[x0, y0, ...]` once for repeated native calls."""

    return CFlatSequence(
        values=(c_double * len(flattened_xy))(*flattened_xy),
        pair_count=len(flattened_xy) // 2,
    )


def as_c_flat_candidate_matrix(flattened_candidates: list[list[float]], roles: list[int]) -> CFlatCandidateMatrix:
    """Convert same-length candidates once for native batch scoring."""

    if len(flattened_candidates) != len(roles):
        raise ValueError("candidate and role counts must match")
    if not flattened_candidates:
        raise ValueError("candidate matrix must not be empty")
    width = len(flattened_candidates[0])
    for candidate in flattened_candidates:
        if len(candidate) != width:
            raise ValueError("candidate matrix rows must have equal length")
    flattened_values = [value for candidate in flattened_candidates for value in candidate]
    return CFlatCandidateMatrix(
        values=(c_double * len(flattened_values))(*flattened_values),
        roles=(c_int * len(roles))(*roles),
        candidate_count=len(flattened_candidates),
        pair_count=width // 2,
    )


def make_c_dtw_workspace(max_pair_count: int) -> CDtwWorkspace:
    """Allocate reusable DP rows for native bounded DTW scoring."""

    if max_pair_count < 0:
        raise ValueError("max_pair_count must be non-negative")
    size = max_pair_count + 1
    return CDtwWorkspace(
        previous_costs=(c_double * size)(),
        current_costs=(c_double * size)(),
        previous_counts=(c_int * size)(),
        current_counts=(c_int * size)(),
    )


def score_candidates_margin_proof_workspace_c(
    observed: CFlatSequence,
    candidates: CFlatCandidateMatrix,
    workspace: CDtwWorkspace,
    *,
    diagnostic_margin: float = 0.020000000001,
    skip_penalty: float = 0.22,
    repeat_penalty: float = 0.14,
) -> CBatchMarginProofResult:
    best_owner_index = c_int(-1)
    best_owner_score = c_double(0.0)
    best_wrong_index = c_int(-1)
    best_wrong_score = c_double(0.0)
    scored_count = c_int(0)
    abandoned_count = c_int(0)
    result = _library().sc_sstw_score_candidates_margin_proof_workspace(
        observed.values,
        observed.pair_count,
        candidates.values,
        candidates.roles,
        candidates.candidate_count,
        candidates.pair_count,
        diagnostic_margin,
        skip_penalty,
        repeat_penalty,
        workspace.previous_costs,
        workspace.current_costs,
        workspace.previous_counts,
        workspace.current_counts,
        byref(best_owner_index),
        byref(best_owner_score),
        byref(best_wrong_index),
        byref(best_wrong_score),
        byref(scored_count),
        byref(abandoned_count),
    )
    if result != 0:
        raise MemoryError("native batch margin proof scorer failed")
    return CBatchMarginProofResult(
        best_owner_index=best_owner_index.value,
        best_owner_score=best_owner_score.value,
        best_wrong_index=best_wrong_index.value,
        best_wrong_score=best_wrong_score.value,
        scored_count=scored_count.value,
        abandoned_count=abandoned_count.value,
    )


def dynamic_time_sync_score_bounded_prepared_c(
    observed: CFlatSequence,
    candidate: CFlatSequence,
    *,
    min_score_to_beat: float,
    skip_penalty: float = 0.22,
    repeat_penalty: float = 0.14,
) -> tuple[float, bool]:
    score = c_double(0.0)
    abandoned = c_int(0)
    result = _library().sc_sstw_dynamic_time_sync_score_bounded_flat(
        observed.values,
        observed.pair_count,
        candidate.values,
        candidate.pair_count,
        min_score_to_beat,
        skip_penalty,
        repeat_penalty,
        byref(score),
        byref(abandoned),
    )
    if result != 0:
        raise MemoryError("native DTW scorer allocation failed")
    return score.value, bool(abandoned.value)


def dynamic_time_sync_score_bounded_prepared_workspace_c(
    observed: CFlatSequence,
    candidate: CFlatSequence,
    workspace: CDtwWorkspace,
    *,
    min_score_to_beat: float,
    skip_penalty: float = 0.22,
    repeat_penalty: float = 0.14,
) -> tuple[float, bool]:
    score = c_double(0.0)
    abandoned = c_int(0)
    result = _library().sc_sstw_dynamic_time_sync_score_bounded_flat_workspace(
        observed.values,
        observed.pair_count,
        candidate.values,
        candidate.pair_count,
        min_score_to_beat,
        skip_penalty,
        repeat_penalty,
        workspace.previous_costs,
        workspace.current_costs,
        workspace.previous_counts,
        workspace.current_counts,
        byref(score),
        byref(abandoned),
    )
    if result != 0:
        raise MemoryError("native DTW scorer allocation failed")
    return score.value, bool(abandoned.value)


def dynamic_time_sync_score_bounded_flat_c(
    observed_xy: list[float],
    candidate_xy: list[float],
    *,
    min_score_to_beat: float,
    skip_penalty: float = 0.22,
    repeat_penalty: float = 0.14,
) -> tuple[float, bool]:
    return dynamic_time_sync_score_bounded_prepared_c(
        as_c_flat_sequence(observed_xy),
        as_c_flat_sequence(candidate_xy),
        min_score_to_beat=min_score_to_beat,
        skip_penalty=skip_penalty,
        repeat_penalty=repeat_penalty,
    )
