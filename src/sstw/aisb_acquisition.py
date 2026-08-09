"""Single-template affine-invariant public AISB capture."""

from __future__ import annotations

from dataclasses import dataclass

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class AISBTemplate:
    points: tuple[Vector2, Vector2, Vector2, Vector2]
    barycentric: tuple[float, float, float]
    threshold: float
    max_candidates: int


@dataclass(frozen=True)
class PublicCandidate:
    start: int
    residual: float


def _residual(window: tuple[Vector2, Vector2, Vector2, Vector2], weights: tuple[float, float, float]) -> float:
    predicted = tuple(sum(weights[k] * window[k][axis] for k in range(3)) for axis in range(2))
    return sum((window[3][axis] - predicted[axis]) ** 2 for axis in range(2))


def capture_candidates(observations: tuple[Vector2, ...], template: AISBTemplate) -> tuple[PublicCandidate, ...]:
    """Capture public candidates without a channel fit or key access."""
    scored = [
        PublicCandidate(start, _residual(tuple(observations[start : start + 4]), template.barycentric))
        for start in range(max(0, len(observations) - 3))
    ]
    accepted = sorted((item for item in scored if item.residual <= template.threshold), key=lambda item: (item.residual, item.start))
    return tuple(accepted[: template.max_candidates])
