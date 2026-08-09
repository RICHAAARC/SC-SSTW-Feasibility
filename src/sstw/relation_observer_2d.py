"""Frozen, non-learning saved-MP4 2D relation-observer boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence

Vector2 = tuple[float, float]


def observe_saved_mp4(video_path: str, decoder: Callable[[str], Sequence[object]], relation_readout: Callable[[object], Vector2]) -> tuple[Vector2, ...]:
    """Decode one MP4 and apply a caller-frozen two-coordinate relation readout.

    The decoder/readout must be fixed before S2; this module intentionally offers
    no training, OFF input, key input, or adaptive fitting hook.
    """
    observations = tuple(tuple(map(float, relation_readout(frame))) for frame in decoder(video_path))
    if not observations or any(len(value) != 2 for value in observations):
        raise ValueError("frozen observer must return a non-empty T x 2 observation")
    return observations  # type: ignore[return-value]
