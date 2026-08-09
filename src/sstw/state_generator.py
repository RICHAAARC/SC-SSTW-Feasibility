"""Frozen key-conditioned two-dimensional SSTW state trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class FrozenStateConfig:
    omega_0: float
    delta_phi: float
    initial_phase: float


def _sign(key: bytes, index: int) -> float:
    digest = hmac.new(key, f"SSTW-state:{index}".encode(), hashlib.sha256).digest()
    return 1.0 if digest[0] & 1 else -1.0


def keyed_trajectory(key: bytes, count: int, delta_times: tuple[float, ...], config: FrozenStateConfig) -> tuple[Vector2, ...]:
    if count <= 0 or len(delta_times) != max(0, count - 1):
        raise ValueError("count and delta_times are inconsistent")
    phase = config.initial_phase
    states: list[Vector2] = []
    for index in range(count):
        states.append((math.cos(phase), math.sin(phase)))
        if index + 1 < count:
            phase += config.omega_0 * delta_times[index] + config.delta_phi * _sign(key, index)
    return tuple(states)
