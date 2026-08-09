"""Bounded state-constrained Viterbi scorer; ordinary DTW is not a detector."""

from __future__ import annotations

from dataclasses import dataclass
import math

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class ViterbiConfig:
    max_skip: int
    emission_scale: float
    state_penalty: float
    edit_penalty: float


def _angle(value: Vector2) -> float:
    return math.atan2(value[1], value[0])


def _circle_distance(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


def score_state_trajectory(observations: tuple[Vector2, ...], trajectory: tuple[Vector2, ...], config: ViterbiConfig) -> float:
    if not observations or not trajectory or config.max_skip < 0 or config.emission_scale <= 0:
        raise ValueError("invalid Viterbi inputs")
    previous: dict[int, float] = {}
    for state, expected in enumerate(trajectory):
        previous[state] = -sum((observations[0][axis] - expected[axis]) ** 2 for axis in range(2)) / config.emission_scale
    for index in range(1, len(observations)):
        current: dict[int, float] = {}
        for state, expected in enumerate(trajectory):
            emission = -sum((observations[index][axis] - expected[axis]) ** 2 for axis in range(2)) / config.emission_scale
            choices: list[float] = []
            for prior, prior_score in previous.items():
                delta = state - prior
                if delta < 0 or delta > config.max_skip + 1:
                    continue
                observed_delta = _circle_distance(_angle(observations[index]), _angle(observations[index - 1]))
                expected_delta = _circle_distance(_angle(trajectory[state]), _angle(trajectory[prior]))
                choices.append(prior_score - config.state_penalty * _circle_distance(observed_delta, expected_delta) ** 2 - config.edit_penalty * abs(delta - 1))
            if choices:
                current[state] = emission + max(choices)
        if not current:
            raise ValueError("no legal bounded Viterbi path")
        previous = current
    return max(previous.values())
