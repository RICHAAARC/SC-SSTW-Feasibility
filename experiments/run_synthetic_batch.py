"""Run a small CPU-only robustness batch for the synthetic SC-SSTW probe."""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.acquisition import acquire_pilots_by_periodic_beam
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from sc_sstw_feasibility.scoring import score_key, score_key_with_calibration
from sc_sstw_feasibility.state import PUBLIC_SYNC_DIRECTIONS, PilotPattern, generate_state_sequence


ASYMMETRIC_PILOT_DIRECTIONS = (
    (1.0, 0.0),
    (0.31, 0.95),
    (-0.82, 0.57),
    (-0.19, -0.98),
    (0.91, -0.41),
)


def _edited_observations[T](values: list[T], *, crop_start: int, crop_end: int, deleted_local_indices: set[int]) -> tuple[list[T], list[int]]:
    edited_values = []
    source_indices = []
    for source_index, value in enumerate(values[crop_start:crop_end], start=crop_start):
        local_index = source_index - crop_start
        if local_index in deleted_local_indices:
            continue
        edited_values.append(value)
        source_indices.append(source_index)
    return edited_values, source_indices


def _case(case_index: int, *, pattern: PilotPattern) -> dict[str, object]:
    owner_key = f"owner_key_{case_index % 3}"
    wrong_keys = [f"wrong_key_{case_index}_{index}" for index in range(10)]
    length = 64
    states = generate_state_sequence(owner_key, length, pilot_pattern=pattern)
    channel = make_random_channel(1000 + case_index, relation_count=16, noise_std=0.035)
    observations = generate_observations(states, channel, seed=2000 + case_index)
    crop_start = 2 + (case_index % 6)
    crop_end = 58 + (case_index % 5)
    deleted = {
        5 + (case_index % 3),
        12 + (case_index % 5),
        24 + (case_index % 7),
        39 + (case_index % 4),
    }
    edited, source_indices = _edited_observations(
        observations,
        crop_start=crop_start,
        crop_end=crop_end,
        deleted_local_indices=deleted,
    )

    oracle_owner, oracle_condition, oracle_mse = score_key(
        edited,
        owner_key,
        pilot_pattern=pattern,
        candidate_length=length,
        source_indices=source_indices,
    )
    oracle_wrong = [
        score_key(
            edited,
            wrong_key,
            pilot_pattern=pattern,
            candidate_length=length,
            source_indices=source_indices,
        )[0].score
        for wrong_key in wrong_keys
    ]
    acquired = acquire_pilots_by_periodic_beam(
        edited,
        pilot_pattern=pattern,
        beam_width=80,
    )
    acquired_owner = score_key_with_calibration(
        edited,
        owner_key,
        calibration=acquired.calibration,
        pilot_pattern=pattern,
        candidate_length=length,
    )
    acquired_wrong = [
        score_key_with_calibration(
            edited,
            wrong_key,
            calibration=acquired.calibration,
            pilot_pattern=pattern,
            candidate_length=length,
        ).score
        for wrong_key in wrong_keys
    ]
    label_checks = []
    for observed_index, acquired_direction_index in zip(
        acquired.observed_indices,
        acquired.pilot_direction_indices,
        strict=True,
    ):
        source_index = source_indices[observed_index]
        if source_index % pattern.period != 0:
            label_checks.append(False)
            continue
        pilot_ordinal = source_index // pattern.period
        if pattern.cyclic:
            expected_direction_index = pilot_ordinal % len(pattern.directions)
        elif pilot_ordinal < len(pattern.directions):
            expected_direction_index = pilot_ordinal
        else:
            label_checks.append(False)
            continue
        label_checks.append(acquired_direction_index == expected_direction_index)
    label_alignment = sum(1 for value in label_checks if value) / max(1, len(label_checks))
    return {
        "case_index": case_index,
        "edited_observation_count": len(edited),
        "oracle_margin": oracle_owner.score - max(oracle_wrong),
        "oracle_condition_number": oracle_condition,
        "oracle_pilot_mse": oracle_mse,
        "oracle_pass": oracle_owner.score > max(oracle_wrong) and oracle_mse < 0.01,
        "acquired_margin": acquired_owner.score - max(acquired_wrong),
        "acquired_condition_number": acquired.calibration.condition_number,
        "acquired_pilot_mse": acquired.calibration.pilot_reconstruction_mse,
        "acquired_pilot_count": len(acquired.observed_indices),
        "acquired_label_alignment": label_alignment,
        "acquired_pass": (
            acquired_owner.score > max(acquired_wrong)
            and acquired.calibration.pilot_reconstruction_mse < 0.01
            and acquired.calibration.condition_number < 10.0
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _run_batch(pattern_name: str, pattern: PilotPattern) -> dict[str, object]:
    cases = [_case(index, pattern=pattern) for index in range(4)]
    acquired_margins = [float(case["acquired_margin"]) for case in cases]
    oracle_margins = [float(case["oracle_margin"]) for case in cases]
    return {
        "pattern_name": pattern_name,
        "case_count": len(cases),
        "oracle_pass_count": sum(1 for case in cases if case["oracle_pass"]),
        "acquired_pass_count": sum(1 for case in cases if case["acquired_pass"]),
        "oracle_margin_mean": _mean(oracle_margins),
        "oracle_margin_min": min(oracle_margins),
        "acquired_margin_mean": _mean(acquired_margins),
        "acquired_margin_min": min(acquired_margins),
        "cases": cases,
    }


def main() -> None:
    report = {
        "status": "synthetic_batch_only_no_video_no_gpu_no_claim",
        "batches": [
            _run_batch("cardinal_four_direction_cycle", PilotPattern(period=4)),
            _run_batch(
                "asymmetric_five_direction_cycle",
                PilotPattern(period=4, directions=ASYMMETRIC_PILOT_DIRECTIONS),
            ),
            _run_batch(
                "public_sync_noncyclic_code",
                PilotPattern(
                    period=4,
                    directions=PUBLIC_SYNC_DIRECTIONS,
                    cyclic=False,
                ),
            ),
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
