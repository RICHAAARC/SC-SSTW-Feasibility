"""Run the first CPU-only SC-SSTW synthetic feasibility probe."""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.acquisition import acquire_pilots_by_periodic_beam
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from sc_sstw_feasibility.scoring import score_key, score_key_with_calibration
from sc_sstw_feasibility.state import PilotPattern, generate_state_sequence


def delete_and_crop[T](values: list[T]) -> tuple[list[T], list[int]]:
    edited_values = []
    source_indices = []
    for source_index, value in enumerate(values[4:60], start=4):
        local_index = source_index - 4
        if local_index in {7, 8, 21, 35}:
            continue
        edited_values.append(value)
        source_indices.append(source_index)
    return edited_values, source_indices


def main() -> None:
    owner_key = "owner_key_example"
    wrong_keys = [f"wrong_key_{index}" for index in range(12)]
    pattern = PilotPattern(period=4)
    owner_states = generate_state_sequence(owner_key, 64, pilot_pattern=pattern)
    channel = make_random_channel(20260730, relation_count=16, noise_std=0.03)
    observations = generate_observations(owner_states, channel, seed=17)
    edited_observations, source_indices = delete_and_crop(observations)

    oracle_owner_score, condition_number, pilot_mse = score_key(
        edited_observations,
        owner_key,
        pilot_pattern=pattern,
        candidate_length=64,
        source_indices=source_indices,
    )
    oracle_wrong_scores = [
        score_key(
            edited_observations,
            wrong_key,
            pilot_pattern=pattern,
            candidate_length=64,
            source_indices=source_indices,
        )[0].score
        for wrong_key in wrong_keys
    ]
    acquired = acquire_pilots_by_periodic_beam(
        edited_observations,
        pilot_pattern=pattern,
    )
    acquired_owner_score = score_key_with_calibration(
        edited_observations,
        owner_key,
        calibration=acquired.calibration,
        pilot_pattern=pattern,
        candidate_length=64,
    )
    acquired_wrong_scores = [
        score_key_with_calibration(
            edited_observations,
            wrong_key,
            calibration=acquired.calibration,
            pilot_pattern=pattern,
            candidate_length=64,
        ).score
        for wrong_key in wrong_keys
    ]
    report = {
        "status": "synthetic_probe_only_no_video_no_gpu_no_claim",
        "oracle": {
            "pilot_reacquisition": "oracle_source_indices",
            "owner_score": oracle_owner_score.score,
            "best_wrong_score": max(oracle_wrong_scores),
            "score_margin": oracle_owner_score.score - max(oracle_wrong_scores),
            "channel_condition_number": condition_number,
            "pilot_reconstruction_mse": pilot_mse,
            "feasibility_pass": (
                oracle_owner_score.score > max(oracle_wrong_scores)
                and condition_number < 10.0
                and pilot_mse < 0.01
            ),
        },
        "acquired": {
            "pilot_reacquisition": "periodic_beam_observation_only",
            "owner_score": acquired_owner_score.score,
            "best_wrong_score": max(acquired_wrong_scores),
            "score_margin": acquired_owner_score.score - max(acquired_wrong_scores),
            "channel_condition_number": acquired.calibration.condition_number,
            "pilot_reconstruction_mse": acquired.calibration.pilot_reconstruction_mse,
            "acquired_pilot_count": len(acquired.observed_indices),
            "acquisition_score": acquired.acquisition_score,
            "feasibility_pass": (
                acquired_owner_score.score > max(acquired_wrong_scores)
                and acquired.calibration.condition_number < 10.0
                and acquired.calibration.pilot_reconstruction_mse < 0.01
            ),
        },
        "edited_observation_count": len(edited_observations),
        "wrong_key_count": len(wrong_keys),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
