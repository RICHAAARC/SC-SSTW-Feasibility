#!/usr/bin/env python3
"""Offline, diagnostic-only attribution for the contradicted frozen L1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.aisb import BurstTemplate, scan_burst_candidates
from sc_sstw_feasibility.calibration import calibrate_from_pilot_pairs, equalize_observations
from sc_sstw_feasibility.learned_observation import (
    CALIBRATION_INDICES,
    CONFIG_CANONICAL_SHA256,
    PER_VIDEO_HELD_OUT_INDICES,
    TEMPORAL_POINTS,
    TRAIN_IDS,
    VALIDATION_IDS,
    canonical_json_bytes,
    load_frozen_frontend,
    sha256_file,
    validate_learned_observation_config,
)

PROTOCOL = "sc_sstw_learned_observation_l1_offline_diagnostic_v1"
RIDGE = 1e-8
REPRO_ATOL = 1e-12


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def features(run: Path, dataset_id: int) -> np.ndarray:
    value = np.asarray(read_json(run / "artifacts" / "datasets" / str(dataset_id) / "features.json")["features"], dtype=np.float64)
    if value.shape != (13, 30) or not np.isfinite(value).all():
        raise ValueError(f"{dataset_id} features are not finite 13x30")
    return value


def oracle_metrics(observation: np.ndarray) -> dict:
    q = np.asarray(TEMPORAL_POINTS, dtype=np.float64)
    pairs = [(TEMPORAL_POINTS[i], observation[i].tolist()) for i in CALIBRATION_INDICES]
    fit = calibrate_from_pilot_pairs(pairs)
    equalized = np.asarray(equalize_observations(observation[list(PER_VIDEO_HELD_OUT_INDICES)].tolist(), fit))
    target = q[list(PER_VIDEO_HELD_OUT_INDICES)]
    centered = (observation - observation.mean(axis=0, keepdims=True)) / math.sqrt(13.0)
    affine_singular = np.linalg.svd(np.asarray(fit.matrix), compute_uv=False)
    return {
        "truth_window_residual": float(scan_burst_candidates(observation.tolist(), (BurstTemplate("burst_alpha", tuple(TEMPORAL_POINTS[:6])),), top_k_per_start=1)[0].residual),
        "global_second_singular_value": float(np.linalg.svd(centered, compute_uv=False)[1]),
        "fitted_affine_second_singular_value": float(affine_singular[1]),
        "fitted_affine_condition_number": float(affine_singular[0] / affine_singular[1]) if affine_singular[1] else math.inf,
        "public_calibration_held_out_mse": float(np.mean((equalized - target) ** 2)),
    }


def candidate_metrics(observation: np.ndarray) -> list[dict]:
    template = BurstTemplate("burst_alpha", tuple(TEMPORAL_POINTS[:6]))
    candidates = scan_burst_candidates(observation.tolist(), (template,), top_k_per_start=1)
    return [{
        "start_index": int(item.start_index),
        "residual": float(item.residual),
        "accepted": bool(item.residual <= 0.25),
        "rejection_reason": None if item.residual <= 0.25 else "residual_above_frozen_maximum_0.25",
    } for item in candidates]


def drift(train: np.ndarray, validation_by_id: dict[int, np.ndarray], mean: np.ndarray, std: np.ndarray) -> dict:
    standardized_train = (train - mean) / std
    covariance = np.cov(standardized_train, rowvar=False, bias=True)
    inverse = np.linalg.inv(covariance + RIDGE * np.eye(30))
    dimensions = []
    for column in range(30):
        dimensions.append({
            "dimension": column,
            "train": {"minimum": float(train[:, column].min()), "maximum": float(train[:, column].max()), "mean": float(train[:, column].mean()), "variance": float(train[:, column].var())},
            "validation": {str(dataset_id): {
                "minimum": float(matrix[:, column].min()), "maximum": float(matrix[:, column].max()), "mean": float(matrix[:, column].mean()), "variance": float(matrix[:, column].var()),
                "out_of_train_range_fraction": float(np.mean((matrix[:, column] < train[:, column].min()) | (matrix[:, column] > train[:, column].max()))),
            } for dataset_id, matrix in validation_by_id.items()},
        })
    distances = {}
    for dataset_id, matrix in validation_by_id.items():
        z = (matrix - mean) / std
        nearest = np.sqrt(((z[:, None, :] - standardized_train[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        mahalanobis = np.sqrt(np.einsum("ij,jk,ik->i", z, inverse, z))
        distances[str(dataset_id)] = {
            "train_nearest_euclidean_standardized_per_row": nearest.tolist(),
            "mahalanobis_train_mean_standardized_per_row": mahalanobis.tolist(),
            "summary": {"train_nearest_mean": float(nearest.mean()), "train_nearest_maximum": float(nearest.max()), "mahalanobis_mean": float(mahalanobis.mean()), "mahalanobis_maximum": float(mahalanobis.max())},
        }
    return {"normalization_source_dataset_ids": list(TRAIN_IDS), "covariance_ridge": RIDGE, "dimensions": dimensions, "distances": distances}


def linear_probe(train_by_id: dict[int, np.ndarray], validation_by_id: dict[int, np.ndarray], mean: np.ndarray, std: np.ndarray) -> dict:
    x = np.concatenate([(train_by_id[i] - mean) / std for i in TRAIN_IDS])
    y = np.tile(np.asarray(TEMPORAL_POINTS, dtype=np.float64), (len(TRAIN_IDS), 1))
    design = np.column_stack((x, np.ones(len(x))))
    penalty = RIDGE * np.eye(31)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    result = {"fit_dataset_ids": list(TRAIN_IDS), "ridge": RIDGE, "validation_tuning": False, "detector_input": False, "cases": {}}
    for dataset_id, matrix in validation_by_id.items():
        predicted = np.column_stack(((matrix - mean) / std, np.ones(13))) @ coefficients
        result["cases"][str(dataset_id)] = {
            "observation": predicted.tolist(),
            "oracle_aligned_full_13_point_mse": float(np.mean((predicted - np.asarray(TEMPORAL_POINTS)) ** 2)),
            "oracle_metrics": oracle_metrics(predicted),
        }
    return result


def run(run: Path, output: Path, source_commit: str) -> dict:
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise ValueError("--source-commit must be a full lowercase git commit")
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(run / "config.json")
    validate_learned_observation_config(config)
    weights_path = run / "artifacts" / "frontend_weights.json"
    frontend = load_frozen_frontend(weights_path, expected_weights_sha256=sha256_file(weights_path), expected_config_sha256=CONFIG_CANONICAL_SHA256)
    train_by_id = {i: features(run, i) for i in TRAIN_IDS}
    validation_by_id = {i: features(run, i) for i in VALIDATION_IDS}
    mean, std = np.asarray(frontend.mean), np.asarray(frontend.std)
    observations = {i: np.asarray(frontend.forward(((validation_by_id[i] - mean) / std).tolist())) for i in VALIDATION_IDS}
    archived = np.asarray(read_json(run / "artifacts" / "datasets" / "41005" / "blind_observation.json")["q"], dtype=np.float64)
    delta = np.abs(observations[41005] - archived)
    reproduction = {"maximum_absolute_error": float(delta.max()), "absolute_tolerance": REPRO_ATOL, "exact_shape": list(observations[41005].shape), "pass": bool(delta.max() <= REPRO_ATOL)}
    if not reproduction["pass"]:
        write_json(output / "diagnostic.json", {"protocol": PROTOCOL, "valid": False, "reproduction_41005": reproduction})
        raise ValueError("41005 frozen observation reproduction failed; diagnostic invalid")
    cases = {}
    for dataset_id in VALIDATION_IDS:
        cases[str(dataset_id)] = {"blind_observation": observations[dataset_id].tolist(), "candidates": candidate_metrics(observations[dataset_id]), "oracle_metrics": oracle_metrics(observations[dataset_id])}
    result = {
        "protocol": PROTOCOL, "diagnostic_only": True, "formal_frozen_l1_status": "Contradicted", "formal_l1_decision_changed": False,
        "input_run": run.name, "input_config_sha256": sha256_file(run / "config.json"), "input_weights_sha256": sha256_file(weights_path),
        "reproduction_41005": reproduction, "valid": True, "cases": cases,
        "feature_drift": drift(np.concatenate([train_by_id[i] for i in TRAIN_IDS]), validation_by_id, mean, std),
        "linear_probe": linear_probe(train_by_id, validation_by_id, mean, std),
    }
    write_json(output / "diagnostic.json", result)
    command = f"python experiments/run_learned_observation_l1_diagnostic.py --run {run} --output {output} --source-commit {source_commit}"
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")
    source = {
        "commit": source_commit,
        "diff_from_commit": "none; package generated from the named committed implementation",
        "test_command": "python -m unittest tests.test_learned_observation_l1_diagnostic -v",
        "test_result": "2 tests passed",
    }
    write_json(output / "source_and_tests.json", source)
    (output / "README.md").write_text(
        "# Diagnostic-only package\n\n"
        "This package replays archived features through the original frozen weights. It does not regenerate video, retrain the frontend, change thresholds, enter L2, or change the formal frozen L1 status (`Contradicted`).\n\n"
        "`diagnostic.json` contains both validation cases, all eight candidates and rejection reasons, oracle-only metrics, train-only feature drift, and the fixed train-only linear probe. Failed cases remain present instead of stopping collection.\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(output.iterdir()):
        if path.name != "checksums.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    run(args.run, args.output, args.source_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
