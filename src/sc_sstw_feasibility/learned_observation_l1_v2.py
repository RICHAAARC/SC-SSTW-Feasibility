"""Fail-closed CPU development gate for the frozen Observation L1-v2 protocol.

This module is implementation infrastructure only. It does not authorize a
formal CPU development run, fresh held-out access, GPU work, or any method
claim. A run is valid only when an independent authorization manifest matches
the repository and all six permitted feature inputs exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "sc_sstw_learned_observation_l1_v2_development"
MANIFEST_SCHEMA_VERSION = 1
OUTPUT_SCHEMA = "sc_sstw_l1_v2_development_audit_v1"
FIT_IDS = (41001, 41002, 41003, 41004)
DEVELOPMENT_IDS = (41005, 41006)
FORBIDDEN_HELD_OUT_IDS = (41007, 41008)
PERMITTED_INPUT_IDS = FIT_IDS + DEVELOPMENT_IDS
CANDIDATE_ORDER = ("A1", "A2")
START_INDICES = tuple(range(8))
CORRECT_START_INDEX = 0
SUCCESS_STATUS = "READY_TO_PREREGISTER_FRESH_GPU_GATE"
FAILURE_STATUS = "STOP_DEVELOPMENT_GATE_FAILED"
INVALID_STATUS = "INVALID_EXPERIMENT"
CONFIG_PATH = "configs/learned_observation_l1_v2_development.json"
RUNNER_PATH = "experiments/run_learned_observation_l1_v2_development.py"
PROTOCOL_PATH = "protocols/gpu_learned_observation_l1_v2.md"
LIBRARY_PATH = "src/sc_sstw_feasibility/learned_observation_l1_v2.py"
TEST_PATH = "tests/test_learned_observation_l1_v2_development.py"
REQUIRED_SOURCE_PATHS = (CONFIG_PATH, RUNNER_PATH, PROTOCOL_PATH, LIBRARY_PATH, TEST_PATH)

ABSOLUTE_THRESHOLDS = {
    "max_residual": 0.25,
    "min_global_s2": 0.10,
    "min_affine_s2": 0.05,
    "max_condition": 10.0,
    "max_held_out_mse": 0.02,
}

TEMPORAL_POINTS = (
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (0.25, 0.35),
    (0.75, 0.2),
    (0.2, 0.8),
    (-0.9238795325, 0.3826834324),
    (0.3826834324, 0.9238795325),
    (-0.3826834324, -0.9238795325),
    (0.9238795325, -0.3826834324),
    (-0.7071067812, 0.7071067812),
    (0.7071067812, 0.7071067812),
    (0.0, -1.0),
)
CALIBRATION_INDICES = (0, 1, 2, 3)
PER_VIDEO_HELD_OUT_INDICES = (4, 5)

FROZEN_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "status": "development_only",
    "partitions": {
        "training_loo": list(FIT_IDS),
        "development_diagnosis": list(DEVELOPMENT_IDS),
        "held_out_forbidden": list(FORBIDDEN_HELD_OUT_IDS),
    },
    "candidate_order": list(CANDIDATE_ORDER),
    "candidate_rule": "A1_then_A2_only_if_A1_fails_then_stop",
    "normalization": {
        "mad_scale": 1.4826,
        "mad_floor": 1e-6,
        "clip_min": -6.0,
        "clip_max": 6.0,
    },
    "readout": {
        "kind": "linear_affine",
        "ridge": 1e-6,
        "fit_intercept": True,
        "target": "public_13_point_temporal_schedule",
    },
    "acquisition": {
        "template_length": 6,
        "candidate_start_indices": list(START_INDICES),
        "correct_start_index": CORRECT_START_INDEX,
        "deletions": False,
    },
    "absolute_thresholds": ABSOLUTE_THRESHOLDS,
    "states": {
        "success": SUCCESS_STATUS,
        "failure": FAILURE_STATUS,
        "invalid": INVALID_STATUS,
    },
    "output_schema": OUTPUT_SCHEMA,
    "old_l1_status": "Contradicted",
    "formal_result": False,
    "stage_progression_allowed": False,
}


class InvalidExperiment(RuntimeError):
    """A stable fail-closed precondition or integrity failure."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class SourceState:
    head: str
    tree: str
    dirty: bool
    status_porcelain: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "git_readable": True,
            "head": self.head,
            "tree": self.tree,
            "dirty": self.dirty,
            "status_porcelain": list(self.status_porcelain),
        }


@dataclass(frozen=True)
class Preflight:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    config: dict[str, Any]
    config_bytes: bytes
    source_state: SourceState
    source_hashes: dict[str, str]
    input_hashes: dict[str, str]
    features: dict[int, np.ndarray]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    if dict(config) != FROZEN_CONFIG:
        raise InvalidExperiment("CONFIG_NOT_FROZEN", "configuration differs from the immutable L1-v2 protocol")


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise InvalidExperiment("GIT_STATE_UNREADABLE", "Git source state could not be read") from exc
    return completed.stdout.strip()


def read_source_state(repo_root: Path) -> SourceState:
    head = _git(repo_root, "rev-parse", "HEAD")
    tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    status_text = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if not _is_commit(head) or not _is_commit(tree):
        raise InvalidExperiment("GIT_STATE_UNREADABLE", "Git returned a malformed HEAD or tree")
    status = tuple(line for line in status_text.splitlines() if line)
    return SourceState(head=head, tree=tree, dirty=bool(status), status_porcelain=status)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], reason: str) -> None:
    if set(value) != expected:
        raise InvalidExperiment(reason, "object keys do not match the frozen schema")


def validate_manifest_schema(manifest: Mapping[str, Any]) -> None:
    _require_exact_keys(
        manifest,
        {"schema_version", "protocol_id", "expected_source", "source_files", "config", "inputs", "command_schema", "output_schema"},
        "MANIFEST_SCHEMA_MISMATCH",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION or manifest["protocol_id"] != PROTOCOL_ID:
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "manifest version or protocol changed")
    if manifest["output_schema"] != OUTPUT_SCHEMA:
        raise InvalidExperiment("OUTPUT_SCHEMA_MISMATCH", "output schema is not authorized")

    expected_source = manifest["expected_source"]
    if not isinstance(expected_source, Mapping):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "expected_source must be an object")
    _require_exact_keys(expected_source, {"head", "tree"}, "MANIFEST_SCHEMA_MISMATCH")
    if not _is_commit(expected_source["head"]) or not _is_commit(expected_source["tree"]):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "expected HEAD or tree is malformed")

    source_files = manifest["source_files"]
    if not isinstance(source_files, Mapping) or tuple(sorted(source_files)) != tuple(sorted(REQUIRED_SOURCE_PATHS)):
        raise InvalidExperiment("SOURCE_FILE_SET_MISMATCH", "source file authorization set changed")
    if not all(_is_sha256(digest) for digest in source_files.values()):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "source file digest is malformed")

    config = manifest["config"]
    if not isinstance(config, Mapping):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "config identity must be an object")
    _require_exact_keys(config, {"path", "sha256"}, "MANIFEST_SCHEMA_MISMATCH")
    if config["path"] != CONFIG_PATH or not _is_sha256(config["sha256"]):
        raise InvalidExperiment("CONFIG_IDENTITY_MISMATCH", "config identity is not frozen")

    inputs = manifest["inputs"]
    if not isinstance(inputs, list) or not all(isinstance(item, Mapping) for item in inputs):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "inputs must be a list of objects")
    for item in inputs:
        _require_exact_keys(item, {"dataset_id", "path", "sha256"}, "MANIFEST_SCHEMA_MISMATCH")
    input_ids = tuple(item["dataset_id"] for item in inputs)
    if any(dataset_id in FORBIDDEN_HELD_OUT_IDS for dataset_id in input_ids):
        raise InvalidExperiment("FORBIDDEN_HELD_OUT_INPUT", "held-out IDs 41007-41008 are forbidden")
    if input_ids != PERMITTED_INPUT_IDS:
        raise InvalidExperiment("INPUT_SET_MISMATCH", "exact ordered inputs 41001-41006 are required")
    if any(not isinstance(item["path"], str) or not item["path"] or not _is_sha256(item["sha256"]) for item in inputs):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "input path or digest is malformed")
    forbidden_path_parts = {str(dataset_id) for dataset_id in FORBIDDEN_HELD_OUT_IDS}
    if any(forbidden_path_parts.intersection(Path(item["path"]).parts) for item in inputs):
        raise InvalidExperiment("FORBIDDEN_HELD_OUT_PATH", "held-out path segments 41007-41008 are forbidden")

    command = manifest["command_schema"]
    if not isinstance(command, Mapping):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "command schema must be an object")
    _require_exact_keys(command, {"runner", "required_arguments", "optional_arguments"}, "MANIFEST_SCHEMA_MISMATCH")
    if command != {
        "runner": RUNNER_PATH,
        "required_arguments": ["--manifest", "--output"],
        "optional_arguments": ["--source-commit"],
    }:
        raise InvalidExperiment("COMMAND_SCHEMA_MISMATCH", "command schema is not authorized")


def _load_json_object(raw: bytes, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidExperiment(reason, "JSON could not be decoded") from exc
    if not isinstance(value, dict):
        raise InvalidExperiment(reason, "JSON root must be an object")
    return value


def _resolve_manifest_input(manifest_path: Path, declared_path: str) -> Path:
    path = Path(declared_path)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def preflight(
    repo_root: Path,
    manifest_path: Path,
    *,
    declared_source_commit: str | None = None,
) -> Preflight:
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise InvalidExperiment("MANIFEST_UNREADABLE", "authorization manifest could not be read") from exc
    manifest = _load_json_object(manifest_bytes, "MANIFEST_INVALID_JSON")
    validate_manifest_schema(manifest)

    source_state = read_source_state(repo_root)
    if source_state.dirty:
        raise InvalidExperiment("DIRTY_WORKTREE", "the repository must be clean before any input is read")
    expected_source = manifest["expected_source"]
    if source_state.head != expected_source["head"]:
        raise InvalidExperiment("HEAD_MISMATCH", "actual HEAD differs from the authorization manifest")
    if source_state.tree != expected_source["tree"]:
        raise InvalidExperiment("TREE_MISMATCH", "actual tree differs from the authorization manifest")
    if declared_source_commit is not None:
        if not _is_commit(declared_source_commit) or declared_source_commit != source_state.head:
            raise InvalidExperiment("SOURCE_COMMIT_DECLARATION_MISMATCH", "declared source commit is not actual HEAD")
        if declared_source_commit != expected_source["head"]:
            raise InvalidExperiment("SOURCE_COMMIT_DECLARATION_MISMATCH", "declared source commit is not authorized")

    source_hashes: dict[str, str] = {}
    for relative_path in REQUIRED_SOURCE_PATHS:
        path = repo_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise InvalidExperiment("SOURCE_FILE_UNREADABLE", f"required source file unavailable: {relative_path}")
        digest = sha256_file(path)
        source_hashes[relative_path] = digest
        if digest != manifest["source_files"][relative_path]:
            raise InvalidExperiment("SOURCE_FILE_HASH_MISMATCH", f"source digest mismatch: {relative_path}")

    config_path = repo_root / CONFIG_PATH
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise InvalidExperiment("CONFIG_UNREADABLE", "frozen config could not be read") from exc
    config_digest = sha256_bytes(config_bytes)
    if config_digest != manifest["config"]["sha256"] or config_digest != source_hashes[CONFIG_PATH]:
        raise InvalidExperiment("CONFIG_IDENTITY_MISMATCH", "config digest differs from authorization")
    config = _load_json_object(config_bytes, "CONFIG_INVALID_JSON")
    validate_frozen_config(config)

    input_hashes: dict[str, str] = {}
    feature_bytes: dict[int, bytes] = {}
    for item in manifest["inputs"]:
        dataset_id = int(item["dataset_id"])
        path = _resolve_manifest_input(manifest_path, item["path"])
        if path.is_symlink() or not path.is_file():
            raise InvalidExperiment("INPUT_UNREADABLE", f"authorized input unavailable: {dataset_id}")
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        input_hashes[str(dataset_id)] = digest
        if digest != item["sha256"]:
            raise InvalidExperiment("INPUT_HASH_MISMATCH", f"input digest mismatch: {dataset_id}")
        feature_bytes[dataset_id] = raw

    features: dict[int, np.ndarray] = {}
    for dataset_id in PERMITTED_INPUT_IDS:
        payload = _load_json_object(feature_bytes[dataset_id], "INPUT_INVALID_JSON")
        try:
            array = np.asarray(payload["features"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidExperiment("INPUT_SCHEMA_MISMATCH", f"invalid feature payload: {dataset_id}") from exc
        if array.shape != (13, 30) or not np.isfinite(array).all():
            raise InvalidExperiment("INPUT_SCHEMA_MISMATCH", f"invalid feature matrix: {dataset_id}")
        features[dataset_id] = array

    return Preflight(
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        config=config,
        config_bytes=config_bytes,
        source_state=source_state,
        source_hashes=source_hashes,
        input_hashes=input_hashes,
        features=features,
    )


def transform(features: np.ndarray, candidate: str) -> np.ndarray:
    if candidate not in CANDIDATE_ORDER:
        raise InvalidExperiment("CANDIDATE_NOT_AUTHORIZED", "only A1 and A2 are authorized")
    normalization = FROZEN_CONFIG["normalization"]
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    normalized = np.clip(
        (features - median) / np.maximum(normalization["mad_scale"] * mad, normalization["mad_floor"]),
        normalization["clip_min"],
        normalization["clip_max"],
    )
    if candidate == "A1":
        return normalized
    high_pass = np.empty_like(normalized)
    high_pass[0] = normalized[0] - normalized[1]
    high_pass[-1] = normalized[-1] - normalized[-2]
    high_pass[1:-1] = normalized[1:-1] - (normalized[:-2] + normalized[2:]) / 2.0
    return high_pass


def fit_readout(feature_sets: Sequence[np.ndarray]) -> np.ndarray:
    features = np.concatenate(feature_sets)
    targets = np.tile(np.asarray(TEMPORAL_POINTS, dtype=np.float64), (len(feature_sets), 1))
    design = np.column_stack((features, np.ones(len(features))))
    penalty = FROZEN_CONFIG["readout"]["ridge"] * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ targets)


def observe(features: np.ndarray, readout: np.ndarray) -> np.ndarray:
    return np.column_stack((features, np.ones(len(features)))) @ readout


def _second_singular_value(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return float(singular_values[1])


def _affine_residual(window: np.ndarray) -> float:
    template = np.asarray(TEMPORAL_POINTS[:6], dtype=np.float64)
    augmented_anchors = np.column_stack((template[:3], np.ones(3)))
    weights = np.linalg.solve(augmented_anchors.T, np.column_stack((template[3:], np.ones(3))).T).T
    predicted = weights @ window[:3]
    numerator = float(np.sum((window[3:] - predicted) ** 2))
    centered = window - window.mean(axis=0)
    return numerator / (float(np.sum(centered**2)) + 1e-9)


def _calibration(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray([TEMPORAL_POINTS[index] for index in CALIBRATION_INDICES], dtype=np.float64)
    design = np.column_stack((target, np.ones(len(target))))
    beta, *_ = np.linalg.lstsq(design, observations, rcond=None)
    return beta[:2].T, beta[2]


def _equalize(observations: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    ridge = 1e-4
    projector = np.linalg.inv(matrix.T @ matrix + ridge * np.eye(2)) @ matrix.T
    return (projector @ (observations - bias).T).T


def window_metrics(observation: np.ndarray, start: int) -> dict[str, float]:
    if start not in START_INDICES:
        raise InvalidExperiment("START_INDEX_NOT_AUTHORIZED", "window start is outside the frozen set")
    window = observation[start : start + 6]
    residual = _affine_residual(window)
    calibration_observations = observation[[start + index for index in CALIBRATION_INDICES]]
    matrix, bias = _calibration(calibration_observations)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    held_out_observations = observation[[start + index for index in PER_VIDEO_HELD_OUT_INDICES]]
    equalized = _equalize(held_out_observations, matrix, bias)
    held_out_target = np.asarray([TEMPORAL_POINTS[index] for index in PER_VIDEO_HELD_OUT_INDICES])
    centered = (observation - observation.mean(axis=0)) / math.sqrt(13)
    affine_s2 = float(singular_values[1])
    return {
        "residual": residual,
        "global_s2": _second_singular_value(centered),
        "affine_s2": affine_s2,
        "condition": float(singular_values[0] / affine_s2) if affine_s2 else 1e300,
        "held_out_mse": float(np.mean((equalized - held_out_target) ** 2)),
    }


def threshold_checks(metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> dict[str, bool]:
    return {
        "residual": metrics["residual"] <= thresholds["max_residual"],
        "global_s2": metrics["global_s2"] >= thresholds["min_global_s2"],
        "affine_s2": metrics["affine_s2"] >= thresholds["min_affine_s2"],
        "condition": metrics["condition"] <= thresholds["max_condition"],
        "held_out_mse": metrics["held_out_mse"] <= thresholds["max_held_out_mse"],
    }


def passes_thresholds(metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> bool:
    return all(threshold_checks(metrics, thresholds).values())


def derive_envelope(loo_metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if len(loo_metrics) != len(FIT_IDS):
        raise InvalidExperiment("TRAINING_LOO_COUNT_MISMATCH", "exactly four training LOO results are required")
    for index, metrics in enumerate(loo_metrics):
        if not passes_thresholds(metrics, ABSOLUTE_THRESHOLDS):
            raise InvalidExperiment("TRAINING_ABSOLUTE_GATE_FAILED", f"training LOO {index} failed an absolute threshold")
    return {
        "max_residual": min(ABSOLUTE_THRESHOLDS["max_residual"], max(item["residual"] for item in loo_metrics)),
        "min_global_s2": max(ABSOLUTE_THRESHOLDS["min_global_s2"], min(item["global_s2"] for item in loo_metrics)),
        "min_affine_s2": max(ABSOLUTE_THRESHOLDS["min_affine_s2"], min(item["affine_s2"] for item in loo_metrics)),
        "max_condition": min(ABSOLUTE_THRESHOLDS["max_condition"], max(item["condition"] for item in loo_metrics)),
        "max_held_out_mse": min(ABSOLUTE_THRESHOLDS["max_held_out_mse"], max(item["held_out_mse"] for item in loo_metrics)),
    }


def evaluate_candidate(raw_features: Mapping[int, np.ndarray], candidate: str) -> dict[str, Any]:
    transformed = {dataset_id: transform(raw_features[dataset_id], candidate) for dataset_id in PERMITTED_INPUT_IDS}
    loo: list[dict[str, float]] = []
    for held_out_id in FIT_IDS:
        readout = fit_readout([transformed[dataset_id] for dataset_id in FIT_IDS if dataset_id != held_out_id])
        loo.append(window_metrics(observe(transformed[held_out_id], readout), CORRECT_START_INDEX))
    absolute_checks = [threshold_checks(item, ABSOLUTE_THRESHOLDS) for item in loo]
    if not all(all(checks.values()) for checks in absolute_checks):
        return {
            "candidate": candidate,
            "training_absolute_gate_pass": False,
            "training_loo": [
                {"dataset_id": dataset_id, "metrics": metrics, "absolute_checks": checks}
                for dataset_id, metrics, checks in zip(FIT_IDS, loo, absolute_checks, strict=True)
            ],
            "development_evaluated": False,
            "development_gate_pass": False,
        }

    envelope = derive_envelope(loo)
    readout = fit_readout([transformed[dataset_id] for dataset_id in FIT_IDS])
    development_cases: dict[str, Any] = {}
    for dataset_id in DEVELOPMENT_IDS:
        observation = observe(transformed[dataset_id], readout)
        windows: list[dict[str, Any]] = []
        for start in START_INDICES:
            metrics = window_metrics(observation, start)
            checks = threshold_checks(metrics, envelope)
            windows.append(
                {
                    "start": start,
                    "correct": start == CORRECT_START_INDEX,
                    "metrics": metrics,
                    "checks": checks,
                    "accepted": all(checks.values()),
                }
            )
        development_cases[str(dataset_id)] = {"windows": windows}
    gate_pass = all(
        development_cases[str(dataset_id)]["windows"][0]["accepted"]
        and not any(window["accepted"] for window in development_cases[str(dataset_id)]["windows"][1:])
        for dataset_id in DEVELOPMENT_IDS
    )
    return {
        "candidate": candidate,
        "training_absolute_gate_pass": True,
        "training_loo": [
            {"dataset_id": dataset_id, "metrics": metrics, "absolute_checks": checks}
            for dataset_id, metrics, checks in zip(FIT_IDS, loo, absolute_checks, strict=True)
        ],
        "derived_envelope": envelope,
        "development_evaluated": True,
        "development_cases": development_cases,
        "development_gate_pass": gate_pass,
    }


def evaluate_gate(raw_features: Mapping[int, np.ndarray]) -> tuple[list[dict[str, Any]], str | None]:
    if tuple(sorted(raw_features)) != PERMITTED_INPUT_IDS:
        raise InvalidExperiment("INPUT_SET_MISMATCH", "science path requires exactly inputs 41001-41006")
    results: list[dict[str, Any]] = []
    for candidate in CANDIDATE_ORDER:
        result = evaluate_candidate(raw_features, candidate)
        results.append(result)
        if result["development_gate_pass"]:
            return results, candidate
    return results, None


def base_audit() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "output_schema": OUTPUT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "formal_result": False,
        "stage_progression_allowed": False,
        "old_l1_status": "Contradicted",
        "fresh_held_out_read": False,
        "candidate_order": list(CANDIDATE_ORDER),
    }


def invalid_audit(error: InvalidExperiment, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    audit = base_audit()
    audit.update(
        {
            "status": INVALID_STATUS,
            "valid_experiment": False,
            "reason_code": error.reason_code,
            "reason_detail": error.detail,
            "science_metrics_present": False,
        }
    )
    if context:
        audit["integrity_context"] = dict(context)
    return audit


def valid_audit(preflight_result: Preflight, candidates: list[dict[str, Any]], selected: str | None) -> dict[str, Any]:
    status = SUCCESS_STATUS if selected else FAILURE_STATUS
    audit = base_audit()
    audit.update(
        {
            "status": status,
            "valid_experiment": True,
            "reason_code": "DEVELOPMENT_GATE_PASS" if selected else "DEVELOPMENT_GATE_FAILED",
            "manifest_identity": {"sha256": sha256_bytes(preflight_result.manifest_bytes)},
            "config_identity": {"path": CONFIG_PATH, "sha256": sha256_bytes(preflight_result.config_bytes)},
            "source_state": preflight_result.source_state.as_dict(),
            "source_file_sha256": preflight_result.source_hashes,
            "input_sha256": preflight_result.input_hashes,
            "candidate_attempts": [result["candidate"] for result in candidates],
            "candidates": candidates,
            "selected_candidate": selected,
        }
    )
    return audit
