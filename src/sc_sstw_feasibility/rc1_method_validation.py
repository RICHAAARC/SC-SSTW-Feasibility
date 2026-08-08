"""Fail-closed RC1 saved-MP4 relation validation preparation.

The module implements protocol identity, prerequisite consumption, matched
triplet integrity, and a schedule-parameterized blind evaluator.  It does not
authorize a formal CPU/GPU run and never trains or tunes from RC1 inputs.
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

from .learned_observation_l1_v2 import (
    ABSOLUTE_THRESHOLDS as G0_ABSOLUTE_THRESHOLDS,
    CONFIG_PATH as G0_CONFIG_PATH,
    OUTPUT_SCHEMA as G0_OUTPUT_SCHEMA,
    PERMITTED_INPUT_IDS as G0_INPUT_IDS,
    PROTOCOL_ID as G0_PROTOCOL_ID,
    REQUIRED_SOURCE_PATHS as G0_REQUIRED_SOURCE_PATHS,
    SUCCESS_STATUS as G0_SUCCESS_STATUS,
    TEMPORAL_POINTS as G0_TEMPORAL_POINTS,
    transform as g0_transform,
)


PROTOCOL_ID = "sc_sstw_rc1_saved_mp4_relation_validation"
MANIFEST_SCHEMA = "sc_sstw_rc1_authorization_manifest_v1"
EXECUTION_SCHEMA = "sc_sstw_rc1_matched_triplet_execution_v1"
OUTPUT_SCHEMA = "sc_sstw_rc1_method_validation_audit_v1"
CONFIG_PATH = "configs/rc1_method_validation.json"
PLAN_PATH = "plans/rc1_matched_triplets.json"
PROTOCOL_PATH = "protocols/rc1_method_validation.md"
LIBRARY_PATH = "src/sc_sstw_feasibility/rc1_method_validation.py"
GPU_LIBRARY_PATH = "src/sc_sstw_feasibility/rc1_gpu_generation.py"
RUNNER_PATH = "experiments/run_rc1_method_validation.py"
NOTEBOOK_PATH = "notebooks/sc_sstw_rc1_method_validation.ipynb"
TEST_PATH = "tests/test_rc1_method_validation.py"
G0_LIBRARY_PATH = "src/sc_sstw_feasibility/learned_observation_l1_v2.py"
REQUIRED_SOURCE_PATHS = (
    CONFIG_PATH,
    PLAN_PATH,
    PROTOCOL_PATH,
    LIBRARY_PATH,
    GPU_LIBRARY_PATH,
    RUNNER_PATH,
    NOTEBOOK_PATH,
    TEST_PATH,
    G0_LIBRARY_PATH,
)

CONFIG_RAW_SHA256 = "8accea693798e2dd2ad4451ea14df71651116b89e3d8707cfe344886a57bcf15"
PLAN_RAW_SHA256 = "d5396107f00e68eba8f972a08996608421f02a4c205339a34b3616636e41a8c7"
G0_BASELINE_COMMIT = "3d7913f01f0315094a223a930bd71239f5ea56a5"
G0_BASELINE_TREE = "0814b3dd63383315239f22873f7b6aff2cdf8d7e"
G0_IMPLEMENTATION_GATE = "NOT_PASSED"

STATUS_PASS = "RC1_VALID_PASS"
STATUS_FAIL = "RC1_VALID_FAIL"
STATUS_INVALID = "INVALID_EXPERIMENT"
STATUS_PREREQUISITE = "PREREQUISITE_NOT_MET"
ALLOWED_STATES = (STATUS_PASS, STATUS_FAIL, STATUS_INVALID, STATUS_PREREQUISITE)
PASS_CONCLUSION = "causal saved-MP4 public relation mechanism screen passed under this frozen RC1 protocol"
CONDITIONS = ("OFF", "A", "B")
TEMPLATES = ("A", "B")
START_INDICES = tuple(range(8))
CROSS_RESIDUAL_EXPECTED = 0.761311945935564
CROSS_RESIDUAL_TOLERANCE = 1e-9
FORBIDDEN_FORMAL_IDS = tuple(range(41001, 41009))
FORBIDDEN_CLAIM_TOKENS = (
    "method_validated",
    "gpu_ready",
    "l2",
    "owner_wrong_key",
    "robust",
    "fixed_fpr",
    "paper_claim",
)
THRESHOLD_KEYS = tuple(G0_ABSOLUTE_THRESHOLDS)
PREREQUISITE_FILES = ("audit.json", "frozen_frontend.json", "readout.json")
ARTIFACT_NAMES = ("video", "features", "stdout", "stderr", "config", "environment", "command", "integrity")


class InvalidExperiment(RuntimeError):
    def __init__(self, reason_code: str, detail: str):
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


class PrerequisiteNotMet(InvalidExperiment):
    pass


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
class FrozenPrerequisite:
    package_path: Path
    checksums_sha256: str
    audit_sha256: str
    frontend_sha256: str
    readout_sha256: str
    selected_candidate: str
    thresholds: dict[str, float]
    frontend_definition_sha256: str
    readout: np.ndarray
    source_head: str
    source_tree: str

    def identity(self) -> dict[str, Any]:
        return {
            "checksums_sha256": self.checksums_sha256,
            "audit_sha256": self.audit_sha256,
            "frontend_sha256": self.frontend_sha256,
            "readout_sha256": self.readout_sha256,
            "selected_candidate": self.selected_candidate,
            "thresholds": self.thresholds,
            "frontend_definition_sha256": self.frontend_definition_sha256,
            "source_head": self.source_head,
            "source_tree": self.source_tree,
        }


@dataclass(frozen=True)
class Preflight:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    config: dict[str, Any]
    config_bytes: bytes
    plan: dict[str, Any]
    plan_bytes: bytes
    source_state: SourceState
    source_hashes: dict[str, str]
    prerequisite: FrozenPrerequisite


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], reason: str) -> None:
    if set(value) != expected:
        raise InvalidExperiment(reason, "object keys do not match the frozen schema")


def _json_object(raw: bytes, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidExperiment(reason, "JSON could not be decoded") from exc
    if not isinstance(value, dict):
        raise InvalidExperiment(reason, "JSON root must be an object")
    return value


def schedule_a() -> tuple[tuple[float, float], ...]:
    """Return A from the existing burst-alpha source of truth."""

    return tuple((float(x), float(y)) for x, y in G0_TEMPORAL_POINTS)


def schedule_b() -> tuple[tuple[float, float], ...]:
    points = list(schedule_a())
    points[4], points[5] = points[5], points[4]
    return tuple(points)


def affine_residual(observation_window: Sequence[Sequence[float]], template_points: Sequence[Sequence[float]]) -> float:
    observation = np.asarray(observation_window, dtype=np.float64)
    template = np.asarray(template_points, dtype=np.float64)
    if observation.shape != (6, 2) or template.shape != (6, 2) or not np.isfinite(observation).all() or not np.isfinite(template).all():
        raise InvalidExperiment("SCHEDULE_SHAPE_MISMATCH", "residual requires finite 6x2 observation and template")
    augmented_anchors = np.column_stack((template[:3], np.ones(3)))
    try:
        weights = np.linalg.solve(augmented_anchors.T, np.column_stack((template[3:], np.ones(3))).T).T
    except np.linalg.LinAlgError as exc:
        raise InvalidExperiment("SCHEDULE_ANCHORS_DEGENERATE", "template anchors are degenerate") from exc
    predicted = weights @ observation[:3]
    centered = observation - observation.mean(axis=0)
    return float(np.sum((observation[3:] - predicted) ** 2) / (np.sum(centered**2) + 1e-9))


def schedule_preflight() -> dict[str, Any]:
    a = schedule_a()
    b = schedule_b()
    if len(a) != 13 or len(b) != 13:
        raise InvalidExperiment("SCHEDULE_STRUCTURE_MISMATCH", "A and B must contain 13 points")
    if a[:3] != b[:3] or a[6:] != b[6:]:
        raise InvalidExperiment("SCHEDULE_STRUCTURE_MISMATCH", "anchors or points 6-12 changed")
    differing = tuple(index for index, (left, right) in enumerate(zip(a, b, strict=True)) if left != right)
    if differing != (4, 5) or b[4] != a[5] or b[5] != a[4]:
        raise InvalidExperiment("SCHEDULE_STRUCTURE_MISMATCH", "only indices 4 and 5 may be exchanged")
    expected_b_prefix = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.25, 0.35), (0.20, 0.80), (0.75, 0.20))
    if b[:6] != expected_b_prefix:
        raise InvalidExperiment("SCHEDULE_B_MISMATCH", "schedule B does not match the frozen six-point definition")
    residual_ab = affine_residual(a[:6], b[:6])
    residual_ba = affine_residual(b[:6], a[:6])
    if abs(residual_ab - CROSS_RESIDUAL_EXPECTED) > CROSS_RESIDUAL_TOLERANCE or abs(residual_ba - CROSS_RESIDUAL_EXPECTED) > CROSS_RESIDUAL_TOLERANCE:
        raise InvalidExperiment("CROSS_RESIDUAL_MISMATCH", "A/B cross residual changed")
    if affine_residual(a[:6], a[:6]) > 1e-12 or affine_residual(b[:6], b[:6]) > 1e-12:
        raise InvalidExperiment("SAME_SCHEDULE_RESIDUAL_MISMATCH", "same-template residual is not zero")
    return {
        "evidence_label": "protocol_synthetic_preflight_only",
        "schedule_a_sha256": sha256_bytes(canonical_json_bytes(a)),
        "schedule_b_sha256": sha256_bytes(canonical_json_bytes(b)),
        "A_observation_by_B_template": residual_ab,
        "B_observation_by_A_template": residual_ba,
        "expected": CROSS_RESIDUAL_EXPECTED,
        "absolute_tolerance": CROSS_RESIDUAL_TOLERANCE,
        "passed": True,
    }


def validate_frozen_config_bytes(raw: bytes) -> dict[str, Any]:
    if sha256_bytes(raw) != CONFIG_RAW_SHA256:
        raise InvalidExperiment("CONFIG_NOT_FROZEN", "RC1 config raw digest changed")
    config = _json_object(raw, "CONFIG_INVALID_JSON")
    if config.get("protocol_id") != PROTOCOL_ID or config.get("output_schema") != OUTPUT_SCHEMA:
        raise InvalidExperiment("CONFIG_NOT_FROZEN", "RC1 config protocol or output schema changed")
    if config.get("g0_baseline") != {"commit": G0_BASELINE_COMMIT, "tree": G0_BASELINE_TREE, "independent_implementation_gate": G0_IMPLEMENTATION_GATE}:
        raise InvalidExperiment("G0_BASELINE_MISMATCH", "G0 baseline or independent-gate state changed")
    if tuple(config["triplets"]["condition_order"]) != CONDITIONS or tuple(config["evaluation"]["start_indices"]) != START_INDICES:
        raise InvalidExperiment("CONFIG_NOT_FROZEN", "condition order or scan budget changed")
    if tuple(config["evaluation"]["template_order"]) != TEMPLATES or config["evaluation"]["averaging_or_majority_vote"] is not False:
        raise InvalidExperiment("CONFIG_NOT_FROZEN", "template order or aggregation rule changed")
    if config["states"] != {"pass": STATUS_PASS, "fail": STATUS_FAIL, "invalid": STATUS_INVALID, "prerequisite_missing": STATUS_PREREQUISITE}:
        raise InvalidExperiment("CONFIG_NOT_FROZEN", "allowed states changed")
    if config["pass_conclusion_limit"] != PASS_CONCLUSION or tuple(config["forbidden_claims"]) != FORBIDDEN_CLAIM_TOKENS:
        raise InvalidExperiment("CONFIG_NOT_FROZEN", "conclusion ceiling or forbidden claims changed")
    if config["formal_result"] is not False or config["stage_progression_allowed"] is not False:
        raise InvalidExperiment("CONFIG_NOT_FROZEN", "evidence boundary changed")
    schedule_preflight()
    return config


def validate_frozen_plan_bytes(raw: bytes, config: Mapping[str, Any]) -> dict[str, Any]:
    if sha256_bytes(raw) != PLAN_RAW_SHA256:
        raise InvalidExperiment("PLAN_NOT_FROZEN", "RC1 triplet plan raw digest changed")
    plan = _json_object(raw, "PLAN_INVALID_JSON")
    if plan.get("protocol_id") != PROTOCOL_ID or tuple(plan.get("group_order", ())) != ("orbital_glass", "articulated_paper"):
        raise InvalidExperiment("PLAN_NOT_FROZEN", "triplet protocol or group order changed")
    groups = plan.get("groups")
    if not isinstance(groups, list) or len(groups) != int(config["triplets"]["required_group_count"]):
        raise InvalidExperiment("PLAN_NOT_FROZEN", "exactly two triplet groups are required")
    grammars = tuple(group.get("content_grammar") for group in groups)
    if len(set(grammars)) != len(grammars) or grammars != ("multi_object_counter_rotation", "articulated_surface_deformation"):
        raise InvalidExperiment("CONTENT_GRAMMAR_NOT_HETEROGENEOUS", "triplet content grammars are not heterogeneous")
    for group in groups:
        _require_exact_keys(group, {"group_id", "content_grammar", "prompt", "seed", "initial_noise_policy", "conditions"}, "PLAN_NOT_FROZEN")
        if tuple(group["conditions"]) != CONDITIONS or not isinstance(group["prompt"], str) or not group["prompt"].strip():
            raise InvalidExperiment("PLAN_NOT_FROZEN", "group prompt or conditions changed")
        if "single" in group["content_grammar"] and "center" in group["content_grammar"]:
            raise InvalidExperiment("CONTENT_GRAMMAR_NOT_HETEROGENEOUS", "legacy single-center grammar is forbidden")
    if plan["only_allowed_difference"] != "carrier_condition_and_schedule" or plan["formal_result"] is not False or plan["stage_progression_allowed"] is not False:
        raise InvalidExperiment("PLAN_NOT_FROZEN", "matched-triplet or evidence boundary changed")
    return plan


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(("git", *arguments), cwd=repo_root, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise InvalidExperiment("GIT_STATE_UNREADABLE", "Git source state could not be read") from exc
    return completed.stdout.strip()


def read_source_state(repo_root: Path) -> SourceState:
    head = _git(repo_root, "rev-parse", "HEAD")
    tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    status = tuple(line for line in _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all").splitlines() if line)
    if not _is_commit(head) or not _is_commit(tree):
        raise InvalidExperiment("GIT_STATE_UNREADABLE", "Git returned malformed source identity")
    return SourceState(head=head, tree=tree, dirty=bool(status), status_porcelain=status)


def _forbidden_path(path_text: str) -> bool:
    return any(str(dataset_id) in path_text for dataset_id in FORBIDDEN_FORMAL_IDS)


def validate_manifest_schema(manifest: Mapping[str, Any]) -> None:
    _require_exact_keys(
        manifest,
        {"schema_version", "manifest_schema", "protocol_id", "expected_source", "source_files", "config", "plan", "prerequisite", "command_schema", "output_schema"},
        "MANIFEST_SCHEMA_MISMATCH",
    )
    if manifest["schema_version"] != 1 or manifest["manifest_schema"] != MANIFEST_SCHEMA or manifest["protocol_id"] != PROTOCOL_ID:
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "manifest version or protocol changed")
    if manifest["output_schema"] != OUTPUT_SCHEMA:
        raise InvalidExperiment("OUTPUT_SCHEMA_MISMATCH", "output schema is not authorized")
    expected = manifest["expected_source"]
    if not isinstance(expected, Mapping):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "expected_source must be an object")
    _require_exact_keys(expected, {"head", "tree"}, "MANIFEST_SCHEMA_MISMATCH")
    if not _is_commit(expected["head"]) or not _is_commit(expected["tree"]):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "expected source identity is malformed")
    sources = manifest["source_files"]
    if not isinstance(sources, Mapping) or tuple(sorted(sources)) != tuple(sorted(REQUIRED_SOURCE_PATHS)) or not all(_is_sha256(item) for item in sources.values()):
        raise InvalidExperiment("SOURCE_FILE_SET_MISMATCH", "source file authorization set or digest changed")
    for key, path, expected_sha in (("config", CONFIG_PATH, CONFIG_RAW_SHA256), ("plan", PLAN_PATH, PLAN_RAW_SHA256)):
        identity = manifest[key]
        if not isinstance(identity, Mapping):
            raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", f"{key} identity must be an object")
        _require_exact_keys(identity, {"path", "sha256"}, "MANIFEST_SCHEMA_MISMATCH")
        if identity != {"path": path, "sha256": expected_sha}:
            raise InvalidExperiment(f"{key.upper()}_IDENTITY_MISMATCH", f"{key} identity is not frozen")
    prerequisite = manifest["prerequisite"]
    if not isinstance(prerequisite, Mapping):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "prerequisite identity must be an object")
    _require_exact_keys(prerequisite, {"package_path", "checksums_sha256"}, "MANIFEST_SCHEMA_MISMATCH")
    if not isinstance(prerequisite["package_path"], str) or not prerequisite["package_path"] or not _is_sha256(prerequisite["checksums_sha256"]):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "prerequisite path or digest is malformed")
    if _forbidden_path(prerequisite["package_path"]):
        raise InvalidExperiment("FORBIDDEN_FORMAL_PATH", "formal 41001-41008 path segments are forbidden")
    command = manifest["command_schema"]
    if command != {
        "runner": RUNNER_PATH,
        "required_arguments": ["--manifest", "--output"],
        "optional_arguments": ["--source-commit", "--execution-package", "--generate", "--synthetic-fixture"],
    }:
        raise InvalidExperiment("COMMAND_SCHEMA_MISMATCH", "command schema is not authorized")


def frontend_definition(candidate: str) -> dict[str, Any]:
    if candidate not in {"A1", "A2"}:
        raise InvalidExperiment("PREREQUISITE_CANDIDATE_INVALID", "selected candidate must be A1 or A2")
    return {
        "candidate": candidate,
        "normalization": {"mad_scale": 1.4826, "mad_floor": 1e-6, "clip_min": -6.0, "clip_max": 6.0},
        "high_pass": candidate == "A2",
        "readout": {"kind": "linear_affine", "shape": [31, 2], "fit_intercept": True},
    }


def _parse_checksums(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise InvalidExperiment("PREREQUISITE_CHECKSUMS_INVALID", "checksum file is not UTF-8") from exc
    parsed: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or not _is_sha256(parts[0]) or not parts[1] or "/" in parts[1] or parts[1] in parsed:
            raise InvalidExperiment("PREREQUISITE_CHECKSUMS_INVALID", "checksum line is malformed")
        parsed[parts[1]] = parts[0]
    if tuple(sorted(parsed)) != tuple(sorted(PREREQUISITE_FILES)):
        raise InvalidExperiment("PREREQUISITE_CHECKSUMS_INVALID", "prerequisite file set changed")
    return parsed


def _validate_thresholds(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or tuple(value) != THRESHOLD_KEYS:
        raise InvalidExperiment("PREREQUISITE_THRESHOLDS_INVALID", "threshold key set or order changed")
    thresholds = {key: float(value[key]) for key in THRESHOLD_KEYS}
    if not all(math.isfinite(item) for item in thresholds.values()):
        raise InvalidExperiment("PREREQUISITE_THRESHOLDS_INVALID", "thresholds must be finite")
    if thresholds["max_residual"] > G0_ABSOLUTE_THRESHOLDS["max_residual"] or thresholds["min_global_s2"] < G0_ABSOLUTE_THRESHOLDS["min_global_s2"] or thresholds["min_affine_s2"] < G0_ABSOLUTE_THRESHOLDS["min_affine_s2"] or thresholds["max_condition"] > G0_ABSOLUTE_THRESHOLDS["max_condition"] or thresholds["max_held_out_mse"] > G0_ABSOLUTE_THRESHOLDS["max_held_out_mse"]:
        raise InvalidExperiment("PREREQUISITE_THRESHOLDS_INVALID", "prerequisite thresholds are wider than absolute limits")
    return thresholds


def validate_prerequisite_package(package_path: Path, expected_checksums_sha256: str) -> FrozenPrerequisite:
    if not package_path.exists():
        raise PrerequisiteNotMet("PREREQUISITE_PACKAGE_MISSING", "no frozen selected-candidate package is available")
    if package_path.is_symlink() or not package_path.is_dir():
        raise InvalidExperiment("PREREQUISITE_PACKAGE_INVALID", "prerequisite package must be a real directory")
    checksums_path = package_path / "checksums.sha256"
    if checksums_path.is_symlink() or not checksums_path.is_file():
        raise InvalidExperiment("PREREQUISITE_CHECKSUMS_MISSING", "prerequisite checksums are missing")
    checksums_raw = checksums_path.read_bytes()
    checksums_sha = sha256_bytes(checksums_raw)
    if checksums_sha != expected_checksums_sha256:
        raise InvalidExperiment("PREREQUISITE_CHECKSUMS_HASH_MISMATCH", "prerequisite checksum-file digest changed")
    declared = _parse_checksums(checksums_raw)
    raw_files: dict[str, bytes] = {}
    for name in PREREQUISITE_FILES:
        path = package_path / name
        if path.is_symlink() or not path.is_file():
            raise InvalidExperiment("PREREQUISITE_FILE_MISSING", f"prerequisite file missing: {name}")
        raw = path.read_bytes()
        if sha256_bytes(raw) != declared[name]:
            raise InvalidExperiment("PREREQUISITE_FILE_HASH_MISMATCH", f"prerequisite file digest changed: {name}")
        raw_files[name] = raw

    audit = _json_object(raw_files["audit.json"], "PREREQUISITE_AUDIT_INVALID")
    if audit.get("protocol_id") != G0_PROTOCOL_ID or audit.get("output_schema") != G0_OUTPUT_SCHEMA:
        raise InvalidExperiment("PREREQUISITE_AUDIT_IDENTITY_MISMATCH", "L1-v2 protocol or schema changed")
    if audit.get("valid_experiment") is not True or audit.get("status") != G0_SUCCESS_STATUS:
        raise PrerequisiteNotMet("PREREQUISITE_SELECTED_CANDIDATE_NOT_READY", "L1-v2 package has no qualifying selected candidate")
    if audit.get("formal_result") is not False or audit.get("stage_progression_allowed") is not False:
        raise InvalidExperiment("PREREQUISITE_EVIDENCE_BOUNDARY_INVALID", "L1-v2 package overstates its evidence boundary")
    selected = audit.get("selected_candidate")
    if selected not in {"A1", "A2"}:
        raise InvalidExperiment("PREREQUISITE_CANDIDATE_INVALID", "selected candidate is forged or missing")
    source_state = audit.get("source_state")
    if not isinstance(source_state, Mapping) or source_state.get("dirty") is not False or not _is_commit(source_state.get("head")) or not _is_commit(source_state.get("tree")):
        raise InvalidExperiment("PREREQUISITE_SOURCE_IDENTITY_INVALID", "L1-v2 source identity is incomplete or dirty")
    source_hashes = audit.get("source_file_sha256")
    if not isinstance(source_hashes, Mapping) or tuple(sorted(source_hashes)) != tuple(sorted(G0_REQUIRED_SOURCE_PATHS)) or not all(_is_sha256(item) for item in source_hashes.values()):
        raise InvalidExperiment("PREREQUISITE_SOURCE_IDENTITY_INVALID", "L1-v2 source file identity is incomplete")
    config_identity = audit.get("config_identity")
    if not isinstance(config_identity, Mapping) or config_identity.get("path") != G0_CONFIG_PATH or not _is_sha256(config_identity.get("sha256")):
        raise InvalidExperiment("PREREQUISITE_CONFIG_IDENTITY_INVALID", "L1-v2 config identity is incomplete")
    input_hashes = audit.get("input_sha256")
    if not isinstance(input_hashes, Mapping) or tuple(sorted(input_hashes)) != tuple(str(item) for item in G0_INPUT_IDS) or not all(_is_sha256(item) for item in input_hashes.values()):
        raise InvalidExperiment("PREREQUISITE_INPUT_IDENTITY_INVALID", "L1-v2 input identity is incomplete")
    candidates = audit.get("candidates")
    if not isinstance(candidates, list):
        raise InvalidExperiment("PREREQUISITE_CANDIDATE_INVALID", "L1-v2 candidate trace is missing")
    selected_traces = [item for item in candidates if isinstance(item, Mapping) and item.get("candidate") == selected]
    if len(selected_traces) != 1 or selected_traces[0].get("development_gate_pass") is not True:
        raise InvalidExperiment("PREREQUISITE_CANDIDATE_INVALID", "selected-candidate trace is contradictory")
    thresholds = _validate_thresholds(selected_traces[0].get("derived_envelope"))

    frontend = _json_object(raw_files["frozen_frontend.json"], "PREREQUISITE_FRONTEND_INVALID")
    _require_exact_keys(frontend, {"schema_version", "selected_candidate", "frontend_definition_sha256", "readout_path", "readout_sha256", "thresholds", "g0_audit_sha256", "source_head", "source_tree"}, "PREREQUISITE_FRONTEND_INVALID")
    expected_frontend_sha = sha256_bytes(canonical_json_bytes(frontend_definition(selected)))
    if frontend["schema_version"] != 1 or frontend["selected_candidate"] != selected or frontend["frontend_definition_sha256"] != expected_frontend_sha:
        raise InvalidExperiment("PREREQUISITE_FRONTEND_INVALID", "frontend definition or selected candidate changed")
    if frontend["readout_path"] != "readout.json" or frontend["readout_sha256"] != declared["readout.json"]:
        raise InvalidExperiment("PREREQUISITE_READOUT_IDENTITY_INVALID", "readout identity changed")
    if frontend["g0_audit_sha256"] != declared["audit.json"] or frontend["source_head"] != source_state["head"] or frontend["source_tree"] != source_state["tree"]:
        raise InvalidExperiment("PREREQUISITE_SOURCE_IDENTITY_INVALID", "frontend binding to G0 audit/source changed")
    if _validate_thresholds(frontend["thresholds"]) != thresholds:
        raise InvalidExperiment("PREREQUISITE_THRESHOLDS_INVALID", "frontend thresholds differ from selected-candidate trace")

    readout_payload = _json_object(raw_files["readout.json"], "PREREQUISITE_READOUT_INVALID")
    _require_exact_keys(readout_payload, {"schema_version", "selected_candidate", "shape", "coefficients"}, "PREREQUISITE_READOUT_INVALID")
    if readout_payload["schema_version"] != 1 or readout_payload["selected_candidate"] != selected or readout_payload["shape"] != [31, 2]:
        raise InvalidExperiment("PREREQUISITE_READOUT_INVALID", "readout metadata changed")
    try:
        readout = np.asarray(readout_payload["coefficients"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidExperiment("PREREQUISITE_READOUT_INVALID", "readout coefficients cannot be decoded") from exc
    if readout.shape != (31, 2) or not np.isfinite(readout).all():
        raise InvalidExperiment("PREREQUISITE_READOUT_INVALID", "readout coefficients must be finite 31x2")
    return FrozenPrerequisite(
        package_path=package_path,
        checksums_sha256=checksums_sha,
        audit_sha256=declared["audit.json"],
        frontend_sha256=declared["frozen_frontend.json"],
        readout_sha256=declared["readout.json"],
        selected_candidate=selected,
        thresholds=thresholds,
        frontend_definition_sha256=expected_frontend_sha,
        readout=readout,
        source_head=str(source_state["head"]),
        source_tree=str(source_state["tree"]),
    )


def _resolve_relative(base_file: Path, declared_path: str) -> Path:
    path = Path(declared_path)
    return path.resolve() if path.is_absolute() else (base_file.parent / path).resolve()


def preflight(repo_root: Path, manifest_path: Path, *, declared_source_commit: str | None = None) -> Preflight:
    if _forbidden_path(str(manifest_path)):
        raise InvalidExperiment("FORBIDDEN_FORMAL_PATH", "manifest path uses a forbidden formal ID")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise InvalidExperiment("MANIFEST_UNREADABLE", "RC1 authorization manifest could not be read") from exc
    manifest = _json_object(manifest_bytes, "MANIFEST_INVALID_JSON")
    validate_manifest_schema(manifest)
    source_state = read_source_state(repo_root)
    if source_state.dirty:
        raise InvalidExperiment("DIRTY_WORKTREE", "RC1 requires a clean checkout before prerequisite or generation access")
    if source_state.head != manifest["expected_source"]["head"]:
        raise InvalidExperiment("HEAD_MISMATCH", "actual HEAD differs from authorization")
    if source_state.tree != manifest["expected_source"]["tree"]:
        raise InvalidExperiment("TREE_MISMATCH", "actual tree differs from authorization")
    if declared_source_commit is not None and (declared_source_commit != source_state.head or declared_source_commit != manifest["expected_source"]["head"]):
        raise InvalidExperiment("SOURCE_COMMIT_DECLARATION_MISMATCH", "caller-declared commit is not actual authorized HEAD")
    source_hashes: dict[str, str] = {}
    for relative_path in REQUIRED_SOURCE_PATHS:
        path = repo_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise InvalidExperiment("SOURCE_FILE_UNREADABLE", f"required source file unavailable: {relative_path}")
        digest = sha256_file(path)
        source_hashes[relative_path] = digest
        if digest != manifest["source_files"][relative_path]:
            raise InvalidExperiment("SOURCE_FILE_HASH_MISMATCH", f"source digest changed: {relative_path}")
    config_bytes = (repo_root / CONFIG_PATH).read_bytes()
    plan_bytes = (repo_root / PLAN_PATH).read_bytes()
    config = validate_frozen_config_bytes(config_bytes)
    plan = validate_frozen_plan_bytes(plan_bytes, config)
    if source_hashes[CONFIG_PATH] != CONFIG_RAW_SHA256 or source_hashes[PLAN_PATH] != PLAN_RAW_SHA256:
        raise InvalidExperiment("CONFIG_OR_PLAN_IDENTITY_MISMATCH", "source map differs from frozen config or plan")
    prerequisite_path = _resolve_relative(manifest_path, manifest["prerequisite"]["package_path"])
    prerequisite = validate_prerequisite_package(prerequisite_path, manifest["prerequisite"]["checksums_sha256"])
    return Preflight(
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        config=config,
        config_bytes=config_bytes,
        plan=plan,
        plan_bytes=plan_bytes,
        source_state=source_state,
        source_hashes=source_hashes,
        prerequisite=prerequisite,
    )


def expected_matched_parameters(config: Mapping[str, Any], group: Mapping[str, Any]) -> dict[str, Any]:
    generation = config["generation"]
    encoding = config["encoding"]
    return {
        "prompt": group["prompt"],
        "prompt_sha256": sha256_bytes(group["prompt"].encode("utf-8")),
        "seed": int(group["seed"]),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "scheduler": generation["scheduler"],
        "sampler": generation["sampler"],
        "inference_steps": generation["inference_steps"],
        "guidance_scale": generation["guidance_scale"],
        "height": generation["height"],
        "width": generation["width"],
        "frame_count": generation["frame_count"],
        "fps": generation["fps"],
        "dtype": generation["dtype"],
        "negative_prompt": generation["negative_prompt"],
        "encoder_call": encoding["call"],
        "container": encoding["container"],
        "codec": encoding["codec"],
        "pixel_format": encoding["pixel_format"],
        "quality": encoding["quality"],
        "bitrate": encoding["bitrate"],
        "macro_block_size": encoding["macro_block_size"],
        "execution_path": "repository_rc1_gpu_generation_helper",
    }


def _artifact_path(execution_path: Path, artifact: Mapping[str, Any]) -> Path:
    path_text = artifact.get("path")
    if not isinstance(path_text, str) or not path_text or _forbidden_path(path_text):
        raise InvalidExperiment("EXECUTION_ARTIFACT_PATH_INVALID", "artifact path is malformed or disguised as a formal input")
    path = Path(path_text)
    resolved = path.resolve() if path.is_absolute() else (execution_path.parent / path).resolve()
    try:
        resolved.relative_to(execution_path.parent.resolve())
    except ValueError as exc:
        raise InvalidExperiment("EXECUTION_ARTIFACT_PATH_INVALID", "artifact path escapes the execution package") from exc
    return resolved


def validate_execution_record(record_path: Path, preflight_result: Preflight, *, synthetic_fixture: bool) -> tuple[dict[str, Any], dict[tuple[str, str], Path]]:
    if _forbidden_path(str(record_path)):
        raise InvalidExperiment("FORBIDDEN_FORMAL_PATH", "execution record path uses a forbidden formal ID")
    if record_path.is_symlink() or not record_path.is_file():
        raise InvalidExperiment("EXECUTION_RECORD_MISSING", "matched-triplet execution record is missing")
    record = _json_object(record_path.read_bytes(), "EXECUTION_RECORD_INVALID_JSON")
    _require_exact_keys(record, {"schema_version", "execution_schema", "protocol_id", "plan_sha256", "prerequisite_identity", "synthetic_fixture", "groups"}, "EXECUTION_SCHEMA_MISMATCH")
    if record["schema_version"] != 1 or record["execution_schema"] != EXECUTION_SCHEMA or record["protocol_id"] != PROTOCOL_ID:
        raise InvalidExperiment("EXECUTION_SCHEMA_MISMATCH", "execution version or protocol changed")
    if record["plan_sha256"] != PLAN_RAW_SHA256 or record["prerequisite_identity"] != preflight_result.prerequisite.identity():
        raise InvalidExperiment("EXECUTION_IDENTITY_MISMATCH", "execution plan or prerequisite identity changed")
    if record["synthetic_fixture"] is not synthetic_fixture:
        raise InvalidExperiment("EXECUTION_MODE_MISMATCH", "synthetic/formal execution mode is mislabeled")
    groups = record["groups"]
    plan_groups = preflight_result.plan["groups"]
    if not isinstance(groups, list) or [item.get("group_id") for item in groups if isinstance(item, Mapping)] != [item["group_id"] for item in plan_groups]:
        raise InvalidExperiment("TRIPLET_GROUP_SET_MISMATCH", "execution groups do not match the frozen plan")

    artifact_paths: dict[tuple[str, str], Path] = {}
    pending_hashes: list[tuple[str, Path, str]] = []
    for group_record, plan_group in zip(groups, plan_groups, strict=True):
        if not isinstance(group_record, Mapping):
            raise InvalidExperiment("TRIPLET_SCHEMA_MISMATCH", "triplet group must be an object")
        _require_exact_keys(group_record, {"group_id", "content_grammar", "prompt", "prompt_sha256", "seed", "matched_parameters", "conditions", "attempts"}, "TRIPLET_SCHEMA_MISMATCH")
        expected_parameters = expected_matched_parameters(preflight_result.config, plan_group)
        if group_record["group_id"] != plan_group["group_id"] or group_record["content_grammar"] != plan_group["content_grammar"]:
            raise InvalidExperiment("TRIPLET_CONTENT_IDENTITY_MISMATCH", "group identity or content grammar changed")
        if group_record["prompt"] != plan_group["prompt"] or group_record["prompt_sha256"] != expected_parameters["prompt_sha256"] or group_record["seed"] != plan_group["seed"]:
            raise InvalidExperiment("TRIPLET_PROMPT_OR_SEED_MISMATCH", "prompt or seed changed within the execution plan")
        if group_record["matched_parameters"] != expected_parameters:
            raise InvalidExperiment("TRIPLET_PARAMETER_MISMATCH", "model/sampling/encoding parameters changed")
        parameter_sha = sha256_bytes(canonical_json_bytes(expected_parameters))
        conditions = group_record["conditions"]
        if not isinstance(conditions, list) or [item.get("condition") for item in conditions if isinstance(item, Mapping)] != list(CONDITIONS):
            raise InvalidExperiment("TRIPLET_CONDITION_SET_MISMATCH", "triplet must contain exactly ordered OFF/A/B")
        latent_hashes: set[str] = set()
        for condition_record in conditions:
            _require_exact_keys(condition_record, {"condition", "schedule_id", "carrier_enabled", "initial_latent_sha256", "matched_parameters_sha256", "artifacts"}, "TRIPLET_SCHEMA_MISMATCH")
            condition = condition_record["condition"]
            expected_schedule = {"OFF": "NONE", "A": "A", "B": "B"}[condition]
            if condition_record["schedule_id"] != expected_schedule or condition_record["carrier_enabled"] is not (condition != "OFF"):
                raise InvalidExperiment("TRIPLET_CONDITION_LABEL_MISMATCH", "carrier condition or schedule is mislabeled")
            if not _is_sha256(condition_record["initial_latent_sha256"]):
                raise InvalidExperiment("TRIPLET_LATENT_IDENTITY_INVALID", "initial latent digest is malformed")
            latent_hashes.add(condition_record["initial_latent_sha256"])
            if condition_record["matched_parameters_sha256"] != parameter_sha:
                raise InvalidExperiment("TRIPLET_PARAMETER_MISMATCH", "condition parameters differ from the group")
            artifacts = condition_record["artifacts"]
            if not isinstance(artifacts, Mapping) or tuple(artifacts) != ARTIFACT_NAMES:
                raise InvalidExperiment("TRIPLET_ARTIFACT_SET_MISMATCH", "condition artifact set changed")
            for artifact_name in ARTIFACT_NAMES:
                identity = artifacts[artifact_name]
                if not isinstance(identity, Mapping):
                    raise InvalidExperiment("TRIPLET_ARTIFACT_IDENTITY_INVALID", "artifact identity must be an object")
                _require_exact_keys(identity, {"path", "sha256"}, "TRIPLET_ARTIFACT_IDENTITY_INVALID")
                if not _is_sha256(identity["sha256"]):
                    raise InvalidExperiment("TRIPLET_ARTIFACT_IDENTITY_INVALID", "artifact digest is malformed")
                path = _artifact_path(record_path, identity)
                pending_hashes.append((f"{group_record['group_id']}:{condition}:{artifact_name}", path, identity["sha256"]))
                if artifact_name == "features":
                    artifact_paths[(group_record["group_id"], condition)] = path
        if len(latent_hashes) != 1:
            raise InvalidExperiment("TRIPLET_LATENT_MISMATCH", "OFF/A/B do not share one actual initial latent identity")
        attempts = group_record["attempts"]
        if not isinstance(attempts, list):
            raise InvalidExperiment("TRIPLET_ATTEMPT_LOG_INVALID", "attempt log must be a list")
        for condition in CONDITIONS:
            condition_attempts = [item for item in attempts if isinstance(item, Mapping) and item.get("condition") == condition]
            if not condition_attempts or sum(item.get("outcome") == "success" for item in condition_attempts) != 1:
                raise InvalidExperiment("TRIPLET_ATTEMPT_LOG_INVALID", "each condition requires exactly one successful recorded attempt")
            for attempt_index, attempt in enumerate(condition_attempts):
                _require_exact_keys(attempt, {"condition", "attempt_index", "outcome", "matched_parameters_sha256"}, "TRIPLET_ATTEMPT_LOG_INVALID")
                if attempt["attempt_index"] != attempt_index or attempt["outcome"] not in {"success", "failed"} or attempt["matched_parameters_sha256"] != parameter_sha:
                    raise InvalidExperiment("TRIPLET_ATTEMPT_LOG_INVALID", "retry changed parameters or ordering")

    # All structure and identities are checked before any artifact is opened.
    for label, path, expected_sha in pending_hashes:
        if path.is_symlink() or not path.is_file():
            raise InvalidExperiment("TRIPLET_ARTIFACT_MISSING", f"execution artifact missing: {label}")
        if sha256_file(path) != expected_sha:
            raise InvalidExperiment("TRIPLET_ARTIFACT_HASH_MISMATCH", f"execution artifact digest changed: {label}")
    return record, artifact_paths


def _second_singular_value(matrix: np.ndarray) -> float:
    return float(np.linalg.svd(matrix, compute_uv=False)[1])


def _calibration(observations: np.ndarray, schedule: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(schedule[:4], dtype=np.float64)
    design = np.column_stack((target, np.ones(4)))
    beta, *_ = np.linalg.lstsq(design, observations, rcond=None)
    return beta[:2].T, beta[2]


def _equalize(observations: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    projector = np.linalg.inv(matrix.T @ matrix + 1e-4 * np.eye(2)) @ matrix.T
    return (projector @ (observations - bias).T).T


def window_metrics(observation: np.ndarray, start: int, schedule: Sequence[Sequence[float]]) -> dict[str, float]:
    if observation.shape != (13, 2) or not np.isfinite(observation).all() or start not in START_INDICES:
        raise InvalidExperiment("EVALUATOR_INPUT_INVALID", "observation shape or start index is invalid")
    schedule_array = np.asarray(schedule, dtype=np.float64)
    window = observation[start : start + 6]
    matrix, bias = _calibration(observation[[start, start + 1, start + 2, start + 3]], schedule)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    affine_s2 = float(singular_values[1])
    equalized = _equalize(observation[[start + 4, start + 5]], matrix, bias)
    centered = (observation - observation.mean(axis=0)) / math.sqrt(13)
    return {
        "residual": affine_residual(window, schedule_array[:6]),
        "global_s2": _second_singular_value(centered),
        "affine_s2": affine_s2,
        "condition": float(singular_values[0] / affine_s2) if affine_s2 else 1e300,
        "held_out_mse": float(np.mean((equalized - schedule_array[[4, 5]]) ** 2)),
    }


def threshold_checks(metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> dict[str, bool]:
    return {
        "residual": metrics["residual"] <= thresholds["max_residual"],
        "global_s2": metrics["global_s2"] >= thresholds["min_global_s2"],
        "affine_s2": metrics["affine_s2"] >= thresholds["min_affine_s2"],
        "condition": metrics["condition"] <= thresholds["max_condition"],
        "held_out_mse": metrics["held_out_mse"] <= thresholds["max_held_out_mse"],
    }


def observe_features(features: np.ndarray, prerequisite: FrozenPrerequisite) -> np.ndarray:
    if features.shape != (13, 30) or not np.isfinite(features).all():
        raise InvalidExperiment("FEATURE_MATRIX_INVALID", "saved-MP4 feature matrix must be finite 13x30")
    transformed = g0_transform(features, prerequisite.selected_candidate)
    return np.column_stack((transformed, np.ones(13))) @ prerequisite.readout


def _load_feature_artifact(path: Path, expected_video_sha: str, *, synthetic_fixture: bool) -> np.ndarray:
    payload = _json_object(path.read_bytes(), "FEATURE_ARTIFACT_INVALID")
    _require_exact_keys(payload, {"source", "video_sha256", "features"}, "FEATURE_ARTIFACT_INVALID")
    expected_source = "pure_synthetic_saved_mp4_fixture" if synthetic_fixture else "single_saved_mp4_only"
    if payload.get("source") != expected_source or payload.get("video_sha256") != expected_video_sha:
        raise InvalidExperiment("FEATURE_ARTIFACT_IDENTITY_MISMATCH", "feature source or saved-MP4 binding changed")
    try:
        features = np.asarray(payload["features"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidExperiment("FEATURE_MATRIX_INVALID", "feature matrix cannot be decoded") from exc
    if features.shape != (13, 30) or not np.isfinite(features).all():
        raise InvalidExperiment("FEATURE_MATRIX_INVALID", "feature matrix must be finite 13x30")
    return features


def evaluate_execution(record: Mapping[str, Any], record_path: Path, artifact_paths: Mapping[tuple[str, str], Path], preflight_result: Preflight, *, synthetic_fixture: bool) -> tuple[list[dict[str, Any]], bool]:
    schedules = {"A": schedule_a(), "B": schedule_b()}
    group_results: list[dict[str, Any]] = []
    for group in record["groups"]:
        condition_results: list[dict[str, Any]] = []
        for condition_record in group["conditions"]:
            condition = condition_record["condition"]
            video_sha = condition_record["artifacts"]["video"]["sha256"]
            features = _load_feature_artifact(artifact_paths[(group["group_id"], condition)], video_sha, synthetic_fixture=synthetic_fixture)
            observation = observe_features(features, preflight_result.prerequisite)
            templates: dict[str, list[dict[str, Any]]] = {}
            for template_id in TEMPLATES:
                windows: list[dict[str, Any]] = []
                for start in START_INDICES:
                    metrics = window_metrics(observation, start, schedules[template_id])
                    checks = threshold_checks(metrics, preflight_result.prerequisite.thresholds)
                    windows.append({"start": start, "metrics": metrics, "checks": checks, "accepted": all(checks.values())})
                templates[template_id] = windows
            if condition == "A":
                case_pass = templates["A"][0]["accepted"] and not any(item["accepted"] for item in templates["A"][1:]) and not any(item["accepted"] for item in templates["B"])
            elif condition == "B":
                case_pass = templates["B"][0]["accepted"] and not any(item["accepted"] for item in templates["B"][1:]) and not any(item["accepted"] for item in templates["A"])
            else:
                case_pass = not any(item["accepted"] for template in templates.values() for item in template)
            condition_results.append({"condition": condition, "observation_sha256": sha256_bytes(canonical_json_bytes(observation.tolist())), "templates": templates, "case_pass": case_pass})
        group_pass = all(item["case_pass"] for item in condition_results)
        group_results.append({"group_id": group["group_id"], "conditions": condition_results, "group_pass": group_pass})
    return group_results, all(item["group_pass"] for item in group_results)


def base_audit() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "output_schema": OUTPUT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "g0_independent_implementation_gate": G0_IMPLEMENTATION_GATE,
        "formal_result": False,
        "stage_progression_allowed": False,
    }


def prerequisite_audit(error: PrerequisiteNotMet, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    audit = base_audit()
    audit.update({"status": STATUS_PREREQUISITE, "valid_experiment": False, "reason_code": error.reason_code, "reason_detail": error.detail, "generation_inputs_accessed": False, "science_metrics_present": False})
    if context:
        audit["integrity_context"] = dict(context)
    return audit


def invalid_audit(error: InvalidExperiment, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    audit = base_audit()
    audit.update({"status": STATUS_INVALID, "valid_experiment": False, "reason_code": error.reason_code, "reason_detail": error.detail, "science_metrics_present": False})
    if context:
        audit["integrity_context"] = dict(context)
    return audit


def valid_audit(preflight_result: Preflight, execution_record: Mapping[str, Any], execution_record_sha256: str, groups: list[dict[str, Any]], passed: bool) -> dict[str, Any]:
    execution_identity = [
        {
            "group_id": group["group_id"],
            "content_grammar": group["content_grammar"],
            "prompt_sha256": group["prompt_sha256"],
            "seed": group["seed"],
            "matched_parameters": group["matched_parameters"],
            "conditions": [
                {
                    "condition": condition["condition"],
                    "schedule_id": condition["schedule_id"],
                    "initial_latent_sha256": condition["initial_latent_sha256"],
                    "artifacts": condition["artifacts"],
                }
                for condition in group["conditions"]
            ],
            "attempts": group["attempts"],
        }
        for group in execution_record["groups"]
    ]
    audit = base_audit()
    audit.update(
        {
            "status": STATUS_PASS if passed else STATUS_FAIL,
            "valid_experiment": True,
            "reason_code": "ALL_MATCHED_TRIPLETS_PASSED" if passed else "AT_LEAST_ONE_CASE_FAILED",
            "conclusion": PASS_CONCLUSION if passed else "frozen RC1 screen did not pass every required case",
            "source_state": preflight_result.source_state.as_dict(),
            "source_file_sha256": preflight_result.source_hashes,
            "manifest_sha256": sha256_bytes(preflight_result.manifest_bytes),
            "config_sha256": sha256_bytes(preflight_result.config_bytes),
            "plan_sha256": sha256_bytes(preflight_result.plan_bytes),
            "prerequisite_identity": preflight_result.prerequisite.identity(),
            "frozen_frontend": frontend_definition(preflight_result.prerequisite.selected_candidate),
            "frozen_thresholds": preflight_result.prerequisite.thresholds,
            "execution_record_sha256": execution_record_sha256,
            "triplet_execution_identity": execution_identity,
            "synthetic_fixture": execution_record["synthetic_fixture"],
            "schedule_preflight": schedule_preflight(),
            "evaluation_budget": {"templates": list(TEMPLATES), "start_indices": list(START_INDICES), "equal_for_all_conditions": True},
            "groups": groups,
            "aggregation": "all_cases_in_all_groups_no_averaging_or_majority_vote",
        }
    )
    return audit
