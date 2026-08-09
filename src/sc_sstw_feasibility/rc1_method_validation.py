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
import shlex
import subprocess
import shutil
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
from .rc1_prerequisite import (
    FRONTEND_SCHEMA,
    G0_SUCCESS_PACKAGE_FILES,
    PREREQUISITE_FILES,
    READOUT_SCHEMA,
    frontend_definition,
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
PREREQUISITE_LIBRARY_PATH = "src/sc_sstw_feasibility/rc1_prerequisite.py"
EXTRACTOR_LIBRARY_PATH = "src/sc_sstw_feasibility/learned_observation.py"
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
    PREREQUISITE_LIBRARY_PATH,
    EXTRACTOR_LIBRARY_PATH,
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
ARTIFACT_NAMES = ("video", "features", "stdout", "stderr", "config", "environment", "command", "integrity")
EVIDENCE_PRODUCTION = "production_saved_mp4"
EVIDENCE_SYNTHETIC = "test_only_synthetic"
EVIDENCE_CPU_HARNESS = "cpu_test_harness_saved_mp4"
PRODUCTION_EVIDENCE_SCHEMA = "sc_sstw_rc1_execution_production_saved_mp4_v1"
SYNTHETIC_EVIDENCE_SCHEMA = "sc_sstw_rc1_execution_test_only_synthetic_v1"
CPU_HARNESS_EVIDENCE_SCHEMA = "sc_sstw_rc1_execution_cpu_test_harness_saved_mp4_v1"
FEATURE_CACHE_SCHEMA = "sc_sstw_rc1_feature_cache_v1"
FEATURE_EXTRACTOR_ID = "sc_sstw_learned_observation_saved_mp4_13x30_v1"
FEATURE_CACHE_ATOL = 1e-12


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
        {"schema_version", "manifest_schema", "protocol_id", "expected_source", "source_files", "config", "plan", "prerequisite", "evidence_policy", "command_schema", "output_schema"},
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
    policy = manifest["evidence_policy"]
    if not isinstance(policy, Mapping):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "evidence policy must be an object")
    _require_exact_keys(policy, {"default_mode", "synthetic_fixture_permitted"}, "MANIFEST_SCHEMA_MISMATCH")
    if policy["default_mode"] != EVIDENCE_PRODUCTION or not isinstance(policy["synthetic_fixture_permitted"], bool):
        raise InvalidExperiment("EXECUTION_MODE_MISMATCH", "manifest evidence policy changed")
    command = manifest["command_schema"]
    if command != {
        "runner": RUNNER_PATH,
        "required_arguments": ["--manifest", "--output"],
        "optional_arguments": ["--source-commit", "--execution-package", "--generate", "--synthetic-fixture"],
    }:
        raise InvalidExperiment("COMMAND_SCHEMA_MISMATCH", "command schema is not authorized")


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
    return parsed


def _validate_thresholds(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(THRESHOLD_KEYS):
        raise InvalidExperiment("PREREQUISITE_THRESHOLDS_INVALID", "threshold key set changed")
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
    if "audit.json" not in declared:
        raise InvalidExperiment("PREREQUISITE_CHECKSUMS_INVALID", "G0 audit is absent from the checksum set")
    raw_files: dict[str, bytes] = {}
    for name in declared:
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
    if tuple(sorted(declared)) != tuple(sorted(G0_SUCCESS_PACKAGE_FILES)):
        raise InvalidExperiment("PREREQUISITE_CHECKSUMS_INVALID", "successful G0 package file set changed")
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
    manifest_identity = audit.get("manifest_identity")
    if not isinstance(manifest_identity, Mapping) or not _is_sha256(manifest_identity.get("sha256")):
        raise InvalidExperiment("PREREQUISITE_MANIFEST_IDENTITY_INVALID", "L1-v2 manifest identity is incomplete")
    if sha256_bytes(raw_files["authorization_manifest.json"]) != manifest_identity["sha256"]:
        raise InvalidExperiment("PREREQUISITE_MANIFEST_IDENTITY_INVALID", "packaged G0 manifest differs from the audit binding")
    if sha256_bytes(raw_files["config.json"]) != config_identity["sha256"]:
        raise InvalidExperiment("PREREQUISITE_CONFIG_IDENTITY_INVALID", "packaged G0 config differs from the audit binding")
    try:
        command_parts = shlex.split(raw_files["command.txt"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise InvalidExperiment("PREREQUISITE_COMMAND_INVALID", "packaged G0 command is malformed") from exc
    if not any(part.endswith(G0_REQUIRED_SOURCE_PATHS[1]) for part in command_parts) or "--manifest" not in command_parts or "--output" not in command_parts:
        raise InvalidExperiment("PREREQUISITE_COMMAND_INVALID", "packaged G0 command does not identify the authorized runner")
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
    _require_exact_keys(frontend, {"schema_version", "frontend_schema", "selected_candidate", "frontend_definition_sha256", "readout_path", "readout_sha256", "thresholds", "g0_identity"}, "PREREQUISITE_FRONTEND_INVALID")
    expected_frontend_sha = sha256_bytes(canonical_json_bytes(frontend_definition(selected)))
    if frontend["schema_version"] != 2 or frontend["frontend_schema"] != FRONTEND_SCHEMA or frontend["selected_candidate"] != selected or frontend["frontend_definition_sha256"] != expected_frontend_sha:
        raise InvalidExperiment("PREREQUISITE_FRONTEND_INVALID", "frontend definition or selected candidate changed")
    if frontend["readout_path"] != "readout.json" or frontend["readout_sha256"] != declared["readout.json"]:
        raise InvalidExperiment("PREREQUISITE_READOUT_IDENTITY_INVALID", "readout identity changed")
    expected_g0_identity = {
        "protocol_id": G0_PROTOCOL_ID,
        "output_schema": G0_OUTPUT_SCHEMA,
        "audit_sha256": declared["audit.json"],
        "manifest_sha256": manifest_identity["sha256"],
        "config_identity": dict(config_identity),
        "source_state": {"head": source_state["head"], "tree": source_state["tree"], "dirty": False},
        "source_file_sha256": dict(source_hashes),
        "input_sha256": dict(input_hashes),
    }
    if frontend["g0_identity"] != expected_g0_identity:
        raise InvalidExperiment("PREREQUISITE_SOURCE_IDENTITY_INVALID", "frontend binding to G0 audit/source/input changed")
    if _validate_thresholds(frontend["thresholds"]) != thresholds:
        raise InvalidExperiment("PREREQUISITE_THRESHOLDS_INVALID", "frontend thresholds differ from selected-candidate trace")

    readout_payload = _json_object(raw_files["readout.json"], "PREREQUISITE_READOUT_INVALID")
    _require_exact_keys(readout_payload, {"schema_version", "readout_schema", "selected_candidate", "shape", "fit_dataset_ids", "input_sha256", "source_head", "source_tree", "config_sha256", "coefficients"}, "PREREQUISITE_READOUT_INVALID")
    if (
        readout_payload["schema_version"] != 2
        or readout_payload["readout_schema"] != READOUT_SCHEMA
        or readout_payload["selected_candidate"] != selected
        or readout_payload["shape"] != [31, 2]
        or readout_payload["fit_dataset_ids"] != list(G0_INPUT_IDS[:4])
        or readout_payload["input_sha256"] != {str(value): input_hashes[str(value)] for value in G0_INPUT_IDS[:4]}
        or readout_payload["source_head"] != source_state["head"]
        or readout_payload["source_tree"] != source_state["tree"]
        or readout_payload["config_sha256"] != config_identity["sha256"]
    ):
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


def condition_config_payload(preflight_result: Preflight, group: Mapping[str, Any], condition: str, latent_sha: str, saved_video_path: str, evidence_mode: str) -> dict[str, Any]:
    parameters = expected_matched_parameters(preflight_result.config, group)
    return {
        "schema_version": 1,
        "artifact_schema": "sc_sstw_rc1_condition_config_v1",
        "protocol_id": PROTOCOL_ID,
        "evidence_mode": evidence_mode,
        "group_id": group["group_id"],
        "condition": condition,
        "schedule_id": {"OFF": "NONE", "A": "A", "B": "B"}[condition],
        "carrier_enabled": condition != "OFF",
        "initial_latent_sha256": latent_sha,
        "matched_parameters": parameters,
        "matched_parameters_sha256": sha256_bytes(canonical_json_bytes(parameters)),
        "frozen_prerequisite_identity": preflight_result.prerequisite.identity(),
        "saved_video_path": saved_video_path,
    }


def command_artifact_payload(preflight_result: Preflight, evidence_mode: str) -> dict[str, Any]:
    action = {
        EVIDENCE_PRODUCTION: "generate",
        EVIDENCE_CPU_HARNESS: "cpu_test_harness_generate",
        EVIDENCE_SYNTHETIC: "validate_test_only_synthetic_execution",
    }[evidence_mode]
    return {
        "schema_version": 1,
        "artifact_schema": "sc_sstw_rc1_execution_command_v1",
        "evidence_mode": evidence_mode,
        "runner": RUNNER_PATH,
        "action": action,
        "source_commit": preflight_result.source_state.head,
        "manifest_sha256": sha256_bytes(preflight_result.manifest_bytes),
        "synthetic_fixture_authorized": evidence_mode == EVIDENCE_SYNTHETIC,
    }


def synthetic_environment_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_schema": "sc_sstw_rc1_execution_environment_v1",
        "evidence_mode": EVIDENCE_SYNTHETIC,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "runtime": {"kind": "pure_synthetic_test_runtime"},
        "scheduler": {"kind": "pure_synthetic_test_only"},
        "sampler": {"kind": "pure_synthetic_test_only"},
    }


def integrity_artifact_payload(
    preflight_result: Preflight,
    group: Mapping[str, Any],
    condition: str,
    latent_sha: str,
    evidence_mode: str,
    carrier_records: Sequence[Mapping[str, Any]],
    codec_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parameters = expected_matched_parameters(preflight_result.config, group)
    extractor = (
        {
            "id": FEATURE_EXTRACTOR_ID,
            "source_path": EXTRACTOR_LIBRARY_PATH,
            "source_sha256": preflight_result.source_hashes[EXTRACTOR_LIBRARY_PATH],
            "cache_atol": FEATURE_CACHE_ATOL,
            "cache_rtol": 0.0,
        }
        if evidence_mode in {EVIDENCE_PRODUCTION, EVIDENCE_CPU_HARNESS}
        else {"id": "test_only_synthetic_feature_matrix_v1"}
    )
    return {
        "schema_version": 1,
        "artifact_schema": "sc_sstw_rc1_execution_integrity_v1",
        "evidence_mode": evidence_mode,
        "group_id": group["group_id"],
        "condition": condition,
        "prompt_sha256": parameters["prompt_sha256"],
        "seed": parameters["seed"],
        "initial_latent_sha256": latent_sha,
        "model_id": parameters["model_id"],
        "model_revision": parameters["model_revision"],
        "matched_parameters_sha256": sha256_bytes(canonical_json_bytes(parameters)),
        "schedule_id": {"OFF": "NONE", "A": "A", "B": "B"}[condition],
        "carrier_enabled": condition != "OFF",
        "carrier_records": [dict(item) for item in carrier_records],
        "codec_identity": dict(codec_identity) if codec_identity is not None else None,
        "feature_extractor": extractor,
    }


def _validate_environment_payload(payload: Mapping[str, Any], config: Mapping[str, Any], evidence_mode: str) -> None:
    _require_exact_keys(payload, {"schema_version", "artifact_schema", "evidence_mode", "model_id", "model_revision", "runtime", "scheduler", "sampler"}, "EXECUTION_ENVIRONMENT_INVALID")
    if payload["schema_version"] != 1 or payload["artifact_schema"] != "sc_sstw_rc1_execution_environment_v1" or payload["evidence_mode"] != evidence_mode:
        raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "environment schema or evidence mode changed")
    if payload["model_id"] != config["model"]["id"] or payload["model_revision"] != config["model"]["revision"]:
        raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "environment model identity changed")
    if evidence_mode == EVIDENCE_SYNTHETIC:
        if payload != synthetic_environment_payload(config):
            raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "synthetic environment identity changed")
        return
    runtime = payload["runtime"]
    scheduler = payload["scheduler"]
    sampler = payload["sampler"]
    if not isinstance(runtime, Mapping) or set(runtime) != {"python", "platform", "torch", "diffusers", "cuda", "gpu", "versions"}:
        raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "production runtime identity is incomplete")
    if any(not isinstance(runtime[key], str) or not runtime[key].strip() for key in ("python", "platform", "torch", "diffusers", "cuda", "gpu")):
        raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "production runtime contains an empty identity")
    placeholder_tokens = ("placeholder", "unknown", "test-only", "synthetic", "not-executed")
    if any(any(token in runtime[key].lower() for token in placeholder_tokens) for key in ("platform", "torch", "diffusers", "cuda", "gpu")):
        raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "production runtime contains a placeholder identity")
    if not isinstance(runtime["versions"], Mapping) or not runtime["versions"] or any(not isinstance(value, str) or not value for value in runtime["versions"].values()):
        raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "production dependency identity is incomplete")
    for value in (scheduler, sampler):
        if not isinstance(value, Mapping) or set(value) != {"class", "config_sha256", "bound_to_model_revision"}:
            raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "scheduler or sampler identity is incomplete")
        if not isinstance(value["class"], str) or not value["class"].strip() or not _is_sha256(value["config_sha256"]) or value["bound_to_model_revision"] != config["model"]["revision"]:
            raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "scheduler or sampler identity is contradictory")
        if any(token in value["class"].lower() for token in placeholder_tokens):
            raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "scheduler or sampler class is a placeholder")
    if scheduler != sampler:
        raise InvalidExperiment("EXECUTION_ENVIRONMENT_INVALID", "scheduler and sampler frozen identity differ")


def _validate_carrier_records(records: Any, condition: str, config: Mapping[str, Any], evidence_mode: str) -> None:
    if condition == "OFF":
        if records != []:
            raise InvalidExperiment("EXECUTION_CARRIER_RECORDS_INVALID", "OFF condition must have no carrier records")
        return
    if not isinstance(records, list) or len(records) != 16:
        raise InvalidExperiment("EXECUTION_CARRIER_RECORDS_INVALID", "A/B condition requires exactly 16 carrier records")
    expected_schedule = schedule_a() if condition == "A" else schedule_b()
    expected_schedule_sha = sha256_bytes(canonical_json_bytes(expected_schedule))
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise InvalidExperiment("EXECUTION_CARRIER_RECORDS_INVALID", "carrier record must be an object")
        required = {"call_index", "module_path", "schedule_sha256", "effective_relative_rms", "evidence_mode"}
        if not required.issubset(record) or record["call_index"] != index or record["module_path"] != config["carrier"]["module_path"] or record["schedule_sha256"] != expected_schedule_sha or record["evidence_mode"] != evidence_mode:
            raise InvalidExperiment("EXECUTION_CARRIER_RECORDS_INVALID", "carrier record identity or order changed")
        try:
            rms = float(record["effective_relative_rms"])
        except (TypeError, ValueError) as exc:
            raise InvalidExperiment("EXECUTION_CARRIER_RECORDS_INVALID", "carrier RMS is malformed") from exc
        if not math.isfinite(rms) or abs(rms - float(config["carrier"]["target_relative_rms"])) > float(config["carrier"]["target_relative_rms_absolute_tolerance"]):
            raise InvalidExperiment("EXECUTION_CARRIER_RECORDS_INVALID", "carrier RMS differs from the frozen target")
        if evidence_mode in {EVIDENCE_PRODUCTION, EVIDENCE_CPU_HARNESS}:
            production_fields = {"output_shape", "output_dtype", "input_tensor_sha256", "modified_tensor_sha256", "distinct_storage"}
            if (
                not production_fields.issubset(record)
                or record.get("distinct_storage") is not True
                or not isinstance(record.get("output_shape"), list)
                or not record["output_shape"]
                or any(not isinstance(item, int) or item <= 0 for item in record["output_shape"])
                or record.get("output_dtype") != config["generation"]["dtype"]
            ):
                raise InvalidExperiment("EXECUTION_CARRIER_RECORDS_INVALID", "production carrier record lacks actual tensor integrity")
            if not _is_sha256(record["input_tensor_sha256"]) or not _is_sha256(record["modified_tensor_sha256"]) or record["input_tensor_sha256"] == record["modified_tensor_sha256"]:
                raise InvalidExperiment("EXECUTION_CARRIER_RECORDS_INVALID", "production carrier tensor hashes are malformed or show no hook effect")
        if evidence_mode == EVIDENCE_SYNTHETIC and record.get("test_only_synthetic") is not True:
            raise InvalidExperiment("EXECUTION_CARRIER_RECORDS_INVALID", "synthetic carrier record lacks test-only labeling")


def _load_artifact_json(path: Path, reason: str) -> dict[str, Any]:
    try:
        return _json_object(path.read_bytes(), reason)
    except OSError as exc:
        raise InvalidExperiment(reason, "semantic artifact could not be read") from exc


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


def validate_execution_record(
    record_path: Path,
    preflight_result: Preflight,
    *,
    synthetic_fixture: bool,
    generation_receipt: object | None = None,
    allow_cpu_test_harness: bool = False,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Path]]]:
    receipt: dict[str, Any] | None = None
    if synthetic_fixture:
        expected_mode = EVIDENCE_SYNTHETIC
        expected_evidence_schema = SYNTHETIC_EVIDENCE_SCHEMA
        if generation_receipt is not None:
            raise InvalidExperiment("GENERATION_RECEIPT_MODE_MISMATCH", "synthetic execution cannot consume a production generation receipt")
    else:
        try:
            from .rc1_gpu_generation import RC1GPUGenerationError, consume_generation_receipt

            receipt = consume_generation_receipt(generation_receipt, record_path)
        except (RC1GPUGenerationError, AttributeError, TypeError) as exc:
            raise InvalidExperiment("GENERATION_RECEIPT_INVALID", "production execution requires a live same-process generation receipt") from exc
        expected_preflight = {
            "manifest_sha256": sha256_bytes(preflight_result.manifest_bytes),
            "config_sha256": sha256_bytes(preflight_result.config_bytes),
            "plan_sha256": sha256_bytes(preflight_result.plan_bytes),
            "prerequisite_identity": preflight_result.prerequisite.identity(),
            "source_head": preflight_result.source_state.head,
            "source_tree": preflight_result.source_state.tree,
            "source_hashes": preflight_result.source_hashes,
            "command_schema": preflight_result.manifest["command_schema"],
            "output_schema": preflight_result.manifest["output_schema"],
        }
        expected_implementation = {
            "generation_module_path": GPU_LIBRARY_PATH,
            "generation_module_sha256": preflight_result.source_hashes[GPU_LIBRARY_PATH],
            "extractor_path": EXTRACTOR_LIBRARY_PATH,
            "extractor_sha256": preflight_result.source_hashes[EXTRACTOR_LIBRARY_PATH],
        }
        if receipt.get("trust_boundary") != "same_clean_source_runner_process_observed_generation_call" or receipt.get("preflight") != expected_preflight or receipt.get("implementation") != expected_implementation:
            raise InvalidExperiment("GENERATION_RECEIPT_IDENTITY_MISMATCH", "generation receipt differs from the frozen source or preflight")
        loaded = receipt.get("loaded_identity")
        receipt_environment = receipt.get("environment")
        expected_dtype = str(preflight_result.config["generation"]["dtype"])
        if (
            not isinstance(loaded, Mapping)
            or not isinstance(receipt_environment, Mapping)
            or loaded.get("model_id") != preflight_result.config["model"]["id"]
            or loaded.get("loaded_revision") != preflight_result.config["model"]["revision"]
            or loaded.get("scheduler") != receipt_environment.get("scheduler")
            or loaded.get("dtype") not in {expected_dtype, f"torch.{expected_dtype}"}
            or not isinstance(loaded.get("device"), str)
            or not loaded["device"]
        ):
            raise InvalidExperiment("GENERATION_RECEIPT_RUNTIME_MISMATCH", "loaded model/runtime identity differs from the frozen generation")
        if receipt.get("record_path") != str(record_path.resolve()):
            raise InvalidExperiment("GENERATION_RECEIPT_PATH_MISMATCH", "generation receipt is not bound to this execution package")
        authority_provenance = receipt.get("authority_provenance_class")
        if authority_provenance == "production_generation":
            if receipt.get("provenance_class") != "production_generation" or receipt.get("cpu_only_test_harness") is not False:
                raise InvalidExperiment("GENERATION_RECEIPT_PROVENANCE_MISMATCH", "production authority metadata is contradictory")
            expected_mode = EVIDENCE_PRODUCTION
            expected_evidence_schema = PRODUCTION_EVIDENCE_SCHEMA
        elif authority_provenance == "cpu_test_harness":
            if receipt.get("provenance_class") != "cpu_test_harness" or receipt.get("cpu_only_test_harness") is not True:
                raise InvalidExperiment("GENERATION_RECEIPT_PROVENANCE_MISMATCH", "CPU harness authority metadata is contradictory")
            if not allow_cpu_test_harness:
                raise InvalidExperiment("CPU_TEST_HARNESS_FORBIDDEN", "CPU-only generation harness cannot enter the production CLI")
            if loaded["device"] != "cpu":
                raise InvalidExperiment("GENERATION_RECEIPT_RUNTIME_MISMATCH", "CPU harness device observation changed")
            expected_mode = EVIDENCE_CPU_HARNESS
            expected_evidence_schema = CPU_HARNESS_EVIDENCE_SCHEMA
        else:
            raise InvalidExperiment("GENERATION_RECEIPT_PROVENANCE_MISMATCH", "receipt provenance class is not authority registered")
    if _forbidden_path(str(record_path)):
        raise InvalidExperiment("FORBIDDEN_FORMAL_PATH", "execution record path uses a forbidden formal ID")
    if record_path.is_symlink() or not record_path.is_file():
        raise InvalidExperiment("EXECUTION_RECORD_MISSING", "matched-triplet execution record is missing")
    record_bytes = record_path.read_bytes()
    if receipt is not None and receipt.get("execution_record_sha256") != sha256_bytes(record_bytes):
        raise InvalidExperiment("GENERATION_RECEIPT_DISK_MISMATCH", "execution record differs from the in-process generation receipt")
    record = _json_object(record_bytes, "EXECUTION_RECORD_INVALID_JSON")
    _require_exact_keys(record, {"schema_version", "execution_schema", "protocol_id", "plan_sha256", "prerequisite_identity", "evidence_mode", "evidence_schema", "groups"}, "EXECUTION_SCHEMA_MISMATCH")
    if record["schema_version"] != 1 or record["execution_schema"] != EXECUTION_SCHEMA or record["protocol_id"] != PROTOCOL_ID:
        raise InvalidExperiment("EXECUTION_SCHEMA_MISMATCH", "execution version or protocol changed")
    if record["plan_sha256"] != PLAN_RAW_SHA256 or record["prerequisite_identity"] != preflight_result.prerequisite.identity():
        raise InvalidExperiment("EXECUTION_IDENTITY_MISMATCH", "execution plan or prerequisite identity changed")
    if record["evidence_mode"] != expected_mode or record["evidence_schema"] != expected_evidence_schema:
        raise InvalidExperiment("EXECUTION_MODE_MISMATCH", "synthetic/formal execution mode is mislabeled")
    groups = record["groups"]
    plan_groups = preflight_result.plan["groups"]
    if not isinstance(groups, list) or [item.get("group_id") for item in groups if isinstance(item, Mapping)] != [item["group_id"] for item in plan_groups]:
        raise InvalidExperiment("TRIPLET_GROUP_SET_MISMATCH", "execution groups do not match the frozen plan")

    receipt_groups = receipt.get("groups") if receipt is not None else None
    if receipt is not None and (not isinstance(receipt_groups, list) or [item.get("group_id") for item in receipt_groups if isinstance(item, Mapping)] != [item["group_id"] for item in plan_groups]):
        raise InvalidExperiment("GENERATION_RECEIPT_TRIPLET_MISMATCH", "receipt triplet set differs from the frozen plan")
    artifact_paths: dict[tuple[str, str], dict[str, Path]] = {}
    pending_hashes: list[tuple[str, Path, str]] = []
    for group_index, (group_record, plan_group) in enumerate(zip(groups, plan_groups, strict=True)):
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
        receipt_group = receipt_groups[group_index] if receipt_groups is not None else None
        if receipt_group is not None and receipt_group.get("initial_latent_identity", {}).get("sha256") is None:
            raise InvalidExperiment("GENERATION_RECEIPT_LATENT_MISMATCH", "receipt lacks the actual initial latent tensor identity")
        group_environment: dict[str, Any] | None = None
        group_command: dict[str, Any] | None = None
        for condition_index, condition_record in enumerate(conditions):
            _require_exact_keys(condition_record, {"condition", "schedule_id", "carrier_enabled", "initial_latent_sha256", "matched_parameters_sha256", "artifacts"}, "TRIPLET_SCHEMA_MISMATCH")
            condition = condition_record["condition"]
            expected_schedule = {"OFF": "NONE", "A": "A", "B": "B"}[condition]
            if condition_record["schedule_id"] != expected_schedule or condition_record["carrier_enabled"] is not (condition != "OFF"):
                raise InvalidExperiment("TRIPLET_CONDITION_LABEL_MISMATCH", "carrier condition or schedule is mislabeled")
            if not _is_sha256(condition_record["initial_latent_sha256"]):
                raise InvalidExperiment("TRIPLET_LATENT_IDENTITY_INVALID", "initial latent digest is malformed")
            latent_hashes.add(condition_record["initial_latent_sha256"])
            receipt_condition = receipt_group["conditions"][condition_index] if receipt_group is not None and isinstance(receipt_group.get("conditions"), list) and len(receipt_group["conditions"]) == len(CONDITIONS) else None
            if receipt_group is not None:
                if not isinstance(receipt_condition, Mapping) or receipt_condition.get("condition") != condition:
                    raise InvalidExperiment("GENERATION_RECEIPT_TRIPLET_MISMATCH", "receipt condition set or order changed")
                initial_identity = receipt_group["initial_latent_identity"]
                if receipt_condition.get("initial_latent_identity") != initial_identity or receipt_condition.get("condition_latent_identity") != initial_identity or condition_record["initial_latent_sha256"] != initial_identity.get("sha256"):
                    raise InvalidExperiment("GENERATION_RECEIPT_LATENT_MISMATCH", "receipt and disk latent identities differ")
                if receipt_condition.get("carrier_hook_effect") is not (condition != "OFF"):
                    raise InvalidExperiment("GENERATION_RECEIPT_HOOK_MISMATCH", "receipt carrier hook effect differs from the condition")
            if condition_record["matched_parameters_sha256"] != parameter_sha:
                raise InvalidExperiment("TRIPLET_PARAMETER_MISMATCH", "condition parameters differ from the group")
            artifacts = condition_record["artifacts"]
            if not isinstance(artifacts, Mapping) or tuple(artifacts) != ARTIFACT_NAMES:
                raise InvalidExperiment("TRIPLET_ARTIFACT_SET_MISMATCH", "condition artifact set changed")
            condition_paths: dict[str, Path] = {}
            for artifact_name in ARTIFACT_NAMES:
                identity = artifacts[artifact_name]
                if not isinstance(identity, Mapping):
                    raise InvalidExperiment("TRIPLET_ARTIFACT_IDENTITY_INVALID", "artifact identity must be an object")
                _require_exact_keys(identity, {"path", "sha256"}, "TRIPLET_ARTIFACT_IDENTITY_INVALID")
                if receipt_condition is not None and receipt_condition.get("artifacts", {}).get(artifact_name) != identity:
                    raise InvalidExperiment("GENERATION_RECEIPT_DISK_MISMATCH", "receipt and disk artifact identities differ")
                if not _is_sha256(identity["sha256"]):
                    raise InvalidExperiment("TRIPLET_ARTIFACT_IDENTITY_INVALID", "artifact digest is malformed")
                path = _artifact_path(record_path, identity)
                pending_hashes.append((f"{group_record['group_id']}:{condition}:{artifact_name}", path, identity["sha256"]))
                condition_paths[artifact_name] = path
            artifact_paths[(group_record["group_id"], condition)] = condition_paths
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
    for group_index, (group_record, plan_group) in enumerate(zip(groups, plan_groups, strict=True)):
        group_environment = None
        group_command = None
        for condition_index, condition_record in enumerate(group_record["conditions"]):
            condition = condition_record["condition"]
            receipt_condition = receipt_groups[group_index]["conditions"][condition_index] if receipt_groups is not None else None
            paths = artifact_paths[(group_record["group_id"], condition)]
            config_payload = _load_artifact_json(paths["config"], "EXECUTION_CONFIG_INVALID")
            expected_config = condition_config_payload(
                preflight_result,
                plan_group,
                condition,
                condition_record["initial_latent_sha256"],
                condition_record["artifacts"]["video"]["path"],
                expected_mode,
            )
            if config_payload != expected_config:
                raise InvalidExperiment("EXECUTION_CONFIG_INVALID", "condition config differs from frozen parameters")
            environment_payload = _load_artifact_json(paths["environment"], "EXECUTION_ENVIRONMENT_INVALID")
            _validate_environment_payload(environment_payload, preflight_result.config, expected_mode)
            if receipt is not None and environment_payload != receipt.get("environment"):
                raise InvalidExperiment("GENERATION_RECEIPT_RUNTIME_MISMATCH", "disk runtime/model/scheduler differs from generation observation")
            if group_environment is None:
                group_environment = environment_payload
            elif environment_payload != group_environment:
                raise InvalidExperiment("TRIPLET_PARAMETER_MISMATCH", "OFF/A/B runtime environment differs")
            command_payload = _load_artifact_json(paths["command"], "EXECUTION_COMMAND_INVALID")
            if command_payload != command_artifact_payload(preflight_result, expected_mode):
                raise InvalidExperiment("EXECUTION_COMMAND_INVALID", "execution command identity changed")
            if receipt is not None and command_payload != receipt.get("command_artifact"):
                raise InvalidExperiment("GENERATION_RECEIPT_COMMAND_MISMATCH", "disk command differs from generation observation")
            if group_command is None:
                group_command = command_payload
            elif command_payload != group_command:
                raise InvalidExperiment("TRIPLET_PARAMETER_MISMATCH", "OFF/A/B command identity differs")
            integrity_payload = _load_artifact_json(paths["integrity"], "EXECUTION_INTEGRITY_INVALID")
            expected_integrity = integrity_artifact_payload(
                preflight_result,
                plan_group,
                condition,
                condition_record["initial_latent_sha256"],
                expected_mode,
                integrity_payload.get("carrier_records", []) if isinstance(integrity_payload, Mapping) else [],
                integrity_payload.get("codec_identity") if isinstance(integrity_payload, Mapping) else None,
            )
            if integrity_payload != expected_integrity:
                raise InvalidExperiment("EXECUTION_INTEGRITY_INVALID", "integrity metadata differs from frozen execution identity")
            _validate_carrier_records(integrity_payload["carrier_records"], condition, preflight_result.config, expected_mode)
            if receipt_condition is not None:
                if integrity_payload["carrier_records"] != receipt_condition.get("carrier_records"):
                    raise InvalidExperiment("GENERATION_RECEIPT_HOOK_MISMATCH", "disk carrier records differ from generation observation")
                if integrity_payload["codec_identity"] != receipt_condition.get("codec_identity"):
                    raise InvalidExperiment("GENERATION_RECEIPT_VIDEO_MISMATCH", "disk codec identity differs from generation observation")
                if receipt_condition.get("video_saved_complete") is not True or condition_record["artifacts"]["video"]["sha256"] != receipt_condition.get("video_sha256"):
                    raise InvalidExperiment("GENERATION_RECEIPT_VIDEO_MISMATCH", "saved video completion or digest differs from generation observation")
            if expected_mode == EVIDENCE_SYNTHETIC:
                synthetic_video = _load_artifact_json(paths["video"], "SYNTHETIC_VIDEO_IDENTITY_INVALID")
                expected_synthetic_video = {"schema_version": 1, "evidence_mode": EVIDENCE_SYNTHETIC, "artifact_kind": "synthetic_video_identity", "group_id": group_record["group_id"], "condition": condition}
                if synthetic_video != expected_synthetic_video:
                    raise InvalidExperiment("SYNTHETIC_VIDEO_IDENTITY_INVALID", "synthetic video identity is malformed")
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


def _inspect_and_decode_saved_mp4(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    if path.suffix.lower() != ".mp4":
        raise InvalidExperiment("SAVED_MP4_INVALID", "production video does not use an MP4 path")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise InvalidExperiment("SAVED_MP4_DECODER_UNAVAILABLE", "ffprobe is unavailable")
    try:
        completed = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames:format=format_name", "-of", "json", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        probe = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise InvalidExperiment("SAVED_MP4_INVALID", "saved MP4 could not be probed") from exc
    streams = probe.get("streams")
    format_name = probe.get("format", {}).get("format_name") if isinstance(probe.get("format"), Mapping) else None
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(format_name, str) or "mp4" not in format_name.split(","):
        raise InvalidExperiment("SAVED_MP4_INVALID", "saved MP4 container or stream count changed")
    stream = streams[0]
    expected_stream = {"codec_name": "h264", "pix_fmt": "yuv420p", "width": 512, "height": 320, "avg_frame_rate": "8/1", "nb_frames": "49"}
    if {key: stream.get(key) for key in expected_stream} != expected_stream:
        raise InvalidExperiment("SAVED_MP4_CODEC_OR_GEOMETRY_MISMATCH", "codec, dimensions, FPS, or frame count changed")
    try:
        from .learned_observation import decode_saved_mp4, extract_feature_matrix

        frames = decode_saved_mp4(path)
        features = np.asarray(extract_feature_matrix(frames), dtype=np.float64)
    except Exception as exc:
        raise InvalidExperiment("SAVED_MP4_DECODE_OR_EXTRACT_FAILURE", "saved MP4 decode or frozen feature extraction failed") from exc
    identity = {"container": "mp4", "codec_name": "h264", "pixel_format": "yuv420p", "width": 512, "height": 320, "fps": "8/1", "frame_count": 49, "decodable": True}
    return identity, features


def _load_feature_artifact(paths: Mapping[str, Path], expected_video_sha: str, preflight_result: Preflight, evidence_mode: str) -> np.ndarray:
    payload = _json_object(paths["features"].read_bytes(), "FEATURE_ARTIFACT_INVALID")
    _require_exact_keys(payload, {"schema_version", "feature_cache_schema", "evidence_mode", "source", "video_sha256", "extractor_identity", "comparison", "features"}, "FEATURE_ARTIFACT_INVALID")
    if payload["schema_version"] != 1 or payload["feature_cache_schema"] != FEATURE_CACHE_SCHEMA or payload["evidence_mode"] != evidence_mode or payload["video_sha256"] != expected_video_sha:
        raise InvalidExperiment("FEATURE_ARTIFACT_IDENTITY_MISMATCH", "feature schema, mode, or saved-MP4 binding changed")
    if evidence_mode in {EVIDENCE_PRODUCTION, EVIDENCE_CPU_HARNESS}:
        expected_extractor = {"id": FEATURE_EXTRACTOR_ID, "source_path": EXTRACTOR_LIBRARY_PATH, "source_sha256": preflight_result.source_hashes[EXTRACTOR_LIBRARY_PATH]}
        if payload["source"] != "recomputed_from_single_saved_mp4" or payload["extractor_identity"] != expected_extractor or payload["comparison"] != {"atol": FEATURE_CACHE_ATOL, "rtol": 0.0}:
            raise InvalidExperiment("FEATURE_EXTRACTOR_IDENTITY_MISMATCH", "frozen production extractor identity changed")
    else:
        if payload["source"] != "pure_synthetic_feature_fixture" or payload["extractor_identity"] != {"id": "test_only_synthetic_feature_matrix_v1"} or payload["comparison"] != {"atol": 0.0, "rtol": 0.0}:
            raise InvalidExperiment("FEATURE_ARTIFACT_IDENTITY_MISMATCH", "synthetic feature cache is not explicitly test-only")
    try:
        features = np.asarray(payload["features"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidExperiment("FEATURE_MATRIX_INVALID", "feature matrix cannot be decoded") from exc
    if features.shape != (13, 30) or not np.isfinite(features).all():
        raise InvalidExperiment("FEATURE_MATRIX_INVALID", "feature matrix must be finite 13x30")
    if evidence_mode in {EVIDENCE_PRODUCTION, EVIDENCE_CPU_HARNESS}:
        codec_identity, recomputed = _inspect_and_decode_saved_mp4(paths["video"])
        if recomputed.shape != (13, 30) or not np.isfinite(recomputed).all():
            raise InvalidExperiment("FEATURE_RECOMPUTE_INVALID", "frozen extractor did not return finite 13x30 features")
        if not np.allclose(features, recomputed, rtol=0.0, atol=FEATURE_CACHE_ATOL):
            raise InvalidExperiment("FEATURE_CACHE_RECOMPUTE_MISMATCH", "stored feature cache differs from saved-MP4 recomputation")
        integrity = _load_artifact_json(paths["integrity"], "EXECUTION_INTEGRITY_INVALID")
        if integrity.get("codec_identity") != codec_identity:
            raise InvalidExperiment("SAVED_MP4_IDENTITY_MISMATCH", "integrity metadata differs from independently decoded MP4")
        return recomputed
    return features


def evaluate_execution(record: Mapping[str, Any], record_path: Path, artifact_paths: Mapping[tuple[str, str], Mapping[str, Path]], preflight_result: Preflight, *, synthetic_fixture: bool) -> tuple[list[dict[str, Any]], bool]:
    schedules = {"A": schedule_a(), "B": schedule_b()}
    evidence_mode = record["evidence_mode"]
    if evidence_mode == EVIDENCE_SYNTHETIC and not synthetic_fixture:
        raise InvalidExperiment("EXECUTION_MODE_MISMATCH", "synthetic execution requires explicit test-only admission")
    if evidence_mode != EVIDENCE_SYNTHETIC and synthetic_fixture:
        raise InvalidExperiment("EXECUTION_MODE_MISMATCH", "saved-MP4 execution cannot enter the synthetic feature path")
    group_results: list[dict[str, Any]] = []
    for group in record["groups"]:
        condition_results: list[dict[str, Any]] = []
        for condition_record in group["conditions"]:
            condition = condition_record["condition"]
            video_sha = condition_record["artifacts"]["video"]["sha256"]
            features = _load_feature_artifact(artifact_paths[(group["group_id"], condition)], video_sha, preflight_result, evidence_mode)
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


def valid_audit(
    preflight_result: Preflight,
    execution_record: Mapping[str, Any],
    execution_record_sha256: str,
    groups: list[dict[str, Any]],
    passed: bool,
    *,
    cpu_only_test_harness: bool = False,
) -> dict[str, Any]:
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
            "conclusion": (
                "CPU-only control-flow harness exercised the passing branch; this is not an RC1 result"
                if passed and cpu_only_test_harness
                else PASS_CONCLUSION if passed else "frozen RC1 screen did not pass every required case"
            ),
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
            "evidence_mode": execution_record["evidence_mode"],
            "test_only_synthetic": execution_record["evidence_mode"] == EVIDENCE_SYNTHETIC,
            "cpu_only_test_harness": cpu_only_test_harness,
            "generation_trust_boundary": (
                "external_test_only_synthetic_fixture"
                if execution_record["evidence_mode"] == EVIDENCE_SYNTHETIC
                else "cpu_only_control_flow_harness" if cpu_only_test_harness
                else "same_clean_source_runner_process_observed_generation_call"
            ),
            "schedule_preflight": schedule_preflight(),
            "evaluation_budget": {"templates": list(TEMPLATES), "start_indices": list(START_INDICES), "equal_for_all_conditions": True},
            "groups": groups,
            "aggregation": "all_cases_in_all_groups_no_averaging_or_majority_vote",
        }
    )
    return audit
