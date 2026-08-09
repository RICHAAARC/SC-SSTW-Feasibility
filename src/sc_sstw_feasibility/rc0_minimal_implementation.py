"""Fail-closed RC0 causal-localization implementation primitives.

This module consumes either a future independently issued authorization manifest
or a self-identifying DIAGNOSTIC_ONLY bootstrap from an exact clean checkout,
validates a same-process generation record, reopens the saved MP4 files, and
applies the frozen RC0 Level P then conditional Level R formulas.  It neither
creates an authorization manifest nor authorizes execution by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from .rc0_causal_localization_v2 import (
    CONDITION_ORDER,
    GROUP_ORDER,
    STATUS_INSUFFICIENT,
    STATUS_INVALID,
    STATUS_P_FAIL,
    STATUS_P_PASS_R_FAIL,
    STATUS_P_PASS_R_PASS,
    TEMPLATES,
    level_p_metrics,
    level_r_metrics,
    validate_config,
    validate_plan,
)


PROTOCOL_ID = "sc_sstw_rc0_causal_localization_v2"
IMPLEMENTATION_SCHEMA = "sc_sstw_rc0_minimal_implementation_v1"
MANIFEST_SCHEMA = "sc_sstw_rc0_authorization_manifest_v1"
DIAGNOSTIC_BOOTSTRAP_SCHEMA = "sc_sstw_rc0_diag_fast_bootstrap_v1"
EXECUTION_SCHEMA = "sc_sstw_rc0_matched_quartet_execution_v1"
OUTPUT_SCHEMA = "sc_sstw_rc0_audit_package_v1"
FEATURE_CACHE_SCHEMA = "sc_sstw_rc0_saved_mp4_feature_cache_v1"

CONFIG_PATH = "configs/rc0_causal_localization_v2.json"
PLAN_PATH = "plans/rc0_matched_quartets_v2.json"
PROTOCOL_PATH = "protocols/rc0_causal_localization_v2.md"
FORMULA_PATH = "src/sc_sstw_feasibility/rc0_causal_localization_v2.py"
IMPLEMENTATION_PATH = "src/sc_sstw_feasibility/rc0_minimal_implementation.py"
GENERATION_PATH = "src/sc_sstw_feasibility/rc0_generation.py"
RUNNER_PATH = "experiments/run_rc0_causal_localization_v2.py"
NOTEBOOK_PATH = "notebooks/sc_sstw_rc0_causal_localization_v2.ipynb"
TEST_PATH = "tests/test_rc0_minimal_implementation.py"
EXTRACTOR_PATH = "src/sc_sstw_feasibility/learned_observation.py"
SCHEDULE_SOURCE_PATH = "src/sc_sstw_feasibility/learned_observation_l1_v2.py"
CARRIER_HELPER_PATH = "src/sc_sstw_feasibility/gpu_internal_challenger.py"
DIAGNOSTIC_PROTOCOL_PATH = "protocols/rc0_diag_fast.md"
DIAGNOSTIC_CONFIG_PATH = "configs/rc0_diag_fast.json"
REQUIRED_SOURCE_PATHS = (
    CONFIG_PATH,
    PLAN_PATH,
    PROTOCOL_PATH,
    FORMULA_PATH,
    IMPLEMENTATION_PATH,
    GENERATION_PATH,
    RUNNER_PATH,
    NOTEBOOK_PATH,
    TEST_PATH,
    EXTRACTOR_PATH,
    SCHEDULE_SOURCE_PATH,
    CARRIER_HELPER_PATH,
    DIAGNOSTIC_PROTOCOL_PATH,
    DIAGNOSTIC_CONFIG_PATH,
)

CONFIG_RAW_SHA256 = "17e8d2c6772f5c551deec560b7b5725d005e75ad2d27520950d8259be9ccdc6f"
PLAN_RAW_SHA256 = "cecd02c57a2528ae8afc3db38fea4e1cb9d541d58ed919e942515ca328bf372f"
PROTOCOL_RAW_SHA256 = "680e2cc78412c0ed8803e666ad4bc181c0dddca9695dc8e5969ebeedd31a1469"
EXTRACTOR_IDENTITY = {
    "path": EXTRACTOR_PATH,
    "symbol": "extract_feature_matrix",
    "source_commit": "fe8bc36461fdf40db917a3772a30ce6969a6c3a8",
    "source_tree": "90e50f107685c768600686239c922242e660d20a",
    "git_blob": "6288d954a1bdaded5fd2f92ed78b463bc11a6a18",
    "raw_sha256": "9c7fd37995d49344c2200a4855eaf6a547ea27336fdfe4fc652c62ac327b9866",
}

EVIDENCE_PRODUCTION = "production_saved_mp4"
EVIDENCE_CPU_HARNESS = "cpu_test_harness_saved_mp4"
PROVENANCE_PRODUCTION = "production_generation"
PROVENANCE_CPU_HARNESS = "cpu_test_harness"
ARTIFACT_NAMES = ("video", "features", "config", "environment", "command", "integrity", "stdout", "stderr")
FEATURE_CACHE_ATOL = 1e-12
FORBIDDEN_FORMAL_IDS = tuple(range(41001, 41009))

PACKAGE_STATUS_P_FAIL = "RC0_P_FAIL"
PACKAGE_STATUS_P_PASS_R_FAIL = "RC0_P_PASS_R_FAIL"
PACKAGE_STATUS_P_PASS_R_PASS = "RC0_P_PASS_R_PASS"
PACKAGE_STATUS_INVALID = STATUS_INVALID
PACKAGE_STATUS_INSUFFICIENT = STATUS_INSUFFICIENT
PACKAGE_TO_PROTOCOL_OUTCOME = {
    PACKAGE_STATUS_P_FAIL: STATUS_P_FAIL,
    PACKAGE_STATUS_P_PASS_R_FAIL: STATUS_P_PASS_R_FAIL,
    PACKAGE_STATUS_P_PASS_R_PASS: STATUS_P_PASS_R_PASS,
    PACKAGE_STATUS_INVALID: STATUS_INVALID,
    PACKAGE_STATUS_INSUFFICIENT: STATUS_INSUFFICIENT,
}
PROTOCOL_TO_PACKAGE_STATUS = {value: key for key, value in PACKAGE_TO_PROTOCOL_OUTCOME.items()}
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
DIAGNOSTIC_DECISION_CARRIER_NOT_FEASIBLE = "CURRENT_CARRIER_NOT_FEASIBLE"
DIAGNOSTIC_DECISION_SWITCH_VAE = "KEEP_CARRIER_SWITCH_TO_VAE_READOUT"
DIAGNOSTIC_DECISION_BUILD_BLIND = "KEEP_CARRIER_BUILD_BLIND_READOUT"
DIAGNOSTIC_DECISION_REDESIGN_RELATION = "REDESIGN_RELATION_CARRIER"
DIAGNOSTIC_DECISION_INSUFFICIENT = "DIAGNOSTIC_INSUFFICIENT"
DIAGNOSTIC_DECISIONS = (
    DIAGNOSTIC_DECISION_CARRIER_NOT_FEASIBLE,
    DIAGNOSTIC_DECISION_SWITCH_VAE,
    DIAGNOSTIC_DECISION_BUILD_BLIND,
    DIAGNOSTIC_DECISION_REDESIGN_RELATION,
    DIAGNOSTIC_DECISION_INSUFFICIENT,
)
DIAGNOSTIC_FEASIBLE = "FEASIBLE"
DIAGNOSTIC_NOT_FEASIBLE = "NOT_FEASIBLE"
DIAGNOSTIC_INSUFFICIENT_TO_DECIDE = "INSUFFICIENT_TO_DECIDE"


class InvalidExperiment(RuntimeError):
    """An integrity or schema violation with a stable reason code."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


class InsufficientEvidence(RuntimeError):
    """A validly identified run that cannot supply the frozen evidence."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class Preflight:
    repo_root: Path
    manifest: dict[str, Any]
    manifest_bytes: bytes
    config: dict[str, Any]
    config_bytes: bytes
    plan: dict[str, Any]
    plan_bytes: bytes
    protocol_bytes: bytes
    source_state: SourceState
    source_hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class Evaluation:
    package_status: str
    protocol_outcome: str
    reason_code: str
    level_p: tuple[dict[str, Any], ...]
    level_r: tuple[dict[str, Any], ...]
    science_metrics_present: bool


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_git_object(value: Any) -> bool:
    return type(value) is str and len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _exact_keys(value: Any, expected: Sequence[str], reason: str) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != tuple(expected):
        raise InvalidExperiment(reason, "object keys, order, or JSON object type changed")
    return value


def _exact_list(value: Any, expected: Sequence[Any], reason: str) -> None:
    if type(value) is not list or len(value) != len(expected):
        raise InvalidExperiment(reason, "array type or length changed")
    for actual_item, expected_item in zip(value, expected, strict=True):
        if type(actual_item) is not type(expected_item) or actual_item != expected_item:
            raise InvalidExperiment(reason, "array value, order, or JSON type changed")


def _require_exact_json(actual: Any, expected: Any, reason: str) -> None:
    if type(actual) is not type(expected):
        raise InvalidExperiment(reason, "JSON type changed")
    if type(expected) is dict:
        if tuple(actual) != tuple(expected):
            raise InvalidExperiment(reason, "object keys or order changed")
        for key in expected:
            _require_exact_json(actual[key], expected[key], reason)
        return
    if type(expected) is list:
        if len(actual) != len(expected):
            raise InvalidExperiment(reason, "array length changed")
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _require_exact_json(actual_item, expected_item, reason)
        return
    if actual != expected:
        raise InvalidExperiment(reason, "frozen JSON value changed")


def _json_object(raw: bytes, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidExperiment(reason, "JSON could not be decoded") from exc
    if type(value) is not dict:
        raise InvalidExperiment(reason, "JSON root must be an object")
    return value


def _forbidden_text(value: str) -> bool:
    lowered = value.lower()
    return any(str(dataset_id) in lowered for dataset_id in FORBIDDEN_FORMAL_IDS) or "diagnostic_staging" in lowered


def validate_status_mapping(package_status: Any, protocol_outcome: Any) -> None:
    if type(package_status) is not str or type(protocol_outcome) is not str:
        raise InvalidExperiment("STATUS_MAPPING_INVALID", "status mapping values must be strings")
    if PACKAGE_TO_PROTOCOL_OUTCOME.get(package_status) != protocol_outcome:
        raise InvalidExperiment("STATUS_MAPPING_INVALID", "package status and protocol outcome do not match the frozen mapping")


def validate_manifest_schema(manifest: Mapping[str, Any]) -> None:
    """Validate a future authorization object without creating one."""

    top = _exact_keys(
        manifest,
        (
            "schema_version",
            "manifest_schema",
            "implementation_schema",
            "protocol_id",
            "expected_source",
            "source_files",
            "frozen_identities",
            "experiment",
            "evidence_policy",
            "command_schema",
            "output_schema",
        ),
        "MANIFEST_SCHEMA_MISMATCH",
    )
    if (
        type(top["schema_version"]) is not int
        or top["schema_version"] != 1
        or top["manifest_schema"] != MANIFEST_SCHEMA
        or top["implementation_schema"] != IMPLEMENTATION_SCHEMA
        or top["protocol_id"] != PROTOCOL_ID
        or top["output_schema"] != OUTPUT_SCHEMA
    ):
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "manifest identity or JSON scalar type changed")

    source = _exact_keys(top["expected_source"], ("head", "tree", "dirty"), "MANIFEST_SCHEMA_MISMATCH")
    if not _is_git_object(source["head"]) or not _is_git_object(source["tree"]) or type(source["dirty"]) is not bool or source["dirty"] is not False:
        raise InvalidExperiment("MANIFEST_SCHEMA_MISMATCH", "expected clean source identity is malformed")

    source_files = _exact_keys(top["source_files"], REQUIRED_SOURCE_PATHS, "SOURCE_FILE_SET_MISMATCH")
    if not all(_is_sha256(source_files[path]) for path in REQUIRED_SOURCE_PATHS):
        raise InvalidExperiment("SOURCE_FILE_SET_MISMATCH", "required source digest is malformed")

    identities = _exact_keys(top["frozen_identities"], ("config", "plan", "protocol", "notebook", "extractor"), "MANIFEST_SCHEMA_MISMATCH")
    expected_simple = (
        ("config", CONFIG_PATH, CONFIG_RAW_SHA256),
        ("plan", PLAN_PATH, PLAN_RAW_SHA256),
        ("protocol", PROTOCOL_PATH, PROTOCOL_RAW_SHA256),
    )
    for name, path, digest in expected_simple:
        identity = _exact_keys(identities[name], ("path", "sha256"), "MANIFEST_SCHEMA_MISMATCH")
        if identity != {"path": path, "sha256": digest}:
            raise InvalidExperiment(f"{name.upper()}_IDENTITY_MISMATCH", f"{name} identity changed")
    notebook = _exact_keys(identities["notebook"], ("path", "sha256"), "MANIFEST_SCHEMA_MISMATCH")
    if notebook["path"] != NOTEBOOK_PATH or not _is_sha256(notebook["sha256"]) or notebook["sha256"] != source_files[NOTEBOOK_PATH]:
        raise InvalidExperiment("NOTEBOOK_IDENTITY_MISMATCH", "notebook identity changed")
    extractor = _exact_keys(
        identities["extractor"],
        ("path", "symbol", "source_commit", "source_tree", "git_blob", "raw_sha256"),
        "MANIFEST_SCHEMA_MISMATCH",
    )
    if dict(extractor) != EXTRACTOR_IDENTITY:
        raise InvalidExperiment("EXTRACTOR_IDENTITY_MISMATCH", "frozen extractor source identity changed")

    experiment = _exact_keys(
        top["experiment"],
        ("group_order", "condition_order", "attempt_order", "attempt_budget", "retry_policy"),
        "MANIFEST_SCHEMA_MISMATCH",
    )
    _exact_list(experiment["group_order"], list(GROUP_ORDER), "EXPERIMENT_IDENTITY_MISMATCH")
    _exact_list(experiment["condition_order"], list(CONDITION_ORDER), "EXPERIMENT_IDENTITY_MISMATCH")
    expected_attempts = [
        {"attempt_index": index, "group_id": group, "condition": condition}
        for index, (group, condition) in enumerate(
            ((group, condition) for group in GROUP_ORDER for condition in CONDITION_ORDER),
            start=1,
        )
    ]
    _require_exact_json(experiment["attempt_order"], expected_attempts, "EXPERIMENT_IDENTITY_MISMATCH")
    if type(experiment["attempt_budget"]) is not int or experiment["attempt_budget"] != 8 or experiment["retry_policy"] != "no_retry_no_replacement_no_additional_attempts":
        raise InvalidExperiment("EXPERIMENT_IDENTITY_MISMATCH", "budget or retry policy changed")

    policy = _exact_keys(
        top["evidence_policy"],
        ("production_entry", "external_production_package_permitted", "formal_result", "stage_progression_allowed"),
        "MANIFEST_SCHEMA_MISMATCH",
    )
    expected_policy = {
        "production_entry": "same_process_runner_generate_only",
        "external_production_package_permitted": False,
        "formal_result": False,
        "stage_progression_allowed": False,
    }
    _require_exact_json(policy, expected_policy, "EVIDENCE_POLICY_MISMATCH")

    command = _exact_keys(top["command_schema"], ("runner", "required_arguments", "forbidden_arguments"), "COMMAND_SCHEMA_MISMATCH")
    expected_command = {
        "runner": RUNNER_PATH,
        "required_arguments": ["--generate", "--manifest", "--output", "--source-commit"],
        "forbidden_arguments": ["--execution-package", "--synthetic-fixture", "--backend", "--device", "--receipt"],
    }
    _require_exact_json(command, expected_command, "COMMAND_SCHEMA_MISMATCH")


def validate_diagnostic_bootstrap_schema(bootstrap: Mapping[str, Any]) -> None:
    """Validate a non-authorizing exact-checkout diagnostic identity object."""

    top = _exact_keys(
        bootstrap,
        (
            "schema_version",
            "bootstrap_schema",
            "diagnostic_class",
            "implementation_schema",
            "protocol_id",
            "expected_source",
            "source_files",
            "frozen_identities",
            "diagnostic_identities",
            "experiment",
            "evidence_policy",
            "command_schema",
            "output_schema",
        ),
        "DIAGNOSTIC_BOOTSTRAP_SCHEMA_MISMATCH",
    )
    if (
        type(top["schema_version"]) is not int
        or top["schema_version"] != 1
        or top["bootstrap_schema"] != DIAGNOSTIC_BOOTSTRAP_SCHEMA
        or top["diagnostic_class"] != DIAGNOSTIC_CLASS
    ):
        raise InvalidExperiment("DIAGNOSTIC_BOOTSTRAP_SCHEMA_MISMATCH", "diagnostic bootstrap identity changed")
    diagnostic_identities = _exact_keys(
        top["diagnostic_identities"],
        ("protocol", "config"),
        "DIAGNOSTIC_BOOTSTRAP_SCHEMA_MISMATCH",
    )
    for name, path in (("protocol", DIAGNOSTIC_PROTOCOL_PATH), ("config", DIAGNOSTIC_CONFIG_PATH)):
        identity = _exact_keys(diagnostic_identities[name], ("path", "sha256"), "DIAGNOSTIC_BOOTSTRAP_SCHEMA_MISMATCH")
        if identity != {"path": path, "sha256": top["source_files"].get(path)} or not _is_sha256(identity["sha256"]):
            raise InvalidExperiment("DIAGNOSTIC_IDENTITY_MISMATCH", f"diagnostic {name} identity changed")
    policy = _exact_keys(
        top["evidence_policy"],
        (
            "diagnostic_only",
            "authorization_claimed",
            "gate_claimed",
            "production_entry",
            "external_production_package_permitted",
            "formal_result",
            "stage_progression_allowed",
        ),
        "DIAGNOSTIC_BOOTSTRAP_SCHEMA_MISMATCH",
    )
    _require_exact_json(
        policy,
        {
            "diagnostic_only": True,
            "authorization_claimed": False,
            "gate_claimed": False,
            "production_entry": "same_process_runner_generate_only",
            "external_production_package_permitted": False,
            "formal_result": False,
            "stage_progression_allowed": False,
        },
        "DIAGNOSTIC_EVIDENCE_POLICY_MISMATCH",
    )
    command = _exact_keys(top["command_schema"], ("runner", "required_arguments", "forbidden_arguments"), "COMMAND_SCHEMA_MISMATCH")
    _require_exact_json(
        command,
        {
            "runner": RUNNER_PATH,
            "required_arguments": ["--generate", "--diagnostic-bootstrap", "--output", "--source-commit"],
            "forbidden_arguments": ["--execution-package", "--synthetic-fixture", "--backend", "--device", "--receipt"],
        },
        "COMMAND_SCHEMA_MISMATCH",
    )
    projected = {
        "schema_version": top["schema_version"],
        "manifest_schema": MANIFEST_SCHEMA,
        "implementation_schema": top["implementation_schema"],
        "protocol_id": top["protocol_id"],
        "expected_source": top["expected_source"],
        "source_files": top["source_files"],
        "frozen_identities": top["frozen_identities"],
        "experiment": top["experiment"],
        "evidence_policy": {
            "production_entry": "same_process_runner_generate_only",
            "external_production_package_permitted": False,
            "formal_result": False,
            "stage_progression_allowed": False,
        },
        "command_schema": {
            "runner": RUNNER_PATH,
            "required_arguments": ["--generate", "--manifest", "--output", "--source-commit"],
            "forbidden_arguments": ["--execution-package", "--synthetic-fixture", "--backend", "--device", "--receipt"],
        },
        "output_schema": top["output_schema"],
    }
    validate_manifest_schema(projected)

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
    status = tuple(line for line in _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all").splitlines() if line)
    if not _is_git_object(head) or not _is_git_object(tree):
        raise InvalidExperiment("GIT_STATE_UNREADABLE", "Git returned malformed source identity")
    return SourceState(head=head, tree=tree, dirty=bool(status), status_porcelain=status)


def preflight(
    repo_root: Path,
    manifest_path: Path,
    *,
    declared_source_commit: str | None,
    diagnostic_only: bool = False,
) -> Preflight:
    """Validate an identity object then exact clean source before generation."""

    if _forbidden_text(os.fspath(manifest_path)):
        raise InvalidExperiment("FORBIDDEN_FORMAL_INPUT_REFERENCE", "manifest path contains a forbidden formal input identifier")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise InvalidExperiment("MANIFEST_UNREADABLE", "authorization manifest could not be read") from exc
    manifest = _json_object(manifest_bytes, "MANIFEST_INVALID_JSON")
    if diagnostic_only:
        validate_diagnostic_bootstrap_schema(manifest)
    else:
        validate_manifest_schema(manifest)

    source_state = read_source_state(repo_root)
    if source_state.dirty:
        raise InvalidExperiment("DIRTY_WORKTREE", "RC0 generation requires an exact clean checkout")
    if source_state.head != manifest["expected_source"]["head"]:
        raise InvalidExperiment("HEAD_MISMATCH", "actual HEAD differs from authorization")
    if source_state.tree != manifest["expected_source"]["tree"]:
        raise InvalidExperiment("TREE_MISMATCH", "actual tree differs from authorization")
    if declared_source_commit is None or declared_source_commit != source_state.head:
        raise InvalidExperiment("SOURCE_COMMIT_DECLARATION_MISMATCH", "explicit source ref is absent or differs from actual HEAD")

    source_hashes: dict[str, str] = {}
    for relative_path in REQUIRED_SOURCE_PATHS:
        path = repo_root / relative_path
        try:
            entry = path.lstat()
        except OSError as exc:
            raise InvalidExperiment("SOURCE_FILE_UNREADABLE", f"required source unavailable: {relative_path}") from exc
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
            raise InvalidExperiment("SOURCE_FILE_UNREADABLE", f"required source is not a regular file: {relative_path}")
        digest = sha256_file(path)
        source_hashes[relative_path] = digest
        if digest != manifest["source_files"][relative_path]:
            raise InvalidExperiment("SOURCE_FILE_HASH_MISMATCH", f"required source digest changed: {relative_path}")

    config_bytes = (repo_root / CONFIG_PATH).read_bytes()
    plan_bytes = (repo_root / PLAN_PATH).read_bytes()
    protocol_bytes = (repo_root / PROTOCOL_PATH).read_bytes()
    if sha256_bytes(config_bytes) != CONFIG_RAW_SHA256 or sha256_bytes(plan_bytes) != PLAN_RAW_SHA256 or sha256_bytes(protocol_bytes) != PROTOCOL_RAW_SHA256:
        raise InvalidExperiment("FROZEN_SCIENCE_IDENTITY_MISMATCH", "config, plan, or protocol raw identity changed")
    config = _json_object(config_bytes, "CONFIG_INVALID_JSON")
    plan = _json_object(plan_bytes, "PLAN_INVALID_JSON")
    try:
        validate_config(config)
        validate_plan(plan)
    except Exception as exc:
        raise InvalidExperiment("FROZEN_SCIENCE_SCHEMA_MISMATCH", "config or plan failed its sealed validator") from exc
    if source_hashes[EXTRACTOR_PATH] != EXTRACTOR_IDENTITY["raw_sha256"]:
        raise InvalidExperiment("EXTRACTOR_IDENTITY_MISMATCH", "extractor raw identity changed")
    if _git(repo_root, "hash-object", EXTRACTOR_PATH) != EXTRACTOR_IDENTITY["git_blob"]:
        raise InvalidExperiment("EXTRACTOR_IDENTITY_MISMATCH", "extractor Git blob identity changed")

    return Preflight(
        repo_root=repo_root,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        config=config,
        config_bytes=config_bytes,
        plan=plan,
        plan_bytes=plan_bytes,
        protocol_bytes=protocol_bytes,
        source_state=source_state,
        source_hashes=source_hashes,
    )


def expected_matched_parameters(preflight_result: Preflight, group: Mapping[str, Any]) -> dict[str, Any]:
    generation = preflight_result.config["generation"]
    encoding = preflight_result.config["encoding"]
    return {
        "prompt": group["prompt"],
        "prompt_sha256": group["prompt_sha256"],
        "seed": group["seed"],
        "model_id": preflight_result.config["model"]["id"],
        "model_revision": preflight_result.config["model"]["revision"],
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
        "execution_path": "same_process_runner_generate_only",
    }


def command_artifact(preflight_result: Preflight, evidence_mode: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runner": RUNNER_PATH,
        "mode": "--generate",
        "evidence_mode": evidence_mode,
        "source_commit": preflight_result.source_state.head,
        "manifest_sha256": sha256_bytes(preflight_result.manifest_bytes),
        "argument_names": ["--generate", "--manifest", "--output", "--source-commit"],
    }


def condition_config(
    preflight_result: Preflight,
    group: Mapping[str, Any],
    condition: str,
    latent_identity: Mapping[str, Any],
    video_path: str,
    evidence_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_schema": "sc_sstw_rc0_condition_config_v1",
        "evidence_mode": evidence_mode,
        "protocol_id": PROTOCOL_ID,
        "group_id": group["group_id"],
        "condition": condition,
        "schedule_id": "NONE" if condition.startswith("OFF_") else condition,
        "matched_parameters": expected_matched_parameters(preflight_result, group),
        "actual_initial_latent_identity": dict(latent_identity),
        "saved_video_path": video_path,
        "formal_result": False,
        "stage_progression_allowed": False,
    }


def _artifact_path(record_path: Path, identity: Mapping[str, Any]) -> Path:
    if type(identity) is not dict or tuple(identity) != ("path", "sha256") or not _is_sha256(identity.get("sha256")):
        raise InvalidExperiment("ARTIFACT_IDENTITY_INVALID", "artifact identity is malformed")
    declared = identity["path"]
    if type(declared) is not str or not declared or Path(declared).is_absolute() or ".." in Path(declared).parts or _forbidden_text(declared):
        raise InvalidExperiment("ARTIFACT_PATH_INVALID", "artifact path is absolute, escaping, or forbidden")
    base = record_path.parent.resolve()
    current = base
    parts = Path(declared).parts
    for part in parts[:-1]:
        current /= part
        try:
            entry = current.lstat()
        except OSError as exc:
            raise InvalidExperiment("ARTIFACT_PATH_INVALID", "artifact parent is unavailable") from exc
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
            raise InvalidExperiment("ARTIFACT_PATH_INVALID", "artifact parent is not a real directory")
    candidate = current / parts[-1]
    if candidate.parent != base and base not in candidate.parents:
        raise InvalidExperiment("ARTIFACT_PATH_INVALID", "artifact path escapes the execution package")
    return candidate


def _validate_carrier_events(events: Any, condition: str, config: Mapping[str, Any], evidence_mode: str) -> None:
    if type(events) is not list:
        raise InvalidExperiment("CARRIER_RECORD_INVALID", "carrier events must be an array")
    if condition.startswith("OFF_"):
        if events:
            raise InvalidExperiment("OFF_CARRIER_EFFECT_PRESENT", "OFF attempt recorded a carrier effect")
        return
    if len(events) != 16:
        raise InvalidExperiment("CARRIER_CALL_BUDGET_MISMATCH", "A/B carrier call budget is not exactly 16")
    from .rc0_causal_localization_v2 import schedule_a, schedule_b

    schedule = schedule_a() if condition == "A" else schedule_b()
    schedule_sha = sha256_bytes(canonical_json_bytes(schedule))
    expected_keys = (
        "condition",
        "call_index",
        "schedule_step",
        "schedule_sha256",
        "output_shape",
        "output_dtype",
        "input_tensor_sha256",
        "modified_tensor_sha256",
        "distinct_storage",
        "effective_relative_rms",
        "evidence_mode",
    )
    for index, event in enumerate(events):
        item = _exact_keys(event, expected_keys, "CARRIER_RECORD_INVALID")
        if (
            item["condition"] != condition
            or type(item["call_index"]) is not int
            or item["call_index"] != index
            or type(item["schedule_step"]) is not int
            or item["schedule_step"] != index // 2
            or item["schedule_sha256"] != schedule_sha
            or type(item["output_shape"]) is not list
            or not item["output_shape"]
            or any(type(value) is not int or value <= 0 for value in item["output_shape"])
            or type(item["output_dtype"]) is not str
            or not item["output_dtype"]
            or not _is_sha256(item["input_tensor_sha256"])
            or not _is_sha256(item["modified_tensor_sha256"])
            or item["input_tensor_sha256"] == item["modified_tensor_sha256"]
            or item["distinct_storage"] is not True
            or type(item["effective_relative_rms"]) is not float
            or abs(item["effective_relative_rms"] - config["carrier"]["target_relative_rms"]) > config["carrier"]["target_relative_rms_absolute_tolerance"]
            or item["evidence_mode"] != evidence_mode
        ):
            raise InvalidExperiment("CARRIER_RECORD_INVALID", "carrier event differs from the frozen schedule, budget, or tensor contract")


def validate_execution_record(
    record_path: Path,
    preflight_result: Preflight,
    generation_receipt: object,
    *,
    allow_cpu_test_harness: bool,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Path]], str]:
    """Consume receipt first, then validate disk evidence before evaluation."""

    try:
        from .rc0_generation import RC0GenerationError, consume_generation_receipt

        receipt = consume_generation_receipt(generation_receipt, record_path)
    except Exception as exc:
        reason = "GENERATION_RECEIPT_INVALID"
        if isinstance(exc, InvalidExperiment):
            raise
        raise InvalidExperiment(reason, "same-process generation receipt is absent, invalid, or replayed") from exc

    provenance = receipt.get("authority_provenance_class")
    expected_provenance = PROVENANCE_CPU_HARNESS if allow_cpu_test_harness else PROVENANCE_PRODUCTION
    expected_mode = EVIDENCE_CPU_HARNESS if allow_cpu_test_harness else EVIDENCE_PRODUCTION
    if provenance != expected_provenance or receipt.get("provenance_class") != expected_provenance:
        raise InvalidExperiment("GENERATION_PROVENANCE_MISMATCH", "receipt provenance does not match the controlled runner branch")
    if allow_cpu_test_harness:
        if receipt.get("cpu_only_test_harness") is not True or receipt.get("loaded_identity", {}).get("device") != "cpu":
            raise InvalidExperiment("CPU_HARNESS_IDENTITY_MISMATCH", "CPU test harness identity is contradictory")
    elif receipt.get("cpu_only_test_harness") is not False:
        raise InvalidExperiment("GENERATION_PROVENANCE_MISMATCH", "production receipt is marked as a CPU test harness")
    loaded_identity = receipt.get("loaded_identity")
    if type(loaded_identity) is not dict:
        raise InvalidExperiment("LOADED_RUNTIME_IDENTITY_MISMATCH", "loaded model/runtime identity is missing")
    if allow_cpu_test_harness:
        if loaded_identity != {
            "model_id": preflight_result.config["model"]["id"],
            "loaded_revision": preflight_result.config["model"]["revision"],
            "scheduler": receipt.get("environment", {}).get("scheduler"),
            "dtype": "cpu_test_harness_numpy_float32",
            "device": "cpu",
            "cpu_only_test_harness": True,
        }:
            raise InvalidExperiment("LOADED_RUNTIME_IDENTITY_MISMATCH", "CPU harness loaded identity changed")
    elif (
        loaded_identity.get("model_id") != preflight_result.config["model"]["id"]
        or loaded_identity.get("loaded_revision") != preflight_result.config["model"]["revision"]
        or loaded_identity.get("scheduler") != receipt.get("environment", {}).get("scheduler")
        or type(loaded_identity.get("dtype")) is not str
        or not loaded_identity["dtype"]
        or type(loaded_identity.get("device")) is not str
        or "cuda" not in loaded_identity["device"].lower()
        or loaded_identity.get("cpu_only_test_harness") is not False
    ):
        raise InvalidExperiment("LOADED_RUNTIME_IDENTITY_MISMATCH", "production loaded model, scheduler, dtype, or CUDA identity changed")

    expected_preflight = {
        "manifest_sha256": sha256_bytes(preflight_result.manifest_bytes),
        "config_sha256": sha256_bytes(preflight_result.config_bytes),
        "plan_sha256": sha256_bytes(preflight_result.plan_bytes),
        "protocol_sha256": sha256_bytes(preflight_result.protocol_bytes),
        "source_head": preflight_result.source_state.head,
        "source_tree": preflight_result.source_state.tree,
        "source_hashes": preflight_result.source_hashes,
        "command_schema": preflight_result.manifest["command_schema"],
        "output_schema": preflight_result.manifest["output_schema"],
    }
    expected_implementation = {
        "generation_module_path": GENERATION_PATH,
        "generation_module_sha256": preflight_result.source_hashes[GENERATION_PATH],
        "implementation_module_path": IMPLEMENTATION_PATH,
        "implementation_module_sha256": preflight_result.source_hashes[IMPLEMENTATION_PATH],
        "extractor": EXTRACTOR_IDENTITY,
    }
    if (
        receipt.get("schema_version") != 1
        or receipt.get("trust_boundary") != "same_clean_source_runner_process_observed_generation_call"
        or receipt.get("completed") is not True
        or receipt.get("record_path") != str(record_path.resolve())
        or receipt.get("preflight") != expected_preflight
        or receipt.get("implementation") != expected_implementation
        or receipt.get("command_artifact") != command_artifact(preflight_result, expected_mode)
    ):
        raise InvalidExperiment("GENERATION_RECEIPT_IDENTITY_MISMATCH", "receipt differs from preflight, source, command, or implementation identity")

    try:
        entry = record_path.lstat()
    except OSError as exc:
        raise InvalidExperiment("EXECUTION_RECORD_MISSING", "execution record is unavailable") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise InvalidExperiment("EXECUTION_RECORD_INVALID", "execution record must be a regular non-symlink file")
    if receipt.get("execution_record_sha256") != sha256_file(record_path):
        raise InvalidExperiment("GENERATION_RECEIPT_DISK_MISMATCH", "execution record digest differs from the in-process receipt")
    record = _json_object(record_path.read_bytes(), "EXECUTION_RECORD_INVALID")
    _exact_keys(record, ("schema_version", "execution_schema", "protocol_id", "plan_sha256", "evidence_mode", "evidence_schema", "groups"), "EXECUTION_RECORD_INVALID")
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["execution_schema"] != EXECUTION_SCHEMA
        or record["protocol_id"] != PROTOCOL_ID
        or record["plan_sha256"] != PLAN_RAW_SHA256
        or record["evidence_mode"] != expected_mode
        or record["evidence_schema"] != f"sc_sstw_rc0_{expected_mode}_v1"
    ):
        raise InvalidExperiment("EXECUTION_RECORD_INVALID", "execution record identity changed")

    groups = record["groups"]
    if type(groups) is not list or len(groups) != 2 or type(receipt.get("groups")) is not list or len(receipt["groups"]) != 2:
        raise InvalidExperiment("QUARTET_STRUCTURE_INVALID", "exactly two receipt-bound groups are required")
    artifact_paths: dict[tuple[str, str], dict[str, Path]] = {}
    pending_hashes: list[tuple[str, Path, str]] = []
    expected_global_attempt = 1
    for group_index, (group_record, plan_group, receipt_group) in enumerate(zip(groups, preflight_result.plan["groups"], receipt["groups"], strict=True)):
        _exact_keys(group_record, ("group_id", "content_grammar", "prompt", "prompt_sha256", "seed", "matched_parameters", "initial_latent_identity", "clone_identities", "conditions", "attempts"), "QUARTET_STRUCTURE_INVALID")
        if (
            group_record["group_id"] != plan_group["group_id"]
            or group_record["content_grammar"] != plan_group["content_grammar"]
            or group_record["prompt"] != plan_group["prompt"]
            or group_record["prompt_sha256"] != plan_group["prompt_sha256"]
            or group_record["seed"] != plan_group["seed"]
            or group_record["matched_parameters"] != expected_matched_parameters(preflight_result, plan_group)
            or receipt_group.get("group_id") != plan_group["group_id"]
        ):
            raise InvalidExperiment("QUARTET_IDENTITY_MISMATCH", "group identity or matched parameters changed")
        latent = group_record["initial_latent_identity"]
        if type(latent) is not dict or tuple(latent) != ("sha256", "shape", "dtype") or not _is_sha256(latent.get("sha256")) or type(latent.get("shape")) is not list or type(latent.get("dtype")) is not str:
            raise InvalidExperiment("LATENT_IDENTITY_INVALID", "source latent identity is malformed")
        clones = group_record["clone_identities"]
        if type(clones) is not list or len(clones) != 4:
            raise InvalidExperiment("LATENT_CLONE_MISMATCH", "four pre-created latent clone identities are required")
        storage_tokens: set[str] = set()
        for condition, clone in zip(CONDITION_ORDER, clones, strict=True):
            _exact_keys(clone, ("condition", "sha256", "shape", "dtype", "distinct_storage", "storage_token"), "LATENT_CLONE_MISMATCH")
            if (
                clone["condition"] != condition
                or clone["sha256"] != latent["sha256"]
                or clone["shape"] != latent["shape"]
                or clone["dtype"] != latent["dtype"]
                or clone["distinct_storage"] is not True
                or type(clone["storage_token"]) is not str
                or not clone["storage_token"]
                or clone["storage_token"] in storage_tokens
            ):
                raise InvalidExperiment("LATENT_CLONE_MISMATCH", "clone bytes, shape, dtype, condition, or storage identity changed")
            storage_tokens.add(clone["storage_token"])
        if receipt_group.get("initial_latent_identity") != latent or receipt_group.get("clone_identities") != clones:
            raise InvalidExperiment("GENERATION_RECEIPT_LATENT_MISMATCH", "disk latent identities differ from the generation receipt")

        conditions = group_record["conditions"]
        attempts = group_record["attempts"]
        receipt_conditions = receipt_group.get("conditions")
        if type(conditions) is not list or type(attempts) is not list or type(receipt_conditions) is not list or len(conditions) != 4 or len(attempts) != 4 or len(receipt_conditions) != 4:
            raise InvalidExperiment("QUARTET_STRUCTURE_INVALID", "each group requires four exact conditions and attempts")
        parameter_sha = sha256_bytes(canonical_json_bytes(group_record["matched_parameters"]))
        for condition_index, (condition_record, attempt, receipt_condition) in enumerate(zip(conditions, attempts, receipt_conditions, strict=True)):
            condition = CONDITION_ORDER[condition_index]
            _exact_keys(condition_record, ("condition", "schedule_id", "carrier_enabled", "clone_identity", "matched_parameters_sha256", "artifacts"), "QUARTET_STRUCTURE_INVALID")
            _exact_keys(attempt, ("attempt_index", "group_id", "condition", "started", "completed", "outcome", "retry_index", "matched_parameters_sha256"), "ATTEMPT_LOG_INVALID")
            if (
                condition_record["condition"] != condition
                or condition_record["schedule_id"] != ("NONE" if condition.startswith("OFF_") else condition)
                or condition_record["carrier_enabled"] is not (not condition.startswith("OFF_"))
                or condition_record["clone_identity"] != clones[condition_index]
                or condition_record["matched_parameters_sha256"] != parameter_sha
                or attempt != {
                    "attempt_index": expected_global_attempt,
                    "group_id": plan_group["group_id"],
                    "condition": condition,
                    "started": True,
                    "completed": True,
                    "outcome": "success",
                    "retry_index": 0,
                    "matched_parameters_sha256": parameter_sha,
                }
            ):
                raise InvalidExperiment("ATTEMPT_LOG_INVALID", "attempt order, completion, retry, or matched identity changed")
            expected_global_attempt += 1
            if receipt_condition.get("condition") != condition or receipt_condition.get("clone_identity") != clones[condition_index]:
                raise InvalidExperiment("GENERATION_RECEIPT_DISK_MISMATCH", "receipt condition or clone identity differs from disk")
            artifacts = _exact_keys(condition_record["artifacts"], ARTIFACT_NAMES, "ARTIFACT_IDENTITY_INVALID")
            receipt_artifacts = receipt_condition.get("artifacts")
            if artifacts != receipt_artifacts:
                raise InvalidExperiment("GENERATION_RECEIPT_DISK_MISMATCH", "receipt and disk artifact identities differ")
            condition_paths: dict[str, Path] = {}
            for artifact_name in ARTIFACT_NAMES:
                identity = artifacts[artifact_name]
                path = _artifact_path(record_path, identity)
                pending_hashes.append((f"{plan_group['group_id']}:{condition}:{artifact_name}", path, identity["sha256"]))
                condition_paths[artifact_name] = path
            artifact_paths[(plan_group["group_id"], condition)] = condition_paths
            _validate_carrier_events(receipt_condition.get("carrier_events"), condition, preflight_result.config, expected_mode)
            if receipt_condition.get("video_saved_complete") is not True or receipt_condition.get("video_sha256") != artifacts["video"]["sha256"]:
                raise InvalidExperiment("GENERATION_RECEIPT_VIDEO_MISMATCH", "receipt lacks a completed saved-video binding")

    # No artifact path is opened before all receipt/record structure and path declarations pass.
    for label, path, expected_sha in pending_hashes:
        try:
            entry = path.lstat()
        except OSError as exc:
            raise InvalidExperiment("ARTIFACT_MISSING", f"execution artifact missing: {label}") from exc
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
            raise InvalidExperiment("ARTIFACT_PATH_INVALID", f"artifact is not a regular non-symlink file: {label}")
        if sha256_file(path) != expected_sha:
            raise InvalidExperiment("ARTIFACT_HASH_MISMATCH", f"execution artifact digest changed: {label}")

    for group_index, (group_record, plan_group, receipt_group) in enumerate(zip(groups, preflight_result.plan["groups"], receipt["groups"], strict=True)):
        common_environment = None
        common_command = None
        for condition_index, condition_record in enumerate(group_record["conditions"]):
            condition = condition_record["condition"]
            receipt_condition = receipt_group["conditions"][condition_index]
            paths = artifact_paths[(plan_group["group_id"], condition)]
            config_payload = _json_object(paths["config"].read_bytes(), "CONDITION_CONFIG_INVALID")
            expected_config = condition_config(
                preflight_result,
                plan_group,
                condition,
                group_record["initial_latent_identity"],
                condition_record["artifacts"]["video"]["path"],
                expected_mode,
            )
            if config_payload != expected_config:
                raise InvalidExperiment("CONDITION_CONFIG_INVALID", "condition config differs from frozen identity")
            environment = _json_object(paths["environment"].read_bytes(), "ENVIRONMENT_INVALID")
            if environment != receipt.get("environment"):
                raise InvalidExperiment("GENERATION_RECEIPT_RUNTIME_MISMATCH", "disk environment differs from in-process generation observation")
            if common_environment is None:
                common_environment = environment
            elif environment != common_environment:
                raise InvalidExperiment("QUARTET_PARAMETER_MISMATCH", "quartet runtime environment differs")
            command = _json_object(paths["command"].read_bytes(), "COMMAND_ARTIFACT_INVALID")
            if command != command_artifact(preflight_result, expected_mode) or command != receipt.get("command_artifact"):
                raise InvalidExperiment("COMMAND_ARTIFACT_INVALID", "disk command differs from authorization or receipt")
            if common_command is None:
                common_command = command
            elif command != common_command:
                raise InvalidExperiment("QUARTET_PARAMETER_MISMATCH", "quartet command identity differs")
            integrity = _json_object(paths["integrity"].read_bytes(), "INTEGRITY_ARTIFACT_INVALID")
            expected_integrity = {
                "schema_version": 1,
                "artifact_schema": "sc_sstw_rc0_condition_integrity_v1",
                "evidence_mode": expected_mode,
                "group_id": plan_group["group_id"],
                "condition": condition,
                "initial_latent_identity": group_record["initial_latent_identity"],
                "clone_identity": group_record["clone_identities"][condition_index],
                "carrier_events": receipt_condition["carrier_events"],
                "codec_identity": receipt_condition["codec_identity"],
                "video_saved_complete": True,
                "video_sha256": receipt_condition["video_sha256"],
            }
            if integrity != expected_integrity:
                raise InvalidExperiment("INTEGRITY_ARTIFACT_INVALID", "disk integrity record differs from receipt")
            _validate_carrier_events(integrity["carrier_events"], condition, preflight_result.config, expected_mode)
    return record, artifact_paths, expected_mode


def _inspect_and_decode_saved_mp4(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    if path.suffix.lower() != ".mp4":
        raise InvalidExperiment("SAVED_MP4_INVALID", "video artifact is not an MP4 path")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise InvalidExperiment("SAVED_MP4_DECODER_UNAVAILABLE", "ffprobe is unavailable")
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames:format=format_name",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        probe = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise InvalidExperiment("SAVED_MP4_INVALID", "saved MP4 could not be independently probed") from exc
    streams = probe.get("streams")
    format_data = probe.get("format")
    format_name = format_data.get("format_name") if type(format_data) is dict else None
    if type(streams) is not list or len(streams) != 1 or type(format_name) is not str or "mp4" not in format_name.split(","):
        raise InvalidExperiment("SAVED_MP4_INVALID", "container or stream count changed")
    expected_stream = {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 512,
        "height": 320,
        "avg_frame_rate": "8/1",
        "nb_frames": "49",
    }
    if {key: streams[0].get(key) for key in expected_stream} != expected_stream:
        raise InvalidExperiment("SAVED_MP4_CODEC_OR_GEOMETRY_MISMATCH", "codec, pixel format, geometry, FPS, or frame count changed")
    try:
        from .learned_observation import decode_saved_mp4

        frames = np.asarray(decode_saved_mp4(path))
    except Exception as exc:
        raise InvalidExperiment("SAVED_MP4_DECODE_FAILURE", "saved MP4 decode failed") from exc
    if frames.shape != (49, 320, 512, 3) or frames.dtype != np.uint8:
        raise InvalidExperiment("SAVED_MP4_DECODE_SHAPE_MISMATCH", "decoded RGB24 tensor changed")
    return {
        "container": "mp4",
        "codec_name": "h264",
        "pixel_format": "yuv420p",
        "width": 512,
        "height": 320,
        "fps": "8/1",
        "frame_count": 49,
        "decodable": True,
    }, frames


def _extract_and_verify_cache(paths: Mapping[str, Path], expected_video_sha: str, preflight_result: Preflight, evidence_mode: str, frames: np.ndarray) -> np.ndarray:
    try:
        from .learned_observation import extract_feature_matrix

        recomputed = np.asarray(extract_feature_matrix(frames), dtype=np.float64)
    except Exception as exc:
        raise InvalidExperiment("FEATURE_RECOMPUTE_FAILURE", "frozen 30D extractor failed") from exc
    if recomputed.shape != (13, 30) or not np.isfinite(recomputed).all():
        raise InvalidExperiment("FEATURE_RECOMPUTE_INVALID", "frozen extractor did not return finite 13x30 features")
    payload = _json_object(paths["features"].read_bytes(), "FEATURE_CACHE_INVALID")
    _exact_keys(payload, ("schema_version", "feature_cache_schema", "evidence_mode", "source", "video_sha256", "extractor_identity", "comparison", "features"), "FEATURE_CACHE_INVALID")
    expected_extractor = {"id": "sc_sstw_frozen_public_30d_v1", **EXTRACTOR_IDENTITY}
    if (
        payload["schema_version"] != 1
        or payload["feature_cache_schema"] != FEATURE_CACHE_SCHEMA
        or payload["evidence_mode"] != evidence_mode
        or payload["source"] != "recomputed_from_single_saved_mp4"
        or payload["video_sha256"] != expected_video_sha
        or payload["extractor_identity"] != expected_extractor
        or payload["comparison"] != {"atol": FEATURE_CACHE_ATOL, "rtol": 0.0}
    ):
        raise InvalidExperiment("FEATURE_CACHE_IDENTITY_MISMATCH", "feature cache identity or comparison policy changed")
    try:
        stored = np.asarray(payload["features"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidExperiment("FEATURE_CACHE_INVALID", "feature cache cannot be decoded") from exc
    if stored.shape != (13, 30) or not np.isfinite(stored).all() or not np.allclose(stored, recomputed, rtol=0.0, atol=FEATURE_CACHE_ATOL):
        raise InvalidExperiment("FEATURE_CACHE_RECOMPUTE_MISMATCH", "stored features differ from saved-MP4 recomputation")
    return recomputed


def evaluate_execution(
    record: Mapping[str, Any],
    artifact_paths: Mapping[tuple[str, str], Mapping[str, Path]],
    preflight_result: Preflight,
    evidence_mode: str,
) -> Evaluation:
    """Decode all MP4s, run all Level P cells, then conditionally run Level R."""

    p_results: list[dict[str, Any]] = []
    p_cells: dict[str, bool] = {}
    off_controls_valid = True
    for group in record["groups"]:
        group_id = group["group_id"]
        group_frames: dict[str, np.ndarray] = {}
        for condition_record in group["conditions"]:
            condition = condition_record["condition"]
            paths = artifact_paths[(group_id, condition)]
            codec, frames = _inspect_and_decode_saved_mp4(paths["video"])
            integrity = _json_object(paths["integrity"].read_bytes(), "INTEGRITY_ARTIFACT_INVALID")
            if codec != integrity["codec_identity"]:
                raise InvalidExperiment("SAVED_MP4_IDENTITY_MISMATCH", "independent probe differs from the receipt-bound codec identity")
            group_frames[condition] = frames
        normalized_off1 = group_frames["OFF_R1"].astype(np.float64) / 255.0
        normalized_off2 = group_frames["OFF_R2"].astype(np.float64) / 255.0
        for condition in TEMPLATES:
            metrics = level_p_metrics(
                normalized_off1,
                normalized_off2,
                group_frames[condition].astype(np.float64) / 255.0,
            )
            p_results.append({"group_id": group_id, "condition": condition, "metrics": metrics, "cell_pass": metrics["cell_pass"]})
            p_cells[f"{group_id}:{condition}"] = metrics["cell_pass"]
            off_controls_valid = off_controls_valid and metrics["off_repeat_control_valid"]
        del normalized_off1, normalized_off2, group_frames

    if not off_controls_valid:
        return Evaluation(
            package_status=PACKAGE_STATUS_INSUFFICIENT,
            protocol_outcome=STATUS_INSUFFICIENT,
            reason_code="OFF_REPEAT_FLOOR_EXCEEDED",
            level_p=tuple(p_results),
            level_r=(),
            science_metrics_present=True,
        )
    if not all(p_cells.values()):
        return Evaluation(
            package_status=PACKAGE_STATUS_P_FAIL,
            protocol_outcome=STATUS_P_FAIL,
            reason_code="AT_LEAST_ONE_LEVEL_P_CELL_FAILED",
            level_p=tuple(p_results),
            level_r=(),
            science_metrics_present=True,
        )

    features: dict[tuple[str, str], np.ndarray] = {}
    for group in record["groups"]:
        group_id = group["group_id"]
        for condition_record in group["conditions"]:
            condition = condition_record["condition"]
            paths = artifact_paths[(group_id, condition)]
            _codec, frames = _inspect_and_decode_saved_mp4(paths["video"])
            features[(group_id, condition)] = _extract_and_verify_cache(
                paths,
                condition_record["artifacts"]["video"]["sha256"],
                preflight_result,
                evidence_mode,
                frames,
            )

    r_results: list[dict[str, Any]] = []
    r_cells: dict[str, bool] = {}
    for group in record["groups"]:
        group_id = group["group_id"]
        for condition in TEMPLATES:
            metrics = level_r_metrics(
                features[(group_id, "OFF_R1")],
                features[(group_id, "OFF_R2")],
                features[(group_id, condition)],
                condition,
            )
            r_results.append({"group_id": group_id, "condition": condition, "metrics": metrics, "cell_pass": metrics["cell_pass"]})
            r_cells[f"{group_id}:{condition}"] = metrics["cell_pass"]
    if not all(r_cells.values()):
        return Evaluation(
            package_status=PACKAGE_STATUS_P_PASS_R_FAIL,
            protocol_outcome=STATUS_P_PASS_R_FAIL,
            reason_code="AT_LEAST_ONE_LEVEL_R_CELL_FAILED",
            level_p=tuple(p_results),
            level_r=tuple(r_results),
            science_metrics_present=True,
        )
    return Evaluation(
        package_status=PACKAGE_STATUS_P_PASS_R_PASS,
        protocol_outcome=STATUS_P_PASS_R_PASS,
        reason_code="ALL_LEVEL_P_AND_R_CELLS_PASSED",
        level_p=tuple(p_results),
        level_r=tuple(r_results),
        science_metrics_present=True,
    )


def base_audit() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "output_schema": OUTPUT_SCHEMA,
        "implementation_schema": IMPLEMENTATION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "attempt_budget": 8,
        "formal_result": False,
        "stage_progression_allowed": False,
    }


def is_diagnostic_preflight(preflight_result: Preflight) -> bool:
    return preflight_result.manifest.get("diagnostic_class") == DIAGNOSTIC_CLASS


def _apply_diagnostic_boundary(
    audit: dict[str, Any],
    decision: str,
    *,
    carrier_survival: str,
    relation_readout: str,
) -> None:
    if decision not in DIAGNOSTIC_DECISIONS:
        raise InvalidExperiment("DIAGNOSTIC_DECISION_INVALID", "diagnostic decision is outside the frozen set")
    audit.update(
        {
            "diagnostic_class": DIAGNOSTIC_CLASS,
            "diagnostic_only": True,
            "authorization_claimed": False,
            "gate_claimed": False,
            "step1_carrier_survival": carrier_survival,
            "step2_relation_readout": relation_readout,
            "route_decision": decision,
            "formal_result": False,
            "stage_progression_allowed": False,
        }
    )


def invalid_audit(reason_code: str, failure_phase: str, exception_type: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    audit = base_audit()
    audit.update(
        {
            "status": PACKAGE_STATUS_INVALID,
            "protocol_outcome": STATUS_INVALID,
            "valid_experiment": False,
            "reason_code": reason_code,
            "failure_phase": failure_phase,
            "exception_type": exception_type,
            "science_metrics_present": False,
            "attempts_started": None,
            "partial_attempt_log": [],
            "cpu_only_test_harness": bool(context and context.get("cpu_only_test_harness") is True),
            "test_only_non_evidence": bool(context and context.get("cpu_only_test_harness") is True),
        }
    )
    if context:
        audit["integrity_context"] = dict(context)
        if context.get("diagnostic_only") is True:
            _apply_diagnostic_boundary(
                audit,
                DIAGNOSTIC_DECISION_INSUFFICIENT,
                carrier_survival=DIAGNOSTIC_INSUFFICIENT_TO_DECIDE,
                relation_readout=DIAGNOSTIC_INSUFFICIENT_TO_DECIDE,
            )
    validate_status_mapping(audit["status"], audit["protocol_outcome"])
    return audit


def insufficient_audit(reason_code: str, failure_phase: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    audit = base_audit()
    audit.update(
        {
            "status": PACKAGE_STATUS_INSUFFICIENT,
            "protocol_outcome": STATUS_INSUFFICIENT,
            "valid_experiment": True,
            "reason_code": reason_code,
            "failure_phase": failure_phase,
            "science_metrics_present": False,
            "attempts_started": None,
            "partial_attempt_log": [],
            "cpu_only_test_harness": False,
            "test_only_non_evidence": False,
        }
    )
    if context:
        audit["integrity_context"] = dict(context)
        if context.get("diagnostic_only") is True:
            _apply_diagnostic_boundary(
                audit,
                DIAGNOSTIC_DECISION_INSUFFICIENT,
                carrier_survival=DIAGNOSTIC_INSUFFICIENT_TO_DECIDE,
                relation_readout=DIAGNOSTIC_INSUFFICIENT_TO_DECIDE,
            )
    validate_status_mapping(audit["status"], audit["protocol_outcome"])
    return audit


def evaluation_audit(
    preflight_result: Preflight,
    record: Mapping[str, Any],
    record_sha256: str,
    evaluation: Evaluation,
    *,
    cpu_only_test_harness: bool,
) -> dict[str, Any]:
    validate_status_mapping(evaluation.package_status, evaluation.protocol_outcome)
    if cpu_only_test_harness:
        conclusion = "CPU-only test harness traversed the frozen RC0 control flow; this is test-only non-evidence"
    elif evaluation.protocol_outcome == STATUS_P_FAIL:
        conclusion = "current frozen carrier stopped by the saved-MP4 pixel bridge screen"
    elif evaluation.protocol_outcome == STATUS_P_PASS_R_FAIL:
        conclusion = "pixel bridge present under the screen; frozen public relation readout remains blocked"
    elif evaluation.protocol_outcome == STATUS_P_PASS_R_PASS:
        conclusion = "oracle-paired saved-MP4 relation bridge present under the frozen RC0 screen"
    else:
        conclusion = "frozen RC0 evidence is insufficient"
    audit = base_audit()
    audit.update(
        {
            "status": evaluation.package_status,
            "protocol_outcome": evaluation.protocol_outcome,
            "valid_experiment": True,
            "reason_code": evaluation.reason_code,
            "conclusion": conclusion,
            "science_metrics_present": evaluation.science_metrics_present,
            "cpu_only_test_harness": cpu_only_test_harness,
            "test_only_non_evidence": cpu_only_test_harness,
            "evidence_mode": record["evidence_mode"],
            "source_state": preflight_result.source_state.as_dict(),
            "source_file_sha256": preflight_result.source_hashes,
            "manifest_sha256": sha256_bytes(preflight_result.manifest_bytes),
            "config_sha256": sha256_bytes(preflight_result.config_bytes),
            "plan_sha256": sha256_bytes(preflight_result.plan_bytes),
            "protocol_sha256": sha256_bytes(preflight_result.protocol_bytes),
            "execution_record_sha256": record_sha256,
            "attempt_order": preflight_result.plan["attempts"],
            "attempt_log": [attempt for group in record["groups"] for attempt in group["attempts"]],
            "aggregation": "all_cells_no_averaging_no_majority_vote",
            "level_p": list(evaluation.level_p),
            "level_r": list(evaluation.level_r),
        }
    )
    if is_diagnostic_preflight(preflight_result):
        if evaluation.protocol_outcome == STATUS_P_FAIL:
            decision = DIAGNOSTIC_DECISION_CARRIER_NOT_FEASIBLE
            carrier_survival = DIAGNOSTIC_NOT_FEASIBLE
            relation_readout = DIAGNOSTIC_INSUFFICIENT_TO_DECIDE
        elif evaluation.protocol_outcome == STATUS_P_PASS_R_PASS:
            decision = DIAGNOSTIC_DECISION_BUILD_BLIND
            carrier_survival = DIAGNOSTIC_FEASIBLE
            relation_readout = DIAGNOSTIC_FEASIBLE
        else:
            # The frozen VAE re-encode branch is declared unavailable in this
            # minimal delivery, so an R failure cannot yet distinguish a VAE
            # readout switch from relation-carrier redesign.
            decision = DIAGNOSTIC_DECISION_INSUFFICIENT
            carrier_survival = (
                DIAGNOSTIC_INSUFFICIENT_TO_DECIDE
                if evaluation.protocol_outcome == STATUS_INSUFFICIENT
                else DIAGNOSTIC_FEASIBLE
            )
            relation_readout = DIAGNOSTIC_INSUFFICIENT_TO_DECIDE
        _apply_diagnostic_boundary(
            audit,
            decision,
            carrier_survival=carrier_survival,
            relation_readout=relation_readout,
        )
        audit.update(
            {
                "diagnostic_bootstrap_sha256": sha256_bytes(preflight_result.manifest_bytes),
                "diagnostic_observation_availability": {
                    "raw_saved_mp4_effect": True,
                    "paired_public_30d": bool(evaluation.level_r),
                    "vae_reencode_relation": False,
                },
            }
        )
        audit.pop("manifest_sha256", None)
    validate_status_mapping(audit["status"], audit["protocol_outcome"])
    return audit
