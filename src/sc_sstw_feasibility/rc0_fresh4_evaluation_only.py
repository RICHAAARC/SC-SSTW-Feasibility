"""Read-only closure over the fixed 80 Phase-C saved MP4 artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .learned_observation import decode_saved_mp4
from .rc0_fresh4_effectiveness_cpu import (
    CONDITION_ORDER,
    GROUP_ORDER,
    TRANSFORM_ORDER,
    audit_target,
    canonical_json_bytes,
    evaluate_transformed_target,
    sha256_file,
)
from .rc0_pixel_chroma_fast_cpu import probe_mp4


SCHEMA = "sc_sstw_rc0_fresh4_evaluation_only_v1"
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
RUN_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-fresh4-effectiveness-cpu-run1")
OUTPUT = RUN_ROOT / "evaluation-only-result"
SOURCE_ZIP = Path("/mnt/g/我的云端硬盘/SC-SSTW-Feasibility/rc0-fresh-exact4-base/rc0-fresh4-0471c4ae283025ad.zip")
SOURCE_ZIP_SIZE = 146503
SOURCE_ZIP_SHA256 = "6076a9645c3ef064ef387c0460305fbbc340bce78049262419124c50d6aa1878"
SOURCE_COMMIT = "add1e966e9bf8778a502e40532558185d5f64d3a"
SOURCE_TREE = "0b48705929038f9769adf3ba3b53af1b29d1fded"
FROZEN_CONFIG_SHA256 = "913b761f848940d0f17ddb71a8424e91d5dad0eb094a81eaaff163df06fb1618"
FROZEN_PROTOCOL_SHA256 = "f563f7dc83c5d3bc3391280968557ab3ee8282503fb0902d5d849af6146af34e"
ORIGINAL_AUDIT = RUN_ROOT / "audit.json"
ORIGINAL_AUDIT_SHA256 = "cc19fc0a097adfdee8d0257004789cc7aafe5eb4eb47c1f765fd94916ea46684"
ORIGINAL_CHECKSUMS = RUN_ROOT / "checksums.sha256"
ORIGINAL_CHECKSUMS_SHA256 = "5223461f7fa336e8879dc102eff61f33ed06389aad878d351ab19aa55568bef4"
INDEPENDENT = Path(str(RUN_ROOT) + ".independent.json")
INDEPENDENT_SHA256 = "f25f56f73f71c5dcb5c7f61ab1cbcdea1c9b12c7dff8be07500f4a9141943e99"
FROZEN_FORMULA = Path(str(RUN_ROOT) + ".independent_frozen_formula.json")
FROZEN_FORMULA_SHA256 = "e0b99ed46e60a9757678af43acfd01839bb9ee31ef3e89c943031d5304dd977a"


class EvaluationOnlyError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _require_bound_file(path: Path, digest: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise EvaluationOnlyError(f"bound file identity mismatch: {path.name}")


def _expected_artifact_paths() -> tuple[set[str], set[str]]:
    conditions = {f"conditions/{group}/{condition}/saved.mp4" for group in GROUP_ORDER for condition in CONDITION_ORDER}
    transforms = {f"transforms/{group}/{condition}/{transform}/saved.mp4" for group in GROUP_ORDER for condition in CONDITION_ORDER for transform in TRANSFORM_ORDER}
    return conditions, transforms


def _compare_clean_output(actual: Mapping[str, Any], original: Mapping[str, Any]) -> bool:
    common = ("observation", "presence", "downstream_executed", "calibration")
    if any(canonical_json_bytes(actual.get(key)) != canonical_json_bytes(original.get(key)) for key in common):
        return False
    actual_phase = actual.get("phase")
    original_phase = original.get("phase")
    if actual_phase is None or original_phase is None:
        return actual_phase is None and original_phase is None
    return isinstance(original_phase, dict) and canonical_json_bytes(actual_phase) == canonical_json_bytes(original_phase.get("identity"))


def _validate_all_identities(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path], dict[str, Path]]:
    """Finish every file/hash/stream check before any scientific decode."""

    if SOURCE_ZIP.is_symlink() or not SOURCE_ZIP.is_file() or SOURCE_ZIP.stat().st_size != SOURCE_ZIP_SIZE or sha256_file(SOURCE_ZIP) != SOURCE_ZIP_SHA256:
        raise EvaluationOnlyError("frozen fresh4 ZIP identity mismatch")
    config = repo_root / "configs/rc0_fresh_exact4_base_gpu.json"
    protocol = repo_root / "protocols/rc0_fresh_exact4_base_gpu.md"
    _require_bound_file(config, FROZEN_CONFIG_SHA256)
    _require_bound_file(protocol, FROZEN_PROTOCOL_SHA256)
    if _git(repo_root, "rev-parse", f"{SOURCE_COMMIT}^{{tree}}") != SOURCE_TREE:
        raise EvaluationOnlyError("fresh4 source commit/tree mismatch")
    _require_bound_file(ORIGINAL_AUDIT, ORIGINAL_AUDIT_SHA256)
    _require_bound_file(ORIGINAL_CHECKSUMS, ORIGINAL_CHECKSUMS_SHA256)
    _require_bound_file(INDEPENDENT, INDEPENDENT_SHA256)
    _require_bound_file(FROZEN_FORMULA, FROZEN_FORMULA_SHA256)
    audit = json.loads(ORIGINAL_AUDIT.read_text(encoding="utf-8"))
    independent = json.loads(INDEPENDENT.read_text(encoding="utf-8"))
    formula = json.loads(FROZEN_FORMULA.read_text(encoding="utf-8"))
    if (
        audit.get("status"), audit.get("condition_count"), audit.get("transform_cell_count"), audit.get("retry_count"),
        audit.get("input_identity", {}).get("source_commit"), audit.get("input_identity", {}).get("source_tree"),
        audit.get("config_sha256"), audit.get("protocol_sha256"),
    ) != (
        "FRESH_SMALL_SAMPLE_NOT_FEASIBLE", 16, 64, 0, SOURCE_COMMIT, SOURCE_TREE,
        FROZEN_CONFIG_SHA256, FROZEN_PROTOCOL_SHA256,
    ):
        raise EvaluationOnlyError("original runner audit identity changed")
    if not all(independent.get(key) is True for key in (
        "source_zip_match", "condition_outputs_exact_match", "transform_outputs_exact_match",
        "condition_truth_exact_match", "transform_truth_exact_match", "artifact_hashes_match",
    )):
        raise EvaluationOnlyError("first independent replay identity changed")
    if (
        formula.get("all_clean_conditions_pass"), formula.get("all_real_transform_cells_pass"),
        formula.get("runner_condition_truth_match"), formula.get("runner_transform_truth_match"),
    ) != (True, True, False, True):
        raise EvaluationOnlyError("frozen-formula crosscheck identity changed")
    expected_conditions, expected_transforms = _expected_artifact_paths()
    condition_paths: dict[str, Path] = {}
    transform_paths: dict[str, Path] = {}
    observed_conditions: set[str] = set()
    observed_transforms: set[str] = set()
    for group in GROUP_ORDER:
        group_artifacts = audit.get("artifacts", {}).get(group, {})
        for condition in CONDITION_ORDER:
            record = group_artifacts.get("conditions", {}).get(condition, {})
            relative = record.get("path")
            expected_relative = f"conditions/{group}/{condition}/saved.mp4"
            if relative != expected_relative:
                raise EvaluationOnlyError("condition artifact path binding changed")
            path = RUN_ROOT / relative
            _require_bound_file(path, record.get("sha256"))
            if probe_mp4(path) != record.get("stream"):
                raise EvaluationOnlyError("condition MP4 stream identity changed")
            condition_paths[f"{group}:{condition}"] = path
            observed_conditions.add(relative)
            for transform in TRANSFORM_ORDER:
                transformed = group_artifacts.get("transforms", {}).get(condition, {}).get(transform, {})
                transformed_relative = transformed.get("path")
                expected_transformed = f"transforms/{group}/{condition}/{transform}/saved.mp4"
                if transformed_relative != expected_transformed:
                    raise EvaluationOnlyError("transform artifact path binding changed")
                transformed_path = RUN_ROOT / transformed_relative
                _require_bound_file(transformed_path, transformed.get("sha256"))
                if probe_mp4(transformed_path) != transformed.get("stream"):
                    raise EvaluationOnlyError("transform MP4 stream identity changed")
                transform_paths[f"{group}:{condition}:{transform}"] = transformed_path
                observed_transforms.add(transformed_relative)
    if observed_conditions != expected_conditions or observed_transforms != expected_transforms:
        raise EvaluationOnlyError("80-artifact closed set changed")
    checksum_lines = ORIGINAL_CHECKSUMS.read_text(encoding="utf-8").splitlines()
    if len(checksum_lines) != 83:
        raise EvaluationOnlyError("original checksum inventory count changed")
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        _require_bound_file(RUN_ROOT / relative, digest)
    return audit, independent, formula, condition_paths, transform_paths


def run_evaluation_only_once(*, repo_root: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise EvaluationOnlyError("evaluation-only result already exists")
    if not RUN_ROOT.is_dir() or RUN_ROOT.is_symlink():
        raise EvaluationOnlyError("bound Phase-C run root unavailable")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise EvaluationOnlyError("evaluation-only closure requires a clean checkout")
    started_at = utc_now()
    audit, independent, formula, condition_paths, transform_paths = _validate_all_identities(repo_root)
    clean_outputs: dict[str, Any] = {}
    clean_truth: dict[str, Any] = {}
    transform_outputs: dict[str, Any] = {}
    transform_truth: dict[str, Any] = {}
    mismatches: list[str] = []
    for group in GROUP_ORDER:
        for condition in CONDITION_ORDER:
            key = f"{group}:{condition}"
            output = evaluate_transformed_target(decode_saved_mp4(condition_paths[key]), "identity")
            clean_outputs[key] = output
            clean_truth[key], _stage = audit_target(output, condition, transformed=True)
            if not _compare_clean_output(output, audit["condition_outputs"][key]):
                mismatches.append(f"clean_output:{key}")
            for transform in TRANSFORM_ORDER:
                transformed_key = f"{group}:{condition}:{transform}"
                transformed_output = evaluate_transformed_target(decode_saved_mp4(transform_paths[transformed_key]), transform)
                transform_outputs[transformed_key] = transformed_output
                transform_truth[transformed_key], _stage = audit_target(transformed_output, condition, transformed=True)
                if canonical_json_bytes(transformed_output) != canonical_json_bytes(audit["transform_outputs"][transformed_key]):
                    mismatches.append(f"transform_output:{transformed_key}")
    if canonical_json_bytes(clean_truth) != canonical_json_bytes(formula["condition_truth"]):
        mismatches.append("clean_truth:frozen_formula")
    if canonical_json_bytes(transform_truth) != canonical_json_bytes(audit["transform_truth_after_outputs_frozen"]):
        mismatches.append("transform_truth:original_and_independent")
    if independent.get("transform_outputs_exact_match") is not True or independent.get("transform_truth_exact_match") is not True:
        mismatches.append("transform_truth:first_independent")
    clean_off_pass = sum(clean_truth[f"{group}:{condition}"]["passed"] for group in GROUP_ORDER for condition in ("OFF_R1", "OFF_R2"))
    clean_positive_pass = sum(clean_truth[f"{group}:{condition}"]["passed"] for group in GROUP_ORDER for condition in ("A", "B"))
    transformed_off_pass = sum(transform_truth[f"{group}:{condition}:{transform}"]["passed"] for group in GROUP_ORDER for condition in ("OFF_R1", "OFF_R2") for transform in TRANSFORM_ORDER)
    transformed_positive_pass = sum(transform_truth[f"{group}:{condition}:{transform}"]["passed"] for group in GROUP_ORDER for condition in ("A", "B") for transform in TRANSFORM_ORDER)
    all_pass = (clean_off_pass, clean_positive_pass, transformed_off_pass, transformed_positive_pass) == (8, 8, 32, 32)
    effective = all_pass and not mismatches
    status = "EFFECTIVE_ON_FRESH_SMALL_SAMPLE" if effective else "INSUFFICIENT"
    phase_c = "FEASIBLE" if effective else "INSUFFICIENT_TO_DECIDE"
    route = "METHOD_EFFECTIVENESS_SMALL_SAMPLE_CLOSED" if effective else "INSUFFICIENT"
    result = {"schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status, "phase_c": phase_c,
              "route": route, "original_runner_result_classification": "ENGINEERING_EVALUATION_INVALID",
              "original_runner_audit_sha256": ORIGINAL_AUDIT_SHA256,
              "independent_replay_sha256": INDEPENDENT_SHA256, "frozen_formula_crosscheck_sha256": FROZEN_FORMULA_SHA256,
              "input_zip": {"absolute_path": str(SOURCE_ZIP), "size": SOURCE_ZIP_SIZE, "sha256": SOURCE_ZIP_SHA256},
              "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE},
              "artifacts_checked_before_decode": 80, "checksum_entries_verified": 83,
              "clean_off_pass": clean_off_pass, "clean_off_total": 8,
              "clean_positive_identity_pass": clean_positive_pass, "clean_positive_total": 8,
              "transformed_off_pass": transformed_off_pass, "transformed_off_total": 32,
              "transformed_positive_pass": transformed_positive_pass, "transformed_positive_total": 32,
              "mismatch_paths": mismatches, "clean_outputs": clean_outputs, "transform_outputs": transform_outputs,
              "clean_truth_after_outputs_frozen": clean_truth, "transform_truth_after_outputs_frozen": transform_truth,
              "generation_was_run": False, "new_artifact_derivation_was_run": False,
              "formal_result": False, "stage_progression_allowed": False,
              "started_at": started_at, "ended_at": utc_now()}
    OUTPUT.mkdir(mode=0o755)
    for name in ("audit.json", "result.json"):
        (OUTPUT / name).write_bytes(canonical_json_bytes(result) + b"\n")
    (OUTPUT / "command.json").write_bytes(canonical_json_bytes({"argv": list(argv), "cwd": str(cwd.resolve()), "exit_code": 0 if effective else 2}) + b"\n")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file())
    (OUTPUT / "checksums.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in files) + "\n", encoding="utf-8")
    return {"status": status, "diagnostic_class": DIAGNOSTIC_CLASS, "phase_c": phase_c, "route": route,
            "original_runner_result_classification": "ENGINEERING_EVALUATION_INVALID", "actual_result_path": str(OUTPUT),
            "clean_off_pass": clean_off_pass, "clean_positive_identity_pass": clean_positive_pass,
            "transformed_off_pass": transformed_off_pass, "transformed_positive_pass": transformed_positive_pass,
            "mismatch_count": len(mismatches)}
