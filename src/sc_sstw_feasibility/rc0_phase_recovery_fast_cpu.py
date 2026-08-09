"""Frozen Step-5 phase-recovery diagnostic over Step-4 calibrations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import CalibrationResult, equalize_observations
from .learned_observation import decode_saved_mp4, extract_feature_matrix
from .rc0_aisb_capture_fast_cpu import BOUND_ARTIFACT_SHA256, CONDITION_ORDER, GROUP_ORDER
from .rc0_causal_localization_v2 import schedule_a, schedule_b
from .rc0_pixel_chroma_fast_cpu import probe_mp4
from .sync import dynamic_time_sync, dynamic_time_sync_score_bounded


SCHEMA = "sc_sstw_rc0_phase_recovery_fast_cpu_v1"
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
STEP4_EXECUTION_COMMIT = "e57151a502343d3a365df708be366c5f0040283a"
STEP4_EXECUTION_TREE = "78c6206948de76cb525aecc823d88c16e597a22e"
STEP4_CLARIFICATION_COMMIT = "dd24f4d9ee2841c94a1ef491efa2791f4585bfa7"
STEP4_CLARIFICATION_TREE = "6b0f29833b7be59d4489203b8dfd1ba7bef735e5"
STEP4_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-candidate-self-calibration-fast-cpu-b5c2cf6-run1")
STEP4_AUDIT = STEP4_ROOT / "audit.json"
STEP4_AUDIT_SHA256 = "572cc1a37a25e8a425125ee31db39e1c864c3e67604cbd964d6dc876740b3a8c"
VIDEO_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-pixel-chroma-fast-cpu-9e19d31-run1/output/videos")
RUN_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-phase-recovery-fast-cpu-dd24f4d-run1")
SKIP_PENALTY = 0.22
REPEAT_PENALTY = 0.14
EQUALIZATION_RIDGE = 1e-4
MAX_EDIT_BUDGET = 2
SCORE_TOLERANCE = 1e-12
PERTURBATION_SOURCE_INDICES = {
    "identity": tuple(range(13)),
    "delete6_duplicate12": (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 12),
    "local_phase_plus1": (0, 1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12),
    "local_phase_minus1": (0, 1, 2, 3, 4, 5, 5, 7, 8, 9, 10, 11, 12),
}
PERTURBATION_ORDER = tuple(PERTURBATION_SOURCE_INDICES)


class PhaseRecoveryDiagnosticError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or config.get("diagnostic_class") != DIAGNOSTIC_CLASS:
        raise PhaseRecoveryDiagnosticError("Step5 config identity mismatch")
    projection = config.get("projection", {})
    dp = config.get("dp", {})
    if projection.get("observation_shape") != [13, 30] or projection.get("projected_shape") != [13, 2]:
        raise PhaseRecoveryDiagnosticError("projection shape changed")
    if projection.get("equalization_ridge") != EQUALIZATION_RIDGE:
        raise PhaseRecoveryDiagnosticError("equalization ridge changed")
    if dp.get("skip_penalty") != SKIP_PENALTY or dp.get("repeat_penalty") != REPEAT_PENALTY:
        raise PhaseRecoveryDiagnosticError("DP penalty changed")
    if dp.get("band") is not None or dp.get("maximum_edit_budget") != MAX_EDIT_BUDGET:
        raise PhaseRecoveryDiagnosticError("DP band/edit budget changed")
    if dp.get("transition_tie_order") != ["MATCH", "SKIP_TEMPLATE", "REPEAT_TEMPLATE"]:
        raise PhaseRecoveryDiagnosticError("DP tie rule changed")
    perturbations = config.get("perturbations", [])
    if [item.get("id") for item in perturbations] != list(PERTURBATION_ORDER):
        raise PhaseRecoveryDiagnosticError("perturbation order changed")
    for item in perturbations:
        if item.get("source_indices") != list(PERTURBATION_SOURCE_INDICES[item["id"]]):
            raise PhaseRecoveryDiagnosticError("perturbation mapping changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise PhaseRecoveryDiagnosticError("diagnostic boundary changed")
    return config


def apply_time_perturbation(states: Any, perturbation: str) -> list[tuple[float, float]]:
    array = np.asarray(states, dtype=np.float64)
    if array.shape != (13, 2) or not np.isfinite(array).all():
        raise PhaseRecoveryDiagnosticError("projected trajectory must be finite 13x2")
    if perturbation not in PERTURBATION_SOURCE_INDICES:
        raise PhaseRecoveryDiagnosticError("unknown time perturbation")
    return [tuple(map(float, array[index])) for index in PERTURBATION_SOURCE_INDICES[perturbation]]


def expected_operation_path(perturbation: str) -> list[dict[str, Any]]:
    if perturbation not in PERTURBATION_SOURCE_INDICES:
        raise PhaseRecoveryDiagnosticError("unknown time perturbation")
    operations: list[dict[str, Any]] = []
    consumed_template = 0
    for observed_index, template_index in enumerate(PERTURBATION_SOURCE_INDICES[perturbation]):
        while consumed_template < template_index:
            operations.append({"operation": "SKIP_TEMPLATE", "observed_index": None, "template_index": consumed_template})
            consumed_template += 1
        if template_index == consumed_template:
            operations.append({"operation": "MATCH", "observed_index": observed_index, "template_index": template_index})
            consumed_template += 1
        elif template_index == consumed_template - 1:
            operations.append({"operation": "REPEAT_TEMPLATE", "observed_index": observed_index, "template_index": template_index})
        else:
            raise PhaseRecoveryDiagnosticError("perturbation mapping is not representable by frozen DP")
    while consumed_template < 13:
        operations.append({"operation": "SKIP_TEMPLATE", "observed_index": None, "template_index": consumed_template})
        consumed_template += 1
    return operations


def recover_complete_path(observed: Sequence[tuple[float, float]], template: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Duplicate the existing recurrence only to expose skip/repeat operations."""

    rows, cols = len(observed), len(template)
    if rows != 13 or cols != 13:
        raise PhaseRecoveryDiagnosticError("DP requires exact 13x13 inputs")
    dp = [[math.inf] * (cols + 1) for _ in range(rows + 1)]
    parent: list[list[tuple[int, int, str] | None]] = [[None] * (cols + 1) for _ in range(rows + 1)]
    dp[0][0] = 0.0
    for col in range(1, cols + 1):
        dp[0][col] = dp[0][col - 1] + SKIP_PENALTY
        parent[0][col] = (0, col - 1, "SKIP_TEMPLATE")
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            dx = observed[row - 1][0] - template[col - 1][0]
            dy = observed[row - 1][1] - template[col - 1][1]
            local = dx * dx + dy * dy
            choices = (
                (dp[row - 1][col - 1] + local, row - 1, col - 1, "MATCH"),
                (dp[row][col - 1] + SKIP_PENALTY, row, col - 1, "SKIP_TEMPLATE"),
                (dp[row - 1][col] + REPEAT_PENALTY + local, row - 1, col, "REPEAT_TEMPLATE"),
            )
            cost, previous_row, previous_col, operation = min(choices, key=lambda item: item[0])
            dp[row][col] = cost
            parent[row][col] = (previous_row, previous_col, operation)
    operations: list[dict[str, Any]] = []
    diagonal: list[list[int]] = []
    row, col = rows, cols
    while row > 0 or col > 0:
        previous = parent[row][col]
        if previous is None:
            raise PhaseRecoveryDiagnosticError("DP backtrack is incomplete")
        previous_row, previous_col, operation = previous
        if operation == "MATCH":
            record = {"operation": operation, "observed_index": row - 1, "template_index": col - 1}
            diagonal.append([row - 1, col - 1])
        elif operation == "SKIP_TEMPLATE":
            record = {"operation": operation, "observed_index": None, "template_index": col - 1}
        else:
            record = {"operation": operation, "observed_index": row - 1, "template_index": col - 1}
        operations.append(record)
        row, col = previous_row, previous_col
    operations.reverse()
    diagonal.reverse()
    match_count = len(diagonal)
    average = dp[rows][cols] / max(1, match_count)
    return {
        "operations": operations, "diagonal_path": diagonal,
        "skip_count": sum(item["operation"] == "SKIP_TEMPLATE" for item in operations),
        "repeat_count": sum(item["operation"] == "REPEAT_TEMPLATE" for item in operations),
        "edit_count": sum(item["operation"] != "MATCH" for item in operations),
        "total_cost": dp[rows][cols], "average_cost": average, "score": -average,
    }


def evaluate_phase_cell(states: Any, own_template: str, perturbation: str) -> dict[str, Any]:
    if own_template not in {"A", "B"}:
        raise PhaseRecoveryDiagnosticError("own template must be A or B")
    observed = apply_time_perturbation(states, perturbation)
    own = [tuple(map(float, point)) for point in (schedule_a() if own_template == "A" else schedule_b())]
    cross_id = "B" if own_template == "A" else "A"
    cross = [tuple(map(float, point)) for point in (schedule_b() if own_template == "A" else schedule_a())]
    primary = dynamic_time_sync(observed, own, skip_penalty=SKIP_PENALTY, repeat_penalty=REPEAT_PENALTY)
    cross_primary = dynamic_time_sync(observed, cross, skip_penalty=SKIP_PENALTY, repeat_penalty=REPEAT_PENALTY)
    complete = recover_complete_path(observed, own)
    bounded_score, bounded_abandoned = dynamic_time_sync_score_bounded(
        observed, own, min_score_to_beat=-math.inf,
        skip_penalty=SKIP_PENALTY, repeat_penalty=REPEAT_PENALTY,
    )
    expected = expected_operation_path(perturbation)
    expected_diagonal = [[item["observed_index"], item["template_index"]] for item in expected if item["operation"] == "MATCH"]
    expected_skip = sum(item["operation"] == "SKIP_TEMPLATE" for item in expected)
    expected_repeat = sum(item["operation"] == "REPEAT_TEMPLATE" for item in expected)
    primary_path = [list(pair) for pair in primary.path]
    score_consistent = abs(primary.score - complete["score"]) <= SCORE_TOLERANCE and abs(primary.score - bounded_score) <= SCORE_TOLERANCE
    path_exact = complete["operations"] == expected and primary_path == expected_diagonal
    edit_exact = complete["skip_count"] == expected_skip and complete["repeat_count"] == expected_repeat and complete["edit_count"] <= MAX_EDIT_BUDGET
    own_beats_cross = primary.score > cross_primary.score
    passed = path_exact and edit_exact and score_consistent and not bounded_abandoned and own_beats_cross
    return {
        "own_template": own_template, "cross_template": cross_id, "perturbation": perturbation,
        "source_indices": list(PERTURBATION_SOURCE_INDICES[perturbation]),
        "expected_operations": expected, "recovered_operations": complete["operations"],
        "primary_diagonal_path": primary_path, "expected_diagonal_path": expected_diagonal,
        "skip_count": complete["skip_count"], "repeat_count": complete["repeat_count"],
        "edit_count": complete["edit_count"], "maximum_edit_budget": MAX_EDIT_BUDGET,
        "own_score": primary.score, "own_average_cost": primary.average_cost,
        "cross_score": cross_primary.score, "cross_average_cost": cross_primary.average_cost,
        "own_beats_cross": own_beats_cross,
        "bounded_score_crosscheck": bounded_score, "bounded_abandoned": bounded_abandoned,
        "complete_backtracker_score": complete["score"], "score_consistent": score_consistent,
        "path_exact": path_exact, "edit_exact": edit_exact, "passed": passed,
    }


def _frozen_calibration(step4: Mapping[str, Any], group: str, condition: str) -> tuple[CalibrationResult, str]:
    keys = [f"{group}:{condition}:identity", f"{group}:{condition}:private_tail_delete6_duplicate12"]
    fits: list[dict[str, Any]] = []
    for key in keys:
        cell = step4.get("cells", {}).get(key)
        if not isinstance(cell, dict) or cell.get("truth_audit", {}).get("passed") is not True:
            raise PhaseRecoveryDiagnosticError(f"Step4 cell is not frozen-pass for {key}")
        selected = cell["truth_audit"].get("selected_candidates")
        ranked = cell.get("ranked_candidates")
        if not isinstance(selected, list) or len(selected) != 1 or not isinstance(ranked, list) or len(ranked) != 8:
            raise PhaseRecoveryDiagnosticError("Step4 K2/candidate record invalid")
        if selected[0].get("start_index") != 0 or selected[0].get("template_id") != condition:
            raise PhaseRecoveryDiagnosticError("Step4 selected candidate identity changed")
        winner = ranked[0]
        if winner.get("capture_identity") != selected[0] or winner.get("calibration_rank") != 1:
            raise PhaseRecoveryDiagnosticError("Step4 winner/calibration identity mismatch")
        fits.append(winner.get("fit"))
    if canonical_json_bytes(fits[0]) != canonical_json_bytes(fits[1]):
        raise PhaseRecoveryDiagnosticError("Step4 perturbations do not share one frozen calibrator")
    fit = fits[0]
    matrix = fit.get("matrix") if isinstance(fit, dict) else None
    bias = fit.get("bias") if isinstance(fit, dict) else None
    if not isinstance(matrix, list) or len(matrix) != 30 or any(not isinstance(row, list) or len(row) != 2 for row in matrix):
        raise PhaseRecoveryDiagnosticError("Step4 calibration matrix invalid")
    if not isinstance(bias, list) or len(bias) != 30:
        raise PhaseRecoveryDiagnosticError("Step4 calibration bias invalid")
    values = np.asarray(matrix + [bias], dtype=np.float64)
    if not np.isfinite(values).all():
        raise PhaseRecoveryDiagnosticError("Step4 calibration is non-finite")
    calibration = CalibrationResult(
        matrix=[[float(value) for value in row] for row in matrix],
        bias=[float(value) for value in bias],
        condition_number=float(fit["condition_number"]),
        pilot_reconstruction_mse=float(fit["pilot_reconstruction_mse"]),
    )
    return calibration, hashlib.sha256(canonical_json_bytes(fit)).hexdigest()


def _validate_inputs(repo_root: Path) -> tuple[dict[str, dict[str, Path]], dict[str, Any], dict[str, Any]]:
    config_path = repo_root / "configs/rc0_phase_recovery_fast_cpu.json"
    protocol_path = repo_root / "protocols/rc0_phase_recovery_fast_cpu.md"
    config = load_and_validate_config(config_path)
    for commit, expected_tree in ((STEP4_EXECUTION_COMMIT, STEP4_EXECUTION_TREE), (STEP4_CLARIFICATION_COMMIT, STEP4_CLARIFICATION_TREE)):
        actual_tree = subprocess.run(["git", "rev-parse", f"{commit}^{{tree}}"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
        if actual_tree != expected_tree:
            raise PhaseRecoveryDiagnosticError("Step4 commit/tree identity mismatch")
    if STEP4_AUDIT.is_symlink() or not STEP4_AUDIT.is_file() or sha256_file(STEP4_AUDIT) != STEP4_AUDIT_SHA256:
        raise PhaseRecoveryDiagnosticError("Step4 audit identity mismatch")
    step4 = json.loads(STEP4_AUDIT.read_text(encoding="utf-8"))
    if (step4.get("status"), step4.get("question4"), step4.get("route"), step4.get("K2")) != (
        "CPU_DIAGNOSTIC_RESULT_READY", "FEASIBLE", "PROCEED_STEP5_PHASE_RECOVERY_CPU", 1,
    ):
        raise PhaseRecoveryDiagnosticError("Step4 outcome identity mismatch")
    paths: dict[str, dict[str, Path]] = {}
    videos: dict[str, Any] = {}
    for group in GROUP_ORDER:
        paths[group] = {}
        videos[group] = {}
        for condition in CONDITION_ORDER:
            path = VIDEO_ROOT / group / condition / "saved.mp4"
            digest = BOUND_ARTIFACT_SHA256[group][condition]
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise PhaseRecoveryDiagnosticError(f"bound video identity mismatch for {group}:{condition}")
            paths[group][condition] = path
            videos[group][condition] = {"absolute_path": str(path), "sha256": digest, "size": path.stat().st_size, "stream": probe_mp4(path)}
    return paths, step4, {
        "step4_execution_commit": STEP4_EXECUTION_COMMIT, "step4_execution_tree": STEP4_EXECUTION_TREE,
        "step4_clarification_commit": STEP4_CLARIFICATION_COMMIT, "step4_clarification_tree": STEP4_CLARIFICATION_TREE,
        "step4_audit_path": str(STEP4_AUDIT), "step4_audit_sha256": STEP4_AUDIT_SHA256,
        "config_sha256": sha256_file(config_path), "protocol_sha256": sha256_file(protocol_path),
        "videos": videos, "config": config,
    }


def run_step5_once(*, repo_root: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if RUN_ROOT.exists() or RUN_ROOT.is_symlink():
        raise PhaseRecoveryDiagnosticError("Step5 run root already exists")
    if not RUN_ROOT.parent.is_dir() or RUN_ROOT.parent.is_symlink():
        raise PhaseRecoveryDiagnosticError("Step5 run parent unavailable")
    started_at = utc_now()
    paths, step4, input_identity = _validate_inputs(repo_root)
    features = {group: {condition: extract_feature_matrix(decode_saved_mp4(paths[group][condition])) for condition in CONDITION_ORDER} for group in GROUP_ORDER}
    cells: dict[str, Any] = {}
    calibrations: dict[str, Any] = {}
    for group in GROUP_ORDER:
        off_ref = (np.asarray(features[group]["OFF_R1"], dtype=np.float64) + np.asarray(features[group]["OFF_R2"], dtype=np.float64)) / 2.0
        for condition in ("A", "B"):
            paired = np.asarray(features[group][condition], dtype=np.float64) - off_ref
            if paired.shape != (13, 30) or not np.isfinite(paired).all():
                raise PhaseRecoveryDiagnosticError("paired 30D matrix invalid")
            calibration, calibration_sha = _frozen_calibration(step4, group, condition)
            projected = equalize_observations(paired.tolist(), calibration, ridge=EQUALIZATION_RIDGE)
            calibrations[f"{group}:{condition}"] = {"frozen_fit_sha256": calibration_sha, "source": "Step4_K2_1_identity_and_tail_equal", "refit": False}
            for perturbation in PERTURBATION_ORDER:
                key = f"{group}:{condition}:{perturbation}"
                cell = evaluate_phase_cell(projected, condition, perturbation)
                cell["group"] = group
                cell["condition_truth_for_post_recovery_audit"] = condition
                cell["frozen_calibration_sha256"] = calibration_sha
                cells[key] = cell
    feasible = len(cells) == 16 and all(cell["passed"] for cell in cells.values())
    question5 = "FEASIBLE" if feasible else "NOT_FEASIBLE"
    status = "CPU_DIAGNOSTIC_RESULT_READY" if feasible else "PHASE_RECOVERY_NOT_FEASIBLE"
    route = "PROCEED_STEP6_TINY_E2E_CPU" if feasible else "PHASE_RECOVERY_NOT_FEASIBLE"
    result = {
        "schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status,
        "question5": question5, "route": route,
        "primary_formula": "dynamic_time_sync_v1", "bounded_crosscheck": "dynamic_time_sync_score_bounded_v1",
        "skip_penalty": SKIP_PENALTY, "repeat_penalty": REPEAT_PENALTY,
        "band": None, "maximum_edit_budget": MAX_EDIT_BUDGET,
        "perturbations": list(PERTURBATION_ORDER), "cells": cells,
        "calibrations": calibrations, "input_identity": input_identity,
        "capture_was_rerun": False, "calibration_was_refit": False,
        "generation_was_run": False, "encoding_was_run": False, "training_was_run": False,
        "formal_result": False, "stage_progression_allowed": False,
        "started_at": started_at, "ended_at": utc_now(),
    }
    RUN_ROOT.mkdir(mode=0o755)
    for name in ("result.json", "audit.json"):
        (RUN_ROOT / name).write_bytes(canonical_json_bytes(result) + b"\n")
    command = {"argv": list(argv), "cwd": str(cwd.resolve()), "started_at": started_at, "ended_at": result["ended_at"], "exit_code": 0}
    (RUN_ROOT / "command.json").write_bytes(canonical_json_bytes(command) + b"\n")
    checksum_paths = sorted(path for path in RUN_ROOT.iterdir() if path.is_file())
    (RUN_ROOT / "checksums.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in checksum_paths) + "\n", encoding="utf-8")
    return {"status": status, "diagnostic_class": DIAGNOSTIC_CLASS, "actual_run_root": str(RUN_ROOT), "question5": question5, "route": route, "cell_count": len(cells)}
