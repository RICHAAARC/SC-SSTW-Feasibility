"""Frozen Step-4 candidate-wise self-calibration diagnostic."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import calibrate_from_pilot_pairs, equalize_observations
from .learned_observation import decode_saved_mp4, extract_feature_matrix
from .rc0_aisb_capture_fast_cpu import (
    BOUND_ARTIFACT_SHA256,
    CONDITION_ORDER,
    GROUP_ORDER,
    PERTURBATION_ORDER,
    perturb_observation,
)
from .rc0_causal_localization_v2 import schedule_a, schedule_b
from .rc0_pixel_chroma_fast_cpu import probe_mp4


SCHEMA = "sc_sstw_rc0_candidate_self_calibration_fast_cpu_v1"
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
CANDIDATE_BUDGET_K = 8
K2 = 1
FIT_INDICES = (0, 1, 2, 3)
HELD_OUT_INDICES = (4, 5)
EQUALIZATION_RIDGE = 1e-4
LEAST_SQUARES_IMPLEMENTATION_RIDGE = 1e-8
STEP3_COMMIT = "b5c2cf62fc21f00f72e5a223af82e9acf19a746e"
STEP3_TREE = "3a5562cdabac1a9adfca404cb89bce6b0976fb1d"
STEP3_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-aisb-capture-fast-cpu-dff72da-run1")
STEP3_AUDIT = STEP3_ROOT / "audit.json"
STEP3_AUDIT_SHA256 = "6e424acbd21a36c0c965581e475a7e03440051ad62925c37fceecb3ac1c8ba6f"
STEP3_INDEPENDENT = STEP3_ROOT.with_name(STEP3_ROOT.name + ".independent.json")
STEP3_INDEPENDENT_SHA256 = "606d868b11fc49ee673040dca93ac6df657831ac215004c4e60500f2531b1059"
VIDEO_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-pixel-chroma-fast-cpu-9e19d31-run1/output/videos")
RUN_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-candidate-self-calibration-fast-cpu-b5c2cf6-run1")


class SelfCalibrationDiagnosticError(RuntimeError):
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
        raise SelfCalibrationDiagnosticError("Step4 config identity mismatch")
    calibration = config.get("calibration", {})
    selection = config.get("selection", {})
    separation = config.get("separation", {})
    if calibration.get("fit_indices") != list(FIT_INDICES) or calibration.get("held_out_indices") != list(HELD_OUT_INDICES):
        raise SelfCalibrationDiagnosticError("fit/held-out split changed")
    if calibration.get("equalization_ridge") != EQUALIZATION_RIDGE or calibration.get("acceptance_threshold") is not None:
        raise SelfCalibrationDiagnosticError("calibration ridge or threshold changed")
    if calibration.get("least_squares_regularization_argument") is not None or calibration.get("least_squares_implementation_default_ridge") != LEAST_SQUARES_IMPLEMENTATION_RIDGE:
        raise SelfCalibrationDiagnosticError("least-squares implementation ridge changed")
    if selection.get("K2") != K2 or selection.get("score_tie") != "cell_failure":
        raise SelfCalibrationDiagnosticError("K2 or tie rule changed")
    if separation.get("capture_candidate_budget_K") != CANDIDATE_BUDGET_K:
        raise SelfCalibrationDiagnosticError("capture budget changed")
    if not separation.get("capture_is_frozen_before_calibration") or separation.get("calibration_may_reorder_or_expand_capture"):
        raise SelfCalibrationDiagnosticError("capture/calibration separation changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise SelfCalibrationDiagnosticError("diagnostic boundary changed")
    return config


def _template_points(template_id: str) -> tuple[tuple[float, float], ...]:
    if template_id == "A":
        return tuple(schedule_a()[:6])
    if template_id == "B":
        return tuple(schedule_b()[:6])
    raise SelfCalibrationDiagnosticError("unknown captured template")


def _capture_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "start_index", "template_id", "residual", "observed_length",
        "missing_template_index", "full_rank", "selected_rank",
    }
    if set(candidate) != expected:
        raise SelfCalibrationDiagnosticError("captured candidate schema changed")
    start = candidate["start_index"]
    template_id = candidate["template_id"]
    residual = candidate["residual"]
    if type(start) is not int or start not in range(8):
        raise SelfCalibrationDiagnosticError("captured start changed")
    if template_id not in {"A", "B"} or type(template_id) is not str:
        raise SelfCalibrationDiagnosticError("captured template changed")
    if type(residual) is not float or not math.isfinite(residual) or residual < 0.0:
        raise SelfCalibrationDiagnosticError("captured residual invalid")
    if candidate["observed_length"] != 6 or candidate["missing_template_index"] is not None:
        raise SelfCalibrationDiagnosticError("captured window semantics changed")
    if type(candidate["full_rank"]) is not int or type(candidate["selected_rank"]) is not int:
        raise SelfCalibrationDiagnosticError("captured rank invalid")
    return {key: candidate[key] for key in (
        "start_index", "template_id", "residual", "observed_length",
        "missing_template_index", "full_rank", "selected_rank",
    )}


def validate_capture_set(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(candidates) != CANDIDATE_BUDGET_K:
        raise SelfCalibrationDiagnosticError("capture set is not K=8")
    records = [_capture_identity(candidate) for candidate in candidates]
    keys = [(record["start_index"], record["template_id"]) for record in records]
    if len(set(keys)) != CANDIDATE_BUDGET_K or sorted(record["start_index"] for record in records) != list(range(8)):
        raise SelfCalibrationDiagnosticError("capture set identities changed")
    return records


def capture_set_sha256(candidates: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(validate_capture_set(candidates))).hexdigest()


def calibrate_capture_candidates(observation: Any, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fit every captured candidate without receiving condition truth."""

    array = np.asarray(observation, dtype=np.float64)
    if array.shape != (13, 30) or not np.isfinite(array).all():
        raise SelfCalibrationDiagnosticError("paired observation must be finite 13x30")
    frozen = validate_capture_set(candidates)
    records: list[dict[str, Any]] = []
    for candidate in frozen:
        start = candidate["start_index"]
        template = _template_points(candidate["template_id"])
        window = array[start : start + 6]
        if window.shape != (6, 30):
            raise SelfCalibrationDiagnosticError("candidate window incomplete")
        calibration = calibrate_from_pilot_pairs([
            (template[index], window[index].tolist()) for index in FIT_INDICES
        ])
        equalized = equalize_observations(
            [window[index].tolist() for index in HELD_OUT_INDICES],
            calibration,
            ridge=EQUALIZATION_RIDGE,
        )
        score = sum(
            (equalized[offset][dimension] - template[index][dimension]) ** 2
            for offset, index in enumerate(HELD_OUT_INDICES)
            for dimension in range(2)
        ) / 4.0
        if not math.isfinite(score):
            raise SelfCalibrationDiagnosticError("candidate calibration score is non-finite")
        records.append({
            "capture_identity": candidate,
            "fit": {
                "matrix": calibration.matrix,
                "bias": calibration.bias,
                "parameter_dimension": len(calibration.matrix) * 2 + len(calibration.bias),
                "condition_number": calibration.condition_number,
                "pilot_reconstruction_mse": calibration.pilot_reconstruction_mse,
                "fit_indices": list(FIT_INDICES),
                "held_out_indices": list(HELD_OUT_INDICES),
                "equalization_ridge": EQUALIZATION_RIDGE,
            },
            "held_out_equalized": [list(point) for point in equalized],
            "held_out_mse": score,
        })
    template_order = {"A": 0, "B": 1}
    records.sort(key=lambda item: (
        item["held_out_mse"], item["capture_identity"]["residual"],
        item["capture_identity"]["start_index"], template_order[item["capture_identity"]["template_id"]],
    ))
    for rank, record in enumerate(records, start=1):
        record["calibration_rank"] = rank
    return records


def audit_ranked_cell(ranked: Sequence[Mapping[str, Any]], own_template: str) -> dict[str, Any]:
    if own_template not in {"A", "B"} or len(ranked) != CANDIDATE_BUDGET_K:
        raise SelfCalibrationDiagnosticError("truth audit input invalid")
    scores = [float(record["held_out_mse"]) for record in ranked]
    unique_winner = scores[0] < scores[1]
    selected = [dict(ranked[0]["capture_identity"])] if unique_winner else []
    target = {"start_index": 0, "template_id": own_template}
    target_record = next(record for record in ranked if (
        record["capture_identity"]["start_index"], record["capture_identity"]["template_id"]
    ) == (0, own_template))
    passed = unique_winner and selected[0]["start_index"] == 0 and selected[0]["template_id"] == own_template
    return {
        "K2": K2,
        "unique_winner": unique_winner,
        "selected_candidates": selected,
        "target": target,
        "target_calibration_rank": target_record["calibration_rank"],
        "target_held_out_mse": target_record["held_out_mse"],
        "runner_up_held_out_mse": scores[1],
        "passed": passed,
    }


def _validate_inputs(repo_root: Path) -> tuple[dict[str, dict[str, Path]], dict[str, Any], dict[str, Any]]:
    config_path = repo_root / "configs/rc0_candidate_self_calibration_fast_cpu.json"
    protocol_path = repo_root / "protocols/rc0_candidate_self_calibration_fast_cpu.md"
    config = load_and_validate_config(config_path)
    tree = subprocess.run(["git", "rev-parse", f"{STEP3_COMMIT}^{{tree}}"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    if tree != STEP3_TREE:
        raise SelfCalibrationDiagnosticError("Step3 commit/tree mismatch")
    for path, digest in ((STEP3_AUDIT, STEP3_AUDIT_SHA256), (STEP3_INDEPENDENT, STEP3_INDEPENDENT_SHA256)):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise SelfCalibrationDiagnosticError("Step3 evidence identity mismatch")
    step3 = json.loads(STEP3_AUDIT.read_text(encoding="utf-8"))
    if (step3.get("status"), step3.get("question3"), step3.get("route")) != (
        "CPU_DIAGNOSTIC_RESULT_READY", "FEASIBLE", "PROCEED_STEP4_SELF_CALIBRATION_CPU",
    ) or len(step3.get("cells", {})) != 8:
        raise SelfCalibrationDiagnosticError("Step3 outcome identity mismatch")
    paths: dict[str, dict[str, Path]] = {}
    videos: dict[str, Any] = {}
    for group in GROUP_ORDER:
        paths[group] = {}
        videos[group] = {}
        for condition in CONDITION_ORDER:
            path = VIDEO_ROOT / group / condition / "saved.mp4"
            digest = BOUND_ARTIFACT_SHA256[group][condition]
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise SelfCalibrationDiagnosticError(f"bound video identity mismatch for {group}:{condition}")
            paths[group][condition] = path
            videos[group][condition] = {
                "absolute_path": str(path), "sha256": digest, "size": path.stat().st_size,
                "stream": probe_mp4(path),
            }
    return paths, step3, {
        "step3_commit": STEP3_COMMIT, "step3_tree": STEP3_TREE,
        "step3_audit_path": str(STEP3_AUDIT), "step3_audit_sha256": STEP3_AUDIT_SHA256,
        "step3_independent_path": str(STEP3_INDEPENDENT), "step3_independent_sha256": STEP3_INDEPENDENT_SHA256,
        "config_sha256": sha256_file(config_path), "protocol_sha256": sha256_file(protocol_path),
        "videos": videos, "config": config,
    }


def run_step4_once(*, repo_root: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if RUN_ROOT.exists() or RUN_ROOT.is_symlink():
        raise SelfCalibrationDiagnosticError("Step4 run root already exists")
    if not RUN_ROOT.parent.is_dir() or RUN_ROOT.parent.is_symlink():
        raise SelfCalibrationDiagnosticError("Step4 run parent unavailable")
    started_at = utc_now()
    paths, step3, input_identity = _validate_inputs(repo_root)
    features = {
        group: {condition: extract_feature_matrix(decode_saved_mp4(paths[group][condition])) for condition in CONDITION_ORDER}
        for group in GROUP_ORDER
    }
    cells: dict[str, Any] = {}
    for group in GROUP_ORDER:
        off_ref = (np.asarray(features[group]["OFF_R1"], dtype=np.float64) + np.asarray(features[group]["OFF_R2"], dtype=np.float64)) / 2.0
        for condition in ("A", "B"):
            paired = np.asarray(features[group][condition], dtype=np.float64) - off_ref
            for perturbation in PERTURBATION_ORDER:
                key = f"{group}:{condition}:{perturbation}"
                step3_cell = step3["cells"].get(key)
                if not isinstance(step3_cell, dict) or step3_cell.get("target_captured") is not True:
                    raise SelfCalibrationDiagnosticError(f"Step3 cell invalid for {key}")
                candidates = validate_capture_set(step3_cell.get("candidate_set", []))
                before_sha = capture_set_sha256(candidates)
                edited = perturb_observation(paired, perturbation)
                ranked = calibrate_capture_candidates(edited, candidates)
                after_candidates = [record["capture_identity"] for record in ranked]
                after_sha = capture_set_sha256(sorted(after_candidates, key=lambda item: item["selected_rank"]))
                if before_sha != after_sha:
                    raise SelfCalibrationDiagnosticError("calibration changed capture identities")
                cells[key] = {
                    "group": group, "condition_truth_for_post_ranking_audit": condition,
                    "perturbation": perturbation, "capture_set_sha256_before": before_sha,
                    "capture_set_sha256_after": after_sha, "capture_set_unchanged": True,
                    "candidate_count_calibrated": len(ranked), "ranked_candidates": ranked,
                    "truth_audit": audit_ranked_cell(ranked, condition),
                }
    feasible = len(cells) == 8 and all(cell["truth_audit"]["passed"] for cell in cells.values())
    question4 = "FEASIBLE" if feasible else "NOT_FEASIBLE"
    status = "CPU_DIAGNOSTIC_RESULT_READY" if feasible else "SELF_CALIBRATION_NOT_FEASIBLE"
    route = "PROCEED_STEP5_PHASE_RECOVERY_CPU" if feasible else "SELF_CALIBRATION_NOT_FEASIBLE"
    result = {
        "schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status,
        "question4": question4, "route": route, "capture_candidate_budget_K": CANDIDATE_BUDGET_K,
        "K2": K2, "formula": "candidate_wise_four_pilot_affine_fit_two_point_heldout_mse_v1",
        "cells": cells, "input_identity": input_identity,
        "capture_was_rerun": False, "generation_was_run": False, "encoding_was_run": False,
        "training_was_run": False, "formal_result": False, "stage_progression_allowed": False,
        "started_at": started_at, "ended_at": utc_now(),
    }
    RUN_ROOT.mkdir(mode=0o755)
    for name in ("result.json", "audit.json"):
        (RUN_ROOT / name).write_bytes(canonical_json_bytes(result) + b"\n")
    command = {"argv": list(argv), "cwd": str(cwd.resolve()), "started_at": started_at, "ended_at": result["ended_at"], "exit_code": 0}
    (RUN_ROOT / "command.json").write_bytes(canonical_json_bytes(command) + b"\n")
    checksum_paths = sorted(path for path in RUN_ROOT.iterdir() if path.is_file())
    (RUN_ROOT / "checksums.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in checksum_paths) + "\n", encoding="utf-8")
    return {"status": status, "diagnostic_class": DIAGNOSTIC_CLASS, "actual_run_root": str(RUN_ROOT), "question4": question4, "route": route, "K2": K2}
