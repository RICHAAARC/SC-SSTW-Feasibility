"""Thin Step-6 orchestration over the frozen Step-1 through Step-5 functions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import CalibrationResult, equalize_observations
from .learned_observation import decode_saved_mp4, extract_feature_matrix
from .rc0_aisb_capture_fast_cpu import BOUND_ARTIFACT_SHA256, CONDITION_ORDER, GROUP_ORDER, capture_cell
from .rc0_candidate_self_calibration_fast_cpu import calibrate_capture_candidates, capture_set_sha256
from .rc0_causal_localization_v2 import level_p_metrics
from .rc0_phase_recovery_fast_cpu import EQUALIZATION_RIDGE, PERTURBATION_ORDER, evaluate_phase_cell
from .rc0_pixel_chroma_fast_cpu import _compare_recursive, probe_mp4


SCHEMA = "sc_sstw_rc0_tiny_e2e_fast_cpu_v1"
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
STEP_AUDITS = {
    "step12": (
        Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-pixel-chroma-fast-cpu-9e19d31-run1/evaluation-only-result/audit.json"),
        "9b08c88aad5378c2b0c94574585906e96a9608d094002248fddd6761b5e1dead",
    ),
    "step3": (
        Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-aisb-capture-fast-cpu-dff72da-run1/audit.json"),
        "6e424acbd21a36c0c965581e475a7e03440051ad62925c37fceecb3ac1c8ba6f",
    ),
    "step4": (
        Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-candidate-self-calibration-fast-cpu-b5c2cf6-run1/audit.json"),
        "572cc1a37a25e8a425125ee31db39e1c864c3e67604cbd964d6dc876740b3a8c",
    ),
    "step5": (
        Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-phase-recovery-fast-cpu-dd24f4d-run2/audit.json"),
        "9e8b6fa4127d9293a50d64e888e6e311d33b1309f5791920c575d9d23f704a46",
    ),
}
VIDEO_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-pixel-chroma-fast-cpu-9e19d31-run1/output/videos")
RUN_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-tiny-e2e-fast-cpu-071d054-run1")


class TinyE2EDiagnosticError(RuntimeError):
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
        raise TinyE2EDiagnosticError("Step6 config identity mismatch")
    if config.get("input", {}).get("groups") != list(GROUP_ORDER) or config.get("input", {}).get("conditions") != list(CONDITION_ORDER):
        raise TinyE2EDiagnosticError("tiny sample identity changed")
    expected_order = ["mp4_identity_decode", "level_p_presence", "paired_30D", "AISB_16_to_K8", "candidate_wise_calibration_K2_1", "DP_phase_recovery"]
    pipeline = config.get("pipeline", {})
    if pipeline.get("stage_order") != expected_order or (pipeline.get("capture_universe"), pipeline.get("capture_K"), pipeline.get("calibration_K2")) != (16, 8, 1):
        raise TinyE2EDiagnosticError("pipeline order or budget changed")
    if config.get("timing", {}).get("all_phase_perturbations") != list(PERTURBATION_ORDER):
        raise TinyE2EDiagnosticError("phase perturbations changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise TinyE2EDiagnosticError("diagnostic boundary changed")
    return config


def capture_candidates_without_truth(paired: Any) -> list[dict[str, Any]]:
    """Return only the candidate set after proving it is audit-label invariant."""

    from_a = capture_cell(paired, "A", "identity")["candidate_set"]
    from_b = capture_cell(paired, "B", "identity")["candidate_set"]
    if canonical_json_bytes(from_a) != canonical_json_bytes(from_b):
        raise TinyE2EDiagnosticError("capture candidate set depends on truth audit label")
    return [dict(candidate) for candidate in from_a]


def _calibration_from_ranked_winner(winner: Mapping[str, Any]) -> CalibrationResult:
    fit = winner.get("fit")
    if not isinstance(fit, dict):
        raise TinyE2EDiagnosticError("calibration fit missing")
    matrix = np.asarray(fit.get("matrix"), dtype=np.float64)
    bias = np.asarray(fit.get("bias"), dtype=np.float64)
    if matrix.shape != (30, 2) or bias.shape != (30,) or not np.isfinite(matrix).all() or not np.isfinite(bias).all():
        raise TinyE2EDiagnosticError("calibration fit identity invalid")
    return CalibrationResult(
        matrix=matrix.tolist(), bias=bias.tolist(),
        condition_number=float(fit["condition_number"]),
        pilot_reconstruction_mse=float(fit["pilot_reconstruction_mse"]),
    )


def run_condition_pipeline(
    off_r1_frames: np.ndarray,
    off_r2_frames: np.ndarray,
    condition_frames: np.ndarray,
    off_r1_features: np.ndarray,
    off_r2_features: np.ndarray,
    condition_features: np.ndarray,
) -> dict[str, Any]:
    """Run stages without receiving expected OFF/A/B truth."""

    presence = level_p_metrics(
        off_r1_frames.astype(np.float64) / 255.0,
        off_r2_frames.astype(np.float64) / 255.0,
        condition_frames.astype(np.float64) / 255.0,
    )
    output: dict[str, Any] = {"presence": presence, "downstream_executed": False, "paired_30D_sha256": None, "capture": None, "calibration": None, "phase": None}
    if not presence["off_repeat_control_valid"] or not presence["cell_pass"]:
        return output
    paired = np.asarray(condition_features, dtype=np.float64) - 0.5 * (
        np.asarray(off_r1_features, dtype=np.float64) + np.asarray(off_r2_features, dtype=np.float64)
    )
    if paired.shape != (13, 30) or not np.isfinite(paired).all():
        raise TinyE2EDiagnosticError("paired feature matrix invalid")
    candidates = capture_candidates_without_truth(paired)
    ranked = calibrate_capture_candidates(paired, candidates)
    unique = len(ranked) == 8 and ranked[0]["held_out_mse"] < ranked[1]["held_out_mse"]
    selected = [ranked[0]["capture_identity"]] if unique else []
    output.update({
        "downstream_executed": True,
        "paired_30D_sha256": hashlib.sha256(np.ascontiguousarray(paired).tobytes(order="C")).hexdigest(),
        "capture": {"candidate_universe": 16, "K": 8, "candidate_set": candidates, "candidate_set_sha256": capture_set_sha256(candidates)},
        "calibration": {"K2": 1, "unique_winner": unique, "selected_candidates": selected, "ranked_candidates": ranked},
    })
    if not unique:
        return output
    calibration = _calibration_from_ranked_winner(ranked[0])
    projected = equalize_observations(paired.tolist(), calibration, ridge=EQUALIZATION_RIDGE)
    predicted_template = str(selected[0]["template_id"])
    output["phase"] = {
        perturbation: evaluate_phase_cell(projected, predicted_template, perturbation)
        for perturbation in PERTURBATION_ORDER
    }
    return output


def _validate_inputs(repo_root: Path) -> tuple[dict[str, dict[str, Path]], dict[str, Any], dict[str, Any]]:
    config_path = repo_root / "configs/rc0_tiny_e2e_fast_cpu.json"
    protocol_path = repo_root / "protocols/rc0_tiny_e2e_fast_cpu.md"
    config = load_and_validate_config(config_path)
    audits: dict[str, Any] = {}
    audit_identity: dict[str, Any] = {}
    for name, (path, digest) in STEP_AUDITS.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise TinyE2EDiagnosticError(f"{name} audit identity mismatch")
        audits[name] = json.loads(path.read_text(encoding="utf-8"))
        audit_identity[name] = {"absolute_path": str(path), "sha256": digest}
    expected_outcomes = {
        "step12": ("CPU_DIAGNOSTIC_RESULT_READY",),
        "step3": ("CPU_DIAGNOSTIC_RESULT_READY", "FEASIBLE"),
        "step4": ("CPU_DIAGNOSTIC_RESULT_READY", "FEASIBLE"),
        "step5": ("CPU_DIAGNOSTIC_RESULT_READY", "FEASIBLE"),
    }
    actual_outcomes = {
        "step12": (audits["step12"].get("status"),),
        "step3": (audits["step3"].get("status"), audits["step3"].get("question3")),
        "step4": (audits["step4"].get("status"), audits["step4"].get("question4")),
        "step5": (audits["step5"].get("status"), audits["step5"].get("question5")),
    }
    if actual_outcomes != expected_outcomes:
        raise TinyE2EDiagnosticError("prior diagnostic outcome identity mismatch")
    paths: dict[str, dict[str, Path]] = {}
    videos: dict[str, Any] = {}
    for group in GROUP_ORDER:
        paths[group] = {}
        videos[group] = {}
        for condition in CONDITION_ORDER:
            path = VIDEO_ROOT / group / condition / "saved.mp4"
            digest = BOUND_ARTIFACT_SHA256[group][condition]
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise TinyE2EDiagnosticError(f"bound video identity mismatch for {group}:{condition}")
            paths[group][condition] = path
            videos[group][condition] = {"absolute_path": str(path), "sha256": digest, "size": path.stat().st_size, "stream": probe_mp4(path)}
    return paths, audits, {"audits": audit_identity, "videos": videos, "config_sha256": sha256_file(config_path), "protocol_sha256": sha256_file(protocol_path), "config": config}


def _stage_identity_checks(outputs: Mapping[str, Any], audits: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for group in GROUP_ORDER:
        for condition in ("A", "B"):
            key = f"{group}:{condition}"
            output = outputs[key]
            comparisons: dict[str, Any] = {}
            prior_p = audits["step12"]["evaluation"]["p_cells"][key]
            comparisons["presence"] = _compare_recursive(output["presence"], prior_p, f"{key}.presence")
            prior_capture = audits["step3"]["cells"][f"{key}:identity"]["candidate_set"]
            comparisons["capture"] = _compare_recursive(output["capture"]["candidate_set"], prior_capture, f"{key}.capture")
            prior_ranked = audits["step4"]["cells"][f"{key}:identity"]["ranked_candidates"]
            comparisons["calibration"] = _compare_recursive(output["calibration"]["ranked_candidates"], prior_ranked, f"{key}.calibration")
            for perturbation in PERTURBATION_ORDER:
                prior_phase = audits["step5"]["cells"][f"{key}:{perturbation}"]
                comparisons[f"phase:{perturbation}"] = _compare_recursive(output["phase"][perturbation], prior_phase, f"{key}.phase.{perturbation}")
            mismatches = [path for _, paths in comparisons.values() for path in paths]
            checks[key] = {"all_match": not mismatches, "maximum_absolute_difference": max((difference for difference, _ in comparisons.values()), default=0.0), "mismatch_paths": mismatches}
    return checks


def _truth_audit(outputs: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    truth: dict[str, Any] = {}
    first_failure: str | None = None
    for group in GROUP_ORDER:
        for condition in CONDITION_ORDER:
            key = f"{group}:{condition}"
            output = outputs[key]
            expected_presence = condition in {"A", "B"}
            presence_correct = bool(output["presence"]["cell_pass"]) == expected_presence
            short_circuit_correct = output["downstream_executed"] == expected_presence
            selected_correct = False
            phase_correct = False
            if expected_presence and output["calibration"] is not None:
                selected = output["calibration"]["selected_candidates"]
                selected_correct = len(selected) == 1 and selected[0]["start_index"] == 0 and selected[0]["template_id"] == condition
                phase_correct = output["phase"] is not None and all(cell["passed"] for cell in output["phase"].values())
            elif not expected_presence:
                selected_correct = output["calibration"] is None
                phase_correct = output["phase"] is None
            passed = presence_correct and short_circuit_correct and selected_correct and phase_correct
            if first_failure is None and not presence_correct:
                first_failure = "presence"
            elif first_failure is None and not short_circuit_correct:
                first_failure = "capture"
            elif first_failure is None and not selected_correct:
                first_failure = "calibration"
            elif first_failure is None and not phase_correct:
                first_failure = "phase"
            truth[key] = {"expected_presence": expected_presence, "presence_correct": presence_correct, "short_circuit_correct": short_circuit_correct, "selected_correct": selected_correct, "phase_correct": phase_correct, "passed": passed}
    return truth, first_failure


def run_step6_once(*, repo_root: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if RUN_ROOT.exists() or RUN_ROOT.is_symlink():
        raise TinyE2EDiagnosticError("Step6 run root already exists")
    if not RUN_ROOT.parent.is_dir() or RUN_ROOT.parent.is_symlink():
        raise TinyE2EDiagnosticError("Step6 run parent unavailable")
    started_at = utc_now()
    paths, audits, input_identity = _validate_inputs(repo_root)
    decoded = {group: {condition: decode_saved_mp4(paths[group][condition]) for condition in CONDITION_ORDER} for group in GROUP_ORDER}
    features = {group: {condition: np.asarray(extract_feature_matrix(decoded[group][condition]), dtype=np.float64) for condition in CONDITION_ORDER} for group in GROUP_ORDER}
    outputs: dict[str, Any] = {}
    for group in GROUP_ORDER:
        for condition in CONDITION_ORDER:
            outputs[f"{group}:{condition}"] = run_condition_pipeline(
                decoded[group]["OFF_R1"], decoded[group]["OFF_R2"], decoded[group][condition],
                features[group]["OFF_R1"], features[group]["OFF_R2"], features[group][condition],
            )
    stage_identity = _stage_identity_checks(outputs, audits)
    if not all(check["all_match"] for check in stage_identity.values()):
        raise TinyE2EDiagnosticError("recomputed stage output differs from frozen prior audit")
    truth, first_failure = _truth_audit(outputs)
    feasible = len(truth) == 8 and all(item["passed"] for item in truth.values())
    status = "CPU_DIAGNOSTIC_RESULT_READY" if feasible else "TINY_E2E_NOT_FEASIBLE"
    question6 = "FEASIBLE" if feasible else "NOT_FEASIBLE"
    overall = "FEASIBLE_FOR_TINY_MATCHED_CONTROL_DIAGNOSTIC" if feasible else "NOT_FEASIBLE_FOR_TINY_MATCHED_CONTROL_DIAGNOSTIC"
    route = "METHOD_FEASIBILITY_CORE_CLOSED" if feasible else "TINY_E2E_NOT_FEASIBLE"
    result = {
        "schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status,
        "question6": question6, "overall": overall, "route": route,
        "first_failed_stage": first_failure, "condition_outputs": outputs,
        "truth_audit_after_outputs_frozen": truth, "prior_stage_identity": stage_identity,
        "original_condition_count": len(outputs),
        "phase_cell_count": sum(len(output["phase"] or {}) for output in outputs.values()),
        "remaining_unanswered": ["blind_public_deployment", "robustness"],
        "input_identity": input_identity,
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
    return {"status": status, "diagnostic_class": DIAGNOSTIC_CLASS, "actual_run_root": str(RUN_ROOT), "question6": question6, "overall": overall, "route": route, "condition_count": len(outputs), "phase_cell_count": result["phase_cell_count"]}
