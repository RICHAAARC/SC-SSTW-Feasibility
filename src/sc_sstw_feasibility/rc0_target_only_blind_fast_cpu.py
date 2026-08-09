"""Target-only blind observation diagnostic over the frozen pixel-chroma MP4s."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .aisb import BurstCandidate, BurstTemplate, affine_burst_residual, scan_burst_candidates
from .calibration import CalibrationResult, calibrate_from_pilot_pairs, equalize_observations
from .learned_observation import decode_saved_mp4
from .rc0_aisb_capture_fast_cpu import BOUND_ARTIFACT_SHA256, CONDITION_ORDER, GROUP_ORDER
from .rc0_causal_localization_v2 import schedule_a, schedule_b
from .rc0_phase_recovery_fast_cpu import EQUALIZATION_RIDGE, PERTURBATION_ORDER, evaluate_phase_cell
from .rc0_pixel_chroma_fast_cpu import probe_mp4


SCHEMA = "sc_sstw_rc0_target_only_blind_fast_cpu_v1"
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
HEIGHT = 320
WIDTH = 512
FRAME_COUNT = 49
FRAME_GROUPS = ((0,),) + tuple(tuple(range(1 + 4 * index, 5 + 4 * index)) for index in range(12))
PROJECTION_EPSILON = 1e-15
ABSOLUTE_PRESENCE_THRESHOLD = 2.0 / 255.0
RELATION_RESIDUAL_THRESHOLD = 0.25
START_INDICES = tuple(range(8))
CANDIDATE_BUDGET_K = 8
K2 = 1
FIT_INDICES = (0, 1, 2, 3)
HELD_OUT_INDICES = (4, 5)
VIDEO_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-pixel-chroma-fast-cpu-9e19d31-run1/output/videos")
STEP6_AUDIT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-tiny-e2e-fast-cpu-071d054-run2/audit.json")
STEP6_AUDIT_SHA256 = "fce19cfecfb9dd81de5979a472fd2a047892f35e208dfb7c790030d5384926e0"
RUN_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-target-only-blind-fast-cpu-run1")


class TargetOnlyDiagnosticError(RuntimeError):
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
        raise TargetOnlyDiagnosticError("target-only config identity mismatch")
    input_config = config.get("input", {})
    if input_config.get("groups") != list(GROUP_ORDER) or input_config.get("conditions") != list(CONDITION_ORDER):
        raise TargetOnlyDiagnosticError("input group/condition identity changed")
    if input_config.get("shape") != [FRAME_COUNT, HEIGHT, WIDTH, 3] or input_config.get("off_reference") != "forbidden":
        raise TargetOnlyDiagnosticError("decode shape or target-only boundary changed")
    observation = config.get("observation", {})
    if observation.get("denominator_epsilon") != PROJECTION_EPSILON or observation.get("frame_groups") != [list(group) for group in FRAME_GROUPS]:
        raise TargetOnlyDiagnosticError("projection epsilon or frame mapping changed")
    presence = config.get("presence", {})
    if presence.get("absolute_lower_bound") != ABSOLUTE_PRESENCE_THRESHOLD or presence.get("relation_upper_bound") != RELATION_RESIDUAL_THRESHOLD:
        raise TargetOnlyDiagnosticError("presence threshold changed")
    downstream = config.get("downstream", {})
    expected = (16, CANDIDATE_BUDGET_K, list(FIT_INDICES), list(HELD_OUT_INDICES), EQUALIZATION_RIDGE, K2, list(PERTURBATION_ORDER))
    actual = (
        downstream.get("capture_universe"), downstream.get("capture_K"), downstream.get("calibration_fit_indices"),
        downstream.get("calibration_held_out_indices"), downstream.get("calibration_equalization_ridge"),
        downstream.get("calibration_K2"), downstream.get("phase_perturbations"),
    )
    if actual != expected:
        raise TargetOnlyDiagnosticError("downstream frozen parameter changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise TargetOnlyDiagnosticError("diagnostic boundary changed")
    return config


def _spatial_bases() -> tuple[np.ndarray, np.ndarray]:
    phi_x = np.cos(2.0 * math.pi * np.arange(WIDTH, dtype=np.float64) / WIDTH)
    phi_y = np.cos(2.0 * math.pi * np.arange(HEIGHT, dtype=np.float64) / HEIGHT)
    u = np.asarray((1.0, -1.0, 0.0), dtype=np.float64) / math.sqrt(2.0)
    v = np.asarray((1.0, 1.0, -2.0), dtype=np.float64) / math.sqrt(6.0)
    basis_x = np.broadcast_to(phi_x[None, :, None] * u[None, None, :], (HEIGHT, WIDTH, 3))
    basis_y = np.broadcast_to(phi_y[:, None, None] * v[None, None, :], (HEIGHT, WIDTH, 3))
    return basis_x, basis_y


def project_centered_frame(normalized_rgb: Any) -> tuple[float, float]:
    frame = np.asarray(normalized_rgb, dtype=np.float64)
    if frame.shape != (HEIGHT, WIDTH, 3) or not np.isfinite(frame).all() or frame.min() < 0.0 or frame.max() > 1.0:
        raise TargetOnlyDiagnosticError("normalized target frame must be finite [320,512,3] in [0,1]")
    centered = frame - float(frame.mean())
    basis_x, basis_y = _spatial_bases()
    qx = float(np.sum(centered * basis_x) / (np.sum(basis_x * basis_x) + PROJECTION_EPSILON))
    qy = float(np.sum(centered * basis_y) / (np.sum(basis_y * basis_y) + PROJECTION_EPSILON))
    if not (math.isfinite(qx) and math.isfinite(qy)):
        raise TargetOnlyDiagnosticError("target-only frame projection is non-finite")
    return qx, qy


def aggregate_frame_projections(frame_projections: Any) -> np.ndarray:
    array = np.asarray(frame_projections, dtype=np.float64)
    if array.shape != (FRAME_COUNT, 2) or not np.isfinite(array).all():
        raise TargetOnlyDiagnosticError("frame projections must be finite 49x2")
    return np.asarray([array[list(group)].mean(axis=0) for group in FRAME_GROUPS], dtype=np.float64)


def target_only_observation(decoded_frames: Any) -> np.ndarray:
    frames = np.asarray(decoded_frames)
    if frames.shape != (FRAME_COUNT, HEIGHT, WIDTH, 3) or frames.dtype != np.uint8:
        raise TargetOnlyDiagnosticError("decoded target MP4 must be RGB24 uint8 [49,320,512,3]")
    frame_projections = np.asarray([project_centered_frame(frame.astype(np.float64) / 255.0) for frame in frames], dtype=np.float64)
    observation = aggregate_frame_projections(frame_projections)
    if observation.shape != (13, 2) or not np.isfinite(observation).all():
        raise TargetOnlyDiagnosticError("target-only observation must be finite 13x2")
    return observation


def _templates() -> tuple[BurstTemplate, BurstTemplate]:
    return BurstTemplate("A", tuple(schedule_a()[:6])), BurstTemplate("B", tuple(schedule_b()[:6]))


def _candidate_record(candidate: BurstCandidate, selected_rank: int) -> dict[str, Any]:
    return {
        "start_index": candidate.start_index, "template_id": candidate.template_id,
        "residual": candidate.residual, "observed_length": candidate.observed_length,
        "missing_template_index": candidate.missing_template_index, "selected_rank": selected_rank,
    }


def capture_without_truth(observation: Any) -> dict[str, Any]:
    array = np.asarray(observation, dtype=np.float64)
    if array.shape != (13, 2) or not np.isfinite(array).all():
        raise TargetOnlyDiagnosticError("capture observation must be finite 13x2")
    templates = _templates()
    selected = scan_burst_candidates(array.tolist(), templates, top_k_per_start=1)
    if len(selected) != CANDIDATE_BUDGET_K or tuple(sorted(item.start_index for item in selected)) != START_INDICES:
        raise TargetOnlyDiagnosticError("capture budget or start coverage changed")
    template_order = {"A": 0, "B": 1}
    selected = sorted(selected, key=lambda item: (item.residual, item.start_index, template_order[item.template_id]))
    records = [_candidate_record(item, rank) for rank, item in enumerate(selected, start=1)]
    full_residuals = [
        affine_burst_residual(array[start : start + 6].tolist(), template)
        for start in START_INDICES for template in templates
    ]
    centered = array - array.mean(axis=0, keepdims=True)
    energy = float(np.sqrt(np.mean(centered * centered)))
    minimum = float(min(full_residuals))
    energy_pass = energy >= ABSOLUTE_PRESENCE_THRESHOLD
    relation_pass = minimum <= RELATION_RESIDUAL_THRESHOLD
    return {
        "candidate_universe": 16, "K": CANDIDATE_BUDGET_K, "candidate_set": records,
        "trajectory_centered_rms": energy, "absolute_threshold": ABSOLUTE_PRESENCE_THRESHOLD,
        "minimum_affine_residual": minimum, "residual_threshold": RELATION_RESIDUAL_THRESHOLD,
        "energy_pass": energy_pass, "relation_pass": relation_pass, "presence_pass": energy_pass and relation_pass,
    }


def _template_points(template_id: str) -> tuple[tuple[float, float], ...]:
    if template_id == "A":
        return tuple(schedule_a()[:6])
    if template_id == "B":
        return tuple(schedule_b()[:6])
    raise TargetOnlyDiagnosticError("candidate template must be A or B")


def calibrate_candidates_without_truth(observation: Any, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    array = np.asarray(observation, dtype=np.float64)
    if array.shape != (13, 2) or not np.isfinite(array).all() or len(candidates) != CANDIDATE_BUDGET_K:
        raise TargetOnlyDiagnosticError("calibration input identity invalid")
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        start = int(candidate["start_index"])
        template_id = str(candidate["template_id"])
        template = _template_points(template_id)
        window = array[start : start + 6]
        if window.shape != (6, 2):
            raise TargetOnlyDiagnosticError("candidate window incomplete")
        calibration = calibrate_from_pilot_pairs([(template[index], window[index].tolist()) for index in FIT_INDICES])
        equalized = equalize_observations([window[index].tolist() for index in HELD_OUT_INDICES], calibration, ridge=EQUALIZATION_RIDGE)
        score = sum((equalized[offset][dimension] - template[index][dimension]) ** 2 for offset, index in enumerate(HELD_OUT_INDICES) for dimension in range(2)) / 4.0
        if not math.isfinite(score):
            raise TargetOnlyDiagnosticError("candidate calibration score is non-finite")
        records.append({
            "capture_identity": dict(candidate),
            "fit": {"matrix": calibration.matrix, "bias": calibration.bias, "condition_number": calibration.condition_number,
                    "pilot_reconstruction_mse": calibration.pilot_reconstruction_mse, "fit_indices": list(FIT_INDICES),
                    "held_out_indices": list(HELD_OUT_INDICES), "equalization_ridge": EQUALIZATION_RIDGE},
            "held_out_equalized": [list(point) for point in equalized], "held_out_mse": score,
        })
    template_order = {"A": 0, "B": 1}
    records.sort(key=lambda item: (item["held_out_mse"], item["capture_identity"]["residual"], item["capture_identity"]["start_index"], template_order[item["capture_identity"]["template_id"]]))
    for rank, record in enumerate(records, start=1):
        record["calibration_rank"] = rank
    return records


def _winner_calibration(winner: Mapping[str, Any]) -> CalibrationResult:
    fit = winner["fit"]
    return CalibrationResult(matrix=fit["matrix"], bias=fit["bias"], condition_number=float(fit["condition_number"]), pilot_reconstruction_mse=float(fit["pilot_reconstruction_mse"]))


def run_target_pipeline(observation: Any) -> dict[str, Any]:
    """Run without receiving condition truth or any other video's observation."""

    array = np.asarray(observation, dtype=np.float64)
    capture = capture_without_truth(array)
    output: dict[str, Any] = {"observation": array.tolist(), "presence": capture, "downstream_executed": False, "calibration": None, "phase": None}
    if not capture["presence_pass"]:
        return output
    ranked = calibrate_candidates_without_truth(array, capture["candidate_set"])
    unique = len(ranked) == CANDIDATE_BUDGET_K and ranked[0]["held_out_mse"] < ranked[1]["held_out_mse"]
    selected = [ranked[0]["capture_identity"]] if unique else []
    output["downstream_executed"] = True
    output["calibration"] = {"K2": K2, "unique_winner": unique, "selected_candidates": selected, "ranked_candidates": ranked}
    if not unique:
        return output
    projected = equalize_observations(array.tolist(), _winner_calibration(ranked[0]), ridge=EQUALIZATION_RIDGE)
    output["phase"] = {name: evaluate_phase_cell(projected, selected[0]["template_id"], name) for name in PERTURBATION_ORDER}
    return output


def truth_audit_after_outputs(outputs: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    audit: dict[str, Any] = {}
    first_failure: str | None = None
    for group in GROUP_ORDER:
        for condition in CONDITION_ORDER:
            key = f"{group}:{condition}"
            output = outputs[key]
            positive = condition in {"A", "B"}
            presence_correct = bool(output["presence"]["presence_pass"]) == positive
            short_circuit_correct = bool(output["downstream_executed"]) == positive
            selected_correct = output["calibration"] is None
            phase_correct = output["phase"] is None
            if positive and output["calibration"] is not None:
                selected = output["calibration"]["selected_candidates"]
                selected_correct = len(selected) == 1 and selected[0]["start_index"] == 0 and selected[0]["template_id"] == condition
                phase_correct = output["phase"] is not None and all(item["passed"] for item in output["phase"].values())
            passed = presence_correct and short_circuit_correct and selected_correct and phase_correct
            for stage, value in (("presence", presence_correct), ("capture_calibration", short_circuit_correct and selected_correct), ("phase", phase_correct)):
                if first_failure is None and not value:
                    first_failure = stage
            audit[key] = {"expected_presence": positive, "presence_correct": presence_correct, "short_circuit_correct": short_circuit_correct,
                          "selected_correct": selected_correct, "phase_correct": phase_correct, "passed": passed}
    return audit, first_failure


def _validate_inputs(repo_root: Path) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    config_path = repo_root / "configs/rc0_target_only_blind_fast_cpu.json"
    protocol_path = repo_root / "protocols/rc0_target_only_blind_fast_cpu.md"
    config = load_and_validate_config(config_path)
    if STEP6_AUDIT.is_symlink() or not STEP6_AUDIT.is_file() or sha256_file(STEP6_AUDIT) != STEP6_AUDIT_SHA256:
        raise TargetOnlyDiagnosticError("Step6 audit identity mismatch")
    step6 = json.loads(STEP6_AUDIT.read_text(encoding="utf-8"))
    if (step6.get("status"), step6.get("question6"), step6.get("route")) != ("CPU_DIAGNOSTIC_RESULT_READY", "FEASIBLE", "METHOD_FEASIBILITY_CORE_CLOSED"):
        raise TargetOnlyDiagnosticError("Step6 outcome identity mismatch")
    paths: dict[str, dict[str, Path]] = {}
    identities: dict[str, Any] = {}
    for group in GROUP_ORDER:
        paths[group] = {}
        identities[group] = {}
        for condition in CONDITION_ORDER:
            path = VIDEO_ROOT / group / condition / "saved.mp4"
            digest = BOUND_ARTIFACT_SHA256[group][condition]
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise TargetOnlyDiagnosticError(f"bound video identity mismatch for {group}:{condition}")
            stream = probe_mp4(path)
            paths[group][condition] = path
            identities[group][condition] = {"absolute_path": str(path), "sha256": digest, "size": path.stat().st_size, "stream": stream}
    return paths, {"step6_audit_path": str(STEP6_AUDIT), "step6_audit_sha256": STEP6_AUDIT_SHA256,
                   "config_sha256": sha256_file(config_path), "protocol_sha256": sha256_file(protocol_path), "videos": identities, "config": config}


def run_phase_a_once(*, repo_root: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if RUN_ROOT.exists() or RUN_ROOT.is_symlink():
        raise TargetOnlyDiagnosticError("target-only run root already exists")
    if not RUN_ROOT.parent.is_dir() or RUN_ROOT.parent.is_symlink():
        raise TargetOnlyDiagnosticError("target-only run parent unavailable")
    started_at = utc_now()
    paths, input_identity = _validate_inputs(repo_root)
    outputs: dict[str, Any] = {}
    for group in GROUP_ORDER:
        for condition in CONDITION_ORDER:
            observation = target_only_observation(decode_saved_mp4(paths[group][condition]))
            outputs[f"{group}:{condition}"] = run_target_pipeline(observation)
    truth, first_failure = truth_audit_after_outputs(outputs)
    feasible = len(truth) == 8 and all(item["passed"] for item in truth.values())
    status = "TARGET_ONLY_BLIND_READOUT_FEASIBLE" if feasible else "BLIND_READOUT_NOT_FEASIBLE"
    phase_a = "FEASIBLE" if feasible else "NOT_FEASIBLE"
    result = {
        "schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status, "phase_a": phase_a,
        "route": "PROCEED_FRESH_EXACT4_BASE_NOTEBOOK" if feasible else "BLIND_READOUT_NOT_FEASIBLE",
        "first_failed_stage": first_failure, "condition_outputs": outputs, "truth_audit_after_outputs_frozen": truth,
        "input_identity": input_identity, "condition_count": len(outputs),
        "off_reference_used": False, "paired_difference_used": False, "training_was_run": False,
        "generation_was_run": False, "encoding_was_run": False,
        "formal_result": False, "stage_progression_allowed": False,
        "started_at": started_at, "ended_at": utc_now(),
    }
    RUN_ROOT.mkdir(mode=0o755)
    for name in ("result.json", "audit.json"):
        (RUN_ROOT / name).write_bytes(canonical_json_bytes(result) + b"\n")
    command = {"argv": list(argv), "cwd": str(cwd.resolve()), "started_at": started_at, "ended_at": result["ended_at"], "exit_code": 0 if feasible else 3}
    (RUN_ROOT / "command.json").write_bytes(canonical_json_bytes(command) + b"\n")
    checksum_paths = sorted(path for path in RUN_ROOT.iterdir() if path.is_file())
    (RUN_ROOT / "checksums.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in checksum_paths) + "\n", encoding="utf-8")
    return {"status": status, "diagnostic_class": DIAGNOSTIC_CLASS, "actual_run_root": str(RUN_ROOT), "phase_a": phase_a,
            "route": result["route"], "first_failed_stage": first_failure, "condition_count": len(outputs), "exit_code": command["exit_code"]}
