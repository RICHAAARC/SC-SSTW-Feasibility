"""Frozen Step-3 AISB candidate-capture diagnostic over eight existing MP4s."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from .aisb import BurstCandidate, BurstTemplate, affine_burst_residual, scan_burst_candidates
from .learned_observation import decode_saved_mp4, extract_feature_matrix
from .rc0_causal_localization_v2 import schedule_a, schedule_b
from .rc0_pixel_chroma_fast_cpu import probe_mp4


SCHEMA = "sc_sstw_rc0_aisb_capture_fast_cpu_v1"
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
GROUP_ORDER = ("orbital_glass", "articulated_paper")
CONDITION_ORDER = ("OFF_R1", "OFF_R2", "A", "B")
PERTURBATION_ORDER = ("identity", "private_tail_delete6_duplicate12")
START_INDICES = tuple(range(8))
TOP_K_PER_START = 1
CANDIDATE_BUDGET_K = 8
INPUT_CLOSURE_COMMIT = "dff72dadb57e5e732f27b22a6497afcac67da850"
INPUT_CLOSURE_TREE = "b32f14cda2deb0793f28060049631bb45f2ee370"
INPUT_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-pixel-chroma-fast-cpu-9e19d31-run1")
INPUT_AUDIT = INPUT_ROOT / "evaluation-only-result/audit.json"
INPUT_RESULT = INPUT_ROOT / "evaluation-only-result/result.json"
INPUT_AUDIT_SHA256 = "9b08c88aad5378c2b0c94574585906e96a9608d094002248fddd6761b5e1dead"
RUN_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-aisb-capture-fast-cpu-dff72da-run1")
BOUND_ARTIFACT_SHA256 = {
    "orbital_glass": {
        "OFF_R1": "1459ce027d94e1446883cebad82eaad6cc3f004be56601fe4dcd6e4149d82a57",
        "OFF_R2": "1459ce027d94e1446883cebad82eaad6cc3f004be56601fe4dcd6e4149d82a57",
        "A": "a0c8f23d86923a375da0d5fd9740c74ed0bb24642653cc3b51abf3004cb90e98",
        "B": "2c9f5813c34729f18b2b81d0f7de694120a13afae0a172ee9336cd0824f281da",
    },
    "articulated_paper": {
        "OFF_R1": "1e9e210f3afda91fe592e61dcb5f1e01167facef13edbd244610b9c8693b3ba9",
        "OFF_R2": "1e9e210f3afda91fe592e61dcb5f1e01167facef13edbd244610b9c8693b3ba9",
        "A": "a9f62f7c5ca0aa762f164a599aac763b46e8a2b143085b0c1cfbaffd1ca4cd30",
        "B": "caa2138ca9e45742e776cba7eb1a4dbdd0c314e96fcd2187d1a5f57b7b097b9c",
    },
}


class AISBDiagnosticError(RuntimeError):
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
        raise AISBDiagnosticError("Step3 config identity mismatch")
    capture = config.get("capture", {})
    if capture.get("top_k_per_start") != TOP_K_PER_START or capture.get("candidate_budget_K") != CANDIDATE_BUDGET_K:
        raise AISBDiagnosticError("candidate budget changed")
    if capture.get("start_indices") != list(START_INDICES) or capture.get("acceptance_threshold") is not None:
        raise AISBDiagnosticError("capture starts or threshold changed")
    if capture.get("templates") != ["A_first_six_points", "B_first_six_points"]:
        raise AISBDiagnosticError("capture templates changed")
    if tuple(item.get("id") for item in config.get("time_perturbations", [])) != PERTURBATION_ORDER:
        raise AISBDiagnosticError("time perturbation family changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise AISBDiagnosticError("diagnostic boundary changed")
    return config


def _templates() -> tuple[BurstTemplate, BurstTemplate]:
    return (
        BurstTemplate("A", tuple(schedule_a()[:6])),
        BurstTemplate("B", tuple(schedule_b()[:6])),
    )


def perturb_observation(observation: Any, perturbation: str) -> np.ndarray:
    array = np.asarray(observation, dtype=np.float64)
    if array.shape != (13, 30) or not np.isfinite(array).all():
        raise AISBDiagnosticError("paired observation must be finite 13x30")
    if perturbation == "identity":
        return array.copy()
    if perturbation == "private_tail_delete6_duplicate12":
        return np.concatenate((array[:6], array[7:], array[12:13]), axis=0)
    raise AISBDiagnosticError("unknown time perturbation")


def _candidate_record(candidate: BurstCandidate, *, full_rank: int, selected_rank: int) -> dict[str, Any]:
    return {
        "start_index": candidate.start_index,
        "template_id": candidate.template_id,
        "residual": candidate.residual,
        "observed_length": candidate.observed_length,
        "missing_template_index": candidate.missing_template_index,
        "full_rank": full_rank,
        "selected_rank": selected_rank,
    }


def capture_cell(observation: Any, own_template: str, perturbation: str) -> dict[str, Any]:
    if own_template not in {"A", "B"}:
        raise AISBDiagnosticError("own template must be A or B")
    edited = perturb_observation(observation, perturbation)
    templates = _templates()
    selected = scan_burst_candidates(
        edited.tolist(), templates, top_k_per_start=TOP_K_PER_START,
        allow_single_deletion=False, allow_double_deletion=False,
    )
    if len(selected) != CANDIDATE_BUDGET_K or tuple(sorted(candidate.start_index for candidate in selected)) != START_INDICES:
        raise AISBDiagnosticError("finite candidate budget or start coverage changed")
    template_order = {"A": 0, "B": 1}
    full: list[BurstCandidate] = []
    for start in START_INDICES:
        for template in templates:
            full.append(BurstCandidate(
                start_index=start,
                template_id=template.template_id,
                residual=affine_burst_residual(edited[start : start + 6].tolist(), template),
                observed_length=6,
                missing_template_index=None,
            ))
    sort_key = lambda candidate: (candidate.residual, candidate.start_index, template_order[candidate.template_id])
    full_sorted = sorted(full, key=sort_key)
    selected_sorted = sorted(selected, key=sort_key)
    full_rank = {(item.start_index, item.template_id): index for index, item in enumerate(full_sorted, start=1)}
    selected_rank = {(item.start_index, item.template_id): index for index, item in enumerate(selected_sorted, start=1)}
    records = [
        _candidate_record(
            candidate,
            full_rank=full_rank[(candidate.start_index, candidate.template_id)],
            selected_rank=selected_rank[(candidate.start_index, candidate.template_id)],
        )
        for candidate in selected_sorted
    ]
    target_key = (0, own_template)
    target = next((record for record in records if (record["start_index"], record["template_id"]) == target_key), None)
    cross_template = "B" if own_template == "A" else "A"
    target_unpruned = next(item for item in full_sorted if (item.start_index, item.template_id) == target_key)
    cross_unpruned = next(item for item in full_sorted if (item.start_index, item.template_id) == (0, cross_template))
    return {
        "own_template": own_template,
        "perturbation": perturbation,
        "candidate_budget_K": CANDIDATE_BUDGET_K,
        "top_k_per_start": TOP_K_PER_START,
        "candidate_set": records,
        "target": target,
        "start0_competition": {
            "own_template": own_template,
            "own_residual": target_unpruned.residual,
            "cross_template": cross_template,
            "cross_residual": cross_unpruned.residual,
            "own_beats_cross": target_unpruned.residual < cross_unpruned.residual,
        },
        "target_captured": target is not None and target_unpruned.residual < cross_unpruned.residual,
    }


def _validate_inputs(repo_root: Path) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    config = load_and_validate_config(repo_root / "configs/rc0_aisb_capture_fast_cpu.json")
    tree = subprocess.run(
        ["git", "rev-parse", f"{INPUT_CLOSURE_COMMIT}^{{tree}}"], cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if tree != INPUT_CLOSURE_TREE:
        raise AISBDiagnosticError("input closure commit/tree mismatch")
    for path in (INPUT_AUDIT, INPUT_RESULT):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != INPUT_AUDIT_SHA256:
            raise AISBDiagnosticError("input evaluation audit/result identity mismatch")
    audit = json.loads(INPUT_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "CPU_DIAGNOSTIC_RESULT_READY":
        raise AISBDiagnosticError("input evaluation is not ready")
    evaluation = audit.get("evaluation", {})
    if (evaluation.get("step1"), evaluation.get("step2"), evaluation.get("route")) != (
        "FEASIBLE", "FEASIBLE", "KEEP_CARRIER_BUILD_BLIND_READOUT",
    ):
        raise AISBDiagnosticError("input Step1/Step2 identity mismatch")
    paths: dict[str, dict[str, Path]] = {}
    identities: dict[str, Any] = {}
    for group in GROUP_ORDER:
        paths[group] = {}
        identities[group] = {}
        for condition in CONDITION_ORDER:
            path = INPUT_ROOT / "output/videos" / group / condition / "saved.mp4"
            expected_sha = BOUND_ARTIFACT_SHA256[group][condition]
            if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
                raise AISBDiagnosticError(f"bound video identity mismatch for {group}:{condition}")
            stream = probe_mp4(path)
            audit_item = audit.get("artifacts", {}).get(group, {}).get(condition, {})
            if audit_item.get("sha256") != expected_sha or audit_item.get("absolute_path") != str(path):
                raise AISBDiagnosticError(f"audit/video binding mismatch for {group}:{condition}")
            paths[group][condition] = path
            identities[group][condition] = {
                "absolute_path": str(path), "sha256": expected_sha,
                "size": path.stat().st_size, "stream": stream,
            }
    return paths, {
        "closure_commit": INPUT_CLOSURE_COMMIT,
        "closure_tree": INPUT_CLOSURE_TREE,
        "evaluation_audit_path": str(INPUT_AUDIT),
        "evaluation_audit_sha256": INPUT_AUDIT_SHA256,
        "config_sha256": sha256_file(repo_root / "configs/rc0_aisb_capture_fast_cpu.json"),
        "protocol_sha256": sha256_file(repo_root / "protocols/rc0_aisb_capture_fast_cpu.md"),
        "videos": identities,
        "config": config,
    }


def run_step3_once(*, repo_root: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if RUN_ROOT.exists() or RUN_ROOT.is_symlink():
        raise AISBDiagnosticError("Step3 run root already exists")
    if not RUN_ROOT.parent.is_dir() or RUN_ROOT.parent.is_symlink():
        raise AISBDiagnosticError("Step3 run parent is unavailable")
    started_at = utc_now()
    paths, input_identity = _validate_inputs(repo_root)
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
                cells[key] = capture_cell(paired, condition, perturbation)
    feasible = len(cells) == 8 and all(bool(cell["target_captured"]) for cell in cells.values())
    question3 = "FEASIBLE" if feasible else "NOT_FEASIBLE"
    route = "PROCEED_STEP4_SELF_CALIBRATION_CPU" if feasible else "AISB_CAPTURE_NOT_FEASIBLE"
    result = {
        "schema": SCHEMA,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "status": "CPU_DIAGNOSTIC_RESULT_READY" if feasible else "AISB_CAPTURE_NOT_FEASIBLE",
        "question3": question3,
        "route": route,
        "formula": "affine_burst_residual_v1",
        "candidate_budget_K": CANDIDATE_BUDGET_K,
        "top_k_per_start": TOP_K_PER_START,
        "perturbations": list(PERTURBATION_ORDER),
        "cells": cells,
        "input_identity": input_identity,
        "generation_was_run": False,
        "encoding_was_run": False,
        "training_was_run": False,
        "formal_result": False,
        "stage_progression_allowed": False,
        "started_at": started_at,
        "ended_at": utc_now(),
    }
    RUN_ROOT.mkdir(mode=0o755)
    (RUN_ROOT / "result.json").write_bytes(canonical_json_bytes(result) + b"\n")
    (RUN_ROOT / "audit.json").write_bytes(canonical_json_bytes(result) + b"\n")
    command = {
        "argv": list(argv), "cwd": str(cwd.resolve()), "started_at": started_at,
        "ended_at": result["ended_at"], "exit_code": 0,
    }
    (RUN_ROOT / "command.json").write_bytes(canonical_json_bytes(command) + b"\n")
    checksum_paths = sorted(path for path in RUN_ROOT.iterdir() if path.is_file())
    (RUN_ROOT / "checksums.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in checksum_paths) + "\n",
        encoding="utf-8",
    )
    return {
        "status": result["status"], "diagnostic_class": DIAGNOSTIC_CLASS,
        "actual_run_root": str(RUN_ROOT), "question3": question3,
        "route": route, "candidate_budget_K": CANDIDATE_BUDGET_K,
    }
