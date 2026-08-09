"""One-shot frozen Phase-C effectiveness diagnostic over four ZIP-bound bases."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from .calibration import equalize_observations
from .learned_observation import decode_saved_mp4
from .rc0_causal_localization_v2 import schedule_a, schedule_b
from .rc0_phase_recovery_fast_cpu import (
    EQUALIZATION_RIDGE,
    MAX_EDIT_BUDGET,
    PERTURBATION_ORDER,
    REPEAT_PENALTY,
    SCORE_TOLERANCE,
    SKIP_PENALTY,
    expected_operation_path,
    recover_complete_path,
)
from .rc0_pixel_chroma_fast_cpu import (
    VIDEO_SHAPE,
    apply_carrier,
    encode_mp4,
    probe_mp4,
)
from .rc0_target_only_blind_fast_cpu import (
    _winner_calibration,
    calibrate_candidates_without_truth,
    capture_without_truth,
    target_only_observation,
)
from .sync import dynamic_time_sync, dynamic_time_sync_score_bounded


SCHEMA = "sc_sstw_rc0_fresh4_effectiveness_cpu_v1"
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
GROUP_ORDER = ("liquid_mosaic", "clockwork_escapement", "steam_fins", "magnetic_filings")
CONDITION_ORDER = ("OFF_R1", "OFF_R2", "A", "B")
TRANSFORM_ORDER = ("identity", "delete6_duplicate12", "local_phase_plus1", "local_phase_minus1")
SOURCE_ZIP = Path("/mnt/g/我的云端硬盘/SC-SSTW-Feasibility/rc0-fresh-exact4-base/rc0-fresh4-0471c4ae283025ad.zip")
SOURCE_ZIP_SIZE = 146503
SOURCE_ZIP_SHA256 = "6076a9645c3ef064ef387c0460305fbbc340bce78049262419124c50d6aa1878"
SOURCE_RUN_ID = "0471c4ae283025ad"
SOURCE_COMMIT = "add1e966e9bf8778a502e40532558185d5f64d3a"
SOURCE_TREE = "0b48705929038f9769adf3ba3b53af1b29d1fded"
FROZEN_CONFIG_SHA256 = "913b761f848940d0f17ddb71a8424e91d5dad0eb094a81eaaff163df06fb1618"
FROZEN_PROTOCOL_SHA256 = "f563f7dc83c5d3bc3391280968557ab3ee8282503fb0902d5d849af6146af34e"
BASE_MEMBERS = {
    "liquid_mosaic": ("output/bases/liquid_mosaic/base.mp4", 20940, "3b6e71777dc2f87e6a8108ab727c666f2103ff9b308513bfa258402492916a56"),
    "clockwork_escapement": ("output/bases/clockwork_escapement/base.mp4", 44369, "b2d4fdaaed8da388b366b3dae36ab850c177770da68a9638999b7be34733dd7c"),
    "steam_fins": ("output/bases/steam_fins/base.mp4", 22698, "f371fb003ff71f6366060b51f630effea4770bef19f9f20e5d83e002b1340283"),
    "magnetic_filings": ("output/bases/magnetic_filings/base.mp4", 50868, "e91db82b6be2a6334773131b30003ea071c1cfcd9dd0f69aa98f55ec00aaecb8"),
}
RUN_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-fresh4-effectiveness-cpu-run1")


class FreshEffectivenessError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def load_frozen_phase_c(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = repo_root / "configs/rc0_fresh_exact4_base_gpu.json"
    protocol_path = repo_root / "protocols/rc0_fresh_exact4_base_gpu.md"
    if sha256_file(config_path) != FROZEN_CONFIG_SHA256 or sha256_file(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise FreshEffectivenessError("frozen Phase-C config or protocol changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    phase_c = config.get("phase_c_frozen_not_executed")
    if not isinstance(phase_c, dict):
        raise FreshEffectivenessError("frozen Phase-C plan missing")
    transforms = phase_c.get("transforms")
    if not isinstance(transforms, list) or tuple(item.get("id") for item in transforms) != TRANSFORM_ORDER:
        raise FreshEffectivenessError("frozen transform order changed")
    for item in transforms:
        if len(item.get("symbol_source_indices", [])) != 13 or len(item.get("frame_source_indices", [])) != 49:
            raise FreshEffectivenessError("frozen transform index budget changed")
    expected = (list(CONDITION_ORDER), 6.0 / 255.0, ["A", "B"], "frozen_target_only_blind_observation_e0744c8")
    actual = (phase_c.get("condition_order"), phase_c.get("pixel_chroma_relative_amplitude"), phase_c.get("schedules"), phase_c.get("detector"))
    if actual != expected:
        raise FreshEffectivenessError("frozen condition/carrier/detector identity changed")
    return config, phase_c


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise FreshEffectivenessError("could not materialize ZIP member in anonymous memory")
        view = view[written:]
    os.lseek(fd, 0, os.SEEK_SET)


def probe_and_decode_mp4_bytes(data: bytes) -> tuple[dict[str, Any], np.ndarray]:
    """Probe/decode a ZIP member through a seekable anonymous memory file."""

    fd = os.memfd_create("rc0-fresh-base", flags=0)
    try:
        _write_all(fd, data)
        source = f"/proc/self/fd/{fd}"
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames", "-of", "json", source],
            check=True, capture_output=True, text=True, pass_fds=(fd,),
        )
        streams = json.loads(probe.stdout).get("streams", [])
        expected = {"codec_name": "h264", "pix_fmt": "yuv420p", "width": 512, "height": 320, "avg_frame_rate": "8/1", "nb_frames": "49"}
        if len(streams) != 1 or streams[0] != expected:
            raise FreshEffectivenessError("base MP4 stream identity changed")
        os.lseek(fd, 0, os.SEEK_SET)
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", source, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            check=True, capture_output=True, pass_fds=(fd,),
        ).stdout
        expected_bytes = int(np.prod(VIDEO_SHAPE))
        if len(decoded) != expected_bytes:
            raise FreshEffectivenessError("base MP4 decoded byte count changed")
        frames = np.frombuffer(decoded, dtype=np.uint8).reshape(VIDEO_SHAPE).copy()
        return streams[0], frames
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise FreshEffectivenessError("base MP4 anonymous-memory probe/decode failed") from exc
    finally:
        os.close(fd)


def read_bound_bases(repo_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if SOURCE_ZIP.is_symlink() or not SOURCE_ZIP.is_file() or SOURCE_ZIP.stat().st_size != SOURCE_ZIP_SIZE or sha256_file(SOURCE_ZIP) != SOURCE_ZIP_SHA256:
        raise FreshEffectivenessError("frozen fresh4 ZIP identity mismatch")
    config, _phase_c = load_frozen_phase_c(repo_root)
    frames: dict[str, np.ndarray] = {}
    identities: dict[str, Any] = {}
    with zipfile.ZipFile(SOURCE_ZIP, "r") as archive:
        if archive.testzip() is not None:
            raise FreshEffectivenessError("fresh4 ZIP CRC failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise FreshEffectivenessError("fresh4 ZIP contains duplicate member names")
        audit = json.loads(archive.read("output/audit.json"))
        embedded_config = json.loads(archive.read("output/config.json"))
        state = json.loads(archive.read("notebook_state.json"))
        if canonical_json_bytes(embedded_config) != canonical_json_bytes(config):
            raise FreshEffectivenessError("embedded config differs from frozen config")
        if (
            audit.get("status"), audit.get("source"), audit.get("config_sha256"), audit.get("protocol_sha256"),
            audit.get("attempts_started"), audit.get("attempts_completed"), audit.get("retry_count"),
        ) != (
            "BASE_EXACT4_READY", {"head": SOURCE_COMMIT, "tree": SOURCE_TREE, "dirty": False}, FROZEN_CONFIG_SHA256,
            FROZEN_PROTOCOL_SHA256, list(GROUP_ORDER), list(GROUP_ORDER), 0,
        ):
            raise FreshEffectivenessError("embedded fresh4 audit identity changed")
        if state.get("run_id") != SOURCE_RUN_ID or state.get("authorized_ref") != SOURCE_COMMIT or state.get("runner_return_code") != 0:
            raise FreshEffectivenessError("notebook run identity changed")
        checksums = archive.read("output/checksums.sha256").decode("utf-8").splitlines()
        checksum_map = {relative: digest for digest, relative in (line.split("  ", 1) for line in checksums)}
        for group in GROUP_ORDER:
            member, expected_size, expected_sha = BASE_MEMBERS[group]
            if names.count(member) != 1:
                raise FreshEffectivenessError(f"base member count mismatch for {group}")
            info = archive.getinfo(member)
            mode = info.external_attr >> 16
            data = archive.read(member)
            if info.is_dir() or stat.S_ISLNK(mode) or len(data) != expected_size or sha256_bytes(data) != expected_sha:
                raise FreshEffectivenessError(f"base member identity mismatch for {group}")
            relative = member.removeprefix("output/")
            if checksum_map.get(relative) != expected_sha or audit.get("artifacts", {}).get(group, {}).get("sha256") != expected_sha:
                raise FreshEffectivenessError(f"base checksum/audit binding mismatch for {group}")
            stream, decoded = probe_and_decode_mp4_bytes(data)
            frames[group] = decoded
            identities[group] = {"member": member, "size": expected_size, "sha256": expected_sha, "stream": stream,
                                 "decoded_array_sha256": sha256_bytes(np.ascontiguousarray(decoded).tobytes(order="C"))}
    return frames, {"zip": {"absolute_path": str(SOURCE_ZIP), "size": SOURCE_ZIP_SIZE, "sha256": SOURCE_ZIP_SHA256},
                    "run_id": SOURCE_RUN_ID, "source_commit": SOURCE_COMMIT, "source_tree": SOURCE_TREE, "bases": identities}


def _derive_condition_frames(base: np.ndarray, condition: str) -> tuple[np.ndarray, dict[str, Any]]:
    if condition in {"OFF_R1", "OFF_R2"}:
        return base.copy(), {"residual": "zero", "target_rms": 0.0}
    if condition == "A":
        return apply_carrier(base, schedule_a())
    if condition == "B":
        return apply_carrier(base, schedule_b())
    raise FreshEffectivenessError("unknown condition")


def apply_frozen_frame_transform(frames: np.ndarray, phase_c: Mapping[str, Any], transform_id: str) -> np.ndarray:
    if frames.shape != VIDEO_SHAPE or frames.dtype != np.uint8:
        raise FreshEffectivenessError("transform input must be uint8 49x320x512x3")
    records = {item["id"]: item for item in phase_c["transforms"]}
    if transform_id not in records:
        raise FreshEffectivenessError("unknown frozen MP4 transform")
    indices = records[transform_id]["frame_source_indices"]
    if len(indices) != 49 or min(indices) < 0 or max(indices) >= 49:
        raise FreshEffectivenessError("frozen frame transform indices invalid")
    return frames[np.asarray(indices, dtype=np.int64)].copy()


def _phase_already_transformed(states: Any, selected_template: str, transform_id: str) -> dict[str, Any]:
    observed = [tuple(map(float, point)) for point in np.asarray(states, dtype=np.float64)]
    if len(observed) != 13 or any(len(point) != 2 or not all(math.isfinite(value) for value in point) for point in observed):
        raise FreshEffectivenessError("transformed projected trajectory must be finite 13x2")
    own = [tuple(map(float, point)) for point in (schedule_a() if selected_template == "A" else schedule_b())]
    cross_id = "B" if selected_template == "A" else "A"
    cross = [tuple(map(float, point)) for point in (schedule_b() if selected_template == "A" else schedule_a())]
    primary = dynamic_time_sync(observed, own, skip_penalty=SKIP_PENALTY, repeat_penalty=REPEAT_PENALTY)
    cross_primary = dynamic_time_sync(observed, cross, skip_penalty=SKIP_PENALTY, repeat_penalty=REPEAT_PENALTY)
    complete = recover_complete_path(observed, own)
    bounded_score, bounded_abandoned = dynamic_time_sync_score_bounded(observed, own, min_score_to_beat=-math.inf, skip_penalty=SKIP_PENALTY, repeat_penalty=REPEAT_PENALTY)
    expected = expected_operation_path(transform_id)
    expected_diagonal = [[item["observed_index"], item["template_index"]] for item in expected if item["operation"] == "MATCH"]
    expected_skip = sum(item["operation"] == "SKIP_TEMPLATE" for item in expected)
    expected_repeat = sum(item["operation"] == "REPEAT_TEMPLATE" for item in expected)
    primary_path = [list(pair) for pair in primary.path]
    score_consistent = abs(primary.score - complete["score"]) <= SCORE_TOLERANCE and abs(primary.score - bounded_score) <= SCORE_TOLERANCE
    path_exact = complete["operations"] == expected and primary_path == expected_diagonal
    edit_exact = complete["skip_count"] == expected_skip and complete["repeat_count"] == expected_repeat and complete["edit_count"] <= MAX_EDIT_BUDGET
    own_beats_cross = primary.score > cross_primary.score
    passed = path_exact and edit_exact and score_consistent and not bounded_abandoned and own_beats_cross
    return {"selected_template": selected_template, "cross_template": cross_id, "transform_id": transform_id,
            "expected_operations": expected, "recovered_operations": complete["operations"], "primary_diagonal_path": primary_path,
            "expected_diagonal_path": expected_diagonal, "skip_count": complete["skip_count"], "repeat_count": complete["repeat_count"],
            "edit_count": complete["edit_count"], "maximum_edit_budget": MAX_EDIT_BUDGET,
            "own_score": primary.score, "cross_score": cross_primary.score, "own_beats_cross": own_beats_cross,
            "bounded_score_crosscheck": bounded_score, "bounded_abandoned": bounded_abandoned,
            "complete_backtracker_score": complete["score"], "score_consistent": score_consistent,
            "path_exact": path_exact, "edit_exact": edit_exact, "passed": passed}


def evaluate_transformed_target(decoded_frames: np.ndarray, transform_id: str) -> dict[str, Any]:
    """Evaluate one target without receiving OFF/A/B truth."""

    observation = target_only_observation(decoded_frames)
    capture = capture_without_truth(observation)
    output: dict[str, Any] = {"observation": observation.tolist(), "presence": capture, "downstream_executed": False,
                              "calibration": None, "phase": None}
    if not capture["presence_pass"]:
        return output
    ranked = calibrate_candidates_without_truth(observation, capture["candidate_set"])
    unique = ranked[0]["held_out_mse"] < ranked[1]["held_out_mse"]
    selected = [ranked[0]["capture_identity"]] if unique else []
    output["downstream_executed"] = True
    output["calibration"] = {"K2": 1, "unique_winner": unique, "selected_candidates": selected, "ranked_candidates": ranked}
    if not unique:
        return output
    states = equalize_observations(observation.tolist(), _winner_calibration(ranked[0]), ridge=EQUALIZATION_RIDGE)
    output["phase"] = _phase_already_transformed(states, selected[0]["template_id"], transform_id)
    return output


def audit_target(output: Mapping[str, Any], condition: str, *, transformed: bool) -> tuple[dict[str, Any], str | None]:
    positive = condition in {"A", "B"}
    presence_correct = bool(output["presence"]["presence_pass"]) == positive
    short_circuit_correct = bool(output["downstream_executed"]) == positive
    selected_correct = output["calibration"] is None
    phase_correct = output["phase"] is None
    if positive and output["calibration"] is not None:
        selected = output["calibration"]["selected_candidates"]
        selected_correct = len(selected) == 1 and selected[0]["start_index"] == 0 and selected[0]["template_id"] == condition
        if transformed:
            phase_correct = output["phase"] is not None and bool(output["phase"]["passed"])
        else:
            phase_correct = output["phase"] is not None and all(bool(item["passed"]) for item in output["phase"].values())
    passed = presence_correct and short_circuit_correct and selected_correct and phase_correct
    first = None
    if not presence_correct:
        first = "presence"
    elif not short_circuit_correct or not selected_correct:
        first = "capture_calibration"
    elif not phase_correct:
        first = "phase"
    return {"expected_presence": positive, "presence_correct": presence_correct, "short_circuit_correct": short_circuit_correct,
            "selected_correct": selected_correct, "phase_correct": phase_correct, "passed": passed}, first


def run_phase_c_once(*, repo_root: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if RUN_ROOT.exists() or RUN_ROOT.is_symlink():
        raise FreshEffectivenessError("Phase-C run root already exists")
    if not RUN_ROOT.parent.is_dir() or RUN_ROOT.parent.is_symlink():
        raise FreshEffectivenessError("Phase-C run parent unavailable")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise FreshEffectivenessError("Phase-C requires a clean checkout")
    config, phase_c = load_frozen_phase_c(repo_root)
    RUN_ROOT.mkdir(mode=0o755)
    started_at = utc_now()
    condition_attempts: list[dict[str, Any]] = []
    transform_attempts: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    condition_outputs: dict[str, Any] = {}
    transform_outputs: dict[str, Any] = {}
    condition_truth: dict[str, Any] = {}
    transform_truth: dict[str, Any] = {}
    first_failure: dict[str, Any] | None = None
    try:
        bases, input_identity = read_bound_bases(repo_root)
        for group in GROUP_ORDER:
            artifacts[group] = {"conditions": {}, "transforms": {}}
            base = bases[group]
            for condition in CONDITION_ORDER:
                key = f"{group}:{condition}"
                condition_attempts.append({"index": len(condition_attempts) + 1, "group": group, "condition": condition,
                                           "retry_index": 0, "started": True, "completed": False})
                encoded_frames, carrier_identity = _derive_condition_frames(base, condition)
                condition_path = RUN_ROOT / "conditions" / group / condition / "saved.mp4"
                encode_mp4(encoded_frames, condition_path)
                condition_stream = probe_mp4(condition_path)
                condition_decoded = decode_saved_mp4(condition_path)
                condition_attempts[-1]["completed"] = True
                condition_attempts[-1]["sha256"] = sha256_file(condition_path)
                clean_output = evaluate_transformed_target(condition_decoded, "identity")
                if clean_output["phase"] is not None:
                    selected = clean_output["calibration"]["selected_candidates"]
                    states = equalize_observations(clean_output["observation"], _winner_calibration(clean_output["calibration"]["ranked_candidates"][0]), ridge=EQUALIZATION_RIDGE)
                    clean_output["phase"] = {
                        name: _phase_already_transformed(states, selected[0]["template_id"], name)
                        for name in PERTURBATION_ORDER
                    }
                condition_outputs[key] = clean_output
                condition_truth[key], stage = audit_target(clean_output, condition, transformed=False)
                if first_failure is None and stage is not None:
                    first_failure = {"kind": "condition", "group": group, "condition": condition, "stage": stage}
                artifacts[group]["conditions"][condition] = {"path": str(condition_path.relative_to(RUN_ROOT)),
                    "sha256": sha256_file(condition_path), "size": condition_path.stat().st_size,
                    "stream": condition_stream, "carrier_identity": carrier_identity}
                artifacts[group]["transforms"][condition] = {}
                for transform_id in TRANSFORM_ORDER:
                    transform_attempts.append({"index": len(transform_attempts) + 1, "group": group, "condition": condition,
                                               "transform": transform_id, "retry_index": 0, "started": True, "completed": False})
                    transformed_frames = apply_frozen_frame_transform(condition_decoded, phase_c, transform_id)
                    transformed_path = RUN_ROOT / "transforms" / group / condition / transform_id / "saved.mp4"
                    encode_mp4(transformed_frames, transformed_path)
                    transformed_stream = probe_mp4(transformed_path)
                    transformed_decoded = decode_saved_mp4(transformed_path)
                    transform_attempts[-1]["completed"] = True
                    transform_attempts[-1]["sha256"] = sha256_file(transformed_path)
                    transformed_output = evaluate_transformed_target(transformed_decoded, transform_id)
                    transformed_key = f"{group}:{condition}:{transform_id}"
                    transform_outputs[transformed_key] = transformed_output
                    transform_truth[transformed_key], stage = audit_target(transformed_output, condition, transformed=True)
                    if first_failure is None and stage is not None:
                        first_failure = {"kind": "transform", "group": group, "condition": condition, "transform": transform_id, "stage": stage}
                    artifacts[group]["transforms"][condition][transform_id] = {"path": str(transformed_path.relative_to(RUN_ROOT)),
                        "sha256": sha256_file(transformed_path), "size": transformed_path.stat().st_size, "stream": transformed_stream,
                        "frame_source_indices": next(item["frame_source_indices"] for item in phase_c["transforms"] if item["id"] == transform_id)}
        feasible = all(item["passed"] for item in condition_truth.values()) and all(item["passed"] for item in transform_truth.values())
        status = "EFFECTIVE_ON_FRESH_SMALL_SAMPLE" if feasible else "FRESH_SMALL_SAMPLE_NOT_FEASIBLE"
        phase_c_result = "FEASIBLE" if feasible else "NOT_FEASIBLE"
        route = "METHOD_EFFECTIVENESS_SMALL_SAMPLE_CLOSED" if feasible else "FRESH_SMALL_SAMPLE_NOT_FEASIBLE"
        result = {"schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status, "phase_c": phase_c_result,
                  "route": route, "first_failure": first_failure, "input_identity": input_identity,
                  "source_code": {"head": _git(repo_root, "rev-parse", "HEAD"), "tree": _git(repo_root, "rev-parse", "HEAD^{tree}"), "dirty": False},
                  "config_sha256": FROZEN_CONFIG_SHA256, "protocol_sha256": FROZEN_PROTOCOL_SHA256,
                  "condition_attempts_started": len(condition_attempts), "condition_attempts_completed": sum(item["completed"] for item in condition_attempts),
                  "transform_attempts_started": len(transform_attempts), "transform_attempts_completed": sum(item["completed"] for item in transform_attempts),
                  "retry_count": 0, "condition_attempts": condition_attempts, "transform_attempts": transform_attempts,
                  "artifacts": artifacts, "condition_outputs": condition_outputs, "transform_outputs": transform_outputs,
                  "condition_truth_after_outputs_frozen": condition_truth, "transform_truth_after_outputs_frozen": transform_truth,
                  "condition_count": len(condition_truth), "transform_cell_count": len(transform_truth),
                  "generation_was_run": False, "GPU_was_run": False, "formal_result": False, "stage_progression_allowed": False,
                  "started_at": started_at, "ended_at": utc_now()}
        (RUN_ROOT / "audit.json").write_bytes(canonical_json_bytes(result) + b"\n")
        (RUN_ROOT / "result.json").write_bytes(canonical_json_bytes(result) + b"\n")
        response = {"status": status, "diagnostic_class": DIAGNOSTIC_CLASS, "phase_c": phase_c_result, "route": route,
                    "first_failure": first_failure, "actual_run_root": str(RUN_ROOT), "condition_count": len(condition_truth),
                    "transform_cell_count": len(transform_truth), "retry_count": 0}
        (RUN_ROOT / "command.json").write_bytes(canonical_json_bytes({"argv": list(argv), "cwd": str(cwd.resolve()), "exit_code": 0 if feasible else 3}) + b"\n")
        checksum_files = sorted(path for path in RUN_ROOT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
        (RUN_ROOT / "checksums.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.relative_to(RUN_ROOT).as_posix()}" for path in checksum_files) + "\n", encoding="utf-8")
        return response
    except BaseException as exc:
        failure = {"schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": "INSUFFICIENT",
                   "phase_c": "INSUFFICIENT_TO_DECIDE", "route": "INSUFFICIENT", "reason": type(exc).__name__,
                   "message": str(exc), "condition_attempts": condition_attempts, "transform_attempts": transform_attempts,
                   "science_outputs_present": bool(condition_outputs or transform_outputs), "formal_result": False,
                   "stage_progression_allowed": False, "ended_at": utc_now()}
        (RUN_ROOT / "audit.json").write_bytes(canonical_json_bytes(failure) + b"\n")
        raise
