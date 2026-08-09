"""One-shot CPU pixel-chroma diagnostic built from two frozen OFF_R1 MP4s."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from .learned_observation import decode_saved_mp4, extract_feature_matrix
from .rc0_causal_localization_v2 import level_p_metrics, level_r_metrics, schedule_a, schedule_b


DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
SCHEMA = "sc_sstw_rc0_pixel_chroma_fast_cpu_v1"
GROUP_ORDER = ("orbital_glass", "articulated_paper")
CONDITION_ORDER = ("OFF_R1", "OFF_R2", "A", "B")
ATTEMPT_ORDER = tuple((group, condition) for group in GROUP_ORDER for condition in CONDITION_ORDER)
VIDEO_SHAPE = (49, 320, 512, 3)
FRAME_POINT_INDICES = (0,) + tuple(point for point in range(1, 13) for _ in range(4))
TARGET_RMS = 6.0 / 255.0
SOURCE_ZIP_SHA256 = "76ddc314132dfc208ebc6dbc04bdbe77ca5da731b16c0345eb139a661c378a4f"
SOURCE_ZIP_SIZE = 397216
SOURCE_ZIP_PATH = Path("/mnt/g/我的云端硬盘/SC-SSTW-Feasibility/rc0-diag-vae-latent-inference-mode/rc0-dfa1c782d409fa02.zip")
SOURCE_MEMBERS = {
    "orbital_glass": {
        "path": "actual-package/generation/orbital_glass/OFF_R1/saved.mp4",
        "size": 25630,
        "sha256": "2eb2f33e8b90ec9b8a4e3079425439abc3c7984d667b68a7b7b4f6ec59860e40",
    },
    "articulated_paper": {
        "path": "actual-package/generation/articulated_paper/OFF_R1/saved.mp4",
        "size": 42053,
        "sha256": "1f99673db08015a5afcf26c985d5c8d84601735b7c14f9b1ca40ff549ba2a276",
    },
}


class DiagnosticError(RuntimeError):
    """The frozen diagnostic could not produce interpretable evidence."""


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


def load_and_validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or config.get("diagnostic_class") != DIAGNOSTIC_CLASS:
        raise DiagnosticError("diagnostic config identity mismatch")
    if config.get("source") != {
        "run_id": "dfa1c782d409fa02",
        "zip_path": str(SOURCE_ZIP_PATH),
        "zip_size": SOURCE_ZIP_SIZE,
        "zip_sha256": SOURCE_ZIP_SHA256,
        "members": SOURCE_MEMBERS,
    }:
        raise DiagnosticError("frozen source identity changed")
    experiment = config.get("experiment", {})
    if experiment != {
        "group_order": list(GROUP_ORDER),
        "condition_order": list(CONDITION_ORDER),
        "attempt_budget": 8,
        "retry_policy": "no_retry_no_replacement_no_additional_attempts",
        "source_frame_policy": "one_exact_decoded_OFF_R1_frame_array_reused_for_all_four_conditions_in_group",
    }:
        raise DiagnosticError("experiment order or budget changed")
    carrier = config.get("carrier", {})
    expected_carrier = {
        "kind": "post_decode_pre_h264_pixel_chroma_relation_residual",
        "input": "normalized_RGB_float32_shape_49x320x512x3",
        "phi_x": "cos(2*pi*x/512)",
        "phi_y": "cos(2*pi*y/320)",
        "u": [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0],
        "v": [1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0)],
        "frame_point_indices": list(FRAME_POINT_INDICES),
        "schedule_A_source": "sc_sstw_feasibility.rc0_causal_localization_v2.schedule_a",
        "schedule_B_source": "sc_sstw_feasibility.rc0_causal_localization_v2.schedule_b",
        "raw_formula": "schedule_x_times_phi_x_times_u_plus_schedule_y_times_phi_y_times_v",
        "normalization": "subtract_one_global_mean_then_divide_by_one_global_RMS_over_entire_clip",
        "target_rms": TARGET_RMS,
        "target_rms_expression": "6/255",
        "clip_range": [0.0, 1.0],
        "off_residual": 0.0,
        "dtype": "float32",
    }
    if carrier != expected_carrier:
        raise DiagnosticError("frozen pixel-chroma carrier changed")
    encoding = config.get("encoding", {})
    if encoding != {
        "call": "imageio.v3.imwrite", "plugin": "FFMPEG", "container": "mp4",
        "codec": "libx264", "required_codec_name": "h264", "pixel_format": "yuv420p",
        "width": 512, "height": 320, "frame_count": 49, "fps": 8,
        "quality": 5.0, "macro_block_size": 16,
    }:
        raise DiagnosticError("encoding identity changed")
    level_p = config.get("level_p", {})
    if level_p.get("absolute_effect_min") != 2.0 / 255.0 or level_p.get("relative_noise_multiplier") != 3.0 or level_p.get("off_noise_max") != 1.0 / 255.0:
        raise DiagnosticError("Level P thresholds changed")
    level_r = config.get("level_r", {})
    if level_r.get("maximum_relation_residual") != 0.25 or level_r.get("start_indices") != list(range(8)):
        raise DiagnosticError("Level R thresholds or budget changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise DiagnosticError("diagnostic boundary changed")
    return config


def validate_source_zip(zip_path: Path = SOURCE_ZIP_PATH) -> dict[str, Any]:
    if zip_path.is_symlink() or not zip_path.is_file():
        raise DiagnosticError("frozen source ZIP is unavailable or not a regular file")
    size = zip_path.stat().st_size
    digest = sha256_file(zip_path)
    if size != SOURCE_ZIP_SIZE or digest != SOURCE_ZIP_SHA256:
        raise DiagnosticError("frozen source ZIP identity mismatch")
    members: dict[str, Any] = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        for group in GROUP_ORDER:
            expected = SOURCE_MEMBERS[group]
            member = expected["path"]
            if names.count(member) != 1:
                raise DiagnosticError(f"source member count mismatch for {group}")
            info = archive.getinfo(member)
            data = archive.read(member)
            if info.is_dir() or info.file_size != expected["size"] or len(data) != expected["size"] or sha256_bytes(data) != expected["sha256"]:
                raise DiagnosticError(f"source member identity mismatch for {group}")
            members[group] = {"path": member, "size": len(data), "sha256": sha256_bytes(data)}
    return {"path": str(zip_path.resolve()), "size": size, "sha256": digest, "members": members}


def decode_source_member(zip_path: Path, group: str, temp_dir: Path) -> np.ndarray:
    if group not in GROUP_ORDER:
        raise DiagnosticError("unexpected source group")
    expected = SOURCE_MEMBERS[group]
    with zipfile.ZipFile(zip_path, "r") as archive:
        data = archive.read(expected["path"])
    if len(data) != expected["size"] or sha256_bytes(data) != expected["sha256"]:
        raise DiagnosticError(f"source member changed before decode for {group}")
    path = temp_dir / f"{group}.source.mp4"
    path.write_bytes(data)
    frames = decode_saved_mp4(path)
    if frames.shape != VIDEO_SHAPE or frames.dtype != np.uint8:
        raise DiagnosticError(f"decoded source identity mismatch for {group}")
    return frames


def construct_unit_residual(schedule: Sequence[Sequence[float]]) -> tuple[np.ndarray, dict[str, float]]:
    points = np.asarray(schedule, dtype=np.float32)
    if points.shape != (13, 2) or not np.isfinite(points).all():
        raise DiagnosticError("schedule must be finite 13x2")
    x = np.arange(512, dtype=np.float32)
    y = np.arange(320, dtype=np.float32)
    phi_x = np.cos(np.float32(2.0 * math.pi) * x / np.float32(512.0))
    phi_y = np.cos(np.float32(2.0 * math.pi) * y / np.float32(320.0))
    u = np.asarray((1.0, -1.0, 0.0), dtype=np.float32) / np.float32(math.sqrt(2.0))
    v = np.asarray((1.0, 1.0, -2.0), dtype=np.float32) / np.float32(math.sqrt(6.0))
    basis_x = phi_x[None, :, None] * u[None, None, :]
    basis_y = phi_y[:, None, None] * v[None, None, :]
    residual = np.empty(VIDEO_SHAPE, dtype=np.float32)
    for frame, point_index in enumerate(FRAME_POINT_INDICES):
        residual[frame] = points[point_index, 0] * basis_x + points[point_index, 1] * basis_y
    raw_mean = float(np.mean(residual, dtype=np.float64))
    residual -= np.float32(raw_mean)
    raw_rms = float(np.sqrt(np.mean(np.square(residual, dtype=np.float32), dtype=np.float64)))
    if not math.isfinite(raw_rms) or raw_rms <= 1e-12:
        raise DiagnosticError("pixel residual is degenerate")
    residual /= np.float32(raw_rms)
    return residual, {
        "raw_global_mean_before_centering": raw_mean,
        "centered_global_rms_before_unit_normalization": raw_rms,
        "unit_global_mean": float(np.mean(residual, dtype=np.float64)),
        "unit_global_rms": float(np.sqrt(np.mean(np.square(residual, dtype=np.float32), dtype=np.float64))),
    }


def apply_carrier(source_frames: np.ndarray, schedule: Sequence[Sequence[float]]) -> tuple[np.ndarray, dict[str, float]]:
    if source_frames.shape != VIDEO_SHAPE or source_frames.dtype != np.uint8:
        raise DiagnosticError("carrier source must be uint8 49x320x512x3")
    unit, identity = construct_unit_residual(schedule)
    source = source_frames.astype(np.float32) / np.float32(255.0)
    preclip = source + unit * np.float32(TARGET_RMS)
    clipped = np.clip(preclip, 0.0, 1.0)
    actual = clipped - source
    encoded_input = np.rint(clipped * np.float32(255.0)).astype(np.uint8)
    identity.update({
        "preclip_target_residual_rms": TARGET_RMS,
        "postclip_actual_residual_rms": float(np.sqrt(np.mean(np.square(actual, dtype=np.float32), dtype=np.float64))),
        "clipped_element_fraction": float(np.mean((preclip < 0.0) | (preclip > 1.0))),
    })
    return encoded_input, identity


def encode_mp4(frames: np.ndarray, path: Path) -> None:
    if frames.shape != VIDEO_SHAPE or frames.dtype != np.uint8:
        raise DiagnosticError("encoder input must be uint8 49x320x512x3")
    try:
        import imageio
        import imageio_ffmpeg
        import imageio.v3 as iio
    except Exception as exc:
        raise DiagnosticError("locked imageio encoder dependencies unavailable") from exc
    if (imageio.__version__, imageio_ffmpeg.__version__) != ("2.37.0", "0.6.0"):
        raise DiagnosticError("encoder dependency version mismatch")
    path.parent.mkdir(parents=True, exist_ok=False)
    iio.imwrite(path, frames, plugin="FFMPEG", fps=8, codec="libx264", pixelformat="yuv420p", quality=5.0, macro_block_size=16)
    if not path.is_file() or path.is_symlink():
        raise DiagnosticError("encoder did not create a regular MP4")


def probe_mp4(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise DiagnosticError("saved MP4 must contain one video stream")
    stream = streams[0]
    expected = {"codec_name": "h264", "pix_fmt": "yuv420p", "width": 512, "height": 320, "avg_frame_rate": "8/1", "nb_frames": "49"}
    if stream != expected:
        raise DiagnosticError(f"saved MP4 stream identity mismatch: {stream}")
    return stream


def _sha256_array(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def evaluate_saved_videos(video_paths: Mapping[str, Mapping[str, Path]]) -> dict[str, Any]:
    decoded: dict[str, dict[str, np.ndarray]] = {}
    for group in GROUP_ORDER:
        decoded[group] = {condition: decode_saved_mp4(video_paths[group][condition]) for condition in CONDITION_ORDER}
    p_cells: dict[str, Any] = {}
    off_noise_valid = True
    for group in GROUP_ORDER:
        for condition in ("A", "B"):
            metrics = level_p_metrics(
                decoded[group]["OFF_R1"].astype(np.float64) / 255.0,
                decoded[group]["OFF_R2"].astype(np.float64) / 255.0,
                decoded[group][condition].astype(np.float64) / 255.0,
            )
            p_cells[f"{group}:{condition}"] = metrics
            off_noise_valid = off_noise_valid and bool(metrics["off_repeat_control_valid"])
    if not off_noise_valid:
        return {
            "step1": "INSUFFICIENT_TO_DECIDE", "step2": "INSUFFICIENT_TO_DECIDE",
            "route": "DIAGNOSTIC_INSUFFICIENT", "next_question": "repeat_only_the_minimum_invalid_OFF_control",
            "p_cells": p_cells, "r_executed": False, "r_cells": None,
        }
    if not all(bool(cell["cell_pass"]) for cell in p_cells.values()):
        return {
            "step1": "NOT_FEASIBLE", "step2": "INSUFFICIENT_TO_DECIDE",
            "route": "CURRENT_CARRIER_NOT_FEASIBLE", "next_question": "stop_current_pixel_chroma_carrier",
            "p_cells": p_cells, "r_executed": False, "r_cells": None,
        }
    features = {
        group: {condition: extract_feature_matrix(decoded[group][condition]) for condition in CONDITION_ORDER}
        for group in GROUP_ORDER
    }
    r_cells: dict[str, Any] = {}
    for group in GROUP_ORDER:
        for condition in ("A", "B"):
            r_cells[f"{group}:{condition}"] = level_r_metrics(
                features[group]["OFF_R1"], features[group]["OFF_R2"], features[group][condition], condition
            )
    if all(bool(cell["cell_pass"]) for cell in r_cells.values()):
        return {
            "step1": "FEASIBLE", "step2": "FEASIBLE", "route": "KEEP_CARRIER_BUILD_BLIND_READOUT",
            "next_question": "STEP3_AISB_CPU", "p_cells": p_cells, "r_executed": True, "r_cells": r_cells,
        }
    return {
        "step1": "FEASIBLE", "step2": "NOT_FEASIBLE", "route": "KEEP_CARRIER_SWITCH_TO_VAE_READOUT",
        "next_question": "prepare_minimum_VAE_reencode_GPU_diagnostic_separately",
        "p_cells": p_cells, "r_executed": True, "r_cells": r_cells,
    }


def run_once(*, repo_root: Path, output: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise DiagnosticError("output path already exists")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise DiagnosticError("output parent must be an existing regular directory")
    config_path = repo_root / "configs/rc0_pixel_chroma_fast_cpu.json"
    config = load_and_validate_config(config_path)
    source_identity = validate_source_zip()
    output.mkdir(mode=0o755)
    started_at = utc_now()
    attempts: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    video_paths: dict[str, dict[str, Path]] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="rc0-pixel-source-", dir=str(output.parent)) as temporary:
            temp_dir = Path(temporary)
            for group in GROUP_ORDER:
                source_frames = decode_source_member(SOURCE_ZIP_PATH, group, temp_dir)
                source_array_sha = _sha256_array(source_frames)
                video_paths[group] = {}
                artifacts[group] = {}
                for group_name, condition in ATTEMPT_ORDER:
                    if group_name != group:
                        continue
                    attempt = {
                        "attempt_index": len(attempts) + 1, "group": group, "condition": condition,
                        "retry_index": 0, "started": True, "completed": False,
                        "source_frame_array_sha256": source_array_sha,
                    }
                    attempts.append(attempt)
                    if condition.startswith("OFF_"):
                        encoded_frames = source_frames
                        carrier_identity: dict[str, Any] = {"residual": "zero", "preclip_target_residual_rms": 0.0, "postclip_actual_residual_rms": 0.0}
                    else:
                        encoded_frames, carrier_identity = apply_carrier(source_frames, schedule_a() if condition == "A" else schedule_b())
                    path = output / "videos" / group / condition / "saved.mp4"
                    encode_mp4(encoded_frames, path)
                    stream = probe_mp4(path)
                    decoded = decode_saved_mp4(path)
                    attempt["completed"] = True
                    attempt["saved_mp4_sha256"] = sha256_file(path)
                    artifacts[group][condition] = {
                        "path": str(path.relative_to(output)), "size": path.stat().st_size,
                        "sha256": sha256_file(path), "stream": stream,
                        "decoded_array_sha256": _sha256_array(decoded), "carrier_identity": carrier_identity,
                    }
        if len(attempts) != 8 or not all(item["completed"] and item["retry_index"] == 0 for item in attempts):
            raise DiagnosticError("exact eight-attempt execution was not completed")
        evaluation = evaluate_saved_videos(video_paths)
        result = {
            "schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS,
            "status": "CPU_DIAGNOSTIC_COMPLETE", "formal_result": False, "stage_progression_allowed": False,
            "source_identity": source_identity, "config_sha256": sha256_file(config_path),
            "started_at": started_at, "ended_at": utc_now(), "attempts_started": 8, "attempts_completed": 8,
            "retry_count": 0, "attempts": attempts, "artifacts": artifacts, "evaluation": evaluation,
        }
        (output / "config.json").write_bytes(canonical_json_bytes(config) + b"\n")
        (output / "source_identity.json").write_bytes(canonical_json_bytes(source_identity) + b"\n")
        (output / "audit.json").write_bytes(canonical_json_bytes(result) + b"\n")
        response = {
            "status": result["status"], "diagnostic_class": DIAGNOSTIC_CLASS,
            "actual_package_path": str(output.resolve()), "step1": evaluation["step1"],
            "step2": evaluation["step2"], "route": evaluation["route"],
        }
        command = {
            "argv": list(argv), "cwd": str(cwd.resolve()), "started_at": started_at,
            "ended_at": result["ended_at"], "exit_code": 0,
        }
        (output / "command.json").write_bytes(canonical_json_bytes(command) + b"\n")
        (output / "invocation.stdout").write_bytes(canonical_json_bytes(response) + b"\n")
        (output / "invocation.stderr").write_bytes(b"")
        checksum_paths = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "checksums.sha256")
        lines = [f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in checksum_paths]
        (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return response
    except BaseException:
        # The caller owns the one-shot stop rule. Preserve partial evidence; do not retry.
        raise


def verify_checksums(output: Path) -> bool:
    lines = (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, relative = line.split("  ", 1)
        path = output / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            return False
    return True
