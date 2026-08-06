"""Real-Wan direct model-output carrier and saved-MP4 propagation probe.

This module deliberately stops at propagation diagnostics.  Paired clean versus
watermarked quantities are written only as diagnostic localization evidence;
the blind observation is computed from the single saved watermarked MP4.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from typing import Any, Sequence
from .aisb import affine_burst_residual, make_default_templates
from .linalg import condition_number_2d_columns



class CarrierPropagationError(RuntimeError):
    pass


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_id") != "sc_sstw_gpu_carrier_propagation_v1":
        raise CarrierPropagationError("unexpected carrier propagation protocol")
    carrier = config["carrier"]
    expected_shape = carrier["expected_tensor_shape"]
    points = carrier["temporal_points"]
    roles = carrier["temporal_point_roles"]
    groups = config["diagnostic_readouts"]["temporal_frame_groups"]
    if expected_shape != [1, 16, 13, 40, 64]:
        raise CarrierPropagationError("carrier shape must match audited real-Wan shape")
    if len(points) != expected_shape[2] or len(roles) != len(points) or len(groups) != len(points):
        raise CarrierPropagationError("points, roles, and frame groups must match 13 temporal tokens")
    if points[:6] != [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.25, 0.35], [0.75, 0.2], [0.2, 0.8]]:
        raise CarrierPropagationError("first six points must be the frozen burst_alpha AISB")
    flat_groups = [item for group in groups for item in group]
    if flat_groups != list(range(config["generation"]["frame_count"])):
        raise CarrierPropagationError("temporal frame groups must partition all saved frames in order")
    if carrier["kind"] != "direct_model_output_residual":
        raise CarrierPropagationError("only direct model-output residual is admitted")
    if carrier["scheduler_mutation_permitted"] is not False or carrier["in_place_mutation_permitted"] is not False:
        raise CarrierPropagationError("scheduler and in-place mutation must remain forbidden")
    ratio = float(carrier["target_relative_rms"])
    tolerance = float(carrier["target_relative_rms_absolute_tolerance"])
    if not (0.0 < ratio <= 0.03) or not (0.0 < tolerance <= 0.00005):
        raise CarrierPropagationError("energy constraint is outside the frozen admissible range")
    paired_thresholds = config["paired_diagnostic_thresholds"]
    if paired_thresholds != {"maximum_public_aisb_residual": 0.25, "maximum_public_condition_number": 10.0}:
        raise CarrierPropagationError("paired diagnostic thresholds must remain exactly frozen")

    if any(len(point) != 2 or not all(math.isfinite(float(value)) for value in point) for point in points):
        raise CarrierPropagationError("all temporal points must be finite and two-dimensional")


def build_analytic_basis(torch: Any, sample: Any) -> tuple[Any, Any]:
    """Build two public, key-independent, orthogonal low-frequency bases."""

    if list(sample.shape) != [1, 16, 13, 40, 64]:
        raise CarrierPropagationError(f"unexpected transformer output shape: {list(sample.shape)}")
    _batch, channels, temporal, height, width = sample.shape
    x = (torch.arange(width, device=sample.device, dtype=torch.float32) + 0.5) / float(width)
    y = (torch.arange(height, device=sample.device, dtype=torch.float32) + 0.5) / float(height)
    scale = math.sqrt(2.0)
    bx = scale * torch.cos(2.0 * math.pi * x).reshape(1, 1, 1, 1, width)
    by = scale * torch.cos(2.0 * math.pi * y).reshape(1, 1, 1, height, 1)
    bx = bx.expand(1, channels, temporal, height, width).to(dtype=sample.dtype)
    by = by.expand(1, channels, temporal, height, width).to(dtype=sample.dtype)
    return bx, by


def construct_residual(torch: Any, sample: Any, points: Sequence[Sequence[float]], target_relative_rms: float) -> tuple[Any, dict[str, float]]:
    bx, by = build_analytic_basis(torch, sample)
    q = torch.tensor(points, device=sample.device, dtype=torch.float32)
    q1 = q[:, 0].reshape(1, 1, -1, 1, 1).to(dtype=sample.dtype)
    q2 = q[:, 1].reshape(1, 1, -1, 1, 1).to(dtype=sample.dtype)
    raw = q1 * bx + q2 * by
    raw_rms = raw.float().square().mean().sqrt()
    output_rms = sample.detach().float().square().mean().sqrt()
    if not bool(torch.isfinite(raw_rms)) or not bool(torch.isfinite(output_rms)) or float(raw_rms) <= 0.0 or float(output_rms) <= 0.0:
        raise CarrierPropagationError("non-finite or zero carrier/output RMS")
    gain = float(target_relative_rms) * output_rms / raw_rms
    residual = raw * gain.to(dtype=sample.dtype)
    relative_rms = residual.float().square().mean().sqrt() / output_rms
    return residual, {
        "raw_rms": float(raw_rms.item()),
        "output_rms": float(output_rms.item()),
        "gain": float(gain.item()),
        "relative_rms": float(relative_rms.item()),
    }


def project_tensor_q(torch: Any, tensor: Any) -> list[list[float]]:
    """Project any audited-shape tensor onto the same public two-dimensional basis."""

    bx, by = build_analytic_basis(torch, tensor)
    value = tensor.detach().float()
    reduce_dims = (0, 1, 3, 4)
    bx_float, by_float = bx.float(), by.float()
    qx = (value * bx_float).mean(dim=reduce_dims) / bx_float.square().mean(dim=reduce_dims)
    qy = (value * by_float).mean(dim=reduce_dims) / by_float.square().mean(dim=reduce_dims)
    return [[float(x), float(y)] for x, y in zip(qx.tolist(), qy.tolist(), strict=True)]


def replace_result_sample(result: Any, modified_sample: Any) -> Any:
    """Return a new transformer result object without mutating the original."""

    if hasattr(result, "sample"):
        try:
            return type(result)(sample=modified_sample)
        except Exception as exc:
            raise CarrierPropagationError("could not reconstruct transformer output object") from exc
    if isinstance(result, tuple) and result:
        return (modified_sample, *result[1:])
    if isinstance(result, list) and result:
        return [modified_sample, *result[1:]]
    raise CarrierPropagationError("Wan transformer result has no replaceable sample")


def result_sample(result: Any) -> Any:
    if hasattr(result, "sample"):
        return result.sample
    if isinstance(result, (tuple, list)) and result:
        return result[0]
    raise CarrierPropagationError("Wan transformer result has no sample")


def saved_video_observation(frames: Any, frame_groups: Sequence[Sequence[int]]) -> list[list[float]]:
    """Key-independent continuous 2D readout from one decoded saved MP4."""

    try:
        import numpy as np
    except ModuleNotFoundError:
        return _saved_video_observation_python(frames, frame_groups)

    array = np.stack([np.asarray(frame, dtype=np.float64) for frame in frames], axis=0)
    if array.ndim != 4 or array.shape[-1] < 3:
        raise CarrierPropagationError(f"saved video must have [F,H,W,C] frames, got {array.shape}")
    height, width = int(array.shape[1]), int(array.shape[2])
    xmask = math.sqrt(2.0) * np.cos(2.0 * math.pi * (np.arange(width) + 0.5) / width)
    ymask = math.sqrt(2.0) * np.cos(2.0 * math.pi * (np.arange(height) + 0.5) / height)
    luma = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    observations: list[list[float]] = []
    for group in frame_groups:
        if not group or min(group) < 0 or max(group) >= array.shape[0]:
            raise CarrierPropagationError("frame group is outside decoded saved video")
        image = luma[list(group)].mean(axis=0)
        centered = image - image.mean()
        qx = float((centered * xmask.reshape(1, width)).mean())
        qy = float((centered * ymask.reshape(height, 1)).mean())
        observations.append([qx, qy])
    if not np.isfinite(np.asarray(observations)).all():
        raise CarrierPropagationError("saved-video observation is non-finite")
    return observations


def _saved_video_observation_python(frames: Any, frame_groups: Sequence[Sequence[int]]) -> list[list[float]]:
    """Small dependency-free equivalent used by CPU contract tests."""

    frame_count = len(frames)
    if frame_count <= 0:
        raise CarrierPropagationError("saved video must not be empty")
    height = len(frames[0])
    width = len(frames[0][0]) if height else 0
    if height <= 0 or width <= 0:
        raise CarrierPropagationError("saved video dimensions must be positive")
    xmask = [math.sqrt(2.0) * math.cos(2.0 * math.pi * (x + 0.5) / width) for x in range(width)]
    ymask = [math.sqrt(2.0) * math.cos(2.0 * math.pi * (y + 0.5) / height) for y in range(height)]
    observations: list[list[float]] = []
    for group in frame_groups:
        if not group or min(group) < 0 or max(group) >= frame_count:
            raise CarrierPropagationError("frame group is outside decoded saved video")
        luma: list[list[float]] = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                total = 0.0
                for frame_index in group:
                    pixel = frames[frame_index][y][x]
                    total += 0.2126 * float(pixel[0]) + 0.7152 * float(pixel[1]) + 0.0722 * float(pixel[2])
                row.append(total / len(group))
            luma.append(row)
        mean = sum(sum(row) for row in luma) / float(height * width)
        qx = sum((luma[y][x] - mean) * xmask[x] for y in range(height) for x in range(width)) / float(height * width)
        qy = sum((luma[y][x] - mean) * ymask[y] for y in range(height) for x in range(width)) / float(height * width)
        if not (math.isfinite(qx) and math.isfinite(qy)):
            raise CarrierPropagationError("saved-video observation is non-finite")
        observations.append([qx, qy])
    return observations


def paired_difference_observation(clean_frames: Any, watermarked_frames: Any, frame_groups: Sequence[Sequence[int]]) -> list[list[float]]:
    """Diagnostic-only paired readout; never eligible as the blind observation."""

    try:
        import numpy as np
    except ModuleNotFoundError:
        if len(clean_frames) != len(watermarked_frames):
            raise CarrierPropagationError("paired videos must have identical decoded shapes")
        difference = []
        for clean_frame, watermarked_frame in zip(clean_frames, watermarked_frames, strict=True):
            if len(clean_frame) != len(watermarked_frame):
                raise CarrierPropagationError("paired videos must have identical decoded shapes")
            difference_frame = []
            for clean_row, watermarked_row in zip(clean_frame, watermarked_frame, strict=True):
                if len(clean_row) != len(watermarked_row):
                    raise CarrierPropagationError("paired videos must have identical decoded shapes")
                difference_frame.append([
                    [float(watermarked_pixel[channel]) - float(clean_pixel[channel]) + 128.0 for channel in range(3)]
                    for clean_pixel, watermarked_pixel in zip(clean_row, watermarked_row, strict=True)
                ])
            difference.append(difference_frame)
        return _saved_video_observation_python(difference, frame_groups)

    clean = np.asarray(clean_frames)
    watermarked = np.asarray(watermarked_frames)
    if clean.shape != watermarked.shape:
        raise CarrierPropagationError("paired videos must have identical decoded shapes")
    difference = watermarked.astype(np.float64) - clean.astype(np.float64) + 128.0
    return saved_video_observation(difference, frame_groups)


def _finite_matrix(value: Any, rows: int, columns: int) -> bool:
    return isinstance(value, list) and len(value) == rows and all(isinstance(row, list) and len(row) == columns and all(math.isfinite(float(item)) for item in row) for row in value)


def run_carrier_propagation(config_path: Path, output_dir: Path, expected_commit: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    repo = Path(__file__).resolve().parents[2]
    commit = _git(repo, "rev-parse", "HEAD")
    dirty_lines = _git(repo, "status", "--porcelain").splitlines()
    if commit != expected_commit:
        raise CarrierPropagationError(f"commit mismatch: observed={commit}, expected={expected_commit}")
    if dirty_lines:
        raise CarrierPropagationError("formal carrier propagation requires a clean checkout")
    try:
        import torch
        import diffusers
        import transformers
        import accelerate
        import imageio
        import imageio.v3 as iio
        import imageio_ffmpeg
        import ftfy
        import safetensors
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
        from huggingface_hub import model_info
    except Exception as exc:
        raise CarrierPropagationError("locked GPU dependencies are unavailable") from exc
    if not torch.cuda.is_available():
        raise CarrierPropagationError("CUDA GPU is required")
    observed_versions = {"diffusers": diffusers.__version__, "transformers": transformers.__version__, "accelerate": accelerate.__version__, "imageio": imageio.__version__, "imageio_ffmpeg": imageio_ffmpeg.__version__, "ftfy": ftfy.__version__, "safetensors": safetensors.__version__}
    mismatches = {key: [observed_versions.get(key), value] for key, value in config["software"].items() if observed_versions.get(key) != value}
    if mismatches:
        raise CarrierPropagationError(f"locked dependency mismatch: {mismatches}")
    token = os.environ.get("HF_TOKEN") or None
    resolved_revision = str(model_info(config["model"]["id"], revision=config["model"]["revision"], token=token).sha)
    if resolved_revision != config["model"]["revision"]:
        raise CarrierPropagationError("model revision mismatch")

    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    _write(output_dir / "git_state.json", {"commit": commit, "dirty": False, "dirty_lines": [], "remote": _git(repo, "remote", "get-url", "origin")})
    _write(output_dir / "config.json", config)
    _write(output_dir / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda, "torch": torch.__version__, **observed_versions, "model_id": config["model"]["id"], "model_revision_requested": config["model"]["revision"], "model_revision_resolved": resolved_revision})
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = WanPipeline.from_pretrained(config["model"]["id"], revision=config["model"]["revision"], torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    scheduler_step_function_before = getattr(pipe.scheduler.step, "__func__", pipe.scheduler.step)
    generation = config["generation"]

    def generate(with_carrier: bool) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
        injection_records: list[dict[str, Any]] = []
        latent_records: list[dict[str, Any]] = []
        original_forward = pipe.transformer.forward

        def injected_forward(*args: Any, **kwargs: Any) -> Any:
            result = original_forward(*args, **kwargs)
            sample = result_sample(result)
            before = getattr(sample, "_version", None)
            residual, energy = construct_residual(torch, sample, config["carrier"]["temporal_points"], float(config["carrier"]["target_relative_rms"]))
            modified = sample + residual
            pre_injection_q = project_tensor_q(torch, sample)
            scaled_q = project_tensor_q(torch, residual)
            reconstructed_q = [[value / energy["gain"] for value in point] for point in scaled_q]
            source_q = config["carrier"]["temporal_points"]
            reconstruction_mse = sum(
                (reconstructed_q[index][dimension] - float(source_q[index][dimension])) ** 2
                for index in range(13) for dimension in range(2)
            ) / 26.0
            after = getattr(sample, "_version", None)
            injection_records.append({"call_index": len(injection_records), "timestep_value": float(kwargs["timestep"].detach().float().flatten()[0].item()), "sample_shape": [int(item) for item in sample.shape], "sample_dtype": str(sample.dtype), "original_tensor_version_before": before, "original_tensor_version_after": after, "original_tensor_version_unchanged": before == after, "modified_has_distinct_storage": int(modified.data_ptr()) != int(sample.data_ptr()), "pre_injection_output_q_diagnostic": pre_injection_q, "residual_q_reconstructed": reconstructed_q, "residual_q_reconstruction_mse": reconstruction_mse, **energy})
            return replace_result_sample(result, modified)

        def callback(_pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            latents = callback_kwargs["latents"]
            latent_records.append({"step_index": int(step_index), "timestep_value": float(timestep.detach().float().flatten()[0].item()), "shape": [int(item) for item in latents.shape], "dtype": str(latents.dtype), "finite": bool(torch.isfinite(latents).all().item()), "q_diagnostic": project_tensor_q(torch, latents)})
            return callback_kwargs

        if with_carrier:
            pipe.transformer.forward = injected_forward
        try:
            generator = torch.Generator(device="cuda").manual_seed(int(generation["seed"]))
            result = pipe(prompt=generation["prompt"], negative_prompt=generation["negative_prompt"], num_frames=int(generation["frame_count"]), height=int(generation["height"]), width=int(generation["width"]), guidance_scale=float(generation["guidance_scale"]), num_inference_steps=int(generation["inference_steps"]), generator=generator, callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
            raw_frames = result.frames
            if len(raw_frames) == 1:
                first = raw_frames[0]
                first_shape = getattr(first, "shape", None)
                if isinstance(first, list) or (first_shape is not None and len(first_shape) == 4):
                    frames = list(first)
                else:
                    frames = list(raw_frames)
            else:
                frames = list(raw_frames)
            return list(frames), injection_records, latent_records
        finally:
            pipe.transformer.forward = original_forward

    clean_frames, clean_injections, clean_latents = generate(False)
    if clean_injections:
        raise CarrierPropagationError("clean generation unexpectedly injected a carrier")
    watermarked_frames, injection_records, watermarked_latents = generate(True)
    scheduler_step_function_after = getattr(pipe.scheduler.step, "__func__", pipe.scheduler.step)
    clean_path = artifacts / "clean.mp4"
    watermarked_path = artifacts / "watermarked.mp4"
    export_to_video(clean_frames, str(clean_path), fps=int(generation["fps"]))
    export_to_video(watermarked_frames, str(watermarked_path), fps=int(generation["fps"]))
    clean_saved = list(iio.imiter(clean_path))
    watermarked_saved = list(iio.imiter(watermarked_path))
    groups = config["diagnostic_readouts"]["temporal_frame_groups"]
    rgb_pre_mp4_q = saved_video_observation(watermarked_frames, groups)
    blind_q = saved_video_observation(watermarked_saved, groups)
    paired_q = paired_difference_observation(clean_saved, watermarked_saved, groups)
    paired_public = paired_q[:6]
    paired_public_mean = [sum(point[dimension] for point in paired_public) / 6.0 for dimension in range(2)]
    paired_public_centered = [[point[dimension] - paired_public_mean[dimension] for dimension in range(2)] for point in paired_public]
    paired_public_condition = condition_number_2d_columns(paired_public_centered)
    paired_public_aisb_residual = affine_burst_residual([list(point) for point in paired_public], make_default_templates()[0])
    _write(artifacts / "injection_records.json", injection_records)
    _write(artifacts / "clean_latent_callback.json", clean_latents)
    _write(artifacts / "watermarked_latent_callback.json", watermarked_latents)
    _write(artifacts / "blind_rgb_pre_mp4_observation.json", {"input": "single_watermarked_rgb_output_before_mp4_only", "key_independent": True, "q": rgb_pre_mp4_q})
    _write(artifacts / "blind_watermarked_mp4_observation.json", {"input": "single_saved_watermarked_mp4_only", "key_independent": True, "q": blind_q})
    _write(artifacts / "paired_difference_observation_diagnostic_only.json", {"eligible_for_blind_pass": False, "q": paired_q})

    tolerance = float(config["carrier"]["target_relative_rms_absolute_tolerance"])
    target = float(config["carrier"]["target_relative_rms"])
    checks = {
        "all_injection_records_finite": len(injection_records) > 0 and all(all(math.isfinite(float(record[key])) for key in ("raw_rms", "output_rms", "gain", "relative_rms")) for record in injection_records),
        "blind_saved_mp4_observation_finite_13_by_2": _finite_matrix(blind_q, 13, 2),
        "clean_and_watermarked_mp4_each_have_49_frames": len(clean_saved) == 49 and len(watermarked_saved) == 49,
        "direct_transformer_return_injection_used": bool(injection_records) and config["carrier"]["kind"] == "direct_model_output_residual",
        "exact_commit_required": commit == expected_commit,
        "injection_call_count_is_16": len(injection_records) == 16,
        "latent_callback_shape_is_1_16_13_40_64": len(watermarked_latents) == 8 and all(record["shape"] == [1, 16, 13, 40, 64] and record["finite"] for record in watermarked_latents),
        "model_revision_exact": resolved_revision == config["model"]["revision"],
        "original_output_tensor_version_unchanged": bool(injection_records) and all(record["original_tensor_version_unchanged"] and record["modified_has_distinct_storage"] for record in injection_records),
        "paired_difference_observation_finite_13_by_2_diagnostic_only": _finite_matrix(paired_q, 13, 2),
        "paired_difference_public_aisb_residual_at_most_0_25_diagnostic_only": math.isfinite(paired_public_aisb_residual) and paired_public_aisb_residual <= float(config["paired_diagnostic_thresholds"]["maximum_public_aisb_residual"]),
        "paired_difference_public_condition_number_at_most_10_diagnostic_only": math.isfinite(paired_public_condition) and paired_public_condition <= float(config["paired_diagnostic_thresholds"]["maximum_public_condition_number"]),
        "residual_q_reconstruction_mse_at_most_0_0001": bool(injection_records) and all(record["residual_q_reconstruction_mse"] <= 0.0001 for record in injection_records),
        "repository_clean": not dirty_lines,
        "rgb_pre_mp4_observation_finite_13_by_2": _finite_matrix(rgb_pre_mp4_q, 13, 2),
        "scheduler_mutation_absent": config["carrier"]["scheduler_mutation_permitted"] is False and scheduler_step_function_after is scheduler_step_function_before,
        "target_relative_rms_within_tolerance": bool(injection_records) and all(abs(record["relative_rms"] - target) <= tolerance for record in injection_records),
    }
    if set(checks) != set(config["pass_conditions"]):
        raise CarrierPropagationError("implemented checks do not exactly match predeclared pass conditions")
    metrics = {"evidence_kind": "real_wan_direct_model_output_residual_propagation_diagnostic", "checks": checks, "gate_pass": all(checks.values()), "injection_call_count": len(injection_records), "clean_saved_frame_count": len(clean_saved), "watermarked_saved_frame_count": len(watermarked_saved), "blind_observation_shape": [len(blind_q), 2], "paired_public_aisb_residual_diagnostic_only": paired_public_aisb_residual, "paired_public_condition_number_diagnostic_only": paired_public_condition, "paired_difference_is_diagnostic_only": True, "saved_mp4_method_claim": False, "owner_wrong_key_claim": False, "temporal_edit_claim": False, "method_claim": False}
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "gate_decision.json", {"gate": "GPU direct model-output carrier propagation diagnostic", "predeclared_conditions": config["pass_conditions"], "implementer_decision": "GATE_PASS" if metrics["gate_pass"] else "GATE_FAIL", "auditor_decision": "PENDING", "saved_mp4_method_gate_admitted": False})
    (output_dir / "stdout.log").write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "stderr.log").write_text("", encoding="utf-8")
    (output_dir / "README.md").write_text("# GPU carrier propagation diagnostic\n\nDirect model-output injection is exercised on a real Wan trajectory. Paired differences are diagnostic only; the single-watermarked-MP4 blind observation is stored separately. This package makes no owner/wrong-key, temporal-edit, or method claim.\n", encoding="utf-8")
    finalize_package(output_dir)
    return metrics


def write_failure_package(config_path: Path, output_dir: Path, expected_commit: str, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(exist_ok=True)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        config = {"config_read_failure": True}
    _write(output_dir / "config.json", config)
    _write(output_dir / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "cuda_status": "not_confirmed_failure_path"})
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = _git(repo, "rev-parse", "HEAD")
        dirty_lines = _git(repo, "status", "--porcelain").splitlines()
        remote = _git(repo, "remote", "get-url", "origin")
    except Exception as exc:
        commit, dirty_lines, remote = "unavailable", [type(exc).__name__], "unavailable"
    _write(output_dir / "git_state.json", {"commit": commit, "expected_commit": expected_commit, "dirty": bool(dirty_lines), "dirty_lines": dirty_lines, "remote": remote, "status": "failure_before_or_during_formal_run"})
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    metrics = {"evidence_kind": "gpu_carrier_propagation_runtime_failure", "gate_pass": False, "reason": reason, "saved_mp4_method_claim": False, "method_claim": False}
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "gate_decision.json", {"gate": "GPU direct model-output carrier propagation diagnostic", "implementer_decision": "GATE_FAIL", "auditor_decision": "PENDING", "reason": reason, "saved_mp4_method_gate_admitted": False})
    (output_dir / "stdout.log").touch()
    (output_dir / "stderr.log").write_text(reason + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text("# GPU carrier propagation failure\n\nThe run failed and supports no carrier or method claim.\n", encoding="utf-8")
    _write(output_dir / "artifacts" / "failure.json", {"reason": reason})
    finalize_package(output_dir)


def finalize_package(output_dir: Path) -> None:
    checksum_path = output_dir / "checksums.sha256"
    targets = [path for path in output_dir.rglob("*") if path.is_file() and path != checksum_path]
    checksum_path.write_text("\n".join(f"{_sha(path)}  {path.relative_to(output_dir).as_posix()}" for path in sorted(targets)) + "\n", encoding="utf-8")
    with tarfile.open(output_dir.with_suffix(".tar.gz"), "w:gz") as handle:
        handle.add(output_dir, arcname=output_dir.name)
