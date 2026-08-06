"""Independent v2 of the real-Wan carrier propagation probe.

v2 fixes only post-cast energy calibration and adds paired RGB-before-MP4
localization.  Paired and internal observations remain diagnostic only.  The
blind relation gate reads one saved watermarked MP4.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Sequence

from .aisb import affine_burst_residual, make_default_templates
from .gpu_carrier import (
    CarrierPropagationError,
    _finite_matrix,
    _git,
    _write,
    build_analytic_basis,
    finalize_package,
    paired_difference_observation,
    project_tensor_q,
    replace_result_sample,
    result_sample,
    saved_video_observation,
)
from .linalg import condition_number_2d_columns


V1_COMMIT = "17eb0cab4e0a0dc9ec4a6abce0692c13a18695fb"
EXPECTED_SHAPE = [1, 16, 13, 40, 64]
FROZEN_POINTS = [
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    [0.25, 0.35], [0.75, 0.2], [0.2, 0.8],
    [-0.9238795325, 0.3826834324], [0.3826834324, 0.9238795325],
    [-0.3826834324, -0.9238795325], [0.9238795325, -0.3826834324],
    [-0.7071067812, 0.7071067812], [0.7071067812, 0.7071067812], [0.0, -1.0],
]
FROZEN_ROLES = [
    "public_aisb_anchor_0", "public_aisb_anchor_1", "public_aisb_anchor_2",
    "public_aisb_checksum_3", "public_aisb_checksum_4", "public_aisb_checksum_5",
    "propagation_probe_6", "propagation_probe_7", "propagation_probe_8",
    "propagation_probe_9", "propagation_probe_10", "propagation_probe_11",
    "propagation_probe_12",
]
FROZEN_GROUPS = [
    [0], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
    [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24],
    [25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36],
    [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48],
]
FROZEN_BASIS = [
    "sqrt_2_cos_2pi_x_centered_over_width_all_channels",
    "sqrt_2_cos_2pi_y_centered_over_height_all_channels",
]


def validate_config_v2(config: dict[str, Any]) -> None:
    if config.get("protocol_id") != "sc_sstw_gpu_carrier_propagation_v2":
        raise CarrierPropagationError("unexpected carrier propagation v2 protocol")
    carrier = config["carrier"]
    generation = config["generation"]
    groups = config["diagnostic_readouts"]["temporal_frame_groups"]
    points = carrier["temporal_points"]
    if carrier["expected_tensor_shape"] != EXPECTED_SHAPE:
        raise CarrierPropagationError("v2 shape must match audited real-Wan shape")
    if len(points) != 13 or len(carrier["temporal_point_roles"]) != 13 or len(groups) != 13:
        raise CarrierPropagationError("v2 points, roles, and groups must contain 13 entries")
    if points != FROZEN_POINTS:
        raise CarrierPropagationError("v2 temporal points differ from v1")
    if carrier["temporal_point_roles"] != FROZEN_ROLES:
        raise CarrierPropagationError("v2 temporal roles differ from v1")
    if groups != FROZEN_GROUPS:
        raise CarrierPropagationError("v2 frame group boundaries differ from v1")
    if carrier["analytic_basis"] != FROZEN_BASIS:
        raise CarrierPropagationError("v2 analytic basis differs from v1")
    if carrier["apply_to"] != "every_transformer_return_both_cfg_branches":
        raise CarrierPropagationError("v2 apply target differs from v1")
    if carrier["injection_location"] != "Wan_transformer_forward_return_before_pipeline_CFG_and_scheduler_step":
        raise CarrierPropagationError("v2 injection location differs from v1")
    if carrier["kind"] != "direct_model_output_residual":
        raise CarrierPropagationError("v2 admits only direct model-output residual")
    if carrier["gain_solver"] != "deterministic_effective_delta_multiplicative_correction":
        raise CarrierPropagationError("v2 requires the frozen post-cast gain solver")
    if int(carrier["gain_solver_max_iterations"]) != 12:
        raise CarrierPropagationError("v2 gain iteration count must remain 12")
    if carrier["required_runtime_dtype"] != "torch.bfloat16":
        raise CarrierPropagationError("v2 runtime dtype must remain torch.bfloat16")
    if carrier["scheduler_mutation_permitted"] is not False or carrier["in_place_mutation_permitted"] is not False:
        raise CarrierPropagationError("scheduler and in-place mutation remain forbidden")
    if float(carrier["target_relative_rms"]) != 0.03 or float(carrier["target_relative_rms_absolute_tolerance"]) != 0.00005:
        raise CarrierPropagationError("v2 energy target and tolerance must equal v1")
    if config["relation_thresholds"] != {"maximum_public_aisb_residual": 0.25, "maximum_public_condition_number": 10.0}:
        raise CarrierPropagationError("v2 relation thresholds must remain frozen")
    expected_generation = {
        "fps": 8, "frame_count": 49, "guidance_scale": 5.0, "height": 320,
        "inference_steps": 8,
        "negative_prompt": "text, watermark, logo, camera motion, cuts, multiple subjects, flicker",
        "prompt": "locked camera, dark matte background, a single bright white cube moving slowly across the center, simple studio lighting, no text, no cuts",
        "seed": 1275, "width": 512,
    }
    if generation != expected_generation:
        raise CarrierPropagationError("v2 generation inputs must remain exactly equal to v1")
    if config["model"] != {"id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "revision": "0fad780a534b6463e45facd96134c9f345acfa5b"}:
        raise CarrierPropagationError("v2 model identity must remain exactly equal to v1")
    expected_software = {
        "accelerate": "1.4.0", "diffusers": "0.35.2", "ftfy": "6.3.1",
        "imageio": "2.37.0", "imageio_ffmpeg": "0.6.0", "safetensors": "0.5.3",
        "transformers": "4.49.0",
    }
    if config["software"] != expected_software:
        raise CarrierPropagationError("v2 software lock differs from v1")
    if any(len(point) != 2 or not all(math.isfinite(float(v)) for v in point) for point in points):
        raise CarrierPropagationError("v2 temporal points must be finite and two-dimensional")


def solve_quantized_scalar_gain(
    initial_gain: float,
    target_relative_rms: float,
    absolute_tolerance: float,
    max_iterations: int,
    measure_relative_rms: Callable[[float], float],
) -> dict[str, float | int | bool]:
    """Correct a scalar using only measurements after all low-precision ops."""

    gain = float(initial_gain)
    best: tuple[float, float, float, int] | None = None
    converged = False
    for iteration in range(int(max_iterations) + 1):
        relative_rms = float(measure_relative_rms(gain))
        error = relative_rms - float(target_relative_rms)
        if not math.isfinite(relative_rms) or relative_rms <= 0.0:
            raise CarrierPropagationError("effective injected RMS is non-finite or zero")
        if best is None or abs(error) < abs(best[0]):
            best = (error, relative_rms, gain, iteration)
        if abs(error) <= float(absolute_tolerance):
            converged = True
            break
        gain *= float(target_relative_rms) / relative_rms
    assert best is not None
    best_error, best_relative_rms, best_gain, best_iteration = best
    return {
        "applied_gain": best_gain,
        "gain_iterations": best_iteration,
        "effective_relative_rms": best_relative_rms,
        "effective_absolute_error": abs(best_error),
        "gain_solver_converged": bool(converged or abs(best_error) <= float(absolute_tolerance)),
    }


def construct_quantization_aware_residual(
    torch: Any,
    sample: Any,
    points: Sequence[Sequence[float]],
    target_relative_rms: float,
    absolute_tolerance: float,
    max_iterations: int,
) -> tuple[Any, Any, dict[str, float | int | bool]]:
    """Solve one scalar against the effective low-precision injected delta."""

    bx, by = build_analytic_basis(torch, sample)
    q = torch.tensor(points, device=sample.device, dtype=torch.float32)
    q1 = q[:, 0].reshape(1, 1, -1, 1, 1)
    q2 = q[:, 1].reshape(1, 1, -1, 1, 1)
    raw_float = q1 * bx.float() + q2 * by.float()
    raw_rms = raw_float.square().mean().sqrt()
    output_rms = sample.detach().float().square().mean().sqrt()
    if not bool(torch.isfinite(raw_rms)) or not bool(torch.isfinite(output_rms)) or float(raw_rms) <= 0.0 or float(output_rms) <= 0.0:
        raise CarrierPropagationError("non-finite or zero carrier/output RMS")

    initial_gain = float((float(target_relative_rms) * output_rms / raw_rms).item())
    gain = initial_gain
    best: tuple[float, Any, Any, float, float, int] | None = None
    converged = False
    for iteration in range(int(max_iterations) + 1):
        candidate = (raw_float * gain).to(dtype=sample.dtype)
        modified = sample + candidate
        effective_delta = modified - sample
        relative_rms = float((effective_delta.float().square().mean().sqrt() / output_rms).item())
        error = relative_rms - float(target_relative_rms)
        if not math.isfinite(relative_rms) or relative_rms <= 0.0:
            raise CarrierPropagationError("effective injected RMS is non-finite or zero")
        if best is None or abs(error) < abs(best[0]):
            best = (error, modified, effective_delta, relative_rms, gain, iteration)
        if abs(error) <= float(absolute_tolerance):
            converged = True
            break
        gain *= float(target_relative_rms) / relative_rms

    assert best is not None
    best_error, best_modified, best_effective_delta, best_relative_rms, best_gain, best_iteration = best
    return best_modified, best_effective_delta, {
        "raw_float_rms": float(raw_rms.item()),
        "output_rms": float(output_rms.item()),
        "initial_gain": initial_gain,
        "applied_gain": best_gain,
        "gain_iterations": best_iteration,
        "effective_relative_rms": best_relative_rms,
        "effective_absolute_error": abs(best_error),
        "gain_solver_converged": bool(converged or abs(best_error) <= float(absolute_tolerance)),
    }


def run_effective_delta_preflight(torch: Any, config: dict[str, Any], dtype: Any) -> dict[str, Any]:
    """Exercise the exact low-precision add/subtract path before model loading."""

    required_dtype = config["carrier"]["required_runtime_dtype"]
    observed_dtype = str(dtype)
    if observed_dtype != required_dtype:
        return {
            "passed": False,
            "required_dtype": required_dtype,
            "observed_dtype": observed_dtype,
            "records": [],
            "reason": "required BF16 runtime dtype is unavailable",
        }
    target = float(config["carrier"]["target_relative_rms"])
    tolerance = float(config["carrier"]["target_relative_rms_absolute_tolerance"])
    points = config["carrier"]["temporal_points"]
    scales = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0] * 2
    generator = torch.Generator(device="cuda").manual_seed(20260807)
    records: list[dict[str, Any]] = []
    for case_index, scale in enumerate(scales):
        bias = (0.25 if case_index % 2 == 0 else -0.25) * scale
        try:
            base = torch.randn(EXPECTED_SHAPE, generator=generator, device="cuda", dtype=torch.float32)
            sample = (scale * base + bias).to(dtype=dtype)
            modified, effective_delta, energy = construct_quantization_aware_residual(
                torch, sample, points, target, tolerance,
                int(config["carrier"]["gain_solver_max_iterations"]),
            )
            reconstructed_q = [[value / float(energy["applied_gain"]) for value in point] for point in project_tensor_q(torch, effective_delta)]
            reconstruction_mse = sum(
                (reconstructed_q[i][d] - float(points[i][d])) ** 2
                for i in range(13) for d in range(2)
            ) / 26.0
            finite = bool(torch.isfinite(modified).all().item() and torch.isfinite(effective_delta).all().item())
            exact_effective_delta = bool(torch.equal(effective_delta, modified - sample))
            case_pass = bool(
                finite
                and exact_effective_delta
                and energy["gain_solver_converged"]
                and abs(float(energy["effective_relative_rms"]) - target) <= tolerance
                and reconstruction_mse <= 0.0001
            )
            records.append({
                "case_index": case_index, "scale": scale, "bias": bias,
                "sample_dtype": str(sample.dtype), "modified_dtype": str(modified.dtype),
                "effective_delta_dtype": str(effective_delta.dtype),
                "modified_and_effective_delta_finite": finite,
                "effective_delta_exactly_modified_minus_sample": exact_effective_delta,
                "residual_q_reconstruction_mse": reconstruction_mse,
                "case_pass": case_pass, **energy,
            })
        except Exception as exc:
            records.append({
                "case_index": case_index, "scale": scale, "bias": bias,
                "case_pass": False, "error": f"{type(exc).__name__}: {exc}",
            })
    return {
        "passed": len(records) == 16 and all(record["case_pass"] for record in records),
        "required_dtype": required_dtype,
        "observed_dtype": observed_dtype,
        "target_relative_rms": target,
        "absolute_tolerance": tolerance,
        "records": records,
    }

def _relation_metrics(q: list[list[float]]) -> dict[str, float | bool]:
    public = q[:6]
    mean = [sum(point[d] for point in public) / 6.0 for d in range(2)]
    centered = [[point[d] - mean[d] for d in range(2)] for point in public]
    condition = condition_number_2d_columns(centered)
    residual = affine_burst_residual([list(point) for point in public], make_default_templates()[0])
    return {
        "public_aisb_residual": float(residual),
        "public_condition_number": float(condition),
        "aisb_residual_at_most_0_25": math.isfinite(residual) and residual <= 0.25,
        "condition_number_at_most_10": math.isfinite(condition) and condition <= 10.0,
    }


def run_carrier_propagation_v2(config_path: Path, output_dir: Path, expected_commit: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config_v2(config)
    repo = Path(__file__).resolve().parents[2]
    commit = _git(repo, "rev-parse", "HEAD")
    dirty_lines = _git(repo, "status", "--porcelain").splitlines()
    if commit != expected_commit:
        raise CarrierPropagationError(f"commit mismatch: observed={commit}, expected={expected_commit}")
    if dirty_lines:
        raise CarrierPropagationError("formal carrier propagation v2 requires a clean checkout")
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
    observed_versions = {
        "diffusers": diffusers.__version__, "transformers": transformers.__version__,
        "accelerate": accelerate.__version__, "imageio": imageio.__version__,
        "imageio_ffmpeg": imageio_ffmpeg.__version__, "ftfy": ftfy.__version__,
        "safetensors": safetensors.__version__,
    }
    mismatches = {k: [observed_versions.get(k), v] for k, v in config["software"].items() if observed_versions.get(k) != v}
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
    effective_delta_preflight = run_effective_delta_preflight(torch, config, dtype)
    _write(artifacts / "effective_delta_preflight.json", effective_delta_preflight)
    if not effective_delta_preflight["passed"]:
        raise CarrierPropagationError("BF16 effective-delta preflight failed before model loading")
    pipe = WanPipeline.from_pretrained(config["model"]["id"], revision=config["model"]["revision"], torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    scheduler_step_function_before = getattr(pipe.scheduler.step, "__func__", pipe.scheduler.step)
    generation = config["generation"]

    def generate(with_carrier: bool) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
        injections: list[dict[str, Any]] = []
        latents_out: list[dict[str, Any]] = []
        original_forward = pipe.transformer.forward

        def injected_forward(*args: Any, **kwargs: Any) -> Any:
            result = original_forward(*args, **kwargs)
            sample = result_sample(result)
            before = getattr(sample, "_version", None)
            modified, effective_delta, energy = construct_quantization_aware_residual(
                torch, sample, config["carrier"]["temporal_points"],
                float(config["carrier"]["target_relative_rms"]),
                float(config["carrier"]["target_relative_rms_absolute_tolerance"]),
                int(config["carrier"]["gain_solver_max_iterations"]),
            )
            reconstructed_q = [[value / float(energy["applied_gain"]) for value in point] for point in project_tensor_q(torch, effective_delta)]
            source_q = config["carrier"]["temporal_points"]
            reconstruction_mse = sum((reconstructed_q[i][d] - float(source_q[i][d])) ** 2 for i in range(13) for d in range(2)) / 26.0
            after = getattr(sample, "_version", None)
            injections.append({
                "call_index": len(injections),
                "timestep_value": float(kwargs["timestep"].detach().float().flatten()[0].item()),
                "sample_shape": [int(item) for item in sample.shape], "sample_dtype": str(sample.dtype),
                "effective_delta_dtype": str(effective_delta.dtype),
                "original_tensor_version_before": before, "original_tensor_version_after": after,
                "original_tensor_version_unchanged": before == after,
                "modified_has_distinct_storage": int(modified.data_ptr()) != int(sample.data_ptr()),
                "pre_injection_output_q_diagnostic": project_tensor_q(torch, sample),
                "residual_q_reconstructed": reconstructed_q,
                "residual_q_reconstruction_mse": reconstruction_mse,
                **energy,
            })
            return replace_result_sample(result, modified)

        def callback(_pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            latents = callback_kwargs["latents"]
            latents_out.append({"step_index": int(step_index), "timestep_value": float(timestep.detach().float().flatten()[0].item()), "shape": [int(item) for item in latents.shape], "dtype": str(latents.dtype), "finite": bool(torch.isfinite(latents).all().item()), "q_diagnostic": project_tensor_q(torch, latents)})
            return callback_kwargs

        if with_carrier:
            pipe.transformer.forward = injected_forward
        try:
            generator = torch.Generator(device="cuda").manual_seed(int(generation["seed"]))
            result = pipe(prompt=generation["prompt"], negative_prompt=generation["negative_prompt"], num_frames=49, height=320, width=512, guidance_scale=5.0, num_inference_steps=8, generator=generator, callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
            raw_frames = result.frames
            if len(raw_frames) == 1:
                first = raw_frames[0]
                first_shape = getattr(first, "shape", None)
                frames = list(first) if isinstance(first, list) or (first_shape is not None and len(first_shape) == 4) else list(raw_frames)
            else:
                frames = list(raw_frames)
            return frames, injections, latents_out
        finally:
            pipe.transformer.forward = original_forward

    clean_frames, clean_injections, clean_latents = generate(False)
    if clean_injections:
        raise CarrierPropagationError("clean generation unexpectedly injected a carrier")
    watermarked_frames, injection_records, watermarked_latents = generate(True)
    scheduler_step_function_after = getattr(pipe.scheduler.step, "__func__", pipe.scheduler.step)
    clean_path, watermarked_path = artifacts / "clean.mp4", artifacts / "watermarked.mp4"
    export_to_video(clean_frames, str(clean_path), fps=8)
    export_to_video(watermarked_frames, str(watermarked_path), fps=8)
    clean_saved, watermarked_saved = list(iio.imiter(clean_path)), list(iio.imiter(watermarked_path))
    groups = config["diagnostic_readouts"]["temporal_frame_groups"]
    blind_q = saved_video_observation(watermarked_saved, groups)
    paired_rgb_q = paired_difference_observation(clean_frames, watermarked_frames, groups)
    paired_mp4_q = paired_difference_observation(clean_saved, watermarked_saved, groups)
    blind_relation = _relation_metrics(blind_q)
    paired_rgb_relation = _relation_metrics(paired_rgb_q)
    paired_mp4_relation = _relation_metrics(paired_mp4_q)

    _write(artifacts / "injection_records.json", injection_records)
    _write(artifacts / "clean_latent_callback.json", clean_latents)
    _write(artifacts / "watermarked_latent_callback.json", watermarked_latents)
    _write(artifacts / "blind_single_watermarked_mp4_observation.json", {"input": "single_saved_watermarked_mp4_only", "key_independent": True, "uses_clean_video": False, "q": blind_q, "relation": blind_relation})
    _write(artifacts / "paired_rgb_pre_mp4_observation_diagnostic_only.json", {"eligible_for_blind_pass": False, "input": "watermarked_rgb_minus_clean_rgb_before_mp4", "q": paired_rgb_q, "relation": paired_rgb_relation})
    _write(artifacts / "paired_mp4_observation_diagnostic_only.json", {"eligible_for_blind_pass": False, "input": "watermarked_saved_mp4_minus_clean_saved_mp4", "q": paired_mp4_q, "relation": paired_mp4_relation})

    target = float(config["carrier"]["target_relative_rms"])
    tolerance = float(config["carrier"]["target_relative_rms_absolute_tolerance"])
    checks = {
        "all_injection_records_finite": bool(injection_records) and all(all(math.isfinite(float(r[k])) for k in ("raw_float_rms", "output_rms", "initial_gain", "applied_gain", "effective_relative_rms", "effective_absolute_error")) for r in injection_records),
        "blind_saved_mp4_observation_finite_13_by_2": _finite_matrix(blind_q, 13, 2),
        "blind_saved_mp4_public_aisb_residual_at_most_0_25": bool(blind_relation["aisb_residual_at_most_0_25"]),
        "blind_saved_mp4_public_condition_number_at_most_10": bool(blind_relation["condition_number_at_most_10"]),
        "clean_and_watermarked_mp4_each_have_49_frames": len(clean_saved) == 49 and len(watermarked_saved) == 49,
        "direct_transformer_return_injection_used": bool(injection_records) and config["carrier"]["kind"] == "direct_model_output_residual",
        "effective_delta_preflight_16_of_16": bool(effective_delta_preflight["passed"]) and len(effective_delta_preflight["records"]) == 16,
        "exact_commit_required": commit == expected_commit,
        "injection_call_count_is_16": len(injection_records) == 16,
        "latent_callback_shape_is_1_16_13_40_64": len(watermarked_latents) == 8 and all(r["shape"] == EXPECTED_SHAPE and r["finite"] for r in watermarked_latents),
        "model_revision_exact": resolved_revision == config["model"]["revision"],
        "original_output_tensor_version_unchanged": bool(injection_records) and all(r["original_tensor_version_unchanged"] and r["modified_has_distinct_storage"] for r in injection_records),
        "paired_mp4_observation_finite_13_by_2_diagnostic_only": _finite_matrix(paired_mp4_q, 13, 2),
        "paired_rgb_pre_mp4_observation_finite_13_by_2_diagnostic_only": _finite_matrix(paired_rgb_q, 13, 2),
        "repository_clean": not dirty_lines,
        "residual_q_reconstruction_mse_at_most_0_0001": bool(injection_records) and all(r["residual_q_reconstruction_mse"] <= 0.0001 for r in injection_records),
        "scheduler_mutation_absent": scheduler_step_function_after is scheduler_step_function_before,
        "target_relative_rms_within_tolerance_16_of_16": len(injection_records) == 16 and all(r["gain_solver_converged"] and abs(r["effective_relative_rms"] - target) <= tolerance for r in injection_records),
    }
    if set(checks) != set(config["pass_conditions"]):
        raise CarrierPropagationError("v2 implemented checks do not match frozen pass conditions")
    blind_keys = {"blind_saved_mp4_public_aisb_residual_at_most_0_25", "blind_saved_mp4_public_condition_number_at_most_10"}
    execution_integrity_pass = all(value for key, value in checks.items() if key not in blind_keys)
    blind_saved_mp4_relation_pass = all(checks[key] for key in blind_keys)
    gate_pass = execution_integrity_pass and blind_saved_mp4_relation_pass
    metrics = {
        "evidence_kind": "real_wan_direct_model_output_residual_public_relation_propagation_v2",
        "checks": checks, "execution_integrity_pass": execution_integrity_pass,
        "blind_saved_mp4_relation": blind_relation,
        "blind_saved_mp4_relation_pass": blind_saved_mp4_relation_pass,
        "paired_propagation_diagnostic_only": {"rgb_pre_mp4": paired_rgb_relation, "saved_mp4": paired_mp4_relation},
        "gate_pass": gate_pass, "injection_call_count": len(injection_records),
        "clean_saved_frame_count": len(clean_saved), "watermarked_saved_frame_count": len(watermarked_saved),
        "public_relation_propagation_bridge_claim": gate_pass,
        "calibration_claim": False, "held_out_claim": False, "owner_wrong_key_claim": False,
        "temporal_edit_claim": False, "method_claim": False,
    }
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "gate_decision.json", {"gate": "GPU direct model-output carrier propagation v2", "predeclared_conditions": config["pass_conditions"], "implementer_decision": "GATE_PASS" if gate_pass else "GATE_FAIL", "auditor_decision": "PENDING", "maximum_claim_if_pass": "public_relation_propagation_bridge_only"})
    (output_dir / "stdout.log").write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "stderr.log").write_text("", encoding="utf-8")
    (output_dir / "README.md").write_text("# GPU carrier propagation v2\n\nQuantization-aware direct model-output injection. Paired RGB and MP4 differences are diagnostic only. The blind relation is computed only from the single saved watermarked MP4. No calibration, owner/wrong-key, edit, or method claim.\n", encoding="utf-8")
    finalize_package(output_dir)
    return metrics


def write_failure_package_v2(config_path: Path, output_dir: Path, expected_commit: str, reason: str) -> None:
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
        commit, dirty_lines, remote = _git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain").splitlines(), _git(repo, "remote", "get-url", "origin")
    except Exception as exc:
        commit, dirty_lines, remote = "unavailable", [type(exc).__name__], "unavailable"
    _write(output_dir / "git_state.json", {"commit": commit, "expected_commit": expected_commit, "dirty": bool(dirty_lines), "dirty_lines": dirty_lines, "remote": remote, "status": "v2_failure_before_or_during_formal_run"})
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    metrics = {"evidence_kind": "gpu_carrier_propagation_v2_runtime_failure", "gate_pass": False, "reason": reason, "public_relation_propagation_bridge_claim": False, "method_claim": False}
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "gate_decision.json", {"gate": "GPU direct model-output carrier propagation v2", "implementer_decision": "GATE_FAIL", "auditor_decision": "PENDING", "reason": reason})
    (output_dir / "stdout.log").write_text("", encoding="utf-8")
    (output_dir / "stderr.log").write_text(reason + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text("# GPU carrier propagation v2 failure\n\nNo bridge or method claim.\n", encoding="utf-8")
    _write(output_dir / "artifacts" / "failure.json", {"reason": reason})
    finalize_package(output_dir)
