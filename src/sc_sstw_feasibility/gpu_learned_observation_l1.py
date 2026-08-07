"""Formal GPU L1 runner for the frozen learned saved-MP4 observation front-end.

This module generates exactly the four training and two validation videos from
the repository's frozen config.  It injects only at Wan block 29 ``attn1``
outputs, writes each saved MP4 and its independently hashed injection record,
trains the public relation front-end, and evaluates validation from each single
saved MP4.  It does not access held-out, null, private-key, or edit cases.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping

from .calibration import calibrate_from_pilot_pairs, equalize_observations
from .gpu_carrier import _git, _write, finalize_package, result_sample
from .gpu_internal_challenger import (
    EXPECTED_BRANCHES,
    EXPECTED_INTERNAL_SHAPE,
    EXPECTED_LATENT_SHAPE,
    EXPECTED_TIMESTEPS,
    construct_internal_residual,
    observed_cfg_sequence_matches,
    project_internal_q,
    run_internal_preflight,
)
from .learned_observation import (
    CALIBRATION_INDICES,
    CONFIG_CANONICAL_SHA256,
    L1_IDS,
    PER_VIDEO_HELD_OUT_INDICES,
    TEMPORAL_POINTS,
    TRAIN_IDS,
    VALIDATION_IDS,
    acquire_and_freeze_ambiguity,
    assert_stage_dataset_access,
    audit_truth_success_after_freeze,
    calibrate_from_frozen_ambiguity,
    canonical_json_bytes,
    decode_saved_mp4,
    encode_saved_mp4,
    extract_feature_matrix,
    load_frozen_frontend,
    sha256_bytes,
    sha256_file,
    train_public_relation_frontend,
    validate_learned_observation_config,
    validate_new_dataset_artifact,
)


class LearnedObservationL1Error(RuntimeError):
    pass


GPU_SOFTWARE = {
    "accelerate": "1.4.0",
    "diffusers": "0.35.2",
    "ftfy": "6.3.1",
    "imageio": "2.37.0",
    "huggingface_hub": "0.35.3",
    "imageio_ffmpeg": "0.6.0",
    "numpy": "1.26.4",
    "safetensors": "0.5.3",
    "transformers": "4.49.0",
}
EXPECTED_STRUCTURE = {
    "transformer_type": "WanTransformer3DModel",
    "block_count": 30,
    "block_type": "WanTransformerBlock",
    "attention_type": "WanAttention",
    "patch_size": [1, 2, 2],
    "internal_dim": 1536,
}


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_injection_records(records: Any, config: Mapping[str, Any]) -> bool:
    """Validate the actual persisted injection records, not a manifest claim."""

    if not isinstance(records, list) or len(records) != 16:
        return False
    target = float(config["carrier"]["target_relative_rms"])
    tolerance = float(config["carrier"]["target_relative_rms_absolute_tolerance"])
    labels = [record.get("branch_label") for record in records]
    timesteps = [int(float(record.get("timestep_value", -1))) for record in records]
    if labels != EXPECTED_BRANCHES or timesteps != EXPECTED_TIMESTEPS:
        return False
    numeric = (
        "raw_float_rms", "output_rms", "initial_gain", "applied_gain",
        "effective_relative_rms", "effective_absolute_error",
        "residual_q_reconstruction_mse",
    )
    return all(
        record.get("call_index") == index
        and record.get("step_index") == index // 2
        and record.get("branch") == ("cond" if index % 2 == 0 else "uncond")
        and record.get("module_path") == "transformer.blocks[29].attn1"
        and record.get("output_shape") == EXPECTED_INTERNAL_SHAPE
        and record.get("output_dtype") == "torch.bfloat16"
        and isinstance(record.get("original_tensor_version_before"), int)
        and record.get("original_tensor_version_after") == record.get("original_tensor_version_before")
        and record.get("original_tensor_version_unchanged") is True
        and record.get("modified_has_distinct_storage") is True
        and record.get("gain_solver_converged") is True
        and all(_finite_number(record.get(key)) for key in numeric)
        and abs(float(record["effective_relative_rms"]) - target) <= tolerance
        and float(record["residual_q_reconstruction_mse"]) <= 0.0001
        for index, record in enumerate(records)
    )


def inspect_saved_mp4_codec(path: Path) -> dict[str, Any]:
    """Fail closed unless ffprobe reports the frozen H.264/yuv420p stream."""

    executable = shutil.which("ffprobe")
    if executable is None:
        raise LearnedObservationL1Error("ffprobe is required for codec verification")
    completed = subprocess.run(
        [executable, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,width,height", "-of", "json", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise LearnedObservationL1Error("saved MP4 must contain exactly one video stream")
    stream = streams[0]
    expected = {"codec_name": "h264", "pix_fmt": "yuv420p", "width": 512, "height": 320}
    observed = {key: stream.get(key) for key in expected}
    if observed != expected:
        raise LearnedObservationL1Error(f"saved MP4 codec mismatch: {observed}")
    return observed


def _binary_version(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise LearnedObservationL1Error(f"{name} is required")
    completed = subprocess.run([executable, "-version"], check=True, text=True, capture_output=True)
    first_line = (completed.stdout or completed.stderr).splitlines()
    if not first_line:
        raise LearnedObservationL1Error(f"{name} version output is empty")
    return first_line[0]


def _model_structure(transformer: Any) -> dict[str, Any]:
    return {
        "transformer_type": type(transformer).__name__,
        "block_count": len(transformer.blocks),
        "block_type": type(transformer.blocks[29]).__name__ if len(transformer.blocks) > 29 else None,
        "attention_type": type(transformer.blocks[29].attn1).__name__ if len(transformer.blocks) > 29 else None,
        "patch_size": list(transformer.config.patch_size),
        "internal_dim": int(transformer.config.num_attention_heads * transformer.config.attention_head_dim),
    }


def generate_l1_video(pipe: Any, torch: Any, item: Mapping[str, Any], config: Mapping[str, Any], mp4_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate one watermarked video through the frozen real Wan trajectory."""

    if item["carrier"] != "watermarked" or int(item["dataset_id"]) not in L1_IDS:
        raise LearnedObservationL1Error("L1 generation accepts only the six frozen watermarked IDs")
    transformer = pipe.transformer
    carrier = config["carrier"]
    generation = config["generation"]
    records: list[dict[str, Any]] = []
    final_calls: list[dict[str, Any]] = []
    latent_records: list[dict[str, Any]] = []
    context: dict[str, Any] = {"actual_cache_context": None, "timestep": None}
    original_forward = transformer.forward
    original_cache_context = transformer.cache_context
    scheduler_step_before = getattr(pipe.scheduler.step, "__func__", pipe.scheduler.step)
    target_attention = transformer.blocks[29].attn1
    parameter_prefixes = ("to_q.", "to_k.", "to_v.", "to_out.")
    target_before = {
        "processor_object_id": id(target_attention.processor),
        "parameters": {
            name: (id(parameter), getattr(parameter, "_version", None))
            for name, parameter in target_attention.named_parameters()
            if name.startswith(parameter_prefixes)
        },
    }

    @contextmanager
    def observed_cache_context(name: str, *args: Any, **kwargs: Any) -> Any:
        if context["actual_cache_context"] is not None:
            raise LearnedObservationL1Error("nested cache contexts are not admitted")
        context["actual_cache_context"] = str(name)
        try:
            with original_cache_context(name, *args, **kwargs):
                yield
        finally:
            context["actual_cache_context"] = None

    def attention_output_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        call_index = len(records)
        branch = context.get("actual_cache_context")
        if call_index >= 16 or branch not in {"cond", "uncond"}:
            raise LearnedObservationL1Error("attention output arrived outside the frozen call sequence")
        if list(output.shape) != EXPECTED_INTERNAL_SHAPE or str(output.dtype) != "torch.bfloat16":
            raise LearnedObservationL1Error(f"unexpected attention output: {list(output.shape)} {output.dtype}")
        version_before = getattr(output, "_version", None)
        modified, delta, energy = construct_internal_residual(
            torch,
            output,
            carrier["temporal_points"],
            float(carrier["target_relative_rms"]),
            float(carrier["target_relative_rms_absolute_tolerance"]),
            12,
        )
        reconstructed = [[value / float(energy["applied_gain"]) for value in point] for point in project_internal_q(torch, delta)]
        mse = sum(
            (reconstructed[index][dimension] - float(carrier["temporal_points"][index][dimension])) ** 2
            for index in range(13)
            for dimension in range(2)
        ) / 26.0
        version_after = getattr(output, "_version", None)
        records.append({
            "call_index": call_index,
            "step_index": call_index // 2,
            "branch": str(branch),
            "branch_label": f"step_{call_index // 2}_{branch}",
            "timestep_value": float(context["timestep"]),
            "module_path": carrier["module_path"],
            "output_shape": [int(value) for value in output.shape],
            "output_dtype": str(output.dtype),
            "original_tensor_version_before": version_before,
            "original_tensor_version_after": version_after,
            "original_tensor_version_unchanged": version_before == version_after,
            "modified_has_distinct_storage": int(modified.data_ptr()) != int(output.data_ptr()),
            "residual_q_reconstruction_mse": mse,
            **energy,
        })
        return modified

    def observed_forward(*args: Any, **kwargs: Any) -> Any:
        call_index = len(final_calls)
        branch = context.get("actual_cache_context")
        if call_index >= 16 or branch not in {"cond", "uncond"}:
            raise LearnedObservationL1Error("transformer call arrived outside the frozen call sequence")
        timestep = kwargs.get("timestep")
        if timestep is None:
            raise LearnedObservationL1Error("transformer timestep is missing")
        context["timestep"] = float(timestep.detach().float().flatten()[0].item())
        result = original_forward(*args, **kwargs)
        sample = result_sample(result)
        final_calls.append({
            "branch_label": f"step_{call_index // 2}_{branch}",
            "timestep_value": context["timestep"],
            "shape": [int(value) for value in sample.shape],
            "finite": bool(torch.isfinite(sample).all().item()),
        })
        return result

    def callback(_pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        latents = callback_kwargs["latents"]
        latent_records.append({
            "step_index": int(step_index),
            "timestep_value": float(timestep.detach().float().flatten()[0].item()),
            "shape": [int(value) for value in latents.shape],
            "finite": bool(torch.isfinite(latents).all().item()),
        })
        return callback_kwargs

    transformer.forward = observed_forward
    transformer.cache_context = observed_cache_context
    handle = target_attention.register_forward_hook(attention_output_hook)
    hook_removed = False
    try:
        generator = torch.Generator(device="cuda").manual_seed(int(item["seed"]))
        result = pipe(
            prompt=item["prompt"],
            negative_prompt=generation["negative_prompt"],
            num_frames=int(generation["frame_count"]),
            height=int(generation["height"]),
            width=int(generation["width"]),
            guidance_scale=float(generation["guidance_scale"]),
            num_inference_steps=int(generation["inference_steps"]),
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    finally:
        transformer.forward = original_forward
        transformer.cache_context = original_cache_context
        handle.remove()
        hook_removed = True
    restoration = {
        "transformer_forward_restored": transformer.forward is original_forward,
        "cache_context_restored": transformer.cache_context is original_cache_context,
        "attention_hook_removed": hook_removed,
    }
    raw_frames = result.frames
    if len(raw_frames) == 1:
        first = raw_frames[0]
        first_shape = getattr(first, "shape", None)
        frames = list(first) if isinstance(first, list) or (first_shape is not None and len(first_shape) == 4) else list(raw_frames)
    else:
        frames = list(raw_frames)
    target_after = {
        "processor_object_id": id(target_attention.processor),
        "parameters": {
            name: (id(parameter), getattr(parameter, "_version", None))
            for name, parameter in target_attention.named_parameters()
            if name.startswith(parameter_prefixes)
        },
    }
    scheduler_step_after = getattr(pipe.scheduler.step, "__func__", pipe.scheduler.step)
    target_unchanged = bool(target_before["parameters"]) and target_before == target_after
    scheduler_unchanged = scheduler_step_after is scheduler_step_before
    sequence_matches = observed_cfg_sequence_matches(records, final_calls)
    final_calls_valid = len(final_calls) == 16 and all(record["shape"] == EXPECTED_LATENT_SHAPE and record["finite"] for record in final_calls)
    latent_callbacks_valid = len(latent_records) == 8 and all(record["shape"] == EXPECTED_LATENT_SHAPE and record["finite"] for record in latent_records)
    if not target_unchanged or not scheduler_unchanged or not all(restoration.values()):
        raise LearnedObservationL1Error("carrier mutated state or failed hook restoration")
    if not sequence_matches:
        raise LearnedObservationL1Error("observed cond/uncond sequence changed")
    if not final_calls_valid or not latent_callbacks_valid:
        raise LearnedObservationL1Error("final-call or latent callback integrity failed")
    if not validate_injection_records(records, config):
        raise LearnedObservationL1Error("persisted injection record contract failed")
    encode_saved_mp4(frames, mp4_path)
    integrity = {
        "final_transformer_calls": final_calls,
        "latent_callbacks": latent_records,
        "target_attention_before": target_before,
        "target_attention_after": target_after,
        "target_attention_unchanged": target_unchanged,
        "scheduler_function_unchanged": scheduler_unchanged,
        "observed_cfg_sequence_matches": sequence_matches,
        "final_transformer_calls_valid": final_calls_valid,
        "latent_callbacks_valid": latent_callbacks_valid,
        **restoration,
        "execution_integrity_pass": target_unchanged and scheduler_unchanged and sequence_matches and final_calls_valid and latent_callbacks_valid and all(restoration.values()),
        "full_activations_saved": False,
    }
    return records, integrity


def _write_generation_manifest(
    manifest_path: Path,
    item: Mapping[str, Any],
    config: Mapping[str, Any],
    commit: str,
    mp4_path: Path,
    injection_path: Path,
) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "config_sha256": CONFIG_CANONICAL_SHA256,
        "repository_commit": commit,
        "dataset_id": int(item["dataset_id"]),
        "prompt": item["prompt"],
        "seed": int(item["seed"]),
        "carrier": item["carrier"],
        "artifact_sha256": sha256_file(mp4_path),
        "derived_from": None,
        "generation_call": "repository_formal_cli",
        "injection_records_sha256": sha256_file(injection_path),
    }
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
    return sha256_file(manifest_path)


def _validation_metrics(
    observation: list[list[float]],
    ambiguity_path: Path,
    config: dict[str, Any],
    *,
    weights_sha256: str,
    mp4_sha256: str,
) -> tuple[dict[str, Any], str]:
    frozen = acquire_and_freeze_ambiguity(observation, ambiguity_path, config)
    ambiguity_sha = frozen["artifact_sha256"]
    metrics = calibrate_from_frozen_ambiguity(observation, ambiguity_path, ambiguity_sha, config)
    truth_success = audit_truth_success_after_freeze(ambiguity_path, ambiguity_sha, config)
    accepted = frozen["envelope"]["payload"]["accepted_candidates"]
    if len(accepted) != 1:
        raise LearnedObservationL1Error("validation calibration evidence requires one frozen candidate")
    start = int(accepted[0]["start_index"])
    calibration_pairs = [
        {"template_index": index, "observation_index": start + index, "q": list(TEMPORAL_POINTS[index]), "observation": list(map(float, observation[start + index]))}
        for index in CALIBRATION_INDICES
    ]
    calibration = calibrate_from_pilot_pairs([(TEMPORAL_POINTS[index], list(map(float, observation[start + index]))) for index in CALIBRATION_INDICES])
    held_observations = [list(map(float, observation[start + index])) for index in PER_VIDEO_HELD_OUT_INDICES]
    held_equalized = equalize_observations(held_observations, calibration)
    thresholds = config["gate_thresholds"]
    checks = {
        "truth_acquisition_success": truth_success,
        "aisb_residual_at_most_0_25": metrics["accepted_aisb_residual"] <= thresholds["maximum_aisb_residual"],
        "global_second_singular_value_at_least_0_10": metrics["global_centered_output_second_singular_value"] >= thresholds["minimum_global_centered_output_second_singular_value"],
        "affine_second_singular_value_at_least_0_05": metrics["fitted_affine_second_singular_value"] >= thresholds["minimum_fitted_affine_second_singular_value"],
        "affine_condition_number_at_most_10": metrics["fitted_affine_condition_number"] <= thresholds["maximum_fitted_affine_condition_number"],
        "c_only_h_mse_at_most_0_02": metrics["public_calibration_held_out_mse"] <= thresholds["maximum_public_calibration_held_out_mse"],
    }
    evidence = {
        "observation_sha256": sha256_bytes(canonical_json_bytes(observation)),
        "ambiguity_sha256": ambiguity_sha,
        "weights_sha256": weights_sha256,
        "mp4_sha256": mp4_sha256,
        "candidate_start_index": start,
        "calibration_template_indices": list(CALIBRATION_INDICES),
        "calibration_pairs": calibration_pairs,
        "affine_matrix_A": calibration.matrix,
        "affine_bias_b": calibration.bias,
        "held_out_indices": list(PER_VIDEO_HELD_OUT_INDICES),
        "held_out_observations": held_observations,
        "held_out_equalized": [list(point) for point in held_equalized],
        "held_out_targets": [list(TEMPORAL_POINTS[index]) for index in PER_VIDEO_HELD_OUT_INDICES],
    }
    return {**metrics, "checks": checks, "case_pass": all(checks.values()), "calibration_evidence": evidence}, ambiguity_sha


def _runtime_imports() -> dict[str, Any]:
    try:
        import accelerate
        import diffusers
        import ftfy
        import huggingface_hub
        import imageio
        import imageio_ffmpeg
        import numpy
        import safetensors
        import torch
        import transformers
        from diffusers import WanPipeline
        from huggingface_hub import model_info
    except Exception as exc:
        raise LearnedObservationL1Error("locked GPU dependencies are unavailable") from exc
    modules = {
        "accelerate": accelerate, "diffusers": diffusers, "ftfy": ftfy,
        "huggingface_hub": huggingface_hub, "imageio": imageio, "imageio_ffmpeg": imageio_ffmpeg,
        "numpy": numpy, "safetensors": safetensors, "torch": torch, "transformers": transformers,
        "WanPipeline": WanPipeline, "model_info": model_info,
    }
    versions = {name: modules[name].__version__ for name in GPU_SOFTWARE}
    mismatches = {name: [versions[name], expected] for name, expected in GPU_SOFTWARE.items() if versions[name] != expected}
    if mismatches:
        raise LearnedObservationL1Error(f"locked dependency mismatch: {mismatches}")
    if str(torch.__version__).split(".", 1)[0] != "2":
        raise LearnedObservationL1Error(f"validated torch major version must be 2, observed {torch.__version__}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise LearnedObservationL1Error("a BF16-capable CUDA GPU is required")
    modules["versions"] = versions
    return modules


def _build_internal_preflight_config(config: dict[str, Any]) -> dict[str, Any]:
    """Add only the two frozen fields required by the historical helper API."""

    validate_learned_observation_config(config)
    adapted = copy.deepcopy(config)
    adapted["carrier"]["required_runtime_dtype"] = "torch.bfloat16"
    adapted["carrier"]["gain_solver_max_iterations"] = 12
    return adapted


def _run_internal_preflight_adapter(torch: Any, config: dict[str, Any]) -> dict[str, Any]:
    adapted = _build_internal_preflight_config(config)
    return run_internal_preflight(torch, adapted, torch.bfloat16)


def run_gpu_learned_observation_l1(config_path: Path, output_dir: Path, expected_commit: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_learned_observation_config(config)
    assert_stage_dataset_access("gpu_train_validation", L1_IDS)
    repo = Path(__file__).resolve().parents[2]
    commit = _git(repo, "rev-parse", "HEAD")
    dirty_lines = _git(repo, "status", "--porcelain").splitlines()
    if commit != expected_commit or len(expected_commit) != 40:
        raise LearnedObservationL1Error(f"commit mismatch: observed={commit}, expected={expected_commit}")
    if dirty_lines:
        raise LearnedObservationL1Error("formal GPU L1 requires a clean checkout")
    runtime = _runtime_imports()
    torch = runtime["torch"]
    token = os.environ.get("HF_TOKEN") or None
    resolved_revision = str(runtime["model_info"](config["model"]["id"], revision=config["model"]["revision"], token=token).sha)
    if resolved_revision != config["model"]["revision"]:
        raise LearnedObservationL1Error("model revision mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = output_dir / "artifacts"
    datasets = artifacts / "datasets"
    datasets.mkdir(parents=True)
    _write(output_dir / "git_state.json", {"commit": commit, "dirty": False, "dirty_lines": [], "remote": _git(repo, "remote", "get-url", "origin")})
    _write(output_dir / "config.json", config)
    _write(output_dir / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__, "torch_version_policy": "major_version_2_and_bf16_cuda_required",
        **runtime["versions"],
        "imageio_ffmpeg_binary_version": runtime["imageio_ffmpeg"].get_ffmpeg_version(),
        "system_ffmpeg_version": _binary_version("ffmpeg"),
        "system_ffprobe_version": _binary_version("ffprobe"),
        "model_id": config["model"]["id"], "model_revision_requested": config["model"]["revision"],
        "model_revision_resolved": resolved_revision,
    })
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    preflight = _run_internal_preflight_adapter(torch, config)
    _write(artifacts / "effective_delta_preflight.json", preflight)
    if not preflight["passed"]:
        raise LearnedObservationL1Error("internal effective-delta preflight failed")
    pipe = runtime["WanPipeline"].from_pretrained(config["model"]["id"], revision=config["model"]["revision"], torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    structure = _model_structure(pipe.transformer)
    _write(artifacts / "model_structure.json", {**structure, "exact": structure == EXPECTED_STRUCTURE})
    if structure != EXPECTED_STRUCTURE:
        raise LearnedObservationL1Error(f"locked Wan structure mismatch: {structure}")

    items = {int(item["dataset_id"]): item for item in config["dataset"]["items"]}
    features_by_id: dict[int, list[list[float]]] = {}
    mp4_sha_by_id: dict[int, str] = {}
    generation_evidence: list[dict[str, Any]] = []

    def generate_and_record(dataset_id: int) -> dict[str, Any]:
        item = items[dataset_id]
        case_dir = datasets / str(dataset_id)
        case_dir.mkdir()
        mp4_path = case_dir / "watermarked.mp4"
        injection_path = case_dir / "injection_records.json"
        integrity_path = case_dir / "execution_integrity.json"
        manifest_path = case_dir / "generation_manifest.json"
        records, integrity = generate_l1_video(pipe, torch, item, config, mp4_path)
        _write(injection_path, records)
        _write(integrity_path, integrity)
        injection_sha = sha256_file(injection_path)
        if not validate_injection_records(json.loads(injection_path.read_text(encoding="utf-8")), config):
            raise LearnedObservationL1Error(f"dataset {dataset_id} persisted injection records failed")
        if not json.loads(integrity_path.read_text(encoding="utf-8"))["execution_integrity_pass"]:
            raise LearnedObservationL1Error(f"dataset {dataset_id} execution integrity failed")
        manifest_sha = _write_generation_manifest(manifest_path, item, config, commit, mp4_path, injection_path)
        validate_new_dataset_artifact(mp4_path, manifest_path, config, expected_repository_commit=commit, expected_manifest_sha256=manifest_sha)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["injection_records_sha256"] != injection_sha:
            raise LearnedObservationL1Error("manifest injection digest does not match the actual record file")
        codec = inspect_saved_mp4_codec(mp4_path)
        frames = decode_saved_mp4(mp4_path)
        features = extract_feature_matrix(frames)
        features_by_id[dataset_id] = features
        mp4_sha_by_id[dataset_id] = sha256_file(mp4_path)
        _write(case_dir / "features.json", {"input": "single_saved_mp4_only", "mp4_sha256": mp4_sha_by_id[dataset_id], "shape": [13, 30], "features": features})
        return {
            "dataset_id": dataset_id, "split": item["split"], "mp4_sha256": mp4_sha_by_id[dataset_id],
            "injection_records_sha256": injection_sha, "execution_integrity_sha256": sha256_file(integrity_path),
            "manifest_sha256": manifest_sha, "codec": codec, "decoded_frame_count": int(frames.shape[0]),
        }

    for dataset_id in TRAIN_IDS:
        generation_evidence.append(generate_and_record(dataset_id))

    weights_path = artifacts / "frontend_weights.json"
    training_result = train_public_relation_frontend(
        {dataset_id: features_by_id[dataset_id] for dataset_id in TRAIN_IDS},
        {dataset_id: mp4_sha_by_id[dataset_id] for dataset_id in TRAIN_IDS},
        config,
        weights_path,
    )
    weights_sha = training_result["weights_sha256"]
    frontend = load_frozen_frontend(weights_path, expected_weights_sha256=weights_sha, expected_config_sha256=CONFIG_CANONICAL_SHA256)
    weights_payload = json.loads(weights_path.read_text(encoding="utf-8"))
    expected_train_digests = {str(dataset_id): mp4_sha_by_id[dataset_id] for dataset_id in TRAIN_IDS}
    if weights_payload["train_artifact_sha256_by_dataset_id"] != expected_train_digests:
        raise LearnedObservationL1Error("weights train-video digests do not match the actual four MP4 files")
    if any((datasets / str(dataset_id)).exists() for dataset_id in VALIDATION_IDS):
        raise LearnedObservationL1Error("validation artifact existed before weights freeze")
    _write(artifacts / "weights_freeze.json", {
        "weights_sha256": weights_sha, "config_sha256": CONFIG_CANONICAL_SHA256,
        "train_ids": list(TRAIN_IDS), "train_mp4_sha256_by_dataset_id": expected_train_digests,
        "validation_ids_not_created_before_freeze": True,
    })

    training_observations: list[dict[str, Any]] = []
    for dataset_id in TRAIN_IDS:
        case_dir = datasets / str(dataset_id)
        mp4_path = case_dir / "watermarked.mp4"
        observation = frontend.observe_saved_mp4(mp4_path)
        observation_payload = {
            "input": "single_saved_mp4_only", "uses_clean_video": False, "uses_internal_state": False,
            "uses_key_or_truth": False, "weights_sha256": weights_sha,
            "mp4_sha256": sha256_file(mp4_path), "shape": [13, 2], "q": observation,
        }
        _write(case_dir / "blind_observation.json", observation_payload)
        training_observations.append({
            "dataset_id": dataset_id, "mp4_sha256": observation_payload["mp4_sha256"],
            "observation_sha256": sha256_bytes(canonical_json_bytes(observation)), "q": observation,
        })
    _write(artifacts / "training_blind_observations.json", training_observations)

    for dataset_id in VALIDATION_IDS:
        generation_evidence.append(generate_and_record(dataset_id))
    _write(artifacts / "generation_evidence.json", generation_evidence)

    validation_results: list[dict[str, Any]] = []
    for dataset_id in VALIDATION_IDS:
        case_dir = datasets / str(dataset_id)
        mp4_path = case_dir / "watermarked.mp4"
        observation = frontend.observe_saved_mp4(mp4_path)
        observation_payload = {
            "input": "single_saved_mp4_only", "uses_clean_video": False, "uses_internal_state": False,
            "uses_key_or_truth": False, "weights_sha256": weights_sha,
            "mp4_sha256": sha256_file(mp4_path), "shape": [13, 2], "q": observation,
        }
        _write(case_dir / "blind_observation.json", observation_payload)
        metrics, ambiguity_sha = _validation_metrics(
            observation, case_dir / "ambiguity.json", config,
            weights_sha256=weights_sha, mp4_sha256=observation_payload["mp4_sha256"],
        )
        _write(case_dir / "calibration_metrics.json", {**metrics, "ambiguity_sha256": ambiguity_sha})
        validation_results.append({"dataset_id": dataset_id, **metrics, "ambiguity_sha256": ambiguity_sha})

    validation_pass_count = sum(bool(result["case_pass"]) for result in validation_results)
    checks = {
        "exact_commit_required": commit == expected_commit,
        "repository_clean": not dirty_lines,
        "model_revision_and_structure_exact": resolved_revision == config["model"]["revision"] and structure == EXPECTED_STRUCTURE,
        "effective_delta_preflight_16_of_16": preflight["passed"] and len(preflight["records"]) == 16,
        "exact_l1_dataset_ids_and_order": [entry["dataset_id"] for entry in generation_evidence] == list(L1_IDS),
        "six_saved_mp4_codec_and_frame_checks": len(generation_evidence) == 6 and all(entry["codec"] == {"codec_name": "h264", "pix_fmt": "yuv420p", "width": 512, "height": 320} and entry["decoded_frame_count"] == 49 for entry in generation_evidence),
        "six_actual_injection_record_digests_cross_checked": len(generation_evidence) == 6 and all(json.loads((datasets / str(entry["dataset_id"]) / "generation_manifest.json").read_text(encoding="utf-8"))["injection_records_sha256"] == sha256_file(datasets / str(entry["dataset_id"]) / "injection_records.json") for entry in generation_evidence),
        "six_execution_integrity_records_pass": len(generation_evidence) == 6 and all(json.loads((datasets / str(entry["dataset_id"]) / "execution_integrity.json").read_text(encoding="utf-8"))["execution_integrity_pass"] is True for entry in generation_evidence),
        "weights_external_sha_and_schema_valid": frontend.artifact_sha256 == weights_sha,
        "weights_four_train_mp4_digests_cross_checked": weights_payload["train_artifact_sha256_by_dataset_id"] == expected_train_digests,
        "validation_pass_count_is_2_of_2": validation_pass_count == 2 and len(validation_results) == 2,
    }
    gate_pass = all(checks.values())
    metrics = {
        "evidence_kind": "real_wan_learned_saved_mp4_observation_gpu_l1_train_validation",
        "checks": checks,
        "training": training_result,
        "weights_sha256": weights_sha,
        "validation_results": validation_results,
        "validation_pass_count": validation_pass_count,
        "gate_pass": gate_pass,
        "gpu_l2_admission": False,
        "l2_candidate": gate_pass,
        "held_out_claim": False,
        "null_claim": False,
        "owner_wrong_key_claim": False,
        "temporal_edit_claim": False,
        "method_claim": False,
    }
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "gate_decision.json", {
        "gate": "GPU learned saved-MP4 observation L1 train/validation",
        "predeclared_conditions": {key: True for key in checks},
        "actual_results": checks,
        "implementer_decision": "GATE_PASS" if gate_pass else "GATE_FAIL",
        "auditor_decision": "PENDING",
        "maximum_claim_if_pass": "gpu_l1_train_validation_only",
    })
    (output_dir / "README.md").write_text(
        "# GPU learned observation L1\n\nSix new real-Wan watermarked MP4s: four train and two validation. "
        "Validation uses one saved MP4 at a time. No held-out, null, private-key, edit, or method claim.\n",
        encoding="utf-8",
    )
    return metrics


def write_l1_failure_package(config_path: Path, output_dir: Path, expected_commit: str, reason: str, *, stdout_text: str, stderr_text: str) -> None:
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
    _write(output_dir / "git_state.json", {"commit": commit, "expected_commit": expected_commit, "dirty": bool(dirty_lines), "dirty_lines": dirty_lines, "remote": remote, "status": "gpu_l1_failure"})
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    metrics = {
        "evidence_kind": "gpu_learned_observation_l1_runtime_failure", "gate_pass": False,
        "gpu_l2_admission": False, "reason": reason, "method_claim": False,
    }
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "gate_decision.json", {"gate": "GPU learned saved-MP4 observation L1 train/validation", "implementer_decision": "GATE_FAIL", "auditor_decision": "PENDING", "reason": reason})
    (output_dir / "stdout.log").write_text(stdout_text, encoding="utf-8")
    (output_dir / "stderr.log").write_text(stderr_text, encoding="utf-8")
    (output_dir / "README.md").write_text("# GPU learned observation L1 failure\n\nNo L1, L2, or method claim.\n", encoding="utf-8")
    _write(output_dir / "artifacts" / "failure.json", {"reason": reason})
    finalize_package(output_dir)


def finalize_l1_success_package(output_dir: Path, *, stdout_text: str, stderr_text: str) -> None:
    """Persist the exact CLI byte streams before checksums and archive."""

    if (output_dir / "checksums.sha256").exists() or output_dir.with_suffix(".tar.gz").exists():
        raise LearnedObservationL1Error("success package was already finalized")
    (output_dir / "stdout.log").write_text(stdout_text, encoding="utf-8")
    (output_dir / "stderr.log").write_text(stderr_text, encoding="utf-8")
    finalize_package(output_dir)
