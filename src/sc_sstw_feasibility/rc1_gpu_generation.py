"""GPU generation interface for the frozen RC1 matched triplets.

This module is imported only after :func:`rc1_method_validation.preflight` has
validated source identity and the frozen L1-v2 selected-candidate package.  It
never selects a candidate or derives a threshold from fresh videos.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .gpu_internal_challenger import construct_internal_residual
from .learned_observation import decode_saved_mp4, encode_saved_mp4, extract_feature_matrix
from .rc1_method_validation import (
    ARTIFACT_NAMES,
    CONDITIONS,
    EXECUTION_SCHEMA,
    PLAN_RAW_SHA256,
    PROTOCOL_ID,
    Preflight,
    canonical_json_bytes,
    expected_matched_parameters,
    schedule_a,
    schedule_b,
    sha256_bytes,
    sha256_file,
)


class RC1GPUGenerationError(RuntimeError):
    """Raised when generation cannot preserve the frozen matched-triplet contract."""


LOCKED_GPU_PACKAGES = {
    "accelerate": "1.4.0",
    "diffusers": "0.35.2",
    "imageio": "2.37.0",
    "imageio_ffmpeg": "0.6.0",
    "numpy": "1.26.4",
    "safetensors": "0.5.3",
    "transformers": "4.49.0",
}


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def tensor_sha256(tensor: Any) -> str:
    """Hash the actual float32 CPU values and shape of a latent tensor."""

    contiguous = tensor.detach().to(device="cpu", dtype=getattr(__import__("torch"), "float32")).contiguous()
    envelope = canonical_json_bytes({"shape": list(contiguous.shape), "dtype": "float32"}) + contiguous.numpy().tobytes()
    return sha256_bytes(envelope)


def _environment(torch: Any, diffusers: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RC1GPUGenerationError("CUDA runtime is unavailable")
    versions: dict[str, str] = {}
    for package, expected in LOCKED_GPU_PACKAGES.items():
        module = __import__(package)
        observed = str(module.__version__)
        if observed != expected:
            raise RC1GPUGenerationError(f"dependency mismatch for {package}: {observed}")
        versions[package] = observed
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "versions": versions,
    }


def _inspect_saved_mp4(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RC1GPUGenerationError("ffprobe is required for saved-MP4 verification")
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise RC1GPUGenerationError("saved MP4 must contain exactly one video stream")
    observed = streams[0]
    expected = {"codec_name": "h264", "pix_fmt": "yuv420p", "width": 512, "height": 320, "avg_frame_rate": "8/1", "nb_frames": "49"}
    if {key: observed.get(key) for key in expected} != expected:
        raise RC1GPUGenerationError("saved-MP4 codec, geometry, frame-count, or FPS mismatch")
    return expected


def _prepare_initial_latents(pipe: Any, torch: Any, parameters: Mapping[str, Any]) -> Any:
    generator = torch.Generator(device="cuda").manual_seed(int(parameters["seed"]))
    try:
        latents = pipe.prepare_latents(
            1,
            int(pipe.transformer.config.in_channels),
            int(parameters["height"]),
            int(parameters["width"]),
            int(parameters["frame_count"]),
            torch.float32,
            pipe._execution_device,
            generator,
            None,
        )
    except Exception as exc:
        raise RC1GPUGenerationError("Wan initial-latent preparation contract changed") from exc
    if not torch.isfinite(latents).all().item():
        raise RC1GPUGenerationError("initial latent contains non-finite values")
    return latents


def _carrier_hook(pipe: Any, torch: Any, schedule: Sequence[Sequence[float]], carrier: Mapping[str, Any], records: list[dict[str, Any]]) -> Any:
    target = pipe.transformer.blocks[int(carrier["block_index"])].attn1

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        modified, _delta, energy = construct_internal_residual(
            torch,
            output,
            schedule,
            float(carrier["target_relative_rms"]),
            float(carrier["target_relative_rms_absolute_tolerance"]),
            12,
        )
        records.append({
            "call_index": len(records),
            "module_path": carrier["module_path"],
            "schedule_sha256": sha256_bytes(canonical_json_bytes(schedule)),
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "distinct_storage": int(modified.data_ptr()) != int(output.data_ptr()),
            "effective_relative_rms": float(energy["effective_relative_rms"]),
        })
        if abs(float(energy["effective_relative_rms"]) - float(carrier["target_relative_rms"])) > float(carrier["target_relative_rms_absolute_tolerance"]):
            raise RC1GPUGenerationError("carrier RMS differs from the frozen target")
        return modified

    return target.register_forward_hook(hook)


def _generate_condition(pipe: Any, torch: Any, preflight: Preflight, parameters: Mapping[str, Any], initial_latents: Any, condition: str, mp4_path: Path) -> dict[str, Any]:
    carrier_records: list[dict[str, Any]] = []
    schedule = None if condition == "OFF" else (schedule_a() if condition == "A" else schedule_b())
    handle = None if schedule is None else _carrier_hook(pipe, torch, schedule, preflight.config["carrier"], carrier_records)
    latent_before = tensor_sha256(initial_latents)
    try:
        result = pipe(
            prompt=parameters["prompt"],
            negative_prompt=parameters["negative_prompt"],
            num_frames=int(parameters["frame_count"]),
            height=int(parameters["height"]),
            width=int(parameters["width"]),
            guidance_scale=float(parameters["guidance_scale"]),
            num_inference_steps=int(parameters["inference_steps"]),
            latents=initial_latents.clone(),
        )
    finally:
        if handle is not None:
            handle.remove()
    if tensor_sha256(initial_latents) != latent_before:
        raise RC1GPUGenerationError("shared initial latent was mutated")
    if condition == "OFF" and carrier_records:
        raise RC1GPUGenerationError("OFF condition recorded carrier injection")
    if condition != "OFF" and len(carrier_records) != 16:
        raise RC1GPUGenerationError("carrier call count differs from frozen 8-step cond/uncond budget")
    frames = result.frames[0] if len(result.frames) == 1 else result.frames
    encode_saved_mp4(frames, mp4_path)
    codec_identity = _inspect_saved_mp4(mp4_path)
    features = extract_feature_matrix(decode_saved_mp4(mp4_path))
    return {"carrier_records": carrier_records, "features": features, "latent_sha256": latent_before, "codec_identity": codec_identity}


def run_gpu_generation(preflight: Preflight, output_dir: Path, command: Sequence[str]) -> Path:
    """Generate exactly two OFF/A/B groups and return the execution record path."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RC1GPUGenerationError("generation output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import diffusers
        import torch
        from diffusers import WanPipeline
    except Exception as exc:
        raise RC1GPUGenerationError("locked GPU dependencies are unavailable") from exc
    environment = _environment(torch, diffusers)
    pipe = WanPipeline.from_pretrained(
        preflight.config["model"]["id"],
        revision=preflight.config["model"]["revision"],
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    scheduler_config = dict(pipe.scheduler.config)
    environment["scheduler"] = {
        "class": type(pipe.scheduler).__name__,
        "config_sha256": sha256_bytes(canonical_json_bytes(scheduler_config)),
        "bound_to_model_revision": preflight.config["model"]["revision"],
    }
    environment["sampler"] = environment["scheduler"].copy()
    group_records: list[dict[str, Any]] = []
    for plan_group in preflight.plan["groups"]:
        parameters = expected_matched_parameters(preflight.config, plan_group)
        initial_latents = _prepare_initial_latents(pipe, torch, parameters)
        latent_sha = tensor_sha256(initial_latents)
        parameter_sha = sha256_bytes(canonical_json_bytes(parameters))
        condition_records: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        for condition in CONDITIONS:
            condition_dir = output_dir / plan_group["group_id"] / condition
            condition_dir.mkdir(parents=True, exist_ok=False)
            paths = {
                "video": condition_dir / "saved.mp4",
                "features": condition_dir / "features.json",
                "stdout": condition_dir / "stdout.txt",
                "stderr": condition_dir / "stderr.txt",
                "config": condition_dir / "frozen_config.json",
                "environment": condition_dir / "environment.json",
                "command": condition_dir / "command.json",
                "integrity": condition_dir / "integrity.json",
            }
            try:
                generated = _generate_condition(pipe, torch, preflight, parameters, initial_latents, condition, paths["video"])
                if generated["latent_sha256"] != latent_sha:
                    raise RC1GPUGenerationError("condition did not consume the shared latent identity")
                _write(paths["features"], canonical_json_bytes({"source": "single_saved_mp4_only", "video_sha256": sha256_file(paths["video"]), "features": generated["features"]}) + b"\n")
                _write(paths["stdout"], b"generation completed\n")
                _write(paths["stderr"], b"")
                _write(paths["config"], preflight.config_bytes)
                _write(paths["environment"], canonical_json_bytes(environment) + b"\n")
                _write(paths["command"], canonical_json_bytes(list(command)) + b"\n")
                _write(paths["integrity"], canonical_json_bytes({"initial_latent_sha256": latent_sha, "condition": condition, "carrier_records": generated["carrier_records"], "codec_identity": generated["codec_identity"]}) + b"\n")
                attempts.append({"condition": condition, "attempt_index": 0, "outcome": "success", "matched_parameters_sha256": parameter_sha})
            except Exception:
                attempts.append({"condition": condition, "attempt_index": 0, "outcome": "failed", "matched_parameters_sha256": parameter_sha})
                raise
            artifacts = {name: {"path": str(paths[name].relative_to(output_dir)), "sha256": sha256_file(paths[name])} for name in ARTIFACT_NAMES}
            condition_records.append({
                "condition": condition,
                "schedule_id": {"OFF": "NONE", "A": "A", "B": "B"}[condition],
                "carrier_enabled": condition != "OFF",
                "initial_latent_sha256": latent_sha,
                "matched_parameters_sha256": parameter_sha,
                "artifacts": artifacts,
            })
        group_records.append({
            "group_id": plan_group["group_id"],
            "content_grammar": plan_group["content_grammar"],
            "prompt": plan_group["prompt"],
            "prompt_sha256": parameters["prompt_sha256"],
            "seed": plan_group["seed"],
            "matched_parameters": parameters,
            "conditions": condition_records,
            "attempts": attempts,
        })
    record = {
        "schema_version": 1,
        "execution_schema": EXECUTION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": PLAN_RAW_SHA256,
        "prerequisite_identity": preflight.prerequisite.identity(),
        "synthetic_fixture": False,
        "groups": group_records,
    }
    record_path = output_dir / "execution.json"
    _write(record_path, json.dumps(record, indent=2, sort_keys=False, allow_nan=False).encode("utf-8") + b"\n")
    return record_path
