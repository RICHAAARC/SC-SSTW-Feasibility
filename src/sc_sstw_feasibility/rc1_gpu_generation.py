"""GPU generation interface for the frozen RC1 matched triplets.

This module is imported only after :func:`rc1_method_validation.preflight` has
validated source identity and the frozen L1-v2 selected-candidate package.  It
never selects a candidate or derives a threshold from fresh videos.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    EVIDENCE_PRODUCTION,
    EXECUTION_SCHEMA,
    FEATURE_CACHE_ATOL,
    FEATURE_CACHE_SCHEMA,
    FEATURE_EXTRACTOR_ID,
    EXTRACTOR_LIBRARY_PATH,
    GPU_LIBRARY_PATH,
    PLAN_RAW_SHA256,
    PRODUCTION_EVIDENCE_SCHEMA,
    PROTOCOL_ID,
    Preflight,
    canonical_json_bytes,
    command_artifact_payload,
    condition_config_payload,
    expected_matched_parameters,
    integrity_artifact_payload,
    schedule_a,
    schedule_b,
    sha256_bytes,
    sha256_file,
)


class RC1GPUGenerationError(RuntimeError):
    """Raised when generation cannot preserve the frozen matched-triplet contract."""


@dataclass(frozen=True, slots=True, init=False)
class GenerationReceipt:
    """Opaque, same-process generation capability; it is never serialized."""

    _seal: object
    _snapshot: bytes

    def __new__(cls) -> "GenerationReceipt":
        raise TypeError("GenerationReceipt is issued only by the generation module")

    def __reduce__(self) -> object:
        raise TypeError("GenerationReceipt cannot be serialized")


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    record_path: Path
    receipt: GenerationReceipt
    cpu_only_test_harness: bool


_ACTIVE_RECEIPTS: dict[object, Path] = {}


def _issue_generation_receipt(record_path: Path, snapshot: Mapping[str, Any]) -> GenerationReceipt:
    seal = object()
    receipt = object.__new__(GenerationReceipt)
    object.__setattr__(receipt, "_seal", seal)
    object.__setattr__(receipt, "_snapshot", canonical_json_bytes(snapshot))
    _ACTIVE_RECEIPTS[seal] = record_path.resolve()
    return receipt


def _consume_generation_receipt(receipt: object, record_path: Path) -> dict[str, Any]:
    """Consume a receipt exactly once before any execution-package read."""

    if type(receipt) is not GenerationReceipt:
        raise RC1GPUGenerationError("generation receipt has the wrong exact type")
    bound_path = _ACTIVE_RECEIPTS.pop(receipt._seal, None)
    if bound_path is None or bound_path != record_path.resolve():
        raise RC1GPUGenerationError("generation receipt is absent, replayed, or path-mismatched")
    try:
        snapshot = json.loads(receipt._snapshot)
    except Exception as exc:  # pragma: no cover - object is created only by the private factory
        raise RC1GPUGenerationError("generation receipt snapshot is malformed") from exc
    if not isinstance(snapshot, dict) or snapshot.get("completed") is not True:
        raise RC1GPUGenerationError("generation receipt is incomplete")
    return snapshot


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
    """Hash exact tensor bytes with bounded host memory."""

    import hashlib

    torch = __import__("torch")
    contiguous = tensor.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"shape": list(contiguous.shape), "dtype": str(contiguous.dtype)}))
    byte_view = contiguous.view(torch.uint8).reshape(-1)
    chunk_bytes = 8 * 1024 * 1024
    for start in range(0, int(byte_view.numel()), chunk_bytes):
        chunk = byte_view[start : start + chunk_bytes].to(device="cpu").contiguous().numpy().tobytes()
        digest.update(chunk)
    return digest.hexdigest()


def _tensor_identity(tensor: Any) -> dict[str, Any]:
    return {
        "sha256": tensor_sha256(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


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
    return {"container": "mp4", "codec_name": "h264", "pixel_format": "yuv420p", "width": 512, "height": 320, "fps": "8/1", "frame_count": 49, "decodable": True}


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
            "input_tensor_sha256": tensor_sha256(output),
            "modified_tensor_sha256": tensor_sha256(modified),
            "distinct_storage": int(modified.data_ptr()) != int(output.data_ptr()),
            "effective_relative_rms": float(energy["effective_relative_rms"]),
            "evidence_mode": EVIDENCE_PRODUCTION,
        })
        if abs(float(energy["effective_relative_rms"]) - float(carrier["target_relative_rms"])) > float(carrier["target_relative_rms_absolute_tolerance"]):
            raise RC1GPUGenerationError("carrier RMS differs from the frozen target")
        return modified

    return target.register_forward_hook(hook)


def _generate_condition(pipe: Any, torch: Any, preflight: Preflight, parameters: Mapping[str, Any], initial_latents: Any, condition: str, mp4_path: Path) -> dict[str, Any]:
    carrier_records: list[dict[str, Any]] = []
    schedule = None if condition == "OFF" else (schedule_a() if condition == "A" else schedule_b())
    handle = None if schedule is None else _carrier_hook(pipe, torch, schedule, preflight.config["carrier"], carrier_records)
    initial_identity = _tensor_identity(initial_latents)
    condition_latents = initial_latents.clone()
    condition_identity = _tensor_identity(condition_latents)
    if condition_identity != initial_identity:
        raise RC1GPUGenerationError("condition latent clone differs from the shared initial tensor")
    try:
        result = pipe(
            prompt=parameters["prompt"],
            negative_prompt=parameters["negative_prompt"],
            num_frames=int(parameters["frame_count"]),
            height=int(parameters["height"]),
            width=int(parameters["width"]),
            guidance_scale=float(parameters["guidance_scale"]),
            num_inference_steps=int(parameters["inference_steps"]),
            latents=condition_latents,
        )
    finally:
        if handle is not None:
            handle.remove()
    if _tensor_identity(initial_latents) != initial_identity:
        raise RC1GPUGenerationError("shared initial latent was mutated")
    if condition == "OFF" and carrier_records:
        raise RC1GPUGenerationError("OFF condition recorded carrier injection")
    if condition != "OFF" and len(carrier_records) != 16:
        raise RC1GPUGenerationError("carrier call count differs from frozen 8-step cond/uncond budget")
    frames = result.frames[0] if len(result.frames) == 1 else result.frames
    encode_saved_mp4(frames, mp4_path)
    codec_identity = _inspect_saved_mp4(mp4_path)
    features = extract_feature_matrix(decode_saved_mp4(mp4_path))
    return {
        "carrier_records": carrier_records,
        "features": features,
        "initial_latent_identity": initial_identity,
        "condition_latent_identity": condition_identity,
        "codec_identity": codec_identity,
        "video_saved_complete": mp4_path.is_file(),
        "video_sha256": sha256_file(mp4_path),
    }


def run_gpu_generation(
    preflight: Preflight,
    output_dir: Path,
    command: Sequence[str],
    *,
    _test_backend: Any | None = None,
) -> GenerationOutcome:
    """Generate two OFF/A/B groups and issue an opaque same-process receipt.

    ``_test_backend`` is dependency injection for the CPU-only control-flow
    harness.  The CLI never exposes it and receipts issued through it retain an
    explicit test-only boundary.
    """

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RC1GPUGenerationError("generation output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    cpu_only_test_harness = _test_backend is not None
    if _test_backend is None:
        try:
            import diffusers
            import torch
            from diffusers import WanPipeline
        except Exception as exc:
            raise RC1GPUGenerationError("locked GPU dependencies are unavailable") from exc
        runtime = _environment(torch, diffusers)
        pipe = WanPipeline.from_pretrained(
            preflight.config["model"]["id"],
            revision=preflight.config["model"]["revision"],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        observed_revision = getattr(pipe.config, "_commit_hash", None) or getattr(pipe.transformer.config, "_commit_hash", None)
        if observed_revision != preflight.config["model"]["revision"]:
            raise RC1GPUGenerationError("loaded model revision could not be observed or differs from the frozen revision")
        scheduler_config = dict(pipe.scheduler.config)
        scheduler_identity = {
            "class": type(pipe.scheduler).__name__,
            "config_sha256": sha256_bytes(canonical_json_bytes(scheduler_config)),
            "bound_to_model_revision": preflight.config["model"]["revision"],
        }
        environment = {
            "schema_version": 1,
            "artifact_schema": "sc_sstw_rc1_execution_environment_v1",
            "evidence_mode": EVIDENCE_PRODUCTION,
            "model_id": preflight.config["model"]["id"],
            "model_revision": preflight.config["model"]["revision"],
            "runtime": runtime,
            "scheduler": scheduler_identity,
            "sampler": scheduler_identity.copy(),
        }
        loaded_identity = {
            "model_id": preflight.config["model"]["id"],
            "loaded_revision": observed_revision,
            "scheduler": scheduler_identity,
            "dtype": str(next(pipe.transformer.parameters()).dtype),
            "device": str(pipe._execution_device),
        }
    else:
        pipe = None
        torch = None
        environment = _test_backend.load_environment(preflight)
        if not isinstance(environment, Mapping):
            raise RC1GPUGenerationError("CPU test backend returned an invalid environment")
        environment = dict(environment)
        loaded_identity = dict(_test_backend.loaded_identity(preflight, environment))
    if environment.get("model_id") != preflight.config["model"]["id"] or environment.get("model_revision") != preflight.config["model"]["revision"]:
        raise RC1GPUGenerationError("loaded model identity differs from the frozen revision")
    group_records: list[dict[str, Any]] = []
    receipt_groups: list[dict[str, Any]] = []
    for plan_group in preflight.plan["groups"]:
        parameters = expected_matched_parameters(preflight.config, plan_group)
        initial_latents = (
            _prepare_initial_latents(pipe, torch, parameters)
            if _test_backend is None
            else _test_backend.prepare_initial_latents(parameters)
        )
        initial_identity = (
            _tensor_identity(initial_latents)
            if _test_backend is None
            else dict(_test_backend.latent_identity(initial_latents))
        )
        if set(initial_identity) != {"sha256", "shape", "dtype"} or not isinstance(initial_identity["sha256"], str):
            raise RC1GPUGenerationError("initial latent identity is incomplete")
        latent_sha = initial_identity["sha256"]
        parameter_sha = sha256_bytes(canonical_json_bytes(parameters))
        condition_records: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        receipt_conditions: list[dict[str, Any]] = []
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
                generated = (
                    _generate_condition(pipe, torch, preflight, parameters, initial_latents, condition, paths["video"])
                    if _test_backend is None
                    else _test_backend.generate_condition(preflight, parameters, initial_latents, condition, paths["video"])
                )
                if generated.get("initial_latent_identity") != initial_identity or generated.get("condition_latent_identity") != initial_identity:
                    raise RC1GPUGenerationError("condition did not consume the shared latent identity")
                if generated.get("video_saved_complete") is not True or not paths["video"].is_file() or generated.get("video_sha256") != sha256_file(paths["video"]):
                    raise RC1GPUGenerationError("saved MP4 completion or identity differs from generation observation")
                _write(paths["features"], canonical_json_bytes({
                    "schema_version": 1,
                    "feature_cache_schema": FEATURE_CACHE_SCHEMA,
                    "evidence_mode": EVIDENCE_PRODUCTION,
                    "source": "recomputed_from_single_saved_mp4",
                    "video_sha256": sha256_file(paths["video"]),
                    "extractor_identity": {"id": FEATURE_EXTRACTOR_ID, "source_path": EXTRACTOR_LIBRARY_PATH, "source_sha256": preflight.source_hashes[EXTRACTOR_LIBRARY_PATH]},
                    "comparison": {"atol": FEATURE_CACHE_ATOL, "rtol": 0.0},
                    "features": generated["features"],
                }) + b"\n")
                _write(paths["stdout"], b"generation completed\n")
                _write(paths["stderr"], b"")
                saved_video_path = str(paths["video"].relative_to(output_dir))
                _write(paths["config"], canonical_json_bytes(condition_config_payload(preflight, plan_group, condition, latent_sha, saved_video_path, EVIDENCE_PRODUCTION)) + b"\n")
                _write(paths["environment"], canonical_json_bytes(environment) + b"\n")
                _write(paths["command"], canonical_json_bytes(command_artifact_payload(preflight, EVIDENCE_PRODUCTION)) + b"\n")
                _write(paths["integrity"], canonical_json_bytes(integrity_artifact_payload(preflight, plan_group, condition, latent_sha, EVIDENCE_PRODUCTION, generated["carrier_records"], generated["codec_identity"])) + b"\n")
                attempts.append({"condition": condition, "attempt_index": 0, "outcome": "success", "matched_parameters_sha256": parameter_sha})
            except Exception:
                attempts.append({"condition": condition, "attempt_index": 0, "outcome": "failed", "matched_parameters_sha256": parameter_sha})
                raise
            artifacts = {name: {"path": str(paths[name].relative_to(output_dir)), "sha256": sha256_file(paths[name])} for name in ARTIFACT_NAMES}
            receipt_conditions.append({
                "condition": condition,
                "initial_latent_identity": initial_identity,
                "condition_latent_identity": generated["condition_latent_identity"],
                "carrier_hook_effect": condition != "OFF",
                "carrier_records": generated["carrier_records"],
                "video_saved_complete": generated["video_saved_complete"],
                "video_sha256": generated["video_sha256"],
                "codec_identity": generated["codec_identity"],
                "artifacts": artifacts,
            })
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
        receipt_groups.append({
            "group_id": plan_group["group_id"],
            "initial_latent_identity": initial_identity,
            "conditions": receipt_conditions,
        })
    record = {
        "schema_version": 1,
        "execution_schema": EXECUTION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": PLAN_RAW_SHA256,
        "prerequisite_identity": preflight.prerequisite.identity(),
        "evidence_mode": EVIDENCE_PRODUCTION,
        "evidence_schema": PRODUCTION_EVIDENCE_SCHEMA,
        "groups": group_records,
    }
    record_path = output_dir / "execution.json"
    _write(record_path, json.dumps(record, indent=2, sort_keys=False, allow_nan=False).encode("utf-8") + b"\n")
    receipt_snapshot = {
        "schema_version": 1,
        "trust_boundary": "same_clean_source_runner_process_observed_generation_call",
        "completed": True,
        "cpu_only_test_harness": cpu_only_test_harness,
        "record_path": str(record_path.resolve()),
        "execution_record_sha256": sha256_file(record_path),
        "preflight": {
            "manifest_sha256": sha256_bytes(preflight.manifest_bytes),
            "config_sha256": sha256_bytes(preflight.config_bytes),
            "plan_sha256": sha256_bytes(preflight.plan_bytes),
            "prerequisite_identity": preflight.prerequisite.identity(),
            "source_head": preflight.source_state.head,
            "source_tree": preflight.source_state.tree,
            "source_hashes": preflight.source_hashes,
            "command_schema": preflight.manifest["command_schema"],
            "output_schema": preflight.manifest["output_schema"],
        },
        "implementation": {
            "generation_module_path": GPU_LIBRARY_PATH,
            "generation_module_sha256": preflight.source_hashes[GPU_LIBRARY_PATH],
            "extractor_path": EXTRACTOR_LIBRARY_PATH,
            "extractor_sha256": preflight.source_hashes[EXTRACTOR_LIBRARY_PATH],
        },
        "loaded_identity": loaded_identity,
        "environment": environment,
        "command_artifact": command_artifact_payload(preflight, EVIDENCE_PRODUCTION),
        "groups": receipt_groups,
    }
    receipt = _issue_generation_receipt(record_path, receipt_snapshot)
    return GenerationOutcome(record_path=record_path, receipt=receipt, cpu_only_test_harness=cpu_only_test_harness)
