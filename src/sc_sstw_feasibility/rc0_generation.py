"""Controlled same-process generation boundary for RC0 matched quartets.

The public production entry is :func:`run_rc0_generation`.  Receipt issuance is
closure-local and a receipt is usable exactly once by the validator in the same
Python process.  The CPU backend parameter is deliberately not exposed by the
CLI and can only produce a test-only provenance class.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .learned_observation import decode_saved_mp4, encode_saved_mp4, extract_feature_matrix
from .rc0_causal_localization_v2 import CONDITION_ORDER, schedule_a, schedule_b
from .rc0_minimal_implementation import (
    ARTIFACT_NAMES,
    EVIDENCE_CPU_HARNESS,
    EVIDENCE_PRODUCTION,
    EXECUTION_SCHEMA,
    EXTRACTOR_IDENTITY,
    FEATURE_CACHE_ATOL,
    FEATURE_CACHE_SCHEMA,
    GENERATION_PATH,
    IMPLEMENTATION_PATH,
    PLAN_RAW_SHA256,
    PROTOCOL_ID,
    PROVENANCE_CPU_HARNESS,
    PROVENANCE_PRODUCTION,
    Preflight,
    canonical_json_bytes,
    command_artifact,
    condition_config,
    diagnostic_carrier_config,
    expected_matched_parameters,
    sha256_bytes,
    sha256_file,
)


class RC0GenerationError(RuntimeError):
    """A controlled generation integrity failure."""


@dataclass(frozen=True, slots=True, init=False)
class GenerationReceipt:
    """Opaque one-use same-process generation capability."""

    _seal: object
    _snapshot: bytes

    def __new__(cls) -> "GenerationReceipt":
        raise TypeError("GenerationReceipt is issued only inside run_rc0_generation")

    def __reduce__(self) -> object:
        raise TypeError("GenerationReceipt cannot be copied or serialized")


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    record_path: Path
    receipt: GenerationReceipt
    cpu_only_test_harness: bool


def _ordered_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _close_receipt_authority() -> tuple[Any, Any]:
    authority = object()
    active: dict[int, dict[str, Any]] = {}

    def issue(record_path: Path, snapshot: Mapping[str, Any], provenance_class: str) -> GenerationReceipt:
        if provenance_class not in {PROVENANCE_PRODUCTION, PROVENANCE_CPU_HARNESS}:
            raise RC0GenerationError("receipt provenance is not a controlled branch")
        snapshot_bytes = _ordered_json_bytes(snapshot)
        seal = object()
        receipt = object.__new__(GenerationReceipt)
        object.__setattr__(receipt, "_seal", seal)
        object.__setattr__(receipt, "_snapshot", snapshot_bytes)
        active[id(receipt)] = {
            "receipt": receipt,
            "path": record_path.resolve(),
            "authority": authority,
            "seal": seal,
            "snapshot": snapshot_bytes,
            "digest": sha256_bytes(snapshot_bytes),
            "provenance_class": provenance_class,
            "completed": True,
        }
        return receipt

    def consume(receipt: object, record_path: Path) -> dict[str, Any]:
        if type(receipt) is not GenerationReceipt:
            raise RC0GenerationError("generation receipt has the wrong exact type")
        entry = active.pop(id(receipt), None)
        if (
            entry is None
            or entry["receipt"] is not receipt
            or entry["path"] != record_path.resolve()
            or entry["authority"] is not authority
            or entry["seal"] is not receipt._seal
            or entry["snapshot"] != receipt._snapshot
            or entry["digest"] != sha256_bytes(receipt._snapshot)
            or entry["completed"] is not True
        ):
            raise RC0GenerationError("generation receipt is absent, copied, replayed, tampered, or path-mismatched")
        try:
            snapshot = json.loads(entry["snapshot"])
        except Exception as exc:  # pragma: no cover - local canonical data only
            raise RC0GenerationError("generation receipt snapshot is malformed") from exc
        if (
            type(snapshot) is not dict
            or snapshot.get("completed") is not True
            or snapshot.get("provenance_class") != entry["provenance_class"]
        ):
            raise RC0GenerationError("generation receipt provenance is incomplete")
        snapshot["authority_provenance_class"] = entry["provenance_class"]
        return snapshot

    def bind(control_flow: Any) -> Any:
        def run_rc0_generation(
            preflight_result: Preflight,
            output_dir: Path,
            command: Sequence[str],
            *,
            _test_backend: Any | None = None,
        ) -> GenerationOutcome:
            return control_flow(
                preflight_result,
                output_dir,
                command,
                _test_backend=_test_backend,
                _authority_issue=issue,
            )

        run_rc0_generation.__name__ = "run_rc0_generation"
        run_rc0_generation.__qualname__ = "run_rc0_generation"
        return run_rc0_generation

    return bind, consume


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


def _tensor_bytes(value: Any) -> tuple[bytes, list[int], str]:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return contiguous.tobytes(order="C"), list(contiguous.shape), str(contiguous.dtype)
    try:
        torch = __import__("torch")
        if isinstance(value, torch.Tensor):
            contiguous = value.detach().contiguous()
            byte_view = contiguous.view(torch.uint8).reshape(-1).to(device="cpu").contiguous().numpy()
            return byte_view.tobytes(), list(contiguous.shape), str(contiguous.dtype)
    except Exception:
        pass
    raise RC0GenerationError("initial latent must be a NumPy array or torch tensor")


def tensor_identity(value: Any) -> dict[str, Any]:
    raw, shape, dtype = _tensor_bytes(value)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"shape": shape, "dtype": dtype}))
    digest.update(raw)
    return {"sha256": digest.hexdigest(), "shape": shape, "dtype": dtype}


def _clone(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    clone = getattr(value, "clone", None)
    if not callable(clone):
        raise RC0GenerationError("initial latent does not support an independent clone")
    return clone()


def _storage_pointer(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.__array_interface__["data"][0])
    data_ptr = getattr(value, "data_ptr", None)
    if not callable(data_ptr):
        raise RC0GenerationError("latent storage identity is unavailable")
    return int(data_ptr())


def _storage_token(value: Any) -> str:
    return hashlib.sha256(f"rc0-storage:{_storage_pointer(value)}".encode("ascii")).hexdigest()


def _environment(torch: Any, diffusers: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RC0GenerationError("CUDA runtime is unavailable")
    if not str(torch.__version__).split(".", 1)[0] == "2" or not torch.cuda.is_bf16_supported():
        raise RC0GenerationError("torch major version or CUDA BF16 capability is incompatible")
    versions: dict[str, str] = {}
    for package, expected in LOCKED_GPU_PACKAGES.items():
        module = __import__(package)
        observed = str(module.__version__)
        if observed != expected:
            raise RC0GenerationError(f"dependency mismatch for {package}: {observed}")
        versions[package] = observed
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "diffusers": str(diffusers.__version__),
        "cuda": str(torch.version.cuda),
        "gpu": str(torch.cuda.get_device_name(0)),
        "versions": versions,
    }


def _inspect_saved_mp4(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RC0GenerationError("ffprobe is required for saved-MP4 verification")
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames:format=format_name",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RC0GenerationError("saved MP4 cannot be independently probed") from exc
    streams = payload.get("streams")
    format_payload = payload.get("format")
    format_name = format_payload.get("format_name") if type(format_payload) is dict else None
    expected = {"codec_name": "h264", "pix_fmt": "yuv420p", "width": 512, "height": 320, "avg_frame_rate": "8/1", "nb_frames": "49"}
    if type(streams) is not list or len(streams) != 1 or type(format_name) is not str or "mp4" not in format_name.split(",") or {key: streams[0].get(key) for key in expected} != expected:
        raise RC0GenerationError("saved-MP4 container, codec, geometry, FPS, or frame count changed")
    return {"container": "mp4", "codec_name": "h264", "pixel_format": "yuv420p", "width": 512, "height": 320, "fps": "8/1", "frame_count": 49, "decodable": True}


def _prepare_initial_latent(pipe: Any, torch: Any, parameters: Mapping[str, Any]) -> Any:
    generator = torch.Generator(device="cuda").manual_seed(int(parameters["seed"]))
    try:
        latent = pipe.prepare_latents(
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
        raise RC0GenerationError("Wan initial-latent preparation contract changed") from exc
    if not torch.isfinite(latent).all().item():
        raise RC0GenerationError("initial latent contains non-finite values")
    return latent


def _load_production_pipeline(wan_pipeline: Any, torch: Any, resolved_snapshot: Path) -> Any:
    """Load the exact local Wan snapshot with the single frozen offload policy."""

    pipe = wan_pipeline.from_pretrained(
        str(resolved_snapshot),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling(
        tile_sample_min_height=192,
        tile_sample_min_width=192,
        tile_sample_stride_height=128,
        tile_sample_stride_width=128,
    )
    if not str(pipe._execution_device).startswith("cuda"):
        raise RC0GenerationError("Wan CPU offload did not retain a CUDA execution device")
    return pipe


def construct_final_latent_relation_residual(
    final_latent: Any,
    schedule: Sequence[Sequence[float]],
    carrier: Mapping[str, Any],
) -> tuple[Any, dict[str, float]]:
    """Apply the single frozen low-frequency relation residual."""

    required_shape = list(carrier["required_shape"])
    if list(final_latent.shape) != required_shape or len(schedule) != 13 or any(len(point) != 2 for point in schedule):
        raise RC0GenerationError("final latent shape or schedule geometry changed")
    epsilon = float(carrier["numeric_epsilon"])
    target = float(carrier["target_relative_rms"])
    if isinstance(final_latent, np.ndarray):
        if not np.isfinite(final_latent).all():
            raise RC0GenerationError("final latent contains non-finite values")
        batch, channels, times, height, width = required_shape
        x = np.arange(width, dtype=np.float32)
        y = np.arange(height, dtype=np.float32)
        phi_x = np.cos(2.0 * np.pi * x / width).astype(np.float32)[None, :]
        phi_y = np.cos(2.0 * np.pi * y / height).astype(np.float32)[:, None]
        raw = np.zeros((batch, channels, times, height, width), dtype=np.float32)
        for time_index, point in enumerate(schedule):
            raw[:, :8, time_index, :, :] = np.float32(point[0]) * phi_x
            raw[:, 8:, time_index, :, :] = np.float32(point[1]) * phi_y
        centered = raw - np.mean(raw, dtype=np.float64)
        raw_rms = float(np.sqrt(np.mean(np.square(centered, dtype=np.float64))))
        if not np.isfinite(raw_rms) or raw_rms <= epsilon:
            raise RC0GenerationError("final-latent relation basis is degenerate")
        unit = centered / np.float32(raw_rms)
        latent_float = final_latent.astype(np.float32, copy=False)
        latent_rms = float(np.sqrt(np.mean(np.square(latent_float, dtype=np.float64))))
        if not np.isfinite(latent_rms) or latent_rms <= epsilon:
            raise RC0GenerationError("final latent RMS is degenerate")
        delta = unit * np.float32(target * latent_rms)
        modified = final_latent + delta.astype(final_latent.dtype, copy=False)
        actual_delta = modified.astype(np.float32) - latent_float
        effective = float(np.sqrt(np.mean(np.square(actual_delta, dtype=np.float64))) / latent_rms)
        unit_mean = float(np.mean(unit, dtype=np.float64))
        unit_rms = float(np.sqrt(np.mean(np.square(unit, dtype=np.float64))))
    else:
        try:
            torch = __import__("torch")
        except Exception as exc:  # pragma: no cover - production dependency
            raise RC0GenerationError("torch is unavailable for final-latent carrier") from exc
        if not isinstance(final_latent, torch.Tensor) or not torch.isfinite(final_latent).all().item():
            raise RC0GenerationError("final latent must be a finite torch tensor")
        batch, channels, times, height, width = required_shape
        x = torch.arange(width, device=final_latent.device, dtype=torch.float32)
        y = torch.arange(height, device=final_latent.device, dtype=torch.float32)
        phi_x = torch.cos(2.0 * torch.pi * x / width).view(1, width)
        phi_y = torch.cos(2.0 * torch.pi * y / height).view(height, 1)
        raw = torch.zeros((batch, channels, times, height, width), device=final_latent.device, dtype=torch.float32)
        for time_index, point in enumerate(schedule):
            raw[:, :8, time_index, :, :] = float(point[0]) * phi_x
            raw[:, 8:, time_index, :, :] = float(point[1]) * phi_y
        centered = raw - raw.mean()
        raw_rms_tensor = centered.square().mean().sqrt()
        raw_rms = float(raw_rms_tensor.item())
        if not np.isfinite(raw_rms) or raw_rms <= epsilon:
            raise RC0GenerationError("final-latent relation basis is degenerate")
        unit = centered / raw_rms_tensor
        latent_float = final_latent.float()
        latent_rms_tensor = latent_float.square().mean().sqrt()
        latent_rms = float(latent_rms_tensor.item())
        if not np.isfinite(latent_rms) or latent_rms <= epsilon:
            raise RC0GenerationError("final latent RMS is degenerate")
        delta = unit * (target * latent_rms_tensor)
        modified = final_latent + delta.to(dtype=final_latent.dtype)
        actual_delta = modified.float() - latent_float
        effective = float((actual_delta.square().mean().sqrt() / latent_rms_tensor).item())
        unit_mean = float(unit.mean().item())
        unit_rms = float(unit.square().mean().sqrt().item())
    if (
        abs(unit_mean) > float(carrier["zero_mean_absolute_tolerance"])
        or abs(unit_rms - 1.0) > float(carrier["unit_rms_absolute_tolerance"])
        or abs(effective - target) > float(carrier["target_relative_rms_absolute_tolerance"])
    ):
        raise RC0GenerationError("final-latent carrier normalization missed its frozen tolerance")
    return modified, {
        "residual_global_mean": unit_mean,
        "residual_global_rms": unit_rms,
        "effective_relative_rms": effective,
    }


def _decode_final_wan_latent(pipe: Any, torch: Any, final_latent: Any) -> Any:
    latents = final_latent.to(pipe.vae.dtype)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    video = pipe.vae.decode(latents / latents_std + latents_mean, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type="np")


def _run_production_condition(pipe: Any, torch: Any, preflight_result: Preflight, parameters: Mapping[str, Any], latent: Any, condition: str, video_path: Path) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    schedule = None if condition.startswith("OFF_") else (schedule_a() if condition == "A" else schedule_b())
    result = pipe(
        prompt=parameters["prompt"],
        negative_prompt=parameters["negative_prompt"],
        num_frames=int(parameters["frame_count"]),
        height=int(parameters["height"]),
        width=int(parameters["width"]),
        guidance_scale=float(parameters["guidance_scale"]),
        num_inference_steps=int(parameters["inference_steps"]),
        latents=latent,
        output_type="latent",
    )
    final_latent = result.frames
    if schedule is not None:
        carrier = diagnostic_carrier_config(preflight_result)["carrier"]
        modified, metrics = construct_final_latent_relation_residual(final_latent, schedule, carrier)
        primitives.append(
            {
                "call_index": 0,
                "injection_stage": carrier["injection_stage"],
                "latent_layout": carrier["tensor_layout"],
                "schedule_point_count": 13,
                "output_shape": list(final_latent.shape),
                "output_dtype": str(final_latent.dtype),
                "input_tensor_sha256": tensor_identity(final_latent)["sha256"],
                "modified_tensor_sha256": tensor_identity(modified)["sha256"],
                "distinct_storage": _storage_pointer(modified) != _storage_pointer(final_latent),
                **metrics,
            }
        )
        final_latent = modified
    decoded = _decode_final_wan_latent(pipe, torch, final_latent)
    frames = decoded[0] if len(decoded) == 1 else decoded
    encode_saved_mp4(frames, video_path)
    return primitives


def _finalize_events(primitives: Any, condition: str, evidence_mode: str) -> list[dict[str, Any]]:
    if type(primitives) is not list:
        raise RC0GenerationError("generation backend event primitives must be an array")
    if condition.startswith("OFF_"):
        if primitives:
            raise RC0GenerationError("OFF condition produced carrier event primitives")
        return []
    if len(primitives) != 1:
        raise RC0GenerationError("A/B final-latent carrier event budget is not exactly one")
    schedule = schedule_a() if condition == "A" else schedule_b()
    schedule_sha = sha256_bytes(canonical_json_bytes(schedule))
    expected_keys = {
        "call_index",
        "injection_stage",
        "latent_layout",
        "schedule_point_count",
        "output_shape",
        "output_dtype",
        "input_tensor_sha256",
        "modified_tensor_sha256",
        "distinct_storage",
        "residual_global_mean",
        "residual_global_rms",
        "effective_relative_rms",
    }
    events: list[dict[str, Any]] = []
    for index, primitive in enumerate(primitives):
        if type(primitive) is not dict or set(primitive) != expected_keys:
            raise RC0GenerationError("generation backend returned an expanded or malformed event primitive")
        if primitive["call_index"] != index:
            raise RC0GenerationError("carrier event call order changed")
        events.append(
            {
                "condition": condition,
                "call_index": index,
                "injection_stage": primitive["injection_stage"],
                "latent_layout": primitive["latent_layout"],
                "schedule_point_count": primitive["schedule_point_count"],
                "schedule_sha256": schedule_sha,
                "output_shape": primitive["output_shape"],
                "output_dtype": primitive["output_dtype"],
                "input_tensor_sha256": primitive["input_tensor_sha256"],
                "modified_tensor_sha256": primitive["modified_tensor_sha256"],
                "distinct_storage": primitive["distinct_storage"],
                "residual_global_mean": primitive["residual_global_mean"],
                "residual_global_rms": primitive["residual_global_rms"],
                "effective_relative_rms": primitive["effective_relative_rms"],
                "evidence_mode": evidence_mode,
            }
        )
    return events


def _run_generation_control_flow(
    preflight_result: Preflight,
    output_dir: Path,
    command: Sequence[str],
    *,
    _test_backend: Any | None,
    _authority_issue: Any,
) -> GenerationOutcome:
    if output_dir.exists() or output_dir.is_symlink():
        raise RC0GenerationError("generation output directory must not already exist")
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance_class = PROVENANCE_PRODUCTION if _test_backend is None else PROVENANCE_CPU_HARNESS
    evidence_mode = EVIDENCE_PRODUCTION if _test_backend is None else EVIDENCE_CPU_HARNESS
    cpu_only = _test_backend is not None

    if _test_backend is None:
        try:
            import diffusers
            import torch
            from diffusers import WanPipeline
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RC0GenerationError("locked GPU dependencies are unavailable") from exc
        runtime = _environment(torch, diffusers)
        frozen_revision = preflight_result.config["model"]["revision"]
        snapshot_path = Path(
            snapshot_download(
                repo_id=preflight_result.config["model"]["id"],
                revision=frozen_revision,
                local_files_only=True,
            )
        )
        resolved_snapshot = snapshot_path.resolve(strict=True)
        if (
            snapshot_path.is_symlink()
            or not snapshot_path.is_dir()
            or snapshot_path.name != frozen_revision
            or resolved_snapshot.name != frozen_revision
        ):
            raise RC0GenerationError("local model snapshot differs from the frozen revision")
        pipe = _load_production_pipeline(WanPipeline, torch, resolved_snapshot)
        vae_memory_policy = {
            "vae_use_tiling": bool(pipe.vae.use_tiling),
            "tile_sample_min_height": int(pipe.vae.tile_sample_min_height),
            "tile_sample_min_width": int(pipe.vae.tile_sample_min_width),
            "tile_sample_stride_height": int(pipe.vae.tile_sample_stride_height),
            "tile_sample_stride_width": int(pipe.vae.tile_sample_stride_width),
            "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        }
        if vae_memory_policy != {
            "vae_use_tiling": True,
            "tile_sample_min_height": 192,
            "tile_sample_min_width": 192,
            "tile_sample_stride_height": 128,
            "tile_sample_stride_width": 128,
            "allocator": "expandable_segments:True",
        }:
            raise RC0GenerationError("Wan VAE memory policy differs from the frozen diagnostic setting")
        observed_revision = frozen_revision
        scheduler = {
            "class": type(pipe.scheduler).__name__,
            "config_sha256": sha256_bytes(canonical_json_bytes(dict(pipe.scheduler.config))),
            "bound_to_model_revision": observed_revision,
        }
        loaded_identity = {
            "model_id": preflight_result.config["model"]["id"],
            "loaded_revision": observed_revision,
            "scheduler": scheduler,
            "dtype": str(next(pipe.transformer.parameters()).dtype),
            "device": str(pipe._execution_device),
            "vae_memory_policy": vae_memory_policy,
            "cpu_only_test_harness": False,
        }
    else:
        pipe = None
        torch = None
        runtime = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": "cpu_test_harness",
            "diffusers": "cpu_test_harness",
            "cuda": "unavailable_cpu_test_harness",
            "gpu": "cpu_test_harness",
            "versions": {"adapter": "cpu_test_harness_v1"},
        }
        scheduler = {
            "class": "CPUControlFlowHarnessScheduler",
            "config_sha256": sha256_bytes(canonical_json_bytes({"kind": "cpu_test_harness_v1"})),
            "bound_to_model_revision": preflight_result.config["model"]["revision"],
        }
        loaded_identity = {
            "model_id": preflight_result.config["model"]["id"],
            "loaded_revision": preflight_result.config["model"]["revision"],
            "scheduler": scheduler,
            "dtype": "cpu_test_harness_numpy_float32",
            "device": "cpu",
            "cpu_only_test_harness": True,
        }
    environment = {
        "schema_version": 1,
        "artifact_schema": "sc_sstw_rc0_execution_environment_v1",
        "evidence_mode": evidence_mode,
        "model_id": preflight_result.config["model"]["id"],
        "model_revision": preflight_result.config["model"]["revision"],
        "runtime": runtime,
        "scheduler": scheduler,
        "sampler": dict(scheduler),
        "loaded_identity": loaded_identity,
        "codec": {"container": "mp4", "codec": "h264", "pixel_format": "yuv420p", "fps": 8},
        "cpu_only_test_harness": cpu_only,
    }

    group_records: list[dict[str, Any]] = []
    receipt_groups: list[dict[str, Any]] = []
    global_attempt = 1
    for plan_group in preflight_result.plan["groups"]:
        parameters = expected_matched_parameters(preflight_result, plan_group)
        initial = _prepare_initial_latent(pipe, torch, parameters) if _test_backend is None else _test_backend.prepare_initial_latent(parameters)
        initial_identity = tensor_identity(initial)
        source_storage = _storage_pointer(initial)
        clones = [_clone(initial) for _condition in CONDITION_ORDER]
        if any(tensor_identity(clone) != initial_identity for clone in clones):
            raise RC0GenerationError("one or more pre-created condition clones differ from the source latent")
        clone_pointers = [_storage_pointer(clone) for clone in clones]
        if source_storage in clone_pointers or len(set(clone_pointers)) != 4:
            raise RC0GenerationError("condition latent clones do not have four independent storages")
        clone_identities = [
            {
                "condition": condition,
                **initial_identity,
                "distinct_storage": True,
                "storage_token": _storage_token(clone),
            }
            for condition, clone in zip(CONDITION_ORDER, clones, strict=True)
        ]
        parameter_sha = sha256_bytes(canonical_json_bytes(parameters))
        condition_records: list[dict[str, Any]] = []
        attempt_records: list[dict[str, Any]] = []
        receipt_conditions: list[dict[str, Any]] = []
        for condition_index, (condition, condition_latent) in enumerate(zip(CONDITION_ORDER, clones, strict=True)):
            condition_dir = output_dir / plan_group["group_id"] / condition
            condition_dir.mkdir(parents=True, exist_ok=False)
            paths = {
                "video": condition_dir / "saved.mp4",
                "features": condition_dir / "features.json",
                "config": condition_dir / "frozen_config.json",
                "environment": condition_dir / "environment.json",
                "command": condition_dir / "command.json",
                "integrity": condition_dir / "integrity.json",
                "stdout": condition_dir / "stdout.txt",
                "stderr": condition_dir / "stderr.txt",
            }
            attempt = {
                "attempt_index": global_attempt,
                "group_id": plan_group["group_id"],
                "condition": condition,
                "started": True,
                "completed": False,
                "outcome": "failed",
                "retry_index": 0,
                "matched_parameters_sha256": parameter_sha,
            }
            try:
                if tensor_identity(condition_latent) != initial_identity or tensor_identity(initial) != initial_identity:
                    raise RC0GenerationError("latent bytes changed before condition execution")
                if _test_backend is None:
                    primitives = _run_production_condition(pipe, torch, preflight_result, parameters, condition_latent, condition, paths["video"])
                else:
                    returned = _test_backend.generate_condition(
                        parameters,
                        condition_latent,
                        condition,
                        None if condition.startswith("OFF_") else (schedule_a() if condition == "A" else schedule_b()),
                        paths["video"],
                    )
                    if type(returned) is not dict or tuple(returned) != ("carrier_event_primitives",):
                        raise RC0GenerationError("CPU test backend may return only carrier_event_primitives")
                    primitives = returned["carrier_event_primitives"]
                events = _finalize_events(primitives, condition, evidence_mode)
                if tensor_identity(initial) != initial_identity or any(tensor_identity(clone) != initial_identity for clone in clones):
                    raise RC0GenerationError("source or condition latent bytes changed during generation")
                if not paths["video"].is_file() or paths["video"].is_symlink():
                    raise RC0GenerationError("generation did not save a regular MP4")
                codec_identity = _inspect_saved_mp4(paths["video"])
                frames = decode_saved_mp4(paths["video"])
                features = extract_feature_matrix(frames)
                video_sha = sha256_file(paths["video"])
                feature_payload = {
                    "schema_version": 1,
                    "feature_cache_schema": FEATURE_CACHE_SCHEMA,
                    "evidence_mode": evidence_mode,
                    "source": "recomputed_from_single_saved_mp4",
                    "video_sha256": video_sha,
                    "extractor_identity": {"id": "sc_sstw_frozen_public_30d_v1", **EXTRACTOR_IDENTITY},
                    "comparison": {"atol": FEATURE_CACHE_ATOL, "rtol": 0.0},
                    "features": features,
                }
                saved_video_path = str(paths["video"].relative_to(output_dir))
                _write(paths["features"], _ordered_json_bytes(feature_payload) + b"\n")
                _write(paths["config"], _ordered_json_bytes(condition_config(preflight_result, plan_group, condition, initial_identity, saved_video_path, evidence_mode)) + b"\n")
                _write(paths["environment"], _ordered_json_bytes(environment) + b"\n")
                _write(paths["command"], _ordered_json_bytes(command_artifact(preflight_result, evidence_mode)) + b"\n")
                integrity = {
                    "schema_version": 1,
                    "artifact_schema": "sc_sstw_rc0_condition_integrity_v1",
                    "evidence_mode": evidence_mode,
                    "group_id": plan_group["group_id"],
                    "condition": condition,
                    "initial_latent_identity": initial_identity,
                    "clone_identity": clone_identities[condition_index],
                    "carrier_events": events,
                    "codec_identity": codec_identity,
                    "video_saved_complete": True,
                    "video_sha256": video_sha,
                }
                _write(paths["integrity"], _ordered_json_bytes(integrity) + b"\n")
                _write(paths["stdout"], b"generation attempt completed\n")
                _write(paths["stderr"], b"")
                attempt.update({"completed": True, "outcome": "success"})
            except BaseException:
                attempt_records.append(attempt)
                raise
            artifacts = {
                name: {"path": str(paths[name].relative_to(output_dir)), "sha256": sha256_file(paths[name])}
                for name in ARTIFACT_NAMES
            }
            receipt_conditions.append(
                {
                    "condition": condition,
                    "clone_identity": clone_identities[condition_index],
                    "carrier_events": events,
                    "video_saved_complete": True,
                    "video_sha256": video_sha,
                    "codec_identity": codec_identity,
                    "artifacts": artifacts,
                }
            )
            condition_records.append(
                {
                    "condition": condition,
                    "schedule_id": "NONE" if condition.startswith("OFF_") else condition,
                    "carrier_enabled": not condition.startswith("OFF_"),
                    "clone_identity": clone_identities[condition_index],
                    "matched_parameters_sha256": parameter_sha,
                    "artifacts": artifacts,
                }
            )
            attempt_records.append(attempt)
            global_attempt += 1
        group_records.append(
            {
                "group_id": plan_group["group_id"],
                "content_grammar": plan_group["content_grammar"],
                "prompt": plan_group["prompt"],
                "prompt_sha256": plan_group["prompt_sha256"],
                "seed": plan_group["seed"],
                "matched_parameters": parameters,
                "initial_latent_identity": initial_identity,
                "clone_identities": clone_identities,
                "conditions": condition_records,
                "attempts": attempt_records,
            }
        )
        receipt_groups.append(
            {
                "group_id": plan_group["group_id"],
                "initial_latent_identity": initial_identity,
                "clone_identities": clone_identities,
                "conditions": receipt_conditions,
            }
        )
    if global_attempt != 9:
        raise RC0GenerationError("exact eight-attempt budget was not consumed")

    evidence_schema = f"sc_sstw_rc0_{evidence_mode}_v1"
    record = {
        "schema_version": 1,
        "execution_schema": EXECUTION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": PLAN_RAW_SHA256,
        "evidence_mode": evidence_mode,
        "evidence_schema": evidence_schema,
        "groups": group_records,
    }
    record_path = output_dir / "execution.json"
    _write(record_path, json.dumps(record, indent=2, sort_keys=False, allow_nan=False).encode("utf-8") + b"\n")
    snapshot = {
        "schema_version": 1,
        "trust_boundary": "same_clean_source_runner_process_observed_generation_call",
        "completed": True,
        "provenance_class": provenance_class,
        "cpu_only_test_harness": cpu_only,
        "record_path": str(record_path.resolve()),
        "execution_record_sha256": sha256_file(record_path),
        "preflight": {
            "manifest_sha256": sha256_bytes(preflight_result.manifest_bytes),
            "config_sha256": sha256_bytes(preflight_result.config_bytes),
            "plan_sha256": sha256_bytes(preflight_result.plan_bytes),
            "protocol_sha256": sha256_bytes(preflight_result.protocol_bytes),
            "source_head": preflight_result.source_state.head,
            "source_tree": preflight_result.source_state.tree,
            "source_hashes": preflight_result.source_hashes,
            "command_schema": preflight_result.manifest["command_schema"],
            "output_schema": preflight_result.manifest["output_schema"],
        },
        "implementation": {
            "generation_module_path": GENERATION_PATH,
            "generation_module_sha256": preflight_result.source_hashes[GENERATION_PATH],
            "implementation_module_path": IMPLEMENTATION_PATH,
            "implementation_module_sha256": preflight_result.source_hashes[IMPLEMENTATION_PATH],
            "extractor": EXTRACTOR_IDENTITY,
        },
        "loaded_identity": loaded_identity,
        "environment": environment,
        "command_artifact": command_artifact(preflight_result, evidence_mode),
        "groups": receipt_groups,
    }
    receipt = _authority_issue(record_path, snapshot, provenance_class)
    return GenerationOutcome(record_path=record_path, receipt=receipt, cpu_only_test_harness=cpu_only)


_bind_generation_control_flow, consume_generation_receipt = _close_receipt_authority()
run_rc0_generation = _bind_generation_control_flow(_run_generation_control_flow)
del _bind_generation_control_flow
del _close_receipt_authority
del _run_generation_control_flow
