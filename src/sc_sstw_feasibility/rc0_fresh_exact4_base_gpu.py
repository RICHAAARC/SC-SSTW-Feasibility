"""Minimal exact-four carrier-free Wan base generation for Phase B."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .learned_observation import decode_saved_mp4, encode_saved_mp4
from .rc0_pixel_chroma_fast_cpu import probe_mp4


SCHEMA = "sc_sstw_rc0_fresh_exact4_base_gpu_v1"
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
GROUP_ORDER = ("liquid_mosaic", "clockwork_escapement", "steam_fins", "magnetic_filings")
SEEDS = (53011, 53012, 53013, 53014)
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"


class FreshBaseDiagnosticError(RuntimeError):
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


def load_and_validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or config.get("diagnostic_class") != DIAGNOSTIC_CLASS:
        raise FreshBaseDiagnosticError("fresh exact4 config identity mismatch")
    if config.get("attempt_order") != list(GROUP_ORDER) or config.get("attempt_budget") != 4 or config.get("retry_policy") != "no_retry_no_replacement_no_additional_sample":
        raise FreshBaseDiagnosticError("attempt identity or retry policy changed")
    groups = config.get("groups")
    if not isinstance(groups, list) or [item.get("group_id") for item in groups] != list(GROUP_ORDER) or tuple(item.get("seed") for item in groups) != SEEDS:
        raise FreshBaseDiagnosticError("fresh group identity changed")
    for item in groups:
        if sha256_bytes(item["prompt"].encode("utf-8")) != item.get("prompt_sha256"):
            raise FreshBaseDiagnosticError("prompt digest changed")
    if config.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise FreshBaseDiagnosticError("model identity changed")
    generation = config.get("generation", {})
    expected_generation = (8, 5.0, 320, 512, 49, 8, "torch.bfloat16", "none", "none", "official_pipeline_default_output_no_manual_latent_decode")
    actual_generation = (generation.get("inference_steps"), generation.get("guidance_scale"), generation.get("height"), generation.get("width"), generation.get("frame_count"), generation.get("fps"), generation.get("dtype"), generation.get("carrier"), generation.get("hook"), generation.get("decode_path"))
    if actual_generation != expected_generation:
        raise FreshBaseDiagnosticError("generation parameter changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise FreshBaseDiagnosticError("diagnostic boundary changed")
    return config


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _normalize_pipeline_frames(result: Any) -> list[Any]:
    raw = getattr(result, "frames", None)
    if raw is None or len(raw) == 0:
        raise FreshBaseDiagnosticError("Wan pipeline returned no decoded frames")
    if len(raw) == 1:
        first = raw[0]
        first_shape = getattr(first, "shape", None)
        frames = list(first) if isinstance(first, list) or (first_shape is not None and len(first_shape) == 4) else list(raw)
    else:
        frames = list(raw)
    if len(frames) != 49:
        raise FreshBaseDiagnosticError("Wan pipeline did not return exactly 49 decoded frames")
    return frames


def generate_official_base_frames(pipe: Any, torch: Any, item: Mapping[str, Any], generation: Mapping[str, Any]) -> list[Any]:
    """Call only the official no-carrier decoded-video pipeline path."""

    generator = torch.Generator(device="cuda").manual_seed(int(item["seed"]))
    result = pipe(
        prompt=item["prompt"],
        negative_prompt=generation["negative_prompt"],
        num_frames=49,
        height=320,
        width=512,
        guidance_scale=5.0,
        num_inference_steps=8,
        generator=generator,
    )
    return _normalize_pipeline_frames(result)


def _load_runtime() -> tuple[Any, Any, dict[str, Any]]:
    try:
        import accelerate
        import diffusers
        import imageio
        import imageio_ffmpeg
        import torch
        import transformers
        from diffusers import WanPipeline
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - Colab production dependency
        raise FreshBaseDiagnosticError("locked GPU dependencies are unavailable") from exc
    observed = {
        "accelerate": accelerate.__version__, "diffusers": diffusers.__version__,
        "imageio": imageio.__version__, "imageio_ffmpeg": imageio_ffmpeg.__version__,
        "transformers": transformers.__version__, "torch": torch.__version__,
    }
    expected = {"accelerate": "1.4.0", "diffusers": "0.35.2", "imageio": "2.37.0", "imageio_ffmpeg": "0.6.0", "transformers": "4.49.0"}
    if any(observed[name] != version for name, version in expected.items()):
        raise FreshBaseDiagnosticError("locked dependency version mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise FreshBaseDiagnosticError("CUDA BF16 runtime is unavailable")
    snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    resolved = snapshot.resolve(strict=True)
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.name != MODEL_REVISION or resolved.name != MODEL_REVISION:
        raise FreshBaseDiagnosticError("local model snapshot identity mismatch")
    pipe = WanPipeline.from_pretrained(str(resolved), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload()
    if not str(pipe._execution_device).startswith("cuda"):
        raise FreshBaseDiagnosticError("Wan offload did not retain CUDA execution")
    runtime = {**observed, "python": platform.python_version(), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
               "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "snapshot_path": str(resolved),
               "scheduler_class": type(pipe.scheduler).__name__, "pipeline_decode": "official_default"}
    return pipe, torch, runtime


def run_exact4_base_once(*, repo_root: Path, output: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FreshBaseDiagnosticError("output target must not exist")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FreshBaseDiagnosticError("output parent unavailable")
    config_path = repo_root / "configs/rc0_fresh_exact4_base_gpu.json"
    protocol_path = repo_root / "protocols/rc0_fresh_exact4_base_gpu.md"
    config = load_and_validate_config(config_path)
    head = _git(repo_root, "rev-parse", "HEAD")
    tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise FreshBaseDiagnosticError("fresh base generation requires a clean exact checkout")
    output.mkdir(mode=0o755)
    bases_root = output / "bases"
    bases_root.mkdir()
    started_at = utc_now()
    attempts_started: list[str] = []
    attempts_completed: list[str] = []
    artifacts: dict[str, Any] = {}
    try:
        pipe, torch, runtime = _load_runtime()
        generation = config["generation"]
        for item in config["groups"]:
            group = item["group_id"]
            attempts_started.append(group)
            group_root = bases_root / group
            group_root.mkdir()
            frames = generate_official_base_frames(pipe, torch, item, generation)
            video = group_root / "base.mp4"
            encode_saved_mp4(frames, video)
            stream = probe_mp4(video)
            decoded = decode_saved_mp4(video)
            if list(decoded.shape) != [49, 320, 512, 3]:
                raise FreshBaseDiagnosticError("saved base decode shape changed")
            artifacts[group] = {"path": str(video.relative_to(output)), "sha256": sha256_file(video), "size": video.stat().st_size,
                                "stream": stream, "prompt_sha256": item["prompt_sha256"], "seed": item["seed"],
                                "carrier": False, "condition": "BASE"}
            attempts_completed.append(group)
        status = "BASE_EXACT4_READY"
        failure = None
    except Exception as exc:
        status = "DIAGNOSTIC_INSUFFICIENT"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        runtime = locals().get("runtime", {"python": platform.python_version()})
        audit = {"schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status,
                 "attempts_started": attempts_started, "attempts_completed": attempts_completed, "failure": failure,
                 "formal_result": False, "stage_progression_allowed": False, "ended_at": utc_now()}
        (output / "audit.json").write_bytes(canonical_json_bytes(audit) + b"\n")
        raise
    audit = {
        "schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status,
        "source": {"head": head, "tree": tree, "dirty": False},
        "config_sha256": sha256_file(config_path), "protocol_sha256": sha256_file(protocol_path),
        "model": config["model"], "runtime": runtime, "attempt_budget": 4,
        "attempts_started": attempts_started, "attempts_completed": attempts_completed,
        "retry_count": 0, "artifacts": artifacts, "carrier_was_used": False,
        "manual_decode_was_used": False, "detector_was_run": False,
        "formal_result": False, "stage_progression_allowed": False,
        "started_at": started_at, "ended_at": utc_now(),
    }
    (output / "config.json").write_bytes(canonical_json_bytes(config) + b"\n")
    (output / "audit.json").write_bytes(canonical_json_bytes(audit) + b"\n")
    (output / "command.json").write_bytes(canonical_json_bytes({"argv": list(argv), "cwd": str(cwd.resolve()), "exit_code": 0}) + b"\n")
    checksum_files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "checksums.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in checksum_files) + "\n", encoding="utf-8")
    return {"status": status, "diagnostic_class": DIAGNOSTIC_CLASS, "output": str(output), "attempts_started": 4, "attempts_completed": 4, "retry_count": 0}
