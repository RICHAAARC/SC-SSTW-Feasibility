"""Real-Wan GPU shape discovery; no injection and no method claim."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from typing import Any


class ShapePreflightError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ShapePreflightError("observed tensor has no shape")
    result = [int(item) for item in shape]
    if not result or any(item <= 0 for item in result):
        raise ShapePreflightError(f"invalid tensor shape: {result}")
    return result


def _result_sample(result: Any) -> Any:
    if hasattr(result, "sample"):
        return result.sample
    if isinstance(result, (tuple, list)) and result:
        return result[0]
    raise ShapePreflightError("Wan transformer result has no sample tensor")


def run_shape_preflight(config_path: Path, output_dir: Path, expected_commit: str) -> dict[str, Any]:
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    if config.get("protocol_id") != "sc_sstw_gpu_shape_preflight_v1":
        raise ShapePreflightError("unexpected shape-preflight protocol")
    if config["hook"]["mutation_permitted"] is not False:
        raise ShapePreflightError("shape preflight must forbid tensor mutation")
    repo = Path(__file__).resolve().parents[2]
    commit = _git(repo, "rev-parse", "HEAD")
    dirty_lines = _git(repo, "status", "--porcelain").splitlines()
    if commit != expected_commit:
        raise ShapePreflightError(f"commit mismatch: observed={commit}, expected={expected_commit}")
    if dirty_lines:
        raise ShapePreflightError("formal shape preflight requires a clean checkout")

    try:
        import torch
        import diffusers
        import transformers
        import accelerate
        import imageio
        import imageio_ffmpeg
        import ftfy
        import safetensors
        from diffusers import WanPipeline
        from huggingface_hub import model_info
    except Exception as exc:
        raise ShapePreflightError("locked GPU dependencies are unavailable") from exc
    if not torch.cuda.is_available():
        raise ShapePreflightError("CUDA GPU is required")
    expected_versions = config["software"]
    observed_versions = {
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "imageio": imageio.__version__,
        "imageio_ffmpeg": imageio_ffmpeg.__version__,
        "ftfy": ftfy.__version__,
        "safetensors": safetensors.__version__,
    }
    mismatches = {key: [observed_versions.get(key), value] for key, value in expected_versions.items() if observed_versions.get(key) != value}
    if mismatches:
        raise ShapePreflightError(f"locked dependency mismatch: {mismatches}")

    token = os.environ.get("HF_TOKEN") or None
    resolved_revision = str(model_info(config["model"]["id"], revision=config["model"]["revision"], token=token).sha)
    if resolved_revision != config["model"]["revision"]:
        raise ShapePreflightError(f"model revision mismatch: resolved={resolved_revision}")

    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        **observed_versions,
        "model_id": config["model"]["id"],
        "model_revision_requested": config["model"]["revision"],
        "model_revision_resolved": resolved_revision,
    }
    _write(output_dir / "environment.json", environment)
    _write(output_dir / "git_state.json", {"commit": commit, "dirty": False, "dirty_lines": [], "remote": _git(repo, "remote", "get-url", "origin")})
    _write(output_dir / "config.json", config)
    command = " ".join(sys.argv)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = WanPipeline.from_pretrained(config["model"]["id"], revision=config["model"]["revision"], torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    calls: list[dict[str, Any]] = []
    original_forward = pipe.transformer.forward

    def traced_forward(*args: Any, **kwargs: Any) -> Any:
        hidden = kwargs.get("hidden_states")
        timestep = kwargs.get("timestep")
        result = original_forward(*args, **kwargs)
        sample = _result_sample(result)
        version_before = getattr(sample, "_version", None)
        record = {
            "call_index": len(calls),
            "hidden_shape": _shape(hidden),
            "hidden_dtype": str(hidden.dtype),
            "output_shape": _shape(sample),
            "output_dtype": str(sample.dtype),
            "timestep_shape": _shape(timestep),
            "timestep_value": float(timestep.detach().float().flatten()[0].item()),
            "output_tensor_version_before_readout": version_before,
        }
        version_after = getattr(sample, "_version", None)
        record["output_tensor_version_after_readout"] = version_after
        record["output_tensor_version_unchanged_during_hook_readout"] = version_before == version_after
        record["hook_returned_original_result_object_static_code_fact"] = True
        calls.append(record)
        return result

    pipe.transformer.forward = traced_forward
    generation = config["generation"]
    try:
        generator = torch.Generator(device="cuda").manual_seed(int(generation["seed"]))
        result = pipe(
            prompt=generation["prompt"],
            negative_prompt=generation["negative_prompt"],
            num_frames=int(generation["frame_count"]),
            height=int(generation["height"]),
            width=int(generation["width"]),
            guidance_scale=float(generation["guidance_scale"]),
            num_inference_steps=int(generation["inference_steps"]),
            generator=generator,
            output_type="latent",
        )
    finally:
        pipe.transformer.forward = original_forward
    latent = result.frames[0] if isinstance(result.frames, (tuple, list)) else result.frames
    hidden_shapes = {tuple(call["hidden_shape"]) for call in calls}
    output_shapes = {tuple(call["output_shape"]) for call in calls}
    all_shapes = [call["hidden_shape"] for call in calls] + [call["output_shape"] for call in calls] + [call["timestep_shape"] for call in calls]
    checks = {
        "cuda_available": bool(torch.cuda.is_available()),
        "repository_clean": not dirty_lines,
        "exact_commit_required": commit == expected_commit,
        "model_revision_exact": resolved_revision == config["model"]["revision"],
        "transformer_call_count_positive": len(calls) > 0,
        "all_shapes_finite_positive_integers": bool(all_shapes) and all(all(isinstance(item, int) and item > 0 for item in shape) for shape in all_shapes),
        "hidden_shape_constant_across_calls": len(hidden_shapes) == 1,
        "output_shape_constant_across_calls": len(output_shapes) == 1,
        "output_tensor_version_unchanged_during_hook_readout": all(call["output_tensor_version_unchanged_during_hook_readout"] for call in calls),
    }
    if set(checks) != set(config["pass_conditions"]):
        raise ShapePreflightError("implemented checks do not exactly match predeclared pass conditions")
    metrics = {
        "evidence_kind": "real_wan_shape_discovery_only_no_injection",
        "configured_denoising_step_count": int(generation["inference_steps"]),
        "transformer_call_count_not_denoising_step_count": len(calls),
        "unique_hidden_shapes": [list(item) for item in sorted(hidden_shapes)],
        "unique_output_shapes": [list(item) for item in sorted(output_shapes)],
        "returned_latent_shape": _shape(latent),
        "checks": checks,
        "gate_pass": all(checks.values()),
        "carrier_claim": False,
        "saved_mp4_claim": False,
        "method_claim": False,
    }
    _write(artifacts / "transformer_calls.json", calls)
    _write(artifacts / "returned_latent_shape.json", {"shape": metrics["returned_latent_shape"]})
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "gate_decision.json", {"gate": "GPU shape/interface preflight", "predeclared_conditions": config["pass_conditions"], "implementer_decision": "GATE_PASS" if metrics["gate_pass"] else "GATE_FAIL", "auditor_decision": "PENDING", "gpu_method_gate_admitted": False})
    (output_dir / "stdout.log").write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "stderr.log").write_text("", encoding="utf-8")
    (output_dir / "README.md").write_text("# GPU shape preflight\n\nDiagnostic only. Inspect `metrics.json` and `artifacts/transformer_calls.json`. No injection or method claim is present.\n", encoding="utf-8")
    _finalize_package(output_dir)
    return metrics


def write_failure_package(config_path: Path, output_dir: Path, expected_commit: str, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = _git(repo, "rev-parse", "HEAD")
        dirty_lines = _git(repo, "status", "--porcelain").splitlines()
        remote = _git(repo, "remote", "get-url", "origin")
    except Exception as exc:
        commit, dirty_lines, remote = "unavailable", [type(exc).__name__], "unavailable"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        config = {"config_read_failure": True}
    _write(output_dir / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "cuda_status": "not_confirmed_failure_path"})
    _write(output_dir / "git_state.json", {"commit": commit, "expected_commit": expected_commit, "dirty": bool(dirty_lines), "dirty_lines": dirty_lines, "remote": remote})
    _write(output_dir / "config.json", config)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    metrics = {"evidence_kind": "gpu_shape_preflight_runtime_failure", "gate_pass": False, "reason": reason, "carrier_claim": False, "saved_mp4_claim": False, "method_claim": False}
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "gate_decision.json", {"gate": "GPU shape/interface preflight", "implementer_decision": "GATE_FAIL", "auditor_decision": "PENDING", "reason": reason, "gpu_method_gate_admitted": False})
    (output_dir / "stdout.log").touch()
    (output_dir / "stderr.log").write_text(reason + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text("# GPU shape preflight failure\n\nThe run failed and supports no injection or method claim.\n", encoding="utf-8")
    _write(output_dir / "artifacts" / "failure.json", {"reason": reason})
    _finalize_package(output_dir)


def _finalize_package(output_dir: Path) -> None:
    checksum_path = output_dir / "checksums.sha256"
    checksum_targets = [path for path in output_dir.rglob("*") if path.is_file() and path != checksum_path]
    lines = [f"{_sha(path)}  {path.relative_to(output_dir).as_posix()}" for path in sorted(checksum_targets)]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive = output_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_dir, arcname=output_dir.name)
