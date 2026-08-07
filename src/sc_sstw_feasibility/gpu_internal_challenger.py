"""Audited Wan internal-attention-output propagation challenger.

The sole carrier hook replaces ``transformer.blocks[29].attn1`` output. Internal
and paired quantities are diagnostic only; the Gate relation is read from one
saved watermarked MP4.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

from .aisb import affine_burst_residual, make_default_templates
from .gpu_carrier import (
    CarrierPropagationError,
    _finite_matrix,
    _git,
    _write,
    finalize_package,
    paired_difference_observation,
    project_tensor_q,
    result_sample,
    saved_video_observation,
)
from .gpu_carrier_v2 import FROZEN_GROUPS, FROZEN_POINTS, FROZEN_ROLES
from .linalg import condition_number_2d_columns


EXPECTED_PATCH_SHAPE = [1, 1536, 13, 20, 32]
EXPECTED_INTERNAL_SHAPE = [1, 8320, 1536]
EXPECTED_LATENT_SHAPE = [1, 16, 13, 40, 64]
EXPECTED_GRID = [13, 20, 32]
EXPECTED_TIMESTEPS = [999, 999, 954, 954, 899, 899, 832, 832, 749, 749, 642, 642, 499, 499, 299, 299]
EXPECTED_BRANCHES = [f"step_{step}_{branch}" for step in range(8) for branch in ("cond", "uncond")]
V2_FILE_DIGESTS = {
    "configs/gpu_carrier_propagation_v2.json": "d6252179aee3375f7bbe38810350d20e1b8d7ccdb0576cd8ad01576bd00ede74",
    "experiments/run_gpu_carrier_propagation_v2.py": "239ca438d139190bd30cceeba9b2764cf5ca1793e9b256e00fb70a902df7fb90",
    "notebooks/sc_sstw_gpu_carrier_propagation_v2.ipynb": "5d161b198af7fae6bba95aecc1748a70eec5381d63012a06ff04bb7aff0fb715",
    "protocols/gpu_carrier_propagation_v2.md": "d2b2397f0f21cb6c6a0d785a047c2d95f4fdf2b9b66cc461598a61dfb29f976a",
    "src/sc_sstw_feasibility/gpu_carrier_v2.py": "0e90d5f1960339a6966bdf31801126aa77d0fe055a897f2e75a84a567cfc4602",
    "tests/test_gpu_carrier_v2.py": "397d81ee9289275fc63a02c69e451b5c2c8e38689f64d802bf69b763f9485b22",
}


def validate_internal_config(config: dict[str, Any]) -> None:
    if config.get("protocol_id") != "sc_sstw_gpu_internal_output_challenger_v1":
        raise CarrierPropagationError("unexpected internal challenger protocol")
    history = config["history_lock"]
    if history != {
        "direct_v2_commit": "9c1be010353367f0ee1a95afc372999dafbaf812",
        "direct_v2_drive_run": "gpu_carrier_propagation_v2/20260807T010521Z_9c1be010",
        "direct_v2_archive_sha256": "5b4e8bf820dc7e12f0eed2520e88098a3d3a0c1b7b3e01ec178337e7f4d08b26",
        "direct_v2_conclusion": "NO_GO",
        "overall_method_conclusion": "NOT_DETERMINED",
    }:
        raise CarrierPropagationError("direct v2 history lock changed")
    carrier = config["carrier"]
    frozen = {
        "kind": "dit_internal_self_attention_output_residual",
        "module_path": "transformer.blocks[29].attn1",
        "block_index": 29,
        "expected_transformer_type": "WanTransformer3DModel",
        "expected_block_type": "WanTransformerBlock",
        "expected_attention_type": "WanAttention",
        "expected_block_count": 30,
        "expected_patch_size": [1, 2, 2],
        "expected_patch_embedding_shape": EXPECTED_PATCH_SHAPE,
        "expected_token_grid": EXPECTED_GRID,
        "expected_internal_dim": 1536,
        "expected_hook_output_shape": EXPECTED_INTERNAL_SHAPE,
        "token_order": "index=((t*20)+h)*32+w_width_fastest_c_order",
        "basis_formula_x": "sqrt(2)*cos(2*pi*(w+0.5)/32)_all_t_h_d",
        "basis_formula_y": "sqrt(2)*cos(2*pi*(h+0.5)/20)_all_t_w_d",
        "required_runtime_dtype": "torch.bfloat16",
        "hook_mode": "forward_output_replacement_new_tensor",
        "apply_to": "8_steps_each_cond_then_uncond_16_calls",
        "expected_branch_sequence": EXPECTED_BRANCHES,
        "expected_timestep_values": EXPECTED_TIMESTEPS,
        "gain_solver": "deterministic_effective_delta_multiplicative_correction",
        "gain_solver_max_iterations": 12,
        "in_place_mutation_permitted": False,
        "processor_or_qkv_mutation_permitted": False,
        "scheduler_mutation_permitted": False,
        "guidance_or_final_output_carrier_permitted": False,
        "target_relative_rms": 0.03,
        "target_relative_rms_absolute_tolerance": 0.00005,
        "temporal_points": FROZEN_POINTS,
        "temporal_point_roles": FROZEN_ROLES,
    }
    if carrier != frozen:
        raise CarrierPropagationError("internal challenger carrier contract changed")
    if config["diagnostic_readouts"]["temporal_frame_groups"] != FROZEN_GROUPS:
        raise CarrierPropagationError("frame groups changed from direct v2")
    if config["relation_thresholds"] != {"maximum_public_aisb_residual": 0.25, "maximum_public_condition_number": 10.0}:
        raise CarrierPropagationError("relation thresholds changed")
    expected_generation = {
        "fps": 8, "frame_count": 49, "guidance_scale": 5.0, "height": 320,
        "inference_steps": 8,
        "negative_prompt": "text, watermark, logo, camera motion, cuts, multiple subjects, flicker",
        "prompt": "locked camera, dark matte background, a single bright white cube moving slowly across the center, simple studio lighting, no text, no cuts",
        "seed": 1275, "width": 512,
    }
    if config["generation"] != expected_generation:
        raise CarrierPropagationError("generation inputs changed from direct v2")
    if config["model"] != {"id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "revision": "0fad780a534b6463e45facd96134c9f345acfa5b"}:
        raise CarrierPropagationError("model identity changed")
    expected_software = {
        "accelerate": "1.4.0", "diffusers": "0.35.2", "ftfy": "6.3.1",
        "imageio": "2.37.0", "imageio_ffmpeg": "0.6.0", "safetensors": "0.5.3",
        "transformers": "4.49.0",
    }
    if config["software"] != expected_software:
        raise CarrierPropagationError("software lock changed")


def build_internal_basis(torch: Any, output: Any) -> tuple[Any, Any]:
    """Return the frozen width- and height-cosine bases in token order."""

    if list(output.shape) != EXPECTED_INTERNAL_SHAPE:
        raise CarrierPropagationError(f"unexpected internal attention output shape: {list(output.shape)}")
    temporal, height, width = EXPECTED_GRID
    x = torch.arange(width, device=output.device, dtype=torch.float32) + 0.5
    y = torch.arange(height, device=output.device, dtype=torch.float32) + 0.5
    bx_grid = math.sqrt(2.0) * torch.cos(2.0 * math.pi * x / float(width)).reshape(1, 1, width, 1)
    by_grid = math.sqrt(2.0) * torch.cos(2.0 * math.pi * y / float(height)).reshape(1, height, 1, 1)
    bx = bx_grid.expand(temporal, height, width, EXPECTED_INTERNAL_SHAPE[2]).reshape(1, -1, EXPECTED_INTERNAL_SHAPE[2])
    by = by_grid.expand(temporal, height, width, EXPECTED_INTERNAL_SHAPE[2]).reshape(1, -1, EXPECTED_INTERNAL_SHAPE[2])
    return bx.to(dtype=output.dtype), by.to(dtype=output.dtype)


def project_internal_q(torch: Any, tensor: Any) -> list[list[float]]:
    bx, by = build_internal_basis(torch, tensor)
    temporal, height, width = EXPECTED_GRID
    value = tensor.detach().float().reshape(1, temporal, height, width, EXPECTED_INTERNAL_SHAPE[2])
    bx_grid = bx.float().reshape(1, temporal, height, width, EXPECTED_INTERNAL_SHAPE[2])
    by_grid = by.float().reshape(1, temporal, height, width, EXPECTED_INTERNAL_SHAPE[2])
    reduce_dims = (0, 2, 3, 4)
    qx = (value * bx_grid).mean(dim=reduce_dims) / bx_grid.square().mean(dim=reduce_dims)
    qy = (value * by_grid).mean(dim=reduce_dims) / by_grid.square().mean(dim=reduce_dims)
    return [[float(x), float(y)] for x, y in zip(qx.tolist(), qy.tolist(), strict=True)]


def construct_internal_residual(
    torch: Any,
    output: Any,
    points: Sequence[Sequence[float]],
    target_relative_rms: float,
    absolute_tolerance: float,
    max_iterations: int,
) -> tuple[Any, Any, dict[str, float | int | bool]]:
    bx, by = build_internal_basis(torch, output)
    q = torch.tensor(points, device=output.device, dtype=torch.float32)
    temporal, height, width = EXPECTED_GRID
    q1 = q[:, 0].reshape(1, temporal, 1, 1, 1)
    q2 = q[:, 1].reshape(1, temporal, 1, 1, 1)
    bx_grid = bx.float().reshape(1, temporal, height, width, EXPECTED_INTERNAL_SHAPE[2])
    by_grid = by.float().reshape(1, temporal, height, width, EXPECTED_INTERNAL_SHAPE[2])
    raw_float = (q1 * bx_grid + q2 * by_grid).reshape(EXPECTED_INTERNAL_SHAPE)
    raw_rms = raw_float.square().mean().sqrt()
    output_rms = output.detach().float().square().mean().sqrt()
    if not bool(torch.isfinite(raw_rms)) or not bool(torch.isfinite(output_rms)) or float(raw_rms) <= 0.0 or float(output_rms) <= 0.0:
        raise CarrierPropagationError("non-finite or zero internal carrier/output RMS")
    initial_gain = float((float(target_relative_rms) * output_rms / raw_rms).item())
    gain = initial_gain
    best: tuple[float, Any, Any, float, float, int] | None = None
    converged = False
    for iteration in range(int(max_iterations) + 1):
        candidate = (raw_float * gain).to(dtype=output.dtype)
        modified = output + candidate
        effective_delta = modified - output
        relative_rms = float((effective_delta.float().square().mean().sqrt() / output_rms).item())
        error = relative_rms - float(target_relative_rms)
        if not math.isfinite(relative_rms) or relative_rms <= 0.0:
            raise CarrierPropagationError("effective internal RMS is non-finite or zero")
        if best is None or abs(error) < abs(best[0]):
            best = (error, modified, effective_delta, relative_rms, gain, iteration)
        if abs(error) <= float(absolute_tolerance):
            converged = True
            break
        gain *= float(target_relative_rms) / relative_rms
    assert best is not None
    error, modified, effective_delta, relative_rms, applied_gain, iteration = best
    return modified, effective_delta, {
        "raw_float_rms": float(raw_rms.item()), "output_rms": float(output_rms.item()),
        "initial_gain": initial_gain, "applied_gain": applied_gain,
        "gain_iterations": iteration, "effective_relative_rms": relative_rms,
        "effective_absolute_error": abs(error),
        "gain_solver_converged": bool(converged or abs(error) <= float(absolute_tolerance)),
    }


def run_internal_preflight(torch: Any, config: dict[str, Any], dtype: Any) -> dict[str, Any]:
    if str(dtype) != config["carrier"]["required_runtime_dtype"]:
        return {"passed": False, "records": [], "reason": "required BF16 runtime dtype is unavailable"}
    target = float(config["carrier"]["target_relative_rms"])
    tolerance = float(config["carrier"]["target_relative_rms_absolute_tolerance"])
    generator = torch.Generator(device="cuda").manual_seed(20260807)
    records: list[dict[str, Any]] = []
    for case_index, scale in enumerate([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0] * 2):
        bias = (0.25 if case_index % 2 == 0 else -0.25) * scale
        try:
            output = (scale * torch.randn(EXPECTED_INTERNAL_SHAPE, generator=generator, device="cuda") + bias).to(dtype)
            modified, delta, energy = construct_internal_residual(
                torch, output, FROZEN_POINTS, target, tolerance, int(config["carrier"]["gain_solver_max_iterations"]),
            )
            reconstructed = [[v / float(energy["applied_gain"]) for v in point] for point in project_internal_q(torch, delta)]
            mse = sum((reconstructed[i][d] - FROZEN_POINTS[i][d]) ** 2 for i in range(13) for d in range(2)) / 26.0
            finite = bool(torch.isfinite(modified).all().item() and torch.isfinite(delta).all().item())
            exact = bool(torch.equal(delta, modified - output))
            passed = finite and exact and bool(energy["gain_solver_converged"]) and abs(float(energy["effective_relative_rms"]) - target) <= tolerance and mse <= 0.0001
            records.append({"case_index": case_index, "scale": scale, "bias": bias, "dtype": str(output.dtype), "finite": finite, "effective_delta_exact": exact, "residual_q_reconstruction_mse": mse, "case_pass": passed, **energy})
        except Exception as exc:
            records.append({"case_index": case_index, "scale": scale, "bias": bias, "case_pass": False, "error": f"{type(exc).__name__}: {exc}"})
    return {"passed": len(records) == 16 and all(r["case_pass"] for r in records), "records": records, "target_relative_rms": target, "absolute_tolerance": tolerance}


def _relation_metrics(q: list[list[float]]) -> dict[str, float | bool]:
    public = q[:6]
    mean = [sum(point[d] for point in public) / 6.0 for d in range(2)]
    centered = [[point[d] - mean[d] for d in range(2)] for point in public]
    condition = condition_number_2d_columns(centered)
    residual = affine_burst_residual([list(point) for point in public], make_default_templates()[0])
    return {
        "public_aisb_residual": float(residual), "public_condition_number": float(condition),
        "aisb_residual_at_most_0_25": math.isfinite(residual) and residual <= 0.25,
        "condition_number_at_most_10": math.isfinite(condition) and condition <= 10.0,
    }


def _subtract_q(clean: Sequence[Sequence[float]], marked: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(clean) != 13 or len(marked) != 13:
        raise CarrierPropagationError("paired diagnostic q must contain 13 points")
    return [[float(marked[i][d]) - float(clean[i][d]) for d in range(2)] for i in range(13)]


def observed_cfg_sequence_matches(*record_sets: Sequence[dict[str, Any]]) -> bool:
    """Validate only branch labels recorded from actual cache-context names."""

    return bool(record_sets) and all(
        len(records) == 16
        and [record["branch_label"] for record in records] == EXPECTED_BRANCHES
        and [int(record["timestep_value"]) for record in records] == EXPECTED_TIMESTEPS
        for records in record_sets
    )


def _sample_digest(torch: Any, tensor: Any) -> str:
    flat = tensor.detach().reshape(-1)
    indices = torch.linspace(0, flat.numel() - 1, steps=64, device=flat.device).long()
    values = [float(v) for v in flat.index_select(0, indices).float().cpu().tolist()]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()


def _v2_files_intact(repo: Path) -> bool:
    return all(hashlib.sha256((repo / relative).read_bytes()).hexdigest() == digest for relative, digest in V2_FILE_DIGESTS.items())


def run_internal_challenger(config_path: Path, output_dir: Path, expected_commit: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_internal_config(config)
    repo = Path(__file__).resolve().parents[2]
    if not _v2_files_intact(repo):
        raise CarrierPropagationError("preserved direct v2 files changed")
    commit = _git(repo, "rev-parse", "HEAD")
    dirty_lines = _git(repo, "status", "--porcelain").splitlines()
    if commit != expected_commit:
        raise CarrierPropagationError(f"commit mismatch: observed={commit}, expected={expected_commit}")
    if dirty_lines:
        raise CarrierPropagationError("formal internal challenger requires a clean checkout")
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
    versions = {
        "diffusers": diffusers.__version__, "transformers": transformers.__version__,
        "accelerate": accelerate.__version__, "imageio": imageio.__version__,
        "imageio_ffmpeg": imageio_ffmpeg.__version__, "ftfy": ftfy.__version__,
        "safetensors": safetensors.__version__,
    }
    mismatches = {k: [versions.get(k), v] for k, v in config["software"].items() if versions.get(k) != v}
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
    _write(output_dir / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda, "torch": torch.__version__, **versions, "model_id": config["model"]["id"], "model_revision_requested": config["model"]["revision"], "model_revision_resolved": resolved_revision})
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    preflight = run_internal_preflight(torch, config, dtype)
    _write(artifacts / "effective_delta_preflight.json", preflight)
    if not preflight["passed"]:
        raise CarrierPropagationError("BF16 internal effective-delta preflight failed before model loading")
    pipe = WanPipeline.from_pretrained(config["model"]["id"], revision=config["model"]["revision"], torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    transformer = pipe.transformer
    carrier = config["carrier"]
    structure = {
        "transformer_type": type(transformer).__name__,
        "block_count": len(transformer.blocks),
        "block_type": type(transformer.blocks[29]).__name__ if len(transformer.blocks) > 29 else None,
        "attention_type": type(transformer.blocks[29].attn1).__name__ if len(transformer.blocks) > 29 else None,
        "patch_size": list(transformer.config.patch_size),
        "internal_dim": int(transformer.config.num_attention_heads * transformer.config.attention_head_dim),
    }
    structure_exact = structure == {
        "transformer_type": carrier["expected_transformer_type"], "block_count": 30,
        "block_type": carrier["expected_block_type"], "attention_type": carrier["expected_attention_type"],
        "patch_size": [1, 2, 2], "internal_dim": 1536,
    }
    _write(artifacts / "model_structure.json", {**structure, "exact": structure_exact})
    if not structure_exact:
        raise CarrierPropagationError(f"locked Wan structure mismatch: {structure}")
    scheduler_step_before = getattr(pipe.scheduler.step, "__func__", pipe.scheduler.step)
    generation = config["generation"]

    def generate(with_carrier: bool) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        injections: list[dict[str, Any]] = []
        latents_out: list[dict[str, Any]] = []
        final_outputs: list[dict[str, Any]] = []
        mapping_records: list[dict[str, Any]] = []
        context: dict[str, Any] = {"call_index": None, "patch_output": None, "actual_cache_context": None}
        original_forward = transformer.forward
        original_cache_context = transformer.cache_context

        @contextmanager
        def observed_cache_context(name: str, *args: Any, **kwargs: Any) -> Any:
            if context["actual_cache_context"] is not None:
                raise CarrierPropagationError("nested transformer cache contexts are not admitted")
            context["actual_cache_context"] = str(name)
            try:
                with original_cache_context(name, *args, **kwargs):
                    yield
            finally:
                context["actual_cache_context"] = None

        def patch_hook(_module: Any, _inputs: Any, output: Any) -> None:
            context["patch_output"] = output

        def block0_pre_hook(_module: Any, inputs: Any) -> None:
            patch_output = context.get("patch_output")
            if patch_output is None or not inputs:
                raise CarrierPropagationError("patch output or block-0 input missing")
            actual = inputs[0]
            expected = patch_output.flatten(2).transpose(1, 2)
            equal = list(patch_output.shape) == EXPECTED_PATCH_SHAPE and list(actual.shape) == EXPECTED_INTERNAL_SHAPE and bool(torch.equal(expected, actual))
            mapping_records.append({
                "call_index": int(context["call_index"]), "patch_shape": [int(v) for v in patch_output.shape],
                "block0_input_shape": [int(v) for v in actual.shape], "full_value_equal": equal,
                "fixed_sample_digest": _sample_digest(torch, actual),
                "saved_tensor_values": False,
            })
            context["patch_output"] = None
            if not equal:
                raise CarrierPropagationError("patch embedding to block-0 token mapping is not value-exact")

        def attention_output_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if not with_carrier:
                return output
            call_index = int(context["call_index"])
            actual_branch = context.get("actual_cache_context")
            if actual_branch not in {"cond", "uncond"}:
                raise CarrierPropagationError(f"attention hook is outside an observed cond/uncond cache context: {actual_branch!r}")
            if list(output.shape) != EXPECTED_INTERNAL_SHAPE or str(output.dtype) != "torch.bfloat16":
                raise CarrierPropagationError(f"unexpected attn1 hook output: shape={list(output.shape)} dtype={output.dtype}")
            version_before = getattr(output, "_version", None)
            modified, delta, energy = construct_internal_residual(
                torch, output, FROZEN_POINTS, float(carrier["target_relative_rms"]),
                float(carrier["target_relative_rms_absolute_tolerance"]), int(carrier["gain_solver_max_iterations"]),
            )
            reconstructed = [[v / float(energy["applied_gain"]) for v in point] for point in project_internal_q(torch, delta)]
            mse = sum((reconstructed[i][d] - FROZEN_POINTS[i][d]) ** 2 for i in range(13) for d in range(2)) / 26.0
            version_after = getattr(output, "_version", None)
            injections.append({
                "call_index": call_index, "step_index": call_index // 2,
                "branch": str(actual_branch),
                "branch_label": f"step_{call_index // 2}_{actual_branch}",
                "timestep_value": float(context["timestep"]),
                "module_path": carrier["module_path"], "output_shape": [int(v) for v in output.shape],
                "output_dtype": str(output.dtype), "effective_delta_dtype": str(delta.dtype),
                "original_tensor_version_before": version_before, "original_tensor_version_after": version_after,
                "original_tensor_version_unchanged": version_before == version_after,
                "modified_has_distinct_storage": int(modified.data_ptr()) != int(output.data_ptr()),
                "effective_delta_q_diagnostic_only": project_internal_q(torch, delta),
                "residual_q_reconstructed": reconstructed, "residual_q_reconstruction_mse": mse,
                **energy,
            })
            return modified

        patch_handle = transformer.patch_embedding.register_forward_hook(patch_hook)
        block0_handle = transformer.blocks[0].register_forward_pre_hook(block0_pre_hook)
        attention_handle = transformer.blocks[29].attn1.register_forward_hook(attention_output_hook)

        def observed_forward(*args: Any, **kwargs: Any) -> Any:
            call_index = len(final_outputs)
            if call_index >= 16:
                raise CarrierPropagationError("more than 16 transformer calls")
            context["call_index"] = call_index
            actual_branch = context.get("actual_cache_context")
            if actual_branch not in {"cond", "uncond"}:
                raise CarrierPropagationError(f"transformer call is outside an observed cond/uncond cache context: {actual_branch!r}")
            timestep = kwargs.get("timestep")
            if timestep is None:
                raise CarrierPropagationError("transformer timestep missing")
            context["timestep"] = float(timestep.detach().float().flatten()[0].item())
            result = original_forward(*args, **kwargs)
            sample = result_sample(result)
            final_outputs.append({
                "call_index": call_index, "step_index": call_index // 2,
                "branch": str(actual_branch),
                "branch_label": f"step_{call_index // 2}_{actual_branch}", "timestep_value": context["timestep"],
                "shape": [int(v) for v in sample.shape], "dtype": str(sample.dtype),
                "finite": bool(torch.isfinite(sample).all().item()), "q_diagnostic_only": project_tensor_q(torch, sample),
            })
            return result

        transformer.forward = observed_forward
        transformer.cache_context = observed_cache_context

        def callback(_pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            latents = callback_kwargs["latents"]
            latents_out.append({"step_index": int(step_index), "timestep_value": float(timestep.detach().float().flatten()[0].item()), "shape": [int(v) for v in latents.shape], "dtype": str(latents.dtype), "finite": bool(torch.isfinite(latents).all().item()), "q_diagnostic_only": project_tensor_q(torch, latents)})
            return callback_kwargs

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
            return frames, injections, latents_out, final_outputs, mapping_records
        finally:
            transformer.forward = original_forward
            transformer.cache_context = original_cache_context
            attention_handle.remove()
            block0_handle.remove()
            patch_handle.remove()

    target_attention = transformer.blocks[29].attn1
    target_parameter_prefixes = ("to_q.", "to_k.", "to_v.", "to_out.")
    target_state_before = {
        "processor_object_id": id(target_attention.processor),
        "parameters": {
            name: {"object_id": id(parameter), "version": getattr(parameter, "_version", None)}
            for name, parameter in target_attention.named_parameters()
            if name.startswith(target_parameter_prefixes)
        },
    }
    clean_frames, clean_injections, clean_latents, clean_outputs, clean_mappings = generate(False)
    if clean_injections:
        raise CarrierPropagationError("clean generation unexpectedly injected")
    marked_frames, injections, marked_latents, marked_outputs, marked_mappings = generate(True)
    target_state_after = {
        "processor_object_id": id(target_attention.processor),
        "parameters": {
            name: {"object_id": id(parameter), "version": getattr(parameter, "_version", None)}
            for name, parameter in target_attention.named_parameters()
            if name.startswith(target_parameter_prefixes)
        },
    }
    target_state_unchanged = bool(target_state_before["parameters"]) and target_state_after == target_state_before
    _write(artifacts / "target_attention_immutability.json", {"before": target_state_before, "after": target_state_after, "unchanged": target_state_unchanged})
    scheduler_step_after = getattr(pipe.scheduler.step, "__func__", pipe.scheduler.step)
    clean_path, marked_path = artifacts / "clean.mp4", artifacts / "watermarked.mp4"
    export_to_video(clean_frames, str(clean_path), fps=8)
    export_to_video(marked_frames, str(marked_path), fps=8)
    clean_saved, marked_saved = list(iio.imiter(clean_path)), list(iio.imiter(marked_path))
    groups = config["diagnostic_readouts"]["temporal_frame_groups"]
    blind_q = saved_video_observation(marked_saved, groups)
    paired_rgb_q = paired_difference_observation(clean_frames, marked_frames, groups)
    paired_mp4_q = paired_difference_observation(clean_saved, marked_saved, groups)
    paired_outputs = [_subtract_q(c["q_diagnostic_only"], m["q_diagnostic_only"]) for c, m in zip(clean_outputs, marked_outputs, strict=True)]
    paired_latents = [_subtract_q(c["q_diagnostic_only"], m["q_diagnostic_only"]) for c, m in zip(clean_latents, marked_latents, strict=True)]
    blind_relation = _relation_metrics(blind_q)

    _write(artifacts / "injection_records.json", injections)
    _write(artifacts / "patch_token_mapping_checks.json", {"clean": clean_mappings, "watermarked": marked_mappings, "full_tensor_values_saved": False})
    _write(artifacts / "paired_final_transformer_output_diagnostic_only.json", {"eligible_for_blind_pass": False, "q_by_call": paired_outputs, "relation_by_call": [_relation_metrics(q) for q in paired_outputs]})
    _write(artifacts / "paired_latent_diagnostic_only.json", {"eligible_for_blind_pass": False, "q_by_step": paired_latents, "relation_by_step": [_relation_metrics(q) for q in paired_latents]})
    _write(artifacts / "blind_single_watermarked_mp4_observation.json", {"input": "single_saved_watermarked_mp4_only", "key_independent": True, "uses_clean_video": False, "q": blind_q, "relation": blind_relation})
    _write(artifacts / "paired_rgb_pre_mp4_observation_diagnostic_only.json", {"eligible_for_blind_pass": False, "q": paired_rgb_q, "relation": _relation_metrics(paired_rgb_q)})
    _write(artifacts / "paired_mp4_observation_diagnostic_only.json", {"eligible_for_blind_pass": False, "q": paired_mp4_q, "relation": _relation_metrics(paired_mp4_q)})

    target = float(carrier["target_relative_rms"])
    tolerance = float(carrier["target_relative_rms_absolute_tolerance"])
    mappings = clean_mappings + marked_mappings
    sequence_ok = observed_cfg_sequence_matches(injections, clean_outputs, marked_outputs)
    paired_finite = all(_finite_matrix(q, 13, 2) for q in paired_outputs + paired_latents + [paired_rgb_q, paired_mp4_q])
    checks = {
        "all_injection_records_finite": bool(injections) and all(all(math.isfinite(float(r[k])) for k in ("raw_float_rms", "output_rms", "initial_gain", "applied_gain", "effective_relative_rms", "effective_absolute_error")) for r in injections),
        "blind_saved_mp4_observation_finite_13_by_2": _finite_matrix(blind_q, 13, 2),
        "blind_saved_mp4_public_aisb_residual_at_most_0_25": bool(blind_relation["aisb_residual_at_most_0_25"]),
        "blind_saved_mp4_public_condition_number_at_most_10": bool(blind_relation["condition_number_at_most_10"]),
        "call_sequence_cond_then_uncond_for_8_steps": sequence_ok,
        "clean_and_watermarked_mp4_each_have_49_frames": len(clean_saved) == 49 and len(marked_saved) == 49,
        "effective_delta_preflight_16_of_16": bool(preflight["passed"]) and len(preflight["records"]) == 16,
        "exact_commit_required": commit == expected_commit,
        "hook_output_shape_and_dtype_16_of_16": len(injections) == 16 and all(r["output_shape"] == EXPECTED_INTERNAL_SHAPE and r["output_dtype"] == "torch.bfloat16" for r in injections),
        "internal_attn1_output_injection_used": len(injections) == 16 and all(r["module_path"] == "transformer.blocks[29].attn1" for r in injections),
        "latent_callback_shape_is_1_16_13_40_64": len(marked_latents) == 8 and all(r["shape"] == EXPECTED_LATENT_SHAPE and r["finite"] for r in marked_latents),
        "model_revision_and_structure_exact": resolved_revision == config["model"]["revision"] and structure_exact,
        "original_attention_output_tensor_version_unchanged": bool(injections) and all(r["original_tensor_version_unchanged"] and r["modified_has_distinct_storage"] for r in injections),
        "paired_diagnostics_finite": paired_finite,
        "patch_embedding_to_block0_value_mapping_exact_32_of_32": len(mappings) == 32 and all(r["full_value_equal"] for r in mappings),
        "repository_clean": not dirty_lines,
        "residual_q_reconstruction_mse_at_most_0_0001": bool(injections) and all(r["residual_q_reconstruction_mse"] <= 0.0001 for r in injections),
        "scheduler_mutation_absent": scheduler_step_after is scheduler_step_before,
        "target_relative_rms_within_tolerance_16_of_16": len(injections) == 16 and all(r["gain_solver_converged"] and abs(r["effective_relative_rms"] - target) <= tolerance for r in injections),
        "target_attention_processor_and_qkvout_parameters_unchanged": target_state_unchanged,
    }
    if set(checks) != set(config["pass_conditions"]):
        raise CarrierPropagationError("implemented checks do not match frozen pass conditions")
    blind_keys = {"blind_saved_mp4_public_aisb_residual_at_most_0_25", "blind_saved_mp4_public_condition_number_at_most_10"}
    execution_integrity_pass = all(value for key, value in checks.items() if key not in blind_keys)
    blind_relation_pass = all(checks[key] for key in blind_keys)
    gate_pass = execution_integrity_pass and blind_relation_pass
    metrics = {
        "evidence_kind": "real_wan_internal_attn1_output_residual_public_relation_propagation_challenger_v1",
        "checks": checks, "execution_integrity_pass": execution_integrity_pass,
        "blind_saved_mp4_relation": blind_relation, "blind_saved_mp4_relation_pass": blind_relation_pass,
        "paired_propagation_diagnostic_only": {"final_output_last": _relation_metrics(paired_outputs[-1]), "latent_last": _relation_metrics(paired_latents[-1]), "rgb_pre_mp4": _relation_metrics(paired_rgb_q), "saved_mp4": _relation_metrics(paired_mp4_q)},
        "gate_pass": gate_pass, "injection_call_count": len(injections),
        "clean_saved_frame_count": len(clean_saved), "watermarked_saved_frame_count": len(marked_saved),
        "public_relation_propagation_bridge_claim": gate_pass,
        "calibration_claim": False, "held_out_claim": False, "owner_wrong_key_claim": False,
        "temporal_edit_claim": False, "method_claim": False,
    }
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "gate_decision.json", {"gate": "GPU internal attention-output challenger v1", "predeclared_conditions": config["pass_conditions"], "implementer_decision": "GATE_PASS" if gate_pass else "GATE_FAIL", "auditor_decision": "PENDING", "maximum_claim_if_pass": "public_relation_propagation_bridge_only"})
    (output_dir / "stdout.log").write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "stderr.log").write_text("", encoding="utf-8")
    (output_dir / "README.md").write_text("# GPU internal-output challenger v1\n\nThe sole carrier is block 29 attn1 output. Internal and paired artifacts are diagnostic only. The Gate relation uses one saved watermarked MP4. No calibration, held-out, owner/wrong-key, edit, or method claim.\n", encoding="utf-8")
    finalize_package(output_dir)
    return metrics


def write_internal_failure_package(config_path: Path, output_dir: Path, expected_commit: str, reason: str) -> None:
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
    _write(output_dir / "git_state.json", {"commit": commit, "expected_commit": expected_commit, "dirty": bool(dirty_lines), "dirty_lines": dirty_lines, "remote": remote, "status": "internal_challenger_failure_before_or_during_formal_run"})
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    metrics = {"evidence_kind": "gpu_internal_output_challenger_v1_runtime_failure", "gate_pass": False, "reason": reason, "public_relation_propagation_bridge_claim": False, "method_claim": False}
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "gate_decision.json", {"gate": "GPU internal attention-output challenger v1", "implementer_decision": "GATE_FAIL", "auditor_decision": "PENDING", "reason": reason})
    (output_dir / "stdout.log").write_text("", encoding="utf-8")
    (output_dir / "stderr.log").write_text(reason + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text("# GPU internal-output challenger v1 failure\n\nNo bridge or method claim.\n", encoding="utf-8")
    _write(output_dir / "artifacts" / "failure.json", {"reason": reason})
    finalize_package(output_dir)
