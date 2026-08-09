"""S1 real-Wan relation primitive runner and preregistered statistics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from .relation_injector import TARGET_QUERY_INDICES, install_s1_processor


SCHEMA = "sstw.s1.real_dit_relation_primitive.v1"
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
CONDITION_ORDER = ("OFF_R1", "OFF_R2", "PLUS_B1", "MINUS_B1", "PLUS_B2", "MINUS_B2")
BRANCH_ORDER = ("cond", "uncond")
LAYER_ORDER = ("relation", "block", "velocity")
CONDITION_STATES = {
    "OFF_R1": (0.0, 0.0), "OFF_R2": (0.0, 0.0),
    "PLUS_B1": (1.0, 0.0), "MINUS_B1": (-1.0, 0.0),
    "PLUS_B2": (0.0, 1.0), "MINUS_B2": (0.0, -1.0),
}


class S1InstrumentationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: Any) -> str:
    array = tensor.detach().float().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def sha256_parameters(module: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(parameter.detach().float().cpu().numpy()).tobytes(order="C"))
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def load_frozen_inputs(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = repo_root / "configs/s1_real_dit_relation_primitive.json"
    plan_path = repo_root / "plans/s1_real_dit_relation_primitive.json"
    authority_path = repo_root / "SSTW_METHOD_AUTHORITY.md"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or config.get("diagnostic_class") != DIAGNOSTIC_CLASS:
        raise S1InstrumentationError("S1 config identity mismatch")
    if sha256_file(authority_path) != config.get("authority", {}).get("raw_sha256"):
        raise S1InstrumentationError("method authority identity mismatch")
    if config.get("conditions") != list(CONDITION_ORDER) or config.get("condition_axes") != {key: list(value) for key, value in CONDITION_STATES.items()}:
        raise S1InstrumentationError("S1 condition identity changed")
    identity = config.get("transformer_identity", {})
    if (identity.get("block_count"), identity.get("block_index"), identity.get("head_count"), identity.get("head_dim"), identity.get("token_grid"), identity.get("token_count")) != (30, 14, 12, 128, [13, 20, 32], 8320):
        raise S1InstrumentationError("S1 transformer identity changed")
    support = config.get("flow_support", {})
    if (support.get("scheduler_index"), support.get("expected_timestep"), support.get("lambda")) != (4, 749, 1.0):
        raise S1InstrumentationError("S1 Flow support changed")
    budget = config.get("call_budget", {})
    if (budget.get("prefix_transformer_calls"), budget.get("probe_transformer_calls"), budget.get("total_transformer_calls"), budget.get("retry_count")) != (8, 12, 20, 0):
        raise S1InstrumentationError("S1 transformer-call budget changed")
    if plan.get("condition_order") != list(CONDITION_ORDER) or plan.get("exact_transformer_calls") != 20:
        raise S1InstrumentationError("S1 plan identity changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise S1InstrumentationError("S1 diagnostic boundary changed")
    return config, plan


def odd_even_noise(records: Mapping[str, Any], axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plus_name, minus_name = (("PLUS_B1", "MINUS_B1") if axis == "B1" else ("PLUS_B2", "MINUS_B2"))
    plus = np.asarray(records[plus_name], dtype=np.float64)
    minus = np.asarray(records[minus_name], dtype=np.float64)
    off_1 = np.asarray(records["OFF_R1"], dtype=np.float64)
    off_2 = np.asarray(records["OFF_R2"], dtype=np.float64)
    if plus.shape != minus.shape or plus.shape != off_1.shape or plus.shape != off_2.shape or not all(np.isfinite(value).all() for value in (plus, minus, off_1, off_2)):
        raise S1InstrumentationError("O/E/N inputs differ or are non-finite")
    odd = (plus - minus) / 2.0
    off_mean = (off_1 + off_2) / 2.0
    even = (plus + minus) / 2.0 - off_mean
    noise = (off_1 - off_2) / math.sqrt(2.0)
    return odd, even, noise, off_mean


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64), dtype=np.float64)))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left.ravel()) * np.linalg.norm(right.ravel()))
    return 0.0 if denominator == 0.0 else float(np.dot(left.ravel(), right.ravel()) / denominator)


def evaluate_preregistered_statistics(
    probe: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    global_relative_overrides: Mapping[tuple[str, str, str], float] | None = None,
) -> dict[str, Any]:
    """Evaluate every layer x branch cell without cross-cell averaging."""

    branch_inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for branch in (*BRANCH_ORDER, "guidance"):
        branch_inputs[branch] = {}
        for condition in CONDITION_ORDER:
            branch_inputs[branch][condition] = {}
            for layer in LAYER_ORDER:
                if branch == "guidance":
                    cond = np.asarray(probe[condition]["cond"][layer], dtype=np.float64)
                    uncond = np.asarray(probe[condition]["uncond"][layer], dtype=np.float64)
                    value = uncond + 5.0 * (cond - uncond)
                else:
                    value = np.asarray(probe[condition][branch][layer], dtype=np.float64)
                branch_inputs[branch][condition][layer] = value

    cells: dict[str, Any] = {}
    all_pass = True
    for branch in (*BRANCH_ORDER, "guidance"):
        for layer in LAYER_ORDER:
            layer_records = {condition: branch_inputs[branch][condition][layer] for condition in CONDITION_ORDER}
            axes: dict[str, Any] = {}
            odd_vectors: dict[str, np.ndarray] = {}
            for axis in ("B1", "B2"):
                odd, even, noise, baseline = odd_even_noise(layer_records, axis)
                odd_vectors[axis] = odd
                odd_rms, even_rms, noise_rms, baseline_rms = map(_rms, (odd, even, noise, baseline))
                ulp_floor = max(float(np.max(np.abs(baseline))) * (2.0 ** -7), 2.0 ** -133)
                separation = odd_rms / max(noise_rms, ulp_floor)
                even_ratio = math.inf if odd_rms == 0.0 else even_rms / odd_rms
                temporal_tv = math.inf if odd_rms == 0.0 else _rms(np.diff(odd, axis=0)) / odd_rms
                relative_rms = math.inf if baseline_rms == 0.0 else odd_rms / baseline_rms
                if global_relative_overrides is not None and (branch, layer, axis) in global_relative_overrides:
                    relative_rms = float(global_relative_overrides[(branch, layer, axis)])
                axis_pass = (
                    separation >= float(thresholds["odd_to_noise_or_ulp_minimum"])
                    and even_ratio < float(thresholds["even_to_odd_strict_maximum"])
                    and temporal_tv < float(thresholds["temporal_total_variation_strict_maximum"])
                )
                if layer in {"block", "velocity"}:
                    axis_pass = axis_pass and math.isfinite(relative_rms) and relative_rms < float(thresholds[f"{layer}_global_relative_rms_strict_maximum"])
                axes[axis] = {
                    "odd_rms": odd_rms, "even_rms": even_rms, "noise_rms": noise_rms,
                    "baseline_rms": baseline_rms, "bfloat16_ulp_floor": ulp_floor,
                    "odd_to_noise_or_ulp": separation, "even_to_odd": even_ratio,
                    "temporal_total_variation": temporal_tv, "global_relative_rms": relative_rms,
                    "passed": bool(axis_pass),
                }
            axis_cosine = abs(_cosine(odd_vectors["B1"], odd_vectors["B2"]))
            gain_ratio = axes["B1"]["odd_rms"] / axes["B2"]["odd_rms"] if axes["B2"]["odd_rms"] else math.inf
            pair_pass = (
                axis_cosine < float(thresholds["axis_absolute_cosine_strict_maximum"])
                and float(thresholds["axis_gain_ratio_inclusive"][0]) <= gain_ratio <= float(thresholds["axis_gain_ratio_inclusive"][1])
            )
            relation_jacobian = None
            if layer == "relation":
                j11 = float(np.mean(odd_vectors["B1"][..., 0]))
                j21 = float(np.mean(odd_vectors["B1"][..., 1]))
                j12 = float(np.mean(odd_vectors["B2"][..., 0]))
                j22 = float(np.mean(odd_vectors["B2"][..., 1]))
                cross = max(abs(j21) / max(abs(j11), 1e-30), abs(j12) / max(abs(j22), 1e-30))
                jacobian_pass = j11 > 0.0 and j22 > 0.0 and cross <= float(thresholds["relation_jacobian_cross_ratio_inclusive_maximum"])
                relation_jacobian = {"matrix_columns_B1_B2": [[j11, j12], [j21, j22]], "maximum_cross_ratio": cross, "passed": jacobian_pass}
                pair_pass = pair_pass and jacobian_pass
            cell_pass = pair_pass and all(record["passed"] for record in axes.values())
            all_pass = all_pass and cell_pass
            cells[f"{layer}:{branch}"] = {
                "axes": axes, "axis_absolute_cosine": axis_cosine, "axis_gain_ratio": gain_ratio,
                "relation_jacobian": relation_jacobian, "passed": bool(cell_pass),
            }
    return {"cells": cells, "all_cells_pass": bool(all_pass)}


def _extract_velocity_slice(value: Any) -> Any:
    torch = __import__("torch")
    slices = []
    for query in TARGET_QUERY_INDICES:
        temporal = query // (20 * 32)
        remainder = query % (20 * 32)
        height = remainder // 32
        width = remainder % 32
        slices.append(value[0, :, temporal, height * 2 : height * 2 + 2, width * 2 : width * 2 + 2])
    return torch.stack(slices, dim=0).detach().float().cpu()


def _load_runtime(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import accelerate, diffusers, ftfy, huggingface_hub, safetensors, torch, transformers
        from diffusers import WanPipeline, WanTransformer3DModel
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise S1InstrumentationError("locked S1 CUDA dependencies unavailable") from exc
    software = config["software"]
    versions = {
        "python": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "accelerate": accelerate.__version__, "diffusers": diffusers.__version__, "ftfy": ftfy.__version__,
        "huggingface_hub": huggingface_hub.__version__, "numpy": np.__version__,
        "safetensors": safetensors.__version__, "transformers": transformers.__version__,
    }
    if versions != software:
        raise S1InstrumentationError(f"locked software identity mismatch: {versions}")
    transformer_source = Path(inspect.getsourcefile(WanTransformer3DModel) or "").resolve(strict=True)
    pipeline_source = Path(inspect.getsourcefile(WanPipeline) or "").resolve(strict=True)
    topology = config["source_topology"]
    if sha256_file(transformer_source) != topology["transformer_wan_raw_sha256"] or sha256_file(pipeline_source) != topology["pipeline_wan_raw_sha256"]:
        raise S1InstrumentationError("installed diffusers Wan source identity mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.get_device_name(0) != config["model"]["required_gpu"]:
        raise S1InstrumentationError("exact NVIDIA L4 BF16 runtime unavailable")
    snapshot = Path(snapshot_download(repo_id=config["model"]["id"], revision=config["model"]["revision"], local_files_only=True))
    resolved = snapshot.resolve(strict=True)
    if snapshot.is_symlink() or not snapshot.is_dir() or resolved.name != config["model"]["revision"]:
        raise S1InstrumentationError("frozen model snapshot identity mismatch")
    pipe = WanPipeline.from_pretrained(str(resolved), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload()
    transformer = pipe.transformer
    identity = config["transformer_identity"]
    target = transformer.blocks[identity["block_index"]].attn1
    if (
        not isinstance(transformer, WanTransformer3DModel)
        or len(transformer.blocks) != identity["block_count"]
        or tuple(transformer.config.patch_size) != tuple(identity["patch_size"])
        or target.heads != identity["head_count"]
        or target.inner_dim != identity["inner_dim"]
        or target.to_q.out_features // target.heads != identity["head_dim"]
        or str(transformer.dtype) != config["model"]["dtype"]
    ):
        raise S1InstrumentationError("real Wan transformer structure mismatch")
    runtime = {**versions, "gpu": torch.cuda.get_device_name(0), "model_snapshot": str(resolved),
               "transformer_type": type(transformer).__name__, "block_type": type(transformer.blocks[14]).__name__,
               "attention_type": type(target).__name__, "execution_device": str(pipe._execution_device),
               "transformer_source": str(transformer_source), "transformer_source_sha256": sha256_file(transformer_source),
               "pipeline_source": str(pipeline_source), "pipeline_source_sha256": sha256_file(pipeline_source)}
    return pipe, torch, runtime


def run_s1_once(*, repo_root: Path, output: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink() or not output.parent.is_dir() or output.parent.is_symlink():
        raise S1InstrumentationError("output must be absent under an existing regular parent")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise S1InstrumentationError("S1 requires a clean exact checkout")
    config, plan = load_frozen_inputs(repo_root)
    output.mkdir(mode=0o755)
    started_at = utc_now()
    call_records: list[dict[str, Any]] = []
    probe: dict[str, Any] = {condition: {} for condition in CONDITION_ORDER}
    try:
        pipe, torch, runtime = _load_runtime(config)
        transformer = pipe.transformer
        target_block = transformer.blocks[14]
        attention_parameter_sha256_before = sha256_parameters(target_block.attn1)
        processor, original_processor = install_s1_processor(transformer, torch, block_index=14, strength=1.0)
        captured_blocks: list[tuple[Any, float, Any]] = []

        def block_hook(_module: Any, _inputs: Any, block_output: Any) -> None:
            indices = torch.tensor(TARGET_QUERY_INDICES, device=block_output.device, dtype=torch.long)
            full = block_output.detach().float().cpu()
            captured_blocks.append((full.index_select(1, indices), float(full.square().mean().sqrt().item()), full))

        hook = target_block.register_forward_hook(block_hook)
        device = pipe._execution_device
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=config["generation"]["prompt"], negative_prompt=config["generation"]["negative_prompt"],
            do_classifier_free_guidance=True, num_videos_per_prompt=1, device=device,
        )
        prompt_embeds = prompt_embeds.to(transformer.dtype)
        negative_embeds = negative_embeds.to(transformer.dtype)
        pipe.scheduler.set_timesteps(config["generation"]["inference_steps"], device=device)
        timesteps = pipe.scheduler.timesteps
        actual_timesteps = [int(value.item()) for value in timesteps]
        if len(actual_timesteps) != 8 or actual_timesteps[4] != config["flow_support"]["expected_timestep"]:
            raise S1InstrumentationError(f"scheduler identity mismatch: {actual_timesteps}")
        generator = torch.Generator(device="cuda").manual_seed(config["generation"]["seed"])
        latents = pipe.prepare_latents(1, transformer.config.in_channels, 320, 512, 49, torch.float32, device, generator, None)

        def one_call(latent_input: Any, timestep: Any, scheduler_index: int, condition: str, branch: str, state: tuple[float, float], keep: bool) -> tuple[Any, dict[str, Any]]:
            call_index = len(call_records) + 1
            processor.set_context(call_index=call_index, scheduler_index=scheduler_index, timestep=int(timestep.item()), condition=condition, branch=branch, state=state)
            before_relation = len(processor.records)
            before_block = len(captured_blocks)
            embeddings = prompt_embeds if branch == "cond" else negative_embeds
            with transformer.cache_context(branch):
                velocity = transformer(hidden_states=latent_input.clone().to(transformer.dtype), timestep=timestep.expand(1), encoder_hidden_states=embeddings, attention_kwargs=None, return_dict=False)[0]
            if len(processor.records) != before_relation + 1 or len(captured_blocks) != before_block + 1:
                raise S1InstrumentationError("processor/block instrumentation call count mismatch")
            relation_record = processor.records[-1]
            block_slice, block_global_rms, block_full = captured_blocks.pop()
            velocity_slice = _extract_velocity_slice(velocity)
            call_identity = {"call_index": call_index, "scheduler_index": scheduler_index, "timestep": int(timestep.item()),
                             "condition": condition, "branch": branch, "state": list(state), "kept_for_probe": keep}
            call_records.append(call_identity)
            captured = {
                "relation": relation_record["relation"], "block": block_slice.tolist(), "velocity": velocity_slice.tolist(),
                "block_global_rms": block_global_rms, "_block_full": block_full, "_velocity_full": velocity.detach().float().cpu(),
                "processor": {key: value for key, value in relation_record.items() if key != "relation"},
            }
            return velocity, captured

        for scheduler_index in range(4):
            timestep = timesteps[scheduler_index]
            cond_velocity, _ = one_call(latents, timestep, scheduler_index, f"PREFIX_{scheduler_index}", "cond", (0.0, 0.0), False)
            uncond_velocity, _ = one_call(latents, timestep, scheduler_index, f"PREFIX_{scheduler_index}", "uncond", (0.0, 0.0), False)
            guided = uncond_velocity + config["generation"]["guidance_scale"] * (cond_velocity - uncond_velocity)
            latents = pipe.scheduler.step(guided, timestep, latents, return_dict=False)[0]

        probe_latent = latents.detach().clone()
        probe_latent_sha256 = sha256_tensor(probe_latent)
        timestep = timesteps[4]
        for condition in CONDITION_ORDER:
            probe[condition] = {}
            for branch in BRANCH_ORDER:
                _velocity, captured = one_call(probe_latent, timestep, 4, condition, branch, CONDITION_STATES[condition], True)
                probe[condition][branch] = captured
            if condition == "OFF_R1":
                equivalence_records = [probe[condition][branch]["processor"] for branch in BRANCH_ORDER]
                if not all(
                    item["lambda_zero_reference_relative_rms"] <= config["thresholds"]["lambda_zero_relative_rms_maximum"]
                    and item["lambda_zero_reference_max_bfloat16_ulp"] <= config["thresholds"]["lambda_zero_max_bfloat16_ulp"]
                    for item in equivalence_records
                ):
                    raise S1InstrumentationError("lambda-zero sparse-row recompute equivalence failed before injection")
        hook.remove()
        target_block.attn1.set_processor(original_processor)
        attention_parameter_sha256_after = sha256_parameters(target_block.attn1)
        if attention_parameter_sha256_after != attention_parameter_sha256_before:
            raise S1InstrumentationError("target attention parameters changed during S1")
        if len(call_records) != 20 or sum(item["kept_for_probe"] for item in call_records) != 12:
            raise S1InstrumentationError("exact20 runtime call budget mismatch")

        numeric_probe = {
            condition: {
                branch: {layer: probe[condition][branch][layer] for layer in LAYER_ORDER}
                for branch in BRANCH_ORDER
            }
            for condition in CONDITION_ORDER
        }
        global_overrides: dict[tuple[str, str, str], float] = {}
        for branch in (*BRANCH_ORDER, "guidance"):
            for axis in ("B1", "B2"):
                plus_name, minus_name = (("PLUS_B1", "MINUS_B1") if axis == "B1" else ("PLUS_B2", "MINUS_B2"))
                if branch == "guidance":
                    full_block = {
                        condition: probe[condition]["uncond"]["_block_full"] + 5.0 * (probe[condition]["cond"]["_block_full"] - probe[condition]["uncond"]["_block_full"])
                        for condition in CONDITION_ORDER
                    }
                    full_velocity = {
                        condition: probe[condition]["uncond"]["_velocity_full"] + 5.0 * (probe[condition]["cond"]["_velocity_full"] - probe[condition]["uncond"]["_velocity_full"])
                        for condition in CONDITION_ORDER
                    }
                else:
                    full_block = {condition: probe[condition][branch]["_block_full"] for condition in CONDITION_ORDER}
                    full_velocity = {condition: probe[condition][branch]["_velocity_full"] for condition in CONDITION_ORDER}
                block_odd = (full_block[plus_name] - full_block[minus_name]) / 2.0
                block_baseline = (full_block["OFF_R1"] + full_block["OFF_R2"]) / 2.0
                global_overrides[(branch, "block", axis)] = float(
                    block_odd.square().mean().sqrt().item() / max(block_baseline.square().mean().sqrt().item(), 1e-30)
                )
                velocity_odd = (full_velocity[plus_name] - full_velocity[minus_name]) / 2.0
                velocity_baseline = (full_velocity["OFF_R1"] + full_velocity["OFF_R2"]) / 2.0
                global_overrides[(branch, "velocity", axis)] = float(
                    velocity_odd.square().mean().sqrt().item() / max(velocity_baseline.square().mean().sqrt().item(), 1e-30)
                )
        evaluation = evaluate_preregistered_statistics(numeric_probe, config["thresholds"], global_overrides)
        structural = {
            "all_processor_records_finite": all(np.isfinite(np.asarray(record["relation"], dtype=np.float64)).all() for record in processor.records),
            "all_pair_sums_zero": all(record["pair_sums_zero"] for record in processor.records),
            "all_non_target_rows_unreplaced": all(record["non_target_rows_replaced"] == 0 for record in processor.records),
            "dense_bias_never_materialized": all(record["dense_bias_materialized"] is False for record in processor.records),
            "probe_bias_counts": [probe[condition][branch]["processor"]["changed_logit_count"] for condition in CONDITION_ORDER for branch in BRANCH_ORDER],
            "attention_parameter_sha256_before": attention_parameter_sha256_before,
            "attention_parameter_sha256_after": attention_parameter_sha256_after,
        }
        structural["passed"] = (
            structural["all_processor_records_finite"] and structural["all_pair_sums_zero"]
            and structural["all_non_target_rows_unreplaced"] and structural["dense_bias_never_materialized"]
            and structural["probe_bias_counts"] == [0, 0, 0, 0, 26, 26, 26, 26, 26, 26, 26, 26]
        )
        status = "S1_GO" if evaluation["all_cells_pass"] and structural["passed"] else "S1_NO_GO_THIS_CONSTRUCTION"
        serializable_probe = {
            condition: {
                branch: {key: value for key, value in probe[condition][branch].items() if not key.startswith("_")}
                for branch in BRANCH_ORDER
            }
            for condition in CONDITION_ORDER
        }
        stats = {"schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "probe_latent_sha256": probe_latent_sha256,
                 "scheduler_timesteps": actual_timesteps, "call_records": call_records, "probe": serializable_probe,
                 "evaluation": evaluation, "structural": structural}
        audit = {"schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status,
                 "source": {"head": _git(repo_root, "rev-parse", "HEAD"), "tree": _git(repo_root, "rev-parse", "HEAD^{tree}"), "dirty": False},
                 "config_sha256": sha256_file(repo_root / "configs/s1_real_dit_relation_primitive.json"),
                 "plan_sha256": sha256_file(repo_root / "plans/s1_real_dit_relation_primitive.json"),
                 "authority_sha256": sha256_file(repo_root / "SSTW_METHOD_AUTHORITY.md"),
                 "runtime": runtime, "transformer_calls": len(call_records), "scheduler_steps": 4,
                 "VAE_was_run": False, "MP4_was_written": False, "formal_result": False, "stage_progression_allowed": False,
                 "started_at": started_at, "ended_at": utc_now()}
        (output / "stats.json").write_bytes(canonical_json_bytes(stats) + b"\n")
        (output / "audit.json").write_bytes(canonical_json_bytes(audit) + b"\n")
        (output / "config.json").write_bytes(canonical_json_bytes(config) + b"\n")
        (output / "plan.json").write_bytes(canonical_json_bytes(plan) + b"\n")
        (output / "command.json").write_bytes(canonical_json_bytes({"argv": list(argv), "cwd": str(cwd.resolve()), "exit_code": 0}) + b"\n")
        files = sorted(path for path in output.iterdir() if path.is_file())
        (output / "checksums.sha256").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
        return {"status": status, "diagnostic_class": DIAGNOSTIC_CLASS, "output": str(output), "transformer_calls": 20,
                "formal_result": False, "stage_progression_allowed": False}
    except Exception:
        raise
