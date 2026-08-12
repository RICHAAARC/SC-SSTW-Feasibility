"""Fresh-content saved-MP4 diagnostic for the SSTW-v2 Flow carrier."""

from __future__ import annotations

import copy
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .flow_guidance_embedder import (
    G0InstrumentationError,
    OBSERVER_FRAME_INDICES,
    _continue_condition,
    _emit_progress,
    _tensor_cosine_float,
    _tensor_rms_float,
    _transformer_velocity,
    capture_unipc_snapshot,
    decode_wan_latents_torch,
    fixed_observer_torch,
    normalized_guidance_update_torch,
    observer_projection_loss_torch,
    runtime_capability_diagnostics,
    wan_checkpoint_canary_torch,
)
from .state_generator import FrozenStateConfig, keyed_trajectory


CONDITIONS = ("OFF_R1", "OFF_R2", "A", "B")
ALLOWED_STATUSES = (
    "FLOW_GUIDANCE_SAVED_MP4_FEASIBLE",
    "FLOW_GUIDANCE_SAVED_MP4_NOT_FEASIBLE",
    "INSTRUMENTATION_INSUFFICIENT",
)


class G1InstrumentationError(G0InstrumentationError):
    """A runtime failure that cannot answer the saved-MP4 question."""


def load_g1_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise G1InstrumentationError("G1 config is unavailable") from exc
    if (
        value.get("protocol_id") != "sstw_v2_g1_flow_guidance_saved_mp4"
        or value.get("diagnostic_class") != "DIAGNOSTIC_ONLY"
        or value.get("model", {}).get("repository") != "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        or value.get("model", {}).get("revision") != "0fad780a534b6463e45facd96134c9f345acfa5b"
        or value.get("guidance", {}).get("conditions") != list(CONDITIONS)
        or value.get("guidance", {}).get("active_after_scheduler_index") != 5
        or value.get("guidance", {}).get("normal_steps_remaining") != [6, 7]
        or value.get("guidance", {}).get("relative_update_rms") != 0.005
        or value.get("allowed_statuses") != list(ALLOWED_STATUSES)
    ):
        raise G1InstrumentationError("G1 frozen construction changed")
    groups = value.get("generation", {}).get("groups")
    if not isinstance(groups, list) or [item.get("id") for item in groups] != ["fresh_lighthouse", "fresh_glass_garden"]:
        raise G1InstrumentationError("G1 fresh group set changed")
    trajectory = value.get("guidance", {}).get("trajectory", {})
    if trajectory.get("length") != 13 or trajectory.get("delta_times") != [1] * 12:
        raise G1InstrumentationError("G1 trajectory identity changed")
    criteria = value.get("criteria", {})
    if criteria != {
        "minimum_effect_to_floor_ratio": 8.0,
        "numeric_floor_float32_eps_multiplier": 32.0,
        "maximum_channel_condition_number": 10.0,
        "maximum_fit_relative_residual": 0.5,
        "maximum_final_rgb_relative_rms": 0.02,
    }:
        raise G1InstrumentationError("G1 criteria changed")
    return value


def frozen_trajectories(config: Mapping[str, Any]) -> dict[str, tuple[tuple[float, float], ...]]:
    item = config["guidance"]["trajectory"]
    common = dict(omega_0=float(item["omega_0"]), delta_phi=float(item["delta_phi"]))
    delta_times = tuple(float(v) for v in item["delta_times"])
    return {
        "A": keyed_trajectory(bytes.fromhex(item["key_a_hex"]), 13, delta_times, FrozenStateConfig(**common, initial_phase=float(item["initial_phase_a"]))),
        "B": keyed_trajectory(bytes.fromhex(item["key_b_hex"]), 13, delta_times, FrozenStateConfig(**common, initial_phase=float(item["initial_phase_b"]))),
    }


def trajectory_gradients_torch(base_latent: Any, vae: Any, trajectories: Mapping[str, Sequence[Sequence[float]]], torch_module: Any):
    gradients: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for name in ("A", "B"):
        differentiable = base_latent.detach().clone().requires_grad_(True)
        _emit_progress(f"g1_vae_vjp_{name}_start", torch_module)
        try:
            from .flow_guidance_embedder import _checkpoint_wan_decoder_frames
            with torch_module.enable_grad(), _checkpoint_wan_decoder_frames(vae, torch_module):
                decoded = decode_wan_latents_torch(differentiable, vae, torch_module)
                loss = observer_projection_loss_torch(decoded, trajectories[name], torch_module)
                gradient = torch_module.autograd.grad(loss, differentiable)[0]
            gradients[name] = gradient.detach()
            diagnostics[name] = {"loss": float(loss.detach().float().item()), "gradient_rms": _tensor_rms_float(gradient, torch_module)}
        except Exception as exc:
            raise G1InstrumentationError(f"G1 {name} VJP failed: {type(exc).__name__}: {exc}") from exc
        finally:
            differentiable = decoded = loss = gradient = None
            vae.clear_cache()
            gc.collect()
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        _emit_progress(f"g1_vae_vjp_{name}_complete", torch_module)
    diagnostics["absolute_cosine"] = abs(_tensor_cosine_float(gradients["A"], gradients["B"], torch_module))
    return gradients, diagnostics


def condition_latents(base: Any, gradients: Mapping[str, Any], relative_rms: float, torch_module: Any):
    update_a, diag_a = normalized_guidance_update_torch(base, gradients["A"], torch_module, relative_update_rms=relative_rms)
    update_b, diag_b = normalized_guidance_update_torch(base, gradients["B"], torch_module, relative_update_rms=relative_rms)
    values = {"OFF_R1": base.detach().clone(), "OFF_R2": base.detach().clone(), "A": (base + update_a).detach(), "B": (base + update_b).detach()}
    if tuple(values) != CONDITIONS or values["OFF_R1"].data_ptr() == values["OFF_R2"].data_ptr():
        raise G1InstrumentationError("G1 condition forks are invalid")
    return values, {"A": diag_a, "B": diag_b}


def _observer_numpy(frames: Any, numpy_module: Any) -> list[list[float]]:
    frames = numpy_module.asarray(frames)
    if frames.shape != (49, 320, 512, 3):
        raise G1InstrumentationError(f"saved MP4 shape changed: {frames.shape}")
    rgb = frames.astype(numpy_module.float32) / 127.5 - 1.0
    rgb = rgb[numpy_module.asarray(OBSERVER_FRAME_INDICES)]
    h = numpy_module.cos(2 * numpy_module.pi * (numpy_module.arange(512) + 0.5) / 512); h = (h-h.mean()) / numpy_module.sqrt(numpy_module.mean((h-h.mean())**2))
    v = numpy_module.cos(2 * numpy_module.pi * (numpy_module.arange(320) + 0.5) / 320); v = (v-v.mean()) / numpy_module.sqrt(numpy_module.mean((v-v.mean())**2))
    q1 = numpy_module.mean((rgb[...,0]-rgb[...,1]) * h[None,None,:], axis=(1,2))
    q2 = numpy_module.mean((rgb[...,2]-0.5*(rgb[...,0]+rgb[...,1])) * v[None,:,None], axis=(1,2))
    return numpy_module.stack((q1,q2),axis=1).astype(float).tolist()


def evaluate_group(observations: Mapping[str, Sequence[Sequence[float]]], trajectories: Mapping[str, Sequence[Sequence[float]]], rgb_quality: Mapping[str, float], criteria: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    q = {name: np.asarray(observations[name], dtype=np.float64) for name in CONDITIONS}
    off = (q["OFF_R1"] + q["OFF_R2"]) / 2.0
    off_floor = float(np.sqrt(np.mean((q["OFF_R1"]-q["OFF_R2"])**2)))
    numeric = float(criteria["numeric_floor_float32_eps_multiplier"]) * 2.0**-23 * max(1.0, float(np.sqrt(np.mean(off**2))))
    floor = max(off_floor, numeric)
    design = np.concatenate([np.asarray(trajectories["A"]), np.asarray(trajectories["B"])], axis=0)
    response = np.concatenate([q["A"]-off, q["B"]-off], axis=0)
    matrix_t, *_ = np.linalg.lstsq(design, response, rcond=None)
    matrix = matrix_t.T
    singular = np.linalg.svd(matrix, compute_uv=False)
    condition = float("inf") if singular[-1] == 0 else float(singular[0]/singular[-1])
    predicted = design @ matrix_t
    residual = float(np.sqrt(np.mean((response-predicted)**2)) / max(float(np.sqrt(np.mean(response**2))), 1e-30))
    effects = {name: float(np.sqrt(np.mean((q[name]-off)**2))) for name in ("A","B")}
    checks = {
        "matrix_finite_full_rank": bool(np.isfinite(matrix).all() and singular[-1] > 0),
        "condition_at_most_10": math.isfinite(condition) and condition <= float(criteria["maximum_channel_condition_number"]),
        "fit_relative_residual": residual <= float(criteria["maximum_fit_relative_residual"]),
        "A_effect_above_floor": effects["A"] >= float(criteria["minimum_effect_to_floor_ratio"])*floor,
        "B_effect_above_floor": effects["B"] >= float(criteria["minimum_effect_to_floor_ratio"])*floor,
        "A_rgb_quality": float(rgb_quality["A"]) <= float(criteria["maximum_final_rgb_relative_rms"]),
        "B_rgb_quality": float(rgb_quality["B"]) <= float(criteria["maximum_final_rgb_relative_rms"]),
    }
    return {"checks": checks, "matrix": matrix.tolist(), "singular_values": singular.tolist(), "condition": condition, "fit_relative_residual": residual, "effect_rms": effects, "floor": {"off_repeat_rms": off_floor, "numeric": numeric, "effective": floor}, "passed": all(checks.values())}


def run_g1_once(repo_root: Path, output: Path, config_path: Path | None = None) -> dict[str, Any]:
    if output.exists() or not output.parent.is_dir():
        raise G1InstrumentationError("G1 output must be absent")
    config = load_g1_config(config_path or repo_root / "configs/g1_flow_guidance_saved_mp4.json")
    try:
        import diffusers, imageio.v3 as iio, numpy as np, torch
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.utils import export_to_video
    except Exception as exc:
        raise G1InstrumentationError("G1 dependencies unavailable") from exc
    runtime = runtime_capability_diagnostics(torch); runtime["diffusers"] = diffusers.__version__
    pipe = WanPipeline.from_pretrained(config["model"]["repository"], revision=config["model"]["revision"], torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload(); device = pipe._execution_device
    canary = wan_checkpoint_canary_torch(torch, AutoencoderKLWan, device)
    trajectories = frozen_trajectories(config)
    output.mkdir(); videos = output / "videos"; videos.mkdir()
    groups: dict[str, Any] = {}; transformer_calls = vae_decodes = 0
    generation = config["generation"]
    try:
        for group in generation["groups"]:
            group_id = group["id"]; _emit_progress(f"g1_{group_id}_prefix", torch)
            with torch.inference_mode():
                cond, uncond = pipe.encode_prompt(prompt=group["prompt"], negative_prompt=generation["negative_prompt"], do_classifier_free_guidance=True, num_videos_per_prompt=1, max_sequence_length=generation["max_sequence_length"], device=device)
                cond=cond.to(pipe.transformer.dtype).detach(); uncond=uncond.to(pipe.transformer.dtype).detach()
                pipe.scheduler.set_timesteps(generation["inference_steps"], device=device)
                if hasattr(pipe.scheduler,"set_begin_index"): pipe.scheduler.set_begin_index(0)
                ts=pipe.scheduler.timesteps
                latent=pipe.prepare_latents(1,int(pipe.transformer.config.in_channels),generation["height"],generation["width"],generation["frames"],torch.float32,device,torch.Generator(device=device).manual_seed(group["seed"]),None).detach()
                for index in range(6):
                    cv=_transformer_velocity(pipe,latent,ts[index],cond,"cond",torch); uv=_transformer_velocity(pipe,latent,ts[index],uncond,"uncond",torch); transformer_calls+=2
                    guided=uv+generation["guidance_scale"]*(cv-uv); latent=pipe.scheduler.step(guided,ts[index],latent,return_dict=False)[0].detach()
            snapshot, summary = capture_unipc_snapshot(pipe.scheduler,torch,expected_step_index=6)
            gradients, grad_diag = trajectory_gradients_torch(latent,pipe.vae,trajectories,torch); vae_decodes+=2
            forks, update_diag = condition_latents(latent,gradients,config["guidance"]["relative_update_rms"],torch); del gradients
            pipe.maybe_free_model_hooks(); gc.collect(); torch.cuda.empty_cache()
            final={}
            for name in CONDITIONS:
                final[name],calls=_continue_condition(pipe,forks[name],snapshot,summary,ts,cond,uncond,config,torch); transformer_calls+=calls
            observations={}; rgb={}; mp4s={}
            with torch.inference_mode():
                for name in CONDITIONS:
                    decoded=decode_wan_latents_torch(final[name],pipe.vae,torch); vae_decodes+=1
                    array=((decoded[0].permute(1,2,3,0).float().clamp(-1,1)+1)*127.5).round().to(torch.uint8).cpu().numpy()
                    path=videos/f"{group_id}_{name}.mp4"; export_to_video(list(array),str(path),fps=generation["fps"],quality=5.0,bitrate=None,macro_block_size=16)
                    saved=np.stack(list(iio.imiter(path,plugin="FFMPEG")),axis=0)
                    observations[name]=_observer_numpy(saved,np); rgb[name]=decoded.detach().float().cpu(); mp4s[name]={"path":str(path.relative_to(output)),"size":path.stat().st_size}
            rgb_quality={name: float(((rgb[name]-rgb["OFF_R1"]).square().mean().sqrt()/rgb["OFF_R1"].square().mean().sqrt()).item()) for name in ("A","B")}
            evaluation=evaluate_group(observations,trajectories,rgb_quality,config["criteria"])
            groups[group_id]={"prompt":group["prompt"],"seed":group["seed"],"gradient":grad_diag,"updates":update_diag,"observations":observations,"rgb_quality":rgb_quality,"videos":mp4s,"evaluation":evaluation}
        status="FLOW_GUIDANCE_SAVED_MP4_FEASIBLE" if all(v["evaluation"]["passed"] for v in groups.values()) else "FLOW_GUIDANCE_SAVED_MP4_NOT_FEASIBLE"
        result={"schema_version":1,"protocol_id":config["protocol_id"],"diagnostic_class":"DIAGNOSTIC_ONLY","status":status,"runtime":runtime,"checkpoint_canary":canary,"trajectories":{k:[list(p) for p in v] for k,v in trajectories.items()},"groups":groups,"execution":{"transformer_calls":transformer_calls,"vae_decodes":vae_decodes,"mp4_count":8}}
        if transformer_calls != 56 or vae_decodes != 12: raise G1InstrumentationError("G1 execution budget changed")
        data=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode(); temp=output/".result.json.tmp"
        with temp.open("xb") as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp,output/"result.json"); return result
    finally:
        try: pipe.maybe_free_model_hooks()
        except Exception: pass


def g1_exit_code(status: str) -> int:
    return 0 if status == ALLOWED_STATUSES[0] else 3 if status == ALLOWED_STATUSES[1] else 2
