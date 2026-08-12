"""Single frozen structured-initial-noise saved-MP4 diagnostic."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

from .flow_guidance_embedder import (
    G0InstrumentationError,
    _emit_progress,
    _transformer_velocity,
    decode_wan_latents_torch,
    runtime_capability_diagnostics,
)
from .flow_guidance_saved_mp4 import _saved_rgb_quality
from .state_generator import FrozenStateConfig, keyed_trajectory

CONDITIONS = ("OFF_R1", "OFF_R2", "A", "B")


class NoiseInstrumentationError(G0InstrumentationError):
    pass

def load_noise_config(path: Path):
    try:
        c = json.loads(path.read_text())
    except Exception as exc:
        raise NoiseInstrumentationError("noise config unavailable") from exc
    if (
        c.get("protocol_id") != "sstw_noise_g0_structured_initial_noise"
        or c["carrier"]["relative_initial_noise_rms"] != 0.03
        or c["carrier"]["conditions"] != list(CONDITIONS)
        or c["carrier"]["basis"]
        != {"a_channels": [0, 1], "b_channels": [2, 3], "spatial_cycles": 1}
        or c["generation"]["inference_steps"] != 8
        or (c["generation"]["frames"], c["generation"]["height"], c["generation"]["width"])
        != (49, 320, 512)
    ):
        raise NoiseInstrumentationError("noise construction changed")
    return c

def trajectories(c):
    x=c["carrier"]["trajectory"]; common=dict(omega_0=x["omega_0"],delta_phi=x["delta_phi"]); dt=x["delta_times"]
    return {"A":keyed_trajectory(bytes.fromhex(x["key_a_hex"]),13,dt,FrozenStateConfig(**common,initial_phase=x["initial_phase_a"])),"B":keyed_trajectory(bytes.fromhex(x["key_b_hex"]),13,dt,FrozenStateConfig(**common,initial_phase=x["initial_phase_b"]))}

def latent_basis(torch, z, trajectory):
    """Map a public 2-D state trajectory to two antisymmetric channel pairs."""

    if tuple(z.shape) != (1, 16, 13, 40, 64):
        raise NoiseInstrumentationError("Wan initial latent shape changed")
    tr = torch.as_tensor(trajectory, device=z.device, dtype=z.dtype)
    if tuple(tr.shape) != (13, 2) or not bool(torch.isfinite(tr).all().item()):
        raise NoiseInstrumentationError("noise trajectory identity changed")
    b = torch.zeros_like(z)
    x = torch.arange(64, device=z.device, dtype=z.dtype)
    y = torch.arange(40, device=z.device, dtype=z.dtype)
    h = torch.cos(2 * torch.pi * (x + 0.5) / 64)
    h = (h - h.mean()) / h.square().mean().sqrt()
    v = torch.cos(2 * torch.pi * (y + 0.5) / 40)
    v = (v - v.mean()) / v.square().mean().sqrt()
    b[:, 0] = tr[:, 0][None, :, None, None] * h[None, None, None, :]
    b[:, 1] = -b[:, 0]
    b[:, 2] = tr[:, 1][None, :, None, None] * v[None, None, :, None]
    b[:, 3] = -b[:, 2]
    if not bool(torch.isfinite(b).all().item()) or float(b.square().mean().item()) <= 0.0:
        raise NoiseInstrumentationError("noise basis is non-finite or degenerate")
    return b

def conditions(torch, base, trajectories_by_name, ratio):
    if not bool(torch.isfinite(base).all().item()) or float(base.square().mean().item()) <= 0.0:
        raise NoiseInstrumentationError("initial latent is non-finite or degenerate")
    out = {"OFF_R1": base.detach().clone(), "OFF_R2": base.detach().clone()}
    target_rms = float(ratio) * base.float().square().mean().sqrt()
    for name in ("A", "B"):
        basis = latent_basis(torch, base, trajectories_by_name[name])
        update = basis * (target_rms / basis.float().square().mean().sqrt()).to(basis.dtype)
        actual = update.float().square().mean().sqrt() / base.float().square().mean().sqrt()
        if abs(float(actual.item()) - float(ratio)) > 1e-6:
            raise NoiseInstrumentationError("initial-noise update RMS changed")
        out[name] = (base + update).detach()
    if not torch.equal(out["OFF_R1"], out["OFF_R2"]):
        raise NoiseInstrumentationError("OFF repeats did not start identically")
    return out

def inverted_observer(torch, vae, frames, device):
    """Blindly map one saved MP4 to the fixed 13x2 inversion observation."""

    if tuple(frames.shape) != (49, 320, 512, 3):
        raise NoiseInstrumentationError("saved MP4 frame identity changed")
    video = (
        torch.from_numpy(frames)
        .permute(3, 0, 1, 2)[None]
        .to(device=device, dtype=vae.dtype)
        / 127.5
        - 1.0
    )
    with torch.inference_mode():
        posterior = vae.encode(video, return_dict=False)[0]
        z = posterior.mode()
    mean = torch.tensor(vae.config.latents_mean, device=z.device, dtype=z.dtype).view(
        1, 16, 1, 1, 1
    )
    std = torch.tensor(vae.config.latents_std, device=z.device, dtype=z.dtype).view(
        1, 16, 1, 1, 1
    )
    z = ((z - mean) / std).float()
    if tuple(z.shape) != (1, 16, 13, 40, 64) or not bool(torch.isfinite(z).all().item()):
        raise NoiseInstrumentationError("Wan VAE inversion identity changed")
    h = torch.cos(2 * torch.pi * (torch.arange(64, device=z.device) + 0.5) / 64)
    h = (h - h.mean()) / h.square().mean().sqrt()
    v = torch.cos(2 * torch.pi * (torch.arange(40, device=z.device) + 0.5) / 40)
    v = (v - v.mean()) / v.square().mean().sqrt()
    q1 = ((z[:, 0] - z[:, 1]) * h[None, None, None, :]).mean((2, 3))
    q2 = ((z[:, 2] - z[:, 3]) * v[None, None, :, None]).mean((2, 3))
    observation = torch.stack((q1, q2), -1)[0]
    if tuple(observation.shape) != (13, 2) or not bool(torch.isfinite(observation).all().item()):
        raise NoiseInstrumentationError("inversion observation is invalid")
    return observation.cpu().tolist()

def evaluate(obs,t,quality,criteria):
    import numpy as np
    q={k:np.asarray(v,float) for k,v in obs.items()}; off=(q["OFF_R1"]+q["OFF_R2"])/2; offn=float(np.sqrt(np.mean((q["OFF_R1"]-q["OFF_R2"])**2))); floor=max(offn,criteria["numeric_floor_float32_eps_multiplier"]*2**-23*max(1,float(np.sqrt(np.mean(off**2)))))
    design=np.concatenate([t["A"],t["B"]]); response=np.concatenate([q["A"]-off,q["B"]-off]); mt,*_=np.linalg.lstsq(design,response,rcond=None); m=mt.T; s=np.linalg.svd(m,compute_uv=False)
    condition_value=None if s[-1]<=0 or not np.isfinite(s).all() else float(s[0]/s[-1])
    response_rms=float(np.sqrt(np.mean(response**2)))
    residual_value=None if response_rms<=0 or not math.isfinite(response_rms) else float(np.sqrt(np.mean((response-design@mt)**2))/response_rms)
    effects={n:float(np.sqrt(np.mean((q[n]-off)**2))) for n in ("A","B")}
    checks={"full_rank":bool(np.isfinite(m).all() and s[-1]>0),"condition":condition_value is not None and condition_value<=criteria["maximum_channel_condition_number"],"fit":residual_value is not None and residual_value<=criteria["maximum_fit_relative_residual"],"A_effect":effects["A"]>=criteria["minimum_effect_to_floor_ratio"]*floor,"B_effect":effects["B"]>=criteria["minimum_effect_to_floor_ratio"]*floor,"A_quality":quality["A"]<=criteria["maximum_saved_mp4_relative_rms"],"B_quality":quality["B"]<=criteria["maximum_saved_mp4_relative_rms"]}
    return {"passed":all(checks.values()),"checks":checks,"matrix":m.tolist(),"condition":condition_value,"fit_relative_residual":residual_value,"effect_rms":effects,"floor":floor}

def run(repo:Path,output:Path):
    if output.exists(): raise NoiseInstrumentationError("output exists")
    c=load_noise_config(repo/"configs/g0_structured_initial_noise.json")
    try:
        import diffusers,imageio.v3 as iio,numpy as np,torch
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
    except Exception as e: raise NoiseInstrumentationError("dependencies unavailable") from e
    g=c["generation"]; pipe=WanPipeline.from_pretrained(c["model"]["repository"],revision=c["model"]["revision"],torch_dtype=torch.bfloat16);pipe.enable_model_cpu_offload();device=pipe._execution_device;t=trajectories(c)
    output.mkdir();(output/"videos").mkdir();calls=decodes=encodes=0
    try:
      with torch.inference_mode():
        cond,uncond=pipe.encode_prompt(prompt=g["prompt"],negative_prompt=g["negative_prompt"],do_classifier_free_guidance=True,num_videos_per_prompt=1,max_sequence_length=g["max_sequence_length"],device=device);cond=cond.to(pipe.transformer.dtype);uncond=uncond.to(pipe.transformer.dtype);base=pipe.prepare_latents(1,int(pipe.transformer.config.in_channels),g["height"],g["width"],g["frames"],torch.float32,device,torch.Generator(device=device).manual_seed(g["seed"]),None).detach()
      starts=conditions(torch,base,t,c["carrier"]["relative_initial_noise_rms"]); finals={}
      for name in CONDITIONS:
        _emit_progress(f"noise_generate_{name}",torch); pipe.scheduler=pipe.scheduler.__class__.from_config(pipe.scheduler.config); pipe.scheduler.set_timesteps(g["inference_steps"],device=device); pipe.scheduler.set_begin_index(0) if hasattr(pipe.scheduler,"set_begin_index") else None; z=starts[name].detach().clone()
        for ts in pipe.scheduler.timesteps:
          cv=_transformer_velocity(pipe,z,ts,cond,"cond",torch);uv=_transformer_velocity(pipe,z,ts,uncond,"uncond",torch);calls+=2
          with torch.inference_mode(): z=pipe.scheduler.step(uv+g["guidance_scale"]*(cv-uv),ts,z,return_dict=False)[0].detach()
        finals[name]=z.cpu(); del z
      starts=base=cond=uncond=None;pipe.maybe_free_model_hooks();gc.collect();torch.cuda.empty_cache()
      paths={}
      for name in CONDITIONS:
        _emit_progress(f"noise_decode_encode_{name}",torch);z=finals[name].to(device);rgb=decode_wan_latents_torch(z,pipe.vae,torch);decodes+=1;array=((rgb[0].permute(1,2,3,0).float().clamp(-1,1)+1)*127.5).round().byte().cpu().numpy();p=output/"videos"/f"noise_{name}.mp4";export_to_video(list(array),str(p),fps=g["fps"],quality=5.0,bitrate=None,macro_block_size=16);paths[name]=p;del z,rgb,array;pipe.maybe_free_model_hooks();gc.collect();torch.cuda.empty_cache()
      saved={n:np.stack(list(iio.imiter(p,plugin="FFMPEG"))) for n,p in paths.items()};obs={}
      for n in CONDITIONS: obs[n]=inverted_observer(torch,pipe.vae,saved[n],device);encodes+=1;pipe.maybe_free_model_hooks();gc.collect();torch.cuda.empty_cache()
      values={n:saved[n].astype(np.float32)/127.5-1 for n in CONDITIONS};quality=_saved_rgb_quality(values,np);ev=evaluate(obs,t,quality,c["criteria"]);status="STRUCTURED_NOISE_PRIMITIVE_FEASIBLE" if ev["passed"] else "STRUCTURED_NOISE_PRIMITIVE_NOT_FEASIBLE";result={"schema_version":1,"protocol_id":c["protocol_id"],"diagnostic_class":"DIAGNOSTIC_ONLY","status":status,"runtime":runtime_capability_diagnostics(torch)|{"diffusers":diffusers.__version__},"observations":obs,"quality":quality,"evaluation":ev,"execution":{"transformer_calls":calls,"vae_decodes":decodes,"vae_encodes":encodes,"mp4_count":4}}
      if (calls,decodes,encodes)!=(64,4,4): raise NoiseInstrumentationError("execution budget changed")
      (output/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n");return result
    finally:
      try:pipe.maybe_free_model_hooks()
      except Exception:pass
