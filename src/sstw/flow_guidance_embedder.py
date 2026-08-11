"""Minimal SSTW-v2 Flow-guidance primitive.

The functions in this module only implement the frozen observer projection and
the bounded latent update.  They do not load a model, choose a video, fit an
observer, or inspect a result in order to tune the construction.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence


OBSERVER_FRAME_INDICES = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
G0_CONDITION_ORDER = (
    "OFF_R1",
    "OFF_R2",
    "PLUS_G1",
    "MINUS_G1",
    "PLUS_G2",
    "MINUS_G2",
)
G0_ACTIVE_CONDITIONS = G0_CONDITION_ORDER[2:]
G0_ALLOWED_STATUSES = (
    "FLOW_GUIDANCE_PRIMITIVE_FEASIBLE",
    "FLOW_GUIDANCE_PRIMITIVE_NOT_FEASIBLE",
    "INSTRUMENTATION_INSUFFICIENT",
)


class G0InstrumentationError(RuntimeError):
    """A runtime or interface failure that cannot answer the G0 question."""


@dataclass(frozen=True, slots=True)
class FrozenGuidanceConfig:
    """Frozen G0 construction and decision thresholds."""

    active_after_scheduler_index: int = 5
    relative_update_rms: float = 0.005
    maximum_gradient_cosine: float = 0.5
    maximum_cross_ratio: float = 0.5
    maximum_even_ratio: float = 0.5
    maximum_rgb_relative_rms: float = 0.02

    def validate(self) -> None:
        if self.active_after_scheduler_index != 5:
            raise ValueError("G0 guidance boundary drift")
        for name, value in (
            ("relative_update_rms", self.relative_update_rms),
            ("maximum_gradient_cosine", self.maximum_gradient_cosine),
            ("maximum_cross_ratio", self.maximum_cross_ratio),
            ("maximum_even_ratio", self.maximum_even_ratio),
            ("maximum_rgb_relative_rms", self.maximum_rgb_relative_rms),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def scalar_rms(values: Iterable[float]) -> float:
    data = tuple(float(value) for value in values)
    if not data or any(not math.isfinite(value) for value in data):
        raise ValueError("RMS input must be non-empty and finite")
    return math.sqrt(sum(value * value for value in data) / len(data))


def scalar_cosine(left: Iterable[float], right: Iterable[float]) -> float:
    x = tuple(float(value) for value in left)
    y = tuple(float(value) for value in right)
    if not x or len(x) != len(y):
        raise ValueError("cosine inputs must have equal non-zero length")
    denominator = scalar_rms(x) * scalar_rms(y) * len(x)
    if denominator == 0.0:
        raise ValueError("cosine is undefined for a zero axis")
    return sum(a * b for a, b in zip(x, y, strict=True)) / denominator


def _normalized_cosine(length: int, torch_module, *, device, dtype):
    coordinate = torch_module.arange(length, device=device, dtype=dtype) + 0.5
    basis = torch_module.cos((2.0 * math.pi / float(length)) * coordinate)
    basis = basis - basis.mean()
    rms = torch_module.sqrt(torch_module.mean(basis.square()))
    if not bool(torch_module.isfinite(rms).item()) or float(rms.item()) <= 0.0:
        raise ValueError("observer cosine basis is degenerate")
    return basis / rms


def fixed_observer_torch(decoded_rgb, torch_module):
    """Return the frozen T=13, R=2 observer from decoded RGB.

    ``decoded_rgb`` must be ``[B,3,49,320,512]``.  The calculation is forced
    to float32 but remains differentiable with respect to the input.
    """

    if tuple(decoded_rgb.shape[1:]) != (3, 49, 320, 512):
        raise ValueError("decoded RGB must have shape [B,3,49,320,512]")
    if decoded_rgb.ndim != 5 or int(decoded_rgb.shape[0]) < 1:
        raise ValueError("decoded RGB must have a non-empty batch")
    rgb = decoded_rgb.float().index_select(
        2,
        torch_module.tensor(
            OBSERVER_FRAME_INDICES,
            device=decoded_rgb.device,
            dtype=torch_module.long,
        ),
    )
    horizontal = _normalized_cosine(
        512, torch_module, device=rgb.device, dtype=rgb.dtype
    ).view(1, 1, 1, 512)
    vertical = _normalized_cosine(
        320, torch_module, device=rgb.device, dtype=rgb.dtype
    ).view(1, 1, 320, 1)
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    q1 = ((red - green) * horizontal).mean(dim=(-2, -1))
    q2 = ((blue - 0.5 * (red + green)) * vertical).mean(dim=(-2, -1))
    observation = torch_module.stack((q1, q2), dim=-1)
    if tuple(observation.shape[1:]) != (13, 2):
        raise ValueError("observer output shape drift")
    if not bool(torch_module.isfinite(observation).all().item()):
        raise ValueError("observer produced non-finite values")
    return observation


def observer_projection_loss_torch(decoded_rgb, target_states, torch_module):
    """Return ``-mean(q dot u)`` for the frozen differentiable observer."""

    observation = fixed_observer_torch(decoded_rgb, torch_module)
    target = torch_module.as_tensor(
        target_states,
        device=observation.device,
        dtype=observation.dtype,
    )
    if tuple(target.shape) != (13, 2):
        raise ValueError("target states must have shape [13,2]")
    if not bool(torch_module.isfinite(target).all().item()):
        raise ValueError("target states must be finite")
    return -(observation[0] * target).sum(dim=-1).mean()


def normalized_guidance_update_torch(
    latent,
    gradient,
    torch_module,
    *,
    relative_update_rms: float = 0.005,
):
    """Construct the frozen relative-RMS descent update without applying it."""

    if tuple(latent.shape) != tuple(gradient.shape):
        raise ValueError("latent and gradient shapes differ")
    if not math.isfinite(relative_update_rms) or relative_update_rms <= 0.0:
        raise ValueError("relative_update_rms must be finite and positive")
    latent32 = latent.float()
    gradient32 = gradient.float()
    if not bool(torch_module.isfinite(latent32).all().item()):
        raise ValueError("latent is non-finite")
    if not bool(torch_module.isfinite(gradient32).all().item()):
        raise ValueError("observer gradient is non-finite")
    latent_rms = torch_module.sqrt(torch_module.mean(latent32.square()))
    gradient_rms = torch_module.sqrt(torch_module.mean(gradient32.square()))
    if float(latent_rms.item()) <= 0.0 or float(gradient_rms.item()) <= 0.0:
        raise ValueError("latent or observer gradient RMS is zero")
    update32 = -relative_update_rms * latent_rms * gradient32 / gradient_rms
    actual_ratio = torch_module.sqrt(torch_module.mean(update32.square())) / latent_rms
    if not bool(torch_module.isfinite(update32).all().item()):
        raise ValueError("guidance update is non-finite")
    return update32.to(dtype=latent.dtype), {
        "latent_rms": float(latent_rms.item()),
        "gradient_rms": float(gradient_rms.item()),
        "actual_relative_update_rms": float(actual_ratio.item()),
    }


def load_g0_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen method-diagnostic construction."""

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise G0InstrumentationError("G0 config is unavailable or invalid JSON") from exc
    expected_generation = {
        "prompt": "locked camera, dark matte background, a single bright white cube moving slowly across the center, simple studio lighting, no text, no cuts",
        "negative_prompt": "text, watermark, logo, camera motion, cuts, multiple subjects, flicker",
        "seed": 1275,
        "guidance_scale": 5.0,
        "max_sequence_length": 226,
        "inference_steps": 8,
        "frames": 49,
        "height": 320,
        "width": 512,
    }
    generation = config.get("generation", {})
    guidance = config.get("guidance", {})
    criteria = config.get("criteria", {})
    if (
        config.get("protocol_id") != "sstw_v2_g0_flow_guidance_primitive"
        or config.get("diagnostic_class") != "DIAGNOSTIC_ONLY"
        or config.get("model", {}).get("repository")
        != "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        or config.get("model", {}).get("revision")
        != "0fad780a534b6463e45facd96134c9f345acfa5b"
        or any(generation.get(name) != value for name, value in expected_generation.items())
        or guidance.get("active_after_scheduler_index") != 5
        or guidance.get("normal_steps_remaining") != [6, 7]
        or guidance.get("relative_update_rms") != 0.005
        or guidance.get("conditions") != list(G0_CONDITION_ORDER)
        or criteria
        != {
            "minimum_odd_to_floor_ratio": 8.0,
            "numeric_floor_float32_eps_multiplier": 32.0,
            "maximum_absolute_gradient_cosine": 0.5,
            "maximum_cross_to_own_ratio": 0.5,
            "maximum_even_to_odd_ratio": 0.5,
            "maximum_final_rgb_relative_rms": 0.02,
        }
        or config.get("allowed_statuses") != list(G0_ALLOWED_STATUSES)
    ):
        raise G0InstrumentationError("G0 frozen construction changed")
    observer = config.get("observer", {})
    if (
        observer.get("version") != "opponent_chroma_cosine_v2_0"
        or observer.get("frame_indices") != list(OBSERVER_FRAME_INDICES)
    ):
        raise G0InstrumentationError("G0 observer identity changed")
    return config


def runtime_capability_diagnostics(torch_module: Any) -> dict[str, Any]:
    """Gate only the CUDA and BF16 capabilities used by the method diagnostic."""

    cuda_available = bool(torch_module.cuda.is_available())
    bf16_supported = bool(cuda_available and torch_module.cuda.is_bf16_supported())
    result = {
        "cuda_available": cuda_available,
        "bf16_supported": bf16_supported,
        "torch": str(torch_module.__version__),
        "cuda": None if torch_module.version.cuda is None else str(torch_module.version.cuda),
        "gpu": None if not cuda_available else str(torch_module.cuda.get_device_name(0)),
        "python": platform.python_version(),
    }
    if not cuda_available or not bf16_supported:
        raise G0InstrumentationError("G0 requires CUDA and BF16 capabilities")
    return result


def _tensor_rms_float(value: Any, torch_module: Any) -> float:
    value32 = value.detach().float()
    if value32.numel() == 0 or not bool(torch_module.isfinite(value32).all().item()):
        raise G0InstrumentationError("tensor RMS input is empty or non-finite")
    return float(value32.square().mean().sqrt().item())


def _tensor_cosine_float(left: Any, right: Any, torch_module: Any) -> float:
    left32 = left.detach().float()
    right32 = right.detach().float()
    if tuple(left32.shape) != tuple(right32.shape) or left32.numel() == 0:
        raise G0InstrumentationError("gradient cosine inputs differ")
    denominator = _tensor_rms_float(left32, torch_module) * _tensor_rms_float(
        right32, torch_module
    )
    if denominator <= 0.0:
        raise G0InstrumentationError("gradient cosine has a zero axis")
    cosine = float((left32 * right32).mean().item()) / denominator
    if not math.isfinite(cosine):
        raise G0InstrumentationError("gradient cosine is non-finite")
    return cosine


def decode_wan_latents_torch(latents: Any, vae: Any, torch_module: Any) -> Any:
    """Use the official Wan latent denormalization followed by VAE decode.

    Diffusers WanPipeline v0.35.2 computes ``latents / (1/std) + mean`` before
    ``vae.decode``.  Keeping this in one function makes the generation-time
    gradient and the terminal RGB measurement use exactly the same path.
    """

    required = ("latents_mean", "latents_std", "z_dim")
    if any(not hasattr(vae.config, name) for name in required):
        raise G0InstrumentationError("Wan VAE latent normalization metadata is missing")
    latent_value = latents.to(dtype=vae.dtype)
    mean = torch_module.tensor(
        vae.config.latents_mean,
        device=latent_value.device,
        dtype=latent_value.dtype,
    ).view(1, int(vae.config.z_dim), 1, 1, 1)
    inverse_std = 1.0 / torch_module.tensor(
        vae.config.latents_std,
        device=latent_value.device,
        dtype=latent_value.dtype,
    ).view(1, int(vae.config.z_dim), 1, 1, 1)
    denormalized = latent_value / inverse_std + mean
    try:
        decoded = vae.decode(denormalized, return_dict=False)[0]
    except Exception as exc:
        raise G0InstrumentationError("Wan VAE decode failed") from exc
    if tuple(decoded.shape[1:]) != (3, 49, 320, 512):
        raise G0InstrumentationError("Wan VAE RGB shape is not [B,3,49,320,512]")
    if not bool(torch_module.isfinite(decoded).all().item()):
        raise G0InstrumentationError("Wan VAE RGB is non-finite")
    return decoded


def common_axis_gradients_torch(
    base_latent: Any,
    vae: Any,
    torch_module: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute both active axes once at the same post-index-5 Flow state."""

    parameters = tuple(vae.parameters())
    for parameter in parameters:
        parameter.requires_grad_(False)
        parameter.grad = None
    targets = {
        "G1": tuple((1.0, 0.0) for _ in OBSERVER_FRAME_INDICES),
        "G2": tuple((0.0, 1.0) for _ in OBSERVER_FRAME_INDICES),
    }
    gradients: dict[str, Any] = {}
    losses: dict[str, float] = {}
    # Deliberately execute two independent VAE graphs.  Saving the full Wan VAE
    # graph to pinned host RAM and retaining it for a second VJP can exhaust the
    # Colab host and kill the whole runtime instead of raising a CUDA OOM.
    for axis in ("G1", "G2"):
        differentiable = base_latent.detach().clone().requires_grad_(True)
        decoded = None
        loss = None
        gradient = None
        _emit_progress(f"vae_vjp_{axis}_start", torch_module)
        try:
            save_on_cpu = getattr(torch_module.autograd.graph, "save_on_cpu", None)
            saved_tensor_context = (
                save_on_cpu(pin_memory=False)
                if save_on_cpu is not None
                else nullcontext()
            )
            # One axis at a time, with saved activations in ordinary host RAM.
            # pin_memory=False avoids the locked-host-memory failure mode seen
            # in the superseded two-axis retained-graph implementation.
            with torch_module.enable_grad(), saved_tensor_context:
                decoded = decode_wan_latents_torch(differentiable, vae, torch_module)
                if not bool(decoded.requires_grad) or decoded.grad_fn is None:
                    raise G0InstrumentationError("Wan VAE decode is outside autograd")
                loss = observer_projection_loss_torch(
                    decoded, targets[axis], torch_module
                )
                gradient = torch_module.autograd.grad(
                    loss,
                    differentiable,
                    retain_graph=False,
                    create_graph=False,
                )[0]
            gradients[axis] = gradient.detach()
            losses[axis] = float(loss.detach().float().item())
        except G0InstrumentationError:
            raise
        except Exception as exc:
            raise G0InstrumentationError(
                f"fixed observer {axis} gradient is disconnected"
            ) from exc
        finally:
            # Ensure the axis-local graph is dead before constructing the next.
            decoded = None
            loss = None
            gradient = None
            differentiable = None
            gc.collect()
            if bool(torch_module.cuda.is_available()):
                torch_module.cuda.empty_cache()
        _emit_progress(f"vae_vjp_{axis}_complete", torch_module)
    if any(parameter.grad is not None for parameter in parameters):
        raise G0InstrumentationError("G0 accumulated a VAE parameter gradient")
    rms = {
        axis: _tensor_rms_float(gradient, torch_module)
        for axis, gradient in gradients.items()
    }
    if any(value <= 0.0 for value in rms.values()):
        raise G0InstrumentationError("fixed observer gradient RMS is zero")
    cosine = _tensor_cosine_float(gradients["G1"], gradients["G2"], torch_module)
    return gradients, {
        "loss": losses,
        "rms": rms,
        "cosine": cosine,
        "absolute_cosine": abs(cosine),
        "common_base_latent_rms": _tensor_rms_float(base_latent, torch_module),
    }


def condition_latents_from_common_gradients_torch(
    base_latent: Any,
    gradients: Mapping[str, Any],
    torch_module: Any,
    *,
    relative_update_rms: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create OFF and signed active forks without recomputing either gradient."""

    update_g1, diagnostics_g1 = normalized_guidance_update_torch(
        base_latent,
        gradients["G1"],
        torch_module,
        relative_update_rms=relative_update_rms,
    )
    update_g2, diagnostics_g2 = normalized_guidance_update_torch(
        base_latent,
        gradients["G2"],
        torch_module,
        relative_update_rms=relative_update_rms,
    )
    condition_latents = {
        "OFF_R1": base_latent.detach().clone(),
        "OFF_R2": base_latent.detach().clone(),
        "PLUS_G1": (base_latent + update_g1).detach(),
        "MINUS_G1": (base_latent - update_g1).detach(),
        "PLUS_G2": (base_latent + update_g2).detach(),
        "MINUS_G2": (base_latent - update_g2).detach(),
    }
    return condition_latents, {"G1": diagnostics_g1, "G2": diagnostics_g2}


def _scheduler_state_value(value: Any, torch_module: Any) -> Any:
    if torch_module.is_tensor(value):
        contiguous = value.detach().contiguous().to(device="cpu")
        # A 0-D Long tensor cannot directly change element size through view().
        # Flatten first so scalars and non-scalars share the same exact-byte path.
        raw = contiguous.reshape(-1).view(torch_module.uint8).numpy().tobytes()
        return {
            "kind": "tensor",
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _scheduler_state_value(item, torch_module)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_scheduler_state_value(item, torch_module) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _emit_progress(stage: str, torch_module: Any | None = None) -> None:
    report: dict[str, Any] = {
        "event": "G0_PROGRESS",
        "stage": stage,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if torch_module is not None and bool(torch_module.cuda.is_available()):
        report["cuda_allocated_gib"] = round(
            float(torch_module.cuda.memory_allocated()) / (2**30), 3
        )
        report["cuda_reserved_gib"] = round(
            float(torch_module.cuda.memory_reserved()) / (2**30), 3
        )
    print(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False),
        file=sys.stderr,
        flush=True,
    )


def scheduler_state_summary(scheduler: Any, torch_module: Any) -> dict[str, Any]:
    """Fingerprint the complete UniPC instance, including multistep history."""

    missing = [
        name
        for name in ("_step_index", "lower_order_nums", "model_outputs")
        if not hasattr(scheduler, name)
    ]
    if missing:
        raise G0InstrumentationError(f"UniPC scheduler history is unavailable: {missing}")
    state = {
        str(name): _scheduler_state_value(value, torch_module)
        for name, value in sorted(vars(scheduler).items())
    }
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "state_sha256": hashlib.sha256(encoded).hexdigest(),
        "state_keys": sorted(state),
        "step_index": int(scheduler._step_index),
    }


def capture_unipc_snapshot(
    scheduler: Any,
    torch_module: Any,
    *,
    expected_step_index: int,
) -> tuple[Any, dict[str, Any]]:
    if type(scheduler).__name__ != "UniPCMultistepScheduler":
        raise G0InstrumentationError("G0 requires the real UniPCMultistepScheduler")
    if getattr(scheduler, "_step_index", None) != expected_step_index:
        raise G0InstrumentationError("guidance boundary scheduler index is incorrect")
    expected = scheduler_state_summary(scheduler, torch_module)
    try:
        snapshot = copy.deepcopy(scheduler)
    except Exception as exc:
        raise G0InstrumentationError("complete UniPC snapshot failed") from exc
    if scheduler_state_summary(snapshot, torch_module) != expected:
        raise G0InstrumentationError("complete UniPC snapshot lost state")
    return snapshot, expected


def fork_unipc_snapshot(
    snapshot: Any,
    expected: Mapping[str, Any],
    torch_module: Any,
) -> Any:
    if scheduler_state_summary(snapshot, torch_module) != expected:
        raise G0InstrumentationError("retained UniPC snapshot was mutated")
    try:
        fork = copy.deepcopy(snapshot)
    except Exception as exc:
        raise G0InstrumentationError("condition-local UniPC fork failed") from exc
    if scheduler_state_summary(fork, torch_module) != expected:
        raise G0InstrumentationError("condition-local UniPC fork lost state")
    return fork


def _finite_vectors(
    value: Sequence[Sequence[float]], *, expected_length: int = 13
) -> tuple[tuple[float, float], ...]:
    converted = tuple(tuple(float(coordinate) for coordinate in point) for point in value)
    if (
        len(converted) != expected_length
        or any(len(point) != 2 for point in converted)
        or any(not math.isfinite(coordinate) for point in converted for coordinate in point)
    ):
        raise G0InstrumentationError("G0 observation must be finite [13,2]")
    return converted  # type: ignore[return-value]


def _flat_rms(values: Iterable[float]) -> float:
    data = tuple(float(value) for value in values)
    if not data:
        raise G0InstrumentationError("metric input is empty")
    result = math.sqrt(sum(value * value for value in data) / len(data))
    if not math.isfinite(result):
        raise G0InstrumentationError("metric is non-finite")
    return result


def evaluate_g0_metrics(
    observations: Mapping[str, Sequence[Sequence[float]]],
    *,
    gradient_rms: Mapping[str, float],
    gradient_cosine: float,
    rgb_relative_rms: Mapping[str, Mapping[str, float]],
    off_rgb_relative_rms: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every frozen G0 criterion without averaging the OFF repeats."""

    q = {name: _finite_vectors(observations[name]) for name in G0_CONDITION_ORDER}
    criteria = config["criteria"]
    off_difference_rms = _flat_rms(
        q["OFF_R1"][index][coordinate] - q["OFF_R2"][index][coordinate]
        for index in range(13)
        for coordinate in range(2)
    )
    off_baseline_rms = _flat_rms(
        (q["OFF_R1"][index][coordinate] + q["OFF_R2"][index][coordinate]) / 2.0
        for index in range(13)
        for coordinate in range(2)
    )
    numeric_floor = (
        float(criteria["numeric_floor_float32_eps_multiplier"])
        * (2.0 ** -23)
        * max(1.0, off_baseline_rms)
    )
    effect_floor = max(off_difference_rms, numeric_floor)
    axes: dict[str, Any] = {}
    checks: dict[str, bool] = {
        "gradient_g1_finite_nonzero": math.isfinite(float(gradient_rms["G1"]))
        and float(gradient_rms["G1"]) > 0.0,
        "gradient_g2_finite_nonzero": math.isfinite(float(gradient_rms["G2"]))
        and float(gradient_rms["G2"]) > 0.0,
        "gradient_absolute_cosine": math.isfinite(gradient_cosine)
        and abs(gradient_cosine) < float(criteria["maximum_absolute_gradient_cosine"]),
    }
    for axis_name, plus_name, minus_name, own_index in (
        ("G1", "PLUS_G1", "MINUS_G1", 0),
        ("G2", "PLUS_G2", "MINUS_G2", 1),
    ):
        odd = tuple(
            tuple((q[plus_name][index][coordinate] - q[minus_name][index][coordinate]) / 2.0 for coordinate in range(2))
            for index in range(13)
        )
        own = tuple(point[own_index] for point in odd)
        cross = tuple(point[1 - own_index] for point in odd)
        own_mean = sum(own) / len(own)
        own_rms = _flat_rms(own)
        cross_rms = _flat_rms(cross)
        cross_ratio = None if own_rms == 0.0 else cross_rms / own_rms
        odd_total_rms = _flat_rms(value for point in odd for value in point)
        even_by_off: dict[str, Any] = {}
        for off_name in ("OFF_R1", "OFF_R2"):
            even = tuple(
                tuple(
                    (q[plus_name][index][coordinate] + q[minus_name][index][coordinate]) / 2.0
                    - q[off_name][index][coordinate]
                    for coordinate in range(2)
                )
                for index in range(13)
            )
            even_rms = _flat_rms(value for point in even for value in point)
            even_by_off[off_name] = {
                "rms": even_rms,
                "ratio_to_odd": None if odd_total_rms == 0.0 else even_rms / odd_total_rms,
            }
        finite_even_ratios = [
            value["ratio_to_odd"]
            for value in even_by_off.values()
            if value["ratio_to_odd"] is not None
        ]
        worst_even_ratio = max(finite_even_ratios) if len(finite_even_ratios) == 2 else None
        axes[axis_name] = {
            "own_mean": own_mean,
            "own_rms": own_rms,
            "cross_rms": cross_rms,
            "cross_to_own_ratio": cross_ratio,
            "odd_total_rms": odd_total_rms,
            "even_by_off": even_by_off,
            "worst_even_to_odd_ratio": worst_even_ratio,
            "effect_floor": effect_floor,
            "own_odd_to_floor_ratio": own_rms / effect_floor,
        }
        checks[f"{axis_name.lower()}_target_direction_positive"] = own_mean > 0.0
        checks[f"{axis_name.lower()}_odd_above_floor"] = own_rms >= float(
            criteria["minimum_odd_to_floor_ratio"]
        ) * effect_floor
        checks[f"{axis_name.lower()}_cross_to_own"] = cross_ratio is not None and cross_ratio <= float(
            criteria["maximum_cross_to_own_ratio"]
        )
        checks[f"{axis_name.lower()}_even_to_odd"] = worst_even_ratio is not None and worst_even_ratio <= float(
            criteria["maximum_even_to_odd_ratio"]
        )
    quality: dict[str, Any] = {}
    for condition in G0_ACTIVE_CONDITIONS:
        values = {
            off: float(rgb_relative_rms[condition][off]) for off in ("OFF_R1", "OFF_R2")
        }
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise G0InstrumentationError("final RGB relative RMS is invalid")
        worst = max(values.values())
        quality[condition] = {"by_off": values, "worst": worst}
        checks[f"{condition.lower()}_rgb_quality"] = worst <= float(
            criteria["maximum_final_rgb_relative_rms"]
        )
    if not math.isfinite(off_rgb_relative_rms) or off_rgb_relative_rms < 0.0:
        raise G0InstrumentationError("OFF RGB floor is invalid")
    status = (
        "FLOW_GUIDANCE_PRIMITIVE_FEASIBLE"
        if all(checks.values())
        else "FLOW_GUIDANCE_PRIMITIVE_NOT_FEASIBLE"
    )
    return {
        "status": status,
        "checks": checks,
        "axes": axes,
        "quality": quality,
        "off_floor": {"rgb_relative_rms": off_rgb_relative_rms},
        "observation_floor": {
            "off_repeat_rms": off_difference_rms,
            "off_baseline_rms": off_baseline_rms,
            "numeric_floor": numeric_floor,
            "effective_floor": effect_floor,
        },
    }


def _transformer_velocity(
    pipe: Any,
    latent: Any,
    timestep: Any,
    embeddings: Any,
    branch: str,
    torch_module: Any,
) -> Any:
    with torch_module.inference_mode():
        with pipe.transformer.cache_context(branch):
            return pipe.transformer(
                hidden_states=latent.to(dtype=pipe.transformer.dtype),
                timestep=timestep.expand(latent.shape[0]),
                encoder_hidden_states=embeddings,
                attention_kwargs=None,
                return_dict=False,
            )[0].detach()


def _continue_condition(
    pipe: Any,
    latent: Any,
    scheduler_snapshot: Any,
    scheduler_summary: Mapping[str, Any],
    timesteps: Any,
    prompt_embeddings: Any,
    negative_embeddings: Any,
    config: Mapping[str, Any],
    torch_module: Any,
) -> tuple[Any, int]:
    scheduler = fork_unipc_snapshot(scheduler_snapshot, scheduler_summary, torch_module)
    result = latent.detach().clone()
    calls = 0
    for index in config["guidance"]["normal_steps_remaining"]:
        conditional = _transformer_velocity(
            pipe, result, timesteps[index], prompt_embeddings, "cond", torch_module
        )
        unconditional = _transformer_velocity(
            pipe, result, timesteps[index], negative_embeddings, "uncond", torch_module
        )
        calls += 2
        with torch_module.inference_mode():
            guided = unconditional + float(config["generation"]["guidance_scale"]) * (
                conditional - unconditional
            )
            result = scheduler.step(
                guided, timesteps[index], result, return_dict=False
            )[0].detach()
        del conditional, unconditional, guided
    if scheduler_state_summary(scheduler_snapshot, torch_module) != scheduler_summary:
        raise G0InstrumentationError("retained UniPC snapshot changed during continuation")
    return result, calls


def _relative_rgb_rms(left: Any, right: Any, torch_module: Any) -> float:
    denominator = _tensor_rms_float(right, torch_module)
    if denominator <= 0.0:
        raise G0InstrumentationError("OFF RGB RMS is zero")
    return _tensor_rms_float(left - right, torch_module) / denominator


def run_g0_once(
    *,
    repo_root: Path,
    output: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run the single frozen real-Wan G0 diagnostic and write JSON only."""

    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise G0InstrumentationError("output must be absent below an existing directory")
    config_file = config_path or repo_root / "configs/g0_flow_guidance_primitive.json"
    config = load_g0_config(config_file)
    try:
        import diffusers
        import torch
        from diffusers import WanPipeline
    except Exception as exc:
        raise G0InstrumentationError("G0 runtime dependencies are unavailable") from exc
    runtime = runtime_capability_diagnostics(torch)
    runtime["diffusers"] = str(diffusers.__version__)
    generation = config["generation"]
    try:
        pipe = WanPipeline.from_pretrained(
            config["model"]["repository"],
            revision=config["model"]["revision"],
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
    except Exception as exc:
        raise G0InstrumentationError("real Wan pipeline could not be loaded") from exc
    device = pipe._execution_device
    transformer_calls = 0
    vae_decodes = 0
    final_latents: dict[str, Any] = {}
    stage = "runtime_ready"
    try:
        stage = "encode_prompt_and_prepare_latent"
        _emit_progress(stage, torch)
        with torch.inference_mode():
            prompt_embeddings, negative_embeddings = pipe.encode_prompt(
                prompt=generation["prompt"],
                negative_prompt=generation["negative_prompt"],
                do_classifier_free_guidance=True,
                num_videos_per_prompt=1,
                max_sequence_length=int(generation["max_sequence_length"]),
                device=device,
            )
            prompt_embeddings = prompt_embeddings.to(pipe.transformer.dtype).detach()
            negative_embeddings = negative_embeddings.to(pipe.transformer.dtype).detach()
            pipe.scheduler.set_timesteps(int(generation["inference_steps"]), device=device)
            if hasattr(pipe.scheduler, "set_begin_index"):
                pipe.scheduler.set_begin_index(0)
            timesteps = pipe.scheduler.timesteps
            if len(timesteps) != 8:
                raise G0InstrumentationError("Wan scheduler did not expose exactly eight steps")
            generator = torch.Generator(device="cuda").manual_seed(int(generation["seed"]))
            latent = pipe.prepare_latents(
                1,
                int(pipe.transformer.config.in_channels),
                int(generation["height"]),
                int(generation["width"]),
                int(generation["frames"]),
                torch.float32,
                device,
                generator,
                None,
            ).detach()
            for index in range(int(config["guidance"]["active_after_scheduler_index"]) + 1):
                stage = f"prefix_transformer_step_{index}"
                _emit_progress(stage, torch)
                conditional = _transformer_velocity(
                    pipe, latent, timesteps[index], prompt_embeddings, "cond", torch
                )
                unconditional = _transformer_velocity(
                    pipe, latent, timesteps[index], negative_embeddings, "uncond", torch
                )
                transformer_calls += 2
                guided = unconditional + float(generation["guidance_scale"]) * (
                    conditional - unconditional
                )
                latent = pipe.scheduler.step(
                    guided, timesteps[index], latent, return_dict=False
                )[0].detach()
                del conditional, unconditional, guided
        boundary_latent = latent.detach().clone()
        stage = "capture_unipc_after_step_5"
        _emit_progress(stage, torch)
        scheduler_snapshot, scheduler_summary = capture_unipc_snapshot(
            pipe.scheduler, torch, expected_step_index=6
        )
        stage = "differentiable_vae_observer_gradient"
        _emit_progress(stage, torch)
        gradients, gradient_diagnostics = common_axis_gradients_torch(
            boundary_latent, pipe.vae, torch
        )
        vae_decodes += 1
        condition_latents, update_diagnostics = condition_latents_from_common_gradients_torch(
            boundary_latent,
            gradients,
            torch,
            relative_update_rms=float(config["guidance"]["relative_update_rms"]),
        )
        del gradients
        for condition in G0_CONDITION_ORDER:
            stage = f"continue_condition_{condition}"
            _emit_progress(stage, torch)
            final_latents[condition], calls = _continue_condition(
                pipe,
                condition_latents[condition],
                scheduler_snapshot,
                scheduler_summary,
                timesteps,
                prompt_embeddings,
                negative_embeddings,
                config,
                torch,
            )
            transformer_calls += calls
        observations: dict[str, Any] = {}
        rgb_relative: dict[str, dict[str, float]] = {}
        with torch.inference_mode():
            stage = "decode_final_OFF_R1"
            _emit_progress(stage, torch)
            off_rgb_1 = decode_wan_latents_torch(final_latents["OFF_R1"], pipe.vae, torch)
            vae_decodes += 1
            observations["OFF_R1"] = fixed_observer_torch(off_rgb_1, torch)[0].detach().float().cpu().tolist()
            stage = "decode_final_OFF_R2"
            _emit_progress(stage, torch)
            off_rgb_2 = decode_wan_latents_torch(final_latents["OFF_R2"], pipe.vae, torch)
            vae_decodes += 1
            observations["OFF_R2"] = fixed_observer_torch(off_rgb_2, torch)[0].detach().float().cpu().tolist()
            off_rgb_floor = _relative_rgb_rms(off_rgb_2, off_rgb_1, torch)
            for condition in G0_ACTIVE_CONDITIONS:
                stage = f"decode_final_{condition}"
                _emit_progress(stage, torch)
                active_rgb = decode_wan_latents_torch(final_latents[condition], pipe.vae, torch)
                vae_decodes += 1
                observations[condition] = fixed_observer_torch(active_rgb, torch)[0].detach().float().cpu().tolist()
                rgb_relative[condition] = {
                    "OFF_R1": _relative_rgb_rms(active_rgb, off_rgb_1, torch),
                    "OFF_R2": _relative_rgb_rms(active_rgb, off_rgb_2, torch),
                }
                del active_rgb
        stage = "evaluate_frozen_g0_metrics"
        _emit_progress(stage, torch)
        metrics = evaluate_g0_metrics(
            observations,
            gradient_rms=gradient_diagnostics["rms"],
            gradient_cosine=float(gradient_diagnostics["cosine"]),
            rgb_relative_rms=rgb_relative,
            off_rgb_relative_rms=off_rgb_floor,
            config=config,
        )
        result = {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "diagnostic_class": "DIAGNOSTIC_ONLY",
            "status": metrics["status"],
            "runtime": runtime,
            "scheduler": {
                "class": type(pipe.scheduler).__name__,
                "timesteps": [float(value.detach().float().item()) for value in timesteps],
                "boundary_after_index": 5,
                "snapshot": scheduler_summary,
                "condition_local_forks": list(G0_CONDITION_ORDER),
            },
            "gradient": gradient_diagnostics,
            "updates": update_diagnostics,
            "observations": observations,
            "metrics": metrics,
            "execution": {
                "transformer_calls": transformer_calls,
                "vae_decodes": vae_decodes,
                "normal_steps_remaining": [6, 7],
                "MP4_was_written": False,
                "AISB_was_run": False,
                "calibration_was_run": False,
                "Viterbi_was_run": False,
            },
        }
        if transformer_calls != 36 or vae_decodes != 7:
            raise G0InstrumentationError("G0 execution budget changed")
        output.mkdir()
        (output / "result.json").write_text(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return result
    except G0InstrumentationError as exc:
        raise G0InstrumentationError(f"{stage}: {exc}") from exc
    except Exception as exc:
        raise G0InstrumentationError(
            f"{stage}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            pipe.maybe_free_model_hooks()
        except Exception:
            pass


def g0_exit_code(status: str) -> int:
    if status == "FLOW_GUIDANCE_PRIMITIVE_FEASIBLE":
        return 0
    if status == "FLOW_GUIDANCE_PRIMITIVE_NOT_FEASIBLE":
        return 3
    if status == "INSTRUMENTATION_INSUFFICIENT":
        return 2
    raise G0InstrumentationError("unknown G0 status")
