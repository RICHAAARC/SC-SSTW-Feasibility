"""Minimal SSTW-v2 Flow-guidance primitive.

The functions in this module only implement the frozen observer projection and
the bounded latent update.  They do not load a model, choose a video, fit an
observer, or inspect a result in order to tune the construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


OBSERVER_FRAME_INDICES = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)


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
