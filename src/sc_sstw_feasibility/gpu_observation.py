"""Minimal GPU/saved-video observation probe helpers.

This module is deliberately narrow. It does not implement detection, fixed-FPR
calibration, observer logic, attacks, or a paper claim. Its only purpose is to
answer whether a real generated/saved video can yield a stable relation
observation ``q_i`` for the SC-SSTW feasibility route.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .aisb import affine_burst_residual, make_default_templates


Vector2 = tuple[float, float]

CLAIM_BOUNDARY = "gpu_observation_probe_only_not_detection_evidence"
DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass(frozen=True)
class ObservationProbeConfig:
    """Small fixed probe contract."""

    template_id: str = "alpha"
    repeat_count: int = 2
    state_window_count: int = 2
    frame_count: int = 16
    width: int = 512
    height: int = 320
    fps: int = 8
    seed: int = 1275
    model_id: str = DEFAULT_MODEL_ID
    dry_run: bool = False


def public_aisb_points() -> tuple[Vector2, ...]:
    """Return the fixed short public AISB-like pattern for the GPU probe."""

    return make_default_templates()[0].points


def relation_prompts() -> tuple[dict[str, Any], ...]:
    """Return fixed prompt records for repeats, AISB points, and state probes."""

    points = public_aisb_points()
    prompts: list[dict[str, Any]] = []
    repeat_prompt = (
        "minimal geometric scene, dark background, one bright square centered, "
        "flat lighting, no camera motion"
    )
    for repeat_index in range(2):
        prompts.append(
            {
                "probe_id": f"repeat_{repeat_index}",
                "kind": "repeat_floor",
                "target_q": (0.5, 0.5),
                "prompt": repeat_prompt,
            }
        )
    for point_index, point in enumerate(points):
        prompts.append(
            {
                "probe_id": f"aisb_{point_index}",
                "kind": "public_aisb",
                "target_q": point,
                "prompt": _point_prompt(point),
            }
        )
    prompts.extend(
        (
            {
                "probe_id": "state_positive",
                "kind": "state_window",
                "target_q": (0.78, 0.24),
                "state_sign": 1,
                "prompt": _point_prompt((0.78, 0.24)),
            },
            {
                "probe_id": "state_negative",
                "kind": "state_window",
                "target_q": (0.22, 0.76),
                "state_sign": -1,
                "prompt": _point_prompt((0.22, 0.76)),
            },
        )
    )
    return tuple(prompts)


def _point_prompt(point: Vector2) -> str:
    x, y = point
    horizontal = "left" if x < 0.33 else "right" if x > 0.67 else "center"
    vertical = "upper" if y < 0.33 else "lower" if y > 0.67 else "middle"
    return (
        "minimal geometric scene, dark background, one bright square in the "
        f"{vertical} {horizontal} region, flat lighting, no camera motion"
    )


def readout_q_from_rgb_frames(frames: Sequence[Any]) -> Vector2:
    """Compute a two-dimensional relation observation from RGB frames."""

    if not frames:
        raise ValueError("frames must not be empty")
    start = max(0, len(frames) // 2 - 2)
    stop = min(len(frames), start + 4)
    selected = frames[start:stop]
    horizontal_values: list[float] = []
    vertical_values: list[float] = []
    for frame in selected:
        width, height = _frame_size(frame)
        patch = max(4, min(width, height) // 8)
        left = _patch_mean(frame, width // 4 - patch // 2, height // 2 - patch // 2, patch)
        right = _patch_mean(frame, 3 * width // 4 - patch // 2, height // 2 - patch // 2, patch)
        top = _patch_mean(frame, width // 2 - patch // 2, height // 4 - patch // 2, patch)
        bottom = _patch_mean(frame, width // 2 - patch // 2, 3 * height // 4 - patch // 2, patch)
        horizontal_values.append(right - left)
        vertical_values.append(bottom - top)
    return (_mean(horizontal_values), _mean(vertical_values))


def _frame_size(frame: Any) -> tuple[int, int]:
    if hasattr(frame, "size"):
        width, height = frame.size
        return int(width), int(height)
    height = len(frame)
    width = len(frame[0]) if height else 0
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    return width, height


def _pixel_rgb(frame: Any, x: int, y: int) -> tuple[float, float, float]:
    if hasattr(frame, "getpixel"):
        pixel = frame.getpixel((x, y))
    else:
        pixel = frame[y][x]
    if len(pixel) < 3:
        raise ValueError("pixel must have at least 3 channels")
    return (float(pixel[0]), float(pixel[1]), float(pixel[2]))


def _patch_mean(frame: Any, x0: int, y0: int, size: int) -> float:
    width, height = _frame_size(frame)
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x0 + size))
    y1 = max(y0 + 1, min(height, y0 + size))
    total = 0.0
    count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            r, g, b = _pixel_rgb(frame, x, y)
            total += (r + g + b) / (3.0 * 255.0)
            count += 1
    if count == 0:
        raise ValueError("empty patch")
    return total / count


def summarize_observations(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute the minimal JSON summary for probe records."""

    q_by_id = {record["probe_id"]: tuple(record["q"][:2]) for record in records}
    required = [record["probe_id"] for record in relation_prompts()]
    missing = [probe_id for probe_id in required if probe_id not in q_by_id]
    if missing:
        raise ValueError(f"missing observation records: {missing}")
    repeat_floor = _distance(q_by_id["repeat_0"], q_by_id["repeat_1"])
    aisb_points = [q_by_id[f"aisb_{index}"] for index in range(len(public_aisb_points()))]
    template = make_default_templates()[0]
    public_residual = affine_burst_residual([list(point) for point in aisb_points], template)
    public_energy = _mean(_norm(point) for point in aisb_points)
    state_delta = _sub(q_by_id["state_positive"], q_by_id["state_negative"])
    state_energy = _norm(state_delta)
    snr_floor = max(repeat_floor, 1e-12)
    visibility_snr = public_energy / snr_floor
    state_snr = state_energy / snr_floor
    observable = bool(
        math.isfinite(public_residual)
        and math.isfinite(visibility_snr)
        and public_residual < 0.25
        and visibility_snr > 3.0
    )
    return {
        "claim_support_status": CLAIM_BOUNDARY,
        "probe_decision": "observable_candidate" if observable else "no_observable_signal",
        "formal_result": False,
        "paper_claim": False,
        "fixed_fpr": False,
        "observer_or_detector": False,
        "record_count": len(records),
        "repeatability_floor_l2": repeat_floor,
        "public_aisb_residual": public_residual,
        "public_relation_energy_mean": public_energy,
        "public_visibility_snr_over_repeat_floor": visibility_snr,
        "state_relation_delta_l2": state_energy,
        "state_relation_snr_over_repeat_floor": state_snr,
        "readout_shape": [2],
        "readout_dtype": "float64_json_number",
        "readout_finite": all(_finite_pair(q_by_id[probe_id]) for probe_id in q_by_id),
        "failure_reason": None if observable else "saved-video relation readout did not clear minimal diagnostic visibility checks",
    }


def make_mock_frames(target_q: Vector2, *, width: int = 64, height: int = 40, frame_count: int = 8) -> list[list[list[tuple[int, int, int]]]]:
    """Create deterministic tiny mock RGB frames for local dry-run tests.

    The mock is constructed in readout space: horizontal patch contrast equals
    ``target_q[0]`` and vertical patch contrast equals ``target_q[1]``. This
    keeps dry-run useful for validating plumbing without pretending to be real
    generated-video evidence.
    """

    x, y = target_q
    frames = []
    for _ in range(frame_count):
        frame = [[(102, 102, 102) for _x in range(width)] for _y in range(height)]
        patch = max(4, min(width, height) // 8)
        _paint_readout_patch(frame, width // 4 - patch // 2, height // 2 - patch // 2, patch, 0.45 - 0.45 * x)
        _paint_readout_patch(frame, 3 * width // 4 - patch // 2, height // 2 - patch // 2, patch, 0.45 + 0.55 * x)
        _paint_readout_patch(frame, width // 2 - patch // 2, height // 4 - patch // 2, patch, 0.45 - 0.45 * y)
        _paint_readout_patch(frame, width // 2 - patch // 2, 3 * height // 4 - patch // 2, patch, 0.45 + 0.55 * y)
        frames.append(frame)
    return frames


def _paint_readout_patch(frame: list[list[tuple[int, int, int]]], x0: int, y0: int, size: int, value: float) -> None:
    height = len(frame)
    width = len(frame[0])
    byte = max(0, min(255, round(value * 255.0)))
    for y in range(max(0, y0), min(height, y0 + size)):
        for x in range(max(0, x0), min(width, x0 + size)):
            frame[y][x] = (byte, byte, byte)


def dry_run_records() -> list[dict[str, Any]]:
    records = []
    for prompt in relation_prompts():
        frames = make_mock_frames(prompt["target_q"])
        records.append(
            {
                "probe_id": prompt["probe_id"],
                "kind": prompt["kind"],
                "target_q": list(prompt["target_q"]),
                "q": list(readout_q_from_rgb_frames(frames)),
                "video_path": None,
            }
        )
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _distance(a: Vector2, b: Vector2) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _sub(a: Vector2, b: Vector2) -> Vector2:
    return (a[0] - b[0], a[1] - b[1])


def _norm(a: Vector2) -> float:
    return math.sqrt(a[0] ** 2 + a[1] ** 2)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def _finite_pair(pair: Vector2) -> bool:
    return math.isfinite(pair[0]) and math.isfinite(pair[1])
