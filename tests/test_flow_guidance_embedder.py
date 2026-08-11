from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from sstw.flow_guidance_embedder import (
    FrozenGuidanceConfig,
    OBSERVER_FRAME_INDICES,
    evaluate_g0_metrics,
    load_g0_config,
    scalar_cosine,
    scalar_rms,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_g0_config_and_frames() -> None:
    config = FrozenGuidanceConfig()
    config.validate()
    assert OBSERVER_FRAME_INDICES == tuple(range(0, 49, 4))
    assert config.relative_update_rms == 0.005
    loaded = load_g0_config(ROOT / "configs/g0_flow_guidance_primitive.json")
    assert loaded["generation"]["max_sequence_length"] == 226
    assert loaded["criteria"]["minimum_odd_to_floor_ratio"] == 8.0


def test_scalar_axis_diagnostics() -> None:
    assert scalar_rms((3.0, 4.0)) == pytest.approx(math.sqrt(12.5))
    assert scalar_cosine((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert scalar_cosine((1.0, -1.0), (-1.0, 1.0)) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "left,right",
    [
        ((), ()),
        ((1.0,), (1.0, 2.0)),
        ((0.0, 0.0), (1.0, 0.0)),
        ((float("nan"),), (1.0,)),
    ],
)
def test_bad_axis_diagnostics_reject(left, right) -> None:
    with pytest.raises(ValueError):
        scalar_cosine(left, right)


def _observations(amplitude: float) -> dict[str, list[list[float]]]:
    zero = [[0.0, 0.0] for _ in range(13)]
    return {
        "OFF_R1": [row[:] for row in zero],
        "OFF_R2": [row[:] for row in zero],
        "PLUS_G1": [[amplitude, 0.0] for _ in range(13)],
        "MINUS_G1": [[-amplitude, 0.0] for _ in range(13)],
        "PLUS_G2": [[0.0, amplitude] for _ in range(13)],
        "MINUS_G2": [[0.0, -amplitude] for _ in range(13)],
    }


def test_effect_must_exceed_frozen_off_and_numeric_floor() -> None:
    config = load_g0_config(ROOT / "configs/g0_flow_guidance_primitive.json")
    quality = {
        condition: {"OFF_R1": 0.001, "OFF_R2": 0.001}
        for condition in ("PLUS_G1", "MINUS_G1", "PLUS_G2", "MINUS_G2")
    }
    passed = evaluate_g0_metrics(
        _observations(1.0e-3),
        gradient_rms={"G1": 1.0, "G2": 1.0},
        gradient_cosine=0.0,
        rgb_relative_rms=quality,
        off_rgb_relative_rms=0.0,
        config=config,
    )
    assert passed["status"] == "FLOW_GUIDANCE_PRIMITIVE_FEASIBLE"
    failed = evaluate_g0_metrics(
        _observations(1.0e-8),
        gradient_rms={"G1": 1.0, "G2": 1.0},
        gradient_cosine=0.0,
        rgb_relative_rms=quality,
        off_rgb_relative_rms=0.0,
        config=config,
    )
    assert failed["status"] == "FLOW_GUIDANCE_PRIMITIVE_NOT_FEASIBLE"
    assert failed["checks"]["g1_odd_above_floor"] is False
    json.dumps(failed, allow_nan=False)


def test_scheduler_tensor_hash_flattens_before_uint8_view() -> None:
    source = (ROOT / "src/sstw/flow_guidance_embedder.py").read_text()
    assert "contiguous.reshape(-1).view(torch_module.uint8)" in source
    assert "contiguous.view(torch_module.uint8)" not in source


def test_vjp_memory_path_checkpoints_recurrent_frames_without_offload() -> None:
    source = (ROOT / "src/sstw/flow_guidance_embedder.py").read_text()
    assert "save_on_cpu" not in source
    assert "pin_memory=True" not in source
    assert "retain_graph=True" not in source
    assert 'for axis in ("G1", "G2")' in source
    assert "_checkpoint_wan_decoder_frames" in source
    assert "use_reentrant=False" in source
    assert "call_index != 13" in source
