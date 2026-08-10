from __future__ import annotations

import math

import pytest

from sstw.flow_guidance_embedder import (
    FrozenGuidanceConfig,
    OBSERVER_FRAME_INDICES,
    scalar_cosine,
    scalar_rms,
)


def test_frozen_g0_config_and_frames() -> None:
    config = FrozenGuidanceConfig()
    config.validate()
    assert OBSERVER_FRAME_INDICES == tuple(range(0, 49, 4))
    assert config.relative_update_rms == 0.005


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
