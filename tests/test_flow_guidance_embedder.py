from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from sstw.flow_guidance_embedder import (
    FrozenGuidanceConfig,
    G0_CONDITION_ORDER,
    OBSERVER_FRAME_INDICES,
    _continue_condition,
    capture_unipc_snapshot,
    condition_latents_from_common_gradients_torch,
    evaluate_g0_metrics,
    load_g0_config,
    scalar_cosine,
    scalar_rms,
    wan_checkpoint_canary_torch,
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
    assert "if call_index == 0" not in source
    assert "decoded_rgb.index_select" in source
    assert ").float()" in source
    assert "vae_decodes += 2" in source
    assert "vae_decodes != 8" in source
    assert "pipe.maybe_free_model_hooks()" in source


def test_real_wan_decoder_checkpoint_output_and_gradient_match() -> None:
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    diagnostics = wan_checkpoint_canary_torch(
        torch, diffusers.AutoencoderKLWan, torch.device("cpu")
    )
    assert diagnostics["latent_frames"] == 13
    assert diagnostics["decoded_frames"] == 49
    assert diagnostics["maximum_output_absolute_error"] == 0.0
    assert diagnostics["maximum_gradient_absolute_error"] == 0.0
    assert diagnostics["gradient_rms"] > 0.0


def test_condition_latents_and_six_stateful_unipc_tails() -> None:
    torch = pytest.importorskip("torch")

    base = torch.ones((1, 16, 13, 40, 64), dtype=torch.float32)
    gradients = {
        "G1": torch.ones_like(base),
        "G2": torch.linspace(-1.0, 1.0, base.numel()).reshape_as(base),
    }
    conditions, _ = condition_latents_from_common_gradients_torch(
        base, gradients, torch, relative_update_rms=0.005
    )
    assert tuple(conditions) == G0_CONDITION_ORDER
    assert torch.equal(conditions["OFF_R1"], conditions["OFF_R2"])
    assert conditions["OFF_R1"].data_ptr() != conditions["OFF_R2"].data_ptr()

    class UniPCMultistepScheduler:
        def __init__(self) -> None:
            self._step_index = 6
            self.lower_order_nums = 3
            self.model_outputs = [torch.zeros(1)]

        def step(self, model_output, timestep, sample, return_dict=False):
            assert int(timestep.item()) == self._step_index
            self._step_index += 1
            return (sample + model_output * 0.0,)

    class CacheContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class Transformer:
        dtype = torch.float32

        def __init__(self) -> None:
            self.calls = 0

        def cache_context(self, branch):
            assert branch in {"cond", "uncond"}
            return CacheContext()

        def __call__(self, *, hidden_states, **kwargs):
            self.calls += 1
            return (torch.zeros_like(hidden_states),)

    class Pipe:
        def __init__(self) -> None:
            self.transformer = Transformer()

    retained = UniPCMultistepScheduler()
    snapshot, summary = capture_unipc_snapshot(retained, torch, expected_step_index=6)
    pipe = Pipe()
    config = {
        "guidance": {"normal_steps_remaining": [6, 7]},
        "generation": {"guidance_scale": 5.0},
    }
    timesteps = torch.arange(8)
    embeddings = torch.zeros((1, 1, 1))
    for condition in G0_CONDITION_ORDER:
        result, calls = _continue_condition(
            pipe,
            conditions[condition],
            snapshot,
            summary,
            timesteps,
            embeddings,
            embeddings,
            config,
            torch,
        )
        assert calls == 4
        assert tuple(result.shape) == tuple(base.shape)
    assert pipe.transformer.calls == 24
    assert retained._step_index == 6


def test_full_fake_tail_writes_result_with_exact_budgets(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    import sstw.flow_guidance_embedder as module

    class UniPCMultistepScheduler:
        def __init__(self) -> None:
            self._step_index = 0
            self.lower_order_nums = 0
            self.model_outputs = [torch.zeros(1)]
            self.timesteps = torch.arange(8)

        def set_timesteps(self, count, device=None):
            assert count == 8
            self.timesteps = torch.arange(8, device=device)
            self._step_index = 0

        def set_begin_index(self, index):
            assert index == 0

        def step(self, model_output, timestep, sample, return_dict=False):
            assert int(timestep.item()) == self._step_index
            self._step_index += 1
            return (sample,)

    class CacheContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class Transformer:
        dtype = torch.float32
        config = type("Config", (), {"in_channels": 16})()

        def __init__(self) -> None:
            self.calls = 0

        def cache_context(self, branch):
            assert branch in {"cond", "uncond"}
            return CacheContext()

        def __call__(self, *, hidden_states, **kwargs):
            self.calls += 1
            return (torch.zeros_like(hidden_states),)

    class Pipe:
        def __init__(self) -> None:
            self._execution_device = torch.device("cpu")
            self.transformer = Transformer()
            self.scheduler = UniPCMultistepScheduler()
            self.vae = object()

        def enable_model_cpu_offload(self):
            return None

        def maybe_free_model_hooks(self):
            return None

        def encode_prompt(self, **kwargs):
            value = torch.zeros((1, 1, 1))
            return value, value.clone()

        def prepare_latents(self, *args):
            return torch.ones((1, 16, 13, 40, 64), dtype=torch.float32)

    pipe = Pipe()
    monkeypatch.setattr(
        diffusers.WanPipeline,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: pipe),
    )
    monkeypatch.setattr(
        module,
        "runtime_capability_diagnostics",
        lambda torch_module: {"cuda_available": True, "bf16_supported": True},
    )
    monkeypatch.setattr(module, "wan_checkpoint_canary_torch", lambda *args: {"ok": 1})

    def gradients(base, vae, torch_module):
        first = torch.zeros_like(base)
        second = torch.zeros_like(base)
        split = base.numel() // 2
        first.reshape(-1)[:split] = -1.0
        second.reshape(-1)[split:] = -1.0
        return {"G1": first, "G2": second}, {
            "loss": {"G1": 1.0, "G2": 1.0},
            "rms": {"G1": module._tensor_rms_float(first, torch), "G2": module._tensor_rms_float(second, torch)},
            "cosine": 0.0,
            "absolute_cosine": 0.0,
            "common_base_latent_rms": 1.0,
        }

    def observer(value, torch_module):
        flat = value.reshape(value.shape[0], -1)
        split = flat.shape[1] // 2
        point = torch.stack((flat[:, :split].mean(1), flat[:, split:].mean(1)), dim=1)
        return point[:, None, :].repeat(1, 13, 1)

    monkeypatch.setattr(module, "common_axis_gradients_torch", gradients)
    monkeypatch.setattr(module, "decode_wan_latents_torch", lambda latent, vae, torch_module: latent)
    monkeypatch.setattr(module, "fixed_observer_torch", observer)
    monkeypatch.setattr(module, "_relative_rgb_rms", lambda *args: 0.0)
    output = tmp_path / "output"
    result = module.run_g0_once(repo_root=ROOT, output=output)
    assert all(result["metrics"]["checks"].values()), result["metrics"]["checks"]
    assert result["status"] == "FLOW_GUIDANCE_PRIMITIVE_FEASIBLE"
    assert result["execution"]["transformer_calls"] == 36
    assert result["execution"]["vae_decodes"] == 8
    assert pipe.transformer.calls == 36
    assert json.loads((output / "result.json").read_text())["status"] == result["status"]
    assert not (output / ".result.json.tmp").exists()
