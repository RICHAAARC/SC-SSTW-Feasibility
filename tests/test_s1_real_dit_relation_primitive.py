from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sstw.relation_injector import (
    B1_KEY_PAIRS,
    B2_KEY_PAIRS,
    PAIR_COEFFICIENTS,
    TARGET_QUERY_INDICES,
    S1WanRelationProcessor,
    apply_sparse_logit_bias_numpy,
    install_s1_processor,
    relation_probability_contrasts_numpy,
    validate_frozen_sparse_geometry,
)
from sstw.s1_real_dit_relation_primitive import (
    BRANCH_ORDER,
    capture_block_output,
    CONDITION_ORDER,
    LAYER_ORDER,
    evaluate_preregistered_statistics,
    load_frozen_inputs,
    runtime_capability_diagnostics,
    S1InstrumentationError,
    transformer_forward_inference,
    validate_method_runtime_interface,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeIndex:
    def __init__(self, values: tuple[int, ...], device: str) -> None:
        self.values = np.asarray(values, dtype=np.int64)
        self.device = device


class _FakeDeviceTensor:
    def __init__(self, value: np.ndarray, device: str, index_log: list[tuple[str, str]]) -> None:
        self.value = np.asarray(value, dtype=np.float64)
        self.device = device
        self.index_log = index_log

    def index_select(self, dimension: int, index: _FakeIndex) -> "_FakeDeviceTensor":
        self.index_log.append((self.device, index.device))
        if self.device != index.device:
            raise RuntimeError("device mismatch")
        return _FakeDeviceTensor(np.take(self.value, index.values, axis=dimension), self.device, self.index_log)

    def detach(self) -> "_FakeDeviceTensor":
        return self

    def float(self) -> "_FakeDeviceTensor":
        return self

    def cpu(self) -> "_FakeDeviceTensor":
        return _FakeDeviceTensor(self.value.copy(), "cpu", self.index_log)

    def square(self) -> "_FakeDeviceTensor":
        return _FakeDeviceTensor(np.square(self.value), self.device, self.index_log)

    def mean(self) -> "_FakeDeviceTensor":
        return _FakeDeviceTensor(np.asarray(self.value.mean()), self.device, self.index_log)

    def sqrt(self) -> "_FakeDeviceTensor":
        return _FakeDeviceTensor(np.sqrt(self.value), self.device, self.index_log)

    def item(self) -> float:
        return float(self.value.item())


class _FakeCaptureTorch:
    long = "long"

    @staticmethod
    def tensor(values: tuple[int, ...], *, device: str, dtype: str) -> _FakeIndex:
        assert dtype == "long"
        return _FakeIndex(values, device)


def test_block_capture_indexes_on_target_device_then_moves_slice_to_cpu() -> None:
    values = np.arange(1 * 8320 * 3, dtype=np.float64).reshape(1, 8320, 3) / 100.0
    index_log: list[tuple[str, str]] = []
    block_output = _FakeDeviceTensor(values, "cuda:7", index_log)
    target_rows, global_rms, full = capture_block_output(block_output, _FakeCaptureTorch())
    assert index_log == [("cuda:7", "cuda:7")]
    assert target_rows.device == "cpu" and full.device == "cpu"
    assert np.array_equal(target_rows.value, values[:, TARGET_QUERY_INDICES, :])
    assert np.array_equal(full.value, values)
    assert global_rms == pytest.approx(float(np.sqrt(np.mean(np.square(values)))))


class _FakeInferenceContext:
    def __init__(self, torch: "_FakeInferenceTorch") -> None:
        self.torch = torch

    def __enter__(self) -> None:
        self.previous = (self.torch.grad_enabled, self.torch.inference_enabled)
        self.torch.grad_enabled = False
        self.torch.inference_enabled = True

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.torch.grad_enabled, self.torch.inference_enabled = self.previous


class _FakeInferenceTorch:
    def __init__(self) -> None:
        self.grad_enabled = True
        self.inference_enabled = False

    def inference_mode(self) -> _FakeInferenceContext:
        return _FakeInferenceContext(self)

    def is_grad_enabled(self) -> bool:
        return self.grad_enabled

    def is_inference_mode_enabled(self) -> bool:
        return self.inference_enabled


class _FakeCacheContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        return None


class _FakeVelocity:
    def __init__(self, grad_fn: object | None) -> None:
        self.grad_fn = grad_fn

    def detach(self) -> "_FakeVelocity":
        return _FakeVelocity(None)


class _FakeInferenceTransformer:
    def __init__(self, torch: _FakeInferenceTorch) -> None:
        self.torch = torch
        self.processor_flags: list[tuple[bool, bool]] = []

    def cache_context(self, branch: str) -> _FakeCacheContext:
        assert branch in {"cond", "uncond"}
        return _FakeCacheContext()

    def __call__(self, **kwargs: object) -> tuple[_FakeVelocity]:
        assert kwargs["return_dict"] is False and kwargs["attention_kwargs"] is None
        self.processor_flags.append((self.torch.is_grad_enabled(), self.torch.is_inference_mode_enabled()))
        grad_fn = object() if self.torch.is_grad_enabled() else None
        return (_FakeVelocity(grad_fn),)


def test_transformer_fake_processor_path_runs_in_inference_mode_without_grad_fn() -> None:
    torch = _FakeInferenceTorch()
    transformer = _FakeInferenceTransformer(torch)
    output = transformer_forward_inference(
        torch,
        transformer,
        branch="cond",
        hidden_states=object(),
        timestep=object(),
        encoder_hidden_states=object(),
    )
    assert transformer.processor_flags == [(False, True)]
    assert output.grad_fn is None
    assert torch.is_grad_enabled() is True and torch.is_inference_mode_enabled() is False


def test_authority_config_conditions_and_exact20_are_frozen() -> None:
    authority = ROOT / "SSTW_METHOD_AUTHORITY.md"
    assert hashlib.sha256(authority.read_bytes()).hexdigest() == "a34cad7f174ab8103ac7c54795393164f1ed74e559927da1198e18abb8836d18"
    authority_text = authority.read_text(encoding="utf-8")
    assert authority_text.count("## 项目推进第一性原则") == 1
    assert "所有证据均为 `DIAGNOSTIC_ONLY`" in authority_text
    config, plan = load_frozen_inputs(ROOT)
    assert config["runtime_capabilities"] == {"cuda_available": True, "bf16_supported": True}
    assert "required_gpu" not in config["model"] and "required_cuda_bf16" not in config["model"]
    assert config["method_interface"] == {"diffusers": "0.35.2"}
    assert config["notebook_install_suggestions"] == {
        "python": "3.11", "accelerate": "1.4.0", "ftfy": "6.3.1",
        "huggingface_hub": "0.35.3", "numpy": "1.26.4", "safetensors": "0.5.3",
        "transformers": "4.49.0",
    }
    assert config["conditions"] == list(CONDITION_ORDER)
    assert config["call_budget"] == {
        "prefix_steps": [0, 1, 2, 3], "prefix_branches": ["cond", "uncond"],
        "prefix_transformer_calls": 8, "probe_step": 4, "probe_conditions": 6,
        "probe_branches": ["cond", "uncond"], "probe_transformer_calls": 12,
        "total_transformer_calls": 20, "scheduler_steps_executed": 4, "retry_count": 0,
    }
    assert plan["exact_transformer_calls"] == 20
    assert config["flow_support"] == {
        "scheduler_index": 4, "expected_timestep": 749, "support": [4], "lambda": 1.0,
        "inject_cond_branch": True, "inject_uncond_branch": True,
    }
    assert config["generation"]["prompt"] == "locked camera, dark matte background, a single bright white cube moving slowly across the center, simple studio lighting, no text, no cuts"
    assert config["generation"]["seed"] == 1275


def test_method_construction_and_plan_have_zero_drift() -> None:
    config, _ = load_frozen_inputs(ROOT)
    method_keys = (
        "model", "source_topology", "generation", "transformer_identity", "flow_support",
        "relation_basis", "conditions", "condition_axes", "call_budget",
        "response_definitions", "thresholds", "outcomes", "formal_result",
        "stage_progression_allowed",
    )
    frozen_method = {key: config[key] for key in method_keys}
    digest = hashlib.sha256(
        json.dumps(frozen_method, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert digest == "9ca66eb07aa9bc5468e04e1c7a212a000cb14650d959a4c17f7b80fd51c0e7c3"
    assert hashlib.sha256((ROOT / "plans/s1_real_dit_relation_primitive.json").read_bytes()).hexdigest() == "1de73796528c574ba9e8502824cbbb9ee495cbcf1e8f35dcd78642a99dce96e7"


class _FakeCuda:
    def __init__(self, *, available: bool, bf16: bool, name: str) -> None:
        self.available = available
        self.bf16 = bf16
        self.name = name

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def get_device_name(self, _index: int) -> str:
        return self.name


class _FakeTorch:
    class version:
        cuda = "13.7-diagnostic"

    __version__ = "2.99.0+future"

    def __init__(self, *, available: bool, bf16: bool, name: str) -> None:
        self.cuda = _FakeCuda(available=available, bf16=bf16, name=name)


def test_runtime_gate_is_capability_only_and_keeps_environment_diagnostic() -> None:
    torch = _FakeTorch(available=True, bf16=True, name="Arbitrary Future GPU")
    assert runtime_capability_diagnostics(
        torch,
        python_version="3.12.99",
        auxiliary_packages={"accelerate": "99.1.7", "transformers": "88.4.2"},
    ) == {
        "cuda_available": True,
        "bf16_supported": True,
        "torch": "2.99.0+future",
        "cuda": "13.7-diagnostic",
        "gpu": "Arbitrary Future GPU",
        "python": "3.12.99",
        "auxiliary_packages": {"accelerate": "99.1.7", "transformers": "88.4.2"},
    }
    with pytest.raises(S1InstrumentationError, match="CUDA and BF16"):
        runtime_capability_diagnostics(_FakeTorch(available=False, bf16=True, name="CPU"))
    with pytest.raises(S1InstrumentationError, match="CUDA and BF16"):
        runtime_capability_diagnostics(_FakeTorch(available=True, bf16=False, name="Any GPU"))


def _method_interface_kwargs(config: dict[str, object]) -> dict[str, str]:
    topology = config["source_topology"]
    assert isinstance(topology, dict)
    return {
        "diffusers_version": "0.35.2",
        "transformer_source_sha256": str(topology["transformer_wan_raw_sha256"]),
        "pipeline_source_sha256": str(topology["pipeline_wan_raw_sha256"]),
        "relation_callsite": "transformer.blocks[14].attn1.processor",
        "processor_class": "WanAttnProcessor",
    }


def test_only_diffusers_and_wan_relation_interface_are_method_gates() -> None:
    config, _ = load_frozen_inputs(ROOT)
    validate_method_runtime_interface(config, **_method_interface_kwargs(config))


@pytest.mark.parametrize(
    ("field", "drift"),
    (
        ("diffusers_version", "0.35.3"),
        ("transformer_source_sha256", "0" * 64),
        ("pipeline_source_sha256", "1" * 64),
        ("relation_callsite", "transformer.blocks[15].attn1.processor"),
        ("processor_class", "DifferentProcessor"),
    ),
)
def test_diffusers_source_topology_and_relation_callsite_drift_are_rejected(field: str, drift: str) -> None:
    config, _ = load_frozen_inputs(ROOT)
    observed = _method_interface_kwargs(config)
    observed[field] = drift
    with pytest.raises(S1InstrumentationError, match="method interface|source topology"):
        validate_method_runtime_interface(config, **observed)


def test_production_runtime_and_notebook_have_no_exact_environment_blocker() -> None:
    paths = (
        ROOT / "configs/s1_real_dit_relation_primitive.json",
        ROOT / "src/sstw/s1_real_dit_relation_primitive.py",
        ROOT / "notebooks/sstw_s1_real_dit_relation_primitive.ipynb",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in ("NVIDIA L4", "2.6.0+cu124", "CUDA 12.4", '"12.4"', "'12.4'"):
        assert forbidden not in combined
    notebook = json.loads(paths[-1].read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells[0]["source"] == [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
    ]
    code = "".join("".join(cell["source"]) for cell in code_cells)
    assert "torch.cuda.is_available()" in code and "torch.cuda.is_bf16_supported()" in code
    assert "diffusers==0.35.2" in code
    assert "0fad780a534b6463e45facd96134c9f345acfa5b" in code
    source = paths[1].read_text(encoding="utf-8")
    assert "notebook_install_suggestions" not in source
    assert "software_diagnostics !=" not in source and "locked_software" not in source


def test_frozen_geometry_has_13_queries_two_orthogonal_unit_pairs() -> None:
    validate_frozen_sparse_geometry()
    assert TARGET_QUERY_INDICES == tuple(336 + 640 * index for index in range(13))
    assert all(pair == (query - 1, query + 1) for query, pair in zip(TARGET_QUERY_INDICES, B1_KEY_PAIRS, strict=True))
    assert all(pair == (query - 32, query + 32) for query, pair in zip(TARGET_QUERY_INDICES, B2_KEY_PAIRS, strict=True))
    assert sum(value * value for value in PAIR_COEFFICIENTS) == pytest.approx(1.0)
    assert sum(PAIR_COEFFICS := PAIR_COEFFICIENTS) == pytest.approx(0.0)
    assert set(B1_KEY_PAIRS[0]).isdisjoint(B2_KEY_PAIRS[0]) and len(PAIR_COEFFICS) == 2


def test_sparse_numpy_reference_changes_only_four_keys_and_is_odd() -> None:
    logits = np.zeros((3, 3, 8), dtype=np.float64)
    query = (1,)
    b1 = ((2, 3),)
    b2 = ((5, 6),)
    plus, plus_updates = apply_sparse_logit_bias_numpy(logits, query, b1, b2, (1.0, 1.0), 1.0)
    minus, minus_updates = apply_sparse_logit_bias_numpy(logits, query, b1, b2, (-1.0, -1.0), 1.0)
    changed = np.argwhere(plus != logits)
    assert changed.shape[0] == 3 * 4
    assert set(changed[:, 1]) == {1} and set(changed[:, 2]) == {2, 3, 5, 6}
    assert np.array_equal(plus + minus, logits * 2.0)
    assert all(left[0] == right[0] and left[2] + right[2] == pytest.approx(0.0) for left, right in zip(plus_updates[::2], plus_updates[1::2], strict=True))
    assert len(plus_updates) == len(minus_updates) == 4
    zero, updates = apply_sparse_logit_bias_numpy(logits, query, b1, b2, (0.0, 0.0), 1.0)
    assert np.array_equal(zero, logits) and updates == ()


def test_sparse_probability_jacobian_has_correct_sign_low_cross_and_common_mode() -> None:
    logits = np.zeros((2, 3, 8), dtype=np.float64)
    query, b1, b2 = (1,), ((2, 3),), ((5, 6),)
    responses = {}
    for name, state in {"p1": (1.0, 0.0), "m1": (-1.0, 0.0), "p2": (0.0, 1.0), "m2": (0.0, -1.0)}.items():
        biased, _ = apply_sparse_logit_bias_numpy(logits, query, b1, b2, state, 1.0)
        probabilities = np.exp(biased - biased.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        responses[name] = relation_probability_contrasts_numpy(probabilities, query, b1, b2)
    odd_1 = (responses["p1"] - responses["m1"]) / 2.0
    odd_2 = (responses["p2"] - responses["m2"]) / 2.0
    even_1 = (responses["p1"] + responses["m1"]) / 2.0
    even_2 = (responses["p2"] + responses["m2"]) / 2.0
    assert odd_1[..., 0].mean() > 0 and odd_2[..., 1].mean() > 0
    assert np.max(np.abs(odd_1[..., 1])) < 1e-15 and np.max(np.abs(odd_2[..., 0])) < 1e-15
    assert np.max(np.abs(even_1)) < 1e-15 and np.max(np.abs(even_2)) < 1e-15


def _synthetic_probe() -> dict[str, object]:
    probe: dict[str, object] = {}
    relation_basis = {
        "B1": np.tile(np.asarray([[[0.1, 0.0]]]), (13, 1, 1)),
        "B2": np.tile(np.asarray([[[0.0, 0.1]]]), (13, 1, 1)),
    }
    flat_basis = {
        "B1": np.tile(np.asarray([[0.1, 0.0]]), (13, 1)),
        "B2": np.tile(np.asarray([[0.0, 0.1]]), (13, 1)),
    }
    for condition in CONDITION_ORDER:
        if condition.endswith("B1"):
            sign, axis = (-1.0 if condition.startswith("MINUS") else 1.0), "B1"
        elif condition.endswith("B2"):
            sign, axis = (-1.0 if condition.startswith("MINUS") else 1.0), "B2"
        else:
            sign, axis = 0.0, "B1"
        probe[condition] = {}
        for branch in BRANCH_ORDER:
            base_relation = np.zeros((13, 1, 2))
            base_flat = np.ones((13, 2))
            probe[condition][branch] = {
                "relation": (base_relation + sign * relation_basis[axis]).tolist(),
                "block": (base_flat + sign * flat_basis[axis]).tolist(),
                "velocity": (base_flat + sign * flat_basis[axis]).tolist(),
            }
    return probe


def test_preregistered_O_E_N_gate_is_per_layer_branch_axis() -> None:
    config, _ = load_frozen_inputs(ROOT)
    overrides = {(branch, layer, axis): 0.005 for branch in (*BRANCH_ORDER, "guidance") for layer in ("block", "velocity") for axis in ("B1", "B2")}
    result = evaluate_preregistered_statistics(_synthetic_probe(), config["thresholds"], overrides)
    assert result["all_cells_pass"] is True
    assert set(result["cells"]) == {f"{layer}:{branch}" for branch in (*BRANCH_ORDER, "guidance") for layer in LAYER_ORDER}
    assert all(cell["axis_absolute_cosine"] < 0.5 and cell["passed"] for cell in result["cells"].values())


def test_fake_wan_adapter_installs_only_block14_processor() -> None:
    class Attention:
        def __init__(self) -> None:
            self.processor = object()

        def set_processor(self, value: object) -> None:
            self.processor = value

    class Block:
        def __init__(self) -> None:
            self.attn1 = Attention()

    class Transformer:
        def __init__(self) -> None:
            self.blocks = [Block() for _ in range(30)]

    transformer = Transformer()
    originals = [block.attn1.processor for block in transformer.blocks]
    processor, original = install_s1_processor(transformer, object(), block_index=14, strength=1.0)
    assert isinstance(processor, S1WanRelationProcessor) and original is originals[14]
    assert transformer.blocks[14].attn1.processor is processor
    assert all(block.attn1.processor is originals[index] for index, block in enumerate(transformer.blocks) if index != 14)


def test_real_processor_source_has_qkv_norm_rope_sparse_rows_and_no_proxy() -> None:
    source = inspect.getsource(S1WanRelationProcessor)
    for token in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.norm_q", "attn.norm_k", "_apply_rotary", "dispatch_attention_fn", "torch.softmax", "TARGET_QUERY_INDICES", "torch.is_grad_enabled()", "torch.is_inference_mode_enabled()"):
        assert token in source
    assert "8320, 8320" not in source and "register_forward_hook" not in source
    forbidden = ("output_residual", "latent_carrier", "vae.decode", "pixel_carrier")
    assert all(token not in source.lower() for token in forbidden)


def test_runner_source_freezes_prefix_fork_and_no_VAE_or_MP4_execution() -> None:
    source = (ROOT / "src/sstw/s1_real_dit_relation_primitive.py").read_text(encoding="utf-8")
    assert "for scheduler_index in range(4)" in source
    assert "probe_latent = latents.detach().clone()" in source
    assert "for condition in CONDITION_ORDER" in source
    assert source.count("pipe.scheduler.step(") == 1
    assert "with torch.inference_mode():" in source
    assert "S1 transformer call escaped inference mode" in source
    assert "S1 transformer output retained an autograd graph" in source
    assert "pipe.scheduler.step(guided, timestep, latents, return_dict=False)[0].detach()" in source
    assert ".vae.decode(" not in source and "encode_saved_mp4" not in source
